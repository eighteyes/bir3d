// fluid-budget.spec.ts — Task 4: the make-or-break budget spike (§8.1).
// Runs GpuFluid headless at representative window grids (128², 256²; single 2D layer) with a
// deterministic scripted force, times every pass via PassTimer, and reports the WARM-MEDIAN
// per-stage ms over a steady window (first ~40 frames discarded — shader compile is 10-100×).
// SWEEPS Jacobi iters × grid; for each cell, one-shot readback OUTSIDE the loop → residual
// max|divergence| via the SAME central-difference metric as the oracle (project.rs::max_abs_div).
// Tags the adapter label from acquireDevice. Compares TOTAL fluid ms vs the §3 fluid sub-budget
// (M-series row 3.5-6ms on this Apple-Metal machine) → PASS/MARGINAL/OVER per (grid,iters), and
// reports per-layer ms (2.5D = ×N_layers is one multiply away). Writes the findings to
// tests/fixtures/fluid/budget-findings.json (Node-side). The verdict is DATA, not a hard pass/
// fail: this test asserts ONLY that the run produced finite numbers.
// Responsibilities:
//   - Browser-side (page.evaluate): build GpuFluid per config, warm + steady timed frames,
//     per-stage median, one-shot residual readback, adapter label. Returns plain JSON.
//   - Node-side: classify vs the platform sub-budget, assemble + write findings.json, assert finite.

import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { expect, test } from "@playwright/test";

const HERE = dirname(fileURLToPath(import.meta.url));
const FINDINGS_PATH = join(HERE, "..", "fixtures", "fluid", "budget-findings.json");

// Sweep matrix (§8.1): representative window grids × Jacobi iters.
const GRIDS = [128, 256];
const ITERS = [10, 20, 40, 80];

// Warm-median discipline: discard the first WARMUP frames (compile + ramp), median over STEADY.
const WARMUP = 40;
const STEADY = 80;
const DT = 0.1;

// §3 fluid 2.5D moving-window sub-budget rows (ms). This machine is Apple Metal → M-series row.
// The budget is a RANGE for a reason: treat it as the MARGINAL zone, not a slack PASS band.
// PASS ≤ lo (fits with room), MARGINAL in [lo,hi] (fits only at the budget's edge), OVER > hi.
// This conservative reading uses BOTH band edges and is the honest default for a kill-gate whose
// own spec keeps noting that an isolated-fluid PASS is already optimistic. (Vs the fluid
// SUB-budget, NOT 16.6ms.)
const SUBBUDGET = {
  "m-series": { lo: 3.5, hi: 6.0 },
  discrete: { lo: 1.5, hi: 2.5 },
} as const;

const STAGES = ["forces", "set_bnd", "divergence", "jacobi", "subtract_grad", "advect"] as const;

interface CellResult {
  grid: number;
  iters: number;
  passesPerFrame: number;
  steadyFrames: number;
  perStageMedianMs: Record<string, number>;
  jacobiTotalMs: number;
  totalMedianMs: number;
  sumOfStagesMs: number;
  interPassOverheadMs: number;
  wallClockMedianMs: number;
  residualMaxDiv: number;
}

// ---- Browser-side sweep runner ---------------------------------------------------
// Builds GpuFluid per (grid,iters), runs warm+steady timed frames, returns medians + residual.
// All timing is GPU-side (timestamp-query); per-stage ms = SUM of all passes tagged that stage
// (stages are interleaved/repeated), TOTAL = max(end)-min(begin) over the frame's passes.

const RUN = async (cfg: { grids: number[]; iters: number[]; warmup: number; steady: number; dt: number }) => {
  const { acquireDevice } = await import("/src/host/gpu/device.ts");
  const { GpuFluid } = await import("/src/host/gpu/fluid.ts");
  const { PassTimer } = await import("/src/host/gpu/passtimer.ts");

  const get = (f: string) => fetch(`/src/host/shaders/fluid/${f}.wgsl`).then((r) => r.text());
  const [advect, divergence, jacobi, subtractGrad, setBnd, forces] = await Promise.all([
    get("advect"), get("divergence"), get("jacobi"), get("subtract_grad"), get("set_bnd"), get("forces"),
  ]);
  const shaders = { forces, divergence, jacobi, subtractGrad, advect, setBnd };

  const { adapter, device, hasTimestampQuery } = await acquireDevice();

  // Adapter label — a fluid ms without a platform tag is uninterpretable vs a platform-split budget.
  let adapterLabel = "unknown";
  try {
    const info = (adapter as any).info ?? (await (adapter as any).requestAdapterInfo?.());
    if (info) {
      adapterLabel = [info.vendor, info.architecture, info.device, info.description]
        .filter(Boolean)
        .join(" / ") || "unknown";
    }
  } catch {
    /* adapter info unavailable — leave "unknown" */
  }

  const median = (xs: number[]): number => {
    if (xs.length === 0) return NaN;
    const s = [...xs].sort((a, b) => a - b);
    const mid = s.length >> 1;
    return s.length % 2 ? s[mid] : 0.5 * (s[mid - 1] + s[mid]);
  };

  // Worst-case passes per frame for capacity sizing: 30 + 6*iters (forces 1, three setBndVel ×4,
  // two project[1 div + iters*(1 jacobi + 2 setBnd) + 1 subtract + 4 setBndVel], advect ×2 + 1 dye,
  // final setBndScalar ×2). Sized dynamically, never hardcoded.
  const passesFor = (iters: number) => 30 + 6 * iters;

  // One-shot residual readback (OUTSIDE the timed loop): copy u,v to MAP_READ staging.
  const readField = async (buf: GPUBuffer, bytes: number): Promise<Float32Array> => {
    const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = device.createCommandEncoder();
    e.copyBufferToBuffer(buf, 0, staging, 0, bytes);
    device.queue.submit([e.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  };

  // max|0.5(u[i+1,j]-u[i-1,j]) + 0.5(v[i,j+1]-v[i,j-1])| over interior — oracle max_abs_div.
  const maxAbsDiv = (u: Float32Array, v: Float32Array, w: number, h: number): number => {
    const stride = w + 2;
    const at = (a: Float32Array, i: number, j: number) => a[i + stride * j];
    let m = 0;
    for (let j = 1; j <= h; j++) {
      for (let i = 1; i <= w; i++) {
        const d = 0.5 * (at(u, i + 1, j) - at(u, i - 1, j)) + 0.5 * (at(v, i, j + 1) - at(v, i, j - 1));
        const a = Math.abs(d);
        if (a > m) m = a;
      }
    }
    return m;
  };

  const STAGE_KEYS = ["forces", "set_bnd", "divergence", "jacobi", "subtract_grad", "advect"];
  const cells: any[] = [];

  for (const grid of cfg.grids) {
    for (const iters of cfg.iters) {
      const w = grid;
      const h = grid;
      const bytes = (w + 2) * (h + 2) * 4;

      // Fresh solver per config so this config's own pipeline-compile is absorbed by its warmup.
      const fluid = new GpuFluid(device, w, h, shaders);
      // Deterministic continuous vortex-ish jet from the centre so the field is non-trivial and
      // carries real divergence into project() (otherwise residual is meaninglessly ~0).
      fluid.setForce({
        fx: 40, fy: 12,
        dyeX: (w + 2) / 2, dyeY: (h + 2) / 2,
        dyeR: grid / 8, dyeAmt: 60,
        forceR: grid / 6,
      });

      const timer = new PassTimer(device, hasTimestampQuery, passesFor(iters));

      // Per-stage + total ms samples across the steady window.
      const perStageSamples: Record<string, number[]> = {};
      for (const k of STAGE_KEYS) perStageSamples[k] = [];
      const totalSamples: number[] = [];
      const sumSamples: number[] = [];
      let lastPasses = 0;

      const total = cfg.warmup + cfg.steady;
      for (let frame = 0; frame < total; frame++) {
        timer.reset();
        const enc = device.createCommandEncoder();
        fluid.step(enc, cfg.dt, iters, timer);
        timer.resolve(enc);
        device.queue.submit([enc.finish()]);
        // Awaiting the 8-byte×K timestamp map per frame is fine: GPU timestamps measure GPU
        // execution regardless of CPU serialization. The "no sync readback" rule targets large
        // FIELD readback in the production loop, not this measurement instrument.
        const t = await timer.readStages();

        if (frame >= cfg.warmup) {
          for (const k of STAGE_KEYS) perStageSamples[k].push(t.perStage[k] ?? 0);
          totalSamples.push(t.total);
          let s = 0;
          for (const k of STAGE_KEYS) s += t.perStage[k] ?? 0;
          sumSamples.push(s);
          lastPasses = t.passes;
        }
      }

      const perStageMedian: Record<string, number> = {};
      for (const k of STAGE_KEYS) perStageMedian[k] = median(perStageSamples[k]);
      const totalMedian = median(totalSamples);
      const sumOfStages = median(sumSamples);

      // One-shot residual readback (outside the timed loop) on the final velocity field.
      const u = await readField(fluid.velocityX, bytes);
      const v = await readField(fluid.velocityY, bytes);
      const residual = maxAbsDiv(u, v, w, h);

      cells.push({
        grid,
        iters,
        passesPerFrame: lastPasses,
        steadyFrames: cfg.steady,
        perStageMedianMs: perStageMedian,
        jacobiTotalMs: perStageMedian["jacobi"],
        totalMedianMs: totalMedian,
        sumOfStagesMs: sumOfStages,
        interPassOverheadMs: totalMedian - sumOfStages,
        residualMaxDiv: residual,
      });

      // Cross-check: re-run this config UNINSTRUMENTED (no per-pass timestampWrites) and time the
      // wall-clock submit→onSubmittedWorkDone with performance.now(). The instrumented TOTAL folds
      // in the timestamp barrier + 4 clearBuffer ops, so it is an UPPER bound on production cost;
      // if this uninstrumented wall time tracks (≤) the instrumented TOTAL, the headline number is
      // verified faithful rather than merely "probably". Same warm-median discipline.
      const wallSamples: number[] = [];
      for (let frame = 0; frame < total; frame++) {
        const enc = device.createCommandEncoder();
        fluid.step(enc, cfg.dt, iters); // untimed (no PassTimer)
        const t0 = performance.now();
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        const dt = performance.now() - t0;
        if (frame >= cfg.warmup) wallSamples.push(dt);
      }
      const wallMedian = median(wallSamples);

      cells[cells.length - 1].wallClockMedianMs = wallMedian;

      timer.destroy();
      fluid.destroy();
    }
  }

  return { adapterLabel, hasTimestampQuery, cells };
};

// ---- The test --------------------------------------------------------------------

test("fluid budget spike: warm-median per-stage ms + iters×grid sweep + residual → findings.json", async ({ page }) => {
  test.setTimeout(180_000); // 2 grids × 4 iters × (40 warm + 80 steady) frames + readbacks

  const pageErrors: string[] = [];
  page.on("pageerror", (e) => pageErrors.push(e.message));
  await page.goto("/");

  const out = await page.evaluate(RUN, {
    grids: GRIDS, iters: ITERS, warmup: WARMUP, steady: STEADY, dt: DT,
  });

  expect(out.hasTimestampQuery, "timestamp-query unavailable — budget ms would be NaN").toBe(true);
  expect(pageErrors, "page errors during budget run").toEqual([]);

  // Classify each (grid,iters) vs the platform fluid sub-budget. Apple Metal → M-series row.
  const platform = /apple|metal|m1|m2|m3|m-series/i.test(out.adapterLabel) ? "m-series" : "discrete";
  const band = SUBBUDGET[platform as keyof typeof SUBBUDGET];

  const classify = (ms: number): "PASS" | "MARGINAL" | "OVER" => {
    if (ms <= band.lo) return "PASS";
    if (ms <= band.hi) return "MARGINAL";
    return "OVER";
  };

  const results: CellResult[] = out.cells as CellResult[];
  const classified = results.map((c) => ({
    ...c,
    verdict: classify(c.totalMedianMs),
    // Per-layer ms == the single-layer measurement here; 2.5D N-layer projection is ×N_layers.
    perLayerMs: c.totalMedianMs,
  }));

  const findings = {
    task: "Plan 3 Task 4 — fluid budget spike (§8.1)",
    generatedAt: new Date().toISOString(),
    machine: {
      adapterLabel: out.adapterLabel,
      platformRow: platform,
      hasTimestampQuery: out.hasTimestampQuery,
    },
    method: {
      warmupFrames: WARMUP,
      steadyFrames: STEADY,
      dt: DT,
      reportedStatistic: "median over steady window",
      perStageMs: "SUM of all passes tagged that stage (stages interleaved/repeated)",
      totalMs: "max(end) − min(begin) over the frame's timed passes (one GPU timeline). UPPER bound: " +
        "folds in inter-pass gaps, the 4 clearBuffer ops, and the per-pass timestampWrites barrier " +
        "that production (uninstrumented) does NOT pay — so a config near the band edge that reads OVER " +
        "here may still fit in production. Cross-checked by wallClockMedianMs.",
      wallClockMedianMs: "uninstrumented re-run, performance.now() around submit→onSubmittedWorkDone " +
        "(no timestampWrites, no querySet). Tracks the instrumented TOTAL when faithful; the honest " +
        "production-cost estimate sits between sumOfStagesMs (lower) and totalMedianMs (upper).",
      interPassOverheadMs: "totalMedianMs − sumOfStagesMs: the architecture is pass-count-bound, not " +
        "compute-bound (this gap, not the kernel math, dominates). Includes clearBuffer + dispatch fixed cost.",
      residualMetric: "max|0.5(u[i+1,j]-u[i-1,j]) + 0.5(v[i,j+1]-v[i,j-1])| over interior (oracle max_abs_div)",
      stages: STAGES,
    },
    subBudget: {
      row: platform,
      loMs: band.lo,
      hiMs: band.hi,
      classification: "PASS ≤ lo, MARGINAL in [lo,hi] (fits only at the budget's edge), OVER > hi. " +
        "Conservative reading: the budget RANGE is the marginal zone, not slack PASS headroom.",
      note: "Compared vs the §3 fluid SUB-budget, NOT the 16.6ms whole frame. " +
        "Fluid IN ISOLATION: on M-series unified memory the concurrent Plan-4 render shares the bus, " +
        "so an isolated PASS is necessarily optimistic. Per-layer ms × N_layers gives the 2.5D total.",
    },
    sweep: classified,
  };

  writeFileSync(FINDINGS_PATH, JSON.stringify(findings, null, 2));

  // The verdict is DATA. This test asserts ONLY that the run produced finite numbers.
  expect(results.length, "no sweep cells produced").toBe(GRIDS.length * ITERS.length);
  for (const c of classified) {
    const tag = `grid=${c.grid} iters=${c.iters}`;
    expect(Number.isFinite(c.totalMedianMs), `${tag}: total ms not finite`).toBe(true);
    expect(Number.isFinite(c.jacobiTotalMs), `${tag}: jacobi-total ms not finite`).toBe(true);
    expect(Number.isFinite(c.wallClockMedianMs), `${tag}: wall-clock ms not finite`).toBe(true);
    expect(Number.isFinite(c.residualMaxDiv), `${tag}: residual not finite`).toBe(true);
    expect(c.totalMedianMs, `${tag}: total ms must be > 0`).toBeGreaterThan(0);
    expect(c.residualMaxDiv, `${tag}: residual must be ≥ 0`).toBeGreaterThanOrEqual(0);
    for (const s of STAGES) {
      expect(Number.isFinite(c.perStageMedianMs[s]), `${tag}: stage ${s} ms not finite`).toBe(true);
    }
  }
});
