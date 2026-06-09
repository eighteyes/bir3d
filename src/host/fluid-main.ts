// fluid-main.ts — live debug-viz bootstrap for the Plan-3 fluid spike ("I want to see something").
// Drives GpuFluid.step + a fullscreen-quad render pass each frame so the dye field visibly swirls,
// and reports the per-stage warm-median ms + the §3 fluid-sub-budget verdict in an overlay.
// Responsibilities:
//   - Acquire the device; build GpuFluid (compute) + the visualize.wgsl render pipeline; configure
//     the canvas context (explicit pixel size so headless layout never yields a 0x0 texture).
//   - Per frame: rotate the scripted jet/dye source (deterministic from frame INDEX) for a watchable
//     swirl, record fluid.step into a caller-owned encoder, then a render pass that REBUILDS the
//     render bind group from fluid.dyeField (the dye ping-pong swaps inside step → the current dye
//     buffer changes every frame; a once-built bind group would render a stale buffer).
//   - Sample the PassTimer on a cadence (pending-guard, like main.ts) for per-stage warm-median ms;
//     classify TOTAL vs the §3 fluid sub-budget (NOT 16.6ms); overlay adapter/grid/iters/ms/verdict/dt.
//   - One-shot, NON-blocking 1x1 center-pixel readback (copyTextureToBuffer in the SAME encoder,
//     mapAsync guarded) → window.__centerPixel for the viz test's non-blank assertion. This is a
//     single pixel, not the large field readback the risk register forbids in the loop.
//   - Set window.__fluidBooted=true once the loop is running.

import { acquireDevice } from "./gpu/device";
import { GpuFluid } from "./gpu/fluid";
import { PassTimer } from "./gpu/passtimer";
import { FrameLoop } from "./frameloop";

const CANVAS_PX = 512; // explicit backing size — headless layout can give 0x0 otherwise.
const GRID = 128; // window grid (single 2D layer), matching the budget spike's small grid.
const ITERS = 20; // Jacobi sweeps per project (budget-affordable; under-converged vs the oracle).
const DT = 0.1;

// Warm-median discipline (same as the budget spike): discard the first WARMUP timed samples,
// report the median over the steady window. Timer is sampled on a cadence, not every frame.
const TIMER_WARMUP = 8;
const TIMER_WINDOW = 30;
const SAMPLE_EVERY = 2; // sample the per-stage timer every Nth frame (readStages is async).

const STAGE_KEYS = ["forces", "set_bnd", "divergence", "jacobi", "subtract_grad", "advect"] as const;

// §3 fluid 2.5D moving-window sub-budget (ms). Apple Metal here → m-series row. PASS ≤ lo,
// MARGINAL in [lo,hi], OVER > hi (conservative: the band is the marginal zone, not slack).
const SUBBUDGET = {
  "m-series": { lo: 3.5, hi: 6.0 },
  discrete: { lo: 1.5, hi: 2.5 },
} as const;

const median = (xs: number[]): number => {
  if (xs.length === 0) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid]! : 0.5 * (s[mid - 1]! + s[mid]!);
};

// Worst-case passes per frame, for PassTimer capacity (mirrors the budget spec; never hardcoded).
const passesFor = (iters: number) => 30 + 6 * iters;

async function adapterLabelOf(adapter: GPUAdapter): Promise<string> {
  try {
    const info = (adapter as any).info ?? (await (adapter as any).requestAdapterInfo?.());
    if (info) {
      return (
        [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" / ") ||
        "unknown"
      );
    }
  } catch {
    /* adapter info unavailable */
  }
  return "unknown";
}

async function boot() {
  const overlay = document.getElementById("overlay")!;
  const canvas = document.getElementById("fluid") as HTMLCanvasElement;
  canvas.width = CANVAS_PX;
  canvas.height = CANVAS_PX;

  const { adapter, device, hasTimestampQuery } = await acquireDevice();
  device.lost.then((info) => {
    overlay.textContent = `WebGPU device lost: ${info.reason} — ${info.message}`;
    console.error("[WebGPU lost]", info.reason, info.message);
  });
  const adapterLabel = await adapterLabelOf(adapter);
  const platform = /apple|metal|m1|m2|m3|m-series/i.test(adapterLabel) ? "m-series" : "discrete";
  const band = SUBBUDGET[platform as keyof typeof SUBBUDGET];

  // Configure the canvas context. COPY_SRC enables the 1x1 center-pixel readback for the test.
  const ctx = canvas.getContext("webgpu") as GPUCanvasContext;
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({
    device,
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    alphaMode: "opaque",
  });

  // Load shaders (compute kernels + the render viz) from the dev server.
  const get = (f: string) => fetch(`/src/host/shaders/fluid/${f}.wgsl`).then((r) => r.text());
  const [advect, divergence, jacobi, subtractGrad, setBnd, forces, visualize] = await Promise.all([
    get("advect"), get("divergence"), get("jacobi"), get("subtract_grad"),
    get("set_bnd"), get("forces"), get("visualize"),
  ]);
  const fluid = new GpuFluid(device, GRID, GRID, { forces, divergence, jacobi, subtractGrad, advect, setBnd });

  // Render pipeline: fullscreen triangle (visualize.wgsl vs/fs), no vertex buffer.
  const vizModule = device.createShaderModule({ code: visualize });
  const renderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: vizModule, entryPoint: "vs" },
    fragment: { module: vizModule, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  // 1x1 center-pixel readback staging. copyTextureToBuffer needs bytesPerRow a multiple of 256.
  const PIXEL_ROW_BYTES = 256;
  const pixelBuf = device.createBuffer({
    size: PIXEL_ROW_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let pixelPending = false;

  const timer = new PassTimer(device, hasTimestampQuery, passesFor(ITERS));
  const perStageSamples: Record<string, number[]> = {};
  for (const k of STAGE_KEYS) perStageSamples[k] = [];
  const totalSamples: number[] = [];
  let timedFrames = 0;
  let timerPending = false;
  let lastFluidMs = NaN; // most-recent single TOTAL sample (shows a number within a few frames).
  let warmMedianTotal = NaN;
  const warmPerStage: Record<string, number> = {};

  let frame = 0;

  const loop = new FrameLoop((dt) => {
    // Deterministic swirl: rotate the jet vector by frame INDEX (not wall time) so the field
    // evolves the same way every run. Dye stays injected at center so the center pixel lights up.
    const theta = frame * 0.08;
    const amp = 45;
    const cx = (GRID + 2) / 2;
    const cy = (GRID + 2) / 2;
    fluid.setForce({
      fx: amp * Math.cos(theta),
      fy: amp * Math.sin(theta),
      dyeX: cx, dyeY: cy,
      dyeR: GRID / 7, dyeAmt: 70,
      forceR: GRID / 5,
    });

    const sampling = hasTimestampQuery && !timerPending && frame % SAMPLE_EVERY === 0;
    if (sampling) timer.reset();

    const enc = device.createCommandEncoder();
    fluid.step(enc, DT, ITERS, sampling ? timer : null);

    // Render pass — REBUILD the bind group from fluid.dyeField each frame (the dye ping-pong
    // swapped inside step(), so the "current" dye buffer differs frame to frame).
    const view = ctx.getCurrentTexture().createView();
    const renderBg = device.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: fluid.paramsBuffer } },
        { binding: 1, resource: { buffer: fluid.dyeField } },
      ],
    });
    const rp = enc.beginRenderPass({
      colorAttachments: [
        { view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
      ],
    });
    rp.setPipeline(renderPipeline);
    rp.setBindGroup(0, renderBg);
    rp.draw(3);
    rp.end();

    // Center-pixel readback: copy the 1x1 center of THIS frame's canvas texture (same encoder).
    const readPixel = !pixelPending;
    if (readPixel) {
      enc.copyTextureToBuffer(
        { texture: ctx.getCurrentTexture(), origin: { x: CANVAS_PX >> 1, y: CANVAS_PX >> 1 } },
        { buffer: pixelBuf, bytesPerRow: PIXEL_ROW_BYTES },
        { width: 1, height: 1, depthOrArrayLayers: 1 }
      );
    }

    if (sampling) timer.resolve(enc);
    device.queue.submit([enc.finish()]);

    if (sampling) {
      timerPending = true;
      timer.readStages()
        .then((t) => {
          timedFrames++;
          if (timedFrames > TIMER_WARMUP && totalSamples.length < TIMER_WINDOW) {
            for (const k of STAGE_KEYS) perStageSamples[k]!.push(t.perStage[k] ?? 0);
            totalSamples.push(t.total);
            for (const k of STAGE_KEYS) warmPerStage[k] = median(perStageSamples[k]!);
            warmMedianTotal = median(totalSamples);
          }
          lastFluidMs = t.total;
          timerPending = false;
        })
        .catch(() => { timerPending = false; });
    }

    if (readPixel) {
      pixelPending = true;
      pixelBuf.mapAsync(GPUMapMode.READ, 0, 4)
        .then(() => {
          const px = new Uint8Array(pixelBuf.getMappedRange(0, 4).slice(0));
          pixelBuf.unmap();
          // green is byte index 1 in BOTH rgba8 and bgra8 → format-agnostic.
          (window as any).__centerPixel = [px[0], px[1], px[2], px[3]];
          pixelPending = false;
        })
        .catch(() => { pixelPending = false; });
    }

    frame++;

    // Overlay: last-sample fluid ms shows within a few frames; warm-median + verdict once ready.
    const verdict = !Number.isFinite(warmMedianTotal)
      ? "warming…"
      : warmMedianTotal <= band.lo ? "PASS" : warmMedianTotal <= band.hi ? "MARGINAL" : "OVER";
    const fluidLine = Number.isFinite(lastFluidMs)
      ? `fluid: ${lastFluidMs.toFixed(3)} ms (last)`
      : hasTimestampQuery ? "fluid: warming…" : "fluid: n/a (no timestamp-query)";
    const stageLine = STAGE_KEYS.map((k) => {
      const v = warmPerStage[k];
      return `${k} ${v !== undefined && Number.isFinite(v) ? v.toFixed(3) : "—"}`;
    }).join("  ");
    overlay.textContent =
      `vector-system fluid debug viz\n` +
      `adapter: ${adapterLabel}\n` +
      `grid: ${GRID}² (+2 border)   iters: ${ITERS}\n` +
      `${fluidLine}\n` +
      `fluid warm-median: ${Number.isFinite(warmMedianTotal) ? warmMedianTotal.toFixed(3) + " ms" : "…"}` +
      `   §3 sub-budget [${band.lo}-${band.hi}ms ${platform}]: ${verdict}\n` +
      `per-stage ms: ${stageLine}\n` +
      `cpu dt: ${(dt * 1000).toFixed(2)} ms`;
  });

  loop.start();
  (window as any).__fluidBooted = true; // test signal
}

boot().catch((e) => {
  const overlay = document.getElementById("overlay");
  if (overlay) overlay.textContent = "boot error: " + (e as Error).message;
  throw e;
});
