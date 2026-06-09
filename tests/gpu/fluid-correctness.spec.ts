// fluid-correctness.spec.ts — Task 3: GPU≈CPU correctness gate (THE port-trust gate).
// Loads each CPU-oracle fixture (tests/fixtures/fluid/*.json), uploads its inputs to GPU
// buffers, runs the SINGLE corresponding WGSL kernel ONCE (per-kernel) or the full composed
// Fluid2D::step x N (composed), reads back ONE-SHOT, and asserts the GPU result matches the
// oracle cell-for-cell over the FULL (W+2)*(H+2) buffer.
// Responsibilities:
//   - Per-kernel fixtures (advect/divergence/jacobi/subtract_grad/set_bnd x3): max|GPU-CPU| < 1e-5.
//     The tight tol means any failure is a REAL port bug (sign/index/bilinear/boundary), not f32 drift.
//   - Composed N-step fixture: max|GPU-CPU| < 1e-3 (accumulated f32 drift tol).
//   - On mismatch, report the argmax cell (i,j) so a structured 4-corner set_bnd mismatch (stale
//     edges read by a non-two-pass corner step) is legible, not hidden under a bare max-norm.
//   - Compare the FULL buffer (border included): a kernel that wrongly writes the border fails here.
//   - Fixtures read Node-side via fs; GPU outputs returned to Node, which owns the assertion math.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { expect, test } from "@playwright/test";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIX_DIR = join(HERE, "..", "fixtures", "fluid");

function loadFixture(name: string): any {
  return JSON.parse(readFileSync(join(FIX_DIR, `${name}.json`), "utf8"));
}

const PER_KERNEL_TOL = 1e-5;
const COMPOSED_TOL = 1e-3;

/** Max |gpu-cpu| over the full (w+2)*(h+2) buffer + the (i,j) where it occurs. */
function maxAbsDiff(gpu: number[], cpu: number[], w: number) {
  expect(gpu.length, "gpu/cpu length mismatch").toBe(cpu.length);
  const stride = w + 2;
  let max = 0;
  let at = 0;
  for (let k = 0; k < cpu.length; k++) {
    const d = Math.abs(gpu[k] - cpu[k]);
    if (d > max) {
      max = d;
      at = k;
    }
  }
  return { max, k: at, i: at % stride, j: Math.floor(at / stride) };
}

function assertClose(name: string, gpu: number[], cpu: number[], w: number, tol: number) {
  const d = maxAbsDiff(gpu, cpu, w);
  expect(
    d.max,
    `${name}: max|GPU-CPU|=${d.max.toExponential(3)} at cell (i=${d.i}, j=${d.j}) [idx ${d.k}] ` +
      `gpu=${gpu[d.k]} cpu=${cpu[d.k]} (tol ${tol.toExponential(0)})`
  ).toBeLessThan(tol);
}

// ---- Shared browser-side runner -------------------------------------------------
// A single page.evaluate entry point that builds the raw pipelines, dispatches the
// requested kernel(s), and returns the full output buffer(s). Kept as a string-injected
// closure so each test passes only its fixture + a kernel selector.

const RUN = async (job: { kernel: string; fx: any }) => {
  const { acquireDevice } = await import("/src/host/gpu/device.ts");
  const SHADERS = [
    "advect",
    "divergence",
    "jacobi",
    "subtract_grad",
    "set_bnd",
    "add_force_field",
  ] as const;
  const src: Record<string, string> = {};
  await Promise.all(
    SHADERS.map(async (s) => {
      src[s] = await (await fetch(`/src/host/shaders/fluid/${s}.wgsl`)).text();
    })
  );

  const { device } = await acquireDevice();

  const pipe = (code: string, entry = "main") =>
    device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code }), entryPoint: entry },
    });

  const P = {
    advect: pipe(src.advect),
    divergence: pipe(src.divergence),
    jacobi: pipe(src.jacobi),
    subtractGrad: pipe(src.subtract_grad),
    addForce: pipe(src.add_force_field),
    scalarEdges: pipe(src.set_bnd, "scalar_edges"),
    scalarCorners: pipe(src.set_bnd, "scalar_corners"),
    velxEdges: pipe(src.set_bnd, "velx_edges"),
    velxCorners: pipe(src.set_bnd, "velx_corners"),
    velyEdges: pipe(src.set_bnd, "vely_edges"),
    velyCorners: pipe(src.set_bnd, "vely_corners"),
  };

  const fx = job.fx;
  const w: number = fx.w;
  const h: number = fx.h;
  const cells = (w + 2) * (h + 2);
  const bytes = cells * 4;

  // Params uniform — identical layout to GpuFluid (w,h u32; dt f32 at [2]; forces unused here).
  const PARAMS_FLOATS = 12;
  const paramsHost = new ArrayBuffer(PARAMS_FLOATS * 4);
  const pu32 = new Uint32Array(paramsHost);
  const pf32 = new Float32Array(paramsHost);
  pu32[0] = w;
  pu32[1] = h;
  pf32[2] = fx.params?.dt ?? 0;
  const params = device.createBuffer({
    size: paramsHost.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(params, 0, paramsHost);

  // Storage buffer factory. Fresh buffers are zero-initialised by WebGPU, matching the
  // oracle's Grid2D::new (zero border) for output/scratch grids.
  const storage = (init?: number[]) => {
    const b = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    if (init) device.queue.writeBuffer(b, 0, new Float32Array(init));
    return b;
  };

  const enc = device.createCommandEncoder();
  const pass = (pipeline: GPUComputePipeline, bindings: GPUBuffer[]) => {
    const bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const p = enc.beginComputePass();
    p.setPipeline(pipeline);
    p.setBindGroup(0, bg);
    p.dispatchWorkgroups(Math.ceil(cells / 64));
    p.end();
  };

  // set_bnd as TWO passes (edges then corners) — corners read freshly-written edges.
  const setBnd = (kind: "scalar" | "velx" | "vely", g: GPUBuffer) => {
    if (kind === "scalar") {
      pass(P.scalarEdges, [params, g]);
      pass(P.scalarCorners, [params, g]);
    } else if (kind === "velx") {
      pass(P.velxEdges, [params, g]);
      pass(P.velxCorners, [params, g]);
    } else {
      pass(P.velyEdges, [params, g]);
      pass(P.velyCorners, [params, g]);
    }
  };

  // Output buffers to read back, keyed by the fixture's expected member name.
  const outputs: Record<string, GPUBuffer> = {};

  switch (job.kernel) {
    case "advect": {
      const field = storage(fx.inputs.field);
      const u = storage(fx.inputs.u);
      const v = storage(fx.inputs.v);
      const dst = storage(); // fresh => zero border, matching oracle Grid2D::new
      pass(P.advect, [params, field, u, v, dst]);
      outputs.field = dst;
      break;
    }
    case "divergence": {
      const u = storage(fx.inputs.u);
      const v = storage(fx.inputs.v);
      const div = storage();
      pass(P.divergence, [params, u, v, div]);
      outputs.div = div;
      break;
    }
    case "jacobi": {
      const p = storage(fx.inputs.p);
      const div = storage(fx.inputs.div);
      const pNext = storage();
      pass(P.jacobi, [params, p, div, pNext]);
      outputs.p = pNext;
      break;
    }
    case "subtract_grad": {
      const p = storage(fx.inputs.p);
      const u = storage(fx.inputs.u);
      const v = storage(fx.inputs.v);
      pass(P.subtractGrad, [params, p, u, v]); // in-place on u,v
      outputs.u = u;
      outputs.v = v;
      break;
    }
    case "set_bnd_scalar":
    case "set_bnd_velx":
    case "set_bnd_vely": {
      const kind: "scalar" | "velx" | "vely" = fx.params.kind;
      const g = storage(fx.inputs.field);
      setBnd(kind, g);
      outputs.field = g;
      break;
    }
    case "composed_step": {
      // Mirror Fluid2D::step x steps EXACTLY (solver.rs). Per-cell force add (add_force_field),
      // project (divergence -> [jacobi -> set_bnd scalar] x iters -> subtract_grad -> set_bnd vel),
      // self-advect u,v through pre-advection clones, set_bnd vel, project again. Scalar s is NOT
      // moved by the 2D step, so it is read straight from its input buffer.
      const iters: number = fx.params.iters;
      const steps: number = fx.params.steps;
      const fxBuf = storage(fx.inputs.force_x);
      const fyBuf = storage(fx.inputs.force_y);

      // Velocity ping-pong (advect needs separate src/dst), pressure ping-pong, divergence scratch.
      let uCur = storage(fx.inputs.u);
      let uNxt = storage();
      let vCur = storage(fx.inputs.v);
      let vNxt = storage();
      const div = storage();
      let pCur = storage();
      let pNxt = storage();

      const project = () => {
        pass(P.divergence, [params, uCur, vCur, div]);
        // zero pressure guess
        enc.clearBuffer(pCur);
        enc.clearBuffer(pNxt);
        for (let it = 0; it < iters; it++) {
          pass(P.jacobi, [params, pCur, div, pNxt]);
          [pCur, pNxt] = [pNxt, pCur];
          setBnd("scalar", pCur);
        }
        pass(P.subtractGrad, [params, pCur, uCur, vCur]);
        setBnd("velx", uCur);
        setBnd("vely", vCur);
      };

      for (let s = 0; s < steps; s++) {
        // 1. add per-cell force (in-place on uCur,vCur).
        pass(P.addForce, [params, fxBuf, fyBuf, uCur, vCur]);
        // 2. project.
        project();
        // 3. self-advect: both components read the SAME pre-advection snapshot (uCur,vCur).
        pass(P.advect, [params, uCur, uCur, vCur, uNxt]);
        pass(P.advect, [params, vCur, uCur, vCur, vNxt]);
        [uCur, uNxt] = [uNxt, uCur];
        [vCur, vNxt] = [vNxt, vCur];
        // 4. set_bnd advected velocities.
        setBnd("velx", uCur);
        setBnd("vely", vCur);
        // 5. final project.
        project();
      }

      // Scalar s is NOT advected by the 2D step (solver.rs): expected s == input s. Round-trip
      // it through a GPU buffer so the assertion still covers it cell-for-cell (and would catch
      // any accidental write to s if the composed sequence ever touched it).
      const sBuf = storage(fx.inputs.s);

      outputs.u = uCur;
      outputs.v = vCur;
      outputs.s = sBuf;
      break;
    }
    default:
      throw new Error(`unknown kernel ${job.kernel}`);
  }

  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();

  // One-shot readback (OUTSIDE any loop): copy each output to a MAP_READ staging buffer.
  const read = async (buf: GPUBuffer): Promise<number[]> => {
    const staging = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const e = device.createCommandEncoder();
    e.copyBufferToBuffer(buf, 0, staging, 0, bytes);
    device.queue.submit([e.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = Array.from(new Float32Array(staging.getMappedRange().slice(0)));
    staging.unmap();
    staging.destroy();
    return out;
  };

  const result: Record<string, number[]> = {};
  for (const key of Object.keys(outputs)) {
    result[key] = await read(outputs[key]);
  }
  return result;
};

// ---- Per-kernel tests (@ 1e-5) --------------------------------------------------

const PER_KERNEL = [
  "advect",
  "divergence",
  "jacobi",
  "subtract_grad",
  "set_bnd_scalar",
  "set_bnd_velx",
  "set_bnd_vely",
] as const;

for (const name of PER_KERNEL) {
  test(`${name}: GPU matches CPU oracle within ${PER_KERNEL_TOL.toExponential(0)}`, async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto("/");
    const fx = loadFixture(name);
    const out = await page.evaluate(RUN, { kernel: name, fx });
    for (const key of Object.keys(fx.expected)) {
      assertClose(`${name}.${key}`, out[key], fx.expected[key], fx.w, PER_KERNEL_TOL);
    }
    expect(errors, "page errors during GPU run").toEqual([]);
  });
}

// ---- Composed N-step test (@ 1e-3) ----------------------------------------------

test(`composed_step: GPU Fluid2D::step x N matches CPU oracle within ${COMPOSED_TOL.toExponential(0)}`, async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  const fx = loadFixture("composed_step");
  const out = await page.evaluate(RUN, { kernel: "composed_step", fx });
  for (const key of Object.keys(fx.expected)) {
    assertClose(`composed_step.${key}`, out[key], fx.expected[key], fx.w, COMPOSED_TOL);
  }
  expect(errors, "page errors during GPU run").toEqual([]);
});
