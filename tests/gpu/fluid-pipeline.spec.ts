// fluid-pipeline.spec.ts — Task 2 smoke: every fluid WGSL kernel compiles and every GpuFluid
// pipeline builds + records a step with zero WebGPU validation errors (compute only; no readback).
// Responsibilities:
//   - Per-shader getCompilationInfo() check so a compile failure localizes to a filename.
//   - Build GpuFluid (all pipelines + buffers) under one validation error scope -> expect null.
//   - Record + submit ONE step(encoder, dt, iters) under a validation scope; onSubmittedWorkDone
//     before popErrorScope. No buffer readback (risk-register hard rule). page errors must stay empty.

import { expect, test } from "@playwright/test";

const SHADER_FILES = [
  "advect",
  "divergence",
  "jacobi",
  "subtract_grad",
  "set_bnd",
  "forces",
] as const;

test("every fluid WGSL kernel compiles with no error-severity diagnostics", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async (files) => {
    const { acquireDevice } = await import("/src/host/gpu/device.ts");
    const { device } = await acquireDevice();
    const out: { file: string; errors: string[] }[] = [];
    for (const f of files) {
      const code = await (await fetch(`/src/host/shaders/fluid/${f}.wgsl`)).text();
      const module = device.createShaderModule({ code });
      const info = await module.getCompilationInfo();
      const errors = info.messages
        .filter((m) => m.type === "error")
        .map((m) => `${m.lineNum}:${m.linePos} ${m.message}`);
      out.push({ file: f, errors });
    }
    return out;
  }, SHADER_FILES);

  for (const { file, errors } of result) {
    expect(errors, `${file}.wgsl compile errors`).toEqual([]);
  }
});

test("GpuFluid builds all pipelines + records & submits one step with no validation error", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (e) => pageErrors.push(e.message));
  await page.goto("/");

  const result = await page.evaluate(async (files) => {
    const { acquireDevice } = await import("/src/host/gpu/device.ts");
    const { GpuFluid } = await import("/src/host/gpu/fluid.ts");
    const { device } = await acquireDevice();

    const get = (f: string) => fetch(`/src/host/shaders/fluid/${f}.wgsl`).then((r) => r.text());
    const [advect, divergence, jacobi, subtractGrad, setBnd, forces] = await Promise.all([
      get("advect"), get("divergence"), get("jacobi"), get("subtract_grad"), get("set_bnd"), get("forces"),
    ]);
    const shaders = { forces, divergence, jacobi, subtractGrad, advect, setBnd };

    // (1) Construction (all pipelines + buffers) under a validation scope.
    device.pushErrorScope("validation");
    const fluid = new GpuFluid(device, 64, 64, shaders);
    fluid.setForce({ fx: 1, fy: 0, dyeX: 32, dyeY: 32, dyeR: 6, dyeAmt: 50, forceR: 0 });
    const buildErr = await device.popErrorScope();

    // (2) Record + submit one full step (exercises every bindgroup<->layout contract). No readback.
    device.pushErrorScope("validation");
    const encoder = device.createCommandEncoder();
    fluid.step(encoder, 0.1, 4);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const stepErr = await device.popErrorScope();

    fluid.destroy();
    return { buildErr: buildErr ? String(buildErr.message) : null, stepErr: stepErr ? String(stepErr.message) : null };
  }, SHADER_FILES);

  expect(result.buildErr, "GpuFluid construction validation error").toBeNull();
  expect(result.stepErr, "GpuFluid.step validation error").toBeNull();
  expect(pageErrors).toEqual([]);
});
