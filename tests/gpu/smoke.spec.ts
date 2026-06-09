import { expect, test } from "@playwright/test";

test("acquires a WebGPU device and reports timestamp-query support", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const { acquireDevice } = await import("/src/host/gpu/device.ts");
    const ctx = await acquireDevice();
    return { ok: !!ctx.device, hasTimestamp: ctx.hasTimestampQuery };
  });
  expect(result.ok).toBe(true);
  expect(typeof result.hasTimestamp).toBe("boolean");
});

test("profiler reports a plausible positive ms for a compute pass (when supported)", async ({ page }) => {
  await page.goto("/");
  const ms = await page.evaluate(async () => {
    const { acquireDevice } = await import("/src/host/gpu/device.ts");
    const { makeComputePipeline } = await import("/src/host/gpu/dispatch.ts");
    const { GpuProfiler } = await import("/src/host/gpu/profiler.ts");
    const addone = await (await fetch("/src/host/shaders/addone.wgsl")).text();
    const { device, hasTimestampQuery } = await acquireDevice();
    if (!hasTimestampQuery) return 0; // pass trivially where unsupported
    const n = 1 << 16;
    const buf = (usage: number) => device.createBuffer({ size: n * 4, usage });
    const inBuf = buf(GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const outBuf = buf(GPUBufferUsage.STORAGE);
    const pipeline = makeComputePipeline(device, addone);
    const prof = new GpuProfiler(device, true);
    const enc = device.createCommandEncoder();
    const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }] });
    const pass = enc.beginComputePass({ timestampWrites: prof.timestampWrites() });
    pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(Math.ceil(n / 64)); pass.end();
    prof.resolve(enc);
    device.queue.submit([enc.finish()]);
    return prof.readMs();
  });
  expect(ms).toBeGreaterThanOrEqual(0);
  expect(ms).toBeLessThan(100);
});

test("app boots, runs a frame loop, and shows a ms readout without errors", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page.waitForFunction(() => (window as any).__vsBooted === true, { timeout: 10000 });
  await page.waitForTimeout(800); // let several frames + a sample run
  const text = await page.locator("#overlay").innerText();
  expect(text).toContain("gpu addone:");
  expect(errors).toEqual([]);
});

test("add-one compute kernel maps [1,2,3] -> [2,3,4]", async ({ page }) => {
  await page.goto("/");
  const out = await page.evaluate(async () => {
    const { acquireDevice } = await import("/src/host/gpu/device.ts");
    const { makeComputePipeline, dispatchCompute } = await import("/src/host/gpu/dispatch.ts");
    const addone = await (await fetch("/src/host/shaders/addone.wgsl")).text();
    const { device } = await acquireDevice();
    const data = new Float32Array([1, 2, 3]);
    const inBuf = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(inBuf, 0, data);
    const outBuf = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const pipeline = makeComputePipeline(device, addone);
    dispatchCompute(device, pipeline, [inBuf, outBuf], data.length);
    const staging = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(outBuf, 0, staging, 0, data.byteLength);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    return Array.from(new Float32Array(staging.getMappedRange().slice(0)));
  });
  expect(out).toEqual([2, 3, 4]);
});
