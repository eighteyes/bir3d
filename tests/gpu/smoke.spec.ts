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
