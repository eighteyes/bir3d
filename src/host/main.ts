// main.ts — bootstrap: acquire device, run the add-one pass each frame, show per-pass ms.
// Proves the whole foundation runs in a live frame loop and the instrument reports numbers.
import { acquireDevice } from "./gpu/device";
import { makeComputePipeline } from "./gpu/dispatch";
import { GpuProfiler } from "./gpu/profiler";
import { FrameLoop } from "./frameloop";

const overlay = document.getElementById("overlay")!;
const SAMPLE_EVERY = 30; // profile once per ~half-second; readMs is async and must not run every frame

async function boot() {
  const { device, hasTimestampQuery } = await acquireDevice();
  device.lost.then((info) => {
    const msg = `WebGPU device lost: ${info.reason} — ${info.message}`;
    overlay.textContent = msg;
    console.error("[WebGPU lost]", info.reason, info.message);
  });

  const code = await (await fetch("/src/host/shaders/addone.wgsl")).text();
  const pipeline = makeComputePipeline(device, code);
  const prof = new GpuProfiler(device, hasTimestampQuery);

  const n = 1 << 16;
  const inBuf = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outBuf = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE });
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }],
  });

  let gpuMs = NaN;
  let frame = 0;
  let profilePending = false;

  const loop = new FrameLoop((dt) => {
    // Only sample timing on a sampling frame: resolve() writes readBuf and readMs() maps it,
    // so doing both every frame races the single staging buffer. Guard with profilePending.
    const sampling = hasTimestampQuery && !profilePending && frame % SAMPLE_EVERY === 0;

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass(sampling ? { timestampWrites: prof.timestampWrites() } : {});
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
    pass.end();
    if (sampling) prof.resolve(enc);
    device.queue.submit([enc.finish()]);

    if (sampling) {
      profilePending = true;
      prof.readMs()
        .then((ms) => { gpuMs = ms; profilePending = false; })
        .catch(() => { profilePending = false; });
    }

    frame++;
    overlay.textContent =
      `vector-system foundation\n` +
      `cpu dt: ${(dt * 1000).toFixed(2)} ms\n` +
      `gpu addone: ${Number.isNaN(gpuMs) ? "n/a (no timestamp-query)" : gpuMs.toFixed(3) + " ms"}`;
  });
  loop.start();
  (window as any).__vsBooted = true; // test signal
}

boot().catch((e) => { overlay.textContent = "boot error: " + (e as Error).message; throw e; });
