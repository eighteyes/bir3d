// bird-main.ts — minimal flyable-bird bootstrap: reuse GpuFluid for wind, drive the GPU bird, render.
// Vertical slice for /index-bird.html. Control scheme 1 only (flick-to-impulse). No goal, no score.
// Responsibilities:
//   - Acquire device; build GpuFluid (sim grid > viewport) + Bird; configure the canvas context.
//   - Seed several drifting gusts across the world (cycled seed positions) so there is always field.
//   - Per frame (one encoder, one submit): fluid.step → bird.step → bird.render; resolve async readback.
//   - Scheme 1 input: mouse drag→release = a length-capped velocity-burst impulse in the drag
//     direction; draw an aim indicator (DOM) while dragging. Impulse is one-shot.
//   - Overlay: active scheme + cpu dt + bird pos. Set window.__birdBooted / __birdPos / __birdVel.

import { acquireDevice } from "./gpu/device";
import { GpuFluid } from "./gpu/fluid";
import { Bird } from "./gpu/bird";
import { FrameLoop } from "./frameloop";

const CANVAS_PX = 768;
const SIM = 256; // sim grid (toroidal world), larger than the viewport
const VIEW = 128; // view spans ~half the world in grid units
const ITERS = 16; // Jacobi sweeps per project (feel proto; budget-affordable)
const DT = 0.1;

// Several gusts seeded across the world: each is a localized force+dye source. We cycle the active
// source position every frame (the fluid's setForce is a single localized source) so over a few
// frames the whole world gets stirred and dyed — always new field to glide into.
const GUSTS = [
  { x: 0.22, y: 0.25, ang: 0.0 },
  { x: 0.72, y: 0.30, ang: 2.1 },
  { x: 0.30, y: 0.72, ang: 4.0 },
  { x: 0.78, y: 0.74, ang: 1.0 },
  { x: 0.5, y: 0.5, ang: 3.1 },
];

async function boot() {
  const overlay = document.getElementById("overlay")!;
  const aim = document.getElementById("aim") as HTMLDivElement;
  const canvas = document.getElementById("bird") as HTMLCanvasElement;
  canvas.width = CANVAS_PX;
  canvas.height = CANVAS_PX;

  const { device } = await acquireDevice();
  device.lost.then((info) => {
    overlay.textContent = `WebGPU device lost: ${info.reason} — ${info.message}`;
    console.error("[WebGPU lost]", info.reason, info.message);
  });

  const ctx = canvas.getContext("webgpu") as GPUCanvasContext;
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({
    device,
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    alphaMode: "opaque",
  });

  // Load shaders from the dev server.
  const fget = (f: string) => fetch(`/src/host/shaders/fluid/${f}.wgsl`).then((r) => r.text());
  const bget = (f: string) => fetch(`/src/host/shaders/bird/${f}.wgsl`).then((r) => r.text());
  const [advect, divergence, jacobi, subtractGrad, setBnd, forces, birdUpdate, scene] =
    await Promise.all([
      fget("advect"), fget("divergence"), fget("jacobi"), fget("subtract_grad"),
      fget("set_bnd"), fget("forces"), bget("bird_update"), bget("scene"),
    ]);

  const fluid = new GpuFluid(device, SIM, SIM, { forces, divergence, jacobi, subtractGrad, advect, setBnd });

  const bird = new Bird(device, fluid, birdUpdate, scene, format, {
    viewSize: [VIEW, VIEW],
    startPos: [SIM / 2, SIM / 2],
    deadzone: VIEW * 0.15,
    followStiffness: 0.22,
    // Fluid velocities are large; keep coupling low and drag firm so the bird glides, not rockets.
    tuning: { windCoupling: 0.12, drag: 0.96, flickStrength: 14 },
  });

  // Pre-warm: stir the field with a few steps so the backdrop is alive at boot.
  for (let i = 0; i < 24; i++) {
    const gu = GUSTS[i % GUSTS.length]!;
    const amp = 60;
    fluid.setForce({
      fx: amp * Math.cos(gu.ang), fy: amp * Math.sin(gu.ang),
      dyeX: gu.x * SIM, dyeY: gu.y * SIM,
      dyeR: SIM / 9, dyeAmt: 90, forceR: SIM / 6,
    });
    const enc = device.createCommandEncoder();
    fluid.step(enc, DT, ITERS);
    device.queue.submit([enc.finish()]);
  }

  // ---- Scheme 1 input: flick-to-impulse ----
  let dragging = false;
  let dragStart: [number, number] = [0, 0];
  let dragCur: [number, number] = [0, 0];
  let pendingImpulse: [number, number] | null = null;

  const setAim = (x0: number, y0: number, x1: number, y1: number, on: boolean) => {
    if (!on) { aim.style.display = "none"; return; }
    const dx = x1 - x0, dy = y1 - y0;
    const len = Math.hypot(dx, dy);
    const ang = (Math.atan2(dy, dx) * 180) / Math.PI;
    aim.style.display = "block";
    aim.style.left = `${x0}px`;
    aim.style.top = `${y0}px`;
    aim.style.width = `${len}px`;
    aim.style.transform = `rotate(${ang}deg)`;
  };

  canvas.addEventListener("mousedown", (e) => {
    dragging = true;
    dragStart = [e.clientX, e.clientY];
    dragCur = [e.clientX, e.clientY];
  });
  canvas.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    dragCur = [e.clientX, e.clientY];
    setAim(dragStart[0], dragStart[1], dragCur[0], dragCur[1], true);
  });
  const endDrag = (e: MouseEvent) => {
    if (!dragging) return;
    dragging = false;
    setAim(0, 0, 0, 0, false);
    // Screen drag → world drag. Flick pushes the bird in the drag direction.
    // Scale screen pixels to world units by the view span; flip y (screen down = world down inverted).
    const sx = (dragCur[0] - dragStart[0]) / CANVAS_PX * VIEW;
    const sy = -((dragCur[1] - dragStart[1]) / CANVAS_PX) * VIEW;
    pendingImpulse = [sx, sy];
  };
  canvas.addEventListener("mouseup", endDrag);
  window.addEventListener("mouseup", endDrag);

  let frame = 0;
  const loop = new FrameLoop((dt) => {
    // Keep the world stirred: cycle the active gust each frame.
    const gu = GUSTS[frame % GUSTS.length]!;
    const amp = 55;
    const drift = 0.6 * Math.sin(frame * 0.01);
    fluid.setForce({
      fx: amp * Math.cos(gu.ang + drift), fy: amp * Math.sin(gu.ang + drift),
      dyeX: gu.x * SIM, dyeY: gu.y * SIM,
      dyeR: SIM / 9, dyeAmt: 55, forceR: SIM / 6,
    });

    // Build the per-frame intent. Scheme 1: a one-shot flick impulse, else zero.
    let intent = { impulse: [0, 0] as [number, number], turn: 0 };
    if (pendingImpulse) {
      intent = bird.flickToIntent(pendingImpulse);
      pendingImpulse = null;
    }

    const enc = device.createCommandEncoder();
    fluid.step(enc, DT, ITERS);          // wind (u,v ping-pong swaps here)
    bird.step(enc, DT, intent);          // GPU bird reads CURRENT u,v in-shader
    const view = ctx.getCurrentTexture().createView();
    bird.render(enc, view);              // backdrop + trail + chevron
    device.queue.submit([enc.finish()]);
    bird.resolveReadback();              // async pos → camera + verification

    (window as any).__birdPos = bird.lastPos;
    (window as any).__birdVel = bird.lastVel;

    frame++;
    overlay.textContent =
      `vector-system — bird (feel proto)\n` +
      `scheme: 1 — flick to impulse (drag→release)\n` +
      `sim ${SIM}²  view ${VIEW}  iters ${ITERS}\n` +
      `cpu dt: ${(dt * 1000).toFixed(2)} ms\n` +
      `bird pos: ${bird.lastPos[0].toFixed(1)}, ${bird.lastPos[1].toFixed(1)}  ` +
      `vel: ${bird.lastVel[0].toFixed(2)}, ${bird.lastVel[1].toFixed(2)}`;
  });

  loop.start();
  (window as any).__birdBooted = true;
}

boot().catch((e) => {
  const overlay = document.getElementById("overlay");
  if (overlay) overlay.textContent = "boot error: " + (e as Error).message;
  throw e;
});
