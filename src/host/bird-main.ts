// bird-main.ts — flyable-bird bootstrap: reuse GpuFluid for wind, drive the GPU bird, render.
// Vertical slice for /index-bird.html. 4 switchable control schemes (the taste test). No goal/score.
// Responsibilities:
//   - Acquire device; build GpuFluid (sim grid > viewport) + Bird; configure the canvas context.
//   - Seed several drifting gusts across the world (cycled seed positions) so there is always field.
//   - Per frame (one encoder, one submit): fluid.step → bird.step → bird.render; resolve async readback.
//   - Input: number keys 1-4 swap the active scheme (the GPU sim is scheme-agnostic; only the CPU
//     mapping → (impulse,turn) intent differs). 1 flick-to-impulse, 2 hold-toward-cursor,
//     3 tap-to-bank (arrows), 4 flap-forward (space/click). Mouse dispatch branches on active scheme.
//   - Tuning overlay: live HTML sliders (windCoupling, drag, flick/thrust/flap strength, turn/bank
//     rate, deadzone, follow stiffness) feed the bird tuning + camera each frame.
//   - Overlay/expose: __birdBooted, __birdPos, __birdVel, __birdScheme, __birdTuning (verification).

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
    tuning: { windCoupling: 0.12, drag: 0.96, flickStrength: 14, thrust: 90, flapStrength: 8, bankRate: 0.45 },
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

  // ---- Control schemes + live switcher (1-4) ----
  const SCHEME_LABELS = [
    "1 — flick to impulse (drag→release)",
    "2 — hold toward cursor (hold mouse)",
    "3 — tap to bank (ArrowLeft/ArrowRight)",
    "4 — flap forward (Space / click)",
  ];
  let activeScheme = 1; // 1..4
  (window as any).__birdScheme = activeScheme;

  // Scheme 1 (flick) state.
  let dragging = false;
  let dragStart: [number, number] = [0, 0];
  let dragCur: [number, number] = [0, 0];
  let pendingImpulse: [number, number] | null = null; // one-shot (schemes 1 & 4)
  let pendingTurn = 0; // one-shot (scheme 3)

  // Scheme 2 (hold toward cursor) state: track cursor always + a held flag.
  let mouseScreen: [number, number] = [CANVAS_PX / 2, CANVAS_PX / 2];
  let held = false;

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

  // Screen pixel → world: relative to the (eased) camera center, scaled by the view span, y flipped.
  const screenToWorld = (sx: number, sy: number): [number, number] => {
    const cam = bird.getCameraPos();
    const wx = cam[0] + (sx / CANVAS_PX - 0.5) * VIEW;
    const wy = cam[1] - (sy / CANVAS_PX - 0.5) * VIEW;
    return [wx, wy];
  };

  canvas.addEventListener("mousedown", (e) => {
    mouseScreen = [e.clientX, e.clientY];
    if (activeScheme === 1) {
      dragging = true;
      dragStart = [e.clientX, e.clientY];
      dragCur = [e.clientX, e.clientY];
    } else if (activeScheme === 2) {
      held = true;
    } else if (activeScheme === 4) {
      pendingImpulse = null; // flap reads heading; mark a tap
      pendingImpulse = bird.flapToIntent().impulse;
    }
  });
  canvas.addEventListener("mousemove", (e) => {
    mouseScreen = [e.clientX, e.clientY];
    if (activeScheme === 1 && dragging) {
      dragCur = [e.clientX, e.clientY];
      setAim(dragStart[0], dragStart[1], dragCur[0], dragCur[1], true);
    }
  });
  const endDrag = () => {
    held = false;
    if (activeScheme !== 1 || !dragging) return;
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

  // Keys: 1-4 switch scheme; arrows = scheme 3 bank; space = scheme 4 flap. Attach to window so
  // Playwright's keyboard.press reaches them.
  window.addEventListener("keydown", (e) => {
    if (e.key >= "1" && e.key <= "4") {
      activeScheme = Number(e.key);
      (window as any).__birdScheme = activeScheme;
      held = false;
      dragging = false;
      setAim(0, 0, 0, 0, false);
      return;
    }
    if (activeScheme === 3) {
      if (e.key === "ArrowLeft") { pendingTurn = bird.bankToIntent(+1).turn; e.preventDefault(); }
      else if (e.key === "ArrowRight") { pendingTurn = bird.bankToIntent(-1).turn; e.preventDefault(); }
    }
    if (activeScheme === 4 && (e.key === " " || e.code === "Space")) {
      pendingImpulse = bird.flapToIntent().impulse;
      e.preventDefault();
    }
  });

  // ---- Tuning overlay: live sliders bound to bird tuning + camera ----
  buildTuningOverlay(bird, VIEW);

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

    // Build the per-frame intent from the active scheme. Schemes 1/3/4 fire one-shot pendings;
    // scheme 2 produces a fresh continuous impulse each held frame.
    let intent = { impulse: [0, 0] as [number, number], turn: 0 };
    if (activeScheme === 1) {
      if (pendingImpulse) { intent = bird.flickToIntent(pendingImpulse); pendingImpulse = null; }
    } else if (activeScheme === 2) {
      if (held) {
        const cursorWorld = screenToWorld(mouseScreen[0], mouseScreen[1]);
        intent = bird.thrustToIntent(cursorWorld, DT);
      }
    } else if (activeScheme === 3) {
      if (pendingTurn !== 0) { intent.turn = pendingTurn; pendingTurn = 0; }
    } else if (activeScheme === 4) {
      if (pendingImpulse) { intent.impulse = pendingImpulse; pendingImpulse = null; }
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
    (window as any).__birdTuning = bird.getTuning();

    frame++;
    overlay.textContent =
      `vector-system — bird (feel proto)\n` +
      `scheme [1-4]: ${SCHEME_LABELS[activeScheme - 1]}\n` +
      `sim ${SIM}²  view ${VIEW}  iters ${ITERS}\n` +
      `cpu dt: ${(dt * 1000).toFixed(2)} ms\n` +
      `bird pos: ${bird.lastPos[0].toFixed(1)}, ${bird.lastPos[1].toFixed(1)}  ` +
      `vel: ${bird.lastVel[0].toFixed(2)}, ${bird.lastVel[1].toFixed(2)}`;
  });

  loop.start();
  (window as any).__birdBooted = true;
}

// Build the live tuning panel: one slider per feel constant, bound to bird tuning + camera each
// change. Sliders are the point of the proto — fly and tune. id'd for verification (#tune-drag etc).
function buildTuningOverlay(bird: Bird, view: number): void {
  const panel = document.getElementById("tuning")!;
  // [key, label, min, max, step, get→initial, apply(value)]
  const t = bird.getTuning();
  type Row = {
    id: string; label: string; min: number; max: number; step: number; value: number;
    apply: (v: number) => void;
  };
  const rows: Row[] = [
    { id: "windCoupling", label: "windCoupling", min: 0, max: 0.6, step: 0.005, value: t.windCoupling,
      apply: (v) => bird.setTuning({ windCoupling: v }) },
    { id: "drag", label: "drag", min: 0.85, max: 1.0, step: 0.002, value: t.drag,
      apply: (v) => bird.setTuning({ drag: v }) },
    { id: "flickStrength", label: "flick str", min: 1, max: 40, step: 0.5, value: t.flickStrength,
      apply: (v) => bird.setTuning({ flickStrength: v }) },
    { id: "thrust", label: "thrust str", min: 5, max: 300, step: 1, value: t.thrust,
      apply: (v) => bird.setTuning({ thrust: v }) },
    { id: "flapStrength", label: "flap str", min: 1, max: 30, step: 0.5, value: t.flapStrength,
      apply: (v) => bird.setTuning({ flapStrength: v }) },
    { id: "bankRate", label: "bank rate", min: 0.05, max: 1.2, step: 0.01, value: t.bankRate,
      apply: (v) => bird.setTuning({ bankRate: v }) },
    { id: "deadzone", label: "cam deadzone", min: 0, max: view * 0.45, step: 0.5, value: view * 0.15,
      apply: (v) => bird.setCamera({ deadzone: v }) },
    { id: "followStiffness", label: "cam follow", min: 0.02, max: 0.9, step: 0.01, value: 0.22,
      apply: (v) => bird.setCamera({ followStiffness: v }) },
  ];

  for (const r of rows) {
    const label = document.createElement("label");
    const name = document.createElement("span");
    name.textContent = r.label;
    const input = document.createElement("input");
    input.type = "range";
    input.id = `tune-${r.id}`;
    input.min = String(r.min);
    input.max = String(r.max);
    input.step = String(r.step);
    input.value = String(r.value);
    const val = document.createElement("span");
    val.className = "val";
    val.textContent = r.value.toFixed(2);
    input.addEventListener("input", () => {
      const v = Number(input.value);
      r.apply(v);
      val.textContent = v.toFixed(2);
    });
    label.append(name, input, val);
    panel.append(label);
  }

  const schemes = document.createElement("div");
  schemes.id = "schemes";
  schemes.textContent = "schemes: 1 flick · 2 hold-cursor · 3 bank ←→ · 4 flap space/click";
  panel.append(schemes);
}

boot().catch((e) => {
  const overlay = document.getElementById("overlay");
  if (overlay) overlay.textContent = "boot error: " + (e as Error).message;
  throw e;
});
