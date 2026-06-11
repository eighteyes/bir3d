// bird-main.ts — 3D flapping-V bird over the neon ridgeline terrain. Entry for /index-bird.html.
// Responsibilities:
//   - Acquire device; configure canvas color + matching depth target (recreate on resize).
//   - Build Terrain3D (neon ridgeline) and Bird3D (CPU-integrated flapping-V); ChaseCamera follows
//     the BIRD: target=bird.pos, forward=bird.forwardVec(), camOffset=(bird.x,bird.z) so the terrain
//     grid recenters under the bird.
//   - ONE control: mouse-steer (cursor offset from screen-center → yaw+pitch rate); click/Space=flap.
//   - Per frame (one encoder, one submit): read input → bird.integrate → camera follow → viewProj;
//     terrain.draw (clears color+depth) → bird.draw (LOADS color+depth, second pass, depth-tested).
//   - Overlay: altitude-over-terrain, speed, heading, wind, fps. Expose window.__birdBooted.

import { acquireDevice } from "./gpu/device";
import { Terrain3D } from "./gpu/terrain";
import { Bird3D, type BirdInput } from "./gpu/bird3d";
import { ChaseCamera } from "./gpu/camera";
import { perspective, multiply } from "./gpu/mat4";
import { FrameLoop } from "./frameloop";

const FOV_Y = (60 * Math.PI) / 180;
const NEAR = 1;
const FAR = 12000;

// Dark hazy-horizon sky / fog color. Terrain fog mixes toward this; clear color == fog color.
// Dim indigo (not pure black) so receding ridges dissolve into a visible haze band at the horizon
// and the grid's far edge is hidden. Kept low to respect the brightness cap (fills the screen).
const SKY: [number, number, number] = [0.06, 0.05, 0.12];

// Mouse-steer gains: cursor offset from screen-center (normalized -1..1) → rate.
const YAW_GAIN = 1.1;   // rad/s at full deflection
const PITCH_GAIN = 0.8; // rad/s at full deflection
const DEADZONE = 0.06;

async function boot() {
  const overlay = document.getElementById("overlay")!;
  const canvas = document.getElementById("bird") as HTMLCanvasElement;

  const { device } = await acquireDevice();
  device.lost.then((info) => {
    overlay.textContent = `WebGPU device lost: ${info.reason} — ${info.message}`;
    console.error("[WebGPU lost]", info.reason, info.message);
  });

  const ctx = canvas.getContext("webgpu") as GPUCanvasContext;
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  let pxW = Math.floor(canvas.clientWidth * dpr) || 900;
  let pxH = Math.floor(canvas.clientHeight * dpr) || 640;
  canvas.width = pxW;
  canvas.height = pxH;

  let depthTex = device.createTexture({
    size: [pxW, pxH],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const terrainShader = await fetch("/src/host/shaders/terrain3d.wgsl").then((r) => r.text());
  const terrain = new Terrain3D(device, terrainShader, format, {
    n: 201,
    cellSize: 24,
    fogColor: SKY,
    fogDensity: 1 / 700, // denser: far ridges haze into the horizon band sooner (hide grid edge)
  });

  const birdShader = await fetch("/src/host/shaders/bird3d.wgsl").then((r) => r.text());
  const startH = terrain.sampleHeight(0, 0);
  // start moderately low so near crests can cross in front of the bird (occlusion reads).
  const bird = new Bird3D(device, birdShader, format, terrain, [0, startH + 55, 0]);

  // Chase cam follows the bird: behind + slightly above, looking ahead/slightly down.
  const cam = new ChaseCamera({
    followDist: 110, // crest more often sits between camera and bird (occlusion); <140 keeps eye clear
    followHeight: 9, // low eye → flat sightline so foreground crests cross above the bird (occlusion)
    lookAhead: 140,
    smooth: 0.14,
    cruiseHeight: 0, // unused: target set explicitly to bird.pos each frame
    lookDrop: 6,     // near-level pitch
  });

  // --- input: mouse-steer + flap ---
  const input: BirdInput = { yawRate: 0, pitchRate: 0, flap: false };
  // normalized cursor offset from screen-center (-1..1); start centered (no steer before first move).
  let mouseX = 0, mouseY = 0;
  let flapHeld = false;
  let flapTap = false;

  canvas.addEventListener("mousemove", (e) => {
    const r = canvas.getBoundingClientRect();
    mouseX = ((e.clientX - r.left) / r.width) * 2 - 1;  // -1..1
    mouseY = ((e.clientY - r.top) / r.height) * 2 - 1;  // -1..1
  });
  canvas.addEventListener("mousedown", () => { flapTap = true; flapHeld = true; });
  window.addEventListener("mouseup", () => { flapHeld = false; });
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") { e.preventDefault(); flapTap = true; flapHeld = true; }
  });
  window.addEventListener("keyup", (e) => { if (e.code === "Space") flapHeld = false; });

  const applyDead = (v: number) =>
    Math.abs(v) < DEADZONE ? 0 : (v - Math.sign(v) * DEADZONE) / (1 - DEADZONE);

  const resize = () => {
    pxW = Math.floor(canvas.clientWidth * dpr) || pxW;
    pxH = Math.floor(canvas.clientHeight * dpr) || pxH;
    canvas.width = pxW;
    canvas.height = pxH;
    depthTex.destroy();
    depthTex = device.createTexture({
      size: [pxW, pxH],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  };
  window.addEventListener("resize", resize);

  let frame = 0;
  let fps = 0;
  const loop = new FrameLoop((dt) => {
    fps = fps * 0.9 + (1 / Math.max(dt, 1e-3)) * 0.1;

    // map input
    input.yawRate = applyDead(mouseX) * YAW_GAIN;
    input.pitchRate = -applyDead(mouseY) * PITCH_GAIN; // mouse-up = nose-up
    input.flap = flapTap || flapHeld;
    flapTap = false;

    bird.integrate(dt, input);

    // camera follows the bird
    cam.target = [bird.pos[0], bird.pos[1], bird.pos[2]];
    const fwd = bird.forwardVec();
    cam.forward = [fwd[0], fwd[1], fwd[2]];
    cam.update();

    const proj = perspective(FOV_Y, pxW / pxH, NEAR, FAR);
    const view = cam.viewMatrix();
    const viewProj = multiply(proj, view);

    const colorView = ctx.getCurrentTexture().createView();
    const depthView = depthTex.createView();
    const eye = cam.getEye();

    const enc = device.createCommandEncoder();
    // terrain pass: clears color+depth.
    terrain.draw(enc, colorView, depthView, viewProj, [bird.pos[0], bird.pos[2]], eye, {
      r: SKY[0], g: SKY[1], b: SKY[2], a: 1,
    });
    // bird pass: loads color+depth, depth-tested → ridges occlude the bird.
    bird.draw(enc, colorView, depthView, viewProj);
    device.queue.submit([enc.finish()]);

    (window as any).__camPos = eye;
    (window as any).__birdPos = bird.pos;
    frame++;
    const headingDeg = ((bird.heading * 180) / Math.PI) % 360;
    overlay.textContent =
      `vector-system — bird3d (flapping-V chase)\n` +
      `alt over terrain: ${bird.lastClearance.toFixed(0)} m   speed: ${bird.lastSpeed.toFixed(0)} m/s\n` +
      `heading: ${headingDeg.toFixed(0)}°   pitch: ${((bird.pitch * 180) / Math.PI).toFixed(0)}°   bank: ${((bird.bank * 180) / Math.PI).toFixed(0)}°\n` +
      `wind: ${bird.lastWind[0].toFixed(1)}, ${bird.lastWind[1].toFixed(1)} m/s\n` +
      `fps: ${fps.toFixed(0)}   frame ${frame}   (mouse=steer, click/Space=flap)`;
  });

  loop.start();
  (window as any).__birdBooted = true;
}

boot().catch((e) => {
  const overlay = document.getElementById("overlay");
  if (overlay) overlay.textContent = "boot error: " + (e as Error).message;
  throw e;
});
