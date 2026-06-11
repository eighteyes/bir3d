// bird-main.ts — 3D terrain + chase-cam look validation (NO bird). Entry for /index-bird.html.
// Responsibilities:
//   - Acquire device; configure the canvas context + an explicit-sized color/depth target.
//   - Build Terrain3D (neon ridgeline heightfield) and a ChaseCamera that auto-advances forward.
//   - Per frame (one encoder, one submit): advance + update camera; viewProj = proj*view;
//     terrain.draw(...) into the canvas with the depth attachment and a dark hazy-sky clear.
//   - Recreate the depth texture on canvas resize (must match the color attachment size).
//   - Overlay: cam pos + fps. Expose window.__birdBooted for the screenshot driver.

import { acquireDevice } from "./gpu/device";
import { Terrain3D } from "./gpu/terrain";
import { ChaseCamera } from "./gpu/camera";
import { perspective, multiply } from "./gpu/mat4";
import { FrameLoop } from "./frameloop";

const FOV_Y = (60 * Math.PI) / 180;
const NEAR = 1;
const FAR = 12000;

// Dark hazy-horizon sky / fog color (near-black blue haze). Terrain fog mixes toward this so far
// ridges fade into the horizon — keep the clear color == fog color so the horizon is seamless.
const SKY: [number, number, number] = [0.03, 0.04, 0.09];

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

  // Explicit pixel size (don't trust CSS/dpr). Depth texture sized to match.
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

  const shader = await fetch("/src/host/shaders/terrain3d.wgsl").then((r) => r.text());
  const terrain = new Terrain3D(device, shader, format, {
    n: 201,
    cellSize: 24,        // 201*24 ≈ 4.8km — many 350m ridges to a horizon
    fogColor: SKY,
    fogDensity: 1 / 900, // far ridges vanish before the grid edge (no hard seam)
  });

  // Cruise the camera near crest height (~220m relief) looking nearly level so near crests
  // silhouette the valleys behind them — the receding-ridgeline occlusion.
  // Eye must sit INSIDE the relief band (crests reach ~220) so near crests rise above the
  // sightline and occlude the terrain behind them — that is the layered-ridgeline effect.
  const cam = new ChaseCamera({
    followDist: 140,
    followHeight: 28,    // eye ≈ cruise + 28 ≈ 168 — below the tallest crests
    lookAhead: 420,      // long look-ahead → near-level pitch
    smooth: 0.12,
    cruiseHeight: 140,   // inside the band; tallest crests rise ~50-80m above the eye
    lookDrop: -15,       // look-target slightly ABOVE cruise → ~2-3° down (near level)
  });

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

    cam.autoAdvance(dt);
    cam.update();
    const proj = perspective(FOV_Y, pxW / pxH, NEAR, FAR);
    const view = cam.viewMatrix();
    const viewProj = multiply(proj, view);

    const colorView = ctx.getCurrentTexture().createView();
    const depthView = depthTex.createView();
    const eye = cam.getEye();

    const enc = device.createCommandEncoder();
    terrain.draw(enc, colorView, depthView, viewProj, cam.camOffset(), eye, {
      r: SKY[0], g: SKY[1], b: SKY[2], a: 1,
    });
    device.queue.submit([enc.finish()]);

    (window as any).__camPos = eye;
    frame++;
    overlay.textContent =
      `vector-system — terrain3d (look validation)\n` +
      `cam eye: ${eye[0].toFixed(0)}, ${eye[1].toFixed(0)}, ${eye[2].toFixed(0)}\n` +
      `target z: ${cam.target[2].toFixed(0)} m   grid ${terrain.n}² @ ${terrain.cellSize}m (${(terrain.worldSpan / 1000).toFixed(1)}km)\n` +
      `fps: ${fps.toFixed(0)}   frame ${frame}`;
  });

  loop.start();
  (window as any).__birdBooted = true;
}

boot().catch((e) => {
  const overlay = document.getElementById("overlay");
  if (overlay) overlay.textContent = "boot error: " + (e as Error).message;
  throw e;
});
