// bird-main.ts — 3D gliding bird over the neon ridgeline terrain. Entry for /index-bird.html.
// Responsibilities:
//   - Acquire device; configure canvas color + matching depth target (recreate on resize).
//   - Build TerrainEKG (neon ridgeline) and Bird3D (CPU-integrated glider); ChaseCamera follows
//     the BIRD: target=bird.pos, forward=bird.forwardVec(), camOffset=(bird.x,bird.z) so the terrain
//     grid recenters under the bird.
//   - ONE control: mouse-steer (cursor offset from screen-center → yaw+pitch rate). Pure glide —
//     dive to gain speed, pull up to zoom-climb, ride ridge lift to stay aloft. No flap input.
//   - Per frame (one encoder, one submit): read input → bird.integrate → camera follow → viewProj;
//     terrain.draw (clears color+depth) → wind.draw (streamline comets, LOADS, depth-test no-write)
//     → bird.draw (LOADS color+depth, depth-tested).
//   - Overlay: altitude, airspeed, vario, updraft, heading vs GROUND-TRACK + DRIFT, wind, fps.
//   - Compass canvas (bottom-right): large heading/ground-track/wind vectors — the felt-wind proof
//     (cyan heading vs yellow ground-track gap = visible cross-track drift from wind).
//   - Tuning panel ('T' toggles): live sliders writing straight into bird.tuning (feel dial-in).
//   - Expose window.__birdBooted.

import { acquireDevice } from "./gpu/device";
import { TerrainEKG } from "./gpu/terrain";
import { Bird3D, type BirdInput } from "./gpu/bird3d";
import { Wind } from "./gpu/wind";
import { ChaseCamera } from "./gpu/camera";
import { perspective, multiply } from "./gpu/mat4";
import { FrameLoop } from "./frameloop";

const FOV_Y = (60 * Math.PI) / 180;
const NEAR = 1;
const FAR = 12000;

// Near-black ground/haze. Clear color == fog color. The terrain exists PURELY as glowing lines on
// this dark ground (NO fill); far lines fade into this haze. Very dark so the neon lines read.
const SKY: [number, number, number] = [0.01, 0.012, 0.03];

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

  const terrainShader = await fetch("/src/host/shaders/terrain_ekg.wgsl").then((r) => r.text());
  const terrain = new TerrainEKG(device, terrainShader, format, {
    rows: 64,         // stacked EKG depth rows
    cols: 256,        // samples per row (polyline resolution)
    rowSpacing: 36,   // m between rows — tight stack so the lower frame fills with EKG traces
    rowStart: -150,   // BEHIND the camera ground point. Rows are built ahead of the camera; start a
                      // little behind so the near-ground under the camera isn't empty black.
    halfWidth: 1500,  // horizontal extent per row (m)
    maxDist: 950,     // CLEAN HORIZON: hard cutoff — drop rows past ~950 m so the far stack never
                      // tangles into a horizon mess (user is fine losing far detail). ~30 visible rows.
    fogColor: SKY,
    fogDensity: 1 / 700, // strong fog: far rows dissolve well before the cutoff edge.
  });

  const birdShader = await fetch("/src/host/shaders/bird3d.wgsl").then((r) => r.text());
  const startH = terrain.sampleHeight(0, 0);
  // HIGHER START (v4): begin well above terrain (~200 m clearance) so the flight is aerial from
  // the first frame and the EKG stack sits below the eyeline (less horizon tangle).
  const bird = new Bird3D(device, birdShader, format, terrain, [0, startH + 200, 0]);

  // VISIBLE WIND: neon streamline comets over the terrain, integrated from the SAME shared windAt
  // field that pushes the bird (src/host/gpu/wind.ts). Camera-relative like the EKG rows.
  const windShader = await fetch("/src/host/shaders/wind.wgsl").then((r) => r.text());
  const wind = new Wind(device, windShader, format, (x, z) => terrain.sampleHeight(x, z));

  // Chase cam follows the bird POSITION + HEADING only (world-up, ground-locked aim). Looks DOWN
  // on the bird's back at a FIXED angle so the ground ALWAYS fills the lower frame, whatever the
  // bird's pitch — this is the v3 ground-lock fix.
  const cam = new ChaseCamera({
    followDist: 120,
    followHeight: 55, // above the bird → look down on its back (the V reads)
    lookAhead: 160,
    lookPitch: (28 * Math.PI) / 180, // fixed ~28° down; steeper → EKG stack spreads down the frame
    smooth: 0.14,
  });

  // --- input: mouse-steer only (pure glide) ---
  const input: BirdInput = { yawRate: 0, pitchRate: 0 };
  // normalized cursor offset from screen-center (-1..1); start centered (no steer before first move).
  let mouseX = 0, mouseY = 0;

  canvas.addEventListener("mousemove", (e) => {
    const r = canvas.getBoundingClientRect();
    mouseX = ((e.clientX - r.left) / r.width) * 2 - 1;  // -1..1
    mouseY = ((e.clientY - r.top) / r.height) * 2 - 1;  // -1..1
    // a real player moved the mouse → the scripted wobble yields control immediately.
    (window as any).__autoWobble = false;
  });

  // --- tuning panel: sliders write straight into bird.tuning; 'T' toggles visibility ---
  const tunePanel = buildTunePanel(bird.tuning, [
    ["glideSpeed", 14, 40, 0.5],
    ["sinkRate", 0.3, 4, 0.1],
    ["divePower", 0.2, 2, 0.05],
    ["dragK", 0.1, 1.5, 0.05],
    ["liftGain", 0, 6, 0.1],
    ["windGain", 0, 15, 0.5],
    ["windDrift", 0, 2, 0.1],
    ["minSpeed", 8, 20, 0.5],
    ["maxSpeed", 30, 80, 1],
  ]);
  document.body.appendChild(tunePanel);
  window.addEventListener("keydown", (e) => {
    if (e.code === "KeyT") {
      tunePanel.style.display = tunePanel.style.display === "none" ? "block" : "none";
    }
  });

  // --- compass overlay canvas: large heading-vs-ground-track-vs-wind vectors (the felt-wind proof) ---
  const compass = document.createElement("canvas");
  compass.id = "compass";
  compass.width = 200;
  compass.height = 200;
  compass.style.cssText =
    "position:fixed;right:14px;bottom:14px;width:200px;height:200px;z-index:9;" +
    "background:rgba(6,5,18,0.55);border:1px solid #2a2550;border-radius:8px;";
  document.body.appendChild(compass);
  const compassCtx = compass.getContext("2d")!;

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

  // Scripted pitch wobble (THIS task): auto nose up/down so the screenshot proves the camera keeps
  // the ground framed no matter how hard the BIRD pitches. window.__autoWobble defaults on.
  (window as any).__autoWobble = true;
  let wobbleT = 0;

  let frame = 0;
  let fps = 0;
  const loop = new FrameLoop((dt) => {
    fps = fps * 0.9 + (1 / Math.max(dt, 1e-3)) * 0.1;

    // map input
    input.yawRate = applyDead(mouseX) * YAW_GAIN;
    input.pitchRate = -applyDead(mouseY) * PITCH_GAIN; // mouse-up = nose-up

    // scripted pitch wobble drives the bird hard up/down; the camera must NOT follow the pitch.
    // PITCH ONLY (no yaw) → heading stays 0 so the world-axis EKG rows render as clean horizontal
    // stacked lines; the wobble is purely the ground-lock proof.
    if ((window as any).__autoWobble) {
      wobbleT += dt;
      input.pitchRate = Math.sin(wobbleT * 1.1) * PITCH_GAIN * 1.6; // exceeds manual range → hard pitch
      input.yawRate = 0;
    }

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
    // terrain pass: clears color+depth. CAMERA-RELATIVE rows — build them around the camera ground
    // point using the SMOOTHED view basis (forward/right) so the stack stays screen-horizontal.
    const camGround = cam.groundPos();
    const camFwd = cam.forwardHoriz();
    const camRight = cam.rightHoriz();
    terrain.draw(enc, colorView, depthView, viewProj, camGround, camFwd, camRight, eye, {
      r: SKY[0], g: SKY[1], b: SKY[2], a: 1,
    });
    // wind pass: loads color+depth (no clear); neon streamline comets over the ridges (depth-tested,
    // no depth-write) — uses the bird's sim time so the drawn field matches the field that pushes.
    wind.draw(enc, colorView, depthView, viewProj, camGround, camFwd, camRight, eye,
      bird.simTime, SKY, 1 / 700);
    // bird pass: loads color+depth, depth-tested → ridges occlude the bird.
    bird.draw(enc, colorView, depthView, viewProj);
    device.queue.submit([enc.finish()]);

    (window as any).__camPos = eye;
    (window as any).__birdPos = bird.pos;
    (window as any).__birdPitch = bird.pitch;     // live pitch (rad)
    (window as any).__birdHeading = bird.heading; // live heading (rad) — capture harness waits for a real turn
    (window as any).__birdGroundTrack = bird.lastGroundTrack; // actual travel dir (rad) — drift proof
    (window as any).__birdWind = bird.lastWind;   // [wx,wz] m/s — overlay/diagnostics
    (window as any).__birdVario = bird.lastVario; // climb m/s — lift proof
    frame++;
    const headingDeg = ((bird.heading * 180) / Math.PI) % 360;
    const trackDeg = (bird.lastGroundTrack * 180) / Math.PI;
    // signed drift = ground-track − heading wrapped to (-180,180]; this is the felt-wind number.
    let drift = trackDeg - (bird.heading * 180) / Math.PI;
    drift = ((((drift + 180) % 360) + 360) % 360) - 180;
    const vario = bird.lastVario;
    const varioStr = `${vario >= 0 ? "+" : ""}${vario.toFixed(1)}`;
    const windSpeed = Math.hypot(bird.lastWind[0], bird.lastWind[1]);
    overlay.textContent =
      `vector-system — bird3d (soaring glider chase)\n` +
      `alt over terrain: ${bird.lastClearance.toFixed(0)} m   air: ${bird.lastSpeed.toFixed(0)} m/s\n` +
      `vario: ${varioStr} m/s ${vario > 0.5 ? "▲" : vario < -0.5 ? "▼" : "—"}   updraft: +${bird.lastUpdraft.toFixed(1)} m/s\n` +
      `heading: ${headingDeg.toFixed(0)}°   ground-track: ${trackDeg.toFixed(0)}°   DRIFT: ${drift >= 0 ? "+" : ""}${drift.toFixed(0)}°\n` +
      `wind: ${bird.lastWind[0].toFixed(1)}, ${bird.lastWind[1].toFixed(1)} m/s  (|${windSpeed.toFixed(1)}|)\n` +
      `fps: ${fps.toFixed(0)}   frame ${frame}   (mouse=steer: down=dive/speed, up=zoom-climb, T=tuning)`;

    // compass overlay: large vectors — heading (cyan), ground-track (yellow), wind (magenta).
    drawCompass(compassCtx, bird.heading, bird.lastGroundTrack, bird.lastWind, windSpeed, drift);
  });

  loop.start();
  (window as any).__birdBooted = true;
}

// Build a floating slider panel bound directly to a live tuning object.
// rows: [key, min, max, step][] — each slider writes tuning[key] on input.
function buildTunePanel(
  tuning: Record<string, number>,
  rows: [string, number, number, number][]
): HTMLDivElement {
  const panel = document.createElement("div");
  panel.id = "tune";
  panel.style.cssText =
    "position:fixed;right:12px;top:12px;display:none;padding:10px 12px;" +
    "background:rgba(8,6,20,0.85);border:1px solid #3a3360;border-radius:6px;" +
    "font:12px/1.6 monospace;color:#9fe8ff;z-index:10;min-width:240px;";
  for (const [key, min, max, step] of rows) {
    const row = document.createElement("div");
    const label = document.createElement("span");
    const val = document.createElement("span");
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(step);
    slider.value = String(tuning[key]);
    slider.style.cssText = "width:110px;vertical-align:middle;margin:0 6px;";
    label.textContent = key.padEnd(11);
    val.textContent = String(tuning[key]);
    slider.addEventListener("input", () => {
      tuning[key] = Number(slider.value);
      val.textContent = slider.value;
    });
    row.append(label, slider, val);
    panel.appendChild(row);
  }
  return panel;
}

// Draw the felt-wind compass: heading (cyan), ground-track (yellow), wind (magenta) as vectors from
// center. North (heading reference) is UP. World X=east → screen +x; world Z=north → screen -y.
// A non-zero gap between cyan (heading) and yellow (ground-track) is the visible drift proof.
function drawCompass(
  ctx: CanvasRenderingContext2D,
  heading: number,
  track: number,
  wind: [number, number],
  windSpeed: number,
  driftDeg: number
): void {
  const w = ctx.canvas.width, h = ctx.canvas.height;
  const cx = w / 2, cy = h / 2;
  ctx.clearRect(0, 0, w, h);

  // ring
  ctx.strokeStyle = "rgba(120,120,180,0.35)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, 72, 0, Math.PI * 2);
  ctx.stroke();

  // a heading angle (rad, +Z=north=up, +X=east=right) → screen vector. dir=(sin,cos) world (x,z).
  const vec = (ang: number, len: number): [number, number] => [
    cx + Math.sin(ang) * len,
    cy - Math.cos(ang) * len, // world +Z (north) is screen up
  ];
  const arrow = (ang: number, len: number, color: string, lw: number) => {
    const [ex, ey] = vec(ang, len);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = lw;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    // arrowhead
    const a = Math.atan2(ey - cy, ex - cx);
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - 9 * Math.cos(a - 0.4), ey - 9 * Math.sin(a - 0.4));
    ctx.lineTo(ex - 9 * Math.cos(a + 0.4), ey - 9 * Math.sin(a + 0.4));
    ctx.closePath();
    ctx.fill();
  };

  // wind vector: angle from world (wx,wz); length scales with speed (capped).
  const windAng = Math.atan2(wind[0], wind[1]); // atan2(x,z) → heading-convention angle
  const windLen = Math.min(70, 14 + windSpeed * 3.0);
  arrow(windAng, windLen, "rgba(230,90,230,0.95)", 5); // wind — magenta, thick
  arrow(heading, 66, "rgba(80,220,255,0.95)", 3);        // heading — cyan
  arrow(track, 66, "rgba(255,225,70,0.95)", 3);          // ground-track — yellow

  ctx.font = "11px monospace";
  ctx.fillStyle = "#9fe8ff"; ctx.fillText("heading", 8, 16);
  ctx.fillStyle = "#ffe146"; ctx.fillText("track", 8, 30);
  ctx.fillStyle = "#e65ae6"; ctx.fillText("wind", 8, 44);
  ctx.fillStyle = "#fff";
  ctx.fillText(`drift ${driftDeg >= 0 ? "+" : ""}${driftDeg.toFixed(0)}°`, 8, h - 10);
}

boot().catch((e) => {
  const overlay = document.getElementById("overlay");
  if (overlay) overlay.textContent = "boot error: " + (e as Error).message;
  throw e;
});
