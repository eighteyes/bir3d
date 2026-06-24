// bird-main.ts — 3D gliding bird over the neon ridgeline terrain. Entry for /index-bird.html.
// Responsibilities:
//   - Acquire device; configure canvas (swapchain) + HDR rgba16float MSAA scene target + single-sample
//     HDR resolve/scene texture + matching depth target (recreate on resize).
//   - BLOOM post-process: scene passes render into the HDR target (rgba16float, so the glow doesn't
//     band on the dark scene), resolve into the scene texture, then bloom (threshold → separable
//     blur → composite/Reinhard tone-map) writes the final glowing image to the swapchain.
//   - Build TerrainEKG (neon ridgeline) and Bird3D (CPU-integrated glider); ChaseCamera follows
//     the BIRD: target=bird.pos, forward=bird.forwardVec(), camOffset=(bird.x,bird.z) so the terrain
//     grid recenters under the bird.
//   - ONE control: mouse-steer (cursor offset from screen-center → yaw+pitch rate). Pure glide —
//     dive to gain speed, pull up to zoom-climb, ride ridge lift to stay aloft. No flap input.
//   - Per frame (one encoder, one submit): read input → bird.integrate → camera follow → viewProj;
//     terrain.draw (clears color+depth) → wind.draw (streamline comets, LOADS, depth-test no-write)
//     → bird.draw (LOADS color+depth, depth-tested).
//   - DEPTH/SWOOP cues: altitude-adaptive chase cam (low clearance → eye drops + look flattens →
//     ground rush; high → v3/v4 god-view), speed FOV kick (dive widens the view), and GroundMarker
//     plumb-line (dashed bird→ground drop-line, ~9 m/dash, + pulsing ground diamond) drawn last.
//   - Overlay: altitude, airspeed, vario, updraft, heading vs GROUND-TRACK + DRIFT, wind, fps.
//   - Compass canvas (bottom-right): large heading/ground-track/wind vectors — the felt-wind proof
//     (cyan heading vs yellow ground-track gap = visible cross-track drift from wind).
//   - Tuning panel ('T' toggles): live sliders writing straight into bird.tuning (feel dial-in).
//   - Expose window.__birdBooted.

import { acquireDevice } from "./gpu/device";
import { TerrainEKG } from "./gpu/terrain";
import { GridTerrain } from "./gpu/terrain-grid";
import { Bird3D, updraftAt, type BirdInput } from "./gpu/bird3d";
import { Wind, windAt, setFluidField, setWindProfile, windProfile, windProfileParams, FAR_MODES, NEAR_MODES, WAKE_MODES } from "./gpu/wind";
import type { FarMode, NearMode, WakeMode } from "./gpu/wind";
import { FluidWind } from "./gpu/fluid-wind";
import { GroundMarker } from "./gpu/marker";
import { Target } from "./gpu/target";
import { Trees } from "./gpu/trees";
import { ChaseCamera } from "./gpu/camera";
import { checkTerrain } from "./gpu/terrain-selfcheck";
import { AutoPilot } from "./autopilot";
import { Bloom } from "./gpu/bloom";
import { perspective, multiply } from "./gpu/mat4";
import { FrameLoop } from "./frameloop";

// AUTOPILOT MODE (this pass): manual controls OFF — the AutoPilot flies, proving autonomous
// soaring (find lift, ride it, never touch the ground) before flapping/controls return.
let autopilot = false; // default MANUAL — YOU fly and feel the wind; press P to hand it to the autopilot
let showWind = true; // neon wind motes drawn over the ridges (window.__showWind toggles); fed by the SAME fluid field that pushes the bird
// terrain renderer: "ekg" = original camera-relative scan-lines (lines run away); "grid" = world-static
// wireframe (parallax toward you); "topo" = world-static topographic contour lines. window.__terrainMode(m).
let terrainMode: "ekg" | "grid" | "topo" = "ekg";

const FOV_Y = (60 * Math.PI) / 180;
const FOV_KICK = (16 * Math.PI) / 180; // extra FOV at dive ceiling — speed reads as widening view
const NEAR = 1;
const FAR = 12000;

// MSAA: render every pass into a 4× multisample color+depth target, then resolve once (on the final
// marker pass) into the single-sample HDR scene texture — smooths the thin neon line/ribbon edges
// (the canvas itself can't be multisampled directly). Single source of truth: threaded into every
// pipeline so counts match.
const SAMPLES = 4;

// BLOOM (neon glow): all scene passes render into an HDR rgba16float MSAA target, resolve into a
// single-sample rgba16float SCENE texture, then the bloom chain (threshold → separable blur →
// composite/tone-map) reads that and writes the final image to the swapchain. rgba16float is
// REQUIRED — rgba8 bands the soft glow on this very dark scene.
const HDR_FORMAT: GPUTextureFormat = "rgba16float";

// Altitude-adaptive camera (the swoop fix): LOW clearance pulls the eye down near the bird and
// flattens the look angle so the ground rushes; HIGH clearance restores the v3/v4 god-view framing.
const CAM_LOW = { clearance: 25, height: 10, pitchDeg: 8 };
const CAM_HIGH = { clearance: 160, height: 55, pitchDeg: 28 };

// Near-black ground/haze. Clear color == fog color. The terrain exists PURELY as glowing lines on
// this dark ground (NO fill); far lines fade into this haze. Very dark so the neon lines read.
const SKY: [number, number, number] = [0.01, 0.012, 0.03];

// Mouse-steer gains. Yaw: cursor offset → turn RATE. Pitch: cursor HEIGHT → nose ATTITUDE
// (holdable — park the cursor, the nose stays put; center = gentle glide trim).
const YAW_GAIN = 1.8; // rad/s at full deflection (v8: crisper, less sluggish maneuvering)
const PITCH_RANGE = 1.0; // rad of nose angle at full vertical deflection — steep dives/climbs reachable mid-screen
const GLIDE_TRIM = -0.03; // rad — centered-cursor attitude: a gentle settling descent
const DEADZONE = 0.05;

// Fly-to-target basis: a target counts as reached within REACH_RADIUS (horizontal).
const REACH_RADIUS = 55;
const START_CLEARANCE = 400; // m above terrain at spawn — the altitude budget a bird spends reaching a
// target (flap to regain it). Raise to reach farther/higher.

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
    sampleCount: SAMPLES,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  // multisample HDR color target — every scene pass renders here; the final marker pass resolves it
  // into the single-sample HDR scene texture (NOT the swapchain — bloom composites that last).
  let msaaTex = device.createTexture({
    size: [pxW, pxH],
    format: HDR_FORMAT,
    sampleCount: SAMPLES,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  // single-sample HDR scene texture — the MSAA resolve target AND the bloom chain's input.
  let sceneTex = device.createTexture({
    size: [pxW, pxH],
    format: HDR_FORMAT,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const terrainShader = await fetch("/src/host/shaders/terrain_ekg.wgsl").then(
    (r) => r.text(),
  );
  const terrain = new TerrainEKG(device, terrainShader, HDR_FORMAT, {
    rows: 512, // stacked EKG depth rows — covers the 2× maxDist at the same 4× spacing
    cols: 1536, // samples per row — 2× density: the wide halfWidth means near rows show only a
    // central slice, so more samples are needed to shrink the chord faceting up close.
    sampleCount: SAMPLES,
    rowSpacing: 2, // near step (m). ~1/4 the rows (266→96) to cut the terrain pass ~14ms→~3ms (GPU-measured).
    nearDenseDepth: 250, // crisp band shortened to 250 m, then the far field thins hard via farSpread.
    farSpread: 100, // beyond 250 m, spacing grows by 7 m per 70 m of depth → aggressive far thinning (the
    // "sparser further away" lever). Lower = thin far harder; raise rowSpacing/nearDenseDepth for crisper near.
    rowStart: -150, // BEHIND the camera ground point. Rows are built ahead of the camera; start a
    // little behind so the near-ground under the camera isn't empty black.
    halfWidth: 2400, // horizontal extent per row (m) — at 1900 m depth the 76° dive-FOV frustum is
    // ~2300 m half-wide; 1500 would expose naked row ends near the new horizon.
    maxDist: 2850, // +50% view (1900→2850) paired with the thinner fog below — fog still dissolves
    // rows before this cutoff so the far edge never shows as a shelf.
    baseline: -300, // fill curtains drop to this world-y (occlusion only; below the frame).
    fogColor: SKY,
    fogDensity: 0.25 / 2200, // +50% view distance (fog ÷1.5 from 0.75/1100): less haze over the far rows.
    // = 1/2200; raise the 0.5 toward 1 for more fog / a shorter view.
  });

  // WORLD-STATIC wireframe terrain (alternative renderer; default ON via gridMode). Lines pinned to world
  // space → fly forward and they flow toward you with real parallax. Same fBm → trees/bird sit on it too.
  const gridShader = await fetch("/src/host/shaders/terrain_grid.wgsl").then((r) => r.text());
  const gridTerrain = new GridTerrain(device, gridShader, HDR_FORMAT, {
    spacing: 26,
    radius: 1650,
    maxDist: 1500,
    fogColor: SKY,
    fogDensity: 0.5 / 1100,
    sampleCount: SAMPLES,
  });

  const birdShader = await fetch("/src/host/shaders/bird3d.wgsl").then((r) =>
    r.text(),
  );
  const startH = terrain.sampleHeight(0, 0);
  // HIGHER START: begin START_CLEARANCE above terrain so the still-air glider has the altitude budget to
  // reach the first target, and the flight is aerial from the first frame (EKG stack below the eyeline).
  const bird = new Bird3D(
    device,
    birdShader,
    HDR_FORMAT,
    terrain,
    [0, startH + START_CLEARANCE, 0],
    {},
    SAMPLES,
  );
  // WIND LIVE: the bird flies the moving fluid field — horizontal drift you must correct, ridge lift +
  // thermals to ride, buffet/gust shake. windGain (bird3d tuning, default 1.6) scales the shove; the
  // visible motes (showWind) draw the SAME windAt field so you SEE the air that's pushing you.
  bird.stillAir = false;

  // VISIBLE WIND: neon streamline comets over the terrain, integrated from the SAME shared windAt
  // field that pushes the bird (src/host/gpu/wind.ts). Camera-relative like the EKG rows.
  const windShader = await fetch("/src/host/shaders/wind.wgsl").then((r) =>
    r.text(),
  );
  const wind = new Wind(
    device,
    windShader,
    HDR_FORMAT,
    (x, z) => terrain.sampleHeight(x, z),
    {},
    // FEWER, BETTER motes (user): cut counts ~2× and make each bigger so they read as distinct wind streaks
    // instead of a faint noisy cloud. dotPx is live-tunable (__wind.dotPx); counts need a reload.
    { nearCount: 200, numMotes: 4000, dotPx: 3.6 },
    SAMPLES,
  );

  // v13: the REAL GPU fluid is the EVOLVING wind SOURCE. windAt() (wind.ts) samples this field as its
  // base horizontal vector + keeps the prevailing drift; bird3d (physics) and the motes (via flowAt) ride
  // it for free. The fluid is stepped + read each frame here over a bird-local moving window; magnitude is
  // regulated to the flyable band by the SCALE (not by cranking force — that is the +61° regression).
  const fluidShaderPaths = {
    forces: "/src/host/shaders/fluid/forces.wgsl",
    divergence: "/src/host/shaders/fluid/divergence.wgsl",
    jacobi: "/src/host/shaders/fluid/jacobi.wgsl",
    subtractGrad: "/src/host/shaders/fluid/subtract_grad.wgsl",
    advect: "/src/host/shaders/fluid/advect.wgsl",
    setBnd: "/src/host/shaders/fluid/set_bnd.wgsl",
  };
  const fluidShaders = Object.fromEntries(
    await Promise.all(
      Object.entries(fluidShaderPaths).map(async ([k, p]) => [
        k,
        await fetch(p).then((r) => r.text()),
      ]),
    ),
  ) as {
    forces: string;
    divergence: string;
    jacobi: string;
    subtractGrad: string;
    advect: string;
    setBnd: string;
  };
  // iters is the perf lever: step() records ~2×iters×3 jacobi/set_bnd passes/frame, and the per-dispatch
  // encode cost (not per-cell GPU work) dominates — so iters, NOT grid, sets the frame budget. A wind
  // field needs no converged divergence-free projection, so low iters is visually fine and holds 60fps.
  const fluidWind = new FluidWind(device, fluidShaders, {
    grid: 256,
    iters: 10,
  });

  // ALTITUDE PLUMB-LINE: dashed neon drop-line bird→ground (one dash per ~9 m = readable altimeter)
  // + pulsing ground diamond. THE direct how-close-is-the-ground cue for swoops.
  const markerShader = await fetch("/src/host/shaders/marker.wgsl").then((r) =>
    r.text(),
  );
  const marker = new GroundMarker(device, markerShader, HDR_FORMAT, SAMPLES);

  // FLIGHT TARGET: an amber beam of light out in the distance — fly to it and it respawns ahead. The
  // playable basis ("see a target, fly toward it"). Drawn always-on-top so it stays visible behind ridges.
  const targetShader = await fetch("/src/host/shaders/target.wgsl").then((r) =>
    r.text(),
  );
  const target = new Target(
    device,
    targetShader,
    HDR_FORMAT,
    (x, z) => terrain.sampleHeight(x, z),
    SAMPLES,
  );

  // TREES: mountaintop neon forests of recursive glow-branch trees, streamed in a grid window around
  // the camera (placed only where terrain clears the peak threshold). Depth-tested so ridges occlude them.
  const treeShader = await fetch("/src/host/shaders/trees.wgsl").then((r) =>
    r.text(),
  );
  const treeGroundShader = await fetch(
    "/src/host/shaders/trees_ground.wgsl",
  ).then((r) => r.text());
  const trees = new Trees(
    device,
    treeShader,
    treeGroundShader,
    HDR_FORMAT,
    (x, z) => terrain.sampleHeight(x, z),
    SAMPLES,
  );

  // BLOOM post-process: reads the resolved HDR scene texture, writes the final image to the swapchain.
  // RE-TUNED for the additive-neon double-count risk (dense comet sphere + 50%-opacity wind + bright
  // bird are blowout candidates): threshold high enough that only bright cores seed the glow; Reinhard
  // tone-map in the composite keeps blown pixels HUE-COLORED instead of smearing to white. Half-res
  // blur (downsample 2) gives a wide soft glow AND holds the 60fps budget.
  const bloomShaders = {
    threshold: await fetch("/src/host/shaders/bloom_threshold.wgsl").then((r) =>
      r.text(),
    ),
    blur: await fetch("/src/host/shaders/bloom_blur.wgsl").then((r) =>
      r.text(),
    ),
    composite: await fetch("/src/host/shaders/bloom_composite.wgsl").then((r) =>
      r.text(),
    ),
  };
  const bloom = new Bloom(device, format, bloomShaders, {
    threshold: 0.85, // only neon cores above this luminance bloom (dark ground / dim far lines do not)
    knee: 0.5, // soft ramp above the threshold so the glow fades in, no hard edge
    intensity: 0.9, // bloom add weight — glow, not wash
    exposure: 1.0, // scene exposure into the tone-map
    downsample: 2, // half-res bloom chain (wide soft glow + perf)
    blurPasses: 2, // H+V iterations — widens the glow; 2 holds 60fps
  });
  bloom.resize(pxW, pxH);

  // hands-off flight controller (AUTOPILOT mode) — emits the same BirdInput the mouse did.
  // "straight" policy = bird-vs-wind eval: locked heading, trim glide, deviate only near ground.
  const auto = new AutoPilot(terrain, "straight");

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
  // v17: give the chase cam the terrain sampler so it can keep its eye OUT of the mountains (the taller
  // RELIEF made the eye embed in peaks → black frames when the bird runs into a mountain).
  cam.terrainHeight = (x, z) => terrain.sampleHeight(x, z);

  // --- input: mouse-steer only (pure glide) ---
  const input: BirdInput = { yawRate: 0, pitchTarget: GLIDE_TRIM, flap: false };
  // normalized cursor offset from screen-center (-1..1); start centered (no steer before first move).
  let mouseX = 0,
    mouseY = 0;
  let flapHeld = false; // Space held → powered wingbeat (tap = one beat, hold = sustained climb)

  // STICKY AUTOPILOT: 'P' toggles autopilot and it HOLDS regardless of mouse motion — so you can press P
  // and move the cursor out of the window to walk away without the bird veering off. Mouse movement updates
  // the steering origin but does NOT cancel autopilot; take manual control back with an EXPLICIT gesture
  // (click the canvas, or press P again).
  canvas.addEventListener("mousemove", (e) => {
    const r = canvas.getBoundingClientRect();
    mouseX = ((e.clientX - r.left) / r.width) * 2 - 1; // -1..1
    mouseY = ((e.clientY - r.top) / r.height) * 2 - 1; // -1..1
    // a real player moved the mouse → the scripted intro wobble yields, but autopilot stays (sticky).
    (window as any).__autoWobble = false;
  });
  // EXPLICIT manual takeback: clicking the canvas grabs the controls from autopilot.
  canvas.addEventListener("mousedown", () => {
    autopilot = false;
    (window as any).__autoWobble = false;
  });
  // INPUT HYGIENE: when the cursor leaves the canvas or the tab loses focus, recenter the steering so
  // MANUAL flight can't freeze on a hard edge-deflection (the old "holds last input" spiral). Autopilot is
  // sticky and reads neither mouseX/mouseY, so it is unaffected.
  const neutralizeSteer = () => { mouseX = 0; mouseY = 0; };
  canvas.addEventListener("mouseleave", neutralizeSteer);
  window.addEventListener("blur", neutralizeSteer);

  // --- tuning panel: sliders write straight into bird.tuning; 'T' toggles visibility ---
  const tunePanel = buildTunePanel(bird.tuning, [
    ["glideSpeed", 14, 40, 0.5],
    ["sinkRate", 0.3, 4, 0.1],
    ["divePower", 0.2, 3, 0.05],
    ["climbPower", 0.3, 2.5, 0.05],
    ["dragK", 0.1, 1.5, 0.05],
    ["liftGain", 0, 6, 0.1],
    ["ridgeLookahead", 0, 150, 5],
    ["ridgeEps", 6, 40, 2],
    ["windGain", 0, 15, 0.5],
    ["windDrift", 0, 2, 0.1],
    ["minSpeed", 8, 20, 0.5],
    ["maxSpeed", 30, 160, 1],
    ["beatLift", 0, 30, 1],
    ["beatThrust", 0, 25, 1],
    ["beatHz", 1, 6, 0.5],
    ["crashSpeed", 5, 40, 1],
  ]);
  // --- terrain render controls in the same panel: a mode button + live topo sliders ---
  const sep = document.createElement("div");
  sep.style.cssText = "border-top:1px solid #3a3360;margin:8px 0 6px;padding-top:6px;color:#c9a8ff;";
  sep.textContent = "terrain render";
  tunePanel.appendChild(sep);
  const modeBtn = document.createElement("button");
  const modes: ("ekg" | "grid" | "topo")[] = ["ekg", "grid", "topo"];
  modeBtn.textContent = `mode: ${terrainMode}  ▸`;
  modeBtn.style.cssText =
    "width:100%;margin:0 0 6px;padding:4px;background:#241d40;color:#9fe8ff;" +
    "border:1px solid #4a4070;border-radius:4px;font:12px monospace;cursor:pointer;";
  modeBtn.onclick = () => {
    terrainMode = modes[(modes.indexOf(terrainMode) + 1) % modes.length]!;
    modeBtn.textContent = `mode: ${terrainMode}  ▸`;
  };
  tunePanel.appendChild(modeBtn);
  // topo line params (live; visible effect in topo mode)
  const gt = gridTerrain as unknown as Record<string, number>;
  sliderRow(tunePanel, gt, "interval", 8, 80, 1);
  sliderRow(tunePanel, gt, "floorFade", 0, 1, 0.02);
  sliderRow(tunePanel, gt, "peakGain", 0.5, 3, 0.1);
  sliderRow(tunePanel, gt, "lineWidth", 0.5, 3, 0.1);

  // --- BIRD BUFFET (phase 3): wind-scaled VISUAL judder of the drawn bird (rock + render-only tremor).
  // Writes straight into bird.tuning; camera is untouched (the tremor never enters bird.pos). ---
  const bt = bird.tuning as unknown as Record<string, number>;
  panelSep(tunePanel, "bird — buffet");
  sliderRow(tunePanel, bt, "buffetGain", 0, 3, 0.1);    // master shake scale (0 = off, 1 = default)
  sliderRow(tunePanel, bt, "buffetWindRef", 4, 30, 1);  // wind m/s mapped to full buffet (lower = judders sooner)
  sliderRow(tunePanel, bt, "rockCapDeg", 0, 25, 1);     // max visual roll from the rock (deg)

  // --- WIND controls: global-wind ACTIVITY (the altitude atmosphere) + RENDERING, then the two OFF layers ---
  const wr = wind as unknown as Record<string, number>; // live access to the Wind instance's tunable fields
  panelSep(tunePanel, "global wind — activity");
  sliderRow(tunePanel, windProfileParams, "loScale", 0, 2, 0.05);   // valley wind fraction
  sliderRow(tunePanel, windProfileParams, "hiScale", 0, 3, 0.05);   // aloft wind strength (also ridge-lift strength)
  sliderRow(tunePanel, windProfileParams, "altLo", 0, 300, 10);     // altitude where calm ends (<altHi)
  sliderRow(tunePanel, windProfileParams, "altHi", 320, 800, 10);   // altitude of full strength
  panelSep(tunePanel, "global wind — render");
  sliderRow(tunePanel, wr, "dotPx", 1, 8, 0.2);                     // mote size
  sliderRow(tunePanel, wr, "clearance", 5, 150, 5);                 // band height above terrain
  sliderRow(tunePanel, wr, "vSpread", 10, 150, 5);                  // band thickness / tail
  sliderRow(tunePanel, wr, "homeBias", 1, 5, 0.2);                  // hug-terrain bias (higher = more hug)
  // per-tier wind RENDER MODES (phase 1): switching is wired end-to-end but B/C currently fall through to
  // the comet/modulate look in the engine — no visible change yet; divergent geometry is a later phase.
  panelSep(tunePanel, "wind — render modes");
  cycleBtn(tunePanel, "FAR", FAR_MODES, "comet", (m) => wind.setFarMode(m));
  cycleBtn(tunePanel, "NEAR", NEAR_MODES, "comet", (m) => wind.setNearMode(m));
  cycleBtn(tunePanel, "WAKE", WAKE_MODES, "modulate", (m) => wind.setWakeMode(m));
  // per-mode tuning DIALS removed from the panel for now (decluttered while running comet/comet/modulate).
  // The fields are still live on __wind.* — e.g. __wind.dashLenM, __wind.spreadAngleDeg, __wind.ringRate.
  panelSep(tunePanel, "local sphere + wake (off — solving global)");
  toggleBtn(tunePanel, "local sphere", false, (v) => wind.setShowNear(v));
  toggleBtn(tunePanel, "wake", false, (v) => wind.setShowWake(v));
  sliderRow(tunePanel, wr, "ambientNearFloor", 0, 1, 0.05);         // sphere stick (1 = full global wind)
  sliderRow(tunePanel, wr, "nearJitter", 0, 0.6, 0.02);            // per-mote direction randomness (rad; 0 = uniform)
  sliderRow(tunePanel, wr, "foreStretch", 1, 5, 0.1);              // sphere forward reach (overlap with the global-wind fade)
  sliderRow(tunePanel, wr, "swirlGain", 0, 2, 0.1);                // wake vortex strength
  sliderRow(tunePanel, wr, "wingSpan", 0, 30, 1);                  // wake vortex tip spacing
  sliderRow(tunePanel, wr, "heatRef", 4, 50, 2);                   // touched-air selectivity (higher = less warm)

  document.body.appendChild(tunePanel);
  window.addEventListener("keydown", (e) => {
    if (e.code === "KeyT") {
      tunePanel.style.display =
        tunePanel.style.display === "none" ? "block" : "none";
    }
    if (e.code === "KeyP") autopilot = !autopilot; // toggle hands-off autopilot <-> manual
    if (e.code === "Space") {
      flapHeld = true;
      e.preventDefault(); // stop Space from scrolling the page
    }
  });
  window.addEventListener("keyup", (e) => {
    if (e.code === "Space") flapHeld = false;
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
      sampleCount: SAMPLES,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    msaaTex.destroy();
    msaaTex = device.createTexture({
      size: [pxW, pxH],
      format: HDR_FORMAT,
      sampleCount: SAMPLES,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    sceneTex.destroy();
    sceneTex = device.createTexture({
      size: [pxW, pxH],
      format: HDR_FORMAT,
      usage:
        GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    bloom.resize(pxW, pxH); // recreate the downsampled bloom-chain textures + bind groups
  };
  window.addEventListener("resize", resize);

  // Scripted pitch wobble (THIS task): auto nose up/down so the screenshot proves the camera keeps
  // the ground framed no matter how hard the BIRD pitches. Off in AUTOPILOT mode (the pilot flies).
  // PLAYABLE: no scripted pitch wobble — the player has clean manual control from the first frame.
  (window as any).__autoWobble = false;
  let wobbleT = 0;

  let frame = 0;
  let fps = 0;
  let reached = 0; // targets reached this run (HUD score)
  let fovCur = FOV_Y; // eased per-frame toward FOV_Y + speed kick
  // smoothed horizontal MOMENTUM (low-passed bird velocity) — the camera aims along this, not heading,
  // so buffet/gust jitter and stall thrashing don't shake the view.
  let momX = bird.vel[0],
    momZ = bird.vel[2];
  const loop = new FrameLoop((dt) => {
    fps = fps * 0.9 + (1 / Math.max(dt, 1e-3)) * 0.1;

    // map input: AUTOPILOT flies (manual controls OFF this pass); else mouse-steer.
    if (autopilot) {
      const cmd = auto.update(bird, dt);
      input.yawRate = cmd.yawRate;
      input.pitchTarget = cmd.pitchTarget;
      (window as any).__autoMode = auto.mode;
    } else {
      input.yawRate = applyDead(mouseX) * YAW_GAIN;
      input.pitchTarget = GLIDE_TRIM + applyDead(mouseY) * PITCH_RANGE; // INVERTED yoke: cursor BELOW center (under bird) = nose-up/climb; ABOVE = dive
    }
    input.flap = !autopilot && flapHeld; // powered wingbeat (manual flight only)

    // scripted pitch wobble drives the bird hard up/down; the camera must NOT follow the pitch.
    // PITCH ONLY (no yaw) → heading stays 0 so the world-axis EKG rows render as clean horizontal
    // stacked lines; the wobble is purely the ground-lock proof. (manual mode only)
    if (!autopilot && (window as any).__autoWobble) {
      wobbleT += dt;
      input.pitchTarget = Math.sin(wobbleT * 1.1) * 0.65; // sweeps near the full attitude range
      input.yawRate = 0;
    }

    // v13: push the latest resolved fluid field into wind.ts BEFORE integrate/draw consume windAt, so the
    // bird physics + motes read the fluid this frame. read() also regulates the scale toward the flyable
    // band. Null until the first readback resolves → windAt falls back to analytic curl-noise (first frames).
    const field = fluidWind.read();
    if (field) {
      const [oX, oZ] = fluidWind.originXZ;
      setFluidField(
        field.u,
        field.v,
        fluidWind.gridW,
        fluidWind.gridH,
        oX,
        oZ,
        fluidWind.cellMeters,
        fluidWind.currentScale,
      );
    }

    bird.integrate(dt, input);

    // fly-to-target: reached → score + respawn ahead. (No ground-reset teleport — the bird now FLAPS to
    // climb out of low passes; the altitude clamp just lets it skim the terrain.)
    if (target.checkReached(bird.pos, REACH_RADIUS)) {
      reached++;
      target.respawn(bird.pos[0], bird.pos[2], bird.heading);
    }

    // altitude-adaptive camera: low clearance → eye drops + look flattens (ground rush);
    // high clearance → exact v3/v4 framing. cam.update() smooths the transition.
    const cf = Math.min(
      1,
      Math.max(
        0,
        (bird.lastClearance - CAM_LOW.clearance) /
          (CAM_HIGH.clearance - CAM_LOW.clearance),
      ),
    );
    cam.followHeight = CAM_LOW.height + cf * (CAM_HIGH.height - CAM_LOW.height);
    cam.lookPitch =
      ((CAM_LOW.pitchDeg + cf * (CAM_HIGH.pitchDeg - CAM_LOW.pitchDeg)) *
        Math.PI) /
      180;

    // camera follows the bird
    cam.target = [bird.pos[0], bird.pos[1], bird.pos[2]];
    // Aim along the bird's MOMENTUM (smoothed horizontal velocity), not heading or the raw ground-track.
    // A low-pass (~0.4s) on the velocity vector filters the buffet/gust jitter and stall thrashing that
    // made the view shake, while still swinging to follow real turns and wind crab.
    const kMom = Math.min(1, dt * 2.5);
    momX += (bird.vel[0] - momX) * kMom;
    momZ += (bird.vel[2] - momZ) * kMom;
    const ml = Math.hypot(momX, momZ) || 1;
    cam.forward = [momX / ml, 0, momZ / ml];
    cam.update();

    // speed FOV kick: diving widens the view (eases, never snaps).
    const speedFrac = Math.min(
      1,
      Math.max(
        0,
        (bird.lastSpeed - bird.tuning.glideSpeed) /
          (bird.tuning.maxSpeed - bird.tuning.glideSpeed),
      ),
    );
    fovCur += (FOV_Y + speedFrac * FOV_KICK - fovCur) * Math.min(1, dt * 5);

    const proj = perspective(fovCur, pxW / pxH, NEAR, FAR);
    const view = cam.viewMatrix();
    const viewProj = multiply(proj, view);

    const colorView = msaaTex.createView(); // all scene passes render into the HDR MSAA target
    const resolveView = sceneTex.createView(); // final scene pass resolves into the HDR scene texture
    const swapView = ctx.getCurrentTexture().createView(); // bloom composite writes the final image here
    const depthView = depthTex.createView();
    const eye = cam.getEye();

    const enc = device.createCommandEncoder();
    // v13: step the GPU fluid (the evolving wind source) over the bird-local moving window, and enqueue
    // its velocity buffers for async readback (consumed next frames via fluidWind.read()). Recorded into
    // the same frame encoder; afterSubmit() (post-submit) kicks the non-awaited maps.
    fluidWind.step(enc, dt, bird.pos[0], bird.pos[2]);
    // terrain pass: clears color+depth. CAMERA-RELATIVE rows — build them around the camera ground
    // point using the SMOOTHED view basis (forward/right) so the stack stays screen-horizontal.
    const camGround = cam.groundPos();
    const camFwd = cam.forwardHoriz();
    const camRight = cam.rightHoriz();
    if (terrainMode !== "ekg") {
      // WORLD-STATIC renderer (grid wireframe or topo contours); first pass, clears color+depth.
      gridTerrain.mode = terrainMode;
      gridTerrain.draw(enc, colorView, depthView, viewProj, camGround, eye, {
        r: SKY[0], g: SKY[1], b: SKY[2], a: 1,
      });
    } else {
      terrain.draw(
        enc,
        colorView,
        depthView,
        viewProj,
        camGround,
        camFwd,
        camRight,
        eye,
        {
          r: SKY[0],
          g: SKY[1],
          b: SKY[2],
          a: 1,
        },
      );
    }
    // wind pass: loads color+depth (no clear); drifting neon DOT motes over the ridges (depth-tested,
    // no depth-write) — advected by the bird's sim time so the drawn field matches the field that pushes.
    // ON by default (window.__showWind(false) hides them); same fluid field drives the bird physics.
    if (showWind) {
      wind.draw(
        enc,
        colorView,
        depthView,
        viewProj,
        camGround,
        camFwd,
        camRight,
        eye,
        bird.simTime,
        SKY,
        0.5 / 1400,
        pxW / pxH,
        bird.pos,
        bird.vel,
      ); // −25% mote fog (was 1/1400), kept coupled to the terrain's; bird.pos = near-sphere center; bird.vel orients the wake stir
    }
    // bird pass: loads color+depth, depth-tested → ridges occlude the bird.
    bird.draw(enc, colorView, depthView, viewProj);
    // trees pass: mountaintop forests, depth-tested → ridges occlude them; rebuilt on cell crossing.
    trees.draw(
      enc,
      colorView,
      depthView,
      viewProj,
      camGround,
      eye,
      bird.simTime,
      0.5 / 1100, // fog density == terrain's → trees haze identically with distance
    );
    // target beam: always-on-top amber waypoint, drawn after the bird so it composites over the scene.
    target.draw(enc, colorView, depthView, viewProj, eye, bird.simTime);
    // altitude plumb-line + ground diamond under the bird (depth-tested → ridges occlude it).
    // LAST scene pass — resolves the HDR MSAA target into the single-sample scene texture.
    marker.draw(
      enc,
      colorView,
      depthView,
      viewProj,
      [bird.pos[0], bird.pos[1], bird.pos[2]],
      bird.pos[1] - bird.lastClearance,
      bird.simTime,
      resolveView,
    );
    // BLOOM: threshold → separable blur → composite/tone-map the HDR scene into the swapchain.
    bloom.apply(enc, resolveView, swapView);
    device.queue.submit([enc.finish()]);
    // v13: kick the non-awaited readback maps for the slots copied this frame (must be AFTER submit).
    fluidWind.afterSubmit();

    (window as any).__camPos = eye;
    (window as any).__birdPos = bird.pos;
    (window as any).__birdPitch = bird.pitch; // live pitch (rad)
    (window as any).__birdHeading = bird.heading; // live heading (rad) — capture harness waits for a real turn
    (window as any).__birdGroundTrack = bird.lastGroundTrack; // actual travel dir (rad) — drift proof
    (window as any).__birdWind = bird.lastWind; // [wx,wz] m/s — overlay/diagnostics
    (window as any).__birdVario = bird.lastVario; // climb m/s — lift proof
    (window as any).__birdBuffet = bird.buffetOffset; // [tx,ty,tz] render-only tremor (NEVER added to bird.pos)
    frame++;
    const headingDeg = ((bird.heading * 180) / Math.PI) % 360;
    const trackDeg = (bird.lastGroundTrack * 180) / Math.PI;
    // signed drift = ground-track − heading wrapped to (-180,180]; this is the felt-wind number.
    let drift = trackDeg - (bird.heading * 180) / Math.PI;
    drift = ((((drift + 180) % 360) + 360) % 360) - 180;
    const vario = bird.lastVario;
    const varioStr = `${vario >= 0 ? "+" : ""}${vario.toFixed(1)}`;
    const windSpeed = Math.hypot(bird.lastWind[0], bird.lastWind[1]);
    // target nav readout: horizontal distance + relative steer bearing (◄ left / ► right / ▲ ahead).
    const tDist = target.distanceTo(bird.pos);
    let tRel =
      ((Math.atan2(target.x - bird.pos[0], target.z - bird.pos[2]) -
        bird.heading) *
        180) /
      Math.PI;
    tRel = ((((tRel + 180) % 360) + 360) % 360) - 180;
    const tArrow = tRel > 5 ? "►" : tRel < -5 ? "◄" : "▲";
    overlay.textContent =
      `vector-system — bird3d (wind glider · fly to target)${autopilot ? `   AUTO: ${auto.mode} (click/P=manual)` : "   MANUAL (P=autopilot)"}${bird.lastFlapping ? "   ▲ FLAP" : ""}${bird.lastCrashing ? "   ✖ CRASH" : ""}\n` +
      `TARGET: ${tDist.toFixed(0)} m   steer ${tArrow} ${Math.abs(tRel).toFixed(0)}°   reached: ${reached}\n` +
      `alt over terrain: ${bird.lastClearance.toFixed(0)} m   air: ${bird.lastSpeed.toFixed(0)} m/s\n` +
      `vario: ${varioStr} m/s ${vario > 0.5 ? "▲" : vario < -0.5 ? "▼" : "—"}   updraft: +${bird.lastUpdraft.toFixed(1)} m/s\n` +
      `heading: ${headingDeg.toFixed(0)}°   ground-track: ${trackDeg.toFixed(0)}°   DRIFT: ${drift >= 0 ? "+" : ""}${drift.toFixed(0)}°\n` +
      `wind: ${bird.lastWind[0].toFixed(1)}, ${bird.lastWind[1].toFixed(1)} m/s  (|${windSpeed.toFixed(1)}|)\n` +
      `fps: ${fps.toFixed(0)}   frame ${frame}   (steer=mouse · SPACE=flap · cursor under bird=climb, over=dive · T=tuning)`;

    // compass overlay: large vectors — heading (cyan), ground-track (yellow), wind (magenta).
    drawCompass(
      compassCtx,
      bird.heading,
      bird.lastGroundTrack,
      bird.lastWind,
      windSpeed,
      drift,
    );
  });

  // v13 EVOLUTION PROBE: sample windAt at a FIXED world point with a FIXED t. Because t is fixed, any
  // change across calls comes ONLY from the fluid readback replacing the field each frame — that IS the
  // proof the wind is the evolving fluid (the old analytic curl-noise at a fixed point was quasi-static).
  (window as any).__windAt = (x: number, z: number) => windAt(x, z, 0);

  // LIFT PROBE: the exact ridge+thermal updraft the bird RIDES (and the autopilot senses) at a world point.
  // tuneOverride lets a test sample the field with different ridgeLookahead/ridgeEps (e.g. lookahead=0) to
  // measure how much WIDER the L+B lift band is than the local-only band. t fixed at 0 for repeatability.
  (window as any).__updraftAt = (
    x: number,
    z: number,
    tuneOverride?: Record<string, number>,
  ) => updraftAt(x, z, 0, terrain, { ...bird.tuning, ...(tuneOverride ?? {}) });

  // SLIPSTREAM live tuning + probes (debug): e.g. __wind.swirlGain = 1.2, __wind.wingSpan = 14,
  // __wind.ambientNearFloor = 0.1, __wind.wingEmitFrac = 0.6. __nearWake(x,y,z) returns the wake velocity at a
  // world point (used by the live gate to prove the two wingtip vortices counter-rotate); __nearFrame() gives
  // the current bird pos + motion axis + wing-right vector.
  (window as any).__wind = wind;
  // GLOBAL WIND atmosphere live tuning: __windProfile({loScale, hiScale, altLo, altHi}). Affects BOTH the bird
  // (drift + ridge lift) and the motes — one shared altitude profile. e.g. deader valleys: {loScale: 0.2}.
  (window as any).__windProfile = (p: Record<string, number>) => setWindProfile(p);
  (window as any).__windProfileAt = (y: number) => windProfile(y); // read the altitude curve (gate + tuning)
  // BUFFET live tuning (phase 3): set buffet tuning fields the same way the T-panel sliders do, e.g.
  // __birdTune({ buffetGain: 0 }) to kill the visual buffet, or { buffetGain: 3 } to slam it. Read-back via
  // __birdBank (visual roll) and __birdBuffet (render-only position tremor — proves it's NOT in bird.pos).
  (window as any).__birdTune = (p: Partial<typeof bird.tuning>) => Object.assign(bird.tuning, p);
  (window as any).__nearWake = (x: number, y: number, z: number) => wind.sampleWake(x, y, z);
  (window as any).__nearFrame = () => wind.nearFrame();
  // per-tier RENDER MODE switches (phase 1): drive the same setters the T-panel buttons call.
  // e.g. __farMode("chevron"), __nearMode("flecks"), __wakeMode("rings"). B/C fall through to comet now.
  (window as any).__farMode = (m: FarMode) => wind.setFarMode(m);
  (window as any).__nearMode = (m: NearMode) => wind.setNearMode(m);
  (window as any).__wakeMode = (m: WakeMode) => wind.setWakeMode(m);

  // perf A/B handle: window.__trees.enabled = false disables the forest pass; window.__trees.treeCount
  // reports how many trees the current window baked.
  (window as any).__trees = trees;
  // toggle the wind motes back on from the console: __showWind(true)
  (window as any).__showWind = (v: boolean) => { showWind = v; };
  // switch terrain renderer from the console: __terrainMode("ekg" | "grid" | "topo")
  (window as any).__terrainMode = (m: "ekg" | "grid" | "topo") => { terrainMode = m; };

  // TERRAIN SELF-CHECK: does the GPU terrain the bird COLLIDES against equal the one it's DRAWN on?
  // Runs once at boot and logs a verdict; re-run live with __terrainCheck(). A FAIL means the bird is
  // hitting invisible terrain (CPU/GPU height fields disagree); a PASS with a stale scene means hard-reload.
  const runTerrainCheck = async () => {
    const r = await checkTerrain(device, (x, z) => terrain.sampleHeight(x, z));
    const tag = r.maxDiff < 1 ? "PASS" : "FAIL";
    console.log(
      `[terrain-selfcheck] ${tag}  points=${r.points}  maxDiff=${r.maxDiff.toFixed(2)}m  meanDiff=${r.meanDiff.toFixed(3)}m\n` +
        `  worst @ (${r.worst.x}, ${r.worst.z})  cpu=${r.worst.cpu.toFixed(1)}m  gpu=${r.worst.gpu.toFixed(1)}m\n` +
        `  ${r.verdict}`,
    );
    return r;
  };
  (window as any).__terrainCheck = runTerrainCheck;
  void runTerrainCheck();

  loop.start();
  (window as any).__birdBooted = true;
}

// Build a floating slider panel bound directly to a live tuning object.
// rows: [key, min, max, step][] — each slider writes tuning[key] on input.
function buildTunePanel(
  tuning: Record<string, number>,
  rows: [string, number, number, number][],
): HTMLDivElement {
  const panel = document.createElement("div");
  panel.id = "tune";
  panel.style.cssText =
    "position:fixed;right:12px;top:12px;display:none;padding:10px 12px;" +
    "background:rgba(8,6,20,0.85);border:1px solid #3a3360;border-radius:6px;" +
    "font:12px/1.6 monospace;color:#9fe8ff;z-index:10;min-width:240px;";
  for (const [key, min, max, step] of rows) sliderRow(panel, tuning, key, min, max, step);
  return panel;
}

// a labelled separator heading inside the tuning panel.
function panelSep(panel: HTMLElement, text: string): void {
  const sep = document.createElement("div");
  sep.style.cssText = "border-top:1px solid #3a3360;margin:8px 0 6px;padding-top:6px;color:#c9a8ff;";
  sep.textContent = text;
  panel.appendChild(sep);
}

// a toggle button bound to a boolean setter; reused for the wind layer toggles.
function toggleBtn(panel: HTMLElement, label: string, initial: boolean, onSet: (v: boolean) => void): void {
  let on = initial;
  const btn = document.createElement("button");
  btn.style.cssText =
    "width:100%;margin:0 0 6px;padding:4px;background:#241d40;color:#9fe8ff;" +
    "border:1px solid #4a4070;border-radius:4px;font:12px monospace;cursor:pointer;";
  const render = () => { btn.textContent = `${label}: ${on ? "ON" : "OFF"}`; };
  render();
  btn.onclick = () => { on = !on; onSet(on); render(); };
  panel.appendChild(btn);
}

// a cycle button that advances through a list of string options on each click (wraps), then fires
// onSet with the new value; reused for the per-tier wind render modes (FAR/NEAR/WAKE).
function cycleBtn<T extends string>(
  panel: HTMLElement,
  label: string,
  opts: readonly T[],
  initial: T,
  onSet: (v: T) => void,
): void {
  let i = Math.max(0, opts.indexOf(initial));
  const btn = document.createElement("button");
  btn.style.cssText =
    "width:100%;margin:0 0 6px;padding:4px;background:#241d40;color:#9fe8ff;" +
    "border:1px solid #4a4070;border-radius:4px;font:12px monospace;cursor:pointer;";
  const render = () => { btn.textContent = `${label}: ${opts[i]} ▸`; };
  render();
  btn.onclick = () => { i = (i + 1) % opts.length; onSet(opts[i]!); render(); };
  panel.appendChild(btn);
}

// one slider row bound to obj[key]; reused for bird.tuning and the terrain render controls.
function sliderRow(
  panel: HTMLElement,
  obj: Record<string, number>,
  key: string,
  min: number,
  max: number,
  step: number,
): void {
  const row = document.createElement("div");
  const label = document.createElement("span");
  const val = document.createElement("span");
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = String(min);
  slider.max = String(max);
  slider.step = String(step);
  slider.value = String(obj[key]);
  slider.style.cssText = "width:110px;vertical-align:middle;margin:0 6px;";
  label.textContent = key.padEnd(11);
  val.textContent = String(obj[key]);
  slider.addEventListener("input", () => {
    obj[key] = Number(slider.value);
    val.textContent = slider.value;
  });
  row.append(label, slider, val);
  panel.appendChild(row);
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
  driftDeg: number,
): void {
  const w = ctx.canvas.width,
    h = ctx.canvas.height;
  const cx = w / 2,
    cy = h / 2;
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
  arrow(heading, 66, "rgba(80,220,255,0.95)", 3); // heading — cyan
  arrow(track, 66, "rgba(255,225,70,0.95)", 3); // ground-track — yellow

  ctx.font = "11px monospace";
  ctx.fillStyle = "#9fe8ff";
  ctx.fillText("heading", 8, 16);
  ctx.fillStyle = "#ffe146";
  ctx.fillText("track", 8, 30);
  ctx.fillStyle = "#e65ae6";
  ctx.fillText("wind", 8, 44);
  ctx.fillStyle = "#fff";
  ctx.fillText(
    `drift ${driftDeg >= 0 ? "+" : ""}${driftDeg.toFixed(0)}°`,
    8,
    h - 10,
  );
}

boot().catch((e) => {
  const overlay = document.getElementById("overlay");
  if (overlay) overlay.textContent = "boot error: " + (e as Error).message;
  throw e;
});
