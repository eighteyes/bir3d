var u=Object.defineProperty;var p=(n,e,t)=>e in n?u(n,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):n[e]=t;var l=(n,e,t)=>p(n,typeof e!="symbol"?e+"":e,t);(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const r of document.querySelectorAll('link[rel="modulepreload"]'))o(r);new MutationObserver(r=>{for(const i of r)if(i.type==="childList")for(const a of i.addedNodes)a.tagName==="LINK"&&a.rel==="modulepreload"&&o(a)}).observe(document,{childList:!0,subtree:!0});function t(r){const i={};return r.integrity&&(i.integrity=r.integrity),r.referrerPolicy&&(i.referrerPolicy=r.referrerPolicy),r.crossOrigin==="use-credentials"?i.credentials="include":r.crossOrigin==="anonymous"?i.credentials="omit":i.credentials="same-origin",i}function o(r){if(r.ep)return;r.ep=!0;const i=t(r);fetch(r.href,i)}})();async function W(){if(!("gpu"in navigator))throw new Error("WebGPU unavailable: navigator.gpu missing");const n=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!n)throw new Error("WebGPU unavailable: no adapter");const e=n.features.has("timestamp-query"),t=e?["timestamp-query"]:[],o=await n.requestDevice({requiredFeatures:t});return o.addEventListener("uncapturederror",r=>console.error("[WebGPU uncaptured]",r.error)),{adapter:n,device:o,hasTimestampQuery:e}}function B(n,e,t="main"){const o=n.createShaderModule({code:e});return n.createComputePipeline({layout:"auto",compute:{module:o,entryPoint:t}})}function M(n,e,t,o,r,i=64,a){const d=n.createBindGroup({layout:t.getBindGroupLayout(0),entries:o.map((c,f)=>({binding:f,resource:{buffer:c}}))}),s=e.beginComputePass(a?{timestampWrites:a}:void 0);s.setPipeline(t),s.setBindGroup(0,d),s.dispatchWorkgroups(Math.ceil(r/i)),s.end()}const h=`// addone.wgsl — proves the compute pipeline end-to-end: out[i] = in[i] + 1.
@group(0) @binding(0) var<storage, read>       inBuf  : array<f32>;
@group(0) @binding(1) var<storage, read_write> outBuf : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&inBuf)) { return; }
  outBuf[i] = inBuf[i] + 1.0;
}
`,g=`// bird_update.wgsl — GPU-integrated single-bird physics over the live fluid wind field.
// One compute pass per frame; reads the live fluid u,v storage buffers in-shader (no readback).
// Responsibilities:
//   - Read bird state {pos:vec2, vel:vec2} + per-frame Intent {impulse, turn} + fluid u,v + BirdParams.
//   - windAt(pos): manual bilinear sample of (u,v) with TOROIDAL wrap (NOT advect's Stam clamp).
//   - Integrate: vel += windAt(pos)*windCoupling*dt + impulse; rotate vel by turn; vel *= drag;
//     pos += vel*dt; toroidal-wrap pos into [0,W)x[0,H).
//   - Write current pos into the trail ring buffer at trailWrite. Single bird: only gid.x==0 runs.

struct BirdState {
  pos : vec2<f32>,
  vel : vec2<f32>,
};

// Per-frame input mapped from the active control scheme on the CPU. impulse is a one-shot
// velocity burst (zeroed by the host the frame after it fires); turn rotates vel (radians).
struct Intent {
  impulse : vec2<f32>,
  turn    : f32,
  pad0    : f32,
};

// w,h = fluid grid (interior cells); the bird lives in world coords [0,w)x[0,h).
struct BirdParams {
  w           : u32,
  h           : u32,
  dt          : f32,
  windCoupling: f32,
  drag        : f32,
  trailWrite  : u32,
  pad0        : f32,
  pad1        : f32,
};

@group(0) @binding(0) var<uniform>             P     : BirdParams;
@group(0) @binding(1) var<uniform>             I     : Intent;
@group(0) @binding(2) var<storage, read>       u     : array<f32>;
@group(0) @binding(3) var<storage, read>       v     : array<f32>;
@group(0) @binding(4) var<storage, read_write> bird  : array<BirdState>;
@group(0) @binding(5) var<storage, read_write> trail : array<vec2<f32>>;

// Bordered (W+2)*(H+2) layout, interior cell (i in 1..=w, j in 1..=h) at idx = i + (W+2)*j.
fn cell(i : u32, j : u32) -> u32 { return i + (P.w + 2u) * j; }

fn wrapI(i : i32, n : i32) -> u32 { return u32(((i % n) + n) % n); }

// Bilinear sample of the velocity field at world (x,y) with toroidal wrap over the interior grid.
// World coords are cell-centered: world x in [0,w) maps to interior column index (1 + floor(x)).
fn windAt(pos : vec2<f32>) -> vec2<f32> {
  let wf = f32(P.w);
  let hf = f32(P.h);
  // Wrap world position into [0,w)x[0,h).
  let px = pos.x - floor(pos.x / wf) * wf;
  let py = pos.y - floor(pos.y / hf) * hf;

  // Cell-center reference: sample between centers at (col 0.5 .. w-0.5).
  let fx = px - 0.5;
  let fy = py - 0.5;
  let i0 = i32(floor(fx));
  let j0 = i32(floor(fy));
  let s1 = fx - f32(i0);
  let t1 = fy - f32(j0);
  let s0 = 1.0 - s1;
  let t0 = 1.0 - t1;

  let wi = i32(P.w);
  let hi = i32(P.h);
  // +1 offset moves interior column 0 to bordered index 1.
  let ia = 1u + wrapI(i0,     wi);
  let ib = 1u + wrapI(i0 + 1, wi);
  let ja = 1u + wrapI(j0,     hi);
  let jb = 1u + wrapI(j0 + 1, hi);

  let uu = s0 * (t0 * u[cell(ia, ja)] + t1 * u[cell(ia, jb)])
         + s1 * (t0 * u[cell(ib, ja)] + t1 * u[cell(ib, jb)]);
  let vv = s0 * (t0 * v[cell(ia, ja)] + t1 * v[cell(ia, jb)])
         + s1 * (t0 * v[cell(ib, ja)] + t1 * v[cell(ib, jb)]);
  return vec2<f32>(uu, vv);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x > 0u) { return; } // single bird

  var s = bird[0];

  // Wind push.
  s.vel = s.vel + windAt(s.pos) * P.windCoupling * P.dt;
  // Input burst (one-shot; host zeroes impulse the next frame).
  s.vel = s.vel + I.impulse;

  // Turn: rotate vel by I.turn (radians).
  let c = cos(I.turn);
  let sn = sin(I.turn);
  s.vel = vec2<f32>(c * s.vel.x - sn * s.vel.y, sn * s.vel.x + c * s.vel.y);

  // Drag, integrate.
  s.vel = s.vel * P.drag;
  s.pos = s.pos + s.vel * P.dt;

  // Toroidal wrap into [0,w)x[0,h).
  let wf = f32(P.w);
  let hf = f32(P.h);
  s.pos.x = s.pos.x - floor(s.pos.x / wf) * wf;
  s.pos.y = s.pos.y - floor(s.pos.y / hf) * hf;

  bird[0] = s;
  trail[P.trailWrite] = s.pos;
}
`,v=`// scene.wgsl — bird scene render passes (backdrop + trail + chevron), camera-relative, neon-on-dark.
// Rough feel-prototype glyphs (NOT the §4.1 vector renderer). Three entry pairs share one Camera
// uniform so all layers align in the same world→NDC mapping.
// Responsibilities:
//   - Camera uniform: cameraPos (world center), viewSize (world units across view), grid w,h.
//   - backdrop_vs/fs: fullscreen triangle; per-fragment map screen→world for the camera sub-window,
//     toroidal-wrap into the grid, sample dye magnitude → dim neon flow, brightness-capped (§7.2).
//   - trail_vs/fs: draw the trail ring buffer as a line strip, fading by recency (alpha ramp).
//   - chevron_vs/fs: read the bird state buffer in-shader, build a neon chevron oriented to vel,
//     drawn relative to the camera (triangle-list, 3 verts).

struct Camera {
  cameraPos : vec2<f32>, // world-space center of the view
  viewSize  : vec2<f32>, // world units spanned by the view (x,y)
  w         : u32,
  h         : u32,
  trailLen  : u32,
  trailHead : u32,       // index of the most-recently-written trail sample
};

struct BirdState {
  pos : vec2<f32>,
  vel : vec2<f32>,
};

const BRIGHTNESS_CEILING : f32 = 0.8;

fn capBrightness(col : vec3<f32>) -> vec3<f32> {
  let lum = dot(col, vec3<f32>(0.2126, 0.7152, 0.0722));
  if (lum > BRIGHTNESS_CEILING) { return col * (BRIGHTNESS_CEILING / lum); }
  return col;
}

// world → NDC given the camera window. y flips so +world-y is up on screen.
fn worldToNdc(world : vec2<f32>, cam : Camera) -> vec2<f32> {
  let rel = (world - cam.cameraPos) / (cam.viewSize * 0.5);
  return vec2<f32>(rel.x, -rel.y);
}

// ---------- backdrop ----------
@group(0) @binding(0) var<uniform>       CAM : Camera;
@group(0) @binding(1) var<storage, read> dye : array<f32>;

struct VsUv {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

@vertex
fn backdrop_vs(@builtin(vertex_index) vi : u32) -> VsUv {
  var out : VsUv;
  let x = f32((vi << 1u) & 2u);
  let y = f32(vi & 2u);
  out.pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2<f32>(x, y); // [0,2] clipped to [0,1] across the screen
  return out;
}

fn dyeCell(i : u32, j : u32) -> u32 { return i + (CAM.w + 2u) * j; }
fn wrapU(i : i32, n : i32) -> u32 { return u32(((i % n) + n) % n); }

@fragment
fn backdrop_fs(in : VsUv) -> @location(0) vec4<f32> {
  // Screen uv [0,1] → world within the camera window (uv.y up).
  let half = CAM.viewSize * 0.5;
  let world = CAM.cameraPos + vec2<f32>((in.uv.x - 0.5) * CAM.viewSize.x,
                                        (0.5 - in.uv.y) * CAM.viewSize.y);
  // Toroidal-wrap into the grid, nearest interior cell.
  let wf = f32(CAM.w);
  let hf = f32(CAM.h);
  let px = world.x - floor(world.x / wf) * wf;
  let py = world.y - floor(world.y / hf) * hf;
  let i = 1u + wrapU(i32(floor(px)), i32(CAM.w));
  let j = 1u + wrapU(i32(floor(py)), i32(CAM.h));
  let d = dye[dyeCell(i, j)];

  let t = clamp(1.0 - exp(-max(d, 0.0)), 0.0, 1.0);
  let base = vec3<f32>(0.015, 0.02, 0.03);
  let flow = vec3<f32>(0.05, 0.28, 0.45); // dim teal flow, dialed back from the debug blob
  var col = base + flow * (t * 0.6);
  return vec4<f32>(capBrightness(col), 1.0);
}

// ---------- trail ----------
@group(0) @binding(0) var<uniform>       CAM_T : Camera;
@group(0) @binding(1) var<storage, read> trail : array<vec2<f32>>;

struct VsTrail {
  @builtin(position) pos : vec4<f32>,
  @location(0) fade : f32,
};

@vertex
fn trail_vs(@builtin(vertex_index) vi : u32) -> VsTrail {
  var out : VsTrail;
  // vi runs oldest→newest along the strip; recency = how close to the head.
  let n = CAM_T.trailLen;
  // age 0 = newest (head), age n-1 = oldest.
  let idx = (CAM_T.trailHead + 1u + vi) % n; // start just after head = oldest
  let age = f32(n - 1u - vi); // 0 at the newest end
  var p = trail[idx];

  // Unwrap the sample to the camera's window so a seam crossing doesn't streak across screen.
  let half = CAM_T.viewSize * 0.5;
  let wf = f32(CAM_T.w);
  let hf = f32(CAM_T.h);
  var rel = p - CAM_T.cameraPos;
  rel.x = rel.x - round(rel.x / wf) * wf;
  rel.y = rel.y - round(rel.y / hf) * hf;
  let ndc = vec2<f32>(rel.x / half.x, -rel.y / half.y);
  out.pos = vec4<f32>(ndc, 0.0, 1.0);
  out.fade = 1.0 - age / f32(max(n - 1u, 1u)); // 1 newest → 0 oldest
  return out;
}

@fragment
fn trail_fs(in : VsTrail) -> @location(0) vec4<f32> {
  let neon = vec3<f32>(0.2, 1.0, 0.7);
  let a = clamp(in.fade, 0.0, 1.0);
  return vec4<f32>(capBrightness(neon * a), a * 0.8);
}

// ---------- chevron ----------
@group(0) @binding(0) var<uniform>       CAM_C : Camera;
@group(0) @binding(1) var<storage, read> bird  : array<BirdState>;

struct VsChev {
  @builtin(position) pos : vec4<f32>,
  @location(0) tip : f32,
};

@vertex
fn chevron_vs(@builtin(vertex_index) vi : u32) -> VsChev {
  var out : VsChev;
  let s = bird[0];
  // Heading from vel (fallback +x if nearly still).
  var dir = vec2<f32>(1.0, 0.0);
  let sp = length(s.vel);
  if (sp > 1e-4) { dir = s.vel / sp; }
  let perp = vec2<f32>(-dir.y, dir.x);

  // Chevron in world units (size in grid cells). Arrowhead: tip forward, two tails back.
  let size = 3.0;
  var local : vec2<f32>;
  if (vi == 0u) {
    local = dir * size;                          // tip
    out.tip = 1.0;
  } else if (vi == 1u) {
    local = -dir * size * 0.6 + perp * size * 0.7; // left tail
    out.tip = 0.0;
  } else {
    local = -dir * size * 0.6 - perp * size * 0.7; // right tail
    out.tip = 0.0;
  }
  let world = s.pos + local;

  // Camera-relative with seam unwrap (bird is near center; keep it stable across wraps).
  let wf = f32(CAM_C.w);
  let hf = f32(CAM_C.h);
  var rel = world - CAM_C.cameraPos;
  rel.x = rel.x - round(rel.x / wf) * wf;
  rel.y = rel.y - round(rel.y / hf) * hf;
  let half = CAM_C.viewSize * 0.5;
  out.pos = vec4<f32>(rel.x / half.x, -rel.y / half.y, 0.0, 1.0);
  return out;
}

@fragment
fn chevron_fs(in : VsChev) -> @location(0) vec4<f32> {
  // Bright neon toward the tip, slightly cooler at the tails.
  let neon = mix(vec3<f32>(0.3, 1.0, 0.9), vec3<f32>(0.9, 1.0, 0.4), in.tip);
  return vec4<f32>(capBrightness(neon), 1.0);
}
`,m=`// bird3d.wgsl — neon gliding-V bird (WebGPU 3D, depth-tested so terrain ridges occlude it).
// Responsibilities:
//   - Vertex: take a procedural bird mesh in LOCAL space (x=span lateral, y=0, z=chord forward).
//     Wings are held OUT (no flap cycle); a SUBTLE flex about the local forward (Z) axis via
//     sin(time*flexHz)*flexAmp (tiny) reads as a living glide, NOT a flap beat. Apply
//     model = T(pos) * Ryaw(heading) * Rroll(bank), then U.viewProj. Body verts do not flex.
//   - Fragment: bold emissive neon ribbons on the dark scene; brightness tapers along the
//     wing so tips glow hot. Depth-tested (less) against the stored terrain depth → occlusion.
//   - Local axes match the world chase convention: +Z = forward (heading), +X = right, +Y = up.

struct Uniforms {
  viewProj : mat4x4<f32>,
  pos : vec3<f32>,        // bird world position
  flexPhase : f32,        // idle living-flex phase (radians) — subtle, NOT a flap beat
  heading : f32,          // yaw about +Y (atan2 forward.x, forward.z)
  bank : f32,             // roll about local +Z (banks into turns)
  flexAmp : f32,          // idle flex amplitude (radians) — small living wobble
  flapPhase : f32,        // POWERED beat phase 0..PI during a downstroke (0 when idle)
  ampL : f32,             // LEFT wing beat amplitude this frame (rad) — independent of right
  ampR : f32,             // RIGHT wing beat amplitude this frame (rad)
  pitch : f32,            // nose attitude (rad), + = nose up — tilts the WHOLE model (climb/dive)
  pad1 : f32,
};

@group(0) @binding(0) var<uniform> U : Uniforms;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) glow : f32,     // 0 body .. 1 wingtip (emissive ramp)
  @location(1) edge : f32,     // ribbon-edge factor for line-like core
};

fn rotZ(a : f32) -> mat3x3<f32> {
  let c = cos(a); let s = sin(a);
  return mat3x3<f32>(vec3<f32>(c, s, 0.0), vec3<f32>(-s, c, 0.0), vec3<f32>(0.0, 0.0, 1.0));
}
fn rotY(a : f32) -> mat3x3<f32> {
  let c = cos(a); let s = sin(a);
  return mat3x3<f32>(vec3<f32>(c, 0.0, -s), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(s, 0.0, c));
}
fn rotX(a : f32) -> mat3x3<f32> { // + a = nose up: +Z (forward) tilts toward +Y
  let c = cos(a); let s = sin(a);
  return mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, c, -s), vec3<f32>(0.0, s, c));
}

// Per-vertex attributes:
//   loc0 = local position (span x, 0, chord z) in meters
//   loc1 = (spanFrac signed -1..1, partFlag: 0=body 1=wing, edgeFrac 0..1)
@vertex
fn vs(@location(0) local : vec3<f32>, @location(1) attr : vec3<f32>) -> VSOut {
  var out : VSOut;
  let spanFrac = attr.x;          // signed, -1 (left tip) .. +1 (right tip)
  let isWing = attr.y;            // 1 for wing verts, 0 for body
  let edgeFrac = attr.z;

  var p = local;

  // IDLE FLEX + POWERED FLAP. The subtle flex (flexAmp) reads as a living glide; the beat (per-wing
  // ampL/ampR, NOT one shared amplitude) is the powered downstroke. Tip lags by phase ∝ |spanFrac| so
  // both feel organic. ampL≠ampR (steering) makes the two wings beat asymmetrically — visibly.
  if (isWing > 0.5) {
    let lag = abs(spanFrac) * 1.4;                    // wingtip phase offset
    let flex = sin(U.flexPhase - lag) * U.flexAmp;    // idle living wobble
    let amp = select(U.ampR, U.ampL, spanFrac < 0.0); // LEFT (span<0) → ampL, RIGHT → ampR
    let beat = sin(U.flapPhase - lag) * amp;          // powered per-wing downstroke
    // dihedral rotation about forward (Z): magnitude per side, sign by side → wings beat up/down.
    let ang = (flex + beat) * sign(spanFrac);
    p = rotZ(ang) * p;
  }

  // Model: bank (roll about forward Z) → pitch (nose up/down about right X) → yaw (heading about Y)
  // → translate. Pitch tilts the whole V so a climb noses up and a dive noses down.
  let rolled = rotZ(U.bank) * p;
  let pitched = rotX(U.pitch) * rolled;
  let yawed = rotY(U.heading) * pitched;
  let world = yawed + U.pos;

  out.clip = U.viewProj * vec4<f32>(world, 1.0);
  out.glow = abs(spanFrac);       // tips glow hottest
  out.edge = edgeFrac;
  return out;
}

const CORE : vec3<f32> = vec3<f32>(0.55, 1.0, 0.9);   // teal-white hot core
const TIP : vec3<f32>  = vec3<f32>(0.95, 0.35, 1.0);  // magenta tips

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  // Hot core → magenta tip ramp; ribbon center brighter than edges for a line-like spine.
  let tint = mix(CORE, TIP, in.glow);
  let spine = smoothstep(0.0, 0.5, 1.0 - abs(in.edge - 0.5) * 2.0); // 1 at ribbon center
  let bright = 0.85 + spine * 0.9 + in.glow * 0.4;
  var color = tint * bright;
  color = min(color, vec3<f32>(1.4, 1.5, 1.6)); // cap (additive blend will bloom it)
  return vec4<f32>(color, 1.0);
}
`,w=`// bloom_blur.wgsl — separable Gaussian blur (one axis per pass) for the bloom chain.
// Responsibilities:
//   - Fullscreen-triangle vertex stage.
//   - 9-tap Gaussian along a per-pass direction (texelStep): horizontal pass then vertical pass.
//   - Direction + texel size come from the uniform so one shader serves H and V at any mip res.

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: VsOut;
  let c = p[vi];
  out.pos = vec4f(c, 0.0, 1.0);
  out.uv = vec2f((c.x + 1.0) * 0.5, (1.0 - c.y) * 0.5);
  return out;
}

struct Params {
  texelStep: vec2f, // (1/w, 0) for horizontal, (0, 1/h) for vertical
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSamp: sampler;
@group(0) @binding(2) var<uniform> P: Params;

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
  // 9-tap Gaussian (normalized weights, sigma ~2.5px) — wide soft falloff for neon glow.
  let w0 = 0.227027;
  let w1 = 0.194595;
  let w2 = 0.121622;
  let w3 = 0.054054;
  let w4 = 0.016216;
  let d = P.texelStep;
  var acc = textureSample(srcTex, srcSamp, in.uv).rgb * w0;
  acc += textureSample(srcTex, srcSamp, in.uv + d * 1.0).rgb * w1;
  acc += textureSample(srcTex, srcSamp, in.uv - d * 1.0).rgb * w1;
  acc += textureSample(srcTex, srcSamp, in.uv + d * 2.0).rgb * w2;
  acc += textureSample(srcTex, srcSamp, in.uv - d * 2.0).rgb * w2;
  acc += textureSample(srcTex, srcSamp, in.uv + d * 3.0).rgb * w3;
  acc += textureSample(srcTex, srcSamp, in.uv - d * 3.0).rgb * w3;
  acc += textureSample(srcTex, srcSamp, in.uv + d * 4.0).rgb * w4;
  acc += textureSample(srcTex, srcSamp, in.uv - d * 4.0).rgb * w4;
  return vec4f(acc, 1.0);
}
`,b=`// bloom_composite.wgsl — final composite + tone-map for the bloom chain.
// Responsibilities:
//   - Fullscreen-triangle vertex stage.
//   - Sample the HDR scene + the blurred bloom; combine scene + bloom*intensity (after exposure).
//   - Reinhard tone-map (soft highlight rolloff, NOT a hard clamp) so bright neon stays HUE-COLORED
//     instead of smearing to white; output to the swapchain (preferred format).

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: VsOut;
  let c = p[vi];
  out.pos = vec4f(c, 0.0, 1.0);
  out.uv = vec2f((c.x + 1.0) * 0.5, (1.0 - c.y) * 0.5);
  return out;
}

struct Params {
  intensity: f32, // bloom add weight
  exposure: f32,  // pre-tonemap scene exposure
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var sceneTex: texture_2d<f32>;
@group(0) @binding(1) var bloomTex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> P: Params;

// Highlight-only rolloff: identity below the knee K, exponential compression above it toward 1.0.
// The 13-round-tuned colors are ALREADY display-referred (written direct to the non-srgb swapchain,
// no gamma), so everything ≤K (background, cool valleys, most lines) MUST pass untouched — only the
// additive-overlap blowout above K rolls off. NO Reinhard-across-the-whole-range, NO gamma (either
// would darken/wash the tuned midtones — that is the regression).
fn rolloff(x: f32, K: f32) -> f32 {
  if (x <= K) { return x; }
  return K + (1.0 - K) * (1.0 - exp(-(x - K) / (1.0 - K)));
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
  let scene = textureSample(sceneTex, samp, in.uv).rgb;
  let bloom = textureSample(bloomTex, samp, in.uv).rgb;
  // additive glow on top of the scene, scaled by exposure first.
  let hdr = scene * P.exposure + bloom * P.intensity;
  // highlight-only rolloff per channel (K~0.8) — preserves tuned colors, compresses only blowout.
  let K = 0.8;
  let outc = vec3f(rolloff(hdr.r, K), rolloff(hdr.g, K), rolloff(hdr.b, K));
  return vec4f(outc, 1.0);
}
`,y=`// bloom_threshold.wgsl — bright-pass extraction for the bloom chain.
// Responsibilities:
//   - Fullscreen-triangle vertex stage (3 verts, no vertex buffer).
//   - Sample the HDR scene texture; output only the energy ABOVE a soft threshold (knee), so
//     only bright neon cores seed the glow and the dark dark ground/dim lines do not bloom.
//   - Runs at the downsampled bloom resolution (sampler is linear → free downsample).

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
  // fullscreen triangle: clip positions (-1,-1),(3,-1),(-1,3); uv derived to cover [0,1].
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: VsOut;
  let c = p[vi];
  out.pos = vec4f(c, 0.0, 1.0);
  out.uv = vec2f((c.x + 1.0) * 0.5, (1.0 - c.y) * 0.5); // flip y → texture space
  return out;
}

struct Params {
  threshold: f32, // luminance below which nothing blooms
  knee: f32,      // soft-knee width above the threshold (smoothstep), avoids a hard edge
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSamp: sampler;
@group(0) @binding(2) var<uniform> P: Params;

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
  let c = textureSample(srcTex, srcSamp, in.uv).rgb;
  // perceptual luminance of the HDR color
  let lum = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  // soft knee: 0 below threshold, ramps to 1 across [threshold, threshold+knee]
  let w = smoothstep(P.threshold, P.threshold + max(P.knee, 1e-4), lum);
  // keep the color hue; scale by the knee weight so cores bloom, mid-tones fade in gently
  return vec4f(c * w, 1.0);
}
`,x=`// add_force_field.wgsl — add a per-cell force field to the velocity (interior, in-place).
// Exact port of the add-forces loop in crates/vs-core/src/fluid/solver.rs::Fluid2D::step
// (u += dt*force_x[i,j], v += dt*force_y[i,j]) using FULL per-cell force buffers, not the
// parametric/localized source in forces.wgsl. Used by the composed correctness gate so the
// GPU step mirrors Fluid2D::step's scripted per-cell force exactly.
// Responsibilities:
//   - u[i,j] += dt*fx[i,j];  v[i,j] += dt*fy[i,j] (interior 1..=w, 1..=h only).
//   - Reads its own cell of fx,fy,u,v; writes its own cell of u,v — in-place safe.
//   - Flat 1D dispatch over (W+2)*(H+2); derive (i,j), guard interior; border untouched.

struct Params {
  w : u32, h : u32, dt : f32, pad0 : f32,
  fx : f32, fy : f32, dye_x : f32, dye_y : f32,
  dye_r : f32, dye_amt : f32, force_r : f32, pad1 : f32,
};

@group(0) @binding(0) var<uniform>             P  : Params;
@group(0) @binding(1) var<storage, read>       fx : array<f32>;
@group(0) @binding(2) var<storage, read>       fy : array<f32>;
@group(0) @binding(3) var<storage, read_write> u  : array<f32>;
@group(0) @binding(4) var<storage, read_write> v  : array<f32>;

fn idx(i : u32, j : u32) -> u32 { return i + (P.w + 2u) * j; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let stride = P.w + 2u;
  let g = gid.x;
  if (g >= stride * (P.h + 2u)) { return; }
  let i = g % stride;
  let j = g / stride;
  if (i < 1u || i > P.w || j < 1u || j > P.h) { return; }

  let k = idx(i, j);
  u[k] = u[k] + P.dt * fx[k];
  v[k] = v[k] + P.dt * fy[k];
}
`,_=`// advect.wgsl — semi-Lagrangian transport with MANUAL bilinear (matches grid.rs sample).
// Port of crates/vs-core/src/fluid/advect.rs::advect (+ grid.rs::sample for the bilinear).
// Responsibilities:
//   - Backtrace (x,y) = (i - dt*u[i,j], j - dt*v[i,j]); bilinearly sample \`src\` at (x,y) -> dst.
//   - Manual bilinear replicates grid.rs::sample VERBATIM: Stam clamp [0.5, w+0.5] x [0.5, h+0.5],
//     i0=floor(x), same s0/s1/t0/t1 weights, same blend-expression order (f32 throughout).
//   - Interior only (1..=w, 1..=h); reads \`src\` (separate buffer) -> ping-pong; border untouched.

struct Params {
  w : u32, h : u32, dt : f32, pad0 : f32,
  fx : f32, fy : f32, dye_x : f32, dye_y : f32,
  dye_r : f32, dye_amt : f32, force_r : f32, pad1 : f32,
};

@group(0) @binding(0) var<uniform>             P   : Params;
@group(0) @binding(1) var<storage, read>       src : array<f32>;
@group(0) @binding(2) var<storage, read>       u   : array<f32>;
@group(0) @binding(3) var<storage, read>       v   : array<f32>;
@group(0) @binding(4) var<storage, read_write> dst : array<f32>;

fn idx(i : u32, j : u32) -> u32 { return i + (P.w + 2u) * j; }

// Bilinear sample of \`src\` at continuous (x,y), matching grid.rs::sample exactly.
fn sample(x_in : f32, y_in : f32) -> f32 {
  let wf = f32(P.w);
  let hf = f32(P.h);
  // Stam clamp to [0.5, w+0.5] x [0.5, h+0.5] so floor()+1 taps stay in-bounds.
  let x = clamp(x_in, 0.5, wf + 0.5);
  let y = clamp(y_in, 0.5, hf + 0.5);

  let i0 = u32(floor(x));
  let j0 = u32(floor(y));
  let i1 = i0 + 1u;
  let j1 = j0 + 1u;
  let s1 = x - f32(i0);
  let s0 = 1.0 - s1;
  let t1 = y - f32(j0);
  let t0 = 1.0 - t1;

  return s0 * (t0 * src[idx(i0, j0)] + t1 * src[idx(i0, j1)])
       + s1 * (t0 * src[idx(i1, j0)] + t1 * src[idx(i1, j1)]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let stride = P.w + 2u;
  let g = gid.x;
  if (g >= stride * (P.h + 2u)) { return; }
  let i = g % stride;
  let j = g / stride;
  if (i < 1u || i > P.w || j < 1u || j > P.h) { return; }

  let x = f32(i) - P.dt * u[idx(i, j)];
  let y = f32(j) - P.dt * v[idx(i, j)];
  dst[idx(i, j)] = sample(x, y);
}
`,P=`// divergence.wgsl — central-difference velocity divergence (Stam, h=1).
// Port of crates/vs-core/src/fluid/project.rs::divergence.
// Responsibilities:
//   - For each interior cell (1..=w, 1..=h): div = 0.5*(u[i+1,j]-u[i-1,j]) + 0.5*(v[i,j+1]-v[i,j-1]).
//   - Flat 1D dispatch over (W+2)*(H+2); derive (i,j), guard interior; border left untouched.
//   - In-place safe: writes only its own cell of \`div\` (a separate buffer from u,v).

struct Params {
  w : u32, h : u32, dt : f32, pad0 : f32,
  fx : f32, fy : f32, dye_x : f32, dye_y : f32,
  dye_r : f32, dye_amt : f32, force_r : f32, pad1 : f32,
};

@group(0) @binding(0) var<uniform>             P   : Params;
@group(0) @binding(1) var<storage, read>       u   : array<f32>;
@group(0) @binding(2) var<storage, read>       v   : array<f32>;
@group(0) @binding(3) var<storage, read_write> div : array<f32>;

fn idx(i : u32, j : u32) -> u32 { return i + (P.w + 2u) * j; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let stride = P.w + 2u;
  let g = gid.x;
  if (g >= stride * (P.h + 2u)) { return; }
  let i = g % stride;
  let j = g / stride;
  if (i < 1u || i > P.w || j < 1u || j > P.h) { return; }

  let d = 0.5 * (u[idx(i + 1u, j)] - u[idx(i - 1u, j)])
        + 0.5 * (v[idx(i, j + 1u)] - v[idx(i, j - 1u)]);
  div[idx(i, j)] = d;
}
`,E=`// force_field.wgsl — add a PER-CELL velocity force field (terrain orographic coupling).
// Responsibilities:
//   - u[k] += dt * fx_field[k]; v[k] += dt * fy_field[k] for interior cells (1..W, 1..H).
//   - The force field is a world-pinned per-cell vector the caller computes from the terrain gradient
//     (deflect flow around/over high terrain; channel through valleys) so the fluid RESPONDS to the
//     real landscape. Complementary to forces.wgsl's scalar disc — applied the same step, in place.
//   - Flat 1D dispatch; derive (i,j), guard interior; reads/writes own cell only (in-place safe).

struct Params {
  w : u32, h : u32, dt : f32, pad0 : f32,
  fx : f32, fy : f32, dye_x : f32, dye_y : f32,
  dye_r : f32, dye_amt : f32, force_r : f32, pad1 : f32,
};

@group(0) @binding(0) var<uniform>             P        : Params;
@group(0) @binding(1) var<storage, read_write> u        : array<f32>;
@group(0) @binding(2) var<storage, read_write> v        : array<f32>;
@group(0) @binding(3) var<storage, read>       fx_field : array<f32>;
@group(0) @binding(4) var<storage, read>       fy_field : array<f32>;

fn idx(i : u32, j : u32) -> u32 { return i + (P.w + 2u) * j; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let stride = P.w + 2u;
  let g = gid.x;
  if (g >= stride * (P.h + 2u)) { return; }
  let i = g % stride;
  let j = g / stride;
  if (i < 1u || i > P.w || j < 1u || j > P.h) { return; }

  let k = idx(i, j);
  u[k] = u[k] + P.dt * fx_field[k];
  v[k] = v[k] + P.dt * fy_field[k];
}
`,S=`// forces.wgsl — add scripted velocity force + dye injection (interior, in-place).
// Generalizes the add-forces loop in crates/vs-core/src/fluid/solver.rs::Fluid2D::step
// (u += dt*force_x, v += dt*force_y) to a localized scripted source for the live viz.
// Responsibilities:
//   - Velocity: u += dt*fx, v += dt*fy within radius force_r of (dye_x, dye_y) [whole interior if force_r<=0].
//   - Dye: dye += dt*dye_amt within radius dye_r of (dye_x, dye_y).
//   - Interior only (1..=w, 1..=h); reads/writes own cell of u,v,dye — in-place safe.
//   - Flat 1D dispatch; derive (i,j), guard interior.

struct Params {
  w : u32, h : u32, dt : f32, pad0 : f32,
  fx : f32, fy : f32, dye_x : f32, dye_y : f32,
  dye_r : f32, dye_amt : f32, force_r : f32, pad1 : f32,
};

@group(0) @binding(0) var<uniform>             P   : Params;
@group(0) @binding(1) var<storage, read_write> u   : array<f32>;
@group(0) @binding(2) var<storage, read_write> v   : array<f32>;
@group(0) @binding(3) var<storage, read_write> dye : array<f32>;

fn idx(i : u32, j : u32) -> u32 { return i + (P.w + 2u) * j; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let stride = P.w + 2u;
  let g = gid.x;
  if (g >= stride * (P.h + 2u)) { return; }
  let i = g % stride;
  let j = g / stride;
  if (i < 1u || i > P.w || j < 1u || j > P.h) { return; }

  let dx = f32(i) - P.dye_x;
  let dy = f32(j) - P.dye_y;
  let r2 = dx * dx + dy * dy;
  let k = idx(i, j);

  // Velocity force: whole interior when force_r <= 0, else only within force_r.
  if (P.force_r <= 0.0 || r2 <= P.force_r * P.force_r) {
    u[k] = u[k] + P.dt * P.fx;
    v[k] = v[k] + P.dt * P.fy;
  }

  // Dye injection within dye_r.
  if (P.dye_r > 0.0 && r2 <= P.dye_r * P.dye_r) {
    dye[k] = dye[k] + P.dt * P.dye_amt;
  }
}
`,j=`// jacobi.wgsl — one Jacobi sweep of the pressure Poisson solve (ping-pong).
// Port of the inner sweep in crates/vs-core/src/fluid/project.rs::project.
// Responsibilities:
//   - p_next[i,j] = 0.25*(p[i-1,j]+p[i+1,j]+p[i,j-1]+p[i,j+1] - div[i,j]) (interior only).
//   - Reads the PREVIOUS pressure buffer, writes the NEXT (caller swaps each sweep).
//   - Flat 1D dispatch; derive (i,j), guard interior; same sign/algorithm as the oracle.

struct Params {
  w : u32, h : u32, dt : f32, pad0 : f32,
  fx : f32, fy : f32, dye_x : f32, dye_y : f32,
  dye_r : f32, dye_amt : f32, force_r : f32, pad1 : f32,
};

@group(0) @binding(0) var<uniform>             P      : Params;
@group(0) @binding(1) var<storage, read>       p      : array<f32>;
@group(0) @binding(2) var<storage, read>       div    : array<f32>;
@group(0) @binding(3) var<storage, read_write> p_next : array<f32>;

fn idx(i : u32, j : u32) -> u32 { return i + (P.w + 2u) * j; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let stride = P.w + 2u;
  let g = gid.x;
  if (g >= stride * (P.h + 2u)) { return; }
  let i = g % stride;
  let j = g / stride;
  if (i < 1u || i > P.w || j < 1u || j > P.h) { return; }

  let sum = p[idx(i - 1u, j)] + p[idx(i + 1u, j)]
          + p[idx(i, j - 1u)] + p[idx(i, j + 1u)];
  p_next[idx(i, j)] = 0.25 * (sum - div[idx(i, j)]);
}
`,R=`// set_bnd.wgsl — Stam boundary conditions as TWO passes (edges, then corners).
// Port of crates/vs-core/src/fluid/boundary.rs::set_bnd, split to avoid the corner race:
// corners read freshly-written edge cells, so edges MUST complete (separate dispatch) first.
// Responsibilities:
//   - 6 entry points: {scalar,velx,vely} x {edges,corners}, one buffer \`g\` (in-place safe).
//   - edges: left/right walls (negate normal x for velx), bottom/top walls (negate normal y for vely).
//   - corners: mean of the two adjacent edge cells (identical for all kinds).
//   - Flat 1D dispatch over (W+2)*(H+2); each thread owns one border cell, interior threads no-op.

struct Params {
  w : u32, h : u32, dt : f32, pad0 : f32,
  fx : f32, fy : f32, dye_x : f32, dye_y : f32,
  dye_r : f32, dye_amt : f32, force_r : f32, pad1 : f32,
};

@group(0) @binding(0) var<uniform>             P : Params;
@group(0) @binding(1) var<storage, read_write> g : array<f32>;

fn idx(i : u32, j : u32) -> u32 { return i + (P.w + 2u) * j; }

// --- Edges: negate the normal component on the two walls selected by (neg_x, neg_y) ---
fn edges(gid : u32, neg_x : bool, neg_y : bool) {
  let w = P.w;
  let h = P.h;
  let stride = w + 2u;
  if (gid >= stride * (h + 2u)) { return; }
  let i = gid % stride;
  let j = gid / stride;

  // Left/right walls: interior rows 1..=h.
  if (j >= 1u && j <= h) {
    if (i == 0u) {
      let left = g[idx(1u, j)];
      g[idx(0u, j)] = select(left, -left, neg_x);
      return;
    }
    if (i == w + 1u) {
      let right = g[idx(w, j)];
      g[idx(w + 1u, j)] = select(right, -right, neg_x);
      return;
    }
  }
  // Bottom/top walls: interior columns 1..=w.
  if (i >= 1u && i <= w) {
    if (j == 0u) {
      let bottom = g[idx(i, 1u)];
      g[idx(i, 0u)] = select(bottom, -bottom, neg_y);
      return;
    }
    if (j == h + 1u) {
      let top = g[idx(i, h)];
      g[idx(i, h + 1u)] = select(top, -top, neg_y);
      return;
    }
  }
}

// --- Corners: mean of the two adjacent edge cells (run after edges complete) ---
fn corners(gid : u32) {
  let w = P.w;
  let h = P.h;
  let stride = w + 2u;
  if (gid >= stride * (h + 2u)) { return; }
  let i = gid % stride;
  let j = gid / stride;

  if (i == 0u && j == 0u) {
    g[idx(0u, 0u)] = 0.5 * (g[idx(1u, 0u)] + g[idx(0u, 1u)]);
  } else if (i == 0u && j == h + 1u) {
    g[idx(0u, h + 1u)] = 0.5 * (g[idx(1u, h + 1u)] + g[idx(0u, h)]);
  } else if (i == w + 1u && j == 0u) {
    g[idx(w + 1u, 0u)] = 0.5 * (g[idx(w, 0u)] + g[idx(w + 1u, 1u)]);
  } else if (i == w + 1u && j == h + 1u) {
    g[idx(w + 1u, h + 1u)] = 0.5 * (g[idx(w, h + 1u)] + g[idx(w + 1u, h)]);
  }
}

@compute @workgroup_size(64)
fn scalar_edges(@builtin(global_invocation_id) gid : vec3<u32>) { edges(gid.x, false, false); }
@compute @workgroup_size(64)
fn velx_edges(@builtin(global_invocation_id) gid : vec3<u32>)   { edges(gid.x, true,  false); }
@compute @workgroup_size(64)
fn vely_edges(@builtin(global_invocation_id) gid : vec3<u32>)   { edges(gid.x, false, true);  }

@compute @workgroup_size(64)
fn scalar_corners(@builtin(global_invocation_id) gid : vec3<u32>) { corners(gid.x); }
@compute @workgroup_size(64)
fn velx_corners(@builtin(global_invocation_id) gid : vec3<u32>)   { corners(gid.x); }
@compute @workgroup_size(64)
fn vely_corners(@builtin(global_invocation_id) gid : vec3<u32>)   { corners(gid.x); }
`,A=`// shift.wgsl — scroll a bordered (W+2)*(H+2) field by an integer cell offset (world-pinned recenter).
// Responsibilities:
//   - dst[i,j] = src[i - dx, j - dz] for interior cells whose source falls inside the interior
//     (the overlapping region is copied 1:1 — no resample, so the existing flow scrolls with NO seam).
//   - Cells whose source falls OUTSIDE the interior (the freshly-exposed leading edge) are seeded by
//     CLAMP-EXTENDING the nearest interior column/row of src (continuous extrapolation, not zero →
//     no hard edge/pop). The caller re-forces these fresh cells from the terrain after the shift.
//   - Border cells (i or j outside 1..W) are left for set_bnd to refill; we only write the interior.
//   - Reads src, writes dst (a separate buffer — the ping-pong .next half — so the copy is race-free).

struct ShiftParams { w : u32, h : u32, dx : i32, dz : i32 };

@group(0) @binding(0) var<uniform>             SP  : ShiftParams;
@group(0) @binding(1) var<storage, read>       src : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;

fn idx(i : u32, j : u32) -> u32 { return i + (SP.w + 2u) * j; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let stride = SP.w + 2u;
  let g = gid.x;
  if (g >= stride * (SP.h + 2u)) { return; }
  let i = g % stride;
  let j = g / stride;
  if (i < 1u || i > SP.w || j < 1u || j > SP.h) { return; } // interior only

  // source interior coord = dst coord - shift, clamped to the interior (clamp-extend the edges).
  let si = clamp(i32(i) - SP.dx, 1, i32(SP.w));
  let sj = clamp(i32(j) - SP.dz, 1, i32(SP.h));
  dst[idx(i, j)] = src[idx(u32(si), u32(sj))];
}
`,O=`// subtract_grad.wgsl — subtract the pressure gradient from the velocity field.
// Port of the gradient-subtraction loop in crates/vs-core/src/fluid/project.rs::project.
// Responsibilities:
//   - u[i,j] -= 0.5*(p[i+1,j]-p[i-1,j]);  v[i,j] -= 0.5*(p[i,j+1]-p[i,j-1]) (interior only).
//   - Reads pressure \`p\` (separate buffer) + own u,v cell; writes own u,v cell — in-place safe.
//   - Flat 1D dispatch; derive (i,j), guard interior. Caller applies set_bnd afterward.

struct Params {
  w : u32, h : u32, dt : f32, pad0 : f32,
  fx : f32, fy : f32, dye_x : f32, dye_y : f32,
  dye_r : f32, dye_amt : f32, force_r : f32, pad1 : f32,
};

@group(0) @binding(0) var<uniform>             P : Params;
@group(0) @binding(1) var<storage, read>       p : array<f32>;
@group(0) @binding(2) var<storage, read_write> u : array<f32>;
@group(0) @binding(3) var<storage, read_write> v : array<f32>;

fn idx(i : u32, j : u32) -> u32 { return i + (P.w + 2u) * j; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let stride = P.w + 2u;
  let g = gid.x;
  if (g >= stride * (P.h + 2u)) { return; }
  let i = g % stride;
  let j = g / stride;
  if (i < 1u || i > P.w || j < 1u || j > P.h) { return; }

  let gx = 0.5 * (p[idx(i + 1u, j)] - p[idx(i - 1u, j)]);
  let gy = 0.5 * (p[idx(i, j + 1u)] - p[idx(i, j - 1u)]);
  u[idx(i, j)] = u[idx(i, j)] - gx;
  v[idx(i, j)] = v[idx(i, j)] - gy;
}
`,C=`// visualize.wgsl — debug viz render pipeline: dye field → neon-green ramp on dark.
// Debug-grade visualization for the Plan-3 fluid spike (NOT the §4.1 neon renderer; that is Plan 4).
// Responsibilities:
//   - Fullscreen triangle (3 verts via vertex_index; no vertex buffer) covering the viewport.
//   - Fragment: map frag UV → interior fluid cell (1..=w, 1..=h) of the bordered (W+2)*(H+2) dye
//     buffer (idx = i + (W+2)*j), sample dye as read-only storage (read_write is compute-only).
//   - Map dye magnitude → neon-green ramp on a dark base; clamp output luminance to a brightness
//     ceiling (blueprint §7.2 photosensitivity) and keep the field continuous (no flashing).
//   - Reuses the shared Params uniform (w,h) so the host binds the same uniform buffer as compute.

struct Params {
  w : u32, h : u32, dt : f32, pad0 : f32,
  fx : f32, fy : f32, dye_x : f32, dye_y : f32,
  dye_r : f32, dye_amt : f32, force_r : f32, pad1 : f32,
};

@group(0) @binding(0) var<uniform>       P   : Params;
@group(0) @binding(1) var<storage, read> dye : array<f32>;

// §7.2 photosensitivity: cap output brightness so the debug viz never hits full-white flashes.
const BRIGHTNESS_CEILING : f32 = 0.8;

struct VsOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

// Fullscreen triangle: 3 oversized verts, clipped to the viewport. uv in [0,1] over the screen.
@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VsOut {
  var out : VsOut;
  let x = f32((vi << 1u) & 2u); // 0,2,0
  let y = f32(vi & 2u);         // 0,0,2
  out.pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2<f32>(x, 1.0 - y); // flip y: row 0 at the top
  return out;
}

@fragment
fn fs(in : VsOut) -> @location(0) vec4<f32> {
  let stride = P.w + 2u;
  // Map screen uv → interior cell index (1..=w, 1..=h).
  let i = 1u + min(P.w - 1u, u32(in.uv.x * f32(P.w)));
  let j = 1u + min(P.h - 1u, u32(in.uv.y * f32(P.h)));
  let d = dye[i + stride * j];

  // Dye magnitude → [0,1] intensity. Soft saturating curve so dense cores read distinct from wisps.
  let t = clamp(1.0 - exp(-max(d, 0.0)), 0.0, 1.0);

  // Neon-green ramp on a dark base: green dominant, faint cyan lift in the highlights.
  let base = vec3<f32>(0.02, 0.03, 0.04);
  let neon = vec3<f32>(0.10, 1.0, 0.45);
  var col = base + neon * t;

  // §7.2: clamp luminance to the brightness ceiling (preserve hue by scaling, not truncating).
  let lum = dot(col, vec3<f32>(0.2126, 0.7152, 0.0722));
  if (lum > BRIGHTNESS_CEILING) {
    col = col * (BRIGHTNESS_CEILING / lum);
  }
  return vec4<f32>(col, 1.0);
}
`,k=`// marker.wgsl — altitude plumb-line + ground diamond for the bird (line-list, additive, depth-tested).
// Responsibilities:
//   - vs: place unit geometry in world space from the uniform. kind 0 = vertical drop line from just
//     under the bird to just above the ground; kind 1 = a small diamond on the ground under the bird,
//     gently pulsing. Carries t (0 at bird, 1 at ground) for the dash pattern.
//   - fs: dash the drop line every DASH_M meters of real height (dash count = readable altimeter);
//     render the diamond as a solid soft-cyan glow. Additive blend; depth-tested (ridges occlude).

const DASH_M: f32 = 9.0;

struct U {
  viewProj : mat4x4<f32>,
  birdPos  : vec3<f32>,
  groundY  : f32,
  time     : f32,
  height   : f32, // bird.y - groundY (m) — drives the dash count
  pad0     : f32,
  pad1     : f32,
};
@group(0) @binding(0) var<uniform> u : U;

struct VOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) t    : f32, // 0..1 along the drop line; -1 for the diamond
};

@vertex
fn vs(@location(0) pos : vec3<f32>, @location(1) attr : vec3<f32>) -> VOut {
  var out : VOut;
  let kind = attr.x;
  let t = attr.y;
  var world : vec3<f32>;
  if (kind < 0.5) {
    // drop line: start a touch below the bird body, end a touch above the ground line.
    world = vec3<f32>(u.birdPos.x, mix(u.birdPos.y - 4.0, u.groundY + 1.0, t), u.birdPos.z);
    out.t = t;
  } else {
    // ground diamond: unit cross scaled with a soft pulse, sitting just above the ground.
    let scale = 7.0 + 1.5 * sin(u.time * 4.0);
    world = vec3<f32>(u.birdPos.x + pos.x * scale, u.groundY + 1.2, u.birdPos.z + pos.z * scale);
    out.t = -1.0;
  }
  out.clip = u.viewProj * vec4<f32>(world, 1.0);
  return out;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4<f32> {
  let cyan = vec3<f32>(0.25, 0.85, 1.0);
  if (in.t < 0.0) {
    // ground diamond: solid soft glow.
    return vec4<f32>(cyan * 0.55, 1.0);
  }
  // drop line: one dash per DASH_M meters of real height; brighter toward the ground end.
  let dash = step(fract(in.t * u.height / DASH_M), 0.55);
  let glow = mix(0.18, 0.45, in.t);
  return vec4<f32>(cyan * glow * dash, 1.0);
}
`,T=`// target.wgsl — flight target beacon: a camera-facing vertical beam of light at a world waypoint.
// Responsibilities:
//   - vs: build a billboarded vertical quad from the uniform (base at ground, up by \`height\`, widened
//     along a CPU-supplied horizontal \`rightAxis\` by \`halfWidth\`). Carries (cx,cy) for the fs shade.
//   - fs: additive amber glow — gaussian horizontal falloff (soft feathered edges), brighter at the
//     base, gentle time pulse. Depth handled by the pipeline (drawn always-on-top for navigation).

struct U {
  viewProj  : mat4x4<f32>,
  basePos   : vec3<f32>,
  height    : f32,
  rightAxis : vec3<f32>,
  halfWidth : f32,
  color     : vec3<f32>,
  time      : f32,
};
@group(0) @binding(0) var<uniform> u : U;

struct VOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) cx : f32, // -1..1 across the beam width
  @location(1) cy : f32, //  0 at base, 1 at top
};

@vertex
fn vs(@location(0) corner : vec3<f32>) -> VOut {
  var out : VOut;
  let cx = corner.x; // -1 or 1
  let cy = corner.y; //  0 or 1
  let world = u.basePos
    + u.rightAxis * (cx * u.halfWidth)
    + vec3<f32>(0.0, 1.0, 0.0) * (cy * u.height);
  out.clip = u.viewProj * vec4<f32>(world, 1.0);
  out.cx = cx;
  out.cy = cy;
  return out;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4<f32> {
  let horiz = exp(-in.cx * in.cx * 3.0);     // soft glowing core, feathered edges
  let vert = mix(1.0, 0.12, in.cy);          // brighter at the base, fades up the column
  let pulse = 0.8 + 0.2 * sin(u.time * 3.0); // gentle "alive / objective" pulse
  let glow = horiz * vert * pulse;
  return vec4<f32>(u.color * glow, 1.0);
}
`,U=`// terrain3d.wgsl — neon receding-ridgeline terrain (WebGPU 3D, NDC z in [0,1]).
// Responsibilities:
//   - Vertex: take a static centered NxN grid (gridXZ), compute worldXZ = gridXZ + camOffset,
//     displace Y by fBm height; project with viewProj; pass world height + view distance to frag.
//   - Fragment: dark surface + emissive glowing contour lines (fract(height/spacing)~0) and
//     ridge-crest glow; exponential distance fog mixes toward the dark haze background.
//   - fBm MUST match the TS sampleHeight() in terrain.ts (same constants/hash) for the bird later.

struct Uniforms {
  viewProj : mat4x4<f32>,
  camOffset : vec2<f32>,   // world (x,z) the grid is recentered onto
  fogColor : vec3<f32>,
  _pad0 : f32,
  eyePos : vec3<f32>,      // world-space camera eye, for view distance
  fogDensity : f32,
};

@group(0) @binding(0) var<uniform> U : Uniforms;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) height : f32,
  @location(1) viewDist : f32,
  @location(2) slope : f32,
};

// --- fBm (5 octaves) — IDENTICAL constants mirrored in terrain.ts sampleHeight ---
const BASE_FREQ : f32 = 0.00285714;  // ~1/350 per meter
const LACUNARITY : f32 = 2.0;
const GAIN : f32 = 0.5;
const OCTAVES : i32 = 3;             // large clean ridges (high octaves fragment the silhouette)
const RELIEF : f32 = 220.0;          // total relief target (meters) — ridged crests

fn hash2(p : vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

// value noise: smooth-interpolated hash of the integer lattice corners.
fn valueNoise(p : vec2<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let a = hash2(i);
  let b = hash2(i + vec2<f32>(1.0, 0.0));
  let c = hash2(i + vec2<f32>(0.0, 1.0));
  let d = hash2(i + vec2<f32>(1.0, 1.0));
  let u = f * f * (3.0 - 2.0 * f);   // smoothstep weights
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p : vec2<f32>) -> f32 {
  var freq = BASE_FREQ;
  var amp = 1.0;
  var sum = 0.0;
  var norm = 0.0;
  for (var k = 0; k < OCTAVES; k = k + 1) {
    // ridged noise: sharp crests + V-valleys (receding-ridgeline signature).
    let n = valueNoise(p * freq);
    let r = 1.0 - abs(2.0 * n - 1.0);
    sum = sum + amp * r;
    norm = norm + amp;
    freq = freq * LACUNARITY;
    amp = amp * GAIN;
  }
  // ridged noise sits in [0,1]; scale to relief (crests up from y=0).
  return (sum / norm) * RELIEF;
}

fn heightAt(xz : vec2<f32>) -> f32 {
  return fbm(xz);
}

@vertex
fn vs(@location(0) gridXZ : vec2<f32>) -> VSOut {
  var out : VSOut;
  let worldXZ = gridXZ + U.camOffset;
  let h = heightAt(worldXZ);

  // central-difference slope magnitude for ridge-crest glow.
  let eps = 4.0;
  let hx = heightAt(worldXZ + vec2<f32>(eps, 0.0)) - heightAt(worldXZ - vec2<f32>(eps, 0.0));
  let hz = heightAt(worldXZ + vec2<f32>(0.0, eps)) - heightAt(worldXZ - vec2<f32>(0.0, eps));
  let slope = length(vec2<f32>(hx, hz) / (2.0 * eps));

  let worldPos = vec3<f32>(worldXZ.x, h, worldXZ.y);
  out.clip = U.viewProj * vec4<f32>(worldPos, 1.0);
  out.height = h;
  out.viewDist = length(worldPos - U.eyePos);
  out.slope = slope;
  return out;
}

const CONTOUR_SPACING : f32 = 12.0;
const NEON_A : vec3<f32> = vec3<f32>(0.10, 0.95, 0.80);  // teal-green
const NEON_B : vec3<f32> = vec3<f32>(0.85, 0.20, 0.95);  // magenta (high ground)
const SURFACE : vec3<f32> = vec3<f32>(0.015, 0.02, 0.05); // near-black surface

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  // contour glow: bright where height crosses a contour level.
  let hn = in.height / CONTOUR_SPACING;
  let d = abs(fract(hn) - 0.5);          // 0.5 at the level, 0 at half-way
  let contour = smoothstep(0.42, 0.5, d); // glow near the band edges (level lines)

  // ridge-crest glow: steep slopes light up.
  let crest = smoothstep(0.18, 0.55, in.slope);

  // height-tinted neon: low ground teal, high crests magenta.
  let tint = mix(NEON_A, NEON_B, clamp(in.height / 220.0, 0.0, 1.0));
  let glow = tint * (contour * 1.0 + crest * 0.6);

  var color = SURFACE + glow;

  // exponential distance fog mixes toward dark haze background — the fog IS the depth.
  let fog = exp(-U.fogDensity * in.viewDist);
  color = mix(U.fogColor, color, clamp(fog, 0.0, 1.0));

  // cap brightness (no full-screen flashes).
  color = min(color, vec3<f32>(0.9, 0.95, 1.0));
  return vec4<f32>(color, 1.0);
}
`,F=`// terrain_ekg.wgsl — EKG/waveform stacked neon trace LINES + opaque hidden-line FILL (WebGPU, NDC z 0..1).
// Responsibilities:
//   - LINES (vsLine/fsLine): render the heightfield as a stack of horizontal neon polylines (Joy
//     Division "Unknown Pleasures" look). Each vertex is one sample of one depth row. Additive glow,
//     depth-test less-equal, NO depth-write (fills own the depth buffer; a line sits on its own
//     curtain's top edge without z-fighting).
//   - FILL (vsFill/fsFill): per-row OPAQUE curtain from the ridge line DOWN to a low baseline, colored
//     the SKY/background so it is invisible as a surface but writes DEPTH. A NEAR curtain writes
//     nearer depth and OCCLUDES the lines of FARTHER rows behind it → hidden-line removal (no horizon
//     tangle). NOT a row-to-row connected mesh (that is the shaded surface the user rejected); each
//     row is its own vertical curtain at constant depth.
//   - CAMERA-RELATIVE ROWS: each row is PERPENDICULAR to the camera. World sample
//     P = camGround + camFwd*depth + camRight*(xFrac*halfWidth); height y = fBm(P.xz). Stacked lines
//     stay SCREEN-HORIZONTAL at every heading.
//   - ELEVATION COLOR: tint each line by terrain HEIGHT — cool deep-teal/blue valleys → warm
//     magenta/white peaks, smooth ramp, brightness-capped.
//   - CLEAN HORIZON: hard maxDist cutoff + exponential fog (applied to lines AND fills).
//   - fBm MUST match terrain.ts sampleHeight() (same constants/hash) — drives the bird ridge-lift.

struct Uniforms {
  viewProj : mat4x4<f32>,
  camGround : vec2<f32>,   // camera ground position (x,z) the line stack is built around
  halfWidth : f32,         // half horizontal extent of each row (m)
  maxDist : f32,           // hard draw-distance cutoff (m) → clean horizon
  camFwd : vec2<f32>,      // horizontal camera forward (x,z), unit
  camRight : vec2<f32>,    // horizontal camera right (x,z), unit
  fogColor : vec3<f32>,    // SKY/background — also the opaque fill color
  fogDensity : f32,
  eyePos : vec3<f32>,      // world-space camera eye, for view distance
  baseline : f32,          // low world-y the fill curtains drop to
};

@group(0) @binding(0) var<uniform> U : Uniforms;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) height : f32,
  @location(1) viewDist : f32,
  @location(2) rowFade : f32,   // 0 near .. 1 far (for brightness falloff)
  @location(3) rowDepth : f32,  // depth ahead of camera (m) — for hard cutoff
};

// --- heightfield — IDENTICAL constants mirrored in terrain.ts sampleHeight ---
const BASE_FREQ : f32 = 0.00142857;  // ~1/700 per meter — 2× wider features
const LACUNARITY : f32 = 2.0;
const GAIN : f32 = 0.5;
const OCTAVES : i32 = 4;
const RELIEF : f32 = 600.0;   // taller, more dramatic peaks (was 320) — MUST mirror terrain.ts
const SHARP : f32 = 1.8;      // valley-deepening / crest-sharpening pow
const TERRACES : f32 = 5.0;   // cliff bands
const RISER_POW : f32 = 4.0;  // riser sharpness
const CLIFF_MIX : f32 = 0.65; // terraced vs smooth blend

// Integer lattice hash — pure u32 ops, identical in WGSL(GPU f32) and JS(CPU f64). The old
// fract(sin(dot)*43758) hash diverged catastrophically f32-vs-f64, so terrain.ts (collision)
// and the GPU (visual) computed DIFFERENT fields — the bird crashed into invisible terrain.
// MUST stay bit-identical to ihash() in terrain.ts.
fn ihash(c : vec2<i32>) -> f32 {
  var h : u32 = bitcast<u32>(c.x) * 374761393u + bitcast<u32>(c.y) * 668265263u;
  h = (h ^ (h >> 13u)) * 1274126177u;
  h = h ^ (h >> 16u);
  return f32(h) / 4294967295.0;
}
fn valueNoise(p : vec2<f32>) -> f32 {
  let pf = floor(p);
  let ci = vec2<i32>(pf);
  let f = p - pf;
  let a = ihash(ci);
  let b = ihash(ci + vec2<i32>(1, 0));
  let c = ihash(ci + vec2<i32>(0, 1));
  let d = ihash(ci + vec2<i32>(1, 1));
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
fn fbm(p : vec2<f32>) -> f32 {
  var freq = BASE_FREQ;
  var amp = 1.0;
  var sum = 0.0;
  var norm = 0.0;
  for (var k = 0; k < OCTAVES; k = k + 1) {
    let n = valueNoise(p * freq);
    let r = 1.0 - abs(2.0 * n - 1.0);
    sum = sum + amp * r;
    norm = norm + amp;
    freq = freq * LACUNARITY;
    amp = amp * GAIN;
  }
  // sharpen (deep valleys, crisp crests) then carve terraced cliff bands (shelf + steep riser).
  let s = pow(sum / norm, SHARP);
  let b = s * TERRACES;
  let fb = b - floor(b);
  let ter = floor(b) / TERRACES + pow(fb, RISER_POW) / TERRACES;
  return (s + (ter - s) * CLIFF_MIX) * RELIEF;
}

// world ridge height at a camera-relative (xFrac, depth) sample.
fn ridgeHeight(xFrac : f32, depth : f32) -> f32 {
  let lateral = xFrac * U.halfWidth;
  let worldX = U.camGround.x + U.camFwd.x * depth + U.camRight.x * lateral;
  let worldZ = U.camGround.y + U.camFwd.y * depth + U.camRight.y * lateral;
  return fbm(vec2<f32>(worldX, worldZ));
}

// --- LINE PASS -------------------------------------------------------------
// vertex: location0 = (xFrac in [-1,1], rowDepth meters); location1 = rowFade.
@vertex
fn vsLine(@location(0) sample : vec2<f32>, @location(1) rowFade : f32) -> VSOut {
  var out : VSOut;
  let depth = sample.y;
  let lateral = sample.x * U.halfWidth;
  let worldX = U.camGround.x + U.camFwd.x * depth + U.camRight.x * lateral;
  let worldZ = U.camGround.y + U.camFwd.y * depth + U.camRight.y * lateral;
  let h = fbm(vec2<f32>(worldX, worldZ));
  let worldPos = vec3<f32>(worldX, h, worldZ);
  out.clip = U.viewProj * vec4<f32>(worldPos, 1.0);
  out.height = h;
  out.viewDist = length(worldPos - U.eyePos);
  out.rowFade = rowFade;
  out.rowDepth = depth;
  return out;
}

// elevation ramp: deep teal/blue valleys → magenta → near-white peaks.
const COOL : vec3<f32> = vec3<f32>(0.05, 0.45, 0.85);  // deep teal-blue (valley)
const MID  : vec3<f32> = vec3<f32>(0.85, 0.20, 0.95);  // magenta (mid/upper slope)
const WARM : vec3<f32> = vec3<f32>(1.00, 0.85, 0.95);  // hot near-white (peak)

@fragment
fn fsLine(in : VSOut) -> @location(0) vec4<f32> {
  // CLEAN HORIZON: hard cutoff beyond the draw distance.
  if (in.rowDepth > U.maxDist) {
    discard;
  }

  // elevation color: two-stage ramp valley→mid→peak by normalized height.
  let e = clamp(in.height / RELIEF, 0.0, 1.0);
  var tint = mix(COOL, MID, smoothstep(0.0, 0.6, e));
  tint = mix(tint, WARM, smoothstep(0.6, 1.0, e));

  // near rows brighter, far rows dimmer (rowFade 0 near .. 1 far).
  let rowBright = mix(1.0, 0.22, in.rowFade);

  // exponential distance fog → far lines dissolve into the dark haze BEFORE the cutoff.
  let fog = exp(-U.fogDensity * in.viewDist);
  // soft cutoff edge over the last 25% of draw distance.
  let edge = clamp((U.maxDist - in.rowDepth) / (U.maxDist * 0.25), 0.0, 1.0);

  var color = tint * rowBright * clamp(fog, 0.0, 1.0) * edge;
  // cap brightness (additive blend → no full-screen blowout).
  color = min(color, vec3<f32>(0.95, 0.97, 1.0));
  return vec4<f32>(color, 1.0);
}

// --- FILL PASS (opaque hidden-line curtains) -------------------------------
// vertex: location0 = (xFrac in [-1,1], rowDepth meters); location1 = topFlag (1=ridge top, 0=baseline).
struct FillOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) rowDepth : f32,
};

@vertex
fn vsFill(@location(0) sample : vec2<f32>, @location(1) topFlag : f32) -> FillOut {
  var out : FillOut;
  let depth = sample.y;
  let lateral = sample.x * U.halfWidth;
  let worldX = U.camGround.x + U.camFwd.x * depth + U.camRight.x * lateral;
  let worldZ = U.camGround.y + U.camFwd.y * depth + U.camRight.y * lateral;
  // top verts sit on the ridge; bottom verts drop to the low baseline (vertical curtain).
  let h = fbm(vec2<f32>(worldX, worldZ));
  let y = mix(U.baseline, h, topFlag);
  let worldPos = vec3<f32>(worldX, y, worldZ);
  out.clip = U.viewProj * vec4<f32>(worldPos, 1.0);
  out.rowDepth = depth;
  return out;
}

@fragment
fn fsFill(in : FillOut) -> @location(0) vec4<f32> {
  // respect the same horizon cutoff so fills never occlude past the visible stack.
  if (in.rowDepth > U.maxDist) {
    discard;
  }
  // opaque SKY/background color — invisible as a surface, writes depth to occlude far lines.
  return vec4<f32>(U.fogColor, 1.0);
}
`,I=`// terrain_grid.wgsl — WORLD-STATIC wireframe terrain. A square grid pinned to WORLD coordinates, draped
// on the terrain fBm, streamed in a window around the camera. Unlike the camera-relative EKG terrain,
// these lines are fixed in the world — fly forward and they flow toward you with real parallax.
// Two passes share this module:
//   - FILL (vsFill/fsFill): the draped surface as dark SKY-colored triangles, depthWrite ON → hides the
//     lines behind ridges (hidden-surface removal, the grid analog of the EKG fill curtains).
//   - LINE (vsLine/fsLine): the grid lines, elevation-ramped neon + distance fog, depth-tested.
// fBm is byte-identical to terrain_ekg.wgsl / trees so trees + bird sit on exactly this surface.

struct U {
  viewProj: mat4x4<f32>,
  eye: vec3<f32>,
  fogDensity: f32,
  fogColor: vec3<f32>,
  maxDist: f32,
  interval: f32,   // topo: m between contour lines
  lineWidth: f32,  // topo: contour line width (screen-relative)
  floorFade: f32,  // topo: brightness at the valley floor (low → dim)
  peakGain: f32,   // topo: brightness at the peaks (high → bright)
};
@group(0) @binding(0) var<uniform> u: U;

const BASE_FREQ: f32 = 0.00142857;
const LACUNARITY: f32 = 2.0;
const GAIN: f32 = 0.5;
const OCTAVES: i32 = 4;
const RELIEF: f32 = 600.0;
const SHARP: f32 = 1.8;
const TERRACES: f32 = 5.0;
const RISER_POW: f32 = 4.0;
const CLIFF_MIX: f32 = 0.65;

// Integer lattice hash — MUST stay bit-identical to ihash() in terrain.ts so grid-mode terrain
// matches the collision field (old sin-hash diverged f32-vs-f64 → bird crashed in clear sky).
fn ihash(c: vec2<i32>) -> f32 {
  var h: u32 = bitcast<u32>(c.x) * 374761393u + bitcast<u32>(c.y) * 668265263u;
  h = (h ^ (h >> 13u)) * 1274126177u;
  h = h ^ (h >> 16u);
  return f32(h) / 4294967295.0;
}
fn valueNoise(p: vec2<f32>) -> f32 {
  let pf = floor(p); let ci = vec2<i32>(pf); let f = p - pf;
  let a = ihash(ci); let b = ihash(ci + vec2<i32>(1, 0));
  let c = ihash(ci + vec2<i32>(0, 1)); let d = ihash(ci + vec2<i32>(1, 1));
  let uu = f * f * (3.0 - 2.0 * f);
  return mix(mix(a, b, uu.x), mix(c, d, uu.x), uu.y);
}
fn fbm(p: vec2<f32>) -> f32 {
  var freq = BASE_FREQ; var amp = 1.0; var sum = 0.0; var norm = 0.0;
  for (var k = 0; k < OCTAVES; k = k + 1) {
    let n = valueNoise(p * freq);
    let r = 1.0 - abs(2.0 * n - 1.0);
    sum = sum + amp * r; norm = norm + amp; freq = freq * LACUNARITY; amp = amp * GAIN;
  }
  let s = pow(sum / norm, SHARP);
  let b = s * TERRACES;
  let fb = b - floor(b);
  let ter = floor(b) / TERRACES + pow(fb, RISER_POW) / TERRACES;
  return (s + (ter - s) * CLIFF_MIX) * RELIEF;
}

// SMOOTH height (skips the terraced cliff bands) → evenly-spaced, consistent topo contours.
fn fbmSmooth(p: vec2<f32>) -> f32 {
  var freq = BASE_FREQ; var amp = 1.0; var sum = 0.0; var norm = 0.0;
  for (var k = 0; k < OCTAVES; k = k + 1) {
    let n = valueNoise(p * freq);
    let r = 1.0 - abs(2.0 * n - 1.0);
    sum = sum + amp * r; norm = norm + amp; freq = freq * LACUNARITY; amp = amp * GAIN;
  }
  return pow(sum / norm, SHARP) * RELIEF;
}

struct VOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) height: f32,
  @location(1) viewDist: f32,
  @location(2) worldXZ: vec2<f32>,
};

fn place(xz: vec2<f32>, lift: f32) -> VOut {
  var o: VOut;
  let y = fbm(xz);
  let world = vec3<f32>(xz.x, y + lift, xz.y);
  o.clip = u.viewProj * vec4<f32>(world, 1.0);
  o.height = y;
  o.viewDist = distance(world, u.eye);
  o.worldXZ = xz;
  return o;
}

const COOL: vec3<f32> = vec3<f32>(0.05, 0.45, 0.85);
const MID: vec3<f32> = vec3<f32>(0.85, 0.20, 0.95);
const WARM: vec3<f32> = vec3<f32>(1.00, 0.85, 0.95);
const CONTOUR_INTERVAL: f32 = 22.0; // m between topo contour lines

fn elevColor(e: f32) -> vec3<f32> {
  let col = mix(COOL, MID, smoothstep(0.0, 0.5, e));
  return mix(col, WARM, smoothstep(0.5, 1.0, e));
}
fn distFog(viewDist: f32) -> f32 {
  let fog = exp(-u.fogDensity * viewDist);
  let edge = clamp((u.maxDist - viewDist) / (u.maxDist * 0.3), 0.0, 1.0); // hide the window boundary
  return fog * edge;
}

// --- WIREFRAME GRID ---
@vertex fn vsFill(@location(0) xz: vec2<f32>) -> VOut { return place(xz, 0.0); }
@fragment fn fsFill(in: VOut) -> @location(0) vec4<f32> {
  if (in.viewDist > u.maxDist) { discard; }
  return vec4<f32>(u.fogColor, 1.0); // dark surface: occludes lines behind ridges, blends into haze
}
@vertex fn vsLine(@location(0) xz: vec2<f32>) -> VOut { return place(xz, 1.5); } // lift lines just above the fill
@fragment fn fsLine(in: VOut) -> @location(0) vec4<f32> {
  if (in.viewDist > u.maxDist) { discard; }
  let e = clamp(in.height / RELIEF, 0.0, 1.0);
  return vec4<f32>(elevColor(e) * distFog(in.viewDist), 1.0);
}

// --- TOPO: contour lines at constant elevation, computed PER-FRAGMENT from the fBm (smooth, not faceted).
// The shared fill mesh provides coverage + depth (hidden-surface removal); the fragment draws the lines.
@fragment fn fsTopo(in: VOut) -> @location(0) vec4<f32> {
  if (in.viewDist > u.maxDist) { discard; }
  let h = fbmSmooth(in.worldXZ);      // smooth per-fragment height → even, consistent contours
  let f = h / u.interval;
  let g = fract(f);
  let dEdge = min(g, 1.0 - g);        // 0 exactly on a contour
  let w = fwidth(f) * u.lineWidth;    // screen-constant line width
  let line = 1.0 - smoothstep(0.0, w, dEdge);
  let e = clamp(h / RELIEF, 0.0, 1.0);
  let bright = mix(u.floorFade, u.peakGain, e); // floor dim → peaks bright (HDR → bloom)
  let col = elevColor(e) * bright * line * distFog(in.viewDist);
  return vec4<f32>(col, 1.0);
}
`,D=`// trees.wgsl — Forest trees render pass. Anchors each vertex to its tree's ground height, which was
// computed ONCE per tree by the trees_ground.wgsl prepass (same fBm + f32 precision as the rendered
// terrain) and stored in \`grounds\`. Vertex carries world XZ + local height offset + its tree index.
// Responsibilities:
//   - worldPos = (worldX, grounds[treeId] + offY, worldZ).
//   - Per-vertex HDR color × exp distance fog × radial fade (1 near → 0 at fadeEnd) so trees fade in/out
//     with the terrain instead of popping at the streaming-window rim. Additive blend → feeds the bloom.

struct U {
  viewProj: mat4x4<f32>,
  eye: vec3<f32>,
  fogDensity: f32,
  fadeStart: f32,
  fadeEnd: f32,
  depthBias: f32, // metres each vertex is pulled toward the eye before projecting (draw-on-top)
  time: f32,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> grounds: array<f32>; // per-tree ground height (from the prepass)

struct VSOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) color: vec3<f32>,
};

@vertex
fn vs(@location(0) wxyz: vec3<f32>, @location(1) treeId: f32, @location(2) col: vec3<f32>) -> VSOut {
  // wxyz = (worldX, worldZ, localOffsetY); ground comes from the per-tree prepass buffer.
  var o: VSOut;
  let ground = grounds[u32(treeId)];
  let pos = vec3<f32>(wxyz.x, ground + wxyz.z, wxyz.y);
  // DRAW-ON-TOP: pull the vertex toward the eye by depthBias metres before projecting so the tree sits ON
  // TOP of the ridge it stands on (kills the coincident-depth z-fight ripple); a ridge genuinely closer
  // than depthBias still occludes it. Fog/fade below use the TRUE position so distance shading is unchanged.
  let toEye = normalize(u.eye - pos);
  o.clip = u.viewProj * vec4<f32>(pos + toEye * u.depthBias, 1.0);
  let dist = distance(pos, u.eye);
  let fog = exp(-dist * u.fogDensity);
  let fade = clamp((u.fadeEnd - dist) / max(u.fadeEnd - u.fadeStart, 1.0), 0.0, 1.0);
  o.color = col * fog * fade;
  return o;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  return vec4<f32>(in.color, 1.0);
}
`,N=`// trees_ground.wgsl — Per-tree ground-height prepass: compute the terrain fBm ONCE per tree (not once
// per vertex) and write it to a buffer the tree vertex shader reads. Runs only on rebuild (cell crossing),
// replacing ~10× redundant per-vertex fBm evals every frame. fBm is byte-identical to terrain_ekg.wgsl so
// trees sit exactly on the terrain.
const BASE_FREQ: f32 = 0.00142857;
const LACUNARITY: f32 = 2.0;
const GAIN: f32 = 0.5;
const OCTAVES: i32 = 4;
const RELIEF: f32 = 600.0;
const SHARP: f32 = 1.8;
const TERRACES: f32 = 5.0;
const RISER_POW: f32 = 4.0;
const CLIFF_MIX: f32 = 0.65;

// Integer lattice hash — MUST stay bit-identical to ihash() in terrain.ts / terrain_ekg.wgsl
// so trees sit on the SAME ground the bird collides with (old sin-hash diverged f32-vs-f64).
fn ihash(c: vec2<i32>) -> f32 {
  var h: u32 = bitcast<u32>(c.x) * 374761393u + bitcast<u32>(c.y) * 668265263u;
  h = (h ^ (h >> 13u)) * 1274126177u;
  h = h ^ (h >> 16u);
  return f32(h) / 4294967295.0;
}
fn valueNoise(p: vec2<f32>) -> f32 {
  let pf = floor(p); let ci = vec2<i32>(pf); let f = p - pf;
  let a = ihash(ci); let b = ihash(ci + vec2<i32>(1, 0));
  let c = ihash(ci + vec2<i32>(0, 1)); let d = ihash(ci + vec2<i32>(1, 1));
  let uu = f * f * (3.0 - 2.0 * f);
  return mix(mix(a, b, uu.x), mix(c, d, uu.x), uu.y);
}
fn fbm(p: vec2<f32>) -> f32 {
  var freq = BASE_FREQ; var amp = 1.0; var sum = 0.0; var norm = 0.0;
  for (var k = 0; k < OCTAVES; k = k + 1) {
    let n = valueNoise(p * freq);
    let r = 1.0 - abs(2.0 * n - 1.0);
    sum = sum + amp * r; norm = norm + amp; freq = freq * LACUNARITY; amp = amp * GAIN;
  }
  let s = pow(sum / norm, SHARP);
  let b = s * TERRACES;
  let fb = b - floor(b);
  let ter = floor(b) / TERRACES + pow(fb, RISER_POW) / TERRACES;
  return (s + (ter - s) * CLIFF_MIX) * RELIEF;
}

@group(0) @binding(0) var<storage, read> bases: array<vec2<f32>>;   // per-tree base world XZ
@group(0) @binding(1) var<storage, read_write> grounds: array<f32>; // per-tree ground height (output)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&grounds)) { return; }
  grounds[i] = fbm(bases[i]);
}
`,L=`// wind.wgsl — drifting neon CURVED-COMET motes showing the terrain-shaped wind field (WebGPU, NDC z in [0,1]).
// Responsibilities:
//   - Render each wind mote as a CURVED multi-segment ribbon: the host (src/host/gpu/wind.ts) integrates
//     each mote's tail BACKWARD along the terrain-shaped flow (flowAt) into a world-space polyline, then
//     emits one quad per segment. This VS receives, per vertex, a segment ENDPOINT (world xyz), a corner
//     {x = near(0)/far(1) endpoint, y = perpendicular ±1}, the segment's world-XZ direction, the head→tail
//     \`along\` fraction, and a CPU-computed \`vis\` (density cull). The ribbon thickness is laid perpendicular
//     to the segment's SCREEN-space direction so the curved comet has constant on-screen width.
//   - Each mote's world path is advected by the SHARED terrain-shaped flow and persisted frame-to-frame, so
//     the curve you SEE arcing over the ridges is the flow that PUSHES the glider. Motes RISE over windward
//     slopes and sink in lees (vertical advection, host-side) — air visibly pours up and over the ridgelines.
//   - v9: tail LENGTH and DENSITY are computed HOST-side (the tail is a real integrated polyline; density is
//     a CPU rank cull passed in \`vis\`), removing the fragile vertex-index hash. SPEED still tints/brightens.
//   - Bright neon head (cyan→white by wind speed) fading to a transparent tail (alpha along the ribbon),
//     additive (host blend), depth-tested (no write) so terrain ridges occlude the comets; distance fog
//     matches the terrain haze so far motes dissolve cleanly.

struct Uniforms {
  viewProj : mat4x4<f32>,
  eyeAspect : vec4<f32>,   // eye.xyz, aspect (pxW/pxH)
  fog : vec4<f32>,         // fogColor.rgb, fogDensity
  misc : vec4<f32>,        // dotSize (NDC ribbon half-width), pad, pad, pad
};

@group(0) @binding(0) var<uniform> U : Uniforms;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) along : f32,          // 0 at head → 1 at tail (for length fade)
  @location(1) across : f32,         // -1..1 perpendicular (for width falloff)
  @location(2) speedFrac : f32,      // 0..1 wind speed
  @location(3) viewDist : f32,
  @location(4) vis : f32,            // 0..1 density-fade visibility (0 = culled in calm air)
  @location(5) heat : f32,           // 0..1 touched-air heat → warm (yellow→red) tint
};

@vertex
fn vs(
  @location(0) endpoint : vec3<f32>, // this segment endpoint, world space
  @location(1) corner : vec2<f32>,   // corner.x in {0,1} near/far endpoint (already picked host-side), .y perp ±1
  @location(2) speedFrac : f32,
  @location(3) segDir : vec2<f32>,   // this segment's world-XZ direction (unit-ish)
  @location(4) along : f32,          // head=0 → tail=1 fade fraction at this endpoint
  @location(5) vis : f32,            // CPU density cull (0 = culled)
  @location(6) heat : f32            // 0..1 touched-air heat (near tier; far tier = 0)
) -> VSOut {
  var out : VSOut;
  let clip = U.viewProj * vec4<f32>(endpoint, 1.0);
  // Project the segment's world direction to clip space to lay the ribbon perpendicular ON SCREEN.
  let world2 = endpoint + vec3<f32>(segDir.x, 0.0, segDir.y);
  let clip2 = U.viewProj * vec4<f32>(world2, 1.0);
  var sdir = vec2<f32>(
    (clip2.x / clip2.w) - (clip.x / clip.w),
    ((clip2.y / clip2.w) - (clip.y / clip.w)) / U.eyeAspect.w,
  );
  let sl = length(sdir);
  if (sl > 1e-5) { sdir = sdir / sl; } else { sdir = vec2<f32>(1.0, 0.0); }
  let sperp = vec2<f32>(-sdir.y, sdir.x);

  let halfW = U.misc.x;        // NDC ribbon half-width
  // collapse fully-culled motes to a degenerate point so they never draw.
  let cull = step(0.001, vis);
  // thin the ribbon toward the tail so the curved comet tapers to a point.
  let taper = (1.0 - 0.7 * clamp(along, 0.0, 1.0));
  let off2 = sperp * (corner.y * halfW * taper * cull);

  var outClip = clip;
  outClip.x += off2.x * clip.w;
  outClip.y += off2.y * U.eyeAspect.w * clip.w;
  out.clip = outClip;
  out.along = along;
  out.across = corner.y;
  out.speedFrac = speedFrac;
  out.viewDist = length(endpoint - U.eyeAspect.xyz);
  out.vis = vis;
  out.heat = heat;
  return out;
}

const CYAN : vec3<f32> = vec3<f32>(0.30, 0.85, 1.0);
const WHITE : vec3<f32> = vec3<f32>(0.85, 0.97, 1.0);
const YELLOW : vec3<f32> = vec3<f32>(1.0, 0.8, 0.2);  // touched air, gentle wake
const RED : vec3<f32> = vec3<f32>(1.0, 0.25, 0.08);   // touched air, hard wake

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  if (in.vis <= 0.001) { discard; }
  // perpendicular soft falloff → rounded ribbon; length fade → bright head, dissolving tail.
  let perp = 1.0 - abs(in.across);
  if (perp <= 0.0) { discard; }
  let lenFade = pow(1.0 - clamp(in.along, 0.0, 1.0), 1.3); // head bright → tail fades to 0
  let glow = pow(perp, 1.6) * lenFade;

  // v17 "it's AIR — always SOME wind, visible everywhere" (user): raise the calm-air base 0.9→1.6 and
  // soften the speed term 1.2→1.0 so slow/calm air READS brightly against the dark (it was dimming below
  // the bloom threshold and reading as absent). Faster wind still brightens to a whiter core, just gentler.
  // density-fade visibility also modulates brightness so motes entering/leaving fade smoothly.
  let intensity = glow * (1.6 + in.speedFrac * 1.0) * in.vis * (1.0 + 0.5 * in.heat); // touched air a touch brighter so red reads
  // TOUCHED AIR: blend the cool speed tint toward a warm yellow→red by heat (the wind the bird physically touched).
  // DEADZONE smoothstep(0.3, 0.6): low heat (the gently-stirred ambient ball) stays PURE cyan — without it, the
  // faint wake everywhere gives every mote a little heat → a partial cyan↔yellow (complementary) blend that
  // desaturates to GREY. Only air the wake genuinely touched (heat ≳ 0.3) warms; redness still ramps with heat.
  let coolTint = mix(CYAN, WHITE, clamp(in.speedFrac, 0.0, 1.0));
  let warmTint = mix(YELLOW, RED, clamp(in.heat, 0.0, 1.0));
  let tint = mix(coolTint, warmTint, smoothstep(0.3, 0.6, in.heat));

  // distance fog → far motes dissolve into the haze.
  let fog = exp(-U.fog.w * in.viewDist);

  var color = tint * intensity * clamp(fog, 0.0, 1.0);
  color = min(color, vec3<f32>(1.2, 1.3, 1.5));
  return vec4<f32>(color * 0.65, 1.0); // brightness (was 0.5): fewer motes → each reads brighter/cleaner (additive blend)
}
`,z=Object.assign({"../shaders/addone.wgsl":h,"../shaders/bird/bird_update.wgsl":g,"../shaders/bird/scene.wgsl":v,"../shaders/bird3d.wgsl":m,"../shaders/bloom_blur.wgsl":w,"../shaders/bloom_composite.wgsl":b,"../shaders/bloom_threshold.wgsl":y,"../shaders/fluid/add_force_field.wgsl":x,"../shaders/fluid/advect.wgsl":_,"../shaders/fluid/divergence.wgsl":P,"../shaders/fluid/force_field.wgsl":E,"../shaders/fluid/forces.wgsl":S,"../shaders/fluid/jacobi.wgsl":j,"../shaders/fluid/set_bnd.wgsl":R,"../shaders/fluid/shift.wgsl":A,"../shaders/fluid/subtract_grad.wgsl":O,"../shaders/fluid/visualize.wgsl":C,"../shaders/marker.wgsl":k,"../shaders/target.wgsl":T,"../shaders/terrain3d.wgsl":U,"../shaders/terrain_ekg.wgsl":F,"../shaders/terrain_grid.wgsl":I,"../shaders/trees.wgsl":D,"../shaders/trees_ground.wgsl":N,"../shaders/wind.wgsl":L});function G(n){const e=n.replace("/src/host",".."),t=z[e];if(t===void 0)throw new Error(`shader not bundled: ${n} (looked for key ${e})`);return t}class H{constructor(e){l(this,"raf",0);l(this,"last",0);l(this,"running",!1);this.onFrame=e}start(){if(this.running)return;this.running=!0;const e=t=>{if(!this.running)return;const o=this.last?(t-this.last)/1e3:1/60;this.last=t,this.onFrame(o),this.raf=requestAnimationFrame(e)};this.raf=requestAnimationFrame(e)}stop(){this.running=!1,cancelAnimationFrame(this.raf)}}export{H as F,W as a,M as e,G as l,B as m};
