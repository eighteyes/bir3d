var fn=Object.defineProperty;var un=(u,e,n)=>e in u?fn(u,e,{enumerable:!0,configurable:!0,writable:!0,value:n}):u[e]=n;var i=(u,e,n)=>un(u,typeof e!="symbol"?e+"":e,n);import{a as pn,F as gn}from"./frameloop-CY_--yfr.js";import{G as mn}from"./fluid-Bf2m7CRU.js";const wn=`// addone.wgsl — proves the compute pipeline end-to-end: out[i] = in[i] + 1.
@group(0) @binding(0) var<storage, read>       inBuf  : array<f32>;
@group(0) @binding(1) var<storage, read_write> outBuf : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&inBuf)) { return; }
  outBuf[i] = inBuf[i] + 1.0;
}
`,bn=`// bird_update.wgsl — GPU-integrated single-bird physics over the live fluid wind field.
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
`,vn=`// scene.wgsl — bird scene render passes (backdrop + trail + chevron), camera-relative, neon-on-dark.
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
`,xn=`// bird3d.wgsl — neon gliding-V bird (WebGPU 3D, depth-tested so terrain ridges occlude it).
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
`,yn=`// bloom_blur.wgsl — separable Gaussian blur (one axis per pass) for the bloom chain.
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
`,_n=`// bloom_composite.wgsl — final composite + tone-map for the bloom chain.
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
`,Mn=`// bloom_threshold.wgsl — bright-pass extraction for the bloom chain.
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
`,Sn=`// add_force_field.wgsl — add a per-cell force field to the velocity (interior, in-place).
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
`,An=`// advect.wgsl — semi-Lagrangian transport with MANUAL bilinear (matches grid.rs sample).
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
`,Rn=`// divergence.wgsl — central-difference velocity divergence (Stam, h=1).
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
`,Pn=`// force_field.wgsl — add a PER-CELL velocity force field (terrain orographic coupling).
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
`,En=`// forces.wgsl — add scripted velocity force + dye injection (interior, in-place).
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
`,Cn=`// jacobi.wgsl — one Jacobi sweep of the pressure Poisson solve (ping-pong).
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
`,kn=`// set_bnd.wgsl — Stam boundary conditions as TWO passes (edges, then corners).
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
`,Fn=`// shift.wgsl — scroll a bordered (W+2)*(H+2) field by an integer cell offset (world-pinned recenter).
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
`,Tn=`// subtract_grad.wgsl — subtract the pressure gradient from the velocity field.
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
`,Bn=`// visualize.wgsl — debug viz render pipeline: dye field → neon-green ramp on dark.
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
`,zn=`// marker.wgsl — altitude plumb-line + ground diamond for the bird (line-list, additive, depth-tested).
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
`,On=`// target.wgsl — flight target beacon: a camera-facing vertical beam of light at a world waypoint.
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
`,Ln=`// terrain3d.wgsl — neon receding-ridgeline terrain (WebGPU 3D, NDC z in [0,1]).
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
`,Dn=`// terrain_ekg.wgsl — EKG/waveform stacked neon trace LINES + opaque hidden-line FILL (WebGPU, NDC z 0..1).
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
`,Un=`// terrain_grid.wgsl — WORLD-STATIC wireframe terrain. A square grid pinned to WORLD coordinates, draped
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
`,Hn=`// trees.wgsl — Forest trees render pass. Anchors each vertex to its tree's ground height, which was
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
`,In=`// trees_ground.wgsl — Per-tree ground-height prepass: compute the terrain fBm ONCE per tree (not once
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
`,Nn=`// wind.wgsl — drifting neon CURVED-COMET motes showing the terrain-shaped wind field (WebGPU, NDC z in [0,1]).
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
`,Gn=.00142857,Vn=2,Wn=.5,jn=4,Xn=600,Yn=1.8,bt=5,Zn=4,qn=.65;class $n{constructor(e,n,r,s={}){i(this,"rows");i(this,"cols");i(this,"rowSpacing");i(this,"nearDenseDepth");i(this,"farSpread");i(this,"rowStart");i(this,"halfWidth");i(this,"maxDist");i(this,"vbuf");i(this,"vertexCount");i(this,"fillBuf");i(this,"fillVertexCount");i(this,"ubuf");i(this,"pipeline");i(this,"fillPipeline");i(this,"bindGroup");i(this,"fillBindGroup");i(this,"uniformHost");i(this,"uniformData");i(this,"fogColor");i(this,"fogDensity");i(this,"baseline");i(this,"rowDepths");this.device=e,this.rows=s.rows??50,this.cols=s.cols??256,this.rowSpacing=s.rowSpacing??70,this.rowStart=s.rowStart??40,this.halfWidth=s.halfWidth??1400,this.maxDist=s.maxDist??this.rowStart+this.rows*this.rowSpacing,this.fogColor=s.fogColor??[.01,.012,.03],this.fogDensity=s.fogDensity??1/1600,this.baseline=s.baseline??-300,this.nearDenseDepth=s.nearDenseDepth??0,this.farSpread=s.farSpread??1/0;const l=[];let t=this.rowStart;for(;t<=this.maxDist&&l.length<this.rows;){l.push(t);const f=Math.max(0,t-this.nearDenseDepth);t+=this.rowSpacing*(1+f/this.farSpread)}this.rowDepths=Float32Array.from(l),this.rows=l.length;const h=this.cols-1;this.vertexCount=this.rows*h*2;const a=3,c=new Float32Array(this.vertexCount*a);let o=0;for(let f=0;f<this.rows;f++){const y=this.rowDepthAt(f),m=this.rows>1?f/(this.rows-1):0;for(let M=0;M<h;M++){const _=M/h*2-1,R=(M+1)/h*2-1;c[o++]=_,c[o++]=y,c[o++]=m,c[o++]=R,c[o++]=y,c[o++]=m}}this.vbuf=e.createBuffer({size:c.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.vbuf,0,c),this.fillVertexCount=this.rows*h*6;const d=new Float32Array(this.fillVertexCount*a);let p=0;const g=(f,y,m)=>{d[p++]=f,d[p++]=y,d[p++]=m};for(let f=0;f<this.rows;f++){const y=this.rowDepthAt(f);for(let m=0;m<h;m++){const M=m/h*2-1,_=(m+1)/h*2-1;g(M,y,1),g(_,y,1),g(M,y,0),g(M,y,0),g(_,y,1),g(_,y,0)}}this.fillBuf=e.createBuffer({size:d.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.fillBuf,0,d),this.uniformHost=new ArrayBuffer(32*4),this.uniformData=new Float32Array(this.uniformHost),this.ubuf=e.createBuffer({size:this.uniformData.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const w=s.sampleCount??1,v=e.createShaderModule({code:n}),b={arrayStride:a*4,attributes:[{shaderLocation:0,offset:0,format:"float32x2"},{shaderLocation:1,offset:8,format:"float32"}]};this.fillPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:v,entryPoint:"vsFill",buffers:[b]},fragment:{module:v,entryPoint:"fsFill",targets:[{format:r}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less",format:"depth24plus"},multisample:{count:w}}),this.pipeline=e.createRenderPipeline({layout:"auto",vertex:{module:v,entryPoint:"vsLine",buffers:[b]},fragment:{module:v,entryPoint:"fsLine",targets:[{format:r,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"line-list",cullMode:"none"},depthStencil:{depthWriteEnabled:!1,depthCompare:"less-equal",format:"depth24plus"},multisample:{count:w}}),this.bindGroup=e.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.ubuf}}]}),this.fillBindGroup=e.createBindGroup({layout:this.fillPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.ubuf}}]})}rowDepthAt(e){return this.rowDepths[e]}ihash(e,n){let r=Math.imul(e|0,374761393)+Math.imul(n|0,668265263)>>>0;return r=Math.imul(r^r>>>13,1274126177)>>>0,r=(r^r>>>16)>>>0,r/4294967295}valueNoise(e,n){const r=Math.floor(e),s=Math.floor(n),l=e-r,t=n-s,h=this.ihash(r,s),a=this.ihash(r+1,s),c=this.ihash(r,s+1),o=this.ihash(r+1,s+1),d=l*l*(3-2*l),p=t*t*(3-2*t),g=h+(a-h)*d,w=c+(o-c)*d;return g+(w-g)*p}sampleHeight(e,n){let r=Gn,s=1,l=0,t=0;for(let d=0;d<jn;d++){const p=this.valueNoise(e*r,n*r),g=1-Math.abs(2*p-1);l+=s*g,t+=s,r*=Vn,s*=Wn}const h=Math.pow(l/t,Yn),a=h*bt,c=a-Math.floor(a),o=Math.floor(a)/bt+Math.pow(c,Zn)/bt;return(h+(o-h)*qn)*Xn}draw(e,n,r,s,l,t,h,a,c){const o=this.uniformData;o.set(s,0),o[16]=l[0],o[17]=l[1],o[18]=this.halfWidth,o[19]=this.maxDist,o[20]=t[0],o[21]=t[1],o[22]=h[0],o[23]=h[1],o[24]=this.fogColor[0],o[25]=this.fogColor[1],o[26]=this.fogColor[2],o[27]=this.fogDensity,o[28]=a[0],o[29]=a[1],o[30]=a[2],o[31]=this.baseline,this.device.queue.writeBuffer(this.ubuf,0,this.uniformHost);const d=e.beginRenderPass({colorAttachments:[{view:n,loadOp:"clear",storeOp:"store",clearValue:c}],depthStencilAttachment:{view:r,depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});d.setPipeline(this.fillPipeline),d.setBindGroup(0,this.fillBindGroup),d.setVertexBuffer(0,this.fillBuf),d.draw(this.fillVertexCount),d.setPipeline(this.pipeline),d.setBindGroup(0,this.bindGroup),d.setVertexBuffer(0,this.vbuf),d.draw(this.vertexCount),d.end()}}const at=2,Tt=112;class Kn{constructor(e,n,r,s={}){i(this,"mode","grid");i(this,"interval",22);i(this,"lineWidth",1.3);i(this,"floorFade",.28);i(this,"peakGain",1.9);i(this,"device");i(this,"spacing");i(this,"radius");i(this,"maxDist");i(this,"fogColor");i(this,"fogDensity");i(this,"fillHost");i(this,"lineHost");i(this,"fillBuf");i(this,"lineBuf");i(this,"fillCount",0);i(this,"lineCount",0);i(this,"ubuf");i(this,"uniformHost",new ArrayBuffer(Tt));i(this,"u",new Float32Array(this.uniformHost));i(this,"fillPipeline");i(this,"topoPipeline");i(this,"linePipeline");i(this,"fillBind");i(this,"topoBind");i(this,"lineBind");i(this,"lastCellX",Number.NaN);i(this,"lastCellZ",Number.NaN);this.device=e,this.spacing=s.spacing??24,this.radius=s.radius??1500,this.maxDist=s.maxDist??1500,this.fogColor=s.fogColor??[.01,.012,.03],this.fogDensity=s.fogDensity??.5/1100;const l=s.sampleCount??1,t=2*Math.ceil(this.radius/this.spacing)+1,h=t*t;this.fillHost=new Float32Array(h*6*at),this.lineHost=new Float32Array(h*4*at),this.fillBuf=e.createBuffer({size:this.fillHost.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),this.lineBuf=e.createBuffer({size:this.lineHost.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),this.ubuf=e.createBuffer({size:Tt,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const a=e.createShaderModule({code:n}),c=[{arrayStride:at*4,attributes:[{shaderLocation:0,offset:0,format:"float32x2"}]}];this.fillPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:a,entryPoint:"vsFill",buffers:c},fragment:{module:a,entryPoint:"fsFill",targets:[{format:r}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less",format:"depth24plus"},multisample:{count:l}}),this.linePipeline=e.createRenderPipeline({layout:"auto",vertex:{module:a,entryPoint:"vsLine",buffers:c},fragment:{module:a,entryPoint:"fsLine",targets:[{format:r,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"line-list"},depthStencil:{depthWriteEnabled:!1,depthCompare:"less-equal",format:"depth24plus"},multisample:{count:l}}),this.topoPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:a,entryPoint:"vsFill",buffers:c},fragment:{module:a,entryPoint:"fsTopo",targets:[{format:r}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less",format:"depth24plus"},multisample:{count:l}}),this.fillBind=e.createBindGroup({layout:this.fillPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.ubuf}}]}),this.topoBind=e.createBindGroup({layout:this.topoPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.ubuf}}]}),this.lineBind=e.createBindGroup({layout:this.linePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.ubuf}}]})}rebuild(e,n){const r=this.spacing,s=this.radius,l=Math.ceil(s/r),t=Math.round(e/r),h=Math.round(n/r),a=this.fillHost,c=this.lineHost;let o=0,d=0;for(let p=-l;p<=l;p++)for(let g=-l;g<=l;g++){const w=(t+g)*r,v=(h+p)*r;if(Math.hypot(w-e,v-n)>s)continue;const b=w+r,f=v+r;a[o++]=w,a[o++]=v,a[o++]=b,a[o++]=v,a[o++]=w,a[o++]=f,a[o++]=b,a[o++]=v,a[o++]=b,a[o++]=f,a[o++]=w,a[o++]=f,c[d++]=w,c[d++]=v,c[d++]=b,c[d++]=v,c[d++]=w,c[d++]=v,c[d++]=w,c[d++]=f}this.fillCount=o/at,this.lineCount=d/at,this.device.queue.writeBuffer(this.fillBuf,0,this.fillHost.buffer,0,o*4),this.device.queue.writeBuffer(this.lineBuf,0,this.lineHost.buffer,0,d*4)}draw(e,n,r,s,l,t,h){const a=Math.round(l[0]/this.spacing),c=Math.round(l[1]/this.spacing);(a!==this.lastCellX||c!==this.lastCellZ)&&(this.rebuild(l[0],l[1]),this.lastCellX=a,this.lastCellZ=c);const o=this.u;o.set(s,0),o[16]=t[0],o[17]=t[1],o[18]=t[2],o[19]=this.fogDensity,o[20]=this.fogColor[0],o[21]=this.fogColor[1],o[22]=this.fogColor[2],o[23]=this.maxDist,o[24]=this.interval,o[25]=this.lineWidth,o[26]=this.floorFade,o[27]=this.peakGain,this.device.queue.writeBuffer(this.ubuf,0,this.uniformHost);const d=e.beginRenderPass({colorAttachments:[{view:n,loadOp:"clear",storeOp:"store",clearValue:h}],depthStencilAttachment:{view:r,depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});this.mode==="topo"?(d.setPipeline(this.topoPipeline),d.setBindGroup(0,this.topoBind),d.setVertexBuffer(0,this.fillBuf),d.draw(this.fillCount)):(d.setPipeline(this.fillPipeline),d.setBindGroup(0,this.fillBind),d.setVertexBuffer(0,this.fillBuf),d.draw(this.fillCount),d.setPipeline(this.linePipeline),d.setBindGroup(0,this.lineBind),d.setVertexBuffer(0,this.lineBuf),d.draw(this.lineCount)),d.end()}}const Ft={curlScale:.0011,curlAmp:7,driftDir:35*Math.PI/180,driftAmp:6,thermalAmp:5};let sn=null;const rn={fluidMax:10};function Jn(u,e,n,r,s,l,t,h){sn={u,v:e,gridW:n,gridH:r,originX:s,originZ:l,cellM:t,scale:h}}function lt(u,e,n){return(Math.sin(u*1+.15*n)*Math.cos(e*1.3)+.6*Math.sin((u+e)*.7-.1*n)+.4*Math.cos(u*.5-e*.9+.07*n))*1}function vt(u,e,n,r,s){const l=s*s,t=l*s;return .5*(2*e+(-u+n)*s+(2*u-5*e+4*n-r)*l+(-u+3*e-3*n+r)*t)}function on(u,e,n,r,s){const l=Math.hypot(n,r);if(l>1e-5){const t=n/l,h=r/l,a=u*t+e*h;if(a>0){const c=Math.min(1,l*4),o=s*c;u-=o*a*t,e-=o*a*h}}return[u,e]}const it=[0,0];function pt(u,e,n,r=Ft){const s=Math.sin(r.driftDir)*r.driftAmp,l=Math.cos(r.driftDir)*r.driftAmp,t=sn;if(t){const g=(u-t.originX)/t.cellM,w=(e-t.originZ)/t.cellM;if(g>=0&&w>=0&&g<=t.gridW-1&&w<=t.gridH-1){const v=g|0,b=w|0,f=v+1<t.gridW?v+1:v,y=b+1<t.gridH?b+1:b,m=g-v,M=w-b,_=t.gridW+2,R=t.u,T=t.v,B=v+1+_*(b+1),P=f+1+_*(b+1),A=v+1+_*(y+1),D=f+1+_*(y+1);let k=((R[B]*(1-m)+R[P]*m)*(1-M)+(R[A]*(1-m)+R[D]*m)*M)*t.scale,S=((T[B]*(1-m)+T[P]*m)*(1-M)+(T[A]*(1-m)+T[D]*m)*M)*t.scale;const C=Math.hypot(k,S),O=rn.fluidMax;if(C>O){const L=O/C;k*=L,S*=L}return it[0]=k+s,it[1]=S+l,it}}const h=r.curlScale,a=.75,c=u*h,o=e*h,d=(lt(c,o+a,n)-lt(c,o-a,n))/(2*a),p=(lt(c+a,o,n)-lt(c-a,o,n))/(2*a);return it[0]=d*r.curlAmp+s,it[1]=-p*r.curlAmp+l,it}function Qn(u,e,n,r=Ft){const l=Math.sin(u*.0013+.05*n)*Math.sin(e*.0013*1.1-.04*n),t=Math.sin((u+e)*.0013*.6+.03*n),h=Math.max(0,l)*Math.max(0,t);return Math.pow(h,1.8)*r.thermalAmp}function xt(u){const e=Math.sin(u*12.9898)*43758.5453;return e-Math.floor(e)}function Fe(u,e,n){if(u===e)return n<u?0:1;const r=Math.min(1,Math.max(0,(n-u)/(e-u)));return r*r*(3-2*r)}const $e={loScale:.4,hiScale:1.4,altLo:100,altHi:500};function ei(u){Object.assign($e,u);const e=$e;e.loScale=Math.max(0,e.loScale),e.hiScale=Math.max(0,e.hiScale),e.altHi<=e.altLo&&(e.altHi=e.altLo+1)}function Ne(u){const e=$e;return e.loScale+(e.hiScale-e.loScale)*Fe(e.altLo,e.altHi,u)}function ti(){return $e.hiScale}const ni=["comet","stipple","chevron"],ii=["comet","flecks","filaments"],si=["modulate","helix","rings"],E=class E{constructor(e,n,r,s,l={},t={},h=1){i(this,"cfg");i(this,"count");i(this,"spanAhead");i(this,"spanBehind");i(this,"spanWide");i(this,"clearance");i(this,"minClear");i(this,"maxClear");i(this,"vSpread");i(this,"homeBias");i(this,"nearBias");i(this,"liftGain");i(this,"relax");i(this,"deflect");i(this,"segments");i(this,"segStep");i(this,"dotPx");i(this,"tailFloor");i(this,"densityFloor");i(this,"speedLo");i(this,"speedHi");i(this,"nearCount");i(this,"nearBodyCount");i(this,"nearWakeCount");i(this,"wakeMoteLen");i(this,"wakeSpeedRef");i(this,"bodyWindRef");i(this,"nearOpacity");i(this,"wakeOpacity");i(this,"_wakeCountNow",0);i(this,"_bodyCountNow",0);i(this,"_curOp",0);i(this,"nearRadius");i(this,"nearSegments");i(this,"nearSegStep");i(this,"nearJitter");i(this,"fadeInTime");i(this,"fadeFarEdge");i(this,"fadeNearEdge");i(this,"bowGain");i(this,"wakeGain");i(this,"swirlGain");i(this,"wingSpan");i(this,"vortexCore");i(this,"wingEmitFrac");i(this,"wingJitter");i(this,"ambientNearFloor");i(this,"heatTau");i(this,"heatRef");i(this,"heatLenGain");i(this,"foreStretch");i(this,"dashCountK");i(this,"dashLenM");i(this,"gapRatio");i(this,"lenByAltitude");i(this,"leadBoost");i(this,"spreadAngleDeg");i(this,"limbLenM");i(this,"apexBoost");i(this,"rakeBySpeed");i(this,"fleckLen");i(this,"shearGain");i(this,"shearRadius");i(this,"fleckTaper");i(this,"orientLerp");i(this,"filSegStep");i(this,"wakeEmitRate");i(this,"wakeLife");i(this,"helixGain");i(this,"wakeSeg");i(this,"wakeSegStep");i(this,"wakeTaper");i(this,"counterRotate");i(this,"ringRate");i(this,"ringGrow");i(this,"ringLife");i(this,"ringSegN");i(this,"ringStartRadius");i(this,"ringTilt");i(this,"twinOffset");i(this,"convectFrac");i(this,"ringWarmBias");i(this,"fleckDirX");i(this,"fleckDirZ");i(this,"_wake",[0,0,0]);i(this,"_ax",0);i(this,"_ay",0);i(this,"_az",0);i(this,"_rx",1);i(this,"_ry",0);i(this,"_rz",0);i(this,"_bs",0);i(this,"_moving",!1);i(this,"_wakeOn",!1);i(this,"showNear",!1);i(this,"showWake",!1);i(this,"farMode","comet");i(this,"nearMode","comet");i(this,"wakeMode","modulate");i(this,"_lastBirdPos",[0,0,0]);i(this,"nx");i(this,"ny");i(this,"nz");i(this,"nearAge");i(this,"nearHeat");i(this,"nearJit");i(this,"nptX");i(this,"nptY");i(this,"nptZ");i(this,"nearSeeded",!1);i(this,"farVertexCount");i(this,"nearVertexCount");i(this,"wakeShedVertexCount");i(this,"wakeShedLiveCount",0);i(this,"helixSeedX");i(this,"helixSeedY");i(this,"helixSeedZ");i(this,"helixAge");i(this,"helixSide");i(this,"helixActive",0);i(this,"helixEmitAcc",0);i(this,"ringCx");i(this,"ringCy");i(this,"ringCz");i(this,"ringRadius");i(this,"ringAge");i(this,"ringSide");i(this,"ringHeat");i(this,"ringActive",0);i(this,"ringEmitAcc",0);i(this,"wakeShedPoolsInit",!1);i(this,"wakeOverrunLogged",!1);i(this,"wsPtX");i(this,"wsPtY");i(this,"wsPtZ");i(this,"_wsWake",[0,0,0]);i(this,"px");i(this,"py");i(this,"pz");i(this,"pHome");i(this,"speedFrac");i(this,"age");i(this,"vbuf");i(this,"vertexCount");i(this,"vertBytes");i(this,"vertHost");i(this,"ubuf");i(this,"uniformHost");i(this,"uniformF32");i(this,"pipeline");i(this,"bindGroup");i(this,"ptX");i(this,"ptY");i(this,"ptZ");i(this,"sptX");i(this,"sptY");i(this,"sptZ");i(this,"lastTime",-1);i(this,"seeded",!1);this.device=e,this.sampleHeight=s,this.cfg={...Ft,...l},this.count=t.numMotes??1600,this.spanAhead=t.spanAhead??950,this.spanBehind=t.spanBehind??260,this.spanWide=t.spanWide??950,this.clearance=t.clearance??30,this.minClear=t.minClear??12,this.maxClear=t.maxClear??200,this.vSpread=t.vSpread??70,this.homeBias=t.homeBias??2.5,this.liftGain=t.liftGain??.6,this.relax=t.relax??.8,this.deflect=t.deflect??.45,this.nearBias=t.nearBias??1.3,this.segments=t.segments??4,this.segStep=t.segStep??.8,this.dotPx=t.dotPx??2.6,this.tailFloor=t.tailFloor??.25,this.densityFloor=t.densityFloor??.6,this.speedLo=t.speedLo??2,this.speedHi=t.speedHi??15,this.nearCount=t.nearCount??1600,this.nearBodyCount=Math.min(t.nearBodyCount??400,this.nearCount),this.nearWakeCount=t.nearWakeCount??1e3,this.wakeMoteLen=t.wakeMoteLen??.5,this.wakeSpeedRef=t.wakeSpeedRef??45,this.bodyWindRef=t.bodyWindRef??15,this.nearOpacity=t.nearOpacity??.5,this.wakeOpacity=t.wakeOpacity??.25,this.nearRadius=t.nearRadius??65,this.nearSegments=t.nearSegments??4,this.nearSegStep=t.nearSegStep??.12,this.nearJitter=t.nearJitter??.12,this.fadeInTime=t.fadeInTime??.55,this.fadeFarEdge=t.fadeFarEdge??120,this.fadeNearEdge=t.fadeNearEdge??.78,this.bowGain=t.bowGain??.45,this.wakeGain=t.wakeGain??.75,this.swirlGain=t.swirlGain??.5,this.wingSpan=t.wingSpan??10,this.vortexCore=t.vortexCore??6,this.wingEmitFrac=t.wingEmitFrac??.5,this.wingJitter=t.wingJitter??3,this.ambientNearFloor=t.ambientNearFloor??1,this.heatTau=t.heatTau??1.5,this.heatRef=t.heatRef??24,this.heatLenGain=t.heatLenGain??1,this.foreStretch=t.foreStretch??2.6,this.dashCountK=t.dashCountK??3,this.dashLenM=t.dashLenM??9,this.gapRatio=t.gapRatio??1.5,this.lenByAltitude=t.lenByAltitude??.6,this.leadBoost=t.leadBoost??1.5,this.spreadAngleDeg=t.spreadAngleDeg??28,this.limbLenM=t.limbLenM??14,this.apexBoost=t.apexBoost??1.5,this.rakeBySpeed=t.rakeBySpeed??.5,this.fleckLen=t.fleckLen??3,this.shearGain=t.shearGain??1.5,this.shearRadius=t.shearRadius??3,this.fleckTaper=t.fleckTaper??.2,this.orientLerp=t.orientLerp??0,this.filSegStep=t.filSegStep??.25,this.wakeEmitRate=t.wakeEmitRate??60,this.wakeLife=t.wakeLife??1.2,this.helixGain=t.helixGain??1,this.wakeSeg=Math.min(E.HELIX_SEGS,Math.max(1,Math.round(t.wakeSeg??3))),this.wakeSegStep=t.wakeSegStep??.1,this.wakeTaper=t.wakeTaper??.7,this.counterRotate=t.counterRotate??!0,this.ringRate=t.ringRate??6,this.ringGrow=t.ringGrow??8,this.ringLife=t.ringLife??1.5,this.ringSegN=Math.min(E.RING_CHORDS,Math.max(3,Math.round(t.ringSegN??24))),this.ringStartRadius=t.ringStartRadius??2,this.ringTilt=t.ringTilt??.3,this.twinOffset=t.twinOffset??10,this.convectFrac=t.convectFrac??.7,this.ringWarmBias=t.ringWarmBias??.5,this.px=new Float32Array(this.count),this.py=new Float32Array(this.count),this.pz=new Float32Array(this.count),this.pHome=new Float32Array(this.count),this.speedFrac=new Float32Array(this.count),this.age=new Float32Array(this.count),this.ptX=new Float32Array(this.segments+1),this.ptY=new Float32Array(this.segments+1),this.ptZ=new Float32Array(this.segments+1);const a=this.segments*E.FAR_SUBDIV;this.sptX=new Float32Array(a+1),this.sptY=new Float32Array(a+1),this.sptZ=new Float32Array(a+1),this.nx=new Float32Array(this.nearCount),this.ny=new Float32Array(this.nearCount),this.nz=new Float32Array(this.nearCount),this.nearAge=new Float32Array(this.nearCount),this.nearHeat=new Float32Array(this.nearCount),this.nearJit=new Float32Array(this.nearCount),this.fleckDirX=new Float32Array(this.nearCount),this.fleckDirZ=new Float32Array(this.nearCount),this.nptX=new Float32Array(this.nearSegments+1),this.nptY=new Float32Array(this.nearSegments+1),this.nptZ=new Float32Array(this.nearSegments+1);const c=E.HELIX_TIPS*E.HELIX_LIVE;this.helixSeedX=new Float32Array(c),this.helixSeedY=new Float32Array(c),this.helixSeedZ=new Float32Array(c),this.helixAge=new Float32Array(c),this.helixSide=new Float32Array(c),this.ringCx=new Float32Array(E.RING_COUNT),this.ringCy=new Float32Array(E.RING_COUNT),this.ringCz=new Float32Array(E.RING_COUNT),this.ringRadius=new Float32Array(E.RING_COUNT),this.ringAge=new Float32Array(E.RING_COUNT),this.ringSide=new Float32Array(E.RING_COUNT),this.ringHeat=new Float32Array(E.RING_COUNT),this.wsPtX=new Float32Array(E.HELIX_SEGS+1),this.wsPtY=new Float32Array(E.HELIX_SEGS+1),this.wsPtZ=new Float32Array(E.HELIX_SEGS+1),this.wakeShedPoolsInit=!0,this.farVertexCount=this.count*this.segments*E.FAR_SUBDIV*6,this.nearVertexCount=this.nearCount*this.nearSegments*6,this.wakeShedVertexCount=E.WAKE_SHED_RESERVE,this.vertexCount=this.farVertexCount+this.nearVertexCount+this.wakeShedVertexCount,this.vertBytes=new ArrayBuffer(this.vertexCount*E.FPV*4),this.vertHost=new Float32Array(this.vertBytes),this.vbuf=e.createBuffer({size:this.vertHost.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),this.uniformHost=new ArrayBuffer(28*4),this.uniformF32=new Float32Array(this.uniformHost),this.ubuf=e.createBuffer({size:this.uniformHost.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const o=e.createShaderModule({code:n});this.pipeline=e.createRenderPipeline({layout:"auto",vertex:{module:o,entryPoint:"vs",buffers:[{arrayStride:E.FPV*4,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x2"},{shaderLocation:2,offset:20,format:"float32"},{shaderLocation:3,offset:24,format:"float32x2"},{shaderLocation:4,offset:32,format:"float32"},{shaderLocation:5,offset:36,format:"float32"},{shaderLocation:6,offset:40,format:"float32"}]}]},fragment:{module:o,entryPoint:"fs",targets:[{format:r,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{depthWriteEnabled:!1,depthCompare:"less",format:"depth24plus"},multisample:{count:h}}),this.bindGroup=e.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.ubuf}}]})}seedMote(e,n,r,s,l,t){const h=Math.max(this.spanAhead*.08,l),a=h+Math.pow(Math.random(),this.nearBias)*(this.spanAhead-h),c=Math.min(1,Math.max(.15,a/this.spanAhead));let o=Math.random(),d=t;d===0&&(d=Math.random()<.5?-1:1);const p=d*o*this.spanWide*c,g=n[0]+r[0]*a+s[0]*p,w=n[1]+r[1]*a+s[1]*p;this.px[e]=g,this.pz[e]=w;const v=Math.max(this.minClear,this.clearance-this.vSpread),b=Math.min(this.maxClear,this.clearance+this.vSpread),f=v+Math.pow(Math.random(),this.homeBias)*(b-v);this.pHome[e]=f,this.py[e]=this.sampleHeight(g,w)+f,this.age[e]=0}seedNearMote(e,n){const r=this.nearRadius;let s,l,t;if(this._wakeOn&&(this.wakeMode==="modulate"&&this.nearMode==="comet"&&e>=this.nearBodyCount||this.nearMode==="filaments"&&Math.random()<this.wingEmitFrac)){const c=Math.random()<.5?1:-1,o=this._ax,d=this._ay,p=this._az,g=this._rx,w=this._ry,v=this._rz,b=d*v-p*w,f=p*g-o*v,y=o*w-d*g,m=r*(.1+.45*Math.random()),M=c*this.wingSpan*(.55+.45*Math.random())+(Math.random()*2-1)*this.wingJitter,_=(Math.random()*2-1)*this.wingJitter;s=n[0]+o*m+g*M+b*_,l=n[1]+d*m+w*M+f*_,t=n[2]+p*m+v*M+y*_}else{const c=r*Math.cbrt(Math.random()),o=2*Math.random()-1,d=Math.sqrt(Math.max(0,1-o*o)),p=2*Math.PI*Math.random();let g=c*d*Math.cos(p),w=c*o,v=c*d*Math.sin(p);if(this._moving){const b=g*this._ax+w*this._ay+v*this._az;if(b>0){const f=b*(this.foreStretch-1);g+=f*this._ax,w+=f*this._ay,v+=f*this._az}}s=n[0]+g,l=n[1]+w,t=n[2]+v}const a=this.sampleHeight(s,t)+this.minClear;this.nx[e]=s,this.ny[e]=l<a?a:l,this.nz[e]=t,this.nearAge[e]=0,this.nearHeat[e]=0,this.nearJit[e]=Math.random()*2-1}flowAt(e,n,r){const[s,l]=pt(e,n,r,this.cfg),t=6,h=(this.sampleHeight(e+t,n)-this.sampleHeight(e-t,n))/(2*t),a=(this.sampleHeight(e,n+t)-this.sampleHeight(e,n-t))/(2*t);let c=this.liftGain*(s*h+l*a);c>E.W_CLAMP?c=E.W_CLAMP:c<-12&&(c=-12);const[o,d]=on(s,l,h,a,this.deflect);return[o,d,c]}step(e,n,r,s,l){if(!this.seeded){for(let a=0;a<this.count;a++)this.seedMote(a,e,n,r,-this.spanBehind,0);this.seeded=!0}let t=this.lastTime<0?0:s-this.lastTime;this.lastTime=s,t<0&&(t=0),t>.05&&(t=.05);let h=0;for(let a=0;a<this.count;a++)switch(this.farMode){case"stipple":h=this.emitFarStipple(a,e,n,r,s,l,t,h);break;case"chevron":h=this.emitFarChevron(a,e,n,r,s,l,t,h);break;case"comet":default:h=this.emitFarComet(a,e,n,r,s,l,t,h);break}}emitFarComet(e,n,r,s,l,t,h,a){const c=this.speedLo,o=this.speedHi,d=this.vertHost,p=E.CORNERS,g=this.segments,w=g*E.FAR_SUBDIV,v=this.ptX,b=this.ptY,f=this.ptZ,y=this.sptX,m=this.sptY,M=this.sptZ;let _=this.px[e],R=this.pz[e],T=this.py[e];const B=_-n[0],P=R-n[1],A=B*r[0]+P*r[1],D=B*s[0]+P*s[1];A<-this.spanBehind||A>this.spanAhead?(this.seedMote(e,n,r,s,this.spanAhead*.6,0),_=this.px[e],R=this.pz[e],T=this.py[e]):Math.abs(D)>this.spanWide&&(this.seedMote(e,n,r,s,-this.spanBehind,D>0?-1:1),_=this.px[e],R=this.pz[e],T=this.py[e]);const[k,S,C]=this.flowAt(_,R,l),O=Ne(T),L=k*O,z=S*O,H=_+L*h,j=R+z*h,G=this.sampleHeight(H,j);let V=T+C*h;V+=(G+this.pHome[e]-V)*Math.min(1,this.relax*h);const Y=G+this.minClear,ie=G+this.maxClear;V<Y&&(V=Y),V>ie&&(V=ie),this.px[e]=H,this.py[e]=V,this.pz[e]=j;const pe=Math.hypot(L,z),ee=Math.min(1,Math.max(0,(pe-c)/(o-c)));let he=ee*ee*(3-2*ee);const ce=Math.min(1,Math.max(0,C/7));he=Math.max(he,ce),this.speedFrac[e]=he;const ge=xt(e),se=this.densityFloor+(1-this.densityFloor)*he;let q=1-Fe(se-.1,se+.02,ge);const $=H-t[0],te=j-t[2],ye=Math.hypot($,te),K=this.nearRadius*1.6;if(ye<K){const F=1-Fe(this.nearRadius,K,ye);q=Math.max(q,.85*F)}this.age[e]+=h;const _e=Fe(0,this.fadeInTime,this.age[e]),be=H-n[0],Ee=j-n[1],N=be*r[0]+Ee*r[1],X=be*s[0]+Ee*s[1],Me=this.spanAhead-N,Z=N+this.spanBehind,Se=this.spanWide-Math.abs(X),Ae=Math.min(Me,Z,Se),ve=Fe(0,this.fadeFarEdge,Ae),ae=V-t[1],xe=this.bubbleFrac($,ae,te),we=Fe(.35,1,xe);if(q*=_e*ve*we,q<=.001){for(let F=0;F<w;F++)for(let W=0;W<6;W++)d[a++]=H,d[a++]=V,d[a++]=j,d[a++]=0,d[a++]=0,d[a++]=he,d[a++]=0,d[a++]=0,d[a++]=1,d[a++]=0,d[a++]=0;return a}const re=this.segStep*(this.tailFloor+(1-this.tailFloor)*he);v[0]=H,b[0]=V,f[0]=j;let le=H,de=V,Ce=j,x=L,I=z,U=C;for(let F=1;F<=g;F++){if(F>1&&(F&1)===1){const J=this.flowAt(le,Ce,l);x=J[0]*O,I=J[1]*O,U=J[2]}le-=x*re,Ce-=I*re,de-=U*re;const W=this.sampleHeight(le,Ce)+this.minClear;de<W&&(de=W),v[F]=le,b[F]=de,f[F]=Ce}for(let F=0;F<g;F++){const W=F>0?F-1:0,J=F+1,Q=F+2<=g?F+2:g;for(let ne=0;ne<E.FAR_SUBDIV;ne++){const fe=ne/E.FAR_SUBDIV,ue=F*E.FAR_SUBDIV+ne;y[ue]=vt(v[W],v[F],v[J],v[Q],fe),m[ue]=vt(b[W],b[F],b[J],b[Q],fe),M[ue]=vt(f[W],f[F],f[J],f[Q],fe);const Te=b[F]*(1-fe)+b[J]*fe;m[ue]<Te&&(m[ue]=Te)}}y[w]=v[g],m[w]=b[g],M[w]=f[g];for(let F=0;F<w;F++){const W=y[F],J=m[F],Q=M[F],ne=y[F+1],fe=m[F+1],ue=M[F+1];let Te=ne-W,ze=ue-Q;const Re=Math.hypot(Te,ze);Re>1e-5?(Te/=Re,ze/=Re):(Te=1,ze=0);const Pe=F/w,me=(F+1)/w;for(let ke=0;ke<6;ke++){const[Be,Oe]=p[ke],He=Be>.5?ne:W,Ue=Be>.5?fe:J,Ie=Be>.5?ue:Q,Ge=Be>.5?me:Pe;d[a++]=He,d[a++]=Ue,d[a++]=Ie,d[a++]=Be,d[a++]=Oe,d[a++]=he,d[a++]=Te,d[a++]=ze,d[a++]=Ge,d[a++]=q,d[a++]=0}}return a}get FAR_VERTS_PER_MOTE(){return this.segments*E.FAR_SUBDIV*6}emitDegenerateFarVert(e){const n=this.vertHost;return n[e++]=0,n[e++]=0,n[e++]=0,n[e++]=0,n[e++]=0,n[e++]=0,n[e++]=0,n[e++]=0,n[e++]=0,n[e++]=0,n[e++]=0,e}emitFarStipple(e,n,r,s,l,t,h,a){const c=a+this.FAR_VERTS_PER_MOTE*E.FPV,o=this.vertHost,d=E.CORNERS,p=this.segments,g=this.ptX,w=this.ptY,v=this.ptZ;let b=this.px[e],f=this.pz[e],y=this.py[e];const m=b-n[0],M=f-n[1],_=m*r[0]+M*r[1],R=m*s[0]+M*s[1];_<-this.spanBehind||_>this.spanAhead?(this.seedMote(e,n,r,s,this.spanAhead*.6,0),b=this.px[e],f=this.pz[e],y=this.py[e]):Math.abs(R)>this.spanWide&&(this.seedMote(e,n,r,s,-this.spanBehind,R>0?-1:1),b=this.px[e],f=this.pz[e],y=this.py[e]);const[T,B,P]=this.flowAt(b,f,l),A=Ne(y),D=T*A,k=B*A,S=b+D*h,C=f+k*h,O=this.sampleHeight(S,C);let L=y+P*h;L+=(O+this.pHome[e]-L)*Math.min(1,this.relax*h);const z=O+this.minClear,H=O+this.maxClear;L<z&&(L=z),L>H&&(L=H),this.px[e]=S,this.py[e]=L,this.pz[e]=C;const j=Math.hypot(D,k),G=Math.min(1,Math.max(0,(j-this.speedLo)/(this.speedHi-this.speedLo)));let V=G*G*(3-2*G);const Y=Math.min(1,Math.max(0,P/7));V=Math.max(V,Y),this.speedFrac[e]=V;const ie=xt(e),pe=this.densityFloor+(1-this.densityFloor)*V;let ee=1-Fe(pe-.1,pe+.02,ie);const he=S-t[0],ce=C-t[2],ge=Math.hypot(he,ce),se=this.nearRadius*1.6;if(ge<se){const U=1-Fe(this.nearRadius,se,ge);ee=Math.max(ee,.85*U)}this.age[e]+=h;const q=Fe(0,this.fadeInTime,this.age[e]),$=S-n[0],te=C-n[1],ye=$*r[0]+te*r[1],K=$*s[0]+te*s[1],_e=this.spanAhead-ye,be=ye+this.spanBehind,Ee=this.spanWide-Math.abs(K),N=Math.min(_e,be,Ee),X=Fe(0,this.fadeFarEdge,N),Me=L-t[1],Z=this.bubbleFrac(he,Me,ce),Se=Fe(.35,1,Z);if(ee*=q*X*Se,ee<=.001){for(;a<c;)a=this.emitDegenerateFarVert(a);return a}const Ae=this.segStep;g[0]=S,w[0]=L,v[0]=C;let ve=S,ae=L,xe=C,we=D,re=k,le=P;for(let U=1;U<=p;U++){if(U>1&&(U&1)===1){const W=this.flowAt(ve,xe,l);we=W[0]*A,re=W[1]*A,le=W[2]}ve-=we*Ae,xe-=re*Ae,ae-=le*Ae;const F=this.sampleHeight(ve,xe)+this.minClear;ae<F&&(ae=F),g[U]=ve,w[U]=ae,v[U]=xe}const de=Math.max(1,Math.min(this.FAR_VERTS_PER_MOTE/6|0,Math.round(this.dashCountK))),Ce=1-this.lenByAltitude*(1-(.4+.6*V)),x=.5*this.dashLenM*Ce,I=Math.min(.85,.85/(1+.25*(this.gapRatio-1)));for(let U=0;U<de;U++){const W=(de>1?I*(U/(de-1)):0)*p,J=Math.min(p-1,Math.floor(W)),Q=W-J,ne=J+1,fe=g[J]*(1-Q)+g[ne]*Q,ue=w[J]*(1-Q)+w[ne]*Q,Te=v[J]*(1-Q)+v[ne]*Q;let ze=g[ne]-g[J],Re=v[ne]-v[J];const Pe=w[ne]-w[J],me=Math.hypot(ze,Pe,Re);let ke,Be,Oe;me>1e-5?(ke=ze/me,Be=Pe/me,Oe=Re/me):(ke=1,Be=0,Oe=0);const He=fe-ke*x,Ue=ue-Be*x,Ie=Te-Oe*x,Ge=fe+ke*x,Ke=ue+Be*x,qe=Te+Oe*x;let De=ke,Le=Oe;const Je=Math.hypot(De,Le);Je>1e-5?(De/=Je,Le/=Je):(De=1,Le=0);const nt=de>1?.8*(U/(de-1)):0,gt=Math.min(1,nt+.12),mt=U===0?Math.min(1,ee*this.leadBoost):ee;for(let wt=0;wt<6;wt++){const[ot,an]=d[wt],ln=ot>.5?Ge:He,hn=ot>.5?Ke:Ue,cn=ot>.5?qe:Ie,dn=ot>.5?gt:nt;o[a++]=ln,o[a++]=hn,o[a++]=cn,o[a++]=ot,o[a++]=an,o[a++]=V,o[a++]=De,o[a++]=Le,o[a++]=dn,o[a++]=mt,o[a++]=0}}for(;a<c;)a=this.emitDegenerateFarVert(a);return a}emitFarChevron(e,n,r,s,l,t,h,a){const c=a+this.FAR_VERTS_PER_MOTE*E.FPV,o=this.vertHost,d=E.CORNERS;let p=this.px[e],g=this.pz[e],w=this.py[e];const v=p-n[0],b=g-n[1],f=v*r[0]+b*r[1],y=v*s[0]+b*s[1];f<-this.spanBehind||f>this.spanAhead?(this.seedMote(e,n,r,s,this.spanAhead*.6,0),p=this.px[e],g=this.pz[e],w=this.py[e]):Math.abs(y)>this.spanWide&&(this.seedMote(e,n,r,s,-this.spanBehind,y>0?-1:1),p=this.px[e],g=this.pz[e],w=this.py[e]);const[m,M,_]=this.flowAt(p,g,l),R=Ne(w),T=m*R,B=M*R,P=p+T*h,A=g+B*h,D=this.sampleHeight(P,A);let k=w+_*h;k+=(D+this.pHome[e]-k)*Math.min(1,this.relax*h);const S=D+this.minClear,C=D+this.maxClear;k<S&&(k=S),k>C&&(k=C),this.px[e]=P,this.py[e]=k,this.pz[e]=A;const O=Math.hypot(T,B),L=Math.min(1,Math.max(0,(O-this.speedLo)/(this.speedHi-this.speedLo)));let z=L*L*(3-2*L);const H=Math.min(1,Math.max(0,_/7));z=Math.max(z,H),this.speedFrac[e]=z;const j=xt(e),G=this.densityFloor+(1-this.densityFloor)*z;let V=1-Fe(G-.1,G+.02,j);const Y=P-t[0],ie=A-t[2],pe=Math.hypot(Y,ie),ee=this.nearRadius*1.6;if(pe<ee){const x=1-Fe(this.nearRadius,ee,pe);V=Math.max(V,.85*x)}this.age[e]+=h;const he=Fe(0,this.fadeInTime,this.age[e]),ce=P-n[0],ge=A-n[1],se=ce*r[0]+ge*r[1],q=ce*s[0]+ge*s[1],$=this.spanAhead-se,te=se+this.spanBehind,ye=this.spanWide-Math.abs(q),K=Math.min($,te,ye),_e=Fe(0,this.fadeFarEdge,K),be=k-t[1],Ee=this.bubbleFrac(Y,be,ie),N=Fe(.35,1,Ee);if(V*=he*_e*N,V<=.001){for(;a<c;)a=this.emitDegenerateFarVert(a);return a}let X=T,Me=B;const Z=Math.hypot(X,Me);Z>1e-5?(X/=Z,Me/=Z):(X=1,Me=0);const Se=this.rakeBySpeed*z,Ae=this.spreadAngleDeg*(1-.5*Se)*Math.PI/180,ve=this.limbLenM*(1+.6*Se),ae=Math.cos(Ae),xe=Math.sin(Ae),we=P,re=k,le=A,de=-X,Ce=-Me;for(let x=-1;x<=1;x+=2){const I=x*xe,U=de*ae-Ce*I,F=de*I+Ce*ae,W=we+U*ve,J=le+F*ve,Q=re;let ne=U,fe=F;const ue=Math.hypot(ne,fe);ue>1e-5?(ne/=ue,fe/=ue):(ne=1,fe=0);const Te=0,ze=1,Re=Math.min(1,V*this.apexBoost);for(let Pe=0;Pe<6;Pe++){const[me,ke]=d[Pe],Be=me>.5?W:we,Oe=me>.5?Q:re,He=me>.5?J:le,Ue=me>.5?ze:Te,Ie=me>.5?V:Re;o[a++]=Be,o[a++]=Oe,o[a++]=He,o[a++]=me,o[a++]=ke,o[a++]=z,o[a++]=ne,o[a++]=fe,o[a++]=Ue,o[a++]=Ie,o[a++]=0}}for(;a<c;)a=this.emitDegenerateFarVert(a);return a}birdWakeAt(e,n,r,s,l,t,h,a,c){c[0]=0,c[1]=0,c[2]=0;const o=this.nearRadius,d=e-s[0],p=n-s[1],g=r-s[2],v=1-(Math.sqrt(d*d+p*p+g*g)||.001)/o;if(v<=0)return;const b=d*l+p*t+g*h;let f=d-b*l,y=p-b*t,m=g-b*h;const M=Math.sqrt(f*f+y*y+m*m)||.001;f/=M,y/=M,m/=M;const _=Math.min(1,Math.max(0,b)/(.35*o)),R=Math.min(1,Math.max(0,-b)/(.35*o)),T=this.bowGain*a*v*_,B=this.wakeGain*a*v*R;c[0]=f*T+l*B,c[1]=y*T+t*B,c[2]=m*T+h*B;const P=this.swirlGain*a*v*R;if(P>0){const A=this._rx,D=this._ry,k=this._rz,S=this.wingSpan,C=this.vortexCore;for(let O=-1;O<=1;O+=2){const L=d-O*S*A,z=p-O*S*D,H=g-O*S*k,j=L*l+z*t+H*h;let G=L-j*l,V=z-j*t,Y=H-j*h;const ie=Math.sqrt(G*G+V*V+Y*Y)||.001;G/=ie,V/=ie,Y/=ie;const pe=t*Y-h*V,ee=h*G-l*Y,he=l*V-t*G,ce=ie*C/(ie*ie+C*C),ge=P*ce*-O;c[0]+=pe*ge,c[1]+=ee*ge,c[2]+=he*ge}}}sampleWake(e,n,r){const s=[0,0,0];return this._moving&&this.birdWakeAt(e,n,r,this._lastBirdPos,this._ax,this._ay,this._az,this._bs,s),s}nearFrame(){return{pos:[this._lastBirdPos[0],this._lastBirdPos[1],this._lastBirdPos[2]],axis:[this._ax,this._ay,this._az],right:[this._rx,this._ry,this._rz],bs:this._bs,moving:this._moving}}bubbleFrac(e,n,r){const s=this.nearRadius,l=e*e+n*n+r*r;if(!this._moving)return Math.sqrt(l)/s;const t=e*this._ax+n*this._ay+r*this._az,h=Math.max(0,l-t*t),a=t>0?s*this.foreStretch:s;return Math.sqrt(t*t/(a*a)+h/(s*s))}setShowNear(e){this.showNear=e}setShowWake(e){this.showWake=e}setFarMode(e){this.farMode=e}setNearMode(e){this.nearMode=e}setWakeMode(e){this.wakeMode=e}stepNear(e,n,r,s){const l=n[0],t=n[1],h=n[2],a=Math.hypot(l,t,h),c=a>.5,o=c?l/a:0,d=c?t/a:0,p=c?h/a:0;let g=-p,w=o;const v=Math.hypot(g,w);if(v>.001?(g/=v,w/=v):(g=1,w=0),this._ax=o,this._ay=d,this._az=p,this._rx=g,this._ry=0,this._rz=w,this._bs=a,this._moving=c,this._wakeOn=c&&this.showWake,this._lastBirdPos[0]=e[0],this._lastBirdPos[1]=e[1],this._lastBirdPos[2]=e[2],!this.nearSeeded){for(let B=0;B<this.nearCount;B++)this.seedNearMote(B,e);this.nearSeeded=!0}let b=this.farVertexCount*E.FPV;this._wakeCountNow=Math.round(this.nearWakeCount*Math.min(1,a/this.wakeSpeedRef));const f=this.nearRadius,y=this.flowAt(e[0],e[2],r),m=this.flowAt(e[0]+f,e[2],r),M=this.flowAt(e[0]-f,e[2],r),_=this.flowAt(e[0],e[2]+f,r),R=this.flowAt(e[0],e[2]-f,r),T=(Math.hypot(y[0],y[1])+Math.hypot(m[0],m[1])+Math.hypot(M[0],M[1])+Math.hypot(_[0],_[1])+Math.hypot(R[0],R[1]))/5*Ne(e[1]);this._bodyCountNow=Math.round(this.nearBodyCount*Math.min(1,T/this.bodyWindRef));for(let B=0;B<this.nearCount;B++){const P=B<this._bodyCountNow,A=B>=this.nearBodyCount&&B<this.nearBodyCount+this._wakeCountNow,D=this._wakeOn&&(this.wakeMode==="modulate"||this.nearMode==="filaments");if(!(P||A&&D)){let k=b+this.NEAR_VERTS_PER_MOTE*E.FPV;for(;b<k;)b=this.emitDegenerateNearVert(b);continue}switch(this._curOp=A?this.wakeOpacity:this.nearOpacity,this.nearMode){case"flecks":b=this.emitNearFlecks(B,e,r,s,o,d,p,a,c,b);break;case"filaments":b=this.emitNearFilaments(B,e,r,s,o,d,p,a,c,b);break;case"comet":default:b=this.emitNearComet(B,e,r,s,o,d,p,a,c,b);break}}}emitNearComet(e,n,r,s,l,t,h,a,c,o){const d=this.speedLo,p=this.speedHi,g=this.vertHost,w=E.CORNERS,v=this.nearSegments,b=this.nptX,f=this.nptY,y=this.nptZ;let m=this.nx[e],M=this.ny[e],_=this.nz[e];const R=m-n[0],T=M-n[1],B=_-n[2];this.bubbleFrac(R,T,B)>1&&(this.seedNearMote(e,n),m=this.nx[e],M=this.ny[e],_=this.nz[e]);const[P,A,D]=this.flowAt(m,_,r),k=Ne(M),S=this.nearJit[e]*this.nearJitter,C=Math.cos(S),O=Math.sin(S),L=P*k,z=A*k,H=L*C-z*O,j=L*O+z*C;let G=0,V=0,Y=0;this._wakeOn&&(this.birdWakeAt(m,M,_,n,l,t,h,a,this._wake),G=this._wake[0],V=this._wake[1],Y=this._wake[2]);const ie=Math.sqrt(G*G+V*V+Y*Y),pe=Math.min(1,ie/this.heatRef),ee=Math.max(this.nearHeat[e]*Math.exp(-s/this.heatTau),pe);this.nearHeat[e]=ee;const he=m-n[0],ce=M-n[1],ge=_-n[2],se=Math.min(1,Math.sqrt(he*he+ce*ce+ge*ge)/this.nearRadius),q=e>=this.nearBodyCount,$=q?1:c?this.ambientNearFloor+(1-this.ambientNearFloor)*se:1,te=H*$+G,ye=D*$+V,K=j*$+Y,_e=m+te*s,be=_+K*s,Ee=this.sampleHeight(_e,be);let N=M+ye*s;const X=Ee+this.minClear;N<X&&(N=X),this.nx[e]=_e,this.ny[e]=N,this.nz[e]=be,this.nearAge[e]+=s;const Me=Fe(0,this.fadeInTime,this.nearAge[e]),Z=_e-n[0],Se=N-n[1],Ae=be-n[2],ve=this.bubbleFrac(Z,Se,Ae),ae=1-Fe(this.fadeNearEdge,1,ve),xe=Me*ae*this._curOp,we=Math.hypot(te,K),re=Math.min(1,Math.max(0,(we-d)/(p-d)));let le=re*re*(3-2*re);const de=Math.min(1,Math.max(0,D/7));le=Math.max(le,de);const Ce=this.nearSegStep*(q?this.wakeMoteLen:1)*(1+this.heatLenGain*ee);b[0]=_e,f[0]=N,y[0]=be;let x=_e,I=N,U=be,F=te,W=ye,J=K;for(let Q=1;Q<=v;Q++){x-=F*Ce,I-=W*Ce,U-=J*Ce;const ne=this.sampleHeight(x,U)+this.minClear;if(I<ne&&(I=ne),b[Q]=x,f[Q]=I,y[Q]=U,Q<v){const[fe,ue,Te]=this.flowAt(x,U,r),ze=fe*k*C-ue*k*O,Re=fe*k*O+ue*k*C;this._wakeOn?(this.birdWakeAt(x,I,U,n,l,t,h,a,this._wake),F=ze*$+this._wake[0],W=Te*$+this._wake[1],J=Re*$+this._wake[2]):(F=ze,W=Te,J=Re)}}for(let Q=0;Q<v;Q++){const ne=b[Q],fe=f[Q],ue=y[Q],Te=b[Q+1],ze=f[Q+1],Re=y[Q+1];let Pe=Te-ne,me=Re-ue;const ke=Math.hypot(Pe,me);ke>1e-5?(Pe/=ke,me/=ke):(Pe=1,me=0);const Be=Q/v,Oe=(Q+1)/v;for(let He=0;He<6;He++){const[Ue,Ie]=w[He],Ge=Ue>.5?Te:ne,Ke=Ue>.5?ze:fe,qe=Ue>.5?Re:ue,De=Ue>.5?Oe:Be;g[o++]=Ge,g[o++]=Ke,g[o++]=qe,g[o++]=Ue,g[o++]=Ie,g[o++]=le,g[o++]=Pe,g[o++]=me,g[o++]=De,g[o++]=xe,g[o++]=ee}}return o}get NEAR_VERTS_PER_MOTE(){return this.nearSegments*6}emitDegenerateNearVert(e){const n=this.vertHost;return n[e++]=0,n[e++]=0,n[e++]=0,n[e++]=0,n[e++]=0,n[e++]=0,n[e++]=0,n[e++]=0,n[e++]=0,n[e++]=0,n[e++]=0,e}emitNearFlecks(e,n,r,s,l,t,h,a,c,o){const d=o+this.NEAR_VERTS_PER_MOTE*E.FPV,p=this.speedLo,g=this.speedHi,w=this.vertHost,v=E.CORNERS;let b=this.nx[e],f=this.ny[e],y=this.nz[e];const m=b-n[0],M=f-n[1],_=y-n[2];this.bubbleFrac(m,M,_)>1&&(this.seedNearMote(e,n),b=this.nx[e],f=this.ny[e],y=this.nz[e]);const[R,T,B]=this.flowAt(b,y,r),P=Ne(f),A=R*P,D=T*P;let k=0,S=0,C=0;this._wakeOn&&(this.birdWakeAt(b,f,y,n,l,t,h,a,this._wake),k=this._wake[0],S=this._wake[1],C=this._wake[2]);const O=b-n[0],L=f-n[1],z=y-n[2],H=Math.min(1,Math.sqrt(O*O+L*L+z*z)/this.nearRadius),j=c?this.ambientNearFloor+(1-this.ambientNearFloor)*H:1,G=A*j+k,V=B*j+S,Y=D*j+C,ie=Math.sqrt(k*k+S*S+C*C),pe=Math.min(1,ie/this.heatRef),ee=Math.max(this.nearHeat[e]*Math.exp(-s/this.heatTau),pe);this.nearHeat[e]=ee;const he=b+G*s,ce=y+Y*s,ge=this.sampleHeight(he,ce);let se=f+V*s;const q=ge+this.minClear;se<q&&(se=q),this.nx[e]=he,this.ny[e]=se,this.nz[e]=ce,this.nearAge[e]+=s;const $=Fe(0,this.fadeInTime,this.nearAge[e]),te=he-n[0],ye=se-n[1],K=ce-n[2],_e=this.bubbleFrac(te,ye,K),be=1-Fe(this.fadeNearEdge,1,_e),Ee=$*be*this._curOp;let N=G,X=Y,Me=Math.hypot(N,X);Me>1e-5?(N/=Me,X/=Me):c?(N=l,X=h,Me=Math.hypot(N,X),Me>1e-5?(N/=Me,X/=Me):(N=1,X=0)):(N=1,X=0);const Z=this.shearRadius,Se=b+N*Z,Ae=y+X*Z,ve=b-N*Z,ae=y-X*Z,[xe,we]=this.flowAt(Se,Ae,r),re=Ne(f);let le=xe*re,de=we*re;const[Ce,x]=this.flowAt(ve,ae,r);let I=Ce*re,U=x*re;this._wakeOn&&(this.birdWakeAt(Se,f,Ae,n,l,t,h,a,this._wake),le=le*j+this._wake[0],de=de*j+this._wake[2],this.birdWakeAt(ve,f,ae,n,l,t,h,a,this._wake),I=I*j+this._wake[0],U=U*j+this._wake[2]);const F=Math.hypot(le-I,de-U)/(2*Z);this.fleckDirX[e]===0&&this.fleckDirZ[e]===0&&(this.fleckDirX[e]=N,this.fleckDirZ[e]=X);let W,J;if(this.orientLerp>0&&c){const De=Math.min(1,this.orientLerp);W=this.fleckDirX[e]*(1-De)+N*De,J=this.fleckDirZ[e]*(1-De)+X*De;const Le=Math.hypot(W,J);Le>1e-5?(W/=Le,J/=Le):(W=N,J=X)}else W=N,J=X;this.fleckDirX[e]=W,this.fleckDirZ[e]=J;const ne=.5*(this.fleckLen*(1+this.shearGain*F)),fe=he-W*ne,ue=ce-J*ne,Te=se,ze=he+W*ne,Re=ce+J*ne,Pe=se,me=Math.hypot(G,Y),ke=Math.min(1,Math.max(0,(me-p)/(g-p))),Be=ke*ke*(3-2*ke),Oe=Math.min(1,Math.max(0,(F-p)/(g-p))),He=Oe*Oe*(3-2*Oe),Ue=Math.max(Be*.4,He),Ie=0,Ge=Math.min(this.fleckTaper,.5),Ke=W,qe=J;for(let De=0;De<6;De++){const[Le,Je]=v[De],Qe=Le>.5?ze:fe,nt=Le>.5?Pe:Te,gt=Le>.5?Re:ue,mt=Le>.5?Ge:Ie;w[o++]=Qe,w[o++]=nt,w[o++]=gt,w[o++]=Le,w[o++]=Je,w[o++]=Ue,w[o++]=Ke,w[o++]=qe,w[o++]=mt,w[o++]=Ee,w[o++]=ee}for(;o<d;)o=this.emitDegenerateNearVert(o);return o}emitNearFilaments(e,n,r,s,l,t,h,a,c,o){const d=this.speedLo,p=this.speedHi,g=this.vertHost,w=E.CORNERS,v=this.nearSegments,b=this.nptX,f=this.nptY,y=this.nptZ;let m=this.nx[e],M=this.ny[e],_=this.nz[e];const R=m-n[0],T=M-n[1],B=_-n[2];this.bubbleFrac(R,T,B)>1&&(this.seedNearMote(e,n),m=this.nx[e],M=this.ny[e],_=this.nz[e]);const[P,A,D]=this.flowAt(m,_,r),k=Ne(M),S=P*k,C=A*k;let O=0,L=0,z=0;this._wakeOn&&(this.birdWakeAt(m,M,_,n,l,t,h,a,this._wake),O=this._wake[0],L=this._wake[1],z=this._wake[2]);const H=Math.sqrt(O*O+L*L+z*z),j=Math.min(1,H/this.heatRef),G=Math.max(this.nearHeat[e]*Math.exp(-s/this.heatTau),j);this.nearHeat[e]=G;const V=m-n[0],Y=M-n[1],ie=_-n[2],pe=Math.min(1,Math.sqrt(V*V+Y*Y+ie*ie)/this.nearRadius),ee=c?this.ambientNearFloor+(1-this.ambientNearFloor)*pe:1,he=S*ee+O,ce=D*ee+L,ge=C*ee+z,se=m+he*s,q=_+ge*s,$=this.sampleHeight(se,q);let te=M+ce*s;const ye=$+this.minClear;te<ye&&(te=ye),this.nx[e]=se,this.ny[e]=te,this.nz[e]=q,this.nearAge[e]+=s;const K=Fe(0,this.fadeInTime,this.nearAge[e]),_e=se-n[0],be=te-n[1],Ee=q-n[2],N=this.bubbleFrac(_e,be,Ee),X=1-Fe(this.fadeNearEdge,1,N),Me=K*X*this._curOp,Z=Math.hypot(he,ge),Se=Math.min(1,Math.max(0,(Z-d)/(p-d)));let Ae=Se*Se*(3-2*Se);const ve=Math.min(1,Math.max(0,D/7));Ae=Math.max(Ae,ve);const ae=this.filSegStep*(1+this.heatLenGain*G);b[0]=se,f[0]=te,y[0]=q;let xe=se,we=te,re=q,le=he,de=ce,Ce=ge;for(let x=1;x<=v;x++){xe-=le*ae,we-=de*ae,re-=Ce*ae;const I=this.sampleHeight(xe,re)+this.minClear;if(we<I&&(we=I),b[x]=xe,f[x]=we,y[x]=re,x<v){const[U,F,W]=this.flowAt(xe,re,r);this._wakeOn?(this.birdWakeAt(xe,we,re,n,l,t,h,a,this._wake),le=U*k*ee+this._wake[0],de=W*ee+this._wake[1],Ce=F*k*ee+this._wake[2]):(le=U*k,de=W,Ce=F*k)}}for(let x=0;x<v;x++){const I=b[x],U=f[x],F=y[x],W=b[x+1],J=f[x+1],Q=y[x+1];let ne=W-I,fe=Q-F;const ue=Math.hypot(ne,fe);ue>1e-5?(ne/=ue,fe/=ue):(ne=1,fe=0);const Te=x/v,ze=(x+1)/v;for(let Re=0;Re<6;Re++){const[Pe,me]=w[Re],ke=Pe>.5?W:I,Be=Pe>.5?J:U,Oe=Pe>.5?Q:F,He=Pe>.5?ze:Te;g[o++]=ke,g[o++]=Be,g[o++]=Oe,g[o++]=Pe,g[o++]=me,g[o++]=Ae,g[o++]=ne,g[o++]=fe,g[o++]=He,g[o++]=Me,g[o++]=G}}return o}stepWakeShed(e,n,r,s,l,t,h,a,c,o,d){this._ax=n,this._ay=r,this._az=s,this._rx=l,this._ry=t,this._rz=h,this._bs=a,this._moving=c,this._wakeOn=c&&this.showWake,this._lastBirdPos[0]=e[0],this._lastBirdPos[1]=e[1],this._lastBirdPos[2]=e[2];const p=(this.farVertexCount+this.nearVertexCount)*E.FPV;this.wakeMode==="rings"?this.wakeShedLiveCount=this.stepShedRings(e,n,r,s,l,t,h,a,c,o,d,p):this.wakeShedLiveCount=this.stepShedHelix(e,n,r,s,l,t,h,a,c,o,d,p)}stepShedHelix(e,n,r,s,l,t,h,a,c,o,d,p){const g=E.HELIX_TIPS*E.HELIX_LIVE,w=r*h-s*t,v=s*l-n*h,b=n*t-r*l;let f=this.helixActive;for(let S=0;S<f;)this.helixAge[S]+=d,this.helixAge[S]>=this.wakeLife?(f--,this.helixSeedX[S]=this.helixSeedX[f],this.helixSeedY[S]=this.helixSeedY[f],this.helixSeedZ[S]=this.helixSeedZ[f],this.helixAge[S]=this.helixAge[f],this.helixSide[S]=this.helixSide[f]):S++;if(this.helixActive=f,c){this.helixEmitAcc+=this.wakeEmitRate*E.HELIX_TIPS*d;let S=Math.floor(this.helixEmitAcc);for(this.helixEmitAcc-=S;S>0&&this.helixActive<g;){const C=this.helixActive&1?-1:1,O=C*this.wingSpan,L=(Math.random()*2-1)*this.wingJitter,z=e[0]+l*O+w*L,H=e[1]+t*O+v*L,j=e[2]+h*O+b*L,G=this.helixActive++;this.helixSeedX[G]=z,this.helixSeedY[G]=H,this.helixSeedZ[G]=j,this.helixAge[G]=0,this.helixSide[G]=C,S--}}const y=this.swirlGain;this.swirlGain=y*this.helixGain;const m=this.vertHost,M=E.CORNERS,_=this.wakeSeg,R=this.wsPtX,T=this.wsPtY,B=this.wsPtZ,P=this._wsWake,A=p+E.WAKE_SHED_RESERVE*E.FPV;let D=p;const k=this.helixActive;for(let S=0;S<k;S++){let C=this.helixSeedX[S],O=this.helixSeedY[S],L=this.helixSeedZ[S];const z=this.helixSide[S],[H,j,G]=this.flowAt(C,L,o),V=Ne(O);let Y=H*V,ie=G,pe=j*V;this.sampleHelixWake(C,O,L,e,n,r,s,a,z,P),Y+=P[0],ie+=P[1],pe+=P[2],C+=Y*d,O+=ie*d,L+=pe*d;const ee=this.sampleHeight(C,L)+this.minClear;O<ee&&(O=ee),this.helixSeedX[S]=C,this.helixSeedY[S]=O,this.helixSeedZ[S]=L;const he=this.helixAge[S]/this.wakeLife,ce=1-Fe(.6,1,he),ge=Fe(0,this.fadeInTime,this.helixAge[S]),se=ce*ge;if(se<=.001)continue;R[0]=C,T[0]=O,B[0]=L;let q=C,$=O,te=L,ye=Y,K=ie,_e=pe,be=Math.sqrt(P[0]*P[0]+P[1]*P[1]+P[2]*P[2]);for(let Z=1;Z<=_;Z++){q-=ye*this.wakeSegStep,$-=K*this.wakeSegStep,te-=_e*this.wakeSegStep;const Se=this.sampleHeight(q,te)+this.minClear;if($<Se&&($=Se),R[Z]=q,T[Z]=$,B[Z]=te,Z<_){const[Ae,ve,ae]=this.flowAt(q,te,o),xe=Ne($);this.sampleHelixWake(q,$,te,e,n,r,s,a,z,P);const we=Math.sqrt(P[0]*P[0]+P[1]*P[1]+P[2]*P[2]);we>be&&(be=we),ye=Ae*xe+P[0],K=ae+P[1],_e=ve*xe+P[2]}}const Ee=Math.min(1,be/this.heatRef),N=Math.hypot(Y,pe),X=Math.min(1,Math.max(0,(N-this.speedLo)/(this.speedHi-this.speedLo))),Me=X*X*(3-2*X);if(D+_*6*E.FPV>A){this.wakeOverrunLogged||(console.warn("[wind] wake-shed helix hit reserve cap; halting emit this frame"),this.wakeOverrunLogged=!0);break}for(let Z=0;Z<_;Z++){const Se=R[Z],Ae=T[Z],ve=B[Z],ae=R[Z+1],xe=T[Z+1],we=B[Z+1];let re=ae-Se,le=we-ve;const de=Math.hypot(re,le);de>1e-5?(re/=de,le/=de):(re=1,le=0);const Ce=this.wakeTaper*(Z/_),x=this.wakeTaper*((Z+1)/_);for(let I=0;I<6;I++){const[U,F]=M[I],W=U>.5?ae:Se,J=U>.5?xe:Ae,Q=U>.5?we:ve,ne=U>.5?x:Ce;m[D++]=W,m[D++]=J,m[D++]=Q,m[D++]=U,m[D++]=F,m[D++]=Me,m[D++]=re,m[D++]=le,m[D++]=ne,m[D++]=se,m[D++]=Ee}}}return this.swirlGain=y,(D-p)/E.FPV}sampleHelixWake(e,n,r,s,l,t,h,a,c,o){if(!this._wakeOn){o[0]=0,o[1]=0,o[2]=0;return}if(this.counterRotate||c>=0){this.birdWakeAt(e,n,r,s,l,t,h,a,o);return}const d=e-s[0],p=n-s[1],g=r-s[2],w=d*this._rx+p*this._ry+g*this._rz,v=e-2*w*this._rx,b=n-2*w*this._ry,f=r-2*w*this._rz;this.birdWakeAt(v,b,f,s,l,t,h,a,o);const y=o[0]*this._rx+o[1]*this._ry+o[2]*this._rz;o[0]-=2*y*this._rx,o[1]-=2*y*this._ry,o[2]-=2*y*this._rz}stepShedRings(e,n,r,s,l,t,h,a,c,o,d,p){const g=r*h-s*t,w=s*l-n*h,v=n*t-r*l;let b=this.ringActive;for(let A=0;A<b;)this.ringAge[A]+=d,this.ringAge[A]>=this.ringLife?(b--,this.ringCx[A]=this.ringCx[b],this.ringCy[A]=this.ringCy[b],this.ringCz[A]=this.ringCz[b],this.ringRadius[A]=this.ringRadius[b],this.ringAge[A]=this.ringAge[b],this.ringSide[A]=this.ringSide[b],this.ringHeat[A]=this.ringHeat[b]):A++;this.ringActive=b;for(let A=0;A<this.ringActive;A++){this.ringRadius[A]+=this.ringGrow*d;const D=this.ringCx[A],k=this.ringCy[A],S=this.ringCz[A],[C,O,L]=this.flowAt(D,S,o),z=Ne(k),H=this.convectFrac*a;let j=D+(C*z-n*H)*d,G=k+(L-r*H)*d,V=S+(O*z-s*H)*d;const Y=this.sampleHeight(j,V)+this.minClear;G<Y&&(G=Y),this.ringCx[A]=j,this.ringCy[A]=G,this.ringCz[A]=V}if(c){this.ringEmitAcc+=this.ringRate*d;let A=Math.floor(this.ringEmitAcc);for(this.ringEmitAcc-=A;A>0;){const D=this.twinOffset>0?[1,-1]:[0];for(const k of D){if(this.ringActive>=E.RING_COUNT)break;const S=k*this.wingSpan,C=e[0]+l*S,O=e[1]+t*S,L=e[2]+h*S;let z=0;this._wakeOn&&(this.birdWakeAt(C,O,L,e,n,r,s,a,this._wsWake),z=Math.min(1,Math.sqrt(this._wsWake[0]**2+this._wsWake[1]**2+this._wsWake[2]**2)/this.heatRef));const H=this.ringActive++;this.ringCx[H]=C,this.ringCy[H]=O,this.ringCz[H]=L,this.ringRadius[H]=this.ringStartRadius,this.ringAge[H]=0,this.ringSide[H]=k,this.ringHeat[H]=z}A--}}const f=this.vertHost,y=E.CORNERS,m=this.ringSegN,M=p+E.WAKE_SHED_RESERVE*E.FPV;let _=p,R=g+this.ringTilt*n,T=w+this.ringTilt*r,B=v+this.ringTilt*s;const P=Math.hypot(R,T,B);P>1e-5?(R/=P,T/=P,B/=P):(R=g,T=w,B=v);for(let A=0;A<this.ringActive;A++){if(_+m*6*E.FPV>M){this.wakeOverrunLogged||(console.warn("[wind] wake-shed rings hit reserve cap; halting emit this frame"),this.wakeOverrunLogged=!0);break}const D=this.ringCx[A],k=this.ringCy[A],S=this.ringCz[A],C=this.ringRadius[A],O=this.ringAge[A]/this.ringLife,L=1-Fe(.5,1,O),z=Fe(0,this.fadeInTime,this.ringAge[A]),H=L*z,j=this.ringHeat[A],G=this.ringSide[A],[V,Y]=this.flowAt(D,S,o),ie=Ne(k),pe=Math.hypot(V*ie,Y*ie),ee=Math.min(1,Math.max(0,(pe-this.speedLo)/(this.speedHi-this.speedLo))),he=ee*ee*(3-2*ee);let ce=D+l*C,ge=k+t*C,se=S+h*C;for(let q=0;q<m;q++){const $=2*Math.PI*(q+1)/m,te=Math.cos($),ye=Math.sin($),K=D+(l*te+R*ye)*C,_e=k+(t*te+T*ye)*C,be=S+(h*te+B*ye)*C;let Ee=K-ce,N=be-se;const X=Math.hypot(Ee,N);X>1e-5?(Ee/=X,N/=X):(Ee=1,N=0);const Me=l*Math.cos(2*Math.PI*(q+.5)/m)+R*Math.sin(2*Math.PI*(q+.5)/m),Z=Math.min(1,j+this.ringWarmBias*Math.max(0,-G*Me)),Se=.15,Ae=.15;for(let ve=0;ve<6;ve++){const[ae,xe]=y[ve],we=ae>.5?K:ce,re=ae>.5?_e:ge,le=ae>.5?be:se,de=ae>.5?Ae:Se;f[_++]=we,f[_++]=re,f[_++]=le,f[_++]=ae,f[_++]=xe,f[_++]=he,f[_++]=Ee,f[_++]=N,f[_++]=de,f[_++]=H,f[_++]=Z}ce=K,ge=_e,se=be}}return(_-p)/E.FPV}draw(e,n,r,s,l,t,h,a,c,o,d,p,g,w){let v=this.lastTime<0?0:c-this.lastTime;if(v<0&&(v=0),v>.05&&(v=.05),this.step(l,t,h,c,g),this.showNear&&this.stepNear(g,w,c,v),this.showWake&&this.wakeMode!=="modulate"){const M=w[0],_=w[1],R=w[2],T=Math.hypot(M,_,R),B=T>.5,P=B?M/T:0,A=B?_/T:0,D=B?R/T:0;let k=-D,S=P;const C=Math.hypot(k,S);C>.001?(k/=C,S/=C):(k=1,S=0),this.stepWakeShed(g,P,A,D,k,0,S,T,B,c,v)}else this.wakeShedLiveCount=0;this.device.queue.writeBuffer(this.vbuf,0,this.vertBytes);const f=this.dotPx/1e3,y=this.uniformF32;y.set(s,0),y[16]=a[0],y[17]=a[1],y[18]=a[2],y[19]=p,y[20]=o[0],y[21]=o[1],y[22]=o[2],y[23]=d,y[24]=f,this.device.queue.writeBuffer(this.ubuf,0,this.uniformHost);const m=e.beginRenderPass({colorAttachments:[{view:n,loadOp:"load",storeOp:"store"}],depthStencilAttachment:{view:r,depthLoadOp:"load",depthStoreOp:"store"}});m.setPipeline(this.pipeline),m.setBindGroup(0,this.bindGroup),m.setVertexBuffer(0,this.vbuf),m.draw(this.farVertexCount,1,0),this.showNear&&m.draw(this.nearVertexCount,1,this.farVertexCount),this.showWake&&this.wakeMode!=="modulate"&&this.wakeShedLiveCount>0&&m.draw(this.wakeShedLiveCount,1,this.farVertexCount+this.nearVertexCount),m.end()}};i(E,"FPV",11),i(E,"W_CLAMP",12),i(E,"FAR_SUBDIV",2),i(E,"CORNERS",[[0,-1],[1,-1],[1,1],[0,-1],[1,1],[0,1]]),i(E,"QUAD_VERTS",6),i(E,"HELIX_TIPS",2),i(E,"HELIX_LIVE",160),i(E,"HELIX_SEGS",4),i(E,"HELIX_VERTS",E.HELIX_TIPS*E.HELIX_LIVE*E.HELIX_SEGS*E.QUAD_VERTS),i(E,"RING_COUNT",32),i(E,"RING_CHORDS",32),i(E,"RINGS_VERTS",E.RING_COUNT*E.RING_CHORDS*E.QUAD_VERTS),i(E,"WAKE_SHED_RESERVE",Math.max(E.HELIX_VERTS,E.RINGS_VERTS));let Ct=E;function ut(u,e,n,r,s){const l=ti(),[t,h]=pt(u,e,n),a=t*s.windGain*l,c=h*s.windGain*l,o=s.ridgeEps,d=Math.hypot(t,h),p=d>1e-4?1/d:0,g=u+t*p*s.ridgeLookahead,w=e+h*p*s.ridgeLookahead,v=(r.sampleHeight(u+o,e)-r.sampleHeight(u-o,e))/(2*o),b=(r.sampleHeight(u,e+o)-r.sampleHeight(u,e-o))/(2*o),f=(r.sampleHeight(g+o,w)-r.sampleHeight(g-o,w))/(2*o),y=(r.sampleHeight(g,w+o)-r.sampleHeight(g,w-o))/(2*o),m=Math.max(0,a*v+c*b),M=Math.max(0,a*f+c*y),_=Math.max(m,M)*s.liftGain,R=Qn(u,e,n)*1.8;return Math.min(8,_+R)}const Bt=6,zt=112;class ri{constructor(e,n,r,s,l=[0,200,0],t={},h=1){i(this,"pos");i(this,"vel",[0,0,18]);i(this,"speed",26);i(this,"heading",0);i(this,"pitch",0);i(this,"bank",0);i(this,"renderBank",0);i(this,"renderPitch",0);i(this,"buffetOffset",[0,0,0]);i(this,"stallYaw",0);i(this,"tumbleRoll",0);i(this,"tumblePitch",0);i(this,"tumbleRollVel",0);i(this,"tumblePitchVel",0);i(this,"stillAir",!1);i(this,"beatActive",!1);i(this,"beatPhase",0);i(this,"flapBeatPhase",0);i(this,"ampL",0);i(this,"ampR",0);i(this,"lastFlapping",!1);i(this,"crashT",0);i(this,"lastCrashing",!1);i(this,"time",0);i(this,"tuning");i(this,"lastWind",[0,0]);i(this,"lastSpeed",0);i(this,"lastClearance",0);i(this,"lastVario",0);i(this,"lastUpdraft",0);i(this,"lastGroundTrack",0);i(this,"vbuf");i(this,"vertexCount");i(this,"ubuf");i(this,"uniformHost");i(this,"uniformF32");i(this,"pipeline");i(this,"bindGroup");this.device=e,this.terrain=s,this.pos=l,this.tuning={glideSpeed:t.glideSpeed??26,minSpeed:t.minSpeed??13,maxSpeed:t.maxSpeed??120,dragK:t.dragK??.1,divePower:t.divePower??2.4,climbPower:t.climbPower??1,gravity:t.gravity??9,sinkRate:t.sinkRate??.8,windGain:t.windGain??1.6,windDrift:t.windDrift??1,liftGain:t.liftGain??3.5,ridgeLookahead:t.ridgeLookahead??50,ridgeEps:t.ridgeEps??14,deflect:t.deflect??.25,flexHz:t.flexHz??.6,flexAmp:t.flexAmp??.06,beatHz:t.beatHz??3,beatLift:t.beatLift??14,beatThrust:t.beatThrust??10,beatAmp:t.beatAmp??.9,flapAsym:t.flapAsym??.3,flapTurn:t.flapTurn??.6,crashSpeed:t.crashSpeed??16,crashBleed:t.crashBleed??.65,crashTime:t.crashTime??.5,minClearance:t.minClearance??6,buffetGain:t.buffetGain??.3,buffetWindRef:t.buffetWindRef??12,rockCapDeg:t.rockCapDeg??12};const a=oi();this.vertexCount=a.length/Bt;const c=new Float32Array(new ArrayBuffer(a.length*4));c.set(a),this.vbuf=e.createBuffer({size:c.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.vbuf,0,c),this.uniformHost=new ArrayBuffer(zt),this.uniformF32=new Float32Array(this.uniformHost),this.ubuf=e.createBuffer({size:zt,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const o=e.createShaderModule({code:n});this.pipeline=e.createRenderPipeline({layout:"auto",vertex:{module:o,entryPoint:"vs",buffers:[{arrayStride:Bt*4,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"}]}]},fragment:{module:o,entryPoint:"fs",targets:[{format:r,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less",format:"depth24plus"},multisample:{count:h}}),this.bindGroup=e.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.ubuf}}]})}forwardVec(){return[Math.sin(this.heading),0,Math.cos(this.heading)]}get simTime(){return this.time}integrate(e,n){this.time+=e;const r=Math.min(e,1/20),s=this.tuning,l=7,t=28,h=.13,a=s.minSpeed,c=Math.max(0,(a-this.speed)/a),o=c>0,d=this.crashT>0?.3:1;this.heading+=n.yawRate*r*(o?.35:1)*d;let p=Math.max(-1,Math.min(1,n.pitchTarget));if(o){const U=-.05-.18*c;p=Math.min(p,U),this.stallYaw=Math.sign(this.bank||1)*c}else this.stallYaw=0;this.pitch+=(p-this.pitch)*Math.min(1,r*(o?6:3.5));const g=-n.yawRate*.5;this.bank+=(g-this.bank)*Math.min(1,r*4);const w=1/Math.max(.5,s.beatHz);!this.beatActive&&n.flap&&(this.beatActive=!0,this.beatPhase=0);let v=0;this.beatActive&&(this.beatPhase+=r/w,v=Math.max(0,Math.sin(Math.PI*Math.min(1,this.beatPhase))),this.beatPhase>=1&&(n.flap?this.beatPhase=0:(this.beatActive=!1,this.beatPhase=0)));const b=Math.max(-1,Math.min(1,n.yawRate*s.flapAsym)),f=1+b,y=1-b,m=v*(f+y)*.5,M=v*(f-y)*.5,_=m*s.beatLift,R=m*s.beatThrust;this.heading+=M*s.flapTurn*r;const T=-M*.5;this.flapBeatPhase=this.beatActive?Math.PI*this.beatPhase:0,this.ampL=v*s.beatAmp*f,this.ampR=v*s.beatAmp*y,this.lastFlapping=this.beatActive;const B=this.forwardVec(),P=[B[0]*Math.cos(this.pitch),Math.sin(this.pitch),B[2]*Math.cos(this.pitch)],A=Math.sin(this.pitch),D=A<0?s.divePower:s.climbPower;this.speed+=(-s.gravity*A*D-s.dragK*Math.max(0,this.speed-s.glideSpeed)+R)*r,this.speed=Math.max(l,Math.min(s.maxSpeed,this.speed));const[k,S]=this.stillAir?[0,0]:pt(this.pos[0],this.pos[2],this.time),C=Ne(this.pos[1]),O=k*s.windGain*C,L=S*s.windGain*C,z=6,H=this.terrain.sampleHeight(this.pos[0]+z,this.pos[2]),j=this.terrain.sampleHeight(this.pos[0],this.pos[2]+z),G=this.terrain.sampleHeight(this.pos[0]-z,this.pos[2]),V=this.terrain.sampleHeight(this.pos[0],this.pos[2]-z),[Y,ie]=on(O,L,(H-G)/(2*z),(j-V)/(2*z),s.deflect);this.lastWind=[Y,ie];const pe=this.stillAir?0:ut(this.pos[0],this.pos[2],this.time,this.terrain,s);this.lastUpdraft=pe;const ee=Math.min(t,s.sinkRate*(s.glideSpeed/this.speed)**3)+(o?s.sinkRate*c*.6:0),he=P[0]*this.speed+Y*s.windDrift,ce=P[1]*this.speed+pe-ee+_,ge=P[2]*this.speed+ie*s.windDrift,se=1-Math.exp(-r/h);this.vel[0]+=(he-this.vel[0])*se,this.vel[1]+=(ce-this.vel[1])*se,this.vel[2]+=(ge-this.vel[2])*se;const q=this.time*2,$=this.pos[0]*.05+this.pos[2]*.05,te=Math.sin(q*3.1+$),ye=Math.sin(q*5.7-$*1.7+1.3),K=Math.sin(q*1.9+$*.6+2.1),_e=this.stillAir?0:.6*te+.4*ye,be=this.stillAir?0:.6*K+.4*te,Ee=Math.cos(this.heading),N=-Math.sin(this.heading),X=be*1.2,Me=this.vel[0]+Ee*X,Z=this.vel[1]+_e*1.5,Se=this.vel[2]+N*X;this.pos[0]+=Me*r,this.pos[1]+=Z*r,this.pos[2]+=Se*r;const Ae=Math.hypot(Y,ie),ve=Math.max(0,Math.min(1,Ae/this.tuning.buffetWindRef));if(this.stillAir)this.buffetOffset[0]=this.buffetOffset[1]=this.buffetOffset[2]=0;else{const F=1.2*this.tuning.buffetGain*ve,W=.5*te*F;this.buffetOffset[0]=Ee*W,this.buffetOffset[1]=.6*ye*F,this.buffetOffset[2]=N*W}const ae=this.terrain.sampleHeight(this.pos[0],this.pos[2]),xe=ae+s.minClearance,we=xe-this.pos[1];if(we>0){this.pos[1]=xe;const U=we/r;if(U>s.crashSpeed&&this.crashT<=0){this.speed=Math.max(l,this.speed*(1-s.crashBleed)),this.crashT=s.crashTime;const F=Math.min(2.2,U/s.crashSpeed),W=this.renderBank>=0?1:-1;this.tumbleRollVel=W*(7+7*F),this.tumblePitchVel=-(4+4*F)}this.vel[1]<0&&(this.vel[1]=0)}this.crashT>0&&(this.crashT-=r),this.lastCrashing=this.crashT>0,this.lastSpeed=this.speed,this.lastVario=Z,this.lastClearance=this.pos[1]-ae,this.lastGroundTrack=Math.atan2(Me,Se),this.tumbleRoll+=this.tumbleRollVel*r,this.tumblePitch+=this.tumblePitchVel*r,this.tumbleRollVel*=Math.exp(-r/.25),this.tumblePitchVel*=Math.exp(-r/.25),this.tumbleRoll*=Math.exp(-r/.5),this.tumblePitch*=Math.exp(-r/.5);const re=Y*Ee+ie*N,le=Math.max(-.18,Math.min(.18,re*.012));let de=this.stillAir?0:(.6*te-.4*K)*.12*(.4+this.tuning.buffetGain*ve);const Ce=this.tuning.rockCapDeg*Math.PI/180;de=Math.max(-Ce,Math.min(Ce,de));const x=this.stallYaw*.15;this.renderBank=this.bank+le+de+T+x+this.tumbleRoll,typeof window<"u"&&(window.__birdBank=this.renderBank);const I=Math.min(.3,Math.max(0,this.vel[1])*.045);this.renderPitch=this.pitch+I+this.tumblePitch}resetAltitude(e){this.pos[1]=e,this.speed=this.tuning.glideSpeed;const n=this.forwardVec();this.vel=[n[0]*this.speed,0,n[2]*this.speed]}draw(e,n,r,s){const l=this.uniformF32;l.set(s,0),l[16]=this.pos[0]+this.buffetOffset[0],l[17]=this.pos[1]+this.buffetOffset[1],l[18]=this.pos[2]+this.buffetOffset[2],l[19]=this.time*this.tuning.flexHz*Math.PI*2,l[20]=this.heading,l[21]=this.renderBank,l[22]=this.tuning.flexAmp,l[23]=this.flapBeatPhase,l[24]=this.ampL,l[25]=this.ampR,l[26]=this.renderPitch,this.device.queue.writeBuffer(this.ubuf,0,this.uniformHost);const t=e.beginRenderPass({colorAttachments:[{view:n,loadOp:"load",storeOp:"store"}],depthStencilAttachment:{view:r,depthLoadOp:"load",depthStoreOp:"store"}});t.setPipeline(this.pipeline),t.setBindGroup(0,this.bindGroup),t.setVertexBuffer(0,this.vbuf),t.draw(this.vertexCount),t.end()}}function oi(){const u=[],l=(_,R,T,B,P,A,D)=>{const k=(H,j,G)=>[H[0]+P[0]*j*G,H[1]+P[1]*j*G,H[2]+P[2]*j*G],S=k(_,T,-1),C=k(_,T,1),O=k(R,B,-1),L=k(R,B,1),z=(H,j,G)=>u.push(H[0],H[1],H[2],j,1,G);z(S,A,0),z(C,A,1),z(O,D,0),z(O,D,0),z(C,A,1),z(L,D,1)},t=(_,R,T,B)=>{u.push(_[0],_[1],_[2],0,0,B),u.push(R[0],R[1],R[2],0,0,B),u.push(T[0],T[1],T[2],0,0,B)},o=[-1.05,.55,0],d=[1.05,.55,0],p=[-.6,-.95,0],g=[.6,-.95,0],w=[0,0,5.5*.62],v=[0,.1,-5.5*.55],b=.5,f=.2,y=.12,m=.04;t(w,o,d,b),t(v,d,o,b),t(w,d,g,f),t(v,g,d,f),t(w,p,o,y),t(v,o,p,y),t(w,g,p,m),t(v,p,g,m);const M=6;for(const _ of[-1,1]){let R=[0,0,0],T=0,B=1.7;for(let P=1;P<=M;P++){const A=P/M,D=_*9*A,k=-3.2*Math.pow(A,1.5),S=4.5*Math.pow(A,1.35)+.6*Math.pow(A,3),C=[D,S,k],O=_*A,L=1.7*(1-.55*A);l(R,C,B,L,[0,0,1],T,O),R=C,T=O,B=L}}return u}const ai=.00142857,li=2,hi=.5,ci=4,di=600,fi=1.8,yt=5,ui=4,pi=.65;function ht(u,e){const n=Math.sin(u*127.1+e*311.7)*43758.5453;return n-Math.floor(n)}function gi(u,e){const n=Math.floor(u),r=Math.floor(e),s=u-n,l=e-r,t=ht(n,r),h=ht(n+1,r),a=ht(n,r+1),c=ht(n+1,r+1),o=s*s*(3-2*s),d=l*l*(3-2*l),p=t+(h-t)*o,g=a+(c-a)*o;return p+(g-p)*d}function Ot(u,e){let n=ai,r=1,s=0,l=0;for(let o=0;o<ci;o++){const d=gi(u*n,e*n),p=1-Math.abs(2*d-1);s+=r*p,l+=r,n*=li,r*=hi}const t=Math.pow(s/l,fi),h=t*yt,a=h-Math.floor(h),c=Math.floor(h)/yt+Math.pow(a,ui)/yt;return(t+(c-t)*pi)*di}class mi{constructor(e,n,r=3){i(this,"staging");i(this,"inflight");i(this,"slotOrigin");i(this,"i",0);i(this,"pending",null);i(this,"latestU",null);i(this,"latestV",null);i(this,"latestOrigin",[0,0]);this.device=e,this.bytes=n,this.size=r;const s=()=>e.createBuffer({size:n,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});this.staging=Array.from({length:r},()=>({u:s(),v:s()})),this.inflight=Array.from({length:r},()=>!1),this.slotOrigin=Array.from({length:r},()=>[0,0])}enqueue(e,n,r,s,l){const t=this.i;this.inflight[t]?this.pending=null:(e.copyBufferToBuffer(n,0,this.staging[t].u,0,this.bytes),e.copyBufferToBuffer(r,0,this.staging[t].v,0,this.bytes),this.inflight[t]=!0,this.slotOrigin[t]=[s,l],this.pending=t),this.i=(this.i+1)%this.size}afterSubmit(){const e=this.pending;if(e===null)return;this.pending=null;const n=this.staging[e];Promise.all([n.u.mapAsync(GPUMapMode.READ),n.v.mapAsync(GPUMapMode.READ)]).then(()=>{this.latestU=new Float32Array(n.u.getMappedRange().slice(0)),this.latestV=new Float32Array(n.v.getMappedRange().slice(0)),this.latestOrigin=this.slotOrigin[e],n.u.unmap(),n.v.unmap(),this.inflight[e]=!1}).catch(r=>{this.inflight[e]=!1,console.error(`PairedReadback slot ${e} map failed:`,r)})}read(){return!this.latestU||!this.latestV?null:{u:this.latestU,v:this.latestV,origin:this.latestOrigin}}destroy(){for(const e of this.staging)e.u.destroy(),e.v.destroy()}}class wi{constructor(e,n,r={}){i(this,"fluid");i(this,"ring");i(this,"gridW");i(this,"gridH");i(this,"iters");i(this,"worldSpanM");i(this,"cellM");i(this,"forceMag");i(this,"targetBand");i(this,"scaleMin");i(this,"scaleMax");i(this,"recenterFrac");i(this,"terrainGain");i(this,"terrainMax");i(this,"originX",0);i(this,"originZ",0);i(this,"initialized",!1);i(this,"readOriginX",0);i(this,"readOriginZ",0);i(this,"scale",.05);i(this,"t",0);i(this,"recenterFrame",0);i(this,"frame",0);i(this,"rawMean",0);i(this,"fxField");i(this,"fyField");i(this,"heights");i(this,"heightsScratch");i(this,"extraReady",!1);const s=r.grid??256;this.gridW=s,this.gridH=s,this.iters=r.iters??10,this.worldSpanM=r.worldSpanM??2600,this.cellM=this.worldSpanM/s,this.forceMag=r.forceMag??28,this.targetBand=r.targetBand??3,this.scaleMin=r.scaleMin??.02,this.scaleMax=r.scaleMax??50,this.recenterFrac=r.recenterFrac??.18,this.terrainGain=r.terrainGain??22,this.terrainMax=r.terrainMax??60,this.fluid=new mn(e,s,s,n),this.ring=new mi(e,this.fluid.bytes),this.fxField=new Float32Array(this.fluid.cells),this.fyField=new Float32Array(this.fluid.cells),this.heights=new Float32Array(this.fluid.cells),this.heightsScratch=new Float32Array(this.fluid.cells),Promise.all([fetch("/src/host/shaders/fluid/shift.wgsl").then(l=>l.text()),fetch("/src/host/shaders/fluid/force_field.wgsl").then(l=>l.text())]).then(([l,t])=>{this.fluid.initExtraPipelines(l,t),this.extraReady=!0,this.initialized&&this.fluid.setForceField(this.fxField,this.fyField)})}get cellMeters(){return this.cellM}get originXZ(){return[this.readOriginX,this.readOriginZ]}get currentScale(){return this.scale}step(e,n,r,s){this.initialized||(this.originX=r-this.worldSpanM*.5,this.originZ=s-this.worldSpanM*.5,this.initialized=!0,this.fullTerrainField());const l=Math.min(Math.max(n,.001),.05);this.t+=l;const t=this.gridW*.5,h=this.gridW*this.recenterFrac,a=(r-this.originX)/this.cellM,c=(s-this.originZ)/this.cellM;let o=0,d=0;a-t>h?o=Math.floor(a-t-h)+1:t-a>h&&(o=-(Math.floor(t-a-h)+1)),c-t>h?d=Math.floor(c-t-h)+1:t-c>h&&(d=-(Math.floor(t-c-h)+1)),(o!==0||d!==0)&&(this.originX+=o*this.cellM,this.originZ+=d*this.cellM,this.fluid.shift(e,-o,-d),this.recenterTerrainField(o,d),this.recenterFrame++);const p=this.gridW*.5,g=this.gridH*.5,w=this.gridW*.22,v=p+Math.cos(this.t*.13)*w,b=g+Math.sin(this.t*.11)*w,f=this.t*.37,y=this.forceMag*Math.cos(f),m=this.forceMag*Math.sin(f*1.3);this.fluid.setForce({fx:y,fy:m,dyeX:v,dyeY:b,dyeR:0,dyeAmt:0,forceR:this.gridW*.25}),this.fluid.step(e,l,this.iters),this.ring.enqueue(e,this.fluid.velocityX,this.fluid.velocityY,this.originX,this.originZ),this.frame++,globalThis.__fluidWindow={originX:this.originX,originZ:this.originZ,recenterFrame:this.recenterFrame,cellM:this.cellM,gridW:this.gridW,frame:this.frame,scale:this.scale,rawMean:this.rawMean}}afterSubmit(){this.ring.afterSubmit()}read(){const e=this.ring.read();if(!e)return null;this.readOriginX=e.origin[0],this.readOriginZ=e.origin[1];const n=this.meanMagnitude(e.u,e.v);if(this.rawMean=n,n>1e-4){const r=this.targetBand/n,s=Math.min(this.scaleMax,Math.max(this.scaleMin,r));this.scale+=(s-this.scale)*.1}return{u:e.u,v:e.v}}cellWorldX(e){return this.originX+(e-1)*this.cellM}cellWorldZ(e){return this.originZ+(e-1)*this.cellM}fillHeights(){const e=this.gridW+2;for(let n=0;n<this.gridH+2;n++){const r=this.cellWorldZ(n);for(let s=0;s<this.gridW+2;s++)this.heights[s+e*n]=Ot(this.cellWorldX(s),r)}}forcesFromHeights(){const e=this.gridW+2,n=1/(2*this.cellM),r=this.terrainGain,s=this.terrainMax,l=this.heights;for(let t=1;t<=this.gridH;t++)for(let h=1;h<=this.gridW;h++){const a=h+e*t,c=(l[a+1]-l[a-1])*n,o=(l[a+e]-l[a-e])*n;let d=-c*r,p=-o*r;const g=Math.hypot(d,p);if(g>s){const w=s/g;d*=w,p*=w}this.fxField[a]=d,this.fyField[a]=p}this.extraReady&&this.fluid.setForceField(this.fxField,this.fyField)}fullTerrainField(){this.fillHeights(),this.forcesFromHeights()}recenterTerrainField(e,n){const r=this.gridW+2,s=this.gridW+2,l=this.gridH+2,t=this.heights,h=this.heightsScratch;for(let a=0;a<l;a++){const c=a+n;for(let o=0;o<s;o++){const d=o+e;h[o+r*a]=d>=0&&d<s&&c>=0&&c<l?t[d+r*c]:NaN}}for(let a=0;a<l;a++){const c=this.cellWorldZ(a);for(let o=0;o<s;o++){const d=o+r*a;Number.isNaN(h[d])&&(h[d]=Ot(this.cellWorldX(o),c))}}this.heights=h,this.heightsScratch=t,this.forcesFromHeights()}meanMagnitude(e,n){const r=this.gridW+2;let s=0,l=0;const t=Math.max(1,Math.floor(this.gridW/48));for(let h=1;h<=this.gridH;h+=t)for(let a=1;a<=this.gridW;a+=t){const c=a+r*h;s+=Math.hypot(e[c],n[c]),l++}return l>0?s/l:0}destroy(){this.fluid.destroy(),this.ring.destroy()}}const Lt=6,Dt=96;class bi{constructor(e,n,r,s=1){i(this,"vbuf");i(this,"vertexCount");i(this,"ubuf");i(this,"uniformHost",new ArrayBuffer(Dt));i(this,"uniformF32",new Float32Array(this.uniformHost));i(this,"pipeline");i(this,"bindGroup");this.device=e;const l=[],t=(o,d,p)=>l.push(o[0],o[1],o[2],d,p,0);t([0,0,0],0,0),t([0,0,0],0,1);const h=[[1,0],[0,1],[-1,0],[0,-1]];for(let o=0;o<4;o++){const d=h[o],p=h[(o+1)%4];t([d[0],0,d[1]],1,0),t([p[0],0,p[1]],1,0)}this.vertexCount=l.length/Lt;const a=new Float32Array(l);this.vbuf=e.createBuffer({size:a.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.vbuf,0,a),this.ubuf=e.createBuffer({size:Dt,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const c=e.createShaderModule({code:n});this.pipeline=e.createRenderPipeline({layout:"auto",vertex:{module:c,entryPoint:"vs",buffers:[{arrayStride:Lt*4,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"}]}]},fragment:{module:c,entryPoint:"fs",targets:[{format:r,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"line-list"},depthStencil:{depthWriteEnabled:!1,depthCompare:"less",format:"depth24plus"},multisample:{count:s}}),this.bindGroup=e.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.ubuf}}]})}draw(e,n,r,s,l,t,h,a){const c=this.uniformF32;c.set(s,0),c[16]=l[0],c[17]=l[1],c[18]=l[2],c[19]=t,c[20]=h,c[21]=Math.max(0,l[1]-t),this.device.queue.writeBuffer(this.ubuf,0,this.uniformHost);const o=e.beginRenderPass({colorAttachments:[{view:n,resolveTarget:a,loadOp:"load",storeOp:"store"}],depthStencilAttachment:{view:r,depthLoadOp:"load",depthStoreOp:"store"}});o.setPipeline(this.pipeline),o.setBindGroup(0,this.bindGroup),o.setVertexBuffer(0,this.vbuf),o.draw(this.vertexCount),o.end()}}const Ut=3,Ht=112,vi=260,xi=14,It=700,yi=1e3,_i=.7,_t=[1.6,1,.35];class Mi{constructor(e,n,r,s,l=1){i(this,"x",0);i(this,"z",0);i(this,"groundY",0);i(this,"vbuf");i(this,"vertexCount");i(this,"ubuf");i(this,"uniformHost",new ArrayBuffer(Ht));i(this,"uniformF32",new Float32Array(this.uniformHost));i(this,"pipeline");i(this,"bindGroup");this.device=e,this.sampleHeight=s;const t=[[-1,0],[1,0],[-1,1],[-1,1],[1,0],[1,1]],h=[];for(const[o,d]of t)h.push(o,d,0);this.vertexCount=h.length/Ut;const a=new Float32Array(h);this.vbuf=e.createBuffer({size:a.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.vbuf,0,a),this.ubuf=e.createBuffer({size:Ht,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const c=e.createShaderModule({code:n});this.pipeline=e.createRenderPipeline({layout:"auto",vertex:{module:c,entryPoint:"vs",buffers:[{arrayStride:Ut*4,attributes:[{shaderLocation:0,offset:0,format:"float32x3"}]}]},fragment:{module:c,entryPoint:"fs",targets:[{format:r,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{depthWriteEnabled:!1,depthCompare:"always",format:"depth24plus"},multisample:{count:l}}),this.bindGroup=e.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.ubuf}}]}),this.respawn(0,0,0)}respawn(e,n,r){const s=It+Math.random()*(yi-It),l=r+(Math.random()-.5)*_i;this.x=e+Math.sin(l)*s,this.z=n+Math.cos(l)*s,this.groundY=this.sampleHeight(this.x,this.z)}distanceTo(e){return Math.hypot(this.x-e[0],this.z-e[2])}checkReached(e,n){return this.distanceTo(e)<n}draw(e,n,r,s,l,t){const h=this.x-l[0],a=this.z-l[2],c=Math.hypot(h,a)||1,o=a/c,d=-h/c,p=this.uniformF32;p.set(s,0),p[16]=this.x,p[17]=this.groundY,p[18]=this.z,p[19]=vi,p[20]=o,p[21]=0,p[22]=d,p[23]=xi,p[24]=_t[0],p[25]=_t[1],p[26]=_t[2],p[27]=t,this.device.queue.writeBuffer(this.ubuf,0,this.uniformHost);const g=e.beginRenderPass({colorAttachments:[{view:n,loadOp:"load",storeOp:"store"}],depthStencilAttachment:{view:r,depthLoadOp:"load",depthStoreOp:"store"}});g.setPipeline(this.pipeline),g.setBindGroup(0,this.bindGroup),g.setVertexBuffer(0,this.vbuf),g.draw(this.vertexCount),g.end()}}const ct=7,Nt=96,Xe=14,Si=1e3,st=14e3,Gt=600,Vt=.42,Ai=.58,Wt=.86,jt=1/160,Ri=.44,Pi=.54,Ei=9,Ci=12,ki=3.2,Fi=.04,Ti=1.8,Bi=[.9,2.1,.55],zi=[.35,.45,.22],Oi=[.3,1.9,1.15],Li=[.2,.42,.34],Di=.5/1400;function kt(u,e){let n=(Math.imul(u,668265261)^Math.imul(e,374761393))>>>0;return n=Math.imul(n^n>>>15,739982445)>>>0,n=Math.imul(n^n>>>12,695872825)>>>0,(n^n>>>15)>>>0}function dt(u,e){return kt(u,e)/4294967296}function Xt(u){let e=u>>>0;return()=>{e=e+1831565813>>>0;let n=e;return n=Math.imul(n^n>>>15,n|1),n^=n+Math.imul(n^n>>>7,n|61),((n^n>>>14)>>>0)/4294967296}}function Yt(u){return u*u*(3-2*u)}function Ui(u,e){const n=Math.floor(u),r=Math.floor(e),s=Yt(u-n),l=Yt(e-r),t=dt(n,r),h=dt(n+1,r),a=dt(n,r+1),c=dt(n+1,r+1);return(t*(1-s)+h*s)*(1-l)+(a*(1-s)+c*s)*l}function ft(u){const e=Math.hypot(u[0],u[1],u[2])||1;return[u[0]/e,u[1]/e,u[2]/e]}const Ye=320,Hi=.72,Ii=.45,Ni=58,Zt=6,Gi=[.8,2.4,1.4],Vi=[.4,.7,.5];class Wi{constructor(e,n,r,s,l,t=1){i(this,"enabled",!0);i(this,"treeCount",0);i(this,"landmarks",[]);i(this,"tuning",{maxTrees:9e3,coverLo:Ri,coverHi:Pi,sizeScale:1,radius:Si,glow:1,fogDensity:.5/1100,depthBias:3});i(this,"vbuf");i(this,"maxVerts");i(this,"vertexCount",0);i(this,"hostBytes");i(this,"host");i(this,"ubuf");i(this,"uniformHost",new ArrayBuffer(Nt));i(this,"uniformF32",new Float32Array(this.uniformHost));i(this,"pipeline");i(this,"bindGroup");i(this,"baseHost");i(this,"baseBuf");i(this,"groundBuf");i(this,"computePipeline");i(this,"computeBindGroup");i(this,"groundDirty",!1);i(this,"lastCellX",Number.NaN);i(this,"lastCellZ",Number.NaN);i(this,"rebuildSig","");i(this,"cbx",0);i(this,"cbz",0);i(this,"ctid",0);this.device=e,this.sampleHeight=l,this.maxVerts=st*14*2,this.hostBytes=new ArrayBuffer(this.maxVerts*ct*4),this.host=new Float32Array(this.hostBytes),this.vbuf=e.createBuffer({size:this.hostBytes.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),this.ubuf=e.createBuffer({size:Nt,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.baseHost=new Float32Array(st*2),this.baseBuf=e.createBuffer({size:st*2*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.groundBuf=e.createBuffer({size:st*4,usage:GPUBufferUsage.STORAGE});const h=e.createShaderModule({code:r});this.computePipeline=e.createComputePipeline({layout:"auto",compute:{module:h,entryPoint:"main"}}),this.computeBindGroup=e.createBindGroup({layout:this.computePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.baseBuf}},{binding:1,resource:{buffer:this.groundBuf}}]});const a=e.createShaderModule({code:n});this.pipeline=e.createRenderPipeline({layout:"auto",vertex:{module:a,entryPoint:"vs",buffers:[{arrayStride:ct*4,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32"},{shaderLocation:2,offset:16,format:"float32x3"}]}]},fragment:{module:a,entryPoint:"fs",targets:[{format:s,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"line-list"},depthStencil:{depthWriteEnabled:!1,depthCompare:"less",format:"depth24plus"},multisample:{count:t}}),this.bindGroup=e.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.ubuf}},{binding:1,resource:{buffer:this.groundBuf}}]})}seg(e,n,r,s,l){const t=this.host;let h=e.i;if(h+ct*2>t.length)return;const a=this.cbx,c=this.cbz,o=this.ctid,d=this.tuning.glow;t[h++]=a+n[0],t[h++]=c+n[2],t[h++]=n[1],t[h++]=o,t[h++]=s[0]*d,t[h++]=s[1]*d,t[h++]=s[2]*d,t[h++]=a+r[0],t[h++]=c+r[2],t[h++]=r[1],t[h++]=o,t[h++]=l[0]*d,t[h++]=l[1]*d,t[h++]=l[2]*d,e.i=h}conifer(e,n,r,s,l,t,h){this.seg(e,n,[n[0],n[1]+r,n[2]],t,[l[0]*.6,l[1]*.6,l[2]*.6]);const a=h()*Math.PI*2;for(let c=0;c<3;c++){const o=c/2,d=n[1]+r*(.25+.6*o),p=s*(1-.6*o),g=.8+.5*o,w=[l[0]*g,l[1]*g,l[2]*g];for(let v=0;v<2;v++){const b=a+c*1.3+v*Math.PI;this.seg(e,[n[0],d,n[2]],[n[0]+Math.cos(b)*p,d-.3*p,n[2]+Math.sin(b)*p],[l[0]*.7,l[1]*.7,l[2]*.7],w)}}}deciduous(e,n,r,s,l,t){const h=r*.5,a=[n[0],n[1]+h,n[2]];this.seg(e,n,a,l,[s[0]*.5,s[1]*.5,s[2]*.5]);const c=(r-h)*.9,o=t()*Math.PI*2;for(let d=0;d<4;d++){const p=o+d/4*Math.PI*2+(t()-.5)*.5,g=.6+.2*t();this.seg(e,a,[a[0]+Math.cos(p)*c*g,a[1]+c*.85,a[2]+Math.sin(p)*c*g],[s[0]*.55,s[1]*.55,s[2]*.55],s)}}branch(e,n,r,s,l,t,h){const a=[n[0]+r[0]*s,n[1]+r[1]*s,n[2]+r[2]*s],c=.45+.7*(1-l/Zt),o=[t[0]*c,t[1]*c,t[2]*c];if(this.seg(e,n,a,[t[0]*.35,t[1]*.35,t[2]*.35],o),l<=0)return;const d=Math.abs(r[1])<.99?[0,1,0]:[1,0,0],p=ft([r[1]*d[2]-r[2]*d[1],r[2]*d[0]-r[0]*d[2],r[0]*d[1]-r[1]*d[0]]),g=ft([r[1]*p[2]-r[2]*p[1],r[2]*p[0]-r[0]*p[2],r[0]*p[1]-r[1]*p[0]]),w=2+(h()<.4?1:0);for(let v=0;v<w;v++){const b=.5*(.7+.6*h()),f=h()*Math.PI*2,y=Math.sin(b),m=Math.cos(b),M=ft([r[0]*m+(p[0]*Math.cos(f)+g[0]*Math.sin(f))*y,r[1]*m+(p[1]*Math.cos(f)+g[1]*Math.sin(f))*y+.15,r[2]*m+(p[2]*Math.cos(f)+g[2]*Math.sin(f))*y]);this.branch(e,a,M,s*.74*(.82+.3*h()),l-1,t,h)}}landmark(e,n,r,s,l,t){const a=t()*Math.PI*2,c=ft([Math.cos(a)*.07,1,Math.sin(a)*.07]),o=r*.38,d=[n[0]+c[0]*o,n[1]+c[1]*o,n[2]+c[2]*o];this.seg(e,n,d,l,[s[0]*.3,s[1]*.3,s[2]*.3]),this.branch(e,d,c,r*.4,Zt,s,t)}rebuild(e,n){const r={i:0};let s=0;this.landmarks.length=0;const l=Math.min(this.tuning.maxTrees|0,st),t=this.tuning.radius,h=this.tuning.sizeScale,a=this.tuning.coverLo,c=this.tuning.coverHi,o=Math.floor(t/Ye),d=Math.round(e/Ye),p=Math.round(n/Ye);for(let b=-o;b<=o;b++)for(let f=-o;f<=o;f++){const y=Xt(kt((d+f)*131+7,(p+b)*131+7));if(y()>Ii)continue;let m=(d+f)*Ye,M=(p+b)*Ye,_=-1;for(let R=0;R<5;R++){const T=(d+f)*Ye+(y()-.5)*Ye,B=(p+b)*Ye+(y()-.5)*Ye,P=this.sampleHeight(T,B);P>_&&(_=P,m=T,M=B)}if(!(Math.hypot(m-e,M-n)>t)&&!(_/Gt<Hi)){if(s>=l)break;this.cbx=m,this.cbz=M,this.ctid=s,this.baseHost[s*2]=m,this.baseHost[s*2+1]=M,this.landmark(r,[0,-4,0],Ni,Gi,Vi,y),this.landmarks.push([m,M]),s++}}const g=Math.floor(t/Xe),w=Math.round(e/Xe),v=Math.round(n/Xe);for(let b=-g;b<=g&&s<l;b++)for(let f=-g;f<=g&&s<l;f++){const y=w+f,m=v+b,M=Xt(kt(y,m)),_=y*Xe+(M()-.5)*Xe*.85,R=m*Xe+(M()-.5)*Xe*.85;if(Math.hypot(_-e,R-n)>t)continue;const T=(Ui(_*jt,R*jt)-a)/(c-a);if(M()>Math.max(0,Math.min(1,T)))continue;const P=this.sampleHeight(_,R)/Gt;if(P>Wt&&M()>Math.max(0,(1-P)/(1-Wt)))continue;const A=Math.max(0,Math.min(1,(P-Vt)/(Ai-Vt))),D=M()<A,S=(M()<Fi?Ti:1)*(.8+.45*M())*h;this.cbx=_,this.cbz=R,this.ctid=s,this.baseHost[s*2]=_,this.baseHost[s*2+1]=R;const C=[0,-4,0];D?this.conifer(r,C,Ci*S,ki*S,Oi,Li,M):this.deciduous(r,C,Ei*S,Bi,zi,M),s++}this.vertexCount=r.i/ct,this.treeCount=s,this.device.queue.writeBuffer(this.vbuf,0,this.hostBytes,0,r.i*4),this.device.queue.writeBuffer(this.baseBuf,0,this.baseHost.buffer,0,s*2*4),this.groundDirty=!0}draw(e,n,r,s,l,t,h,a=Di,c){if(!this.enabled)return;const o=Math.round(l[0]/Xe),d=Math.round(l[1]/Xe),p=this.tuning,g=`${Math.min(p.maxTrees|0,st)}|${p.coverLo}|${p.coverHi}|${p.sizeScale}|${p.radius}|${p.glow}`;if((o!==this.lastCellX||d!==this.lastCellZ||g!==this.rebuildSig)&&(this.rebuild(l[0],l[1]),this.lastCellX=o,this.lastCellZ=d,this.rebuildSig=g),this.vertexCount===0)return;if(this.groundDirty){const b=e.beginComputePass();b.setPipeline(this.computePipeline),b.setBindGroup(0,this.computeBindGroup),b.dispatchWorkgroups(Math.ceil(this.treeCount/64)),b.end(),this.groundDirty=!1}const w=this.uniformF32;w.set(s,0),w[16]=t[0],w[17]=t[1],w[18]=t[2],w[19]=a,w[20]=this.tuning.radius*.78,w[21]=this.tuning.radius*.98,w[22]=this.tuning.depthBias,w[23]=h,this.device.queue.writeBuffer(this.ubuf,0,this.uniformHost);const v=e.beginRenderPass({colorAttachments:[{view:n,loadOp:"load",storeOp:"store"}],depthStencilAttachment:{view:r,depthLoadOp:"load",depthStoreOp:"store"},timestampWrites:c});v.setPipeline(this.pipeline),v.setBindGroup(0,this.bindGroup),v.setVertexBuffer(0,this.vbuf),v.draw(this.vertexCount),v.end()}}function ji(u,e,n,r){const s=1/Math.tan(u/2),l=new Float32Array(16);return l[0]=s/e,l[5]=s,l[10]=r/(n-r),l[11]=-1,l[14]=n*r/(n-r),l}function Xi(u,e,n){const r=$t(Zi(u,e)),s=$t(qt(n,r)),l=qt(r,s),t=new Float32Array(16);return t[0]=s[0],t[1]=l[0],t[2]=r[0],t[3]=0,t[4]=s[1],t[5]=l[1],t[6]=r[1],t[7]=0,t[8]=s[2],t[9]=l[2],t[10]=r[2],t[11]=0,t[12]=-Mt(s,u),t[13]=-Mt(l,u),t[14]=-Mt(r,u),t[15]=1,t}function Yi(u,e){const n=new Float32Array(16);for(let r=0;r<4;r++)for(let s=0;s<4;s++)n[r*4+s]=u[0*4+s]*e[r*4+0]+u[1*4+s]*e[r*4+1]+u[2*4+s]*e[r*4+2]+u[3*4+s]*e[r*4+3];return n}function Zi(u,e){return[u[0]-e[0],u[1]-e[1],u[2]-e[2]]}function Mt(u,e){return u[0]*e[0]+u[1]*e[1]+u[2]*e[2]}function qt(u,e){return[u[1]*e[2]-u[2]*e[1],u[2]*e[0]-u[0]*e[2],u[0]*e[1]-u[1]*e[0]]}function qi(u){return Math.hypot(u[0],u[1],u[2])}function $t(u){const e=qi(u)||1;return[u[0]/e,u[1]/e,u[2]/e]}function Kt(u,e,n){return[u[0]+(e[0]-u[0])*n,u[1]+(e[1]-u[1])*n,u[2]+(e[2]-u[2])*n]}class $i{constructor(e={}){i(this,"target",[0,0,0]);i(this,"forward",[0,0,1]);i(this,"eye",[0,100,-120]);i(this,"lookTarget",[0,0,200]);i(this,"followDist");i(this,"followHeight");i(this,"lookAhead");i(this,"lookPitch");i(this,"smooth");i(this,"terrainHeight",null);i(this,"eyeMargin",10);this.followDist=e.followDist??60,this.followHeight=e.followHeight??28,this.lookAhead=e.lookAhead??160,this.lookPitch=e.lookPitch??16*Math.PI/180,this.smooth=e.smooth??.14}update(){const e=this.forward[0],n=this.forward[2],r=Math.hypot(e,n)||1,s=e/r,l=n/r,t=[this.target[0]-s*this.followDist,this.target[1]+this.followHeight,this.target[2]-l*this.followDist];if(this.terrainHeight){const c=this.eyeMargin,o=t[0]-this.target[0],d=t[1]-this.target[1],p=t[2]-this.target[2],g=12;let w=1;for(let b=1;b<=g;b++){const f=b/g,y=this.target[0]+o*f,m=this.target[1]+d*f,M=this.target[2]+p*f;if(m<this.terrainHeight(y,M)+c){w=(b-1)/g;break}}w<1&&(t[0]=this.target[0]+o*w,t[1]=this.target[1]+d*w,t[2]=this.target[2]+p*w);const v=this.terrainHeight(t[0],t[2])+c;t[1]<v&&(t[1]=v)}const h=this.lookAhead*Math.tan(this.lookPitch),a=[t[0]+s*this.lookAhead,t[1]-h,t[2]+l*this.lookAhead];if(this.eye=Kt(this.eye,t,this.smooth),this.lookTarget=Kt(this.lookTarget,a,this.smooth),this.terrainHeight){const c=this.terrainHeight(this.eye[0],this.eye[2])+this.eyeMargin;this.eye[1]<c&&(this.eye[1]=c)}}viewMatrix(){return Xi(this.eye,this.lookTarget,[0,1,0])}getEye(){return[this.eye[0],this.eye[1],this.eye[2]]}camOffset(){return[this.target[0],this.target[2]]}groundPos(){return[this.eye[0],this.eye[2]]}forwardHoriz(){const e=this.lookTarget[0]-this.eye[0],n=this.lookTarget[2]-this.eye[2],r=Math.hypot(e,n)||1;return[e/r,n/r]}rightHoriz(){const[e,n]=this.forwardHoriz();return[-n,e]}}const Ki=`
const BASE_FREQ : f32 = 0.00142857;
const LACUNARITY : f32 = 2.0;
const GAIN : f32 = 0.5;
const OCTAVES : i32 = 4;
const RELIEF : f32 = 600.0;
const SHARP : f32 = 1.8;
const TERRACES : f32 = 5.0;
const RISER_POW : f32 = 4.0;
const CLIFF_MIX : f32 = 0.65;

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
  let s = pow(sum / norm, SHARP);
  let b = s * TERRACES;
  let fb = b - floor(b);
  let ter = floor(b) / TERRACES + pow(fb, RISER_POW) / TERRACES;
  return (s + (ter - s) * CLIFF_MIX) * RELIEF;
}

@group(0) @binding(0) var<storage, read> pts : array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> outH : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&pts)) { return; }
  outH[i] = fbm(pts[i]);
}
`;async function Ji(u,e){const n=[];for(let m=-3e3;m<=3e3;m+=250)for(let M=-3e3;M<=3e3;M+=250)n.push([m,M]);for(const m of[4e3,-4e3,8e3,-8e3])n.push([m,m]),n.push([m,-m]);const r=n.length,s=new Float32Array(r*2);for(let m=0;m<r;m++){const M=n[m];s[m*2]=M[0],s[m*2+1]=M[1]}const l=u.createBuffer({size:s.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});u.queue.writeBuffer(l,0,s);const t=u.createBuffer({size:r*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),h=u.createBuffer({size:r*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),a=u.createShaderModule({code:Ki}),c=u.createComputePipeline({layout:"auto",compute:{module:a,entryPoint:"main"}}),o=u.createBindGroup({layout:c.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:l}},{binding:1,resource:{buffer:t}}]}),d=u.createCommandEncoder(),p=d.beginComputePass();p.setPipeline(c),p.setBindGroup(0,o),p.dispatchWorkgroups(Math.ceil(r/64)),p.end(),d.copyBufferToBuffer(t,0,h,0,r*4),u.queue.submit([d.finish()]),await h.mapAsync(GPUMapMode.READ);const g=new Float32Array(h.getMappedRange().slice(0));h.unmap();let w=0,v=0,b={x:0,z:0,cpu:0,gpu:0};for(let m=0;m<r;m++){const M=n[m],_=g[m],R=e(M[0],M[1]),T=Math.abs(R-_);v+=T,T>w&&(w=T,b={x:M[0],z:M[1],cpu:R,gpu:_})}l.destroy(),t.destroy(),h.destroy();const f=v/r,y=w<1?"PASS — GPU and CPU terrain agree (<1 m). If the bird still crashes in clear sky, the SCENE shader is STALE: hard-reload (Cmd+Shift+R) or restart the dev server.":`FAIL — GPU and CPU terrain DISAGREE by up to ${w.toFixed(0)} m. The integer hash is not matching across CPU/GPU; the hash needs to change (e.g. avoid f32(u32) precision loss).`;return{points:r,maxDiff:w,meanDiff:f,worst:b,verdict:y}}const Jt=140,St=2.5,Qi=90,Qt=260,es=45,ts=35;class ns{constructor(e,n="soar"){i(this,"mode","CRUISE");i(this,"pitchCmd",-.03);i(this,"yawCmd",0);i(this,"logT",0);i(this,"lockedHeading",null);this.terrain=e,this.policy=n}update(e,n){const r=e.tuning,[s,l,t]=e.pos,h=e.simTime,a=e.lastClearance,c=e.speed;this.lockedHeading===null&&(this.lockedHeading=e.heading);let o=e.heading,d=-1;if(this.policy==="soar")for(let _=0;_<8;_++){const R=_/8*Math.PI*2,T=ut(s+Math.sin(R)*Jt,t+Math.cos(R)*Jt,h,this.terrain,r);T>d&&(d=T,o=R)}const p=ut(s,t,h,this.terrain,r),g=r.sinkRate*(r.glideSpeed/c)**3,w=s+e.vel[0]*St,v=t+e.vel[2]*St,b=l+e.vel[1]*St-this.terrain.sampleHeight(w,v);let f,y;if(a<es||b<ts){this.mode="AVOID",f=c>r.minSpeed+5?.32:-.08;const _=this.terrain.sampleHeight(s+Math.sin(e.heading-.8)*160,t+Math.cos(e.heading-.8)*160),R=this.terrain.sampleHeight(s+Math.sin(e.heading+.8)*160,t+Math.cos(e.heading+.8)*160);y=e.heading+(_<R?-.8:.8),this.lockedHeading=y}else this.policy==="straight"?(this.mode=c<r.glideSpeed-5?"ENERGY":"STRAIGHT",f=c<r.glideSpeed-5?-.22:-.03,y=this.lockedHeading):p>g+.4?(this.mode="SOAR",f=a>Qt?-.12:.04,y=e.heading+.5):c<r.glideSpeed-5?(this.mode="ENERGY",f=-.22,y=o):a<Qi?(this.mode="CLIMB",f=.1,y=o):a>Qt?(this.mode="DESCEND",f=-.15,y=o):(this.mode="CRUISE",f=-.03,y=o);let m=y-e.heading;m=Math.atan2(Math.sin(m),Math.cos(m));const M=Math.max(-.9,Math.min(.9,m*1.4));return this.pitchCmd+=(f-this.pitchCmd)*Math.min(1,n*4),this.yawCmd+=(M-this.yawCmd)*Math.min(1,n*6),this.logT+=n,this.logT>2&&(this.logT=0,console.log(`[auto] ${this.mode} clr=${a.toFixed(0)}m spd=${c.toFixed(1)} vario=${e.lastVario.toFixed(1)} lift=${p.toFixed(1)} bestProbe=${d.toFixed(1)}`)),{yawRate:this.yawCmd,pitchTarget:this.pitchCmd,flap:!1}}}class is{constructor(e,n,r,s={}){i(this,"device");i(this,"swapFormat");i(this,"hdrFormat","rgba16float");i(this,"threshold");i(this,"knee");i(this,"intensity");i(this,"exposure");i(this,"downsample");i(this,"blurPasses");i(this,"sampler");i(this,"thresholdPipeline");i(this,"blurPipeline");i(this,"compositePipeline");i(this,"thresholdUbuf");i(this,"blurUbuf");i(this,"compositeUbuf");i(this,"bw",1);i(this,"bh",1);i(this,"brightTex");i(this,"pingTex");i(this,"pongTex");i(this,"brightView");i(this,"pingView");i(this,"pongView");this.device=e,this.swapFormat=n,this.threshold=s.threshold??.9,this.knee=s.knee??.4,this.intensity=s.intensity??1,this.exposure=s.exposure??1,this.downsample=Math.max(1,s.downsample??2),this.blurPasses=Math.max(1,s.blurPasses??2),this.sampler=e.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"});const l=e.createShaderModule({code:r.threshold}),t=e.createShaderModule({code:r.blur}),h=e.createShaderModule({code:r.composite});this.thresholdPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:l,entryPoint:"vs"},fragment:{module:l,entryPoint:"fs",targets:[{format:this.hdrFormat}]},primitive:{topology:"triangle-list"}}),this.blurPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:t,entryPoint:"vs"},fragment:{module:t,entryPoint:"fs",targets:[{format:this.hdrFormat}]},primitive:{topology:"triangle-list"}}),this.compositePipeline=e.createRenderPipeline({layout:"auto",vertex:{module:h,entryPoint:"vs"},fragment:{module:h,entryPoint:"fs",targets:[{format:this.swapFormat}]},primitive:{topology:"triangle-list"}}),this.thresholdUbuf=e.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.blurUbuf=e.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.compositeUbuf=e.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}resize(e,n){var s,l,t;this.bw=Math.max(1,Math.floor(e/this.downsample)),this.bh=Math.max(1,Math.floor(n/this.downsample));const r=()=>this.device.createTexture({size:[this.bw,this.bh],format:this.hdrFormat,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.RENDER_ATTACHMENT});(s=this.brightTex)==null||s.destroy(),(l=this.pingTex)==null||l.destroy(),(t=this.pongTex)==null||t.destroy(),this.brightTex=r(),this.pingTex=r(),this.pongTex=r(),this.brightView=this.brightTex.createView(),this.pingView=this.pingTex.createView(),this.pongView=this.pongTex.createView()}setTuning(e){e.threshold!==void 0&&(this.threshold=e.threshold),e.knee!==void 0&&(this.knee=e.knee),e.intensity!==void 0&&(this.intensity=e.intensity),e.exposure!==void 0&&(this.exposure=e.exposure)}apply(e,n,r){const s=this.device;s.queue.writeBuffer(this.thresholdUbuf,0,new Float32Array([this.threshold,this.knee,0,0]));const l=s.createBindGroup({layout:this.thresholdPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:n},{binding:1,resource:this.sampler},{binding:2,resource:{buffer:this.thresholdUbuf}}]});this.pass(e,this.thresholdPipeline,l,this.brightView);const t=1/this.bw,h=1/this.bh;let a=this.brightView;for(let o=0;o<this.blurPasses;o++){s.queue.writeBuffer(this.blurUbuf,0,new Float32Array([t,0,0,0]));const d=s.createBindGroup({layout:this.blurPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:a},{binding:1,resource:this.sampler},{binding:2,resource:{buffer:this.blurUbuf}}]});this.pass(e,this.blurPipeline,d,this.pingView),s.queue.writeBuffer(this.blurUbuf,0,new Float32Array([0,h,0,0]));const p=s.createBindGroup({layout:this.blurPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:this.pingView},{binding:1,resource:this.sampler},{binding:2,resource:{buffer:this.blurUbuf}}]});this.pass(e,this.blurPipeline,p,this.pongView),a=this.pongView}s.queue.writeBuffer(this.compositeUbuf,0,new Float32Array([this.intensity,this.exposure,0,0]));const c=s.createBindGroup({layout:this.compositePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:n},{binding:1,resource:a},{binding:2,resource:this.sampler},{binding:3,resource:{buffer:this.compositeUbuf}}]});this.pass(e,this.compositePipeline,c,r)}pass(e,n,r,s){const l=e.beginRenderPass({colorAttachments:[{view:s,loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});l.setPipeline(n),l.setBindGroup(0,r),l.draw(3),l.end()}}const ss=Object.assign({"./shaders/addone.wgsl":wn,"./shaders/bird/bird_update.wgsl":bn,"./shaders/bird/scene.wgsl":vn,"./shaders/bird3d.wgsl":xn,"./shaders/bloom_blur.wgsl":yn,"./shaders/bloom_composite.wgsl":_n,"./shaders/bloom_threshold.wgsl":Mn,"./shaders/fluid/add_force_field.wgsl":Sn,"./shaders/fluid/advect.wgsl":An,"./shaders/fluid/divergence.wgsl":Rn,"./shaders/fluid/force_field.wgsl":Pn,"./shaders/fluid/forces.wgsl":En,"./shaders/fluid/jacobi.wgsl":Cn,"./shaders/fluid/set_bnd.wgsl":kn,"./shaders/fluid/shift.wgsl":Fn,"./shaders/fluid/subtract_grad.wgsl":Tn,"./shaders/fluid/visualize.wgsl":Bn,"./shaders/marker.wgsl":zn,"./shaders/target.wgsl":On,"./shaders/terrain3d.wgsl":Ln,"./shaders/terrain_ekg.wgsl":Dn,"./shaders/terrain_grid.wgsl":Un,"./shaders/trees.wgsl":Hn,"./shaders/trees_ground.wgsl":In,"./shaders/wind.wgsl":Nn});function Ve(u){const e=u.replace("/src/host","."),n=ss[e];if(n===void 0)throw new Error(`shader not bundled: ${u} (looked for key ${e})`);return n}let et=!1,en=!0,tt="ekg";const tn=60*Math.PI/180,rs=16*Math.PI/180,os=1,as=12e3,We=4,je="rgba16float",rt={clearance:25,height:10,pitchDeg:8},At={clearance:160,height:55,pitchDeg:28},Ze=[.01,.012,.03],ls=1.8,hs=1,nn=-.03,Rt=.05,cs=55,ds=400;async function fs(){const u=document.getElementById("overlay"),e=document.getElementById("bird"),{device:n}=await pn();n.lost.then(x=>{u.textContent=`WebGPU device lost: ${x.reason} — ${x.message}`,console.error("[WebGPU lost]",x.reason,x.message)});const r=e.getContext("webgpu"),s=navigator.gpu.getPreferredCanvasFormat();r.configure({device:n,format:s,alphaMode:"opaque"});const l=Math.min(window.devicePixelRatio||1,2);let t=Math.floor(e.clientWidth*l)||900,h=Math.floor(e.clientHeight*l)||640;e.width=t,e.height=h;let a=n.createTexture({size:[t,h],format:"depth24plus",sampleCount:We,usage:GPUTextureUsage.RENDER_ATTACHMENT}),c=n.createTexture({size:[t,h],format:je,sampleCount:We,usage:GPUTextureUsage.RENDER_ATTACHMENT}),o=n.createTexture({size:[t,h],format:je,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.RENDER_ATTACHMENT});const d=Ve("/src/host/shaders/terrain_ekg.wgsl"),p=new $n(n,d,je,{rows:512,cols:1536,sampleCount:We,rowSpacing:2,nearDenseDepth:250,farSpread:100,rowStart:-150,halfWidth:2400,maxDist:2850,baseline:-300,fogColor:Ze,fogDensity:.25/2200}),g=Ve("/src/host/shaders/terrain_grid.wgsl"),w=new Kn(n,g,je,{spacing:26,radius:1650,maxDist:1500,fogColor:Ze,fogDensity:.5/1100,sampleCount:We}),v=Ve("/src/host/shaders/bird3d.wgsl"),b=p.sampleHeight(0,0),f=new ri(n,v,je,p,[0,b+ds,0],{},We);f.stillAir=!1;const y=Ve("/src/host/shaders/wind.wgsl"),m=new Ct(n,y,je,(x,I)=>p.sampleHeight(x,I),{},{nearCount:1500,numMotes:4e3,dotPx:3.6},We),_=Object.fromEntries(Object.entries({forces:"/src/host/shaders/fluid/forces.wgsl",divergence:"/src/host/shaders/fluid/divergence.wgsl",jacobi:"/src/host/shaders/fluid/jacobi.wgsl",subtractGrad:"/src/host/shaders/fluid/subtract_grad.wgsl",advect:"/src/host/shaders/fluid/advect.wgsl",setBnd:"/src/host/shaders/fluid/set_bnd.wgsl"}).map(([x,I])=>[x,Ve(I)])),R=new wi(n,_,{grid:256,iters:10}),T=Ve("/src/host/shaders/marker.wgsl"),B=new bi(n,T,je,We),P=Ve("/src/host/shaders/target.wgsl"),A=new Mi(n,P,je,(x,I)=>p.sampleHeight(x,I),We),D=Ve("/src/host/shaders/trees.wgsl"),k=Ve("/src/host/shaders/trees_ground.wgsl"),S=new Wi(n,D,k,je,(x,I)=>p.sampleHeight(x,I),We),C={threshold:Ve("/src/host/shaders/bloom_threshold.wgsl"),blur:Ve("/src/host/shaders/bloom_blur.wgsl"),composite:Ve("/src/host/shaders/bloom_composite.wgsl")},O=new is(n,s,C,{threshold:.85,knee:.5,intensity:.9,exposure:1,downsample:2,blurPasses:2});O.resize(t,h);const L=new ns(p,"straight"),z=new $i({followDist:120,followHeight:55,lookAhead:160,lookPitch:28*Math.PI/180,smooth:.14});z.terrainHeight=(x,I)=>p.sampleHeight(x,I);const H={yawRate:0,pitchTarget:nn,flap:!1};let j=0,G=0,V=!1;e.addEventListener("mousemove",x=>{const I=e.getBoundingClientRect();j=(x.clientX-I.left)/I.width*2-1,G=(x.clientY-I.top)/I.height*2-1,window.__autoWobble=!1}),e.addEventListener("mousedown",()=>{et=!1,window.__autoWobble=!1});const Y=()=>{j=0,G=0};e.addEventListener("mouseleave",Y),window.addEventListener("blur",Y);const ie=us(),pe=ps(ie),ee=pe.section("flight"),he=[["glideSpeed",14,40,.5,"Trim airspeed (m/s) that drag relaxes speed toward."],["sinkRate",.3,4,.1,"Base sink at trim speed (m/s); scales (trim/speed)² when slow."],["divePower",.2,3,.05,"Scale on gravity-along-path when diving (nose down)."],["climbPower",.3,2.5,.05,"Scale on gravity-along-path when climbing (nose up)."],["dragK",.1,1.5,.05,"Per-second relaxation of airspeed toward trim."],["liftGain",0,6,.1,"Ridge updraft scale (vertical air-motion m/s per unit wind·slope)."],["ridgeLookahead",0,150,5,"Metres downwind the ridge-lift gradient is sampled (bigger = lift off hills sooner)."],["ridgeEps",6,40,2,"Central-diff half-step (m) for the ridge-lift gradient (broaden to widen the lift band)."],["windGain",0,15,.5,"Analytic wind push scale — how hard wind pushes the bird (cross-track drift)."],["windDrift",0,2,.1,"Fraction of horizontal wind the bird drifts with."],["minSpeed",8,20,.5,"Stall floor (m/s)."],["maxSpeed",30,160,1,"Dive ceiling (m/s)."],["beatLift",0,30,1,"Peak vertical lift from a symmetric wingbeat (m/s into target vertical vel)."],["beatThrust",0,25,1,"Peak forward thrust during a beat (m/s²) — sustains airspeed in a climb."],["beatHz",1,6,.5,"Wingbeats per second while flap is held (sustained-climb cadence)."],["crashSpeed",5,40,1,"Closing speed into terrain (m/s) above which a touch counts as a crash."]];for(const[x,I,U,F,W]of he)oe(ee,f.tuning,x,I,U,F,W);const ce=pe.section("terrain render"),ge=document.createElement("button"),se=["ekg","grid","topo"];ge.textContent=`mode: ${tt}  ▸`,ge.style.cssText="width:100%;margin:0 0 6px;padding:4px;background:#241d40;color:#9fe8ff;border:1px solid #4a4070;border-radius:4px;font:12px monospace;cursor:pointer;",ge.title="Cycle the terrain render style: EKG rows / grid / topo contours.",ge.onclick=()=>{tt=se[(se.indexOf(tt)+1)%se.length],ge.textContent=`mode: ${tt}  ▸`},ce.appendChild(ge);const q=w;oe(ce,q,"interval",8,80,1,"Horizontal spacing (m) of the stacked depth rows in the dense near band."),oe(ce,q,"floorFade",0,1,.02,"Opacity fade for far rows (0..1)."),oe(ce,q,"peakGain",.5,3,.1,"Elevation colour scale for peaks (higher = brighter warm tint)."),oe(ce,q,"lineWidth",.5,3,.1,"Neon line stroke width (px).");const $=pe.section("trees");Pt($,"show trees",S.enabled,x=>{S.enabled=x},"Show/hide the forest."),oe($,S.tuning,"maxTrees",1e3,14e3,250,"Cap on generated trees — density + perf. Rebuilds on change."),oe($,S.tuning,"coverLo",.3,.6,.01,"Density threshold: below = clearings. LOWER = more trees. Rebuilds."),oe($,S.tuning,"coverHi",.4,.7,.01,"Density threshold: at/above = full coverage. Rebuilds."),oe($,S.tuning,"sizeScale",.3,3,.1,"Global tree height + width multiplier. Rebuilds on change."),oe($,S.tuning,"radius",300,1200,20,"How far out trees stream + fade (m). Rebuilds on change."),oe($,S.tuning,"glow",.2,3,.1,"HDR brightness baked into tree colours (bloom glow). Rebuilds."),oe($,S.tuning,"fogDensity",2e-4,.0014,5e-5,"Distance haze over the trees — LIVE (no rebuild)."),oe($,S.tuning,"depthBias",0,10,.5,"Pull trees toward the camera (m) so they draw ON TOP of their ridge — kills the ridgeline ripple. 0 = off. LIVE.");const te=f.tuning,ye=pe.section("bird — buffet");oe(ye,te,"buffetGain",0,3,.1,"Master scale on the wind-scaled visual buffet (rock + render-only tremor; 0 = off)."),oe(ye,te,"buffetWindRef",4,30,1,"Local wind speed (m/s) mapped to full buffet saturation (lower = judders sooner)."),oe(ye,te,"rockCapDeg",0,25,1,"Max visual roll from the buffet rock (degrees) — clamps the judder.");const K=m,_e=pe.section("global wind — activity");oe(_e,$e,"loScale",0,2,.05,"Wind strength fraction in the valleys (low altitude)."),oe(_e,$e,"hiScale",0,3,.05,"Wind strength fraction aloft (also drives ridge-lift strength)."),oe(_e,$e,"altLo",0,300,10,"Altitude (m) where calm ends and wind strength begins rising."),oe(_e,$e,"altHi",320,800,10,"Altitude (m) of full wind strength."),oe(_e,rn,"fluidMax",0,100,.5,"Peak |fluid wind| (m/s) the fluid component is clamped to. Steady drift (~6 m/s) adds on top → felt field peaks ~6 above this. Default 10 → max ~16.");const be=pe.section("global wind — render");oe(be,K,"dotPx",1,8,.2,"On-screen comet-head diameter (px)."),oe(be,K,"clearance",5,150,5,"Nominal metres above terrain the motes relax toward (height is advected)."),oe(be,K,"vSpread",10,150,5,"Half-height (m) of the vertical band for mote home clearance spread."),oe(be,K,"homeBias",1,5,.2,"Power (≥1) biasing far-mote heights toward terrain; >1 clusters motes low.");const Ee=pe.section("wind — render modes");Et(Ee,"FAR",ni,"comet",x=>m.setFarMode(x),"Render style for the FAR (global) wind tier."),Et(Ee,"NEAR",ii,"comet",x=>m.setNearMode(x),"Render style for the NEAR (local sphere) tier."),Et(Ee,"WAKE",si,"modulate",x=>m.setWakeMode(x),"Render style for the WAKE (wing) tier.");const N=pe.section("local sphere + wake (off — solving global)");Pt(N,"local sphere",!1,x=>m.setShowNear(x),"Show/hide the local sphere (body) mote layer."),Pt(N,"wake",!1,x=>m.setShowWake(x),"Show/hide the wake (wing) mote layer."),oe(N,K,"ambientNearFloor",0,1,.05,"Ambient (global) terrain-wind weight at the bird (0..1); 1 = full immersion."),oe(N,K,"nearJitter",0,.6,.02,"Per-mote random direction rotation (rad) so the sphere isn't uniform."),oe(N,K,"foreStretch",1,5,.1,"Forward reach of the near bubble as a multiple of nearRadius (>1 = bigger ahead)."),oe(N,K,"nearBodyCount",0,600,20,"Local sphere (body) mote count cap — scales with global wind speed up to this."),oe(N,K,"bodyWindRef",4,30,1,"Global wind speed (m/s) at which the body count reaches its cap."),oe(N,K,"nearWakeCount",0,1e3,20,"Wake (wing) mote count cap — active count scales with bird speed up to this."),oe(N,K,"wakeSpeedRef",10,70,5,"Bird speed (m/s) at which the wake count reaches its cap."),oe(N,K,"wakeMoteLen",.2,4,.1,"Tail-length multiplier for wake motes vs the body comets (1 = same)."),oe(N,K,"nearOpacity",.1,1,.05,"Opacity multiplier for the local sphere (body) motes."),oe(N,K,"wakeOpacity",.1,1,.05,"Opacity multiplier for the wake (wing) motes."),oe(N,K,"swirlGain",0,2,.1,"Tangential swirl strength in the wake (twin-vortex circulation)."),oe(N,K,"wingSpan",0,30,1,"Half-span (m) lateral offset of each wingtip vortex core from the centerline."),oe(N,K,"heatRef",4,50,2,"Wake speed (m/s) mapped to full heat (red + max tail length)."),document.body.appendChild(ie),window.addEventListener("keydown",x=>{x.code==="KeyT"&&(ie.style.display=ie.style.display==="none"?"block":"none"),x.code==="KeyP"&&(et=!et),x.code==="Space"&&(V=!0,x.preventDefault())}),window.addEventListener("keyup",x=>{x.code==="Space"&&(V=!1)});const X=document.createElement("canvas");X.id="compass",X.width=200,X.height=200,X.style.cssText="position:fixed;right:14px;bottom:14px;width:200px;height:200px;z-index:9;background:rgba(6,5,18,0.55);border:1px solid #2a2550;border-radius:8px;",document.body.appendChild(X);const Me=X.getContext("2d"),Z=x=>Math.abs(x)<Rt?0:(x-Math.sign(x)*Rt)/(1-Rt),Se=()=>{t=Math.floor(e.clientWidth*l)||t,h=Math.floor(e.clientHeight*l)||h,e.width=t,e.height=h,a.destroy(),a=n.createTexture({size:[t,h],format:"depth24plus",sampleCount:We,usage:GPUTextureUsage.RENDER_ATTACHMENT}),c.destroy(),c=n.createTexture({size:[t,h],format:je,sampleCount:We,usage:GPUTextureUsage.RENDER_ATTACHMENT}),o.destroy(),o=n.createTexture({size:[t,h],format:je,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.RENDER_ATTACHMENT}),O.resize(t,h)};window.addEventListener("resize",Se),window.__autoWobble=!1;let Ae=0,ve=0,ae=0,xe=0,we=tn,re=f.vel[0],le=f.vel[2];const de=new gn(x=>{if(ae=ae*.9+1/Math.max(x,.001)*.1,et){const Qe=L.update(f,x);H.yawRate=Qe.yawRate,H.pitchTarget=Qe.pitchTarget,window.__autoMode=L.mode}else H.yawRate=Z(j)*ls,H.pitchTarget=nn+Z(G)*hs;H.flap=!et&&V,!et&&window.__autoWobble&&(Ae+=x,H.pitchTarget=Math.sin(Ae*1.1)*.65,H.yawRate=0);const I=R.read();if(I){const[Qe,nt]=R.originXZ;Jn(I.u,I.v,R.gridW,R.gridH,Qe,nt,R.cellMeters,R.currentScale)}f.integrate(x,H),A.checkReached(f.pos,cs)&&(xe++,A.respawn(f.pos[0],f.pos[2],f.heading));const U=Math.min(1,Math.max(0,(f.lastClearance-rt.clearance)/(At.clearance-rt.clearance)));z.followHeight=rt.height+U*(At.height-rt.height),z.lookPitch=(rt.pitchDeg+U*(At.pitchDeg-rt.pitchDeg))*Math.PI/180,z.target=[f.pos[0],f.pos[1],f.pos[2]];const F=Math.min(1,x*2.5);re+=(f.vel[0]-re)*F,le+=(f.vel[2]-le)*F;const W=Math.hypot(re,le)||1;z.forward=[re/W,0,le/W],z.update();const J=Math.min(1,Math.max(0,(f.lastSpeed-f.tuning.glideSpeed)/(f.tuning.maxSpeed-f.tuning.glideSpeed)));we+=(tn+J*rs-we)*Math.min(1,x*5);const Q=ji(we,t/h,os,as),ne=z.viewMatrix(),fe=Yi(Q,ne),ue=c.createView(),Te=o.createView(),ze=r.getCurrentTexture().createView(),Re=a.createView(),Pe=z.getEye(),me=n.createCommandEncoder();R.step(me,x,f.pos[0],f.pos[2]);const ke=z.groundPos(),Be=z.forwardHoriz(),Oe=z.rightHoriz();tt!=="ekg"?(w.mode=tt,w.draw(me,ue,Re,fe,ke,Pe,{r:Ze[0],g:Ze[1],b:Ze[2],a:1})):p.draw(me,ue,Re,fe,ke,Be,Oe,Pe,{r:Ze[0],g:Ze[1],b:Ze[2],a:1}),en&&m.draw(me,ue,Re,fe,ke,Be,Oe,Pe,f.simTime,Ze,.5/1400,t/h,f.pos,f.vel),f.draw(me,ue,Re,fe),S.draw(me,ue,Re,fe,ke,Pe,f.simTime,S.tuning.fogDensity),A.draw(me,ue,Re,fe,Pe,f.simTime),B.draw(me,ue,Re,fe,[f.pos[0],f.pos[1],f.pos[2]],f.pos[1]-f.lastClearance,f.simTime,Te),O.apply(me,Te,ze),n.queue.submit([me.finish()]),R.afterSubmit(),window.__camPos=Pe,window.__birdPos=f.pos,window.__birdPitch=f.pitch,window.__birdHeading=f.heading,window.__birdGroundTrack=f.lastGroundTrack,window.__birdWind=f.lastWind,window.__birdVario=f.lastVario,window.__birdBuffet=f.buffetOffset,ve++;const He=f.heading*180/Math.PI%360,Ue=f.lastGroundTrack*180/Math.PI;let Ie=Ue-f.heading*180/Math.PI;Ie=((Ie+180)%360+360)%360-180;const Ge=f.lastVario,Ke=`${Ge>=0?"+":""}${Ge.toFixed(1)}`,qe=Math.hypot(f.lastWind[0],f.lastWind[1]),De=A.distanceTo(f.pos);let Le=(Math.atan2(A.x-f.pos[0],A.z-f.pos[2])-f.heading)*180/Math.PI;Le=((Le+180)%360+360)%360-180;const Je=Le>5?"►":Le<-5?"◄":"▲";u.textContent=`vector-system — bird3d (wind glider · fly to target)${et?`   AUTO: ${L.mode} (click/P=manual)`:"   MANUAL (P=autopilot)"}${f.lastFlapping?"   ▲ FLAP":""}${f.lastCrashing?"   ✖ CRASH":""}
TARGET: ${De.toFixed(0)} m   steer ${Je} ${Math.abs(Le).toFixed(0)}°   reached: ${xe}
alt over terrain: ${f.lastClearance.toFixed(0)} m   air: ${f.lastSpeed.toFixed(0)} m/s
vario: ${Ke} m/s ${Ge>.5?"▲":Ge<-.5?"▼":"—"}   updraft: +${f.lastUpdraft.toFixed(1)} m/s
heading: ${He.toFixed(0)}°   ground-track: ${Ue.toFixed(0)}°   DRIFT: ${Ie>=0?"+":""}${Ie.toFixed(0)}°
wind: ${f.lastWind[0].toFixed(1)}, ${f.lastWind[1].toFixed(1)} m/s  (|${qe.toFixed(1)}|)
fps: ${ae.toFixed(0)}   frame ${ve}   (steer=mouse · SPACE=flap · cursor under bird=climb, over=dive · T=tuning)`,gs(Me,f.heading,f.lastGroundTrack,f.lastWind,qe,Ie)});window.__windAt=(x,I)=>pt(x,I,0),window.__updraftAt=(x,I,U)=>ut(x,I,0,p,{...f.tuning,...U??{}}),window.__wind=m,window.__windProfile=x=>ei(x),window.__windProfileAt=x=>Ne(x),window.__birdTune=x=>Object.assign(f.tuning,x),window.__nearWake=(x,I,U)=>m.sampleWake(x,I,U),window.__nearFrame=()=>m.nearFrame(),window.__farMode=x=>m.setFarMode(x),window.__nearMode=x=>m.setNearMode(x),window.__wakeMode=x=>m.setWakeMode(x),window.__trees=S,window.__showWind=x=>{en=x},window.__terrainMode=x=>{tt=x};const Ce=async()=>{const x=await Ji(n,(U,F)=>p.sampleHeight(U,F)),I=x.maxDiff<1?"PASS":"FAIL";return console.log(`[terrain-selfcheck] ${I}  points=${x.points}  maxDiff=${x.maxDiff.toFixed(2)}m  meanDiff=${x.meanDiff.toFixed(3)}m
  worst @ (${x.worst.x}, ${x.worst.z})  cpu=${x.worst.cpu.toFixed(1)}m  gpu=${x.worst.gpu.toFixed(1)}m
  ${x.verdict}`),x};window.__terrainCheck=Ce,Ce(),de.start(),window.__birdBooted=!0}function us(){const u=document.createElement("div");return u.id="tune",u.style.cssText="position:fixed;right:12px;top:12px;display:none;padding:10px 12px;max-height:92vh;overflow-y:auto;background:rgba(8,6,20,0.85);border:1px solid #3a3360;border-radius:6px;font:12px/1.6 monospace;color:#9fe8ff;z-index:10;min-width:240px;",u}function ps(u){const e=[];return{section:r=>{const s=document.createElement("div"),l=document.createElement("span"),t=document.createElement("span"),h=document.createElement("div");l.style.cssText="display:inline-block;width:14px;",t.textContent=r,s.append(l,t),s.style.cssText="color:#c9a8ff;font-weight:bold;margin:6px 0 4px;padding:4px 2px;border-top:1px solid #3a3360;cursor:pointer;user-select:none;",h.style.cssText="padding:2px 0 6px;";const a=c=>{h.style.display=c?"block":"none",l.textContent=c?"▾ ":"▸ "};return s.addEventListener("click",()=>{const c=h.style.display==="none";for(const o of e)o(!1);a(c)}),a(e.length===0),e.push(a),u.append(s,h),h}}}function Pt(u,e,n,r,s){let l=n;const t=document.createElement("button");t.style.cssText="width:100%;margin:0 0 6px;padding:4px;background:#241d40;color:#9fe8ff;border:1px solid #4a4070;border-radius:4px;font:12px monospace;cursor:pointer;",s&&(t.title=s);const h=()=>{t.textContent=`${e}: ${l?"ON":"OFF"}`};h(),t.onclick=()=>{l=!l,r(l),h()},u.appendChild(t)}function Et(u,e,n,r,s,l){let t=Math.max(0,n.indexOf(r));const h=document.createElement("button");h.style.cssText="width:100%;margin:0 0 6px;padding:4px;background:#241d40;color:#9fe8ff;border:1px solid #4a4070;border-radius:4px;font:12px monospace;cursor:pointer;",l&&(h.title=l);const a=()=>{h.textContent=`${e}: ${n[t]} ▸`};a(),h.onclick=()=>{t=(t+1)%n.length,s(n[t]),a()},u.appendChild(h)}function oe(u,e,n,r,s,l,t){const h=document.createElement("div"),a=document.createElement("span"),c=document.createElement("span"),o=document.createElement("input");o.type="range",o.min=String(r),o.max=String(s),o.step=String(l),o.value=String(e[n]),o.style.cssText="width:110px;vertical-align:middle;margin:0 6px;",a.textContent=n.padEnd(11),c.textContent=String(e[n]),t&&(h.title=t),o.addEventListener("input",()=>{e[n]=Number(o.value),c.textContent=o.value}),h.append(a,o,c),u.appendChild(h)}function gs(u,e,n,r,s,l){const t=u.canvas.width,h=u.canvas.height,a=t/2,c=h/2;u.clearRect(0,0,t,h),u.strokeStyle="rgba(120,120,180,0.35)",u.lineWidth=1,u.beginPath(),u.arc(a,c,72,0,Math.PI*2),u.stroke();const o=(w,v)=>[a+Math.sin(w)*v,c-Math.cos(w)*v],d=(w,v,b,f)=>{const[y,m]=o(w,v);u.strokeStyle=b,u.fillStyle=b,u.lineWidth=f,u.beginPath(),u.moveTo(a,c),u.lineTo(y,m),u.stroke();const M=Math.atan2(m-c,y-a);u.beginPath(),u.moveTo(y,m),u.lineTo(y-9*Math.cos(M-.4),m-9*Math.sin(M-.4)),u.lineTo(y-9*Math.cos(M+.4),m-9*Math.sin(M+.4)),u.closePath(),u.fill()},p=Math.atan2(r[0],r[1]),g=Math.min(70,14+s*3);d(p,g,"rgba(230,90,230,0.95)",5),d(e,66,"rgba(80,220,255,0.95)",3),d(n,66,"rgba(255,225,70,0.95)",3),u.font="11px monospace",u.fillStyle="#9fe8ff",u.fillText("heading",8,16),u.fillStyle="#ffe146",u.fillText("track",8,30),u.fillStyle="#e65ae6",u.fillText("wind",8,44),u.fillStyle="#fff",u.fillText(`drift ${l>=0?"+":""}${l.toFixed(0)}°`,8,h-10)}fs().catch(u=>{const e=document.getElementById("overlay");throw e&&(e.textContent="boot error: "+u.message),u});
