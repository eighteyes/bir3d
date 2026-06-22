// terrain_ekg.wgsl — EKG/waveform stacked neon trace LINES + opaque hidden-line FILL (WebGPU, NDC z 0..1).
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
