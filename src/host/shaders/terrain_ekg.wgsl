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

// --- fBm — IDENTICAL constants mirrored in terrain.ts sampleHeight ---
const BASE_FREQ : f32 = 0.00285714;  // ~1/350 per meter
const LACUNARITY : f32 = 2.0;
const GAIN : f32 = 0.5;
const OCTAVES : i32 = 3;
const RELIEF : f32 = 220.0;

fn hash2(p : vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}
fn valueNoise(p : vec2<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let a = hash2(i);
  let b = hash2(i + vec2<f32>(1.0, 0.0));
  let c = hash2(i + vec2<f32>(0.0, 1.0));
  let d = hash2(i + vec2<f32>(1.0, 1.0));
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
  return (sum / norm) * RELIEF;
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
