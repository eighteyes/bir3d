// terrain_ekg.wgsl — EKG/waveform stacked neon trace LINES terrain (WebGPU, NDC z in [0,1]).
// Responsibilities:
//   - NO FILL. Render the heightfield purely as a stack of horizontal neon polylines (Joy Division
//     "Unknown Pleasures" / oscilloscope look). Each vertex is one sample of one depth row.
//   - CAMERA-RELATIVE ROWS (v4): each row is PERPENDICULAR to the camera. A row at depth d spans
//     camRight from -halfWidth..+halfWidth. World sample P = camGround + camFwd*d + camRight*(xFrac*
//     halfWidth); height y = fBm(P.xz). So the stacked lines stay SCREEN-HORIZONTAL at every heading
//     (v3 locked rows to world-East → turning skewed them diagonally; this fixes the skew).
//   - Fragment: emissive neon, height-tinted; near rows brighter, far rows compress and fade into
//     the dark haze. CLEAN HORIZON (v4): hard distance cutoff (maxDist) + exponential fog discard
//     far fragments so the horizon never tangles. Additive blend (host) gives glow.
//   - fBm MUST match terrain.ts sampleHeight() (same constants/hash) — drives the bird ridge-lift.

struct Uniforms {
  viewProj : mat4x4<f32>,
  camGround : vec2<f32>,   // camera ground position (x,z) the line stack is built around
  halfWidth : f32,         // half horizontal extent of each row (m)
  maxDist : f32,           // hard draw-distance cutoff (m) → clean horizon
  camFwd : vec2<f32>,      // horizontal camera forward (x,z), unit
  camRight : vec2<f32>,    // horizontal camera right (x,z), unit
  fogColor : vec3<f32>,
  fogDensity : f32,
  eyePos : vec3<f32>,      // world-space camera eye, for view distance
  _pad1 : f32,
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

@vertex
fn vs(@location(0) sample : vec2<f32>, @location(1) rowFade : f32) -> VSOut {
  // sample.x = xFrac in [-1,1]; sample.y = rowDepth in meters ahead of the camera.
  // CAMERA-RELATIVE: P = camGround + camFwd*depth + camRight*(xFrac*halfWidth). The row is
  // perpendicular to camFwd and spans camRight → screen-horizontal whichever way we fly.
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

const NEON_A : vec3<f32> = vec3<f32>(0.10, 0.95, 0.80);  // teal-green (low ground)
const NEON_B : vec3<f32> = vec3<f32>(0.85, 0.20, 0.95);  // magenta (high crests)

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  // CLEAN HORIZON: hard cutoff — discard any fragment beyond the draw distance so the far rows
  // never pile into a horizon tangle. Reliable lever (additive thin lines don't depth-occlude well).
  if (in.rowDepth > U.maxDist) {
    discard;
  }

  // height-tinted neon line color.
  let tint = mix(NEON_A, NEON_B, clamp(in.height / RELIEF, 0.0, 1.0));

  // near rows brighter, far rows dimmer (rowFade 0 near .. 1 far).
  let rowBright = mix(1.0, 0.18, in.rowFade);

  // exponential distance fog → far lines dissolve into the dark haze BEFORE the cutoff.
  let fog = exp(-U.fogDensity * in.viewDist);
  // extra ramp: fade hard over the last 25% of draw distance so the cutoff edge isn't a hard line.
  let edge = clamp((U.maxDist - in.rowDepth) / (U.maxDist * 0.25), 0.0, 1.0);

  var color = tint * rowBright * clamp(fog, 0.0, 1.0) * edge;
  // cap brightness (additive blend → no full-screen blowout).
  color = min(color, vec3<f32>(0.9, 0.95, 1.0));
  return vec4<f32>(color, 1.0);
}
