// terrain3d.wgsl — neon receding-ridgeline terrain (WebGPU 3D, NDC z in [0,1]).
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
const OCTAVES : i32 = 5;
const RELIEF : f32 = 120.0;          // total relief target (meters)

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
    sum = sum + amp * valueNoise(p * freq);
    norm = norm + amp;
    freq = freq * LACUNARITY;
    amp = amp * GAIN;
  }
  // normalize to ~[0,1] then scale to relief, recenter so mean ~0.
  return (sum / norm - 0.5) * RELIEF;
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

  // height-tinted neon: low ground teal, high ground magenta.
  let tint = mix(NEON_A, NEON_B, clamp(in.height / 80.0 + 0.5, 0.0, 1.0));
  let glow = tint * (contour * 1.0 + crest * 0.6);

  var color = SURFACE + glow;

  // exponential distance fog mixes toward dark haze background — the fog IS the depth.
  let fog = exp(-U.fogDensity * in.viewDist);
  color = mix(U.fogColor, color, clamp(fog, 0.0, 1.0));

  // cap brightness (no full-screen flashes).
  color = min(color, vec3<f32>(0.9, 0.95, 1.0));
  return vec4<f32>(color, 1.0);
}
