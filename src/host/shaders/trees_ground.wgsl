// trees_ground.wgsl — Per-tree ground-height prepass: compute the terrain fBm ONCE per tree (not once
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

fn hash2(p: vec2<f32>) -> f32 { return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453); }
fn valueNoise(p: vec2<f32>) -> f32 {
  let i = floor(p); let f = fract(p);
  let a = hash2(i); let b = hash2(i + vec2<f32>(1.0, 0.0));
  let c = hash2(i + vec2<f32>(0.0, 1.0)); let d = hash2(i + vec2<f32>(1.0, 1.0));
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
