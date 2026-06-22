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
