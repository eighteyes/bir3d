// terrain_grid.wgsl — WORLD-STATIC wireframe terrain. A square grid pinned to WORLD coordinates, draped
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
