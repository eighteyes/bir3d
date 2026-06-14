// bloom_composite.wgsl — final composite + tone-map for the bloom chain.
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
