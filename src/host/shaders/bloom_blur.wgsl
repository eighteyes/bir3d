// bloom_blur.wgsl — separable Gaussian blur (one axis per pass) for the bloom chain.
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
