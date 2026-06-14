// bloom_threshold.wgsl — bright-pass extraction for the bloom chain.
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
