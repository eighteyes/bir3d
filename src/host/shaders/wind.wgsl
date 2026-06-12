// wind.wgsl — drifting neon DOT particles showing the wind field (WebGPU, NDC z in [0,1]).
// Responsibilities:
//   - Render wind motes as camera-facing additive quads. Per-dot world center (repeated 6×) +
//     a corner offset (±1) expand into a constant screen-size billboard in the vertex stage.
//   - Each dot's world position is advected CPU-side by the SHARED windAt field (src/host/gpu/wind.ts)
//     and persisted frame-to-frame, so the field you SEE drifting is the field that PUSHES the glider.
//   - Soft round neon glow (radial falloff from corner coords), cyan→white by wind speed; additive
//     (host blend). Depth-tested (no write) so terrain ridges occlude the motes. Distance fog matches
//     the terrain haze so far motes dissolve cleanly.

struct Uniforms {
  viewProj : mat4x4<f32>,
  eyeAspect : vec4<f32>,   // eye.xyz, aspect (pxW/pxH)
  fog : vec4<f32>,         // fogColor.rgb, fogDensity
  misc : vec4<f32>,        // dotSize(NDC half-extent), 0,0,0
};

@group(0) @binding(0) var<uniform> U : Uniforms;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) corner : vec2<f32>,   // -1..1 within the quad (for radial falloff)
  @location(1) speedFrac : f32,      // 0..1 wind speed
  @location(2) viewDist : f32,
};

@vertex
fn vs(
  @location(0) center : vec3<f32>,   // world center (repeated per quad vertex)
  @location(1) corner : vec2<f32>,   // (-1,-1)..(1,1) quad corner
  @location(2) speedFrac : f32
) -> VSOut {
  var out : VSOut;
  var clip = U.viewProj * vec4<f32>(center, 1.0);
  // expand to a constant screen-size billboard: offset in clip space scaled by w (so it stays the
  // same pixel size at any depth), aspect-corrected so the dot is round not stretched.
  let size = U.misc.x;
  clip.x += corner.x * size * clip.w;
  clip.y += corner.y * size * U.eyeAspect.w * clip.w;
  out.clip = clip;
  out.corner = corner;
  out.speedFrac = speedFrac;
  out.viewDist = length(center - U.eyeAspect.xyz);
  return out;
}

const CYAN : vec3<f32> = vec3<f32>(0.30, 0.85, 1.0);
const WHITE : vec3<f32> = vec3<f32>(0.85, 0.97, 1.0);

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  // radial soft falloff → round glowing mote; discard outside the disc to avoid square edges.
  let r = length(in.corner);
  if (r > 1.0) { discard; }
  // softer falloff (1.6 vs 2.2) → a wider visible glow disc so each mote reads as a dot, not a pinprick.
  let glow = pow(1.0 - r, 1.6);

  // faster wind → brighter + whiter core. Raised base so slow-wind motes still read against the dark.
  let intensity = glow * (0.85 + in.speedFrac * 1.1);
  let tint = mix(CYAN, WHITE, clamp(in.speedFrac, 0.0, 1.0));

  // distance fog → far motes dissolve into the haze.
  let fog = exp(-U.fog.w * in.viewDist);

  var color = tint * intensity * clamp(fog, 0.0, 1.0);
  color = min(color, vec3<f32>(1.2, 1.3, 1.5));
  return vec4<f32>(color, 1.0);
}
