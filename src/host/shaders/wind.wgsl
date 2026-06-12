// wind.wgsl — neon streamline "comet" ribbons showing the wind field (WebGPU, NDC z in [0,1]).
// Responsibilities:
//   - Draw camera-relative wind streamlines (line-list) integrated CPU-side from the SHARED windAt
//     field (src/host/gpu/wind.ts) — the field you SEE is the field that PUSHES the glider.
//   - Animate comets scrolling ALONG each static field line: a moving bright pulse over a dim base,
//     so direction + flow are unmistakable. Brightness/length scale with wind speed (speedFrac).
//   - Cyan→white emissive, additive (host blend) for glow; depth-tested (no write) so terrain ridges
//     occlude the traces. Distance fog matches the terrain haze so far traces dissolve cleanly.

struct Uniforms {
  viewProj : mat4x4<f32>,
  eyePhase : vec4<f32>,    // eye.xyz, phase(time)
  fog : vec4<f32>,         // fogColor.rgb, fogDensity
};

@group(0) @binding(0) var<uniform> U : Uniforms;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) arc : f32,        // 0..1 along the streamline
  @location(1) speedFrac : f32,  // 0..1 wind speed
  @location(2) viewDist : f32,
};

@vertex
fn vs(
  @location(0) world : vec3<f32>,
  @location(1) arc : f32,
  @location(2) speedFrac : f32
) -> VSOut {
  var out : VSOut;
  out.clip = U.viewProj * vec4<f32>(world, 1.0);
  out.arc = arc;
  out.speedFrac = speedFrac;
  out.viewDist = length(world - U.eyePhase.xyz);
  return out;
}

const CYAN : vec3<f32> = vec3<f32>(0.20, 0.85, 1.0);
const WHITE : vec3<f32> = vec3<f32>(0.9, 0.98, 1.0);

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  let phase = U.eyePhase.w;
  // base dim trace so the field lines are always faintly visible.
  let base = 0.10;
  // scrolling comet: a moving bright band along arc. Speed of scroll ∝ wind speed so faster wind
  // visibly streaks faster. Multiple comets per line via fract() wrap.
  let scroll = fract(in.arc * 3.0 - phase * (0.25 + in.speedFrac * 0.6));
  // sharp leading pulse: bright near scroll≈0, decaying tail.
  let comet = pow(1.0 - scroll, 6.0);
  let intensity = base + comet * (0.5 + in.speedFrac * 1.4);

  // color shifts cyan→white at the comet head.
  let tint = mix(CYAN, WHITE, clamp(comet, 0.0, 1.0));

  // distance fog → far traces dissolve into the haze.
  let fogD = U.fog.w;
  let fog = exp(-fogD * in.viewDist);

  var color = tint * intensity * clamp(fog, 0.0, 1.0);
  color = min(color, vec3<f32>(1.2, 1.3, 1.5));
  return vec4<f32>(color, 1.0);
}
