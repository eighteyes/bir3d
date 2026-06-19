// target.wgsl — flight target beacon: a camera-facing vertical beam of light at a world waypoint.
// Responsibilities:
//   - vs: build a billboarded vertical quad from the uniform (base at ground, up by `height`, widened
//     along a CPU-supplied horizontal `rightAxis` by `halfWidth`). Carries (cx,cy) for the fs shade.
//   - fs: additive amber glow — gaussian horizontal falloff (soft feathered edges), brighter at the
//     base, gentle time pulse. Depth handled by the pipeline (drawn always-on-top for navigation).

struct U {
  viewProj  : mat4x4<f32>,
  basePos   : vec3<f32>,
  height    : f32,
  rightAxis : vec3<f32>,
  halfWidth : f32,
  color     : vec3<f32>,
  time      : f32,
};
@group(0) @binding(0) var<uniform> u : U;

struct VOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) cx : f32, // -1..1 across the beam width
  @location(1) cy : f32, //  0 at base, 1 at top
};

@vertex
fn vs(@location(0) corner : vec3<f32>) -> VOut {
  var out : VOut;
  let cx = corner.x; // -1 or 1
  let cy = corner.y; //  0 or 1
  let world = u.basePos
    + u.rightAxis * (cx * u.halfWidth)
    + vec3<f32>(0.0, 1.0, 0.0) * (cy * u.height);
  out.clip = u.viewProj * vec4<f32>(world, 1.0);
  out.cx = cx;
  out.cy = cy;
  return out;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4<f32> {
  let horiz = exp(-in.cx * in.cx * 3.0);     // soft glowing core, feathered edges
  let vert = mix(1.0, 0.12, in.cy);          // brighter at the base, fades up the column
  let pulse = 0.8 + 0.2 * sin(u.time * 3.0); // gentle "alive / objective" pulse
  let glow = horiz * vert * pulse;
  return vec4<f32>(u.color * glow, 1.0);
}
