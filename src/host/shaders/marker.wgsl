// marker.wgsl — altitude plumb-line + ground diamond for the bird (line-list, additive, depth-tested).
// Responsibilities:
//   - vs: place unit geometry in world space from the uniform. kind 0 = vertical drop line from just
//     under the bird to just above the ground; kind 1 = a small diamond on the ground under the bird,
//     gently pulsing. Carries t (0 at bird, 1 at ground) for the dash pattern.
//   - fs: dash the drop line every DASH_M meters of real height (dash count = readable altimeter);
//     render the diamond as a solid soft-cyan glow. Additive blend; depth-tested (ridges occlude).

const DASH_M: f32 = 9.0;

struct U {
  viewProj : mat4x4<f32>,
  birdPos  : vec3<f32>,
  groundY  : f32,
  time     : f32,
  height   : f32, // bird.y - groundY (m) — drives the dash count
  pad0     : f32,
  pad1     : f32,
};
@group(0) @binding(0) var<uniform> u : U;

struct VOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) t    : f32, // 0..1 along the drop line; -1 for the diamond
};

@vertex
fn vs(@location(0) pos : vec3<f32>, @location(1) attr : vec3<f32>) -> VOut {
  var out : VOut;
  let kind = attr.x;
  let t = attr.y;
  var world : vec3<f32>;
  if (kind < 0.5) {
    // drop line: start a touch below the bird body, end a touch above the ground line.
    world = vec3<f32>(u.birdPos.x, mix(u.birdPos.y - 4.0, u.groundY + 1.0, t), u.birdPos.z);
    out.t = t;
  } else {
    // ground diamond: unit cross scaled with a soft pulse, sitting just above the ground.
    let scale = 7.0 + 1.5 * sin(u.time * 4.0);
    world = vec3<f32>(u.birdPos.x + pos.x * scale, u.groundY + 1.2, u.birdPos.z + pos.z * scale);
    out.t = -1.0;
  }
  out.clip = u.viewProj * vec4<f32>(world, 1.0);
  return out;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4<f32> {
  let cyan = vec3<f32>(0.25, 0.85, 1.0);
  if (in.t < 0.0) {
    // ground diamond: solid soft glow.
    return vec4<f32>(cyan * 0.55, 1.0);
  }
  // drop line: one dash per DASH_M meters of real height; brighter toward the ground end.
  let dash = step(fract(in.t * u.height / DASH_M), 0.55);
  let glow = mix(0.18, 0.45, in.t);
  return vec4<f32>(cyan * glow * dash, 1.0);
}
