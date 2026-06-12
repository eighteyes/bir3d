// wind.wgsl — drifting neon CURVED-COMET motes showing the terrain-shaped wind field (WebGPU, NDC z in [0,1]).
// Responsibilities:
//   - Render each wind mote as a CURVED multi-segment ribbon: the host (src/host/gpu/wind.ts) integrates
//     each mote's tail BACKWARD along the terrain-shaped flow (flowAt) into a world-space polyline, then
//     emits one quad per segment. This VS receives, per vertex, a segment ENDPOINT (world xyz), a corner
//     {x = near(0)/far(1) endpoint, y = perpendicular ±1}, the segment's world-XZ direction, the head→tail
//     `along` fraction, and a CPU-computed `vis` (density cull). The ribbon thickness is laid perpendicular
//     to the segment's SCREEN-space direction so the curved comet has constant on-screen width.
//   - Each mote's world path is advected by the SHARED terrain-shaped flow and persisted frame-to-frame, so
//     the curve you SEE arcing over the ridges is the flow that PUSHES the glider. Motes RISE over windward
//     slopes and sink in lees (vertical advection, host-side) — air visibly pours up and over the ridgelines.
//   - v9: tail LENGTH and DENSITY are computed HOST-side (the tail is a real integrated polyline; density is
//     a CPU rank cull passed in `vis`), removing the fragile vertex-index hash. SPEED still tints/brightens.
//   - Bright neon head (cyan→white by wind speed) fading to a transparent tail (alpha along the ribbon),
//     additive (host blend), depth-tested (no write) so terrain ridges occlude the comets; distance fog
//     matches the terrain haze so far motes dissolve cleanly.

struct Uniforms {
  viewProj : mat4x4<f32>,
  eyeAspect : vec4<f32>,   // eye.xyz, aspect (pxW/pxH)
  fog : vec4<f32>,         // fogColor.rgb, fogDensity
  misc : vec4<f32>,        // dotSize (NDC ribbon half-width), pad, pad, pad
};

@group(0) @binding(0) var<uniform> U : Uniforms;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) along : f32,          // 0 at head → 1 at tail (for length fade)
  @location(1) across : f32,         // -1..1 perpendicular (for width falloff)
  @location(2) speedFrac : f32,      // 0..1 wind speed
  @location(3) viewDist : f32,
  @location(4) vis : f32,            // 0..1 density-fade visibility (0 = culled in calm air)
};

@vertex
fn vs(
  @location(0) endpoint : vec3<f32>, // this segment endpoint, world space
  @location(1) corner : vec2<f32>,   // corner.x in {0,1} near/far endpoint (already picked host-side), .y perp ±1
  @location(2) speedFrac : f32,
  @location(3) segDir : vec2<f32>,   // this segment's world-XZ direction (unit-ish)
  @location(4) along : f32,          // head=0 → tail=1 fade fraction at this endpoint
  @location(5) vis : f32             // CPU density cull (0 = culled)
) -> VSOut {
  var out : VSOut;
  let clip = U.viewProj * vec4<f32>(endpoint, 1.0);
  // Project the segment's world direction to clip space to lay the ribbon perpendicular ON SCREEN.
  let world2 = endpoint + vec3<f32>(segDir.x, 0.0, segDir.y);
  let clip2 = U.viewProj * vec4<f32>(world2, 1.0);
  var sdir = vec2<f32>(
    (clip2.x / clip2.w) - (clip.x / clip.w),
    ((clip2.y / clip2.w) - (clip.y / clip.w)) / U.eyeAspect.w,
  );
  let sl = length(sdir);
  if (sl > 1e-5) { sdir = sdir / sl; } else { sdir = vec2<f32>(1.0, 0.0); }
  let sperp = vec2<f32>(-sdir.y, sdir.x);

  let halfW = U.misc.x;        // NDC ribbon half-width
  // collapse fully-culled motes to a degenerate point so they never draw.
  let cull = step(0.001, vis);
  // thin the ribbon toward the tail so the curved comet tapers to a point.
  let taper = (1.0 - 0.7 * clamp(along, 0.0, 1.0));
  let off2 = sperp * (corner.y * halfW * taper * cull);

  var outClip = clip;
  outClip.x += off2.x * clip.w;
  outClip.y += off2.y * U.eyeAspect.w * clip.w;
  out.clip = outClip;
  out.along = along;
  out.across = corner.y;
  out.speedFrac = speedFrac;
  out.viewDist = length(endpoint - U.eyeAspect.xyz);
  out.vis = vis;
  return out;
}

const CYAN : vec3<f32> = vec3<f32>(0.30, 0.85, 1.0);
const WHITE : vec3<f32> = vec3<f32>(0.85, 0.97, 1.0);

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  if (in.vis <= 0.001) { discard; }
  // perpendicular soft falloff → rounded ribbon; length fade → bright head, dissolving tail.
  let perp = 1.0 - abs(in.across);
  if (perp <= 0.0) { discard; }
  let lenFade = pow(1.0 - clamp(in.along, 0.0, 1.0), 1.3); // head bright → tail fades to 0
  let glow = pow(perp, 1.6) * lenFade;

  // faster wind → brighter + whiter core. Raised base so slow-wind motes still read against the dark.
  // density-fade visibility also modulates brightness so motes entering/leaving fade smoothly.
  let intensity = glow * (0.9 + in.speedFrac * 1.2) * in.vis;
  let tint = mix(CYAN, WHITE, clamp(in.speedFrac, 0.0, 1.0));

  // distance fog → far motes dissolve into the haze.
  let fog = exp(-U.fog.w * in.viewDist);

  var color = tint * intensity * clamp(fog, 0.0, 1.0);
  color = min(color, vec3<f32>(1.2, 1.3, 1.5));
  return vec4<f32>(color, 1.0);
}
