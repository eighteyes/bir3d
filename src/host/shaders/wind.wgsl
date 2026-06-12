// wind.wgsl — drifting neon COMET motes showing the wind field (WebGPU, NDC z in [0,1]).
// Responsibilities:
//   - Render each wind mote as a small camera-facing comet: a quad stretched along the screen-space
//     wind direction. Per-mote world center (repeated 6×) + a corner offset where corner.x is the
//     ALONG-streak axis (head 0 → tail 1 via the +1/-1 encoding) and corner.y is the perpendicular
//     half-width. The streak is oriented in screen space from the mote's world-space wind direction.
//   - Each mote's world position is advected CPU-side by the SHARED windAt field (src/host/gpu/wind.ts)
//     and persisted frame-to-frame, so the field you SEE streaking is the field that PUSHES the glider.
//   - v8: SPEED is read off the field two ways, both scaling with the mote's local |windAt| (speedFrac):
//       * TAIL LENGTH — the streak length scales from a calm-air stub (tailFloor·base) up to the full
//         base tail in fast air, so long streaks mark fast lanes and short stubs mark calm air.
//       * DENSITY — a stable per-mote hash gives each mote a rank in 0..1; a mote survives the
//         speed-fade only if its rank falls under densityFloor + (1-densityFloor)·speedFrac. Calm air
//         keeps a faint floor of motes (wind EVERYWHERE); fast air shows far more. The cutoff is
//         smoothstepped so motes brighten/dim across speed contours instead of popping.
//   - Bright neon head (cyan→white by wind speed) fading to a transparent tail (alpha along streak),
//     additive (host blend), so drift + direction read even in a still frame and the motes stay
//     distinct from the sky starfield. Depth-tested (no write) so terrain ridges occlude them; distance
//     fog matches the terrain haze so far motes dissolve cleanly.

struct Uniforms {
  viewProj : mat4x4<f32>,
  eyeAspect : vec4<f32>,   // eye.xyz, aspect (pxW/pxH)
  fog : vec4<f32>,         // fogColor.rgb, fogDensity
  misc : vec4<f32>,        // dotSize(NDC half-width), tailLen(NDC base streak), tailFloor, densityFloor
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

// stable per-mote hash → rank in 0..1 (deterministic, frame-stable: depends only on mote index).
fn hash11(n : f32) -> f32 {
  return fract(sin(n * 12.9898) * 43758.5453);
}

@vertex
fn vs(
  @builtin(vertex_index) vidx : u32,
  @location(0) center : vec3<f32>,   // world center (repeated per quad vertex)
  @location(1) corner : vec2<f32>,   // corner.x in {-1,+1} along streak, corner.y in {-1,+1} perp
  @location(2) speedFrac : f32,
  @location(3) windDir : vec2<f32>   // world XZ wind direction at the mote (unit-ish)
) -> VSOut {
  var out : VSOut;
  let clip = U.viewProj * vec4<f32>(center, 1.0);
  // Project the world wind direction to clip space to orient the streak. Sample a second clip point a
  // little down-wind and take the screen-space delta (perspective-correct enough for a short streak).
  let world2 = center + vec3<f32>(windDir.x, 0.0, windDir.y);
  let clip2 = U.viewProj * vec4<f32>(world2, 1.0);
  // screen-space wind direction (NDC, aspect-corrected so it's a true on-screen direction).
  var sdir = vec2<f32>(
    (clip2.x / clip2.w) - (clip.x / clip.w),
    ((clip2.y / clip2.w) - (clip.y / clip.w)) / U.eyeAspect.w,
  );
  let sl = length(sdir);
  if (sl > 1e-5) { sdir = sdir / sl; } else { sdir = vec2<f32>(1.0, 0.0); }
  let sperp = vec2<f32>(-sdir.y, sdir.x);

  let tailFloor = U.misc.z;
  let densityFloor = U.misc.w;
  // TAIL ∝ SPEED: stub (tailFloor·base) in calm air → full base tail in fast air.
  let tailLen = U.misc.y * (tailFloor + (1.0 - tailFloor) * speedFrac);
  let halfW = U.misc.x;     // NDC half-width

  // DENSITY ∝ SPEED: stable per-mote rank; survive if rank < densityFloor + (1-densityFloor)*speedFrac.
  // smoothstep the cutoff over a small band → motes fade in/out across speed contours (no pop).
  let moteId = f32(vidx / 6u);
  let rank = hash11(moteId);
  let cutoff = densityFloor + (1.0 - densityFloor) * speedFrac;
  out.vis = 1.0 - smoothstep(cutoff - 0.10, cutoff + 0.02, rank);

  // corner.x == +1 → head (lead, down-wind), corner.x == -1 → tail (up-wind, opposite motion).
  let alongN = (corner.x * 0.5 + 0.5);          // head=1, tail=0 in raw corner space
  out.along = 1.0 - alongN;                      // head=0 → tail=1 for the fade
  // head sits at the mote center and leads by halfW; tail extends back by the per-mote tailLen.
  // collapse fully-culled motes to a degenerate point (offset 0) so they never draw.
  let cull = step(0.001, out.vis);
  let alongAmt = select(-tailLen, halfW, corner.x > 0.0) * cull;
  let off2 = (sdir * alongAmt + sperp * (corner.y * halfW * cull));
  // back to clip space: aspect-correct y, scale by w so size is depth-independent on screen.
  var outClip = clip;
  outClip.x += off2.x * clip.w;
  outClip.y += off2.y * U.eyeAspect.w * clip.w;
  out.clip = outClip;
  out.across = corner.y;
  out.speedFrac = speedFrac;
  out.viewDist = length(center - U.eyeAspect.xyz);
  return out;
}

const CYAN : vec3<f32> = vec3<f32>(0.30, 0.85, 1.0);
const WHITE : vec3<f32> = vec3<f32>(0.85, 0.97, 1.0);

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  if (in.vis <= 0.001) { discard; }
  // perpendicular soft falloff → rounded streak; length fade → bright head, dissolving tail.
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
