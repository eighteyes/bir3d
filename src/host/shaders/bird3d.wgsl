// bird3d.wgsl — neon gliding-V bird (WebGPU 3D, depth-tested so terrain ridges occlude it).
// Responsibilities:
//   - Vertex: take a procedural bird mesh in LOCAL space (x=span lateral, y=0, z=chord forward).
//     Wings are held OUT (no flap cycle); a SUBTLE flex about the local forward (Z) axis via
//     sin(time*flexHz)*flexAmp (tiny) reads as a living glide, NOT a flap beat. Apply
//     model = T(pos) * Ryaw(heading) * Rroll(bank), then U.viewProj. Body verts do not flex.
//   - Fragment: bold emissive neon ribbons on the dark scene; brightness tapers along the
//     wing so tips glow hot. Depth-tested (less) against the stored terrain depth → occlusion.
//   - Local axes match the world chase convention: +Z = forward (heading), +X = right, +Y = up.

struct Uniforms {
  viewProj : mat4x4<f32>,
  pos : vec3<f32>,        // bird world position
  flexPhase : f32,        // time * flexHz (radians) — subtle living flex, NOT a flap beat
  heading : f32,          // yaw about +Y (atan2 forward.x, forward.z)
  bank : f32,             // roll about local +Z (banks into turns)
  flexHz : f32,           // unused in shader (kept for parity)
  flexAmp : f32,          // subtle flex angle (radians) — wings held OUT, no flap cycle
};

@group(0) @binding(0) var<uniform> U : Uniforms;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) glow : f32,     // 0 body .. 1 wingtip (emissive ramp)
  @location(1) edge : f32,     // ribbon-edge factor for line-like core
};

fn rotZ(a : f32) -> mat3x3<f32> {
  let c = cos(a); let s = sin(a);
  return mat3x3<f32>(vec3<f32>(c, s, 0.0), vec3<f32>(-s, c, 0.0), vec3<f32>(0.0, 0.0, 1.0));
}
fn rotY(a : f32) -> mat3x3<f32> {
  let c = cos(a); let s = sin(a);
  return mat3x3<f32>(vec3<f32>(c, 0.0, -s), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(s, 0.0, c));
}

// Per-vertex attributes:
//   loc0 = local position (span x, 0, chord z) in meters
//   loc1 = (spanFrac signed -1..1, partFlag: 0=body 1=wing, edgeFrac 0..1)
@vertex
fn vs(@location(0) local : vec3<f32>, @location(1) attr : vec3<f32>) -> VSOut {
  var out : VSOut;
  let spanFrac = attr.x;          // signed, -1 (left tip) .. +1 (right tip)
  let isWing = attr.y;            // 1 for wing verts, 0 for body
  let edgeFrac = attr.z;

  var p = local;

  // GLIDE, NO FLAP: wings held OUT. A SUBTLE flex about local forward (Z) reads as a living
  // glide, not a flap beat. Tip lags by phase ∝ |spanFrac| so the flex feels organic.
  if (isWing > 0.5) {
    let lag = abs(spanFrac) * 1.4;                 // wingtip phase offset
    let flex = sin(U.flexPhase - lag) * U.flexAmp; // tiny dihedral wobble (flexAmp small)
    // both wings flex together: rotate +span up, -span up → sign by side, magnitude by |span|.
    let ang = flex * sign(spanFrac);
    p = rotZ(ang) * p;
  }

  // Model: bank (roll about forward Z) → yaw (heading about Y) → translate.
  let rolled = rotZ(U.bank) * p;
  let yawed = rotY(U.heading) * rolled;
  let world = yawed + U.pos;

  out.clip = U.viewProj * vec4<f32>(world, 1.0);
  out.glow = abs(spanFrac);       // tips glow hottest
  out.edge = edgeFrac;
  return out;
}

const CORE : vec3<f32> = vec3<f32>(0.55, 1.0, 0.9);   // teal-white hot core
const TIP : vec3<f32>  = vec3<f32>(0.95, 0.35, 1.0);  // magenta tips

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  // Hot core → magenta tip ramp; ribbon center brighter than edges for a line-like spine.
  let tint = mix(CORE, TIP, in.glow);
  let spine = smoothstep(0.0, 0.5, 1.0 - abs(in.edge - 0.5) * 2.0); // 1 at ribbon center
  let bright = 0.85 + spine * 0.9 + in.glow * 0.4;
  var color = tint * bright;
  color = min(color, vec3<f32>(1.4, 1.5, 1.6)); // cap (additive blend will bloom it)
  return vec4<f32>(color, 1.0);
}
