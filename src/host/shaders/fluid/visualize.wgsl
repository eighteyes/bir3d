// visualize.wgsl — debug viz render pipeline: dye field → neon-green ramp on dark.
// Debug-grade visualization for the Plan-3 fluid spike (NOT the §4.1 neon renderer; that is Plan 4).
// Responsibilities:
//   - Fullscreen triangle (3 verts via vertex_index; no vertex buffer) covering the viewport.
//   - Fragment: map frag UV → interior fluid cell (1..=w, 1..=h) of the bordered (W+2)*(H+2) dye
//     buffer (idx = i + (W+2)*j), sample dye as read-only storage (read_write is compute-only).
//   - Map dye magnitude → neon-green ramp on a dark base; clamp output luminance to a brightness
//     ceiling (blueprint §7.2 photosensitivity) and keep the field continuous (no flashing).
//   - Reuses the shared Params uniform (w,h) so the host binds the same uniform buffer as compute.

struct Params {
  w : u32, h : u32, dt : f32, pad0 : f32,
  fx : f32, fy : f32, dye_x : f32, dye_y : f32,
  dye_r : f32, dye_amt : f32, force_r : f32, pad1 : f32,
};

@group(0) @binding(0) var<uniform>       P   : Params;
@group(0) @binding(1) var<storage, read> dye : array<f32>;

// §7.2 photosensitivity: cap output brightness so the debug viz never hits full-white flashes.
const BRIGHTNESS_CEILING : f32 = 0.8;

struct VsOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

// Fullscreen triangle: 3 oversized verts, clipped to the viewport. uv in [0,1] over the screen.
@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VsOut {
  var out : VsOut;
  let x = f32((vi << 1u) & 2u); // 0,2,0
  let y = f32(vi & 2u);         // 0,0,2
  out.pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2<f32>(x, 1.0 - y); // flip y: row 0 at the top
  return out;
}

@fragment
fn fs(in : VsOut) -> @location(0) vec4<f32> {
  let stride = P.w + 2u;
  // Map screen uv → interior cell index (1..=w, 1..=h).
  let i = 1u + min(P.w - 1u, u32(in.uv.x * f32(P.w)));
  let j = 1u + min(P.h - 1u, u32(in.uv.y * f32(P.h)));
  let d = dye[i + stride * j];

  // Dye magnitude → [0,1] intensity. Soft saturating curve so dense cores read distinct from wisps.
  let t = clamp(1.0 - exp(-max(d, 0.0)), 0.0, 1.0);

  // Neon-green ramp on a dark base: green dominant, faint cyan lift in the highlights.
  let base = vec3<f32>(0.02, 0.03, 0.04);
  let neon = vec3<f32>(0.10, 1.0, 0.45);
  var col = base + neon * t;

  // §7.2: clamp luminance to the brightness ceiling (preserve hue by scaling, not truncating).
  let lum = dot(col, vec3<f32>(0.2126, 0.7152, 0.0722));
  if (lum > BRIGHTNESS_CEILING) {
    col = col * (BRIGHTNESS_CEILING / lum);
  }
  return vec4<f32>(col, 1.0);
}
