// shift.wgsl — scroll a bordered (W+2)*(H+2) field by an integer cell offset (world-pinned recenter).
// Responsibilities:
//   - dst[i,j] = src[i - dx, j - dz] for interior cells whose source falls inside the interior
//     (the overlapping region is copied 1:1 — no resample, so the existing flow scrolls with NO seam).
//   - Cells whose source falls OUTSIDE the interior (the freshly-exposed leading edge) are seeded by
//     CLAMP-EXTENDING the nearest interior column/row of src (continuous extrapolation, not zero →
//     no hard edge/pop). The caller re-forces these fresh cells from the terrain after the shift.
//   - Border cells (i or j outside 1..W) are left for set_bnd to refill; we only write the interior.
//   - Reads src, writes dst (a separate buffer — the ping-pong .next half — so the copy is race-free).

struct ShiftParams { w : u32, h : u32, dx : i32, dz : i32 };

@group(0) @binding(0) var<uniform>             SP  : ShiftParams;
@group(0) @binding(1) var<storage, read>       src : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;

fn idx(i : u32, j : u32) -> u32 { return i + (SP.w + 2u) * j; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let stride = SP.w + 2u;
  let g = gid.x;
  if (g >= stride * (SP.h + 2u)) { return; }
  let i = g % stride;
  let j = g / stride;
  if (i < 1u || i > SP.w || j < 1u || j > SP.h) { return; } // interior only

  // source interior coord = dst coord - shift, clamped to the interior (clamp-extend the edges).
  let si = clamp(i32(i) - SP.dx, 1, i32(SP.w));
  let sj = clamp(i32(j) - SP.dz, 1, i32(SP.h));
  dst[idx(i, j)] = src[idx(u32(si), u32(sj))];
}
