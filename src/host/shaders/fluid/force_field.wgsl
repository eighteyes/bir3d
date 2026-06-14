// force_field.wgsl — add a PER-CELL velocity force field (terrain orographic coupling).
// Responsibilities:
//   - u[k] += dt * fx_field[k]; v[k] += dt * fy_field[k] for interior cells (1..W, 1..H).
//   - The force field is a world-pinned per-cell vector the caller computes from the terrain gradient
//     (deflect flow around/over high terrain; channel through valleys) so the fluid RESPONDS to the
//     real landscape. Complementary to forces.wgsl's scalar disc — applied the same step, in place.
//   - Flat 1D dispatch; derive (i,j), guard interior; reads/writes own cell only (in-place safe).

struct Params {
  w : u32, h : u32, dt : f32, pad0 : f32,
  fx : f32, fy : f32, dye_x : f32, dye_y : f32,
  dye_r : f32, dye_amt : f32, force_r : f32, pad1 : f32,
};

@group(0) @binding(0) var<uniform>             P        : Params;
@group(0) @binding(1) var<storage, read_write> u        : array<f32>;
@group(0) @binding(2) var<storage, read_write> v        : array<f32>;
@group(0) @binding(3) var<storage, read>       fx_field : array<f32>;
@group(0) @binding(4) var<storage, read>       fy_field : array<f32>;

fn idx(i : u32, j : u32) -> u32 { return i + (P.w + 2u) * j; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let stride = P.w + 2u;
  let g = gid.x;
  if (g >= stride * (P.h + 2u)) { return; }
  let i = g % stride;
  let j = g / stride;
  if (i < 1u || i > P.w || j < 1u || j > P.h) { return; }

  let k = idx(i, j);
  u[k] = u[k] + P.dt * fx_field[k];
  v[k] = v[k] + P.dt * fy_field[k];
}
