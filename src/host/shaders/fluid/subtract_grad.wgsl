// subtract_grad.wgsl — subtract the pressure gradient from the velocity field.
// Port of the gradient-subtraction loop in crates/vs-core/src/fluid/project.rs::project.
// Responsibilities:
//   - u[i,j] -= 0.5*(p[i+1,j]-p[i-1,j]);  v[i,j] -= 0.5*(p[i,j+1]-p[i,j-1]) (interior only).
//   - Reads pressure `p` (separate buffer) + own u,v cell; writes own u,v cell — in-place safe.
//   - Flat 1D dispatch; derive (i,j), guard interior. Caller applies set_bnd afterward.

struct Params {
  w : u32, h : u32, dt : f32, pad0 : f32,
  fx : f32, fy : f32, dye_x : f32, dye_y : f32,
  dye_r : f32, dye_amt : f32, force_r : f32, pad1 : f32,
};

@group(0) @binding(0) var<uniform>             P : Params;
@group(0) @binding(1) var<storage, read>       p : array<f32>;
@group(0) @binding(2) var<storage, read_write> u : array<f32>;
@group(0) @binding(3) var<storage, read_write> v : array<f32>;

fn idx(i : u32, j : u32) -> u32 { return i + (P.w + 2u) * j; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let stride = P.w + 2u;
  let g = gid.x;
  if (g >= stride * (P.h + 2u)) { return; }
  let i = g % stride;
  let j = g / stride;
  if (i < 1u || i > P.w || j < 1u || j > P.h) { return; }

  let gx = 0.5 * (p[idx(i + 1u, j)] - p[idx(i - 1u, j)]);
  let gy = 0.5 * (p[idx(i, j + 1u)] - p[idx(i, j - 1u)]);
  u[idx(i, j)] = u[idx(i, j)] - gx;
  v[idx(i, j)] = v[idx(i, j)] - gy;
}
