// add_force_field.wgsl — add a per-cell force field to the velocity (interior, in-place).
// Exact port of the add-forces loop in crates/vs-core/src/fluid/solver.rs::Fluid2D::step
// (u += dt*force_x[i,j], v += dt*force_y[i,j]) using FULL per-cell force buffers, not the
// parametric/localized source in forces.wgsl. Used by the composed correctness gate so the
// GPU step mirrors Fluid2D::step's scripted per-cell force exactly.
// Responsibilities:
//   - u[i,j] += dt*fx[i,j];  v[i,j] += dt*fy[i,j] (interior 1..=w, 1..=h only).
//   - Reads its own cell of fx,fy,u,v; writes its own cell of u,v — in-place safe.
//   - Flat 1D dispatch over (W+2)*(H+2); derive (i,j), guard interior; border untouched.

struct Params {
  w : u32, h : u32, dt : f32, pad0 : f32,
  fx : f32, fy : f32, dye_x : f32, dye_y : f32,
  dye_r : f32, dye_amt : f32, force_r : f32, pad1 : f32,
};

@group(0) @binding(0) var<uniform>             P  : Params;
@group(0) @binding(1) var<storage, read>       fx : array<f32>;
@group(0) @binding(2) var<storage, read>       fy : array<f32>;
@group(0) @binding(3) var<storage, read_write> u  : array<f32>;
@group(0) @binding(4) var<storage, read_write> v  : array<f32>;

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
  u[k] = u[k] + P.dt * fx[k];
  v[k] = v[k] + P.dt * fy[k];
}
