// divergence.wgsl — central-difference velocity divergence (Stam, h=1).
// Port of crates/vs-core/src/fluid/project.rs::divergence.
// Responsibilities:
//   - For each interior cell (1..=w, 1..=h): div = 0.5*(u[i+1,j]-u[i-1,j]) + 0.5*(v[i,j+1]-v[i,j-1]).
//   - Flat 1D dispatch over (W+2)*(H+2); derive (i,j), guard interior; border left untouched.
//   - In-place safe: writes only its own cell of `div` (a separate buffer from u,v).

struct Params {
  w : u32, h : u32, dt : f32, pad0 : f32,
  fx : f32, fy : f32, dye_x : f32, dye_y : f32,
  dye_r : f32, dye_amt : f32, force_r : f32, pad1 : f32,
};

@group(0) @binding(0) var<uniform>             P   : Params;
@group(0) @binding(1) var<storage, read>       u   : array<f32>;
@group(0) @binding(2) var<storage, read>       v   : array<f32>;
@group(0) @binding(3) var<storage, read_write> div : array<f32>;

fn idx(i : u32, j : u32) -> u32 { return i + (P.w + 2u) * j; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let stride = P.w + 2u;
  let g = gid.x;
  if (g >= stride * (P.h + 2u)) { return; }
  let i = g % stride;
  let j = g / stride;
  if (i < 1u || i > P.w || j < 1u || j > P.h) { return; }

  let d = 0.5 * (u[idx(i + 1u, j)] - u[idx(i - 1u, j)])
        + 0.5 * (v[idx(i, j + 1u)] - v[idx(i, j - 1u)]);
  div[idx(i, j)] = d;
}
