// jacobi.wgsl — one Jacobi sweep of the pressure Poisson solve (ping-pong).
// Port of the inner sweep in crates/vs-core/src/fluid/project.rs::project.
// Responsibilities:
//   - p_next[i,j] = 0.25*(p[i-1,j]+p[i+1,j]+p[i,j-1]+p[i,j+1] - div[i,j]) (interior only).
//   - Reads the PREVIOUS pressure buffer, writes the NEXT (caller swaps each sweep).
//   - Flat 1D dispatch; derive (i,j), guard interior; same sign/algorithm as the oracle.

struct Params {
  w : u32, h : u32, dt : f32, pad0 : f32,
  fx : f32, fy : f32, dye_x : f32, dye_y : f32,
  dye_r : f32, dye_amt : f32, force_r : f32, pad1 : f32,
};

@group(0) @binding(0) var<uniform>             P      : Params;
@group(0) @binding(1) var<storage, read>       p      : array<f32>;
@group(0) @binding(2) var<storage, read>       div    : array<f32>;
@group(0) @binding(3) var<storage, read_write> p_next : array<f32>;

fn idx(i : u32, j : u32) -> u32 { return i + (P.w + 2u) * j; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let stride = P.w + 2u;
  let g = gid.x;
  if (g >= stride * (P.h + 2u)) { return; }
  let i = g % stride;
  let j = g / stride;
  if (i < 1u || i > P.w || j < 1u || j > P.h) { return; }

  let sum = p[idx(i - 1u, j)] + p[idx(i + 1u, j)]
          + p[idx(i, j - 1u)] + p[idx(i, j + 1u)];
  p_next[idx(i, j)] = 0.25 * (sum - div[idx(i, j)]);
}
