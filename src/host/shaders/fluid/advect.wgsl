// advect.wgsl — semi-Lagrangian transport with MANUAL bilinear (matches grid.rs sample).
// Port of crates/vs-core/src/fluid/advect.rs::advect (+ grid.rs::sample for the bilinear).
// Responsibilities:
//   - Backtrace (x,y) = (i - dt*u[i,j], j - dt*v[i,j]); bilinearly sample `src` at (x,y) -> dst.
//   - Manual bilinear replicates grid.rs::sample VERBATIM: Stam clamp [0.5, w+0.5] x [0.5, h+0.5],
//     i0=floor(x), same s0/s1/t0/t1 weights, same blend-expression order (f32 throughout).
//   - Interior only (1..=w, 1..=h); reads `src` (separate buffer) -> ping-pong; border untouched.

struct Params {
  w : u32, h : u32, dt : f32, pad0 : f32,
  fx : f32, fy : f32, dye_x : f32, dye_y : f32,
  dye_r : f32, dye_amt : f32, force_r : f32, pad1 : f32,
};

@group(0) @binding(0) var<uniform>             P   : Params;
@group(0) @binding(1) var<storage, read>       src : array<f32>;
@group(0) @binding(2) var<storage, read>       u   : array<f32>;
@group(0) @binding(3) var<storage, read>       v   : array<f32>;
@group(0) @binding(4) var<storage, read_write> dst : array<f32>;

fn idx(i : u32, j : u32) -> u32 { return i + (P.w + 2u) * j; }

// Bilinear sample of `src` at continuous (x,y), matching grid.rs::sample exactly.
fn sample(x_in : f32, y_in : f32) -> f32 {
  let wf = f32(P.w);
  let hf = f32(P.h);
  // Stam clamp to [0.5, w+0.5] x [0.5, h+0.5] so floor()+1 taps stay in-bounds.
  let x = clamp(x_in, 0.5, wf + 0.5);
  let y = clamp(y_in, 0.5, hf + 0.5);

  let i0 = u32(floor(x));
  let j0 = u32(floor(y));
  let i1 = i0 + 1u;
  let j1 = j0 + 1u;
  let s1 = x - f32(i0);
  let s0 = 1.0 - s1;
  let t1 = y - f32(j0);
  let t0 = 1.0 - t1;

  return s0 * (t0 * src[idx(i0, j0)] + t1 * src[idx(i0, j1)])
       + s1 * (t0 * src[idx(i1, j0)] + t1 * src[idx(i1, j1)]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let stride = P.w + 2u;
  let g = gid.x;
  if (g >= stride * (P.h + 2u)) { return; }
  let i = g % stride;
  let j = g / stride;
  if (i < 1u || i > P.w || j < 1u || j > P.h) { return; }

  let x = f32(i) - P.dt * u[idx(i, j)];
  let y = f32(j) - P.dt * v[idx(i, j)];
  dst[idx(i, j)] = sample(x, y);
}
