// forces.wgsl — add scripted velocity force + dye injection (interior, in-place).
// Generalizes the add-forces loop in crates/vs-core/src/fluid/solver.rs::Fluid2D::step
// (u += dt*force_x, v += dt*force_y) to a localized scripted source for the live viz.
// Responsibilities:
//   - Velocity: u += dt*fx, v += dt*fy within radius force_r of (dye_x, dye_y) [whole interior if force_r<=0].
//   - Dye: dye += dt*dye_amt within radius dye_r of (dye_x, dye_y).
//   - Interior only (1..=w, 1..=h); reads/writes own cell of u,v,dye — in-place safe.
//   - Flat 1D dispatch; derive (i,j), guard interior.

struct Params {
  w : u32, h : u32, dt : f32, pad0 : f32,
  fx : f32, fy : f32, dye_x : f32, dye_y : f32,
  dye_r : f32, dye_amt : f32, force_r : f32, pad1 : f32,
};

@group(0) @binding(0) var<uniform>             P   : Params;
@group(0) @binding(1) var<storage, read_write> u   : array<f32>;
@group(0) @binding(2) var<storage, read_write> v   : array<f32>;
@group(0) @binding(3) var<storage, read_write> dye : array<f32>;

fn idx(i : u32, j : u32) -> u32 { return i + (P.w + 2u) * j; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let stride = P.w + 2u;
  let g = gid.x;
  if (g >= stride * (P.h + 2u)) { return; }
  let i = g % stride;
  let j = g / stride;
  if (i < 1u || i > P.w || j < 1u || j > P.h) { return; }

  let dx = f32(i) - P.dye_x;
  let dy = f32(j) - P.dye_y;
  let r2 = dx * dx + dy * dy;
  let k = idx(i, j);

  // Velocity force: whole interior when force_r <= 0, else only within force_r.
  if (P.force_r <= 0.0 || r2 <= P.force_r * P.force_r) {
    u[k] = u[k] + P.dt * P.fx;
    v[k] = v[k] + P.dt * P.fy;
  }

  // Dye injection within dye_r.
  if (P.dye_r > 0.0 && r2 <= P.dye_r * P.dye_r) {
    dye[k] = dye[k] + P.dt * P.dye_amt;
  }
}
