// set_bnd.wgsl — Stam boundary conditions as TWO passes (edges, then corners).
// Port of crates/vs-core/src/fluid/boundary.rs::set_bnd, split to avoid the corner race:
// corners read freshly-written edge cells, so edges MUST complete (separate dispatch) first.
// Responsibilities:
//   - 6 entry points: {scalar,velx,vely} x {edges,corners}, one buffer `g` (in-place safe).
//   - edges: left/right walls (negate normal x for velx), bottom/top walls (negate normal y for vely).
//   - corners: mean of the two adjacent edge cells (identical for all kinds).
//   - Flat 1D dispatch over (W+2)*(H+2); each thread owns one border cell, interior threads no-op.

struct Params {
  w : u32, h : u32, dt : f32, pad0 : f32,
  fx : f32, fy : f32, dye_x : f32, dye_y : f32,
  dye_r : f32, dye_amt : f32, force_r : f32, pad1 : f32,
};

@group(0) @binding(0) var<uniform>             P : Params;
@group(0) @binding(1) var<storage, read_write> g : array<f32>;

fn idx(i : u32, j : u32) -> u32 { return i + (P.w + 2u) * j; }

// --- Edges: negate the normal component on the two walls selected by (neg_x, neg_y) ---
fn edges(gid : u32, neg_x : bool, neg_y : bool) {
  let w = P.w;
  let h = P.h;
  let stride = w + 2u;
  if (gid >= stride * (h + 2u)) { return; }
  let i = gid % stride;
  let j = gid / stride;

  // Left/right walls: interior rows 1..=h.
  if (j >= 1u && j <= h) {
    if (i == 0u) {
      let left = g[idx(1u, j)];
      g[idx(0u, j)] = select(left, -left, neg_x);
      return;
    }
    if (i == w + 1u) {
      let right = g[idx(w, j)];
      g[idx(w + 1u, j)] = select(right, -right, neg_x);
      return;
    }
  }
  // Bottom/top walls: interior columns 1..=w.
  if (i >= 1u && i <= w) {
    if (j == 0u) {
      let bottom = g[idx(i, 1u)];
      g[idx(i, 0u)] = select(bottom, -bottom, neg_y);
      return;
    }
    if (j == h + 1u) {
      let top = g[idx(i, h)];
      g[idx(i, h + 1u)] = select(top, -top, neg_y);
      return;
    }
  }
}

// --- Corners: mean of the two adjacent edge cells (run after edges complete) ---
fn corners(gid : u32) {
  let w = P.w;
  let h = P.h;
  let stride = w + 2u;
  if (gid >= stride * (h + 2u)) { return; }
  let i = gid % stride;
  let j = gid / stride;

  if (i == 0u && j == 0u) {
    g[idx(0u, 0u)] = 0.5 * (g[idx(1u, 0u)] + g[idx(0u, 1u)]);
  } else if (i == 0u && j == h + 1u) {
    g[idx(0u, h + 1u)] = 0.5 * (g[idx(1u, h + 1u)] + g[idx(0u, h)]);
  } else if (i == w + 1u && j == 0u) {
    g[idx(w + 1u, 0u)] = 0.5 * (g[idx(w, 0u)] + g[idx(w + 1u, 1u)]);
  } else if (i == w + 1u && j == h + 1u) {
    g[idx(w + 1u, h + 1u)] = 0.5 * (g[idx(w, h + 1u)] + g[idx(w + 1u, h)]);
  }
}

@compute @workgroup_size(64)
fn scalar_edges(@builtin(global_invocation_id) gid : vec3<u32>) { edges(gid.x, false, false); }
@compute @workgroup_size(64)
fn velx_edges(@builtin(global_invocation_id) gid : vec3<u32>)   { edges(gid.x, true,  false); }
@compute @workgroup_size(64)
fn vely_edges(@builtin(global_invocation_id) gid : vec3<u32>)   { edges(gid.x, false, true);  }

@compute @workgroup_size(64)
fn scalar_corners(@builtin(global_invocation_id) gid : vec3<u32>) { corners(gid.x); }
@compute @workgroup_size(64)
fn velx_corners(@builtin(global_invocation_id) gid : vec3<u32>)   { corners(gid.x); }
@compute @workgroup_size(64)
fn vely_corners(@builtin(global_invocation_id) gid : vec3<u32>)   { corners(gid.x); }
