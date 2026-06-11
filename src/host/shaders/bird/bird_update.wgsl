// bird_update.wgsl — GPU-integrated single-bird physics over the live fluid wind field.
// One compute pass per frame; reads the live fluid u,v storage buffers in-shader (no readback).
// Responsibilities:
//   - Read bird state {pos:vec2, vel:vec2} + per-frame Intent {impulse, turn} + fluid u,v + BirdParams.
//   - windAt(pos): manual bilinear sample of (u,v) with TOROIDAL wrap (NOT advect's Stam clamp).
//   - Integrate: vel += windAt(pos)*windCoupling*dt + impulse; rotate vel by turn; vel *= drag;
//     pos += vel*dt; toroidal-wrap pos into [0,W)x[0,H).
//   - Write current pos into the trail ring buffer at trailWrite. Single bird: only gid.x==0 runs.

struct BirdState {
  pos : vec2<f32>,
  vel : vec2<f32>,
};

// Per-frame input mapped from the active control scheme on the CPU. impulse is a one-shot
// velocity burst (zeroed by the host the frame after it fires); turn rotates vel (radians).
struct Intent {
  impulse : vec2<f32>,
  turn    : f32,
  pad0    : f32,
};

// w,h = fluid grid (interior cells); the bird lives in world coords [0,w)x[0,h).
struct BirdParams {
  w           : u32,
  h           : u32,
  dt          : f32,
  windCoupling: f32,
  drag        : f32,
  trailWrite  : u32,
  pad0        : f32,
  pad1        : f32,
};

@group(0) @binding(0) var<uniform>             P     : BirdParams;
@group(0) @binding(1) var<uniform>             I     : Intent;
@group(0) @binding(2) var<storage, read>       u     : array<f32>;
@group(0) @binding(3) var<storage, read>       v     : array<f32>;
@group(0) @binding(4) var<storage, read_write> bird  : array<BirdState>;
@group(0) @binding(5) var<storage, read_write> trail : array<vec2<f32>>;

// Bordered (W+2)*(H+2) layout, interior cell (i in 1..=w, j in 1..=h) at idx = i + (W+2)*j.
fn cell(i : u32, j : u32) -> u32 { return i + (P.w + 2u) * j; }

fn wrapI(i : i32, n : i32) -> u32 { return u32(((i % n) + n) % n); }

// Bilinear sample of the velocity field at world (x,y) with toroidal wrap over the interior grid.
// World coords are cell-centered: world x in [0,w) maps to interior column index (1 + floor(x)).
fn windAt(pos : vec2<f32>) -> vec2<f32> {
  let wf = f32(P.w);
  let hf = f32(P.h);
  // Wrap world position into [0,w)x[0,h).
  let px = pos.x - floor(pos.x / wf) * wf;
  let py = pos.y - floor(pos.y / hf) * hf;

  // Cell-center reference: sample between centers at (col 0.5 .. w-0.5).
  let fx = px - 0.5;
  let fy = py - 0.5;
  let i0 = i32(floor(fx));
  let j0 = i32(floor(fy));
  let s1 = fx - f32(i0);
  let t1 = fy - f32(j0);
  let s0 = 1.0 - s1;
  let t0 = 1.0 - t1;

  let wi = i32(P.w);
  let hi = i32(P.h);
  // +1 offset moves interior column 0 to bordered index 1.
  let ia = 1u + wrapI(i0,     wi);
  let ib = 1u + wrapI(i0 + 1, wi);
  let ja = 1u + wrapI(j0,     hi);
  let jb = 1u + wrapI(j0 + 1, hi);

  let uu = s0 * (t0 * u[cell(ia, ja)] + t1 * u[cell(ia, jb)])
         + s1 * (t0 * u[cell(ib, ja)] + t1 * u[cell(ib, jb)]);
  let vv = s0 * (t0 * v[cell(ia, ja)] + t1 * v[cell(ia, jb)])
         + s1 * (t0 * v[cell(ib, ja)] + t1 * v[cell(ib, jb)]);
  return vec2<f32>(uu, vv);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x > 0u) { return; } // single bird

  var s = bird[0];

  // Wind push.
  s.vel = s.vel + windAt(s.pos) * P.windCoupling * P.dt;
  // Input burst (one-shot; host zeroes impulse the next frame).
  s.vel = s.vel + I.impulse;

  // Turn: rotate vel by I.turn (radians).
  let c = cos(I.turn);
  let sn = sin(I.turn);
  s.vel = vec2<f32>(c * s.vel.x - sn * s.vel.y, sn * s.vel.x + c * s.vel.y);

  // Drag, integrate.
  s.vel = s.vel * P.drag;
  s.pos = s.pos + s.vel * P.dt;

  // Toroidal wrap into [0,w)x[0,h).
  let wf = f32(P.w);
  let hf = f32(P.h);
  s.pos.x = s.pos.x - floor(s.pos.x / wf) * wf;
  s.pos.y = s.pos.y - floor(s.pos.y / hf) * hf;

  bird[0] = s;
  trail[P.trailWrite] = s.pos;
}
