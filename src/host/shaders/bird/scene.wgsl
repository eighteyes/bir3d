// scene.wgsl — bird scene render passes (backdrop + trail + chevron), camera-relative, neon-on-dark.
// Rough feel-prototype glyphs (NOT the §4.1 vector renderer). Three entry pairs share one Camera
// uniform so all layers align in the same world→NDC mapping.
// Responsibilities:
//   - Camera uniform: cameraPos (world center), viewSize (world units across view), grid w,h.
//   - backdrop_vs/fs: fullscreen triangle; per-fragment map screen→world for the camera sub-window,
//     toroidal-wrap into the grid, sample dye magnitude → dim neon flow, brightness-capped (§7.2).
//   - trail_vs/fs: draw the trail ring buffer as a line strip, fading by recency (alpha ramp).
//   - chevron_vs/fs: read the bird state buffer in-shader, build a neon chevron oriented to vel,
//     drawn relative to the camera (triangle-list, 3 verts).

struct Camera {
  cameraPos : vec2<f32>, // world-space center of the view
  viewSize  : vec2<f32>, // world units spanned by the view (x,y)
  w         : u32,
  h         : u32,
  trailLen  : u32,
  trailHead : u32,       // index of the most-recently-written trail sample
};

struct BirdState {
  pos : vec2<f32>,
  vel : vec2<f32>,
};

const BRIGHTNESS_CEILING : f32 = 0.8;

fn capBrightness(col : vec3<f32>) -> vec3<f32> {
  let lum = dot(col, vec3<f32>(0.2126, 0.7152, 0.0722));
  if (lum > BRIGHTNESS_CEILING) { return col * (BRIGHTNESS_CEILING / lum); }
  return col;
}

// world → NDC given the camera window. y flips so +world-y is up on screen.
fn worldToNdc(world : vec2<f32>, cam : Camera) -> vec2<f32> {
  let rel = (world - cam.cameraPos) / (cam.viewSize * 0.5);
  return vec2<f32>(rel.x, -rel.y);
}

// ---------- backdrop ----------
@group(0) @binding(0) var<uniform>       CAM : Camera;
@group(0) @binding(1) var<storage, read> dye : array<f32>;

struct VsUv {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

@vertex
fn backdrop_vs(@builtin(vertex_index) vi : u32) -> VsUv {
  var out : VsUv;
  let x = f32((vi << 1u) & 2u);
  let y = f32(vi & 2u);
  out.pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2<f32>(x, y); // [0,2] clipped to [0,1] across the screen
  return out;
}

fn dyeCell(i : u32, j : u32) -> u32 { return i + (CAM.w + 2u) * j; }
fn wrapU(i : i32, n : i32) -> u32 { return u32(((i % n) + n) % n); }

@fragment
fn backdrop_fs(in : VsUv) -> @location(0) vec4<f32> {
  // Screen uv [0,1] → world within the camera window (uv.y up).
  let half = CAM.viewSize * 0.5;
  let world = CAM.cameraPos + vec2<f32>((in.uv.x - 0.5) * CAM.viewSize.x,
                                        (0.5 - in.uv.y) * CAM.viewSize.y);
  // Toroidal-wrap into the grid, nearest interior cell.
  let wf = f32(CAM.w);
  let hf = f32(CAM.h);
  let px = world.x - floor(world.x / wf) * wf;
  let py = world.y - floor(world.y / hf) * hf;
  let i = 1u + wrapU(i32(floor(px)), i32(CAM.w));
  let j = 1u + wrapU(i32(floor(py)), i32(CAM.h));
  let d = dye[dyeCell(i, j)];

  let t = clamp(1.0 - exp(-max(d, 0.0)), 0.0, 1.0);
  let base = vec3<f32>(0.015, 0.02, 0.03);
  let flow = vec3<f32>(0.05, 0.28, 0.45); // dim teal flow, dialed back from the debug blob
  var col = base + flow * (t * 0.6);
  return vec4<f32>(capBrightness(col), 1.0);
}

// ---------- trail ----------
@group(0) @binding(0) var<uniform>       CAM_T : Camera;
@group(0) @binding(1) var<storage, read> trail : array<vec2<f32>>;

struct VsTrail {
  @builtin(position) pos : vec4<f32>,
  @location(0) fade : f32,
};

@vertex
fn trail_vs(@builtin(vertex_index) vi : u32) -> VsTrail {
  var out : VsTrail;
  // vi runs oldest→newest along the strip; recency = how close to the head.
  let n = CAM_T.trailLen;
  // age 0 = newest (head), age n-1 = oldest.
  let idx = (CAM_T.trailHead + 1u + vi) % n; // start just after head = oldest
  let age = f32(n - 1u - vi); // 0 at the newest end
  var p = trail[idx];

  // Unwrap the sample to the camera's window so a seam crossing doesn't streak across screen.
  let half = CAM_T.viewSize * 0.5;
  let wf = f32(CAM_T.w);
  let hf = f32(CAM_T.h);
  var rel = p - CAM_T.cameraPos;
  rel.x = rel.x - round(rel.x / wf) * wf;
  rel.y = rel.y - round(rel.y / hf) * hf;
  let ndc = vec2<f32>(rel.x / half.x, -rel.y / half.y);
  out.pos = vec4<f32>(ndc, 0.0, 1.0);
  out.fade = 1.0 - age / f32(max(n - 1u, 1u)); // 1 newest → 0 oldest
  return out;
}

@fragment
fn trail_fs(in : VsTrail) -> @location(0) vec4<f32> {
  let neon = vec3<f32>(0.2, 1.0, 0.7);
  let a = clamp(in.fade, 0.0, 1.0);
  return vec4<f32>(capBrightness(neon * a), a * 0.8);
}

// ---------- chevron ----------
@group(0) @binding(0) var<uniform>       CAM_C : Camera;
@group(0) @binding(1) var<storage, read> bird  : array<BirdState>;

struct VsChev {
  @builtin(position) pos : vec4<f32>,
  @location(0) tip : f32,
};

@vertex
fn chevron_vs(@builtin(vertex_index) vi : u32) -> VsChev {
  var out : VsChev;
  let s = bird[0];
  // Heading from vel (fallback +x if nearly still).
  var dir = vec2<f32>(1.0, 0.0);
  let sp = length(s.vel);
  if (sp > 1e-4) { dir = s.vel / sp; }
  let perp = vec2<f32>(-dir.y, dir.x);

  // Chevron in world units (size in grid cells). Arrowhead: tip forward, two tails back.
  let size = 3.0;
  var local : vec2<f32>;
  if (vi == 0u) {
    local = dir * size;                          // tip
    out.tip = 1.0;
  } else if (vi == 1u) {
    local = -dir * size * 0.6 + perp * size * 0.7; // left tail
    out.tip = 0.0;
  } else {
    local = -dir * size * 0.6 - perp * size * 0.7; // right tail
    out.tip = 0.0;
  }
  let world = s.pos + local;

  // Camera-relative with seam unwrap (bird is near center; keep it stable across wraps).
  let wf = f32(CAM_C.w);
  let hf = f32(CAM_C.h);
  var rel = world - CAM_C.cameraPos;
  rel.x = rel.x - round(rel.x / wf) * wf;
  rel.y = rel.y - round(rel.y / hf) * hf;
  let half = CAM_C.viewSize * 0.5;
  out.pos = vec4<f32>(rel.x / half.x, -rel.y / half.y, 0.0, 1.0);
  return out;
}

@fragment
fn chevron_fs(in : VsChev) -> @location(0) vec4<f32> {
  // Bright neon toward the tip, slightly cooler at the tails.
  let neon = mix(vec3<f32>(0.3, 1.0, 0.9), vec3<f32>(0.9, 1.0, 0.4), in.tip);
  return vec4<f32>(capBrightness(neon), 1.0);
}
