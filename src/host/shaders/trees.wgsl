// trees.wgsl — Forest trees render pass. Anchors each vertex to its tree's ground height, which was
// computed ONCE per tree by the trees_ground.wgsl prepass (same fBm + f32 precision as the rendered
// terrain) and stored in `grounds`. Vertex carries world XZ + local height offset + its tree index.
// Responsibilities:
//   - worldPos = (worldX, grounds[treeId] + offY, worldZ).
//   - Per-vertex HDR color × exp distance fog × radial fade (1 near → 0 at fadeEnd) so trees fade in/out
//     with the terrain instead of popping at the streaming-window rim. Additive blend → feeds the bloom.

struct U {
  viewProj: mat4x4<f32>,
  eye: vec3<f32>,
  fogDensity: f32,
  fadeStart: f32,
  fadeEnd: f32,
  depthBias: f32, // metres each vertex is pulled toward the eye before projecting (draw-on-top)
  time: f32,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> grounds: array<f32>; // per-tree ground height (from the prepass)

struct VSOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) color: vec3<f32>,
};

@vertex
fn vs(@location(0) wxyz: vec3<f32>, @location(1) treeId: f32, @location(2) col: vec3<f32>) -> VSOut {
  // wxyz = (worldX, worldZ, localOffsetY); ground comes from the per-tree prepass buffer.
  var o: VSOut;
  let ground = grounds[u32(treeId)];
  let pos = vec3<f32>(wxyz.x, ground + wxyz.z, wxyz.y);
  // DRAW-ON-TOP: pull the vertex toward the eye by depthBias metres before projecting so the tree sits ON
  // TOP of the ridge it stands on (kills the coincident-depth z-fight ripple); a ridge genuinely closer
  // than depthBias still occludes it. Fog/fade below use the TRUE position so distance shading is unchanged.
  let toEye = normalize(u.eye - pos);
  o.clip = u.viewProj * vec4<f32>(pos + toEye * u.depthBias, 1.0);
  let dist = distance(pos, u.eye);
  let fog = exp(-dist * u.fogDensity);
  let fade = clamp((u.fadeEnd - dist) / max(u.fadeEnd - u.fadeStart, 1.0), 0.0, 1.0);
  o.color = col * fog * fade;
  return o;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  return vec4<f32>(in.color, 1.0);
}
