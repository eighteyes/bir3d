// addone.wgsl — proves the compute pipeline end-to-end: out[i] = in[i] + 1.
@group(0) @binding(0) var<storage, read>       inBuf  : array<f32>;
@group(0) @binding(1) var<storage, read_write> outBuf : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&inBuf)) { return; }
  outBuf[i] = inBuf[i] + 1.0;
}
