// terrain-selfcheck.ts — GPU-vs-CPU terrain-height equality probe.
// One-line description: evaluate the terrain fBm on the GPU at many world points and diff it
// against the CPU sampleHeight twin, proving whether the two fields actually agree at runtime.
// Responsibilities:
//   - Carry a FRESH copy of the integer-hash fBm (identical to terrain_ekg.wgsl / terrain.ts).
//   - Dispatch a compute pass over a set of world XZ points, read the heights back to the CPU.
//   - Compare against the supplied CPU height fn and report max/mean disagreement + worst point.
//   - Pure diagnostic: allocates its own buffers, touches no scene state, safe to call anytime.

// WGSL constants below MUST match terrain.ts / terrain_ekg.wgsl. If the scene shader is stale,
// this probe (freshly compiled) will agree with the CPU while the SCENE still disagrees → stale build.
const PROBE_WGSL = /* wgsl */ `
const BASE_FREQ : f32 = 0.00142857;
const LACUNARITY : f32 = 2.0;
const GAIN : f32 = 0.5;
const OCTAVES : i32 = 4;
const RELIEF : f32 = 600.0;
const SHARP : f32 = 1.8;
const TERRACES : f32 = 5.0;
const RISER_POW : f32 = 4.0;
const CLIFF_MIX : f32 = 0.65;

fn ihash(c : vec2<i32>) -> f32 {
  var h : u32 = bitcast<u32>(c.x) * 374761393u + bitcast<u32>(c.y) * 668265263u;
  h = (h ^ (h >> 13u)) * 1274126177u;
  h = h ^ (h >> 16u);
  return f32(h) / 4294967295.0;
}
fn valueNoise(p : vec2<f32>) -> f32 {
  let pf = floor(p);
  let ci = vec2<i32>(pf);
  let f = p - pf;
  let a = ihash(ci);
  let b = ihash(ci + vec2<i32>(1, 0));
  let c = ihash(ci + vec2<i32>(0, 1));
  let d = ihash(ci + vec2<i32>(1, 1));
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
fn fbm(p : vec2<f32>) -> f32 {
  var freq = BASE_FREQ;
  var amp = 1.0;
  var sum = 0.0;
  var norm = 0.0;
  for (var k = 0; k < OCTAVES; k = k + 1) {
    let n = valueNoise(p * freq);
    let r = 1.0 - abs(2.0 * n - 1.0);
    sum = sum + amp * r;
    norm = norm + amp;
    freq = freq * LACUNARITY;
    amp = amp * GAIN;
  }
  let s = pow(sum / norm, SHARP);
  let b = s * TERRACES;
  let fb = b - floor(b);
  let ter = floor(b) / TERRACES + pow(fb, RISER_POW) / TERRACES;
  return (s + (ter - s) * CLIFF_MIX) * RELIEF;
}

@group(0) @binding(0) var<storage, read> pts : array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> outH : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&pts)) { return; }
  outH[i] = fbm(pts[i]);
}
`;

export interface TerrainCheckResult {
  points: number;
  maxDiff: number;
  meanDiff: number;
  worst: { x: number; z: number; cpu: number; gpu: number };
  verdict: string;
}

// Run the GPU fBm at a spread of world points and diff against the CPU twin.
export async function checkTerrain(
  device: GPUDevice,
  cpuHeight: (x: number, z: number) => number,
): Promise<TerrainCheckResult> {
  // Sample spread: a grid out to ±3000 m plus a few far points — covers near-spawn and the far
  // field where any divergence grows. Deterministic, no RNG (so the report is reproducible).
  const pts: Array<[number, number]> = [];
  for (let gx = -3000; gx <= 3000; gx += 250) {
    for (let gz = -3000; gz <= 3000; gz += 250) pts.push([gx, gz]);
  }
  for (const far of [4000, -4000, 8000, -8000]) { pts.push([far, far]); pts.push([far, -far]); }
  const N = pts.length;

  const ptData = new Float32Array(N * 2);
  for (let i = 0; i < N; i++) { const p = pts[i]!; ptData[i * 2] = p[0]; ptData[i * 2 + 1] = p[1]; }

  const ptBuf = device.createBuffer({ size: ptData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(ptBuf, 0, ptData);
  const outBuf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const module = device.createShaderModule({ code: PROBE_WGSL });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: ptBuf } },
      { binding: 1, resource: { buffer: outBuf } },
    ],
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(N / 64));
  pass.end();
  enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, N * 4);
  device.queue.submit([enc.finish()]);

  await readBuf.mapAsync(GPUMapMode.READ);
  const gpu = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();

  let maxDiff = 0, sumDiff = 0;
  let worst = { x: 0, z: 0, cpu: 0, gpu: 0 };
  for (let i = 0; i < N; i++) {
    const p = pts[i]!;
    const gh = gpu[i]!;
    const cpu = cpuHeight(p[0], p[1]);
    const d = Math.abs(cpu - gh);
    sumDiff += d;
    if (d > maxDiff) { maxDiff = d; worst = { x: p[0], z: p[1], cpu, gpu: gh }; }
  }
  ptBuf.destroy(); outBuf.destroy(); readBuf.destroy();

  const meanDiff = sumDiff / N;
  // RELIEF is 600 m; agreement should be sub-meter. A multi-hundred-metre max = fields disagree.
  const verdict =
    maxDiff < 1
      ? "PASS — GPU and CPU terrain agree (<1 m). If the bird still crashes in clear sky, the SCENE shader is STALE: hard-reload (Cmd+Shift+R) or restart the dev server."
      : `FAIL — GPU and CPU terrain DISAGREE by up to ${maxDiff.toFixed(0)} m. The integer hash is not matching across CPU/GPU; the hash needs to change (e.g. avoid f32(u32) precision loss).`;
  return { points: N, maxDiff, meanDiff, worst, verdict };
}
