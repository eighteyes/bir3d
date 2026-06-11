// terrain.ts — Terrain3D: static centered grid mesh + neon-ridgeline render pipeline (depth-tested).
// Responsibilities:
//   - Build an N x N grid of vertices (gridXZ in world meters, centered on the camera) + a uint32
//     triangle index buffer (N>256 so 16-bit indices would overflow — uint32 is mandatory).
//   - Create the render pipeline from terrain3d.wgsl WITH depthStencil (depth24plus, less, write on).
//   - Own the uniform buffer (viewProj, camOffset, fog, eye) and upload it per draw.
//   - sampleHeight(x,z): TS mirror of the WGSL fBm (same constants/hash) for the bird (later task).
//   - draw(encoder, view, depthView, viewProj, camOffset, eye): record one indexed draw.

const BASE_FREQ = 0.00285714; // ~1/350 per meter (mirrors WGSL)
const LACUNARITY = 2.0;
const GAIN = 0.5;
const OCTAVES = 5;
const RELIEF = 120.0;

export interface TerrainParams {
  n?: number;          // grid resolution (verts per side)
  cellSize?: number;   // world meters per cell
  fogColor?: [number, number, number];
  fogDensity?: number;
}

export class Terrain3D {
  readonly n: number;
  readonly cellSize: number;
  readonly worldSpan: number; // total grid extent in meters

  private vbuf: GPUBuffer;
  private ibuf: GPUBuffer;
  private indexCount: number;
  private ubuf: GPUBuffer;
  private pipeline: GPURenderPipeline;
  private bindGroup: GPUBindGroup;
  private uniformHost: ArrayBuffer;
  private uniformData: Float32Array;

  private fogColor: [number, number, number];
  private fogDensity: number;

  constructor(
    private device: GPUDevice,
    shader: string,
    colorFormat: GPUTextureFormat,
    p: TerrainParams = {}
  ) {
    this.n = p.n ?? 201;
    this.cellSize = p.cellSize ?? 24; // 201*24 ≈ 4.8km window — spans many 350m wavelengths
    this.worldSpan = (this.n - 1) * this.cellSize;
    this.fogColor = p.fogColor ?? [0.02, 0.03, 0.07];
    this.fogDensity = p.fogDensity ?? 1 / 1800;

    // --- centered grid vertices: gridXZ in [-span/2, span/2] meters ---
    const n = this.n;
    const verts = new Float32Array(n * n * 2);
    const half = this.worldSpan / 2;
    let vi = 0;
    for (let row = 0; row < n; row++) {
      const z = row * this.cellSize - half;
      for (let col = 0; col < n; col++) {
        const x = col * this.cellSize - half;
        verts[vi++] = x;
        verts[vi++] = z;
      }
    }
    this.vbuf = device.createBuffer({
      size: verts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.vbuf, 0, verts);

    // --- uint32 index buffer (two triangles per cell) ---
    const quads = (n - 1) * (n - 1);
    const indices = new Uint32Array(quads * 6);
    let ii = 0;
    for (let row = 0; row < n - 1; row++) {
      for (let col = 0; col < n - 1; col++) {
        const a = row * n + col;
        const b = a + 1;
        const c = a + n;
        const d = c + 1;
        indices[ii++] = a; indices[ii++] = c; indices[ii++] = b;
        indices[ii++] = b; indices[ii++] = c; indices[ii++] = d;
      }
    }
    this.indexCount = indices.length;
    this.ibuf = device.createBuffer({
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.ibuf, 0, indices);

    // --- uniform buffer: mat4(64) + camOffset(8)+pad(8) + fogColor(12)+pad(4) + eye(12)+density(4) = 112 ---
    this.uniformHost = new ArrayBuffer(28 * 4);
    this.uniformData = new Float32Array(this.uniformHost);
    this.ubuf = device.createBuffer({
      size: this.uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // --- pipeline with depthStencil ---
    const module = device.createShaderModule({ code: shader });
    this.pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module,
        entryPoint: "vs",
        buffers: [
          {
            arrayStride: 2 * 4,
            attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
          },
        ],
      },
      fragment: { module, entryPoint: "fs", targets: [{ format: colorFormat }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
    });
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.ubuf } }],
    });
  }

  // TS mirror of the WGSL fBm. NOTE: sin()-hash diverges between f64 (JS) and f32 (GPU); this is the
  // bird-task's concern. Used here only as the CPU height reference; the render is 100% WGSL.
  private hash2(px: number, py: number): number {
    const s = Math.sin(px * 127.1 + py * 311.7) * 43758.5453;
    return s - Math.floor(s);
  }
  private valueNoise(px: number, py: number): number {
    const ix = Math.floor(px), iy = Math.floor(py);
    const fx = px - ix, fy = py - iy;
    const a = this.hash2(ix, iy);
    const b = this.hash2(ix + 1, iy);
    const c = this.hash2(ix, iy + 1);
    const d = this.hash2(ix + 1, iy + 1);
    const ux = fx * fx * (3 - 2 * fx);
    const uy = fy * fy * (3 - 2 * fy);
    const mx0 = a + (b - a) * ux;
    const mx1 = c + (d - c) * ux;
    return mx0 + (mx1 - mx0) * uy;
  }
  sampleHeight(x: number, z: number): number {
    let freq = BASE_FREQ, amp = 1, sum = 0, norm = 0;
    for (let k = 0; k < OCTAVES; k++) {
      sum += amp * this.valueNoise(x * freq, z * freq);
      norm += amp;
      freq *= LACUNARITY;
      amp *= GAIN;
    }
    return (sum / norm - 0.5) * RELIEF;
  }

  draw(
    encoder: GPUCommandEncoder,
    colorView: GPUTextureView,
    depthView: GPUTextureView,
    viewProj: Float32Array,
    camOffset: [number, number],
    eye: [number, number, number],
    clearColor: GPUColor
  ): void {
    // pack uniforms (std140-ish layout matching the WGSL struct)
    const u = this.uniformData;
    u.set(viewProj, 0);              // [0..16) mat4
    u[16] = camOffset[0]; u[17] = camOffset[1]; u[18] = 0; u[19] = 0; // camOffset + pad
    u[20] = this.fogColor[0]; u[21] = this.fogColor[1]; u[22] = this.fogColor[2]; u[23] = 0; // fogColor + pad
    u[24] = eye[0]; u[25] = eye[1]; u[26] = eye[2]; u[27] = this.fogDensity; // eye + density
    this.device.queue.writeBuffer(this.ubuf, 0, this.uniformHost);

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        { view: colorView, loadOp: "clear", storeOp: "store", clearValue: clearColor },
      ],
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vbuf);
    pass.setIndexBuffer(this.ibuf, "uint32");
    pass.drawIndexed(this.indexCount);
    pass.end();
  }
}
