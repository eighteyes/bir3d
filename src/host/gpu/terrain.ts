// terrain.ts — TerrainEKG: stacked horizontal neon trace LINES (no fill) + depth-tested pipeline.
// Responsibilities:
//   - Build a STACK of ~rows horizontal polylines (line-list). Each row is a fixed depth ahead of
//     the camera; each vertex is (xFrac in [-1,1], rowDepth in meters) + rowFade. Segments are
//     emitted as line-list pairs (no strip restart). There is NO filled surface — lines only.
//   - CAMERA-RELATIVE ROWS (v4): the WGSL builds each world sample as camGround + camFwd*depth +
//     camRight*(xFrac*halfWidth). Rows are perpendicular to camForward → screen-horizontal at every
//     heading (v3 locked rows to world-East, so turning skewed them diagonally; fixed here).
//   - Create the render pipeline from terrain_ekg.wgsl, topology "line-list", WITH depthStencil
//     (depth24plus, less, write on) so terrain lines occlude the bird, and additive blend for glow.
//   - Own the uniform buffer (viewProj, camGround, halfWidth, maxDist, camFwd, camRight, fog, eye);
//     upload per draw. maxDist gives a hard horizon cutoff in the fragment shader (clean horizon).
//   - sampleHeight(x,z): TS mirror of the WGSL fBm (same constants/hash) for the bird ridge-lift.
//   - draw(...): record one line-list draw that CLEARS color+depth (first pass of the frame).

const BASE_FREQ = 0.00285714; // ~1/350 per meter (mirrors WGSL)
const LACUNARITY = 2.0;
const GAIN = 0.5;
const OCTAVES = 3;
const RELIEF = 220.0;

export interface TerrainParams {
  rows?: number;       // number of stacked depth rows
  cols?: number;       // horizontal samples per row (polyline resolution)
  rowSpacing?: number; // world meters between rows (depth step)
  rowStart?: number;   // depth of the nearest row ahead of the camera (m)
  halfWidth?: number;  // half horizontal extent of each row (m)
  maxDist?: number;    // hard draw-distance cutoff (m) → clean horizon
  fogColor?: [number, number, number];
  fogDensity?: number;
}

export class TerrainEKG {
  readonly rows: number;
  readonly cols: number;
  readonly rowSpacing: number;
  readonly rowStart: number;
  readonly halfWidth: number;
  readonly maxDist: number;

  private vbuf: GPUBuffer;
  private vertexCount: number;
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
    this.rows = p.rows ?? 50;
    this.cols = p.cols ?? 256;
    this.rowSpacing = p.rowSpacing ?? 70;
    this.rowStart = p.rowStart ?? 40;
    this.halfWidth = p.halfWidth ?? 1400;
    // hard horizon cutoff. Default: just past the last row so the full stack draws.
    this.maxDist = p.maxDist ?? (this.rowStart + this.rows * this.rowSpacing);
    this.fogColor = p.fogColor ?? [0.01, 0.012, 0.03];
    this.fogDensity = p.fogDensity ?? 1 / 1600;

    // --- line-list vertices: per row, (cols-1) segments = 2 verts each. ---
    // each vertex: xFrac(1) + rowDepth(1) + rowFade(1) = 3 floats.
    const segsPerRow = this.cols - 1;
    this.vertexCount = this.rows * segsPerRow * 2;
    const FPV = 3;
    const verts = new Float32Array(this.vertexCount * FPV);
    let vi = 0;
    for (let r = 0; r < this.rows; r++) {
      const rowDepth = this.rowStart + r * this.rowSpacing;
      const rowFade = this.rows > 1 ? r / (this.rows - 1) : 0; // 0 near .. 1 far
      for (let c = 0; c < segsPerRow; c++) {
        const xA = (c / segsPerRow) * 2 - 1;       // [-1,1]
        const xB = ((c + 1) / segsPerRow) * 2 - 1;
        verts[vi++] = xA; verts[vi++] = rowDepth; verts[vi++] = rowFade;
        verts[vi++] = xB; verts[vi++] = rowDepth; verts[vi++] = rowFade;
      }
    }
    this.vbuf = device.createBuffer({
      size: verts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.vbuf, 0, verts);

    // --- uniform buffer (std140-aligned, 128 bytes / 32 floats):
    //   [0..16)  mat4 viewProj
    //   [16,17]  camGround.xz   [18] halfWidth   [19] maxDist
    //   [20,21]  camFwd.xz      [22,23] camRight.xz
    //   [24..27) fogColor.rgb   [27] fogDensity
    //   [28..31) eye.xyz        [31] pad
    this.uniformHost = new ArrayBuffer(32 * 4);
    this.uniformData = new Float32Array(this.uniformHost);
    this.ubuf = device.createBuffer({
      size: this.uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // --- pipeline: line-list, depthStencil, additive blend for neon glow ---
    const module = device.createShaderModule({ code: shader });
    this.pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module,
        entryPoint: "vs",
        buffers: [
          {
            arrayStride: FPV * 4,
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x2" }, // xFrac, rowDepth
              { shaderLocation: 1, offset: 8, format: "float32" },   // rowFade
            ],
          },
        ],
      },
      fragment: {
        module,
        entryPoint: "fs",
        targets: [
          {
            format: colorFormat,
            blend: {
              color: { srcFactor: "one", dstFactor: "one", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "line-list", cullMode: "none" },
      depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
    });
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.ubuf } }],
    });
  }

  // TS mirror of the WGSL fBm — CPU height reference for the bird ridge-lift.
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
      const n = this.valueNoise(x * freq, z * freq);
      const r = 1 - Math.abs(2 * n - 1);
      sum += amp * r;
      norm += amp;
      freq *= LACUNARITY;
      amp *= GAIN;
    }
    return (sum / norm) * RELIEF;
  }

  draw(
    encoder: GPUCommandEncoder,
    colorView: GPUTextureView,
    depthView: GPUTextureView,
    viewProj: Float32Array,
    camGround: [number, number],            // camera ground (x,z) the stack is built around
    camFwd: [number, number],               // horizontal camera forward (x,z), unit
    camRight: [number, number],             // horizontal camera right (x,z), unit
    eye: [number, number, number],
    clearColor: GPUColor
  ): void {
    const u = this.uniformData;
    u.set(viewProj, 0);                                              // [0..16) mat4
    u[16] = camGround[0]; u[17] = camGround[1];                      // camGround.xz
    u[18] = this.halfWidth; u[19] = this.maxDist;                   // halfWidth, maxDist
    u[20] = camFwd[0]; u[21] = camFwd[1];                            // camFwd.xz
    u[22] = camRight[0]; u[23] = camRight[1];                        // camRight.xz
    u[24] = this.fogColor[0]; u[25] = this.fogColor[1]; u[26] = this.fogColor[2]; u[27] = this.fogDensity;
    u[28] = eye[0]; u[29] = eye[1]; u[30] = eye[2]; u[31] = 0;       // eye + pad
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
    pass.draw(this.vertexCount);
    pass.end();
  }
}
