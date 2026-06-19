// terrain-grid.ts — GridTerrain: WORLD-STATIC wireframe terrain (alternative to the camera-relative EKG).
// Responsibilities:
//   - Build a square grid of (x,z) vertices snapped to WORLD coordinates within a radius of the camera;
//     emit a fill mesh (2 triangles/cell) AND grid line segments. Heights are draped in the shader (fBm),
//     so the lines are pinned in the world — flying forward, they flow toward you with real parallax.
//   - Stream the window: rebuild only when the camera crosses a grid cell (like the trees).
//   - draw(): one render pass — FILL first (dark, depthWrite ON → hides lines behind ridges), then the
//     elevation-ramped neon LINES (depth-tested, additive → bloom). First pass of the frame (clears).
//   - fBm matches terrain_ekg / trees exactly, so trees + bird sit on this surface too.

const FLOATS_PER_VERT = 2; // world x, z (height draped in the shader)
const UNIFORM_BYTES = 112; // mat4(64) + eye+fogDensity(16) + fogColor+maxDist(16) + topo params(16)

export interface GridParams {
  spacing?: number; // m between grid lines
  radius?: number; // m — window radius around the camera
  maxDist?: number; // m — fog/draw cutoff
  fogColor?: [number, number, number];
  fogDensity?: number;
  sampleCount?: number;
}

export class GridTerrain {
  mode: "grid" | "topo" = "grid"; // wireframe grid, or topographic contour lines
  // live-tunable topo params (written into the uniform each draw)
  interval = 22; // m between contour lines
  lineWidth = 1.3; // contour line width (screen-relative)
  floorFade = 0.28; // brightness at the valley floor (low)
  peakGain = 1.9; // brightness at the peaks (high)
  private device: GPUDevice;
  private spacing: number;
  private radius: number;
  private maxDist: number;
  private fogColor: [number, number, number];
  private fogDensity: number;

  private fillHost: Float32Array;
  private lineHost: Float32Array;
  private fillBuf: GPUBuffer;
  private lineBuf: GPUBuffer;
  private fillCount = 0;
  private lineCount = 0;

  private ubuf: GPUBuffer;
  private uniformHost = new ArrayBuffer(UNIFORM_BYTES);
  private u = new Float32Array(this.uniformHost);
  private fillPipeline: GPURenderPipeline;
  private topoPipeline: GPURenderPipeline;
  private linePipeline: GPURenderPipeline;
  private fillBind: GPUBindGroup;
  private topoBind: GPUBindGroup;
  private lineBind: GPUBindGroup;

  private lastCellX = Number.NaN;
  private lastCellZ = Number.NaN;

  constructor(device: GPUDevice, shader: string, colorFormat: GPUTextureFormat, p: GridParams = {}) {
    this.device = device;
    this.spacing = p.spacing ?? 24;
    this.radius = p.radius ?? 1500;
    this.maxDist = p.maxDist ?? 1500;
    this.fogColor = p.fogColor ?? [0.01, 0.012, 0.03];
    this.fogDensity = p.fogDensity ?? 0.5 / 1100;
    const sampleCount = p.sampleCount ?? 1;

    const side = 2 * Math.ceil(this.radius / this.spacing) + 1;
    const maxCells = side * side;
    this.fillHost = new Float32Array(maxCells * 6 * FLOATS_PER_VERT); // 2 tris/cell
    this.lineHost = new Float32Array(maxCells * 4 * FLOATS_PER_VERT); // 2 segments/cell
    this.fillBuf = device.createBuffer({ size: this.fillHost.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    this.lineBuf = device.createBuffer({ size: this.lineHost.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    this.ubuf = device.createBuffer({ size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const module = device.createShaderModule({ code: shader });
    const vbLayout = [{ arrayStride: FLOATS_PER_VERT * 4, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" as const }] }];
    this.fillPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vsFill", buffers: vbLayout },
      fragment: { module, entryPoint: "fsFill", targets: [{ format: colorFormat }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
      multisample: { count: sampleCount },
    });
    this.linePipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vsLine", buffers: vbLayout },
      fragment: {
        module, entryPoint: "fsLine",
        targets: [{ format: colorFormat, blend: { color: { srcFactor: "one", dstFactor: "one", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "one", operation: "add" } } }],
      },
      primitive: { topology: "line-list" },
      depthStencil: { depthWriteEnabled: false, depthCompare: "less-equal", format: "depth24plus" },
      multisample: { count: sampleCount },
    });
    // TOPO fill pipeline: same draped mesh, but the fragment draws elevation contour lines.
    this.topoPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vsFill", buffers: vbLayout },
      fragment: { module, entryPoint: "fsTopo", targets: [{ format: colorFormat }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
      multisample: { count: sampleCount },
    });
    this.fillBind = device.createBindGroup({ layout: this.fillPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.ubuf } }] });
    this.topoBind = device.createBindGroup({ layout: this.topoPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.ubuf } }] });
    this.lineBind = device.createBindGroup({ layout: this.linePipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.ubuf } }] });
  }

  private rebuild(camX: number, camZ: number): void {
    const S = this.spacing, R = this.radius;
    const n = Math.ceil(R / S);
    const c0x = Math.round(camX / S), c0z = Math.round(camZ / S);
    const f = this.fillHost, l = this.lineHost;
    let fi = 0, li = 0;
    for (let dz = -n; dz <= n; dz++) {
      for (let dx = -n; dx <= n; dx++) {
        const x = (c0x + dx) * S, z = (c0z + dz) * S;
        if (Math.hypot(x - camX, z - camZ) > R) continue;
        const x1 = x + S, z1 = z + S;
        // fill: two triangles for the cell
        f[fi++] = x; f[fi++] = z; f[fi++] = x1; f[fi++] = z; f[fi++] = x; f[fi++] = z1;
        f[fi++] = x1; f[fi++] = z; f[fi++] = x1; f[fi++] = z1; f[fi++] = x; f[fi++] = z1;
        // lines: the cell's +x and +z edges
        l[li++] = x; l[li++] = z; l[li++] = x1; l[li++] = z;
        l[li++] = x; l[li++] = z; l[li++] = x; l[li++] = z1;
      }
    }
    this.fillCount = fi / FLOATS_PER_VERT;
    this.lineCount = li / FLOATS_PER_VERT;
    this.device.queue.writeBuffer(this.fillBuf, 0, this.fillHost.buffer, 0, fi * 4);
    this.device.queue.writeBuffer(this.lineBuf, 0, this.lineHost.buffer, 0, li * 4);
  }

  draw(
    encoder: GPUCommandEncoder,
    colorView: GPUTextureView,
    depthView: GPUTextureView,
    viewProj: Float32Array,
    camGround: [number, number],
    eye: [number, number, number],
    clearColor: GPUColor,
  ): void {
    const cx = Math.round(camGround[0] / this.spacing);
    const cz = Math.round(camGround[1] / this.spacing);
    if (cx !== this.lastCellX || cz !== this.lastCellZ) {
      this.rebuild(camGround[0], camGround[1]);
      this.lastCellX = cx; this.lastCellZ = cz;
    }

    const u = this.u;
    u.set(viewProj, 0);
    u[16] = eye[0]; u[17] = eye[1]; u[18] = eye[2]; u[19] = this.fogDensity;
    u[20] = this.fogColor[0]; u[21] = this.fogColor[1]; u[22] = this.fogColor[2]; u[23] = this.maxDist;
    u[24] = this.interval; u[25] = this.lineWidth; u[26] = this.floorFade; u[27] = this.peakGain;
    this.device.queue.writeBuffer(this.ubuf, 0, this.uniformHost);

    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: colorView, loadOp: "clear", storeOp: "store", clearValue: clearColor }],
      depthStencilAttachment: { view: depthView, depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" },
    });
    if (this.mode === "topo") {
      // TOPO: just the draped mesh with per-fragment contour shading (no wireframe lines).
      pass.setPipeline(this.topoPipeline);
      pass.setBindGroup(0, this.topoBind);
      pass.setVertexBuffer(0, this.fillBuf);
      pass.draw(this.fillCount);
    } else {
      // GRID: dark fill (depth/occlusion) + neon wireframe lines on top.
      pass.setPipeline(this.fillPipeline);
      pass.setBindGroup(0, this.fillBind);
      pass.setVertexBuffer(0, this.fillBuf);
      pass.draw(this.fillCount);
      pass.setPipeline(this.linePipeline);
      pass.setBindGroup(0, this.lineBind);
      pass.setVertexBuffer(0, this.lineBuf);
      pass.draw(this.lineCount);
    }
    pass.end();
  }
}
