// terrain.ts — TerrainEKG: stacked horizontal neon trace LINES + opaque hidden-line FILL curtains.
// Responsibilities:
//   - Build a STACK of ~rows horizontal polylines (line-list). Each row is a fixed depth ahead of
//     the camera; each vertex is (xFrac in [-1,1], rowDepth in meters) + rowFade. Segments are
//     emitted as line-list pairs (no strip restart).
//   - Build a matching STACK of opaque FILL curtains (triangle-list): per row, a vertical quad from
//     the ridge line DOWN to a low baseline. Drawn FIRST, opaque, SKY-colored, depthWrite ON. A near
//     curtain writes nearer depth and OCCLUDES the lines of farther rows → hidden-line removal (no
//     horizon tangle). NOT a row-to-row connected mesh (that is the shaded surface the user rejected).
//   - Lines drawn AFTER the fills with depthCompare "less-equal" and depthWrite OFF, so a line sits
//     on its own curtain's top edge without z-fighting while NEARER curtains still occlude it.
//   - CAMERA-RELATIVE ROWS (v4): the WGSL builds each world sample as camGround + camFwd*depth +
//     camRight*(xFrac*halfWidth). Rows are perpendicular to camForward → screen-horizontal at every
//     heading (v3 locked rows to world-East, so turning skewed them diagonally; fixed here).
//   - Own the uniform buffer (viewProj, camGround, halfWidth, maxDist, camFwd, camRight, fog, eye,
//     baseline); upload per draw. maxDist gives a hard horizon cutoff (clean horizon).
//   - Elevation color is in the line fragment shader (cool valleys → warm/bright peaks).
//   - sampleHeight(x,z): TS mirror of the WGSL fBm (same constants/hash) for the bird ridge-lift.
//   - draw(...): record fill pass (clears color+depth) then line pass (loads) — first pass of frame.

// Heightfield constants — MUST mirror terrain_ekg.wgsl exactly (bird physics samples this twin).
const BASE_FREQ = 0.00142857; // ~1/700 per meter — features 2× wider; a valley dwarfs the 7 m bird
const LACUNARITY = 2.0;
const GAIN = 0.5;
const OCTAVES = 4;            // extra octave restores mid-scale detail at the wider base
const RELIEF = 600.0;         // taller, more dramatic peaks (was 320) — the bird threads between mountains
const SHARP = 1.8;            // pow on the normalized ridge sum — deepens valleys, sharpens crests
const TERRACES = 5.0;         // cliff bands: flat shelves with steep risers (geology, not noise)
const RISER_POW = 4.0;        // riser sharpness within each band — higher = cliffier
const CLIFF_MIX = 0.65;       // blend terraced vs smooth (1 = hard ledges everywhere)

export interface TerrainParams {
  rows?: number;       // number of stacked depth rows
  cols?: number;       // horizontal samples per row (polyline resolution)
  rowSpacing?: number; // world meters between rows in the DENSE near band (the close-up step)
  nearDenseDepth?: number; // rows stay at the full rowSpacing density out to this depth (m) → crisp foreground
  farSpread?: number;  // beyond nearDenseDepth, spacing grows by one rowSpacing per this many meters of depth
                       // (linear) → far rows spread out and declutter. Larger = gentler far thinning. ∞ = uniform.
  rowStart?: number;   // depth of the nearest row ahead of the camera (m)
  halfWidth?: number;  // half horizontal extent of each row (m)
  maxDist?: number;    // hard draw-distance cutoff (m) → clean horizon
  fogColor?: [number, number, number];
  fogDensity?: number;
  baseline?: number;   // low world-y the fill curtains drop to (occlusion only)
  sampleCount?: number; // MSAA samples per pixel — MUST match the render target + every pipeline in the pass
}

export class TerrainEKG {
  readonly rows: number;
  readonly cols: number;
  readonly rowSpacing: number;
  readonly nearDenseDepth: number;
  readonly farSpread: number;
  readonly rowStart: number;
  readonly halfWidth: number;
  readonly maxDist: number;

  private vbuf: GPUBuffer;
  private vertexCount: number;
  private fillBuf: GPUBuffer;
  private fillVertexCount: number;
  private ubuf: GPUBuffer;
  private pipeline: GPURenderPipeline;
  private fillPipeline: GPURenderPipeline;
  private bindGroup: GPUBindGroup;
  private fillBindGroup: GPUBindGroup;
  private uniformHost: ArrayBuffer;
  private uniformData: Float32Array;

  private fogColor: [number, number, number];
  private fogDensity: number;
  private baseline: number;

  // Precomputed per-row depth (m ahead of camera). Built in the constructor: a DENSE near band at the full
  // rowSpacing, then spacing that grows linearly with depth in the far field (declutter the far mush).
  private rowDepths!: Float32Array;

  // Depth (m ahead of the camera) of stacked row r — read from the precomputed rowDepths.
  private rowDepthAt(r: number): number {
    return this.rowDepths[r]!;
  }

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
    this.baseline = p.baseline ?? -300; // terrain min is 0; drop curtains well below frame.

    // GRADUATED DEPTH (declutter the far field WITHOUT thinning the foreground): hold the full rowSpacing
    // density through the DENSE near band (depth ≤ nearDenseDepth — the crisp close-up the eye reads), then
    // grow the step linearly with depth so far rows spread out where they'd otherwise compress into mush.
    // Defaults (nearDenseDepth 0, farSpread ∞) reproduce uniform spacing exactly.
    this.nearDenseDepth = p.nearDenseDepth ?? 0;
    this.farSpread = p.farSpread ?? Infinity;
    // Build the row depths by walking outward from rowStart to maxDist; the explicit `rows` is an upper cap.
    // step(d) = rowSpacing·(1 + max(0, d − nearDenseDepth)/farSpread): flat (= rowSpacing) through the near
    // band, then linearly increasing — fewer total rows, SAME far horizon, foreground density untouched.
    const depths: number[] = [];
    let d = this.rowStart;
    while (d <= this.maxDist && depths.length < this.rows) {
      depths.push(d);
      const over = Math.max(0, d - this.nearDenseDepth);
      d += this.rowSpacing * (1 + over / this.farSpread);
    }
    this.rowDepths = Float32Array.from(depths);
    this.rows = depths.length;

    // --- line-list vertices: per row, (cols-1) segments = 2 verts each. ---
    // each vertex: xFrac(1) + rowDepth(1) + rowFade(1) = 3 floats.
    const segsPerRow = this.cols - 1;
    this.vertexCount = this.rows * segsPerRow * 2;
    const FPV = 3;
    const verts = new Float32Array(this.vertexCount * FPV);
    let vi = 0;
    for (let r = 0; r < this.rows; r++) {
      const rowDepth = this.rowDepthAt(r);
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

    // --- fill curtains: per row, (cols-1) quads = 2 tris = 6 verts each. ---
    // each vertex: xFrac(1) + rowDepth(1) + topFlag(1) = 3 floats. topFlag 1=ridge, 0=baseline.
    // A quad spans [xA,xB] horizontally, ridge-top down to baseline vertically (vertical curtain).
    this.fillVertexCount = this.rows * segsPerRow * 6;
    const fverts = new Float32Array(this.fillVertexCount * FPV);
    let fi = 0;
    const pushFV = (x: number, d: number, top: number) => {
      fverts[fi++] = x; fverts[fi++] = d; fverts[fi++] = top;
    };
    for (let r = 0; r < this.rows; r++) {
      const rowDepth = this.rowDepthAt(r);
      for (let c = 0; c < segsPerRow; c++) {
        const xA = (c / segsPerRow) * 2 - 1;
        const xB = ((c + 1) / segsPerRow) * 2 - 1;
        // tri 1: topA, topB, botA   tri 2: botA, topB, botB
        pushFV(xA, rowDepth, 1); pushFV(xB, rowDepth, 1); pushFV(xA, rowDepth, 0);
        pushFV(xA, rowDepth, 0); pushFV(xB, rowDepth, 1); pushFV(xB, rowDepth, 0);
      }
    }
    this.fillBuf = device.createBuffer({
      size: fverts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.fillBuf, 0, fverts);

    // --- uniform buffer (std140-aligned, 128 bytes / 32 floats):
    //   [0..16)  mat4 viewProj
    //   [16,17]  camGround.xz   [18] halfWidth   [19] maxDist
    //   [20,21]  camFwd.xz      [22,23] camRight.xz
    //   [24..27) fogColor.rgb   [27] fogDensity
    //   [28..31) eye.xyz        [31] baseline (fill curtain bottom, world-y)
    this.uniformHost = new ArrayBuffer(32 * 4);
    this.uniformData = new Float32Array(this.uniformHost);
    this.ubuf = device.createBuffer({
      size: this.uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const sampleCount = p.sampleCount ?? 1; // 1 = no MSAA; >1 must match the render target + all pipelines.
    const module = device.createShaderModule({ code: shader });
    // shared vertex layout: 3 floats (xFrac, rowDepth, flag) — flag is rowFade for lines, topFlag for fill.
    const vbLayout: GPUVertexBufferLayout = {
      arrayStride: FPV * 4,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x2" }, // xFrac, rowDepth
        { shaderLocation: 1, offset: 8, format: "float32" },   // rowFade / topFlag
      ],
    };

    // --- FILL pipeline: opaque SKY curtains, triangle-list, depthWrite ON (owns the depth buffer). ---
    this.fillPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vsFill", buffers: [vbLayout] },
      fragment: {
        module,
        entryPoint: "fsFill",
        targets: [{ format: colorFormat }], // no blend → opaque
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
      multisample: { count: sampleCount },
    });

    // --- LINE pipeline: additive neon glow, depthCompare less-equal, depthWrite OFF. ---
    // less-equal lets a line sit on its own curtain top edge; nearer curtains (strictly smaller
    // depth, written by the fill pass) still occlude it → hidden-line removal without z-fighting.
    this.pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vsLine", buffers: [vbLayout] },
      fragment: {
        module,
        entryPoint: "fsLine",
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
      depthStencil: { depthWriteEnabled: false, depthCompare: "less-equal", format: "depth24plus" },
      multisample: { count: sampleCount },
    });

    // separate bind group per pipeline (layout:"auto" → distinct layouts), both pointing at ubuf.
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.ubuf } }],
    });
    this.fillBindGroup = device.createBindGroup({
      layout: this.fillPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.ubuf } }],
    });
  }

  // TS mirror of the WGSL fBm — CPU height reference the bird COLLIDES against. The hash must be
  // bit-identical to the GPU's, or the bird crashes into invisible terrain. The old fract(sin()*N)
  // hash could NOT be mirrored: the *43758 amplifies the f32(GPU)-vs-f64(JS) sin difference into a
  // totally different field (measured: ~100 m mean, 460 m max disagreement on a 600 m relief).
  // An integer lattice hash uses only exact uint32 ops, so GPU and CPU agree to <1 mm.
  // MUST stay identical to ihash() in terrain_ekg.wgsl / terrain_grid.wgsl / trees_ground.wgsl.
  private ihash(ix: number, iy: number): number {
    let h = (Math.imul(ix | 0, 374761393) + Math.imul(iy | 0, 668265263)) >>> 0;
    h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0;
    h = (h ^ (h >>> 16)) >>> 0;
    return h / 4294967295;
  }
  private valueNoise(px: number, py: number): number {
    const ix = Math.floor(px), iy = Math.floor(py);
    const fx = px - ix, fy = py - iy;
    const a = this.ihash(ix, iy);
    const b = this.ihash(ix + 1, iy);
    const c = this.ihash(ix, iy + 1);
    const d = this.ihash(ix + 1, iy + 1);
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
    // sharpen (deep valleys, crisp crests) then carve terraced cliff bands (shelf + steep riser).
    const s = Math.pow(sum / norm, SHARP);
    const b = s * TERRACES;
    const fb = b - Math.floor(b);
    const ter = Math.floor(b) / TERRACES + Math.pow(fb, RISER_POW) / TERRACES;
    return (s + (ter - s) * CLIFF_MIX) * RELIEF;
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
    u[28] = eye[0]; u[29] = eye[1]; u[30] = eye[2]; u[31] = this.baseline; // eye + fill baseline
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
    // FILL FIRST: opaque SKY curtains write depth → near rows occlude far lines (hidden-line removal).
    pass.setPipeline(this.fillPipeline);
    pass.setBindGroup(0, this.fillBindGroup);
    pass.setVertexBuffer(0, this.fillBuf);
    pass.draw(this.fillVertexCount);
    // LINES SECOND: additive neon, depth-test less-equal against the curtains, no depth-write.
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vbuf);
    pass.draw(this.vertexCount);
    pass.end();
  }
}
