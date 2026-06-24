// trees.ts — Trees: an elevation-banded neon forest streamed around the camera.
// Responsibilities:
//   - Place trees on a deterministic grid window around the camera, CLUSTERED by a low-frequency density
//     field (clumps + clearings, not a uniform scatter), thinning out above the alpine line.
//   - Choose species by terrain elevation: deciduous (oak/maple, rounded crown, warm green) in the
//     valleys, conifers (fir/pine, whorled cone, cool blue-green) on the tops, blended through a treeline.
//   - Generate each tree CPU-side as line segments with per-vertex HDR color, rooted slightly INTO the
//     terrain so it never floats, baked into ONE batched line-list buffer (no instancing).
//   - Rebuild that buffer only when the camera crosses a grid cell (infrequent), not per frame.
//   - Own the uniform + render pipeline (trees.wgsl); draw() records one depth-tested pass that LOADS
//     color+depth (ridges occlude) and fades trees in/out with distance so the window edge never pops.

type Vec3 = [number, number, number];
type HeightFn = (x: number, z: number) => number;

const FLOATS_PER_VERT = 7; // worldX, worldZ, localOffsetY, treeId, color.rgb (ground from the prepass buffer)
const UNIFORM_BYTES = 96; // mat4(64) + eye+fogDensity(16) + fadeStart+fadeEnd+pad+time(16)

// --- placement window ---
const CELL = 14; // m — grid spacing of candidate tree slots
const RADIUS = 680; // m — how far out trees stream around the camera
const MAX_TREES = 14000; // ALLOCATION ceiling (vertex buffer + CPU rebuild cost). The live tuning.maxTrees
// (default 9000) caps how many trees actually generate, up to this ceiling.
const PEAK_RELIEF = 600; // m — terrain RELIEF constant (max height); mirror of terrain.ts
const ROOT_SINK = 4; // m — plant the trunk base this far BELOW the sampled surface so it never floats

// --- species by elevation fraction (height / PEAK_RELIEF) ---
const DECID_MAX = 0.42; // below this fraction → all deciduous (valleys)
const CONIFER_MIN = 0.58; // above this fraction → all conifer (tops); blended in between
const ALPINE = 0.86; // above this the tops go rocky → density thins toward zero

// --- clustering (forest density field) ---
const CLUMP_FREQ = 1 / 160; // 1/m — clump wavelength (~240 m forests/clearings)
const COVER_LO = 0.44; // density-field value below which there are no trees (clearings)
const COVER_HI = 0.54; // density-field value at/above which coverage is full

// --- tree size (MUCH smaller than v1) ---
const DECID_H = 9; // m — deciduous trunk+crown reach
const CONIFER_H = 12; // m — conifer height
const CONIFER_RAD = 3.2; // m — conifer base half-width
const ANCIENT_FRAC = 0.04; // a few "big ole" specimens
const ANCIENT_SCALE = 1.8; // ancient size multiplier (still small in absolute terms)

// --- HDR colors (>1 green so the bloom picks it up); additive blend ---
const DECID_FOLIAGE: Vec3 = [0.9, 2.1, 0.55]; // warm yellow-green (oak/maple)
const DECID_TRUNK: Vec3 = [0.35, 0.45, 0.22];
const CONIFER_FOLIAGE: Vec3 = [0.3, 1.9, 1.15]; // cool blue-green (fir/pine)
const CONIFER_TRUNK: Vec3 = [0.2, 0.42, 0.34];

const FOG_DENSITY = 0.5 / 1400; // matches the scene's mote/terrain fog band

// deterministic per-cell hash → uint32.
function hash2(ix: number, iz: number): number {
  let h = (Math.imul(ix, 0x27d4eb2d) ^ Math.imul(iz, 0x165667b1)) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d) >>> 0;
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39) >>> 0;
  return (h ^ (h >>> 15)) >>> 0;
}
function hashf(ix: number, iz: number): number {
  return hash2(ix, iz) / 4294967296;
}

// mulberry32 — small seeded RNG so each tree's shape is stable across rebuilds.
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// smooth value noise on a lattice → clustered forest density.
function smooth(t: number): number {
  return t * t * (3 - 2 * t);
}
function vnoise(x: number, z: number): number {
  const xi = Math.floor(x), zi = Math.floor(z);
  const fx = smooth(x - xi), fz = smooth(z - zi);
  const a = hashf(xi, zi), b = hashf(xi + 1, zi);
  const c = hashf(xi, zi + 1), d = hashf(xi + 1, zi + 1);
  return (a * (1 - fx) + b * fx) * (1 - fz) + (c * (1 - fx) + d * fx) * fz;
}

function norm(v: Vec3): Vec3 {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

// --- landmark giants: rare big recursive (L-system) trees at peaks, as waypoints ---
const LANDMARK_CELL = 320; // m — coarse grid; ~one candidate per cell
const LANDMARK_MIN_E = 0.72; // only on high ground (fraction of PEAK_RELIEF) → peaks
const LANDMARK_RARITY = 0.45; // fraction of qualifying coarse cells that grow a giant ("a few")
const LANDMARK_H = 58; // m — trunk reach (towers over the ~10 m forest)
const LANDMARK_DEPTH = 6; // recursion depth (many branches)
const LANDMARK_FOLIAGE: Vec3 = [0.8, 2.4, 1.4]; // bright cyan-white-green → reads as a waypoint
const LANDMARK_TRUNK: Vec3 = [0.4, 0.7, 0.5];


export class Trees {
  enabled = true;
  treeCount = 0;
  landmarks: [number, number][] = []; // world XZ of the landmark giants placed this rebuild (debug/waypoints)

  // LIVE-TUNABLE — the host panel writes straight into this. The rebuild-affecting fields (maxTrees,
  // coverLo, coverHi, sizeScale, radius, glow) trigger a buffer rebuild on the next draw() via the tuning
  // signature; fogDensity is read per-frame by the caller (live, no rebuild). Defaults mirror the originals.
  tuning = {
    maxTrees: 9000,         // live cap on generated trees (density + perf); ≤ MAX_TREES allocation
    coverLo: COVER_LO,      // density-field threshold: below = clearings (LOWER → more trees)
    coverHi: COVER_HI,      // density-field threshold: at/above = full coverage
    sizeScale: 1,           // global multiplier on regular-tree height + width
    radius: RADIUS,         // stream + fade radius (m) around the camera
    glow: 1,                // HDR colour multiplier baked into every tree vertex (bloom brightness)
    fogDensity: 0.5 / 1100, // distance haze (matches the terrain's); higher = shorter view
  };

  private vbuf: GPUBuffer;
  private maxVerts: number;
  private vertexCount = 0;
  private hostBytes: ArrayBuffer;
  private host: Float32Array;

  private ubuf: GPUBuffer;
  private uniformHost = new ArrayBuffer(UNIFORM_BYTES);
  private uniformF32 = new Float32Array(this.uniformHost);
  private pipeline: GPURenderPipeline;
  private bindGroup: GPUBindGroup;

  // per-tree ground prepass: base XZ in → ground height out (computed once per tree on rebuild).
  private baseHost: Float32Array; // [x,z] per tree
  private baseBuf: GPUBuffer; // storage: per-tree base XZ
  private groundBuf: GPUBuffer; // storage: per-tree ground height (compute output, vertex input)
  private computePipeline: GPUComputePipeline;
  private computeBindGroup: GPUBindGroup;
  private groundDirty = false;

  private lastCellX = Number.NaN;
  private lastCellZ = Number.NaN;
  private rebuildSig = ""; // signature of the rebuild-affecting tuning values at the last rebuild
  private cbx = 0; // current tree base world X
  private cbz = 0; // current tree base world Z
  private ctid = 0; // current tree index (written per vertex → indexes the ground buffer)

  constructor(
    private device: GPUDevice,
    shader: string,
    groundShader: string,
    colorFormat: GPUTextureFormat,
    private sampleHeight: HeightFn,
    sampleCount = 1,
  ) {
    this.maxVerts = MAX_TREES * 14 * 2; // generous per-tree segment budget
    this.hostBytes = new ArrayBuffer(this.maxVerts * FLOATS_PER_VERT * 4);
    this.host = new Float32Array(this.hostBytes);

    this.vbuf = device.createBuffer({
      size: this.hostBytes.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.ubuf = device.createBuffer({
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.baseHost = new Float32Array(MAX_TREES * 2);
    this.baseBuf = device.createBuffer({ size: MAX_TREES * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.groundBuf = device.createBuffer({ size: MAX_TREES * 4, usage: GPUBufferUsage.STORAGE });

    // ground prepass compute pipeline
    const cmod = device.createShaderModule({ code: groundShader });
    this.computePipeline = device.createComputePipeline({ layout: "auto", compute: { module: cmod, entryPoint: "main" } });
    this.computeBindGroup = device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.baseBuf } },
        { binding: 1, resource: { buffer: this.groundBuf } },
      ],
    });

    const module = device.createShaderModule({ code: shader });
    this.pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module,
        entryPoint: "vs",
        buffers: [
          {
            arrayStride: FLOATS_PER_VERT * 4,
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x3" }, // worldX, worldZ, offY
              { shaderLocation: 1, offset: 12, format: "float32" },   // treeId
              { shaderLocation: 2, offset: 16, format: "float32x3" }, // color
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
      primitive: { topology: "line-list" },
      depthStencil: {
        depthWriteEnabled: false,
        depthCompare: "less",
        format: "depth24plus",
      },
      multisample: { count: sampleCount },
    });
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.ubuf } },
        { binding: 1, resource: { buffer: this.groundBuf } },
      ],
    });
  }

  // append one colored line segment. a,b are LOCAL offsets from the tree base; stored as world XZ + local
  // height + treeId (the prepass-computed ground for that tree is added in the vertex shader).
  private seg(out: { i: number }, a: Vec3, b: Vec3, ca: Vec3, cb: Vec3): void {
    const h = this.host;
    let i = out.i;
    if (i + FLOATS_PER_VERT * 2 > h.length) return;
    const bx = this.cbx, bz = this.cbz, id = this.ctid, g = this.tuning.glow;
    h[i++] = bx + a[0]; h[i++] = bz + a[2]; h[i++] = a[1]; h[i++] = id; h[i++] = ca[0] * g; h[i++] = ca[1] * g; h[i++] = ca[2] * g;
    h[i++] = bx + b[0]; h[i++] = bz + b[2]; h[i++] = b[1]; h[i++] = id; h[i++] = cb[0] * g; h[i++] = cb[1] * g; h[i++] = cb[2] * g;
    out.i = i;
  }

  // CONIFER (simple): a vertical trunk + 3 whorls of 2 short down-arms → a fir glyph in ~7 segments.
  private conifer(out: { i: number }, base: Vec3, h: number, rad: number, fol: Vec3, trunk: Vec3, r: () => number): void {
    this.seg(out, base, [base[0], base[1] + h, base[2]], trunk, [fol[0] * 0.6, fol[1] * 0.6, fol[2] * 0.6]);
    const a0 = r() * Math.PI * 2;
    for (let w = 0; w < 3; w++) {
      const t = w / 2; // 0 bottom → 1 top
      const y = base[1] + h * (0.25 + 0.6 * t);
      const len = rad * (1 - 0.6 * t);
      const bright = 0.8 + 0.5 * t;
      const cb: Vec3 = [fol[0] * bright, fol[1] * bright, fol[2] * bright];
      for (let b = 0; b < 2; b++) {
        const az = a0 + w * 1.3 + b * Math.PI; // opposite arms, rotated per whorl
        this.seg(out, [base[0], y, base[2]], [base[0] + Math.cos(az) * len, y - 0.3 * len, base[2] + Math.sin(az) * len],
          [fol[0] * 0.7, fol[1] * 0.7, fol[2] * 0.7], cb);
      }
    }
  }

  // DECIDUOUS (simple): a trunk + 4 short crown limbs splayed up-and-out → an oak glyph in ~5 segments.
  private deciduous(out: { i: number }, base: Vec3, h: number, fol: Vec3, trunk: Vec3, r: () => number): void {
    const forkH = h * 0.5;
    const fork: Vec3 = [base[0], base[1] + forkH, base[2]];
    this.seg(out, base, fork, trunk, [fol[0] * 0.5, fol[1] * 0.5, fol[2] * 0.5]);
    const crown = (h - forkH) * 0.9;
    const a0 = r() * Math.PI * 2;
    for (let l = 0; l < 4; l++) {
      const az = a0 + (l / 4) * Math.PI * 2 + (r() - 0.5) * 0.5;
      const outw = 0.6 + 0.2 * r();
      this.seg(out, fork, [fork[0] + Math.cos(az) * crown * outw, fork[1] + crown * 0.85, fork[2] + Math.sin(az) * crown * outw],
        [fol[0] * 0.55, fol[1] * 0.55, fol[2] * 0.55], fol);
    }
  }

  // LANDMARK recursive branch (L-system): the original fractal — deep recursion, many branches.
  private branch(out: { i: number }, base: Vec3, dir: Vec3, len: number, depth: number, fol: Vec3, r: () => number): void {
    const tip: Vec3 = [base[0] + dir[0] * len, base[1] + dir[1] * len, base[2] + dir[2] * len];
    const bright = 0.45 + 0.7 * (1 - depth / LANDMARK_DEPTH); // trunk/inner brighter
    const c: Vec3 = [fol[0] * bright, fol[1] * bright, fol[2] * bright];
    this.seg(out, base, tip, [fol[0] * 0.35, fol[1] * 0.35, fol[2] * 0.35], c);
    if (depth <= 0) return;
    const up: Vec3 = Math.abs(dir[1]) < 0.99 ? [0, 1, 0] : [1, 0, 0];
    const a = norm([dir[1] * up[2] - dir[2] * up[1], dir[2] * up[0] - dir[0] * up[2], dir[0] * up[1] - dir[1] * up[0]]);
    const b = norm([dir[1] * a[2] - dir[2] * a[1], dir[2] * a[0] - dir[0] * a[2], dir[0] * a[1] - dir[1] * a[0]]);
    const kids = 2 + (r() < 0.4 ? 1 : 0);
    for (let k = 0; k < kids; k++) {
      const split = 0.5 * (0.7 + 0.6 * r());
      const phi = r() * Math.PI * 2;
      const st = Math.sin(split), ct = Math.cos(split);
      const cd = norm([
        dir[0] * ct + (a[0] * Math.cos(phi) + b[0] * Math.sin(phi)) * st,
        dir[1] * ct + (a[1] * Math.cos(phi) + b[1] * Math.sin(phi)) * st + 0.15,
        dir[2] * ct + (a[2] * Math.cos(phi) + b[2] * Math.sin(phi)) * st,
      ]);
      this.branch(out, tip, cd, len * 0.74 * (0.82 + 0.3 * r()), depth - 1, fol, r);
    }
  }

  // a big honking landmark giant: a tall trunk forking into a deep recursive crown.
  private landmark(out: { i: number }, base: Vec3, h: number, fol: Vec3, trunk: Vec3, r: () => number): void {
    const lean = 0.07;
    const la = r() * Math.PI * 2;
    const dir = norm([Math.cos(la) * lean, 1, Math.sin(la) * lean]);
    // trunk to the first fork, then recurse into the crown.
    const forkH = h * 0.38;
    const fork: Vec3 = [base[0] + dir[0] * forkH, base[1] + dir[1] * forkH, base[2] + dir[2] * forkH];
    this.seg(out, base, fork, trunk, [fol[0] * 0.3, fol[1] * 0.3, fol[2] * 0.3]);
    this.branch(out, fork, dir, h * 0.4, LANDMARK_DEPTH, fol, r);
  }

  // Rebuild the batched vertex buffer for the window centered on the camera ground position.
  private rebuild(camX: number, camZ: number): void {
    const out = { i: 0 };
    let trees = 0;
    this.landmarks.length = 0;
    const cap = Math.min(this.tuning.maxTrees | 0, MAX_TREES); // live count cap (≤ allocation)
    const radius = this.tuning.radius;                          // live stream radius
    const sizeScale = this.tuning.sizeScale;
    const coverLo = this.tuning.coverLo, coverHi = this.tuning.coverHi;

    // LANDMARK pass FIRST (reserve the giants before the forest fills the cap): a coarse grid drops a
    // rare big recursive tree near the local high point of qualifying cells → waypoints on the peaks.
    const lcells = Math.floor(radius / LANDMARK_CELL);
    const l0x = Math.round(camX / LANDMARK_CELL), l0z = Math.round(camZ / LANDMARK_CELL);
    for (let dz = -lcells; dz <= lcells; dz++) {
      for (let dx = -lcells; dx <= lcells; dx++) {
        const lr = rng(hash2((l0x + dx) * 131 + 7, (l0z + dz) * 131 + 7)); // distinct seed stream
        if (lr() > LANDMARK_RARITY) continue;
        // nudge toward the local high point: sample a few spots in the cell, keep the highest.
        let bx = (l0x + dx) * LANDMARK_CELL, bz = (l0z + dz) * LANDMARK_CELL, bh = -1;
        for (let s = 0; s < 5; s++) {
          const sx = (l0x + dx) * LANDMARK_CELL + (lr() - 0.5) * LANDMARK_CELL;
          const sz = (l0z + dz) * LANDMARK_CELL + (lr() - 0.5) * LANDMARK_CELL;
          const hh = this.sampleHeight(sx, sz);
          if (hh > bh) { bh = hh; bx = sx; bz = sz; }
        }
        if (Math.hypot(bx - camX, bz - camZ) > radius) continue;
        if (bh / PEAK_RELIEF < LANDMARK_MIN_E) continue; // peaks only
        if (trees >= cap) break;
        this.cbx = bx; this.cbz = bz; this.ctid = trees;
        this.baseHost[trees * 2] = bx; this.baseHost[trees * 2 + 1] = bz;
        this.landmark(out, [0, -ROOT_SINK, 0], LANDMARK_H, LANDMARK_FOLIAGE, LANDMARK_TRUNK, lr);
        this.landmarks.push([bx, bz]);
        trees++;
      }
    }

    const cells = Math.floor(radius / CELL);
    const c0x = Math.round(camX / CELL);
    const c0z = Math.round(camZ / CELL);

    for (let dz = -cells; dz <= cells && trees < cap; dz++) {
      for (let dx = -cells; dx <= cells && trees < cap; dx++) {
        const ix = c0x + dx;
        const iz = c0z + dz;
        const r = rng(hash2(ix, iz));
        const wx = ix * CELL + (r() - 0.5) * CELL * 0.85;
        const wz = iz * CELL + (r() - 0.5) * CELL * 0.85;
        if (Math.hypot(wx - camX, wz - camZ) > radius) continue;

        // clustering: low-freq density field carves clumps and clearings.
        const cover = (vnoise(wx * CLUMP_FREQ, wz * CLUMP_FREQ) - coverLo) / (coverHi - coverLo);
        if (r() > Math.max(0, Math.min(1, cover))) continue;

        const gy = this.sampleHeight(wx, wz);
        const e = gy / PEAK_RELIEF; // elevation fraction
        // alpine thinning: above the treeline the tops turn rocky.
        if (e > ALPINE && r() > Math.max(0, (1 - e) / (1 - ALPINE))) continue;

        // species by elevation, blended through the treeline band.
        const coniferP = Math.max(0, Math.min(1, (e - DECID_MAX) / (CONIFER_MIN - DECID_MAX)));
        const isConifer = r() < coniferP;

        const ancient = r() < ANCIENT_FRAC;
        const scale = (ancient ? ANCIENT_SCALE : 1) * (0.8 + 0.45 * r()) * sizeScale;
        // base is LOCAL (origin at the trunk foot, sunk ROOT_SINK); the prepass adds fbm(wx,wz) as ground.
        this.cbx = wx; this.cbz = wz; this.ctid = trees;
        this.baseHost[trees * 2] = wx; this.baseHost[trees * 2 + 1] = wz;
        const base: Vec3 = [0, -ROOT_SINK, 0];
        if (isConifer) {
          this.conifer(out, base, CONIFER_H * scale, CONIFER_RAD * scale, CONIFER_FOLIAGE, CONIFER_TRUNK, r);
        } else {
          this.deciduous(out, base, DECID_H * scale, DECID_FOLIAGE, DECID_TRUNK, r);
        }
        trees++;
      }
    }

    this.vertexCount = out.i / FLOATS_PER_VERT;
    this.treeCount = trees;
    this.device.queue.writeBuffer(this.vbuf, 0, this.hostBytes, 0, out.i * 4);
    this.device.queue.writeBuffer(this.baseBuf, 0, this.baseHost.buffer, 0, trees * 2 * 4);
    this.groundDirty = true; // ground prepass must recompute for the new tree set
  }

  draw(
    encoder: GPUCommandEncoder,
    colorView: GPUTextureView,
    depthView: GPUTextureView,
    viewProj: Float32Array,
    camGround: [number, number],
    eye: Vec3,
    time: number,
    fogDensity = FOG_DENSITY, // pass the terrain's fog density so trees haze identically with distance
    timestampWrites?: GPURenderPassTimestampWrites, // optional profiling hook (GpuProfiler)
  ): void {
    if (!this.enabled) return;

    const cx = Math.round(camGround[0] / CELL);
    const cz = Math.round(camGround[1] / CELL);
    // rebuild on a cell-cross OR when a rebuild-affecting tuning value changed (panel sliders).
    const t = this.tuning;
    const sig = `${Math.min(t.maxTrees | 0, MAX_TREES)}|${t.coverLo}|${t.coverHi}|${t.sizeScale}|${t.radius}|${t.glow}`;
    if (cx !== this.lastCellX || cz !== this.lastCellZ || sig !== this.rebuildSig) {
      this.rebuild(camGround[0], camGround[1]);
      this.lastCellX = cx;
      this.lastCellZ = cz;
      this.rebuildSig = sig;
    }
    if (this.vertexCount === 0) return;

    // ground prepass: compute each tree's terrain height ONCE (only when the tree set changed).
    if (this.groundDirty) {
      const cpass = encoder.beginComputePass();
      cpass.setPipeline(this.computePipeline);
      cpass.setBindGroup(0, this.computeBindGroup);
      cpass.dispatchWorkgroups(Math.ceil(this.treeCount / 64));
      cpass.end();
      this.groundDirty = false;
    }

    const u = this.uniformF32;
    u.set(viewProj, 0);
    u[16] = eye[0]; u[17] = eye[1]; u[18] = eye[2];
    u[19] = fogDensity;
    u[20] = this.tuning.radius * 0.78; // fadeStart (tracks the live stream radius)
    u[21] = this.tuning.radius * 0.98; // fadeEnd — trees fade fully out before the window rim → no pop
    u[22] = 0;
    u[23] = time;
    this.device.queue.writeBuffer(this.ubuf, 0, this.uniformHost);

    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: colorView, loadOp: "load", storeOp: "store" }],
      depthStencilAttachment: {
        view: depthView,
        depthLoadOp: "load",
        depthStoreOp: "store",
      },
      timestampWrites,
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vbuf);
    pass.draw(this.vertexCount);
    pass.end();
  }
}
