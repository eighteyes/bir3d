// wind.ts — shared analytic wind field + Wind DOT-particle overlay (drifting neon motes).
// Responsibilities:
//   - windAt(x,z,t): SINGLE SOURCE OF TRUTH for the wind vector at any world XZ + time. A
//     divergence-free curl-noise flow plus a slow large-scale drift, CRANKED so the lateral
//     component is unmistakable (~8-12 m/s vs the bird's ~26 m/s forward → clear cross-track).
//     Closed-form so the SAME field drives the bird physics (CPU), this dot overlay (CPU advection),
//     and the overlay compass — zero GPU sync. FLAGGED: stand-in for the GPU fluid
//     (src/host/gpu/fluid.ts), which is compute-only and would need frame-laggy async readback +
//     a grid→world mapping and STILL need a closed form for advection/overlay; the analytic field
//     keeps one coherent, arbitrarily-sampleable source so the field you SEE is the field that PUSHES.
//   - Wind: a render pipeline (wind.wgsl) that draws a field of drifting neon COMET motes over the
//     terrain. Each mote's WORLD position is PERSISTED and advected every frame by windAt (p += w*dt),
//     so the dots literally drift on the wind (motion + density = the flow). Each mote is a small
//     comet — a quad stretched along the screen-space wind direction with a bright head and a fading
//     tail — so drift + direction read even in a still frame and the motes stay distinct from the sky
//     stars. Motes that age out or leave the camera-relative span are recycled ahead. Rendered as
//     additive billboards, depth-tested against terrain (ridges occlude them). NOT regenerated each
//     frame — that would freeze them in place; persistence is what makes them move.
//   - thermalAt(x,z,t): explicit vertical updraft (m/s) the bird rides — a few broad thermals so the
//     vario reads clearly positive somewhere, independent of ridge-lift geometry.

export interface WindConfig {
  curlScale?: number;   // spatial frequency of the curl-noise (1/m)
  curlAmp?: number;     // m/s amplitude of the curl-noise component
  driftDir?: number;    // radians — prevailing large-scale wind heading
  driftAmp?: number;    // m/s amplitude of the steady prevailing drift
  thermalAmp?: number;  // m/s peak vertical updraft of the broad thermals
}

const DEFAULTS: Required<WindConfig> = {
  curlScale: 0.0011,
  curlAmp: 7.0,
  driftDir: (35 * Math.PI) / 180, // prevailing wind toward ~ENE
  driftAmp: 6.0,
  thermalAmp: 4.0,
};

// Closed-form curl-noise potential. Sum of sines → smooth scalar field; wind = curl(potential)
// = (dPot/dz, -dPot/dx), which is divergence-free (looks like real flow, no sources/sinks).
function potential(x: number, z: number, t: number): number {
  const s = 1.0; // scale baked into caller-supplied coords
  return (
    Math.sin(x * 1.0 + 0.15 * t) * Math.cos(z * 1.3) +
    0.6 * Math.sin((x + z) * 0.7 - 0.1 * t) +
    0.4 * Math.cos(x * 0.5 - z * 0.9 + 0.07 * t)
  ) * s;
}

// Horizontal wind [wx, wz] m/s at world (x,z), time t. Curl-noise + steady prevailing drift.
export function windAt(
  x: number,
  z: number,
  t: number,
  cfg: Required<WindConfig> = DEFAULTS
): [number, number] {
  const sc = cfg.curlScale;
  const e = 0.75; // finite-diff step in SCALED units
  const px = x * sc, pz = z * sc;
  // curl of scalar potential
  const dPot_dz =
    (potential(px, pz + e, t) - potential(px, pz - e, t)) / (2 * e);
  const dPot_dx =
    (potential(px + e, pz, t) - potential(px - e, pz, t)) / (2 * e);
  const cx = dPot_dz * cfg.curlAmp;
  const cz = -dPot_dx * cfg.curlAmp;
  // steady prevailing drift (large-scale, ever-present cross-track shove)
  const dx = Math.sin(cfg.driftDir) * cfg.driftAmp;
  const dz = Math.cos(cfg.driftDir) * cfg.driftAmp;
  return [cx + dx, cz + dz];
}

// Vertical thermal updraft (m/s) at world (x,z): a few broad, slowly-drifting columns of rising
// air. Always non-negative; broad enough that the glider passes through them often.
export function thermalAt(
  x: number,
  z: number,
  t: number,
  cfg: Required<WindConfig> = DEFAULTS
): number {
  const s = 0.0013;
  // moving bump field — overlapping broad gaussians via a sine lattice raised to a power.
  const a = Math.sin(x * s + 0.05 * t) * Math.sin(z * s * 1.1 - 0.04 * t);
  const b = Math.sin((x + z) * s * 0.6 + 0.03 * t);
  // SPARSE columns: product of both lobes raised to a power → ~0 across most of the world and
  // strong only in narrow rising cores you must HUNT for (a glider sinks by default; lift is local).
  const core = Math.max(0, a) * Math.max(0, b);
  return Math.pow(core, 2.2) * cfg.thermalAmp;
}

// --- dot-particle overlay pipeline ---

interface DotParams {
  count?: number;     // number of drifting motes
  spanAhead?: number; // how far ahead of the camera ground point the field reaches (m)
  spanBehind?: number;// margin behind the camera before a mote is recycled (m)
  spanWide?: number;  // half-width of the camera-relative field (m)
  clearance?: number; // meters above terrain the motes float
  dotPx?: number;     // on-screen comet head diameter (px) — converted to NDC half-width per frame
  tailMul?: number;   // streak length as a multiple of the head width (comet tail length)
}

export class Wind {
  private cfg: Required<WindConfig>;
  private count: number;
  private spanAhead: number;
  private spanBehind: number;
  private spanWide: number;
  private clearance: number;
  private dotPx: number;
  private tailMul: number;

  // PERSISTENT world-space particle state (NOT regenerated each frame — that is what makes them drift).
  private px: Float32Array;   // world x
  private pz: Float32Array;   // world z
  private speedFrac: Float32Array; // cached 0..1 wind speed at the mote (for color/brightness)

  private vbuf: GPUBuffer;     // per-vertex instanced quad data, rebuilt from particle state each frame
  private vertexCount: number; // count * 6 (two triangles per quad)
  private vertBytes: ArrayBuffer;
  private vertHost: Float32Array;
  private ubuf: GPUBuffer;
  private uniformHost: ArrayBuffer;
  private uniformF32: Float32Array;
  private pipeline: GPURenderPipeline;
  private bindGroup: GPUBindGroup;

  private lastTime = -1;       // for per-frame dt derivation from bird.simTime
  private seeded = false;      // first draw seeds the field around the camera span

  // floats/vertex: center.xyz(3) + corner.xy(2) + speedFrac(1) + windDir.xy(2) = 8
  private static FPV = 8;
  // fixed quad corners (two tris) reused for every mote.
  private static CORNERS: ReadonlyArray<[number, number]> = [
    [-1, -1], [1, -1], [1, 1],
    [-1, -1], [1, 1], [-1, 1],
  ];

  constructor(
    private device: GPUDevice,
    shader: string,
    colorFormat: GPUTextureFormat,
    private sampleHeight: (x: number, z: number) => number,
    cfg: WindConfig = {},
    p: DotParams = {}
  ) {
    this.cfg = { ...DEFAULTS, ...cfg };
    this.count = p.count ?? 1300;
    // spanAhead matches the terrain maxDist (~950) so every mote lives in the DRAWN terrain band —
    // dots past the cutoff floated over a void and read as detached sky specks, not wind in the scene.
    this.spanAhead = p.spanAhead ?? 950;
    this.spanBehind = p.spanBehind ?? 260;
    this.spanWide = p.spanWide ?? 950;
    // clearance keeps motes above the near fill-curtain crests (which write depth and would otherwise
    // occlude low dots) but LOW enough to hug the ridges — too high lifts them into the pure-sky band
    // above the terrain silhouette where they conflict with the starfield. 55 is the readable middle.
    this.clearance = p.clearance ?? 55;
    // smaller head than v5 (11 → 5) so motes read as streaking comets, not star-like blobs.
    this.dotPx = p.dotPx ?? 5;
    // tail a few head-widths long → drift direction reads even in a still frame.
    this.tailMul = p.tailMul ?? 5;

    this.px = new Float32Array(this.count);
    this.pz = new Float32Array(this.count);
    this.speedFrac = new Float32Array(this.count);

    this.vertexCount = this.count * 6;
    this.vertBytes = new ArrayBuffer(this.vertexCount * Wind.FPV * 4);
    this.vertHost = new Float32Array(this.vertBytes);
    this.vbuf = device.createBuffer({
      size: this.vertHost.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    // uniform: mat4 viewProj(16) + eye.xyz(3) + aspect(1) + fogColor.rgb(3) + fogDensity(1)
    //          + dotSize(1) + tailLen(1) + pad(2) = 28
    this.uniformHost = new ArrayBuffer(28 * 4);
    this.uniformF32 = new Float32Array(this.uniformHost);
    this.ubuf = device.createBuffer({
      size: this.uniformHost.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const module = device.createShaderModule({ code: shader });
    this.pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module,
        entryPoint: "vs",
        buffers: [
          {
            arrayStride: Wind.FPV * 4,
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x3" },  // world center
              { shaderLocation: 1, offset: 12, format: "float32x2" }, // quad corner ±1 (along, perp)
              { shaderLocation: 2, offset: 20, format: "float32" },   // speedFrac
              { shaderLocation: 3, offset: 24, format: "float32x2" }, // windDir.xz at the mote
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
      primitive: { topology: "triangle-list", cullMode: "none" },
      // depth-test (no write) so terrain ridges occlude the motes but they don't z-fight each other.
      depthStencil: { depthWriteEnabled: false, depthCompare: "less", format: "depth24plus" },
    });
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.ubuf } }],
    });
  }

  // Place one mote at a fresh world position within the camera-relative span (random ahead+lateral).
  // Used to seed the whole field at start and to recycle motes that drift out.
  private spawn(
    i: number,
    camGround: [number, number],
    camFwd: [number, number],
    camRight: [number, number],
    aheadMin: number
  ): void {
    const ahead = aheadMin + Math.random() * (this.spanAhead - aheadMin);
    const lateral = (Math.random() * 2 - 1) * this.spanWide;
    this.px[i] = camGround[0] + camFwd[0] * ahead + camRight[0] * lateral;
    this.pz[i] = camGround[1] + camFwd[1] * ahead + camRight[1] * lateral;
  }

  // Advect every mote by windAt (p += w*dt), recycle ones that leave the camera-relative span, then
  // rebuild the billboard vertex buffer from the PERSISTED positions. dt is derived from sim time and
  // clamped so the first frame and any tab-stall don't fling the motes.
  private step(
    camGround: [number, number],
    camFwd: [number, number],
    camRight: [number, number],
    t: number
  ): void {
    // seed the field once, spread across the whole span so frame-0 is already populated and mid-flow.
    if (!this.seeded) {
      for (let i = 0; i < this.count; i++) {
        this.spawn(i, camGround, camFwd, camRight, -this.spanBehind);
      }
      this.seeded = true;
    }

    let dt = this.lastTime < 0 ? 0 : t - this.lastTime;
    this.lastTime = t;
    if (dt < 0) dt = 0;
    if (dt > 0.05) dt = 0.05; // clamp stalls

    const refSpeed = this.cfg.curlAmp + this.cfg.driftAmp; // for speedFrac normalization
    const v = this.vertHost;
    const corners = Wind.CORNERS;
    let vi = 0;
    for (let i = 0; i < this.count; i++) {
      let x = this.px[i]!;
      let z = this.pz[i]!;

      // advect by the SAME shared field (no amplification — drift accumulates over the mote lifetime).
      const [wx, wz] = windAt(x, z, t, this.cfg);
      x += wx * dt;
      z += wz * dt;

      // recycle: project onto the camera-relative basis; respawn ahead if out of the span.
      const rx = x - camGround[0];
      const rz = z - camGround[1];
      const fwdDist = rx * camFwd[0] + rz * camFwd[1];   // along view
      const sideDist = rx * camRight[0] + rz * camRight[1]; // lateral
      if (fwdDist < -this.spanBehind || fwdDist > this.spanAhead || Math.abs(sideDist) > this.spanWide) {
        // reseed near the FAR edge of the span so the field stays full as the camera advances.
        this.spawn(i, camGround, camFwd, camRight, this.spanAhead * 0.55);
        x = this.px[i]!;
        z = this.pz[i]!;
      } else {
        this.px[i] = x;
        this.pz[i] = z;
      }

      const wspeed = Math.hypot(wx, wz);
      const sp = Math.min(1, wspeed / (refSpeed * 1.4));
      this.speedFrac[i] = sp;
      const y = this.sampleHeight(x, z) + this.clearance;
      // unit wind direction (world XZ) → orients the comet streak in screen space (VS projects it).
      const inv = wspeed > 1e-4 ? 1 / wspeed : 0;
      const wdx = wx * inv;
      const wdz = wz * inv;

      // emit two triangles (6 verts) — same world center+windDir, four corners.
      for (let c = 0; c < 6; c++) {
        const [cx, cy] = corners[c]!;
        v[vi++] = x; v[vi++] = y; v[vi++] = z;
        v[vi++] = cx; v[vi++] = cy;
        v[vi++] = sp;
        v[vi++] = wdx; v[vi++] = wdz;
      }
    }
    this.device.queue.writeBuffer(this.vbuf, 0, this.vertBytes);
  }

  // SECOND pass: LOAD terrain color+depth (no clear); draw the drifting dot motes over the ridges.
  // Signature is UNCHANGED from the streamline version so the bird-main.ts call site is untouched.
  draw(
    encoder: GPUCommandEncoder,
    colorView: GPUTextureView,
    depthView: GPUTextureView,
    viewProj: Float32Array,
    camGround: [number, number],
    camFwd: [number, number],
    camRight: [number, number],
    eye: [number, number, number],
    time: number,
    fogColor: [number, number, number],
    fogDensity: number,
    aspect = 1
  ): void {
    this.step(camGround, camFwd, camRight, time);

    // dotPx (screen px diameter) → NDC half-extent: full clip span is 2 across pxW. The VS scales the
    // x offset by clip.w only and the y offset by clip.w*aspect, so this half-extent uses the X axis.
    // (pxW is recovered as aspect maps x/y; pass an effective px width via aspect = pxW/pxH and a fixed
    //  reference width baked here.) Use viewport-independent sizing: half-extent = dotPx / refWidth.
    const REF_W = 1000; // reference canvas width the dotPx was tuned against
    const dotSize = this.dotPx / REF_W;          // NDC half-width of the comet head
    const tailLen = dotSize * this.tailMul;      // NDC streak length (head → tail)

    const u = this.uniformF32;
    u.set(viewProj, 0);
    u[16] = eye[0]; u[17] = eye[1]; u[18] = eye[2];
    u[19] = aspect;                               // pxW/pxH for un-skewed streaks
    u[20] = fogColor[0]; u[21] = fogColor[1]; u[22] = fogColor[2];
    u[23] = fogDensity;
    u[24] = dotSize;                              // NDC half-width
    u[25] = tailLen;                              // NDC streak length
    this.device.queue.writeBuffer(this.ubuf, 0, this.uniformHost);

    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: colorView, loadOp: "load", storeOp: "store" }],
      depthStencilAttachment: { view: depthView, depthLoadOp: "load", depthStoreOp: "store" },
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vbuf);
    pass.draw(this.vertexCount);
    pass.end();
  }
}
