// wind.ts — shared analytic wind field + Wind streamline overlay (neon drifting traces).
// Responsibilities:
//   - windAt(x,z,t): SINGLE SOURCE OF TRUTH for the wind vector at any world XZ + time. A
//     divergence-free curl-noise flow plus a slow large-scale drift, CRANKED so the lateral
//     component is unmistakable (~8-12 m/s vs the bird's ~26 m/s forward → clear cross-track).
//     Closed-form so the SAME field drives the bird physics (CPU), this streamline overlay (CPU
//     vertex build), and the overlay compass — zero GPU sync. FLAGGED: stand-in for the GPU fluid
//     (src/host/gpu/fluid.ts), which is compute-only and would need frame-laggy async readback +
//     a grid→world mapping and STILL need a closed form for streamlines/overlay; the analytic field
//     keeps one coherent, arbitrarily-sampleable source so the field you SEE is the field that PUSHES.
//   - Wind: a render pipeline (wind.wgsl) that draws animated neon streamline "comet" ribbons over
//     the terrain. Streamlines are integrated each frame from windAt along a camera-relative grid
//     ahead of the camera (so they follow the flight like the v4 EKG rows); comets scroll along the
//     static field lines to show direction + strength. Depth-tested against terrain, additive glow.
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
  thermalAmp: 3.0,
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
  const m = Math.max(0, a) * 0.7 + Math.max(0, b) * 0.5;
  return m * cfg.thermalAmp;
}

// --- streamline overlay pipeline ---

interface StreamParams {
  lines?: number;    // streamlines across the width
  steps?: number;    // integration steps per streamline (segments = steps-1)
  spanAhead?: number;// how far ahead of the camera the seed grid starts (m)
  spanWide?: number; // half-width of the seed grid (m)
  stepLen?: number;  // world meters advanced per integration step
}

export class Wind {
  private cfg: Required<WindConfig>;
  private lines: number;
  private steps: number;
  private spanAhead: number;
  private spanWide: number;
  private stepLen: number;

  private vbuf: GPUBuffer;
  private vertexCount: number;
  private vertBytes: ArrayBuffer;  // backing buffer uploaded each frame
  private vertHost: Float32Array;  // view over vertBytes, rebuilt each frame (camera-relative)
  private ubuf: GPUBuffer;
  private uniformHost: ArrayBuffer;
  private uniformF32: Float32Array;
  private pipeline: GPURenderPipeline;
  private bindGroup: GPUBindGroup;

  // floats/vertex: world.xyz(3) + arc(1, 0..1 along the streamline) + speedFrac(1) = 5
  private static FPV = 5;

  constructor(
    private device: GPUDevice,
    shader: string,
    colorFormat: GPUTextureFormat,
    private sampleHeight: (x: number, z: number) => number,
    cfg: WindConfig = {},
    p: StreamParams = {}
  ) {
    this.cfg = { ...DEFAULTS, ...cfg };
    this.lines = p.lines ?? 26;
    this.steps = p.steps ?? 40;
    this.spanAhead = p.spanAhead ?? -200;
    this.spanWide = p.spanWide ?? 1300;
    this.stepLen = p.stepLen ?? 36;

    const segsPerLine = this.steps - 1;
    this.vertexCount = this.lines * segsPerLine * 2; // line-list pairs
    this.vertBytes = new ArrayBuffer(this.vertexCount * Wind.FPV * 4);
    this.vertHost = new Float32Array(this.vertBytes);
    this.vbuf = device.createBuffer({
      size: this.vertHost.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    // uniform: mat4 viewProj(16) + eye.xyz(3) + phase(1) + fogColor.rgb(3) + fogDensity(1) = 24
    this.uniformHost = new ArrayBuffer(24 * 4);
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
              { shaderLocation: 0, offset: 0, format: "float32x3" },  // world xyz
              { shaderLocation: 1, offset: 12, format: "float32" },   // arc 0..1
              { shaderLocation: 2, offset: 16, format: "float32" },   // speedFrac
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
      // depth-test (no write) so terrain ridges occlude the wind traces but traces don't z-fight.
      depthStencil: { depthWriteEnabled: false, depthCompare: "less", format: "depth24plus" },
    });
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.ubuf } }],
    });
  }

  // Build camera-relative streamlines by integrating windAt from a seed grid ahead of the camera.
  // Each streamline rides ABOVE the terrain (clearance) so it reads as airflow, not a ground line.
  private rebuild(
    camGround: [number, number],
    camFwd: [number, number],
    camRight: [number, number],
    t: number
  ): void {
    const v = this.vertHost;
    const fpv = Wind.FPV;
    const CLEAR = 45; // meters above terrain the streamlines float
    const refSpeed = this.cfg.curlAmp + this.cfg.driftAmp; // for speedFrac normalization
    let vi = 0;
    for (let l = 0; l < this.lines; l++) {
      // seed across the camera-right axis, ahead of the camera ground point.
      const lateralFrac = this.lines > 1 ? (l / (this.lines - 1)) * 2 - 1 : 0;
      const lateral = lateralFrac * this.spanWide;
      let x = camGround[0] + camFwd[0] * this.spanAhead + camRight[0] * lateral;
      let z = camGround[1] + camFwd[1] * this.spanAhead + camRight[1] * lateral;
      // integrate the streamline forward through the field.
      let prevX = x, prevZ = z;
      let prevY = this.sampleHeight(prevX, prevZ) + CLEAR;
      let prevSpeed = Math.hypot(...windAt(prevX, prevZ, t, this.cfg));
      for (let s = 1; s < this.steps; s++) {
        const [wx, wz] = windAt(x, z, t, this.cfg);
        const wl = Math.hypot(wx, wz) || 1;
        x += (wx / wl) * this.stepLen;
        z += (wz / wl) * this.stepLen;
        const y = this.sampleHeight(x, z) + CLEAR;
        const arcPrev = (s - 1) / (this.steps - 1);
        const arc = s / (this.steps - 1);
        const sp = Math.min(1, prevSpeed / (refSpeed * 1.6));
        // segment prev→cur as a line-list pair
        v[vi++] = prevX; v[vi++] = prevY; v[vi++] = prevZ; v[vi++] = arcPrev; v[vi++] = sp;
        v[vi++] = x;     v[vi++] = y;     v[vi++] = z;     v[vi++] = arc;     v[vi++] = sp;
        prevX = x; prevZ = z; prevY = y;
        prevSpeed = Math.hypot(wx, wz);
      }
    }
    this.device.queue.writeBuffer(this.vbuf, 0, this.vertBytes);
  }

  // SECOND pass: LOAD terrain color+depth (no clear); draw streamline comets over the ridges.
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
    fogDensity: number
  ): void {
    this.rebuild(camGround, camFwd, camRight, time);

    const u = this.uniformF32;
    u.set(viewProj, 0);
    u[16] = eye[0]; u[17] = eye[1]; u[18] = eye[2];
    u[19] = time;                                 // comet scroll phase
    u[20] = fogColor[0]; u[21] = fogColor[1]; u[22] = fogColor[2];
    u[23] = fogDensity;
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
