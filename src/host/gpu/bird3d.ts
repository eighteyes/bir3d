// bird3d.ts — Bird3D: one CPU-integrated 3D bird + neon flapping-V render pipeline (depth-tested).
// Responsibilities:
//   - Hold bird state {pos:vec3, speed (scalar airspeed), heading, pitch, bank}; integrate(dt, input):
//     energy-exchange glider — pitch trades airspeed for altitude (dive to gain speed, pull up to
//     zoom-climb), drag relaxes speed toward trim, base sink rate (mushes when slow), RIDGE LIFT as
//     vertical AIR MOTION the bird rides (wind · uphill gradient, finite-diff of sampleHeight),
//     horizontal wind drift (analytic curl-noise — FLAGGED stand-in for fluid sampling), flap as a
//     decaying boost (speed kick + vertical surge), altitude clamp above ground.
//   - Expose public `tuning` (live-writable) so a host overlay can slide feel parameters at runtime.
//   - Build a procedural V mesh (body + two swept dihedral wing ribbons) as triangle strips packed
//     to a triangle list; each vertex carries (signed spanFrac, wingFlag, edgeFrac) for shader flap.
//   - Own the bird uniform + vertex buffer + render pipeline (bird3d.wgsl) WITH depthStencil
//     (depth24plus, less, write on) so terrain ridges occlude the bird.
//   - draw(encoder, colorView, depthView, viewProj, time): record one bird draw in a SECOND pass
//     that LOADS the terrain color+depth (no clear) — terrain must already be drawn this encoder.
//   - forwardVec()/heading expose the chase convention (+Z forward) for the camera.

import type { TerrainEKG } from "./terrain";

export interface BirdInput {
  yawRate: number;   // rad/s, from mouse-x offset
  pitchRate: number; // rad/s, from mouse-y offset
}

export interface BirdTuning {
  glideSpeed?: number;   // trim airspeed (m/s) — drag relaxes speed toward this
  minSpeed?: number;     // stall floor (m/s)
  maxSpeed?: number;     // dive ceiling (m/s)
  dragK?: number;        // per-second relaxation of airspeed toward trim
  divePower?: number;    // scale on gravity-along-path energy exchange (dive↔zoom)
  gravity?: number;      // m/s^2 — only enters via sin(pitch) energy exchange + sink
  sinkRate?: number;     // base sink at trim speed (m/s); scales (trim/speed)^2 when slow
  windGain?: number;     // analytic wind push scale
  windDrift?: number;    // fraction of horizontal wind the bird drifts with
  liftGain?: number;     // ridge updraft scale (vertical air-motion m/s per unit wind·slope)
  flexHz?: number;       // subtle wing-flex frequency (NOT a flap beat)
  flexAmp?: number;      // subtle wing-flex amplitude (rad) — wings held OUT, no flap cycle
  minClearance?: number; // min meters above terrain
}

type Vec3 = [number, number, number];

const FLOATS_PER_VERT = 6; // local.xyz + attr.xyz
const UNIFORM_BYTES = 96;  // mat4(64) + pos(12)+flapPhase(4) + heading,bank,flapHz,flapAmp(16) = 96

export class Bird3D {
  pos: Vec3;
  vel: Vec3 = [0, 0, 18];
  speed = 26;    // scalar airspeed (m/s) — the energy store
  heading = 0;   // yaw, +Z forward at 0
  pitch = 0;     // radians, + = nose up
  bank = 0;      // roll, banks into turns
  private time = 0;

  tuning: Required<BirdTuning>;

  // latest derived telemetry for the overlay
  lastWind: [number, number] = [0, 0];
  lastSpeed = 0;
  lastClearance = 0;
  lastVario = 0;   // vertical speed (m/s)
  lastUpdraft = 0; // ridge updraft being ridden (m/s)

  private vbuf: GPUBuffer;
  private vertexCount: number;
  private ubuf: GPUBuffer;
  private uniformHost: ArrayBuffer;
  private uniformF32: Float32Array;
  private pipeline: GPURenderPipeline;
  private bindGroup: GPUBindGroup;

  constructor(
    private device: GPUDevice,
    shader: string,
    colorFormat: GPUTextureFormat,
    private terrain: TerrainEKG,
    startPos: Vec3 = [0, 200, 0],
    t: BirdTuning = {}
  ) {
    this.pos = startPos;
    this.tuning = {
      glideSpeed: t.glideSpeed ?? 26,
      minSpeed: t.minSpeed ?? 13,
      maxSpeed: t.maxSpeed ?? 55,
      dragK: t.dragK ?? 0.4,
      divePower: t.divePower ?? 0.9,
      gravity: t.gravity ?? 9.0,
      sinkRate: t.sinkRate ?? 1.4,
      windGain: t.windGain ?? 6,
      windDrift: t.windDrift ?? 1.0,
      liftGain: t.liftGain ?? 2.2,
      flexHz: t.flexHz ?? 0.6,   // slow, subtle flex — wings stay OUT (no flap beat)
      flexAmp: t.flexAmp ?? 0.06, // tiny → static gliding V
      minClearance: t.minClearance ?? 6,
    };

    const meshArr = buildVMesh(); // number[]
    this.vertexCount = meshArr.length / FLOATS_PER_VERT;
    const mesh = new Float32Array(new ArrayBuffer(meshArr.length * 4));
    mesh.set(meshArr);
    this.vbuf = device.createBuffer({
      size: mesh.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.vbuf, 0, mesh);

    this.uniformHost = new ArrayBuffer(UNIFORM_BYTES);
    this.uniformF32 = new Float32Array(this.uniformHost);
    this.ubuf = device.createBuffer({
      size: UNIFORM_BYTES,
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
            arrayStride: FLOATS_PER_VERT * 4,
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x3" },
              { shaderLocation: 1, offset: 12, format: "float32x3" },
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
            // additive blend → neon ribbons bloom over the dark terrain.
            blend: {
              color: { srcFactor: "one", dstFactor: "one", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      // depth-test against the stored terrain depth so ridges occlude the bird.
      depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
    });
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.ubuf } }],
    });
  }

  forwardVec(): Vec3 {
    return [Math.sin(this.heading), 0, Math.cos(this.heading)];
  }

  // Analytic curl-noise horizontal wind (FLAGGED: stand-in for the GPU fluid; fluid sampling
  // needs async readback which is deferred this pass). Divergence-free 2D flow over (x,z).
  private windAt(x: number, z: number): [number, number] {
    const s = 0.0016;
    const e = 1.0;
    const pot = (px: number, pz: number) =>
      Math.sin(px * s) * Math.cos(pz * s * 1.3) + 0.5 * Math.sin((px + pz) * s * 0.6);
    // wind = curl of scalar potential = (dPot/dz, -dPot/dx)
    const dz = (pot(x, z + e) - pot(x, z - e)) / (2 * e);
    const dx = (pot(x + e, z) - pot(x - e, z)) / (2 * e);
    return [dz * this.tuning.windGain * 200, -dx * this.tuning.windGain * 200];
  }

  integrate(dt: number, input: BirdInput): void {
    this.time += dt;
    const clamped = Math.min(dt, 1 / 20);
    const T = this.tuning;

    // --- steering: mouse offsets drive yaw & pitch rate; bank eases toward yaw rate ---
    this.heading += input.yawRate * clamped;
    this.pitch += input.pitchRate * clamped;
    this.pitch = Math.max(-0.7, Math.min(0.7, this.pitch));
    const targetBank = -input.yawRate * 0.5; // roll into the turn
    this.bank += (targetBank - this.bank) * Math.min(1, clamped * 4);

    const fwd = this.forwardVec();
    const dir: Vec3 = [
      fwd[0] * Math.cos(this.pitch),
      Math.sin(this.pitch),
      fwd[2] * Math.cos(this.pitch),
    ];

    // --- airspeed energy exchange: gravity along the flight path + drag toward trim ---
    // pitch down → speed builds; pull up → speed bleeds into climb (zoom). This is the soar.
    this.speed +=
      (-T.gravity * Math.sin(this.pitch) * T.divePower -
        T.dragK * (this.speed - T.glideSpeed)) *
      clamped;

    // GLIDE, NO FLAP: no thrust input this pass. Airspeed is sustained by the dive↔zoom energy
    // exchange above and bled by drag toward trim; lift comes from glide + ridge updraft below.
    this.speed = Math.max(T.minSpeed, Math.min(T.maxSpeed, this.speed));

    // --- wind + ridge updraft: vertical air motion the bird RIDES ---
    const [wx, wz] = this.windAt(this.pos[0], this.pos[2]);
    this.lastWind = [wx, wz];
    const eps = 6;
    const hC = this.terrain.sampleHeight(this.pos[0], this.pos[2]);
    const hX = this.terrain.sampleHeight(this.pos[0] + eps, this.pos[2]);
    const hZ = this.terrain.sampleHeight(this.pos[0], this.pos[2] + eps);
    const gx = (hX - hC) / eps; // uphill gradient
    const gz = (hZ - hC) / eps;
    const into = wx * gx + wz * gz; // wind · uphill
    const updraft = Math.max(0, into) * T.liftGain;
    this.lastUpdraft = updraft;

    // --- sink: minimal at trim, mushes quadratically when slow (stall teaches itself) ---
    const sink = T.sinkRate * (T.glideSpeed / this.speed) ** 2;

    // --- compose velocity: flight path + horizontal wind drift + ridge updraft − sink ---
    this.vel[0] = dir[0] * this.speed + wx * T.windDrift;
    this.vel[1] = dir[1] * this.speed + updraft - sink;
    this.vel[2] = dir[2] * this.speed + wz * T.windDrift;

    this.pos[0] += this.vel[0] * clamped;
    this.pos[1] += this.vel[1] * clamped;
    this.pos[2] += this.vel[2] * clamped;

    // --- altitude clamp above terrain ---
    const floorY = hC + T.minClearance;
    if (this.pos[1] < floorY) this.pos[1] = floorY;

    this.lastSpeed = this.speed;
    this.lastVario = this.vel[1];
    this.lastClearance = this.pos[1] - hC;
  }

  draw(
    encoder: GPUCommandEncoder,
    colorView: GPUTextureView,
    depthView: GPUTextureView,
    viewProj: Float32Array
  ): void {
    const u = this.uniformF32;
    u.set(viewProj, 0);                 // [0..16)
    u[16] = this.pos[0]; u[17] = this.pos[1]; u[18] = this.pos[2];
    u[19] = this.time * this.tuning.flexHz * Math.PI * 2; // flexPhase (subtle, not a flap beat)
    u[20] = this.heading;
    u[21] = this.bank;
    u[22] = this.tuning.flexHz;
    u[23] = this.tuning.flexAmp;
    this.device.queue.writeBuffer(this.ubuf, 0, this.uniformHost);

    // SECOND pass: LOAD terrain color+depth (no clear) so the bird composites over the ridges
    // and the stored depth occludes it behind crests.
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: colorView, loadOp: "load", storeOp: "store" }],
      depthStencilAttachment: {
        view: depthView,
        depthLoadOp: "load",
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

// --- procedural V mesh: body spine + two swept dihedral wing ribbons (triangle list) ---
// Local frame: +X right, +Y up, +Z forward (heading). Wings sweep back (-Z) toward the tips.
function buildVMesh(): number[] {
  const verts: number[] = [];
  const SPAN = 18;     // half-wingspan (m) → ~36m tip-to-tip, bold + readable at followDist 120
  const SWEEP = 6;     // how far back the tip sits (-Z)
  const DIHEDRAL = 9;  // tip rise (m) at full span → clear static gliding V
  const RIBBON = 3.2;  // ribbon half-width (m) — bold neon, not a hairline
  const BODY_LEN = 11; // body spine length (m)
  const BODY_W = 2.0;

  // push a quad (two tris) as a ribbon between two centerline points pA,pB.
  // attr = (signed spanFrac, wingFlag, edgeFrac); edgeFrac 0/1 marks ribbon edges.
  const quad = (
    pA: Vec3, pB: Vec3, halfW: number, axis: Vec3,
    spanA: number, spanB: number, wing: number
  ) => {
    const a0: Vec3 = [pA[0] - axis[0] * halfW, pA[1] - axis[1] * halfW, pA[2] - axis[2] * halfW];
    const a1: Vec3 = [pA[0] + axis[0] * halfW, pA[1] + axis[1] * halfW, pA[2] + axis[2] * halfW];
    const b0: Vec3 = [pB[0] - axis[0] * halfW, pB[1] - axis[1] * halfW, pB[2] - axis[2] * halfW];
    const b1: Vec3 = [pB[0] + axis[0] * halfW, pB[1] + axis[1] * halfW, pB[2] + axis[2] * halfW];
    const v = (p: Vec3, span: number, edge: number) =>
      verts.push(p[0], p[1], p[2], span, wing, edge);
    // tri 1: a0,a1,b0   tri 2: b0,a1,b1
    v(a0, spanA, 0); v(a1, spanA, 1); v(b0, spanB, 0);
    v(b0, spanB, 0); v(a1, spanA, 1); v(b1, spanB, 1);
  };

  // --- body spine ribbon (along Z), spanFrac 0, wing flag 0 ---
  quad([0, 0, BODY_LEN * 0.5], [0, 0, -BODY_LEN * 0.5], BODY_W, [1, 0, 0], 0, 0, 0);

  // --- each wing: a swept ribbon from root (near body) to tip; subdivided so flap bends it ---
  const SEGS = 4;
  for (const side of [-1, 1]) {
    let prev: Vec3 = [0, 0, 0];
    let prevSpan = 0;
    for (let i = 1; i <= SEGS; i++) {
      const f = i / SEGS;
      const x = side * SPAN * f;
      const z = -SWEEP * f;           // sweep back toward tip
      const y = DIHEDRAL * f;         // base dihedral rise → static V even mid-flap
      const cur: Vec3 = [x, y, z];
      const span = side * f;          // signed spanFrac -1..1
      // ribbon width axis = forward (Z): gives the swept wing a flat chord facing the camera.
      quad(prev, cur, RIBBON, [0, 0, 1], prevSpan, span, 1);
      prev = cur; prevSpan = span;
    }
  }

  return verts;
}
