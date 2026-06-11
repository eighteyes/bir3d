// bird3d.ts — Bird3D: one CPU-integrated 3D bird + neon flapping-V render pipeline (depth-tested).
// Responsibilities:
//   - Hold bird state {pos:vec3, vel:vec3, heading, pitch, bank}; integrate(dt, input, terrain):
//     forward glide + drag, flap thrust+lift, gravity -Y, horizontal wind push (analytic curl-noise
//     this pass — FLAGGED; fluid sampling needs async readback, deferred), RIDGE LIFT from the
//     terrain uphill gradient (finite-diff of Terrain3D.sampleHeight), altitude clamp above ground.
//   - Build a procedural V mesh (body + two swept dihedral wing ribbons) as triangle strips packed
//     to a triangle list; each vertex carries (signed spanFrac, wingFlag, edgeFrac) for shader flap.
//   - Own the bird uniform + vertex buffer + render pipeline (bird3d.wgsl) WITH depthStencil
//     (depth24plus, less, write on) so terrain ridges occlude the bird.
//   - draw(encoder, colorView, depthView, viewProj, time): record one bird draw in a SECOND pass
//     that LOADS the terrain color+depth (no clear) — terrain must already be drawn this encoder.
//   - forwardVec()/heading expose the chase convention (+Z forward) for the camera.

import type { Terrain3D } from "./terrain";

export interface BirdInput {
  yawRate: number;   // rad/s, from mouse-x offset
  pitchRate: number; // rad/s, from mouse-y offset
  flap: boolean;     // click/Space → flap impulse this frame
}

export interface BirdTuning {
  glideSpeed?: number;   // baseline forward airspeed (m/s)
  drag?: number;         // velocity damping per second
  gravity?: number;      // m/s^2 downward
  flapThrust?: number;   // forward+up impulse per flap (m/s)
  flapLift?: number;     // vertical component of a flap (m/s)
  windGain?: number;     // analytic wind push scale
  liftGain?: number;     // ridge-lift scale
  flapHz?: number;       // wingbeat frequency
  flapAmp?: number;      // max flap angle (rad)
  minClearance?: number; // min meters above terrain
}

type Vec3 = [number, number, number];

const FLOATS_PER_VERT = 6; // local.xyz + attr.xyz
const UNIFORM_BYTES = 96;  // mat4(64) + pos(12)+flapPhase(4) + heading,bank,flapHz,flapAmp(16) = 96

export class Bird3D {
  pos: Vec3;
  vel: Vec3 = [0, 0, 18];
  heading = 0;   // yaw, +Z forward at 0
  pitch = 0;     // radians, + = nose up
  bank = 0;      // roll, banks into turns
  private time = 0;

  private glideSpeed: number;
  private drag: number;
  private gravity: number;
  private flapThrust: number;
  private flapLift: number;
  private windGain: number;
  private liftGain: number;
  private flapHz: number;
  private flapAmp: number;
  private minClearance: number;

  // latest derived telemetry for the overlay
  lastWind: [number, number] = [0, 0];
  lastSpeed = 0;
  lastClearance = 0;

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
    private terrain: Terrain3D,
    startPos: Vec3 = [0, 200, 0],
    t: BirdTuning = {}
  ) {
    this.pos = startPos;
    this.glideSpeed = t.glideSpeed ?? 26;
    this.drag = t.drag ?? 0.6;
    this.gravity = t.gravity ?? 9.0;
    this.flapThrust = t.flapThrust ?? 14;
    this.flapLift = t.flapLift ?? 16;
    this.windGain = t.windGain ?? 6;
    this.liftGain = t.liftGain ?? 0.5;
    this.flapHz = t.flapHz ?? 2.4;
    this.flapAmp = t.flapAmp ?? 0.55; // shallower beat → reads as a flapping V, not a deep U
    this.minClearance = t.minClearance ?? 20;

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
    return [dz * this.windGain * 200, -dx * this.windGain * 200];
  }

  integrate(dt: number, input: BirdInput, _terrain?: Terrain3D): void {
    this.time += dt;
    const clamped = Math.min(dt, 1 / 20);

    // --- steering: mouse offsets drive yaw & pitch rate; bank eases toward yaw rate ---
    this.heading += input.yawRate * clamped;
    this.pitch += input.pitchRate * clamped;
    this.pitch = Math.max(-0.7, Math.min(0.7, this.pitch));
    const targetBank = -input.yawRate * 0.5; // roll into the turn
    this.bank += (targetBank - this.bank) * Math.min(1, clamped * 4);

    const fwd = this.forwardVec();

    // --- forward glide toward target airspeed along heading+pitch ---
    const dir: Vec3 = [
      fwd[0] * Math.cos(this.pitch),
      Math.sin(this.pitch),
      fwd[2] * Math.cos(this.pitch),
    ];
    // accelerate velocity toward glide along the look direction
    const accel = 6;
    this.vel[0] += (dir[0] * this.glideSpeed - this.vel[0]) * Math.min(1, clamped * accel * 0.16);
    this.vel[1] += (dir[1] * this.glideSpeed - this.vel[1]) * Math.min(1, clamped * accel * 0.16);
    this.vel[2] += (dir[2] * this.glideSpeed - this.vel[2]) * Math.min(1, clamped * accel * 0.16);

    // --- flap: forward thrust + vertical lift impulse ---
    if (input.flap) {
      this.vel[0] += dir[0] * this.flapThrust;
      this.vel[2] += dir[2] * this.flapThrust;
      this.vel[1] += this.flapLift;
    }

    // --- gravity ---
    this.vel[1] -= this.gravity * clamped;

    // --- horizontal wind push (analytic curl-noise, FLAGGED) ---
    const [wx, wz] = this.windAt(this.pos[0], this.pos[2]);
    this.lastWind = [wx, wz];
    this.vel[0] += wx * clamped;
    this.vel[2] += wz * clamped;

    // --- RIDGE LIFT: wind into an uphill slope → upward velocity ---
    const eps = 6;
    const hC = this.terrain.sampleHeight(this.pos[0], this.pos[2]);
    const hX = this.terrain.sampleHeight(this.pos[0] + eps, this.pos[2]);
    const hZ = this.terrain.sampleHeight(this.pos[0], this.pos[2] + eps);
    // uphill gradient (points toward rising terrain)
    const gx = (hX - hC) / eps;
    const gz = (hZ - hC) / eps;
    const into = wx * gx + wz * gz; // wind · uphill
    const lift = Math.max(0, into) * this.liftGain;
    this.vel[1] += lift * clamped;

    // --- drag ---
    const d = Math.max(0, 1 - this.drag * clamped);
    this.vel[0] *= d; this.vel[1] *= d; this.vel[2] *= d;

    // --- integrate position ---
    this.pos[0] += this.vel[0] * clamped;
    this.pos[1] += this.vel[1] * clamped;
    this.pos[2] += this.vel[2] * clamped;

    // --- altitude clamp above terrain ---
    const floorY = hC + this.minClearance;
    if (this.pos[1] < floorY) {
      this.pos[1] = floorY;
      if (this.vel[1] < 0) this.vel[1] = 0;
    }

    this.lastSpeed = Math.hypot(this.vel[0], this.vel[1], this.vel[2]);
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
    u[19] = this.time * this.flapHz * Math.PI * 2; // flapPhase
    u[20] = this.heading;
    u[21] = this.bank;
    u[22] = this.flapHz;
    u[23] = this.flapAmp;
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
  const SPAN = 12;     // half-wingspan (m) → ~24m tip-to-tip, reads at followDist 80
  const SWEEP = 5;     // how far back the tip sits (-Z)
  const DIHEDRAL = 7;  // tip rise (m) at full span → clear static V even mid-flap
  const RIBBON = 1.8;  // ribbon half-width (m)
  const BODY_LEN = 8;  // body spine length (m)
  const BODY_W = 1.2;

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
