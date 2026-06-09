// fluid.ts — GpuFluid: GPU Stam fluid solver (compute only; no readback, no viz).
// Ports crates/vs-core/src/fluid (Plan 2 oracle) to WGSL compute over the host convention.
// Responsibilities:
//   - Allocate storage buffers in the exact (W+2)*(H+2) bordered layout (u,v,dye ping-pong;
//     p ping-pong; div; params uniform), build one pipeline per kernel/entry via makeComputePipeline.
//   - step(encoder, dt, iters): record the full Stam pass sequence into the CALLER's encoder
//     (one submit/frame) via encodeComputePass — add-forces -> set_bnd(vel) -> project ->
//     set_bnd(vel) -> self-advect u,v -> set_bnd(vel) -> project -> advect dye -> set_bnd(scalar).
//   - Hold/expose current buffers + params for downstream readback/viz; NO readback in step().

import { makeComputePipeline, encodeComputePass } from "./dispatch";
import { PingPong } from "./pingpong";

// Params uniform layout (std140-friendly, 48 bytes, 16-byte aligned). Field order MUST match
// the `struct Params` declared in every fluid WGSL kernel.
//   [0]=w(u32) [1]=h(u32) [2]=dt(f32) [3]=pad
//   [4]=fx     [5]=fy     [6]=dye_x   [7]=dye_y
//   [8]=dye_r  [9]=dye_amt [10]=force_r [11]=pad
const PARAMS_FLOATS = 12;
const PARAMS_BYTES = PARAMS_FLOATS * 4;

export interface ForceParams {
  fx: number;
  fy: number;
  dyeX: number;
  dyeY: number;
  dyeR: number;
  dyeAmt: number;
  forceR: number;
}

const ZERO_FORCE: ForceParams = { fx: 0, fy: 0, dyeX: 0, dyeY: 0, dyeR: 0, dyeAmt: 0, forceR: 0 };

interface FluidShaders {
  forces: string;
  divergence: string;
  jacobi: string;
  subtractGrad: string;
  advect: string;
  setBnd: string;
}

export class GpuFluid {
  readonly w: number;
  readonly h: number;
  readonly cells: number; // (w+2)*(h+2)
  readonly bytes: number; // cells*4

  private device: GPUDevice;

  // Storage buffers (all (W+2)*(H+2)*4 bytes).
  private u: PingPong<GPUBuffer>;
  private v: PingPong<GPUBuffer>;
  private dye: PingPong<GPUBuffer>;
  private p: PingPong<GPUBuffer>;
  private div: GPUBuffer;
  private params: GPUBuffer;

  // Scratch for per-step uniform upload (w/h fixed; dt/forces vary).
  private paramsHost = new ArrayBuffer(PARAMS_BYTES);
  private paramsU32 = new Uint32Array(this.paramsHost);
  private paramsF32 = new Float32Array(this.paramsHost);

  private force: ForceParams = { ...ZERO_FORCE };

  // Pipelines.
  private pForces: GPUComputePipeline;
  private pDivergence: GPUComputePipeline;
  private pJacobi: GPUComputePipeline;
  private pSubtractGrad: GPUComputePipeline;
  private pAdvect: GPUComputePipeline;
  private pScalarEdges: GPUComputePipeline;
  private pScalarCorners: GPUComputePipeline;
  private pVelxEdges: GPUComputePipeline;
  private pVelxCorners: GPUComputePipeline;
  private pVelyEdges: GPUComputePipeline;
  private pVelyCorners: GPUComputePipeline;

  constructor(device: GPUDevice, w: number, h: number, shaders: FluidShaders) {
    this.device = device;
    this.w = w;
    this.h = h;
    this.cells = (w + 2) * (h + 2);
    this.bytes = this.cells * 4;

    const storage = () =>
      device.createBuffer({
        size: this.bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });

    this.u = new PingPong(storage(), storage());
    this.v = new PingPong(storage(), storage());
    this.dye = new PingPong(storage(), storage());
    this.p = new PingPong(storage(), storage());
    this.div = storage();
    this.params = device.createBuffer({
      size: PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // w/h are constant for the solver's lifetime; write once.
    this.paramsU32[0] = w;
    this.paramsU32[1] = h;

    this.pForces = makeComputePipeline(device, shaders.forces);
    this.pDivergence = makeComputePipeline(device, shaders.divergence);
    this.pJacobi = makeComputePipeline(device, shaders.jacobi);
    this.pSubtractGrad = makeComputePipeline(device, shaders.subtractGrad);
    this.pAdvect = makeComputePipeline(device, shaders.advect);
    this.pScalarEdges = makeComputePipeline(device, shaders.setBnd, "scalar_edges");
    this.pScalarCorners = makeComputePipeline(device, shaders.setBnd, "scalar_corners");
    this.pVelxEdges = makeComputePipeline(device, shaders.setBnd, "velx_edges");
    this.pVelxCorners = makeComputePipeline(device, shaders.setBnd, "velx_corners");
    this.pVelyEdges = makeComputePipeline(device, shaders.setBnd, "vely_edges");
    this.pVelyCorners = makeComputePipeline(device, shaders.setBnd, "vely_corners");
  }

  /** Set the scripted force / dye-injection source applied each step (until changed). */
  setForce(force: Partial<ForceParams>): void {
    this.force = { ...this.force, ...force };
  }

  /** Current (most-recently-written) buffers — for one-shot readback / viz OUTSIDE the loop. */
  get velocityX(): GPUBuffer { return this.u.current; }
  get velocityY(): GPUBuffer { return this.v.current; }
  get dyeField(): GPUBuffer { return this.dye.current; }
  get pressure(): GPUBuffer { return this.p.current; }
  get divergence(): GPUBuffer { return this.div; }
  get paramsBuffer(): GPUBuffer { return this.params; }

  /** Upload the per-step params (dt + current force). queue op orders before submitted commands. */
  private uploadParams(dt: number): void {
    this.paramsF32[2] = dt;
    this.paramsF32[4] = this.force.fx;
    this.paramsF32[5] = this.force.fy;
    this.paramsF32[6] = this.force.dyeX;
    this.paramsF32[7] = this.force.dyeY;
    this.paramsF32[8] = this.force.dyeR;
    this.paramsF32[9] = this.force.dyeAmt;
    this.paramsF32[10] = this.force.forceR;
    this.device.queue.writeBuffer(this.params, 0, this.paramsHost);
  }

  private pass(encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, bindings: GPUBuffer[]): void {
    encodeComputePass(this.device, encoder, pipeline, bindings, this.cells);
  }

  /** set_bnd on velocity: velx on u, vely on v — each two-pass (edges then corners). */
  private setBndVel(encoder: GPUCommandEncoder): void {
    this.pass(encoder, this.pVelxEdges, [this.params, this.u.current]);
    this.pass(encoder, this.pVelxCorners, [this.params, this.u.current]);
    this.pass(encoder, this.pVelyEdges, [this.params, this.v.current]);
    this.pass(encoder, this.pVelyCorners, [this.params, this.v.current]);
  }

  /** set_bnd(Scalar) on an arbitrary field buffer — two-pass. */
  private setBndScalar(encoder: GPUCommandEncoder, field: GPUBuffer): void {
    this.pass(encoder, this.pScalarEdges, [this.params, field]);
    this.pass(encoder, this.pScalarCorners, [this.params, field]);
  }

  /**
   * project: divergence -> (jacobi sweep -> set_bnd scalar) x iters -> subtract_grad.
   * Reads/writes current u,v in place; uses div + p ping-pong (cleared to a zero pressure guess).
   */
  private project(encoder: GPUCommandEncoder, iters: number): void {
    // div = divergence(u, v).
    this.pass(encoder, this.pDivergence, [this.params, this.u.current, this.v.current, this.div]);

    // Zero pressure guess (both ping-pong halves).
    encoder.clearBuffer(this.p.current);
    encoder.clearBuffer(this.p.next);

    for (let it = 0; it < iters; it++) {
      // p_next = jacobi(p, div); reads prev, writes next.
      this.pass(encoder, this.pJacobi, [this.params, this.p.current, this.div, this.p.next]);
      this.p.swap();
      // set_bnd(Scalar) on the freshly written pressure (now current).
      this.setBndScalar(encoder, this.p.current);
    }

    // u,v -= grad(p) using the final pressure (p.current).
    this.pass(encoder, this.pSubtractGrad, [this.params, this.p.current, this.u.current, this.v.current]);
    this.setBndVel(encoder);
  }

  /** Self-advect u,v through the pre-advection (u,v) snapshot, then swap both together. */
  private advectVelocity(encoder: GPUCommandEncoder, _dt: number): void {
    const u0 = this.u.current;
    const v0 = this.v.current;
    // advect u: src=u0, vel=(u0,v0) -> u.next.
    this.pass(encoder, this.pAdvect, [this.params, u0, u0, v0, this.u.next]);
    // advect v: src=v0, vel=(u0,v0) -> v.next. Both read the SAME pre-advection velocity.
    this.pass(encoder, this.pAdvect, [this.params, v0, u0, v0, this.v.next]);
    this.u.swap();
    this.v.swap();
  }

  /**
   * Record one Stam fluid step into the caller's encoder (one submit/frame; no readback).
   * Sequence (spec-locked): add-forces -> set_bnd(vel) -> project -> set_bnd(vel) ->
   * self-advect u,v -> set_bnd(vel) -> project -> advect dye -> set_bnd(scalar, dye).
   */
  step(encoder: GPUCommandEncoder, dt: number, iters: number): void {
    this.uploadParams(dt);

    // add-forces + dye injection (interior, in place).
    this.pass(encoder, this.pForces, [this.params, this.u.current, this.v.current, this.dye.current]);
    this.setBndVel(encoder);

    // project the post-force field.
    this.project(encoder, iters);
    this.setBndVel(encoder);

    // self-advect velocity, then boundaries.
    this.advectVelocity(encoder, dt);
    this.setBndVel(encoder);

    // final projection.
    this.project(encoder, iters);

    // advect the dye through the (now divergence-free) velocity, then scalar boundaries.
    const dye0 = this.dye.current;
    this.pass(encoder, this.pAdvect, [this.params, dye0, this.u.current, this.v.current, this.dye.next]);
    this.dye.swap();
    this.setBndScalar(encoder, this.dye.current);
  }

  /** Release all GPU buffers. */
  destroy(): void {
    for (const b of [
      this.u.current, this.u.next, this.v.current, this.v.next,
      this.dye.current, this.dye.next, this.p.current, this.p.next,
      this.div, this.params,
    ]) {
      b.destroy();
    }
  }
}
