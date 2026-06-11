// bird.ts — Bird module: GPU-integrated single bird gliding over a reused GpuFluid wind field.
// Owns the bird-state buffer, trail ring, intent uniform, CPU deadzone-follow camera, and the
// compute + render pipelines. No synchronous readback in the loop; an async pending-guarded copy
// of the 16-byte bird buffer feeds the CPU camera (1–2 frames stale is fine) and verification.
// Responsibilities:
//   - Allocate bird state {pos,vel}, trail ring (N vec2), intent uniform, bird/camera param uniforms.
//   - step(encoder, dt, intent): upload intent + bird params, rebuild the bird_update bind group from
//     the fluid's CURRENT u,v (they ping-pong inside fluid.step), record the compute pass, advance
//     the trail write index, and record the async bird-pos readback (guarded).
//   - render(encoder, view): update the camera (deadzone ease toward last-known bird pos), upload the
//     camera uniform, and record backdrop → trail → chevron render passes into the caller's encoder.
//   - Map raw input → (impulse,turn) intent per control scheme (flick/thrust/bank/flap). The GPU sim
//     is scheme-agnostic; only these CPU mappers differ. Live-tunable via setTuning/setCamera.
//   - Expose lastPos/lastVel/getCameraPos/getTuning (CPU mirror) for input, overlay, and verification.

import { makeComputePipeline, encodeComputePass } from "./dispatch";
import type { GpuFluid } from "./fluid";

export interface BirdTuning {
  windCoupling: number;
  drag: number;
  flickStrength: number; // max impulse magnitude (world units/frame) for scheme 1 flick
  thrust: number; // scheme 2: continuous thrust magnitude toward cursor (per fixed step)
  flapStrength: number; // scheme 4: impulse magnitude along heading per tap
  bankRate: number; // scheme 3: turn (radians) applied to vel per arrow tap
}

export interface Intent {
  impulse: [number, number];
  turn: number;
}

const TRAIL_LEN = 96;

// BirdParams uniform (std140): w,h,dt,windCoupling, drag,trailWrite,pad,pad = 32 bytes.
const BIRD_PARAMS_BYTES = 32;
// Intent uniform: impulse.xy, turn, pad = 16 bytes.
const INTENT_BYTES = 16;
// Camera uniform: cameraPos.xy, viewSize.xy, w,h,trailLen,trailHead = 32 bytes.
const CAMERA_BYTES = 32;

const BIRD_STATE_BYTES = 16; // pos.xy + vel.xy

export class Bird {
  private device: GPUDevice;
  private fluid: GpuFluid;
  readonly w: number;
  readonly h: number;

  // World/camera. deadzone + followStiffness are live-tunable (the tuning overlay sets them).
  private cameraPos: [number, number];
  private readonly viewSize: [number, number];
  private deadzone: number; // world-unit radius around center before camera eases
  private followStiffness: number; // 0..1 ease per frame

  // GPU resources.
  private birdBuf: GPUBuffer;
  private trailBuf: GPUBuffer;
  private intentBuf: GPUBuffer;
  private birdParamsBuf: GPUBuffer;
  private cameraBuf: GPUBuffer;

  private updatePipeline: GPUComputePipeline;
  private backdropPipeline: GPURenderPipeline;
  private trailPipeline: GPURenderPipeline;
  private chevronPipeline: GPURenderPipeline;

  // Host scratch.
  private birdParamsHost = new ArrayBuffer(BIRD_PARAMS_BYTES);
  private intentHost = new ArrayBuffer(INTENT_BYTES);
  private cameraHost = new ArrayBuffer(CAMERA_BYTES);

  // Async pos readback (CPU camera + verification). One 16-byte staging buffer, pending-guarded.
  private posStaging: GPUBuffer;
  private posPending = false;
  lastPos: [number, number];
  lastVel: [number, number] = [0, 0];

  private trailWrite = 0;
  private tuning: BirdTuning;

  constructor(
    device: GPUDevice,
    fluid: GpuFluid,
    updateShader: string,
    sceneShader: string,
    canvasFormat: GPUTextureFormat,
    opts: {
      viewSize: [number, number];
      startPos: [number, number];
      deadzone: number;
      followStiffness: number;
      tuning: BirdTuning;
    }
  ) {
    this.device = device;
    this.fluid = fluid;
    this.w = fluid.w;
    this.h = fluid.h;
    this.viewSize = opts.viewSize;
    this.deadzone = opts.deadzone;
    this.followStiffness = opts.followStiffness;
    this.tuning = opts.tuning;
    this.lastPos = [...opts.startPos];
    this.cameraPos = [...opts.startPos]; // start centered on the bird so the chevron is in frame

    // Bird state buffer, seeded with the start pos and a small initial drift.
    this.birdBuf = device.createBuffer({
      size: BIRD_STATE_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const initState = new Float32Array([opts.startPos[0], opts.startPos[1], 2.0, 1.0]);
    device.queue.writeBuffer(this.birdBuf, 0, initState);

    this.trailBuf = device.createBuffer({
      size: TRAIL_LEN * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // Seed the trail with the start pos so it doesn't render at (0,0).
    const trailInit = new Float32Array(TRAIL_LEN * 2);
    for (let k = 0; k < TRAIL_LEN; k++) {
      trailInit[k * 2] = opts.startPos[0];
      trailInit[k * 2 + 1] = opts.startPos[1];
    }
    device.queue.writeBuffer(this.trailBuf, 0, trailInit);

    this.intentBuf = device.createBuffer({ size: INTENT_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.birdParamsBuf = device.createBuffer({ size: BIRD_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.cameraBuf = device.createBuffer({ size: CAMERA_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    this.posStaging = device.createBuffer({ size: BIRD_STATE_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    this.updatePipeline = makeComputePipeline(device, updateShader);

    const sceneModule = device.createShaderModule({ code: sceneShader });
    const target = { format: canvasFormat };
    const blendTarget = {
      format: canvasFormat,
      blend: {
        color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
        alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
      },
    } as GPUColorTargetState;

    this.backdropPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: sceneModule, entryPoint: "backdrop_vs" },
      fragment: { module: sceneModule, entryPoint: "backdrop_fs", targets: [target] },
      primitive: { topology: "triangle-list" },
    });
    this.trailPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: sceneModule, entryPoint: "trail_vs" },
      fragment: { module: sceneModule, entryPoint: "trail_fs", targets: [blendTarget] },
      primitive: { topology: "line-strip" },
    });
    this.chevronPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: sceneModule, entryPoint: "chevron_vs" },
      fragment: { module: sceneModule, entryPoint: "chevron_fs", targets: [target] },
      primitive: { topology: "triangle-list" },
    });
  }

  setTuning(t: Partial<BirdTuning>): void {
    this.tuning = { ...this.tuning, ...t };
  }

  getTuning(): BirdTuning {
    return { ...this.tuning };
  }

  /** Live-tunable camera params (the tuning overlay sets these). */
  setCamera(c: { deadzone?: number; followStiffness?: number }): void {
    if (c.deadzone !== undefined) this.deadzone = c.deadzone;
    if (c.followStiffness !== undefined) this.followStiffness = c.followStiffness;
  }

  /** Current camera center (world). Schemes need this to map screen-cursor → world. */
  getCameraPos(): [number, number] {
    return [this.cameraPos[0], this.cameraPos[1]];
  }

  /** Scheme 1 — flick: drag vector (world units) → a length-capped one-shot impulse. */
  flickToIntent(dragWorld: [number, number]): Intent {
    const len = Math.hypot(dragWorld[0], dragWorld[1]);
    const cap = this.tuning.flickStrength;
    let imp: [number, number] = [0, 0];
    if (len > 1e-4) {
      const scale = Math.min(len, cap) / len;
      imp = [dragWorld[0] * scale, dragWorld[1] * scale];
    }
    return { impulse: imp, turn: 0 };
  }

  /**
   * Scheme 2 — hold toward cursor: while held, a gentle continuous impulse toward the cursor.
   * impulse = normalize(cursorWorld - birdWorld) * thrust * dt. Fresh each frame (the shader's
   * vel += impulse does the accumulation; do NOT pre-accumulate or it grows unbounded).
   */
  thrustToIntent(cursorWorld: [number, number], dt: number): Intent {
    const dx = cursorWorld[0] - this.lastPos[0];
    const dy = cursorWorld[1] - this.lastPos[1];
    const len = Math.hypot(dx, dy);
    if (len < 1e-4) return { impulse: [0, 0], turn: 0 };
    const m = (this.tuning.thrust * dt) / len;
    return { impulse: [dx * m, dy * m], turn: 0 };
  }

  /** Scheme 3 — tap to bank: a one-shot turn (radians) applied to vel; sign from the tapped arrow. */
  bankToIntent(sign: number): Intent {
    return { impulse: [0, 0], turn: sign * this.tuning.bankRate };
  }

  /** Scheme 4 — flap forward: a one-shot impulse along the current heading (normalize(vel)). */
  flapToIntent(): Intent {
    const len = Math.hypot(this.lastVel[0], this.lastVel[1]);
    const dir: [number, number] = len > 1e-4 ? [this.lastVel[0] / len, this.lastVel[1] / len] : [1, 0];
    return { impulse: [dir[0] * this.tuning.flapStrength, dir[1] * this.tuning.flapStrength], turn: 0 };
  }

  /** Record the bird compute pass into the caller's encoder. MUST run after fluid.step (u,v swap). */
  step(encoder: GPUCommandEncoder, dt: number, intent: Intent): void {
    // Upload intent.
    const iF = new Float32Array(this.intentHost);
    iF[0] = intent.impulse[0];
    iF[1] = intent.impulse[1];
    iF[2] = intent.turn;
    iF[3] = 0;
    this.device.queue.writeBuffer(this.intentBuf, 0, this.intentHost);

    // Upload bird params (w,h as u32; rest f32).
    const pU = new Uint32Array(this.birdParamsHost);
    const pF = new Float32Array(this.birdParamsHost);
    pU[0] = this.w;
    pU[1] = this.h;
    pF[2] = dt;
    pF[3] = this.tuning.windCoupling;
    pF[4] = this.tuning.drag;
    pU[5] = this.trailWrite;
    this.device.queue.writeBuffer(this.birdParamsBuf, 0, this.birdParamsHost);

    // Rebuild the bind group from the fluid's CURRENT u,v (they ping-pong inside fluid.step).
    const bg = this.device.createBindGroup({
      layout: this.updatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.birdParamsBuf } },
        { binding: 1, resource: { buffer: this.intentBuf } },
        { binding: 2, resource: { buffer: this.fluid.velocityX } },
        { binding: 3, resource: { buffer: this.fluid.velocityY } },
        { binding: 4, resource: { buffer: this.birdBuf } },
        { binding: 5, resource: { buffer: this.trailBuf } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.updatePipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1);
    pass.end();

    // Async bird-pos readback (camera + verification). Copy this frame's state to staging.
    if (!this.posPending) {
      encoder.copyBufferToBuffer(this.birdBuf, 0, this.posStaging, 0, BIRD_STATE_BYTES);
    }

    this.trailWrite = (this.trailWrite + 1) % TRAIL_LEN;
  }

  /** Resolve the pending pos readback (call once after queue.submit, like the fluid pixel readback). */
  resolveReadback(): void {
    if (this.posPending) return;
    this.posPending = true;
    this.posStaging
      .mapAsync(GPUMapMode.READ, 0, BIRD_STATE_BYTES)
      .then(() => {
        const f = new Float32Array(this.posStaging.getMappedRange(0, BIRD_STATE_BYTES).slice(0));
        this.posStaging.unmap();
        this.lastPos = [f[0]!, f[1]!];
        this.lastVel = [f[2]!, f[3]!];
        this.posPending = false;
      })
      .catch(() => {
        this.posPending = false;
      });
  }

  /** Ease the camera toward the bird using a soft deadzone (only moves when the bird nears the edge). */
  private updateCamera(): void {
    // Toroidal offset from camera center to the bird (shortest wrap).
    let dx = this.lastPos[0] - this.cameraPos[0];
    let dy = this.lastPos[1] - this.cameraPos[1];
    dx -= Math.round(dx / this.w) * this.w;
    dy -= Math.round(dy / this.h) * this.h;
    const dist = Math.hypot(dx, dy);
    if (dist > this.deadzone) {
      // Pull the deadzone edge to the bird: target offset = (dist - deadzone) along (dx,dy).
      const over = dist - this.deadzone;
      const nx = dx / dist;
      const ny = dy / dist;
      this.cameraPos[0] += nx * over * this.followStiffness;
      this.cameraPos[1] += ny * over * this.followStiffness;
      // Keep camera in [0,w)x[0,h).
      this.cameraPos[0] = ((this.cameraPos[0] % this.w) + this.w) % this.w;
      this.cameraPos[1] = ((this.cameraPos[1] % this.h) + this.h) % this.h;
    }
  }

  /** Record backdrop → trail → chevron into the caller's encoder. trailHead = last-written index. */
  render(encoder: GPUCommandEncoder, view: GPUTextureView): void {
    this.updateCamera();

    const cF = new Float32Array(this.cameraHost);
    const cU = new Uint32Array(this.cameraHost);
    cF[0] = this.cameraPos[0];
    cF[1] = this.cameraPos[1];
    cF[2] = this.viewSize[0];
    cF[3] = this.viewSize[1];
    cU[4] = this.w;
    cU[5] = this.h;
    cU[6] = TRAIL_LEN;
    cU[7] = (this.trailWrite + TRAIL_LEN - 1) % TRAIL_LEN; // most-recently-written index
    this.device.queue.writeBuffer(this.cameraBuf, 0, this.cameraHost);

    const backdropBg = this.device.createBindGroup({
      layout: this.backdropPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuf } },
        { binding: 1, resource: { buffer: this.fluid.dyeField } },
      ],
    });
    const trailBg = this.device.createBindGroup({
      layout: this.trailPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuf } },
        { binding: 1, resource: { buffer: this.trailBuf } },
      ],
    });
    const chevronBg = this.device.createBindGroup({
      layout: this.chevronPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuf } },
        { binding: 1, resource: { buffer: this.birdBuf } },
      ],
    });

    const rp = encoder.beginRenderPass({
      colorAttachments: [
        { view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
      ],
    });
    rp.setPipeline(this.backdropPipeline);
    rp.setBindGroup(0, backdropBg);
    rp.draw(3);

    rp.setPipeline(this.trailPipeline);
    rp.setBindGroup(0, trailBg);
    rp.draw(TRAIL_LEN);

    rp.setPipeline(this.chevronPipeline);
    rp.setBindGroup(0, chevronBg);
    rp.draw(3);
    rp.end();
  }
}
