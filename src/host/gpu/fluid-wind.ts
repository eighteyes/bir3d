// fluid-wind.ts — drive the GPU Stam fluid as the EVOLVING wind source for the bird sandbox (v13).
// Responsibilities:
//   - Own a GpuFluid (~256²) over a BIRD-LOCAL window that MOVES with the bird (so the bird never
//     flies out into dead air; the world-pinned moving-window is deferred).
//   - Each frame FORCE the fluid with a CONTINUOUS BOUNDED structure: a slowly ORBITING + ROTATING
//     localized force disc (forces.wgsl: fx,fy within force_r of dye_x,dye_y). Localized injection →
//     projection spreads it into circulation → advection carries it → the field gains gradients
//     (self-bounding, no viscosity needed) AND structure that EVOLVES. A weak prevailing component is
//     folded in via the disc direction; the strong steady cross-track drift stays analytic in wind.ts.
//   - Record fluid.step() into the caller's encoder; enqueue velocityX/velocityY into two ReadbackRings.
//   - After submit, kick both rings' maps; expose read() of the latest (2-3 frame stale) u/v arrays +
//     the window mapping so the caller can push them to wind.setFluidField.
//   - Magnitude REGULATOR: the equilibrium speed of a forceless-viscosity solver is unpredictable, so the
//     world→m/s SCALE is calibrated from the MEASURED readback magnitude toward a target flyable band —
//     force = structure/evolution, scale = band (decoupled, per the advisor).

import { GpuFluid } from "./fluid";
import { ReadbackRing } from "./readback";

interface FluidShaders {
  forces: string;
  divergence: string;
  jacobi: string;
  subtractGrad: string;
  advect: string;
  setBnd: string;
}

export interface FluidWindConfig {
  grid?: number;       // interior cells per side (square). 256² default; lower if fps suffers.
  iters?: number;      // Stam projection iterations per project() (×2 per step). Lower if fps suffers.
  worldSpanM?: number; // world meters the grid window spans (window is bird-local, this wide)
  forceMag?: number;   // per-step force magnitude injected at the disc (structure/evolution knob)
  targetBand?: number; // desired |sampled wind| (m/s) the magnitude regulator drives the SCALE toward
  scaleMin?: number;   // clamp the regulated scale (avoid runaway / collapse)
  scaleMax?: number;
}

export class FluidWind {
  private fluid: GpuFluid;
  private ringU: ReadbackRing;
  private ringV: ReadbackRing;

  readonly gridW: number;
  readonly gridH: number;
  readonly iters: number;
  private worldSpanM: number;
  readonly cellM: number;       // meters per grid cell
  private forceMag: number;
  private targetBand: number;
  private scaleMin: number;
  private scaleMax: number;

  // bird-local window origin (world XZ of interior cell (0,0)); recentred on the bird each frame.
  private originX = 0;
  private originZ = 0;

  // regulated grid-velocity → m/s scale (calibrated from the measured readback magnitude). Starts small
  // so the band eases UP into range (never spikes above the flyable band on the first resolved frames).
  private scale = 0.05;

  private t = 0; // accumulated sim time for the slowly-moving force pattern

  constructor(device: GPUDevice, shaders: FluidShaders, cfg: FluidWindConfig = {}) {
    const grid = cfg.grid ?? 256;
    this.gridW = grid;
    this.gridH = grid;
    this.iters = cfg.iters ?? 20;
    this.worldSpanM = cfg.worldSpanM ?? 2600; // ~ matches the visible terrain/mote span around the bird
    this.cellM = this.worldSpanM / grid;
    this.forceMag = cfg.forceMag ?? 90;       // structure knob (NOT the band — band is the scale)
    // target the MEAN |wind| to the analytic field's proven mean (~8.5) it replaced — the regulator pins
    // the mean here; the peaks are clamped to FLUID_MAX in windAt. This matches the shipped-flyable
    // distribution (no +61° blow-around) AND keeps the motes' speed-cull cost at the analytic baseline.
    this.targetBand = cfg.targetBand ?? 3.0;
    this.scaleMin = cfg.scaleMin ?? 0.02;
    this.scaleMax = cfg.scaleMax ?? 50;

    this.fluid = new GpuFluid(device, grid, grid, shaders);
    this.ringU = new ReadbackRing(device, this.fluid.bytes);
    this.ringV = new ReadbackRing(device, this.fluid.bytes);
  }

  /** world meters per grid cell (for the caller's window mapping). */
  get cellMeters(): number { return this.cellM; }
  get originXZ(): [number, number] { return [this.originX, this.originZ]; }
  get currentScale(): number { return this.scale; }

  /**
   * Per-frame: recenter the bird-local window, set a CONTINUOUS bounded force (orbiting+rotating disc),
   * step the fluid into the caller's encoder, and enqueue u/v readback. Call afterSubmit() post-submit.
   */
  step(encoder: GPUCommandEncoder, dt: number, birdX: number, birdZ: number): void {
    // BIRD-LOCAL moving window: origin so the bird sits at the window CENTER.
    this.originX = birdX - (this.worldSpanM * 0.5);
    this.originZ = birdZ - (this.worldSpanM * 0.5);

    const dtc = Math.min(Math.max(dt, 1e-3), 0.05); // clamp first-frame / tab-stall spikes
    this.t += dtc;

    // CONTINUOUS BOUNDED FORCE — a localized disc that slowly ORBITS the grid center while its force
    // direction ROTATES; successive injections + the fluid's persistence integrate into an evolving
    // multi-swirl field. force_r ~¼ grid → localized (global uniform force has no gradient → unbounded
    // blow-up + zero structure; see advisor). Magnitudes are bounded; the BAND is set by the scale, not
    // by cranking this (cranking force = the +61° blown-around regression).
    const cx = this.gridW * 0.5;
    const cz = this.gridH * 0.5;
    const orbitR = this.gridW * 0.22;
    const dyeX = cx + Math.cos(this.t * 0.13) * orbitR;
    const dyeY = cz + Math.sin(this.t * 0.11) * orbitR;
    // force direction: a slow prevailing bias (constant-ish) + a rotating swirl component.
    const ang = this.t * 0.37;
    const fx = this.forceMag * (0.55 + 0.45 * Math.cos(ang));
    const fy = this.forceMag * (0.35 + 0.45 * Math.sin(ang * 1.3));
    this.fluid.setForce({
      fx, fy,
      dyeX, dyeY,
      dyeR: 0,                          // no dye injection (we only need velocity)
      dyeAmt: 0,
      forceR: this.gridW * 0.25,        // localized disc (~¼ grid)
    });

    this.fluid.step(encoder, dtc, this.iters);
    // enqueue this frame's velocity buffers for async readback (resolved 2-3 frames later).
    this.ringU.enqueue(encoder, this.fluid.velocityX);
    this.ringV.enqueue(encoder, this.fluid.velocityY);
  }

  /** Call AFTER device.queue.submit() — kicks the non-awaited maps for this frame's copied slots. */
  afterSubmit(): void {
    this.ringU.afterSubmit();
    this.ringV.afterSubmit();
  }

  /**
   * Latest resolved u/v (bordered (gridW+2)*(gridH+2)) or null until the first readback resolves.
   * Also REGULATES the world→m/s scale toward the target band from the measured mean magnitude, so the
   * sampled wind stays in the flyable band regardless of the solver's (unpredictable) equilibrium speed.
   */
  read(): { u: Float32Array; v: Float32Array } | null {
    const u = this.ringU.read();
    const v = this.ringV.read();
    if (!u || !v) return null;

    // regulate the scale from the mean interior magnitude (proportional, gentle — it eases each frame).
    const mean = this.meanMagnitude(u, v);
    if (mean > 1e-4) {
      const wantScale = this.targetBand / mean;
      const clamped = Math.min(this.scaleMax, Math.max(this.scaleMin, wantScale));
      // ease toward the target scale so the band settles smoothly (not a per-frame jump).
      this.scale += (clamped - this.scale) * 0.1;
    }
    return { u, v };
  }

  /** Mean magnitude of the interior velocity cells (sampled on a stride for cheapness). */
  private meanMagnitude(u: Float32Array, v: Float32Array): number {
    const stride = this.gridW + 2;
    let sum = 0, n = 0;
    const step = Math.max(1, Math.floor(this.gridW / 48)); // ~48² samples — cheap, representative
    for (let j = 1; j <= this.gridH; j += step) {
      for (let i = 1; i <= this.gridW; i += step) {
        const k = i + stride * j;
        sum += Math.hypot(u[k]!, v[k]!);
        n++;
      }
    }
    return n > 0 ? sum / n : 0;
  }

  destroy(): void {
    this.fluid.destroy();
  }
}
