// fluid-wind.ts — drive the GPU Stam fluid as the EVOLVING, WORLD-PINNED wind source for the bird (v15).
// Responsibilities:
//   - Own a GpuFluid (~256²) over a WORLD-PINNED window: the grid origin is anchored to WORLD coordinates
//     so a gust sits over the SAME ridge as the bird flies past (anchored, not bird-relative). As the bird
//     nears the window edge, RECENTER in GRID-ALIGNED whole-cell steps: bump the origin AND GPU-shift the
//     velocity field (u/v) by the same integer cell offset (overlap copied 1:1, leading edge clamp-extended)
//     so the existing flow scrolls with the world — NO seam/pop in the wind on recenter.
//   - TERRAIN COUPLING: compute a per-cell orographic force field from the terrain heightfield (replicated
//     fBm twin of terrain.ts) — flow is deflected around/over high terrain and channelled along the
//     down-gradient so gusts visibly relate to the ridges. BOUNDED to the flyable band (no spike). The
//     force field is recomputed only on recenter (shift the world window → re-evaluate fBm for the cells).
//   - Each frame FORCE the fluid with a weak prevailing component (advects the terrain-anchored structure
//     past the ridges). The bird-relative orbiting disc is cut to near-zero (it dragged structure with the
//     grid and fought world-pinning).
//   - Own a u+v readback ring (resolved together; per-slot ORIGIN tag) so read() returns the field and the
//     CAPTURE-TIME origin as a matched pair — bird-main reads read() then originXZ in the same frame, so
//     the world→grid mapping is always self-consistent (no 2-3-frame-stale mapping pop after a recenter).
//   - Magnitude REGULATOR: the SCALE (grid-velocity→m/s) is calibrated from the measured readback magnitude
//     toward a target band — force = structure, scale = band (decoupled, per the advisor).
//   - Expose window.__fluidWindow {originX,originZ,recenterFrame} so a probe can detect/await a recenter.

import { GpuFluid } from "./fluid";
import { loadShader } from "./shaders";

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
  worldSpanM?: number; // world meters the grid window spans
  forceMag?: number;   // per-step PREVAILING force magnitude (advects the terrain-anchored structure)
  targetBand?: number; // desired mean |sampled wind| (m/s) the magnitude regulator drives the SCALE toward
  scaleMin?: number;   // clamp the regulated scale (avoid runaway / collapse)
  scaleMax?: number;
  recenterFrac?: number; // bird may drift this fraction of half-span from center before a recenter
  terrainGain?: number;  // orographic force strength (per-cell force units); bounded by terrainMax
  terrainMax?: number;   // hard clamp on |per-cell terrain force| (keeps the field in the flyable band)
}

// --- terrain fBm twin (MUST mirror terrain.ts / terrain_ekg.wgsl constants exactly) ---
// Replicated here because terrain.ts exposes sampleHeight only as a TerrainEKG instance method and the
// constants are module-private; bird-main (frozen) does not pass the terrain to FluidWind. Terrain is
// frozen, so this duplication is stable. Render-only TerrainEKG config (rows/cols/fog) does not affect
// the heightfield, so these defaults match the live terrain.
const BASE_FREQ = 0.00142857;
const LACUNARITY = 2.0;
const GAIN = 0.5;
const OCTAVES = 4;
const RELIEF = 600.0;   // taller, more dramatic peaks (was 320) — MUST mirror terrain.ts / terrain_ekg.wgsl
const SHARP = 1.8;
const TERRACES = 5.0;
const RISER_POW = 4.0;
const CLIFF_MIX = 0.65;

function hash2(px: number, py: number): number {
  const s = Math.sin(px * 127.1 + py * 311.7) * 43758.5453;
  return s - Math.floor(s);
}
function valueNoise(px: number, py: number): number {
  const ix = Math.floor(px), iy = Math.floor(py);
  const fx = px - ix, fy = py - iy;
  const a = hash2(ix, iy);
  const b = hash2(ix + 1, iy);
  const c = hash2(ix, iy + 1);
  const d = hash2(ix + 1, iy + 1);
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const mx0 = a + (b - a) * ux;
  const mx1 = c + (d - c) * ux;
  return mx0 + (mx1 - mx0) * uy;
}
function sampleHeight(x: number, z: number): number {
  let freq = BASE_FREQ, amp = 1, sum = 0, norm = 0;
  for (let k = 0; k < OCTAVES; k++) {
    const n = valueNoise(x * freq, z * freq);
    const r = 1 - Math.abs(2 * n - 1);
    sum += amp * r;
    norm += amp;
    freq *= LACUNARITY;
    amp *= GAIN;
  }
  const s = Math.pow(sum / norm, SHARP);
  const b = s * TERRACES;
  const fb = b - Math.floor(b);
  const ter = Math.floor(b) / TERRACES + Math.pow(fb, RISER_POW) / TERRACES;
  return (s + (ter - s) * CLIFF_MIX) * RELIEF;
}

// Self-contained u+v readback ring with a per-slot ORIGIN tag. u and v for a frame resolve TOGETHER
// (Promise.all) so read() never pairs mismatched u/v (a latent bug of two independent rings), and each
// resolved field carries the window origin that was in effect when it was captured — so the world→grid
// mapping that bird-main builds from originXZ matches the field read() handed out, even though the readback
// is 2-3 frames stale and an intervening recenter may have moved the live origin.
class PairedReadback {
  private staging: { u: GPUBuffer; v: GPUBuffer }[];
  private inflight: boolean[];
  private slotOrigin: [number, number][];
  private i = 0;
  private pending: number | null = null;
  private latestU: Float32Array | null = null;
  private latestV: Float32Array | null = null;
  private latestOrigin: [number, number] = [0, 0];

  constructor(private device: GPUDevice, private bytes: number, private size = 3) {
    const mk = () =>
      device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.staging = Array.from({ length: size }, () => ({ u: mk(), v: mk() }));
    this.inflight = Array.from({ length: size }, () => false);
    this.slotOrigin = Array.from({ length: size }, () => [0, 0] as [number, number]);
  }

  /** Record the u/v copies for this frame into a free slot, tagged with the current window origin. */
  enqueue(encoder: GPUCommandEncoder, srcU: GPUBuffer, srcV: GPUBuffer, originX: number, originZ: number): void {
    const slot = this.i;
    if (!this.inflight[slot]) {
      encoder.copyBufferToBuffer(srcU, 0, this.staging[slot]!.u, 0, this.bytes);
      encoder.copyBufferToBuffer(srcV, 0, this.staging[slot]!.v, 0, this.bytes);
      this.inflight[slot] = true;
      this.slotOrigin[slot] = [originX, originZ];
      this.pending = slot;
    } else {
      this.pending = null; // slot still mapping; skip this frame
    }
    this.i = (this.i + 1) % this.size;
  }

  /** Kick the non-awaited maps for this frame's slot (must be AFTER queue.submit()). */
  afterSubmit(): void {
    const slot = this.pending;
    if (slot === null) return;
    this.pending = null;
    const cell = this.staging[slot]!;
    Promise.all([cell.u.mapAsync(GPUMapMode.READ), cell.v.mapAsync(GPUMapMode.READ)])
      .then(() => {
        this.latestU = new Float32Array(cell.u.getMappedRange().slice(0));
        this.latestV = new Float32Array(cell.v.getMappedRange().slice(0));
        this.latestOrigin = this.slotOrigin[slot]!;
        cell.u.unmap();
        cell.v.unmap();
        this.inflight[slot] = false;
      })
      .catch((err: unknown) => {
        this.inflight[slot] = false;
        console.error(`PairedReadback slot ${slot} map failed:`, err);
      });
  }

  /** Latest resolved u/v + the origin captured with them (or null until the first map resolves). */
  read(): { u: Float32Array; v: Float32Array; origin: [number, number] } | null {
    if (!this.latestU || !this.latestV) return null;
    return { u: this.latestU, v: this.latestV, origin: this.latestOrigin };
  }

  destroy(): void {
    for (const c of this.staging) { c.u.destroy(); c.v.destroy(); }
  }
}

export class FluidWind {
  private fluid: GpuFluid;
  private ring: PairedReadback;

  readonly gridW: number;
  readonly gridH: number;
  readonly iters: number;
  private worldSpanM: number;
  readonly cellM: number;       // meters per grid cell
  private forceMag: number;
  private targetBand: number;
  private scaleMin: number;
  private scaleMax: number;
  private recenterFrac: number;
  private terrainGain: number;
  private terrainMax: number;

  // WORLD-PINNED window origin (world XZ of interior cell (0,0)). Moves only in whole-cell steps on
  // recenter; the field is GPU-shifted by the same offset so the flow scrolls with the world (no pop).
  private originX = 0;
  private originZ = 0;
  private initialized = false;

  // origin matched to the field read() last returned (capture-time origin). originXZ returns THIS so
  // bird-main's read()+originXZ pair (same frame, both inside `if (field)`) maps the field correctly.
  private readOriginX = 0;
  private readOriginZ = 0;

  private scale = 0.05; // regulated grid-velocity → m/s scale (starts small → eases up into band)
  private t = 0;        // accumulated sim time for the slow prevailing pattern
  private recenterFrame = 0; // frame counter bumped on each recenter (probe hook)
  private frame = 0;
  private rawMean = 0;  // last measured raw (pre-scale) interior mean magnitude (blow-up diagnostic)

  // CPU scratch for the per-cell terrain force field + a cached height grid (bordered (W+2)*(H+2)).
  // The height grid caches sampleHeight per cell so forces derive from cheap neighbour finite-diffs;
  // on recenter only the freshly-exposed edge strip needs fBm re-eval (the rest is shifted in place).
  private fxField: Float32Array;
  private fyField: Float32Array;
  private heights: Float32Array;
  private heightsScratch: Float32Array;
  private extraReady = false;

  constructor(device: GPUDevice, shaders: FluidShaders, cfg: FluidWindConfig = {}) {
    const grid = cfg.grid ?? 256;
    this.gridW = grid;
    this.gridH = grid;
    this.iters = cfg.iters ?? 10;
    this.worldSpanM = cfg.worldSpanM ?? 2600;
    this.cellM = this.worldSpanM / grid;
    this.forceMag = cfg.forceMag ?? 28;       // weak PREVAILING advection (not the band — band is scale)
    this.targetBand = cfg.targetBand ?? 3.0;  // mean |wind| target; peaks clamped to windTuning.fluidMax in windAt
    this.scaleMin = cfg.scaleMin ?? 0.02;
    this.scaleMax = cfg.scaleMax ?? 50;
    this.recenterFrac = cfg.recenterFrac ?? 0.18; // bird may drift 18% of half-span before recenter
    this.terrainGain = cfg.terrainGain ?? 22;     // orographic force strength
    this.terrainMax = cfg.terrainMax ?? 60;       // hard clamp on |per-cell terrain force|

    this.fluid = new GpuFluid(device, grid, grid, shaders);
    this.ring = new PairedReadback(device, this.fluid.bytes);
    this.fxField = new Float32Array(this.fluid.cells);
    this.fyField = new Float32Array(this.fluid.cells);
    this.heights = new Float32Array(this.fluid.cells);
    this.heightsScratch = new Float32Array(this.fluid.cells);

    // Init the extra kernels (shift + per-cell force field). Shaders are bundled at build time
    // (loadShader), so they're available synchronously — no runtime fetch that would 404 on a
    // static host and break these compute pipelines.
    this.fluid.initExtraPipelines(
      loadShader("/src/host/shaders/fluid/shift.wgsl"),
      loadShader("/src/host/shaders/fluid/force_field.wgsl"),
    );
    this.extraReady = true;
    if (this.initialized) this.fluid.setForceField(this.fxField, this.fyField);
  }

  get cellMeters(): number { return this.cellM; }
  /** Capture-time origin paired with the field read() last returned (NOT the live sim origin). */
  get originXZ(): [number, number] { return [this.readOriginX, this.readOriginZ]; }
  get currentScale(): number { return this.scale; }

  /**
   * Per-frame: world-pin the window (recenter in whole-cell steps + GPU-shift the field on recenter),
   * set the weak prevailing force, step the fluid into the caller's encoder, enqueue u/v readback (tagged
   * with the live origin). Call afterSubmit() post-submit.
   */
  step(encoder: GPUCommandEncoder, dt: number, birdX: number, birdZ: number): void {
    // Initialize the world-pinned window so the bird starts centered (one-time).
    if (!this.initialized) {
      this.originX = birdX - this.worldSpanM * 0.5;
      this.originZ = birdZ - this.worldSpanM * 0.5;
      this.initialized = true;
      this.fullTerrainField();
    }

    const dtc = Math.min(Math.max(dt, 1e-3), 0.05);
    this.t += dtc;

    // WORLD-PINNED RECENTER (grid-aligned). Bird's grid coord vs center; if it drifts past the deadzone,
    // shift the origin by WHOLE cells so the bird returns toward center, and GPU-shift the field by the
    // same integer offset so the existing flow scrolls with the world (overlap copied → no seam/pop).
    const center = this.gridW * 0.5;
    const dead = this.gridW * this.recenterFrac;
    const bgx = (birdX - this.originX) / this.cellM;
    const bgz = (birdZ - this.originZ) / this.cellM;
    let dCellsX = 0, dCellsZ = 0;
    if (bgx - center > dead) dCellsX = Math.floor(bgx - center - dead) + 1;
    else if (center - bgx > dead) dCellsX = -(Math.floor(center - bgx - dead) + 1);
    if (bgz - center > dead) dCellsZ = Math.floor(bgz - center - dead) + 1;
    else if (center - bgz > dead) dCellsZ = -(Math.floor(center - bgz - dead) + 1);

    if (dCellsX !== 0 || dCellsZ !== 0) {
      // origin moves toward the bird by whole cells.
      this.originX += dCellsX * this.cellM;
      this.originZ += dCellsZ * this.cellM;
      // field grid index of a fixed WORLD point decreases by dCells when origin increases → shift by -dCells
      // (dst[i] = src[i + dCells]). The shift shader takes dx as `dst[i]=src[i-dx]` → pass -dCells.
      this.fluid.shift(encoder, -dCellsX, -dCellsZ);
      // shift the cached height grid the SAME way (CPU mirror of the GPU u/v shift), re-eval fBm only for
      // the freshly-exposed edge strip, then re-derive forces — the fBm cost is edge-only (no full hitch).
      this.recenterTerrainField(dCellsX, dCellsZ);
      this.recenterFrame++;
    }

    // EVOLUTION FORCE — a small LOCALIZED rotating disc (NOT whole-interior: a spatially-uniform force has
    // no gradient → advection can't diffuse it → unbounded accumulation; the original code's comment). The
    // disc keeps the field time-EVOLVING (gusts that come and go); the world-pinned TERRAIN force is the
    // anchored structure, and wind.ts's analytic drift is the cross-track sweep. Small + localized = bounded.
    const cx = this.gridW * 0.5;
    const cz = this.gridH * 0.5;
    const orbitR = this.gridW * 0.22;
    const dyeX = cx + Math.cos(this.t * 0.13) * orbitR;
    const dyeY = cz + Math.sin(this.t * 0.11) * orbitR;
    const ang = this.t * 0.37;
    const fx = this.forceMag * Math.cos(ang);
    const fy = this.forceMag * Math.sin(ang * 1.3);
    this.fluid.setForce({
      fx, fy,
      dyeX, dyeY, dyeR: 0, dyeAmt: 0,
      forceR: this.gridW * 0.25, // localized disc (~¼ grid) — gradient-bounded
    });

    this.fluid.step(encoder, dtc, this.iters);
    this.ring.enqueue(encoder, this.fluid.velocityX, this.fluid.velocityY, this.originX, this.originZ);

    this.frame++;
    (globalThis as { __fluidWindow?: unknown }).__fluidWindow = {
      originX: this.originX, originZ: this.originZ, recenterFrame: this.recenterFrame,
      cellM: this.cellM, gridW: this.gridW, frame: this.frame,
      scale: this.scale, rawMean: this.rawMean, // blow-up diagnostics (advisor #1)
    };
  }

  /** Kick the non-awaited readback maps for the slot copied this frame (must be AFTER submit). */
  afterSubmit(): void {
    this.ring.afterSubmit();
  }

  /**
   * Latest resolved u/v (bordered (gridW+2)*(gridH+2)) or null until the first readback resolves, paired
   * with the CAPTURE-TIME origin (stored into readOriginX/Z so originXZ returns it). Also REGULATES the
   * world→m/s scale toward the target band from the measured mean magnitude.
   */
  read(): { u: Float32Array; v: Float32Array } | null {
    const r = this.ring.read();
    if (!r) return null;
    this.readOriginX = r.origin[0];
    this.readOriginZ = r.origin[1];

    const mean = this.meanMagnitude(r.u, r.v);
    this.rawMean = mean;
    if (mean > 1e-4) {
      const wantScale = this.targetBand / mean;
      const clamped = Math.min(this.scaleMax, Math.max(this.scaleMin, wantScale));
      this.scale += (clamped - this.scale) * 0.1;
    }
    return { u: r.u, v: r.v };
  }

  // World XZ of cell index (i,j) (interior cell (1,1) = window origin; border cells extend ±1 cell).
  private cellWorldX(i: number): number { return this.originX + (i - 1) * this.cellM; }
  private cellWorldZ(j: number): number { return this.originZ + (j - 1) * this.cellM; }

  /** Fill the cached height grid for EVERY cell (incl. border) for the current window — full fBm pass. */
  private fillHeights(): void {
    const stride = this.gridW + 2;
    for (let j = 0; j < this.gridH + 2; j++) {
      const wz = this.cellWorldZ(j);
      for (let i = 0; i < this.gridW + 2; i++) {
        this.heights[i + stride * j] = sampleHeight(this.cellWorldX(i), wz);
      }
    }
  }

  /**
   * Derive the per-cell orographic force from the cached height grid (cheap neighbour finite-diff; no fBm).
   * Force flows DOWN-gradient (−uphill) scaled by slope → flow is pushed away from high ground, around
   * peaks / channelled into valleys, visibly relating to the ridges. BOUNDED (clamped to terrainMax) so it
   * never spikes the field out of the flyable band. Edge interior cells use one-sided diffs (border cells
   * are sampled too, so central diffs are valid for all interior cells).
   */
  private forcesFromHeights(): void {
    const stride = this.gridW + 2;
    const inv2e = 1 / (2 * this.cellM);
    const g = this.terrainGain;
    const tmax = this.terrainMax;
    const H = this.heights;
    for (let j = 1; j <= this.gridH; j++) {
      for (let i = 1; i <= this.gridW; i++) {
        const k = i + stride * j;
        const gx = (H[k + 1]! - H[k - 1]!) * inv2e;          // dH/dx
        const gz = (H[k + stride]! - H[k - stride]!) * inv2e; // dH/dz
        let fx = -gx * g;
        let fz = -gz * g;
        const m = Math.hypot(fx, fz);
        if (m > tmax) { const s = tmax / m; fx *= s; fz *= s; }
        this.fxField[k] = fx;
        this.fyField[k] = fz;
      }
    }
    if (this.extraReady) this.fluid.setForceField(this.fxField, this.fyField);
  }

  /** Full recompute (init): all heights + all forces. */
  private fullTerrainField(): void {
    this.fillHeights();
    this.forcesFromHeights();
  }

  /**
   * Recenter recompute: shift the cached height grid by the SAME integer offset as the GPU u/v shift
   * (dst[i] = src[i + dCells] — a fixed world point keeps its height), re-eval fBm only for the freshly-
   * exposed edge strip, then re-derive ALL forces from the (now consistent) height grid and re-upload.
   * The expensive fBm is edge-only; force derivation is cheap neighbour diffs. dCellsX>0 = origin moved +x.
   */
  private recenterTerrainField(dCellsX: number, dCellsZ: number): void {
    const stride = this.gridW + 2;
    const W2 = this.gridW + 2, H2 = this.gridH + 2;
    const src = this.heights, dst = this.heightsScratch;
    // shift: dst[i,j] = src[i+dCellsX, j+dCellsZ] when the source is in range, else mark for re-eval (NaN).
    for (let j = 0; j < H2; j++) {
      const sj = j + dCellsZ;
      for (let i = 0; i < W2; i++) {
        const si = i + dCellsX;
        dst[i + stride * j] = (si >= 0 && si < W2 && sj >= 0 && sj < H2)
          ? src[si + stride * sj]!
          : NaN; // freshly exposed → re-eval below
      }
    }
    // re-eval fBm only for the exposed cells (NaN). |dCells| is small (1 in steady flight) → a thin strip.
    for (let j = 0; j < H2; j++) {
      const wz = this.cellWorldZ(j);
      for (let i = 0; i < W2; i++) {
        const k = i + stride * j;
        if (Number.isNaN(dst[k]!)) dst[k] = sampleHeight(this.cellWorldX(i), wz);
      }
    }
    // swap scratch in as the live height grid, then re-derive forces.
    this.heights = dst;
    this.heightsScratch = src;
    this.forcesFromHeights();
  }

  private meanMagnitude(u: Float32Array, v: Float32Array): number {
    const stride = this.gridW + 2;
    let sum = 0, n = 0;
    const step = Math.max(1, Math.floor(this.gridW / 48));
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
    this.ring.destroy();
  }
}
