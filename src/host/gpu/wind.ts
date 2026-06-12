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
//     FROZEN: windAt/thermalAt/potential/DEFAULTS drive the bird flight physics — do NOT retune them.
//   - Wind: a render pipeline (wind.wgsl) that draws drifting neon COMET motes over the terrain.
//     v9 model — wind INTERACTS WITH TERRAIN + curved longer tails (supersedes v8's flat streaks):
//     each mote carries a persisted 3D world position (x,y,z) advected each frame by flowAt() — the
//     TERRAIN-SHAPED flow built on top of the frozen windAt: (1) VERTICAL — w = horizontalWind ·
//     uphill-gradient (finite-diff of sampleHeight), so motes RISE over windward slopes and SINK in
//     lees; the height is advected (clamped above terrain, with a mild relaxation back toward nominal
//     clearance so the field doesn't deplete or pile at a ceiling). (2) HORIZONTAL DEFLECTION — near
//     steep terrain the into-slope component of the horizontal wind is removed so flow bends AROUND
//     peaks / OVER crests instead of through them. The SAME flowAt feeds both advection and the tail
//     integration so they stay coherent. (3) TAILS — each mote's tail is a CURVED multi-segment
//     polyline integrated BACKWARD along flowAt over several steps (long curved comets arcing over the
//     ridges), longer than v8. SPEED still reads two ways off local |windAt|: DENSITY (a per-mote rank
//     culled below a speed-dependent cutoff, computed CPU-side) and TAIL LENGTH (segment step scales
//     with speed). Rendered as additive billboards, depth-tested against terrain (ridges occlude
//     them). NOT regenerated each frame — persistence is what makes them move.
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

// stable per-mote rank in 0..1 from the integer index (deterministic, frame-stable) — used for the
// CPU-side density cull so a mote can span many ribbon verts without breaking a vertex-index hash.
function hashRank(i: number): number {
  const s = Math.sin(i * 12.9898) * 43758.5453;
  return s - Math.floor(s);
}

function smoothstep(a: number, b: number, x: number): number {
  if (a === b) return x < a ? 0 : 1;
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

// --- dot-particle overlay pipeline ---

interface DotParams {
  numMotes?: number;  // total motes seeded uniformly across the camera-relative wedge
  spanAhead?: number; // how far ahead of the camera ground point the field reaches (m)
  spanBehind?: number;// margin behind the camera before a mote is wrapped to the front (m)
  spanWide?: number;  // half-width of the camera-relative field at the far edge (m)
  clearance?: number; // nominal meters above terrain the motes relax toward (height is advected, not pinned)
  minClear?: number;  // hard floor: motes never sink closer than this above terrain
  maxClear?: number;  // soft ceiling on height above terrain (keeps motes in the readable band)
  nearBias?: number;  // v10 density: ahead = near + (far-near)·rand^nearBias (k>1 → cluster near the bird)
  liftGain?: number;  // multiplier on the vertical flow w = liftGain · (horizontalWind · uphillGrad)
  relax?: number;     // per-second relaxation of height back toward nominal clearance (anti-deplete)
  deflect?: number;   // 0..1 strength of horizontal into-slope deflection near steep terrain
  segments?: number;  // tail polyline segment count (curved comet) — more = longer/smoother arc
  segStep?: number;   // seconds of flow integrated per tail segment (tail length in flow-time)
  dotPx?: number;     // on-screen comet head diameter (px) — converted to NDC half-width per frame
  tailFloor?: number; // tail segment-step fraction in the calmest air (0..1) — short stub when slow
  densityFloor?: number; // fraction of motes that survive the speed-fade even in the calmest air (0..1)
  speedLo?: number;   // |windAt| (m/s) mapped to speedFrac 0 (calm) — calibrated to the field min
  speedHi?: number;   // |windAt| (m/s) mapped to speedFrac 1 (fast) — calibrated to the field max
}

export class Wind {
  private cfg: Required<WindConfig>;
  private count: number;          // total motes
  private spanAhead: number;
  private spanBehind: number;
  private spanWide: number;
  private clearance: number;
  private minClear: number;
  private maxClear: number;
  private nearBias: number;
  private liftGain: number;
  private relax: number;
  private deflect: number;
  private segments: number;       // tail polyline segment count
  private segStep: number;        // seconds of flow per tail segment
  private dotPx: number;
  private tailFloor: number;
  private densityFloor: number;
  private speedLo: number;
  private speedHi: number;

  // PERSISTENT world-space particle state (NOT regenerated each frame — that is what makes them drift).
  // v9: each mote carries a 3D position (x, y, z); y is ADVECTED by the vertical flow w (terrain-shaped),
  // not pinned to a fixed clearance — so motes visibly pour up windward slopes and sink in lees.
  private px: Float32Array;   // world x (per mote)
  private py: Float32Array;   // world y — ADVECTED height (per mote)
  private pz: Float32Array;   // world z (per mote)
  private speedFrac: Float32Array; // cached 0..1 wind speed at the mote (for color/density/tail)

  private vbuf: GPUBuffer;     // per-vertex ribbon-segment data, rebuilt from particle state each frame
  private vertexCount: number; // count * segments * 6 (a quad per tail segment)
  private vertBytes: ArrayBuffer;
  private vertHost: Float32Array;
  private ubuf: GPUBuffer;
  private uniformHost: ArrayBuffer;
  private uniformF32: Float32Array;
  private pipeline: GPURenderPipeline;
  private bindGroup: GPUBindGroup;

  // reusable scratch for the per-mote integrated polyline (segments+1 points): world xyz + along-fraction.
  private ptX: Float32Array;
  private ptY: Float32Array;
  private ptZ: Float32Array;

  private lastTime = -1;       // for per-frame dt derivation from bird.simTime
  private seeded = false;      // first draw seeds the field uniformly across the wedge

  // floats/vertex: center.xyz(3) + corner.xy(2) + speedFrac(1) + segDir.xz(2) + along(1) + vis(1) = 10.
  // The streak is now a CURVED ribbon: each segment is a quad between two integrated polyline points,
  // oriented along that segment's own screen-space direction; `along` is the head→tail fade fraction.
  private static FPV = 10;
  // a quad per ribbon segment: x∈{0,1} picks the segment's near(0)/far(1) endpoint, y∈{-1,1} is the perp.
  private static CORNERS: ReadonlyArray<[number, number]> = [
    [0, -1], [1, -1], [1, 1],
    [0, -1], [1, 1], [0, 1],
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
    // v9: fewer motes than v8 (4200 → 2200) to AFFORD the multi-step curved-tail integration per mote
    // (each mote now integrates `segments` flow steps every frame). Still wind EVERYWHERE across the wedge.
    this.count = p.numMotes ?? 2200;
    // spanAhead matches the terrain maxDist (~950) so every mote lives in the DRAWN terrain band —
    // dots past the cutoff floated over a void and read as detached sky specks, not wind in the scene.
    this.spanAhead = p.spanAhead ?? 950;
    this.spanBehind = p.spanBehind ?? 260;
    this.spanWide = p.spanWide ?? 950;
    // v10 HUG: drop the clearance the height RELAXES toward from 55→16 so the field rides JUST over the
    // ridges and follows the contour up and over each crest (v9's 55 floated a flat sheet above 220m relief).
    // height is advected by w (not pinned) so motes pour up/over ridges, with mild relaxation back to this low
    // nominal so the field doesn't deplete or pile at a ceiling.
    this.clearance = p.clearance ?? 16;
    this.minClear = p.minClear ?? 5;    // v10: hard floor 14→5 — motes hug right down onto the surface.
    this.maxClear = p.maxClear ?? 170;  // v10: RAISE the ceiling 100→170 so actively-CLIMBING motes have room
                                        // to stream up the windward face and SPILL over the crest (in flat/lee
                                        // air w≈0 so relax keeps motes at the LOW clearance — the raised ceiling
                                        // does not lift the baseline, it only frees the pour to read as a plume).
    // v10 POUR: dh/dt = (liftGain−1)·d(terr)/dt along the path, so liftGain>1 lifts climbing motes OFF the
    // windward face into a visible arc; raise 3.2→2.4 — strong enough to pour up + spill (with the now-low
    // clearance the arcs read against the surface) but NOT the v9 "9" that pinned every mote at the ceiling.
    this.liftGain = p.liftGain ?? 2.4;
    this.relax = p.relax ?? 0.1;        // gentle pull back toward nominal clearance (τ~10s) — anti-deplete
                                        // without damping the multi-second pour-over transient.
    // v10 POUR over deflect: v9's strong deflection (0.9) routed motes AROUND ridges ALONG the contour —
    // flat horizontal streaks, the opposite of v10's "stream UP windward faces and SPILL over crests". Drop
    // to 0.25 so most of the into-slope horizontal wind is KEPT → motes drive UP and OVER the crest (the
    // minClear clamp rides them along the surface; once they crest, the gradient flips and w<0 spills them
    // down the lee). `w` is computed pre-deflection so the vertical pour is unchanged — only the horizontal
    // path reorients from along-contour to into-and-over, and more motes dwell in the bright climbing state.
    this.deflect = p.deflect ?? 0.25;
    // v10 DENSITY: bias the seed distance toward the NEAR field so the cloud is THICK around the bird and
    // thins into the distance. ahead = near + (far−near)·rand^nearBias; k=2.6 lands ~70% of motes in the
    // near third (the camera→bird→just-beyond band; bird sits ~followDist=120m ahead) — inverts v9's
    // uniform-world seed where perspective made the FAR field read densest.
    this.nearBias = p.nearBias ?? 2.6;
    // CURVED long tails: segments × segStep seconds of flow integrated backward = the comet arc. Much
    // longer than v8: 10 × 0.5s ≈ 5s of real flow (~35-50m near/mid-field) so the arc spans the deflection
    // zone near a ridge and the curve is unmistakable. segStep is the cheap length knob (no extra verts).
    this.segments = p.segments ?? 10;
    this.segStep = p.segStep ?? 0.5;
    // small head: many tiny motes, not star-like blobs.
    this.dotPx = p.dotPx ?? 2.6;
    // calmest-air segment step = 25% (short stubby arc); fast air = full step (long arc). Kept LOW so the
    // speed contrast stays steep.
    this.tailFloor = p.tailFloor ?? 0.25;
    // even the calmest air keeps ~30% of motes visible → wind EVERYWHERE, just sparse where slow. Raised
    // from 0.2 so the full-frame read stays populated (the curved tails are sparser per-mote than v8 dots).
    this.densityFloor = p.densityFloor ?? 0.3;
    // calibrated to the field's real |windAt| distribution (sampled min ~0.03, max ~16.5, mean ~8.5):
    // smoothstep(2,15) stretches the contrast across the bulk so calm→fast spans the full 0..1.
    this.speedLo = p.speedLo ?? 2.0;
    this.speedHi = p.speedHi ?? 15.0;

    this.px = new Float32Array(this.count);
    this.py = new Float32Array(this.count);
    this.pz = new Float32Array(this.count);
    this.speedFrac = new Float32Array(this.count);
    this.ptX = new Float32Array(this.segments + 1);
    this.ptY = new Float32Array(this.segments + 1);
    this.ptZ = new Float32Array(this.segments + 1);

    this.vertexCount = this.count * this.segments * 6;
    this.vertBytes = new ArrayBuffer(this.vertexCount * Wind.FPV * 4);
    this.vertHost = new Float32Array(this.vertBytes);
    this.vbuf = device.createBuffer({
      size: this.vertHost.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    // uniform: mat4 viewProj(16) + eye.xyz(3) + aspect(1) + fogColor.rgb(3) + fogDensity(1)
    //          + misc.x=dotSize(1) + misc.yzw pad(3) = 28 floats (112 B). The misc vec4 keeps the WGSL
    //          struct std140 size even though only dotSize is used now (tail/density moved CPU-side).
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
              { shaderLocation: 0, offset: 0, format: "float32x3" },  // segment endpoint world pos
              { shaderLocation: 1, offset: 12, format: "float32x2" }, // corner: x=near/far endpoint, y=perp
              { shaderLocation: 2, offset: 20, format: "float32" },   // speedFrac
              { shaderLocation: 3, offset: 24, format: "float32x2" }, // segDir.xz (world XZ of this segment)
              { shaderLocation: 4, offset: 32, format: "float32" },   // along: head=0 → tail=1 fade
              { shaderLocation: 5, offset: 36, format: "float32" },   // vis: CPU-side density cull (0/1·smooth)
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

  // Place mote i at a fresh world position UNIFORMLY within the camera-relative wedge. ahead is drawn in
  // [aheadMin, spanAhead]; lateral spread widens with distance so the wedge matches the view frustum
  // (near motes land on-screen, not off to the sides). Used to seed the whole field and to wrap motes
  // that drift out. A nonzero aheadMin (recycle) reseeds nearer the far edge to refill as the cam moves.
  private seedMote(
    i: number,
    camGround: [number, number],
    camFwd: [number, number],
    camRight: [number, number],
    aheadMin: number,
    lateralSign: number // 0 = either side; ±1 = force this side (for side-exit wrap)
  ): void {
    // v10 DENSITY: bias the distance toward the NEAR field — ahead = base + (far−base)·rand^nearBias,
    // k>1 clusters most motes near `base`, thinning into the distance. base is floored at ~8% of spanAhead
    // (~75m) so the THICK cloud peaks AT THE BIRD (which sits ~followDist=120m ahead of the camera), NOT
    // under the camera (ahead≈0 motes fall off the bottom edge / off the sides at the 0.15 wedge floor).
    // On a front-exit reseed (aheadMin = spanAhead·0.6) the larger aheadMin wins, refilling near the far edge.
    const base = Math.max(this.spanAhead * 0.08, aheadMin);
    const ahead = base + Math.pow(Math.random(), this.nearBias) * (this.spanAhead - base);
    const wedge = Math.min(1, Math.max(0.15, ahead / this.spanAhead));
    let lat = Math.random();             // 0..1 magnitude fraction
    let sign = lateralSign;
    if (sign === 0) sign = Math.random() < 0.5 ? -1 : 1;
    const lateral = sign * lat * this.spanWide * wedge;
    const x = camGround[0] + camFwd[0] * ahead + camRight[0] * lateral;
    const z = camGround[1] + camFwd[1] * ahead + camRight[1] * lateral;
    this.px[i] = x;
    this.pz[i] = z;
    // seed height at nominal clearance over the terrain; advection then pours it up/over the ridges.
    this.py[i] = this.sampleHeight(x, z) + this.clearance;
  }

  // TERRAIN-SHAPED flow at world (x,z): the frozen horizontal windAt, plus a VERTICAL component and a
  // HORIZONTAL into-slope DEFLECTION derived from the terrain gradient (finite-diff of sampleHeight).
  // Returns [wx, wz, w] (m/s). The SAME function feeds both mote advection and the curved-tail integration
  // so the streaks trace exactly the flow that carries the motes. windAt itself is untouched (frozen).
  private flowAt(x: number, z: number, t: number): [number, number, number] {
    let [wx, wz] = windAt(x, z, t, this.cfg);
    // terrain gradient via central finite-diff: grad points UPHILL; magnitude ~ slope.
    const e = 6.0; // meters
    const hxp = this.sampleHeight(x + e, z);
    const hxm = this.sampleHeight(x - e, z);
    const hzp = this.sampleHeight(x, z + e);
    const hzm = this.sampleHeight(x, z - e);
    const gx = (hxp - hxm) / (2 * e); // dH/dx (rise per meter east)
    const gz = (hzp - hzm) / (2 * e); // dH/dz (rise per meter north)
    // VERTICAL: w = horizontalWind · uphill-gradient → wind climbing a windward slope rises, descends a lee.
    const w = this.liftGain * (wx * gx + wz * gz);
    // HORIZONTAL DEFLECTION: remove the component of the wind pointing INTO a rising slope (downhill of
    // grad), scaled by deflect and by slope steepness so flow bends AROUND peaks / OVER crests, not through.
    const gmag = Math.hypot(gx, gz);
    if (gmag > 1e-5) {
      const nx = gx / gmag, nz = gz / gmag;   // unit uphill normal
      const into = wx * nx + wz * nz;         // +into = blowing UP the slope (the part to shed)
      if (into > 0) {
        // steeper slope → deflect more; cap the steepness term so gentle ground barely bends.
        const steep = Math.min(1, gmag * 4.0);
        const k = this.deflect * steep;
        wx -= k * into * nx;
        wz -= k * into * nz;
      }
    }
    return [wx, wz, w];
  }

  // Advect every mote by windAt (p += w*dt), boundary-WRAP ones that leave the camera-relative wedge,
  // then rebuild the billboard vertex buffer from the PERSISTED positions. dt is derived from sim time
  // and clamped so the first frame and any tab-stall don't fling the motes.
  private step(
    camGround: [number, number],
    camFwd: [number, number],
    camRight: [number, number],
    t: number
  ): void {
    // seed once: scatter ALL motes uniformly across the whole wedge so frame-0 is already full coverage.
    if (!this.seeded) {
      for (let i = 0; i < this.count; i++) {
        this.seedMote(i, camGround, camFwd, camRight, -this.spanBehind, 0);
      }
      this.seeded = true;
    }

    let dt = this.lastTime < 0 ? 0 : t - this.lastTime;
    this.lastTime = t;
    if (dt < 0) dt = 0;
    if (dt > 0.05) dt = 0.05; // clamp stalls

    const lo = this.speedLo;
    const hi = this.speedHi;
    const v = this.vertHost;
    const corners = Wind.CORNERS;
    const seg = this.segments;
    const ptX = this.ptX, ptY = this.ptY, ptZ = this.ptZ;
    let vi = 0;
    for (let i = 0; i < this.count; i++) {
      let x0 = this.px[i]!;
      let z0 = this.pz[i]!;
      let y0 = this.py[i]!;

      // Boundary-WRAP: if the mote has left the wedge, reseed it back into the opposite boundary so
      // coverage stays full with no persistent gap and no mid-view pop-in (reseeds happen at the edges).
      const rx = x0 - camGround[0];
      const rz = z0 - camGround[1];
      const fwdDist = rx * camFwd[0] + rz * camFwd[1];
      const sideDist = rx * camRight[0] + rz * camRight[1];
      if (fwdDist < -this.spanBehind || fwdDist > this.spanAhead) {
        // exited front/back → reseed near the far-ahead edge (refills as the camera advances).
        this.seedMote(i, camGround, camFwd, camRight, this.spanAhead * 0.6, 0);
        x0 = this.px[i]!; z0 = this.pz[i]!; y0 = this.py[i]!;
      } else if (Math.abs(sideDist) > this.spanWide) {
        // exited a side → reseed on the OPPOSITE side at a fresh distance (wedge stays balanced).
        this.seedMote(i, camGround, camFwd, camRight, -this.spanBehind, sideDist > 0 ? -1 : 1);
        x0 = this.px[i]!; z0 = this.pz[i]!; y0 = this.py[i]!;
      }

      // advect by the TERRAIN-SHAPED flow: horizontal (deflected) + vertical pour over the slope.
      const [wx, wz, w] = this.flowAt(x0, z0, t);
      const x = x0 + wx * dt;
      const z = z0 + wz * dt;
      // height advected by w, then a mild relaxation toward nominal clearance (anti-deplete) and clamps.
      const terr = this.sampleHeight(x, z);
      let y = y0 + w * dt;
      y += (terr + this.clearance - y) * Math.min(1, this.relax * dt);
      const loY = terr + this.minClear;
      const hiY = terr + this.maxClear;
      if (y < loY) y = loY;
      if (y > hiY) y = hiY;
      this.px[i] = x;
      this.py[i] = y;
      this.pz[i] = z;

      const wspeed = Math.hypot(wx, wz);
      // calibrated smoothstep(lo,hi): calm→fast spans the full 0..1 so density+tail contrast reads.
      const u = Math.min(1, Math.max(0, (wspeed - lo) / (hi - lo)));
      let sp = u * u * (3 - 2 * u);
      // v10 POUR-READS: brightness + tail-length + density-survival all key off `sp`, but on a windward
      // face the into-slope DEFLECTION collapses the HORIZONTAL speed exactly where the VERTICAL pour w is
      // strongest — so climbing motes would go dim+short, hiding the pour. Fold the upward climb into sp so
      // a mote streaming UP a ridge is bright, long-tailed, and survives the cull. wScale≈7 m/s maps a strong
      // pour to ~full. The backward tail (integrated along flowAt) then arcs DOWN the windward face behind the
      // bright climbing head — that descending bright arc IS the visible pour over the crest.
      const climbFrac = Math.min(1, Math.max(0, w / 7.0));
      sp = Math.max(sp, climbFrac);
      this.speedFrac[i] = sp;

      // CPU-side DENSITY cull (replaces the fragile vidx/6u hash in the shader, which broke once a mote
      // spans many verts): stable per-mote rank vs a speed-dependent cutoff. Calm air keeps a floor.
      const rank = hashRank(i);
      const cutoff = this.densityFloor + (1 - this.densityFloor) * sp;
      const vis = 1 - smoothstep(cutoff - 0.1, cutoff + 0.02, rank);
      if (vis <= 0.001) {
        // emit degenerate (collapsed) verts for every segment so the draw count stays fixed.
        for (let s = 0; s < seg; s++) {
          for (let c = 0; c < 6; c++) {
            v[vi++] = x; v[vi++] = y; v[vi++] = z;
            v[vi++] = 0; v[vi++] = 0;
            v[vi++] = sp;
            v[vi++] = 0; v[vi++] = 0;
            v[vi++] = 1; v[vi++] = 0; // along=1 (tail), vis=0
          }
        }
        continue;
      }

      // CURVED TAIL: integrate BACKWARD along flowAt from the head, building a polyline of seg+1 points.
      // tail length scales with speed (calm = short stub, fast = long arc). Each step is RK-lite (single
      // eval) upwind; the path bends as flowAt changes over the terrain → the comet arcs over the ridges.
      const stepLen = this.segStep * (this.tailFloor + (1 - this.tailFloor) * sp);
      ptX[0] = x; ptY[0] = y; ptZ[0] = z;
      let cx = x, cy = y, cz = z;
      for (let s = 1; s <= seg; s++) {
        const [fwx, fwz, fw] = this.flowAt(cx, cz, t);
        cx -= fwx * stepLen;
        cz -= fwz * stepLen;
        cy -= fw * stepLen;
        // keep the tail above terrain so it doesn't sink into the ridge behind a crest.
        const tb = this.sampleHeight(cx, cz) + this.minClear;
        if (cy < tb) cy = tb;
        ptX[s] = cx; ptY[s] = cy; ptZ[s] = cz;
      }

      // emit a quad per segment between consecutive polyline points. corner.x picks near(0)/far(1) point.
      for (let s = 0; s < seg; s++) {
        const ax = ptX[s]!, ay = ptY[s]!, az = ptZ[s]!;       // near (toward head)
        const bx = ptX[s + 1]!, by = ptY[s + 1]!, bz = ptZ[s + 1]!; // far (toward tail)
        // segment direction in world XZ (for screen-space perp orientation in the VS).
        let sdx = bx - ax, sdz = bz - az;
        const sdl = Math.hypot(sdx, sdz);
        if (sdl > 1e-5) { sdx /= sdl; sdz /= sdl; } else { sdx = 1; sdz = 0; }
        const alongN = s / seg;       // head=0 → tail=1 across the whole ribbon
        const alongF = (s + 1) / seg;
        for (let c = 0; c < 6; c++) {
          const [pick, perp] = corners[c]!; // pick: 0=near endpoint, 1=far endpoint
          const ex = pick > 0.5 ? bx : ax;
          const ey = pick > 0.5 ? by : ay;
          const ez = pick > 0.5 ? bz : az;
          const al = pick > 0.5 ? alongF : alongN;
          v[vi++] = ex; v[vi++] = ey; v[vi++] = ez;
          v[vi++] = pick; v[vi++] = perp;
          v[vi++] = sp;
          v[vi++] = sdx; v[vi++] = sdz;
          v[vi++] = al; v[vi++] = vis;
        }
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

    // dotPx (screen px diameter) → NDC half-width for the ribbon perpendicular thickness. The curved
    // tail LENGTH now lives in the integrated world-space polyline (built in step()), not in a uniform —
    // so the only screen-space sizing left here is the ribbon half-width. Viewport-independent: dotPx/refW.
    const REF_W = 1000; // reference canvas width the dotPx was tuned against
    const dotSize = this.dotPx / REF_W;          // NDC half-width of the comet ribbon

    const u = this.uniformF32;
    u.set(viewProj, 0);
    u[16] = eye[0]; u[17] = eye[1]; u[18] = eye[2];
    u[19] = aspect;                               // pxW/pxH for un-skewed perpendicular thickness
    u[20] = fogColor[0]; u[21] = fogColor[1]; u[22] = fogColor[2];
    u[23] = fogDensity;
    u[24] = dotSize;                              // NDC half-width of the ribbon
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
