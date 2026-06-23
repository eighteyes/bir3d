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
//   - Wind: a render pipeline (wind.wgsl) that draws drifting neon COMET motes over the terrain in TWO
//     TIERS for legibility (v11):
//       FAR / DISTANCE = the existing persistent population of LONG curved streamline lines (the v9/v10
//         model below) — distant wind reads as long flowing arcs over the ridges.
//       NEAR / UP-CLOSE = a DENSE WIND SPHERE of LITTLE short-tailed comets seeded in a BALL centered on
//         the BIRD (radius ~80m), advected by the same terrain-aware flowAt; comets that drift outside the
//         ball are recycled back inside so the sphere follows the bird. Rendered DENSE (no density cull —
//         vis=1 for every near mote) so the local air is unmistakably legible right where the bird is.
//     Both tiers share one pipeline/shader/vertex-format and draw in a single combined vertex buffer.
//     RENDER MODES (phase-1 scaffold): each tier has a selectable MODE (FarMode/NearMode/WakeMode) that
//       switches which geometry is emitted into that tier's buffer SPAN — the "A" defaults
//       (comet/comet/modulate) reproduce today's look; divergent B/C geometries land in a later phase. The
//       buffer carries a THIRD reserved span (worst-case wake-shed quads) so the helix/ring shed geometry
//       has a home + the draw plumbing is verified early; draw() issues up to three offset draws into it.
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
  thermalAmp: 5.0,
};

// --- v13: GPU-fluid field (the structured horizontal SOURCE) ---
// Module-level handle to the latest async-readback fluid velocity, plus its bird-local world→grid
// mapping. windAt() uses this as its BASE horizontal vector when set (else falls back to the analytic
// curl-noise below — covers the first frames before the first readback resolves). bird3d (physics) and
// the Wind motes (via flowAt→windAt) BOTH read windAt, so wiring the field here makes both ride the
// fluid with NO change to either. setFluidField is called once per frame from bird-main.ts.
interface FluidField {
  u: Float32Array;    // bordered (gridW+2)*(gridH+2) f32 — fluid velocity x (grid-space)
  v: Float32Array;    // bordered (gridW+2)*(gridH+2) f32 — fluid velocity y (grid-space)
  gridW: number;      // interior cells across (x)
  gridH: number;      // interior cells across (z)
  originX: number;    // world X at grid interior cell (0,*) center — window origin (moves with bird)
  originZ: number;    // world Z at grid interior cell (*,0) center
  cellM: number;      // meters per grid cell (world span = gridW*cellM)
  scale: number;      // grid-velocity → m/s scale (calibrated so |sampled| lands in the flyable band)
}
let fluidField: FluidField | null = null;

// Peak |fluid wind| (m/s) the FLUID component is clamped to (drift is added on top). Tuned with the
// regulator's targetBand so the SAMPLED field lands mean ~10 / max ~16 — in the spec's 10-15 flyable
// band, no +61° blow-around (drift+16° crab preserved), AND matching the shipped analytic field's speed
// distribution so the motes' speed-cull survival (→ the per-mote 10-segment sampleHeight tail loop, the
// dominant CPU cost) does not spike above the analytic baseline.
const FLUID_MAX = 10;

// Per-frame setter (bird-main.ts). uArr/vArr are the latest readback of fluid.velocityX/Y (bordered
// (gridW+2)*(gridH+2)). The window is BIRD-LOCAL: originX/Z is the world position of interior cell
// (0,0); it moves with the bird so the field is always live where the bird is. scale maps the raw
// grid velocity into m/s (calibrated against the measured readback magnitude).
export function setFluidField(
  uArr: Float32Array,
  vArr: Float32Array,
  gridW: number,
  gridH: number,
  originX: number,
  originZ: number,
  cellM: number,
  scale: number
): void {
  fluidField = { u: uArr, v: vArr, gridW, gridH, originX, originZ, cellM, scale };
}

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

// Catmull-Rom spline interpolation: smooth value at t∈[0,1] on the segment p1→p2 with neighbors p0,p3.
// Interpolating (passes through p1 at t=0, p2 at t=1) → resampling a polyline with it keeps the original
// vertices and rounds the corners between them. Used to subdivide the far wind tail into a smooth curve.
function catmullRom(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const t2 = t * t, t3 = t2 * t;
  return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
}

// SHARED terrain into-slope DEFLECTION — the SINGLE implementation used by BOTH Wind.flowAt (the motes)
// and bird3d (the physics) so the bird drifts with EXACTLY the vector the motes ride. PURE: takes the
// horizontal wind [wx,wz] and the terrain gradient [gx,gz] (the caller computes the gradient once — no
// sampleHeight here, so the hot per-segment mote loop is never double-sampled). Sheds the component
// blowing UP the slope so flow bends AROUND peaks / OVER crests instead of through them.
export function flowHorizontal(
  wx: number, wz: number, gx: number, gz: number, deflect: number
): [number, number] {
  const gmag = Math.hypot(gx, gz);
  if (gmag > 1e-5) {
    const nx = gx / gmag, nz = gz / gmag;   // unit uphill normal
    const into = wx * nx + wz * nz;         // +into = blowing UP the slope (the part to shed)
    if (into > 0) {
      const steep = Math.min(1, gmag * 4.0); // steeper → deflect more; capped so gentle ground barely bends
      const k = deflect * steep;
      wx -= k * into * nx;
      wz -= k * into * nz;
    }
  }
  return [wx, wz];
}

// Module-level scratch tuple for windAt's single return allocation. windAt is called tens of thousands
// of times per frame (every mote × every tail segment); returning a fresh array each call was ~7ms/frame
// of GC pressure (measured). Every caller (bird3d, flowAt) destructures the result IMMEDIATELY
// (`const [wx,wz] = windAt(...)`), reading it out before the next call overwrites the scratch — so reuse
// is safe and transparent (no signature change).
const _w: [number, number] = [0, 0];

// Horizontal wind [wx, wz] m/s at world (x,z), time t.
// v13: BASE = the GPU-fluid sample (the structured, EVOLVING horizontal source) when a field is set
// (async readback wired from bird-main.ts), else the analytic curl-noise (fallback for the first frames
// before readback resolves). In BOTH cases the steady PREVAILING DRIFT is ADDED — a zero-mean fluid swirl
// alone would collapse the crab/ground-track onto heading and regress the felt-wind; the drift is the
// ever-present cross-track shove. windAt stays the single source of truth for bird physics + motes.
// Returns the shared _w scratch — destructure it before the next call.
export function windAt(
  x: number,
  z: number,
  t: number,
  cfg: Required<WindConfig> = DEFAULTS
): [number, number] {
  // steady prevailing drift (large-scale, ever-present cross-track shove) — KEPT in every branch.
  const dx = Math.sin(cfg.driftDir) * cfg.driftAmp;
  const dz = Math.cos(cfg.driftDir) * cfg.driftAmp;

  // BASE = the GPU-fluid bilinear sample when a field is set (the real evolving source). Inlined (no
  // intermediate array) so the hot mote loop stays allocation-free. Interior cells are indexed
  // i+(gridW+2)*j with i,j in 1..gridW — the +1 border offset is critical (off-by-one silently shifts
  // the whole field). World (originX,originZ) maps to interior cell (0,0) → grid coord (x-originX)/cellM.
  const f = fluidField;
  if (f) {
    const gx = (x - f.originX) / f.cellM;
    const gz = (z - f.originZ) / f.cellM;
    if (gx >= 0 && gz >= 0 && gx <= f.gridW - 1 && gz <= f.gridH - 1) {
      const i0 = gx | 0, j0 = gz | 0;
      const i1 = i0 + 1 < f.gridW ? i0 + 1 : i0;
      const j1 = j0 + 1 < f.gridH ? j0 + 1 : j0;
      const tx = gx - i0, tz = gz - j0;
      const stride = f.gridW + 2;
      const u = f.u, v = f.v;
      // +1 on both axes → skip the border into the interior.
      const a = (i0 + 1) + stride * (j0 + 1);
      const b = (i1 + 1) + stride * (j0 + 1);
      const c = (i0 + 1) + stride * (j1 + 1);
      const d = (i1 + 1) + stride * (j1 + 1);
      let ux = ((u[a]! * (1 - tx) + u[b]! * tx) * (1 - tz) + (u[c]! * (1 - tx) + u[d]! * tx) * tz) * f.scale;
      let vx = ((v[a]! * (1 - tx) + v[b]! * tx) * (1 - tz) + (v[c]! * (1 - tx) + v[d]! * tx) * tz) * f.scale;
      // Clamp the fluid sample magnitude to match the analytic field's distribution (max ~16.5 m/s) it
      // replaced. The regulator pins the MEAN, but local cells run 2-3× mean (peaky); unclamped peaks
      // (a) push the wind past the proven-flyable band → the +61° blown-around regression, and (b) raise
      // the motes' speed-cull survival → more run the 10-segment sampleHeight tail loop → a CPU spike.
      // Clamping the peaks to the analytic max fixes BOTH (feel stays in-band, cull survival matches).
      const mag = Math.hypot(ux, vx);
      if (mag > FLUID_MAX) { const s = FLUID_MAX / mag; ux *= s; vx *= s; }
      _w[0] = ux + dx;
      _w[1] = vx + dz;
      return _w;
    }
  }

  // FALLBACK: analytic curl-noise (first frames before readback resolves, or outside the window).
  const sc = cfg.curlScale;
  const e = 0.75; // finite-diff step in SCALED units
  const px = x * sc, pz = z * sc;
  // curl of scalar potential
  const dPot_dz =
    (potential(px, pz + e, t) - potential(px, pz - e, t)) / (2 * e);
  const dPot_dx =
    (potential(px + e, pz, t) - potential(px - e, pz, t)) / (2 * e);
  _w[0] = dPot_dz * cfg.curlAmp + dx;
  _w[1] = -dPot_dx * cfg.curlAmp + dz;
  return _w;
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
  // strong only in rising cores you must HUNT for (a glider sinks by default; lift is local).
  // v14: exponent 2.2→1.8 BROADENS the cores so lift is findable (still ~0 over half the world where
  // either lobe is negative — the hunt remains, the needles are just wider).
  const core = Math.max(0, a) * Math.max(0, b);
  return Math.pow(core, 1.8) * cfg.thermalAmp;
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

// --- GLOBAL WIND atmosphere: horizontal wind magnitude scales with ABSOLUTE altitude y (calm low → strong
// high). ONE profile, multiplied into wind at every consumer (bird drift + ridge lift + motes) — uniform, no
// special rules. Absolute altitude means ridges (high) sit in strong wind → ridge soaring stays intact, and
// valleys (low) are calm for free (terrain shelter emerges from altitude). Module-level + a setter so the bird
// (module windAt/updraftAt) and the motes (instance) share the SAME profile and it stays live-tunable. A pure
// function of y — no terrain sampling — so it's cheap in the 10k×/frame hot paths. windAt/thermalAt stay frozen.
export const windProfileParams = { loScale: 0.4, hiScale: 1.4, altLo: 100, altHi: 500 };
export function setWindProfile(p: Partial<typeof windProfileParams>): void {
  Object.assign(windProfileParams, p);
  // GUARDS (the live debug setter accepts arbitrary input): non-negative magnitudes (a negative scale would
  // REVERSE wind/lift sign) and altLo < altHi (altLo ≥ altHi would INVERT the atmosphere — calm high, windy low).
  const q = windProfileParams;
  q.loScale = Math.max(0, q.loScale);
  q.hiScale = Math.max(0, q.hiScale);
  if (q.altHi <= q.altLo) q.altHi = q.altLo + 1;
}
export function windProfile(y: number): number {
  const p = windProfileParams;
  return p.loScale + (p.hiScale - p.loScale) * smoothstep(p.altLo, p.altHi, y);
}
// RIDGE LIFT uses the STRONG free-stream wind ALOFT (the saturated profile), NOT the bird's calm low-altitude
// value — a 150m ridge has strong wind flowing over it just like a 500m peak, so ridge soaring works at ANY
// ridge height and the bird is never stranded low. DRIFT stays altitude-scaled (windProfile); LIFT uses aloft.
export function windAloftScale(): number {
  return windProfileParams.hiScale;
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
  vSpread?: number;   // v14: half-height (m) of the vertical band each mote's HOME clearance is spread across
  homeBias?: number;  // power (≥1) biasing far-mote HOME heights toward the terrain — most hug, a thin tail reaches aloft
                      //      (centered on `clearance`, clamped into [minClear,maxClear]). 0 = a single flat
                      //      sheet at clearance (old behaviour); larger = the wedge field FILLS a volume.
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
  // --- v11 NEAR WIND SPHERE (dense little comets centered on the bird) ---
  nearCount?: number;    // number of little comets packed in the bird-centered ball (DENSE)
  nearRadius?: number;   // radius (m) of the sphere around the bird
  nearSegments?: number; // tail segment count for the LITTLE comets (short → distinct from far lines)
  nearSegStep?: number;  // seconds of flow integrated per near-comet tail segment (short tail)
  // --- v12 FADE envelope (no-pop on recycle): fade IN on (re)seed, fade OUT before recycle ---
  fadeInTime?: number;   // seconds over which a freshly seeded mote ramps vis 0→1 (smoothstep)
  fadeFarEdge?: number;  // meters before a FAR span boundary where the fade-OUT begins (→0 at the edge)
  fadeNearEdge?: number; // fraction (0..1) of the near ball radius at which the fade-OUT begins (→0 at R)
  // --- BIRD WAKE (visuals only): the bird stirs the air INSIDE the near sphere. Added ONLY to the near-mote
  //     advection (+ their tails); windAt / flight physics stay FROZEN. All three scale with bird speed and
  //     fall off to 0 at the ball edge so motes circulate rather than eject. 0 = no stir. ---
  bowGain?: number;    // radial OUTWARD push AHEAD of the bird — motes part at the nose (bow wave)
  wakeGain?: number;   // along-motion DRAG behind the bird — slipstream the wake follows the bird
  swirlGain?: number;  // TANGENTIAL swirl in the wake — now drives the TWIN wingtip vortices (per-core circulation)
  // --- SLIPSTREAM (twin wingtip vortices + wing emission + body attach) — render-only, near sphere ---
  wingSpan?: number;        // half-span (m): lateral offset of each wingtip vortex core from the bird centerline
  vortexCore?: number;      // Rankine core radius rc (m): the trailing vortex swirl peaks at this distance from the core line
  wingEmitFrac?: number;    // fraction of (re)seeded near motes BORN at the wingtips (the two streams); the rest fill the body
  wingJitter?: number;      // random spread (m) around a wingtip emission point so the stream is a soft cord, not a wire
  ambientNearFloor?: number; // ambient (global) terrain-wind weight AT the bird (0..1); 1 = near motes ride the FULL global wind (immersion); <1 makes the near sphere STICK to the bird near its center
  // --- TOUCHED AIR (the wind the bird physically touched glows warm + trails longer) — render-only ---
  heatTau?: number;         // seconds for a touched mote's heat to decay back toward 0 (memory of being touched)
  heatRef?: number;         // wake speed (m/s) that maps to FULL heat (red + max length); gentler wake → yellow + shorter
  heatLenGain?: number;     // extra tail length at full heat (1 = up to 2× longer for the hardest-touched air)
  foreStretch?: number;     // forward reach of the near bubble as a multiple of nearRadius (>1 = bigger bubble AHEAD of the bird so motes read in front)
}

// Per-TIER render MODE selectors (Phase 1 scaffold). Each tier shares the ONE pipeline/shader/vertex
// format; the mode only switches which geometry is emitted into that tier's buffer span. The "A" default
// of each tier ("comet"/"comet"/"modulate") reproduces today's look — the divergent B/C geometries land
// in a later phase (their branches currently fall through to the A emission).
// The mode ARRAYS are the single source of truth; each union type is DERIVED from its array so the
// UI cycle-buttons (which pass these arrays directly) can never drift from the engine's accepted modes.
export const FAR_MODES = ["comet", "stipple", "chevron"] as const;   // FAR / distance long-line tier
export type FarMode = typeof FAR_MODES[number];
export const NEAR_MODES = ["comet", "flecks", "filaments"] as const; // NEAR / local sphere tier
export type NearMode = typeof NEAR_MODES[number];
export const WAKE_MODES = ["modulate", "helix", "rings"] as const;   // WAKE: "modulate" = today's velocity overlay (no own geometry); helix/rings = future shed geometry into the reserved span
export type WakeMode = typeof WAKE_MODES[number];

export class Wind {
  private cfg: Required<WindConfig>;
  private count: number;          // total motes
  private spanAhead: number;
  private spanBehind: number;
  private spanWide: number;
  private clearance: number;
  private minClear: number;
  private maxClear: number;
  private vSpread: number;
  private homeBias: number;
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

  // --- v11 NEAR WIND SPHERE state (bird-centered dense little comets) ---
  private nearCount: number;
  private nearRadius: number;
  private nearSegments: number;
  private nearSegStep: number;
  // v12 FADE envelope tuning.
  private fadeInTime: number;
  private fadeFarEdge: number;
  private fadeNearEdge: number;
  // BIRD WAKE (visuals only) gains — see DotParams.
  private bowGain: number;
  private wakeGain: number;
  private swirlGain: number;
  // SLIPSTREAM tunables (see DotParams).
  private wingSpan: number;
  private vortexCore: number;
  private wingEmitFrac: number;
  private wingJitter: number;
  private ambientNearFloor: number;
  // TOUCHED AIR (see DotParams).
  private heatTau: number;
  private heatRef: number;
  private heatLenGain: number;
  private foreStretch: number;
  // reusable scratch for the per-point bird-wake disturbance (avoids allocation in the hot near-tail loop).
  private readonly _wake: [number, number, number] = [0, 0, 0];
  // per-frame near-wake FRAME (motion axis + wing-right unit + speed + bird pos + moving flag) — set at the top
  // of stepNear, read by birdWakeAt (twin vortices), seedNearMote (wingtip emission), and the sampleWake probe.
  private _ax = 0; private _ay = 0; private _az = 0;
  private _rx = 1; private _ry = 0; private _rz = 0;
  private _bs = 0; private _moving = false;
  private _wakeOn = false;  // moving && showWake — gates the wake disturbance + wing emission within the near sphere
  private showNear = false; // LOCAL SPHERE (near-mote bubble) drawn? OFF by default — solving global wind first
  private showWake = false; // WAKE (bow/drag/twin-vortices/touched-air + wing emission) applied in the sphere? OFF by default
  // Per-tier render MODE (Phase 1 scaffold). "A" defaults reproduce today's geometry; B/C currently
  // fall through to A (see step()/stepNear()). wakeMode "modulate" = today's velocity overlay (no own
  // geometry); helix/rings will emit into the reserved wake-shed span in a later phase.
  private farMode: FarMode = "comet";
  private nearMode: NearMode = "comet";
  private wakeMode: WakeMode = "modulate";
  private readonly _lastBirdPos: [number, number, number] = [0, 0, 0];
  private nx: Float32Array;   // near-comet world x (per near mote)
  private ny: Float32Array;   // near-comet world y (advected height)
  private nz: Float32Array;   // near-comet world z
  private nearAge: Float32Array; // v12 FADE: per-near-comet AGE (s since last seed/recycle) → fade-IN ramp
  private nearHeat: Float32Array; // TOUCHED-AIR: per-near-comet heat (0..1) — decays over heatTau; warm tint + longer tail
  private nptX: Float32Array; // scratch polyline for the near comet (nearSegments+1)
  private nptY: Float32Array;
  private nptZ: Float32Array;
  private nearSeeded = false; // first draw with a valid bird pos seeds the ball
  private farVertexCount: number; // far-tier vertices (the long-line population)
  private nearVertexCount: number; // near-tier vertices (the sphere)
  // THIRD span: dedicated WAKE-SHED geometry (helix cords / shed rings) written AFTER far+near. The full
  // worst-case span is RESERVED in the buffer now (zero-filled this phase); wakeShedLiveCount tracks how
  // many verts were actually emitted this frame so draw() only issues the live count (0 until Phase 2).
  private wakeShedVertexCount: number; // reserved worst-case wake-shed vertices (buffer span)
  private wakeShedLiveCount = 0;       // wake-shed verts ACTUALLY written this frame (0 this phase → draw skipped)

  // PERSISTENT world-space particle state (NOT regenerated each frame — that is what makes them drift).
  // v9: each mote carries a 3D position (x, y, z); y is ADVECTED by the vertical flow w (terrain-shaped),
  // not pinned to a fixed clearance — so motes visibly pour up windward slopes and sink in lees.
  private px: Float32Array;   // world x (per mote)
  private py: Float32Array;   // world y — ADVECTED height (per mote)
  private pz: Float32Array;   // world z (per mote)
  // v14: per-mote HOME clearance (m above terrain) the height RELAXES toward. Distributed across a vertical
  // band at seed time so the wedge field fills a VOLUME instead of collapsing to one sheet at `clearance`.
  private pHome: Float32Array;
  private speedFrac: Float32Array; // cached 0..1 wind speed at the mote (for color/density/tail)
  // v12 FADE: per-mote AGE (seconds since last seed/recycle). Advanced by dt every frame, reset to 0 in
  // seedMote. Drives the fade-IN envelope (young motes ramp up) so a (re)seeded mote never POPS in.
  private age: Float32Array;

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
  // dense scratch polyline: the coarse far tail (segments+1 pts) Catmull-Rom resampled to
  // (segments·FAR_SUBDIV + 1) pts so the ribbon renders as a smooth CURVE, not 6 straight kinked pieces.
  private sptX: Float32Array;
  private sptY: Float32Array;
  private sptZ: Float32Array;

  private lastTime = -1;       // for per-frame dt derivation from bird.simTime
  private seeded = false;      // first draw seeds the field uniformly across the wedge

  // floats/vertex: center.xyz(3) + corner.xy(2) + speedFrac(1) + segDir.xz(2) + along(1) + vis(1) = 10.
  // The streak is now a CURVED ribbon: each segment is a quad between two integrated polyline points,
  // oriented along that segment's own screen-space direction; `along` is the head→tail fade fraction.
  private static FPV = 11; // 10 base + heat (loc 6) for the touched-air warm tint
  // v17 anti-geyser: hard clamp (m/s) on the per-mote VERTICAL flow w. Bounds the up-and-over pour so steep
  // faces (gradient ∝ RELIEF) lift into a believable arc instead of erupting straight up like a fountain.
  private static W_CLAMP = 12;
  // far tail render-time smoothing: each coarse segment is subdivided into FAR_SUBDIV rendered segments
  // via Catmull-Rom (interpolating → passes through the original integrated points). Pure geometry; adds
  // NO flowAt/sampleHeight calls (those stay at the coarse `segments` count — the cited CPU cost).
  // Phase-1 FAR-A short-comet unify: 3→2. With segments 6→4 the far comet is shorter so it can't show
  // hard corners at distance; FAR_SUBDIV still smooths each coarse segment via Catmull-Rom (pure geometry).
  private static FAR_SUBDIV = 2;
  // a quad per ribbon segment: x∈{0,1} picks the segment's near(0)/far(1) endpoint, y∈{-1,1} is the perp.
  private static CORNERS: ReadonlyArray<[number, number]> = [
    [0, -1], [1, -1], [1, 1],
    [0, -1], [1, 1], [0, 1],
  ];

  // --- WAKE-SHED reserved span (Phase 1: plumbing only; zero-filled, never drawn yet) ---
  // The later wake phase emits dedicated shed geometry (helix cords OR shed rings) into a THIRD buffer span
  // after far+near. Reserve the worst-case of the two future pools NOW so the buffer + offset plumbing is
  // built and verified early. 6 verts per quad in both pools (CORNERS quad, shared format).
  private static QUAD_VERTS = 6;
  // helix pool: 2 wingtip cords × 160 live segments-per-cord × 4 ribbon segs each.
  private static HELIX_TIPS = 2;
  private static HELIX_LIVE = 160;
  private static HELIX_SEGS = 4;
  private static HELIX_VERTS = Wind.HELIX_TIPS * Wind.HELIX_LIVE * Wind.HELIX_SEGS * Wind.QUAD_VERTS; // 7680
  // rings pool: 32 shed rings × 32 chords per ring.
  private static RING_COUNT = 32;
  private static RING_CHORDS = 32;
  private static RINGS_VERTS = Wind.RING_COUNT * Wind.RING_CHORDS * Wind.QUAD_VERTS;                  // 6144
  // reserve the larger of the two so EITHER wake mode fits without resizing the buffer.
  private static WAKE_SHED_RESERVE = Math.max(Wind.HELIX_VERTS, Wind.RINGS_VERTS);                    // 7680

  constructor(
    private device: GPUDevice,
    shader: string,
    colorFormat: GPUTextureFormat,
    private sampleHeight: (x: number, z: number) => number,
    cfg: WindConfig = {},
    p: DotParams = {},
    sampleCount = 1 // MSAA samples — must match the render target + every pipeline in the pass
  ) {
    this.cfg = { ...DEFAULTS, ...cfg };
    // v9: fewer motes than v8 (4200 → 2200) to AFFORD the multi-step curved-tail integration per mote
    // (each mote now integrates `segments` flow steps every frame). Still wind EVERYWHERE across the wedge.
    this.count = p.numMotes ?? 1600; // v17: 2200→1600 — fewer motes for fps headroom (per user), paired with a HIGH
                                     // always-on densityFloor below so the FEWER motes read as omnipresent air, not sparse.
    // spanAhead matches the terrain maxDist (~950) so every mote lives in the DRAWN terrain band —
    // dots past the cutoff floated over a void and read as detached sky specks, not wind in the scene.
    this.spanAhead = p.spanAhead ?? 950;
    this.spanBehind = p.spanBehind ?? 260;
    this.spanWide = p.spanWide ?? 950;
    // v10 HUG: drop the clearance the height RELAXES toward from 55→16 so the field rides JUST over the
    // ridges and follows the contour up and over each crest (v9's 55 floated a flat sheet above 220m relief).
    // height is advected by w (not pinned) so motes pour up/over ridges, with mild relaxation back to this low
    // nominal so the field doesn't deplete or pile at a ceiling.
    this.clearance = p.clearance ?? 30; // terrain-hugging band center (user: "motes close to terrain looks cool").
                                        // The altitude ATMOSPHERE is read via mote SPEED (windProfile), not float-height;
                                        // homeBias + vSpread leave a thin tail aloft so altitude isn't a dead void.
    this.minClear = p.minClear ?? 12;   // v17c: 5→12. At 5 the motes hugged RIGHT at the neon ridgelines and read
                                        // as "wind in the terrain" (the terrain is drawn as LINES; the depth-only
                                        // fill curtains have gaps so surface-hugging motes show through). 12 floats
                                        // the field clearly ABOVE the lines while still hugging close enough to interact.
    this.maxClear = p.maxClear ?? 200;  // v17: ceiling 170→200 — headroom for the up-and-over arcs over the taller
                                        // terrain WITHOUT the 260 over-spread that diluted areal density (flat/lee
                                        // air w≈0 → relax keeps motes low, so the ceiling only frees the climbing pour).
    // v10 POUR: dh/dt = (liftGain−1)·d(terr)/dt along the path, so liftGain>1 lifts climbing motes OFF the
    // windward face into a visible arc; raise 3.2→2.4 — strong enough to pour up + spill (with the now-low
    // clearance the arcs read against the surface) but NOT the v9 "9" that pinned every mote at the ceiling.
    this.vSpread = p.vSpread ?? 70;     // taller band for the altitude TAIL (homeBias clusters most motes low) so
                                        //      the wedge reads as a VOLUME of moving air, not a flat sheet. The
                                        //      ridge-pour (liftGain/relax) rides ON TOP of this spread unchanged.
    this.homeBias = p.homeBias ?? 2.5;  // cluster far motes near the terrain (cool hug) with a thin tail to altitude
    this.liftGain = p.liftGain ?? 0.6;  // v17 UP-AND-OVER: was 0.3 (flat slither) / 2.4 (geyser). Moderate pour, BUT
                                        // the vertical w is CLAMPED to ±W_CLAMP in flowAt — so steep faces (gradient ∝
                                        // RELIEF, now 1.9× steeper) lift into a visible arc instead of erupting.
    this.relax = p.relax ?? 0.8;        // v17: was 2.5 (snapped flat to the surface) — slower track (τ~1.25s) lets the
                                        // up-and-over arc BREATHE before it eases back toward home, so the pour reads.
    // v10 POUR over deflect: v9's strong deflection (0.9) routed motes AROUND ridges ALONG the contour —
    // flat horizontal streaks, the opposite of v10's "stream UP windward faces and SPILL over crests". Drop
    // to 0.25 so most of the into-slope horizontal wind is KEPT → motes drive UP and OVER the crest (the
    // minClear clamp rides them along the surface; once they crest, the gradient flips and w<0 spills them
    // down the lee). `w` is computed pre-deflection so the vertical pour is unchanged — only the horizontal
    // path reorients from along-contour to into-and-over, and more motes dwell in the bright climbing state.
    this.deflect = p.deflect ?? 0.45;   // v17 BOTH: was 0.85 (all-around) / 0.25 (all-over). Middle — sheds ~45% of the
                                        // into-slope wind (routes AROUND peaks) while keeping ~55% to drive UP-AND-OVER crests.
    // v11 DISTANCE: the dedicated near comet SPHERE now owns the bird vicinity, so the FAR long-line tier
    // is freed to SPREAD into the distance (v11 wants distant wind = long curved lines). Drop nearBias
    // 2.6→1.3: at 2.6 ~61% of far motes piled in the near third (16% reached the far third) — they crowded
    // the sphere and STARVED the distance; 1.3 rebalances to ~37/34/29% near/mid/far so the long lines
    // populate the distance where they read as flowing arcs. ahead = near + (far−near)·rand^nearBias.
    this.nearBias = p.nearBias ?? 1.3;
    // CURVED long tails: segments × segStep seconds of flow integrated backward = the comet arc. v13
    // PERF: the per-segment flowAt (4 sampleHeight each) tail loop dominated the CPU frame once the
    // livelier fluid wind let more far motes survive the speed cull (fps 60→51). Cut segments 10→6 and
    // bump segStep 0.5→0.8 so the arc keeps the SAME ~5s/35-50m flow span (and visible curve) with 40%
    // fewer flowAt+clamp evals and 40% fewer verts. Tail flowAt is further sub-sampled below (every 2nd
    // segment) to halve the remaining gradient cost while preserving the curve. segStep is the free length
    // knob (no extra verts); the curve survives because flow is still sampled along the path (not reused).
    // Phase-1 FAR-A short-comet unify: default 6→4 (shorter far comet that can't read as corners at
    // distance). segStep stays the free length knob; the buffer-sizing formula references this.segments so
    // the far span auto-updates. The curve still reads (Catmull-Rom over the coarse points).
    this.segments = p.segments ?? 4;
    this.segStep = p.segStep ?? 0.8;
    // small head: many tiny motes, not star-like blobs.
    this.dotPx = p.dotPx ?? 2.6;
    // calmest-air segment step = 25% (short stubby arc); fast air = full step (long arc). Kept LOW so the
    // speed contrast stays steep.
    this.tailFloor = p.tailFloor ?? 0.25;
    // v17 "it's AIR — always SOME wind" (user): 0.3→0.6 so the calmest air STILL keeps ~60% of motes visible.
    // Air is omnipresent; it must never thin to bare patches. Paired with the lower count above so average
    // surviving-mote count (the tail-loop cost) roughly holds 60fps while coverage becomes uniform everywhere.
    this.densityFloor = p.densityFloor ?? 0.6;
    // calibrated to the field's real |windAt| distribution (sampled min ~0.03, max ~16.5, mean ~8.5):
    // smoothstep(2,15) stretches the contrast across the bulk so calm→fast spans the full 0..1.
    this.speedLo = p.speedLo ?? 2.0;
    this.speedHi = p.speedHi ?? 15.0;

    // v11 NEAR SPHERE: a DENSE ball of LITTLE short-tailed comets around the bird. ~1600 motes in a tight
    // ~65m ball makes the local air unmistakably legible AND visually distinct from the far long lines (the
    // v11 fix). SHORT tails (3 × 0.12s ≈ 0.36s of flow, ~3-4m) keep them reading as LITTLE comets/dots, not
    // the far streamlines. Tight radius + high count = a thick visible cloud right at the bird. Count chosen
    // to hold a clean 60fps (2200 read as borderline 56-59); the near tail reuses the head flow + a single
    // tip terrain clamp to stay cheap (no per-segment flowAt/sampleHeight).
    this.nearCount = p.nearCount ?? 1600; // NOTE: bird-main overrides this to 800 (the live near-sphere count);
                                          // this default is the fallback only. Tune the sphere size there, not here.
    this.nearRadius = p.nearRadius ?? 65;
    this.nearSegments = p.nearSegments ?? 4; // a 4th segment → smoother CURLING tail arcs
    this.nearSegStep = p.nearSegStep ?? 0.12;

    // v12 FADE: motes RECYCLE (far = span-boundary reseed; near = ball-edge reseed) and the position
    // SNAPS to a new seed → a hard POP in at the new spot and POP out when it leaves. Fold a fade envelope
    // into the existing `vis`: fadeIn(age) ramps 0→1 over the first ~0.55s of a mote's life; fadeOut ramps
    // 1→0 as the mote nears its recycle condition (far: distance to the nearest span exit; near: distance
    // to the ball edge). Multiplied into vis → every mote eases in on (re)seed and eases out before recycle.
    this.fadeInTime = p.fadeInTime ?? 0.55;   // ~0.4-0.7s ramp; smoothstep keeps the onset gentle.
    this.fadeFarEdge = p.fadeFarEdge ?? 120;  // last 120m before a far span boundary fades to 0 — wide enough
                                              // to soften the outer wedge edge (the ball handoff is separate).
    this.fadeNearEdge = p.fadeNearEdge ?? 0.78; // start fading at 78% of the ball radius → 0 at the edge.

    // BIRD WAKE (visuals only): the bird stirs the air inside the sphere — bow-wave outward AHEAD + drag/swirl
    // slipstream BEHIND. Scales with bird speed (~26 m/s); peak ≈ gain·26·0.65. Balanced so BOW (push-aside,
    // ~15 m/s) and DRAG (trailing slipstream, ~13 m/s) LEAD and SWIRL (~12 m/s) is secondary — reads as a wake
    // that PARTS and TRAILS, not a pinwheel of pure spin (the "just spirals" fix). Drop swirlGain toward 0 for
    // push+trail only; raise it for more tumble; all three to 0 disables. windAt / flight physics are frozen.
    this.bowGain = p.bowGain ?? 0.45; // softened (was 0.9): the strong outward push carved a "split" void directly ahead of the bird
    this.wakeGain = p.wakeGain ?? 0.75;
    this.swirlGain = p.swirlGain ?? 0.7;
    // SLIPSTREAM: two wingtip vortices at ±wingSpan, Rankine core vortexCore; half the motes are born at the
    // tips (wingEmitFrac) to make the streams legible; ambientNearFloor attenuates terrain-wind at the bird so
    // the near sphere rides the bird's own wake (sticks) instead of blowing downwind.
    this.wingSpan = p.wingSpan ?? 10;
    this.vortexCore = p.vortexCore ?? 6;
    this.wingEmitFrac = p.wingEmitFrac ?? 0.5;
    this.wingJitter = p.wingJitter ?? 3;
    this.ambientNearFloor = p.ambientNearFloor ?? 1.0; // 1 = near motes ride the FULL global wind (immersion); the
                                                       // bird connection comes from the wake + warm touched-air trails,
                                                       // NOT from starving the motes of global wind. Lower for "stick".
    // TOUCHED AIR: the wind the bird's wake hits glows yellow→red and trails up to 2× longer, fading over heatTau.
    this.heatTau = p.heatTau ?? 1.5;
    this.heatRef = p.heatRef ?? 24; // wake speed (m/s) for FULL heat. HIGH so only the genuine wake stream warms
                                    // (low values heat the WHOLE ball at speed → everything red). Tune live.
    this.heatLenGain = p.heatLenGain ?? 1.0;
    this.foreStretch = p.foreStretch ?? 1.3; // mild forward reach (was 1.6): with FEWER motes, don't spread them thin

    this.px = new Float32Array(this.count);
    this.py = new Float32Array(this.count);
    this.pz = new Float32Array(this.count);
    this.pHome = new Float32Array(this.count);
    this.speedFrac = new Float32Array(this.count);
    this.age = new Float32Array(this.count);
    this.ptX = new Float32Array(this.segments + 1);
    this.ptY = new Float32Array(this.segments + 1);
    this.ptZ = new Float32Array(this.segments + 1);
    const denseSeg = this.segments * Wind.FAR_SUBDIV;
    this.sptX = new Float32Array(denseSeg + 1);
    this.sptY = new Float32Array(denseSeg + 1);
    this.sptZ = new Float32Array(denseSeg + 1);

    this.nx = new Float32Array(this.nearCount);
    this.ny = new Float32Array(this.nearCount);
    this.nz = new Float32Array(this.nearCount);
    this.nearAge = new Float32Array(this.nearCount);
    this.nearHeat = new Float32Array(this.nearCount); // TOUCHED-AIR: per-mote heat (0..1), warm tint + longer tail
    this.nptX = new Float32Array(this.nearSegments + 1);
    this.nptY = new Float32Array(this.nearSegments + 1);
    this.nptZ = new Float32Array(this.nearSegments + 1);

    // combined vertex buffer: FAR long-line tier + NEAR sphere tier + reserved WAKE-SHED span, drawn in one
    // pass (shared format). FAR uses the SUBDIVIDED segment count (each coarse segment → FAR_SUBDIV quads).
    // The wake-shed span is reserved (worst-case) NOW so the offset plumbing is verified early; it stays
    // zero-filled this phase (wakeShedLiveCount=0 → its draw is never issued — see draw()).
    this.farVertexCount = this.count * this.segments * Wind.FAR_SUBDIV * 6;
    this.nearVertexCount = this.nearCount * this.nearSegments * 6;
    this.wakeShedVertexCount = Wind.WAKE_SHED_RESERVE;
    this.vertexCount = this.farVertexCount + this.nearVertexCount + this.wakeShedVertexCount;
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
              { shaderLocation: 6, offset: 40, format: "float32" },   // heat: 0..1 touched-air (near=real, far=0)
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
      multisample: { count: sampleCount },
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
    // v14 VOLUME: give this mote a HOME clearance drawn uniformly from a vertical band centered on
    // `clearance` (±vSpread), with the band itself clamped into [minClear,maxClear] BEFORE sampling so
    // motes never pile against a clamp (that would just make a new sheet at the floor/ceiling).
    const loHome = Math.max(this.minClear, this.clearance - this.vSpread);
    const hiHome = Math.min(this.maxClear, this.clearance + this.vSpread);
    const home = loHome + Math.pow(Math.random(), this.homeBias) * (hiHome - loHome); // homeBias>1 clusters motes NEAR terrain (cool look) with a thin tail reaching up (altitude not a dead void)
    this.pHome[i] = home;
    // seed height AT the mote's home; advection then pours it up/over the ridges, relax returns it home.
    this.py[i] = this.sampleHeight(x, z) + home;
    this.age[i] = 0; // v12 FADE: fresh seed → age 0 so the fade-IN envelope ramps this mote up from dark.
  }

  // Place / recycle near-comet i. HYBRID seeding: a wingEmitFrac fraction are BORN at the wingtips (the two
  // visible streams), the rest fill the ball UNIFORMLY (body air). Uniform branch keeps r = R·cbrt(rand) +
  // random direction so the body is evenly DENSE, not center-heavy. y is clamped above terrain so a comet
  // never seeds inside a ridge; flowAt + the bird wake then advect it. Wing emission needs the per-frame wing
  // frame (this._a*/_r*), so it only applies when moving (a stationary bird has no slipstream → uniform fill).
  private seedNearMote(i: number, birdPos: [number, number, number]): void {
    const R = this.nearRadius;
    let x: number, y0: number, z: number;
    if (this._wakeOn && Math.random() < this.wingEmitFrac) {
      // WINGTIP EMISSION: born near a wingtip (birdPos ± wingSpan·right), slightly AHEAD along the motion axis
      // so it immediately streams BACK through that tip's vortex → the visible "off the wing" cord.
      const side = Math.random() < 0.5 ? 1 : -1;
      const ax = this._ax, ay = this._ay, az = this._az;
      const rx = this._rx, ry = this._ry, rz = this._rz;
      const ux = ay * rz - az * ry, uy = az * rx - ax * rz, uz = ax * ry - ay * rx; // upW = axis × right (thin vertical jitter axis)
      const lead = R * (0.1 + 0.45 * Math.random());            // ahead of the bird so there is room to trail back
      const offR = side * this.wingSpan + (Math.random() * 2 - 1) * this.wingJitter;
      const offU = (Math.random() * 2 - 1) * this.wingJitter;
      x = birdPos[0] + ax * lead + rx * offR + ux * offU;
      y0 = birdPos[1] + ay * lead + ry * offR + uy * offU;
      z = birdPos[2] + az * lead + rz * offR + uz * offU;
    } else {
      // BODY: UNIFORM VOLUME inside the ball. The attenuated ambient (stepNear) + the twin vortices give this
      // body air its "stick + stir"; no separate bias needed.
      const r = R * Math.cbrt(Math.random());
      const ct = 2 * Math.random() - 1;          // cos(theta) uniform in [-1,1]
      const st = Math.sqrt(Math.max(0, 1 - ct * ct));
      const ph = 2 * Math.PI * Math.random();
      let ox = r * st * Math.cos(ph), oy = r * ct, oz = r * st * Math.sin(ph);
      if (this._moving) {
        // BIGGER BUBBLE FORE: stretch the forward (+axis) component so the body fills further AHEAD of the bird
        // (matches the forward-stretched recycle/fade metric) → motes populate the space in front, not just behind.
        const al = ox * this._ax + oy * this._ay + oz * this._az;
        if (al > 0) {
          const ext = al * (this.foreStretch - 1);
          ox += ext * this._ax; oy += ext * this._ay; oz += ext * this._az;
        }
      }
      x = birdPos[0] + ox;
      y0 = birdPos[1] + oy;
      z = birdPos[2] + oz;
    }
    const floor = this.sampleHeight(x, z) + this.minClear;
    this.nx[i] = x;
    this.ny[i] = y0 < floor ? floor : y0;
    this.nz[i] = z;
    this.nearAge[i] = 0; // v12 FADE: fresh seed → age 0 so the fade-IN envelope ramps this comet up.
    this.nearHeat[i] = 0; // TOUCHED-AIR: a freshly (re)seeded mote is untouched air → cool until the wake hits it.
  }

  // TERRAIN-SHAPED flow at world (x,z): the frozen horizontal windAt, plus a VERTICAL component and a
  // HORIZONTAL into-slope DEFLECTION derived from the terrain gradient (finite-diff of sampleHeight).
  // Returns [wx, wz, w] (m/s). The SAME function feeds both mote advection and the curved-tail integration
  // so the streaks trace exactly the flow that carries the motes. windAt itself is untouched (frozen).
  private flowAt(x: number, z: number, t: number): [number, number, number] {
    const [wx0, wz0] = windAt(x, z, t, this.cfg);
    // terrain gradient via central finite-diff: grad points UPHILL; magnitude ~ slope.
    const e = 6.0; // meters
    const gx = (this.sampleHeight(x + e, z) - this.sampleHeight(x - e, z)) / (2 * e); // dH/dx
    const gz = (this.sampleHeight(x, z + e) - this.sampleHeight(x, z - e)) / (2 * e); // dH/dz
    // VERTICAL: w = RAW (pre-deflection) horizontalWind · uphill-gradient, then CLAMPED to ±W_CLAMP.
    // The clamp is the anti-geyser keystone (v17): gradient ∝ RELIEF, so on steep faces wx·gx can spike
    // huge — unclamped + a real liftGain = the eruption the v16 commit killed by zeroing liftGain. Clamping
    // the OUTPUT (not the gain) lets gentle/moderate faces lift into a visible arc while steep faces cap out
    // at a believable updraft instead of launching motes to the ceiling.
    let w = this.liftGain * (wx0 * gx + wz0 * gz);
    if (w > Wind.W_CLAMP) w = Wind.W_CLAMP; else if (w < -Wind.W_CLAMP) w = -Wind.W_CLAMP;
    // HORIZONTAL: shed the into-slope component via the SHARED fn (identical math, now also used by the bird).
    const [wx, wz] = flowHorizontal(wx0, wz0, gx, gz, this.deflect);
    return [wx, wz, w];
  }

  // Advect every mote by windAt (p += w*dt), boundary-WRAP ones that leave the camera-relative wedge,
  // then rebuild the billboard vertex buffer from the PERSISTED positions. dt is derived from sim time
  // and clamped so the first frame and any tab-stall don't fling the motes.
  private step(
    camGround: [number, number],
    camFwd: [number, number],
    camRight: [number, number],
    t: number,
    birdPos: [number, number, number]
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

    let vi = 0;
    for (let i = 0; i < this.count; i++) {
      // FAR render-MODE branch (Phase 1 scaffold). Only "comet" (today's geometry) is implemented; the
      // divergent stipple/chevron geometries fall through to comet for now so the branch + farMode field
      // exist and are exercised.
      switch (this.farMode) {
        case "stipple":
        case "chevron":
          // Phase 2: divergent far geometry — for now emit the comet ribbon (identical output).
        case "comet":
        default:
          vi = this.emitFarComet(i, camGround, camFwd, camRight, t, birdPos, dt, vi);
          break; // end FAR "comet" mode emission (stipple/chevron fall through to here in Phase 1)
      }
    }
  }

  // FAR "comet" geometry emission for one mote (index i). Advects this.px/py/pz, then integrates a curved,
  // Catmull-Rom-smoothed tail and writes its quads into this.vertHost starting at vertex-float cursor `vi`,
  // RETURNING the advanced cursor. Extracted verbatim from step()'s mode switch (review follow-up): the body
  // is unchanged so the default farMode="comet" output is byte-identical. A degenerate (collapsed) ribbon is
  // emitted when vis≤0.001 so the draw count stays fixed; that early-out RETURNS the advanced cursor (was a
  // loop `continue`). The divergent stipple/chevron geometries will get their own emit methods in Phase 2.
  private emitFarComet(
    i: number,
    camGround: [number, number],
    camFwd: [number, number],
    camRight: [number, number],
    t: number,
    birdPos: [number, number, number],
    dt: number,
    vi: number
  ): number {
    const lo = this.speedLo;
    const hi = this.speedHi;
    const v = this.vertHost;
    const corners = Wind.CORNERS;
    const seg = this.segments;
    const denseSeg = seg * Wind.FAR_SUBDIV;          // rendered (subdivided) segment count per tail
    const ptX = this.ptX, ptY = this.ptY, ptZ = this.ptZ;
    const sptX = this.sptX, sptY = this.sptY, sptZ = this.sptZ; // dense Catmull-Rom polyline
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
      // ATMOSPHERE: scale the horizontal wind by the absolute-altitude profile so a mote skimming a HIGH ridge
      // rips while one in a LOW valley idles — the gradient reads through speed (and the speed-tint) even though
      // motes hug terrain. Scaled at the source so the tint + curling tail downstream use the same value.
      const [fwx0, fwz0, w] = this.flowAt(x0, z0, t);
      const prof = windProfile(y0);
      const wx = fwx0 * prof, wz = fwz0 * prof;
      const x = x0 + wx * dt;
      const z = z0 + wz * dt;
      // height advected by w, then a mild relaxation toward nominal clearance (anti-deplete) and clamps.
      const terr = this.sampleHeight(x, z);
      let y = y0 + w * dt;
      y += (terr + this.pHome[i]! - y) * Math.min(1, this.relax * dt);
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
      let vis = 1 - smoothstep(cutoff - 0.1, cutoff + 0.02, rank);

      // TIER CROSSFADE (no gap): inside the near-ball blend zone, RAISE the far density floor so far motes
      // stay dense through the region the near comets occupy (no calm-air thinning right at the bird).
      // Proximity-SCALED (not a flat clamp) so the floor ramps in smoothly — no hard ring at the zone edge.
      const bdx = x - birdPos[0], bdz = z - birdPos[2];
      const bdist = Math.hypot(bdx, bdz);
      const blendR = this.nearRadius * 1.6;            // blend zone reaches 1.6× the ball radius
      if (bdist < blendR) {
        const proximity = 1 - smoothstep(this.nearRadius, blendR, bdist); // 1 at/inside ball → 0 at blendR
        vis = Math.max(vis, 0.85 * proximity);
      }

      // v12 FADE envelope (no-pop): advance this mote's AGE every frame (BEFORE any cull-skip so a culled
      // mote that later survives the cull still has a real age — otherwise it would pop). Multiply a
      // fadeIn(age)·fadeOut(edge) envelope into the density vis so the mote eases in on (re)seed and eases
      // out before it recycles. Recompute the exit distances from the NEW (advected) position so the
      // fade-out tracks the actual recycle boundary. fadeIn keeps the existing density vis untouched once
      // age > fadeInTime; fadeOut→0 only in the last fadeFarEdge meters before a span exit.
      this.age[i]! += dt;
      const fadeIn = smoothstep(0, this.fadeInTime, this.age[i]!);
      const nrx = x - camGround[0], nrz = z - camGround[1];
      const nFwd = nrx * camFwd[0] + nrz * camFwd[1];
      const nSide = nrx * camRight[0] + nrz * camRight[1];
      // distance to the NEAREST span boundary (front / back / either side); fade over the last fadeFarEdge m.
      const distFront = this.spanAhead - nFwd;
      const distBack = nFwd + this.spanBehind;
      const distSide = this.spanWide - Math.abs(nSide);
      const edgeDist = Math.min(distFront, distBack, distSide);
      const fadeOut = smoothstep(0, this.fadeFarEdge, edgeDist);
      // far→near HANDOFF: fade the far mote DOWN as it enters the near ball so far+near don't STACK into an
      // over-bright core — the near tier owns the ball interior, far owns outside, smooth handoff across the
      // shell. Mirrors the near tier's edge fade so exactly ONE tier carries each region (no pop, no double).
      const bdy = y - birdPos[1];
      const d3 = this.bubbleFrac(bdx, bdy, bdz); // 0 center → 1 at the (forward-stretched) near-bubble edge (matches near tier)
      const nearHandoff = smoothstep(0.35, 1.0, d3); // far full outside the bubble (d3≥1) → ~0 by 35% in
      vis *= fadeIn * fadeOut * nearHandoff;
      if (vis <= 0.001) {
        // emit degenerate (collapsed) verts for every RENDERED segment so the draw count stays fixed.
        for (let s = 0; s < denseSeg; s++) {
          for (let c = 0; c < 6; c++) {
            v[vi++] = x; v[vi++] = y; v[vi++] = z;
            v[vi++] = 0; v[vi++] = 0;
            v[vi++] = sp;
            v[vi++] = 0; v[vi++] = 0;
            v[vi++] = 1; v[vi++] = 0; // along=1 (tail), vis=0
            v[vi++] = 0;              // heat=0 (far air is always cool)
          }
        }
        return vi; // degenerate ribbon only (was a loop `continue`); cursor advanced past the collapsed quads
      }

      // CURVED TAIL: integrate BACKWARD along flowAt from the head, building a polyline of seg+1 points.
      // tail length scales with speed (calm = short stub, fast = long arc). The path bends as flowAt changes
      // over the terrain → the comet arcs over the ridges. v13 PERF: flowAt (4 sampleHeight each) was called
      // PER segment — the dominant CPU cost. Cut it two ways WITHOUT losing the curve: (1) the FIRST step
      // reuses the head flow [wx,wz,w] already computed for advection above (free; the head IS where flowAt
      // was just sampled). (2) thereafter re-evaluate flowAt only every 2nd segment and HOLD it between —
      // halves the gradient cost; over a ~6-8m sub-step the flow barely changes so the arc still reads
      // curved (flow is still re-sampled along the path, not reused for the whole tail like the near comet).
      // The per-POINT terrain clamp is KEPT (the 35-50m far tail crosses ridges and must not sink in).
      const stepLen = this.segStep * (this.tailFloor + (1 - this.tailFloor) * sp);
      ptX[0] = x; ptY[0] = y; ptZ[0] = z;
      let cx = x, cy = y, cz = z;
      let fwx = wx, fwz = wz, fw = w; // step 1 reuses the head flow (sampled at the head above)
      for (let s = 1; s <= seg; s++) {
        // re-sample flow every 2nd segment (steps 1,3,5…); hold the previous sample on the in-between steps.
        if (s > 1 && (s & 1) === 1) { const f = this.flowAt(cx, cz, t); fwx = f[0] * prof; fwz = f[1] * prof; fw = f[2]; } // re-sampled flow profiled to match the head (atmosphere)
        cx -= fwx * stepLen;
        cz -= fwz * stepLen;
        cy -= fw * stepLen;
        // keep the tail above terrain so it doesn't sink into the ridge behind a crest.
        const tb = this.sampleHeight(cx, cz) + this.minClear;
        if (cy < tb) cy = tb;
        ptX[s] = cx; ptY[s] = cy; ptZ[s] = cz;
      }

      // SMOOTH: Catmull-Rom resample the coarse seg+1 points into a dense denseSeg+1 polyline so the
      // ribbon reads as a CURVE, not a few straight pieces with corners at the joints. Interpolating →
      // passes exactly through every original (terrain-clamped) point; endpoints are clamped (no P[-1]/
      // P[seg+1] read). No extra flowAt/sampleHeight — pure interpolation arithmetic.
      for (let i = 0; i < seg; i++) {
        const i0 = i > 0 ? i - 1 : 0;
        const i2 = i + 1;
        const i3 = i + 2 <= seg ? i + 2 : seg;
        for (let j = 0; j < Wind.FAR_SUBDIV; j++) {
          const tt = j / Wind.FAR_SUBDIV;
          const di = i * Wind.FAR_SUBDIV + j;
          sptX[di] = catmullRom(ptX[i0]!, ptX[i]!, ptX[i2]!, ptX[i3]!, tt);
          sptY[di] = catmullRom(ptY[i0]!, ptY[i]!, ptY[i2]!, ptY[i3]!, tt);
          sptZ[di] = catmullRom(ptZ[i0]!, ptZ[i]!, ptZ[i2]!, ptZ[i3]!, tt);
          // ANTI-PENETRATION (v17c): the Catmull-Rom can UNDERSHOOT below the chord of the (terrain-clamped)
          // coarse points → the curved tail dips INTO the terrain (worse over the deep RELIEF-600 valleys —
          // the user's "wind goes directly into terrain"). Clamp each dense point to ≥ the straight line
          // between its bracketing coarse points (both already ≥ terrain+minClear). Overshoot ABOVE the chord
          // (the up-and-over arc) is untouched; no new sampleHeight (reuses the coarse ptY).
          const chordY = ptY[i]! * (1 - tt) + ptY[i2]! * tt;
          if (sptY[di]! < chordY) sptY[di] = chordY;
        }
      }
      sptX[denseSeg] = ptX[seg]!; sptY[denseSeg] = ptY[seg]!; sptZ[denseSeg] = ptZ[seg]!;

      // emit a quad per RENDERED segment between consecutive dense points. corner.x picks near(0)/far(1).
      for (let s = 0; s < denseSeg; s++) {
        const ax = sptX[s]!, ay = sptY[s]!, az = sptZ[s]!;       // near (toward head)
        const bx = sptX[s + 1]!, by = sptY[s + 1]!, bz = sptZ[s + 1]!; // far (toward tail)
        // segment direction in world XZ (for screen-space perp orientation in the VS).
        let sdx = bx - ax, sdz = bz - az;
        const sdl = Math.hypot(sdx, sdz);
        if (sdl > 1e-5) { sdx /= sdl; sdz /= sdl; } else { sdx = 1; sdz = 0; }
        const alongN = s / denseSeg;       // head=0 → tail=1 across the whole ribbon
        const alongF = (s + 1) / denseSeg;
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
          v[vi++] = 0; // heat=0 (far air is always cool)
        }
      }
      return vi;
  }

  // Bird-wake disturbance velocity (visuals only) at world point (px,py,pz): a bow-wave OUTWARD ahead of the
  // bird (parts at the nose), an axial DRAG slipstream behind, and TWO counter-rotating WINGTIP VORTICES that
  // trail off the wing cores at birdPos ± wingSpan·right — the visible streams "off the wings". Relative to the
  // bird at birdPos moving along unit axis at speed bs; the wing-right vector is the per-frame this._r* set in
  // stepNear. Falls off to 0 at the ball edge. Writes [x,y,z] into `out` (NO allocation — called per head AND
  // per tail point so the near comets CURL along the wake). windAt / flight physics are untouched (render-only).
  private birdWakeAt(
    px: number, py: number, pz: number,
    birdPos: [number, number, number],
    axisX: number, axisY: number, axisZ: number,
    bs: number,
    out: [number, number, number]
  ): void {
    out[0] = 0; out[1] = 0; out[2] = 0;
    const R = this.nearRadius;
    const dx = px - birdPos[0], dy = py - birdPos[1], dz = pz - birdPos[2];
    const rr = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-3;
    const fall = 1 - rr / R;                                 // 1 at the bird → 0 at the ball edge
    if (fall <= 0) return;
    const along = dx * axisX + dy * axisY + dz * axisZ;      // signed distance along the motion axis
    let rx = dx - along * axisX, ry = dy - along * axisY, rz = dz - along * axisZ; // radial-from-axis (bow dir)
    const rho = Math.sqrt(rx * rx + ry * ry + rz * rz) || 1e-3;
    rx /= rho; ry /= rho; rz /= rho;                         // unit outward (bow-wave push direction)
    const ahead = Math.min(1, Math.max(0, along) / (0.35 * R));  // saturating ahead/behind weights (full
    const behind = Math.min(1, Math.max(0, -along) / (0.35 * R)); // strength within 35% of R along the axis)
    const push = this.bowGain * bs * fall * ahead;          // bow wave: outward, ahead of the bird
    const drag = this.wakeGain * bs * fall * behind;        // slipstream: along the motion axis, behind
    out[0] = rx * push + axisX * drag;
    out[1] = ry * push + axisY * drag;
    out[2] = rz * push + axisZ * drag;
    // TWIN counter-rotating wingtip vortices (replaces the single central swirl). Each core line runs through
    // birdPos ± wingSpan·right parallel to the motion axis; a Rankine-style falloff peaks near vortexCore and
    // the circulation sign flips per side (−c) → the pair counter-rotates with downwash between the tips.
    // Bounded by the ball-edge `fall` and the `behind` weight so the corkscrews TRAIL the wings.
    const spinBase = this.swirlGain * bs * fall * behind;
    if (spinBase > 0) {
      const rgx = this._rx, rgy = this._ry, rgz = this._rz;
      const hs = this.wingSpan, rc = this.vortexCore;
      for (let c = -1; c <= 1; c += 2) {
        const cdx = dx - c * hs * rgx, cdy = dy - c * hs * rgy, cdz = dz - c * hs * rgz; // P − corePos
        const calong = cdx * axisX + cdy * axisY + cdz * axisZ;
        let crx = cdx - calong * axisX, cry = cdy - calong * axisY, crz = cdz - calong * axisZ; // radial-from-core
        const crho = Math.sqrt(crx * crx + cry * cry + crz * crz) || 1e-3;
        crx /= crho; cry /= crho; crz /= crho;
        const tcx = axisY * crz - axisZ * cry, tcy = axisZ * crx - axisX * crz, tcz = axisX * cry - axisY * crx; // tangential = axis × r̂
        const coreFall = (crho * rc) / (crho * crho + rc * rc); // Rankine: 0 at the core, peak 0.5 at crho=rc, ~0 far
        const s = spinBase * coreFall * -c;                      // −c → the two cores counter-rotate
        out[0] += tcx * s; out[1] += tcy * s; out[2] += tcz * s;
      }
    }
  }

  // DEBUG/TEST probe: the bird-wake disturbance velocity at a world point using the LAST frame's bird state
  // (set by stepNear). Lets a live gate assert the TWIN vortices counter-rotate. Returns a fresh [x,y,z].
  sampleWake(px: number, py: number, pz: number): [number, number, number] {
    const out: [number, number, number] = [0, 0, 0];
    if (this._moving) this.birdWakeAt(px, py, pz, this._lastBirdPos, this._ax, this._ay, this._az, this._bs, out);
    return out;
  }

  // DEBUG/TEST: the current near-wake frame (bird pos + motion axis + wing-right unit + speed + moving flag).
  nearFrame(): { pos: number[]; axis: number[]; right: number[]; bs: number; moving: boolean } {
    return {
      pos: [this._lastBirdPos[0], this._lastBirdPos[1], this._lastBirdPos[2]],
      axis: [this._ax, this._ay, this._az],
      right: [this._rx, this._ry, this._rz],
      bs: this._bs,
      moving: this._moving,
    };
  }

  // Ellipsoidal "bubble" fraction for a near offset (dx,dy,dz from the bird): 0 at the bird → 1 at the bubble
  // edge. The bubble is a sphere of nearRadius EXCEPT stretched foreStretch× FORWARD (along the motion axis) so
  // the near field reaches further AHEAD — the bird flies INTO visible motes. Used by recycle, the fade envelope,
  // and the far→near handoff so all three agree on where the (asymmetric) bubble ends. Spherical when not moving.
  private bubbleFrac(dx: number, dy: number, dz: number): number {
    const R = this.nearRadius;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (!this._moving) return Math.sqrt(d2) / R;
    const along = dx * this._ax + dy * this._ay + dz * this._az;
    const perpSq = Math.max(0, d2 - along * along);
    const aR = along > 0 ? R * this.foreStretch : R; // forward (ahead) reach stretched; lateral/behind = R
    return Math.sqrt((along * along) / (aR * aR) + perpSq / (R * R));
  }

  // RENDER toggles: showNear = the LOCAL SPHERE (near bubble) draws; showWake = the wake disturbance (bow/drag/
  // twin vortices/touched-air + wing emission) is applied inside it. Both default OFF — global wind solved first.
  setShowNear(v: boolean): void { this.showNear = v; }
  setShowWake(v: boolean): void { this.showWake = v; }

  // Per-tier render MODE selectors (Phase 1 scaffold). Switches which geometry each tier emits into its
  // buffer span. Defaults "comet"/"comet"/"modulate" = today's look; the B/C modes currently fall through
  // to the A emission (the branch + field exist so the later divergent-geometry phase drops straight in).
  setFarMode(m: FarMode): void { this.farMode = m; }
  setNearMode(m: NearMode): void { this.nearMode = m; }
  setWakeMode(m: WakeMode): void { this.wakeMode = m; }

  // v11 NEAR SPHERE: advect the dense little comets around the bird, recycle any that leave the ball back
  // inside, and emit their short CURLING tails into the SAME host vertex array (after the far tier's verts).
  // Same terrain-shaped flowAt → coherent with the shared flow + respects terrain. vis is forced to 1 (NO
  // density cull) so the sphere renders DENSE and unmistakably legible right where the bird is.
  private stepNear(
    birdPos: [number, number, number],
    birdVel: [number, number, number],
    t: number,
    dt: number
  ): void {
    // BIRD WAKE setup (per-frame): unit motion axis + speed + WING FRAME. Computed BEFORE the first seed so the
    // wingtip emission (seedNearMote) has the frame on the very first fill. Below a small threshold the bird is
    // ~stationary → relative air is calm, no stir/slipstream. birdWakeAt() builds bow/drag + the TWIN wingtip
    // vortices on this axis; the wing-right vector offsets the two vortex cores.
    const bvx = birdVel[0], bvy = birdVel[1], bvz = birdVel[2];
    const bs = Math.hypot(bvx, bvy, bvz);
    const moving = bs > 0.5;
    const axisX = moving ? bvx / bs : 0, axisY = moving ? bvy / bs : 0, axisZ = moving ? bvz / bs : 0;
    // wing-right = normalize(axis × worldUp); worldUp=(0,1,0) → (−az, 0, ax). Near-vertical axis → fallback world X.
    let rgx = -axisZ, rgz = axisX;
    const rgl = Math.hypot(rgx, rgz);
    if (rgl > 1e-3) { rgx /= rgl; rgz /= rgl; } else { rgx = 1; rgz = 0; }
    this._ax = axisX; this._ay = axisY; this._az = axisZ;
    this._rx = rgx; this._ry = 0; this._rz = rgz;
    this._bs = bs; this._moving = moving; this._wakeOn = moving && this.showWake;
    this._lastBirdPos[0] = birdPos[0]; this._lastBirdPos[1] = birdPos[1]; this._lastBirdPos[2] = birdPos[2];

    // seed once we have a real bird position (first frame): fill the whole ball so it is immediately dense.
    if (!this.nearSeeded) {
      for (let i = 0; i < this.nearCount; i++) this.seedNearMote(i, birdPos);
      this.nearSeeded = true;
    }

    // write after the far tier's verts.
    let vi = this.farVertexCount * Wind.FPV;

    for (let i = 0; i < this.nearCount; i++) {
      // NEAR render-MODE branch (Phase 1 scaffold). Only "comet" (today's geometry) is implemented; the
      // divergent flecks/filaments geometries fall through to comet for now so the branch + nearMode field
      // exist and are exercised.
      switch (this.nearMode) {
        case "flecks":
        case "filaments":
          // Phase 2: divergent near geometry — for now emit the comet ribbon (identical output).
        case "comet":
        default:
          vi = this.emitNearComet(i, birdPos, t, dt, axisX, axisY, axisZ, bs, moving, vi);
          break; // end NEAR "comet" mode emission (flecks/filaments fall through to here in Phase 1)
      }
    }
  }

  // NEAR "comet" geometry emission for one mote (index i). Advects this.nx/ny/nz by the terrain-shaped flow
  // PLUS the bird wake, integrates a curling tail, and writes its quads into this.vertHost starting at the
  // vertex-float cursor `vi`, RETURNING the advanced cursor. Extracted verbatim from stepNear()'s mode switch
  // (review follow-up): the body is unchanged so the default nearMode="comet" output is byte-identical. The
  // per-frame wake frame (axis/speed/moving + this._wake*/_wakeOn) is computed by the caller; passed in here.
  private emitNearComet(
    i: number,
    birdPos: [number, number, number],
    t: number,
    dt: number,
    axisX: number,
    axisY: number,
    axisZ: number,
    bs: number,
    moving: boolean,
    vi: number
  ): number {
    const lo = this.speedLo, hi = this.speedHi;
    const v = this.vertHost;
    const corners = Wind.CORNERS;
    const seg = this.nearSegments;
    const ptX = this.nptX, ptY = this.nptY, ptZ = this.nptZ;
      let x0 = this.nx[i]!, y0 = this.ny[i]!, z0 = this.nz[i]!;

      // recycle: if the comet has drifted outside the bird-centered ball, reseed it back inside (the
      // sphere follows the bird as it moves). 3D distance test against the radius.
      const dxb = x0 - birdPos[0], dyb = y0 - birdPos[1], dzb = z0 - birdPos[2];
      if (this.bubbleFrac(dxb, dyb, dzb) > 1) { // outside the (forward-stretched) bubble → recycle back inside
        this.seedNearMote(i, birdPos);
        x0 = this.nx[i]!; y0 = this.ny[i]!; z0 = this.nz[i]!;
      }

      // advect by the terrain-shaped flow (horizontal deflected + vertical pour) → respects terrain, PLUS the
      // BIRD WAKE disturbance (visuals only): bow-wave outward ahead, drag + swirl slipstream behind. NOT
      // added to windAt (flight physics frozen) — it lives only in this near-mote advection + the curling tail.
      const [fwx0, fwz0, w] = this.flowAt(x0, z0, t);
      const prof = windProfile(y0); // ATMOSPHERE: near motes ride the SAME altitude-scaled global wind as the bird
      const wx = fwx0 * prof, wz = fwz0 * prof;
      let ibx = 0, iby = 0, ibz = 0;
      if (this._wakeOn) {
        this.birdWakeAt(x0, y0, z0, birdPos, axisX, axisY, axisZ, bs, this._wake);
        ibx = this._wake[0]; iby = this._wake[1]; ibz = this._wake[2];
      }
      // TOUCHED-AIR HEAT: how hard the bird's wake is hitting THIS mote, with ~heatTau memory. gain = |wake|/heatRef
      // (0..1); heat = max(decayed previous, gain) so a mote the bird physically touched stays warm then fades over
      // ~heatTau s. Drives the warm yellow→red tint AND a longer tail below. Reset to 0 on reseed (untouched air).
      const wakeMag = Math.sqrt(ibx * ibx + iby * iby + ibz * ibz);
      const gain = Math.min(1, wakeMag / this.heatRef);
      const heat = Math.max(this.nearHeat[i]! * Math.exp(-dt / this.heatTau), gain);
      this.nearHeat[i] = heat;
      // GLOBAL-WIND immersion: ambientNearFloor=1 (default) → near motes ride the FULL global wind, drifting with the
      // field like the far tier so the air reads as real moving air; the bird's wake is ADDED on top (the connection).
      // Lower ambientNearFloor (<1) to attenuate global wind near the bird so the near sphere STICKS to it instead.
      const adx = x0 - birdPos[0], ady = y0 - birdPos[1], adz = z0 - birdPos[2];
      const rrFrac = Math.min(1, Math.sqrt(adx * adx + ady * ady + adz * adz) / this.nearRadius);
      const ambientW = moving ? this.ambientNearFloor + (1 - this.ambientNearFloor) * rrFrac : 1;
      const fwx = wx * ambientW + ibx, fwy = w * ambientW + iby, fwz = wz * ambientW + ibz; // global wind (immersion) + bird wake (connection)
      const x = x0 + fwx * dt;
      const z = z0 + fwz * dt;
      const terr = this.sampleHeight(x, z);
      let y = y0 + fwy * dt;
      const loY = terr + this.minClear;        // stay above ground
      if (y < loY) y = loY;
      this.nx[i] = x; this.ny[i] = y; this.nz[i] = z;

      // v12 FADE envelope (no-pop): advance AGE, then fadeIn(age)·fadeOut(ballEdge). fadeIn ramps a freshly
      // (re)seeded comet up from dark over fadeInTime; fadeOut dims a comet as it nears the ball EDGE so it
      // is already faint when it recycles. Edge distance from the NEW (advected) position vs the bird. This
      // REPLACES the old hard vis=1 — so the sphere now shows a soft brightness gradient (dim toward the
      // edge / when young) instead of hard-edged uniform dots that would pop on recycle.
      this.nearAge[i]! += dt;
      const fadeIn = smoothstep(0, this.fadeInTime, this.nearAge[i]!);
      const ndx = x - birdPos[0], ndy = y - birdPos[1], ndz = z - birdPos[2];
      const distFrac = this.bubbleFrac(ndx, ndy, ndz); // 0 center → 1 at the (forward-stretched) bubble edge
      const fadeOut = 1 - smoothstep(this.fadeNearEdge, 1, distFrac); // 1 inside → 0 at the ball edge
      const nearVis = fadeIn * fadeOut;

      // speed tint (same calibration as the far tier) — little comets brighten in fast air; using the
      // DISTURBED horizontal speed so the bird's stir lights up the air it churns around itself.
      const wspeed = Math.hypot(fwx, fwz);
      const u = Math.min(1, Math.max(0, (wspeed - lo) / (hi - lo)));
      let sp = u * u * (3 - 2 * u);
      const climbFrac = Math.min(1, Math.max(0, w / 7.0));
      sp = Math.max(sp, climbFrac);

      // CURLING tail: integrate BACKWARD along the LOCAL disturbed flow (wind + bird wake), re-sampling at
      // each point so the tail CURVES with the swirl/slipstream instead of being a straight streak. Costs a
      // flowAt + birdWakeAt per segment — affordable now that the near sphere is small. Per-point terrain
      // clamp since a curling tail can swing toward a ridge.
      const stepLen = this.nearSegStep * (1 + this.heatLenGain * heat); // TOUCHED air trails up to (1+heatLenGain)× longer
      ptX[0] = x; ptY[0] = y; ptZ[0] = z;
      let cx = x, cy = y, cz = z;
      let tfx = fwx, tfy = fwy, tfz = fwz; // first backward step reuses the head's disturbed flow (free)
      for (let s = 1; s <= seg; s++) {
        cx -= tfx * stepLen;
        cy -= tfy * stepLen;
        cz -= tfz * stepLen;
        const tb = this.sampleHeight(cx, cz) + this.minClear; // keep the tail above the ridge
        if (cy < tb) cy = tb;
        ptX[s] = cx; ptY[s] = cy; ptZ[s] = cz;
        // re-sample the disturbed flow at the NEW point for the NEXT step → the tail curls along the wake.
        if (s < seg) {
          const [nwx, nwz, nw] = this.flowAt(cx, cz, t);
          if (this._wakeOn) {
            this.birdWakeAt(cx, cy, cz, birdPos, axisX, axisY, axisZ, bs, this._wake);
            tfx = nwx * prof * ambientW + this._wake[0]; tfy = nw * ambientW + this._wake[1]; tfz = nwz * prof * ambientW + this._wake[2];
          } else {
            tfx = nwx * prof; tfy = nw; tfz = nwz * prof;
          }
        }
      }

      // emit a quad per segment; vis forced to 1 → DENSE (no cull). along = head→tail fade over the ribbon.
      for (let s = 0; s < seg; s++) {
        const ax = ptX[s]!, ay = ptY[s]!, az = ptZ[s]!;
        const bx = ptX[s + 1]!, by = ptY[s + 1]!, bz = ptZ[s + 1]!;
        let sdx = bx - ax, sdz = bz - az;
        const sdl = Math.hypot(sdx, sdz);
        if (sdl > 1e-5) { sdx /= sdl; sdz /= sdl; } else { sdx = 1; sdz = 0; }
        const alongN = s / seg;
        const alongF = (s + 1) / seg;
        for (let c = 0; c < 6; c++) {
          const [pick, perp] = corners[c]!;
          const ex = pick > 0.5 ? bx : ax;
          const ey = pick > 0.5 ? by : ay;
          const ez = pick > 0.5 ? bz : az;
          const al = pick > 0.5 ? alongF : alongN;
          v[vi++] = ex; v[vi++] = ey; v[vi++] = ez;
          v[vi++] = pick; v[vi++] = perp;
          v[vi++] = sp;
          v[vi++] = sdx; v[vi++] = sdz;
          v[vi++] = al; v[vi++] = nearVis; // v12 FADE: vis = fadeIn·fadeOut (no hard pop; soft ball edge)
          v[vi++] = heat;                  // TOUCHED-AIR heat (loc 6) → warm tint in fs
        }
      }
      return vi;
  }

  // SECOND pass: LOAD terrain color+depth (no clear); draw the drifting dot motes over the ridges.
  // Takes the bird POS (near-sphere center) and VEL (orients the bird-wake stir inside that sphere).
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
    aspect: number,
    birdPos: [number, number, number],
    birdVel: [number, number, number]
  ): void {
    // derive dt once (shared by both tiers) so the near sphere advances in lock-step with the far field.
    let dt = this.lastTime < 0 ? 0 : time - this.lastTime;
    if (dt < 0) dt = 0;
    if (dt > 0.05) dt = 0.05;
    this.step(camGround, camFwd, camRight, time, birdPos);
    if (this.showNear) this.stepNear(birdPos, birdVel, time, dt); // LOCAL SPHERE off → far-tier (global wind) only
    // single upload of the combined (far + near) vertex buffer.
    this.device.queue.writeBuffer(this.vbuf, 0, this.vertBytes);

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
    // Up to THREE draws into the ONE buffer, addressed by firstVertex offset (the spans are NOT a simple
    // prefix — near can be off while a future wake-shed draw is on, so each tier is its own draw call). Far
    // and near rely on shader vis-culling (degenerate quads) so they always draw their FULL reserved count.
    pass.draw(this.farVertexCount, 1, 0);                                       // FAR (always)
    if (this.showNear) pass.draw(this.nearVertexCount, 1, this.farVertexCount); // NEAR sphere (when on)
    // WAKE-SHED: dedicated shed geometry (helix/rings). wakeShedLiveCount is 0 this phase → this draw is
    // never issued yet, but the buffer span is reserved and the code path exists (verified early).
    if (this.showWake && this.wakeMode !== "modulate" && this.wakeShedLiveCount > 0) {
      pass.draw(this.wakeShedLiveCount, 1, this.farVertexCount + this.nearVertexCount);
    }
    pass.end();
  }
}
