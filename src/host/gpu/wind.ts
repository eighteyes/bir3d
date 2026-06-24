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
  nearBodyCount?: number; // of nearCount, how many are the BALL body (always shown); the rest are the modulate wing slipstream (only when wake on)
  nearWakeCount?: number; // live count of WAKE (wingtip slipstream) motes shown when wake is on — drawn from the buffer ON TOP of the body
  wakeMoteLen?: number;   // tail-length multiplier for the wake (wing) motes vs the body comets (1 = same)
  wakeSpeedRef?: number;  // bird speed (m/s) at which the speed-scaled wake count reaches its cap (nearWakeCount)
  nearOpacity?: number;   // opacity multiplier for the LOCAL sphere + wake motes (1 = same as far; <1 = dimmer)
  nearRadius?: number;   // radius (m) of the sphere around the bird
  nearSegments?: number; // tail segment count for the LITTLE comets (short → distinct from far lines)
  nearSegStep?: number;  // seconds of flow integrated per near-comet tail segment (short tail)
  nearJitter?: number;   // per-mote random rotation (rad) of the near-comet flow direction so motes aren't uniform (0 = off)
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
  // --- FAR-B STIPPLE (disconnected dash streamline) — render-only, far tier ---
  dashCountK?: number;      // number of DISCONNECTED dashes along the over-ridge flow arc (2-4)
  dashLenM?: number;        // world-meters length of each dash along its LOCAL tangent
  gapRatio?: number;        // dash spacing along the arc: gap length as a multiple of dash length (>1 = sparser)
  lenByAltitude?: number;   // 0..1 — how strongly dash length scales with altitude (proxied by mote speedFrac/windProfile)
  leadBoost?: number;       // brightness multiplier on the HEAD (lead) dash
  // --- FAR-C CHEVRON (2-limb arrowhead glyph) — render-only, far tier ---
  spreadAngleDeg?: number;  // half-angle (deg) between the two chevron limbs around the flow direction
  limbLenM?: number;        // world-meters length of each chevron limb from apex to tip
  apexBoost?: number;       // brightness multiplier at the apex (the bright point of the arrow)
  rakeBySpeed?: number;     // 0..1 — how strongly speedFrac sharpens the spread + grows the limbs (faster = sharper dart)
  // --- NEAR-B SHEAR FLECKS (single oriented dash whose length/brightness reads local velocity shear) — render-only, near tier ---
  fleckLen?: number;        // base world-meters length of the dash (full 2-point span = ±0.5·len along the local velocity dir)
  shearGain?: number;       // how strongly local shear magnitude stretches the fleck: len = fleckLen·(1 + shearGain·shearMag)
  shearRadius?: number;     // world-meters half-span of the finite-diff used to measure shear across the flow (head ± shearRadius·dir)
  fleckTaper?: number;      // low `along` ramp ceiling so the dash reads as a FLAT tracer (not a head-bright mini-comet)
  orientLerp?: number;      // 0..1 slerp of the long axis toward the current velocity each frame (0 = snap to current dir)
  // --- NEAR-C CURL FILAMENTS (thin thread that corkscrews around the wingtip vortex cores) — render-only, near tier ---
  filSegStep?: number;      // seconds of flow integrated per filament tail segment (LONGER than nearSegStep → looser, readable spiral)
  // --- WAKE-B WINGTIP HELIX FILAMENTS (counter-rotating cords SHED off the two wingtips) — render-only, wake-shed span ---
  wakeEmitRate?: number;    // arcs SHED per wingtip per second (cadence of new shed elements)
  wakeLife?: number;        // seconds a shed helix element persists before retiring
  helixGain?: number;       // multiplies the tangential/swirl (twin-vortex) term FOR SHED GEOMETRY ONLY → tighter/looser corkscrew
  wakeSeg?: number;         // backward-integrated polyline segments per shed element (KEEP SHORT, ≤4 — anti-corner)
  wakeSegStep?: number;     // seconds of flow integrated per shed-element segment
  wakeTaper?: number;       // head→tail width/brightness taper bias for a shed element (0..1)
  counterRotate?: boolean;  // flip one tip's swirl sense so the two cords spiral in OPPOSITE senses
  // --- WAKE-C SHED PRESSURE RINGS (periodic expanding hoops shed off the wings) — render-only, wake-shed span ---
  ringRate?: number;        // rings SHED per second (cadence)
  ringGrow?: number;        // m/s the ring radius expands each second after spawn
  ringLife?: number;        // seconds a shed ring persists before retiring
  ringSegN?: number;        // chords tessellating each ring loop (12-32)
  ringStartRadius?: number; // ring radius (m) at spawn
  ringTilt?: number;        // tilt of the ring normal by local shear (0 = face-on to the flight axis)
  twinOffset?: number;      // >0 = one ring per wingtip (center at birdPos ± wingSpan·right); 0 = centerline train
  convectFrac?: number;     // fraction of bird speed the ring center convects backward along the axis each frame
  ringWarmBias?: number;    // warms the loaded (downstream) side of the loop (0..1)
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
  private nearBodyCount: number;
  private nearWakeCount: number;
  private wakeMoteLen: number;
  private wakeSpeedRef: number;
  private nearOpacity: number;
  private _wakeCountNow = 0; // per-frame speed-scaled active wake count (≤ nearWakeCount)
  private nearRadius: number;
  private nearSegments: number;
  private nearSegStep: number;
  private nearJitter: number;
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
  // FAR-B STIPPLE dials (see DotParams).
  private dashCountK: number;
  private dashLenM: number;
  private gapRatio: number;
  private lenByAltitude: number;
  private leadBoost: number;
  // FAR-C CHEVRON dials (see DotParams).
  private spreadAngleDeg: number;
  private limbLenM: number;
  private apexBoost: number;
  private rakeBySpeed: number;
  // NEAR-B SHEAR FLECKS dials (see DotParams).
  private fleckLen: number;
  private shearGain: number;
  private shearRadius: number;
  private fleckTaper: number;
  private orientLerp: number;
  // NEAR-C CURL FILAMENTS dials (see DotParams).
  private filSegStep: number;
  // WAKE-B WINGTIP HELIX dials (see DotParams).
  private wakeEmitRate: number;
  private wakeLife: number;
  private helixGain: number;
  private wakeSeg: number;
  private wakeSegStep: number;
  private wakeTaper: number;
  private counterRotate: boolean;
  // WAKE-C SHED RINGS dials (see DotParams).
  private ringRate: number;
  private ringGrow: number;
  private ringLife: number;
  private ringSegN: number;
  private ringStartRadius: number;
  private ringTilt: number;
  private twinOffset: number;
  private convectFrac: number;
  private ringWarmBias: number;
  // NEAR-B per-mote long-axis (XZ unit dir) persisted so orientLerp can slerp it toward velocity each frame.
  private fleckDirX: Float32Array;
  private fleckDirZ: Float32Array;
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
  private nearJit: Float32Array;  // per-near-mote random sign/amount [-1,1] for the direction jitter (set at seed)
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
  private wakeShedLiveCount = 0;       // wake-shed verts ACTUALLY written this frame (0 when wake off / modulate)

  // --- WAKE-B HELIX shed pool (persistent, allocated once). Each live element is a short tube-arc SHED from a
  //     wingtip: a seed point advected by flowAt + birdWakeAt, an age (retire at wakeLife), and which tip it
  //     came from (side ±1, drives the counter-rotating sense). Capacity HELIX_TIPS*HELIX_LIVE so the worst
  //     case (every slot live × wakeSeg quads) stays within WAKE_SHED_RESERVE. `helixActive` = live count. ---
  private helixSeedX!: Float32Array;
  private helixSeedY!: Float32Array;
  private helixSeedZ!: Float32Array;
  private helixAge!: Float32Array;
  private helixSide!: Float32Array; // +1 / −1: which wingtip this cord was shed from
  private helixActive = 0;          // number of live helix elements (≤ HELIX_TIPS*HELIX_LIVE)
  private helixEmitAcc = 0;         // fractional emission accumulator (arcs owed, both tips) across frames
  // --- WAKE-C RINGS shed pool (persistent, allocated once). Each live ring is a center, a radius (grows with
  //     ringGrow), an age (retire at ringLife), a side (±1 wingtip or 0 centerline), and a spawn-time heat
  //     sampled from |birdWakeAt(center)|. Capacity RING_COUNT so worst case (all rings × ringSegN chords)
  //     stays within reserve. `ringActive` = live count. ---
  private ringCx!: Float32Array;
  private ringCy!: Float32Array;
  private ringCz!: Float32Array;
  private ringRadius!: Float32Array;
  private ringAge!: Float32Array;
  private ringSide!: Float32Array;  // +1 / −1 / 0 wingtip-or-centerline marker (drives the warm-bias direction)
  private ringHeat!: Float32Array;  // spawn-time heat (0..1) sampled from |birdWakeAt(center)|
  private ringActive = 0;           // number of live rings (≤ RING_COUNT)
  private ringEmitAcc = 0;          // fractional ring-emission accumulator across frames
  private wakeShedPoolsInit = false;// lazy one-time pool allocation guard
  private wakeOverrunLogged = false;// log the cap-hit warning at most once
  // reusable scratch for a shed element's integrated backward polyline (wakeSeg+1 points) and per-frame wake.
  private wsPtX!: Float32Array;
  private wsPtY!: Float32Array;
  private wsPtZ!: Float32Array;
  private readonly _wsWake: [number, number, number] = [0, 0, 0];

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
    this.nearBodyCount = Math.min(p.nearBodyCount ?? 200, this.nearCount); // BALL body budget; wake adds wing motes ON TOP (never steals)
    this.nearWakeCount = p.nearWakeCount ?? 400; // WAKE motes CAP — the active count scales with bird speed up to this (≤ buffer)
    this.wakeMoteLen = p.wakeMoteLen ?? 1.0;     // wake-mote tail length vs the body comets (1 = same)
    this.wakeSpeedRef = p.wakeSpeedRef ?? 45;    // bird speed (m/s) at which the wake count reaches its cap
    this.nearOpacity = p.nearOpacity ?? 0.8;     // LOCAL sphere + wake opacity — 0.8 = 20% dimmer than the far tier
    this.nearRadius = p.nearRadius ?? 65;
    this.nearSegments = p.nearSegments ?? 4; // a 4th segment → smoother CURLING tail arcs
    this.nearSegStep = p.nearSegStep ?? 0.12;
    this.nearJitter = p.nearJitter ?? 0.12; // a tiny bit of per-mote direction variety so the sphere isn't lockstep

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
    this.foreStretch = p.foreStretch ?? 2.6; // 2× forward reach: the near bubble stretches well ahead so it OVERLAPS the
                                              // forward zone where the global-wind handoff fades out (no seam in front of the bird)

    // FAR-B STIPPLE: K disconnected dashes tracing the over-ridge flow arc (the comet's integrated polyline),
    // lead dash bright, trailing dashes dimmer. Defaults per spec.
    this.dashCountK = p.dashCountK ?? 3;
    this.dashLenM = p.dashLenM ?? 9;
    this.gapRatio = p.gapRatio ?? 1.5;
    this.lenByAltitude = p.lenByAltitude ?? 0.6;
    this.leadBoost = p.leadBoost ?? 1.5;
    // FAR-C CHEVRON: a 2-limb arrowhead glyph at the mote head, opening UPstream along the flow. Defaults per spec.
    this.spreadAngleDeg = p.spreadAngleDeg ?? 28;
    this.limbLenM = p.limbLenM ?? 14;
    this.apexBoost = p.apexBoost ?? 1.5;
    this.rakeBySpeed = p.rakeBySpeed ?? 0.5;

    // NEAR-B SHEAR FLECKS: a single short oriented dash per mote, length+brightness driven by local velocity
    // shear (finite-diff across ±shearRadius). NO backward integration. Defaults per spec.
    this.fleckLen = p.fleckLen ?? 3;
    this.shearGain = p.shearGain ?? 1.5;
    this.shearRadius = p.shearRadius ?? 3;
    this.fleckTaper = p.fleckTaper ?? 0.2;
    this.orientLerp = p.orientLerp ?? 0;
    // NEAR-C CURL FILAMENTS: backward integration (same as the comet) but a LONGER per-segment reach so the
    // wingtip-vortex corkscrew reads as a loose spiral. Default per spec.
    this.filSegStep = p.filSegStep ?? 0.25;

    // WAKE-B WINGTIP HELIX: short tube-arcs SHED from each wingtip, advected by flowAt + birdWakeAt and
    // retired at wakeLife; each frame each live element integrates a SHORT backward polyline through the same
    // disturbed flow (the twin-vortex tangential term makes it corkscrew). counterRotate flips one tip's
    // sense. Cap: emitRate·life·2 tips must stay ≤ HELIX_TIPS·HELIX_LIVE (320) — the defaults below give
    // 60·1.2·2 = 144 ≤ 320 (headroom for the cap, never overruns the reserve).
    this.wakeEmitRate = p.wakeEmitRate ?? 60;
    this.wakeLife = p.wakeLife ?? 1.2;
    this.helixGain = p.helixGain ?? 1.0;
    this.wakeSeg = Math.min(Wind.HELIX_SEGS, Math.max(1, Math.round(p.wakeSeg ?? 3))); // ≤4 (reserve sizing); KEEP SHORT
    this.wakeSegStep = p.wakeSegStep ?? 0.1;
    this.wakeTaper = p.wakeTaper ?? 0.7;
    this.counterRotate = p.counterRotate ?? true;
    // WAKE-C SHED RINGS: periodic expanding hoops shed face-on to the flight axis, center convected backward,
    // radius grows, retire at ringLife. ringRate·ringLife must stay ≤ RING_COUNT (32) and ringSegN ≤ RING_CHORDS
    // (32) so the worst case (all rings × chords × 6) stays within the reserve. Defaults: 6·1.5 = 9 ≤ 32.
    this.ringRate = p.ringRate ?? 6;
    this.ringGrow = p.ringGrow ?? 8;
    this.ringLife = p.ringLife ?? 1.5;
    this.ringSegN = Math.min(Wind.RING_CHORDS, Math.max(3, Math.round(p.ringSegN ?? 24))); // 12-32 (clamped ≤32)
    this.ringStartRadius = p.ringStartRadius ?? 2;
    this.ringTilt = p.ringTilt ?? 0.3;
    this.twinOffset = p.twinOffset ?? 10; // >0 = per-wingtip; 0 = centerline train
    this.convectFrac = p.convectFrac ?? 0.7;
    this.ringWarmBias = p.ringWarmBias ?? 0.5;

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
    this.nearJit = new Float32Array(this.nearCount);  // per-mote direction-jitter amount [-1,1], assigned at seed
    // NEAR-B: persisted per-mote long-axis (XZ unit dir) for the optional orientLerp slerp toward velocity.
    this.fleckDirX = new Float32Array(this.nearCount);
    this.fleckDirZ = new Float32Array(this.nearCount);
    this.nptX = new Float32Array(this.nearSegments + 1);
    this.nptY = new Float32Array(this.nearSegments + 1);
    this.nptZ = new Float32Array(this.nearSegments + 1);

    // WAKE-SHED pools (helix cords + shed rings) — allocated ONCE here, sized to the reserve caps so the
    // worst-case emission can never overrun the third buffer span. The arrays persist across frames (that
    // persistence is what makes the shed geometry trail and live); only the pool matching wakeMode is
    // advanced/emitted each frame. See stepWakeShed.
    const HELIX_CAP = Wind.HELIX_TIPS * Wind.HELIX_LIVE; // 320 live cords (×wakeSeg≤4 ×6 = 7680 = reserve)
    this.helixSeedX = new Float32Array(HELIX_CAP);
    this.helixSeedY = new Float32Array(HELIX_CAP);
    this.helixSeedZ = new Float32Array(HELIX_CAP);
    this.helixAge = new Float32Array(HELIX_CAP);
    this.helixSide = new Float32Array(HELIX_CAP);
    this.ringCx = new Float32Array(Wind.RING_COUNT);
    this.ringCy = new Float32Array(Wind.RING_COUNT);
    this.ringCz = new Float32Array(Wind.RING_COUNT);
    this.ringRadius = new Float32Array(Wind.RING_COUNT);
    this.ringAge = new Float32Array(Wind.RING_COUNT);
    this.ringSide = new Float32Array(Wind.RING_COUNT);
    this.ringHeat = new Float32Array(Wind.RING_COUNT);
    // scratch for a shed element's backward polyline (max segs+1 across both pools = HELIX_SEGS+1).
    this.wsPtX = new Float32Array(Wind.HELIX_SEGS + 1);
    this.wsPtY = new Float32Array(Wind.HELIX_SEGS + 1);
    this.wsPtZ = new Float32Array(Wind.HELIX_SEGS + 1);
    this.wakeShedPoolsInit = true;

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
    // WING SEEDING: in modulate, the dedicated wing-budget slots (i >= nearBodyCount) ARE the wingtip streams, so
    // the body slots [0, nearBodyCount) always stay in the ball — wake never thins the sphere, it ADDS streams.
    // Filaments keeps its random wingEmitFrac core-seeding (it wants most motes at the tips).
    const wingSeed = this._wakeOn && (
      (this.wakeMode === "modulate" && this.nearMode === "comet" && i >= this.nearBodyCount) ||
      (this.nearMode === "filaments" && Math.random() < this.wingEmitFrac)
    );
    if (wingSeed) {
      // WINGTIP EMISSION: born near a wingtip (birdPos ± wingSpan·right), slightly AHEAD along the motion axis
      // so it immediately streams BACK through that tip's vortex → the visible "off the wing" cord.
      const side = Math.random() < 0.5 ? 1 : -1;
      const ax = this._ax, ay = this._ay, az = this._az;
      const rx = this._rx, ry = this._ry, rz = this._rz;
      const ux = ay * rz - az * ry, uy = az * rx - ax * rz, uz = ax * ry - ay * rx; // upW = axis × right (thin vertical jitter axis)
      // EMIT off the wing, MODESTLY ahead: seed ahead along the motion axis so the motes stream BACK past the wingtip
      // as the bird flies INTO them → reads like air coming off the wing. Pulled HALFWAY back toward the wing from the
      // fully-forward version (user). Uniform spread keeps it a CONTINUOUS source (varied lead → staggered exits).
      const lead = R * (0.1 + 0.45 * Math.random());
      const offR = side * this.wingSpan * (0.55 + 0.45 * Math.random()) + (Math.random() * 2 - 1) * this.wingJitter;
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
    this.nearJit[i] = Math.random() * 2 - 1; // fresh per-mote direction-jitter sign/amount so the cloud isn't uniform
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
      // FAR render-MODE branch. Each mode emits into the SAME fixed per-mote stride (FAR_VERTS_PER_MOTE):
      // comet writes the full subdivided ribbon; stipple/chevron emit FEWER real quads and PAD the remainder
      // of the slot with degenerate (vis=0) verts, all returning vi advanced by exactly the comet stride.
      switch (this.farMode) {
        case "stipple":
          vi = this.emitFarStipple(i, camGround, camFwd, camRight, t, birdPos, dt, vi);
          break; // FAR-B: disconnected dash streamline
        case "chevron":
          vi = this.emitFarChevron(i, camGround, camFwd, camRight, t, birdPos, dt, vi);
          break; // FAR-C: 2-limb arrowhead glyph
        case "comet":
        default:
          vi = this.emitFarComet(i, camGround, camFwd, camRight, t, birdPos, dt, vi);
          break; // FAR-A: continuous subdivided comet ribbon
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

  // FIXED per-mote far-tier vertex stride. draw() always draws the FULL farVertexCount, and that span is
  // sized count * segments * FAR_SUBDIV * 6 — so EVERY far mote owns exactly this many vertices regardless
  // of which farMode emitted it. The divergent geometries (stipple/chevron) emit FEWER real quads, so they
  // PAD the remainder of this slot with degenerate (vis=0) verts and advance the cursor by exactly this many
  // so the next mote lands at the correct offset. Matches the comet's denseSeg*6 (= segments*FAR_SUBDIV*6).
  private get FAR_VERTS_PER_MOTE(): number {
    return this.segments * Wind.FAR_SUBDIV * 6;
  }

  // Emit ONE degenerate (collapsed, vis=0) far vertex at cursor `vi` and return the advanced cursor. All 11
  // floats are 0 — the VS collapses vis<0.001 to a point and the FS discards vis<=0.001, so the reserved
  // padding draws nothing. Used by the divergent far geometries to fill their fixed 48-vert slot after the
  // real quads, keeping the per-mote stride constant (see FAR_VERTS_PER_MOTE).
  private emitDegenerateFarVert(vi: number): number {
    const v = this.vertHost;
    v[vi++] = 0; v[vi++] = 0; v[vi++] = 0; // pos
    v[vi++] = 0; v[vi++] = 0;              // corner
    v[vi++] = 0;                           // speedFrac
    v[vi++] = 0; v[vi++] = 0;              // segDir
    v[vi++] = 0; v[vi++] = 0;              // along, vis (vis=0 → discarded)
    v[vi++] = 0;                           // heat
    return vi;
  }

  // FAR-B STIPPLE: the same over-ridge flow STREAMLINE the comet traces, shown as `dashCountK` DISCONNECTED
  // single-segment dashes with dark gaps instead of one continuous ribbon. Reuses emitFarComet's coarse
  // backward-flow integration (advection + recycle + speedFrac + vis + the integrated ptX/ptY/ptZ polyline),
  // then picks K points at arc-fractions and emits ONE short quad (a dash) at each, oriented along the LOCAL
  // tangent, length dashLenM. Lead dash bright (along≈0, ×leadBoost), trailing dashes ramp dimmer (along
  // 0→~0.8). gapRatio spaces the dashes along the arc; lenByAltitude scales dash length with the mote's
  // speedFrac (which already encodes windProfile/altitude). Real quads = K*6 verts; the slot is PADDED to
  // FAR_VERTS_PER_MOTE with degenerate verts and the cursor advanced by exactly that. heat=0 (far air is cool).
  private emitFarStipple(
    i: number,
    camGround: [number, number],
    camFwd: [number, number],
    camRight: [number, number],
    t: number,
    birdPos: [number, number, number],
    dt: number,
    vi: number
  ): number {
    const slotEnd = vi + this.FAR_VERTS_PER_MOTE * Wind.FPV; // exact end of this mote's fixed stride
    const v = this.vertHost;
    const corners = Wind.CORNERS;
    const seg = this.segments;
    const ptX = this.ptX, ptY = this.ptY, ptZ = this.ptZ;

    let x0 = this.px[i]!;
    let z0 = this.pz[i]!;
    let y0 = this.py[i]!;

    // Boundary-WRAP (identical to the comet so motes recycle the same way).
    const rx = x0 - camGround[0];
    const rz = z0 - camGround[1];
    const fwdDist = rx * camFwd[0] + rz * camFwd[1];
    const sideDist = rx * camRight[0] + rz * camRight[1];
    if (fwdDist < -this.spanBehind || fwdDist > this.spanAhead) {
      this.seedMote(i, camGround, camFwd, camRight, this.spanAhead * 0.6, 0);
      x0 = this.px[i]!; z0 = this.pz[i]!; y0 = this.py[i]!;
    } else if (Math.abs(sideDist) > this.spanWide) {
      this.seedMote(i, camGround, camFwd, camRight, -this.spanBehind, sideDist > 0 ? -1 : 1);
      x0 = this.px[i]!; z0 = this.pz[i]!; y0 = this.py[i]!;
    }

    // advect by the terrain-shaped, altitude-profiled flow (identical to the comet).
    const [fwx0, fwz0, w] = this.flowAt(x0, z0, t);
    const prof = windProfile(y0);
    const wx = fwx0 * prof, wz = fwz0 * prof;
    const x = x0 + wx * dt;
    const z = z0 + wz * dt;
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
    const u = Math.min(1, Math.max(0, (wspeed - this.speedLo) / (this.speedHi - this.speedLo)));
    let sp = u * u * (3 - 2 * u);
    const climbFrac = Math.min(1, Math.max(0, w / 7.0));
    sp = Math.max(sp, climbFrac);
    this.speedFrac[i] = sp;

    // DENSITY cull + tier crossfade + fade envelope (identical to the comet so stipple culls/fades the same).
    const rank = hashRank(i);
    const cutoff = this.densityFloor + (1 - this.densityFloor) * sp;
    let vis = 1 - smoothstep(cutoff - 0.1, cutoff + 0.02, rank);
    const bdx = x - birdPos[0], bdz = z - birdPos[2];
    const bdist = Math.hypot(bdx, bdz);
    const blendR = this.nearRadius * 1.6;
    if (bdist < blendR) {
      const proximity = 1 - smoothstep(this.nearRadius, blendR, bdist);
      vis = Math.max(vis, 0.85 * proximity);
    }
    this.age[i]! += dt;
    const fadeIn = smoothstep(0, this.fadeInTime, this.age[i]!);
    const nrx = x - camGround[0], nrz = z - camGround[1];
    const nFwd = nrx * camFwd[0] + nrz * camFwd[1];
    const nSide = nrx * camRight[0] + nrz * camRight[1];
    const distFront = this.spanAhead - nFwd;
    const distBack = nFwd + this.spanBehind;
    const distSide = this.spanWide - Math.abs(nSide);
    const edgeDist = Math.min(distFront, distBack, distSide);
    const fadeOut = smoothstep(0, this.fadeFarEdge, edgeDist);
    const bdy = y - birdPos[1];
    const d3 = this.bubbleFrac(bdx, bdy, bdz);
    const nearHandoff = smoothstep(0.35, 1.0, d3);
    vis *= fadeIn * fadeOut * nearHandoff;
    if (vis <= 0.001) {
      while (vi < slotEnd) vi = this.emitDegenerateFarVert(vi); // collapse the whole slot
      return vi;
    }

    // CURVED STREAMLINE points: integrate BACKWARD along flowAt from the head (same as the comet's coarse
    // tail), building ptX/ptY/ptZ[0..seg]. We trace the WHOLE potential arc (full segStep, not speed-scaled)
    // so the K dashes sample the same over-ridge curve regardless of speed; speed shows through vis + length.
    const stepLen = this.segStep;
    ptX[0] = x; ptY[0] = y; ptZ[0] = z;
    let cx = x, cy = y, cz = z;
    let fwx = wx, fwz = wz, fw = w; // step 1 reuses the head flow
    for (let s = 1; s <= seg; s++) {
      if (s > 1 && (s & 1) === 1) { const f = this.flowAt(cx, cz, t); fwx = f[0] * prof; fwz = f[1] * prof; fw = f[2]; }
      cx -= fwx * stepLen;
      cz -= fwz * stepLen;
      cy -= fw * stepLen;
      const tb = this.sampleHeight(cx, cz) + this.minClear;
      if (cy < tb) cy = tb;
      ptX[s] = cx; ptY[s] = cy; ptZ[s] = cz;
    }

    // DASHES: place K dashes along the arc, spaced by gapRatio (dash : gap). The arc-fraction of dash k's
    // CENTER is laid out so dash + gap tile [0,1]: pitch = (1+gapRatio)·unit, with K dashes the unit is
    // 1/(K + (K-1)·gapRatio) of the arc... but we want the LEAD dash at the head (frac 0). So center_k =
    // k·pitch where pitch = 1/max(1,K-1) when K>1 spaces them evenly head→~near-tail, and gapRatio biases
    // the inter-dash spacing implicitly via where we stop (≤ ~0.85 so the last dash isn't at the very tail).
    const K = Math.max(1, Math.min((this.FAR_VERTS_PER_MOTE / 6) | 0, Math.round(this.dashCountK))); // ≤ slot (no overrun)
    // dash length in world meters, scaled by altitude (proxied by speedFrac) via lenByAltitude:
    // full length when lenByAltitude=0; at =1 the length scales 0.4→1 with sp (low altitude/calm → shorter).
    const altScale = 1 - this.lenByAltitude * (1 - (0.4 + 0.6 * sp));
    const halfLen = 0.5 * this.dashLenM * altScale;
    // arc span the dashes occupy: more gap (gapRatio) packs the dashes toward the head (shorter occupied span).
    const span = Math.min(0.85, 0.85 / (1 + 0.25 * (this.gapRatio - 1)));
    for (let k = 0; k < K; k++) {
      const frac = K > 1 ? span * (k / (K - 1)) : 0; // 0 (head) → span (toward tail)
      // sample the streamline point + local tangent at this arc-fraction (linear interp over the coarse pts).
      const fs = frac * seg;
      const i0 = Math.min(seg - 1, Math.floor(fs));
      const ft = fs - i0;
      const i1 = i0 + 1;
      const ax = ptX[i0]! * (1 - ft) + ptX[i1]! * ft;
      const ay = ptY[i0]! * (1 - ft) + ptY[i1]! * ft;
      const az = ptZ[i0]! * (1 - ft) + ptZ[i1]! * ft;
      // local tangent = direction along the polyline at this point (XZ), normalized.
      let tx = ptX[i1]! - ptX[i0]!, tz = ptZ[i1]! - ptZ[i0]!;
      const ty = ptY[i1]! - ptY[i0]!;
      const tl = Math.hypot(tx, ty, tz);
      let utx: number, uty: number, utz: number;
      if (tl > 1e-5) { utx = tx / tl; uty = ty / tl; utz = tz / tl; } else { utx = 1; uty = 0; utz = 0; }
      // dash endpoints = point ± halfLen·tangent (world meters along the tangent).
      const p0x = ax - utx * halfLen, p0y = ay - uty * halfLen, p0z = az - utz * halfLen; // near (head side)
      const p1x = ax + utx * halfLen, p1y = ay + uty * halfLen, p1z = az + utz * halfLen; // far (tail side)
      // segDir for screen-space perp (XZ), normalized.
      let sdx = utx, sdz = utz;
      const sdl = Math.hypot(sdx, sdz);
      if (sdl > 1e-5) { sdx /= sdl; sdz /= sdl; } else { sdx = 1; sdz = 0; }
      // brightness ramp: lead dash brightest, trailing dashes dimmer. `along` darkens toward the tail
      // (shader lenFade = (1-along)^1.3), so trailing dashes ride a higher base `along` (0→~0.8 across K).
      // The lead dash sits at along≈0 (already brightest by along); leadBoost adds EXTRA brightness on top
      // via vis (the shader scales intensity by vis directly), clamped so additive blend doesn't blow out.
      const baseAlong = K > 1 ? 0.8 * (k / (K - 1)) : 0;
      const along0 = baseAlong;
      const along1 = Math.min(1, along0 + 0.12); // tiny along gradient across the dash so it reads directional
      const dashVis = k === 0 ? Math.min(1, vis * this.leadBoost) : vis; // head dash brightened by leadBoost
      for (let c = 0; c < 6; c++) {
        const [pick, perp] = corners[c]!;
        const ex = pick > 0.5 ? p1x : p0x;
        const ey = pick > 0.5 ? p1y : p0y;
        const ez = pick > 0.5 ? p1z : p0z;
        const al = pick > 0.5 ? along1 : along0;
        v[vi++] = ex; v[vi++] = ey; v[vi++] = ez;
        v[vi++] = pick; v[vi++] = perp;
        v[vi++] = sp;
        v[vi++] = sdx; v[vi++] = sdz;
        v[vi++] = al; v[vi++] = dashVis;
        v[vi++] = 0; // heat=0
      }
    }
    // PAD the remainder of the fixed 48-vert slot with degenerate verts so the stride stays constant.
    while (vi < slotEnd) vi = this.emitDegenerateFarVert(vi);
    return vi;
  }

  // FAR-C CHEVRON: a 2-limb arrowhead glyph per mote, opening UPstream along the flow. Reuses the comet's
  // advection + recycle + speedFrac + vis, samples the flow direction ONCE at the mote head (the same head
  // flow the comet's first backward step uses), then emits TWO quads (one per limb) from the apex (the mote
  // head world pos) back to two tip points = apex - limbLenM·rotate(flowDir, ±spreadAngleDeg). `along`=0 at
  // the apex (bright, ×apexBoost) → 1 at each tip (fades). rakeBySpeed sharpens the spread + grows the limbs
  // with speedFrac (faster = sharper dart). widthPx reuses the global dotPx ribbon half-width. Real quads =
  // 2*6 verts; the slot is PADDED to FAR_VERTS_PER_MOTE with degenerate verts. heat=0 (far air is cool).
  private emitFarChevron(
    i: number,
    camGround: [number, number],
    camFwd: [number, number],
    camRight: [number, number],
    t: number,
    birdPos: [number, number, number],
    dt: number,
    vi: number
  ): number {
    const slotEnd = vi + this.FAR_VERTS_PER_MOTE * Wind.FPV; // exact end of this mote's fixed stride
    const v = this.vertHost;
    const corners = Wind.CORNERS;

    let x0 = this.px[i]!;
    let z0 = this.pz[i]!;
    let y0 = this.py[i]!;

    // Boundary-WRAP (identical to the comet).
    const rx = x0 - camGround[0];
    const rz = z0 - camGround[1];
    const fwdDist = rx * camFwd[0] + rz * camFwd[1];
    const sideDist = rx * camRight[0] + rz * camRight[1];
    if (fwdDist < -this.spanBehind || fwdDist > this.spanAhead) {
      this.seedMote(i, camGround, camFwd, camRight, this.spanAhead * 0.6, 0);
      x0 = this.px[i]!; z0 = this.pz[i]!; y0 = this.py[i]!;
    } else if (Math.abs(sideDist) > this.spanWide) {
      this.seedMote(i, camGround, camFwd, camRight, -this.spanBehind, sideDist > 0 ? -1 : 1);
      x0 = this.px[i]!; z0 = this.pz[i]!; y0 = this.py[i]!;
    }

    // advect by the terrain-shaped, altitude-profiled flow (identical to the comet).
    const [fwx0, fwz0, w] = this.flowAt(x0, z0, t);
    const prof = windProfile(y0);
    const wx = fwx0 * prof, wz = fwz0 * prof;
    const x = x0 + wx * dt;
    const z = z0 + wz * dt;
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
    const u = Math.min(1, Math.max(0, (wspeed - this.speedLo) / (this.speedHi - this.speedLo)));
    let sp = u * u * (3 - 2 * u);
    const climbFrac = Math.min(1, Math.max(0, w / 7.0));
    sp = Math.max(sp, climbFrac);
    this.speedFrac[i] = sp;

    // DENSITY cull + tier crossfade + fade envelope (identical to the comet).
    const rank = hashRank(i);
    const cutoff = this.densityFloor + (1 - this.densityFloor) * sp;
    let vis = 1 - smoothstep(cutoff - 0.1, cutoff + 0.02, rank);
    const bdx = x - birdPos[0], bdz = z - birdPos[2];
    const bdist = Math.hypot(bdx, bdz);
    const blendR = this.nearRadius * 1.6;
    if (bdist < blendR) {
      const proximity = 1 - smoothstep(this.nearRadius, blendR, bdist);
      vis = Math.max(vis, 0.85 * proximity);
    }
    this.age[i]! += dt;
    const fadeIn = smoothstep(0, this.fadeInTime, this.age[i]!);
    const nrx = x - camGround[0], nrz = z - camGround[1];
    const nFwd = nrx * camFwd[0] + nrz * camFwd[1];
    const nSide = nrx * camRight[0] + nrz * camRight[1];
    const distFront = this.spanAhead - nFwd;
    const distBack = nFwd + this.spanBehind;
    const distSide = this.spanWide - Math.abs(nSide);
    const edgeDist = Math.min(distFront, distBack, distSide);
    const fadeOut = smoothstep(0, this.fadeFarEdge, edgeDist);
    const bdy = y - birdPos[1];
    const d3 = this.bubbleFrac(bdx, bdy, bdz);
    const nearHandoff = smoothstep(0.35, 1.0, d3);
    vis *= fadeIn * fadeOut * nearHandoff;
    if (vis <= 0.001) {
      while (vi < slotEnd) vi = this.emitDegenerateFarVert(vi); // collapse the whole slot
      return vi;
    }

    // FLOW DIRECTION at the head (unit world-XZ), the same head flow the comet's first backward step uses.
    let fdx = wx, fdz = wz;
    const fdl = Math.hypot(fdx, fdz);
    if (fdl > 1e-5) { fdx /= fdl; fdz /= fdl; } else { fdx = 1; fdz = 0; }

    // apex = the mote head world pos. The arrowhead OPENS upstream: tips = apex - limbLen·rotate(flowDir,±θ).
    // rakeBySpeed sharpens (narrows) the spread and grows the limbs with speedFrac (faster = longer, sharper).
    const rake = this.rakeBySpeed * sp;
    const spread = (this.spreadAngleDeg * (1 - 0.5 * rake) * Math.PI) / 180; // sharper (narrower) when fast
    const limbLen = this.limbLenM * (1 + 0.6 * rake);                        // longer limbs when fast
    const ca = Math.cos(spread), sa = Math.sin(spread);
    const apx = x, apy = y, apz = z;
    // two backward (upstream) directions: rotate the REVERSED flow dir by ±spread in the XZ plane.
    const bx = -fdx, bz = -fdz; // upstream (opening) base direction
    for (let limb = -1; limb <= 1; limb += 2) {
      const s = limb * sa;
      const rdx = bx * ca - bz * s; // rotate (bx,bz) by ±spread
      const rdz = bx * s + bz * ca;
      const tipx = apx + rdx * limbLen;
      const tipz = apz + rdz * limbLen;
      const tipy = apy; // keep the glyph in a near-horizontal plane (reads as an arrowhead from above/ahead)
      // segDir for the limb's screen-space perp (XZ), normalized.
      let sdx = rdx, sdz = rdz;
      const sdl = Math.hypot(sdx, sdz);
      if (sdl > 1e-5) { sdx /= sdl; sdz /= sdl; } else { sdx = 1; sdz = 0; }
      // apex is the bright point: along=0 (brightest by the shader's head→tail fade) AND boosted further via
      // vis (intensity scales by vis directly), clamped so the additive blend doesn't blow out. The tips fade
      // to along=1 (dark) at the normal vis so the arrowhead reads as a bright point trailing two dim limbs.
      const along0 = 0;
      const along1 = 1;
      const apexVis = Math.min(1, vis * this.apexBoost);
      for (let c = 0; c < 6; c++) {
        const [pick, perp] = corners[c]!;
        const ex = pick > 0.5 ? tipx : apx;
        const ey = pick > 0.5 ? tipy : apy;
        const ez = pick > 0.5 ? tipz : apz;
        const al = pick > 0.5 ? along1 : along0;
        const vv = pick > 0.5 ? vis : apexVis; // apex-side verts brightened by apexBoost
        v[vi++] = ex; v[vi++] = ey; v[vi++] = ez;
        v[vi++] = pick; v[vi++] = perp;
        v[vi++] = sp;
        v[vi++] = sdx; v[vi++] = sdz;
        v[vi++] = al; v[vi++] = vv;
        v[vi++] = 0; // heat=0
      }
    }
    // PAD the remainder of the fixed 48-vert slot with degenerate verts so the stride stays constant.
    while (vi < slotEnd) vi = this.emitDegenerateFarVert(vi);
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

    // WAKE COUNT scales with bird speed (more slipstream when faster), capped at nearWakeCount.
    this._wakeCountNow = Math.round(this.nearWakeCount * Math.min(1, bs / this.wakeSpeedRef));

    for (let i = 0; i < this.nearCount; i++) {
      // LIVE BUDGET: body slots [0,nearBodyCount) always render; wing slots [nearBodyCount, +nearWakeCount) render
      // only while wake is on (modulate) or in filaments; anything beyond is a PARKED spare → degenerate. This makes
      // nearBodyCount / nearWakeCount LIVE-adjustable up to the buffer (nearCount) with no resize.
      const isBodySlot = i < this.nearBodyCount;
      const isWingSlot = i >= this.nearBodyCount && i < this.nearBodyCount + this._wakeCountNow;
      const wingActive = this._wakeOn && (this.wakeMode === "modulate" || this.nearMode === "filaments");
      if (!(isBodySlot || (isWingSlot && wingActive))) {
        let e = vi + this.NEAR_VERTS_PER_MOTE * Wind.FPV;
        while (vi < e) vi = this.emitDegenerateNearVert(vi);
        continue;
      }
      // NEAR render-MODE branch (Phase 2). Each mode fills the SAME fixed per-mote stride
      // (NEAR_VERTS_PER_MOTE = nearSegments·6 verts): flecks emit ONE 2-segment dash (12 verts) then PAD the
      // remainder with degenerate (vis=0) verts; filaments + comet integrate the full nearSegments-segment
      // ribbon (no pad). All three return vi advanced by exactly NEAR_VERTS_PER_MOTE·FPV.
      switch (this.nearMode) {
        case "flecks":
          vi = this.emitNearFlecks(i, birdPos, t, dt, axisX, axisY, axisZ, bs, moving, vi);
          break; // NEAR-B: oriented shear dash
        case "filaments":
          vi = this.emitNearFilaments(i, birdPos, t, dt, axisX, axisY, axisZ, bs, moving, vi);
          break; // NEAR-C: curl filament corkscrewing the wingtip vortices
        case "comet":
        default:
          vi = this.emitNearComet(i, birdPos, t, dt, axisX, axisY, axisZ, bs, moving, vi);
          break; // NEAR-A: short curling comet (today's geometry)
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
      // per-mote DIRECTION JITTER: rotate the AMBIENT flow by a small persistent random angle so the near motes
      // don't all stream in lockstep (nearJitter rad; 0 = off). The bird WAKE added below stays physical/true.
      const ja = this.nearJit[i]! * this.nearJitter;
      const jc = Math.cos(ja), jjs = Math.sin(ja);
      const wx0 = fwx0 * prof, wz0 = fwz0 * prof;
      const wx = wx0 * jc - wz0 * jjs, wz = wx0 * jjs + wz0 * jc;
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
      // WING (wake) motes ride the FULL global wind (ambientW=1) regardless of ambientNearFloor — the user wants
      // the global wind to visibly HIT the wake. Body motes keep the ambientNearFloor stick curve.
      const isWing = i >= this.nearBodyCount;
      const ambientW = isWing ? 1 : (moving ? this.ambientNearFloor + (1 - this.ambientNearFloor) * rrFrac : 1);
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
      const nearVis = fadeIn * fadeOut * this.nearOpacity; // local/wake opacity (dimmer than the far tier)

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
      const stepLen = this.nearSegStep * (isWing ? this.wakeMoteLen : 1) * (1 + this.heatLenGain * heat); // wake length via wakeMoteLen
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
          // rotate the re-sampled ambient flow by the SAME per-mote jitter so the whole tail points consistently.
          const rnx = nwx * prof * jc - nwz * prof * jjs, rnz = nwx * prof * jjs + nwz * prof * jc;
          if (this._wakeOn) {
            this.birdWakeAt(cx, cy, cz, birdPos, axisX, axisY, axisZ, bs, this._wake);
            tfx = rnx * ambientW + this._wake[0]; tfy = nw * ambientW + this._wake[1]; tfz = rnz * ambientW + this._wake[2];
          } else {
            tfx = rnx; tfy = nw; tfz = rnz;
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

  // FIXED per-mote near-tier vertex stride. stepNear() always fills the near span exactly, and that span is
  // sized nearCount * nearSegments * 6 — so EVERY near mote owns exactly this many vertices regardless of which
  // nearMode emitted it. emitNearComet/emitNearFilaments fill it entirely (nearSegments segments × 6);
  // emitNearFlecks emits FEWER real quads and PADS the remainder with degenerate (vis=0) verts, advancing the
  // cursor by exactly this many so the next mote lands at the correct offset. (Mirrors FAR_VERTS_PER_MOTE.)
  private get NEAR_VERTS_PER_MOTE(): number {
    return this.nearSegments * 6;
  }

  // Emit ONE degenerate (collapsed, vis=0) near vertex at cursor `vi` and return the advanced cursor. All 11
  // floats are 0 — the VS collapses vis<0.001 to a point and the FS discards vis<=0.001, so the reserved
  // padding draws nothing. Used by the divergent near geometry (flecks) to fill its fixed slot after the real
  // quads, keeping the per-mote stride constant (see NEAR_VERTS_PER_MOTE). Near equivalent of emitDegenerateFarVert.
  private emitDegenerateNearVert(vi: number): number {
    const v = this.vertHost;
    v[vi++] = 0; v[vi++] = 0; v[vi++] = 0; // pos
    v[vi++] = 0; v[vi++] = 0;              // corner
    v[vi++] = 0;                           // speedFrac
    v[vi++] = 0; v[vi++] = 0;              // segDir
    v[vi++] = 0; v[vi++] = 0;              // along, vis (vis=0 → discarded)
    v[vi++] = 0;                           // heat
    return vi;
  }

  // NEAR-B SHEAR FLECKS geometry emission for one mote (index i). Shares emitNearComet's advection/recycle/heat/
  // fade bookkeeping (so the sphere follows the bird and reads the same touched-air), but the GEOMETRY is a single
  // SHORT 2-segment oriented dash centered on the head along the LOCAL disturbed-velocity dir — NO backward path
  // integration. Length + brightness read local velocity SHEAR (finite-diff of flowAt+birdWakeAt across
  // ±shearRadius). Emits 1 quad (6 verts) and PADS the rest of the fixed slot with degenerate verts; returns the
  // cursor advanced by exactly NEAR_VERTS_PER_MOTE·FPV.
  private emitNearFlecks(
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
    const slotEnd = vi + this.NEAR_VERTS_PER_MOTE * Wind.FPV; // exact end of this mote's fixed stride
    const lo = this.speedLo, hi = this.speedHi;
    const v = this.vertHost;
    const corners = Wind.CORNERS;
    let x0 = this.nx[i]!, y0 = this.ny[i]!, z0 = this.nz[i]!;

    // recycle (identical to the comet): outside the forward-stretched bubble → reseed back inside.
    const dxb = x0 - birdPos[0], dyb = y0 - birdPos[1], dzb = z0 - birdPos[2];
    if (this.bubbleFrac(dxb, dyb, dzb) > 1) {
      this.seedNearMote(i, birdPos);
      x0 = this.nx[i]!; y0 = this.ny[i]!; z0 = this.nz[i]!;
    }

    // LOCAL disturbed flow at the head (windAt-derived terrain flow + bird wake), the SAME sample the comet
    // takes for advection. This is the dash's orientation source AND the center of the shear finite-diff.
    const [fwx0, fwz0, w] = this.flowAt(x0, z0, t);
    const prof = windProfile(y0);
    const wx = fwx0 * prof, wz = fwz0 * prof;
    let ibx = 0, iby = 0, ibz = 0;
    if (this._wakeOn) {
      this.birdWakeAt(x0, y0, z0, birdPos, axisX, axisY, axisZ, bs, this._wake);
      ibx = this._wake[0]; iby = this._wake[1]; ibz = this._wake[2];
    }
    const adx = x0 - birdPos[0], ady = y0 - birdPos[1], adz = z0 - birdPos[2];
    const rrFrac = Math.min(1, Math.sqrt(adx * adx + ady * ady + adz * adz) / this.nearRadius);
    const ambientW = moving ? this.ambientNearFloor + (1 - this.ambientNearFloor) * rrFrac : 1;
    // FULL disturbed velocity at the head (global immersion + bird wake) — the dash orients along its XZ dir.
    const fwx = wx * ambientW + ibx, fwy = w * ambientW + iby, fwz = wz * ambientW + ibz;

    // TOUCHED-AIR heat (identical to the comet): decay + max with the current wake gain; warm tint + length.
    const wakeMag = Math.sqrt(ibx * ibx + iby * iby + ibz * ibz);
    const gain = Math.min(1, wakeMag / this.heatRef);
    const heat = Math.max(this.nearHeat[i]! * Math.exp(-dt / this.heatTau), gain);
    this.nearHeat[i] = heat;

    // advect the mote head (so flecks DRIFT through the field like the comet, just without a trailing tail).
    const x = x0 + fwx * dt;
    const z = z0 + fwz * dt;
    const terr = this.sampleHeight(x, z);
    let y = y0 + fwy * dt;
    const loY = terr + this.minClear;
    if (y < loY) y = loY;
    this.nx[i] = x; this.ny[i] = y; this.nz[i] = z;

    // fade envelope (no-pop) — identical to the comet so flecks ease in/out at the ball edge.
    this.nearAge[i]! += dt;
    const fadeIn = smoothstep(0, this.fadeInTime, this.nearAge[i]!);
    const ndx = x - birdPos[0], ndy = y - birdPos[1], ndz = z - birdPos[2];
    const distFrac = this.bubbleFrac(ndx, ndy, ndz);
    const fadeOut = 1 - smoothstep(this.fadeNearEdge, 1, distFrac);
    const nearVis = fadeIn * fadeOut * this.nearOpacity; // local/wake opacity (dimmer than the far tier)

    // UNIT dash dir = the disturbed-velocity dir in world XZ at the head. Fallback to the motion axis / +X.
    let ux = fwx, uz = fwz;
    let ul = Math.hypot(ux, uz);
    if (ul > 1e-5) { ux /= ul; uz /= ul; } else if (moving) { ux = axisX; uz = axisZ; ul = Math.hypot(ux, uz); if (ul > 1e-5) { ux /= ul; uz /= ul; } else { ux = 1; uz = 0; } } else { ux = 1; uz = 0; }

    // SHEAR magnitude: finite-difference the disturbed velocity across the flow at head ± shearRadius·dir.
    // shearMag = |v(head + d) − v(head − d)| / (2·shearRadius), with d = shearRadius·dashDir (XZ). Two extra
    // flowAt+birdWakeAt samples (the only extra field cost — flecks have no per-segment tail, so net cheaper).
    const sr = this.shearRadius;
    const hpx = x0 + ux * sr, hpz = z0 + uz * sr;   // head + d
    const hmx = x0 - ux * sr, hmz = z0 - uz * sr;   // head − d
    const [pfx0, pfz0] = this.flowAt(hpx, hpz, t);
    const pprof = windProfile(y0);
    let pvx = pfx0 * pprof, pvz = pfz0 * pprof;
    const [mfx0, mfz0] = this.flowAt(hmx, hmz, t);
    let mvx = mfx0 * pprof, mvz = mfz0 * pprof;
    if (this._wakeOn) {
      this.birdWakeAt(hpx, y0, hpz, birdPos, axisX, axisY, axisZ, bs, this._wake);
      pvx = pvx * ambientW + this._wake[0]; pvz = pvz * ambientW + this._wake[2];
      this.birdWakeAt(hmx, y0, hmz, birdPos, axisX, axisY, axisZ, bs, this._wake);
      mvx = mvx * ambientW + this._wake[0]; mvz = mvz * ambientW + this._wake[2];
    }
    const shearMag = Math.hypot(pvx - mvx, pvz - mvz) / (2 * sr);

    // ORIENT: optionally lerp the persisted long-axis toward the current velocity dir (orientLerp=0 → snap).
    if (this.fleckDirX[i] === 0 && this.fleckDirZ[i] === 0) { this.fleckDirX[i] = ux; this.fleckDirZ[i] = uz; }
    let dirX: number, dirZ: number;
    if (this.orientLerp > 0 && moving) {
      const k = Math.min(1, this.orientLerp);
      dirX = this.fleckDirX[i]! * (1 - k) + ux * k;
      dirZ = this.fleckDirZ[i]! * (1 - k) + uz * k;
      const dl = Math.hypot(dirX, dirZ);
      if (dl > 1e-5) { dirX /= dl; dirZ /= dl; } else { dirX = ux; dirZ = uz; }
    } else { dirX = ux; dirZ = uz; }
    this.fleckDirX[i] = dirX; this.fleckDirZ[i] = dirZ;

    // LENGTH: base fleckLen stretched by shear → uniform air = short, shear layers = long. SHORT 2-segment dash
    // = head ± 0.5·len along the long axis (flat, in the mote's near-horizontal plane).
    const len = this.fleckLen * (1 + this.shearGain * shearMag);
    const half = 0.5 * len;
    const p0x = x - dirX * half, p0z = z - dirZ * half; const p0y = y; // one end
    const p1x = x + dirX * half, p1z = z + dirZ * half; const p1y = y; // other end

    // BRIGHTNESS via speedFrac (it already drives intensity): drive it from SHEAR so high-shear flecks brighten.
    // sheared brightness reads the moving shear surfaces; the calm speed read is kept as a floor so still air
    // isn't black. shearMag is normalized through the same speedLo/Hi band the comet uses (m/s-scale).
    const wspeed = Math.hypot(fwx, fwz);
    const us = Math.min(1, Math.max(0, (wspeed - lo) / (hi - lo)));
    const spBase = us * us * (3 - 2 * us);
    const shu = Math.min(1, Math.max(0, (shearMag - lo) / (hi - lo)));
    const spShear = shu * shu * (3 - 2 * shu);
    const sp = Math.max(spBase * 0.4, spShear); // shear leads; a low speed floor keeps uniform air faintly lit

    // FLAT low-taper `along`: clamp the ramp to ≤ fleckTaper so the dash reads as a FLAT tracer (not head-bright).
    // The shader fades intensity by (1-along)^p, so a LOW along on BOTH ends keeps the whole dash near-uniform.
    const along0 = 0;
    const along1 = Math.min(this.fleckTaper, 0.5); // never a full head→tail comet ramp; capped for a flat read
    // segDir for screen-space perp (XZ), normalized (= the dash dir).
    const sdx = dirX, sdz = dirZ;
    // emit ONE quad (the dash). pick: 0 = head-side end (p0), 1 = far end (p1).
    for (let c = 0; c < 6; c++) {
      const [pick, perp] = corners[c]!;
      const ex = pick > 0.5 ? p1x : p0x;
      const ey = pick > 0.5 ? p1y : p0y;
      const ez = pick > 0.5 ? p1z : p0z;
      const al = pick > 0.5 ? along1 : along0;
      v[vi++] = ex; v[vi++] = ey; v[vi++] = ez;
      v[vi++] = pick; v[vi++] = perp;
      v[vi++] = sp;
      v[vi++] = sdx; v[vi++] = sdz;
      v[vi++] = al; v[vi++] = nearVis;
      v[vi++] = heat; // TOUCHED-AIR heat (loc 6) → warm tint, kept from the existing near-heat path
    }
    // PAD the remainder of the fixed slot with degenerate (vis=0) verts so the per-mote stride stays constant.
    while (vi < slotEnd) vi = this.emitDegenerateNearVert(vi);
    return vi;
  }

  // NEAR-C CURL FILAMENTS geometry emission for one mote (index i). The SAME backward integration as
  // emitNearComet (through flowAt + birdWakeAt, whose twin-Rankine tangential term curls the tail around the
  // wingtip cores) BUT with a LONGER per-segment reach (filSegStep vs nearSegStep) so the corkscrew is longer/
  // looser and the curl reads. Integrates over nearSegments segments → fills the slot EXACTLY (no pad, same as
  // the comet). Constant thin width, gentle head→tail fade (reuses the comet's along/vis/heat handling). Seeding
  // toward the two wingtip cores is already honored by seedNearMote via wingEmitFrac when the wake is on.
  private emitNearFilaments(
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

    // recycle (identical to the comet): outside the forward-stretched bubble → reseed back inside (seedNearMote
    // honors wingEmitFrac when the wake is on, so filaments are born preferentially at the two wingtip cores).
    const dxb = x0 - birdPos[0], dyb = y0 - birdPos[1], dzb = z0 - birdPos[2];
    if (this.bubbleFrac(dxb, dyb, dzb) > 1) {
      this.seedNearMote(i, birdPos);
      x0 = this.nx[i]!; y0 = this.ny[i]!; z0 = this.nz[i]!;
    }

    // advect by the terrain-shaped flow PLUS the bird wake (the twin Rankine vortices) — identical to the comet.
    const [fwx0, fwz0, w] = this.flowAt(x0, z0, t);
    const prof = windProfile(y0);
    const wx = fwx0 * prof, wz = fwz0 * prof;
    let ibx = 0, iby = 0, ibz = 0;
    if (this._wakeOn) {
      this.birdWakeAt(x0, y0, z0, birdPos, axisX, axisY, axisZ, bs, this._wake);
      ibx = this._wake[0]; iby = this._wake[1]; ibz = this._wake[2];
    }
    const wakeMag = Math.sqrt(ibx * ibx + iby * iby + ibz * ibz);
    const gain = Math.min(1, wakeMag / this.heatRef);
    const heat = Math.max(this.nearHeat[i]! * Math.exp(-dt / this.heatTau), gain);
    this.nearHeat[i] = heat;
    const adx = x0 - birdPos[0], ady = y0 - birdPos[1], adz = z0 - birdPos[2];
    const rrFrac = Math.min(1, Math.sqrt(adx * adx + ady * ady + adz * adz) / this.nearRadius);
    const ambientW = moving ? this.ambientNearFloor + (1 - this.ambientNearFloor) * rrFrac : 1;
    const fwx = wx * ambientW + ibx, fwy = w * ambientW + iby, fwz = wz * ambientW + ibz;
    const x = x0 + fwx * dt;
    const z = z0 + fwz * dt;
    const terr = this.sampleHeight(x, z);
    let y = y0 + fwy * dt;
    const loY = terr + this.minClear;
    if (y < loY) y = loY;
    this.nx[i] = x; this.ny[i] = y; this.nz[i] = z;

    // fade envelope (no-pop) — identical to the comet.
    this.nearAge[i]! += dt;
    const fadeIn = smoothstep(0, this.fadeInTime, this.nearAge[i]!);
    const ndx = x - birdPos[0], ndy = y - birdPos[1], ndz = z - birdPos[2];
    const distFrac = this.bubbleFrac(ndx, ndy, ndz);
    const fadeOut = 1 - smoothstep(this.fadeNearEdge, 1, distFrac);
    const nearVis = fadeIn * fadeOut * this.nearOpacity; // local/wake opacity (dimmer than the far tier)

    // speed tint (same calibration as the comet) — uses the DISTURBED horizontal speed.
    const wspeed = Math.hypot(fwx, fwz);
    const u = Math.min(1, Math.max(0, (wspeed - lo) / (hi - lo)));
    let sp = u * u * (3 - 2 * u);
    const climbFrac = Math.min(1, Math.max(0, w / 7.0));
    sp = Math.max(sp, climbFrac);

    // CORKSCREW tail: integrate BACKWARD along the LOCAL disturbed flow (wind + bird wake), re-sampling at each
    // point so the thread WRAPS the wingtip vortex cores. Same loop as the comet BUT with the LONGER filSegStep
    // (looser, readable spiral) and the heat-driven length extension. Per-point terrain clamp kept.
    const stepLen = this.filSegStep * (1 + this.heatLenGain * heat);
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
      // re-sample the disturbed flow at the NEW point for the NEXT step → the filament corkscrews the core.
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

    // emit a quad per segment (constant thin width via the shared ribbon); vis = fade envelope, along = head→
    // tail fade. Fills the slot EXACTLY (seg segments × 6 = NEAR_VERTS_PER_MOTE) — no pad.
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
        v[vi++] = al; v[vi++] = nearVis;
        v[vi++] = heat; // TOUCHED-AIR heat (loc 6) → warm tint
      }
    }
    return vi;
  }

  // --- WAKE-SHED stepper (Phase 2): dedicated shed geometry written into the THIRD buffer span (after far+near).
  // Drives ONE of two persistent pools depending on wakeMode: "helix" (WAKE-B wingtip cords) or "rings"
  // (WAKE-C shed hoops). Sets up its OWN per-frame bird/wing frame (stepNear may NOT have run — the local
  // sphere can be off while the shed geometry is on) so birdWakeAt() has the wing-right + axis it reads.
  // Writes quads contiguously from the wake-shed span start (float offset (farVertexCount+nearVertexCount)·FPV),
  // tracks the verts ACTUALLY written in this.wakeShedLiveCount, and NEVER exceeds WAKE_SHED_RESERVE (it stops
  // emitting and logs once rather than overrun). When the wake is off OR wakeMode==="modulate" the caller skips
  // this and zeroes wakeShedLiveCount. windAt / flight physics are untouched (render-only).
  private stepWakeShed(
    birdPos: [number, number, number],
    axisX: number, axisY: number, axisZ: number,
    rgx: number, rgy: number, rgz: number,
    bs: number,
    moving: boolean,
    t: number,
    dt: number
  ): void {
    // Set the per-frame wake frame birdWakeAt() reads (this._r* for the vortex cores; _ax/_bs/_lastBirdPos for
    // the probe). Mirrors the top of stepNear so the shed path is self-contained when the near sphere is off.
    this._ax = axisX; this._ay = axisY; this._az = axisZ;
    this._rx = rgx; this._ry = rgy; this._rz = rgz;
    this._bs = bs; this._moving = moving; this._wakeOn = moving && this.showWake;
    this._lastBirdPos[0] = birdPos[0]; this._lastBirdPos[1] = birdPos[1]; this._lastBirdPos[2] = birdPos[2];

    const base = (this.farVertexCount + this.nearVertexCount) * Wind.FPV; // float cursor at the wake-shed span start
    if (this.wakeMode === "rings") {
      this.wakeShedLiveCount = this.stepShedRings(birdPos, axisX, axisY, axisZ, rgx, rgy, rgz, bs, moving, t, dt, base);
    } else { // "helix"
      this.wakeShedLiveCount = this.stepShedHelix(birdPos, axisX, axisY, axisZ, rgx, rgy, rgz, bs, moving, t, dt, base);
    }
  }

  // WAKE-B WINGTIP HELIX FILAMENTS: continuously SHED short tube-arcs from each wingtip (birdPos ± wingSpan·right)
  // at wakeEmitRate arcs/tip/sec. Each shed element persists (seed advected by flowAt + birdWakeAt; retired at
  // wakeLife). Each frame, per live element, integrate a SHORT backward polyline (wakeSeg ≤ HELIX_SEGS) through
  // flowAt + birdWakeAt (the twin-vortex tangential term corkscrews it around its core) and emit wakeSeg quads via
  // the CORNERS pattern. The two tips spiral in OPPOSITE senses (counter-rotating) — that falls out of the twin
  // -vortex field; counterRotate=false flips one tip's sense back to match. helixGain multiplies the swirl term FOR
  // SHED GEOMETRY ONLY (temporarily scales swirlGain during integration, restored after). Warm via the heat channel
  // where |birdWakeAt| is large. Returns the verts written (capped at WAKE_SHED_RESERVE). NO new physics.
  private stepShedHelix(
    birdPos: [number, number, number],
    axisX: number, axisY: number, axisZ: number,
    rgx: number, rgy: number, rgz: number,
    bs: number,
    moving: boolean,
    t: number,
    dt: number,
    base: number
  ): number {
    const HELIX_CAP = Wind.HELIX_TIPS * Wind.HELIX_LIVE;
    // upW = axis × right (thin vertical jitter axis for the seed spread off the wingtip).
    const ux = axisY * rgz - axisZ * rgy, uy = axisZ * rgx - axisX * rgz, uz = axisX * rgy - axisY * rgx;

    // 1) RETIRE aged elements (compact the live pool in place — order doesn't matter for emission).
    let n = this.helixActive;
    for (let i = 0; i < n; ) {
      this.helixAge[i]! += dt;
      if (this.helixAge[i]! >= this.wakeLife) {
        n--;
        this.helixSeedX[i] = this.helixSeedX[n]!; this.helixSeedY[i] = this.helixSeedY[n]!; this.helixSeedZ[i] = this.helixSeedZ[n]!;
        this.helixAge[i] = this.helixAge[n]!; this.helixSide[i] = this.helixSide[n]!;
      } else i++;
    }
    this.helixActive = n;

    // 2) SHED new arcs (both tips) at wakeEmitRate per tip per second when moving. Accumulate fractional owed.
    if (moving) {
      this.helixEmitAcc += this.wakeEmitRate * Wind.HELIX_TIPS * dt; // both tips
      let owed = Math.floor(this.helixEmitAcc);
      this.helixEmitAcc -= owed;
      while (owed > 0 && this.helixActive < HELIX_CAP) {
        const side = (this.helixActive & 1) === 0 ? 1 : -1; // alternate tips
        const offR = side * this.wingSpan;
        const jU = (Math.random() * 2 - 1) * this.wingJitter;
        const x = birdPos[0] + rgx * offR + ux * jU;
        const y = birdPos[1] + rgy * offR + uy * jU;
        const z = birdPos[2] + rgz * offR + uz * jU;
        const k = this.helixActive++;
        this.helixSeedX[k] = x; this.helixSeedY[k] = y; this.helixSeedZ[k] = z;
        this.helixAge[k] = 0; this.helixSide[k] = side;
        owed--;
      }
    }

    // 3) ADVANCE + EMIT. helixGain scales the swirl term for shed geometry only (restore after).
    const savedSwirl = this.swirlGain;
    this.swirlGain = savedSwirl * this.helixGain;
    const v = this.vertHost;
    const corners = Wind.CORNERS;
    const seg = this.wakeSeg;
    const ptX = this.wsPtX, ptY = this.wsPtY, ptZ = this.wsPtZ;
    const wake = this._wsWake;
    const maxFloats = base + Wind.WAKE_SHED_RESERVE * Wind.FPV; // hard end of the reserved span (float index)
    let vi = base;
    const live = this.helixActive;
    for (let i = 0; i < live; i++) {
      // advect this element's SEED point by flowAt + birdWakeAt (so the cord drifts off the wing and downstream).
      let sx = this.helixSeedX[i]!, sy = this.helixSeedY[i]!, sz = this.helixSeedZ[i]!;
      const side = this.helixSide[i]!;
      const [afx, afz, aw] = this.flowAt(sx, sz, t);
      const aprof = windProfile(sy);
      let vx = afx * aprof, vy = aw, vz = afz * aprof;
      this.sampleHelixWake(sx, sy, sz, birdPos, axisX, axisY, axisZ, bs, side, wake);
      vx += wake[0]; vy += wake[1]; vz += wake[2];
      sx += vx * dt; sy += vy * dt; sz += vz * dt;
      const sfloor = this.sampleHeight(sx, sz) + this.minClear;
      if (sy < sfloor) sy = sfloor;
      this.helixSeedX[i] = sx; this.helixSeedY[i] = sy; this.helixSeedZ[i] = sz;

      // age fade (head bright on shed, fades over wakeLife) folded into vis so a retiring cord doesn't pop out.
      const ageFrac = this.helixAge[i]! / this.wakeLife;             // 0 fresh → 1 retire
      const lifeFade = 1 - smoothstep(0.6, 1.0, ageFrac);            // full life → fades over the last 40%
      const bornFade = smoothstep(0, this.fadeInTime, this.helixAge[i]!); // ease in on shed (no pop)
      const vis = lifeFade * bornFade;
      if (vis <= 0.001) continue; // skip emitting this element (its slot is simply not written → fewer live verts)

      // SHORT backward polyline through flowAt + birdWakeAt (corkscrews around the core). Head = the seed.
      ptX[0] = sx; ptY[0] = sy; ptZ[0] = sz;
      let cx = sx, cy = sy, cz = sz;
      let tfx = vx, tfy = vy, tfz = vz; // first backward step reuses the head's disturbed flow (free)
      let maxWake = Math.sqrt(wake[0] * wake[0] + wake[1] * wake[1] + wake[2] * wake[2]); // peak |wake| → heat
      for (let s = 1; s <= seg; s++) {
        cx -= tfx * this.wakeSegStep;
        cy -= tfy * this.wakeSegStep;
        cz -= tfz * this.wakeSegStep;
        const tb = this.sampleHeight(cx, cz) + this.minClear;
        if (cy < tb) cy = tb;
        ptX[s] = cx; ptY[s] = cy; ptZ[s] = cz;
        if (s < seg) {
          const [nwx, nwz, nw] = this.flowAt(cx, cz, t);
          const nprof = windProfile(cy);
          this.sampleHelixWake(cx, cy, cz, birdPos, axisX, axisY, axisZ, bs, side, wake);
          const wm = Math.sqrt(wake[0] * wake[0] + wake[1] * wake[1] + wake[2] * wake[2]);
          if (wm > maxWake) maxWake = wm;
          tfx = nwx * nprof + wake[0]; tfy = nw + wake[1]; tfz = nwz * nprof + wake[2];
        }
      }
      // heat: warm where the twin-vortex disturbance is strong (the genuine wake), 0..1 vs heatRef.
      const heat = Math.min(1, maxWake / this.heatRef);
      // speed tint from the head disturbed horizontal speed (cyan→white), same calibration as the motes.
      const wspeed = Math.hypot(vx, vz);
      const u = Math.min(1, Math.max(0, (wspeed - this.speedLo) / (this.speedHi - this.speedLo)));
      const sp = u * u * (3 - 2 * u);

      // would this element's quads overrun the reserved span? if so STOP (cap protection) — log once.
      if (vi + seg * 6 * Wind.FPV > maxFloats) {
        if (!this.wakeOverrunLogged) { console.warn("[wind] wake-shed helix hit reserve cap; halting emit this frame"); this.wakeOverrunLogged = true; }
        break;
      }
      // emit a quad per segment via the CORNERS pattern. wakeTaper biases the head→tail along so the cord
      // tapers (the VS taper + FS lenFade already darken toward the tail; wakeTaper raises the base ramp).
      for (let s = 0; s < seg; s++) {
        const axp = ptX[s]!, ayp = ptY[s]!, azp = ptZ[s]!;
        const bxp = ptX[s + 1]!, byp = ptY[s + 1]!, bzp = ptZ[s + 1]!;
        let sdx = bxp - axp, sdz = bzp - azp;
        const sdl = Math.hypot(sdx, sdz);
        if (sdl > 1e-5) { sdx /= sdl; sdz /= sdl; } else { sdx = 1; sdz = 0; }
        const alongN = this.wakeTaper * (s / seg);
        const alongF = this.wakeTaper * ((s + 1) / seg);
        for (let c = 0; c < 6; c++) {
          const [pick, perp] = corners[c]!;
          const ex = pick > 0.5 ? bxp : axp;
          const ey = pick > 0.5 ? byp : ayp;
          const ez = pick > 0.5 ? bzp : azp;
          const al = pick > 0.5 ? alongF : alongN;
          v[vi++] = ex; v[vi++] = ey; v[vi++] = ez;
          v[vi++] = pick; v[vi++] = perp;
          v[vi++] = sp;
          v[vi++] = sdx; v[vi++] = sdz;
          v[vi++] = al; v[vi++] = vis;
          v[vi++] = heat;
        }
      }
    }
    this.swirlGain = savedSwirl; // restore (helixGain is shed-only)
    return (vi - base) / Wind.FPV; // verts actually written
  }

  // Sample the bird-wake disturbance for a HELIX shed point, honoring counterRotate. When counterRotate=true
  // (default) birdWakeAt's twin cores already counter-rotate (the −c sign per side). When false, sample the
  // wake at the point MIRRORED across the centerline for the −side so it picks up the +side's rotational sense
  // → both cords spiral the SAME way. Writes [x,y,z] into `out` (no allocation).
  private sampleHelixWake(
    px: number, py: number, pz: number,
    birdPos: [number, number, number],
    axisX: number, axisY: number, axisZ: number,
    bs: number,
    side: number,
    out: [number, number, number]
  ): void {
    if (!this._wakeOn) { out[0] = 0; out[1] = 0; out[2] = 0; return; }
    if (this.counterRotate || side >= 0) {
      this.birdWakeAt(px, py, pz, birdPos, axisX, axisY, axisZ, bs, out);
      return;
    }
    // counterRotate OFF and this is the −side cord: mirror the sample point across the centerline so the
    // tangential sense matches the +side, then mirror the resulting lateral component back.
    const dx = px - birdPos[0], dy = py - birdPos[1], dz = pz - birdPos[2];
    const lat = dx * this._rx + dy * this._ry + dz * this._rz; // signed lateral offset
    const mx = px - 2 * lat * this._rx, my = py - 2 * lat * this._ry, mz = pz - 2 * lat * this._rz;
    this.birdWakeAt(mx, my, mz, birdPos, axisX, axisY, axisZ, bs, out);
    const olat = out[0] * this._rx + out[1] * this._ry + out[2] * this._rz; // mirror the lateral component back
    out[0] -= 2 * olat * this._rx; out[1] -= 2 * olat * this._ry; out[2] -= 2 * olat * this._rz;
  }

  // WAKE-C SHED PRESSURE RINGS: on a ringRate cadence SHED a closed loop of ringSegN short chords, oriented
  // FACE-ON to the flight axis (plane basis = wingRight and up = axis × wingRight). Each ring persists: radius
  // += ringGrow·dt, center advected by flowAt + convectFrac·axis·bs·dt (backward), age++, retired at ringLife.
  // twinOffset>0 = one ring per wingtip (center at birdPos ± wingSpan·right); 0 = centerline train. Each frame,
  // per live ring, tessellate ringSegN points and emit each consecutive pair as one CORNERS quad with segDir =
  // around-ring tangent (the VS keeps it a clean constant-px hoop). `along` 0..1 around the loop for fade;
  // ringTilt tilts the normal by local shear; heat sampled once per ring at spawn; ringWarmBias warms the loaded
  // (downstream) side. Returns verts written (capped at reserve). windAt / physics untouched.
  private stepShedRings(
    birdPos: [number, number, number],
    axisX: number, axisY: number, axisZ: number,
    rgx: number, rgy: number, rgz: number,
    bs: number,
    moving: boolean,
    t: number,
    dt: number,
    base: number
  ): number {
    // up = axis × wingRight (the ring plane's second basis; with wingRight it spans the plane normal to axis).
    const upx = axisY * rgz - axisZ * rgy, upy = axisZ * rgx - axisX * rgz, upz = axisX * rgy - axisY * rgx;

    // 1) RETIRE aged rings (compact in place), and ADVANCE the survivors (grow + convect backward).
    let n = this.ringActive;
    for (let i = 0; i < n; ) {
      this.ringAge[i]! += dt;
      if (this.ringAge[i]! >= this.ringLife) {
        n--;
        this.ringCx[i] = this.ringCx[n]!; this.ringCy[i] = this.ringCy[n]!; this.ringCz[i] = this.ringCz[n]!;
        this.ringRadius[i] = this.ringRadius[n]!; this.ringAge[i] = this.ringAge[n]!;
        this.ringSide[i] = this.ringSide[n]!; this.ringHeat[i] = this.ringHeat[n]!;
      } else i++;
    }
    this.ringActive = n;
    for (let i = 0; i < this.ringActive; i++) {
      this.ringRadius[i]! += this.ringGrow * dt;
      // center advected by the terrain flow + a backward convection along the axis (the ring trails the bird).
      const cx = this.ringCx[i]!, cy = this.ringCy[i]!, cz = this.ringCz[i]!;
      const [cfx, cfz, cw] = this.flowAt(cx, cz, t);
      const cprof = windProfile(cy);
      const back = this.convectFrac * bs;
      let ncx = cx + (cfx * cprof - axisX * back) * dt;
      let ncy = cy + (cw - axisY * back) * dt;
      let ncz = cz + (cfz * cprof - axisZ * back) * dt;
      const cfloor = this.sampleHeight(ncx, ncz) + this.minClear;
      if (ncy < cfloor) ncy = cfloor;
      this.ringCx[i] = ncx; this.ringCy[i] = ncy; this.ringCz[i] = ncz;
    }

    // 2) SHED new rings on the ringRate cadence when moving. twinOffset>0 → one per wingtip; 0 → centerline.
    if (moving) {
      this.ringEmitAcc += this.ringRate * dt;
      let owed = Math.floor(this.ringEmitAcc);
      this.ringEmitAcc -= owed;
      while (owed > 0) {
        const sides = this.twinOffset > 0 ? [1, -1] : [0];
        for (const side of sides) {
          if (this.ringActive >= Wind.RING_COUNT) break;
          const off = side * this.wingSpan;
          const cx = birdPos[0] + rgx * off;
          const cy = birdPos[1] + rgy * off;
          const cz = birdPos[2] + rgz * off;
          // heat sampled once at spawn from |birdWakeAt(center)| (the genuine wake loading the ring).
          let heat = 0;
          if (this._wakeOn) {
            this.birdWakeAt(cx, cy, cz, birdPos, axisX, axisY, axisZ, bs, this._wsWake);
            heat = Math.min(1, Math.sqrt(this._wsWake[0] ** 2 + this._wsWake[1] ** 2 + this._wsWake[2] ** 2) / this.heatRef);
          }
          const k = this.ringActive++;
          this.ringCx[k] = cx; this.ringCy[k] = cy; this.ringCz[k] = cz;
          this.ringRadius[k] = this.ringStartRadius; this.ringAge[k] = 0;
          this.ringSide[k] = side; this.ringHeat[k] = heat;
        }
        owed--;
      }
    }

    // 3) TESSELLATE + EMIT each live ring. Plane basis (e0,e1) = wingRight, up; tilt the up-axis by ringTilt so
    // the hoop rakes with the local shear (a fixed tilt toward +axis here — cheap, reads as a tilted hoop).
    const v = this.vertHost;
    const corners = Wind.CORNERS;
    const segN = this.ringSegN;
    const maxFloats = base + Wind.WAKE_SHED_RESERVE * Wind.FPV;
    let vi = base;
    // tilted up-axis = normalize(up + ringTilt·axis) → the ring face rakes toward the flight axis.
    let tux = upx + this.ringTilt * axisX, tuy = upy + this.ringTilt * axisY, tuz = upz + this.ringTilt * axisZ;
    const tul = Math.hypot(tux, tuy, tuz);
    if (tul > 1e-5) { tux /= tul; tuy /= tul; tuz /= tul; } else { tux = upx; tuy = upy; tuz = upz; }
    for (let i = 0; i < this.ringActive; i++) {
      // would this ring's quads overrun the reserved span? if so STOP (cap protection) — log once.
      if (vi + segN * 6 * Wind.FPV > maxFloats) {
        if (!this.wakeOverrunLogged) { console.warn("[wind] wake-shed rings hit reserve cap; halting emit this frame"); this.wakeOverrunLogged = true; }
        break;
      }
      const cx = this.ringCx[i]!, cy = this.ringCy[i]!, cz = this.ringCz[i]!;
      const r = this.ringRadius[i]!;
      const ageFrac = this.ringAge[i]! / this.ringLife;     // 0 fresh → 1 retire
      const lifeFade = 1 - smoothstep(0.5, 1.0, ageFrac);   // fade over the last half of life
      const bornFade = smoothstep(0, this.fadeInTime, this.ringAge[i]!);
      const ringVis = lifeFade * bornFade;
      const baseHeat = this.ringHeat[i]!;
      const side = this.ringSide[i]!;
      // speed tint from the local flow speed at the ring center (cyan→white), same calibration as the motes.
      const [rfx, rfz] = this.flowAt(cx, cz, t);
      const rprof = windProfile(cy);
      const wspeed = Math.hypot(rfx * rprof, rfz * rprof);
      const u = Math.min(1, Math.max(0, (wspeed - this.speedLo) / (this.speedHi - this.speedLo)));
      const sp = u * u * (3 - 2 * u);
      // precompute the loop points (segN points around the circle, closed: point segN == point 0).
      // emit a quad per consecutive pair. The around-ring tangent is the segDir (clean constant-px hoop).
      let px0 = cx + rgx * r, py0 = cy + rgy * r, pz0 = cz + rgz * r; // angle 0 (along +wingRight)
      for (let s = 0; s < segN; s++) {
        const a1 = (2 * Math.PI * (s + 1)) / segN;
        const ca = Math.cos(a1), sa = Math.sin(a1);
        const px1 = cx + (rgx * ca + tux * sa) * r;
        const py1 = cy + (rgy * ca + tuy * sa) * r;
        const pz1 = cz + (rgz * ca + tuz * sa) * r;
        let sdx = px1 - px0, sdz = pz1 - pz0;
        const sdl = Math.hypot(sdx, sdz);
        if (sdl > 1e-5) { sdx /= sdl; sdz /= sdl; } else { sdx = 1; sdz = 0; }
        // along around the loop for a gentle fade; ringWarmBias warms the loaded (downstream −axis) side: the
        // chord whose midpoint sits on the −side of the wingRight axis (toward the trailing edge) heats more.
        const midDot = (rgx * (Math.cos(2 * Math.PI * (s + 0.5) / segN)) + tux * (Math.sin(2 * Math.PI * (s + 0.5) / segN))); // wingRight-component of the chord midpoint dir
        const warm = Math.min(1, baseHeat + this.ringWarmBias * Math.max(0, -side * midDot));
        const along0 = 0.15; // flat, low ramp so the whole hoop reads (not head-bright)
        const along1 = 0.15;
        for (let c = 0; c < 6; c++) {
          const [pick, perp] = corners[c]!;
          const ex = pick > 0.5 ? px1 : px0;
          const ey = pick > 0.5 ? py1 : py0;
          const ez = pick > 0.5 ? pz1 : pz0;
          const al = pick > 0.5 ? along1 : along0;
          v[vi++] = ex; v[vi++] = ey; v[vi++] = ez;
          v[vi++] = pick; v[vi++] = perp;
          v[vi++] = sp;
          v[vi++] = sdx; v[vi++] = sdz;
          v[vi++] = al; v[vi++] = ringVis;
          v[vi++] = warm;
        }
        px0 = px1; py0 = py1; pz0 = pz1;
      }
    }
    return (vi - base) / Wind.FPV;
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
    // WAKE-SHED tier (Phase 2): dedicated helix cords / shed rings into the THIRD span. Driven INDEPENDENTLY of
    // the near sphere (showNear may be off while shed geometry is on), gated on showWake && wakeMode!=="modulate".
    // Compute the bird/wing FRAME here (mirrors the top of stepNear) so it's available even when stepNear didn't
    // run, then hand it to stepWakeShed. When the gate is off, wakeShedLiveCount=0 so draw() skips the shed draw.
    if (this.showWake && this.wakeMode !== "modulate") {
      const bvx = birdVel[0], bvy = birdVel[1], bvz = birdVel[2];
      const bs = Math.hypot(bvx, bvy, bvz);
      const moving = bs > 0.5;
      const axisX = moving ? bvx / bs : 0, axisY = moving ? bvy / bs : 0, axisZ = moving ? bvz / bs : 0;
      let rgx = -axisZ, rgz = axisX; // wing-right = normalize(axis × worldUp); fallback world X if axis ~vertical
      const rgl = Math.hypot(rgx, rgz);
      if (rgl > 1e-3) { rgx /= rgl; rgz /= rgl; } else { rgx = 1; rgz = 0; }
      this.stepWakeShed(birdPos, axisX, axisY, axisZ, rgx, 0, rgz, bs, moving, time, dt);
    } else {
      this.wakeShedLiveCount = 0; // wake off / modulate → no shed geometry; the shed draw is skipped
    }
    // single upload of the combined (far + near + wake-shed) vertex buffer.
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
