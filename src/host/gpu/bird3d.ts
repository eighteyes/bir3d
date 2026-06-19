// bird3d.ts — Bird3D: one CPU-integrated 3D bird + neon flapping-V render pipeline (depth-tested).
// Responsibilities:
//   - Hold bird state {pos:vec3, speed (scalar airspeed), heading, pitch, bank}; integrate(dt, input):
//     energy-exchange glider — pitch trades airspeed for altitude (dive to gain speed, pull up to
//     zoom-climb), drag relaxes speed toward trim, base sink rate (mushes when slow), RIDGE LIFT as
//     vertical AIR MOTION the bird rides (wind · uphill gradient, finite-diff of sampleHeight),
//     plus explicit THERMAL updraft, plus CRANKED horizontal wind drift (shared analytic curl-noise
//     field from wind.ts — the SAME field the streamline overlay draws; FLAGGED stand-in for the GPU
//     fluid), altitude clamp above ground. Wind is cranked so cross-track drift is unmistakable.
//     BUFFETING: a fast-varying turbulence term (fast time + spatial phase) rocks the render-bank
//     (±~7°), bobs vel[1] (±~1.5 m/s, so vario oscillates) and shoves horizontal vel in pulses; plus a
//     steady CRAB lean of the render-bank into the cross-wind. Felt, controllable — not violent.
//     Exposes window.__birdBank (the drawn render-bank) for the buffet probe.
//   - Expose public `tuning` (live-writable) so a host overlay can slide feel parameters at runtime.
//   - Build a procedural V mesh (body + two swept dihedral wing ribbons) as triangle strips packed
//     to a triangle list; each vertex carries (signed spanFrac, wingFlag, edgeFrac) for shader flap.
//   - Own the bird uniform + vertex buffer + render pipeline (bird3d.wgsl) WITH depthStencil
//     (depth24plus, less, write on) so terrain ridges occlude the bird.
//   - draw(encoder, colorView, depthView, viewProj, time): record one bird draw in a SECOND pass
//     that LOADS the terrain color+depth (no clear) — terrain must already be drawn this encoder.
//   - forwardVec()/heading expose the chase convention (+Z forward) for the camera.

import type { TerrainEKG } from "./terrain";
import { windAt, thermalAt, flowHorizontal } from "./wind";

export interface BirdInput {
  yawRate: number;     // rad/s, from mouse-x offset (rate: hold to keep turning)
  pitchTarget: number; // rad, from mouse-y offset (ATTITUDE: cursor height = nose angle, holdable)
  flap: boolean;       // wingbeat held (Space): powered climb — tap = one beat, hold = sustained
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
  deflect?: number;      // terrain into-slope deflection of horizontal drift — MUST match Wind.deflect to ride the motes
  flexHz?: number;       // subtle idle wing-flex frequency (living glide, NOT a flap beat)
  flexAmp?: number;      // subtle idle wing-flex amplitude (rad)
  beatHz?: number;       // POWERED FLAP: wingbeats/sec while held (sustained-climb cadence)
  beatLift?: number;     // peak vertical lift from a symmetric beat (m/s into target vertical vel)
  beatThrust?: number;   // peak forward thrust during a beat (m/s^2) — sustains airspeed in a climb
  beatAmp?: number;      // visual flap amplitude at full beat (rad)
  flapAsym?: number;     // steering→wing asymmetry: outer wing beats harder (turn from the difference)
  flapTurn?: number;     // yaw assist (rad/s) from the wing difference while flapping
  crashSpeed?: number;   // CRASH threshold: closing speed into terrain (m/s) above which a touch = a crash
  crashBleed?: number;   // fraction of airspeed lost in a crash (0..1)
  crashTime?: number;    // stumble duration after a crash (s) — steering degraded while > 0
  minClearance?: number; // min meters above terrain
}

type Vec3 = [number, number, number];

// Updraft the PHYSICS applies at world (x,z) at sim time t — ridge lift (wind · uphill, gained)
// + thermal cores, capped. Exported so the autopilot SENSES exactly the air the bird RIDES.
export function updraftAt(
  x: number,
  z: number,
  t: number,
  terrain: TerrainEKG,
  T: Required<BirdTuning>
): number {
  const [bwx, bwz] = windAt(x, z, t);
  const wx = bwx * T.windGain;
  const wz = bwz * T.windGain;
  const eps = 6;
  const hC = terrain.sampleHeight(x, z);
  const gx = (terrain.sampleHeight(x + eps, z) - hC) / eps;
  const gz = (terrain.sampleHeight(x, z + eps) - hC) / eps;
  const ridge = Math.max(0, wx * gx + wz * gz) * T.liftGain;
  const thermal = thermalAt(x, z, t) * 1.8; // stronger sparse cores (see integrate)
  return Math.min(8.0, ridge + thermal);    // cap mirrors integrate's anti-launch clamp (v14: 5.5→8.0)
}

const FLOATS_PER_VERT = 6; // local.xyz + attr.xyz
const UNIFORM_BYTES = 112; // mat4(64) + pos,flexPhase(16) + heading,bank,flexAmp,flapPhase(16) + ampL,ampR,pitch,pad(16)

export class Bird3D {
  pos: Vec3;
  vel: Vec3 = [0, 0, 18];
  speed = 26;    // scalar airspeed (m/s) — the energy store
  heading = 0;   // yaw, +Z forward at 0
  pitch = 0;     // radians, + = nose up
  bank = 0;      // roll, banks into turns
  renderBank = 0; // bank actually drawn: steering bank + crab lean + buffet rock (visual only)
  // STILL_AIR: dead-calm airframe basis — when true the bird ignores wind drift, ridge lift, thermal,
  // and buffet/rock. The wind.ts field + motes keep evolving (rendering untouched); the BIRD just flies
  // through dead air. This is the hook to re-introduce wind as "flair" later — set false to restore the
  // full soaring model.
  stillAir = false;
  // POWERED FLAP state: a beat runs to completion once started (so a tap = one full beat); held input
  // re-triggers the next beat (sustained climb). ampL/ampR drive the per-wing visual beat (independent).
  private beatActive = false;
  private beatPhase = 0;     // 0..1 progress through the current beat
  private flapBeatPhase = 0; // radians 0..PI for the shader downstroke (0 when idle)
  private ampL = 0;          // LEFT wing visual flap amplitude this frame (rad)
  private ampR = 0;          // RIGHT wing visual flap amplitude this frame (rad)
  lastFlapping = false;      // telemetry — actively beating this frame (HUD)
  private crashT = 0;        // crash-stumble timer (s); > 0 = recovering, steering degraded
  lastCrashing = false;      // telemetry — in a crash stumble this frame (HUD)
  private time = 0;

  tuning: Required<BirdTuning>;

  // latest derived telemetry for the overlay
  lastWind: [number, number] = [0, 0];
  lastSpeed = 0;
  lastClearance = 0;
  lastVario = 0;     // vertical speed (m/s)
  lastUpdraft = 0;   // total updraft being ridden (ridge + thermal, m/s)
  lastGroundTrack = 0; // actual horizontal travel direction (rad) — differs from heading under wind

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
    t: BirdTuning = {},
    sampleCount = 1 // MSAA samples — must match the render target + every pipeline in the pass
  ) {
    this.pos = startPos;
    this.tuning = {
      glideSpeed: t.glideSpeed ?? 26,
      minSpeed: t.minSpeed ?? 13,
      maxSpeed: t.maxSpeed ?? 70,  // headroom so a committed dive keeps gaining (was 55 — capped early)
      dragK: t.dragK ?? 0.2,       // low bleed: dive speed HOLDS through the swoop (feeds the zoom)
      divePower: t.divePower ?? 2.0, // STRONG gravity dive — nose-down alone screams (no flap needed)
      gravity: t.gravity ?? 9.0,
      sinkRate: t.sinkRate ?? 1.0,   // v14: gentler sink (1.4→1.0) — glide efficiently, more time to find lift
      windGain: t.windGain ?? 1.6,  // multiplier on the shared windAt field (CRANKED)
      windDrift: t.windDrift ?? 1.0, // fraction of horizontal wind the glider drifts with
      liftGain: t.liftGain ?? 3.5,   // v14: stronger ridge lift (2.5→3.5) — soaring the windward ridges sustains you
      deflect: t.deflect ?? 0.25,    // v17b REVERTED to 0.25 (the no-complaint baseline): syncing to the motes (0.45)
                                     // + the taller RELIEF (steeper gradients saturate flowHorizontal's into-slope shed)
                                     // made the bird's drift SPATIALLY LUMPY — dead between ridges, hard swing crossing
                                     // a steep face = "not responsive / too much / no accel-deaccel". FEEL > visual sync.
      flexHz: t.flexHz ?? 0.6,   // slow, subtle idle flex — living glide (no flap beat)
      flexAmp: t.flexAmp ?? 0.06, // tiny idle flex
      beatHz: t.beatHz ?? 3.0,     // ~3 beats/sec held → smooth sustained climb
      beatLift: t.beatLift ?? 14,  // a held flap climbs ~8 m/s; a tap gives a felt hop
      beatThrust: t.beatThrust ?? 10, // forward thrust so a climb HOLDS airspeed instead of stalling
      beatAmp: t.beatAmp ?? 0.9,   // big visible wingbeat (rad)
      flapAsym: t.flapAsym ?? 0.3, // turn while flapping → outer wing beats harder
      flapTurn: t.flapTurn ?? 0.6, // modest yaw assist from the wing difference
      crashSpeed: t.crashSpeed ?? 16, // below this you skim; above = crash (a fast dive into the deck)
      crashBleed: t.crashBleed ?? 0.65, // a crash dumps ~65% of your airspeed
      crashTime: t.crashTime ?? 0.5,    // ~0.5 s of mushy steering after the hit
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
      multisample: { count: sampleCount },
    });
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.ubuf } }],
    });
  }

  forwardVec(): Vec3 {
    return [Math.sin(this.heading), 0, Math.cos(this.heading)];
  }

  // accumulated sim time — shared with the wind overlay so its streamlines use the SAME field state.
  get simTime(): number { return this.time; }

  integrate(dt: number, input: BirdInput): void {
    this.time += dt;
    const clamped = Math.min(dt, 1 / 20);
    const T = this.tuning;
    const STALL_FLOOR = 7; // m/s — absolute airspeed floor: lets speed DECAY into the stall regime
                           // (below minSpeed) while keeping the cubic sink term from dividing to infinity.
    const SINK_CAP = 28;   // m/s — clamp the cubic stall-sink: a deep stall drops HARD but recoverably,
                           // not a teleport-to-terrain.
    const VEL_TAU = 0.13;  // s — velocity-vector inertia time constant: how fast world velocity ramps
                           // toward the composed flight-path velocity. Larger = smoother/heavier accel
                           // & deaccel; smaller = snappier. LOWERED (0.25→0.13) for a visceral dive/swoop.

    // --- steering: mouse-x drives yaw RATE; mouse-y drives pitch ATTITUDE (holdable — the nose
    // eases toward the cursor's target angle and STAYS there; centered cursor = gentle glide trim,
    // which replaces the old hands-off auto-trim) ---
    // --- STALL (entry airspeed): below stallSpeed the wing stops flying. The nose BREAKS down
    // regardless of the held stick, yaw authority goes mushy, and the cubic sink (below) dumps the
    // bird until the dive rebuilds speed past stall → recover. Manual flight can provoke it by holding
    // the nose up; the autopilot's energy guard (nose-down at glideSpeed−5, well above stall) keeps the
    // hands-off nanny out of it, and the break itself auto-recovers if it ever clips one. ---
    const stallSpeed = T.minSpeed;                                          // wing quits below this airspeed
    const stallDepth = Math.max(0, (stallSpeed - this.speed) / stallSpeed); // 0 flying .. →1 deep stall
    const stalled = stallDepth > 0;
    const stumble = this.crashT > 0 ? 0.3 : 1; // CRASH stumble: steering degraded while recovering

    this.heading += input.yawRate * clamped * (stalled ? 0.35 : 1) * stumble; // mush in stall/crash
    let pitchGoal = Math.max(-1.0, Math.min(1.0, input.pitchTarget)); // ±~57° — steep dives/climbs
    if (stalled) {
      const breakPitch = -0.35 - 0.35 * stallDepth;   // nose drops harder the deeper the stall
      pitchGoal = Math.min(pitchGoal, breakPitch);     // can't hold the nose up past the break
    }
    this.pitch += (pitchGoal - this.pitch) * Math.min(1, clamped * (stalled ? 6 : 3.5)); // snappier break
    const targetBank = -input.yawRate * 0.5; // roll into the turn
    this.bank += (targetBank - this.bank) * Math.min(1, clamped * 4);

    // --- POWERED FLAP (climb engine): per-wing beat, NOT a unified force. A beat starts on input and
    // runs to completion (tap = one beat); while held, beats repeat (sustained climb). The two wings'
    // SUM = climb (lift + forward thrust to hold airspeed); their DIFFERENCE (from steering) = turn. ---
    const beatDur = 1 / Math.max(0.5, T.beatHz);
    if (!this.beatActive && input.flap) { this.beatActive = true; this.beatPhase = 0; }
    let beatShape = 0; // 0..1 power envelope of the downstroke this frame
    if (this.beatActive) {
      this.beatPhase += clamped / beatDur;
      beatShape = Math.max(0, Math.sin(Math.PI * Math.min(1, this.beatPhase)));
      if (this.beatPhase >= 1) {
        if (input.flap) this.beatPhase = 0;                   // held → next beat
        else { this.beatActive = false; this.beatPhase = 0; } // released → glide
      }
    }
    // steering biases the two wings: outer wing beats harder → banks/yaws into the turn.
    const steerBias = Math.max(-1, Math.min(1, input.yawRate * T.flapAsym));
    const effL = 1 + steerBias; // yaw>0 (turning right) → LEFT (outer) wing works harder
    const effR = 1 - steerBias;
    const beatSym = beatShape * (effL + effR) * 0.5;  // = beatShape when symmetric (the SUM → climb)
    const beatDiff = beatShape * (effL - effR) * 0.5; // the DIFFERENCE (→ turn)
    const flapClimb = beatSym * T.beatLift;           // m/s → target vertical velocity (climb)
    const flapThrust = beatSym * T.beatThrust;        // m/s^2 → airspeed (sustains the climb)
    this.heading += beatDiff * T.flapTurn * clamped;  // wing-difference yaw assist
    const flapBank = -beatDiff * 0.5;                 // bank toward the weaker wing (visual only)
    this.flapBeatPhase = this.beatActive ? Math.PI * this.beatPhase : 0;
    this.ampL = beatShape * T.beatAmp * effL;         // per-wing visual beat amplitude (independent)
    this.ampR = beatShape * T.beatAmp * effR;
    this.lastFlapping = this.beatActive;

    const fwd = this.forwardVec();
    const dir: Vec3 = [
      fwd[0] * Math.cos(this.pitch),
      Math.sin(this.pitch),
      fwd[2] * Math.cos(this.pitch),
    ];

    // --- airspeed energy exchange: gravity along the flight path + one-sided drag ---
    // pitch down → speed builds; pull up → speed bleeds into climb (zoom). Drag only bleeds
    // speed ABOVE trim (parasitic): a pure glider gets no free thrust back toward trim —
    // speed lost to a climb is recovered only by diving. This is the soaring energy contract.
    this.speed +=
      (-T.gravity * Math.sin(this.pitch) * T.divePower -
        T.dragK * Math.max(0, this.speed - T.glideSpeed) +
        flapThrust) * // POWERED FLAP adds forward thrust so a climb holds airspeed (no stall-out)
      clamped;

    // GLIDE, NO FLAP: no thrust input this pass. Airspeed is sustained by the dive↔zoom energy
    // exchange above and bled by drag toward trim; lift comes from glide + ridge updraft below.
    // FLOOR at STALL_FLOOR, not minSpeed — airspeed may decay BELOW minSpeed into the stall regime;
    // the floor only keeps the cubic sink finite. (minSpeed is now the stall THRESHOLD, not a hard floor.)
    this.speed = Math.max(STALL_FLOOR, Math.min(T.maxSpeed, this.speed));

    // --- wind + lift: SHARED field (wind.ts) — the SAME field the streamline overlay draws.
    // CRANKED: windGain scales the field; the lateral component visibly shoves the glide off
    // heading (you must correct), and updraft (ridge + thermal) carries it up. ---
    const [bwx, bwz] = this.stillAir
      ? ([0, 0] as [number, number])
      : windAt(this.pos[0], this.pos[2], this.time);
    const rwx = bwx * T.windGain; // RAW gained horizontal — drives RIDGE LIFT (into-slope component intact)
    const rwz = bwz * T.windGain;
    const eps = 6;
    const hC = this.terrain.sampleHeight(this.pos[0], this.pos[2]);
    const hX = this.terrain.sampleHeight(this.pos[0] + eps, this.pos[2]);
    const hZ = this.terrain.sampleHeight(this.pos[0], this.pos[2] + eps);
    const gx = (hX - hC) / eps; // uphill gradient (forward diff) — ridge lift, unchanged
    const gz = (hZ - hC) / eps;
    // DRIFT uses the terrain-DEFLECTED wind (the SAME shared fn the motes ride) so the bird drifts with the
    // VISIBLE flow, not the raw field. Central-diff gradient (e=eps) matches Wind.flowAt exactly — reuse
    // hX/hZ and add the two −e samples (2 extra sampleHeight/frame, negligible). Ridge lift below keeps the
    // RAW wind so windward soaring is unchanged (deflection sheds exactly the into-slope lift component).
    const hXm = this.terrain.sampleHeight(this.pos[0] - eps, this.pos[2]);
    const hZm = this.terrain.sampleHeight(this.pos[0], this.pos[2] - eps);
    const [wx, wz] = flowHorizontal(rwx, rwz, (hX - hXm) / (2 * eps), (hZ - hZm) / (2 * eps), T.deflect);
    this.lastWind = [wx, wz]; // telemetry (overlay + compass) shows the DRIFT the bird feels — matches the motes
    // (kept in sync via the exported updraftAt — same ridge+thermal+cap math; gx/gz reused above)
    const into = rwx * gx + rwz * gz; // RAW wind · uphill → ridge lift (undeflected, soaring intact)
    const ridge = Math.max(0, into) * T.liftGain;
    const thermal = this.stillAir ? 0 : thermalAt(this.pos[0], this.pos[2], this.time) * 1.8; // v8: stronger thermal cores (sparse but rewarding to find); NOT scaled by the horizontal-wind crank
    const updraft = Math.min(8.0, ridge + thermal); // CAP (v14: 5.5→8.0): strong ridge/thermal lift sustains a climb without launching the bird (was spiking +25 on steep windward faces)
    this.lastUpdraft = updraft;

    // --- sink: minimal at trim, mushes CUBICALLY when slow — at minSpeed full-nose-up the
    // sink exceeds sin(pitch)*speed, so a stalled climb falls instead of levitating ---
    const sink = Math.min(SINK_CAP, T.sinkRate * (T.glideSpeed / this.speed) ** 3);

    // --- compose TARGET velocity: flight path + horizontal wind drift + ridge updraft − sink ---
    const tvx = dir[0] * this.speed + wx * T.windDrift;
    const tvy = dir[1] * this.speed + updraft - sink + flapClimb; // + POWERED FLAP climb
    const tvz = dir[2] * this.speed + wz * T.windDrift;

    // --- velocity-vector INERTIA: the world velocity now carries MOMENTUM instead of being SET to
    // dir·speed every frame. Low-pass this.vel toward the target with a dt-correct factor (1−e^(−dt/τ))
    // so accel & deaccel RAMP over ~VEL_TAU rather than snapping to whatever pitch/speed/sink dictate
    // this frame — the fix for "very fast accel/deaccel". Airspeed/pitch/sink physics are untouched;
    // only the composition into world velocity is smoothed, so the soaring energy model is preserved. ---
    const aVel = 1 - Math.exp(-clamped / VEL_TAU);
    this.vel[0] += (tvx - this.vel[0]) * aVel;
    this.vel[1] += (tvy - this.vel[1]) * aVel;
    this.vel[2] += (tvz - this.vel[2]) * aVel;

    // --- BUFFETING: fast-varying turbulence so MOVING AIR is FELT (not the steady mean wind drift).
    // Three decorrelated fast oscillators sampled in (fast time + space) so the gust pattern shifts as
    // the bird flies through it — phase-locked to position, not a pure clock. Tuned felt-but-controllable.
    const bt = this.time * 2.0;                       // fast buffet clock (~rad/s base)
    const ph = this.pos[0] * 0.05 + this.pos[2] * 0.05; // spatial phase → gust texture varies in space
    const g1 = Math.sin(bt * 3.1 + ph);              // ~0.5 Hz-ish primary gust
    const g2 = Math.sin(bt * 5.7 - ph * 1.7 + 1.3);  // faster chop
    const g3 = Math.sin(bt * 1.9 + ph * 0.6 + 2.1);  // slow swell
    const gustV = this.stillAir ? 0 : 0.6 * g1 + 0.4 * g2; // vertical bob driver  (-1..1)
    const gustL = this.stillAir ? 0 : 0.6 * g3 + 0.4 * g1; // lateral shove driver (-1..1)
    // Buffet is an INSTANTANEOUS gust overlay added ON TOP of the smoothed base velocity for the position
    // step — deliberately NOT fed back through the inertia filter, so the crisp gust texture (vertical bob
    // ±~1.5 m/s, lateral shove ±~1.2 m/s perpendicular to heading) isn't low-passed into mush.
    const rightX = Math.cos(this.heading);  // heading-right unit (XZ): (cos h, -sin h)
    const rightZ = -Math.sin(this.heading);
    const shove = gustL * 1.2;
    const cvx = this.vel[0] + rightX * shove;
    const cvy = this.vel[1] + gustV * 1.5;
    const cvz = this.vel[2] + rightZ * shove;

    this.pos[0] += cvx * clamped;
    this.pos[1] += cvy * clamped;
    this.pos[2] += cvz * clamped;

    // --- terrain contact: skim gently, CRASH if you hit too hard ---
    // terrain at the NEW x,z (post-move) so a fast run into a rising slope is caught this frame.
    const hCnow = this.terrain.sampleHeight(this.pos[0], this.pos[2]);
    const floorY = hCnow + T.minClearance;
    const penetration = floorY - this.pos[1]; // >0 → dipped into the surface this frame
    if (penetration > 0) {
      this.pos[1] = floorY;
      const impactRate = penetration / clamped; // m/s closing into the surface (frame-rate independent)
      if (impactRate > T.crashSpeed && this.crashT <= 0) {
        // CRASH — hit too hard: dump most of the airspeed + start the stumble (degraded steering).
        this.speed = Math.max(STALL_FLOOR, this.speed * (1 - T.crashBleed));
        this.crashT = T.crashTime;
      }
      if (this.vel[1] < 0) this.vel[1] = 0; // ride the contour (no into-ground velocity)
    }
    if (this.crashT > 0) this.crashT -= clamped; // stumble timer counts down
    this.lastCrashing = this.crashT > 0;

    this.lastSpeed = this.speed;
    this.lastVario = cvy;
    this.lastClearance = this.pos[1] - hCnow;
    // ground-track: the ACTUAL horizontal travel direction. Wind drift makes it diverge from
    // heading — this is the felt-wind proof (overlay compares heading vs ground-track).
    this.lastGroundTrack = Math.atan2(cvx, cvz);

    // --- render-bank: steady CRAB lean into the cross-wind + buffet ROCK (visual only, no physics) ---
    // Steady DC lean: cross-wind component (wind projected onto heading-right) tips the V into the
    // breeze, like a glider crabbing. Proportional, capped, so it's a visible steady offset (not a wobble).
    const crossWind = wx * rightX + wz * rightZ;       // m/s of wind across the heading (signed)
    const crab = Math.max(-0.18, Math.min(0.18, crossWind * 0.012)); // ~±10° cap, leans with the gust DC
    // Buffet ROCK: fast roll oscillation ±~0.12 rad (~7°) so the V visibly rolls back and forth.
    const rock = this.stillAir ? 0 : (0.6 * g1 - 0.4 * g3) * 0.12;
    this.renderBank = this.bank + crab + rock + flapBank; // + flap-asymmetry bank
    if (typeof window !== "undefined") (window as any).__birdBank = this.renderBank;
  }

  // soft reset for the downhill glide: lift back to world-y `y`, restore trim airspeed and a clean
  // forward velocity. Called when the still-air glider sinks to the deck so a fly-to-target run
  // continues instead of dead-skimming the ground (this glider only ever loses altitude).
  resetAltitude(y: number): void {
    this.pos[1] = y;
    this.speed = this.tuning.glideSpeed;
    const f = this.forwardVec();
    this.vel = [f[0] * this.speed, 0, f[2] * this.speed];
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
    u[19] = this.time * this.tuning.flexHz * Math.PI * 2; // flexPhase (idle living flex)
    u[20] = this.heading;
    u[21] = this.renderBank; // steering bank + crab + buffet rock + flap-asymmetry → the V rolls/leans
    u[22] = this.tuning.flexAmp; // idle flex amplitude
    u[23] = this.flapBeatPhase;  // powered beat phase 0..PI (0 when idle)
    u[24] = this.ampL;           // LEFT wing beat amplitude (independent of right)
    u[25] = this.ampR;           // RIGHT wing beat amplitude
    u[26] = this.pitch;          // nose attitude → tilts the whole model (climb noses up, dive noses down)
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
  // v6: halved vs v5 (SPAN 18→9 etc.) — a SMALL glider dwarfed by the ~120 m ridge relief. Same V
  // shape, just smaller; the chase camera is pulled in proportionally (camera.ts followDist/Height)
  // so it stays a clear readable V while the terrain dominates the frame.
  const SPAN = 9;      // half-wingspan (m) → ~18 m tip-to-tip
  const SWEEP = 3;     // how far back the tip sits (-Z)
  const DIHEDRAL = 4.5; // tip rise (m) at full span → clear static gliding V
  const RIBBON = 1.7;  // ribbon half-width (m) — neon, not a hairline
  const BODY_LEN = 5.5; // body spine length (m)
  const BODY_W = 1.0;

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
