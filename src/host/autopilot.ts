// autopilot.ts — AutoPilot: hands-off flight controller. Emits BirdInput; physics untouched.
// Responsibilities:
//   - POLICY "straight" (bird-vs-wind eval): hold a locked heading at trim glide — no lift-seeking,
//     no orbiting; only ground-avoid (which re-locks to the escape heading) and a stall nose-down
//     deviate. POLICY "soar": full lift-hunting behavior below.
//   - Each frame, SENSE the air via the exported updraftAt (the exact field the physics applies):
//     updraft here + at 8 compass probes (radius ~140 m), terrain look-ahead along velocity.
//   - DECIDE a mode (priority order): AVOID (keep off the ground — hard rule) > SOAR (in lift:
//     orbit to stay in it) > ENERGY (slow: nose down, rebuild airspeed before anything else) >
//     CLIMB→LIFT / DESCEND (altitude band keeping) > CRUISE→LIFT (steer toward best probe).
//   - COMMAND smoothly: eased pitchTarget (attitude) + yawRate toward the desired heading —
//     the same BirdInput contract the mouse used, so controls can swap back in later.
//   - Expose `mode` for the overlay and window.__autoMode; log telemetry every ~2 s for browser logs.

import type { Bird3D, BirdInput } from "./gpu/bird3d";
import { updraftAt } from "./gpu/bird3d";
import type { TerrainEKG } from "./gpu/terrain";

const PROBE_R = 140;      // m — how far out the 8 lift probes sample
const LOOKAHEAD_S = 2.5;  // s — terrain collision horizon along current velocity
const BAND_LO = 90;       // m — climb below this clearance
const BAND_HI = 260;      // m — descend above this clearance
const AVOID_CLR = 45;     // m — hard floor guard: current clearance
const AVOID_AHEAD = 35;   // m — hard floor guard: predicted clearance at the horizon

export class AutoPilot {
  mode = "CRUISE";
  private pitchCmd = -0.03;
  private yawCmd = 0;
  private logT = 0;

  // "straight": hold a locked heading + trim glide (no lift-seeking, no orbiting) — the bird-vs-wind
  // eval policy; only AVOID/ENERGY deviate. "soar": full lift-hunting behavior.
  constructor(private terrain: TerrainEKG, public policy: "straight" | "soar" = "soar") {}
  private lockedHeading: number | null = null;

  update(bird: Bird3D, dt: number): BirdInput {
    const T = bird.tuning;
    const [px, py, pz] = bird.pos;
    const t = bird.simTime;
    const clr = bird.lastClearance;
    const spd = bird.speed;
    if (this.lockedHeading === null) this.lockedHeading = bird.heading;

    // --- sense: best lift among 8 compass probes + lift here vs current sink ---
    let bestAng = bird.heading;
    let bestU = -1;
    if (this.policy === "soar") {
      for (let i = 0; i < 8; i++) {
        const a = (i / 8) * Math.PI * 2;
        const u = updraftAt(px + Math.sin(a) * PROBE_R, pz + Math.cos(a) * PROBE_R, t, this.terrain, T);
        if (u > bestU) { bestU = u; bestAng = a; }
      }
    }
    const here = updraftAt(px, pz, t, this.terrain, T);
    const sinkNow = T.sinkRate * (T.glideSpeed / spd) ** 3;

    // --- terrain look-ahead along current velocity (the keep-off-the-ground guard) ---
    const ax = px + bird.vel[0] * LOOKAHEAD_S;
    const az = pz + bird.vel[2] * LOOKAHEAD_S;
    const aheadClr = py + bird.vel[1] * LOOKAHEAD_S - this.terrain.sampleHeight(ax, az);

    let pitchGoal: number;
    let headingGoal: number;
    if (clr < AVOID_CLR || aheadClr < AVOID_AHEAD) {
      this.mode = "AVOID";
      // climb hard only if there is airspeed to spend; otherwise keep the nose flying.
      pitchGoal = spd > T.minSpeed + 5 ? 0.32 : -0.08;
      // turn toward the lower shoulder (left vs right diagonal terrain sample).
      const hl = this.terrain.sampleHeight(
        px + Math.sin(bird.heading - 0.8) * 160, pz + Math.cos(bird.heading - 0.8) * 160);
      const hr = this.terrain.sampleHeight(
        px + Math.sin(bird.heading + 0.8) * 160, pz + Math.cos(bird.heading + 0.8) * 160);
      headingGoal = bird.heading + (hl < hr ? -0.8 : 0.8);
      this.lockedHeading = headingGoal; // the escape heading becomes the new straight line
    } else if (this.policy === "straight") {
      // STRAIGHT-LINE EVAL: hold the locked heading at trim glide. No lift-seeking, no orbiting —
      // every deviation you observe is the WIND acting on the bird, not the pilot. Only the AVOID
      // branch above (close to the ground) and a stall-guard nose-down may interfere.
      this.mode = spd < T.glideSpeed - 5 ? "ENERGY" : "STRAIGHT";
      pitchGoal = spd < T.glideSpeed - 5 ? -0.22 : -0.03;
      headingGoal = this.lockedHeading;
    } else if (here > sinkNow + 0.4) {
      this.mode = "SOAR";
      // in net lift: orbit gently to stay inside it; shallow climb attitude rides it up.
      pitchGoal = clr > BAND_HI ? -0.12 : 0.04;
      headingGoal = bird.heading + 0.5;
    } else if (spd < T.glideSpeed - 5) {
      this.mode = "ENERGY";
      pitchGoal = -0.22; // airspeed first — a slow glider can do nothing else safely
      headingGoal = bestAng;
    } else if (clr < BAND_LO) {
      this.mode = "CLIMB";
      pitchGoal = 0.1;
      headingGoal = bestAng;
    } else if (clr > BAND_HI) {
      this.mode = "DESCEND";
      pitchGoal = -0.15;
      headingGoal = bestAng;
    } else {
      this.mode = "CRUISE";
      pitchGoal = -0.03;
      headingGoal = bestAng;
    }

    // --- smooth commands: eased attitude, wrapped + clamped turn rate ---
    let dAng = headingGoal - bird.heading;
    dAng = Math.atan2(Math.sin(dAng), Math.cos(dAng));
    const yawGoal = Math.max(-0.9, Math.min(0.9, dAng * 1.4));
    this.pitchCmd += (pitchGoal - this.pitchCmd) * Math.min(1, dt * 4);
    this.yawCmd += (yawGoal - this.yawCmd) * Math.min(1, dt * 6);

    // telemetry for browser logs (~every 2 s)
    this.logT += dt;
    if (this.logT > 2) {
      this.logT = 0;
      console.log(
        `[auto] ${this.mode} clr=${clr.toFixed(0)}m spd=${spd.toFixed(1)} ` +
        `vario=${bird.lastVario.toFixed(1)} lift=${here.toFixed(1)} bestProbe=${bestU.toFixed(1)}`
      );
    }
    return { yawRate: this.yawCmd, pitchTarget: this.pitchCmd };
  }
}
