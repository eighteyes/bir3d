// camera.ts — ChaseCamera: ground-locked follow behind+above a target, world-up always.
// Responsibilities:
//   - Decouple from the bird's pitch/roll: camForward = horizontal(yaw only) = (sin(yaw),0,cos(yaw)).
//     The camera follows the target POSITION and HEADING only; it ignores bird pitch/roll.
//   - eye = target - camForward*followDist + worldUp*followHeight (worldUp = (0,1,0) ALWAYS).
//   - Aim DOWN at a FIXED downward angle (lookPitch) independent of the bird's altitude so the
//     ground ALWAYS fills the lower frame and the camera NEVER points at sky-only:
//     aimY = eye.y - lookAhead*tan(lookPitch).
//   - Smooth: lerp eye/lookTarget toward their goals each frame.
//   - camOffset() returns (target.x, target.z) for the terrain grid/heightfield recentering.
//   - viewMatrix(): lookAt(eye, lookTarget, worldUp); the host multiplies proj*view.

import { lookAt, lerp, type Mat4, type Vec3 } from "./mat4";

export interface ChaseParams {
  followDist?: number;
  followHeight?: number;
  lookAhead?: number;     // horizontal distance ahead the aim point sits
  lookPitch?: number;     // fixed downward look angle (rad); altitude-independent ground lock
  smooth?: number;        // 0..1 lerp factor per frame
}

export class ChaseCamera {
  target: Vec3 = [0, 0, 0];
  // forward is set by the host from the bird HEADING only (yaw); it is horizontal by construction.
  forward: Vec3 = [0, 0, 1]; // +z = north
  private eye: Vec3 = [0, 100, -120];
  private lookTarget: Vec3 = [0, 0, 200];

  followDist: number;
  followHeight: number;
  lookAhead: number;
  lookPitch: number;
  smooth: number;

  constructor(p: ChaseParams = {}) {
    this.followDist = p.followDist ?? 120;
    this.followHeight = p.followHeight ?? 55;
    this.lookAhead = p.lookAhead ?? 160;
    this.lookPitch = p.lookPitch ?? (16 * Math.PI) / 180; // ~16° down → ground fills lower ~75% of frame
    this.smooth = p.smooth ?? 0.14;
  }

  // Recompute goal eye/lookTarget (world-up, fixed downward angle) and ease toward them.
  update(): void {
    // camForward is horizontal (yaw only); flatten any stray Y and renormalize defensively.
    const fx = this.forward[0];
    const fz = this.forward[2];
    const fl = Math.hypot(fx, fz) || 1;
    const hx = fx / fl;
    const hz = fz / fl;

    const goalEye: Vec3 = [
      this.target[0] - hx * this.followDist,
      this.target[1] + this.followHeight,
      this.target[2] - hz * this.followDist,
    ];

    // Aim at a point lookAhead in front of the EYE (horizontal), dropped by a FIXED angle.
    // This is independent of the bird's altitude, so the ground stays framed no matter the pitch.
    const drop = this.lookAhead * Math.tan(this.lookPitch);
    const goalLook: Vec3 = [
      goalEye[0] + hx * this.lookAhead,
      goalEye[1] - drop,
      goalEye[2] + hz * this.lookAhead,
    ];

    this.eye = lerp(this.eye, goalEye, this.smooth);
    this.lookTarget = lerp(this.lookTarget, goalLook, this.smooth);
  }

  viewMatrix(): Mat4 {
    return lookAt(this.eye, this.lookTarget, [0, 1, 0]); // world-up ALWAYS
  }

  getEye(): Vec3 { return [this.eye[0], this.eye[1], this.eye[2]]; }
  camOffset(): [number, number] { return [this.target[0], this.target[2]]; }
}
