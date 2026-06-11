// camera.ts — ChaseCamera: smooth deadzone follow behind+above a target, looking forward.
// Responsibilities:
//   - Hold a target world position + forward heading; per FACTS compute
//     eye = target - forward*followDist + worldUp*followHeight,
//     lookAt = target + forward*lookAhead (worldUp = (0,1,0)).
//   - Smooth: lerp eye/lookTarget toward their goals each frame (deadzone lets the subject drift).
//   - For THIS task: autoAdvance(dt) walks the target forward over the terrain so the shot shows
//     depth/motion. camOffset() returns (target.x, target.z) for the terrain grid recentering.
//   - viewMatrix(): lookAt(eye, lookTarget, up); the host multiplies proj*view.

import { lookAt, lerp, type Mat4, type Vec3 } from "./mat4";

export interface ChaseParams {
  followDist?: number;
  followHeight?: number;
  lookAhead?: number;
  smooth?: number;       // 0..1 lerp factor per frame
  groundHeight?: number; // min eye height above y=0 (kept above terrain relief)
}

export class ChaseCamera {
  target: Vec3 = [0, 0, 0];
  forward: Vec3 = [0, 0, 1]; // +z = north
  private eye: Vec3 = [0, 100, -120];
  private lookTarget: Vec3 = [0, 0, 200];

  followDist: number;
  followHeight: number;
  lookAhead: number;
  smooth: number;
  groundHeight: number;
  speed = 55; // m/s forward auto-advance for the shot

  constructor(p: ChaseParams = {}) {
    this.followDist = p.followDist ?? 120;
    this.followHeight = p.followHeight ?? 70;
    this.lookAhead = p.lookAhead ?? 200;
    this.smooth = p.smooth ?? 0.12;
    this.groundHeight = p.groundHeight ?? 45;
  }

  // Walk the target forward over the terrain (this-task auto motion).
  autoAdvance(dt: number): void {
    this.target = [
      this.target[0] + this.forward[0] * this.speed * dt,
      this.target[1],
      this.target[2] + this.forward[2] * this.speed * dt,
    ];
  }

  // Recompute goal eye/lookTarget and ease toward them.
  update(): void {
    const up: Vec3 = [0, 1, 0];
    const goalEye: Vec3 = [
      this.target[0] - this.forward[0] * this.followDist,
      this.target[1] + this.followHeight,
      this.target[2] - this.forward[2] * this.followDist,
    ];
    const goalLook: Vec3 = [
      this.target[0] + this.forward[0] * this.lookAhead,
      this.target[1],
      this.target[2] + this.forward[2] * this.lookAhead,
    ];
    this.eye = lerp(this.eye, goalEye, this.smooth);
    this.lookTarget = lerp(this.lookTarget, goalLook, this.smooth);
    if (this.eye[1] < this.groundHeight) this.eye[1] = this.groundHeight;
    void up;
  }

  viewMatrix(): Mat4 {
    return lookAt(this.eye, this.lookTarget, [0, 1, 0]);
  }

  getEye(): Vec3 { return [this.eye[0], this.eye[1], this.eye[2]]; }
  camOffset(): [number, number] { return [this.target[0], this.target[2]]; }
}
