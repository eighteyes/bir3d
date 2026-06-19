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

  // v17 CAMERA-TERRAIN COLLISION: the bird is clamped above terrain, but the chase eye sits BEHIND+ABOVE it
  // and — especially with the taller RELIEF — can end up INSIDE a peak, which renders the scene fully black
  // (the user's "black when I run into a mountain"). Wired from bird-main; null = no collision handling.
  terrainHeight: ((x: number, z: number) => number) | null = null;
  eyeMargin = 10; // meters the eye is kept clear of terrain (boom pull-in + hard floor)

  constructor(p: ChaseParams = {}) {
    // v6: pulled in proportionally to the halved bird (SPAN 18→9) so it stays a readable V while the
    // big ridges dominate. followHeight is the real "terrain looms" lever (lower eye → ridges rise
    // against the sky); followDist holds the bird's screen size. lookAhead is direction-only (lookAt
    // normalizes), so it doesn't change framing — left at 160.
    this.followDist = p.followDist ?? 60;
    this.followHeight = p.followHeight ?? 28;
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

    // CAMERA-TERRAIN COLLISION (v17): keep the eye out of the mountains so the scene never goes black.
    // (1) PULL-IN: march the boom from the target out to goalEye; if terrain blocks the view partway, plant
    //     the eye at the last clear point (a tighter over-the-shoulder shot, but the bird stays visible).
    // (2) FLOOR: never let the eye sit below terrain at its own XZ (backstop for the straight-down boom).
    if (this.terrainHeight) {
      const m = this.eyeMargin;
      const dx = goalEye[0] - this.target[0];
      const dy = goalEye[1] - this.target[1];
      const dz = goalEye[2] - this.target[2];
      const steps = 12;
      let clear = 1;
      for (let s = 1; s <= steps; s++) {
        const f = s / steps;
        const px = this.target[0] + dx * f, py = this.target[1] + dy * f, pz = this.target[2] + dz * f;
        if (py < this.terrainHeight(px, pz) + m) { clear = (s - 1) / steps; break; }
      }
      if (clear < 1) {
        goalEye[0] = this.target[0] + dx * clear;
        goalEye[1] = this.target[1] + dy * clear;
        goalEye[2] = this.target[2] + dz * clear;
      }
      const minY = this.terrainHeight(goalEye[0], goalEye[2]) + m;
      if (goalEye[1] < minY) goalEye[1] = minY;
    }

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
    // post-lerp backstop: the lerp can dwell inside terrain for a few frames while easing out to a freshly
    // pulled-in goal — clamp the live eye above terrain at its XZ each frame so no single frame renders black.
    if (this.terrainHeight) {
      const minY = this.terrainHeight(this.eye[0], this.eye[2]) + this.eyeMargin;
      if (this.eye[1] < minY) this.eye[1] = minY;
    }
  }

  viewMatrix(): Mat4 {
    return lookAt(this.eye, this.lookTarget, [0, 1, 0]); // world-up ALWAYS
  }

  getEye(): Vec3 { return [this.eye[0], this.eye[1], this.eye[2]]; }
  camOffset(): [number, number] { return [this.target[0], this.target[2]]; }

  // Horizontal camera basis used to build the CAMERA-RELATIVE terrain rows. Derived from the
  // SMOOTHED eye→lookTarget (the same vectors the view matrix is built from) so the rows track the
  // lerped view, never the instantaneous bird heading — otherwise a hard turn shows residual skew.
  // groundPos() is the eye projected to the ground plane; rows are built ahead of it.
  groundPos(): [number, number] { return [this.eye[0], this.eye[2]]; }
  forwardHoriz(): [number, number] {
    const dx = this.lookTarget[0] - this.eye[0];
    const dz = this.lookTarget[2] - this.eye[2];
    const l = Math.hypot(dx, dz) || 1;
    return [dx / l, dz / l];
  }
  // right = forward × worldUp, kept horizontal. With up=(0,1,0): right = (fz, -fx) for a
  // right-handed cross (fwd × up).x = fy*upz - fz*upy = -fz ... use lookAt convention: in a
  // right-handed lookAt, camera-right = normalize(cross(forward, up)); here forward is the look dir.
  rightHoriz(): [number, number] {
    const [fx, fz] = this.forwardHoriz();
    // cross(forward(fx,0,fz), up(0,1,0)) = (0*0 - fz*1, fz*0 - fx*0, fx*1 - 0*0) = (-fz, 0, fx)
    return [-fz, fx];
  }
}
