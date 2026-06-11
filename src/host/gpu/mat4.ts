// mat4.ts — minimal column-major 4x4 matrix + vec3 math for the WebGPU 3D scene.
// Responsibilities:
//   - perspective(fovY, aspect, near, far) with WebGPU clip-space z in [0,1].
//   - lookAt(eye, target, up) right-handed view matrix.
//   - multiply(a, b) = a*b (column-major), identity().
//   - vec3 ops: sub, add, scale, normalize, cross, dot, length used by the camera.
// Matrices are Float32Array(16) in column-major order (col0 = m[0..3]) — upload-ready for WGSL.

export type Mat4 = Float32Array;
export type Vec3 = [number, number, number];

export function identity(): Mat4 {
  const m = new Float32Array(16);
  m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
  return m;
}

// WebGPU perspective: clip-space z in [0,1] (NOT WebGL's [-1,1]). Column-major.
export function perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2);
  const m = new Float32Array(16);
  // col0
  m[0] = f / aspect;
  // col1
  m[5] = f;
  // col2
  m[10] = far / (near - far);
  m[11] = -1;
  // col3
  m[14] = (near * far) / (near - far);
  return m;
}

// Right-handed lookAt. Forward = normalize(target - eye); camera looks down -z in view space.
export function lookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
  const z = normalize(sub(eye, target)); // +z points back toward the eye (RH)
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  const m = new Float32Array(16);
  // Column-major: columns are basis vectors; translation row uses -dot(axis, eye).
  m[0] = x[0]; m[1] = y[0]; m[2] = z[0]; m[3] = 0;
  m[4] = x[1]; m[5] = y[1]; m[6] = z[1]; m[7] = 0;
  m[8] = x[2]; m[9] = y[2]; m[10] = z[2]; m[11] = 0;
  m[12] = -dot(x, eye); m[13] = -dot(y, eye); m[14] = -dot(z, eye); m[15] = 1;
  return m;
}

// a*b in column-major convention: out = a * b (apply b first, then a).
export function multiply(a: Mat4, b: Mat4): Mat4 {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] =
        a[0 * 4 + r]! * b[c * 4 + 0]! +
        a[1 * 4 + r]! * b[c * 4 + 1]! +
        a[2 * 4 + r]! * b[c * 4 + 2]! +
        a[3 * 4 + r]! * b[c * 4 + 3]!;
    }
  }
  return o;
}

export function sub(a: Vec3, b: Vec3): Vec3 { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
export function add(a: Vec3, b: Vec3): Vec3 { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
export function scale(a: Vec3, s: number): Vec3 { return [a[0] * s, a[1] * s, a[2] * s]; }
export function dot(a: Vec3, b: Vec3): number { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
export function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
export function length(a: Vec3): number { return Math.hypot(a[0], a[1], a[2]); }
export function normalize(a: Vec3): Vec3 {
  const l = length(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}
export function lerp(a: Vec3, b: Vec3, t: number): Vec3 {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}
