// target.ts — Target: a navigable flight waypoint rendered as a billboarded beam of light.
// Responsibilities:
//   - Hold a world waypoint {x, z, groundY (terrain height at x,z)}; place it ahead of the bird via
//     respawn(fromX, fromZ, headingRad) — a randomized distance/bearing cone so the player must steer.
//   - Report navigation state: distanceTo(birdPos) and checkReached(birdPos, radius) — horizontal only,
//     altitude-agnostic (the downhill glider may pass above or below the beam).
//   - Own a static unit-quad vertex buffer + uniform + render pipeline (target.wgsl); draw() records one
//     camera-facing beam pass that LOADS color+depth and draws always-on-top (depthCompare "always") so
//     the waypoint stays visible behind ridges for reliable navigation.

type Vec3 = [number, number, number];
type HeightFn = (x: number, z: number) => number;

const FLOATS_PER_VERT = 3;  // corner.xy + pad
const UNIFORM_BYTES = 112;  // mat4(64) + base+height(16) + right+halfWidth(16) + color+time(16)

const BEAM_HEIGHT = 260;          // m — tall enough to read at distance
const BEAM_HALF_WIDTH = 14;       // m — beam thickness
const SPAWN_MIN = 700;            // m — nearest a fresh target spawns ahead
const SPAWN_MAX = 1000;           // m — farthest
const SPAWN_SPREAD = 0.7;         // rad — total lateral cone (±0.35) so the player turns toward it
const BEAM_COLOR: Vec3 = [1.6, 1.0, 0.35]; // HDR amber (>1 so the bloom picks it up) — warm, contrasts cyan

export class Target {
  x = 0;
  z = 0;
  groundY = 0;

  private vbuf: GPUBuffer;
  private vertexCount: number;
  private ubuf: GPUBuffer;
  private uniformHost = new ArrayBuffer(UNIFORM_BYTES);
  private uniformF32 = new Float32Array(this.uniformHost);
  private pipeline: GPURenderPipeline;
  private bindGroup: GPUBindGroup;

  constructor(
    private device: GPUDevice,
    shader: string,
    colorFormat: GPUTextureFormat,
    private sampleHeight: HeightFn,
    sampleCount = 1
  ) {
    // unit quad (two tris): corner = (cx in {-1,1}, cy in {0,1}); the shader expands it to world space.
    const corners: [number, number][] = [[-1, 0], [1, 0], [-1, 1], [-1, 1], [1, 0], [1, 1]];
    const verts: number[] = [];
    for (const [cx, cy] of corners) verts.push(cx, cy, 0);
    this.vertexCount = verts.length / FLOATS_PER_VERT;
    const mesh = new Float32Array(verts);
    this.vbuf = device.createBuffer({
      size: mesh.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.vbuf, 0, mesh);

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
            attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
          },
        ],
      },
      fragment: {
        module,
        entryPoint: "fs",
        targets: [
          {
            format: colorFormat,
            // additive → the amber beam blooms warm over the dark scene.
            blend: {
              color: { srcFactor: "one", dstFactor: "one", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      // ALWAYS-on-top: depthCompare "always" + write off → the waypoint stays visible behind ridges so
      // navigation is reliable (a target you can't see is a target you can't fly to).
      depthStencil: { depthWriteEnabled: false, depthCompare: "always", format: "depth24plus" },
      multisample: { count: sampleCount },
    });
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.ubuf } }],
    });

    this.respawn(0, 0, 0); // first waypoint dead ahead of the origin
  }

  // place a fresh waypoint ahead of (fromX,fromZ) along `headingRad`, within a randomized
  // distance/bearing cone so the player must actively steer toward it.
  respawn(fromX: number, fromZ: number, headingRad: number): void {
    const dist = SPAWN_MIN + Math.random() * (SPAWN_MAX - SPAWN_MIN);
    const ang = headingRad + (Math.random() - 0.5) * SPAWN_SPREAD;
    this.x = fromX + Math.sin(ang) * dist;
    this.z = fromZ + Math.cos(ang) * dist;
    this.groundY = this.sampleHeight(this.x, this.z);
  }

  distanceTo(birdPos: Vec3): number {
    return Math.hypot(this.x - birdPos[0], this.z - birdPos[2]);
  }

  checkReached(birdPos: Vec3, radius: number): boolean {
    return this.distanceTo(birdPos) < radius;
  }

  draw(
    encoder: GPUCommandEncoder,
    colorView: GPUTextureView,
    depthView: GPUTextureView,
    viewProj: Float32Array,
    eye: Vec3,
    time: number
  ): void {
    // cylindrical billboard: rightAxis = horizontal perpendicular to (eye → base) so the flat beam
    // always faces the camera. right = normalize(cross(up,(dx,_,dz))) = normalize(dz, 0, -dx).
    const dx = this.x - eye[0];
    const dz = this.z - eye[2];
    const len = Math.hypot(dx, dz) || 1;
    const rx = dz / len;
    const rz = -dx / len;

    const u = this.uniformF32;
    u.set(viewProj, 0);
    u[16] = this.x; u[17] = this.groundY; u[18] = this.z;
    u[19] = BEAM_HEIGHT;
    u[20] = rx; u[21] = 0; u[22] = rz;
    u[23] = BEAM_HALF_WIDTH;
    u[24] = BEAM_COLOR[0]; u[25] = BEAM_COLOR[1]; u[26] = BEAM_COLOR[2];
    u[27] = time;
    this.device.queue.writeBuffer(this.ubuf, 0, this.uniformHost);

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
