// marker.ts — GroundMarker: altitude plumb-line + ground diamond under the bird (depth cue).
// Responsibilities:
//   - Own a tiny static line-list vertex buffer: 2 verts for the vertical drop line (kind 0,
//     t 0→1) + 8 verts for a unit ground diamond loop (kind 1). World placement happens in
//     marker.wgsl from the per-frame uniform (birdPos, groundY, time, height).
//   - Create the render pipeline (marker.wgsl): line-list, additive blend, depth-tested with
//     write OFF so ridges occlude the marker (extra parallax cue) without disturbing depth.
//   - draw(enc, colorView, depthView, viewProj, birdPos, groundY, time): LOADS color+depth —
//     terrain/bird must already be drawn this encoder.

type Vec3 = [number, number, number];

const FLOATS_PER_VERT = 6; // pos.xyz + attr.xyz (kind, t, unused)
const UNIFORM_BYTES = 96;  // mat4(64) + birdPos+groundY(16) + time,height,pad,pad(16)

export class GroundMarker {
  private vbuf: GPUBuffer;
  private vertexCount: number;
  private ubuf: GPUBuffer;
  private uniformHost = new ArrayBuffer(UNIFORM_BYTES);
  private uniformF32 = new Float32Array(this.uniformHost);
  private pipeline: GPURenderPipeline;
  private bindGroup: GPUBindGroup;

  constructor(private device: GPUDevice, shader: string, colorFormat: GPUTextureFormat, sampleCount = 1) {
    const verts: number[] = [];
    const v = (pos: Vec3, kind: number, t: number) =>
      verts.push(pos[0], pos[1], pos[2], kind, t, 0);

    // drop line (kind 0): geometry is a placeholder — the shader positions it from the uniform.
    v([0, 0, 0], 0, 0);
    v([0, 0, 0], 0, 1);
    // ground diamond (kind 1): unit cross loop in XZ, scaled in the shader.
    const ring: [number, number][] = [[1, 0], [0, 1], [-1, 0], [0, -1]];
    for (let i = 0; i < 4; i++) {
      const a = ring[i]!, b = ring[(i + 1) % 4]!;
      v([a[0], 0, a[1]], 1, 0);
      v([b[0], 0, b[1]], 1, 0);
    }
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
            blend: {
              color: { srcFactor: "one", dstFactor: "one", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "line-list" },
      // depth-tested so ridges occlude the marker; write OFF — it is an overlay cue, not geometry.
      depthStencil: { depthWriteEnabled: false, depthCompare: "less", format: "depth24plus" },
      multisample: { count: sampleCount },
    });
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.ubuf } }],
    });
  }

  draw(
    encoder: GPUCommandEncoder,
    colorView: GPUTextureView,
    depthView: GPUTextureView,
    viewProj: Float32Array,
    birdPos: Vec3,
    groundY: number,
    time: number,
    resolveTarget?: GPUTextureView // MSAA resolve dest (the swapchain view); this is the LAST pass of the frame
  ): void {
    const u = this.uniformF32;
    u.set(viewProj, 0);
    u[16] = birdPos[0]; u[17] = birdPos[1]; u[18] = birdPos[2];
    u[19] = groundY;
    u[20] = time;
    u[21] = Math.max(0, birdPos[1] - groundY);
    this.device.queue.writeBuffer(this.ubuf, 0, this.uniformHost);

    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: colorView, resolveTarget, loadOp: "load", storeOp: "store" }],
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
