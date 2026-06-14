// bloom.ts — Bloom post-process: HDR scene → bright-pass → separable Gaussian blur → composite.
// Responsibilities:
//   - Own the bloom-chain textures (bright-pass + two ping-pong blur targets) at a DOWNSAMPLED
//     resolution (half-res by default — wider soft glow AND cheaper than full-res blur).
//   - Build the threshold / blur / composite pipelines (fullscreen-triangle passes) + a linear
//     sampler + per-pass uniform buffers.
//   - apply(encoder, sceneView, swapchainView): run threshold → H-blur → V-blur (optionally
//     repeated for a wider glow) → composite scene+bloom with Reinhard tone-map to the swapchain.
//   - resize(w,h): recreate the downsampled textures + their bind groups (call from the host's
//     resize handler alongside the scene/depth targets).
//   - setTuning({threshold,knee,intensity,exposure}): live re-tune the glow without rebuilds.

export interface BloomParams {
  threshold?: number; // luminance above which pixels bloom
  knee?: number;      // soft-knee width above threshold
  intensity?: number; // bloom add weight in the composite
  exposure?: number;  // pre-tonemap scene exposure
  downsample?: number; // bloom-chain resolution divisor (2 = half-res)
  blurPasses?: number; // number of H+V blur iterations (more = wider, costlier)
}

export class Bloom {
  private device: GPUDevice;
  private swapFormat: GPUTextureFormat;
  private hdrFormat: GPUTextureFormat = "rgba16float";

  private threshold: number;
  private knee: number;
  private intensity: number;
  private exposure: number;
  private downsample: number;
  private blurPasses: number;

  private sampler: GPUSampler;
  private thresholdPipeline: GPURenderPipeline;
  private blurPipeline: GPURenderPipeline;
  private compositePipeline: GPURenderPipeline;

  // uniforms
  private thresholdUbuf: GPUBuffer;
  private blurUbuf: GPUBuffer;   // reused per blur pass (texelStep rewritten each pass)
  private compositeUbuf: GPUBuffer;

  // downsampled targets (recreated on resize)
  private bw = 1;
  private bh = 1;
  private brightTex!: GPUTexture; // bright-pass output
  private pingTex!: GPUTexture;   // blur ping
  private pongTex!: GPUTexture;   // blur pong
  private brightView!: GPUTextureView;
  private pingView!: GPUTextureView;
  private pongView!: GPUTextureView;

  constructor(device: GPUDevice, swapFormat: GPUTextureFormat, shaders: {
    threshold: string;
    blur: string;
    composite: string;
  }, p: BloomParams = {}) {
    this.device = device;
    this.swapFormat = swapFormat;
    this.threshold = p.threshold ?? 0.9;
    this.knee = p.knee ?? 0.4;
    this.intensity = p.intensity ?? 1.0;
    this.exposure = p.exposure ?? 1.0;
    this.downsample = Math.max(1, p.downsample ?? 2);
    this.blurPasses = Math.max(1, p.blurPasses ?? 2);

    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    const thresholdMod = device.createShaderModule({ code: shaders.threshold });
    const blurMod = device.createShaderModule({ code: shaders.blur });
    const compositeMod = device.createShaderModule({ code: shaders.composite });

    this.thresholdPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: thresholdMod, entryPoint: "vs" },
      fragment: { module: thresholdMod, entryPoint: "fs", targets: [{ format: this.hdrFormat }] },
      primitive: { topology: "triangle-list" },
    });
    this.blurPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: blurMod, entryPoint: "vs" },
      fragment: { module: blurMod, entryPoint: "fs", targets: [{ format: this.hdrFormat }] },
      primitive: { topology: "triangle-list" },
    });
    this.compositePipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: compositeMod, entryPoint: "vs" },
      fragment: { module: compositeMod, entryPoint: "fs", targets: [{ format: this.swapFormat }] },
      primitive: { topology: "triangle-list" },
    });

    this.thresholdUbuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.blurUbuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.compositeUbuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  // (re)create the downsampled bloom-chain textures for a given full-res framebuffer size.
  resize(fullW: number, fullH: number): void {
    this.bw = Math.max(1, Math.floor(fullW / this.downsample));
    this.bh = Math.max(1, Math.floor(fullH / this.downsample));
    const mk = () => this.device.createTexture({
      size: [this.bw, this.bh],
      format: this.hdrFormat,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.brightTex?.destroy();
    this.pingTex?.destroy();
    this.pongTex?.destroy();
    this.brightTex = mk();
    this.pingTex = mk();
    this.pongTex = mk();
    this.brightView = this.brightTex.createView();
    this.pingView = this.pingTex.createView();
    this.pongView = this.pongTex.createView();
  }

  setTuning(p: BloomParams): void {
    if (p.threshold !== undefined) this.threshold = p.threshold;
    if (p.knee !== undefined) this.knee = p.knee;
    if (p.intensity !== undefined) this.intensity = p.intensity;
    if (p.exposure !== undefined) this.exposure = p.exposure;
  }

  // Run the full bloom chain. sceneView = single-sample HDR scene; swapchainView = canvas target.
  apply(encoder: GPUCommandEncoder, sceneView: GPUTextureView, swapchainView: GPUTextureView): void {
    const dev = this.device;

    // --- bright-pass threshold: scene (full-res) → brightTex (half-res, linear downsample). ---
    dev.queue.writeBuffer(this.thresholdUbuf, 0, new Float32Array([this.threshold, this.knee, 0, 0]));
    const thBind = dev.createBindGroup({
      layout: this.thresholdPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sceneView },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.thresholdUbuf } },
      ],
    });
    this.pass(encoder, this.thresholdPipeline, thBind, this.brightView);

    // --- separable Gaussian blur (ping-pong): H then V, blurPasses iterations. ---
    // first H reads brightTex → ping; thereafter read the last written target.
    const dx = 1 / this.bw;
    const dy = 1 / this.bh;
    let src = this.brightView;
    for (let i = 0; i < this.blurPasses; i++) {
      // horizontal: src → ping
      dev.queue.writeBuffer(this.blurUbuf, 0, new Float32Array([dx, 0, 0, 0]));
      const hBind = dev.createBindGroup({
        layout: this.blurPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: src },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: { buffer: this.blurUbuf } },
        ],
      });
      this.pass(encoder, this.blurPipeline, hBind, this.pingView);
      // vertical: ping → pong
      dev.queue.writeBuffer(this.blurUbuf, 0, new Float32Array([0, dy, 0, 0]));
      const vBind = dev.createBindGroup({
        layout: this.blurPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.pingView },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: { buffer: this.blurUbuf } },
        ],
      });
      this.pass(encoder, this.blurPipeline, vBind, this.pongView);
      src = this.pongView; // next iteration widens the existing blur
    }

    // --- composite: scene + bloom*intensity, Reinhard tone-map → swapchain. ---
    dev.queue.writeBuffer(this.compositeUbuf, 0, new Float32Array([this.intensity, this.exposure, 0, 0]));
    const coBind = dev.createBindGroup({
      layout: this.compositePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sceneView },
        { binding: 1, resource: src }, // last blurred target
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: { buffer: this.compositeUbuf } },
      ],
    });
    this.pass(encoder, this.compositePipeline, coBind, swapchainView);
  }

  private pass(encoder: GPUCommandEncoder, pipeline: GPURenderPipeline, bind: GPUBindGroup, target: GPUTextureView): void {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: target, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.draw(3);
    pass.end();
  }
}
