// trees-live.spec.ts — Confirm trees in the LIVE config: (1) the running app reports a positive tree
// count at spawn, (2) the Trees pipeline renders green in the EXACT live render config (rgba16float +
// MSAA 4 + resolve + depth24plus), closing the earlier rgba8/MSAA-off proof gap.
import { test, expect } from "@playwright/test";

test("live app generates trees + renders green in MSAA4/rgba16float config", async ({ page }) => {
  await page.goto("/index-bird.html");
  await page.waitForFunction(() => (window as any).__birdBooted === true, { timeout: 15000 });
  await page.waitForTimeout(1500); // let the first rebuild run at spawn

  const spawnCount = await page.evaluate(() => (window as any).__trees?.treeCount ?? -1);

  const render = await page.evaluate(async () => {
    const { acquireDevice } = await import("/src/host/gpu/device.ts");
    const { Trees } = await import("/src/host/gpu/trees.ts");
    const { TerrainEKG } = await import("/src/host/gpu/terrain.ts");
    const { perspective, lookAt, multiply } = await import("/src/host/gpu/mat4.ts");
    const shader = await (await fetch("/src/host/shaders/trees.wgsl")).text();
    const groundSrc = await (await fetch("/src/host/shaders/trees_ground.wgsl")).text();
    const tSrc = await (await fetch("/src/host/shaders/terrain_ekg.wgsl")).text();
    const { device } = await acquireDevice();

    const W = 512, H = 512, fmt: GPUTextureFormat = "rgba16float", SAMPLES = 4;
    const SKY: [number, number, number] = [0.01, 0.012, 0.03];
    // real terrain: the height source for trees AND for framing the camera (trees anchor to its fBm).
    const terrain = new TerrainEKG(device, tSrc, fmt, { rows: 512, cols: 1536, sampleCount: SAMPLES, rowSpacing: 4.5, nearDenseDepth: 500, farSpread: 220, rowStart: -150, halfWidth: 2400, maxDist: 2850, baseline: -300, fogColor: SKY, fogDensity: 0.5 / 1100 });

    device.pushErrorScope("validation");
    const trees = new Trees(device, shader, groundSrc, fmt, (x, z) => terrain.sampleHeight(x, z), SAMPLES); // EXACT live params (rgba16float, MSAA 4)
    const createErr = await device.popErrorScope();
    if (createErr) return { gpuError: "CREATE: " + createErr.message, greenDom: -1, lit: -1 };

    const msaa = device.createTexture({ size: [W, H], format: fmt, sampleCount: SAMPLES, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const resolve = device.createTexture({ size: [W, H], format: fmt, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const depth = device.createTexture({ size: [W, H], format: "depth24plus", sampleCount: SAMPLES, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const msaaV = msaa.createView(), resolveV = resolve.createView(), depthV = depth.createView();

    const fwd: [number, number] = [0, 1];
    const half = (h: number) => {
      const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
      if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
      if (e === 31) return NaN;
      return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
    };
    const bpr = Math.ceil((W * 8) / 256) * 256; // rgba16 = 8 bytes/texel
    const buf = device.createBuffer({ size: bpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    // sweep several ground positions — tight clustering means one fixed spot can land on a clearing; this
    // asserts the forest renders green in at least one. Each: eye above terrain, 28° down chase view.
    const spots: [number, number][] = [[0, 0], [420, 320], [-380, 520], [640, -240], [-560, -420], [220, 880]];
    let greenDom = 0, treeCount = 0, gpuError: string | null = null;
    for (const [gx, gz] of spots) {
      const eye: [number, number, number] = [gx, terrain.sampleHeight(gx, gz + 120) + 70, gz - 40];
      const view = lookAt(eye, [eye[0], eye[1] - 320 * Math.tan((28 * Math.PI) / 180), eye[2] + 320], [0, 1, 0]);
      const viewProj = new Float32Array(multiply(perspective((76 * Math.PI) / 180, W / H, 1, 6000), view));
      device.pushErrorScope("validation");
      const enc = device.createCommandEncoder();
      terrain.draw(enc, msaaV, depthV, viewProj, [eye[0], eye[2]], fwd, [-fwd[1], fwd[0]], eye, { r: SKY[0], g: SKY[1], b: SKY[2], a: 1 });
      trees.draw(enc, msaaV, depthV, viewProj, [eye[0], eye[2]], eye, 0);
      enc.beginRenderPass({ colorAttachments: [{ view: msaaV, resolveTarget: resolveV, loadOp: "load", storeOp: "store" }] }).end();
      enc.copyTextureToBuffer({ texture: resolve }, { buffer: buf, bytesPerRow: bpr }, [W, H]);
      device.queue.submit([enc.finish()]);
      const e = await device.popErrorScope();
      if (e) gpuError = e.message;
      await buf.mapAsync(GPUMapMode.READ);
      const u16 = new Uint16Array(buf.getMappedRange());
      const tpr = bpr / 8;
      let g0 = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const o = (y * tpr + x) * 4, r = half(u16[o]), g = half(u16[o + 1]), b = half(u16[o + 2]);
        if (g > 0.1 && g > r * 1.4 && g > b * 1.4) g0++;
      }
      buf.unmap();
      greenDom = Math.max(greenDom, g0);
      treeCount = trees.treeCount;
    }
    return { treeCount, greenDom, gpuError };
  });

  console.log(JSON.stringify({ spawnCount, render }, null, 2));
  expect(spawnCount).toBeGreaterThan(0);          // running app has trees near spawn
  expect(render.gpuError).toBeNull();             // live-config pipeline is valid
  expect(render.greenDom).toBeGreaterThan(50);    // and renders green in MSAA4/rgba16float
});
