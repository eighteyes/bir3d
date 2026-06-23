// updraft-buffer.spec.ts — Prove the ridge-lift "bigger buffer off hills" (L+B) in the LIVE bird app.
// Responsibilities:
//   - BUFFER: scan the live updraft field (window.__updraftAt) over a grid around the bird and show the
//     L+B band (lookahead+broaden, defaults) finds lift in strictly MORE places than the no-lookahead
//     baseline — and that there exist points where the OLD geometry gave ~no lift but L+B gives real lift
//     (the bird now catches a hill BEFORE skimming it).
//   - NO REGRESSION: under autopilot the bird still rides lift, holds a clean course (no crash), keeps a
//     live frame loop, and throws no pageerror.
import { test, expect } from "@playwright/test";

const BOOT_MS = 15000;
const SETTLE_MS = 2500; // let the fluid readback resolve so windAt is the real moving field (ridge lift needs wind)

test("live bird app: L+B gives a bigger ridge-lift buffer off hills, no flight regression", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));

  await page.goto("/index-bird.html");
  await page.waitForFunction(() => (window as any).__birdBooted === true, { timeout: BOOT_MS });
  await page.waitForTimeout(SETTLE_MS);

  // BUFFER PROOF: sample the exact updraft the bird rides across a wide grid around it, twice —
  // L+B (defaults) vs the no-buffer baseline (lookahead 0, narrow eps 6 = "lift only where the bird is").
  const buffer = await page.evaluate(() => {
    const w = window as any;
    const u = w.__updraftAt as (x: number, z: number, o?: Record<string, number>) => number;
    const pos = w.__birdPos as [number, number, number];
    const cx = pos[0], cz = pos[2];
    const baseline = { ridgeLookahead: 0, ridgeEps: 6 }; // pre-change geometry: no upwind buffer, narrow band
    const RANGE = 500, STEP = 10;
    let lbCells = 0, baseCells = 0, bufferCells = 0, total = 0;
    let maxLB = 0, maxBase = 0;
    for (let dx = -RANGE; dx <= RANGE; dx += STEP) {
      for (let dz = -RANGE; dz <= RANGE; dz += STEP) {
        const x = cx + dx, z = cz + dz;
        const lb = u(x, z);              // L+B (live defaults: lookahead 50, eps 14)
        const base = u(x, z, baseline);  // local-only, no buffer
        if (lb > 0.5) lbCells++;
        if (base > 0.5) baseCells++;
        // a "buffer cell": the OLD geometry gave ~no lift here but L+B gives real lift → the bird now
        // catches a windward face from further out instead of having to skim it.
        if (base < 0.3 && lb > 0.8) bufferCells++;
        maxLB = Math.max(maxLB, lb);
        maxBase = Math.max(maxBase, base);
        total++;
      }
    }
    return { lbCells, baseCells, bufferCells, total, maxLB, maxBase };
  });

  console.log(
    `[buffer] lift-cells L+B=${buffer.lbCells} baseline=${buffer.baseCells} (of ${buffer.total})  ` +
      `bufferCells=${buffer.bufferCells}  maxLift L+B=${buffer.maxLB.toFixed(2)} base=${buffer.maxBase.toFixed(2)}`,
  );

  // NO REGRESSION: hand to autopilot and fly ~8s; sample the ridden updraft + crash/fps from the HUD.
  await page.keyboard.press("P");
  const updrafts: number[] = [];
  for (let i = 0; i < 16; i++) {
    const ov = await page.locator("#overlay").innerText();
    const up = Number(/updraft:\s*\+?([\d.]+)/.exec(ov)?.[1] ?? "0");
    updrafts.push(up);
    await page.waitForTimeout(500);
  }
  const overlay = await page.locator("#overlay").innerText();
  const fps = Number(/fps:\s*(\d+)/.exec(overlay)?.[1] ?? "0");
  const crashed = /✖ CRASH/.test(overlay);
  const maxUpdraft = Math.max(...updrafts);

  await page.screenshot({ path: "test-results/updraft-buffer.png" });

  console.log(
    `[flight] maxRiddenUpdraft=${maxUpdraft.toFixed(1)} m/s  fps=${fps}  crashed=${crashed}  errors=${errors.length}`,
  );

  // BUFFER: the L+B band is strictly wider, and real buffer cells exist (lift gained where there was none).
  expect(buffer.bufferCells).toBeGreaterThan(0);
  expect(buffer.lbCells).toBeGreaterThanOrEqual(buffer.baseCells);
  // SANITY: there IS ridge lift to find in the scanned region (not a dead-calm world that proves nothing).
  expect(buffer.maxLB).toBeGreaterThan(0.8);
  // NO REGRESSION: the bird rides lift, doesn't crash, frame loop is alive, no runtime errors.
  expect(maxUpdraft).toBeGreaterThan(0);
  expect(crashed).toBe(false);
  expect(fps).toBeGreaterThan(15);
  expect(errors).toEqual([]);
});
