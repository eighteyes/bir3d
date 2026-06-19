// trees-perf.spec.ts — Performance harness for the mountaintop forest pass.
// Responsibilities:
//   - Boot the bird app, measure TRUE average FPS (frame-counter delta / wall time) over a window.
//   - A/B the trees pass via window.__trees.enabled to isolate its frame cost.
//   - Report tree count for the current window and the per-rebuild CPU cost.

import { test, expect } from "@playwright/test";

// read "frame NNN" out of the overlay → a monotonic frame counter for true-average FPS.
async function frameCount(page: import("@playwright/test").Page): Promise<number> {
  const txt = await page.locator("#overlay").innerText();
  const m = txt.match(/frame (\d+)/);
  return m ? parseInt(m[1], 10) : -1;
}

async function measureFps(page: import("@playwright/test").Page, ms: number): Promise<number> {
  const f0 = await frameCount(page);
  const t0 = await page.evaluate(() => performance.now());
  await page.waitForTimeout(ms);
  const f1 = await frameCount(page);
  const t1 = await page.evaluate(() => performance.now());
  return ((f1 - f0) * 1000) / (t1 - t0);
}

test("forest pass: FPS A/B + tree count + rebuild cost", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/index-bird.html");
  await page.waitForFunction(() => (window as any).__birdBooted === true, { timeout: 15000 });
  await page.waitForTimeout(1500); // let the loop + first tree rebuild settle

  // trees ON
  await page.evaluate(() => ((window as any).__trees.enabled = true));
  const fpsOn = await measureFps(page, 4000);
  const treeCount = await page.evaluate(() => (window as any).__trees.treeCount);

  // trees OFF (baseline)
  await page.evaluate(() => ((window as any).__trees.enabled = false));
  const fpsOff = await measureFps(page, 4000);

  // isolate rebuild cost: force a rebuild by invalidating the cell cache, time it on the CPU.
  const rebuildMs = await page.evaluate(() => {
    const t = (window as any).__trees;
    t.lastCellX = NaN; // force rebuild on next draw
    t.enabled = true;
    const start = performance.now();
    t.rebuild(0, 0); // direct call (same window the loop would build)
    return performance.now() - start;
  });

  console.log(JSON.stringify({
    fpsOn: +fpsOn.toFixed(1),
    fpsOff: +fpsOff.toFixed(1),
    deltaFps: +(fpsOff - fpsOn).toFixed(1),
    treeCount,
    rebuildMs: +rebuildMs.toFixed(2),
    pageErrors: errors,
  }, null, 2));

  expect(errors).toEqual([]);
  expect(treeCount).toBeGreaterThan(0);
  expect(fpsOn).toBeGreaterThan(30);
});
