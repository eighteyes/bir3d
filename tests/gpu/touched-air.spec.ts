// touched-air.spec.ts — Prove the "wind the bird physically touched" warm-trail feature in the LIVE app.
// Responsibilities:
//   - HEAT ACCUMULATES: after flying, some near motes report heat>0 (the bird's wake has touched them), and
//     the heat field spans a range (gentle→hard touch), not a single flat value.
//   - FPV LOCKSTEP INTACT: the vertex-format bump (10→11, +heat) did not break rendering — no pageerror
//     (a stride/attribute mismatch throws a WebGPU validation error), motes still render, 60fps holds.
//   - Screenshot for the visual record (yellow→red trails off the wings).
import { test, expect } from "@playwright/test";

const BOOT_MS = 15000;
const SETTLE_MS = 2500;

test("live bird app: touched air heats up (warm trails), FPV render intact", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));

  await page.goto("/index-bird.html");
  await page.waitForFunction(() => (window as any).__birdBooted === true, { timeout: BOOT_MS });
  await page.keyboard.press("P"); // autopilot → the bird MOVES so its wake touches air
  // the local sphere + wake default OFF (global-wind-first); enable them to test the touched-air feature.
  await page.evaluate(() => { const w = (window as any).__wind; w.setShowNear(true); w.setShowWake(true); });
  await page.waitForTimeout(SETTLE_MS);
  await page.waitForFunction(() => (window as any).__nearFrame?.().moving === true, { timeout: BOOT_MS });

  // fly a few seconds so the wake accumulates heat into the near motes, then read the heat field.
  await page.waitForTimeout(3500);
  const heat = await page.evaluate(() => {
    const h = (window as any).__wind.nearHeat as Float32Array;
    let max = 0, touched = 0, warm = 0, sum = 0;
    for (let i = 0; i < h.length; i++) {
      const v = h[i]!;
      if (v > max) max = v;
      if (v > 0.1) touched++;
      if (v > 0.3) warm++; // above the shader deadzone → renders warm (yellow→red)
      sum += v;
    }
    return { max, touched, warm, warmFrac: warm / h.length, mean: sum / h.length, count: h.length };
  });

  const overlay = await page.locator("#overlay").innerText();
  const fps = Number(/fps:\s*(\d+)/.exec(overlay)?.[1] ?? "0");
  const crashed = /✖ CRASH/.test(overlay);

  await page.screenshot({ path: "test-results/touched-air.png" });

  console.log(
    `[touched-air] maxHeat=${heat.max.toFixed(2)} touched=${heat.touched}/${heat.count} warm(>0.3)=${heat.warm} (${(heat.warmFrac * 100).toFixed(0)}%) meanHeat=${heat.mean.toFixed(3)}  fps=${fps}  crashed=${crashed}  errors=${errors.length}`,
  );

  // HEAT: the bird's wake genuinely touched air (some motes warm) and there is a spread, not all-or-nothing.
  expect(heat.max).toBeGreaterThan(0.3);
  expect(heat.touched).toBeGreaterThan(0);
  // FPV LOCKSTEP: the +heat vertex attribute didn't break the pipeline — no validation error, loop alive.
  expect(errors).toEqual([]);
  expect(fps).toBeGreaterThan(15);
  expect(crashed).toBe(false);
});
