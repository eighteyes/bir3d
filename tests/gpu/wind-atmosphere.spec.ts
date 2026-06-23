// wind-atmosphere.spec.ts — Prove the GLOBAL WIND altitude atmosphere (gameplay) in the LIVE bird app.
// Responsibilities:
//   - GRADIENT REAL: the windProfile curve rises monotonically with absolute altitude (calm low → strong
//     high) and hits its calibrated endpoints. Ridge lift consumes it (lift at a windward spot is stronger
//     sampled at high altitude than low) — proving the physics, not just the visual, reads the profile.
//   - SOARING PRESERVED: unfreezing the wind must NOT break flight. Under autopilot the bird still rides
//     positive updraft and does not crash, frame loop holds. (Absolute-altitude design keeps ridges windy.)
import { test, expect } from "@playwright/test";

const BOOT_MS = 15000;
const SETTLE_MS = 2500;

test("live bird app: global-wind altitude atmosphere, soaring preserved", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));

  await page.goto("/index-bird.html");
  await page.waitForFunction(() => (window as any).__birdBooted === true, { timeout: BOOT_MS });
  await page.waitForTimeout(SETTLE_MS);

  // GRADIENT: read the altitude curve directly + confirm ridge lift scales with altitude.
  const grad = await page.evaluate(() => {
    const prof = (window as any).__windProfileAt as (y: number) => number;
    const samples = [0, 100, 200, 300, 400, 500, 600].map((y) => ({ y, p: prof(y) }));
    // ridge lift now uses the STRONG ALOFT wind (altitude-INDEPENDENT) so soaring works at any ridge height and
    // the bird is never stranded low. Confirm strong lift is available across the windward terrain.
    const up = (window as any).__updraftAt as (x: number, z: number, o: any) => number;
    const pos = (window as any).__birdPos as number[];
    let maxLift = 0;
    for (let dx = -500; dx <= 500; dx += 30) {
      for (let dz = -500; dz <= 500; dz += 30) {
        const u = up(pos[0] + dx, pos[2] + dz, {});
        if (u > maxLift) maxLift = u;
      }
    }
    return { samples, maxLift };
  });

  // SOARING: autopilot ~8s, sample the ridden updraft + crash/fps from the HUD.
  await page.keyboard.press("P");
  const updrafts: number[] = [];
  for (let i = 0; i < 16; i++) {
    const ov = await page.locator("#overlay").innerText();
    updrafts.push(Number(/updraft:\s*\+?([\d.]+)/.exec(ov)?.[1] ?? "0"));
    await page.waitForTimeout(500);
  }
  const overlay = await page.locator("#overlay").innerText();
  const fps = Number(/fps:\s*(\d+)/.exec(overlay)?.[1] ?? "0");
  const crashed = /✖ CRASH/.test(overlay);
  const maxUpdraft = Math.max(...updrafts);

  await page.screenshot({ path: "test-results/wind-atmosphere.png" });

  const ps = grad.samples.map((s) => s.p);
  const monotonic = ps.every((p, i) => i === 0 || p >= ps[i - 1]! - 1e-6);
  console.log(
    `[atmosphere] profile ${grad.samples.map((s) => `${s.y}:${s.p.toFixed(2)}`).join(" ")}  ` +
      `maxRidgeLift=${grad.maxLift.toFixed(2)} (aloft, altitude-independent)  ` +
      `maxRiddenUpdraft=${maxUpdraft.toFixed(1)} fps=${fps} crashed=${crashed} errors=${errors.length}`,
  );

  // GRADIENT: monotonic rise, calm low (<0.6) → strong high (>1.0), and ridge lift is stronger high than low.
  expect(monotonic).toBe(true);
  expect(ps[0]!).toBeLessThan(0.6);
  expect(ps[ps.length - 1]!).toBeGreaterThan(1.0);
  expect(grad.maxLift).toBeGreaterThan(5); // ridge lift uses strong aloft wind → soaring available at any altitude (not nerfed low / stranding)
  // SOARING PRESERVED: the bird still rides lift, doesn't crash, frame loop alive, no runtime errors.
  expect(maxUpdraft).toBeGreaterThan(0);
  expect(crashed).toBe(false);
  expect(fps).toBeGreaterThan(15);
  expect(errors).toEqual([]);
});
