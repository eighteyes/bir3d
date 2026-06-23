// slipstream.spec.ts — Prove the wing slipstream (twin counter-rotating wingtip vortices) in the LIVE bird app.
// Responsibilities:
//   - COUNTER-ROTATION: sample the bird-wake field (window.__nearWake) behind each wingtip and assert the
//     two trailing vortices spin in OPPOSITE directions (the physical wingtip pair), with real magnitude.
//   - NO REGRESSION: the near sphere renders, the frame loop holds, the bird flies under autopilot without
//     crashing, and no pageerror fires. Screenshot captured for the visual record (two streams off the wings).
import { test, expect } from "@playwright/test";

const BOOT_MS = 15000;
const SETTLE_MS = 2500;

test("live bird app: twin wingtip vortices counter-rotate, no flight regression", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));

  await page.goto("/index-bird.html");
  await page.waitForFunction(() => (window as any).__birdBooted === true, { timeout: BOOT_MS });

  // hand to autopilot so the bird MOVES (the slipstream/wake only exists above the moving threshold).
  await page.keyboard.press("P");
  // the local sphere + wake default OFF (global-wind-first); enable them to test the feature.
  await page.evaluate(() => { const w = (window as any).__wind; w.setShowNear(true); w.setShowWake(true); });
  await page.waitForTimeout(SETTLE_MS);
  await page.waitForFunction(() => (window as any).__nearFrame?.().moving === true, { timeout: BOOT_MS });

  // COUNTER-ROTATION PROBE: behind each wingtip core, offset radially by the vortex-core radius, measure the
  // circulation (wake · (axis × right)). The two cores must have OPPOSITE sign and non-trivial magnitude.
  const probe = await page.evaluate(() => {
    const w = window as any;
    const f = w.__nearFrame() as { pos: number[]; axis: number[]; right: number[]; bs: number };
    const wind = w.__wind;
    const hs = wind.wingSpan as number;
    const rc = wind.vortexCore as number;
    const [ax, ay, az] = f.axis;
    const [rx, ry, rz] = f.right;
    const ux = ay * rz - az * ry, uy = az * rx - ax * rz, uz = ax * ry - ay * rx; // upW = axis × right
    const behind = 20; // m behind the bird (within the trailing-wake reach)
    const delta = rc;  // radial offset = core radius → peak swirl
    const circ = (side: number) => {
      // core line at pos + right*(side*hs); step BEHIND along −axis; probe offset by the SAME radial direction
      // (+right*delta) from EACH core. Measuring the tangential component with a common r̂ means an opposite
      // SIGN between the two cores ⇒ opposite circulation SENSE ⇒ genuinely counter-rotating. (Sampling mirror
      // OUTBOARD points instead would read the same sign — upwash on both flanks — which does NOT prove it.)
      const px = f.pos[0] + rx * (side * hs) - ax * behind + rx * delta;
      const py = f.pos[1] + ry * (side * hs) - ay * behind + ry * delta;
      const pz = f.pos[2] + rz * (side * hs) - az * behind + rz * delta;
      const wake = w.__nearWake(px, py, pz) as number[];
      return wake[0] * ux + wake[1] * uy + wake[2] * uz; // tangential / circulation component
    };
    const right = circ(1);
    const left = circ(-1);
    return { right, left, bs: f.bs, hs, rc };
  });

  // NO REGRESSION: fly ~6s under autopilot, read crash/fps from the HUD.
  for (let i = 0; i < 6; i++) await page.waitForTimeout(1000);
  const overlay = await page.locator("#overlay").innerText();
  const fps = Number(/fps:\s*(\d+)/.exec(overlay)?.[1] ?? "0");
  const crashed = /✖ CRASH/.test(overlay);

  await page.screenshot({ path: "test-results/slipstream.png" });

  console.log(
    `[slipstream] circ right=${probe.right.toFixed(2)} left=${probe.left.toFixed(2)} (opposite & nonzero ⇒ counter-rotating)  ` +
      `bs=${probe.bs.toFixed(1)} wingSpan=${probe.hs} core=${probe.rc}  fps=${fps}  crashed=${crashed}  errors=${errors.length}`,
  );

  // COUNTER-ROTATION: the two trailing vortices spin opposite ways (product < 0) with real magnitude.
  expect(probe.right * probe.left).toBeLessThan(0);
  expect(Math.abs(probe.right)).toBeGreaterThan(0.3);
  expect(Math.abs(probe.left)).toBeGreaterThan(0.3);
  // NO REGRESSION: frame loop alive, no crash, no runtime errors.
  expect(fps).toBeGreaterThan(15);
  expect(crashed).toBe(false);
  expect(errors).toEqual([]);
});
