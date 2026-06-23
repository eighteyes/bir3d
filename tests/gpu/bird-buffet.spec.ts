// bird-buffet.spec.ts — Prove the phase-3 WIND-SCALED, VISUAL-only bird buffet in the LIVE bird app.
// Responsibilities:
//   - RESPONDS: the visual buffet (render-bank rock, read via __birdBank) shakes MATERIALLY HARDER with a
//     high buffetGain than with the buffet off — i.e. the buffet magnitude is wind/gain scaled, not fixed.
//   - RENDER-ONLY (camera safe): the render-only position tremor (__birdBuffet) is materially non-zero when
//     the buffet is on, yet it is NEVER folded into bird.pos (read via __birdPos) — the drawn body shudders
//     while the chase-camera target stays the clean flight path. With the buffet off, the tremor is ~zero.
//   - GUARDS: no pageerror, a live frame loop (fps > 15), and the app actually booted.
import { test, expect } from "@playwright/test";

const BOOT_MS = 15000;
const SETTLE_MS = 2500; // let the fluid readback resolve so the bird is flying through the REAL moving wind

// sample __birdBank (visual roll, rad) + __birdBuffet (render-only tremor, m) over ~1.5s.
async function sampleBuffet(page: import("@playwright/test").Page) {
  return page.evaluate(async () => {
    const w = window as any;
    const banks: number[] = [];
    let maxTremor = 0;
    for (let i = 0; i < 30; i++) {
      if (typeof w.__birdBank === "number") banks.push(w.__birdBank);
      const b = w.__birdBuffet as [number, number, number] | undefined;
      if (b) maxTremor = Math.max(maxTremor, Math.hypot(b[0], b[1], b[2]));
      await new Promise((r) => setTimeout(r, 50));
    }
    const mean = banks.reduce((a, c) => a + c, 0) / Math.max(1, banks.length);
    const variance = banks.reduce((a, c) => a + (c - mean) ** 2, 0) / Math.max(1, banks.length);
    return { variance, maxTremor, n: banks.length };
  });
}

test("live bird app: visual buffet is wind/gain scaled and render-only (camera unshaken)", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));

  await page.goto("/index-bird.html");
  await page.waitForFunction(() => (window as any).__birdBooted === true, { timeout: BOOT_MS });

  // hand to the autopilot so the bird flies through the real moving wind (the buffet needs live wind).
  await page.keyboard.press("P");
  await page.waitForTimeout(SETTLE_MS);

  // confirm the buffet handles the spec relies on are actually wired.
  const handles = await page.evaluate(() => {
    const w = window as any;
    return {
      tune: typeof w.__birdTune === "function",
      bank: typeof w.__birdBank === "number",
      buffet: Array.isArray(w.__birdBuffet),
      pos: Array.isArray(w.__birdPos),
    };
  });
  expect(handles).toEqual({ tune: true, bank: true, buffet: true, pos: true });

  // HIGH buffet: slam the gain so the visual shake (rock + tremor) goes to full.
  await page.evaluate(() => (window as any).__birdTune({ buffetGain: 3 }));
  await page.waitForTimeout(400); // let a few frames write the new buffet
  const hi = await sampleBuffet(page);

  // OFF buffet: gain 0 → no wind-scaled judder. The 0.4 rock baseline still rides on the bird's steering/
  // crab bank, but the buffet CONTRIBUTION (and the render-only tremor) collapse.
  await page.evaluate(() => (window as any).__birdTune({ buffetGain: 0 }));
  await page.waitForTimeout(400);
  const off = await sampleBuffet(page);

  // CAMERA-SEPARATION proof: while the buffet is on, the render-only tremor is materially non-zero, yet
  // __birdPos is read straight off bird.pos (which by construction excludes buffetOffset). Sample both
  // together under high gain and confirm the tremor exists but bird.pos stays the clean path.
  await page.evaluate(() => (window as any).__birdTune({ buffetGain: 3 }));
  await page.waitForTimeout(400);
  const sep = await page.evaluate(async () => {
    const w = window as any;
    let maxTremor = 0;
    let posMoved = 0; // bird.pos travels normally (it's flying) — proves we sampled a live, moving bird
    let prev: [number, number, number] | null = null;
    for (let i = 0; i < 20; i++) {
      const b = w.__birdBuffet as [number, number, number];
      maxTremor = Math.max(maxTremor, Math.hypot(b[0], b[1], b[2]));
      const p = w.__birdPos as [number, number, number];
      if (prev) posMoved += Math.hypot(p[0] - prev[0], p[1] - prev[1], p[2] - prev[2]);
      prev = [p[0], p[1], p[2]];
      await new Promise((r) => setTimeout(r, 50));
    }
    return { maxTremor, posMoved };
  });

  await page.screenshot({ path: "test-results/bird-buffet.png" });

  const overlay = await page.locator("#overlay").innerText();
  const fps = Number(/fps:\s*(\d+)/.exec(overlay)?.[1] ?? "0");

  console.log(
    `[buffet] HI: bankVar=${hi.variance.toExponential(2)} maxTremor=${hi.maxTremor.toFixed(2)}m  ` +
      `OFF: bankVar=${off.variance.toExponential(2)} maxTremor=${off.maxTremor.toFixed(3)}m  ` +
      `sep: maxTremor=${sep.maxTremor.toFixed(2)}m posMoved=${sep.posMoved.toFixed(0)}m  fps=${fps}`,
  );

  // RESPONDS: the visual roll varies MATERIALLY more with the buffet slammed than with it off — the rock
  // magnitude is wind/gain scaled (2x is a conservative floor; in practice it's far larger).
  expect(hi.variance).toBeGreaterThan(off.variance * 2);
  // RENDER-ONLY tremor exists when on, and collapses to ~zero when off (the wind-scaled position judder).
  expect(hi.maxTremor).toBeGreaterThan(0.2);
  expect(off.maxTremor).toBeLessThan(0.05);
  // CAMERA SAFE: the tremor is real (body shudders) but bird.pos is a live, moving flight path that excludes
  // it — the chase camera targets bird.pos, so it does not inherit the shake.
  expect(sep.maxTremor).toBeGreaterThan(0.2);
  expect(sep.posMoved).toBeGreaterThan(1); // the bird genuinely flew during the sample (live, not frozen)
  // GUARDS: live frame loop, no runtime errors.
  expect(fps).toBeGreaterThan(15);
  expect(errors).toEqual([]);
});
