// wind-render-modes.spec.ts — Phase-1 smoke test for the per-tier wind RENDER MODE switches.
// Responsibilities:
//   - HANDLES EXIST: __farMode / __nearMode / __wakeMode are functions on window (the T-panel buttons
//     drive the same setters).
//   - SWITCHING IS SAFE: cycling every value of each tier (FAR comet→stipple→chevron, NEAR
//     comet→flecks→filaments, WAKE modulate→helix→rings) does not crash the app or trip a WebGPU
//     validation error — wake is enabled first so the wake tier is actually live while we switch it.
//   - DEMO STILL RUNS: after all switches, zero pageerrors, the loop is alive (fps>15), bird still booted.
// PHASE-1 NOTE: B/C modes currently FALL THROUGH to the comet/modulate look in the engine — divergent
// geometry is a later phase — so we deliberately do NOT assert the visuals change here.
import { test, expect } from "@playwright/test";

const BOOT_MS = 15000;
const SETTLE_MS = 2500;

test("live bird app: per-tier wind render modes switch without crashing (phase 1)", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));

  await page.goto("/index-bird.html");
  await page.waitForFunction(() => (window as any).__birdBooted === true, { timeout: BOOT_MS });
  await page.keyboard.press("P"); // autopilot → the bird MOVES so all tiers (incl. wake) are exercised
  // wake defaults OFF (global-wind-first); enable it so the WAKE tier is live while we switch its mode.
  await page.evaluate(() => { (window as any).__wind.setShowWake(true); });
  await page.waitForTimeout(SETTLE_MS);

  // the mode-switch debug handles must exist and be callable.
  const handlesOk = await page.evaluate(() => {
    const w = window as any;
    return typeof w.__farMode === "function"
      && typeof w.__nearMode === "function"
      && typeof w.__wakeMode === "function";
  });
  expect(handlesOk).toBe(true);

  // cycle EVERY value of each tier with a frame/short wait between each switch.
  const cycle = async (handle: string, modes: string[]) => {
    for (const m of modes) {
      await page.evaluate(([h, v]) => { (window as any)[h](v); }, [handle, m]);
      await page.waitForTimeout(120);
    }
  };
  await cycle("__farMode", ["comet", "stipple", "chevron"]);
  await cycle("__nearMode", ["comet", "flecks", "filaments"]);
  await cycle("__wakeMode", ["modulate", "helix", "rings"]);

  // let a few more frames run after the last switch, then read the live HUD.
  await page.waitForTimeout(1000);
  const overlay = await page.locator("#overlay").innerText();
  const fps = Number(/fps:\s*(\d+)/.exec(overlay)?.[1] ?? "0");
  const crashed = /✖ CRASH/.test(overlay);
  const booted = await page.evaluate(() => (window as any).__birdBooted === true);

  console.log(
    `[wind-render-modes] handlesOk=${handlesOk} fps=${fps} crashed=${crashed} booted=${booted} errors=${errors.length}`,
  );

  // SWITCHING IS SAFE: no runtime errors, no WebGPU validation throws (those surface as pageerrors).
  expect(errors).toEqual([]);
  // DEMO STILL RUNS: loop alive, no crash spiral, app still booted.
  expect(fps).toBeGreaterThan(15);
  expect(crashed).toBe(false);
  expect(booted).toBe(true);
});
