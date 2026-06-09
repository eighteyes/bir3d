// fluid-viz.spec.ts — Task 5: the live debug-viz boot gate ("I want to see something").
// Boots /index-fluid.html, waits for the fluid loop to run, lets it advance ~60 frames, and
// asserts the user-visible contract: the overlay reports a fluid ms readout, the canvas is NOT
// the clear color (the swirling dye actually rendered), and the page produced zero errors.
// Responsibilities:
//   - Navigate to /index-fluid.html and wait for window.__fluidBooted (the loop is running).
//   - Wait ~60 frames via requestAnimationFrame so the field swirls + the timer/pixel readbacks
//     resolve (both are async, guarded; a number/pixel appears within a handful of frames).
//   - Assert the overlay text contains a "fluid … <n>.<n> ms" readout (the per-stage warm-median
//     instrument is reporting, not stuck).
//   - Assert the rendered center pixel is non-blank: window.__centerPixel green channel (byte 1,
//     format-agnostic across rgba8/bgra8) is well above the black clear color.
//   - Assert zero page errors (uncaught exceptions / unhandled rejections) during the run.

import { expect, test } from "@playwright/test";

const GREEN_THRESHOLD = 64; // clear color is black (0); neon-green dye must lift the center well above.

test("fluid debug viz: boots, renders a non-blank swirl, reports fluid ms, zero errors", async ({ page }) => {
  test.setTimeout(60_000);

  const pageErrors: string[] = [];
  page.on("pageerror", (e) => pageErrors.push(e.message));

  await page.goto("/index-fluid.html");

  // The loop is running once boot() sets this flag.
  await page.waitForFunction(() => (window as any).__fluidBooted === true, null, { timeout: 30_000 });

  // Advance ~60 frames so the field develops and the async timer/pixel readbacks resolve.
  await page.evaluate(
    (n) =>
      new Promise<void>((resolve) => {
        let f = 0;
        const tick = () => (++f >= n ? resolve() : requestAnimationFrame(tick));
        requestAnimationFrame(tick);
      }),
    60
  );

  // Overlay reports a fluid ms readout (the instrument is live, not stuck on "warming…").
  await page.waitForFunction(
    () => /fluid[\s\S]*?\d+\.\d+\s*ms/i.test(document.getElementById("overlay")?.textContent ?? ""),
    null,
    { timeout: 20_000 }
  );
  const overlayText = await page.locator("#overlay").textContent();
  expect(overlayText, "overlay missing a fluid ms readout").toMatch(/fluid[\s\S]*?\d+\.\d+\s*ms/i);

  // Canvas is non-blank: the center pixel's green channel is well above the black clear color.
  await page.waitForFunction(
    (thr) => {
      const px = (window as any).__centerPixel as number[] | undefined;
      return Array.isArray(px) && px[1] > thr;
    },
    GREEN_THRESHOLD,
    { timeout: 20_000 }
  );
  const centerPixel = (await page.evaluate(() => (window as any).__centerPixel)) as number[];
  expect(centerPixel, "no center pixel read back").toBeTruthy();
  expect(centerPixel[1], `center pixel green=${centerPixel?.[1]} not above clear color`).toBeGreaterThan(
    GREEN_THRESHOLD
  );

  // Zero page errors throughout the run.
  expect(pageErrors, "page errors during fluid viz run").toEqual([]);
});
