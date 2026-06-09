import { expect, test } from "@playwright/test";

test("acquires a WebGPU device and reports timestamp-query support", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const { acquireDevice } = await import("/src/host/gpu/device.ts");
    const ctx = await acquireDevice();
    return { ok: !!ctx.device, hasTimestamp: ctx.hasTimestampQuery };
  });
  expect(result.ok).toBe(true);
  expect(typeof result.hasTimestamp).toBe("boolean");
});
