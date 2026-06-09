import { expect, test } from "vitest";
import { RingIndex } from "../../src/host/gpu/readback";

test("ring advances modulo size and never returns the in-flight slot", () => {
  const ring = new RingIndex(3);
  const a = ring.acquire(); // slot for THIS frame's copy
  ring.advance();
  const b = ring.acquire();
  ring.advance();
  expect(a).not.toBe(b);
  // after `size` advances we reuse slot a — the frame that wrote it is now safely resolved
  ring.advance();
  expect(ring.acquire()).toBe(a);
});
