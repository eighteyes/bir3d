import { expect, test } from "vitest";
import { PingPong } from "../../src/host/gpu/pingpong";

test("swap alternates current/next without aliasing", () => {
  const pp = new PingPong<string>("A", "B");
  expect(pp.current).toBe("A");
  expect(pp.next).toBe("B");
  pp.swap();
  expect(pp.current).toBe("B");
  expect(pp.next).toBe("A");
  expect(pp.current).not.toBe(pp.next);
});
