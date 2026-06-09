# Engine Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A running WebGPU app (`vs-host`) with a verified GPU-timestamp profiler and the ping-pong / dispatch / async-readback convention, proven end-to-end by a trivial compute pass, with a green TDD pipeline — the measuring rig and plumbing every later subsystem depends on.

**Architecture:** TypeScript/Vite host owns the `GPUDevice` (blueprint §4.2 wiring: JS owns the device, WASM will orchestrate later). A small typed "GPU convention" module provides device+feature acquisition (incl. `timestamp-query`), ping-pong buffer pairs, a compute-dispatch helper, an async readback ring (never awaited in-frame), and a timestamp profiler (per-pass ms). A trivial add-one compute kernel proves the pipeline. Rust/WASM (`vs-core`) is scaffolded but stubbed; it enters when CPU orchestration arrives (a later plan).

**Tech Stack:** TypeScript, Vite, Vitest (pure-logic unit tests, Node), Playwright + headless Chromium with WebGPU enabled (GPU integration tests), WGSL. Cargo workspace stub for `vs-core` (cargo test green now; wasm-pack wiring later).

**This is Plan 1 of N.** Roadmap of subsequent plans (each its own working/testable increment, built on this foundation):
- Plan 2 — Fluid math in Rust (CPU reference): grid, semi-Lagrangian advection, divergence, Jacobi/multigrid pressure projection, 2.5D layer coupling. Pure `cargo test`, no GPU.
- Plan 3 — Fluid GPU port + **feasibility spike** (blueprint §8.1): port kernels to WGSL, validate GPU≈CPU via readback, moving-window + coarse-tier coupling, capture per-pass ms vs the §3 budget on this machine. **The make-or-break gate.**
- Plan 4 — Vector renderer core (§4.1): `submitStroke`, ribbon expansion, segment-SDF, bloom.
- Plan 5+ — Terrain substrate (§4.4), scheduler (§4.3), WorldState format (§4.5), then per-game (Bird first).

---

## File Structure

```
Cargo.toml                          workspace: members = ["crates/vs-core"]
crates/vs-core/Cargo.toml           Rust engine core (stub now)
crates/vs-core/src/lib.rs           one pure unit-tested fn (proves cargo test green)
package.json                        JS workspace + scripts
tsconfig.json
vite.config.ts
vitest.config.ts
playwright.config.ts
index.html                          host entry; #overlay div for the ms readout
src/host/gpu/device.ts              acquire adapter+device, detect 'timestamp-query', error scopes
src/host/gpu/pingpong.ts            PingPong<T>: pair of buffers, current()/next()/swap()
src/host/gpu/dispatch.ts            runComputePass(): encode+dispatch a compute pipeline
src/host/gpu/readback.ts            ReadbackRing: triple-buffered mapAsync, never awaited in-frame
src/host/gpu/profiler.ts            GpuProfiler: timestamp querySet → resolve → per-pass ms
src/host/shaders/addone.wgsl        trivial compute kernel: out[i] = in[i] + 1
src/host/frameloop.ts               rAF loop, max(GPU,CPU) timing hooks
src/host/main.ts                    bootstrap: device → loop → run addone → overlay ms
tests/unit/pingpong.test.ts         vitest: swap logic (no GPU)
tests/unit/readback-ring.test.ts    vitest: ring index math (no GPU)
tests/gpu/smoke.spec.ts             playwright: device acquired, addone correct, timestamp > 0, boots N frames
```

**Responsibility boundaries:** each `gpu/*.ts` file owns exactly one piece of the convention and is independently testable. `device.ts` knows nothing about fluids or rendering; `profiler.ts` measures any pass; `readback.ts` is generic over buffer contents.

---

## Task 0: Repo scaffold + tooling green

**Files:**
- Create: `package.json`, `tsconfig.json`, `vite.config.ts`, `vitest.config.ts`, `index.html`
- Create: `Cargo.toml`, `crates/vs-core/Cargo.toml`, `crates/vs-core/src/lib.rs`
- Create: `tests/unit/smoke.test.ts`

- [ ] **Step 1: Write `package.json`**

```json
{
  "name": "vector-system",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "test": "vitest run",
    "test:gpu": "playwright test"
  },
  "devDependencies": {
    "@playwright/test": "^1.48.0",
    "typescript": "^5.6.0",
    "vite": "^5.4.0",
    "vitest": "^2.1.0"
  }
}
```

- [ ] **Step 2: Write `tsconfig.json`** (WebGPU types ship with TS lib `dom` + `@webgpu/types`; use the bundled lib first, add `@webgpu/types` only if `GPUDevice` is unresolved)

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "types": ["@webgpu/types"],
    "skipLibCheck": true
  },
  "include": ["src", "tests"]
}
```

- [ ] **Step 3: Write `vite.config.ts`, `vitest.config.ts`, `index.html`**

`vite.config.ts`:
```ts
import { defineConfig } from "vite";
export default defineConfig({ server: { port: 5173 } });
```
`vitest.config.ts`:
```ts
import { defineConfig } from "vitest/config";
export default defineConfig({ test: { environment: "node", include: ["tests/unit/**/*.test.ts"] } });
```
`index.html`:
```html
<!doctype html>
<html><head><meta charset="utf-8" /><title>Vector System</title>
<style>body{margin:0;background:#05060a;color:#7fffd4;font:13px monospace}
#overlay{position:fixed;top:8px;left:8px;white-space:pre}</style></head>
<body><div id="overlay">booting…</div><script type="module" src="/src/host/main.ts"></script></body></html>
```

- [ ] **Step 4: Write the Rust workspace stub**

`Cargo.toml`:
```toml
[workspace]
resolver = "2"
members = ["crates/vs-core"]
```
`crates/vs-core/Cargo.toml`:
```toml
[package]
name = "vs-core"
version = "0.0.0"
edition = "2021"
```
`crates/vs-core/src/lib.rs`:
```rust
//! vs-core — Vector System engine core (orchestration brain; stubbed in the foundation increment).

/// Engine ABI version. Bumped when the WASM↔JS command-descriptor layout changes.
pub fn abi_version() -> u32 { 1 }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn abi_version_is_stable() { assert_eq!(abi_version(), 1); }
}
```

- [ ] **Step 5: Write a trivial passing unit test** `tests/unit/smoke.test.ts`

```ts
import { expect, test } from "vitest";
test("tooling is wired", () => { expect(1 + 1).toBe(2); });
```

- [ ] **Step 6: Install + verify tooling green**

Run: `npm install && npm run test && cargo test`
Expected: `npm run test` passes (1 test); `cargo test` passes (`abi_version_is_stable`).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: scaffold vs-host (vite/vitest/playwright) + vs-core cargo workspace"
```

---

## Task 1: WebGPU device acquisition

**Files:**
- Create: `src/host/gpu/device.ts`
- Create: `playwright.config.ts`
- Test: `tests/gpu/smoke.spec.ts` (device sub-test)

- [ ] **Step 1: Write `playwright.config.ts`** (headless Chromium with WebGPU enabled)

```ts
import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "tests/gpu",
  webServer: { command: "npm run dev", url: "http://localhost:5173", reuseExistingServer: true },
  use: {
    baseURL: "http://localhost:5173",
    launchOptions: {
      args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=metal"],
    },
  },
});
```
Note: on macOS, Chromium WebGPU uses Metal via ANGLE; the flags above are the working set as of Chromium 128+. If a runner reports no adapter, fall back to `headless: false` in `launchOptions` to confirm WebGPU works headed, then revisit flags.

- [ ] **Step 2: Write the failing test** — append to `tests/gpu/smoke.spec.ts`

```ts
import { expect, test } from "@playwright/test";

test("acquires a WebGPU device and reports timestamp-query support", async ({ page }) => {
  const result = await page.evaluate(async () => {
    const { acquireDevice } = await import("/src/host/gpu/device.ts");
    const ctx = await acquireDevice();
    return { ok: !!ctx.device, hasTimestamp: ctx.hasTimestampQuery };
  });
  expect(result.ok).toBe(true);
  expect(typeof result.hasTimestamp).toBe("boolean");
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `npm run test:gpu -- -g "acquires a WebGPU device"`
Expected: FAIL — module `device.ts` not found.

- [ ] **Step 4: Implement `src/host/gpu/device.ts`**

```ts
// device.ts — acquire the GPUDevice and detect engine-critical features.
// Responsibilities: adapter/device acquisition; timestamp-query feature gate; install an
// uncaptured-error logger so validation errors are never silent.

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  hasTimestampQuery: boolean;
}

export async function acquireDevice(): Promise<GpuContext> {
  if (!("gpu" in navigator)) throw new Error("WebGPU unavailable: navigator.gpu missing");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU unavailable: no adapter");
  const hasTimestampQuery = adapter.features.has("timestamp-query");
  const requiredFeatures: GPUFeatureName[] = hasTimestampQuery ? ["timestamp-query"] : [];
  const device = await adapter.requestDevice({ requiredFeatures });
  device.addEventListener("uncapturederror", (e) =>
    console.error("[WebGPU uncaptured]", (e as GPUUncapturedErrorEvent).error)
  );
  return { adapter, device, hasTimestampQuery };
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `npm run test:gpu -- -g "acquires a WebGPU device"`
Expected: PASS (`ok=true`).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(gpu): WebGPU device acquisition + timestamp-query detection"
```

---

## Task 2: Ping-pong buffer convention

**Files:**
- Create: `src/host/gpu/pingpong.ts`
- Test: `tests/unit/pingpong.test.ts`

- [ ] **Step 1: Write the failing test** `tests/unit/pingpong.test.ts`

```ts
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test -- pingpong`
Expected: FAIL — `PingPong` not exported.

- [ ] **Step 3: Implement `src/host/gpu/pingpong.ts`**

```ts
// pingpong.ts — a read/write pair for double-buffered GPU state (fluid grids, particle SoA).
// Responsibilities: hold two like resources; expose stable current()/next(); swap each step.
// Generic over T so it unit-tests without a GPU (T can be GPUBuffer at runtime).

export class PingPong<T> {
  private a: T;
  private b: T;
  private flipped = false;
  constructor(a: T, b: T) { this.a = a; this.b = b; }
  get current(): T { return this.flipped ? this.b : this.a; }
  get next(): T { return this.flipped ? this.a : this.b; }
  swap(): void { this.flipped = !this.flipped; }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm run test -- pingpong`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(gpu): ping-pong double-buffer convention"
```

---

## Task 3: Trivial compute kernel + dispatch helper

**Files:**
- Create: `src/host/shaders/addone.wgsl`, `src/host/gpu/dispatch.ts`
- Test: `tests/gpu/smoke.spec.ts` (add-one sub-test)

- [ ] **Step 1: Write `src/host/shaders/addone.wgsl`**

```wgsl
// addone.wgsl — proves the compute pipeline end-to-end: out[i] = in[i] + 1.
@group(0) @binding(0) var<storage, read>       inBuf  : array<f32>;
@group(0) @binding(1) var<storage, read_write> outBuf : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&inBuf)) { return; }
  outBuf[i] = inBuf[i] + 1.0;
}
```

- [ ] **Step 2: Implement `src/host/gpu/dispatch.ts`**

```ts
// dispatch.ts — encode and submit a single compute pass over a 1D workload.
// Responsibilities: build a pipeline from WGSL, bind group from buffers, dispatch ceil(n/wg).

export function makeComputePipeline(device: GPUDevice, code: string, entryPoint = "main"): GPUComputePipeline {
  const module = device.createShaderModule({ code });
  return device.createComputePipeline({ layout: "auto", compute: { module, entryPoint } });
}

export function runComputePass(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindings: GPUBuffer[],
  workItems: number,
  workgroupSize = 64,
  encoder = device.createCommandEncoder(),
  submit = true
): GPUCommandEncoder {
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bindings.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(workItems / workgroupSize));
  pass.end();
  if (submit) device.queue.submit([encoder.finish()]);
  return encoder;
}
```

- [ ] **Step 3: Write the failing test** — append to `tests/gpu/smoke.spec.ts`

```ts
test("add-one compute kernel maps [1,2,3] -> [2,3,4]", async ({ page }) => {
  const out = await page.evaluate(async () => {
    const { acquireDevice } = await import("/src/host/gpu/device.ts");
    const { makeComputePipeline, runComputePass } = await import("/src/host/gpu/dispatch.ts");
    const addone = await (await fetch("/src/host/shaders/addone.wgsl")).text();
    const { device } = await acquireDevice();
    const data = new Float32Array([1, 2, 3]);
    const inBuf = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(inBuf, 0, data);
    const outBuf = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const pipeline = makeComputePipeline(device, addone);
    runComputePass(device, pipeline, [inBuf, outBuf], data.length);
    const staging = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(outBuf, 0, staging, 0, data.byteLength);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    return Array.from(new Float32Array(staging.getMappedRange().slice(0)));
  });
  expect(out).toEqual([2, 3, 4]);
});
```

- [ ] **Step 4: Run test to verify it fails then passes**

Run: `npm run test:gpu -- -g "add-one compute kernel"`
Expected: first run FAIL (no dispatch.ts), then after Step 2 in place, PASS with `[2,3,4]`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(gpu): add-one compute kernel + dispatch helper (pipeline proven)"
```

---

## Task 4: Async readback ring

**Files:**
- Create: `src/host/gpu/readback.ts`
- Test: `tests/unit/readback-ring.test.ts`

- [ ] **Step 1: Write the failing test** `tests/unit/readback-ring.test.ts` (pure ring-index logic, no GPU)

```ts
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test -- readback-ring`
Expected: FAIL — `RingIndex` not exported.

- [ ] **Step 3: Implement `src/host/gpu/readback.ts`**

```ts
// readback.ts — async GPU→CPU aggregate readback that NEVER blocks the frame.
// Responsibilities: round-robin staging buffers (triple by default); kick mapAsync without
// awaiting in-frame; hand the last RESOLVED result to the CPU (2–3 frames stale, by design).
// RingIndex is split out so the index policy is unit-testable without a GPU.

export class RingIndex {
  private i = 0;
  constructor(private readonly size: number) {}
  acquire(): number { return this.i; }
  advance(): void { this.i = (this.i + 1) % this.size; }
}

export class ReadbackRing {
  private readonly ring: RingIndex;
  private readonly staging: GPUBuffer[];
  private readonly mapped: (Float32Array | null)[];
  private readonly inflight: boolean[];
  private latest: Float32Array | null = null;

  constructor(private device: GPUDevice, private byteLength: number, size = 3) {
    this.ring = new RingIndex(size);
    this.staging = Array.from({ length: size }, () =>
      device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    );
    this.mapped = Array.from({ length: size }, () => null);
    this.inflight = Array.from({ length: size }, () => false);
  }

  /** Encode the copy from `src` into this frame's slot, then kick a non-awaited map. */
  enqueue(encoder: GPUCommandEncoder, src: GPUBuffer): void {
    const slot = this.ring.acquire();
    if (!this.inflight[slot]) {
      encoder.copyBufferToBuffer(src, 0, this.staging[slot]!, 0, this.byteLength);
      this.inflight[slot] = true;
      // fire-and-forget: resolves a few frames later
      this.staging[slot]!.mapAsync(GPUMapMode.READ).then(() => {
        this.latest = new Float32Array(this.staging[slot]!.getMappedRange().slice(0));
        this.staging[slot]!.unmap();
        this.inflight[slot] = false;
      });
    }
    this.ring.advance();
  }

  /** Last resolved aggregate (or null until the first map resolves). Never blocks. */
  read(): Float32Array | null { return this.latest; }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm run test -- readback-ring`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(gpu): async readback ring (never awaited in-frame)"
```

---

## Task 5: GPU timestamp profiler

**Files:**
- Create: `src/host/gpu/profiler.ts`
- Test: `tests/gpu/smoke.spec.ts` (timestamp sub-test)

- [ ] **Step 1: Implement `src/host/gpu/profiler.ts`**

```ts
// profiler.ts — per-pass GPU timing via timestamp-query. The instrument the perf-LOD
// controller (blueprint §7.3) and the fluid feasibility spike (§8.1) are built on.
// Responsibilities: allocate a querySet + resolve buffer; wrap a compute pass with begin/end
// timestamps; resolve to milliseconds. Degrades to NaN when timestamp-query is unsupported.

export class GpuProfiler {
  private querySet?: GPUQuerySet;
  private resolveBuf?: GPUBuffer;
  private readBuf?: GPUBuffer;
  readonly enabled: boolean;

  constructor(private device: GPUDevice, enabled: boolean) {
    this.enabled = enabled;
    if (!enabled) return;
    this.querySet = device.createQuerySet({ type: "timestamp", count: 2 });
    this.resolveBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this.readBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  }

  /** Timestamp-wrapped compute pass descriptor; pass null when disabled. */
  timestampWrites(): GPUComputePassTimestampWrites | undefined {
    if (!this.enabled) return undefined;
    return { querySet: this.querySet!, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
  }

  /** Encode resolve+copy after the pass; call once per measured frame. */
  resolve(encoder: GPUCommandEncoder): void {
    if (!this.enabled) return;
    encoder.resolveQuerySet(this.querySet!, 0, 2, this.resolveBuf!, 0);
    encoder.copyBufferToBuffer(this.resolveBuf!, 0, this.readBuf!, 0, 16);
  }

  /** Await the mapped timestamps and return elapsed ms (NaN if disabled). Not for the hot loop. */
  async readMs(): Promise<number> {
    if (!this.enabled) return NaN;
    await this.readBuf!.mapAsync(GPUMapMode.READ);
    const t = new BigUint64Array(this.readBuf!.getMappedRange().slice(0));
    this.readBuf!.unmap();
    return Number(t[1] - t[0]) / 1e6; // ns → ms
  }
}
```

- [ ] **Step 2: Write the failing test** — append to `tests/gpu/smoke.spec.ts`

```ts
test("profiler reports a plausible positive ms for a compute pass (when supported)", async ({ page }) => {
  const ms = await page.evaluate(async () => {
    const { acquireDevice } = await import("/src/host/gpu/device.ts");
    const { makeComputePipeline, runComputePass } = await import("/src/host/gpu/dispatch.ts");
    const { GpuProfiler } = await import("/src/host/gpu/profiler.ts");
    const addone = await (await fetch("/src/host/shaders/addone.wgsl")).text();
    const { device, hasTimestampQuery } = await acquireDevice();
    if (!hasTimestampQuery) return 0; // pass trivially where unsupported
    const n = 1 << 16;
    const buf = (usage: number) => device.createBuffer({ size: n * 4, usage });
    const inBuf = buf(GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const outBuf = buf(GPUBufferUsage.STORAGE);
    const pipeline = makeComputePipeline(device, addone);
    const prof = new GpuProfiler(device, true);
    const enc = device.createCommandEncoder();
    const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }] });
    const pass = enc.beginComputePass({ timestampWrites: prof.timestampWrites() });
    pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(Math.ceil(n / 64)); pass.end();
    prof.resolve(enc);
    device.queue.submit([enc.finish()]);
    return prof.readMs();
  });
  expect(ms).toBeGreaterThanOrEqual(0);
  expect(ms).toBeLessThan(100); // a 64k add-one must be well under 100ms
});
```

- [ ] **Step 3: Run test to verify it fails then passes**

Run: `npm run test:gpu -- -g "profiler reports"`
Expected: first FAIL (no profiler.ts), then PASS after Step 1.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(gpu): timestamp-query profiler (per-pass ms instrument)"
```

---

## Task 6: Frame loop + bootstrap + overlay

**Files:**
- Create: `src/host/frameloop.ts`, `src/host/main.ts`
- Modify: `index.html` (already has `#overlay`)
- Test: `tests/gpu/smoke.spec.ts` (boot sub-test)

- [ ] **Step 1: Implement `src/host/frameloop.ts`**

```ts
// frameloop.ts — requestAnimationFrame driver with the blueprint's frame-timing hooks.
// Responsibilities: call onFrame(realDtSeconds) each rAF; expose start/stop; track a CPU-side
// dt so later the budget model frame_ms = max(GPU, CPU) + derate can be evaluated.

export class FrameLoop {
  private raf = 0;
  private last = 0;
  private running = false;
  constructor(private onFrame: (dtSeconds: number) => void) {}
  start(): void {
    if (this.running) return;
    this.running = true;
    const tick = (t: number) => {
      if (!this.running) return;
      const dt = this.last ? (t - this.last) / 1000 : 1 / 60;
      this.last = t;
      this.onFrame(dt);
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }
  stop(): void { this.running = false; cancelAnimationFrame(this.raf); }
}
```

- [ ] **Step 2: Implement `src/host/main.ts`**

```ts
// main.ts — bootstrap: acquire device, run the add-one pass each frame, show per-pass ms.
// Proves the whole foundation runs in a live frame loop and the instrument reports numbers.
import { acquireDevice } from "./gpu/device";
import { makeComputePipeline } from "./gpu/dispatch";
import { GpuProfiler } from "./gpu/profiler";
import { FrameLoop } from "./frameloop";

const overlay = document.getElementById("overlay")!;

async function boot() {
  const { device, hasTimestampQuery } = await acquireDevice();
  const code = await (await fetch("/src/host/shaders/addone.wgsl")).text();
  const pipeline = makeComputePipeline(device, code);
  const prof = new GpuProfiler(device, hasTimestampQuery);

  const n = 1 << 16;
  const inBuf = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outBuf = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE });
  const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }] });

  let gpuMs = NaN;
  const loop = new FrameLoop((dt) => {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass({ timestampWrites: prof.timestampWrites() });
    pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(Math.ceil(n / 64)); pass.end();
    prof.resolve(enc);
    device.queue.submit([enc.finish()]);
    if (hasTimestampQuery) prof.readMs().then((ms) => { gpuMs = ms; });
    overlay.textContent =
      `vector-system foundation\n` +
      `cpu dt: ${(dt * 1000).toFixed(2)} ms\n` +
      `gpu addone: ${Number.isNaN(gpuMs) ? "n/a (no timestamp-query)" : gpuMs.toFixed(3) + " ms"}`;
  });
  loop.start();
  (window as any).__vsBooted = true; // test signal
}

boot().catch((e) => { overlay.textContent = "boot error: " + (e as Error).message; throw e; });
```

- [ ] **Step 3: Write the failing test** — append to `tests/gpu/smoke.spec.ts`

```ts
test("app boots, runs a frame loop, and shows a ms readout without errors", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page.waitForFunction(() => (window as any).__vsBooted === true, { timeout: 10000 });
  await page.waitForTimeout(500); // let a few frames run
  const text = await page.locator("#overlay").innerText();
  expect(text).toContain("gpu addone:");
  expect(errors).toEqual([]);
});
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm run test:gpu -- -g "app boots"`
Expected: PASS — overlay contains "gpu addone:", no page errors.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(host): frame loop + bootstrap + live per-pass ms overlay"
```

---

## Task 7: Foundation run/verify notes

**Files:**
- Create: `.ai/plan/vector-engine-blueprint/foundation-verify.md`

- [ ] **Step 1: Write the verification notes** `foundation-verify.md`

```markdown
# Foundation — run & verify

Dev server:    npm install && npm run dev   → open http://localhost:5173 (overlay shows live ms)
Unit tests:    npm run test
GPU tests:     npm run test:gpu
Rust core:     cargo test

If GPU tests find no adapter: set launchOptions.headless=false in playwright.config.ts to confirm
WebGPU works headed; if it does, the headless flags need adjusting for this machine/Chromium build.
On a machine WITHOUT timestamp-query, the profiler reports n/a and the boot test still passes — the
fluid spike (Plan 3) then falls back to CPU-side wall timing for budget capture, flagged as less precise.
```

- [ ] **Step 2: Commit**

```bash
git add -A
git commit -m "docs: foundation run/verify notes"
```

---

## Self-Review

**Spec coverage (against blueprint):** This plan builds the §4.2 GPU-resource convention (device, ping-pong, dispatch, readback ring, command encoding) and the §7.3 instrument (timestamp profiler) — the two pieces every later subsystem and every fallback ladder depend on. It does NOT build fluid (Plans 2–3), renderer (Plan 4), terrain/scheduler/WorldState (Plan 5+) — those are explicitly deferred and rostered above. No blueprint requirement is silently dropped; the deferrals are named.

**Placeholder scan:** No "TBD/TODO/handle edge cases". Every code step shows complete code; every run step shows the command + expected result. The Playwright headless-flags caveat is a concrete fallback instruction, not a placeholder.

**Type consistency:** `acquireDevice(): GpuContext{device, adapter, hasTimestampQuery}` is consumed consistently in Tasks 1/3/5/6. `GpuProfiler(device, enabled)`, `.timestampWrites()`, `.resolve(encoder)`, `.readMs()` match across Task 5 and Task 6. `PingPong.current/next/swap` and `RingIndex.acquire/advance` match their tests. `runComputePass`/`makeComputePipeline` signatures match Task 3 usage.

**Known risk carried into execution:** headless WebGPU in Chromium is environment-sensitive (driver/flags). Task 1 ships the working flag set + a headed-mode fallback so a failure is diagnosable, not mysterious. This is the first thing to confirm on the target machine.
