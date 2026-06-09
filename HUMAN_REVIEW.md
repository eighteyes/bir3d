# Human Review Steps

## ReadbackRing submit-while-mapped fix
**Date:** 2026-06-09
**Commit:** f36a84c
**Session:** readback-ring-fix-submit-while-mapped

### What was done
- Rewrote `src/host/gpu/readback.ts`: extracted `mapAsync` call out of `enqueue` into new `afterSubmit()` method
- `enqueue()` now ONLY records `copyBufferToBuffer` and stores `pendingMap` slot — no map
- `afterSubmit()` must be called after `device.queue.submit()` — then kicks the non-awaited map
- Added `ReadbackRing copies src to CPU across batched frames` test to `tests/gpu/smoke.spec.ts`

### Pre-conditions
```
cd /Users/god/projects/ai-jank/vector-system
```

### Verify: unit tests still pass
```
npm run test -- readback-ring
```
Expected: 1 test passes.

### Verify: GPU/Playwright tests (all 5)
```
npm run test:gpu
```
Expected: 5 tests pass. New test "ReadbackRing copies src to CPU across batched frames" returns `[10,20,30,40]` with zero page errors.

### Verify: TypeScript clean
```
npx tsc --noEmit
```
Expected: no output (exit 0).

### Watch for
- Any `pageerror` in browser console containing "used in submit while mapped" — means regression
- `ring.read()` returning non-null within 10 frames in the GPU test



## Task 6 — Frame Loop + Bootstrap + Live ms Overlay
**Date:** 2026-06-09
**Commit:** 2f1bb8b
**Session:** task-6-frameloop-bootstrap

### What was done
- Created `src/host/frameloop.ts` — rAF driver with CPU dt tracking
- Created `src/host/main.ts` — full bootstrap: device acquisition, add-one compute pass, live overlay
- `device.lost` handler wired to overlay + console.error
- Sampled profiling: timestamp writes + resolve + readMs only on frames divisible by 30, guarded by `profilePending` flag to prevent racing the single staging buffer
- Added boot test to `tests/gpu/smoke.spec.ts`

### Pre-conditions
```
cd /Users/god/projects/ai-jank/vector-system
```

### Verify: TypeScript clean
```
npx tsc --noEmit
```
Expected: no output (exit 0).

### Verify: Unit tests
```
npm run test
```
Expected: 3 tests pass.

### Verify: GPU/Playwright tests
```
npm run test:gpu
```
Expected: 4 tests pass including "app boots, runs a frame loop, and shows a ms readout without errors".

### Verify: Live in browser
```
npm run dev
```
Open http://localhost:5173 — overlay should display:
```
vector-system foundation
cpu dt: XX.XX ms
gpu addone: X.XXX ms  (or "n/a (no timestamp-query)" if unsupported)
```

### Watch for
- No `[WebGPU uncaptured]` or `[WebGPU lost]` in browser console
- `gpu addone:` line updates approximately every 30 frames (~0.5s at 60fps)
- No Vite 404 warning for `main.ts` (it now exists)
