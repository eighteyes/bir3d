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



## Plan 2 — CPU fluid reference (vs-core)
**Date:** 2026-06-09
**Commit:** 825124d
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What was done
- Built the pure-CPU, deterministic Stam fluid solver in `crates/vs-core/src/fluid/` as the correctness oracle for the Plan 3 GPU port
- `grid.rs` Grid2D + Stam-clamped bilinear sample; `boundary.rs` set_bnd walls/corners (W×H); `advect.rs` semi-Lagrangian backtrace; `project.rs` divergence + Jacobi ping-pong pressure projection; `solver.rs` Fluid2D::step + Fluid25D vertical coupling
- 27 tests total (21 lib unit + 6 cross-module invariant in `tests/fluid_invariants.rs`); all encode physical invariants, not fabricated value-matches
- Scope held: Jacobi ping-pong only (no Gauss-Seidel, no multigrid), no diffusion/viscosity stage; deterministic (no RNG/threads/time)

### Pre-conditions
```
cd /Users/god/projects/ai-jank/vector-system
```

### Verify: full test suite green (run twice for determinism)
```
cargo test -p vs-core
```
Expected: 21 passed (lib) + 6 passed (fluid_invariants) + 0 doc-tests, 0 failed. Identical pass counts on a second run (the project tests take ~17s at N=128, iters=8000).

### Verify: clean build, no warnings
```
cargo build -p vs-core
```
Expected: `Finished dev profile` with zero warnings.

### Watch for
- Divergence tolerance margins are f32 frequency-floor dependent: project tests sit ~1.7× under the 1e-2 bound, the solver step test ~6.5× under. If the grid size or initial-condition fields ever change, re-sweep the Gaussian amp/sigma — NOT the iteration count (the Jacobi residual plateaus at a frequency floor, so more iters will not help).
- Confirm `project.rs` still uses two buffers (`p` / `p_next`) with `std::mem::swap` — in-place Gauss-Seidel would break the 1:1 GPU port.
