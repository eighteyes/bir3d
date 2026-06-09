# Plan 3 — Tasks

Mostly sequential (shared host wiring + each builds on prior). TDD where testable; GPU tests via Playwright on Metal.

- [ ] **Task 1 — fixtures** `examples/dump_fixture.rs` → per-kernel (one-application) + composed (N-step) JSON fixtures committed under `tests/fixtures/fluid/`. Deterministic, seeded.
- [ ] **Task 2 — WGSL + GpuFluid** kernels (advect/divergence/jacobi/subtract_grad/set_bnd two-pass/forces) + `gpu/fluid.ts` step() recording the pass sequence into a caller encoder. Storage buffers, flat 1D dispatch, manual bilinear. Pipeline-build smoke.
- [ ] **Task 3 — correctness gate** `fluid-correctness.spec.ts`: per-kernel @1e-5, composed @1e-3 vs fixtures. Confirm set_bnd corners clean. Harden to green.
- [ ] **Task 4 — budget spike** `fluid-budget.spec.ts`: warm-median per-stage ms, grid×iter sweep, residual via oracle metric, adapter label, vs §3 sub-budget, per-layer. Emit findings.json.
- [x] **Task 5 — live viz** `visualize.wgsl` + `forces.wgsl` + `fluid-main.ts` + `index-fluid.html`: swirling neon dye + overlay (ms/iters/verdict). Boot test: runs, overlay ms, canvas non-blank, no errors. (`fluid-viz.spec.ts` green; 17/17 GPU tests pass on apple/metal-3.)
- [ ] **Task 6 — gate + findings** all GPU tests green; capture REAL Metal numbers; write `SPIKE-FINDINGS.md` (§8.1 verdict + architecture recommendation); HUMAN_REVIEW entry; this checklist updated.
