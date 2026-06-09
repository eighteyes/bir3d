# Plan 3 — Fluid GPU port + feasibility spike (§8.1) + live debug viz

> Roadmap (vector-engine-blueprint/plan.md): "Plan 3 — Fluid GPU port + feasibility spike (§8.1): port kernels to WGSL, validate GPU≈CPU via readback, moving-window + coarse-tier coupling, capture per-pass ms vs the §3 budget on this machine. THE make-or-break gate."
> User requirement (2026-06-09): "I want to see something" → ship a LIVE debug visualization of the fluid (watchable swirl) alongside the per-pass ms — visuals are debug-grade, NOT the §4.1 neon renderer (that is Plan 4).
> REQUIRED SUB-SKILL for executors: superpowers:subagent-driven-development / test-driven-development.

**Goal:** Port the Plan-2 verified CPU fluid to WGSL compute, prove the port faithful against the CPU oracle, **measure the real per-pass ms on this machine and check the fluid architecture survives its §3 sub-budget**, and render a live watchable debug viz. This replaces the blueprint's `[ESTIMATED]` fluid numbers with `[MEASURED]` ones — the central feasibility question of the whole project.

## The load-bearing distinction: correctness ≠ budget
- **Correctness:** GPU kernels reproduce the CPU oracle at MATCHED params. The CPU oracle is the reference (Plan 2).
- **Budget:** at a realistic window grid, how many Jacobi iters fit the frame, and what residual divergence is left. The CPU oracle used 8000 iters; GPU at 60fps affords ~tens — so the GPU fluid is deliberately UNDER-CONVERGED vs the oracle. Validation matches iters; the budget sweep measures real iters. Never conflate them.

## Budget threshold (do NOT compare against 16.6ms)
16.6ms is the WHOLE frame (render + aero + scheduler + fluid). The spike's verdict is measured-fluid vs the **§3 fluid sub-budget**:
```
                    fluid 2.5D moving-window line item (§3)
RTX-3060 class      ~1.5–2.5 ms   [ESTIMATED → replace with MEASURED]
base M-series       ~3.5–6 ms     [ESTIMATED → replace with MEASURED]  (mandatory 2.5D baseline)
```
Report: measured fluid ms (per stage) vs this sub-budget → PASS / MARGINAL / OVER; AND total-frame headroom (16.6 − fluid) separately. **Honesty note in the verdict:** this is fluid IN ISOLATION; on M-series unified memory the bus is shared, so the concurrent Plan-4 render will inflate it — an isolated PASS is necessarily optimistic.

## Measurement discipline (or the number is noise)
- **Warm, not cold:** discard the first ~30–60 frames (shader compile / pipeline creation is 10–100×), report the **median** over a steady-state window.
- **Label the platform:** tag every ms number with the adapter from `acquireDevice` (Apple Metal here). A fluid ms without a platform label is uninterpretable against a platform-split budget.
- **No synchronous readback in the loop** (risk register HIGH; the budget model assumes pipelining). Validation/residual readback is ONE-SHOT outside the hot loop.

## Conventions (lock these)
- Reuse Plan-1 host convention: `makeComputePipeline` / `encodeComputePass(device, encoder, pipeline, bindings[], workItems, wg=64)` (bindings[i] → `@group(0) @binding(i)`, batched into one caller-owned encoder + one submit), `PingPong`, `FrameLoop`, `acquireDevice`.
- **Storage buffers, exact `(W+2)*(H+2)` bordered layout matching the Rust oracle** (1-cell border, `idx = i + (W+2)*j`). Makes GPU≈CPU validation cell-for-cell. **Manual bilinear in WGSL** (NOT texture hardware sampling — must match the CPU bilinear + Stam clamp exactly).
- **Flat 1D dispatch:** `workItems = (W+2)*(H+2)`, `@workgroup_size(64)`; kernel derives `i = gid % (W+2)`, `j = gid / (W+2)`, guards interior `1..=W, 1..=H`.
- Params (W, H, dt, …) via a uniform buffer bound alongside the storage buffers.
- **Jacobi ping-pong** (two pressure buffers, swap per sweep) — same algorithm/sign as the verified oracle.
- **`set_bnd` as TWO passes** (edges, then corners) so corner cells read the freshly-written edges — matches CPU ordering and avoids the single-dispatch corner race.
- Every TS/WGSL/Rust file starts with the project comment block. Terse, no personality in work files.

## File structure
```
crates/vs-core/examples/dump_fixture.rs   Rust: run deterministic scenarios, emit JSON {input, expected} to stdout
tests/fixtures/fluid/*.json               committed CPU-oracle reference fixtures (per-kernel + composed)
src/host/shaders/fluid/advect.wgsl        semi-Lagrangian + manual bilinear (matches grid.rs sample)
src/host/shaders/fluid/divergence.wgsl
src/host/shaders/fluid/jacobi.wgsl        one Jacobi sweep (ping-pong)
src/host/shaders/fluid/subtract_grad.wgsl
src/host/shaders/fluid/set_bnd.wgsl       entry points for scalar/velx/vely × edges/corners
src/host/shaders/fluid/forces.wgsl        add scripted force + dye injection (for the viz)
src/host/shaders/fluid/visualize.wgsl     fullscreen-quad fragment: sample dye/velocity → neon ramp
src/host/gpu/fluid.ts                     GpuFluid: buffers, pipelines, step(encoder, dt) records the pass sequence
src/host/gpu/passtimer.ts                 multi-stage timestamp timer (querySet count=2*K → ms[] per stage)
src/host/fluid-main.ts                    bootstrap: GpuFluid + scripted force + viz render + warm-median overlay
index-fluid.html                          entry for the fluid spike (canvas + overlay)
tests/gpu/fluid-correctness.spec.ts       per-kernel @1e-5, composed @1e-3, vs fixtures
tests/gpu/fluid-budget.spec.ts            warm-median per-stage ms + iter/grid sweep + residual; emits findings JSON
tests/gpu/fluid-viz.spec.ts              boot: app runs, overlay shows fluid ms, canvas non-blank, no errors
```

---

## Task 1 — CPU oracle reference fixtures (Rust example → JSON)
`examples/dump_fixture.rs` runs deterministic (no RNG), seeded (sinusoid) scenarios and emits JSON to stdout; generated files committed under `tests/fixtures/fluid/`:
- Per-kernel, ONE application each: `advect`, `divergence`, one `jacobi` sweep, `subtract_grad`, `set_bnd` (scalar / velx / vely). Each fixture: `{w, h, params, inputs:{...}, expected:{...}}`.
- Composed: `Fluid2D::step` × N steps (small grid, modest iters), `{ic, params, expected}`.
Calls the existing `pub` kernel fns directly. Document the generator command in the file header.
**Test:** `cargo run -p vs-core --example dump_fixture` produces valid JSON; commit the fixtures.

## Task 2 — WGSL kernels + `GpuFluid` host module (compute only)
Port each kernel to WGSL (storage buffers, flat 1D dispatch, manual bilinear, uniform params). `GpuFluid`: allocate buffers (u, v, dye ping-pong; p ping-pong; div; params uniform), build pipelines, `step(encoder, dt, iters)` records: add-forces → set_bnd(vel) → project[ divergence → (jacobi sweep → set_bnd scalar) ×iters → subtract_grad ] → set_bnd(vel) → advect(vel through clones) → set_bnd(vel) → project → advect(dye) → set_bnd(scalar). All into a caller-owned encoder (one submit/frame). `set_bnd` two-pass (edges then corners). No readback here.
**Test:** pipelines build without validation error (a minimal Playwright pipeline-creation smoke).

## Task 3 — Correctness gate (GPU≈CPU) — THE port-trust gate
`fluid-correctness.spec.ts`: for each per-kernel fixture, upload inputs, run that ONE kernel on GPU, read back, assert `max|GPU−CPU| < 1e-5` (real-bug tol). For the composed fixture, run N matched steps, assert `< 1e-3` (drift tol). Staged so a failure localizes to a kernel vs f32 drift. Expect/verify the `set_bnd` corner case is handled by the two-pass split (no 4-corner structured mismatch). Harden until green.

## Task 4 — Budget spike (§8.1) — THE make-or-break measurement
`fluid-budget.spec.ts`: run `GpuFluid` headless at representative window grids (e.g., 128², 256²), single 2D layer. **Warm-median:** discard first ~40 frames, median per-stage ms (advect, divergence, jacobi-total, subtract_grad, set_bnd, TOTAL) over a window. **Sweep** Jacobi iters (e.g., 10/20/40/80) × grid size. For each cell, one-shot readback → compute **residual `max|div|`** via the oracle metric. Tag adapter label. Emit `findings.json`. Compare TOTAL fluid ms vs the **§3 fluid sub-budget** (1.5–2.5 / 3.5–6), not 16.6ms → PASS/MARGINAL/OVER + isolation-optimism note. Report per-layer ms so 2.5D = ×N_layers is one multiply away.

## Task 5 — Live debug viz (the "see something")
`visualize.wgsl` fullscreen-quad fragment samples dye (and/or velocity magnitude) → neon-green ramp (respect a brightness ceiling per §7.2). `forces.wgsl` scripted continuous vortex/jet force + dye injection so the field swirls deterministically. `fluid-main.ts` + `index-fluid.html`: `FrameLoop` driving `GpuFluid.step` + render + an overlay showing adapter, grid, iters, per-stage warm-median ms, fluid-vs-subbudget verdict, cpu dt. Dev script `dev:fluid` (or reuse vite root) serves it.
**Test:** `fluid-viz.spec.ts` — boots, runs N frames, overlay contains "fluid" ms, canvas is non-blank (sample a pixel), zero page errors.

## Task 6 — Final gate + findings
Run all GPU tests green. **Capture the ACTUAL measured numbers on this Apple Metal machine** (run the budget spec, record the medians + residuals). Write `.ai/plan/fluid-gpu-spike/SPIKE-FINDINGS.md`: the §8.1 verdict — measured fluid ms per stage @ each grid/iter, residual at budget-iters, PASS/MARGINAL/OVER vs §3 sub-budget, isolation-optimism caveat, per-layer → 2.5D projection, and a recommendation (does the moving-window 2.5D architecture survive, or does it need descope?). Update `tasks.md`; append HUMAN_REVIEW.md with verify commands + session id.

## Carried-forward flags
- Textures (vs storage buffers) and in-kernel boundary are bandwidth optimizations deferred until the spike says they're needed.
- wasm32/rustup still deferred (no WASM in this plan; fluid is WGSL + TS host).
- 2.5D vertical coupling on GPU is single-layer-projected here; full stacked GPU coupling is the next increment if the budget survives.
