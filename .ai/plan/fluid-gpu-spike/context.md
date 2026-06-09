# Plan 3 — Context: key files & decisions

**Date:** 2026-06-09 · **Branch:** build/foundation · **Predecessors:** Plan 1 (host GPU convention, verified live on Metal), Plan 2 (CPU fluid oracle, 27 invariant tests, mutation-verified).

## What & why
The make-or-break gate (§8.1): port the verified CPU fluid to WGSL, prove the port faithful vs the oracle, and **measure** whether the moving-window fluid survives its §3 budget on real hardware — replacing `[ESTIMATED]` with `[MEASURED]`. User added a hard requirement: a LIVE watchable debug viz ("I want to see something"), debug-grade (NOT the §4.1 neon renderer — that's Plan 4).

## Decisions (non-obvious, reviewer-forced)
- **Budget gate = measured fluid vs §3 fluid sub-budget (~1.5–2.5ms discrete / ~3.5–6ms M-series), NOT vs 16.6ms.** Comparing against the whole frame would hide an architecture-threatening result (fluid must leave room for render+aero+scheduler). Frame headroom reported separately. Isolated-fluid PASS is optimistic (shared bus + concurrent render inflate on M-series).
- **Correctness ≠ budget.** CPU oracle = 8000 iters; GPU budget affords ~tens → GPU is intentionally under-converged. Validation matches iters (correctness); the sweep measures real iters + residual (budget).
- **Warm-median measurement, platform-labelled.** Cold first frames include shader compile (10–100×); discard ~40, report median, tag adapter. Else the number is noise.
- **Staged validation:** per-kernel after ONE application @1e-5 (real-bug floor) → composed N-step @1e-3 (drift). Localizes port bug vs f32 drift.
- **`set_bnd` two-pass (edges then corners).** Single GPU dispatch races: corner threads read stale edges. Catches the structured 4-corner mismatch the oracle review would otherwise mask under a loosened max-norm.
- **Storage buffers, exact (W+2)(H+2) bordered layout; manual WGSL bilinear; flat 1D dispatch.** Cell-for-cell comparable to the CPU oracle; textures/in-kernel-boundary deferred as optimizations.
- **B reports residual divergence at budget-iters**, not just ms — the real finding is "N iters fit, residual R, is R acceptable?"

## Key files
- Spec: `.ai/plan/fluid-gpu-spike/plan.md` · Oracle: `crates/vs-core/src/fluid/*` (Plan 2)
- Host convention (Plan 1): `src/host/gpu/{device,dispatch,pingpong,profiler,readback}.ts`, `src/host/frameloop.ts`
- Blueprint: §3 budget, §6.1 ladder, §7.3 perf-LOD instrument, §8.1 microbench, §10 risk register (no sync readback).

## Toolchain
- WebGPU verified live on Apple Metal (Plan 1): 60fps, timestamp-query available. Playwright headless WebGPU works on this machine (flags in playwright.config.ts).
- No WASM needed (fluid = WGSL + TS host; oracle stays native Rust). wasm32/rustup still deferred to a later plan.
