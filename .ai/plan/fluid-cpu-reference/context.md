# Plan 2 — Context: key files & decisions

**Date:** 2026-06-09 · **Branch:** build/foundation · **Predecessor:** Plan 1 (foundation) COMPLETE & verified live on Metal.

## Where this fits
Plan 2 of the Vector System engine roadmap. Builds the CPU fluid reference inside `vs-core`. It is consumed by **Plan 3** (GPU/WGSL port + §8.1 feasibility spike — "the make-or-break gate"), which validates GPU≈CPU against this code via readback.

## Decisions (non-obvious, reviewed)
- **CPU-first is correct order**, not a validate-first violation: the GPU port needs a correctness oracle to validate against, and the oracle has zero GPU-budget uncertainty.
- **Jacobi only, multigrid DEFERRED** (advisor-forced). Multigrid is convergence optimization, not correctness; whether it's needed is what the Plan 3 budget spike answers; Jacobi ping-pong is the clean GPU-port target. CPU compensates with more iterations.
- **2.5D coupling minimal** (vertical Laplacian mixing, Neumann ends) — reference, not a tuned weather scheme.
- **No diffusion/viscosity stage** — wind viscosity ≈ 0; same Jacobi machinery covers it later if needed.
- **Tests = physical invariants, not fabricated value-matches.** This is the load-bearing decision: a shape-only test lets the oracle be silently wrong and Plan 3's GPU≈CPU check matches a broken reference. Invariant-testing IS the TDD here (you cannot hand-compute semi-Lagrangian advection to exact values).
- **Conventions locked:** collocated grid, cell units `h=1`, `(W+2)*(H+2)` storage with 1-cell border, Stam clamping `[0.5, N+0.5]`, deterministic (no RNG/threads/time).

## Key files
- Spec: `.ai/plan/fluid-cpu-reference/plan.md`
- Target crate: `crates/vs-core/` (currently only `lib.rs::abi_version`)
- Blueprint refs: `§3` frame budget, `§6.1` fluid fidelity ladder, `§8` validate-first.

## Toolchain reality
- Rust 1.89 Homebrew, cargo present. Installed target: `aarch64-apple-darwin` only. **No rustup, no wasm-pack.**
- Plan 2 needs **none** of the missing pieces — pure native `cargo test`.
- **Forward flag:** Plan 3+ Rust→WASM needs wasm32; without rustup that's NOT a one-liner. Resolve when Plan 3 starts.
