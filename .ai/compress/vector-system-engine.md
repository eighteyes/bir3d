# Compressed Context
date: 2026-06-09
session: 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

## Original Task
"exciting suite of mini games based on wasm and webgl. come up with a fundamental engine to drive these experiences" — for the "Vector System" vision doc: three neon-vector systems-games sharing one engine (Bird/flight-reads-wind, Trees/light-competition-over-decades, Emergency-Services-in-Hexland/ambulance-market-sim). Deliverable scoped by user to: blueprint first, then build.

## Locked Decisions (the forks)
- Blueprint-only first → then "git init and start building".
- Shared engine, 3 SEPARATE game builds; cross-game ties = data handoffs (WorldState file), NOT a live shared sim.
- Full hierarchical fluid is BIRD-ONLY; reframed (review-forced) to full-3D in a flock-following MOVING WINDOW + coarse 2.5D weather; 2.5D-stacked-in-window is the realistic BASELINE, full-3D-window is reach (discrete-GPU only). Domain-wide full-3D is FALSIFIED at 60fps (M-series 150GB/s unified bus).
- WebGPU for BOTH compute (Bird fluid + Bird aero) and render (all games). WASM = default CPU compute. Renderer is the one truly-shared core.
- Engine shape = THIN spine: shared = vector renderer + GPU-resource convention + multi-rate scheduler + terrain substrate + WorldState format. Sims bespoke per game. NO shared solver/ECS/field-interpolator.
- Physical form: vs-core (Rust→WASM) + vs-shaders (WGSL) + vs-host (JS/TS owns GPUDevice), consumed by 3 game crates.

## Files Changed (branch build/foundation, 15 commits off main@821f5ca)
Docs (.ai/plan/vector-engine-blueprint/):
- blueprint.md — full engine blueprint; budget-first, epistemic-tagged (MEASURED/DERIVED/ESTIMATED/FALSIFIED); §3 unified frame budget, §6.1 fluid fidelity ladder, §7 shell (tutorial/accessibility/perf-LOD), §8 validate-first.
- context.md — decision trail + research provenance (workflow wf_28b6bd2c-b80, 8 agents).
- plan.md — Plan 1 Foundation (Tasks 0-7) TDD; roadmap of Plans 2-8.
- tasks.md — all Plan-1 tasks checked done; roadmap unchecked.
Code (src/host/):
- gpu/device.ts — acquireDevice()→{device,adapter,hasTimestampQuery}; uncapturederror logger.
- gpu/pingpong.ts — PingPong<T> double-buffer.
- gpu/dispatch.ts — makeComputePipeline, encodeComputePass(caller-owned encoder, batched), dispatchCompute(standalone).
- gpu/readback.ts — RingIndex + ReadbackRing; CRITICAL FIX: enqueue() records copy, afterSubmit() kicks mapAsync (was mapping pre-submit → "used in submit while mapped" dropped frame).
- gpu/profiler.ts — GpuProfiler timestamp-query → per-pass ms; NaN when disabled.
- frameloop.ts — rAF driver. main.ts — bootstrap: device.lost handler + add-one pass/frame + sampled(every-30-frame, non-re-entrant) profiling + overlay.
- shaders/addone.wgsl — trivial compute kernel.
Config: tsconfig.json (noEmit:true), package.json (build=tsc --noEmit && vite build), Cargo workspace + crates/vs-core (cdylib+rlib stub), playwright.config.ts (WebGPU flags: --enable-unsafe-webgpu --enable-features=Vulkan --use-angle=metal), .gitignore.
HUMAN_REVIEW.md (root) — verify steps. .ai/tmp/drive.mjs — Playwright run-driver (gitignored).

## Current State
Foundation Plan 1 COMPLETE via subagent-driven dev (implementer + spec + quality review per task). All green: 3/3 vitest, 5/5 Playwright GPU, 1/1 cargo, tsc clean. App VERIFIED RUNNING live on real Apple Metal: 60fps (cpu dt ~16.7ms), GPU add-one measured 0.016-0.084ms via timestamp-query, zero errors. Headless+real WebGPU both work; timestamp-query available. Biggest project risk (GPU compute + measurement at framerate) empirically retired. Dev server RUNNING in background (task id b14ocj7sn, http://localhost:5173) — user is trying it in browser.

## Open Threads
- MERGE DECISION PENDING: build/foundation → main (opt 1, recommended) / keep branch (3) / discard (4). PR (2) needs a GitHub remote (none configured). main is 16 commits behind.
- Dev server b14ocj7sn still running — STOP IT when user done (TaskStop) or on "stop server".
- Next: Plan 2 (fluid math in Rust, CPU TDD) → Plan 3 (fluid GPU port + feasibility spike §8.1, the make-or-break budget gate) → Plan 4 (vector renderer §4.1) → Plans 5-8 (terrain/scheduler/WorldState/per-game, Bird first).
- wasm32 target NOT installed (Homebrew rust has no rustup) — needed before Plan 2/3 Rust→WASM wiring.
- Optional: /run-skill-generator to capture drive.mjs run-recipe as a project skill.

## Key Decisions (non-obvious)
- Reviews caught 5 real bugs the plan missed: WASM crate-type; dispatch API footgun (split for batched encoder); readback ring device-loss starvation (.catch); profiler dead field; CRITICAL ReadbackRing submit-while-mapped (split enqueue/afterSubmit, hardware-confirmed). The two-phase readback (record pre-submit, map post-submit) is now the convention the fluid spike must follow.
- main.ts profiler reads are SAMPLED (every 30 frames) + non-re-entrant guarded — never await readMs() in the loop; single readBuf races otherwise.
- Bird aero MUST run as a GPU compute pass (reads wind in-shader), NOT WASM reading stale async-readback wind (33-50ms stale).
- Perf numbers in blueprint are mostly [ESTIMATED] — §8 mandates profiling the 2.5D moving-window fluid FIRST (esp. M-series) before committing the fluid architecture.
