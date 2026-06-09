# Vector System — Build Task Checklist

Detail for each task (files, TDD steps, complete code) lives in [plan.md](file:///Users/god/projects/ai-jank/vector-system/.ai/plan/vector-engine-blueprint/plan.md). This is the high-level tracker.

## Plan 1 — Engine Foundation (current)

- [ ] Task 0 — Repo scaffold + tooling green (vite/vitest/playwright + cargo workspace; `npm run test` and `cargo test` pass)
- [ ] Task 1 — WebGPU device acquisition + timestamp-query detection
- [ ] Task 2 — Ping-pong double-buffer convention
- [ ] Task 3 — Add-one compute kernel + dispatch helper (pipeline proven end-to-end)
- [ ] Task 4 — Async readback ring (never awaited in-frame)
- [ ] Task 5 — GPU timestamp profiler (per-pass ms instrument)
- [ ] Task 6 — Frame loop + bootstrap + live ms overlay
- [ ] Task 7 — Foundation run/verify notes

## Roadmap — subsequent plans (each its own working/testable increment)

- [ ] Plan 2 — Fluid math in Rust (CPU reference), TDD: grid, advection, divergence, pressure projection, 2.5D coupling
- [ ] Plan 3 — Fluid GPU port + **feasibility spike** (blueprint §8.1): GPU≈CPU parity, moving-window + coarse tier, capture ms vs §3 budget — the make-or-break gate
- [ ] Plan 4 — Vector renderer core (§4.1): submitStroke, ribbon expansion, segment-SDF, bloom
- [ ] Plan 5 — Terrain substrate (§4.4): ENU frame, heightfield, sampling API, CDLOD
- [ ] Plan 6 — Multi-rate scheduler (§4.3): f64 clock, affine sim-clocks, due-queue, event-skip
- [ ] Plan 7 — WorldState handoff format (§4.5): versioned, bit-exact contract
- [ ] Plan 8+ — Per-game: Bird vertical slice first, then Trees, then Hexland
