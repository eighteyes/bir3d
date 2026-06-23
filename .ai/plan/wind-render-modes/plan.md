# Plan — Wind render modes (3 divergent primitives per tier + wind-scaled buffet)

Accepted design: [.ai/explore/2026-06-23-wind-render-modes-design.md](../../explore/2026-06-23-wind-render-modes-design.md)

## Approach
Render-only. Add a per-tier render-MODE switch (far/near/wake → A/B/C) that swaps ONLY the CPU geometry
generation; all nine looks reuse the one existing ribbon pipeline + `wind.wgsl` (zero shader edits). Wake
B/C are new dedicated shed-from-wings geometry written into a third reserved vertex-buffer span. Bird buffet
gains a wind-scaled VISUAL component (render-bank rock + a render-only position tremor) so heavy wind shakes
the bird, not the already-decoupled camera. Feel/physics frozen.

## Phases
1. **Baseline + scaffold** — `FarMode`/`NearMode`/`WakeMode` enums + setters + a per-tier branch in the
   emit; implement only `comet`/`comet`/`modulate` (= today's looks, unified onto the short comet); add 3
   cycle buttons to the T-panel; reserve the third (wake-shed) buffer span (empty); add debug handles.
   B/C branches are no-op stubs. This is the "make all 3 render like the local sphere motes" first move.
2. **Divergent modes** — implement FAR-B Stipple, FAR-C Chevron, NEAR-B Shear Flecks, NEAR-C Curl
   Filaments, WAKE-B Wingtip Helix, WAKE-C Shed Rings + their dials (one task each). Address the two
   watch-points: NEAR-C tail smoothing, WAKE-B/C buffer-span sizing.
3. **Buffet** — `buffetGain`/`buffetWindRef`/`rockCapDeg` in `BirdTuning`; scale rock + a render-only
   tremor by local wind magnitude in `integrate`; apply tremor at `draw` (u[16..18]); T-panel dials.

## Verify (each phase)
`tsc --noEmit` clean → live headless gate (mode-cycle: no pageerror, vertex counts change, 60fps on densest
mode; corner check; wake-shed recycles without buffer overrun; buffet amplitude high-vs-low wind; frozen-feel
guard: existing wind specs stay green) → HUMAN_REVIEW entry + session id.

## Grade: A
Reuses one verified pipeline (no shader/pipeline risk); divergence is real (orthogonal fluid quantities per
tier); buffet is render-only so the frozen-feel guarantee is structural; two watch-points are scoped, not
blockers.
