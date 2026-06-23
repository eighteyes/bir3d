# Context — key files, anchors & decisions

All anchors verified against THIS worktree (not the main repo — they diverge: worktree is FPV=11 with the
committed wind work; main is the older FPV=10).

## Files & anchors
- `src/host/gpu/wind.ts`
  - `static FPV = 11` (L433); `static FAR_SUBDIV = 3` (L440); `CORNERS` 6-vert quad table near the FPV block.
  - Buffer sizing L601-604: `farVertexCount = count*segments*FAR_SUBDIV*6`, `nearVertexCount =
    nearCount*nearSegments*6`, `vertexCount = far+near`, `vertBytes = ArrayBuffer(vertexCount*FPV*4)`.
  - Far emit loop in `step()`; near emit loop in `stepNear()` (the per-segment CORNERS quad write, vis=1).
  - Existing wake field `birdWakeAt()` (L1001+, twin Rankine at ±`wingSpan`, core `vortexCore`, L1030-1036).
  - Seed wingtip branch `seedNearMote` L714-722 (`wingEmitFrac`, `wingSpan`, `wingJitter`); `nearHeat` reset L753.
  - Touched-air heat accumulation L1157-1158. DotParams wake fields L311-313; constructor defaults L562-564.
  - Setters: `setShowNear` (L1086), `setShowWake` (L1087) — ADD `setFarMode`/`setNearMode`/`setWakeMode`.
- `src/host/bird-main.ts`
  - T-panel helpers: `panelSep` (L788), `toggleBtn` (L796), `sliderRow` (L809). Wind tuning section L391-408.
  - ADD a `cycleBtn(panel,label,opts[],initial,onSet)` helper (3-state sibling of `toggleBtn`).
  - Debug handles block (`__wind`, `__windProfile`, …) — ADD `__farMode`/`__nearMode`/`__wakeMode`.
- `src/host/gpu/bird3d.ts`
  - `interface BirdTuning` (L32); defaults `this.tuning = {` (L164). Buffet: `gustV` L417, `gustL` L418,
    velocity shove L424/L426, `rock` L481, `renderBank` assembly L483.
  - `draw()` uniform packing: `u[16..18] = pos` (L511), `u[21]=renderBank` (L514), `u[26]=renderPitch` (L519).
  - Horizontal wind components `wx,wz` are already sampled in `integrate` (used for `crossWind`) → buffet
    intensity reuses them; no new `windAt` call.
- `src/host/shaders/wind.wgsl` — UNCHANGED (verify zero edits at the end).

## Contracts (decisions locked here)
- Enums: `FarMode = "comet"|"stipple"|"chevron"`, `NearMode = "comet"|"flecks"|"filaments"`,
  `WakeMode = "modulate"|"helix"|"rings"`. Fields on `Wind`, default `comet`/`comet`/`modulate`.
- `showNear` still gates the near tier; `showWake` gates the wake; `wakeMode` only acts when `showWake` on.
  `wakeMode="modulate"` = today's near-mote stir (no shed geometry); `helix`/`rings` drive the shed pool.
- Buffer: add a third span `wakeShedReserve = max(helixWorstCase, ringWorstCase)` quads;
  `vertexCount = far+near+wakeShedReserve`. Layout order: far | near | wake-shed. `draw()` issues up to 3
  draws with `firstVertex` offsets (far always; near if `showNear`; wake-shed if `showWake && wakeMode!=modulate`),
  each using the LIVE emitted count for its span (≤ reserve). Spans are NOT a simple prefix (near can be off
  while wake on) → separate draw calls, not one `draw(prefixCount)`.
- New dial fields live on `DotParams` + `Wind` privates (same pattern as `wingSpan`); full per-option dial
  lists are in the spec (don't re-list — wire them as named `sliderRow`s). Reuse where noted (FAR-A reuses
  `segments`/`segStep`/`FAR_SUBDIV`; NEAR-C reuses `swirlGain`/`wingEmitFrac`/`vortexCore`).
- Buffet: `BirdTuning` adds `buffetGain` (default 1), `buffetWindRef` (m/s → full buffet, default ~12),
  `rockCapDeg` (max visual roll, default ~12). In `integrate`: `intensity = clamp(hypot(wx,wz)/buffetWindRef,
  0,1)`; scale `rock` by `(1 + buffetGain*intensity)`, clamp to `rockCapDeg`; build a render-only tremor
  vector `this.buffetOffset` from the g-oscillators × `buffetGain*intensity`. In `draw()` add `buffetOffset`
  to `u[16..18]` (NOT to `this.pos`). Velocity shove (L424/426) UNCHANGED → feel frozen, camera (follows
  `pos`) unmoved.

## Test idiom (this repo)
Headless Playwright specs in `tests/gpu/*.spec.ts` probing `window.__*` handles + asserting no pageerror /
fps. New: `tests/gpu/wind-render-modes.spec.ts` (cycle `__farMode`/`__nearMode`/`__wakeMode`; assert vertex
counts change, no crash, fps>15, wake-shed pool bounded) and `tests/gpu/bird-buffet.spec.ts` (assert
`__birdBank` variance higher at high local wind than low; camera aim variance ~flat; `pos` path unchanged by
the render-only tremor).

## Verify gotchas
- Fresh vite from THIS worktree; curl-verify the served bundle before trusting a test.
- Frozen-feel guard: re-run existing `wind-atmosphere` / `updraft-buffer` / `wind-live` specs — must stay green.
- Wake-shed pool must recycle by age (no unbounded growth) and never exceed its reserved span.
