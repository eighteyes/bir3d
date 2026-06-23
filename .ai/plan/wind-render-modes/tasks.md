# Tasks — Wind render modes + wind-scaled buffet

Spec: [../../explore/2026-06-23-wind-render-modes-design.md](../../explore/2026-06-23-wind-render-modes-design.md) ·
Context: [context.md](context.md)

## Phase 1 — Baseline + scaffold
- [ ] Add `FarMode`/`NearMode`/`WakeMode` types + `farMode`/`nearMode`/`wakeMode` fields on `Wind` (default `comet`/`comet`/`modulate`) + `setFarMode`/`setNearMode`/`setWakeMode`
- [ ] Branch the far emit (`step()`) on `farMode` and near emit (`stepNear()`) on `nearMode`; only `comet` implemented, B/C = no-op stubs
- [ ] Unify FAR-A onto the short comet (clamp `segments`→4, `FAR_SUBDIV`→2 path; sparse scatter) — verify no corners at distance
- [ ] Reserve the third wake-shed buffer span (`wakeShedReserve`); `vertexCount = far+near+wakeShed`; refactor `draw()` to up-to-3 `firstVertex` draw calls (far always; near if `showNear`; wake-shed if `showWake && wakeMode!=modulate`)
- [ ] Add `cycleBtn` helper (bird-main.ts) + three mode-cycle buttons under the wind sections; add `__farMode`/`__nearMode`/`__wakeMode` debug handles
- [ ] `tsc --noEmit` clean; live gate: cycle all modes (B/C stubs no-op), no pageerror, 60fps; commit

## Phase 2 — Divergent modes (one task each; dials per spec, wired via `sliderRow`)
- [ ] FAR-B Stipple Streamline — K disconnected dashes along the coarse flow polyline (no shared joints); dials dashCountK/dashLenM/gapRatio/lenByAltitude/leadBoost
- [ ] FAR-C Arrowhead Chevron — 2-limb glyph, one flow sample, apex bright; dials spreadAngleDeg/limbLenM/widthPx/apexBoost/rakeBySpeed
- [ ] NEAR-B Shear Flecks — 2-seg velocity-oriented dash, length/brightness by local shear (finite-diff reuses head sample); dials fleckLen/shearGain/shearRadius/fleckTaper/orientLerp/fleckCount. WATCH: flat `along` remap is new near code
- [ ] NEAR-C Curl Filaments — 5-seg tail integrated through `flowAt+birdWakeAt`, wingtip-seeded, corkscrews the cores; dials filSegStep/filCount + reuse swirlGain/wingEmitFrac/vortexCore/nearSegments. WATCH: add tail smoothing only if kink shows
- [ ] WAKE-B Wingtip Helix — new persistent shed pool (2 tips × ~120), short backward arcs via the existing curl loop, counter-rotating; dials wakeEmitRate/wakeLife/helixGain/wakeSeg/wakeSegStep/wakeTubePx/wakeTaper/counterRotate. WATCH: span sizing
- [ ] WAKE-C Shed Rings — new ring pool (~16-32), grow+convect, tessellate N chords/ring into the shed span; dials ringRate/ringGrow/ringLife/ringSegN/ringTubePx/ringStartRadius/ringTilt/twinOffset/convectFrac/ringWarmBias
- [ ] `tests/gpu/wind-render-modes.spec.ts`: cycle every mode, assert vertex counts change, no crash, fps>15, wake-shed pool bounded ≤ reserve
- [ ] `tsc` clean; live gate (corner check on FAR-B/C; wake-shed appears + recycles); commit

## Phase 3 — Wind-scaled visual buffet
- [ ] `BirdTuning` += `buffetGain` (1) / `buffetWindRef` (~12) / `rockCapDeg` (~12) + defaults
- [ ] `integrate`: `intensity = clamp(hypot(wx,wz)/buffetWindRef,0,1)`; scale `rock` by `(1+buffetGain*intensity)` capped at `rockCapDeg`; build render-only `this.buffetOffset` from the g-oscillators × `buffetGain*intensity` (velocity shove UNCHANGED)
- [ ] `draw()`: add `this.buffetOffset` to `u[16..18]` (NOT `this.pos`)
- [ ] T-panel "bird — buffet" section: `sliderRow` buffetGain/buffetWindRef/rockCapDeg
- [ ] `tests/gpu/bird-buffet.spec.ts`: `__birdBank` variance high-wind > low-wind; camera aim variance ~flat; `pos` path unchanged by tremor
- [ ] `tsc` clean; live gate; commit

## Close-out
- [ ] Frozen-feel guard: existing `wind-atmosphere` / `updraft-buffer` / `wind-live` specs green
- [ ] Confirm `src/host/shaders/wind.wgsl` has ZERO edits
- [ ] HUMAN_REVIEW.md entry per phase (T-panel checklist per tier) + session id
