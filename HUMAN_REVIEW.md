# Human Review Steps

## Bird 3D v3 (EKG lines-only + ground-locked cam + glide)
**Date:** 2026-06-11
**Commit:** 4783a31
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What was done
- Final SHOW gate for v3 (lines-only). Vision: the v3 section of `.ai/explore/2026-06-10-bird-sandbox-flow.md`.
- Tuned the EKG terrain + camera so the line stack fills the lower frame: steepened the fixed camera downtilt `lookPitch` 16°→28° and tightened `rowSpacing` 65→36 m in `src/host/bird-main.ts`; raised far-row fog `fogDensity` 1/1400→1/1100 so the no-fill back-of-stack dissolves into haze before it tangles.
- Exposed `window.__birdPitch` (bird-main) so the capture harness can wait for a HARD nose-up frame.
- No fill anywhere — terrain is purely additive neon trace lines (`terrain_ekg.wgsl`, line-list); bird is a glide-only V (no flap input). Both depth-tested; ground-locked chase camera (world-up always, decoupled from bird pitch/roll) per `src/host/gpu/camera.ts`.
- Capture script `.ai/tmp/myshot-bird3d.mjs` (gitignored verification artifact): mouse-steers, waits `__birdBooted`, holds nose-up to +40° and saves the hero to `.ai/tmp/v3b-final.png`, plus a streaming motion pair and a banked frame.

### Pre-conditions
```
cd /Users/god/projects/ai-jank/vector-system
```
```
npm run dev
```

### Verify: typecheck clean
```
node node_modules/typescript/bin/tsc --noEmit
```
Expected: no output, exit 0.

### Verify: FLY IT — http://localhost:5173/index-bird.html (mouse steers, no flap, glide)
- [ ] Terrain reads as stacked horizontal neon EKG/oscilloscope trace lines on a dark ground — NO fill, NO shaded surface; lines fill the lower frame, far rows fade into haze with a faint horizon strip at top.
- [ ] The lines ARE the ridges: each line's bumps track the fBm terrain height (not flat scanlines).
- [ ] The terrain MOVES: as you glide forward the lines stream toward the camera and recycle at the horizon (never static).
- [ ] Bird is a clean bright gliding V (hot core, magenta tips, real wingspan), wings held OUT, no flap beat; it banks (rolls) into turns.
- [ ] GROUND STAYS IN FRAME even when you pitch the nose up hard (mouse to top of screen → pitch ~+40°): the camera does NOT follow the bird's pitch.
- [ ] You can sustain/gain altitude on lift: cross a windward ridge → `ridge lift` >0 and vario goes positive without diving (the soar). Level glide sinks gently.
- [ ] 60 fps, no page errors.

### Verify: headless hero capture (proves ground-lock at hard nose-up)
```
node .ai/tmp/myshot-bird3d.mjs
```
Expected: `connected: …5174/index-bird.html`, `hero pitch (deg): 40`, overlay dump with `fps: 60`, `=== errors ===` empty. Writes `.ai/tmp/v3b-final.png` (gliding V over EKG stack, pitch 40°, ground filling the lower frame), plus `v3b-motion-0/1.png` (streaming pair) and `v3b-bank.png`.

### Watch for
- KNOWN LIMITATION (ship-A decision): the EKG rows are world-X-locked (built from `camOffset` only, not camera heading). At heading ~0 they read as clean horizontal stacked lines; while BANKING/TURNING they skew diagonally (`v3b-bank.png`) — geometrically-correct perspective on a world-locked feature, not a regression. If the diagonal-on-turn look is undesirable, the fix is camera-relative rows (lay each row perpendicular to camForward) — deferred pending live feedback.
- The mid-distance line tangle is inherent to the no-fill constraint (no hidden-line removal without a fill); the only lever is far-row fade, already applied. Do not add fill.
- Ridge lift uses the analytic curl-noise wind (FLAGGED stand-in for the GPU fluid); lift bands exist where wind blows into uphill slopes.

## Bird3D — soaring glider physics (energy-exchange model + live tuning panel)
**Date:** 2026-06-11
**Commit:** working tree on top of d0ca3c3
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What was done
- Replaced the velocity-servo flight model in `src/host/gpu/bird3d.ts` with an energy-exchange glider: scalar airspeed; pitch trades speed for altitude (dive to gain, pull up to zoom-climb); ONE-SIDED drag (only bleeds speed above trim — no free thrust back toward trim, so sustained climbs are impossible without lift); sink minimal at trim (~1.4 m/s, L/D ≈ 18), rising CUBICALLY when slow so a stalled nose-up falls instead of levitating; ridge updraft is vertical air motion the bird rides (wind · uphill gradient × liftGain 2.2 — now perceptible vs the old ~5%-of-gravity).
- Reconciled `bird-main.ts` to the glide-no-flap contract (d0ca3c3): TerrainEKG import, no flap input, scripted `__autoWobble` now yields to the player on first mousemove.
- New HUD lines: `vario ±x.x m/s ▲/▼/—` and `ridge lift +x.x m/s`. New tuning panel ('T'): 9 sliders writing live into `bird.tuning` (glideSpeed, sinkRate, divePower, dragK, liftGain, windGain, windDrift, minSpeed, maxSpeed).

### Pre-conditions
```
cd /Users/god/projects/ai-jank/vector-system
npm run dev
```

### Verify: typecheck + unit tests green
```
node node_modules/typescript/bin/tsc --noEmit
node node_modules/vitest/vitest.mjs run
```
Expected: no tsc output; 3 test files / 3 tests pass.

### Verify: page boots headless, integrator alive
```
node .ai/tmp/bird-boot-probe.mjs
```
Expected: `booted: true`, moved ≥10 m, overlay dump, `PROBE_OK`, exit 0. (Headless = no mouse, so the wobble flies it — pitch pegged ±40° is expected there.)

### Verify: FLY IT (the actual gate — feel, eyes + hands)
Open http://localhost:5173/index-bird.html and fly with the mouse only:
- [ ] Level glide sinks gently (~-1.4 m/s vario), not a brick.
- [ ] Mouse-down (dive): airspeed climbs toward 40-55; mouse-up after a dive: zoom-climb, vario strongly positive, speed bleeding off.
- [ ] Hold full nose-up: speed mushes to ~13 m/s and the bird FALLS (no levitation) — stall teaches itself.
- [ ] Cross a windward ridge: `ridge lift` reads >0 and the vario goes positive without diving — circling/tracking the lift band gains altitude. THIS is the soar.
- [ ] Press `T`: sliders move feel live (try liftGain up for stronger soaring, sinkRate up for harsher glide).

### Watch for
- Ridge lift depends on the analytic curl-noise wind (FLAGGED stand-in for the GPU fluid) — lift bands exist where wind blows into uphill slopes; if lift feels too rare, raise windGain or liftGain before judging the model.
- `minClearance` is 6 m and the altitude clamp silently floors the bird — no crash state yet (game layer, not this pass).
- `.ai/tmp/bird-boot-probe.mjs` is a verification artifact (gitignored), not a deliverable.

## Bird sandbox flow (vertical slice) — final SHOW gate
**Date:** 2026-06-10
**Commit:** cba5feb
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What was done
- Final SHOW-gate pass on the rough-but-playable Bird vertical slice: one bird gliding over the existing GPU fluid wind, top-down, toroidal world, deadzone-follow camera, neon chevron + ribbon trail, 4 swappable control schemes, live tuning overlay. Design: `.ai/explore/2026-06-10-bird-sandbox-flow.md`.
- Booted `/index-bird.html` headless (WebGPU, Metal flags) and captured a fresh screenshot `.ai/tmp/bird-final.png`: canvas non-blank, bird present, zero page errors, all 4 schemes switch via keys 1-4, all 8 tuning sliders present.
- No source changes this pass — verification + docs only. The deliverable bird files (committed earlier) are `src/host/shaders/bird/bird_update.wgsl`, `src/host/shaders/bird/scene.wgsl`, `src/host/gpu/bird.ts`, `src/host/bird-main.ts`, `index-bird.html`.

### How to run it
```
cd /Users/god/projects/ai-jank/vector-system
npm run dev
```
Then open http://localhost:5173/index-bird.html (vite may pick 5174 if 5173 is busy).
- Keys `1`-`4` swap control schemes live (label shown top-left):
  - `1` flick to impulse — drag-release on the canvas flings the bird in the drag direction.
  - `2` hold toward cursor — hold the mouse and move it; the bird thrusts toward the cursor.
  - `3` tap to bank — `ArrowLeft`/`ArrowRight` rotate the glide a notch; momentum carries.
  - `4` flap forward — tap `Space` or click for an impulse along the bird's heading.
- The tuning panel (top-right) sliders tune feel live: windCoupling, drag, flick str, thrust str, flap str, bank rate, cam deadzone, cam follow.

### Verify: automated SHOW-gate driver (boot, non-blank, switcher, sliders, clean)
```
node .ai/tmp/bird-final.mjs
```
Expected: `RESULT` JSON with `switcherWorks:true`, `switchSeq:[2,3,4,1]`, `allSliders:true`, `nonBlank:true` (litFraction ~0.57, maxLum ~198), `pageErrors:[]`, `bootsClean:true`; exit 0. PNG written to `.ai/tmp/bird-final.png` (neon chevron + ribbon trail over the dim teal wind backdrop, scheme label top-left, 8-slider tuning panel top-right).

### Watch for
- The driver lives under `.ai/tmp` (gitignored) — it is a verification artifact, not a source deliverable. Re-run it after any change to `bird-main.ts`/`bird.ts` to confirm the switcher and sliders still wire up.
- Known feel caveats (acceptable for a feel proto, not bugs): the fluid's `set_bnd` gives reflective walls, so the wind field is not truly periodic — there is a velocity discontinuity at the world seam even though the bird and camera wrap cleanly. Scheme-2 thrust direction reads `lastPos` (async readback), 1-2 frames stale, so it lags slightly at high cursor speed. Default thrust=90 makes scheme 2 hot; dial it with the thrust slider.

## Bird — control taste-test (schemes 2-4, switcher, tuning overlay)
**Date:** 2026-06-10
**Commit:** cba5feb
**Session:** bird-tastetest-tuning

- Added control schemes 2-4 + live 1-4 switcher to the flyable bird; all map raw input → the `(impulse,turn)` intent for the scheme-agnostic GPU bird pass.
- Scheme 2 hold-toward-cursor (held mouse → `thrust*dt` toward cursor each frame); 3 tap-to-bank (Arrow keys → one-shot turn ±bankRate); 4 flap-forward (Space/click → one-shot impulse along heading). Scheme 1 flick path unchanged.
- Tuning overlay: live HTML range sliders (windCoupling, drag, flick/thrust/flap strength, bank rate, camera deadzone, follow stiffness) feed bird tuning + camera each frame.
- This is a rough feel prototype: verified by booting + screenshotting, not by exhaustive tests.

### Pre-conditions
```
cd /Users/god/projects/ai-jank/vector-system
```
```
npm run dev
```

### Verify: page boots and types clean
```
npm run typecheck
```
Expected: no output (tsc --noEmit passes).

### Verify: open the bird page and fly it manually
Open http://localhost:5173/index-bird.html — a neon chevron bird glides over a dim teal wind backdrop; the tuning panel sits top-right, the scheme label top-left.

### Verify: switch schemes and exercise each
- Press `1`, drag-release on the canvas → the bird flings in the drag direction.
- Press `2`, hold the mouse and move it → the bird thrusts toward the cursor; release → momentum + wind only.
- Press `3`, tap ArrowLeft / ArrowRight → the glide banks (vel rotates) and momentum carries.
- Press `4`, tap Space or click → the bird flaps forward along its heading.
Confirm the top-left label updates each press.

### Verify: tuning sliders move live
Drag the `drag` slider down → the bird coasts longer; drag `windCoupling` up → the wind pushes harder. Each slider's value readout updates as you drag.

### Verify: automated driver (boots clean, switcher, schemes 2/3, tuning, screenshot)
```
node .ai/tmp/bird-tastetest.mjs
```
Expected: `RESULT` JSON with `switcherWorks:true`, `scheme2.pass:true`, `scheme3.pass:true`, `tuning.pass:true`, `nonBlank:true`, `pageErrors:[]`; exit 0. PNG at `.ai/tmp/bird-tastetest.png`.

### Watch for
- The CPU camera reads `lastPos`, which is 1-2 frames stale (async readback) — scheme-2 thrust direction lags slightly at high cursor speed. Fine for a proto.
- Mouse dispatch branches on the active scheme in one set of listeners; if you add a 5th scheme, extend those branches (don't stack parallel listeners).

## Plan 3 — GPU fluid spike (final gate, §8.1)
**Date:** 2026-06-09
**Commit:** 4354316
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What was done
- Final gate for the Plan 3 fluid GPU port: ran all GPU + Rust tests green, regenerated the budget findings with REAL apple/metal-3 numbers, and wrote the §8.1 verdict.
- `.ai/plan/fluid-gpu-spike/SPIKE-FINDINGS.md`: measured fluid ms per stage at each grid/iter, dual ms reporting (instrumented `totalMedianMs` upper bound vs production-representative `wallClockMedianMs`), residual `max|div|`, PASS/MARGINAL/OVER vs the §3 M-series sub-budget (3.5–6ms), the 2.5D ×4-layer projection, the isolation-optimism caveat, and the architecture recommendation.
- Verdict: single-layer fluid PASSES on wall-clock at every useful iter count; the moving-window 2.5D ×4 stack is MARGINAL-to-OVER on honest wall-clock at a usefully-sized + usefully-converged operating point (256², ≥20 iters → ~7ms wall-clock vs 6ms ceiling), before the concurrent-render isolation penalty. Bottleneck is pass-count (set_bnd-dominated), not bandwidth → recommend in-kernel boundary before adding layers, else descope layers/grid/iters.

### Pre-conditions
```
cd /Users/god/projects/ai-jank/vector-system
```

### Verify: full GPU suite green (regenerates budget-findings.json)
```
npm run test:gpu
```
Expected: 17 tests pass on `apple / metal-3`, zero page errors. The budget spec writes `tests/fixtures/fluid/budget-findings.json`.

### Verify: Rust oracle tests green
```
cargo test -p vs-core
```
Expected: 21 (lib) + 6 (fluid_invariants) = 27 passed, 0 failed.

### Verify: read the measured numbers
```
jq -r '.machine.adapterLabel, (.sweep[] | "grid=\(.grid) iters=\(.iters) total=\(.totalMedianMs) wall=\(.wallClockMedianMs) residual=\(.residualMaxDiv) verdict=\(.verdict)")' tests/fixtures/fluid/budget-findings.json
```
Expected: adapter `apple / metal-3`; 8 sweep rows (128²/256² × 10/20/40/80 iters); totals ~1.6–13ms, wall ~0.8–4.8ms, residual ~3–34.

### Verify: see the swirl live (manual, eyes-on)
```
npm run dev
```
Then open `/index-fluid.html` in the browser (or `npm run dev:fluid` to auto-open). Expected: a swirling neon-green dye plume on a dark canvas; overlay reports adapter `apple / metal-3`, grid, iters, per-stage warm-median ms, a `§3 sub-budget` PASS/MARGINAL/OVER verdict, and `cpu dt`.

### Watch for
- The JSON `verdict` field classifies `totalMedianMs` (instrumented UPPER bound, ×2–3 the production cost). Do NOT quote it as the architecture verdict without `wallClockMedianMs` alongside — SPIKE-FINDINGS.md tables both.
- Every ms here is fluid IN ISOLATION. The decisive number is fluid + Plan-4 render on the shared M-series bus; any "2.5D fits" claim stays provisional until measured concurrently.
- Residual `max|div|` (5–34) is deeply under-converged vs the oracle's 8000-iter near-zero; "acceptable residual" is a Plan-4 visual judgment, not a timing fact this spike can settle.

## Fluid live debug viz (Plan 3 — Task 5)
**Date:** 2026-06-09
**Commit:** ad7e484
**Session:** plan3-task5-fluid-viz

### What was done
- `src/host/shaders/fluid/visualize.wgsl`: fullscreen-triangle render pipeline; fragment samples the bordered dye storage buffer, neon-green ramp on dark, luminance clamped to a brightness ceiling (§7.2).
- `src/host/fluid-main.ts`: FrameLoop driving `GpuFluid.step` + a render pass; rotating host-side jet/dye source (deterministic from frame index) for a visible swirl; per-frame render bind-group rebuild from `fluid.dyeField`; cadence-sampled warm-median overlay (adapter/grid/iters/ms/§3-verdict/cpu dt); non-blocking 1×1 center-pixel readback to `window.__centerPixel`; sets `window.__fluidBooted`.
- `index-fluid.html`: canvas + overlay entry (served at `/index-fluid.html`); `dev:fluid` npm script.
- `tests/gpu/fluid-viz.spec.ts`: boots, runs ~60 frames, asserts overlay fluid ms readout + non-blank center pixel + zero page errors.

### Pre-conditions
```
cd /Users/god/projects/ai-jank/vector-system
```

### Verify: viz boot test passes
```
npx playwright test tests/gpu/fluid-viz.spec.ts
```
Expected: 1 test passes.

### Verify: full GPU suite still green
```
npm run test:gpu
```
Expected: 17 tests pass, zero page errors.

### Verify: see the swirl live (manual, eyes-on)
```
npm run dev:fluid
```
Expected: a browser opens `/index-fluid.html` showing a swirling neon-green dye plume on a dark canvas; the overlay reports adapter `apple / metal-3`, grid `128²`, iters `20`, a `fluid: <n> ms (last)` line, a `fluid warm-median: <n> ms` line with a `§3 sub-budget` PASS/MARGINAL/OVER verdict, per-stage ms, and `cpu dt`.

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



## Bird 3D (chase cam + ridgeline terrain) — final SHOW gate
**Date:** 2026-06-11
**Commit:** 12d1e49
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What was done
- Final look validation of the 3D Bird: WebGPU perspective scene (NDC z in [0,1], depth24plus, depthCompare less), flapping-V neon bird in a chase cam over a procedural ridged-fBm neon-ridgeline terrain, fog-as-depth hazing to the horizon.
- Tuned for the look: lifted SKY/fog color to a dim indigo (0.06,0.05,0.12) so receding ridges dissolve into a visible haze band (no hard grid edge / black void); `fogDensity` 1/900 -> 1/700; bird `flapAmp` 0.85 -> 0.55 (reads as a flapping V, not a deep U); wing `DIHEDRAL` 5 -> 7 for a clearer static V mid-flap.
- Confirmed all 5 named elements in the hero still (`.ai/tmp/bird3d-final.png`): flapping-V silhouette, chase view, receding neon ridgelines hazing to a horizon, real 3D perspective depth, AND terrain occluding the bird — a foreground crest cuts the bird's lower body (lower V hidden behind the near ridge, upper wings above the crest line). 60fps, zero page errors. Wingbeat motion separately verified via two shots ~half a beat apart (wings open/close).
- Occlusion capture method: dead-center cursor (no steer, NO flap), settle to the ~20m clamp, then straight level glide so the bird crests a ridge and the far side drops; the post-crest clearance SPIKE frames (telemetry) are where the just-crossed crest sits between camera and bird → occlusion. Tap-flaps lift the bird out of the spike and break the occlusion, so the occlusion sweep must be flap-free. Driver: `.ai/tmp/sweep-occlude.mjs`.

### Pre-conditions
```
cd /Users/god/projects/ai-jank/vector-system
```

### Verify: typecheck clean
```
npx tsc --noEmit
```
Expected: exit 0, no output.

### Verify: live in browser
```
npm run dev
```
Open http://localhost:5173/index-bird.html — mouse steers (cursor offset = yaw + pitch rate, bird banks into turns), click or Space flaps for lift, auto forward drift. Overlay shows altitude-over-terrain, speed, heading/pitch/bank, wind, fps.

### Verify: the look reads
Fly low and near-level so the camera looks ACROSS the ridge field (not down from high altitude). Confirm:
- Flapping-V neon bird (teal-white core, magenta tips) reads as a V, drifting in frame.
- Neon ridgelines (magenta near / teal-blue far) recede and fade into a dim indigo haze band at the horizon — the fog is the depth.
- Real 3D perspective; 60fps; no `[WebGPU lost]` / `pageerror` / `console.error`.

### Watch for
- Pitch is rate-controlled (mouse-y), so it does not self-level — to recover from a dive you must actively pull the nose up; centering the cursor freezes pitch.
- Holding flap continuously is a thrust runaway (impulse every frame) — it is a manual-input artifact, not a bug; tap to flap.
- The TS `sampleHeight` (f64 `Math.sin`) and WGSL fBm (f32 `sin`) diverge by an estimated ~tens of meters near the origin (the `*43758.5453` fract amplifies the f32/f64 sin diff into a different hash; flagged terrain.ts:124). The render is 100% WGSL and occlusion is depth-correct regardless; consequence is only that the bird's ground-clamp / ridge-lift run against a slightly different height than is drawn. Fix if the bird ever looks conspicuously pasted over valleys: `Math.fround` the hash intermediates in TS to match f32.
- Screenshot driver: `.ai/tmp/shoot-bird3d.mjs` (Playwright + Metal WebGPU flags, port 5173 then 5174); hero frame at `.ai/tmp/bird3d-final.png`.
