# Human Review Steps

## Bird 3D v7 (grouped wind gusts)
**Date:** 2026-06-11
**Commit:** 65d811b (v7 wind code); HUMAN_REVIEW + spec on top
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What was done
- WIND MOTES ONLY (`src/host/gpu/wind.ts`, `src/host/shaders/wind.wgsl`); bird/camera/terrain/physics untouched.
- GROUPED INTO GUSTS: motes organized as drifting CLUSTERS — `numClusters=70` cluster centers seeded across the camera-relative span (view-wedge lateral spread), each carrying `motesPerCluster=60` motes scattered within `clusterRadius=28 m` (4200 motes total). Both centers and members advect by the SHARED `windAt`; a cluster recycles AS A UNIT (reseeded ahead, members re-scattered) when its center ages out (45 s) or leaves the span — members never recycle individually, so packets stay discrete instead of bleeding into an even speckle.
- MORE + SMALLER: ~4200 motes (up from v6) at `dotPx=2.6` (smaller than v6's head).
- LONGER TAILS: `tailMul=11` → comet streak ≈ 11× head width; drift direction reads in a still.
- Keeps: advection by the same `windAt` the bird flies, depth-test against terrain (ridges occlude motes), additive neon, overlay compass. No synchronous readback in the frame loop.

### Pre-conditions
```
cd /Users/god/projects/ai-jank/vector-system
npm run dev
```

### Verify: page boots, 60fps, zero errors
```
node .ai/tmp/myshot-v7-final.mjs
```
Expected: `connected: ...`, telemetry lines, `=== errors === \n none`, exit 0. Saves `.ai/tmp/v7-final-0.png` and `.ai/tmp/v7-final-1.png` ~0.8s apart.

### Verify: wind reads as drifting GUSTS, not a starfield
Open the saved pair and confirm by eye:
- Motes CLUMP into discrete gusts with dark gaps between — NOT an even all-over speckle.
- Each mote is a tiny cyan COMET with a long fading tail, all oriented along the wind.
- The gust field visibly SHIFTS between frame 0 and frame 1.

### Verify: clustering actually reads (decisive A/B — do NOT use a grid CV/occupancy stat)
A single-frame grid CV / cell-occupancy number does NOT discriminate gusts from a uniform field:
sky-band emptiness + distance fog + perspective concentration alone produce occupancy ~65-75% and
CV ~1.6-1.7 with ZERO clustering (verified — the uniform control scored 74% / 1.58 vs the clustered
65% / 1.72). Use a relative A/B instead:
```
# temporarily make members scatter field-wide (= uniform control), shoot, then revert
sed -i '' 's/p.clusterRadius ?? 28/p.clusterRadius ?? 900/' src/host/gpu/wind.ts
node .ai/tmp/myshot-v7-final.mjs    # this overwrites v7-final-*.png with the UNIFORM control
git checkout src/host/gpu/wind.ts   # restore clustered defaults
node .ai/tmp/myshot-v7-final.mjs    # re-shoot the real CLUSTERED frames
```
Crop both (`(40,360,520,600)`, NEAREST 3×) and compare by eye: the clustered crop must be visibly
KNOTTIER (multiple streaks bunched together) with LARGER empty gaps than the evenly-spread uniform
control. If they look the same, clustering is not reading — tune (`numClusters` down, `clusterRadius`
down, `motesPerCluster` up to hold density) and re-shoot.

### Verify: prior wins intact
- Small gliding-V bird dwarfed by the big EKG ridgeline terrain (chase cam keeps it centered).
- Near ridge crests OCCLUDE motes and far ridge rows behind them; elevation color teal-to-magenta.
- Glider sinks by default (`vario` negative); compass shows heading / ground-track / wind with a drift readout.

### Watch for
- A frozen PNG shows position + density only — judge cluster MOTION across the pair (or live), not a single still.
- If motes ever read as an even speckle again: fewer clusters (`numClusters` down) for more separation, keep density via `motesPerCluster` up, tighten `clusterRadius`.
- Screenshot driver: `.ai/tmp/myshot-v7-final.mjs` (Playwright + Metal WebGPU flags, port 5174 then 5173); waits `window.__birdBooted`.

## Bird3D — depth-to-ground cues for dramatic swoops (adaptive cam + FOV kick + plumb-line)
**Date:** 2026-06-11
**Commit:** working tree on top of 768af6f
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What was done
- ALTITUDE-ADAPTIVE CHASE CAM (`src/host/bird-main.ts`): clearance ≤25 m → eye drops to 10 m above the bird, look flattens to 8° (ground rushes); clearance ≥160 m → exact v3/v4 framing (55 m / 28°), linear blend between, smoothed by the existing cam easing. High-altitude capture harnesses are unaffected.
- SPEED FOV KICK: FOV eases 60°→76° as airspeed runs trim→maxSpeed — dives visibly widen the view.
- GROUND PLUMB-LINE (`src/host/gpu/marker.ts` + `src/host/shaders/marker.wgsl`, new): dashed neon drop-line bird→terrain-below (one dash ≈ 9 m — the dash count IS the altimeter) + pulsing ground diamond at its foot. Additive, depth-tested (ridges occlude it = extra parallax), depth-write off. Drawn last in the frame encoder.

### Pre-conditions
```
cd /Users/god/projects/ai-jank/vector-system
npm run dev
```

### Verify: typecheck clean + page boots headless
```
node node_modules/typescript/bin/tsc --noEmit
node .ai/tmp/bird-boot-probe.mjs
```
Expected: no tsc output; `booted: true`, `PROBE_OK`, exit 0.

### Verify: eyes-on screenshots (high vs low MUST look different)
```
node .ai/tmp/marker-shot.mjs
```
Expected: `errors: []`. `.ai/tmp/marker-high.png` (≈190 m: long dashed plumb-line spanning the frame, god-view framing) vs `.ai/tmp/marker-low.png` (≈6 m: stub line + diamond under the wingtips, camera near the deck, ridge ahead at eye level).

### Verify: FLY THE SWOOP (the actual gate)
Open http://localhost:5173/index-bird.html:
- [ ] Cruise high: long dashed drop-line below; count of dashes shrinks as you descend.
- [ ] Full dive toward a valley: FOV widens with speed, camera sinks toward the bird, terrain lines accelerate past — the ground RUSHES.
- [ ] Pull up at the deck (<25 m): diamond right under you, camera low and flat, near-ridge crests cross above the bird — then zoom-climb out and the god-view eases back.
- [ ] No pop: camera height/angle and FOV all ease, never snap.

### Watch for
- The plumb-line samples terrain via `bird.lastClearance` (same sampleHeight as physics) — if bird and line ever disagree visually, suspect a terrain mesh/sampleHeight divergence, not the marker.
- Marker tunables are constants in `marker.wgsl` (DASH_M=9, diamond scale 7±1.5) and `bird-main.ts` (CAM_LOW/CAM_HIGH, FOV_KICK 16°) — adjust there if the feel is close-but-not-quite.
- `.ai/tmp/marker-shot.mjs` is a verification artifact (gitignored), not a deliverable.

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

### Verify: lift-sustain (climbs on lift WITHOUT diving)
```
node .ai/tmp/probe-lift.mjs
```
Expected: a `LIFT-SUSTAIN FRAME` line with `ridge lift >0`, `vario >0`, `pitch >= 0` (e.g. lift +1.2, vario +3.2, pitch +8°, airspeed steady ~25) and `.ai/tmp/v3b-lift.png`. This is the soar — rising on the updraft, not trading airspeed in a dive/zoom. (The +40° hero frame proves ground-lock, NOT lift; in that frame the climb is a zoom-climb with airspeed bleeding and ridge lift 0.)

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

---

## Bird 3D v5 (denser occluding terrain + elevation color + wind dots)
**Date:** 2026-06-11
**Commit:** 95b6859
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What was done
- **Terrain 2× density:** EKG rows 64→128, cols 256→512, rowSpacing 36→18 m → ~2× visible rows inside the maxDist cutoff and finer ridge profiles. Still camera-relative + screen-horizontal at every heading (terrain.ts / terrain_ekg.wgsl).
- **Black-fill hidden-line occlusion:** per-row OPAQUE curtain from each ridge line DOWN to a low baseline, colored the SKY background, drawn FIRST with depthWrite ON (vsFill/fsFill). Lines drawn AFTER with depthCompare less-equal + depthWrite OFF. A near curtain writes nearer depth and occludes the lines of farther rows → Joy Division hidden-line removal, no horizon tangle. NOT the row-to-row shaded mesh the user rejected — each row is its own vertical curtain at constant depth.
- **Elevation color hints:** lines tinted by terrain height — deep teal/blue valleys → magenta mid-slopes → hot near-white peaks (two-stage smoothstep ramp, brightness-capped) so color reads elevation.
- **Wind = DOTS, not lines:** wind.ts/wind.wgsl replaced the streamline comets with a persistent field of drifting neon DOT particles. Each mote's world position is advected CPU-side by the SAME `windAt` field (p += w·dt) and recycled when it leaves the camera-relative span (reseeded ahead). Rendered as additive billboard quads, depth-tested (no write) so ridges occlude them. Compass/wind-vector overlay retained.
- **Tuning (this commit):** dot `count` 900→1300, `dotPx` 7→11, glow falloff 2.2→1.6 + base intensity 0.55→0.85 (motes read as dots, not pinpricks); `spanAhead`/`spanWide` 1400→950 to match terrain `maxDist` (dots past the cutoff floated over a void and read as detached sky specks); `clearance` 45→55 (clear near crests but hug the ridges, not the pure-sky band); terrain `fogDensity` 1/700→1/550 (far rows dissolve before they compress at the horizon — at 2× density adjacent rows barely self-occlude).
- Flight physics (bird3d.ts integrate — glider sinks by default, lift is local) and the chase camera were NOT touched.

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
Open http://localhost:5173/index-bird.html (vite may use 5174 if 5173 is taken). Mouse = steer (down dive/speed, up zoom-climb); the glider sinks by default and you must hunt lift.

### Verify: the look reads
Fly a moment with the cursor near screen-center, then confirm:
- Denser EKG terrain — many fine horizontal neon ridge lines.
- Near ridge crests cleanly SEVER the lines behind them — no horizon tangle; far rows dissolve into the haze.
- Elevation color — teal/blue valleys ramping to magenta/white peaks.
- Wind shown as drifting cyan DOTS (NOT lines) floating over the terrain — flow reads through their drift + density.
- Readable gliding-V bird, ground framed, compass overlay (heading / track / wind, drift readout).
- 60fps; no `[WebGPU lost]` / `pageerror` / `console.error`.

### Watch for
- The wind-dot drift is invisible in a single still frame — judge the flow in MOTION (a frozen PNG shows position + density only). Sky-band dots above the terrain silhouette read as specks when frozen but as flow when moving.
- Dot `clearance` below ~50 risks re-bunching motes under the near fill curtains (they get depth-occluded) — 55 is the readable floor now that span/glow are fixed.
- Screenshot driver: `.ai/tmp/myshot-v5-final.mjs` (Playwright + Metal WebGPU flags, port 5174 then 5173); waits `window.__birdBooted`; final frame at `.ai/tmp/v5-final.png`.

---

## Bird 3D v8 (wind everywhere, speed-driven density+tail)
**Date:** 2026-06-12
**Commit:** 8b35b60
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What was done
- **Wind EVERYWHERE (no clusters).** Dropped v7's hard mote-clustering that left big empty dark gaps. Motes (`numMotes` 4200) are now seeded UNIFORMLY across the whole camera-relative wedge (`seedMote`), so the entire airspace shows airflow. `windAt` is divergence-free, so a uniform seed STAYS uniform under advection — no clumping, no gaps. Motes that leave the wedge are boundary-WRAPPED (front/back exit → reseed near the far edge; side exit → reseed the opposite side) so coverage stays full with no mid-view pop-in.
- **DENSITY ∝ SPEED.** Each mote gets a stable per-mote hash rank (0..1) in the shader; it survives the speed-fade only if `rank < densityFloor + (1-densityFloor)·speedFrac`. Fast air (high `|windAt|`) keeps far more motes; calm air keeps a faint floor (`densityFloor` 0.18 → ~18% survive) so wind reads EVERYWHERE, just sparser where slow. The cutoff is smoothstepped so motes fade in/out across speed contours instead of popping.
- **TAIL LENGTH ∝ SPEED + longer base tail.** Each comet tail scales from a calm-air stub (`tailFloor` 0.2 × base) up to the full base tail in fast air. Base tail lengthened well beyond v7 (`tailMul` 40, vs v7's ~11–16) so fast lanes read as clear on-screen streaks (~52px) while calm air stays short stubs. `speedFrac` is a calibrated `smoothstep(speedLo 2, speedHi 15)` over the field's real `|windAt|` min/max so calm→fast spans the full 0..1 and the contrast reads.
- **Net read:** fast air = dense long bright streaks; calm air = sparse faint short stubs — viewer reads SPEED off density + tail length, everywhere.
- Same advection by the shared `windAt`; depth-test vs terrain (ridges occlude motes), additive neon, overlay compass, 60fps. NO synchronous readback in the frame loop.
- ONLY `wind.ts` / `wind.wgsl` changed. Bird flight, camera, and terrain were NOT touched.

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
Open http://localhost:5174/index-bird.html (vite falls back to 5173 if 5174 is taken). Mouse = steer; the glider sinks by default and you hunt lift.

### Verify: the look reads
Fly a moment with the cursor near screen-center, then confirm:
- Wind motes fill the WHOLE airspace — left to right, near to far — with NO large empty dark regions (the v7 gap failure is gone).
- Speed reads off the field: FAST air shows DENSE clusters of LONG bright streaks; CALM air shows SPARSE faint SHORT stubs. Density and tail length both track local wind speed.
- Tails are clearly longer than v5/v7 — comet streaks, not dots.
- Streaks drift in MOTION (advected by `windAt`); the compass shows heading (cyan) vs ground-track (yellow) gap with a DRIFT readout (~+27°), and the scene shifts laterally between frames.
- Prior wins intact: small gliding-V bird against large rolling EKG ridges; ridges OCCLUDE the motes; elevation color (teal/blue valleys → magenta/white peaks); the good flight.
- 60fps; no `[WebGPU lost]` / `pageerror` / `console.error`.

### Watch for
- The mote drift is most legible in MOTION; a single still frame shows density + tail length + direction (enough to read SPEED) but not the live flow.
- `tailFloor` / `densityFloor` are the calm-air floors — raising them flattens the speed contrast (calm air starts to look as busy as fast air); lowering `densityFloor` toward 0 reintroduces empty gaps.
- `speedLo` / `speedHi` are calibrated to the current `windAt` (sampled min ~0.03, max ~16.5, mean ~8.5); retuning the field (`curlAmp`/`driftAmp`) would require recalibrating these for the contrast to stay full-range.
- Screenshot driver: `.ai/tmp/v8b-shot.mjs` (Playwright + Metal WebGPU flags, port 5174 then 5173); waits `window.__birdBooted`; captures a pair ~0.8s apart at `.ai/tmp/v8b-final-0.png` / `.ai/tmp/v8b-final-1.png`.
