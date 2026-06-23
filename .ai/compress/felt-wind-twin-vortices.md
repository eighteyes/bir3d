# Compressed Context
date: 2026-06-22
session: f65433b5-2d4a-4594-85b6-036787d8f3af

## Original Task
Two threads. (1) DONE: "time to re-enable wind". (2) ACTIVE — user asks of the live wind viz:
"why does wind primarily make a single spiral out of the back of the bird, shouldn't it do two? i
want to clearly see the wind the bird has touched, maybe give it red / yellow sparkle trails?" and
"the sphere of wind around the bird needs to be more legible, it feels flat, maybe keep the opacity
light until it's close enough to be felt by the bird".

## Files Changed
- src/host/bird-main.ts:44 — `showWind = true` (motes drawn); :182 `bird.stillAir = false` (physics feels wind); HUD label → "wind glider". DONE, verified.
- tests/gpu/wind-live.spec.ts — NEW live gate (wind felt/evolves/drifts/flyable/60fps). DONE, passing.
- HUMAN_REVIEW.md:1-… — "wind re-enabled" section (manual steps, :5273 server caveat). DONE.
- .ai/explore/2026-06-22-felt-wind-twin-vortices-design.md — APPROVED design for the ACTIVE feature.
- .ai/tmp/wind-verify.config.ts — throwaway Playwright config (baseURL :5273, no webServer).

## Current State
Wind re-enable is complete and verified (meanMag 11.8 m/s, drift 23.4°, 60fps, no crash; cyan motes
stream + bird crabs). The felt-wind feature (twin vortices / warm touched-air / depth-graded sphere)
is DESIGNED + APPROVED but NOT YET IMPLEMENTED — wind.wgsl is still FPV=10 cool-only; wind.ts wake is
still the single on-axis swirl. Last turn ended asking user "Build now vs Review spec first"; user ran
/compress. Treat design as approved (no vetoes) — next step is implementation.

## Open Threads
- IMPLEMENT felt-wind per the spec (.ai/explore/2026-06-22-felt-wind-twin-vortices-design.md):
  V) replace single swirl in birdWakeAt (wind.ts:886-912) with TWO counter-rotating wingtip cores at
     birdPos±halfSpan·right (right=normalize(axis×worldUp)); coreFall ~ rho·rc/(rho²+rc²).
  H) add nearHeat:Float32Array; heat=max(heat·exp(-dt/τ), |wake|/heatRef), τ≈1.5s; reset in seedNearMote.
  W) vertex FPV 10→11, append `heat` (loc 6, far tier=0); wind.wgsl fs blend cool→mix(YELLOW,RED,heat) by
     smoothstep(0,0.3,heat); sparkle = 1+sparkleAmp·heat·sin(time·freq+hashRank(i)·TAU) folded into nearVis.
  O) near opacity: nearVis = fadeIn·max(ambientFloor+(1-ambientFloor)·(1-dist/R)^k, heat)·spark; floor≈0.12,k≈1.6.
  Update pipeline vertex attrs/arrayStride for FPV=11; thread tunables via DotParams + bird-main.
- VERIFY: tsc clean → live gate (assert two cores OPPOSITE tangential sign + heat>0 then decays + 60fps
  + no crash) → screenshot (two warm sparkle corkscrews, graded sphere) → HUMAN_REVIEW entry.

## Key Decisions
- RENDER-ONLY: never touch windAt/thermalAt/potential/flowHorizontal/bird3d physics (FROZEN).
- Forks chosen by user: persistent decaying trail (not instantaneous), TRUE wingtip pair (not split swirl),
  trail SPHERE-BOUNDED for v1 (horizon ribbon + bank-aware wing axis = out of scope).
- SERVER GOTCHA: stale vite on :5173 (other worktree) serves OLD code; reuseExistingServer latches onto it.
  Always start a fresh `./node_modules/.bin/vite --port 5273 --strictPort` from THIS worktree and test via
  `--config .ai/tmp/wind-verify.config.ts`. Verify served file with curl before trusting a test result.
- Shell nvm wrapper breaks `node`/`npm`; call binaries directly via ./node_modules/.bin/.
- bird-main passes nearCount:400, nearRadius default 65m. swirlGain/bowGain/wakeGain defaults 0.7/0.9/0.75.
