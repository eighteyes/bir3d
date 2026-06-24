# Compressed Context
date: 2026-06-19
session: 5b3183c3-54f2-4a41-99a5-7dc3551d3cd2

## Original Task
"i'm thinking of stripping this to it's base elements, so i can diagnose the movements, lets get the core mechanics working first, then add flair."
Scope evolved (in order): still-air glider → fly-to-target beacon → keep ALL rendering → powered flap-to-climb (two wings) → view-fog +50% → inverted pitch → punchy dive/swoop → terrain skim (no teleport) → crash penalty on hard impact. Entry: `/index-bird.html` → `bird-main.ts`.

## Files Changed
- src/host/gpu/bird3d.ts — `stillAir` flag (zeros wind/lift/thermal/buffet on the BIRD only); POWERED FLAP in `integrate()` (beat cycle: tap=1 beat / hold=repeat; per-wing `effL=1+steerBias`,`effR=1-steerBias`; SUM→`flapClimb`+`flapThrust`, DIFF→heading yaw + `flapBank`; `ampL`/`ampR`→shader); terrain contact = skim (kill into-ground vel) + CRASH (`impactRate=penetration/dt > crashSpeed` → `speed*=1-crashBleed`, `crashT` steering stumble); `resetAltitude()` (now UNUSED); `UNIFORM_BYTES=112`, draw writes u[19..26]=flexPhase,heading,renderBank,flexAmp,flapBeatPhase,ampL,ampR,pitch
- src/host/gpu/target.ts — NEW. `Target` class: `respawn(fromX,fromZ,heading)` 700–1000m ahead, `distanceTo`/`checkReached` (horizontal), billboarded amber beam `draw()` (depthCompare "always" = always-on-top)
- src/host/shaders/target.wgsl — NEW. Camera-facing vertical beam, gaussian falloff, HDR amber, additive
- src/host/shaders/bird3d.wgsl — `Uniforms`: flexPhase/heading/bank/flexAmp/flapPhase/ampL/ampR/pitch/pad0/pad1; added `rotX()`; vs = idle flex + per-wing beat (`select(ampR,ampL,spanFrac<0)`) + `rotX(pitch)` whole-model tilt
- src/host/bird-main.ts — `bird.stillAir=true`; Target instantiate+draw+reach-loop; Space keydown/keyup→`input.flap`; INVERTED pitch `GLIDE_TRIM + applyDead(mouseY)*PITCH_RANGE`; removed mouse-leave→autopilot; removed ground-reset teleport; fog `fogDensity 0.5/1100`, mote fog `0.5/1400`, `maxDist 2850`; HUD TARGET/▲FLAP/✖CRASH lines; tune-panel rows beatLift/beatThrust/beatHz/crashSpeed; `sampleCount:SAMPLES` in terrain config (CRITICAL — its absence = black screen)
- src/host/autopilot.ts — returns `flap:false`
- HUMAN_REVIEW.md — entries for "still-air glider + fly-to-target" and "two-wing flap + fog" (NOT yet: dive tuning, inverted/steeper pitch, model tilt, terrain skim, crash)
- .ai/plan/still-air-glider-target/ — plan/context/tasks artifacts

## Current Tuning (live on 'T' panel where noted)
- Airframe: glideSpeed 26, minSpeed 13, maxSpeed 70, dragK 0.2, divePower 2.0, gravity 9.0, sinkRate 1.0, VEL_TAU 0.13 (const in integrate), pitch clamp ±1.0 (~57°)
- Flap: beatHz 3.0, beatLift 14, beatThrust 10, beatAmp 0.9, flapAsym 0.3, flapTurn 0.6
- Crash: crashSpeed 16, crashBleed 0.65, crashTime 0.5, stumble factor 0.3
- bird-main consts: PITCH_RANGE 1.0, GLIDE_TRIM -0.03, YAW_GAIN 1.8, START_CLEARANCE 400, REACH_RADIUS 55, SAMPLES 4, rowSpacing 5 (user-set)
- Controls: mouse-x=yaw rate; mouse-y=pitch INVERTED (cursor UNDER bird=climb, OVER=dive); Space=flap (tap/hold); P=autopilot; T=panel

## Current State
tsc clean (exit 0) after every change. Vite dev server RUNNING in background on port 5174 (bg task btwzegau3) via `/opt/homebrew/bin/node node_modules/.bin/vite --port 5174`. App boots/serves (200). All changes verified by typecheck + asset-serve only — live WebGPU render/feel is user's flight test each pass. Last shipped: crash-on-hard-impact mechanic.

## Open Threads
- HUMAN_REVIEW.md needs an entry covering: dive tuning, inverted+steeper pitch, model pitch-tilt, terrain skim, crash mechanic (user: "say the word when the feel settles and I'll write it all in")
- Canyons = deferred "later"; chosen mountain direction beyond crash = "solid walls / canyon nav"
- Feel still being dialed by user — expect more tuning passes (dive swoop, flap strength, crash threshold)
- Offered but NOT done: pitch response curve (gentle center/steep edges), VEL_TAU on T panel, FOV_KICK for more dive rush, pitch neutral pinned to bird's screen position, manual reset key (resetAltitude unused)

## Key Decisions
- "no wind" = no wind ON THE BIRD (stillAir); wind field + motes kept drifting as ambient visual; ALL rendering retained
- Flap = TWO independent wing forces, NOT a unified vector (user requirement): sum climbs, difference turns; both wings visibly beat (ampL/ampR)
- Pitch is INVERTED yoke-style; whole model tilts to nose attitude via rotX
- No auto-teleport/ground-reset — flap to recover; gentle terrain = skim, hard impact = crash penalty (speed dump + stumble), no full reset (user disliked teleport)
- ENV: nvm shim broken in non-interactive shell (FUNCNEST corrupts PATH). Use `/opt/homebrew/bin/node` + `./node_modules/.bin/tsc`; NEVER `export PATH=` (truncates → tools vanish); call binaries absolute (`/usr/bin/curl`). Interactive shell is fine (user can `! npm run dev`)
- All scene render pipelines MUST pass `sampleCount: SAMPLES` (4×) or they mismatch the MSAA target → black screen
