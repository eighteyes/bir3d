---
title: Felt-wind — twin wingtip vortices, warm touched-air, depth-graded sphere
date: 2026-06-22
session: f65433b5-2d4a-4594-85b6-036787d8f3af
status: approved (forks: persistent decaying trail + true wingtip pair + sphere-bounded v1)
scope: render-only — windAt / bird flight physics stay FROZEN
files: src/host/gpu/wind.ts, src/host/shaders/wind.wgsl, src/host/bird-main.ts (wiring/tunables)
---

# Goal

Make the wind-around-the-bird legible and expressive. Three coupled asks, one outcome:
the air the bird cuts **flares warm (red/yellow) + sparkles**, trails off **two counter-rotating
wingtip vortices**, and **condenses from faint to bright as it is felt** (proximity + disturbance),
fixing the flat uniform sphere.

# Diagnosis (current state)

- ONE spiral: `birdWakeAt` (wind.ts:886-912) models swirl as a single tangential rotation about ONE
  central motion axis through `birdPos` (`t = axis × r̂`, line 903). No wingspan, no two cores → one
  corkscrew. Real wings shed a counter-rotating PAIR (one per wingtip).
- FLAT sphere: near brightness `nearVis = fadeIn(age) · fadeOut(distFrac)` keeps FULL brightness from
  center to 78% radius, fading only at the rim (wind.ts:984) → ~78% of the ball is uniform = a flat
  wall of dots. Shader adds only flat 50% (`color*0.5`) + a speed tint. No proximity/depth gradient.
- Touched air is indistinguishable: all motes tint cyan→white by speed only (wind.wgsl:77-94).

# Decisions (resolved with user)

1. "Touched" = PERSISTENT decaying trail (per-mote heat, ~1.5 s decay), not instantaneous flare.
2. Two vortices = TRUE wingtip pair (two offset counter-rotating cores), not a split single swirl.
3. Trail scope = SPHERE-BOUNDED for v1 (fades at the ~65 m ball's rear; a horizon-spanning ribbon is
   a later separate emitter system — explicitly out of scope).

# Design

## V — twin counter-rotating wingtip vortices (replaces the single on-axis swirl)
Keep `bowGain` (parts at nose) and `wakeGain` (axial slipstream behind). REPLACE the single central
`swirlGain` swirl with two cores:
- wing axis `right = normalize(axis × worldUp)` (guard near-vertical axis → fallback to world X).
  v1 uses world-up; using the bird's true up-vector (bank-aware) is a later refinement.
- core positions `birdPos ± halfSpan·right`; each core line runs backward along `−axis` (trails).
- at point P, for each core c∈{+1 right tip, −1 left tip}: radial-from-core
  `rvec = (P−corePos) − ((P−corePos)·axis)·axis`, `rho=|rvec|`, tangential dir `tdir = axis × r̂`,
  circulation sign `= −c` so the pair is COUNTER-rotating with downwash BETWEEN the tips.
- induced tangential speed `= swirlGain · bs · behind · coreFall(rho)`, where `coreFall` peaks near the
  core and decays (e.g. `rho·rc / (rho²+rc²)` Rankine-ish, `rc=vortexCore`), and the existing
  ball-edge `fall` still bounds it. Sum both cores into the wake `out`.
- cost: ~2× wake math on a 400-mote sphere (head + ~4 tail pts) ≈ 4k evals/frame — negligible.

## H — per-mote heat (memory of being touched)
- new `nearHeat: Float32Array(nearCount)`.
- each frame per near mote: `gain = clamp(|wakeVel| / heatRef, 0, 1)`; `heat = max(heat·decay, gain)`
  where `decay = exp(-dt / heatTau)`, `heatTau ≈ 1.5 s`. (gain uses the wake disturbance magnitude
  already computed for advection — free.)
- recycled motes (leave the ball) reset heat to 0 in `seedNearMote`.

## W — warm tint + sparkle (touched air only)
- vertex format FPV 10 → 11: append `heat` (location 6). Far tier writes heat=0 (always cool).
- wind.wgsl fs: `coolTint = mix(CYAN, WHITE, speedFrac)` (unchanged); `warmTint = mix(YELLOW, RED, heat)`;
  `tint = mix(coolTint, warmTint, smoothstep(0, 0.3, heat))`. Palette consts: YELLOW≈(1.0,0.8,0.2),
  RED≈(1.0,0.25,0.08).
- sparkle: per-mote twinkle `spark = 1 + sparkleAmp·heat·sin(time·sparkleFreq + phase_i)` (phase_i from
  `hashRank(i)·TAU`), folded into `nearVis` CPU-side (vis already scales fs brightness — no new uniform).
  Untouched air (heat≈0) does not sparkle.

## O — opacity ramp (kill the flat wall; "light until felt")
Replace the full-to-78%/rim-fade with:
`proximityRamp = ambientFloor + (1-ambientFloor)·pow(1 - distFromBird/nearRadius, k)`  (1 at bird → ambientFloor at edge)
`nearVis = fadeIn · max(proximityRamp, heat) · spark`
So ambient sphere air is faint (light), brightening toward the bird AND wherever the wake has touched
it (heat). `ambientFloor≈0.12`, `k≈1.6`. fadeIn (no-pop on reseed) is kept.

# Tunables (DotParams + bird-main; a few on the `T` panel)
`wingSpan` (half-span, m; default ~10), `vortexCore` (rc, m; ~6), `heatTau` (s; ~1.5),
`heatRef` (m/s → full heat; ~10), `sparkleAmp` (~0.6), `sparkleFreq` (rad/s; ~9), `ambientFloor` (~0.12),
`k` proximity exponent (~1.6). Existing `bowGain`/`wakeGain`/`swirlGain` retained (swirlGain now drives
the pair).

# Constraints
- FROZEN: do NOT touch `windAt`, `thermalAt`, `potential`, `flowHorizontal`, bird3d physics. All new
  behavior lives in the near-mote advection (`stepNear`/`birdWakeAt`), the vertex format, and wind.wgsl.
- 60fps must hold (headless ANGLE proxy; real GPU faster). +1 float/vertex ≈ +730 KB buffer (negligible).

# Verification
1. `tsc --noEmit` clean.
2. Live headless gate (worktree server on :5273, like the wind-reenable):
   - TWO distinct counter-rotating cores: sample the wake field at mirror points ±halfSpan·right behind
     the bird; assert tangential components have OPPOSITE sign (counter-rotation) and non-trivial magnitude.
   - HEAT present: after a few seconds of flight, some near motes report heat>0 (touched), and heat
     DECAYS (a touched mote's heat drops over ~1.5 s once disturbance ends).
   - 60fps held, no crash, no pageerror.
3. Screenshot: two warm (red/yellow) sparkle corkscrews trailing the wingtips; sphere reads graded
   (faint rim → bright/warm core), not a flat wall.
4. HUMAN_REVIEW.md entry (manual steps + session id).

# Out of scope (v1)
- Horizon-spanning persistent trail (needs a separate emitter decoupled from the recycling sphere).
- Bank-aware wing axis (use bird's true up-vector) — world-up for v1.
- Touched-heat on the FAR tier — near sphere only.
