---
title: Global wind — altitude atmosphere (boundary-layer gameplay) + terrain-hugging visual
date: 2026-06-22
session: f65433b5-2d4a-4594-85b6-036787d8f3af
status: approved (direction green-lit; pending written-spec review)
scope: GAMEPLAY (unfreezes windAt consumption) + far-tier mote visual. Local sphere + wake are OUT (separate layers, rendered distinctly later).
files: src/host/gpu/wind.ts (windProfile + far-mote advection/seed), src/host/gpu/bird3d.ts (drift + ridge lift consume the profile), src/host/autopilot.ts (updraftAt signature)
---

# Goal

Make the GLOBAL wind interesting as a gameplay system: a believable boundary-layer **atmosphere** the
bird reads. Calm down low, strong up high — one honest altitude profile, **no special-case rules**. The
far-tier mote visual stays close to the terrain (the cool look) but never leaves altitude a dead void.

# Gameplay intent (locked with user)

"Both — atmosphere": altitude gradient × terrain shelter, "a real sky to read." Climbing into open air =
stronger wind + more drift to manage (risk/reward); valleys = calm shelter. Wind has GAME IMPACT (the
bird flies it); the local sphere + wake are visual dials handled later.

# Key insight — absolute altitude dissolves the "special rules" problem

Profiling wind by **height-above-local-surface** would make wind ~0 at every surface, killing ridge
soaring at the ridge face → forcing a special-case exception. Instead profile by **ABSOLUTE altitude**:

- Ridges ARE high → they sit in strong wind → ridge lift stays strong (climbing the windward face into
  stronger air is rewarded). No exception needed.
- Valleys ARE low → calm. "Terrain shelter" emerges for free from low altitude — no hand-coded lee rule.
- Open high air → strong wind + drift (the risk/reward).

One profile, evaluated at the relevant altitude, multiplied into wind everywhere it's consumed. Uniform.

# Design

## 1. `windProfile(y)` — the one rule (wind.ts, exported)
Scalar multiplier on horizontal wind magnitude by absolute world altitude `y`:
```
windProfile(y) = windLoScale + (windHiScale - windLoScale) * smoothstep(windAltLo, windAltHi, y)
```
- `windLoScale ≈ 0.4` (valley-floor wind fraction), `windHiScale ≈ 1.4` (aloft strength).
- `windAltLo ≈ 100` m (below = full calm), `windAltHi ≈ 500` m (above = full strength). Terrain is 0–600 m
  (RELIEF=600), so valleys land in the calm zone, ridge-tops/open air in the strong zone.
- Pure function of `y` — cheap, no terrain sampling, no upwind raymarch.

## 2. Physics — uniform consumption (bird3d.ts, UNFROZEN)
Evaluate `windProfile(birdY)` ONCE per frame at the bird's absolute altitude and multiply it into the
bird's wind BEFORE it splits into drift + ridge lift — so both scale identically (no special rule):
- Horizontal **drift** scales by the profile → high/open shoves hard, low/valley settles.
- **Ridge lift** (`updraftAt`) scales by the same profile. `updraftAt` gains a `y` param (the bird's
  altitude); the autopilot passes the bird's `y` so it senses exactly what the bird rides. Because the
  bird soaring a ridge is at high `y`, ridge lift stays strong — soaring preserved, even enhanced with height.
- **Thermals** (`thermalAt`) are a SEPARATE vertical system, not "wind" — left unchanged this pass.

## 3. Visual — terrain-hugging, with a tail to altitude (wind.ts far tier)
- **Revert the loft**: far-mote HOME heights go back near the terrain (the cool surface-skimming look) —
  undo clearance 30→60 / vSpread 38→48.
- **But not a dead void aloft**: bias HOME heights toward the terrain with a power curve (most motes hug,
  a thin tail reaches up) over a tall-ish band, so altitude always shows SOME wind. (`home = loHome +
  rand^homeBias · (hiHome−loHome)`, `homeBias ≈ 2.2` clusters low; raise `maxClear`/`vSpread` enough for
  the tail.)
- Motes advect by `windAt × windProfile(moteY)` → sluggish in low valleys, ripping over high ridges and in
  the high tail. The atmosphere reads through SPEED, not float-height.

> Seeing wind AT the bird's altitude is the LOCAL SPHERE's job (separate distinct render, later). Global
> wind stays the terrain-skimming streamlines; this pass makes the field under them interesting.

# Constraints
- UNFREEZES the bird's windAt consumption (drift + ridge lift) — re-verify soaring like the updraft-buffer.
- `windAt`/`thermalAt`/`flowHorizontal` internals stay intact; the profile is a multiplier APPLIED at the
  consumers, not baked into windAt (which has no `y`). `windAt` is called tens of thousands of times/frame —
  the profile is one cheap multiply at each consumer, allocation-free.
- 60fps must hold.

# Tunables (DotParams + bird3d tuning; live where possible)
`windLoScale` (~0.4), `windHiScale` (~1.4), `windAltLo` (~100 m), `windAltHi` (~500 m), `homeBias` (~2.2),
plus the reverted `clearance`/`vSpread`/`maxClear`. Curve shape dialed by feel.

# Verification
1. `tsc --noEmit` clean.
2. Live headless gate (worktree :5273):
   - PROFILE: sample the bird's felt wind |drift| at LOW altitude vs HIGH altitude → high is materially
     stronger (gradient real). Assert `windProfile` monotonic increasing across the band.
   - SOARING PRESERVED: autopilot flies N s, rides positive updraft, does NOT sink-crash (ridge lift still
     works at altitude), 60fps, no pageerror.
   - VISUAL: screenshot — motes hug terrain, some wind visible aloft (not a void), faster over high ridges.
3. HUMAN_REVIEW entry + session id.

# Out of scope (this pass)
- Explicit lee/wind-shadow behind ridges (absolute altitude already gives valley-calm; lee is a later refinement).
- Local sphere + wake layers (rendered distinctly, separate work).
- Altitude-varying thermals.
