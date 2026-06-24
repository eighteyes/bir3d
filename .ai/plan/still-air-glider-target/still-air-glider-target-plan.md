# Plan — still-air glider basis + fly-to-target

## Accepted plan
Strip the bird's FORCE model down to a dead-calm downhill glider (the "basis"), keep all
rendering, and add a target to fly toward. Canyons and flapping wings are explicitly later.

### Decisions (from brainstorming)
- Diagnose: BIRD FLIGHT FEEL (not wind / terrain / whole-loop).
- Baseline: STILL AIR (Layer 0) — no wind, lift, thermal, or buffet on the bird.
- Power model: DOWNHILL GLIDER / WINGSUIT — no thrust; energy management; altitude only decreases.
- Keep ALL rendering (bloom, terrain, motes, marker, compass). Motes keep drifting as ambient
  air; the bird ignores them. ("no wind" applies to the BIRD, not the visuals.)
- Terrain stays as a distance/depth reference (no canyon tuning yet).
- Build approach: minimal diff — a `stillAir` flag on the bird + one self-contained Target
  module. No flair-toggle scaffolding (scope corrected by user: "keep all the rendering").

### Scope
IN: stillAir flag, resetAltitude, Target beam (render + reach/respawn), HUD target line,
disable scripted wobble.
OUT (later): canyon terrain tuning, flapping-wing animation, wind-as-flair re-enable.

## Next
- Fly-test and tune glide feel + target placement (by feel).
- Then: flapping wings.
- Then: canyons.
