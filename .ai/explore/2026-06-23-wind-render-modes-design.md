---
title: Wind render modes — 3 divergent primitives per tier (far / near / wake) + wind-scaled bird buffet
date: 2026-06-23
session: f65433b5-2d4a-4594-85b6-036787d8f3af
status: approved (grid green-lit; pending written-spec review)
scope: RENDER ONLY. Wind feel/physics frozen (windAt/windProfile/updraftAt unchanged). Adds per-tier render-MODE switches (A/B/C), new dedicated wake geometry (B/C), and wind-scaled VISUAL bird buffet. No new shaders, no new pipelines.
files: src/host/gpu/wind.ts (per-tier mode geometry + 2 shed-emitter pools + vertex-buffer span), src/host/bird-main.ts (T-panel mode buttons + dials + buffet dials), src/host/gpu/bird3d.ts (wind-scaled visual buffet), src/host/shaders/wind.wgsl (UNCHANGED — verify zero edits)
---

# Goal

Three divergent ways to render each of the three wind layers — global (far), local sphere (near), and
wake — selectable live as render MODES with T-panel dials. Nine looks total. Plus: in heavy wind the
**bird** shakes more than the **camera** (the camera is already decoupled; only the visual buffet changes).

Feel is good and stays good. This pass touches pixels, not flight.

# Constraints (locked with user)

- **No long continuous lines** — they show CORNERS at segment joints. Every option is corner-proof by
  construction (short bounded tail / disconnected pieces / discrete glyph / closed convex loop).
- **No pure point sprites** — they read as SWIMMING, not flying. Every option is directional/anisotropic
  with a clear sense of heading.
- **Baseline Option A for every tier = the short dense comet** (the legible near-sphere primitive), tuned
  per tier. B and C must be genuinely divergent from A and from each other.
- **Wake = MIX**: A modulates the sphere motes (no geometry of its own); B and C are dedicated geometry
  shed from the wings and persisting behind the bird.
- **Buffet = wind-scaled, visual-bias**: ramp the render-bank rock + a render-only positional tremor by
  LOCAL wind magnitude; keep the dynamic velocity-shove (the feel) modest/unchanged; camera untouched.
- Feel/physics frozen; 60fps must hold; zero shader edits.

# Architecture — the shared spine

All nine reuse the existing `FPV=11` ribbon vertex format and the SINGLE wind pipeline (additive blend,
depth-test `less` / no depth-write, 4× MSAA). Only CPU geometry generation differs per mode, so:

- A per-tier mode enum drives a branch in the geometry emit:
  - `farMode: "comet" | "stipple" | "chevron"` — branch in the far emit loop (`wind.ts` `step()`).
  - `nearMode: "comet" | "flecks" | "filaments"` — branch in `stepNear()`.
  - `wakeMode: "modulate" | "helix" | "rings"` — `modulate` is the existing in-place stir of near motes;
    `helix`/`rings` drive two new persistent shed-emitter pools.
- **Vertex buffer layout grows to reserve three spans**: far + near (as today) + a new WAKE-SHED span sized
  to the worst-case quad count of `helix`/`rings`. `draw()` issues the count for the active spans.
- Wake B/C emitter pools follow the same persistence model as the near motes (seed xyz + age [+ side for
  helix / plane-basis for rings]); both default OFF (mirror today's `showNear`/`showWake`).
- **wind.wgsl is unchanged.** The VS already lays ribbon width screen-perpendicular at constant pixel
  width, so tilted rings and twisting tubes render as clean ribbons with no shader work.

# The nine options

Divergence is by orthogonal physical quantity per tier, not by re-skinning one idea.

## FAR — global wind  (axis: mass → implied-line → glyph)

**FAR-A · Short Dense Comet (baseline, far-tuned)**
- Today's comet shortened (`farSegments` 6→4, `farSubdiv` 3→2, ~8–14 m flow-span) and scattered sparse
  across the ~950 m wedge. Tail too short to accumulate the joint angle that corners. Altitude profile is
  free: `windProfile(y)` already feeds `speedFrac` → fast-aloft reads brighter/whiter, calm valleys dim.
- Geometry: existing far ribbon, reuse. Cost: WIN (~24 v/mote vs 108).
- Dials: `farSegments` 3–5, `farSubdiv` 1–2, `segStep` 0.3–0.6 s, `numMotes` 1600–4000, `densityFloor`
  0.4–0.8, `dotPx` 2–4.

**FAR-B · Stipple Streamline (gapped co-aligned dashes)**
- A streamline drawn as 2–4 DISCONNECTED single-segment dashes with dark gaps, each tangent to the live
  flow at its point — the eye completes the over-ridge arc (road-lane-dash read). Lead dash brightest
  (marching-ants direction). Aloft: dashes lengthen, gaps close toward a fast streak; valleys: short/sparse
  stipple. `windProfile` drives the gap/length modulation as the visible atmosphere readout.
- Geometry: keep the short coarse backward integration for over-ridge points, but emit K disconnected quads
  at arc-fractions instead of a continuous subdivided ribbon. No shared joints → corner-free. Cost: WIN
  (~18 v/mote; can drop integration sub-steps).
- Dials: `dashCountK` 2–4, `dashLenM` 5–16, `gapRatio` 0.5–3, `lenByAltitude` 0–1, `leadBoost` 1–2,
  `numMotes` 1500–4000, `densityFloor` 0.3–0.7.

**FAR-C · Arrowhead Chevron (V-tick heading glyph)**
- Two short limb-quads meeting at a bright nose pointing along the windProfile-scaled flow; heading reads on
  a frozen frame. Aloft: sharp raked darts; valleys: stubby wide marks. Cheapest of all nine (one flow
  sample, 12 v, no integration) → highest affordable coverage = the "whole sky is moving" read at distance.
- Geometry: per mote sample flow dir once; apex = head; two limbs at ±spread; emit 2 quads, `along` 0 at
  apex → 1 at tips. Only shared vertex is the intentional apex (a bright point, not a smoothing joint). Cost:
  BIG WIN.
- Dials: `spreadAngleDeg` 15–45, `limbLenM` 8–22, `widthPx` 1.5–3, `apexBoost` 1–2, `rakeBySpeed` 0–1,
  `numMotes` 800–2500, `densityFloor` 0.3–0.7.

## NEAR — local sphere  (axis: translation → gradient → curl)

**NEAR-A · Short Dense Comet (baseline)**
- The shipped tadpole swarm you fly through: ~800–1600 little 4-segment curling comets in the ~65 m ball,
  tail re-samples `flowAt + birdWakeAt` so it curls. Shows path/history. Cost: baseline (24 v).
- Dials: `nearCount` 400–1600, `nearSegStep` 0.08–0.20 s, `nearSegments` 3–5, `dotPx` 1.8–3.2, `swirlGain`
  0.5–1.0, `heatLenGain` 0–1.5.

**NEAR-B · Shear Flecks**
- One mote = a single short (2-segment, ~2–4 m) oriented dash pointing along its local disturbed velocity;
  LENGTH + BRIGHTNESS driven by local SHEAR (finite-diff of velocity across ~3 m). Uniform air = short/dim;
  where wake meets ambient (a shear layer) flecks stretch + brighten → you see the moving shear SURFACES.
  Shows the velocity field + gradient (state), no tail.
- Geometry: in `stepNear`, build a 2-point polyline along the disturbed-flow dir (no backward integration);
  `speedFrac` carries shear magnitude; flat low-taper `along` ramp so it reads as a tracer, not a mini-comet.
  Finite-diff reuses the `flowAt + birdWakeAt` already sampled at the head. Cost: CHEAPEST (12 v) →
  ~2–4× density headroom.
- Dials: `fleckLen` 1.5–5 m, `shearGain` 0–3, `shearRadius` 1–5 m, `fleckTaper` 0–0.4, `orientLerp` 0–1,
  `fleckCount` 120–260.
- WATCH POINT: the flat (non-head-bright) `along` remap is new near-tier code; the FS taper assumes a
  monotonic head-bright ramp. Small, but not pure reuse.

**NEAR-C · Curl Filaments**
- One mote = a thin 5-segment thread (~6–10 m) that visibly CORKSCREWS around an invisible wingtip vortex
  core. Seeded preferentially at the two cores (`wingEmitFrac`); tail integrates the full `flowAt +
  birdWakeAt` twin-Rankine field, so it genuinely wraps the core. Shows curl/rotation — turns the
  already-built vortex field into the visible subject.
- Geometry: the existing near tail loop re-tuned (`nearSegments` 5, larger step, high `wingEmitFrac`, raised
  `swirlGain`). Scratch buffers auto-size from `nearSegments`. Cost: DEAREST (30 v) — keep `filCount`
  120–220.
- Dials: `filSegStep` 0.12–0.4 s, `swirlGain` 0.5–2.0, `wingEmitFrac` 0.5–0.95, `vortexCore` 3–10 m,
  `nearSegments` 4–6, `filCount` 120–220.
- WATCH POINT: may want a Catmull-Rom smoothing pass on the near tail to kill residual kink — new near-tier
  CPU (subdiv is far-tier only today). Keep step moderate + 5 segs to avoid needing it.

## WAKE — the bird's mark  (axis: stir-in-place → continuous helix → discrete pulse)

**WAKE-A · Comet, wake-modulated (baseline, no new geometry)**
- The existing dense near ball MODULATED by the bird's disturbance: `birdWakeAt` added to advection + tail
  integration (bow-wave lights air ahead, slipstream rakes behind, twin Rankine vortices bend the tails),
  touched-air heat tints hard-hit motes yellow→red and trails them longer. The wake is a tint + curl on
  motes that already exist. Cost: FREE (today's `showWake` path).
- Dials: `showWake` (master), `wakeGain` (drag), `swirlGain` (vortex), `bowGain`, `vortexCore`, `heatRef`,
  `heatTau`, `heatLenGain`, `wingSpan`, `wingEmitFrac`.

**WAKE-B · Wingtip Helix Filaments (counter-rotating shed cords)**
- Dedicated geometry: two streams of short bright tube-arcs shed from each wingtip that visibly TWIST, one
  per tip, counter-rotating (downwash between them). The twist is real — each shed element is advected by the
  same `birdWakeAt` field whose tangential term peaks at `vortexCore`, so it corkscrews around its core while
  drag pulls it back. The visible "tube" is a dense stream of overlapping SHORT arcs, never one long
  filament. Warms yellow→red where the bird hit hardest.
- Geometry: new persistent emitter pool (2 tips × ~120 live elements: seed point + age + side), seeded at
  `birdPos ± wingSpan·right` (reuse the wingtip seed branch). Each frame integrate a SHORT backward polyline
  through `flowAt + birdWakeAt` (the existing curling-tail loop) and emit via the existing 6-vert quad emit,
  written into the new wake-shed span. Retire by age. Cost: per-segment field cost like the near tail
  (~120/tip). Reuse, newPipeline=false.
- Dials: `wakeEmitRate`, `wakeLife`, `helixGain` (twist tightness, shed-only), `wakeSeg` 2–4, `wakeSegStep`,
  `wakeTubePx`, `wakeTaper`, `heatLenGain`, `counterRotate` (toggle).
- WATCH POINT: vertex-buffer span must reserve this pool's worst-case quad count.

**WAKE-C · Shed Pressure Rings (periodic expanding hoops)**
- Dedicated, DISCRETE geometry: thin bright ring-arcs — closed ribbon loops oriented face-on to the flight
  axis — shed on a cadence (`ringRate`, a Strouhal heartbeat), EXPANDING radially while carried backward +
  downward by `flowAt` + axial drag, fading over life. A receding train of growing hoops marks each pressure
  pulse. Encodes shedding frequency + shear (via tilt). Reads as flying via recession + growth +
  heading-locked face-normal.
- Geometry: new ring pool (~16–32 live rings: center xyz, plane basis, radius(age), age, heat). On a timer
  spawn a ring; each frame grow radius + convect center, tessellate N points, emit consecutive pairs as quads
  (`segDir` = around-ring tangent) into the wake-shed span. Pure trig + one vector add per ring → cheapest
  dedicated wake. Short convex chords hidden by the perp rounding. Cost: CHEAP.
- Dials: `ringRate`, `ringGrow`, `ringLife`, `ringSegN` 12–32, `ringTubePx`, `ringStartRadius`, `ringTilt`,
  `twinOffset` (per-tip vs centerline), `convectFrac` 0–1, `ringWarmBias`.

# Bird buffet — wind-scaled, visual-bias

The camera is already decoupled (it aims along a low-passed velocity, τ≈0.4 s, so it ignores fast gusts).
Today's buffet amplitude is FIXED, independent of wind, which is why heavy wind doesn't feel heavier.

- Compute a local wind magnitude per frame: `wMag = |windAt(birdPos)| · windProfile(birdY)` (the field the
  bird is actually in), mapped through `buffetWindRef` to a 0..1 intensity.
- Scale the VISUAL buffet by `buffetGain · intensity`:
  - Ramp the existing render-bank rock (`bird3d.ts` ~L475–481), capped at `rockCapDeg`.
  - Add a render-ONLY positional tremor applied to the bird's model matrix at draw time — NOT integrated
    into `pos`, so physics and the camera target (which follows `pos`) are unaffected. The bird visibly
    judders against a steady frame; calm air stays glassy.
- Leave the dynamic gust velocity-shove (the feel) modest/unchanged.
- Result: bird shakes, camera doesn't — proportional to the wind it's flying.
- Dials: `buffetGain` (overall visual strength), `buffetWindRef` (wind speed m/s → full buffet), `rockCapDeg`
  (max visual roll).

# T-panel wiring

Reuse the existing helpers (`panelSep`, `sliderRow`, `toggleBtn`; `bird-main.ts`):

- Under each wind section header, a mode button cycling `A ▸ B ▸ C` (the `toggleBtn` pattern generalized to
  a 3-state cycle), calling `wind.setFarMode/​setNearMode/​setWakeMode`.
- Per-mode dials shown live (the active mode's dials; inactive-mode dials may stay visible — simplest).
- A new "bird — buffet" section with `buffetGain` / `buffetWindRef` / `rockCapDeg`.
- New `Wind` setters: `setFarMode`, `setNearMode`, `setWakeMode` (+ the shed-pool toggles). Debug handles on
  `window` to set modes from the console.

# Delivery phases

1. **Baseline + scaffold** — unify all three tiers onto the short comet (Option A everywhere), add the mode
   enums + per-tier branch + the three mode buttons + the vertex-buffer wake-shed span (empty for now).
   Ship/verify: comet renders for far/near/wake-modulate, switching modes is a no-op stub. (This is the
   "make all 3 render like the local sphere motes" first move.)
2. **Divergent modes** — implement FAR-B/C, NEAR-B/C, WAKE-B/C and their dials. Verify each renders + dials
   respond; flag the two WATCH POINTs.
3. **Buffet** — wind-scaled visual buffet + dials.

# Verification

1. `tsc --noEmit` clean.
2. Live headless gate (worktree dev server):
   - MODE SWITCH: cycle each tier A/B/C via the debug handles → no pageerror, vertex counts change as
     expected, 60fps held on the densest mode.
   - CORNER CHECK (visual/screenshot): FAR-B/C and the comets show no kinked joints.
   - WAKE SHED: enable WAKE-B/C → shed geometry appears behind the bird and recycles (pool doesn't grow
     unbounded); buffer span not overrun.
   - BUFFET: sample bird visual buffet amplitude at LOW vs HIGH local wind → high materially larger; camera
     aim variance stays ~flat across both (camera unmoved). Physics `pos` path unchanged by the render-only
     tremor.
   - FROZEN-FEEL GUARD: `windAt`/`updraftAt`/drift unchanged (re-run the existing wind-atmosphere /
     updraft-buffer / wind-live specs green).
3. HUMAN_REVIEW entry + session id; manual T-panel checklist per tier.

# Out of scope (this pass)
- Any change to wind feel/physics (`windAt`, `windProfile`, `updraftAt`, drift, thermals).
- New shaders or new render pipelines (all nine reuse the one ribbon pipeline).
- Camera shake (camera stays decoupled by design).
- Per-mode persistence of dial values across reloads.
