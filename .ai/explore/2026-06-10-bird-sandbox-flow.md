# Bird — sandbox flow (rough-but-playable vertical slice)

**Date:** 2026-06-10 · **Branch:** build/foundation · **Project:** Vector System engine.

## Goal
A rough-but-playable Bird game: steer a single bird gliding through a living wind field you can see, top-down. No goal, no score — the joy is the glide and reading the wind. Purpose: pull the project up from engine-internals to an actual playable game, and **feel out which control scheme works** (the design is uncertain about controls on purpose). This deliberately jumps ahead of the remaining engine plans (renderer/terrain/scheduler) to answer "is this fun?" early with a vertical slice.

## World
- The wind is the existing GPU fluid (`src/host/gpu/fluid.ts` + `src/host/shaders/fluid/*`), reused as-is.
- The fluid grid is a fixed **toroidal torus larger than the viewport** (sim ~256², viewport shows a sub-window ~half of it). Wind extends past every screen edge; crossing an edge **wraps** (endless glide).
- Seed the wind across the whole world (several drifting gusts/vortices), not one central blob, so there is always new field to glide into.
- Render the wind as a **subtle flowing backdrop** (dialed back from the bright debug blob; respect the §7.2 brightness ceiling) so the bird reads as the foreground.

## Camera — soft deadzone follow
- The camera follows the bird loosely: the bird **drifts around the center** within a deadzone; the camera only eases over when the bird approaches the deadzone edge. Not pinned to a pixel.
- Tunable: deadzone size, follow stiffness (easing rate).
- This is the *rough* moving window. The blueprint's true moving-window (re-solving the fluid in a window that recenters on the bird, feeding boundaries as it scrolls) is DEFERRED — here the sim is a fixed torus and only the camera moves.

## Bird physics — GPU-integrated
- One bird: state `{ pos, vel }` in a small GPU buffer.
- Integrated on the GPU each frame (a `bird_update.wgsl` compute pass) so it **reads the live wind in-shader** — no stale async-readback, no synchronous readback in the loop (blueprint hard rule). The chevron points along `vel`.
- Update: `vel += windAt(pos) * windCoupling * dt` (wind pushes) `+ intent.impulse` (input burst) ; apply `intent.turn` (rotate vel) ; `vel *= drag` ; `pos += vel * dt` ; toroidal wrap.
- Input arrives as a per-frame `intent = { impulse: vec2, turn: f32 }` uniform written from the CPU. **The GPU sim is control-scheme-agnostic** — only the CPU mapping differs per scheme.

## Controls — 4 switchable schemes (the taste test)
Swap live with number keys `1`–`4`; the active scheme is labeled on screen. Each maps raw input → the `(impulse, turn)` intent:
1. **Flick to impulse** — mouse drag→release = a velocity burst in the drag direction (length-capped); show an aim indicator while dragging. Gestural/floaty.
2. **Hold toward cursor** — hold button = gentle continuous thrust toward the cursor each frame; release = pure momentum + wind. Continuous/precise.
3. **Tap to bank** — tap ←/→ = rotate the glide a notch (turn applied to vel); momentum carries; wind curves it. Piloting feel.
4. **Flap forward** — tap space/click = an impulse along the bird's current heading (`normalize(vel)`). Rhythmic/birdlike.

## Visuals
- Bird: a neon **chevron** (arrowhead) oriented to velocity + a **fading ribbon trail** of recent world positions. Rough glyph drawn with a simple line/triangle pass — NOT the full §4.1 vector renderer (deferred).
- Wind: subtle flow backdrop (see World).

## Tuning overlay (how we find the feel)
On-screen, live-adjustable: wind-coupling, drag, input strength (flick/thrust/flap magnitude), turn rate, deadzone size, follow stiffness. Plus active scheme + cpu dt. These dials are the point — we fly and tune.

## Scope
**In:** one bird, 4 swappable controls, deadzone follow cam, toroidal world bigger than screen, GPU bird reading live wind, neon chevron + trail, live tuning, subtle wind backdrop. Entry at `/index-bird.html`.
**Out (deferred, not dropped):** goal/score, flock (multiple birds), true window-recentering fluid sim, full §4.1 neon renderer, Trees/Hexland, audio, persistence.

## Architecture decisions (judgment calls, all cheap to reverse)
- GPU-side bird integration (correct per blueprint; avoids stale/forbidden readback).
- Endless toroidal world rather than bounded edges (best fit for free-glide + periodic fluid).
- Rough chevron glyph instead of the real renderer (this is a feel prototype).
- Reuse the existing fluid sim unchanged; only add bird + camera + render-window + input.

## First step
Bird-state buffer + `bird_update.wgsl` (intent + wind + drag + integrate + toroidal wrap) + a camera-centered render of the wind sub-window + the chevron glyph + trail, driven by scheme 1 (flick). Get a bird flinging and gliding over the live wind on `/index-bird.html`. Then add schemes 2–4 + the switcher + the tuning overlay.

## Done = playable
Open `/index-bird.html`: a neon bird glides over a living, scrolling wind world; the camera drifts-follows; flick (and schemes 2–4) fling it; the wind visibly carries and curves it; feel constants are tunable live; 60fps; no errors. No goal needed — it is "a whole game, rough."

---

## v2 (2026-06-11) — CORRECTED to the real vision: 3D, not overhead

The v1 above was the wrong vision (flat top-down toy). User correction: **"this is 3d, not overhead"; depth comes from TERRAIN LAYERS; the bird is a V with floppy/flapping wings, not a triangle.** This matches the blueprint (Bird is the full-3D game). v1's 2D files stay in git history; v2 replaces the look.

**Decided (questions answered):**
- **3D perspective scene**, neon-vector on dark.
- **Terrain = a real 3D procedural heightfield** (blueprint §4.4: seed + fBm noise {octaves, lacunarity, gain, base_freq}, ENU meters; rough CDLOD = a recentering grid mesh that follows the bird and samples fBm at world coords). **Rendered in the neon receding-ridgeline LOOK** the user picked — glowing ridges/contours fading into atmospheric haze toward the horizon. "Receding ridgelines" is how this heightfield reads from the chase cam; it is NOT flat painted silhouette planes. Depth = real 3D perspective + fog.
- **Camera = chase behind + slightly above the bird** (§5: Bird is a 3D free-fly controller), looking forward over the terrain toward the hazy horizon; smooth deadzone follow (the bird drifts in frame). The flapping V reads from behind/above.
- **Bird = a 3D V silhouette with floppy flapping wings** (§4.1: 3D stroke, project-then-flatten, neon): body + two swept wings; wings flap with a sine, wingtips LAG the inner wing (the "floppy"); banks (roll) into turns. **Depth-tested so terrain ridges occlude it** (§4.1 Bird = depth-test terrain occlusion).
- **Wind drives flight AND is shaped by terrain** (§6.1 "FEEL ridge lift, thermals, valley eddies"): the existing GPU fluid is the horizontal wind field (bird samples it via async readback, no in-loop sync); ADD terrain-derived **ridge lift** (upwind slope → upward push) so soaring the ridges is the read-the-wind hook — terrain + wind + flight connect.

**This build = VALIDATE THE LOOK first.** One natural flight control only (mouse-steer: cursor steers heading yaw + pitch, bird banks; click/space to flap for lift; auto forward drift; wind pushes). The 4-scheme control taste-test + tuning panel RETURN after the 3D look is approved — don't rebuild them on an unvalidated look.

**Scope (v2):** in = 3D perspective render, chase cam, receding ridgeline terrain + sky/haze, flapping-V bird, wind coupling, one flight control, minimal overlay. Out (deferred) = control taste-test/tuning panel (until look approved), flock, goal/score, true moving-window fluid, full §4.1 renderer, real 3D fluid.

**Architecture notes (v2):** perspective render pipeline (hand-rolled mat4 perspective/lookAt, MVP uniform, depth buffer; WebGPU NDC z∈[0,1]); ridges as true-3D walls in world space at recycling Z distances (camera perspective makes the depth); bird physics may be CPU (one bird, proto) sampling wind via async readback — no synchronous in-loop readback; GPU-aero is the scale path. Entry stays `/index-bird.html` (replaced).

**Done (v2):** open `/index-bird.html` — a flapping-V neon bird flies (chase cam) over layered ridgelines receding into haze with real 3D depth; the wind nudges its flight; 60fps; no errors. Confirm the LOOK, then re-add controls/tuning.

---

## v3 (2026-06-11) — look dial-in: ground-locked camera, EKG terrain, glide-not-flap

v2 result: terrain DEPTH read well (good) but two corrections from the user looking at it live:
- "it looked REALLY good for a moment, then the terrain flew out of camera" → the camera lost the ground when the bird pitched.
- "less topographic, more like a series of EKG lines indicating terrain. more stylized."
- "lets start with a glide, no flapping, i should be able to glide."

**Camera rule (hard):** the BIRD can change orientation freely (pitch/roll/bank — its mesh orients), but **the CAMERA ALWAYS SHOWS THE GROUND.** Implementation: chase cam behind+above, **world-up always** (never rolls with the bird), follows the bird's POSITION and HEADING (yaw) ONLY — it IGNORES the bird's pitch/roll for camera aim. Clamp the camera pitch so terrain always fills the lower portion of the frame; it never points at sky-only. The "terrain flew out" bug = the camera was inheriting the bird's pitch; decouple them.

**Terrain restyle — EKG/waveform, not topographic. NO FILL — LINES ONLY.** Render the terrain as a **series of stacked horizontal neon trace lines** (Joy Division "Unknown Pleasures" / oscilloscope look), NOT contour iso-lines, and with **NO filled or shaded terrain surface AT ALL** — the ground is dark/empty and the terrain exists PURELY as glowing neon lines on it. The filled terrain mesh from v2 is REMOVED, not drawn-under the lines (the fill is what looked bad). Just lines on the dark ground. Each line = a horizontal polyline across the view at a fixed depth row ahead of the camera, vertically displaced by the terrain height profile along that row (the "EKG spike"). ~30–60 rows receding; near rows brighter/taller, far rows compress and fade into haze on near-black. Rows scroll/recycle as the camera advances. The SAME fBm heightfield underneath drives both the line displacement and the bird's ridge-lift — only the rendering style changes. More stylized, less map-like.

**Two hard checks (user, 2026-06-11):** (a) **the horizontal lines ARE the ridges** — each line's spikes are the actual terrain ridge crests (driven by fBm), not flat/decorative scanlines. (b) **the terrain MOVES, never static** — as the glider flies forward the ridge-lines continuously STREAM toward the camera and recycle at the horizon (one continuous scrolling world). Verify motion with TWO captured frames, not one.

**Bird — GLIDE, no flap:** a clean **gliding V** (wings held OUT, no flap cycle), bolder/brighter/readable than v2's hairline (thick neon ribbons, real wingspan), banks (rolls) into turns. **Flight = soaring glide, no flap thrust:** gravity + airspeed-lift (keeps it aloft) + ridge-lift/thermals (so you can sustain/gain altitude — "i should be able to glide"); steer pitch + bank with the mouse. Subtle wing flex is fine; NO flapping beats and NO flap input this pass. The new ground-locked camera (looking down at the bird's back) also makes the V read.

**Done (v3):** open `/index-bird.html` — a readable gliding-V neon bird soars over stacked EKG ridgeline terrain receding into haze; the camera always keeps the ground in frame no matter how the bird pitches; you can steer and sustain a glide on lift; 60fps; no errors.

v3 RESULT: landed well (no-fill EKG ridges, ground-locked cam, readable gliding-V, terrain streams). Three refinements from the user:

---

## v4 (2026-06-11) — camera-relative rows, clean horizon + higher start, FELT wind

1. **Skew fix → EKG rows are CAMERA-RELATIVE (screen-horizontal at every heading).** v3 rows were locked to world-East, so turning tilted them diagonally. Fix: generate each row PERPENDICULAR to camForward — a line at distance d ahead spanning camera-Right (−W/2..W/2), height = fBm at those world XZ. The terrain content still comes from the same world fBm; only the line orientation follows the camera, so the stacked lines stay horizontal on screen no matter which way you fly.

2. **Clean horizon + higher start.** User: "i'm ok occluding lines in the distance" + "i want to start higher from the ground." So: AGGRESSIVELY occlude/cull/fog-out the distant rows (accept losing far detail) to kill the horizon tangle — hard fog cutoff and/or a max draw distance and/or depth-occlude far lines behind near ridges. AND raise the bird's START altitude well above the terrain (more aerial; also pushes the tangle below the eyeline).

3. **Make the WIND FELT.** User: "i'm not feeling the wind at all." The coupling is too weak and invisible. (a) CRANK wind coupling so the glide is visibly shoved/carried/lifted — gusts blow you off heading (you must correct), thermals/ridge-lift clearly carry you up. (b) Make it VISIBLE — exaggerated streamline ribbons / drifting wind traces over the terrain (§7.1 "stylized wind, more legible than literal") so you SEE the field you're feeling. (c) Wire the REAL GPU fluid as the wind source if it can be sampled cleanly (no in-loop sync readback); else keep strong curl-noise and FLAG it. The bird should visibly drift, lift, and fight the air; the overlay's wind vector should be large and clearly acting on the flight.

**Done (v4):** open `/index-bird.html` — EKG ridge-lines stay horizontal on screen in every turn; the horizon is clean (far lines occluded); the bird starts high and aerial; and the WIND is unmistakable — visible streamlines + the glider is clearly pushed/lifted/carried by it.

---

## v5 (2026-06-11) — glider energy fix + denser occluding terrain + wind DOTS

**Glide energy — FIXED (committed 768af6f).** User: "i seem to keep flying up." Cause: v4 cranked the horizontal-wind gain and that ALSO multiplied the vertical thermal updraft, while thermals were broad/frequent — so updraft beat the small sink almost everywhere → constant climb. Fix: decoupled thermal from windGain, made thermals SPARSE narrow cores (pow), raised base sink (1.4→2.2), lowered ridge liftGain (2.2→1.2), added a hands-off pitch auto-trim to a gentle descent. Verified: hands-off the glider sinks ~2.8 m/s (195m→57m/5s); you must HUNT lift to stay up — the soaring contract.

**Terrain (v5):**
- **2× line density** — double the EKG rows (and/or samples per row) for finer ridges.
- **BLACK FILL to occlude (hidden-line removal).** NOT the shaded fill that looked bad — a BACKGROUND-COLORED (black) fill UNDER each ridge line so nearer ridges HIDE the lines behind them (Joy Division technique: draw back-to-front, each line's black fill paints over the far lines; or depth-buffer equivalent). Result stays lines-only in look, but the tangle is gone because far lines are properly occluded by near terrain.
- **Elevation coloration hints** — tint the lines by terrain HEIGHT (e.g., low/valley → cool, high/peak → warm/bright), so color reads elevation.

**Wind (v5): DOTS, not lines.** Replace the streamline comet *lines* with drifting neon DOT particles advected by the SAME `windAt` field — a field of motes carried by the wind, showing flow through their motion and density (recycle as they age/leave). Over the terrain, depth-tested, additive glow.

**Done (v5):** denser EKG terrain with black-fill hidden-line occlusion (clean, no tangle) + elevation-tinted lines; wind shown as drifting dots; and the glider sinks by default, soaring only on found lift.

---

## v6 (2026-06-11) — dot tails + smaller, and fix bird-vs-terrain scale

- **Wind dots → smaller + TAILS.** v5 dots read like a starfield when frozen. Make each mote SMALLER (dotPx ~11 → ~4-6) and give it a short fading COMET TAIL along the wind direction / its recent path (a few dot-widths). Motion + direction then read even in a still, and they're distinct from the sky stars. Still advected by the shared windAt.
- **Bird-vs-terrain SCALE.** User: "the bird seems HUGE compared to the terrain." The bird is ~36 m wingspan against ~120 m relief, so it dominates the frame. Shrink the bird's world size (wingspan ~36 m → ~14-18 m) AND pull the chase camera in proportionally (reduce followDist) so the bird stays readable but is clearly a SMALL glider in a VAST landscape — the terrain should dominate, the bird a small soaring thing within it. Tune by screenshot until the proportion feels right.

**Done (v6):** wind reads as small tailed motes streaking on the breeze; the bird is a small glider dwarfed by big ridgelines. (Committed 0a3b3f6: bird halved + camera pulled in, dots smaller+tailed.)

---

## v7 (2026-06-11) — wind particles: more, smaller, GROUPED, longer tails

On top of v6 (bird scale done — do NOT change it). Refine ONLY the wind motes:
- **MORE** particles (raise the count substantially).
- **SMALLER** (smaller than v6's already-shrunk dots).
- **GROUPED** — cluster the motes into gusts/eddies (a set of cluster centers, each with several motes scattered in a small radius; advect clusters + members by windAt; recycle/reseed clusters ahead). The field should read as discrete gusts/packets drifting through, NOT a uniform even speckle (which read like a starfield).
- **LONGER comet tails** — lengthen the fading tails so the streaking/direction reads clearly.

**Done (v7):** wind reads as drifting CLUSTERS of many tiny long-tailed motes — gusts streaking through the scene, unmistakably wind, not stars.

---

## v8 (2026-06-12) — wind EVERYWHERE, speed-driven density + tail length

v7's hard clustering left big empty gaps. New model (supersedes the gaps):
- **Wind EVERYWHERE** — motes cover the whole view; no large empty dark regions. The whole airspace shows airflow.
- **density + tail length = SPEED.** BOTH encode local wind speed (`|windAt|`): where the wind is FAST → MORE motes (denser) AND LONGER tails; where SLOW → sparser AND shorter. So fast air reads as dense long streaks, slow air as faint short stubs — you read speed off the field directly. The natural spatial variation of windAt then creates gust-like structure WITHOUT hard empty gaps.
- **Longer tails overall** (lengthen the base tail beyond v7's).
Still advected by the shared windAt; only wind.ts/wind.wgsl change. Bird/terrain/camera/physics untouched.

**Done (v8):** wind fills the airspace everywhere; you read its speed from how dense and how long-tailed the streaks are — fast lanes dense and streaky, calm air faint and short. Also fixed flight feel (4958d39): gentler sink, crisper steering, findable+capped lift.

---

## v9 (2026-06-12) — wind FELT on the bird + curved longer trails

- **Wind INTERACTS WITH TERRAIN (the headline).** User: "i'd like to feel like the wind is interacting with the terrain." Today windAt is independent of the landscape. Make the mote flow TERRAIN-AWARE (wind.ts, which already gets sampleHeight): (a) VERTICAL — motes RISE over windward slopes and sink in lees (a vertical flow w = horizontalWind · uphill-gradient, the same ridge-lift the bird rides), so you SEE air pour up and over the ridges; (b) HORIZONTAL DEFLECTION — reduce the into-slope component near steep terrain so flow bends AROUND peaks/over crests rather than through them; optional speed-up over crests. The motes should visibly hug and climb the ridgelines. (The bird already rides this via ridge lift; the buffet below makes it felt.)
- **Trails longer + CURVED, following the terrain-shaped flow.** User: "make the trails even longer and give them some curve." wind.ts/wind.wgsl ONLY: each mote's tail becomes a CURVED multi-segment polyline integrated along the (now terrain-shaped) flow over several steps — long curved comets that arc over the ridges and around peaks, not straight lines; even longer than v8.
- **Wind FELT on the bird (buffeting).** User: "doesn't feel like the wind is really impacting the bird." The drift is large but STEADY, so it's not noticed. Add BUFFETING (bird3d.ts ONLY): a turbulence term that ROCKS the bird's bank (±~6-8°), BOBS it vertically (±~1.5 m/s), and SHOVES it laterally in gusts so flying feels like moving air; plus a visible LEAN/crab into the cross-wind. Keep it controllable. Verify: flying straight with no input, bank + vario OSCILLATE (buffet active) where before they were near-constant.

**Done (v9):** the wind visibly pours over and around the ridges (terrain interaction); its streaks are long curved comets following that flow; and the bird rocks/bobs/crabs so you feel the moving air. (v9 result: curved tails + buffet landed; terrain-interaction too subtle; drift +61° maybe too strong.)

---

## v10 (2026-06-12) — density near the bird; terrain interaction must HUG and POUR

wind.ts/wind.wgsl ONLY (do NOT touch bird physics/camera/terrain).
- **Density: DENSE near the bird, SPARSE far.** v9 seeds motes uniformly in world space, so perspective makes the FAR field look densest and the near field thin — the OPPOSITE of what's wanted. Bias the mote distribution toward the near field (concentrate around the bird/camera ground point; fewer + fading motes with distance) so the wind is thickest right around the bird and thins into the distance.
- **Terrain interaction must HUG + POUR (it's still too weak).** v9 motes float in a flat layer ~45-55 m ABOVE the ridges, so they don't visibly relate to the terrain. Fix: motes HUG the terrain surface (much lower clearance, follow the contour up and over each ridge) AND a STRONGER vertical pour over windward slopes (raise the lift the motes ride; brighten/accelerate motes that are climbing) so the airflow OBVIOUSLY conforms to and pours over the landscape — you should clearly see wind streaming up the windward faces and spilling over the crests, not floating in a flat sheet.

**Done (v10):** the wind is a thick cloud of streaks right around the bird thinning into the distance, and it visibly hugs the terrain — pouring up the windward ridges and spilling over the crests.

---

## Felt wind (2026-06-12, committed 4d1c306) — autopilot off + camera follows ground-track
Root cause of "I don't feel the wind impacting the bird" (3 rounds): the user was watching the AUTOPILOT fly smoothly THROUGH the wind, and the chase camera followed HEADING so drift was invisible. Fix (advisor-guided): default to MANUAL (P toggles autopilot, kept), and point the camera down the GROUND-TRACK (blend 70%) so the bird visibly CRABS into the wind. User confirms "wind is better" now.

## v11 (2026-06-12) — wind LEGIBLE at the bird: two tiers (distant lines + near comet sphere)
User: "i see wind all around, but not next to the bird. keep the long lines for the in-the-distance wind, and the little comets for the up-close wind sphere." Plus the standing note: "wind is a much lower density than what it's rendered as" (air is thin — keep it airy).
- **FAR (distance) = LONG curved streamline LINES** — keep the existing long curved tails for the distant wind. Sparse/airy.
- **NEAR (up-close) = a WIND SPHERE of LITTLE COMETS around the bird** — a ball/volume of small short-tailed comets centered on the BIRD (radius ~60–100 m), advected by the local terrain-aware flow, dense enough to be LEGIBLE right where the bird is (showing the air you're flying through); recycle comets that leave the sphere back in. This is the fix for "wind not legible next to the bird."
- **RENDER at HIGHER density** (user clarification 2026-06-12: "air calculations for fluid are low density, render at a higher density"). The air *calc/physics* is low-density (thin medium — correct), but the VISUAL must be DENSE and clearly visible so you can read it: MORE comets, not fewer — especially a THICK sphere of comets at the bird. (This CORRECTS the earlier "airy/don't crowd" wording — low-density CALCULATION, high-density RENDER.)
- **Respect terrain** (comets stay above ground and flow over it).
Scope: wind.ts + wind.wgsl, plus the single `wind.draw(...)` call in bird-main.ts (to pass the bird position as the sphere center) — nothing else.

**Done (v11):** distant wind reads as long curved lines; right around the bird a sphere of little comets shows the local air, so the wind is legible exactly where you are; airy, terrain-respecting. (+ v12: motes fade in/out on recycle — no pop; + wind 50% opacity. + felt-wind: MANUAL default, camera follows ground-track → bird crabs.)

---

## v13 (2026-06-13) — wire the REAL GPU fluid as the wind source (AUTHENTICITY pass, not a feel fix)

User: "yes, real gpu fluid." Replace the analytic curl-noise BASE with the real GPU Stam fluid (src/host/gpu/fluid.ts GpuFluid — step(encoder,dt,iters), exposes velocityX/Y GPU buffers). Success = real EVOLVING wind, NOTHING regressed, still 60fps. (Feel was already fixed by camera+control — do NOT crank fluid force to chase feel; that's the +61° blown-around bug returning.)

**Architecture (advisor, non-negotiable): FLUID = the new curl-noise (the structured horizontal SOURCE), NOT the new windAt. Everything wrapping windAt STAYS.**
- `windAt(x,z,t)` BASE = bird-local GPU-fluid sample (async readback) + KEEP the PREVAILING DRIFT (driftAmp/driftDir). Fallback to analytic curl-noise until the first readback resolves. (A zero-mean Stam swirl with no drift collapses ground-track onto heading → bird stops crabbing → regresses "I don't feel it." Keep the drift.)
- `flowAt` terrain-coupling (vertical pour over ridges + into-slope deflection) — UNCHANGED, wraps windAt exactly as today. Do NOT return the raw flat 2D fluid — the terrain-pour win must survive.
- `thermalAt` — UNCHANGED. Bird physics (bird3d) + motes (Wind) both call windAt/flowAt → automatically ride the fluid; NO rewrite of either.

**Wiring:** bird-main.ts instantiates GpuFluid (~256²) over a BIRD-LOCAL window that MOVES with the bird (never fly into dead air; world-pinned moving-window deferred); steps it each frame; FORCES it continuously with a PREVAILING flow + BOUNDED structure tuned to the ~10–15 m/s flyable band (NOT raw impulse spikes). Async-readback u/v via ReadbackRing (2–3 frames stale, fine). wind.ts gets a per-frame fluid-field setter + world→grid bilinear sampler. Vertical lift stays terrain-derived ridge lift (fluid is 2D/horizontal, §6.1 2.5D); deep fluid-as-terrain-boundary coupling is LATER.

**NO-REGRESSION GATE (the still looks like v12 — verify by MEASUREMENT):**
(a) 60fps HOLDS with the fluid step ADDED (measure).
(b) terrain pour + crab/drift (non-zero DRIFT) + mote fade + near-sphere legibility ALL still present.
(c) PROOF the field is the fluid: sample one FIXED world point across ~3s and show the wind vector EVOLVES (curl-noise was quasi-static; the fluid changes).

**Scope:** bird-main.ts + wind.ts (+ small fluid-forcing helper if needed); REUSE GpuFluid + ReadbackRing. Do NOT touch bird3d.ts, terrain.ts/terrain_ekg.wgsl, camera.ts, autopilot.ts.

**Done (v13):** the wind is the real GPU fluid — evolves over time (real gusts), motes + bird ride it, terrain-pour + crab + fade all intact, 60fps holds.
