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
