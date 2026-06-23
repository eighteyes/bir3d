# Human Review Steps

## wind render modes — phase 1 scaffold (per-tier A/B/C switch)
**Date:** 2026-06-23
**Commits:** 7c774f4, 06de41e, 4a37c60, 955a53c (Phase 1 of the plan; B/C deliberately inert)
**Session:** wind-render-modes (f65433b5-2d4a-4594-85b6-036787d8f3af)
**Design:** .ai/explore/2026-06-23-wind-render-modes-design.md · **Plan:** .ai/plan/wind-render-modes/

### What changed (RENDER/UI ONLY — feel/physics frozen; wind.wgsl untouched)
- Per-tier render-MODE switch on the `Wind` class: `farMode` (comet|stipple|chevron), `nearMode` (comet|flecks|filaments), `wakeMode` (modulate|helix|rings); setters `setFarMode/setNearMode/setWakeMode`. Mode ARRAYS (`FAR_MODES`/`NEAR_MODES`/`WAKE_MODES`) are the single source of truth; the union types derive from them.
- Phase 1 implements ONLY the "A" look (comet/comet/modulate = today's render). B/C modes are wired end-to-end but FALL THROUGH to A — no divergent geometry yet (that's Phase 2). The comet emission was extracted into `emitFarComet`/`emitNearComet` so Phase 2 geometries slot in as sibling methods.
- Far comet shortened (`segments` 6→4, `FAR_SUBDIV` 3→2) — the ONLY deliberate visual change; the far tier reads shorter / corner-proof.
- Vertex buffer reserves a THIRD span (wake-shed, 7680 v) for Phase 2 helix/rings; `draw()` refactored into up to 3 offset draws (far always; near if `showNear`; wake-shed if `showWake && wakeMode!=="modulate"` — never drawn this phase).
- T-panel: a generic `cycleBtn` + three FAR/NEAR/WAKE mode buttons under a "wind — render modes" group. Console handles `__farMode/__nearMode/__wakeMode`.

### Verified (automated gate, headless — full suite green, tsc exit 0)
- 6/6 GPU specs pass: `wind-render-modes` (smoke, fps 60), `touched-air`, `slipstream`, `wind-live`, `wind-atmosphere`, `updraft-buffer` — no pageerrors, errors=0, fps 35–60.
- Frozen-feel CONFIRMED byte-level: diff `70a4408..955a53c` touches only `wind.ts`, `bird-main.ts`, and the new spec — zero changes to bird3d/physics, autopilot, `windAt`/`windProfile`/`updraftAt`, or `wind.wgsl`.

### Pre-work — launch a server FROM THIS WORKTREE (fresh strict port; stale vites can squat ports)
```
cd /Users/god/projects/ai-jank/vector-system/.claude/worktrees/mountaintop-forests && ./node_modules/.bin/vite --port 5274 --strictPort
```

### Verify — cycle the modes in the panel (manual)
```
open -a "Google Chrome" "http://localhost:5274/index-bird.html"
```
- [ ] Press `T` to open the tuning panel; find the "wind — render modes" group.
- [ ] Click FAR / NEAR / WAKE to cycle each (comet▸stipple▸chevron, etc.). Should NOT crash and should NOT change the look yet — B/C fall through to A by design in Phase 1.
- [ ] Toggle "local sphere" and "wake" on (same panel), cycle NEAR/WAKE again — still no crash.
- [ ] Confirm the far wind comets read SHORTER than before (the one intended visual change).

### Verify — cycle from the console
```
__farMode("chevron"); __nearMode("flecks"); __wakeMode("rings")
```
- [ ] No errors; demo keeps running. Reset: `__farMode("comet"); __nearMode("comet"); __wakeMode("modulate")`.

### Verify — re-run the smoke spec (fresh server running first)
```
cd /Users/god/projects/ai-jank/vector-system/.claude/worktrees/mountaintop-forests && ./node_modules/.bin/playwright test tests/gpu/wind-render-modes.spec.ts --reporter=line
```

### Verify — re-run the frozen-feel guard
```
cd /Users/god/projects/ai-jank/vector-system/.claude/worktrees/mountaintop-forests && ./node_modules/.bin/playwright test tests/gpu/touched-air.spec.ts tests/gpu/wind-live.spec.ts tests/gpu/wind-atmosphere.spec.ts --reporter=line
```

## global wind = altitude atmosphere (gameplay: calm low → strong high)
**Date:** 2026-06-22
**Commit:** (uncommitted — worktree `.claude/worktrees/mountaintop-forests`, branch `worktree-mountaintop-forests`, based on 58b7e44)
**Session:** global-wind-atmosphere (f65433b5-2d4a-4594-85b6-036787d8f3af)
**Design:** .ai/explore/2026-06-22-global-wind-altitude-atmosphere-design.md · **Plan:** .ai/plan/global-wind-atmosphere/

### What changed (GAMEPLAY — unfreezes the bird's windAt consumption)
- New `windProfile(absoluteY)` (wind.ts): one magnitude curve, calm low → strong high (smoothstep `altLo..altHi`). Multiplied into horizontal wind at EVERY consumer — **uniform, no special rules**.
- **Drift** (bird3d.integrate) scales by `windProfile(birdY)` — calm valleys, strong open air.
- **Ridge lift** (`updraftAt`) uses `windAloftScale()` — the STRONG free-stream wind ALOFT, altitude-INDEPENDENT. A 150m ridge has strong wind over it just like a 500m peak, so **ridge soaring works at any ridge height and the bird is never stranded low**. (An earlier draft scaled lift by the bird's absolute altitude — adversarial review caught that it nerfed soaring 25–60% on low/mid ridges and could strand a bird in a valley; corrected to aloft per the original "strong aloft wind" intent.)
- Terrain shelter (calm valleys) emerges from the DRIFT profile; no lee rule.
- **Motes** advect by `windAt × windProfile(moteY)` — peak-huggers rip, valley-huggers idle (gradient reads through speed). Far tier reverted to **terrain-hugging** (clearance 60→30) with `homeBias` clustering low + a thin tail aloft so altitude isn't a dead void.
- Thermals untouched (separate vertical system). Live tuning: `__windProfile({loScale,hiScale,altLo,altHi})`, `__windProfileAt(y)`.

### Verified (automated gate, headless — `tests/gpu/wind-atmosphere.spec.ts`)
- Curve monotonic: `0:0.40 → 600:1.40`. Ridge lift consumes it: unsaturated spot low=2.33 vs high=7.90 (Δ=5.57).
- SOARING PRESERVED: autopilot rides 8.0 m/s updraft, no crash, `fps=60`, no errors. Slipstream + touched-air gates still pass (no regression).

### Pre-work — launch a server FROM THIS WORKTREE
```
cd /Users/god/projects/ai-jank/vector-system/.claude/worktrees/mountaintop-forests && ./node_modules/.bin/vite --port 5273 --strictPort
```

### Verify — FEEL the atmosphere (manual flight)
```
open -a "Google Chrome" "http://localhost:5273/index-bird.html"
```
- [ ] Dive into a valley: drift goes calm (sheltered), the air settles.
- [ ] Climb into open air / over a high ridge: drift shoves harder, `wind` HUD rises — and ridge soaring still lifts you (climb the windward face, vario stays ▲).
- [ ] Press `P` (autopilot) and walk away: it still finds lift and does NOT spiral in.

### Verify — tune the curve live (console)
```
__windProfile({ loScale: 0.2, hiScale: 1.8 })
```
- [ ] Deader valleys / wilder heights. `__windProfileAt(50)` vs `__windProfileAt(450)` shows the curve. Reset: `__windProfile({loScale:0.4, hiScale:1.4, altLo:100, altHi:500})`.

### Verify — re-run the automated gate
```
cd /Users/god/projects/ai-jank/vector-system/.claude/worktrees/mountaintop-forests && ./node_modules/.bin/playwright test tests/gpu/wind-atmosphere.spec.ts --config .ai/tmp/wind-verify.config.ts --reporter=line
```

## wing slipstream (twin wingtip vortices + body that sticks)
**Date:** 2026-06-22
**Commit:** (uncommitted — worktree `.claude/worktrees/mountaintop-forests`, branch `worktree-mountaintop-forests`, based on 58b7e44)
**Session:** felt-wind / slipstream (f65433b5-2d4a-4594-85b6-036787d8f3af)

### What changed (RENDER-ONLY — near-mote sphere; flight physics + far tier + vertex format untouched)
- **V — twin counter-rotating wingtip vortices** replace the single central swirl in `birdWakeAt` (wind.ts). Two cores at `birdPos ± wingSpan·right` (`right = axis × worldUp`), Rankine falloff peaking at `vortexCore`, circulation sign flips per side → the pair counter-rotates and trails BEHIND the wings.
- **B — body attach.** Near the bird the ambient terrain-wind is attenuated (`ambientNearFloor` at the bird, ramping to full at the ball edge) so the near sphere rides the bird's own wake and STICKS instead of blowing downwind. Applied to head advection + the curling tail.
- **C — wingtip emission.** `seedNearMote` now births a `wingEmitFrac` fraction of motes at the wingtips (slightly ahead, with jitter) so the two streams are visibly born at the tips; the rest fill the body uniformly.
- Live tuning hooks (console): `__wind.swirlGain`, `__wind.wingSpan`, `__wind.vortexCore`, `__wind.wingEmitFrac`, `__wind.ambientNearFloor`. Probes `__nearWake(x,y,z)` / `__nearFrame()`.

### Verified (automated gate, headless ANGLE/Metal — `tests/gpu/slipstream.spec.ts`)
- COUNTER-ROTATION: circulation behind the two wingtips reads `right=-3.65`, `left=+12.67` — opposite sign, real magnitude ⇒ the vortices genuinely counter-rotate.
- `fps=60`, no crash, no pageerrors. Screenshot: `test-results/slipstream.png`.

### Pre-work — launch a server FROM THIS WORKTREE (a stale `:5173` from another worktree serves OLD code)
```
cd /Users/god/projects/ai-jank/vector-system/.claude/worktrees/mountaintop-forests && ./node_modules/.bin/vite --port 5273 --strictPort
```

### Verify — SEE the slipstream (manual flight, WebGPU browser)
```
open -a "Google Chrome" "http://localhost:5273/index-bird.html"
```
- [ ] In a fast glide, two mote streams visibly trail off the wingtips (not one central spiral).
- [ ] The near sphere reads as ATTACHED to the bird — motes flow past/with it rather than blowing away downwind.
- [ ] The streams curl (counter-rotating corkscrews), strongest just behind the wings, fading by the ball edge.

### Verify — tune the feel live (browser console)
```
__wind.wingSpan = 16
```
- [ ] Wider tip separation → the two streams move apart. Try `__wind.swirlGain = 1.2` (more curl), `__wind.wingEmitFrac = 0.7` (denser streams), `__wind.ambientNearFloor = 0.05` (sticks harder).

### Verify — re-run the automated gate (needs the worktree server on :5273 from pre-work)
```
cd /Users/god/projects/ai-jank/vector-system/.claude/worktrees/mountaintop-forests && ./node_modules/.bin/playwright test tests/gpu/slipstream.spec.ts --config .ai/tmp/wind-verify.config.ts --reporter=line
```

## updraft buffer off hills (ridge lift kicks in earlier — L+B)
**Date:** 2026-06-22
**Commit:** (uncommitted — worktree `.claude/worktrees/mountaintop-forests`, branch `worktree-mountaintop-forests`, based on 58b7e44)
**Session:** felt-wind / updraft-buffer (f65433b5-2d4a-4594-85b6-036787d8f3af)

### What changed (FLIGHT PHYSICS — previously frozen, unfrozen on request)
- The bird used to need to skim a slope before ridge lift kicked in. Two mechanisms give a buffer so lift appears earlier on the approach:
  - **L (lookahead)** — ridge lift also samples the terrain gradient `ridgeLookahead` m DOWNWIND (toward the windward face the wind compresses into) and takes the MAX with the local value. Lift can only appear EARLIER, never less than the bird's own position already gives.
  - **B (broaden)** — the ridge gradient is a central difference over a wider half-step `ridgeEps` (was a hardcoded 6 m) → a softer, wider lift band reaching off the face.
- De-duplicated the ridge-lift math: `Bird3D.integrate` now CALLS the exported `updraftAt` (single source of truth) instead of a hand-synced copy — so the autopilot senses EXACTLY the air the bird rides. The horizontal-drift deflection gradient is untouched (still eps=6, still matched to the motes).
- New live `T`-panel sliders: `ridgeLookahead` (0–150 m, default 50) and `ridgeEps` (6–40 m, default 14). Dial to taste.
- Debug probe `window.__updraftAt(x, z, tuneOverride?)` returns the exact ridden updraft; pass `{ridgeLookahead:0}` to compare against the no-buffer field.

### Verified (automated gate, headless ANGLE/Metal — `tests/gpu/updraft-buffer.spec.ts`)
- Lift band L+B vs no-buffer baseline: **6397 vs 4631** lift cells (of 10201) — ~38% wider.
- **1823 buffer cells**: spots where the old geometry gave ~no lift but L+B gives real lift (≥0.8 m/s).
- `maxLift = 8.00` both configs — the anti-launch cap (8 m/s) still holds; physics not destabilized.
- Flight under autopilot: rides updraft 8.0 m/s, `fps=60`, no crash, no pageerrors.

### Pre-work — launch a server FROM THIS WORKTREE (a stale `:5173` from another worktree serves OLD code)
```
cd /Users/god/projects/ai-jank/vector-system/.claude/worktrees/mountaintop-forests && ./node_modules/.bin/vite --port 5273 --strictPort
```

### Verify — FEEL the bigger buffer (manual flight, WebGPU browser)
```
open -a "Google Chrome" "http://localhost:5273/index-bird.html"
```
- [ ] Fly toward a windward (into-the-wind) hill at a shallow approach: `vario` goes positive (▲) / `updraft` climbs while you are still well OFF the slope — not only once you are skimming it.
- [ ] Press `T`, drag `ridgeLookahead` to 0: lift should now feel "late" again (you must get close). Drag it back up (50–120): lift returns earlier. That swing IS the buffer.
- [ ] Drag `ridgeEps` higher: the lift band feels broader/softer; lower (6): tighter and peakier.
- [ ] Press `P` (autopilot) and walk away: it still holds course and does NOT spiral into the ground (the autopilot senses the same buffered lift).

### Verify — measure the buffer (browser console)
```
__updraftAt(__birdPos[0], __birdPos[2])
```
- [ ] Compare against the no-buffer field at the same point: `__updraftAt(__birdPos[0], __birdPos[2], {ridgeLookahead:0, ridgeEps:6})` — the first (L+B) should be ≥ the second wherever a windward face is within lookahead.

### Verify — re-run the automated gate (needs the worktree server on :5273 from pre-work)
```
cd /Users/god/projects/ai-jank/vector-system/.claude/worktrees/mountaintop-forests && ./node_modules/.bin/playwright test tests/gpu/updraft-buffer.spec.ts --config .ai/tmp/wind-verify.config.ts --reporter=line
```

## wind re-enabled (bird feels it + motes drawn)
**Date:** 2026-06-21
**Commit:** (uncommitted — worktree `.claude/worktrees/mountaintop-forests`, branch `worktree-mountaintop-forests`, based on 58b7e44)
**Session:** wind-reenable (f65433b5-2d4a-4594-85b6-036787d8f3af)

### What changed
- Both wind consumers turned back ON in `bird-main.ts` (the fluid sim itself never stopped — only its two consumers were parked):
  - PHYSICS: `bird.stillAir = false` (was `true`). The bird now flies the moving fluid field — horizontal drift you must correct, ridge lift + thermals to ride, buffet/gust shake. `windGain` (bird3d tuning, default 1.6) scales the shove.
  - MOTES: `showWind = true` (was `false`). The neon streamline-comet motes draw over the ridges again, fed by the SAME `windAt` field that pushes the bird. Toggle live with `window.__showWind(false|true)`.
  - Comments corrected to match (no longer claim "still-air basis" / "hidden for now"); HUD title relabelled `still-air glider` → `wind glider`.
- GATE (`tests/gpu/wind-live.spec.ts`, NEW): boots the live app, hands control to autopilot, samples telemetry ~3 s, and asserts wind is FELT (non-zero, flyable band), EVOLVES (moving fluid), DRIFTS (heading vs ground-track gap), stays CONTROLLABLE (no crash), runs (fps>15), and errors-free. Writes `test-results/wind-live.png` (gitignored) as the visual mote proof.

### Verified (automated gate, headless ANGLE/Metal)
- `meanMag=11.8 m/s  maxMag=14.3` — dead-center in the documented flyable band (regulator targets mean ~10 / max ~16).
- `evolveΔ=1.35` (field is the moving fluid, not frozen) · `maxDrift=23.4°` (bird genuinely crabs) · `fps=60` · `crashed=false` · no pageerrors.

### Pre-work — launch a server FROM THIS WORKTREE (a stale `:5173` from another worktree serves OLD code)
```
cd /Users/god/projects/ai-jank/vector-system/.claude/worktrees/mountaintop-forests && ./node_modules/.bin/vite --port 5273 --strictPort
```

### Verify — SEE the wind (open in a WebGPU browser: Chrome/Edge)
```
open -a "Google Chrome" "http://localhost:5273/index-bird.html"
```
- [ ] Cyan neon mote-comets stream over the ridges (not a still scene).
- [ ] Bottom-right compass shows a GAP between the cyan (heading) and yellow (ground-track) arrows — that gap is the drift.
- [ ] HUD `wind:` line reads non-zero m/s (≈10–15); HUD title reads `wind glider`.

### Verify — FEEL the wind (manual flight)
- [ ] Mouse-steer across a ridge: the bird visibly crabs sideways (you correct heading) and `DRIFT` on the HUD swings non-zero.
- [ ] Cross a windward face: `vario` goes positive (▲) as ridge lift carries you — climb without flapping.
- [ ] Press `P` (autopilot) and walk away: it holds course and does NOT spiral into the ground.

### Verify — the motes toggle (browser console)
```
window.__showWind(false)
```
- [ ] Motes vanish, bird KEEPS drifting (physics independent of the render toggle). Re-run with `true` to restore.

### Verify — re-run the automated gate (needs the worktree server on :5273 from pre-work)
```
cd /Users/god/projects/ai-jank/vector-system/.claude/worktrees/mountaintop-forests && ./node_modules/.bin/playwright test tests/gpu/wind-live.spec.ts --config .ai/tmp/wind-verify.config.ts --reporter=line
```
NOTE: the `.ai/tmp/wind-verify.config.ts` override is needed ONLY because stale vite servers from other worktrees occupy `:5173` and the committed `playwright.config.ts` has `reuseExistingServer:true` (it would test their OLD code). In a clean environment with nothing on `:5173`, the standard `./node_modules/.bin/playwright test tests/gpu/wind-live.spec.ts` starts its own correct server.

## mountaintop neon forests (algorithmic trees)
**Date:** 2026-06-18
**Commit:** (uncommitted — worktree `.claude/worktrees/mountaintop-forests`, branch `worktree-mountaintop-forests`, based on 99bfc9a)
**Session:** mountaintop-forests (isolated worktree; core was being edited by another agent)

### What changed
- TREES MODULE (`gpu/trees.ts` + `shaders/trees.wgsl`, NEW): a dense, elevation-banded neon forest streamed around the camera, baked CPU-side into ONE line-list vertex buffer (no instancing). SPECIES BY ELEVATION — deciduous (oak/maple, warm-green) in the valleys, conifers (fir/pine, cool blue-green) on the tops, blended through a treeline; density THINS above the alpine line. CLUSTERED into TIGHT isolated stands — a high density-field threshold (only the noise peaks grow forest) with a narrow band for sharp stand edges (treeline feel), not a wide sprinkle. SIMPLE glyph geometry (conifer ≈7 segments, deciduous ≈5) so the forest can be DENSE — ~1900 trees in view (cap 4000). Small (deciduous ≈9 m, conifer ≈12 m; ~4% larger "ancient"). ANCHORING (root-cause fix): trees no longer trust the CPU height mirror — each tree's ground Y is computed in the VERTEX SHADER with the terrain's exact fBm (same function + f32 precision as the rendered terrain). The old `sampleHeight` (TS, f64) diverged from the GPU (f32) in the `sin`-hash and drifted MORE with distance from origin → the "floating" the user saw, worst far out. Now trees sit on the surface at any distance (verified at world x=9000). Vertices store local offsets + base XZ; the shader adds `fbm(baseXZ)`. Bases sunk 4 m. Per-vertex HDR color (additive → bloom). DISTANCE FADE (radial + exp fog) so trees rise from nothing at the window rim instead of popping. Depth-tested (`depthCompare "less"`) so ridges occlude. Vertex buffer rebuilt only on camera cell-crossing (≈2.3 ms CPU — simple geometry kept the rebuild cheap).
- WIRING (`bird-main.ts`): construct `Trees` after the target with `(x,z)=>terrain.sampleHeight(x,z)`; draw depth-tested between bird and target (ridges occlude); `window.__trees` exposed (`.enabled`, `.treeCount`) for the perf A/B.
- TESTS (`tests/gpu/trees-perf.spec.ts`, `tests/gpu/trees-live.spec.ts`, NEW): FPS A/B + tree count + rebuild cost; offscreen GPU-readback proof the trees render green in the EXACT live config (rgba16float + MSAA 4 + resolve + depth24plus) with a `popErrorScope` guard that catches WGSL pipeline-compile regressions. Visual iteration tool: `.ai/tmp/trees-png.spec.ts` (renders a forest PNG to `.ai/tmp/`).

### Render budget (GPU timestamps, headless ANGLE/Metal @ 1280×720 MSAA4 — relative, real HW is faster)
- Frame budget: **16.67 ms** (60 fps). Holds 60 fps.
- Profiling found: `terrain` ≈14 ms (dominant) and the trees' per-vertex 4-octave fBm = **92%** of the trees cost.
- DONE — terrain rows cut to ~1/4 (266→96) via `rowSpacing 7 / nearDenseDepth 250 / farSpread 70` in
  bird-main (leans on the far-thinning lever): terrain **≈14 ms → ≈3 ms**. Foreground a touch coarser.
- DONE — trees fBm moved from per-vertex to a once-per-tree COMPUTE PREPASS (`shaders/trees_ground.wgsl`):
  ground computed once per tree on rebuild, stored in a buffer the vertex shader reads. Anchoring identical
  (same GPU fBm; verified at x=9000). ~2× on this backend (the vertex-stage storage read costs on ANGLE;
  real HW better). Rebuild dropped to ~1.1 ms (CPU no longer touches fBm).
- `trees.draw(...)` takes an optional trailing `timestampWrites` (profiling hook). `terrain.ts` left pristine.
- NOTE: terrain params live in bird-main (`new TerrainEKG({...})`); `terrain.ts`/`terrain_ekg.wgsl` core unchanged.

### Latest iteration (denser / brighter / wind-off / landmarks / fog)
- WAY MORE TREES: `CELL 14`, `MAX_TREES 9000`, lower COVER thresholds → ~3500 in view. 60 fps, rebuild ~2 ms.
- BRIGHTER foliage (HDR > 1 → bloom fattens them, reading as "thicker" in the neon scene).
- TREES FOG = TERRAIN FOG: `trees.draw(..., fogDensity)` now passed `0.5/1100` from bird-main (was 0.5/1400).
- WIND HIDDEN: `showWind=false` in bird-main gates the mote pass (fluid sim still runs); `window.__showWind(true)` restores.
- LANDMARK GIANTS: a few big recursive (L-system, depth 6) trees on PEAKS as waypoints — coarse-grid placement
  at local high points above `LANDMARK_MIN_E`, bright cyan-white. `trees.landmarks` lists their world XZ.
- Terrain animation is camera-relative world sampling (no time term); user set `rowSpacing 2` (denser near lines).
- THREE TERRAIN MODES (`window.__terrainMode("ekg" | "grid" | "topo")`), default **ekg**:
  - `ekg` — original camera-relative scan-lines (lines run AWAY as you fly). `terrain.ts` UNTOUCHED.
  - `grid` — world-static wireframe grid draped on the fBm; flows toward you with parallax.
  - `topo` — world-static TOPOGRAPHIC contour lines (constant-elevation, per-fragment fBm → smooth).
  - Both world-static modes live in `gpu/terrain-grid.ts` + `shaders/terrain_grid.wgsl` (new, separate from
    the EKG core): a windowed draped mesh; FILL gives hidden-surface removal; same fBm so trees/bird sit on
    it. Valid in live MSAA4/rgba16float, 60 fps.
  - TOPO tuning: contours computed from the SMOOTH (pre-terrace) fBm → even spacing; elevation brightness
    ramp (`floorFade` dims lowlands, `peakGain` lights peaks; dark between lines). Live tuning-panel (T)
    controls: a mode button (cycles ekg/grid/topo) + sliders `interval` / `floorFade` / `peakGain` / `lineWidth`.

### Verify — typecheck (from the worktree)
```
cd /Users/god/projects/ai-jank/vector-system/.claude/worktrees/mountaintop-forests && ./node_modules/.bin/tsc --noEmit && echo OK
```

### Verify — perf + render proof (GPU readback; no live server needed beyond the test's own)
```
cd /Users/god/projects/ai-jank/vector-system/.claude/worktrees/mountaintop-forests && ./node_modules/.bin/playwright test tests/gpu/trees-perf.spec.ts tests/gpu/trees-live.spec.ts --reporter=list
```

### Verify — run the sim, then open the bird page
PORT WARNING: this is a worktree and another agent holds 5173–5175, so vite auto-bumps to the next free
port. READ the "Local:" URL vite prints — do NOT assume 5173.
```
cd /Users/god/projects/ai-jank/vector-system/.claude/worktrees/mountaintop-forests && ./node_modules/.bin/vite
```
Then browse to the printed URL + `/index-bird.html` (e.g. http://localhost:5176/index-bird.html). Confirm
trees with `window.__trees.treeCount` in the console (should be > 0) — and check `bird-main.ts` on that
port actually imports Trees if unsure: `curl -s http://localhost:PORT/src/host/bird-main.ts | grep Trees`.

### Visual checklist (human eyeball — fly across valleys AND peaks)
- [ ] SPECIES BY ELEVATION: warm-green broadleaves (oak/maple) in the valleys; cool blue-green conifers (fir/pine) on the high ground; a mixed treeline band between.
- [ ] SMALL + ROOTED: trees are small and sit ON the terrain — no floating, no giants towering over the hills.
- [ ] NATURALISTIC CLUMPS: forests cluster into stands with clearings between, not an even scatter; tops thin out toward bare rock above the alpine line.
- [ ] FADE, NO POP: as you fly, new trees fade UP from nothing at the far edge rather than popping in; distant trees fade into the haze.
- [ ] OCCLUSION: trees behind a nearer ridge are hidden by it (depth-tested).
- [ ] BLOOM: branches glow softly (additive HDR), consistent with the bird/terrain neon.
- [ ] FPS ~60 with the forest on screen (toggle `window.__trees.enabled = false` in the console to A/B).

### Tuning knobs (`gpu/trees.ts`)
- Density / reach: `CELL` (spacing), `RADIUS` (stream distance), `MAX_TREES` (cap).
- Species split: `DECID_MAX` / `CONIFER_MIN` (treeline band, fraction of `PEAK_RELIEF`); `ALPINE` (bare-rock line).
- Clustering: `CLUMP_FREQ` (forest wavelength), `COVER_LO` / `COVER_HI` (clearing↔full-cover thresholds).
- Size: `DECID_H`, `CONIFER_H`, `CONIFER_RAD`, `ANCIENT_FRAC`, `ANCIENT_SCALE`; anchor `ROOT_SINK`.
- Color: `DECID_FOLIAGE` / `CONIFER_FOLIAGE` (+ trunk variants). Fade: `FOG_DENSITY`, fadeStart/End in `draw()`.

---
## powered two-wing flap (climb engine) + view-fog extension
**Date:** 2026-06-18
**Commit:** (uncommitted — working tree on top of 99bfc9a, branch build/foundation)
**Session:** 5b3183c3-54f2-4a41-99a5-7dc3551d3cd2 (flap-to-climb + fog)

### What changed
- POWERED FLAP — climb engine (`gpu/bird3d.ts` + `shaders/bird3d.wgsl`). Modeled as **two wings, NOT a unified force**: each wing makes its own beat force. Their **SUM** = climb (vertical lift + forward thrust so a climb HOLDS airspeed instead of stalling); their **DIFFERENCE** (driven by steering) = turn assist (yaw) + visual bank. **Tap** Space = one beat (runs to completion); **hold** Space = beats repeat → sustained climb. Per-wing visual beat is independent (`ampL`/`ampR`) so an asymmetric beat is *visible* when turning. Bird uniform grown 96→112 B (added `flapPhase`, `ampL`, `ampR`).
- SPACE INPUT + HUD (`bird-main.ts`): `keydown`/`keyup` → `input.flap` (manual flight only; `preventDefault` stops page-scroll). `BirdInput` gained `flap` (autopilot returns `flap:false`). T-panel sliders added: `beatLift`, `beatThrust`, `beatHz`. HUD shows `▲ FLAP` while beating; controls hint updated (`SPACE=flap/climb`).
- VIEW/FOG +50% (`bird-main.ts`): terrain `fogDensity` 0.75/1100→0.5/1100 and mote fog 0.75/1400→0.5/1400 (fog ÷1.5), plus `maxDist` 1900→2850 so the farther-reaching fog doesn't expose a hard horizon shelf.
- `START_CLEARANCE` = 400 (spawn/reset altitude; user-tuned).

### Verify — typecheck (standalone)
```
cd /Users/god/projects/ai-jank/vector-system && ./node_modules/.bin/tsc --noEmit && echo OK
```

### Verify — run the sim, then open the bird page
```
cd /Users/god/projects/ai-jank/vector-system && ./node_modules/.bin/vite
```
Then browse to http://localhost:5173/index-bird.html

### Visual checklist (human eyeball — fly with mouse + Space)
- [ ] FLAP CLIMB: hold Space → the bird climbs steadily and HOLDS the climb (airspeed doesn't bleed to a stall); release → back to the dive/zoom glide. `▲ FLAP` shows in the HUD, vario goes positive.
- [ ] TAP = ONE BEAT: a single Space tap = one wingbeat — a small felt hop up, not nothing.
- [ ] BOTH WINGS BEAT: you can SEE both wings flapping (no longer a static V).
- [ ] ASYMMETRIC TURN: flap while steering hard → the OUTER wing visibly beats harder, banking/yawing into the turn (the "two wings, not unified" proof).
- [ ] CLIMB TO TARGETS: you can now gain altitude to reach high/far beacons (flap up to them).
- [ ] VIEW/FOG: the horizon sits ~50% farther out; no hard shelf/line at the far edge.
- [ ] FPS ~60.

### Tuning knobs
- Flap feel (live, T panel): `beatLift` (climb strength), `beatThrust` (airspeed hold), `beatHz` (beat cadence).
- Flap feel (code, `bird3d.ts` tuning defaults): `beatAmp` (visual amplitude), `flapAsym` (turn-from-asymmetry strength), `flapTurn` (yaw assist).
- View/fog (`bird-main.ts` terrain config): `fogDensity` + `maxDist`; mote fog at the `wind.draw` call.

### Machine verification done (2026-06-18)
- `tsc --noEmit` clean (exit 0).
- Live dev server serves changed assets: `bird3d.wgsl` 200 (new `ampL`/`ampR` uniform present), `bird3d.ts` 200, `bird-main.ts` 200.
- NOT verified by machine: live WebGPU rendering / flap feel — human flight test required.

## still-air glider basis + fly-to-target beacon
**Date:** 2026-06-18
**Commit:** (uncommitted — working tree on top of 99bfc9a, branch build/foundation)
**Session:** 5b3183c3-54f2-4a41-99a5-7dc3551d3cd2 (still-air-glider-fly-to-target)

### What changed
- STILL-AIR FLAG (`gpu/bird3d.ts`): new public `bird.stillAir` (set true in `bird-main.ts`). When true, `integrate()` zeros the wind drift, ridge lift, thermal, and buffet/rock terms — the bird flies a dead-calm downhill glider (gravity + drag + pitch↔speed energy exchange + sink + control). `wind.ts` and the mote overlay are UNTOUCHED (they keep drifting as ambient atmosphere; the bird just ignores them). Set `stillAir = false` to restore the full soaring model.
- SOFT GROUND RESET (`gpu/bird3d.ts`): new `resetAltitude(y)` — lifts the bird back to altitude + restores trim airspeed/forward velocity. Called from the loop when clearance drops to `minClearance + GROUND_RESET` so a downhill run continues instead of dead-skimming the deck.
- FLIGHT TARGET (`gpu/target.ts` + `shaders/target.wgsl`, NEW): `Target` class — a world waypoint rendered as a camera-facing amber beam of light, drawn always-on-top (`depthCompare "always"`) so it stays visible behind ridges. `respawn()` places a fresh target 700–1000 m ahead within a ±0.35 rad cone; `checkReached()` is horizontal-distance only. HDR amber color (>1) so it blooms.
- WIRING (`bird-main.ts`): instantiate Target; `bird.stillAir = true`; per-frame reach-detect → score + respawn and ground-reset; draw the beam after the bird; TARGET HUD line (distance / steer ◄►▲ / reached). Scripted pitch-wobble DISABLED (`__autoWobble = false`) so manual control is clean from frame 1. Constants `REACH_RADIUS=55`, `GROUND_RESET=3`.
- Rendering kept entirely intact: terrain, wind motes, fluid, bloom, ground marker, compass, tuning panel all unchanged.

### Verify — typecheck (standalone)
```
cd /Users/god/projects/ai-jank/vector-system && ./node_modules/.bin/tsc --noEmit && echo OK
```

### Verify — run the sim, then open the bird page
```
cd /Users/god/projects/ai-jank/vector-system && ./node_modules/.bin/vite
```
Then browse to http://localhost:5173/index-bird.html

### Visual checklist (human eyeball — fly with the mouse)
- [ ] STILL AIR: no buffet/jitter; HUD reads `wind: 0.0, 0.0` and `DRIFT 0°`; vario moves only with your own dive/zoom, not random gusts.
- [ ] GLIDE FEEL: mouse-down = nose down, speed builds; mouse-up = zoom-climb then speed bleeds; centered cursor = gentle settling descent.
- [ ] TARGET BEAM: an amber column stands in the distance, pulsing, visible even when a ridge is in front of it.
- [ ] FLY TO IT: HUD `TARGET` distance counts down and `steer ◄/►/▲` points the way; within ~55 m `reached:` increments and a new beam appears ahead (you must turn for it).
- [ ] GROUND RESET: glide all the way to the deck — at min-clearance you're lifted back to altitude and the run continues (no stuck skimming).
- [ ] RENDERING INTACT: terrain ridgelines, drifting wind motes, bloom glow, plumb-line + ground diamond, compass — all still present.
- [ ] MOTES vs BIRD: motes drift (ambient air) but the bird's path is NOT pushed by them (expected — dead air on the bird).
- [ ] FPS ~60.

### Tuning knobs (the by-feel pass)
- Glide feel: `bird.tuning` live via the 'T' panel — `glideSpeed`, `sinkRate`, `divePower`, `dragK` (also `gravity`, `minSpeed`, `maxSpeed`).
- Target: `BEAM_HEIGHT` / `BEAM_HALF_WIDTH` / `BEAM_COLOR` / `SPAWN_MIN` / `SPAWN_MAX` / `SPAWN_SPREAD` in `gpu/target.ts`; `REACH_RADIUS` / `GROUND_RESET` in `bird-main.ts`.
- Undo the strip (wind back on): `bird.stillAir = false` in `bird-main.ts`.

### Machine verification done (2026-06-18)
- `tsc --noEmit` clean (exit 0).
- Live dev server serves new assets: `target.wgsl` 200, `target.ts` 200, `index-bird.html` 200 (no 404; module transforms).
- NOT verified by machine: live WebGPU rendering / flight feel (no browser harness this session) — human flight test required.

## v17 — taller mountains, up-and-over wind, omnipresent air, camera-vs-mountain fix
**Date:** 2026-06-17
**Commit:** (uncommitted — working tree on top of 99bfc9a)
**Session:** 2026-06-17 taller-terrain-wind-interplay

### What changed
- TALLER TERRAIN (`gpu/terrain.ts`, `shaders/terrain_ekg.wgsl`, `gpu/fluid-wind.ts` — 3 mirrors): `RELIEF 320→600`, `SHARP 1.6→1.8`. Dramatic peaks/valleys the bird threads between (was rolling dunes). All three fBm twins kept in lock-step.
- WIND UP-AND-OVER + AROUND (`gpu/wind.ts`): `liftGain 0.3→0.6`, `deflect 0.85→0.45`, `relax 2.5→0.8`. Wind rides up windward faces and arcs over crests AND routes around peaks (was a flat low slither that vanished behind ridges). Anti-geyser keystone: new `W_CLAMP=12` clamps the per-mote vertical flow `w` in `flowAt` so steep faces (gradient ∝ RELIEF) lift into a visible arc instead of erupting.
- OMNIPRESENT AIR (`gpu/wind.ts` + `shaders/wind.wgsl`): "it's air — always some wind, visible everywhere." `densityFloor 0.3→0.6` (calm air keeps ~60% of motes, never bare). Band concentrated (`clearance 40→30`, `vSpread 55→38`, `maxClear 260→200`) so motes aren't diluted across a tall column. Far-mote `count 2200→1600` for fps headroom. Shader brightness floor raised: `intensity = glow*(0.9+sp*1.2)` → `(1.6+sp*1.0)` so calm air READS instead of dimming below the bloom threshold.
- BIRD DRIFT SYNCED (`gpu/bird3d.ts`): `deflect 0.25→0.45` to match the wind motes (the code comment demanded they match; they didn't — bird now drifts with the air you see).
- CAMERA-VS-MOUNTAIN (`gpu/camera.ts` + `bird-main.ts`): chase eye now gets the terrain sampler and (1) PULLS IN along the boom until the view clears terrain, (2) FLOORS above terrain at its XZ, (3) post-lerp backstop. Fixes the fully-black screen when running into a mountain (the taller RELIEF embedded the eye in peaks).
- `nearCount` left at 800 (bird-main override, your QOL-pass value — untouched).

### Verify — typecheck (standalone)
```
cd /Users/god/projects/ai-jank/vector-system && ./node_modules/.bin/tsc --noEmit && echo OK
```

### Verify — run the sim, then open the bird page
```
cd /Users/god/projects/ai-jank/vector-system && ./node_modules/.bin/vite
```
Then browse to http://localhost:5173/index-bird.html

### Visual checklist (human eyeball — fly with the mouse)
- [ ] TERRAIN: peaks are tall and dramatic with deep valleys (not rolling hills); the bird flies among/between mountains, not just on top.
- [ ] AIR EVERYWHERE: wind is visible across the WHOLE frame at any altitude — never bare patches, even in calm air; dense legible cloud around the bird.
- [ ] UP-AND-OVER: motes ride up windward faces and arc over crests; near steep peaks the flow also bends around them. No vertical geyser/fountain eruptions.
- [ ] NOTHING CLIPS: wind motes and the bird stay above the terrain surface (never sink through it).
- [ ] RUN INTO A MOUNTAIN: dive the nose down (mouse to bottom) straight into a slope — the screen must NOT go fully black; you stay looking at the bird against the mountain.
- [ ] FPS: overlay holds ~60 throughout.

### Measured results (2026-06-17, headless WebGPU autopilot + forced-dive capture)
- fps: 60 held across autopilot flight AND a forced nose-down dive to the terrain (clearance floored at 6 m).
- Air visible everywhere at clearance 63-70 m (dense bright field over the ridges); taller relief confirmed in-frame.
- Forced dive to clearance 6 m (pinned to a steep slope) renders terrain+wind — NO black frame (camera-collision fix verified). Bird floored at minClearance (never passed into terrain).
- Typecheck clean. Zero pageerrors.
- Tuning knobs: wind `liftGain`/`deflect`/`relax`/`densityFloor`/`clearance`/`vSpread`/`maxClear` + `W_CLAMP` (vertical clamp) in the `Wind` ctor defaults; shader brightness `(1.6 + sp*1.0)` in `wind.wgsl` fs; camera `eyeMargin` (clearance the eye keeps from terrain) in `camera.ts`.

## QOL pass — mouse-leave autopilot, graduated terrain, halved near-sphere, bird wake
**Date:** 2026-06-17
**Commit:** (uncommitted — branch build/foundation)
**Session:** 2026-06-17 mouse-leave-autopilot QOL

### What changed
- MOUSE-LEAVE AUTOPILOT (`bird-main.ts`): cursor leaving the viewport (`document` mouseleave) sets `autopilot = true`; any canvas mousemove sets it back to `false` (manual). `P` still toggles either way.
- GRADUATED TERRAIN (`gpu/terrain.ts` + `bird-main.ts`): DENSE near band + far spread. `rowSpacing` (4.5 m) holds the FULL original density out to `nearDenseDepth` (500 m) so the foreground is unchanged; beyond it the step grows linearly (`farSpread` 220 m per +4.5 m). Row depths are precomputed by walking rowStart→maxDist; SAME horizon. Result: 4.5 m spacing at 0/250/500 m (identical to original), 14.7 m at 1000 m, 31.2 m at the horizon — **244 rows vs 456**, the cut entirely in the far field. Defaults (nearDenseDepth 0, farSpread ∞) reproduce uniform spacing exactly. One shared `rowDepthAt(r)` feeds both the line and fill-curtain loops so occlusion can't desync. (Supersedes a first geometric `rowSpacingGrowth` attempt that compounded from behind the camera and thinned the foreground — "too sparse up close".)
- DISTANCE FOG −25% (`bird-main.ts`): terrain fog `1/1100 → 0.75/1100` (= 1/1467) and wind-mote fog `1/1400 → 0.75/1400`, kept coupled — the sparser far rows + motes read deeper before dissolving into haze. Raise the `0.75` factors toward 1 for more fog.
- NEAR WIND SPHERE (`bird-main.ts`): `nearCount` 1600 → 800 → 400 (fewer particles).
- BIRD WAKE — VISUALS ONLY (`gpu/wind.ts` + `bird-main.ts`): inside the near sphere the bird now stirs the air it flies through — bow-wave push outward AHEAD + drag/swirl slipstream BEHIND, scaled by bird speed, falling to 0 at the ball edge. Added ONLY to the near-mote advection + their tails (and the speed-tint, so stirred air brightens). `windAt` / flight physics are UNTOUCHED (frozen). New gains `bowGain`/`wakeGain`/`swirlGain` (0 disables). `bird.vel` is now threaded into `wind.draw` to orient the stir.

### Verify — typecheck (standalone)
```
cd /Users/god/projects/ai-jank/vector-system && ./node_modules/.bin/tsc --noEmit && echo OK
```

### Verify — run the sim (standalone), then open the bird page
```
cd /Users/god/projects/ai-jank/vector-system && ./node_modules/.bin/vite
```
Then browse to http://localhost:5173/index-bird.html

### Visual checklist (human eyeball — fly with the mouse)
- [ ] TERRAIN: close-up ridgelines are dense and crisp; lines spread progressively farther apart toward the horizon; the far horizon is in the SAME place as before (not pulled in).
- [ ] NEAR SPHERE: the dense ball of little comets around the bird is about half as thick as before (still legible).
- [ ] BIRD WAKE: while gliding, motes part/push outward just AHEAD of the bird (bow wave) and drag + swirl in a slipstream just BEHIND it; stirred air reads brighter. Bank into a turn — the wake reorients to follow the new heading.
- [ ] WAKE IS LOCAL: only the near sphere is disturbed; the far streamlines and the bird's actual drift/crab (compass: cyan heading vs yellow ground-track gap) are unchanged (physics frozen).

### Verify — mouse-leave autopilot (human)
- [ ] Move the cursor OFF the browser window → the top overlay line flips to `AUTO: <mode>` and the bird flies itself (climbs/avoids ground hands-off).
- [ ] Move the cursor back over the canvas and wiggle → the overlay returns to `MANUAL (P=autopilot)` and you steer again.
- [ ] `P` still toggles autopilot manually regardless of the mouse.

### Measured results (2026-06-17, headless WebGPU boot-check)
- fps: 60 held with graduated terrain + halved sphere + bird-wake all active (no perf regression).
- mouse-leave → overlay `AUTO: ENERGY`; mouse-move → overlay `MANUAL`. Fluid window still stepping. Zero pageerrors / console errors. Typecheck clean.
- WAKE LEGIBILITY FIX (2026-06-17): first wake was invisible — squared falloff × an `along/R` directional weight peaked at ~15% of gain, so the stir was ~1-2 m/s against ~10 m/s ambient wind. Fixed: SATURATING ahead/behind weights (reach full strength within 35% of the ball radius) + LINEAR falloff, gains raised to `bowGain` 0.9 / `wakeGain` 0.75 / `swirlGain` 1.2. Peak stir now ~20 m/s swirl / ~15 m/s bow ≈ 2× ambient → the wake dominates the local flow. 60fps held.
- WAKE v2 — CURLING TAILS + PUSH/TRAIL REBALANCE (2026-06-17): user read v1 as "just spirals". Extracted `birdWakeAt()` (per-point wake velocity) and made the near tail RE-SAMPLE the disturbed flow per segment (4 segments now) so tails CURVE along the wake instead of straight streaks. Rebalanced so bow (push-aside ~15 m/s) + drag (trailing slipstream ~13 m/s) LEAD and swirl dropped 1.2 → 0.7 (~12 m/s) — reads as part+trail, not a pinwheel. `nearCount` 800 → 400. 60fps held (4/4 samples).
- NOTE: tuning knobs — wake character: `swirlGain` → 0 = pure push+trail (no rotation), raise for more tumble; all three (`bowGain`/`wakeGain`/`swirlGain`) → 0 disables the wake entirely; `nearSegStep`/`nearSegments` lengthen/smooth the curl. Terrain declutter — raise `nearDenseDepth` (denser/deeper crisp foreground) or lower `farSpread` (sparser far).

## Bird 3D v15 (world-pinned fluid window + terrain coupling)
**Date:** 2026-06-13
**Commit:** fda3f2f
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What changed (fluid subsystem only — `gpu/fluid-wind.ts`, `gpu/fluid.ts`, `shaders/fluid/{shift,force_field}.wgsl`)
- WORLD-PINNED MOVING WINDOW: the GpuFluid(256) grid origin is anchored to WORLD coords (was bird-local in v13). The window recenters in GRID-ALIGNED whole-cell steps when the bird nears the edge; on recenter the u/v field is GPU-shifted by the integer cell offset (overlap copied 1:1, leading edge clamp-extended) so the flow scrolls with the world — no seam/pop. Readback origin is captured per-slot and returned paired with the field so the world→grid mapping is self-consistent across the 2-3-frame-stale readback even when an intervening recenter moved the live origin.
- TERRAIN COUPLING: per-cell orographic force from an fBm twin of terrain.ts (down-gradient push → flow deflected around/over high terrain, channelled along valleys), clamped to `terrainMax` (flyable band, no spike). Recomputed only on recenter (fBm re-eval is edge-strip-only; force derivation is cheap neighbour diffs).
- Magnitude regulator decoupled: force = structure, scale (grid-vel→m/s) = band.

### Verify — typecheck (standalone)
```
cd /Users/god/projects/ai-jank/vector-system && ./node_modules/.bin/tsc --noEmit && echo OK
```

### Verify — run the sim (standalone)
```
cd /Users/god/projects/ai-jank/vector-system && ./node_modules/.bin/vite
```

### Verify — screenshot (standalone, dev server must be up)
```
cd /Users/god/projects/ai-jank/vector-system && node .ai/tmp/v15-shot.mjs
```

### Verify — full v15 probe: fps + anchor + no-seam + spatial coupling (standalone)
```
cd /Users/god/projects/ai-jank/vector-system && node .ai/tmp/probe-v15.mjs
```

### Verify — 30s bounded-field / blow-up check (standalone)
```
cd /Users/god/projects/ai-jank/vector-system && node .ai/tmp/probe-v15-blowup.mjs
```

### Measured results (2026-06-13)
- fps: 60 held across the full 10s probe (20/20 samples) and the 30s blow-up run, with terrain coupling + bloom both active.
- NO SEAM/POP on recenter: at the recenter crossing (recenterFrame 0→1), wind at a fixed world point went 7.0@79° → 7.3@78°, |Δ|=0.25 m/s (within normal per-100ms fluid evolution). Corroborated by 65 recenters over 30s flight, zero errors, bounded rawMean.
- ANCHORED (not bird-relative): during the 10s anchor loop the bird flew ~270 m (27 m/s, ground-track 21°) while `recenterFrame` stayed 0 — the window origin did NOT follow the bird. The fixed point's wind evolved smoothly (6.7→15.3→4.1 m/s, 57°→127°→84°) because the FLUID evolves, not because the window moved (in v13 the bird-local window slid every frame).
- SPATIAL terrain coupling: 5 fixed points across the window at one instant gave differing directions/magnitudes (2.8@169°, 2.1@96°, 8.8@63°, 7.5@74°, 7.3@70°) — the field is spatially structured, not flat.
- BOUNDED field: rawMean oscillates 15–31, scale tracks inversely 0.10–0.20 over 30s of continuous recentering — no monotonic climb, no divergence.

### Visual checklist (open http://localhost:5174/index-bird.html)
- [ ] 60fps in the overlay with bloom on.
- [ ] Fly straight across ~window-span (≈2600 m); the wind does NOT jump/snap when the window recenters (motes keep streaming continuously).
- [ ] Wind gusts sit over the SAME ridges as you fly past (anchored to the landscape), not random churn that moves with you.
- [ ] No-regression: bloom glow, EKG terrain-pour, crab/DRIFT (overlay DRIFT non-zero), mote fade-in/out, near-bird comet sphere, evolving fluid (overlay wind vector changes over time) all present; zero console errors.

### Rough / caveats
- No-pop sampled continuously across ONE recenter crossing (first, ~1-cell, |Δ|=0.25 m/s); corroborated by 65 recenters/30s zero-error, but it is one continuous crossing, not a full window-span sweep with frame-by-frame sampling.
- The 5-point spatial variation is consistent with terrain coupling but does not isolate it from the fluid's own swirl/drift; the per-cell fBm-gradient force is wired and active, but a point-sample doesn't prove terrain-correlation cleanly.

## Bird snap-accel fix + wind volume-fill (v14)
**Date:** 2026-06-13
**Commit:** N/A (uncommitted working tree)
**Session:** 17dc1ecc-a6c4-4456-bc85-980e0fb1a6b1

### Symptoms (user)
- "the bird has very fast accel / deaccel" — snappy/jerky motion, no inertia.
- "the wind has strange boundaries where particles cluster in planes" — confirmed by user as FLAT HORIZONTAL layers.

### Bird — velocity-vector inertia (`gpu/bird3d.ts`)
- ROOT CAUSE (adversarially confirmed): the bird had no velocity momentum. `integrate()` overwrote `this.vel = dir·speed + wind + updraft − sink` every frame (was bird3d.ts:279-281) — assigned, never integrated — so world velocity instantly tracked pitch/speed/sink → snap accel/deaccel.
- FIX: compose the flight-path velocity into a TARGET (`tvx/tvy/tvz`), then low-pass `this.vel` toward it with a dt-correct factor `1 − e^(−dt/VEL_TAU)`, `VEL_TAU = 0.25 s`. Buffet is added as an instantaneous overlay (`cvx/cvy/cvz`) for the position step + telemetry, deliberately NOT fed back through the filter (keeps the gust texture crisp).
- NOT TOUCHED (refutation caught the investigator over-reaching): cubic stall-sink (intentional "drops HARD" feel) and the pitch-ease (its claimed frame-rate bug was backwards — a raw lerp already preserves its time constant across refresh rates).
- KNOB: `VEL_TAU` at top of `integrate()`. Bigger = heavier/smoother; smaller = snappier.

### Wind — per-mote home height fills a volume (`gpu/wind.ts`)
- ROOT CAUSE: wedge motes were seeded at a single height (`terr + clearance`, 16 m) AND relaxed back to that same single height every frame, while vertical excitation (terrain-pour `w`) is ≈0 over flat ground — so the whole [minClear 5 .. maxClear 170] band collapsed onto one 16 m sheet. (Near-bird "comet" motes already seed in a volume — unaffected.) The original design comment (wind.ts ~386) deliberately pinned them low; user now wants volume.
- FIX: new per-mote `pHome` array + `vSpread` config (default 40 m). Each mote draws a HOME clearance uniformly from `[max(minClear, clearance−vSpread) .. min(maxClear, clearance+vSpread)]` (band clamped BEFORE sampling so motes never pile at a clamp → no new sheet), seeds at it, and relaxes toward its OWN home. Ridge-pour + anti-deplete behaviour preserved on top.
- KNOB: `vSpread` (wind cfg). 0 = old flat sheet; larger = taller volume. `clearance` is now the band CENTER.

### Climb/lift — "can't get higher" + "headwind makes me drop" (`gpu/bird3d.ts` + `gpu/wind.ts`)
- DIAGNOSIS (NOT a regression): the velocity-inertia change did not touch the lift/sink math — steady-state climb/sink is byte-identical. The symptoms were the pure-soaring model working as designed: a glider always sinks (~1.4 m/s at trim); lift comes only from windward ridges (`max(0, wind·uphill)`) or sparse thermals; oncoming/flat-ground wind gives no lift. (The new ~0.25 s velocity lag also mildly softens transient zoom-climbs.)
- DECISION (user): keep the soaring identity, make lift FINDABLE. No headwind-lift term added — oncoming wind over flat ground stays neutral by choice.
- TUNE: sinkRate 1.4→1.0 (bird3d.ts ~119); liftGain 2.5→3.5 (~122); updraft cap 5.5→8.0 at BOTH sites (`updraftAt` ~68 + `integrate` ~274); thermalAmp 4.0→5.0 (wind.ts DEFAULTS); thermal core exponent `pow(core, 2.2→1.8)` (wind.ts `thermalAt`) = broader, findable cores while ~half the world stays calm (the hunt is preserved).
- NEW BALANCE: trim sink ~1.0 m/s; a strong core nets ~+7 m/s climb (capped at 8); modest cores now net slightly positive instead of sinking.

### Verify — typecheck (standalone)
```
cd /Users/god/projects/ai-jank/vector-system && ./node_modules/.bin/tsc --noEmit && echo OK
```

### Verify — run the sim (standalone)
```
cd /Users/god/projects/ai-jank/vector-system && ./node_modules/.bin/vite
```

### Visual checklist (open http://localhost:5173/index-bird.html)
- [ ] BIRD: pitch/dive/recover ramps smoothly — no instant snap to a new speed; still feels connected to the stick (not mushy). Tune `VEL_TAU` if too floaty (lower) or still too sharp (raise).
- [ ] BIRD: stall still breaks (hold nose up → nose drops, vario goes strongly negative) — the inertia must not have erased the stall plummet.
- [ ] WIND: wedge motes now fill a VERTICAL VOLUME of air, not a single flat sheet at one height. Over flat ground you should see motes at a range of heights.
- [ ] WIND: no NEW flat sheet at the floor/ceiling (would mean clamp-piling — raise `minClear` headroom or lower `vSpread`).
- [ ] WIND: ridge-pour still reads — motes still stream up windward faces and spill over crests.
- [ ] LIFT: you can now CLIMB by finding lift — fly around and you should hit thermals (vario/overlay updraft goes positive) that let you gain altitude, not just sink everywhere.
- [ ] LIFT: soaring a windward ridge (wind blowing UP the slope) sustains/climbs you; expect stronger hold than before.
- [ ] LIFT no-launch: outside lift you still sink (~1 m/s) and strong cores don't rocket you (climb caps ~+7 m/s) — it must still feel like a glider, not a jetpack.
- [ ] LIFT (accepted limitation): flying INTO wind over flat ground still gives no lift (by design — that was the deliberate choice, not a bug).

## Bird 3D v14 (neon bloom post-process + re-tune)
**Date:** 2026-06-13
**Commit:** 8e49e36 (bloom wiring) + this doc commit
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What was done (`bird-main.ts` + `gpu/bloom.ts` + `shaders/bloom_*.wgsl`)
- BLOOM post-process for the neon glow. All scene passes (terrain / wind / bird / marker) now render into an HDR `rgba16float` 4× MSAA target, resolve into a single-sample HDR scene texture, then the bloom chain reads that and writes the final image to the swapchain. `rgba16float` is REQUIRED — `rgba8` bands the soft glow on this very dark scene.
- Chain: bright-pass THRESHOLD (soft-knee, luminance-weighted, keeps hue) → SEPARABLE 9-tap Gaussian blur (H then V, 2 iterations) at HALF-RES (downsample 2 = wide soft glow + perf) → COMPOSITE (scene·exposure + bloom·intensity, highlight-only rolloff tone-map at K=0.8 so blowout stays hue-colored, not white). All fullscreen-triangle passes.
- The existing render MODULES were NOT changed — `bird-main` constructs TerrainEKG / Bird3D / Wind / GroundMarker with `'rgba16float'` as their `colorFormat` (instead of the canvas preferred format), so they draw into the HDR target. Canvas context stays the preferred swapchain format; only the final composite writes to it.

### Re-tune (the advisor's blowout risk: bloom + additive neon double-counts bright pixels)
- TUNING (held — no down-tune needed): threshold 0.85, knee 0.5, intensity 0.9, exposure 1.0, downsample 2, blurPasses 2. Verified by reading the captured images against the pre-bloom baseline (`.ai/tmp/fade-final.png`). Per-element read (all PASS):
  - (a) terrain elevation color ramp — cool blue/purple valleys vs magenta peaks STILL DISTINGUISHABLE (colored magenta halos, NOT uniform white).
  - (b) 50%-opacity wind motes — glow as discrete bright specks, SEPARABLE (not smeared into a sheet).
  - (c) dense near-comet sphere — reads as individual comet STREAKS around the bird, NOT a white blob (confirmed on the tight crop `.ai/tmp/v14-bird-tight.png`).
  - (d) bird V — bright cyan-white with a glow halo, fully V-SHAPED (two wings + body notch).
  - (e) mote fade — soft into the dark haze at frame edges, no hard pop.
  - terrain pour + crab/drift (DRIFT +15°) + evolving fluid wind all still present.

### Measured
- 60 fps over 5 s (samples: 60×10, min/median/max = 60/60/60). Bloom is +4 full-screen passes (threshold + 2×(H+V) blur + composite) at half-res — holds the ~60fps budget clawed back in v13. Zero page errors.
- Screenshots: `.ai/tmp/v14-final.png` (full), `.ai/tmp/v14-final-crop.png` (center crop), `.ai/tmp/v14-bird-tight.png` (bird + comet sphere).

### Verify
```
cd /Users/god/projects/ai-jank/vector-system && node .ai/tmp/probe-v14.mjs
```
Open http://localhost:5174/index-bird.html (or 5173): the scene should GLOW like neon — magenta ridge halos, a luminous bird V, glowing wind motes — while every element STILL READS (no white smear). Knobs live in `bird-main.ts` (the `new Bloom(...)` constructor): threshold UP / intensity DOWN if anything blows out; `Bloom.setTuning({...})` re-tunes without a rebuild.

## Wind: bird drifts on visible flow (#3) + far/near tier crossfade (#4)
**Date:** 2026-06-13
**Commit:** N/A (no git repo present in working tree)
**Session:** (current)

### #3 — bird drifts through the terrain-shaped flow the motes ride (`wind.ts` + `bird3d.ts`)
- PROBLEM (user): the wind visual didn't match the wind applied to the bird. ROOT CAUSE: bird drifted on raw `windAt × windGain(1.6)`; motes advect on `flowAt` (terrain-deflected `windAt`). Different vector → mismatch.
- FIX: new module-level `flowHorizontal(wx, wz, gx, gz, deflect)` in wind.ts — the SINGLE into-slope deflection impl, PURE (caller passes the gradient, so no double `sampleHeight` in the hot mote loop). `Wind.flowAt` delegates to it (vertical `w` still computed from RAW pre-deflection wind → byte-equivalent for motes). `bird3d` imports it, adds `deflect` tuning (0.25 == Wind's), computes a central-diff gradient (reusing hX/hZ + 2 new −e samples) and applies the SAME deflection to its horizontal drift. `lastWind` (→ overlay + compass) now reports the deflected drift.
- CRITICAL GUARD: ridge LIFT keeps the RAW gained wind (`rwx/rwz`) — deflection sheds exactly the into-slope component that drives ridge lift, so feeding deflected wind there would kill windward soaring. `into = rwx*gx + rwz*gz` ≡ old behaviour (byte-identical). Exported `updraftAt` (autopilot's lift sense) still uses raw windAt.

### #4 — far/near tier crossfade, no pop (`wind.ts`)
- PROBLEM (user): substantial gap between far streamlines and the near ball; tiers pop in/out instead of fading.
- FIX: plumb `birdPos` into far `step()`. (a) inside `nearRadius*1.6` blend zone, raise the far density floor proximity-scaled (`max(vis, 0.85*proximity)`) so far stays dense into the ball — no calm-air thinning at the bird, no hard ring. (b) far→near handoff: `nearHandoff = smoothstep(0.35, 1, dist3D/nearRadius)` fades far motes out inside the ball so far+near don't stack over-bright; near tier owns the interior. (c) `fadeFarEdge` 50→120 softens the outer wedge perimeter.
- Result: far owns outside, crossfade across the shell, near owns inside — one tier per region.

### VERIFIED
- tsc clean; boots clean over many frames (birdPos plumbing + flowHorizontal export/import valid, no circular import); fps 60 (crossfade adds only ~1 sqrt + smoothsteps per far mote); manual stall still fires (nose breaks under nose-up input, vario ~−12). `.ai/tmp/{msaa-boot-check,stall-check,fps-check,shot}.mjs`.

### Review
```
cd /Users/god/projects/ai-jank/vector-system && ./node_modules/.bin/tsc --noEmit && echo OK
```
Open http://localhost:5173/index-bird.html: (#3) the compass magenta WIND arrow should bend near ridges and the bird's drift should follow the visible streamlines; (#4) fly toward/over a ridge — the streamlines should flow continuously from distance into the near ball with no pop or gap. Knobs: `bird.tuning.deflect` (match `Wind.deflect`), `Wind.fadeFarEdge`, the `blendR`/`nearHandoff` constants in wind.ts step().

## Far wind streamlines: corners → smooth curves
**Date:** 2026-06-13
**Commit:** N/A (no git repo present in working tree)
**Session:** (current)

### `wind.ts` (+ no shader change)
- PROBLEM (user): the distance/far wind lines have visible CORNERS; want smooth curves. ROOT CAUSE (workflow-mapped): far tail = only `segments=6` points, each rendered as an INDEPENDENT un-mitered billboard quad oriented by its own per-segment direction → the ribbon kinks at all 5 interior joints. The flow path is already curved; it was just drawn as too few straight pieces.
- FIX: `FAR_SUBDIV=3` — at render time, Catmull-Rom subdivide the coarse 6-point tail into a dense 19-point polyline (`sptX/sptY/sptZ`), then emit the ribbon over the dense points. Catmull-Rom is interpolating → passes exactly through every original (terrain-clamped) point; endpoints clamped (no out-of-range control reads). Adds ZERO `flowAt`/`sampleHeight` calls (the cited dominant CPU cost stays at the coarse count) — only interpolation + 3× vertex upload (far buffer 3.2MB→9.5MB, once-allocated).
- Near tier (short comets) left UNTOUCHED.
- VERIFIED: tsc clean; boots clean over many frames (no buffer overrun/RangeError); fps 60 (no regression). `.ai/tmp/shot.mjs`, `.ai/tmp/fps-check.mjs`.

### Verify
```
cd /Users/god/projects/ai-jank/vector-system && ./node_modules/.bin/tsc --noEmit && echo OK
```
Open http://localhost:5173/index-bird.html — the long distance streamlines should arc smoothly over the ridges with no angular kinks. Knob: `Wind.FAR_SUBDIV` in src/host/gpu/wind.ts (2 = lighter, 4 = smoother/more verts).

## Manual stall + momentum camera
**Date:** 2026-06-13
**Commit:** N/A (no git repo present in working tree)
**Session:** (current)

### Stall (`bird3d.ts` integrate)
- PROBLEM (user): flight felt "flat", bird couldn't stall. ROOT CAUSE: the `Math.max(minSpeed, …)` speed floor forbade airspeed decaying into a stall; the cubic sink that would drop the bird never engaged.
- FIX: floor lowered to absolute `STALL_FLOOR=7` (keeps cubic sink finite); `minSpeed` is now the stall THRESHOLD. Below it: nose BREAKS down (`breakPitch -0.35..-0.7`, overrides held stick), yaw mushy (×0.35), cubic sink capped at `SINK_CAP=28`. Speed rebuilds in the dive → recover → porpoise if you keep pulling. Autopilot energy guard (nose-down at glideSpeed−5=21, above stall 13) keeps the nanny out of it.
- VERIFIED headless: held nose-up → vario plunged to **−15.3 m/s** (vs ~−3 glide), pitch broke negative under nose-up input. `.ai/tmp/stall-check.mjs`.

### Camera (`bird-main.ts`)
- PROBLEM (user): camera "super shaky"; track momentum not heading. ROOT CAUSE: aimed along `bird.lastGroundTrack` which carries buffet/gust jitter (±2.6°) + stall thrashing.
- FIX: aim along a low-passed (`dt*2.5`, ~0.4s) horizontal velocity vector — momentum. Filters jitter, still swings for real turns + wind crab.
- NOTE: a terrain-dots experiment was tried and reverted (back to lines) per user.

### Verify
```
cd /Users/god/projects/ai-jank/vector-system && ./node_modules/.bin/tsc --noEmit && echo OK
```
```
cd /Users/god/projects/ai-jank/vector-system && (npm run dev >/tmp/v.log 2>&1 &) ; sleep 4 ; /opt/homebrew/bin/node .ai/tmp/stall-check.mjs
```
Open http://localhost:5173/index-bird.html (manual), drag mouse to top to hold nose up → watch it stall and drop; camera should glide, not shake.

## Terrain traces smoother — MSAA + 2× sample density
**Date:** 2026-06-12
**Commit:** N/A (no git repo present in working tree)
**Session:** (current)

### What was done (`bird-main.ts` + 4 GPU drawables; no shader edits)
- PROBLEM (user): "can we make the terrain lines splines, not points? the jaggies are throwing me." Two causes: (1) thin neon lines/ribbons staircase because the render target had NO MSAA (sampleCount 1); (2) near rows show only a central slice of the wide `halfWidth=2400` span, so `cols=768` left visible chord faceting up close.
- FIX:
  - 4× MSAA. New `SAMPLES=4` in `bird-main.ts` (single source of truth). Added a multisample color target (`msaaTex`, swapchain format) + made `depthTex` sampleCount 4; both recreated on resize. Every pass renders into `msaaTex`; the FINAL pass (`marker.draw`) carries a `resolveTarget` = swapchain view, resolving once into the canvas (a WebGPU canvas can't be multisampled directly).
  - Threaded `sampleCount` into every pipeline so counts match the target: `TerrainEKG` (via `TerrainParams.sampleCount`, both fill+line pipelines), `Bird3D`, `Wind`, `GroundMarker` (trailing ctor arg, default 1). `marker.draw` gained a trailing optional `resolveTarget`.
  - Sample density: terrain `cols` 768 → 1536 (shrinks near-field chord faceting).
- UNCHANGED: heightfield function (still terraced/cliffy by design — MSAA/density do NOT soften the geology), flight physics, the EKG aesthetic, all shaders.
- NOTE: chose "more samples + MSAA" over a Catmull-Rom spline (user declined spline) and over softening the cliffs (user likes the aesthetic, esp. distant ridge highlighting).

### Verify
```
cd /Users/god/projects/ai-jank/vector-system && ./node_modules/.bin/tsc --noEmit && echo TYPECHECK_OK
```
```
cd /Users/god/projects/ai-jank/vector-system && (npm run dev >/tmp/vite-bird.log 2>&1 &) ; sleep 4 ; /opt/homebrew/bin/node .ai/tmp/msaa-boot-check.mjs
```
Then open http://localhost:5173/index-bird.html and eyeball: terrain traces read as smooth curves (no staircase on near rows), distant-ridge highlight intact, no black screen. If near-field faceting still bugs you, the levers are `cols` (bird-main.ts:86) and `SAMPLES` (bird-main.ts).

## Wind motes fade in/out on recycle (no pop)
**Date:** 2026-06-12
**Commit:** N/A (no git repo present in working tree)
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What was done (`wind.ts` only; `wind.wgsl` unchanged — `vis` already plumbed)
- PROBLEM (user): "the wind pops in / out, can we have a fade?" Persistent motes RECYCLE — far tier reseeds at the camera-relative span boundaries; near tier (a ~65 m ball of ~1600 little comets around `bird.pos`) reseeds when a comet drifts outside the ball. On recycle the position SNAPS to a new seed → the mote POPS in at the new spot and POPS out when it leaves.
- FIX (CPU-side fade envelope folded into the existing per-vertex `vis`, so the shader needs no change):
  - Per-mote AGE (seconds since last seed). Far tier: `age` array; near tier: `nearAge` array. Advanced by `dt` each frame in the rebuild/update loop, reset to 0 in `seedMote` / `seedNearMote`.
  - `fadeIn = smoothstep(0, fadeInTime, age)` — a freshly (re)seeded mote ramps `vis` 0→1 over the first ~0.55 s of life (`fadeInTime`).
  - `fadeOut` ramps 1→0 as the mote nears its recycle condition:
    - FAR tier: distance to the NEAREST span boundary (front / back / either side); `smoothstep(0, fadeFarEdge, edgeDist)` → 0 in the last `fadeFarEdge` (50 m) before any span exit.
    - NEAR tier: proximity to the ball EDGE; `distFrac = dist(bird)/nearRadius`; `1 - smoothstep(fadeNearEdge, 1, distFrac)` → 1 inside, 0 at the radius (`fadeNearEdge` 0.78 → fade starts at 78 % of R).
  - FAR tier: `vis *= fadeIn * fadeOut` — multiplied INTO the existing density cull (density-fade preserved, not replaced).
  - NEAR tier: `nearVis = fadeIn * fadeOut` REPLACES the old hard `vis = 1` — the sphere now shows a soft brightness gradient (dim toward the edge / when young) instead of hard uniform dots.
  - AGE is advanced BEFORE the cull-skip so a culled mote that later survives the density cull still has a real age — it won't pop when it re-enters.
- Tuning knobs added to `DotParams` / constructor: `fadeInTime` (0.55 s), `fadeFarEdge` (50 m), `fadeNearEdge` (0.78). Fade zones are wide enough that bird/camera relative motion can't skip a mote past them in one frame.
- UNCHANGED: `windAt`/`thermalAt`/flight physics (frozen), the two-tier look, additive neon, terrain occlusion, the v11 distance spread. NO shader edit. NO other files touched.

### Pre-conditions
```
cd /Users/god/projects/ai-jank/vector-system
```

### Verify: typecheck clean
```
npx tsc --noEmit
```
Expected: exit 0, no output.

### Verify: boot + capture (Playwright, Metal WebGPU)
```
npm run dev
```
```
node .ai/tmp/myshot-bird3d.mjs .ai/tmp/fade-final.png .ai/tmp/fade-final-1.png .ai/tmp/fade-final-crop.png
```
Expected: JSON with `"errors": []` and `"booted": true`. Saves a full frame, a second frame, and a centered bird crop.

### Verify: the fade reads (watch in MOTION — a still cannot prove pop-vs-fade)
Open http://localhost:5174/index-bird.html (vite falls back to 5173). Fly with the cursor near screen-center, then confirm:
- NEAR SPHERE has SOFT edges: little comets dim toward the ball boundary and brighten toward the center — NO hard uniform-brightness disc with a crisp rim.
- Watch a comet reach the sphere edge: it FADES OUT before it recycles, and a reseeded comet FADES IN from dark — no hard POP at either end.
- FAR LINES: motes near the wedge front/back/side boundaries are dim and ease out; reseeded far motes ease in. No mid-air pop-in.
- Prior wins intact: two-tier look (long distant arcs + dense near sphere), the small gliding-V bird, EKG ridges that OCCLUDE the motes, the good flight.
- HUD bottom line reads `fps: ~60`; no `[WebGPU lost]` / `pageerror` / `console.error`.

### Watch for
- Fade is a brightness envelope only — it does NOT move/recycle motes any differently; it just eases `vis` at the ends of a mote's life.
- `fadeNearEdge` too low → the whole sphere dims (fade starts too far in); too high (→1) → the edge pops again. `fadeFarEdge` too small → far motes still pop at the span boundary.
- `fadeInTime` ~0.4–0.7 s; longer makes reseeds linger dark, shorter reintroduces a softer pop.
- Static-frame evidence is the brightness GRADIENT on the near sphere (visible in the crop); true pop-vs-fade only reads in motion.

## Bird 3D v11 (two-tier wind: distant lines + dense near comet sphere)
**Date:** 2026-06-12
**Commit:** f37af9d
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What was done (`wind.ts` + `wind.wgsl` + 1 line in `bird-main.ts`)
- TWO-TIER WIND for legibility at the bird (user: "i see wind all around, but not next to the bird"):
  - FAR tier = the existing persistent population of LONG curved streamline lines (2200 motes, multi-segment tails integrated backward along the terrain-shaped flow) — distant wind reads as long flowing arcs over the ridges.
  - NEAR tier = a DENSE WIND SPHERE of ~1600 LITTLE short-tailed comets seeded uniformly in a ~65 m ball centered on the BIRD; advected by the SAME terrain-aware `flowAt`; comets that leave the ball recycle back inside so the sphere follows the bird. `vis=1` for every near comet (NO density cull) → the local air is unmistakably legible right where you fly.
  - Both tiers share one pipeline/shader/vertex-format, drawn in a single combined vertex buffer.
- `bird-main.ts`: the single `wind.draw(...)` call passes `bird.pos` as the near-sphere center (the only change in that file).
- DISTANCE SPREAD (final tune, f37af9d): with the sphere owning the bird vicinity, dropped the far tier's `nearBias` 2.6→1.3 so the long lines spread into the distance (was ~61% near / 16% far; now ~37/34/29% near/mid/far) — the long curved lines now populate the distance where v11 wants them.
- PHYSICS low-density, VISUAL high-density: the near sphere renders as a THICK cloud of many little comets (user: "air calculations for fluid are low density, render at a higher density").
- `windAt`/`thermalAt`/flight physics UNTOUCHED (frozen). Terrain respected: comets stay above ground, hug + pour over the ridges, depth-tested (ridges occlude).

### Verify: typecheck + boot capture
```
node node_modules/typescript/bin/tsc --noEmit
```
```
node .ai/tmp/myshot-v11b.mjs
```
Expected: connects on 5174 (or 5173), `fps: 60`, `errors: none`, `DRIFT` non-zero (~+26° — wind visibly crabs the bird). Saves `.ai/tmp/v11b-final-0.png`, `.ai/tmp/v11b-final-1.png`, `.ai/tmp/v11b-final-crop.png`.

### Verify: WATCH IT (eyes-on)
Open http://localhost:5174/index-bird.html (autopilot off — touch nothing or fly manually):
- [ ] DISTANT wind reads as long curved flowing LINES arcing over the far ridges.
- [ ] A DENSE legible sphere of little cyan comets surrounds the bird — the air next to it is unmistakable (the v11 fix).
- [ ] The near sphere follows the bird as it flies; comets recycle, the cloud stays thick.
- [ ] Prior wins intact: small white V bird dwarfed by big magenta EKG ridges; EKG terrain + elevation color; crab camera; good flight; 60fps; no errors.

### Watch for
- Near-sphere cost: ~1600 comets × 3 tail segments held 60fps headless; 2200 read borderline 56–59. The near tail REUSES the head flow vector (no per-segment `flowAt`/`sampleHeight`) — if you raise `nearCount`, that shortcut is what keeps it cheap; restoring per-segment flow there is the first fps cliff.
- The two-tier distinction is subtle in a downscaled full frame — read tail-length-at-range from a tight DISTANCE crop (e.g. `.ai/tmp/v11b-dist-crop.png`, upper band away from the bird), not the bird-centered crop.

## Bird3D v12 — fog 2×, terraced-cliff terrain, straight-line wind eval
**Date:** 2026-06-12
**Commit:** working tree
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What was done
- FOG EXPANDED 2× (`bird-main.ts`): terrain fogDensity 1/550→1/1100, wind fog 1/700→1/1400. Horizon moved with it: maxDist 950→1900, rows 256→512, cols 512→768, halfWidth 1500→2400 (the 76° dive frustum is ~2300 m half-wide at the new horizon — 1500 would have shown naked row ends).
- TERRAIN GEOLOGY (`terrain.ts` + `terrain_ekg.wgsl`, IDENTICAL twins): BASE_FREQ 1/350→1/700 (features 2× wider — a valley is now ~50 wingspans, the bird reads as a speck in real terrain), OCTAVES 3→4, RELIEF 220→320, then SHARP=1.6 pow (deep valleys, crisp crests) + TERRACED CLIFF BANDS (TERRACES=5, RISER_POW=4, CLIFF_MIX=0.65 — flat shelves with steep risers ≈ every 64 m of height).
- STRAIGHT-LINE EVAL POLICY (`autopilot.ts`): `new AutoPilot(terrain, "straight")` — locked heading at trim glide, NO lift-seeking/orbiting; only AVOID (near ground, re-locks to escape heading) and a stall nose-down deviate. Probes skipped in straight mode. (User's P-key toggles manual ↔ autopilot in bird-main.)

### Measured (60 s straight line, `node .ai/tmp/auto-eval.mjs` → EVAL_PASS)
- clearance min 40 / avg 111 / max 199 m — AVOID fired 3×, recovered, never near the floor.
- vario -3.6…+9.2 m/s, 50% of samples positive — lift bands cross the straight track often.
- |drift| avg 27°, max 69° — the wind is STRONGLY felt on a constant heading (eval headline).

### Verify
```
node node_modules/typescript/bin/tsc --noEmit
node .ai/tmp/auto-eval.mjs
```
Then eyes-on `.ai/tmp/eval-straight.png` / live page: wide valleys dwarfing the bird, terraced shelf-and-riser ridge profiles (not noise fuzz), ridges readable ~2× deeper before the haze.

### Watch for
- TS/WGSL heightfield twins MUST stay identical — any future terrain change edits BOTH or the bird flies a different mountain than you see.
- |drift| avg 27° may be more wind than a player wants; windGain/windDrift sliders are the tamers.
- 512×768 rows ≈ 786k line verts — fps held 60 headless; watch it on battery.

## Bird3D v11 — AUTOPILOT: autonomous soaring, manual controls off
**Date:** 2026-06-12
**Commit:** working tree
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What was done
- NEW `src/host/autopilot.ts`: hands-off soaring controller emitting the SAME BirdInput the mouse used (physics untouched, controls swap at the wire). Mode priority: AVOID (clearance <45 m or predicted <35 m at a 2.5 s velocity look-ahead → climb hard if airspeed allows + turn toward the lower shoulder) > SOAR (net lift here → orbit gently, ride it) > ENERGY (slow → nose down, airspeed first) > CLIMB/DESCEND (90–260 m band) > CRUISE (steer toward the best of 8 lift probes at 140 m). Commands eased; `[auto]` telemetry to console every ~2 s.
- `src/host/gpu/bird3d.ts`: exported `updraftAt(x,z,t,terrain,T)` — the EXACT ridge+thermal+cap math integrate applies — so the autopilot senses the same air the bird rides.
- `src/host/bird-main.ts`: `AUTOPILOT=true` — mouse steering + wobble disabled; overlay shows `AUTO: <mode>`; `window.__autoMode` exposed. Set AUTOPILOT=false to restore manual flight (everything still wired).

### Verify: typecheck + 90 s autonomous soak (the gate)
```
node node_modules/typescript/bin/tsc --noEmit
node .ai/tmp/auto-soak.mjs
```
Expected: `SOAK_PASS` — min clearance >10 m (measured 39 m), lastAvg >40 m (measured 138 m), zero page errors; relayed `[auto]` lines show SOAR/CRUISE/ENERGY/AVOID transitions and recoveries. Saves `.ai/tmp/auto-soak.png`.

### Verify: WATCH IT (eyes-on)
Open http://localhost:5173/index-bird.html and touch nothing:
- [ ] Overlay shows `AUTO: <mode>`; bird banks, hunts, and circles in lift (vario positive in SOAR).
- [ ] It sinks in still air, dives to rebuild speed (ENERGY), and climbs away from ridges (AVOID).
- [ ] Minutes pass; it never grinds along the terrain floor.

### Watch for
- `bestProbe` saturates at the 5.5 updraft cap almost everywhere — probe ranking barely discriminates; if heading choice looks aimless, rank probes on UNCAPPED ridge+thermal in `updraftAt` (return both) before tuning anything else.
- Flapping comes next: a thrust impulse the autopilot (then the player) can spend when lift fails — slot it where ENERGY mode dives today.

## Bird3D v10 — holdable pitch attitude (mouse-y = nose angle) + dive payout
**Date:** 2026-06-12
**Commit:** working tree
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What was done
- PITCH IS NOW ATTITUDE, NOT RATE (`src/host/gpu/bird3d.ts` BirdInput.pitchTarget + `bird-main.ts` mapping): cursor height = nose angle (±0.6 rad at full deflection), nose eases toward it at 3.5/s and STAYS — park the cursor, hold the attitude. Centered cursor = gentle glide trim (-0.03 rad), which REPLACES the old hands-off auto-trim (deleted — it dragged the nose down whenever the mouse rested in the deadzone: the "can't hold a pitch" complaint). Yaw unchanged (rate-based).
- DIVE PAYOUT: divePower default 0.9→1.1 (dives build speed visibly faster). sinkRate left at the 1.4 already set.
- Scripted wobble now sweeps pitchTarget ±0.65 (same visual proof, new contract).

### Verify: typecheck + attitude HOLDS headless
```
node node_modules/typescript/bin/tsc --noEmit
node .ai/tmp/hold-probe.mjs
```
Expected: no tsc output; `pitch after 1.5s: ~12° — after 3.5s: ~12°` then `HOLDS`.

### Verify: FLY IT
- [ ] Park cursor above center: nose sets to a steady climb angle and STAYS — no slump-back.
- [ ] Center the cursor: gentle settling glide (~-1.4 m/s vario), hands-off stable.
- [ ] Cursor low: committed dive, airspeed builds noticeably; pull to center-high: zoom-climb.
- [ ] The dive→zoom→hold-attitude loop feels like flying, not negotiating.

### Watch for
- Attitude ease rate is the `3.5` in bird3d.ts integrate(); raise for snappier nose, lower for heavier bird.
- PITCH_RANGE 0.6 / GLIDE_TRIM -0.03 live at the top of bird-main.ts.

## Bird 3D v10 (near-dense wind + terrain-hugging pour)
**Date:** 2026-06-12
**Commit:** b1df7d3 (v10 wind code); this docs-only entry committed on top
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What was done (wind.ts / wind.wgsl ONLY)
- **Density: DENSE near, SPARSE far.** Seed distance is biased to the near field — `ahead = base + (far−base)·rand^nearBias` with `nearBias` 2.6 (base floored at ~8% of `spanAhead` so the cloud peaks AT the bird ~120 m ahead, not under the camera). Inverts v9's uniform-world seed where perspective made the FAR field read densest. Far motes also thin via the speed/density cull + distance fog.
- **HUG.** Nominal `clearance` 55→16 m, `minClear` 14→5 m — motes ride just over the surface and follow the contour up each ridge instead of floating in a flat sheet.
- **POUR.** The pour reads stronger NOT by raising the vertical gain — `liftGain` was LOWERED 3.2→2.4 (rides `w = liftGain·(wind·uphill-grad)`; v9's higher value pinned every mote at the ceiling). It reads because: lower HUG clearance puts the arcs against the surface; `deflect` dropped 0.9→0.25 so most of the into-slope wind is KEPT, driving motes UP and OVER the crest rather than routing flat along the contour; and `maxClear` 100→170 m gives climbing motes headroom to stream up the windward face and SPILL over. Climbing motes are folded into `speedFrac` so they read BRIGHT + long-tailed while pouring; the backward-integrated curved tail then arcs down the lee = the visible spill.
- FROZEN `windAt`/`thermalAt` untouched (bird imports them). Bird physics/camera/terrain untouched.

### Verify: SHOW gate (the deliverable)
```
node .ai/tmp/myshot-v10.mjs
```
Expected: connects on 5174 (or 5173), zero page errors, `fps: 60`. Captures a 6-frame burst `.ai/tmp/v10-burst-*.png` (+ per-frame crops). The best soaring pair (autopilot in SOAR, dense near-band on a windward face) was then HAND-PICKED and `cp`'d to `.ai/tmp/v10-final-0.png`, `.ai/tmp/v10-final-1.png`, and `.ai/tmp/v10-final-crop.png` — not auto-selected by the script.

### Verify: READ IT (eyes-on)
Open http://localhost:5174/index-bird.html (or `.ai/tmp/v10-final-crop.png`):
- [ ] Wind is a THICK cloud of streaks right around the bird, thinning into the distance (near-dense / far-sparse).
- [ ] Motes HUG the terrain and POUR up the windward faces — bright near-vertical streaks climbing the slopes, the dense band spilling over a crest (best seen in a SOAR frame, vario positive).
- [ ] Curved comet tails (not straight); additive neon cyan→white; ridges occlude motes (depth test).
- [ ] Prior wins intact: small bird vs big EKG ridges, elevation color (cool low → warm/magenta high), terrain streams, glide + buffet.

### Watch for
- The capture runs against the working tree, which currently has an UNCOMMITTED out-of-scope AUTOPILOT effort (`autopilot.ts` + `bird-main.ts`/`bird3d.ts` mods, see v11/holdable-pitch entries above). So the framing in the captures is autopilot-driven, NOT committed mouse-steer. The WIND rendering is identical either way (driven by wind.ts/wind.wgsl + camera).
- The pour reads strongest when the autopilot parks the dense near-band ON a windward face (SOAR mode). A frame over a valley can look flat — that's framing, not the wind code.

## Bird 3D v9 (wind↔terrain + curved tails + felt buffeting)
**Date:** 2026-06-12
**Commit:** 65fafdd (v9 code: wind a5eb19d/65fafdd, buffet 0d58e74); this docs-only entry committed on top
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What was done
- **Wind ↔ TERRAIN.** `flowAt(x,z,t)` wraps the FROZEN `windAt` (untouched — bird import intact): (a) VERTICAL pour `w = liftGain·(horizontalWind · uphill-gradient)` from finite-diff of the Wind class's `sampleHeight` → motes rise over windward slopes, sink in lees; (b) into-slope HORIZONTAL deflection so flow bends around peaks/over crests. Each mote persists a 3D height; motes visibly hug and climb the ridgelines.
- **Curved longer tails.** Each comet tail is a multi-segment polyline integrated along the terrain-shaped `flowAt` over ~10 steps (`tailStep` 0.5) → long curved comets arcing over ridges, not straight lines.
- **Felt buffeting (`bird3d.ts`).** A turbulence term ROCKS render-bank ±~7°, BOBS `vel[1]` ±~1.5 m/s (vario oscillates), pulses a lateral gust SHOVE, and holds a steady CRAB lean into the cross-wind. Controllable. Exposes `window.__birdBank`.
- **v9 tune (65fafdd):** vertical lift gain 9→3.2, tail `step` 0.1→0.5 × 10 seg, `relax` 0.1, `maxClear` 100, `densityFloor` 0.3. The pour survived this cut — verified still visible (see screenshots).
- **Note (scope):** a pre-existing working-tree change to `src/host/bird-main.ts` (terrain EKG density: `rows` 128→256, `rowSpacing` 18→4.5 = 4× line density) is present but REMAINS UNCOMMITTED — this gate was bounded to docs only, staging the source was blocked by the scope guard, and it was NOT reverted (destructive, needs the user's decision). It is NOT a v9 wind/buffet change; it sharpens the ridge profiles and reads clean (no tangle). The captures were taken against the working tree, so they match what `npm run dev` shows today. Flagged for the user to keep or drop.

### Pre-conditions
```
cd /Users/god/projects/ai-jank/vector-system
```

### Verify: typecheck clean
```
npx tsc --noEmit
```
Expected: exit 0, no output. (Verified 2026-06-12.)

### Verify: live in browser
```
npm run dev
```
Open http://localhost:5174/index-bird.html (vite falls back to 5173 if 5174 is taken). Mouse = steer; the glider sinks by default and you hunt lift.

### Verify: the look reads (fly a moment, cursor near screen-center)
- Wind motes POUR up and over / around the ridgelines — curved streaks arc over the crests (terrain interaction), not straight horizontal lines.
- Comet tails are LONG and CURVED, following the terrain-shaped flow.
- The bird ROCKS/BOBS/CRABS with no input: flying straight, bank oscillates ±~7° and vario oscillates (buffet active); a steady crab lean into the cross-wind shows DRIFT (~+26°) on the compass (heading cyan vs ground-track vs wind magenta).
- Prior wins intact: SMALL gliding-V bird dwarfed by LARGE rolling EKG ridges; ridges OCCLUDE the motes; elevation color (teal/blue valleys → magenta/white peaks); good flight (~60fps, default sink ~−2.7 m/s so you hunt lift).
- 60fps; no `[WebGPU lost]` / `pageerror` / `console.error`.

### Captured proof (this gate)
- Driver: `.ai/tmp/myshot-bird3d-v9.mjs` (Playwright + Metal WebGPU flags, port 5174 then 5173; waits `window.__birdBooted`; pair ~0.8s apart, mouse held screen-center = no input).
- Pair: `.ai/tmp/v9-final-0.png` / `.ai/tmp/v9-final-1.png`. Between frames the terrain streams, motes advect, and bank/vario/drift move — world is alive. Zero page errors.

### Watch for
- The pour and curved tails read best in MOTION; a still shows the curve + density but not the live climb.
- `liftGain` (now 3.2) drives the vertical pour AND the bird's ridge-lift — lowering it further would flatten the pour toward invisible.

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

## Bird 3D v13 (real GPU fluid wired as wind source)
**Date:** 2026-06-13
**Commit:** 8488539 (v13 code: `bird-main.ts` + `gpu/fluid-wind.ts` + `gpu/wind.ts`); this entry = doc-only follow-up commit
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What this entry covers
NO-REGRESSION GATE on the already-committed v13 wiring (the GPU Stam fluid replaced the analytic curl-noise as `windAt`'s base horizontal source; prevailing drift kept; `flowAt`/`thermalAt` unchanged). Verified by MEASUREMENT — no code changed for this gate (a temporary `__fluidOff` A/B probe guard was added to `bird-main.ts` and fully reverted; `git status` confirms `bird-main.ts` is clean). FPS/visuals were measured against the CLEAN v13 tree (out-of-scope uncommitted edits to `bird3d.ts`/`terrain.ts`/`terrain_ekg.wgsl` — an unrelated terrain line→dots experiment — were stashed for the fps/screenshot reads, then restored).

### Measured results
- (a) FPS (clean v13): ~45 (raw rAF over 3s), overlay EMA ~49-52. Below 60. The fluid step did NOT regress fps: an in-context A/B (`__fluidOff` true vs false, same page so any headless effect + tree state cancel) shows the fluid's MARGINAL cost ≈ 0 fps (fluid-off ~43.6, fluid-on ~46 — within noise; the off side also swaps `windAt` to the analytic branch which runs `potential()` 4× in the hot mote loop, so the A/B slightly under-counts the step, but net is still negligible). Headed run matched headless (~45.5 raw / 51 EMA) → NOT a headless-throttle artifact. Stashing the out-of-scope edits left fps unchanged (~45.4 raw) → they are not the cost either. CONCLUSION: the ~45fps ceiling is PRE-EXISTING in the committed v13 baseline (likely the motes' 10-segment `sampleHeight` tail loop that `wind.ts` flags as the dominant CPU cost, plus the committed terrain render) — the bottleneck was NOT isolated and is OUT of v13 scope. Did NOT tune fluid grid/iters (256/10) — the A/B proves it is not the cost; tuning would degrade the wind field for nothing.
- (b) NO-REGRESSION (all PASS, read from `.ai/tmp/v13-final.png` + `.ai/tmp/v13-final-crop.png`): terrain pour (motes drape/curve over ridges), crab/drift (overlay DRIFT +10° to +18°, NON-ZERO), mote fade (soft edges, no pop), near-bird comet sphere legible, long distant lines over far ridges, small bird vs big ridges, elevation color (teal→magenta).
- (c) EVOLUTION PROOF (field is the FLUID, not quasi-static curl-noise): `window.__windAt(x,z)` sampled at fixed `t=0` (any change is purely the readback replacing the field).
  - Fixed world point (x0,z0)=(109,316), 6× over ~3s: `[7.47,7.83] [7.6,7.2] [7.79,6.43] [7.98,5.6] [7.92,5.05] [7.24,5.18]`.
  - Discriminator at LIVE bird pos (bird always grid-center → cell fixed → PURE temporal evolution, no window-translation confound): `[11.45,3.66] [11.07,5.3] [9.69,7.11] [8.08,8.01] [6.88,8.23] [5.94,8.16]`.
  - Both sets swing clearly frame-to-frame → real evolving wind, and the non-null readback also confirms the fluid path is actually wired.

### Verify
```
cd /Users/god/projects/ai-jank/vector-system && ./node_modules/.bin/tsc --noEmit && echo TYPECHECK_OK
```
```
cd /Users/god/projects/ai-jank/vector-system && node .ai/tmp/v13-gate.mjs
```
Open http://localhost:5174/index-bird.html (falls back to 5173). Watch the wind motes drift and re-read the overlay over several seconds: DRIFT stays non-zero, the local air swirl visibly CHANGES (gusts), terrain pour + near sphere intact.

### Watch for
- Observed |wind| runs ~15-17 m/s (overlay), slightly above the stated 10-15 band, but DRIFT is only +10-18° — nowhere near the +61° blow-around regression. Feel is intact; do NOT crank fluid force.
- `targetBand=3.0` in `fluid-wind.ts` contradicts its own comment ("mean ~8.5") and `wind.ts` `FLUID_MAX` comment ("mean ~10/max ~16"); the regulated SCALE plus `FLUID_MAX=10` clamp produce the observed ~15-17 peak — comments are stale, the measured number is authoritative.
- A/B probe harnesses: `.ai/tmp/v13-gate.mjs` (full gate), `.ai/tmp/v13-fps-probe.mjs` / `v13-fps-headed.mjs` (raw rAF), `.ai/tmp/v13-ab.mjs` (fluid marginal cost). All Playwright + Metal WebGPU flags, port 5174 then 5173, wait `window.__birdBooted`.
- The ~45fps ceiling is real (pre-existing in committed v13) and worth a separate perf pass; the prime suspect is the motes' per-mote 10-segment `sampleHeight` tail loop (`wind.ts` flags it as the dominant CPU cost), but it was NOT isolated and is OUT of v13 scope.

## Bird 3D wind perf: far-mote tail lightened, fps restored
**Date:** 2026-06-13
**Commit:** eda55a8 (`gpu/wind.ts` only — far-mote curved-tail build cost cut)
**Session:** 4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5

### What was done (`wind.ts` only)
- PROBLEM: v13 wired the live GPU fluid as `windAt`'s base. The fluid STEP is cheap; the cost is the FAR-mote CURVED TAIL — each far mote built a ~10-segment polyline integrated along the flow, calling `sampleHeight`+`flowAt` PER SEGMENT. The livelier fluid let more far motes survive the speed cull, so this loop dominated the CPU frame (fps 60→~51, the v13-gate entry above flagged this exact loop as the prime suspect).
- FIX (far tier only; near comet sphere was already cheap and is unchanged):
  - Far tail segments 10→6, `segStep` 0.5→0.8 — SAME ~5s / 35-50m flow span and visible curve, 40% fewer `flowAt`+clamp evals and 40% fewer verts.
  - First tail step REUSES the head flow `[wx,wz,w]` already computed for advection (free; the head is where `flowAt` was just sampled).
  - Thereafter re-evaluate `flowAt` only every 2nd segment and HOLD between — halves the remaining gradient cost; over a ~6-8m sub-step the flow barely changes so the arc still reads curved (flow is still re-sampled along the path, NOT reused for the whole tail like the near comet).
  - Per-POINT terrain clamp KEPT (the 35-50m far tail crosses ridges and must not sink in).
- UNCHANGED: the wind FIELD / forcing, `windAt`/`flowAt`/`thermalAt` semantics, near comet sphere, fade envelope, two-tier look, terrain occlusion. NO shader edit. NO other file touched.

### Measured results
- FPS before: ~51 (overlay EMA; the v13 regression baseline). FPS after: **60 flat** — 10 overlay samples over ~5s all read 60 (min/mean/max 60/60/60), well above the 58 gate.
- NO look regression (read from `.ai/tmp/perf-final.png` + `.ai/tmp/perf-final-crop.png`): curved distant lines over far ridges PRESENT; dense near-bird comet sphere legible; terrain pour (motes drape/curve over ridges) intact; fade soft (no pop); crab/drift NON-ZERO (overlay DRIFT +15°); evolving fluid present (`__windAt(1000,0)` swings `[9.7,12.7]`→`[2.8,6.9]` over ~5s at fixed t).
- Zero page/console errors.

### Verify
```
cd /Users/god/projects/ai-jank/vector-system && ./node_modules/.bin/tsc --noEmit && echo TYPECHECK_OK
```
```
cd /Users/god/projects/ai-jank/vector-system && node .ai/tmp/perf-final.mjs
```
Open http://localhost:5174/index-bird.html (falls back to 5173). Overlay `fps:` reads ~60; watch far motes still draw long curved arcs over distant ridges, the near sphere stays dense, motes fade (no pop), DRIFT stays non-zero.

### Watch for
- `perf-final.mjs` reads the overlay EMA `fps:` field (smoothed); it pinned 60 here. If you want raw rAF too, the v13 raw probes are at `.ai/tmp/v13-fps-probe.mjs`.
- Levers if it regresses again (all `wind.ts`, far tier): `segments` (6), `segStep` (0.8), the every-2nd-segment subsample cadence, or capping how many far motes build full tails.

---

## bird respects the dirt — f32/f64 terrain-hash divergence (CRASH in clear sky)
**Date:** 2026-06-19
**Commit:** (uncommitted — worktree `.claude/worktrees/mountaintop-forests`, branch `worktree-mountaintop-forests`)
**Session:** mountaintop-forests (bird "says crash when nowhere near the ground")

### Root cause
Terrain height uses `fract(sin(dot(p,(127.1,311.7)))*43758.5453)`. The GPU computes it in f32, the
CPU mirror (`terrain.ts sampleHeight`, what the bird COLLIDES against) in f64. The `*43758` amplifies
the tiny f32-vs-f64 `sin` difference into a totally different random field. Measured divergence:
mean ~104 m, max ~461 m on a 600 m relief — e.g. at (-1500,300) the GPU draws ground at 42 m while
the CPU collision field thinks there's a 364 m peak. Bird at 200 m over visually-empty sky hits an
invisible wall. (Same divergence the trees module already documented and worked around on the GPU.)

### Fix
Replaced the `sin`-hash with an integer lattice hash (`ihash`, pure uint32 ops) in all four coupled
files so CPU and GPU agree to <1 mm:
- `src/host/gpu/terrain.ts` (collision field)
- `src/host/shaders/terrain_ekg.wgsl` (neon terrain)
- `src/host/shaders/terrain_grid.wgsl` (grid/topo mode)
- `src/host/shaders/trees_ground.wgsl` (tree ground)
`terrain3d.wgsl` left untouched (separate scene, not in the bird pipeline).
NOTE: terrain SHAPE changes (it's a new random field) — peaks/valleys move. That's unavoidable to
make CPU match GPU; the bird now respects what you see.

### Verify — divergence collapses (offline, no browser)
```
/opt/homebrew/bin/node .ai/tmp/terrain-fix-probe.mjs
```
Expect: `mean ... diff (m): 0.0000`, `max ... diff (m): 0.0006`.

### Verify — typecheck
```
./node_modules/.bin/tsc --noEmit && echo OK
```

### Verify — in the app (the real test)
```
npm run dev
```
Open `http://localhost:5173/index-bird.html`. Fly far from spawn (the bug worsened with distance).
- Bird's altimeter plumb-line should touch the VISIBLE ground, not stop in mid-air.
- "CRASH" (HUD `lastCrashing` / steering goes mushy) only fires when the bird visibly skims terrain.
- Trees sit ON the neon ridges (already true; confirms tree-ground field still matches).
- Toggle grid/topo terrain mode — same ridgelines, bird still collides correctly.

---

## terrain self-check — diagnose persistent invisible-terrain crash
**Date:** 2026-06-21
**Commit:** (uncommitted — worktree `.claude/worktrees/mountaintop-forests`)
**Session:** mountaintop-forests (bird STILL crashes in clear sky after the ihash fix; "invisible terrain impacting camera/ground")

### Why
The ihash fix is present in the worktree but the symptom persists. Three live hypotheses, different fixes:
H1 stale build / wrong dir (dev server running main repo, not this worktree) → old sin-hash still active.
H2 my WGSL ihash != JS ihash on the real GPU → CPU/GPU fields still disagree.
H3 unrelated bug (camera clamp), not the hash at all.
"Can't tell if terrain changed" + "random invisible walls mid-air" can't disambiguate by eye → added a
runtime GPU-vs-CPU height probe (`gpu/terrain-selfcheck.ts`, wired in `bird-main.ts`).

### Run the probe (decisive)
```
cd /Users/god/projects/ai-jank/vector-system/.claude/worktrees/mountaintop-forests
npm run dev
```
Open the bird page, hard-reload (Cmd+Shift+R), read the browser console for `[terrain-selfcheck]`:
- `PASS  maxDiff<1m`  → GPU==CPU. Fix is correct. Crash-in-sky is then NOT the hash (chase H3 camera clamp).
- `FAIL  maxDiff=NNNm` → GPU!=CPU on this hardware (H2). The `worst @ (x,z)` is the invisible wall; rewrite hash.
- no line at all → wrong directory / stale build (H1). Start dev from the worktree above.
Re-run live anytime in the console: `__terrainCheck()`.

---

## sticky autopilot — P survives the mouse leaving the window
**Date:** 2026-06-21
**Commit:** (uncommitted — worktree `.claude/worktrees/mountaintop-forests`)
**Session:** mountaintop-forests (bird "loses" when pressing P then leaving the window)

### Root cause
`mousemove` set `autopilot = false` on EVERY cursor move. Pressing P then moving the mouse toward the
window edge to leave instantly cancelled autopilot, and the frozen edge cursor position (≈ full
deflection) was held as manual steering → the bird banked hard and spiralled off.

### Fix (bird-main.ts) — "Sticky P"
- `mousemove` no longer cancels autopilot; it only updates the steering origin.
- Manual takeback is now an EXPLICIT gesture: click the canvas (`mousedown`) or press P again.
- `mouseleave` / window `blur` recenter steering (mouseX=mouseY=0) so MANUAL flight can't freeze on a
  hard edge-deflection. Autopilot ignores mouseX/mouseY, so it is unaffected.
- HUD shows `AUTO: <mode> (click/P=manual)` so the takeback is discoverable.

### Verify (dev server already running on :5180 from the worktree)
```
open http://localhost:5180/index-bird.html
```
1. Press `P` → HUD reads `AUTO: ... (click/P=manual)`; bird flies itself.
2. Move the mouse OUT of the window → bird KEEPS flying straight on autopilot (no veer/spiral).
3. Move mouse back over canvas → still autopilot (sticky); HUD still AUTO.
4. Click the canvas (or press P) → HUD flips to `MANUAL (P=autopilot)`; mouse steers again.
5. In MANUAL, fling the cursor off the edge → bird centers (glides straight), does not spiral.

---

## climb nose-up + painful stall (bird feel)
**Date:** 2026-06-21
**Commit:** (uncommitted — worktree `.claude/worktrees/mountaintop-forests`)
**Session:** mountaintop-forests

### Changes (gpu/bird3d.ts)
- CLIMB TILT: model noses UP proportional to real climb rate (vario>0), visual-only, additive to the
  control pitch — mirrors the existing dive nose-down. So ridge-lift/zoom/flap climbs show attitude, not
  just stick-held climbs (which stall). New `renderPitch` drives the uniform (was raw `this.pitch`).
  Tunable: `climbTilt = min(0.30, max(0,vel.y) * 0.045)` (rad/(m/s), cap ~17°).
- SOFT STALL (landing-flare feel): on stall the wing MUSHES, it does not depart. The nose sags only
  gently (hold it high and settle), the airframe sinks softly, and a faint lean hints the mush — NO
  uncommanded yaw, NO wing-drop. Coordinated and controllable; ease off or dive to fly out. (Retuned from
  an earlier harsher "bite" per feedback: a plane settling onto a runway, not a stall-spin.)
  Tunables: nose-sag `breakPitch -0.05 - 0.18*stallDepth`, settle sink `sinkRate*stallDepth*0.6`,
  mush lean `stallYaw*0.15` (heading departure removed).

### Verify (dev server on :5180)
```
open http://localhost:5180/index-bird.html
```
1. Dive (cursor above bird) → nose tilts DOWN (unchanged).
2. Climb — pull up, OR ride ridge lift / thermal / flap (SPACE) to gain altitude → nose tilts UP a little.
   Watch the vario (overlay) positive ⇒ visible nose-up even with the cursor near center.
3. STALL it: hold the nose up in still air until airspeed decays below minSpeed (HUD shows the break).
   Expect a LANDING-FLARE mush: the nose sags only gently, the bird settles/sinks softly with a faint lean,
   wings stay level (no departure). Fully controllable — ease off or dive to fly out.

---

## solid triangular body + curved wings (bird model)
**Date:** 2026-06-21
**Commit:** (uncommitted — worktree `.claude/worktrees/mountaintop-forests`)
**Session:** mountaintop-forests ("bird looks like it's made out of paper")

### Changes (gpu/bird3d.ts buildVMesh)
- BODY: was a single flat ribbon (paper). Now a SOLID faceted triangular spindle — triangular cross-section
  (apex up) tapering to a nose (+Z) and tail (−Z) point, 6 facets / closed volume. Faked top-light via the
  existing edgeFrac→brightness path (no shader change): upper-right facet brightest, upper-left mid, belly
  dim → the facets read as 3D volume. Body stays teal (spanFrac 0), wings still ramp to magenta tips.
- WINGS: were straight linear ribbons. Now gently CURVED — sweep `-SWEEP*f^1.5`, gull dihedral
  `DIHEDRAL*f^1.35 + tip up-curl`, chord tapers to the tip `RIBBON*(1-0.55f)`, SEGS 4→6 for a smooth curve.
- Vertex count 90 (body 18 + wings 72); draw uses dynamic `vertexCount`, nothing hardcoded.

### Verify (dev server on :5180)
```
open http://localhost:5180/index-bird.html
```
- The body reads as a SOLID 3D wedge (visible top/side/belly facet tones), not a flat sheet.
- Wings curve (swept + gull dihedral + slight tip up-curl) and taper toward the tips, not a straight V.
- Flap (SPACE) and idle flex still bend the wings; bank/pitch/stall lean still apply to the whole model.
- Tips still glow magenta; body teal. No z-fighting / no missing faces as it banks (cullMode none).

---

## glide feel pass + uncomfortable crash tumble (windless flight)
**Date:** 2026-06-21
**Commit:** (uncommitted — worktree `.claude/worktrees/mountaintop-forests`)
**Session:** mountaintop-forests ("good glides" + "rotate uncomfortably when I hit terrain")
**Context:** bird.stillAir = true (windless) — tuning PURE energy-glider flight; ridge lift/thermal are
zeroed in still air and come alive only when wind is enabled later.

### Changes (gpu/bird3d.ts)
- GLIDE (floatier + more dynamic): sinkRate 1.0→0.8 (~32:1 glide, more hang time), divePower 2.0→2.4
  (bigger dive↔zoom energy swings), dragK 0.2→0.16 (dive speed holds through the swoop → bigger zooms).
  All three are still live sliders in the T panel for fine-tuning.
- CRASH TUMBLE: a hard terrain hit (impactRate > crashSpeed) now throws the bird into a disorienting
  roll + nose-down lurch, scaled by impact severity, direction following the current lean. Winds up over
  ~0.25 s then settles back to level over ~0.5 s. Visual only (no physics kick); recoverable.
  Tunables: roll kick `7 + 7*hit` rad/s, pitch kick `-(4 + 4*hit)`, decay 0.25 s (vel) / 0.5 s (angle).

### Verify (dev server on :5180)
```
open http://localhost:5180/index-bird.html
```
1. GLIDE: level off (cursor near center) → long floaty descent. Dive (cursor up), build speed, then pull
   up (cursor down) → a big zoom-climb that trades the speed back for altitude. Energy management reads.
2. CRASH: dive hard into a ridge → the bird snaps into an uncomfortable tumble (roll + nose-down), then
   rights itself. Harder hits tumble more violently.

---

## swoosh fix — asymmetric dive↔zoom energy (climb felt heavy) + dive ceiling
**Date:** 2026-06-21
**Commit:** (uncommitted — worktree `.claude/worktrees/mountaintop-forests`)
**Session:** mountaintop-forests ("not getting a good swoosh, too HEAVY on the climb after a dive")

### Root cause
divePower scaled gravity-along-flight-path SYMMETRICALLY — so cranking it for punchy dives (2.4) also
bled climb speed at 2.4× the natural rate, draining the dive's energy instead of carrying it up. Heavy zoom.

### Changes (gpu/bird3d.ts + bird-main.ts)
- Split into ASYMMETRIC energy: DIVE (nose down) uses divePower 2.4 (accelerates hard); CLIMB (nose up)
  uses new climbPower 1.0 (≈ energy-conserving) so the dive's speed CARRIES UP into a swoosh. New
  `climbPower` tuning param + live slider (0.3–2.5). Lower climbPower = floatier/longer zoom.
- maxSpeed 70→120 (slider 30→160): committed dives keep accelerating instead of pinning at the cap.

### Verify (dev server on :5180)
```
open http://localhost:5180/index-bird.html
```
- Dive to build speed, then pull up → a long light ZOOM-climb (swoosh), not a heavy stall-out. Speed
  bleeds gracefully as you trade it for altitude.
- T panel: drop `climbPower` toward ~0.7 for an even floatier zoom; raise toward divePower for heavier.
