# Compressed Context
date: 2026-06-22
session: f65433b5-2d4a-4594-85b6-036787d8f3af

## Original Task
Iterative redesign of the bird-sim wind viz. Final framing (user): THREE layers —
1) **global wind** (gameplay; the field the bird flies), 2) **local sphere** (the bird's sensory
surface, reflects global wind, rendered DISTINCTLY), 3) **wake** (start as 7 long lines off the wings:
tip/mid/inner ×2 + tail, must TOUCH the wings, influenced by the local sphere). User decision:
**"solve for global wind first"** — it has game impact; the other two are visual dials.
Last action: turned local sphere + wake OFF, added T-panel sliders/toggles to dial global wind in isolation.

## Files Changed
- src/host/gpu/wind.ts — `windProfile(absoluteY)` altitude atmosphere (calm low→strong high) + `setWindProfile`(guarded) + `windAloftScale()` + exported `windProfileParams`; far+near mote advection scale horizontal by `windProfile(moteY)`; `showNear`/`showWake` fields + `setShowNear/setShowWake` + draw gated (`draw(showNear?vertexCount:farVertexCount)`, stepNear skipped when off); `_wakeOn=moving&&showWake` gates wake/wing-emission; twin wingtip vortices in `birdWakeAt`; touched-air `nearHeat` (FPV **10→11**, heat=loc6); `foreStretch` bubble + `bubbleFrac`; `homeBias` seed; clearance 30/vSpread 70 (reverted loft).
- src/host/gpu/bird3d.ts — `updraftAt(x,z,t,terrain,T)` (no y): ridge lift uses `windAloftScale()` (strong aloft, altitude-independent → soaring preserved); L+B lookahead/broaden (`ridgeLookahead`/`ridgeEps`); `integrate` drift `rwx/rwz *= windProfile(this.pos[1])`.
- src/host/autopilot.ts — `updraftAt` calls (5-arg, no y).
- src/host/shaders/wind.wgsl — `heat` vtx attr (loc6) + warm tint `mix(YELLOW,RED,heat)` via `smoothstep(0.3,0.6,heat)` deadzone; brightness 0.5→0.65.
- src/host/bird-main.ts — Wind params `{nearCount:200,numMotes:700,dotPx:3.6}`; T-panel WIND section (`panelSep`/`toggleBtn` helpers + activity/render sliders + local-sphere/wake toggle buttons); hooks `__wind __nearWake __nearFrame __updraftAt __windProfile __windProfileAt`.
- tests/gpu/{wind-atmosphere,slipstream,touched-air,updraft-buffer}.spec.ts — NEW live gates (:5273).
- HUMAN_REVIEW.md — entries: updraft-buffer, slipstream, touched-air, global-wind atmosphere.
- .ai/explore/2026-06-22-global-wind-altitude-atmosphere-design.md (design); .ai/plan/global-wind-atmosphere/{plan,context,tasks}.md.

## Current State
Global-wind atmosphere DONE + verified (gradient 0.40→1.40 monotonic, ridge lift 8.0 strong everywhere,
soaring preserved, 60fps). Local sphere + wake (the near tier = old slipstream/touched-air bubble) OFF by
default → global wind alone. T-panel (press `T`) dials it. All 3 gates green; tsc clean. **Nothing committed.**
A 3-lens adversarial workflow caught + I fixed a HIGH (ridge lift was wrongly scaled down low → now aloft).

## Open Threads
- User is iterating on global wind FEEL via the T-panel sliders (loScale/hiScale/altLo/altHi + render). Awaiting their flight verdict.
- **Local sphere layer** — NOT built as "distinct render"; currently it's the old near-tier bubble behind `setShowNear`. Future: a distinct sensory-surface render that reflects global wind.
- **Wake layer** — NOT built as "7 lines off the wings"; currently wake = `birdWakeAt` twin-vortices baked in the near tier behind `setShowWake`. Future: 7 long lines anchored to wing geometry, influenced by local sphere, dial feel then break up.
- Commit when user is happy with the set.

## Key Decisions
- Global wind profile keys on **ABSOLUTE altitude** (not height-above-surface) → no special-case rule; valleys calm because low.
- **Ridge lift uses `windAloftScale()` (constant strong aloft), NOT the bird's altitude** — soaring works at ANY ridge height, no valley-stranding. DRIFT (and motes) use `windProfile(y)`. This is the user's "strong aloft wind, no special rules" instruction; the absolute-altitude-on-lift version was a bug (adversarial-caught).
- `windAt`/`thermalAt` internals stay FROZEN; profile applied AT consumers. Thermals untouched.
- Near tier OFF by default; gates re-enable via `__wind.setShowNear(true)/setShowWake(true)`.
- SERVER GOTCHA: fresh `./node_modules/.bin/vite --port 5273 --strictPort` from THIS worktree; test `--config .ai/tmp/wind-verify.config.ts`; curl-verify served before trusting a result.
- nvm wrapper breaks node/npm → call `./node_modules/.bin/*` directly. vsay voices: bm_lewis grave, bm_fable playful.
- Per CLAUDE.md: design docs in .ai/ (not docs/superpowers/); commit only when asked; render tables as tab-aligned.
