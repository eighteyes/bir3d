# Context — still-air glider basis + fly-to-target

## Files
- `src/host/gpu/bird3d.ts` — CPU glider physics. Added `stillAir` (zeros wind/lift/thermal/
  buffet/rock in `integrate()`) + `resetAltitude(y)`.
- `src/host/gpu/target.ts` (NEW) — `Target` class: waypoint state, billboarded beam render
  (always-on-top), `respawn` / `distanceTo` / `checkReached`.
- `src/host/shaders/target.wgsl` (NEW) — camera-facing vertical beam; gaussian horizontal
  falloff, base-bright, time pulse; additive (blooms).
- `src/host/bird-main.ts` — instantiate Target, set `bird.stillAir = true`, reach/respawn +
  ground reset in loop, draw beam, TARGET HUD line, `__autoWobble = false`. Constants
  `REACH_RADIUS = 55`, `GROUND_RESET = 3`.

## Key technical notes
- Wind is single-source-of-truth (`wind.ts windAt`): bird + motes read it. `stillAir` zeros it
  ONLY for the bird (motes untouched) — that is how "no wind" + "keep all rendering" coexist.
- Target uniform layout (112 B): mat4(64) + basePos+height(16) + rightAxis+halfWidth(16) +
  color+time(16). Float indices 0..27. Matches std140 vec3+f32 packing.
- Always-on-top beam: `depthStencil { depthWriteEnabled:false, depthCompare:"always" }` in a
  pass that shares the depth attachment.
- All scene pipelines must use `multisample.count = SAMPLES (4)`; Target ctor takes sampleCount.
- Project strictness: `noUncheckedIndexedAccess` — index params must be `Vec3` tuples, not
  `ArrayLike<number>` (the first build error; fixed).

## Environment gotcha
- The nvm shim is broken in the non-interactive shell (FUNCNEST recursion corrupts PATH; npm
  fails with `_load_nvm` errors). Use `/opt/homebrew/bin/node` (v25) and `./node_modules/.bin/
  tsc` directly. Avoid `export PATH=...` in a command — it truncates PATH and tools vanish; call
  binaries by absolute path (e.g. `/usr/bin/curl`) instead. The interactive shell is fine.
