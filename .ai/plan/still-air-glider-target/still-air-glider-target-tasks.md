# Tasks — still-air glider basis + fly-to-target

- [x] Add `stillAir` flag + `resetAltitude` to bird3d.ts (zero wind/lift/thermal/buffet/rock).
- [x] Build Target module (target.ts + target.wgsl): beam render, respawn, reach detection.
- [x] Wire into bird-main.ts: instantiate, stillAir=true, reach/respawn + ground reset, draw
      beam, HUD line, scripted wobble off.
- [x] Typecheck (tsc --noEmit clean) + server smoke-test (new assets serve 200) + plan/
      HUMAN_REVIEW docs.
- [ ] (LATER) Fly-test + tune glide feel and target placement.
- [ ] (LATER) Flapping-wing animation.
- [ ] (LATER) Canyon terrain.
