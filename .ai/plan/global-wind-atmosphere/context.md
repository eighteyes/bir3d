# Context — key files & decisions

## Files
- `src/host/gpu/wind.ts` — `windProfile` + setter (module-level, like `setFluidField`); far/near mote advection scales horizontal flow by `windProfile(moteY)`; `seedMote` gets `homeBias`; revert clearance/vSpread loft.
- `src/host/gpu/bird3d.ts` — `integrate` scales drift by `windProfile(birdY)`; `updraftAt(x,z,t,terrain,T,y)` scales ridge lift by `windProfile(y)`.
- `src/host/autopilot.ts` — passes `bird.pos[1]` to `updraftAt` (9 probes + here).
- `src/host/bird-main.ts` — optional `__windProfile` live-tune hook.

## Decisions
- ABSOLUTE altitude (not height-above-surface) → no special rule needed; ridges high = windy = soaring intact.
- Profile is a multiplier APPLIED AT CONSUMERS, not baked into `windAt` (which has no y, called 10k×/frame).
- Module-level mutable profile params + setter so the bird (module funcs) and motes (instance) share ONE profile, live-tunable.
- Thermals (`thermalAt`) = separate vertical system, untouched.
- Terrain altitude 0–600m (RELIEF=600) → defaults: loScale 0.4, hiScale 1.4, altLo 100, altHi 500.
- Visual: motes hug terrain (user: "looks cool") but `homeBias` leaves a tail aloft (user: "odd to see NO wind in altitude"). Wind-at-bird-altitude is the LOCAL SPHERE's job (separate, later).

## Verify gotchas
- Fresh vite on :5273 from THIS worktree; curl-verify served before trusting a test.
- Re-verify SOARING (autopilot rides lift, no sink-crash) — this unfreezes flight physics.
