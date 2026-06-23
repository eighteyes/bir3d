# Tasks — Global wind altitude atmosphere

- [x] `windProfile(y)` + `setWindProfile` + module params in wind.ts (smoothstep curve)
- [x] Profile tunables exposed (defaults: loScale 0.4, hiScale 1.4, altLo 100, altHi 500)
- [x] Bird drift scaled by `windProfile(birdY)` (bird3d.integrate)
- [x] `updraftAt` gains `y` param, scales ridge lift by `windProfile(y)`
- [x] integrate + autopilot pass the bird's `y` to `updraftAt`
- [x] Far + near mote horizontal advection scaled by `windProfile(moteY)`
- [x] Visual revert: clearance 60→30, vSpread→70 + `homeBias` 2.5 (cluster low, tail aloft)
- [x] `__windProfile` / `__windProfileAt` live-tune hooks in bird-main
- [x] tsc clean
- [x] Live gate: gradient real (0.40→1.40 monotonic, ridge lift Δ=5.57 unsaturated) + soaring preserved (rides 8 m/s, no crash) + 60fps
- [~] Adversarial verification workflow (running in background)
- [x] Screenshot (test-results/wind-atmosphere.png) + HUMAN_REVIEW entry
