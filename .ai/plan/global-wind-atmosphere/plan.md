# Plan — Global wind altitude atmosphere

Accepted design: [.ai/explore/2026-06-22-global-wind-altitude-atmosphere-design.md](../../explore/2026-06-22-global-wind-altitude-atmosphere-design.md)

## Approach
One `windProfile(absoluteY)` magnitude curve (calm low → strong high), multiplied into wind everywhere it's
consumed — uniform, no special rules. Soaring survives because ridges are high → windy. Far-mote visual
reverts to terrain-hugging with a tail to altitude. Thermals untouched. Then re-verify flight.

## Steps
1. `windProfile` (module-level, tunable params + setter) in wind.ts; `smoothstep`-based curve.
2. Bird drift: `bird3d.integrate` scales its windAt by `windProfile(birdY)`.
3. Ridge lift: `updraftAt` gains a `y` param, scales its wind by `windProfile(y)`; integrate + autopilot pass the bird's y.
4. Motes: far + near advection scale horizontal flow by `windProfile(moteY)`.
5. Visual revert: clearance/vSpread back toward terrain-hug + `homeBias` power curve (cluster low, tail up).
6. Verify: tsc → live gate (gradient real + soaring preserved + 60fps) → adversarial workflow → screenshot.

## Grade: A (absolute-altitude removes the soaring-break risk; verifiable).
