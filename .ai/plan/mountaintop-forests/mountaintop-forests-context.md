# Context — key files & decisions

## Worktree
- `.claude/worktrees/mountaintop-forests` on branch `worktree-mountaintop-forests`, based on `99bfc9a`
  + the in-flight (uncommitted) scene files copied in. Source checkout left untouched (other agent on core).
- `git config worktree.baseRef head` set so the native worktree tool branched from HEAD (no remote exists).

## New files
- `src/host/gpu/trees.ts` — Trees module: windowed placement, L-system geometry, batched buffer, draw pass.
- `src/host/shaders/trees.wgsl` — transform + distance fog + neon-green glow, additive, line-list.

## Integration (src/host/bird-main.ts)
- import `Trees`; construct after `target` with `(x,z)=>terrain.sampleHeight(x,z)`, HDR_FORMAT, SAMPLES.
- `trees.draw(enc, colorView, depthView, viewProj, camGround, eye, bird.simTime)` between bird and target.
- `window.__trees` exposed for perf A/B (`.enabled`, `.treeCount`).

## Tuning constants (trees.ts)
- CELL 50m, RADIUS 700m, HEIGHT_FRAC 0.55 × PEAK_RELIEF 600, ANCIENT_FRAC 0.03, MAX_TREES 600.
- TRUNK_LEN 16, DECAY 0.72, SPLIT 0.55rad, BRANCHES 2, DEPTH_NORMAL 4, DEPTH_ANCIENT 6, ANCIENT_SCALE 2.5.

## Hard-won bug
- WGSL forbids a uniform var sharing its struct's name. Original `var<uniform> U: U` silently invalidated
  the pipeline (compile errors don't throw — they surface only via `device.popErrorScope`). An invalid
  pipeline in the shared frame encoder drops the WHOLE submit. Fixed to `var<uniform> u: U` (target.wgsl
  convention). Caught only by an offscreen GPU-readback test with an error scope — NOT by FPS/pageerror.

## Verification harness
- `tests/gpu/trees-perf.spec.ts` — true-average FPS A/B + tree count + rebuild ms.
- `tests/gpu/trees-offscreen.spec.ts` — renders Trees to an offscreen rgba8unorm target, GPU-readback
  proof of green pixels + error scope. (Live WebGPU canvas can't be screenshotted in headless.)
