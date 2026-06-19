# Mountaintop neon forests — accepted plan

Add algorithmic trees to the bird3d landscape: glowing green recursive (L-system) branch trees that
cluster on mountaintops, with a few large "ancient" giants interspersed.

## Decisions
- **Form:** recursive branch lines (L-system), baked CPU-side to a line-list. Matches the neon wireframe aesthetic.
- **Color:** neon green HDR `[0.3, 1.4, 0.6]`, additive blend, feeds the existing bloom.
- **Placement:** deterministic grid window around the camera; keep a tree only where terrain height
  clears a mountaintop threshold (0.55 × RELIEF). Rebuild only on camera cell-crossing.
- **Batching:** one vertex buffer for all trees (mirrors the wind-mote batching; no instancing).
- **Render order:** after `bird.draw`, before `target.draw`; depth-tested (`less`, write off) so ridges occlude.

## Constraints
- Single-author hobby render project; must hold 60fps; isolated in a worktree (another agent on core).
- No new rendering machinery (no instancing) — `[BORING AND CORRECT]`.
- One earned complexity: windowed rebuild, solving the infinite camera-relative terrain.

## Grade: B+
