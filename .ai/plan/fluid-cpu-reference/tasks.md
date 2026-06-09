# Plan 2 — Tasks

Sequential (each task compiles on the prior). TDD: invariant tests first, then implement, `cargo test` green, commit.

- [x] **Task 1 — grid.rs** Grid2D + bilinear sample + clamping. Tests: exact-at-center, midpoint-average, out-of-range clamp, idx round-trip.
- [x] **Task 2 — boundary.rs** set_bnd (scalar Neumann, velX/velY wall reflect, corners). Tests: Neumann copy, velocity negation at wall, corner mean.
- [x] **Task 3 — advect.rs** semi-Lagrangian. Tests: zero-vel identity, uniform-shift analytic, symmetry, bounded extrema.
- [x] **Task 4 — project.rs** divergence + Jacobi projection. Tests: max|div|≈0 after project (THE test), divergence reduced ≫, idempotence on divergence-free, symmetry.
- [x] **Task 5 — solver.rs** Fluid2D::step + Fluid25D vertical coupling. Tests: post-step divergence-free, 200-step stability (no NaN/blowup), symmetry, kappa=0 independence, kappa>0 vertical-sum conservation.
- [x] **Final** full `cargo test` green ×2 (determinism), `cargo build` clean, scope held (Jacobi/no-multigrid/no-diffusion), HUMAN_REVIEW.md entry, this checklist updated.
