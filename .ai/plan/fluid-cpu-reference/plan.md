# Plan 2 — Fluid math in Rust (CPU reference)

> Roadmap entry (from `vector-engine-blueprint/plan.md`): "Fluid math in Rust (CPU reference): grid, semi-Lagrangian advection, divergence, Jacobi pressure projection, 2.5D layer coupling. Pure `cargo test`, no GPU."
>
> REQUIRED SUB-SKILL for executors: superpowers:subagent-driven-development / test-driven-development. Steps use `- [ ]` tracking in `tasks.md`.

**Goal:** A pure-CPU, deterministic, `cargo test`-green fluid solver in `vs-core` that is the **correctness oracle** for Plan 3's GPU port (Plan 3 validates GPU≈CPU via readback). No GPU, no WASM, no wasm32 target — builds on the installed `aarch64-apple-darwin` target.

**This solver's entire job is to be correct enough to trust as the reference.** Therefore tests encode **physical invariants**, never fabricated value-matches. A shape-only test ("returns array length N") lets the oracle be silently wrong and Plan 3's GPU≈CPU check will faithfully match a broken reference.

## Scope discipline (validate-first — blueprint §8)

The blueprint mandates profiling the fluid architecture (esp. M-series) *before* committing it. Two parts of the design are exactly what the Plan 3 budget spike is most likely to reshape — so they are **deliberately minimal or deferred** here:

- **Jacobi pressure projection only. NO multigrid.** Multigrid is a convergence optimization, not a correctness requirement; whether it is needed is precisely what the Plan 3 budget spike answers. Jacobi is also the clean GPU-port target (ping-pong double-buffer, no in-iteration data dependencies). On CPU we simply run more iterations; the oracle loses nothing.
- **2.5D coupling kept minimal** — a reference vertical-mixing term, NOT a tuned weather scheme.
- **No viscosity/diffusion step** (wind viscosity ≈ 0). The same Jacobi machinery would implement it later if needed.

## Conventions (lock these so tests are unambiguous)

- **Collocated grid, cell units, grid spacing `h = 1`.** Velocity is in cells/time. Storage is `(W+2)*(H+2)` with a 1-cell border; interior is `i in 1..=W`, `j in 1..=H`. Index `idx(i,j) = i + (W+2)*j`.
- **Algorithm = Stam "Stable Fluids" / "Real-Time Fluid Dynamics for Games" (2003)**, with the linear solver as **Jacobi (ping-pong: read prev buffer, write next, swap)** — NOT Gauss-Seidel in place. This matches the engine's GPU convention and ports 1:1 to Plan 3.
- **Determinism:** no RNG, no threads, no time. Same input → bit-identical output.
- Every Rust file starts with the project comment block (name, one-line description, responsibilities).

## File structure (`crates/vs-core/`)

```
src/lib.rs                  add `pub mod fluid;` (keep abi_version)
src/fluid/mod.rs            re-exports; module doc
src/fluid/grid.rs           Grid2D: storage, idx, bilinear sample, clone/zero
src/fluid/boundary.rs       set_bnd(kind, &mut Grid2D): walls reflect velocity, scalars Neumann
src/fluid/advect.rs         advect(dst, src, u, v, dt): semi-Lagrangian backtrace + bilinear
src/fluid/project.rs        divergence + Jacobi pressure projection (make divergence-free)
src/fluid/solver.rs         Fluid2D::step + Fluid25D (stack + minimal vertical coupling)
tests/fluid_invariants.rs   cross-module physical-invariant integration tests
```

---

## Task 1 — Grid container + bilinear sampling (`grid.rs`)

`Grid2D { w, h, data: Vec<f32> }` sized `(w+2)*(h+2)`; `idx(i,j)`, `at(i,j)`, `set(i,j,v)`, `zero()`, bilinear `sample(x,y)` with Stam clamping of `x∈[0.5, w+0.5]`, `y∈[0.5, h+0.5]`.

**Invariant tests (write first):**
- `sample` at integer cell center returns that cell's exact value.
- `sample` at the midpoint between two cells returns their average (bilinear linearity).
- `sample` outside the interior clamps to the border (no panic, no OOB).
- `idx` round-trips and never aliases two distinct `(i,j)`.

## Task 2 — Boundary conditions (`boundary.rs`)

`set_bnd(kind: Bnd, g: &mut Grid2D)` where `Bnd::Scalar` (Neumann, copy neighbor), `Bnd::VelX` (negate at left/right walls — no penetration), `Bnd::VelY` (negate at top/bottom walls). Corners = average of two adjacent border cells (Stam).

**Invariant tests:**
- Scalar: border cell equals its in-bounds neighbor (zero normal gradient).
- VelX: left/right border = negation of neighbor (wall reflects normal component); top/bottom = copy.
- VelY: symmetric to VelX on the other axis.
- Corner cell = mean of its two edge neighbors.

## Task 3 — Semi-Lagrangian advection (`advect.rs`)

`advect(dst: &mut Grid2D, src: &Grid2D, u: &Grid2D, v: &Grid2D, dt: f32)`: for each interior cell backtrace `x = i - dt*u(i,j)`, `y = j - dt*v(i,j)`, bilinear-sample `src`, write `dst`. Caller applies `set_bnd` after.

**Invariant tests (the oracle's backbone — NO fabricated values):**
- **Zero velocity** → `dst == src` exactly (no spurious motion).
- **Uniform constant velocity** `(c,0)` with integer `c*dt` → field shifts by exactly `c*dt` cells (analytic, exact at integer shift); non-integer shift within bilinear-interpolation error of the analytic shift.
- **Symmetry:** a left-right-symmetric field under a symmetric velocity field stays symmetric.
- Advection does not create new extrema beyond `[min(src), max(src)]` (monotone-ish; bilinear is bounded by its 4 taps) — assert `dst` stays within `[min,max]` of `src`.

## Task 4 — Divergence + Jacobi pressure projection (`project.rs`)

`divergence(u, v) -> Grid2D` (central diff, `h=1`). `project(u: &mut Grid2D, v: &mut Grid2D, iters: usize)`: build `div`, solve `∇²p = div` via **Jacobi ping-pong** for `iters`, `set_bnd(Scalar, p)` each sweep, then subtract `∇p` from `u,v`, `set_bnd` velocities.

**Invariant tests (THE defining tests):**
- **`max|divergence(u,v)| ≈ 0` after `project`** — pick a grid size + `iters` that drives `max|div| < 1e-2` from a divergent initial field; this single test proves projection works. Document the chosen size/iters.
- **Projection reduces divergence by a large factor** (residual_after / residual_before < 1e-2) — guards against a no-op or wrong-sign solve.
- An already-divergence-free field is left ≈ unchanged (idempotence within tol).
- Symmetric divergent IC → symmetric corrected field.

## Task 5 — Fluid2D step + Fluid25D vertical coupling (`solver.rs`)

`Fluid2D { u, v, /* optional scalar s */ , iters }`; `step(dt, force_x: &Grid2D, force_y: &Grid2D)` = add forces → `project` → self-advect `u,v` (advect each through a clone of the pre-advection field) → `set_bnd` → `project`. (Stam velocity step without the diffuse stage.)

`Fluid25D { layers: Vec<Fluid2D>, kappa: f32 }`; `step(dt, forces)` = step each layer's 2D solver, then apply a **minimal vertical mixing** term: a 1D vertical Laplacian across layers, `field[l] += kappa*dt*(field[l-1] - 2*field[l] + field[l+1])`, with no-flux (Neumann) top/bottom. Conserves the vertical sum.

**Invariant tests:**
- After a full `Fluid2D::step`, `max|divergence| < 1e-2` (still divergence-free).
- **Stability:** 200 steps with a bounded force field → no `NaN`, velocity magnitude stays bounded (semi-Lagrangian is unconditionally stable).
- **Symmetry:** symmetric IC + symmetric forces → symmetric state after N steps.
- **`kappa = 0` → layers evolve identically to running each `Fluid2D` standalone** (coupling is off).
- **`kappa > 0` conserves the vertical sum** of a passively-mixed scalar (vertical Laplacian + Neumann conserves total) within tol.

---

## Self-review gates (final phase)

- Full `cargo test` green; `cargo build` clean; no warnings that indicate dead/wrong code.
- Every test above present and asserting a **tight** bound (reviewer rejects trivially-true tests).
- Jacobi (ping-pong), not Gauss-Seidel; no multigrid; no diffusion stage — scope held.
- Determinism: run the suite twice, identical results.
- Update `tasks.md`; append a `HUMAN_REVIEW.md` entry with session id + verify commands.

## Carried-forward flags (do NOT action in Plan 2)

- **wasm32 is not a one-liner later.** Homebrew rust has no rustup, so `rustup target add wasm32-unknown-unknown` is closed. Plan 3+ will need rustup installed alongside, or a Homebrew wasm path. Noted, not blocking.
