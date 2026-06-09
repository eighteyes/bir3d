// fluid_invariants — cross-module physical-invariant tests for the CPU fluid oracle.
// Exercises Fluid2D::step and Fluid25D vertical coupling through the public API.
// Responsibilities:
//   - (a) one Fluid2D::step leaves the velocity (approximately) divergence-free, on a
//         non-trivial field (divergent + solenoidal IC) so a no-op step cannot pass.
//   - (b) 200 steps under a bounded force stay finite and bounded (semi-Lagrangian
//         is unconditionally stable).
//   - (c) symmetric IC + symmetric forces stay mirror-symmetric, on a non-trivial field.
//   - (d) kappa=0 => Fluid25D layers evolve bit-identically to standalone Fluid2D,
//         with distinct per-layer ICs so cross-talk would contaminate.
//   - (e) kappa>0 => the vertical sum of a passively-mixed scalar is conserved, and the
//         mixing actually changes the field (it is not a no-op).

use vs_core::fluid::grid::Grid2D;
use vs_core::fluid::project::divergence;
use vs_core::fluid::solver::{Fluid25D, Fluid2D};

/// Max absolute interior divergence of `(u, v)` — the divergence-free error metric.
fn max_abs_div(u: &Grid2D, v: &Grid2D) -> f32 {
    let d = divergence(u, v);
    let mut m = 0.0f32;
    for j in 1..=d.h {
        for i in 1..=d.w {
            m = m.max(d.at(i, j).abs());
        }
    }
    m
}

/// Max absolute interior velocity magnitude of `(u, v)`.
fn max_speed(u: &Grid2D, v: &Grid2D) -> f32 {
    let mut m = 0.0f32;
    for j in 1..=u.h {
        for i in 1..=u.w {
            let s = (u.at(i, j) * u.at(i, j) + v.at(i, j) * v.at(i, j)).sqrt();
            m = m.max(s);
        }
    }
    m
}

/// Fill the full buffer of a fresh `w x h` grid with `f(i, j)`.
fn filled<F: Fn(usize, usize) -> f32>(w: usize, h: usize, f: F) -> Grid2D {
    let mut g = Grid2D::new(w, h);
    for j in 0..=h + 1 {
        for i in 0..=w + 1 {
            let k = g.idx(i, j);
            g.data[k] = f(i, j);
        }
    }
    g
}

/// (a) After one `Fluid2D::step`, max|divergence| < 1e-2 on a NON-TRIVIAL field.
///
/// IC = grad(phi) + curl(psi) for two centered Gaussian potentials: the gradient part
/// is divergent (so the test is not vacuous — a broken/no-op step leaves divergence),
/// the curl part is solenoidal and survives projection (so the post-step field carries
/// real velocity and the divergence check is not measuring near-zero noise).
/// GRID=32, ITERS=2000, sigma=w/8, amp=5, dt=0.25.
///
/// Why these values: projection's residual floor is set by where the field's
/// divergence energy sits in frequency, not by iteration count (the compact 5-point
/// Jacobi stencil we solve annihilates the smooth part and leaves a high-frequency
/// residual ~ sin^2(theta/2); more iters past the plateau do NOT help — measured:
/// 1200 and 4000 iters give the identical residual). Self-advection injects high-
/// frequency divergence proportional to the velocity magnitude, so the effective lever
/// is a small, SMOOTH velocity: a wide Gaussian (sigma=w/8) with modest amplitude keeps
/// the post-advection divergence low-frequency enough that the compact solve clears it.
/// Measured here: max|div| ~1.5e-3 after one step (< 1e-2 with ~6x margin), max speed
/// ~0.73 (>> the 1e-2 non-vacuity floor, so the surviving solenoidal field is real).
#[test]
fn step_leaves_velocity_divergence_free() {
    let (w, h) = (32usize, 32usize);
    let iters = 2000usize;
    let cx = (w as f32 + 1.0) / 2.0;
    let cy = (h as f32 + 1.0) / 2.0;
    let sigma = w as f32 / 8.0;
    let amp = 5.0f32;
    let gauss = |i: usize, j: usize| {
        let dx = i as f32 - cx;
        let dy = j as f32 - cy;
        amp * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()
    };

    let mut f = Fluid2D::new(w, h, iters);
    for j in 1..=h {
        for i in 1..=w {
            // grad(phi): divergent part.
            let gpx = 0.5 * (gauss(i + 1, j) - gauss(i - 1, j));
            let gpy = 0.5 * (gauss(i, j + 1) - gauss(i, j - 1));
            // curl(psi) with psi==phi: (d psi/dy, -d psi/dx) — solenoidal part.
            let cpx = 0.5 * (gauss(i + 1, j) - gauss(i - 1, j));
            let cpy = 0.5 * (gauss(i, j + 1) - gauss(i, j - 1));
            f.u.set(i, j, gpx + cpy);
            f.v.set(i, j, gpy - cpx);
        }
    }

    let zero = Grid2D::new(w, h);
    f.step(0.25, &zero, &zero);

    // Non-trivial: the surviving solenoidal part keeps real velocity present.
    let speed = max_speed(&f.u, &f.v);
    assert!(speed > 1e-2, "post-step field must be non-trivial (max speed = {speed})");

    let after = max_abs_div(&f.u, &f.v);
    assert!(after < 1e-2, "max|div| after step = {after}, want < 1e-2");
}

/// (b) STABILITY: 200 steps with a bounded force field => no NaN, bounded velocity.
///
/// Semi-Lagrangian advection is unconditionally stable, so velocity cannot blow up to
/// infinity/NaN even with a sustained force. The bound is impulse-based, not magic:
/// a sustained per-cell force f over n steps of dt adds at most |f|*dt*n momentum, and
/// projection/advection do not amplify it, so |velocity| <= |f|*dt*n*safety bounds it.
#[test]
fn two_hundred_steps_stay_finite_and_bounded() {
    let (w, h) = (24usize, 24usize);
    let iters = 40usize;
    let dt = 0.5f32;
    let steps = 200usize;
    let fmax = 1.0f32;

    let mut f = Fluid2D::new(w, h, iters);
    // Bounded, spatially-varying force field (a swirl), |force| <= fmax.
    let cx = (w as f32 + 1.0) / 2.0;
    let cy = (h as f32 + 1.0) / 2.0;
    let r = (cx.max(cy)).max(1.0);
    let force_x = filled(w, h, |_i, j| -fmax * (j as f32 - cy) / r);
    let force_y = filled(w, h, |i, _j| fmax * (i as f32 - cx) / r);

    for _ in 0..steps {
        f.step(dt, &force_x, &force_y);
    }

    // No NaN/Inf anywhere (interior or border).
    for &val in f.u.data.iter().chain(f.v.data.iter()) {
        assert!(val.is_finite(), "velocity went non-finite: {val}");
    }
    // Impulse-based generous bound: |force|<=fmax, n=200 steps of dt.
    let bound = fmax * dt * steps as f32 * 4.0;
    let speed = max_speed(&f.u, &f.v);
    assert!(speed < bound, "max speed {speed} exceeds impulse bound {bound}");
    // Non-vacuous: a sustained force MUST have produced real velocity (a no-op step
    // would leave the field at its zero IC and pass the bound trivially).
    assert!(speed > 1e-2, "force produced no velocity (max speed = {speed}); step is a no-op");
}

/// (c) Symmetric IC + symmetric forces => mirror-symmetric state after N steps.
///
/// Under i <-> w+1-i: u must be antisymmetric, v symmetric, fx antisymmetric, fy
/// symmetric (the parity that makes the velocity field mirror-symmetric as a vector).
/// Even w maps interior cells to interior cells. Asserts symmetry AND non-trivial
/// magnitude so a collapsed/zero field cannot pass vacuously.
#[test]
fn symmetric_ic_and_forces_stay_symmetric() {
    let (w, h) = (16usize, 16usize);
    let iters = 60usize;
    let dt = 0.4f32;
    let n_steps = 6usize;
    let cx = (w as f32 + 1.0) / 2.0;

    let mut f = Fluid2D::new(w, h, iters);
    for j in 1..=h {
        for i in 1..=w {
            // u(i) = (cx - i): antisymmetric under i<->w+1-i. v depends on |i-cx|: symmetric.
            f.u.set(i, j, (cx - i as f32) * 0.2);
            f.v.set(i, j, (1.0 - (i as f32 - cx).abs() / cx) * 0.2);
        }
    }
    // Forces with matching parity: fx antisymmetric, fy symmetric.
    let force_x = filled(w, h, |i, _j| (cx - i as f32) * 0.05);
    let force_y = filled(w, h, |i, _j| (1.0 - (i as f32 - cx).abs() / cx) * 0.05);

    let u_ic = f.u.clone();
    for _ in 0..n_steps {
        f.step(dt, &force_x, &force_y);
    }
    // Non-vacuous: the field must have evolved away from the IC (a no-op step leaves
    // it equal to the symmetric IC and passes the symmetry check trivially).
    let mut max_evolve = 0.0f32;
    for j in 1..=h {
        for i in 1..=w {
            max_evolve = max_evolve.max((f.u.at(i, j) - u_ic.at(i, j)).abs());
        }
    }
    assert!(max_evolve > 1e-3, "step did not evolve the field (max change = {max_evolve})");

    let mut max_u = 0.0f32;
    let mut max_v = 0.0f32;
    let mut err = 0.0f32;
    for j in 1..=h {
        for i in 1..=w {
            let mi = w + 1 - i;
            err = err.max((f.u.at(i, j) + f.u.at(mi, j)).abs()); // u antisymmetric
            err = err.max((f.v.at(i, j) - f.v.at(mi, j)).abs()); // v symmetric
            max_u = max_u.max(f.u.at(i, j).abs());
            max_v = max_v.max(f.v.at(i, j).abs());
        }
    }
    assert!(max_u > 1e-3 && max_v > 1e-3, "field must be non-trivial (u={max_u}, v={max_v})");
    assert!(err < 1e-4, "mirror-symmetry error {err} exceeds tol");
}

/// (d) kappa = 0 => Fluid25D layers evolve BIT-IDENTICALLY to standalone Fluid2D.
///
/// With kappa=0 the vertical mixing adds exactly 0.0 (exact for finite floats) and the
/// per-layer code path is the identical `Fluid2D::step` method, so equality is exact,
/// not within-tolerance. Distinct per-layer ICs ensure any cross-layer contamination
/// (a mixing term that fired) would visibly diverge the two paths.
#[test]
fn kappa_zero_matches_standalone_layers() {
    let (w, h) = (12usize, 12usize);
    let iters = 30usize;
    let dt = 0.5f32;
    let n_layers = 3usize;
    let n_steps = 4usize;

    let make_layer = |seed: f32| {
        let mut f = Fluid2D::new(w, h, iters);
        for j in 1..=h {
            for i in 1..=w {
                // Distinct per-layer field via `seed`.
                f.u.set(i, j, (i as f32 * 0.1 + seed).sin() * 0.3);
                f.v.set(i, j, (j as f32 * 0.1 + seed).cos() * 0.3);
                f.s.set(i, j, i as f32 + 10.0 * j as f32 + 100.0 * seed);
            }
        }
        f
    };

    let force_x = filled(w, h, |_i, j| (j as f32 * 0.05).sin() * 0.2);
    let force_y = filled(w, h, |i, _j| (i as f32 * 0.05).cos() * 0.2);

    // Standalone reference: each layer stepped independently.
    let mut standalone: Vec<Fluid2D> =
        (0..n_layers).map(|l| make_layer(l as f32 + 1.0)).collect();
    for _ in 0..n_steps {
        for f in standalone.iter_mut() {
            f.step(dt, &force_x, &force_y);
        }
    }

    // Coupled with kappa=0: identical layers, mixing must be a no-op.
    let coupled_ic: Vec<Fluid2D> =
        (0..n_layers).map(|l| make_layer(l as f32 + 1.0)).collect();
    let mut stack = Fluid25D::new(coupled_ic.clone(), 0.0);
    for _ in 0..n_steps {
        stack.step(dt, &force_x, &force_y);
    }

    // Non-vacuous: stepping must have evolved each layer away from its IC (else the
    // equality is no-op == no-op and proves nothing about coupling being off).
    for (l, (ic, now)) in coupled_ic.iter().zip(stack.layers.iter()).enumerate() {
        let mut max_evolve = 0.0f32;
        for j in 1..=h {
            for i in 1..=w {
                max_evolve = max_evolve.max((now.u.at(i, j) - ic.u.at(i, j)).abs());
            }
        }
        assert!(max_evolve > 1e-3, "layer {l} did not evolve (max change = {max_evolve}); step is a no-op");
    }

    for (l, (a, b)) in standalone.iter().zip(stack.layers.iter()).enumerate() {
        assert_eq!(a.u.data, b.u.data, "layer {l} u differs (kappa=0 must be no-op)");
        assert_eq!(a.v.data, b.v.data, "layer {l} v differs (kappa=0 must be no-op)");
        assert_eq!(a.s.data, b.s.data, "layer {l} s differs (kappa=0 must be no-op)");
    }
}

/// (e) kappa > 0 conserves the vertical sum of a passively-mixed scalar, AND the mixing
/// actually changes the field (it is not a silent no-op).
///
/// Velocities and forces are zero, so the only thing that moves `s` is vertical mixing.
/// The vertical Laplacian with Neumann (no-flux) ends conserves sum_l s[l] at each cell.
/// f32 telescoping leaves ~eps drift, so the assertion is a relative tolerance.
#[test]
fn kappa_positive_conserves_vertical_sum() {
    let (w, h) = (10usize, 10usize);
    let iters = 10usize;
    let dt = 0.5f32;
    let kappa = 0.3f32;
    let n_layers = 4usize;
    let n_steps = 8usize;

    // Layers with zero velocity; scalar varies sharply across layers so mixing bites.
    let make_layer = |l: usize| {
        let mut f = Fluid2D::new(w, h, iters);
        for j in 1..=h {
            for i in 1..=w {
                // Distinct, non-monotone-per-cell scalar profile across layers.
                let val = if l % 2 == 0 { 10.0 } else { 2.0 } + (i + j) as f32;
                f.s.set(i, j, val);
            }
        }
        f
    };
    let layers: Vec<Fluid2D> = (0..n_layers).map(make_layer).collect();

    // Per-cell vertical sum BEFORE mixing.
    let sum_before: Vec<f32> = {
        let mut sums = vec![0.0f32; (w + 2) * (h + 2)];
        for f in layers.iter() {
            for k in 0..sums.len() {
                sums[k] += f.s.data[k];
            }
        }
        sums
    };
    // Snapshot layer 0 to later prove mixing actually changed something.
    let s0_before = layers[0].s.clone();

    let mut stack = Fluid25D::new(layers, kappa);
    let zero = Grid2D::new(w, h);
    for _ in 0..n_steps {
        stack.step(dt, &zero, &zero);
    }

    // Vertical sum AFTER mixing.
    let mut sum_after = vec![0.0f32; (w + 2) * (h + 2)];
    for f in stack.layers.iter() {
        for k in 0..sum_after.len() {
            sum_after[k] += f.s.data[k];
        }
    }

    // Conservation: per-interior-cell vertical sum unchanged (relative tol).
    let g = &stack.layers[0].s;
    for j in 1..=h {
        for i in 1..=w {
            let k = g.idx(i, j);
            let denom = sum_before[k].abs().max(1.0);
            let rel = (sum_after[k] - sum_before[k]).abs() / denom;
            assert!(
                rel < 1e-4,
                "vertical sum not conserved at ({i},{j}): before={}, after={}, rel={rel}",
                sum_before[k],
                sum_after[k]
            );
        }
    }

    // Mixing must have actually moved the scalar (else conservation is vacuous).
    let mut max_change = 0.0f32;
    for j in 1..=h {
        for i in 1..=w {
            max_change = max_change.max((stack.layers[0].s.at(i, j) - s0_before.at(i, j)).abs());
        }
    }
    assert!(max_change > 1e-2, "vertical mixing did not change the scalar (max change = {max_change})");
}
