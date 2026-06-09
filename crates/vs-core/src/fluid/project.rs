// project — velocity divergence + Jacobi pressure projection (Stam, h=1).
// Makes a velocity field (approximately) divergence-free via a pressure solve.
// Responsibilities:
//   - divergence(u,v): central-difference divergence of the velocity field (h=1).
//   - jacobi_sweep(p_next,p,div): one compact 5-point Jacobi pressure update (interior).
//   - subtract_grad(u,v,p): subtract the central-difference pressure gradient (interior).
//   - project(u,v,iters): solve Laplacian(p)=div by Jacobi ping-pong (two buffers,
//     read prev / write next / swap; set_bnd(Scalar,p) each sweep), then subtract
//     grad(p) from u,v and apply set_bnd(VelX,u)/set_bnd(VelY,v).
//   - Jacobi (NOT Gauss-Seidel) so the solve ports 1:1 to the Plan 3 GPU kernel.

use super::boundary::{set_bnd, Bnd};
use super::grid::Grid2D;

/// Central-difference divergence of the velocity field `(u, v)` at every interior
/// cell (`h = 1`): `div(i,j) = 0.5*(u(i+1,j)-u(i-1,j)) + 0.5*(v(i,j+1)-v(i,j-1))`.
/// The border is left zero; the caller applies `set_bnd(Scalar, ..)` if needed.
pub fn divergence(u: &Grid2D, v: &Grid2D) -> Grid2D {
    let w = u.w;
    let h = u.h;
    let mut div = Grid2D::new(w, h);
    for j in 1..=h {
        for i in 1..=w {
            let d = 0.5 * (u.at(i + 1, j) - u.at(i - 1, j))
                + 0.5 * (v.at(i, j + 1) - v.at(i, j - 1));
            div.set(i, j, d);
        }
    }
    div
}

/// One compact 5-point Jacobi pressure sweep: for every interior cell,
/// `p_next(i,j) = 0.25*(p(i-1,j)+p(i+1,j)+p(i,j-1)+p(i,j+1) - div(i,j))`.
/// Reads `p` and `div`, writes `p_next` (interior only; the border is the caller's
/// concern via `set_bnd(Scalar, ..)`). This is the single iteration the Plan 3 GPU
/// `jacobi.wgsl` kernel reproduces, so it is exposed standalone for the GPU oracle.
pub fn jacobi_sweep(p_next: &mut Grid2D, p: &Grid2D, div: &Grid2D) {
    let w = p.w;
    let h = p.h;
    for j in 1..=h {
        for i in 1..=w {
            let sum = p.at(i - 1, j) + p.at(i + 1, j) + p.at(i, j - 1) + p.at(i, j + 1);
            p_next.set(i, j, 0.25 * (sum - div.at(i, j)));
        }
    }
}

/// Subtract the central-difference pressure gradient from `(u, v)` at every interior
/// cell (`h = 1`): `u -= 0.5*(p(i+1,j)-p(i-1,j))`, `v -= 0.5*(p(i,j+1)-p(i,j-1))`.
/// Interior only; the caller re-applies `set_bnd(VelX,u)/set_bnd(VelY,v)`. Exposed
/// standalone so the Plan 3 GPU `subtract_grad.wgsl` kernel has a 1:1 oracle.
pub fn subtract_grad(u: &mut Grid2D, v: &mut Grid2D, p: &Grid2D) {
    let w = u.w;
    let h = u.h;
    for j in 1..=h {
        for i in 1..=w {
            let gx = 0.5 * (p.at(i + 1, j) - p.at(i - 1, j));
            let gy = 0.5 * (p.at(i, j + 1) - p.at(i, j - 1));
            let nu = u.at(i, j) - gx;
            let nv = v.at(i, j) - gy;
            u.set(i, j, nu);
            v.set(i, j, nv);
        }
    }
}

/// Project `(u, v)` onto its (discretely) divergence-free part.
///
/// Solves the Poisson problem `Laplacian(p) = div(u,v)` (5-point compact stencil,
/// `h = 1`) with **Jacobi ping-pong**: two pressure buffers, each sweep reads the
/// previous buffer and writes the next, then swaps; `set_bnd(Scalar, p)` is applied
/// to the freshly written buffer every sweep. After `iters` sweeps the pressure
/// gradient is subtracted from the velocity (`u -= 0.5*(p(i+1,j)-p(i-1,j))`,
/// `v -= 0.5*(p(i,j+1)-p(i,j-1))`) and the velocity boundaries are re-applied.
///
/// Sign: the Jacobi update is `p_next = (sum_neighbors - div)/4`. Combined with the
/// gradient subtraction this yields `div(u - grad p) = div - Laplacian(p) -> 0`.
pub fn project(u: &mut Grid2D, v: &mut Grid2D, iters: usize) {
    let w = u.w;
    let h = u.h;

    // RHS of the Poisson solve: the current velocity divergence.
    let div = divergence(u, v);

    // Jacobi ping-pong buffers (both start at zero -> zero pressure guess).
    let mut p = Grid2D::new(w, h);
    let mut p_next = Grid2D::new(w, h);

    for _ in 0..iters {
        jacobi_sweep(&mut p_next, &p, &div);
        set_bnd(Bnd::Scalar, &mut p_next);
        std::mem::swap(&mut p, &mut p_next);
    }

    // Subtract the pressure gradient to remove the divergent component.
    subtract_grad(u, v, &p);
    set_bnd(Bnd::VelX, u);
    set_bnd(Bnd::VelY, v);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // ---- Test parameters (documented per the spec) ----------------------------
    //
    // GRID = 128, ITERS = 8000, Gaussian-bump sigma = GRID/10.
    //
    // Why these values (this solver is Plan 3's oracle, so the floor is real and
    // worth understanding, not papering over):
    //   * The divergence we MEASURE and the gradient we SUBTRACT are both central
    //     differences (taps at i±1), but the Poisson stencil we SOLVE is the compact
    //     5-point Laplacian. Per Fourier mode theta, the composition (central div of
    //     central grad) scales the residual divergence by (1 - lambda_compact/
    //     lambda_wide) = sin^2(theta/2). So projection annihilates the SMOOTH part of
    //     the divergence and leaves only the high-frequency part. The residual floor
    //     is therefore set by where the IC's divergence energy sits in frequency.
    //   * IC = grad of a C-infinity Gaussian potential, centered, sigma = N/10. It is
    //     (1) wall-vanishing within >=5 sigma, so set_bnd injects no boundary
    //     divergence; (2) C-infinity, so no broadband content from a kink; (3) mid-
    //     frequency (peak mode ~4-5), so Jacobi converges in a few thousand sweeps and
    //     sin^2(theta/2) at the peak mode is ~3e-3 << 1e-2.
    //   * Jacobi's slowest modes converge in O(N^2) sweeps; 8000 at N=128 is past the
    //     plateau (the residual stops improving), confirming convergence not luck.
    // Measured at GRID=128, ITERS=8000: before~1.21, after~5.9e-3 (test a, <1e-2),
    // ratio~4.9e-3 (test b, <1e-2). A wide stride-2 Laplacian would drive the residual
    // to ~2e-5 but solves the WRONG operator (decoupled odd/even, checkerboard null
    // space) — it would NOT match the compact Jacobi kernel the Plan 3 GPU port runs,
    // so the oracle deliberately keeps the compact stencil.
    const GRID: usize = 128;
    const ITERS: usize = 8000;
    const SIGMA: f32 = GRID as f32 / 10.0;
    const AMP: f32 = 100.0;

    /// Max absolute interior divergence of `(u, v)` (the projection error metric).
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

    /// Centered Gaussian potential `phi = amp*exp(-r^2/(2 sigma^2))` evaluated at
    /// cell `(i,j)`; `cx,cy` is the domain center. C-infinity and (with its gradient)
    /// negligibly small far from the center, so a velocity built from its central-
    /// difference gradient vanishes at the walls (no boundary divergence injection).
    fn gaussian(i: usize, j: usize, cx: f32, cy: f32, amp: f32, sigma: f32) -> f32 {
        let dx = i as f32 - cx;
        let dy = j as f32 - cy;
        amp * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()
    }

    /// Curl-free divergent velocity = central-difference gradient of a centered
    /// Gaussian potential. Smooth, wall-vanishing, mid-frequency: projection removes
    /// nearly all of it (large reduction factor for test b) and the result is ~0.
    fn gaussian_gradient(w: usize, h: usize) -> (Grid2D, Grid2D) {
        let mut u = Grid2D::new(w, h);
        let mut v = Grid2D::new(w, h);
        let cx = (w as f32 + 1.0) / 2.0;
        let cy = (h as f32 + 1.0) / 2.0;
        let phi = |i: usize, j: usize| gaussian(i, j, cx, cy, AMP, SIGMA);
        for j in 1..=h {
            for i in 1..=w {
                u.set(i, j, 0.5 * (phi(i + 1, j) - phi(i - 1, j)));
                v.set(i, j, 0.5 * (phi(i, j + 1) - phi(i, j - 1)));
            }
        }
        (u, v)
    }

    /// (a) THE test: starting from a divergent field, after `project` the maximum
    /// absolute interior divergence is below 1e-2 (GRID=128, ITERS=8000).
    #[test]
    fn project_drives_divergence_below_tol() {
        let (mut u, mut v) = gaussian_gradient(GRID, GRID);
        let before = max_abs_div(&u, &v);
        // IC divergence must start well ABOVE the target, so a no-op/broken solve
        // cannot pass this test by starting already-clean.
        assert!(before > 1e-1, "IC must be strongly divergent (got {before})");
        project(&mut u, &mut v, ITERS);
        let after = max_abs_div(&u, &v);
        assert!(after < 1e-2, "max|div| after project = {after}, want < 1e-2");
    }

    /// (b) Projection reduces divergence by a large factor (after/before < 1e-2).
    /// Guards against a no-op solve (ratio == 1) or a wrong-sign solve (ratio > 1,
    /// divergence grows).
    #[test]
    fn project_reduces_divergence_by_large_factor() {
        let (mut u, mut v) = gaussian_gradient(GRID, GRID);
        let before = max_abs_div(&u, &v);
        project(&mut u, &mut v, ITERS);
        let after = max_abs_div(&u, &v);
        assert!(
            after / before < 1e-2,
            "residual ratio after/before = {} (before={before}, after={after}), want < 1e-2",
            after / before
        );
    }

    /// (c) Idempotence (P^2 = P): projecting an already-projected field leaves it
    /// essentially unchanged.
    ///
    /// The IC is gradient(phi) + curl(psi) for two centered Gaussian potentials: the
    /// gradient part is divergent and removed by projection, the curl part is exactly
    /// solenoidal and SURVIVES. A pure-gradient IC would be a degenerate test here
    /// (projection drives it to ~0, so "unchanged" would be trivially true on noise);
    /// the surviving solenoidal part gives P^2 = P real teeth on a non-trivial field.
    #[test]
    fn project_is_idempotent() {
        let (w, h) = (GRID, GRID);
        let cx = (w as f32 + 1.0) / 2.0;
        let cy = (h as f32 + 1.0) / 2.0;
        let phi = |i: usize, j: usize| gaussian(i, j, cx, cy, AMP, SIGMA);
        let psi = |i: usize, j: usize| gaussian(i, j, cx, cy, AMP, SIGMA);
        let mut u = Grid2D::new(w, h);
        let mut v = Grid2D::new(w, h);
        for j in 1..=h {
            for i in 1..=w {
                // velocity = grad(phi) + curl(psi); curl(psi) = (d psi/dy, -d psi/dx).
                let gpx = 0.5 * (phi(i + 1, j) - phi(i - 1, j));
                let gpy = 0.5 * (phi(i, j + 1) - phi(i, j - 1));
                let cpx = 0.5 * (psi(i + 1, j) - psi(i - 1, j));
                let cpy = 0.5 * (psi(i, j + 1) - psi(i, j - 1));
                u.set(i, j, gpx + cpy);
                v.set(i, j, gpy - cpx);
            }
        }
        project(&mut u, &mut v, ITERS); // remove gradient part; keep solenoidal part
        let u0 = u.clone();
        let v0 = v.clone();
        project(&mut u, &mut v, ITERS); // second projection should be a near no-op

        let mut max_delta = 0.0f32;
        let mut max_mag = 0.0f32;
        for j in 1..=u.h {
            for i in 1..=u.w {
                max_delta = max_delta
                    .max((u.at(i, j) - u0.at(i, j)).abs())
                    .max((v.at(i, j) - v0.at(i, j)).abs());
                max_mag = max_mag.max(u0.at(i, j).abs()).max(v0.at(i, j).abs());
            }
        }
        // Field must carry a non-trivial surviving (solenoidal) magnitude.
        assert!(max_mag > 1e-1, "projected field must be non-trivial (mag={max_mag})");
        // Second projection changes it by << its magnitude.
        assert!(
            max_delta < 1e-2 * max_mag,
            "idempotence: max change {max_delta} not << field magnitude {max_mag}"
        );
    }

    /// (d) A left-right-symmetric divergent IC yields a corrected field with the
    /// correct mirror parity: u is ANTIsymmetric, v is symmetric, under i<->W+1-i.
    /// Uses even W so the mirror maps interior cells to interior cells. Parity is
    /// preserved exactly through the symmetric pipeline (even RHS -> even p ->
    /// odd grad_x p / even grad_y p), independent of convergence.
    #[test]
    fn project_preserves_mirror_parity() {
        let (w, h) = (GRID, GRID); // GRID is even
        let mut u = Grid2D::new(w, h);
        let mut v = Grid2D::new(w, h);
        // u = cos(...)*sin(...): cos is odd under i<->W+1-i  -> u antisymmetric.
        // v = sin(...)*cos(...): sin is even under i<->W+1-i -> v symmetric.
        // A generic divergent field (not a pure gradient), so projection leaves a
        // non-trivial symmetric solenoidal remainder.
        for j in 1..=h {
            for i in 1..=w {
                let cx = (PI * (i as f32 - 0.5) / w as f32).cos();
                let sx = (PI * (i as f32 - 0.5) / w as f32).sin();
                let sy = (PI * (j as f32 - 0.5) / h as f32).sin();
                let cy = (PI * (j as f32 - 0.5) / h as f32).cos();
                u.set(i, j, cx * sy);
                v.set(i, j, sx * cy);
            }
        }
        project(&mut u, &mut v, ITERS);

        let mut max_u = 0.0f32;
        let mut max_v = 0.0f32;
        let mut err = 0.0f32;
        for j in 1..=h {
            for i in 1..=w {
                let mi = w + 1 - i;
                // u antisymmetric: u(i,j) ~= -u(W+1-i,j).
                err = err.max((u.at(i, j) + u.at(mi, j)).abs());
                // v symmetric: v(i,j) ~= v(W+1-i,j).
                err = err.max((v.at(i, j) - v.at(mi, j)).abs());
                max_u = max_u.max(u.at(i, j).abs());
                max_v = max_v.max(v.at(i, j).abs());
            }
        }
        // Field must be non-trivially nonzero, else the parity check is vacuous.
        assert!(max_u > 1e-3 && max_v > 1e-3, "corrected field must be nonzero (u={max_u}, v={max_v})");
        // Mirrored Jacobi sums differ only at the last ULPs -> small tolerance, not ==.
        assert!(err < 1e-5, "mirror-parity error {err} exceeds tol");
    }

    /// divergence sanity: a pure horizontal shear u=u(j) has zero divergence
    /// (the x-derivative of u is zero and v is zero). Guards the divergence stencil
    /// independently of the solve.
    #[test]
    fn divergence_of_shear_is_zero() {
        let (w, h) = (8, 8);
        let mut u = Grid2D::new(w, h);
        let v = Grid2D::new(w, h);
        for j in 0..=h + 1 {
            for i in 0..=w + 1 {
                let k = u.idx(i, j);
                u.data[k] = j as f32; // depends on j only -> du/dx = 0
            }
        }
        let d = divergence(&u, &v);
        for j in 1..=h {
            for i in 1..=w {
                assert!(d.at(i, j).abs() < 1e-6, "shear divergence at ({i},{j}) = {}", d.at(i, j));
            }
        }
    }
}
