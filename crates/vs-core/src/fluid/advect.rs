// advect — semi-Lagrangian (backtrace + bilinear) transport of a field along a velocity.
// The transport stage of Stam's stable-fluids step; caller applies set_bnd afterward.
// Responsibilities:
//   - For each interior cell (i,j) backtrace x = i - dt*u(i,j), y = j - dt*v(i,j).
//   - Bilinearly sample src at (x,y) (Stam-clamped in Grid2D::sample) and write dst.
//   - Touch the interior only (1..=w x 1..=h); never write the border.

use super::grid::Grid2D;

/// Semi-Lagrangian advection: transport `src` along velocity `(u, v)` over `dt`,
/// writing the result into `dst`. `u` is the x/column velocity, `v` the y/row
/// velocity, matching `Grid2D::sample(x = column, y = row)`.
///
/// Interior cells (`1..=w` x `1..=h`) only; the caller is responsible for the
/// border via `set_bnd`. Backtrace `(x, y) = (i - dt*u, j - dt*v)` is bilinearly
/// sampled from `src` (sampling is Stam-clamped, so border taps stay in bounds).
pub fn advect(dst: &mut Grid2D, src: &Grid2D, u: &Grid2D, v: &Grid2D, dt: f32) {
    let w = dst.w;
    let h = dst.h;
    for j in 1..=h {
        for i in 1..=w {
            let x = i as f32 - dt * u.at(i, j);
            let y = j as f32 - dt * v.at(i, j);
            dst.set(i, j, src.sample(x, y));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a velocity grid whose interior is the uniform constant `(value)`.
    /// Border stays zero (advect reads velocity at interior cells only).
    fn uniform(w: usize, h: usize, value: f32) -> Grid2D {
        let mut g = Grid2D::new(w, h);
        for j in 1..=h {
            for i in 1..=w {
                g.set(i, j, value);
            }
        }
        g
    }

    /// Fill the FULL buffer (including border) with `f(i, j)`.
    /// Filling the border matters because `sample`'s Stam clamp makes border
    /// cells live interpolation taps for backtraces that leave the interior.
    fn filled_full<F: Fn(usize, usize) -> f32>(w: usize, h: usize, f: F) -> Grid2D {
        let mut g = Grid2D::new(w, h);
        for j in 0..=h + 1 {
            for i in 0..=w + 1 {
                let k = g.idx(i, j);
                g.data[k] = f(i, j);
            }
        }
        g
    }

    /// (a) Zero velocity => dst == src exactly on the interior.
    /// dt*0 == 0 and i - 0.0 == i bit-exactly, so sample returns the tap exactly.
    /// Compare interior only: advect never writes the border.
    #[test]
    fn zero_velocity_is_identity() {
        let (w, h) = (5, 4);
        let src = filled_full(w, h, |i, j| 10.0 * i as f32 + j as f32);
        let u = Grid2D::new(w, h); // all zero
        let v = Grid2D::new(w, h);
        let mut dst = Grid2D::new(w, h);
        advect(&mut dst, &src, &u, &v, 1.0);
        for j in 1..=h {
            for i in 1..=w {
                assert_eq!(
                    dst.at(i, j),
                    src.at(i, j),
                    "zero velocity must be identity at ({i},{j})"
                );
            }
        }
    }

    /// (b1) Uniform velocity (2,0), dt=1 => integer shift of exactly 2 cells.
    /// For cells whose backtrace stays interior (i-shift >= 1) the result equals
    /// src.at(i-shift, j) exactly. A sign error (x = i + dt*u) shifts the wrong
    /// way and fails this; a linear-in-both field also catches an i/j transpose.
    #[test]
    fn uniform_integer_shift_is_exact() {
        let (w, h) = (6, 4);
        let shift = 2usize;
        let src = filled_full(w, h, |i, j| 10.0 * i as f32 + j as f32);
        let u = uniform(w, h, shift as f32);
        let v = Grid2D::new(w, h);
        let mut dst = Grid2D::new(w, h);
        advect(&mut dst, &src, &u, &v, 1.0);
        for j in 1..=h {
            for i in 1..=w {
                if i >= shift + 1 {
                    // Backtrace x = i - 2 stays interior; integer coord => exact tap.
                    assert_eq!(
                        dst.at(i, j),
                        src.at(i - shift, j),
                        "integer shift must pull from upstream cell at ({i},{j})"
                    );
                }
            }
        }
    }

    /// (b2) Non-integer shift: bilinear is EXACT on an affine field, so advecting
    /// a linear field by 1.5 must match src.sample of the backtraced coordinate to
    /// floating tolerance. This pins the backtrace coordinate, not the blend
    /// (blend correctness is covered by grid::sample_midpoint_is_average).
    #[test]
    fn noninteger_shift_matches_analytic_sample() {
        let (w, h) = (6, 4);
        let dt = 0.75f32;
        let vel = 2.0f32; // shift = dt*vel = 1.5
        let src = filled_full(w, h, |i, j| 10.0 * i as f32 + j as f32);
        let u = uniform(w, h, vel);
        let v = Grid2D::new(w, h);
        let mut dst = Grid2D::new(w, h);
        advect(&mut dst, &src, &u, &v, dt);
        for j in 1..=h {
            for i in 1..=w {
                let expect = src.sample(i as f32 - dt * vel, j as f32);
                assert!(
                    (dst.at(i, j) - expect).abs() < 1e-5,
                    "non-integer shift at ({i},{j}): got {}, want {}",
                    dst.at(i, j),
                    expect
                );
            }
        }
    }

    /// (c) Left-right symmetric field under a mirror-symmetric velocity field
    /// stays symmetric. Mirror i -> w+1-i requires u ANTI-symmetric and v
    /// symmetric for the flow to be mirror-symmetric as a vector field; a
    /// converging flow (u negative on the right half, positive on the left,
    /// zero at center) satisfies this. Even w avoids a fixed center column.
    #[test]
    fn symmetric_field_stays_symmetric() {
        let (w, h) = (6, 4);
        let center = (w as f32 + 1.0) / 2.0; // 3.5 for w=6
        // Symmetric scalar: depends on |i - center|, symmetric in j too for cleanliness.
        let src = filled_full(w, h, |i, _j| {
            let d = (i as f32 - center).abs();
            10.0 - d // peak at center, symmetric under i -> w+1-i
        });
        // Anti-symmetric u: converging flow toward center. u(i) = (center - i)*0.3.
        // u(w+1-i) = (center - (w+1-i))*0.3 = (i - center)*0.3 = -u(i). Anti-symmetric.
        let mut u = Grid2D::new(w, h);
        for j in 1..=h {
            for i in 1..=w {
                u.set(i, j, (center - i as f32) * 0.3);
            }
        }
        let v = Grid2D::new(w, h); // zero is symmetric
        let mut dst = Grid2D::new(w, h);
        advect(&mut dst, &src, &u, &v, 1.0);
        for j in 1..=h {
            for i in 1..=w {
                let mirror = w + 1 - i;
                assert!(
                    (dst.at(i, j) - dst.at(mirror, j)).abs() < 1e-5,
                    "symmetry broken at ({i},{j}) vs ({mirror},{j}): {} vs {}",
                    dst.at(i, j),
                    dst.at(mirror, j)
                );
            }
        }
    }

    /// (d) No spurious extrema: bilinear is bounded by its 4 taps, so every dst
    /// value lies in [min, max] of the FULL src buffer (clamped backtraces sample
    /// border cells). Uses a peaked field + a rotational velocity so the test can
    /// actually fail on extrapolation/NaN bugs; pure bilinear cannot overshoot.
    #[test]
    fn no_spurious_extrema() {
        let (w, h) = (8, 8);
        let cx = (w as f32 + 1.0) / 2.0;
        let cy = (h as f32 + 1.0) / 2.0;
        // Peaked field: a bump at the center over a flat background.
        let src = filled_full(w, h, |i, j| {
            let dx = i as f32 - cx;
            let dy = j as f32 - cy;
            5.0 / (1.0 + 0.5 * (dx * dx + dy * dy))
        });
        // Rotational velocity about the center: u = -(y-cy), v = (x-cx), scaled.
        let mut u = Grid2D::new(w, h);
        let mut v = Grid2D::new(w, h);
        for j in 1..=h {
            for i in 1..=w {
                u.set(i, j, -(j as f32 - cy) * 0.2);
                v.set(i, j, (i as f32 - cx) * 0.2);
            }
        }
        let mut dst = Grid2D::new(w, h);
        advect(&mut dst, &src, &u, &v, 1.0);

        let smin = src.data.iter().cloned().fold(f32::INFINITY, f32::min);
        let smax = src.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for j in 1..=h {
            for i in 1..=w {
                let val = dst.at(i, j);
                assert!(val.is_finite(), "dst non-finite at ({i},{j})");
                assert!(
                    val >= smin - 1e-5 && val <= smax + 1e-5,
                    "spurious extremum at ({i},{j}): {val} outside [{smin}, {smax}]"
                );
            }
        }
    }
}
