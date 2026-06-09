// boundary — Stam set_bnd wall/border conditions for fluid fields.
// Enforces no-penetration velocity walls and zero-normal-gradient scalar walls.
// Responsibilities:
//   - Set the 1-cell border of a Grid2D per the field kind (Scalar / VelX / VelY).
//   - Scalar (Neumann): border copies its in-bounds neighbor (zero normal gradient).
//   - VelX: negate the normal component at left/right walls, copy at top/bottom.
//   - VelY: negate the normal component at top/bottom walls, copy at left/right.
//   - Corners: mean of the two adjacent edge cells (set after the wall pass, Stam).
//   - Generalize Stam's square (N x N) code to rectangular (W x H) interiors.

use super::grid::Grid2D;

/// Boundary kind for [`set_bnd`]. Selects which walls negate the normal component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bnd {
    /// Scalar field (pressure, density): Neumann — every border copies its neighbor.
    Scalar,
    /// X-velocity component: negate at left/right walls (no penetration), copy top/bottom.
    VelX,
    /// Y-velocity component: negate at top/bottom walls (no penetration), copy left/right.
    VelY,
}

/// Apply boundary conditions to the 1-cell border of `g`.
///
/// Stam's `set_bnd`, generalized from a square grid to `g.w` x `g.h`:
/// left/right walls iterate the H-range, top/bottom walls iterate the W-range,
/// then the four corners are set to the mean of their two adjacent edge cells.
pub fn set_bnd(kind: Bnd, g: &mut Grid2D) {
    let w = g.w;
    let h = g.h;

    // Left (i=0) and right (i=w+1) walls: VelX negates the normal (x) component.
    let neg_x = kind == Bnd::VelX;
    for j in 1..=h {
        let left = g.at(1, j);
        let right = g.at(w, j);
        g.set(0, j, if neg_x { -left } else { left });
        g.set(w + 1, j, if neg_x { -right } else { right });
    }

    // Bottom (j=0) and top (j=h+1) walls: VelY negates the normal (y) component.
    let neg_y = kind == Bnd::VelY;
    for i in 1..=w {
        let bottom = g.at(i, 1);
        let top = g.at(i, h);
        g.set(i, 0, if neg_y { -bottom } else { bottom });
        g.set(i, h + 1, if neg_y { -top } else { top });
    }

    // Corners = mean of the two adjacent edge cells (already set above).
    g.set(0, 0, 0.5 * (g.at(1, 0) + g.at(0, 1)));
    g.set(0, h + 1, 0.5 * (g.at(1, h + 1) + g.at(0, h)));
    g.set(w + 1, 0, 0.5 * (g.at(w, 0) + g.at(w + 1, 1)));
    g.set(w + 1, h + 1, 0.5 * (g.at(w, h + 1) + g.at(w + 1, h)));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fill the interior (1..=w, 1..=h) with distinct non-zero values; border stays 0.
    /// v = 100*i + j + 1 guarantees no interior cell is 0, so negate != copy is testable.
    /// W != H (4 x 3) so an axis-swap bug (treating the grid as square) surfaces.
    fn interior_filled(w: usize, h: usize) -> Grid2D {
        let mut g = Grid2D::new(w, h);
        for j in 1..=h {
            for i in 1..=w {
                g.set(i, j, (100 * i + j) as f32 + 1.0);
            }
        }
        g
    }

    /// (a) Scalar: every border cell equals its in-bounds neighbor (zero normal gradient).
    #[test]
    fn scalar_border_copies_neighbor() {
        let mut g = interior_filled(4, 3);
        let (w, h) = (g.w, g.h);
        set_bnd(Bnd::Scalar, &mut g);
        for j in 1..=h {
            assert_eq!(g.at(0, j), g.at(1, j), "left border copies neighbor at j={j}");
            assert_eq!(g.at(w + 1, j), g.at(w, j), "right border copies neighbor at j={j}");
        }
        for i in 1..=w {
            assert_eq!(g.at(i, 0), g.at(i, 1), "bottom border copies neighbor at i={i}");
            assert_eq!(g.at(i, h + 1), g.at(i, h), "top border copies neighbor at i={i}");
        }
    }

    /// (b) VelX: left/right border = negation of neighbor; top/bottom border = copy.
    #[test]
    fn velx_negates_left_right_copies_top_bottom() {
        let mut g = interior_filled(4, 3);
        let (w, h) = (g.w, g.h);
        set_bnd(Bnd::VelX, &mut g);
        for j in 1..=h {
            assert_eq!(g.at(0, j), -g.at(1, j), "left border negates neighbor at j={j}");
            assert_eq!(g.at(w + 1, j), -g.at(w, j), "right border negates neighbor at j={j}");
        }
        for i in 1..=w {
            assert_eq!(g.at(i, 0), g.at(i, 1), "bottom border copies neighbor at i={i}");
            assert_eq!(g.at(i, h + 1), g.at(i, h), "top border copies neighbor at i={i}");
        }
    }

    /// (c) VelY: symmetric counterpart — top/bottom negate; left/right copy.
    #[test]
    fn vely_negates_top_bottom_copies_left_right() {
        let mut g = interior_filled(4, 3);
        let (w, h) = (g.w, g.h);
        set_bnd(Bnd::VelY, &mut g);
        for i in 1..=w {
            assert_eq!(g.at(i, 0), -g.at(i, 1), "bottom border negates neighbor at i={i}");
            assert_eq!(g.at(i, h + 1), -g.at(i, h), "top border negates neighbor at i={i}");
        }
        for j in 1..=h {
            assert_eq!(g.at(0, j), g.at(1, j), "left border copies neighbor at j={j}");
            assert_eq!(g.at(w + 1, j), g.at(w, j), "right border copies neighbor at j={j}");
        }
    }

    /// (d) Each corner equals the mean of its two adjacent edge cells (read post-state,
    /// so the corner's dependence on the wall pass having already run is verified too).
    #[test]
    fn corners_are_mean_of_adjacent_edges() {
        // Run for each kind: corners always average edges regardless of negation.
        for kind in [Bnd::Scalar, Bnd::VelX, Bnd::VelY] {
            let mut g = interior_filled(4, 3);
            let (w, h) = (g.w, g.h);
            set_bnd(kind, &mut g);
            assert_eq!(
                g.at(0, 0),
                0.5 * (g.at(1, 0) + g.at(0, 1)),
                "bottom-left corner mean ({kind:?})"
            );
            assert_eq!(
                g.at(0, h + 1),
                0.5 * (g.at(1, h + 1) + g.at(0, h)),
                "top-left corner mean ({kind:?})"
            );
            assert_eq!(
                g.at(w + 1, 0),
                0.5 * (g.at(w, 0) + g.at(w + 1, 1)),
                "bottom-right corner mean ({kind:?})"
            );
            assert_eq!(
                g.at(w + 1, h + 1),
                0.5 * (g.at(w, h + 1) + g.at(w + 1, h)),
                "top-right corner mean ({kind:?})"
            );
        }
    }
}
