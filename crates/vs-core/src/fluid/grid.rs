// grid — 2D scalar field container with Stam-clamped bilinear sampling.
// Backing store for every fluid quantity (velocity components, pressure, scalars).
// Responsibilities:
//   - Hold a (w+2)*(h+2) f32 grid with a 1-cell border; interior i in 1..=w, j in 1..=h.
//   - Provide idx/at/set/zero accessors over the row-major layout idx(i,j)=i+(w+2)*j.
//   - Sample at continuous coords via bilinear interpolation with Stam clamping
//     (x in [0.5, w+0.5], y in [0.5, h+0.5]) — never panics, never reads OOB.

/// A 2D scalar field of `(w+2)*(h+2)` cells with a one-cell border.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid2D {
    /// Interior width (columns 1..=w are interior; 0 and w+1 are border).
    pub w: usize,
    /// Interior height (rows 1..=h are interior; 0 and h+1 are border).
    pub h: usize,
    /// Row-major storage, length `(w+2)*(h+2)`, addressed by `idx(i,j)`.
    pub data: Vec<f32>,
}

impl Grid2D {
    /// Allocate a zeroed grid with interior dimensions `w` x `h`.
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![0.0; (w + 2) * (h + 2)] }
    }

    /// Flatten `(i, j)` to a storage offset. Layout: `i + (w+2)*j` (row-major).
    #[inline]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        i + (self.w + 2) * j
    }

    /// Read the value at cell `(i, j)`.
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f32 {
        self.data[self.idx(i, j)]
    }

    /// Write `v` to cell `(i, j)`.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f32) {
        let k = self.idx(i, j);
        self.data[k] = v;
    }

    /// Reset every cell (interior and border) to zero.
    pub fn zero(&mut self) {
        for v in self.data.iter_mut() {
            *v = 0.0;
        }
    }

    /// Bilinearly sample the field at continuous coords `(x, y)`.
    /// Integer coords land on cell centers (returns that cell exactly).
    /// Coords are Stam-clamped to `[0.5, w+0.5]` x `[0.5, h+0.5]` so the four
    /// interpolation taps always stay in-bounds — no panic, no OOB.
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        // Stam clamp: keep coords in [0.5, w+0.5] x [0.5, h+0.5] so floor()+1
        // indices stay within [0, w+1] x [0, h+1] (the full storage range).
        let x = x.clamp(0.5, self.w as f32 + 0.5);
        let y = y.clamp(0.5, self.h as f32 + 0.5);

        // Lower-left tap and fractional offsets (x>0 after clamp, so cast is safe).
        let i0 = x.floor() as usize;
        let j0 = y.floor() as usize;
        let i1 = i0 + 1;
        let j1 = j0 + 1;
        let s1 = x - i0 as f32;
        let s0 = 1.0 - s1;
        let t1 = y - j0 as f32;
        let t0 = 1.0 - t1;

        // Bilinear blend of the four surrounding cells.
        s0 * (t0 * self.at(i0, j0) + t1 * self.at(i0, j1))
            + s1 * (t0 * self.at(i1, j0) + t1 * self.at(i1, j1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fill a small grid so each cell carries a distinct, predictable value.
    /// Value encodes the coords: v = 100*i + j, so aliasing/transpose bugs surface.
    fn filled(w: usize, h: usize) -> Grid2D {
        let mut g = Grid2D::new(w, h);
        for j in 0..=h + 1 {
            for i in 0..=w + 1 {
                let k = g.idx(i, j);
                g.data[k] = (100 * i + j) as f32;
            }
        }
        g
    }

    /// (a) sampling at an integer cell center returns that cell's exact value.
    #[test]
    fn sample_at_cell_center_is_exact() {
        let g = filled(4, 3);
        for j in 1..=3 {
            for i in 1..=4 {
                assert_eq!(
                    g.sample(i as f32, j as f32),
                    g.at(i, j),
                    "sample({i},{j}) must equal cell value"
                );
            }
        }
    }

    /// (b) sampling the midpoint between two horizontally-adjacent cells returns
    /// their average (bilinear linearity); likewise vertically.
    #[test]
    fn sample_midpoint_is_average() {
        let g = filled(4, 3);
        // Horizontal midpoint between (2,2) and (3,2).
        let expect_h = 0.5 * (g.at(2, 2) + g.at(3, 2));
        assert!(
            (g.sample(2.5, 2.0) - expect_h).abs() < 1e-6,
            "horizontal midpoint: got {}, want {}",
            g.sample(2.5, 2.0),
            expect_h
        );
        // Vertical midpoint between (2,2) and (2,3).
        let expect_v = 0.5 * (g.at(2, 2) + g.at(2, 3));
        assert!(
            (g.sample(2.0, 2.5) - expect_v).abs() < 1e-6,
            "vertical midpoint: got {}, want {}",
            g.sample(2.0, 2.5),
            expect_v
        );
        // Center of the four-cell block: mean of all four taps.
        let expect_c = 0.25 * (g.at(2, 2) + g.at(3, 2) + g.at(2, 3) + g.at(3, 3));
        assert!(
            (g.sample(2.5, 2.5) - expect_c).abs() < 1e-6,
            "block center: got {}, want {}",
            g.sample(2.5, 2.5),
            expect_c
        );
    }

    /// (c) sampling outside the interior clamps to the border: no panic, no OOB,
    /// and far-outside coords equal the sample at the clamp boundary.
    #[test]
    fn sample_outside_clamps_to_border() {
        let g = filled(4, 3);
        // Must not panic at extreme coords on any side.
        let _ = g.sample(-100.0, 2.0);
        let _ = g.sample(1000.0, 2.0);
        let _ = g.sample(2.0, -100.0);
        let _ = g.sample(2.0, 1000.0);
        let _ = g.sample(-1e9, 1e9);
        // Clamp-equality: far-outside collapses onto the clamp boundary value.
        assert_eq!(
            g.sample(-100.0, 2.0),
            g.sample(0.5, 2.0),
            "left clamp to x=0.5"
        );
        assert_eq!(
            g.sample(1000.0, 2.0),
            g.sample(g.w as f32 + 0.5, 2.0),
            "right clamp to x=w+0.5"
        );
        assert_eq!(
            g.sample(2.0, -100.0),
            g.sample(2.0, 0.5),
            "bottom clamp to y=0.5"
        );
        assert_eq!(
            g.sample(2.0, 1000.0),
            g.sample(2.0, g.h as f32 + 0.5),
            "top clamp to y=h+0.5"
        );
    }

    /// (d) idx is injective over the full (w+2)*(h+2) range and round-trips.
    /// Rectangular grid catches transposed-stride bugs (i+(h+2)*j).
    #[test]
    fn idx_roundtrips_and_never_aliases() {
        use std::collections::HashSet;
        let w = 4;
        let h = 3;
        let g = Grid2D::new(w, h);
        let stride = w + 2;
        let mut seen = HashSet::new();
        for j in 0..=h + 1 {
            for i in 0..=w + 1 {
                let k = g.idx(i, j);
                // Injective: no two distinct (i,j) collide.
                assert!(seen.insert(k), "idx collision at ({i},{j}) -> {k}");
                // In range.
                assert!(k < g.data.len(), "idx({i},{j})={k} out of range");
                // Round-trip recovers (i,j).
                assert_eq!(k % stride, i, "round-trip i at ({i},{j})");
                assert_eq!(k / stride, j, "round-trip j at ({i},{j})");
            }
        }
        assert_eq!(seen.len(), (w + 2) * (h + 2), "idx must cover full range");
    }
}
