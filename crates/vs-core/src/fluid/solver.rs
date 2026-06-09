// solver — Stam velocity step (Fluid2D) + minimal 2.5D vertical layer coupling (Fluid25D).
// Assembles advect/project/set_bnd into the stable-fluids time step and stacks 2D
// solvers into layers coupled by a vertical Laplacian mixing term.
// Responsibilities:
//   - Fluid2D::step(dt, fx, fy): add forces -> project -> self-advect u,v through a
//     clone of the pre-advection (u,v) -> set_bnd velocities -> project. No diffusion.
//   - Hold a passive scalar `s` advected by nothing in the 2D step; it is moved ONLY by
//     Fluid25D vertical mixing (keeps the vertical-sum conservation invariant clean).
//   - Fluid25D::step: step every layer's 2D solver, then mix u,v,s vertically with a
//     Jacobi-across-layers Laplacian (snapshot all layers, compute deltas, apply) and
//     no-flux (Neumann) ends. The Jacobi snapshot is what conserves the vertical sum.

use super::advect::advect;
use super::boundary::{set_bnd, Bnd};
use super::grid::Grid2D;
use super::project::project;

/// A single 2D fluid layer: velocity `(u, v)`, a passive scalar `s`, and the Jacobi
/// pressure-solve iteration count `iters`.
#[derive(Clone, Debug)]
pub struct Fluid2D {
    pub u: Grid2D,
    pub v: Grid2D,
    pub s: Grid2D,
    pub iters: usize,
}

impl Fluid2D {
    /// Allocate a zeroed `w` x `h` layer with `iters` Jacobi pressure iterations.
    pub fn new(w: usize, h: usize, iters: usize) -> Self {
        Self {
            u: Grid2D::new(w, h),
            v: Grid2D::new(w, h),
            s: Grid2D::new(w, h),
            iters,
        }
    }

    /// One stable-fluids velocity step (no diffusion stage):
    /// 1. add forces: `u += dt*force_x`, `v += dt*force_y` (interior);
    /// 2. `project` to remove the divergence the forces introduced;
    /// 3. self-advect: backtrace `u` and `v` through a clone of the *pre-advection*
    ///    `(u, v)` so both components use the same velocity field;
    /// 4. `set_bnd` the advected velocities (advect writes the interior only);
    /// 5. `project` again to restore the divergence-free condition.
    pub fn step(&mut self, dt: f32, force_x: &Grid2D, force_y: &Grid2D) {
        let w = self.u.w;
        let h = self.u.h;

        // 1. Add forces (interior only; project re-applies velocity boundaries).
        for j in 1..=h {
            for i in 1..=w {
                let nu = self.u.at(i, j) + dt * force_x.at(i, j);
                let nv = self.v.at(i, j) + dt * force_y.at(i, j);
                self.u.set(i, j, nu);
                self.v.set(i, j, nv);
            }
        }

        // 2. Project the post-force field.
        project(&mut self.u, &mut self.v, self.iters);

        // 3. Self-advect: both components backtrace through the SAME pre-advection
        //    velocity snapshot (u0, v0).
        let u0 = self.u.clone();
        let v0 = self.v.clone();
        advect(&mut self.u, &u0, &u0, &v0, dt);
        advect(&mut self.v, &v0, &u0, &v0, dt);

        // 4. Boundaries on the advected velocities (advect touched the interior only).
        set_bnd(Bnd::VelX, &mut self.u);
        set_bnd(Bnd::VelY, &mut self.v);

        // 5. Final projection.
        project(&mut self.u, &mut self.v, self.iters);
    }
}

/// A vertical stack of 2D layers coupled by a vertical Laplacian mixing term.
#[derive(Clone, Debug)]
pub struct Fluid25D {
    pub layers: Vec<Fluid2D>,
    pub kappa: f32,
}

impl Fluid25D {
    /// Build from explicit layers and a vertical mixing coefficient.
    pub fn new(layers: Vec<Fluid2D>, kappa: f32) -> Self {
        Self { layers, kappa }
    }

    /// Step every layer's 2D solver, then apply a minimal vertical mixing term across
    /// layers: a 1D vertical Laplacian `field[l] += kappa*dt*(field[l-1]-2*field[l]+
    /// field[l+1])` applied to `u`, `v`, and the passive scalar `s`, with no-flux
    /// (Neumann) top/bottom ends. Conserves the per-cell vertical sum of each field.
    ///
    /// Mixing is **Jacobi across the layer index**: every delta is computed from the
    /// SAME pre-mixing snapshot, then applied. In-place sequential mixing would feed an
    /// already-updated `l-1` into layer `l` and break the telescoping that conserves
    /// the vertical sum. With a single layer (or `kappa==0`) mixing is a no-op.
    pub fn step(&mut self, dt: f32, force_x: &Grid2D, force_y: &Grid2D) {
        // 1. Advance each layer's 2D dynamics independently.
        for layer in self.layers.iter_mut() {
            layer.step(dt, force_x, force_y);
        }

        let n = self.layers.len();
        if n < 2 || self.kappa == 0.0 {
            return; // nothing to couple
        }
        let coef = self.kappa * dt;

        // 2. Snapshot every field of every layer (Jacobi read buffer).
        let snap_u: Vec<Vec<f32>> = self.layers.iter().map(|l| l.u.data.clone()).collect();
        let snap_v: Vec<Vec<f32>> = self.layers.iter().map(|l| l.v.data.clone()).collect();
        let snap_s: Vec<Vec<f32>> = self.layers.iter().map(|l| l.s.data.clone()).collect();
        let len = snap_u[0].len();

        // 3. Apply the vertical Laplacian per layer from the snapshot. Neumann ends:
        //    the missing neighbor is taken to equal the current layer, so the boundary
        //    Laplacian uses only the one interior neighbor (no-flux -> conserves sum).
        for l in 0..n {
            let up = if l == 0 { l } else { l - 1 };
            let down = if l + 1 == n { l } else { l + 1 };
            for k in 0..len {
                let lap_u = snap_u[up][k] - 2.0 * snap_u[l][k] + snap_u[down][k];
                let lap_v = snap_v[up][k] - 2.0 * snap_v[l][k] + snap_v[down][k];
                let lap_s = snap_s[up][k] - 2.0 * snap_s[l][k] + snap_s[down][k];
                self.layers[l].u.data[k] += coef * lap_u;
                self.layers[l].v.data[k] += coef * lap_v;
                self.layers[l].s.data[k] += coef * lap_s;
            }
        }
    }
}
