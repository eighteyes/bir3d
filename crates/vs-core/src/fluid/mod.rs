// fluid — CPU reference fluid solver (Stam "Real-Time Fluid Dynamics for Games", 2003).
// Pure-CPU, deterministic correctness oracle for the Plan 3 GPU port.
// Responsibilities:
//   - Aggregate the fluid solver submodules and re-export their public API.
//   - Define module-wide conventions: collocated grid, cell units (h=1),
//     (w+2)*(h+2) storage with a 1-cell border, interior i in 1..=w / j in 1..=h.

pub mod advect;
pub mod boundary;
pub mod grid;
