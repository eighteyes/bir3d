// dump_fixture — emit CPU-oracle reference fixtures (JSON) for the Plan 3 GPU port.
// Runs deterministic (no-RNG, seeded-sinusoid) scenarios against the verified Plan 2
// fluid kernels and writes per-kernel + composed fixtures the GPU tests compare against.
// Responsibilities:
//   - Seed inputs from a fixed sinusoid over the FULL (w+2)*(h+2) buffer (border too),
//     so edge cells exercise their border taps; no RNG -> bit-deterministic.
//   - Run exactly ONE application of each public kernel (advect, divergence,
//     jacobi_sweep, subtract_grad, set_bnd{scalar,velx,vely}) and one composed
//     Fluid2D::step x N, calling the verified pub fns directly.
//   - Serialize each scenario to { kernel, w, h, params, inputs, expected } JSON with
//     arrays in full (w+2)*(h+2) row-major layout (idx = i + (w+2)*j).
//   - Write one file per fixture under tests/fixtures/fluid/ AND print the combined
//     JSON object to stdout (so the verify command emits valid JSON). Logs go to stderr.
//
// Regenerate (from the repo root):
//   cargo run -p vs-core --example dump_fixture
// This (re)writes tests/fixtures/fluid/*.json. The path is resolved relative to
// CARGO_MANIFEST_DIR so the command works from any cwd.

use std::fmt::Write as _;
use std::path::PathBuf;

use vs_core::fluid::advect::advect;
use vs_core::fluid::boundary::{set_bnd, Bnd};
use vs_core::fluid::grid::Grid2D;
use vs_core::fluid::project::{divergence, jacobi_sweep, subtract_grad};
use vs_core::fluid::solver::Fluid2D;

// ---- Deterministic seeding ---------------------------------------------------

/// Fill the FULL buffer (border included, 0..=w+1 x 0..=h+1) of a fresh grid from
/// `f(i, j)`. Seeding the border matters: edge cells read border taps (advect's
/// Stam clamp, the divergence/jacobi/grad stencils at i=1), so an all-zero border
/// would leave those paths untested.
fn seed_full<F: Fn(usize, usize) -> f32>(w: usize, h: usize, f: F) -> Grid2D {
    let mut g = Grid2D::new(w, h);
    for j in 0..=h + 1 {
        for i in 0..=w + 1 {
            let k = g.idx(i, j);
            g.data[k] = f(i, j);
        }
    }
    g
}

/// Fill the INTERIOR (1..=w x 1..=h) only; border stays zero. Used where the
/// kernel/state semantics expect a zero border on input (e.g. Fluid2D fields,
/// scripted forces) and the kernel itself writes the border via set_bnd.
fn seed_interior<F: Fn(usize, usize) -> f32>(w: usize, h: usize, f: F) -> Grid2D {
    let mut g = Grid2D::new(w, h);
    for j in 1..=h {
        for i in 1..=w {
            g.set(i, j, f(i, j));
        }
    }
    g
}

/// Seeded sinusoid sample with a phase offset so distinct fields differ.
/// Smooth, bounded in [-1, 1], deterministic; coords are 1-based cell centers.
fn sinu(i: usize, j: usize, kx: f32, ky: f32, phase: f32) -> f32 {
    (kx * i as f32 + ky * j as f32 + phase).sin()
}

// ---- JSON serialization (hand-rolled; the crate has no serde dep) ------------

/// Format one f32 as a JSON number using Rust's shortest round-trip repr.
/// `{}` on f32 round-trips exactly (parse back -> identical bits), which the 1e-5
/// GPU tolerance relies on. NaN/Inf cannot occur in these smooth scenarios, but
/// guard anyway so the output is always valid JSON.
fn fnum(x: f32) -> String {
    if x.is_finite() {
        format!("{}", x)
    } else {
        "null".to_string()
    }
}

/// Serialize a full-buffer array (length (w+2)*(h+2), row-major) as a JSON array.
fn arr_json(g: &Grid2D) -> String {
    let mut s = String::with_capacity(g.data.len() * 8);
    s.push('[');
    for (k, v) in g.data.iter().enumerate() {
        if k > 0 {
            s.push(',');
        }
        s.push_str(&fnum(*v));
    }
    s.push(']');
    s
}

/// A `"key": <grid-array>` member.
fn member(key: &str, g: &Grid2D) -> String {
    format!("\"{}\":{}", key, arr_json(g))
}

/// Assemble one fixture object. `params` and `inputs`/`expected` are pre-built
/// JSON member lists (comma-joined `"k":v` strings, no braces).
fn fixture_json(kernel: &str, w: usize, h: usize, params: &str, inputs: &[String], expected: &[String]) -> String {
    let mut s = String::new();
    let _ = write!(
        s,
        "{{\"kernel\":\"{}\",\"w\":{},\"h\":{},\"params\":{{{}}},\"inputs\":{{{}}},\"expected\":{{{}}}}}",
        kernel,
        w,
        h,
        params,
        inputs.join(","),
        expected.join(","),
    );
    s
}

// ---- Scenarios ---------------------------------------------------------------

/// (kernel, w, h, single-line JSON fixture). One application of one kernel each.
fn build_fixtures() -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();

    // advect: transport `field` along velocity (u, v) over dt; interior written.
    {
        let (w, h, dt) = (6usize, 5usize, 0.75f32);
        let field = seed_full(w, h, |i, j| 1.0 + sinu(i, j, 0.7, 0.4, 0.0));
        let u = seed_full(w, h, |i, j| 0.8 * sinu(i, j, 0.3, 0.5, 1.1));
        let v = seed_full(w, h, |i, j| 0.6 * sinu(i, j, 0.5, 0.3, 2.2));
        let mut dst = Grid2D::new(w, h);
        advect(&mut dst, &field, &u, &v, dt);
        let params = format!("\"dt\":{}", fnum(dt));
        let inputs = vec![member("field", &field), member("u", &u), member("v", &v)];
        let expected = vec![member("field", &dst)];
        out.push(("advect".into(), fixture_json("advect", w, h, &params, &inputs, &expected)));
    }

    // divergence: central-difference divergence of (u, v); interior written.
    {
        let (w, h) = (6usize, 5usize);
        let u = seed_full(w, h, |i, j| sinu(i, j, 0.6, 0.2, 0.3));
        let v = seed_full(w, h, |i, j| sinu(i, j, 0.25, 0.55, 1.7));
        let div = divergence(&u, &v);
        let inputs = vec![member("u", &u), member("v", &v)];
        let expected = vec![member("div", &div)];
        out.push(("divergence".into(), fixture_json("divergence", w, h, "", &inputs, &expected)));
    }

    // jacobi: ONE compact 5-point Jacobi sweep. p seeded nonzero (not the zero guess)
    // so all four neighbor taps AND the -div term are exercised. No set_bnd bundled.
    {
        let (w, h) = (6usize, 5usize);
        let p = seed_full(w, h, |i, j| sinu(i, j, 0.4, 0.6, 0.9));
        let div = seed_full(w, h, |i, j| 0.5 * sinu(i, j, 0.7, 0.3, 2.0));
        let mut p_next = Grid2D::new(w, h);
        jacobi_sweep(&mut p_next, &p, &div);
        let inputs = vec![member("p", &p), member("div", &div)];
        let expected = vec![member("p", &p_next)];
        out.push(("jacobi".into(), fixture_json("jacobi", w, h, "", &inputs, &expected)));
    }

    // subtract_grad: subtract central-difference grad(p) from (u, v); interior only.
    // No set_bnd(vel) bundled — that is the set_bnd fixtures' job.
    {
        let (w, h) = (6usize, 5usize);
        let mut u = seed_full(w, h, |i, j| sinu(i, j, 0.35, 0.45, 0.2));
        let mut v = seed_full(w, h, |i, j| sinu(i, j, 0.55, 0.25, 1.3));
        let p = seed_full(w, h, |i, j| sinu(i, j, 0.5, 0.5, 0.6));
        let u_in = u.clone();
        let v_in = v.clone();
        subtract_grad(&mut u, &mut v, &p);
        let inputs = vec![member("u", &u_in), member("v", &v_in), member("p", &p)];
        let expected = vec![member("u", &u), member("v", &v)];
        out.push(("subtract_grad".into(), fixture_json("subtract_grad", w, h, "", &inputs, &expected)));
    }

    // set_bnd: writes the full border (edges + corners). One fixture per kind; the
    // GPU runs this as two passes (edges then corners) but the CPU result is the same.
    for (kind, name) in [(Bnd::Scalar, "set_bnd_scalar"), (Bnd::VelX, "set_bnd_velx"), (Bnd::VelY, "set_bnd_vely")] {
        let (w, h) = (6usize, 5usize);
        // Interior-only seed: border starts zero and is computed by the kernel, so the
        // expected border values come solely from set_bnd (the thing under test).
        let g_in = seed_interior(w, h, |i, j| 1.0 + sinu(i, j, 0.45, 0.65, 0.4));
        let mut g = g_in.clone();
        set_bnd(kind, &mut g);
        let params = format!("\"kind\":\"{}\"", match kind {
            Bnd::Scalar => "scalar",
            Bnd::VelX => "velx",
            Bnd::VelY => "vely",
        });
        let inputs = vec![member("field", &g_in)];
        let expected = vec![member("field", &g)];
        out.push((name.into(), fixture_json(name, w, h, &params, &inputs, &expected)));
    }

    // composed: Fluid2D::step x N on a small grid with modest iters. The full
    // add-forces -> project -> advect -> project pipeline. `s` is NOT moved by the
    // 2D step (only Fluid25D mixes it), so expected s == input s.
    {
        let (w, h) = (8usize, 8usize);
        let iters = 20usize;
        let steps = 4usize;
        let dt = 0.1f32;
        let mut fluid = Fluid2D::new(w, h, iters);
        // Seed velocity + scalar interior from sinusoids (zero border; step's set_bnd
        // writes the border each iteration).
        fluid.u = seed_interior(w, h, |i, j| 0.5 * sinu(i, j, 0.4, 0.3, 0.0));
        fluid.v = seed_interior(w, h, |i, j| 0.5 * sinu(i, j, 0.3, 0.4, 1.0));
        fluid.s = seed_interior(w, h, |i, j| 1.0 + sinu(i, j, 0.6, 0.6, 2.0));
        // Scripted continuous force (constant across the N steps).
        let force_x = seed_interior(w, h, |i, j| 0.3 * sinu(i, j, 0.5, 0.2, 0.5));
        let force_y = seed_interior(w, h, |i, j| 0.3 * sinu(i, j, 0.2, 0.5, 1.5));

        let u_in = fluid.u.clone();
        let v_in = fluid.v.clone();
        let s_in = fluid.s.clone();

        for _ in 0..steps {
            fluid.step(dt, &force_x, &force_y);
        }

        let params = format!("\"dt\":{},\"iters\":{},\"steps\":{}", fnum(dt), iters, steps);
        let inputs = vec![
            member("u", &u_in),
            member("v", &v_in),
            member("s", &s_in),
            member("force_x", &force_x),
            member("force_y", &force_y),
        ];
        let expected = vec![member("u", &fluid.u), member("v", &fluid.v), member("s", &fluid.s)];
        out.push(("composed_step".into(), fixture_json("composed_step", w, h, &params, &inputs, &expected)));
    }

    out
}

// ---- Output ------------------------------------------------------------------

fn fixtures_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = .../crates/vs-core ; fixtures live at repo-root/tests/fixtures/fluid.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.join("../../tests/fixtures/fluid")
}

fn main() -> std::io::Result<()> {
    let fixtures = build_fixtures();
    let dir = fixtures_dir();
    std::fs::create_dir_all(&dir)?;

    // Write one file per fixture (the GPU tests glob tests/fixtures/fluid/*.json).
    for (name, json) in &fixtures {
        let path = dir.join(format!("{}.json", name));
        std::fs::write(&path, format!("{}\n", json))?;
        eprintln!("wrote {}", path.display());
    }

    // Emit the combined object to stdout so the verify command sees valid JSON.
    let mut combined = String::from("{");
    for (k, (name, json)) in fixtures.iter().enumerate() {
        if k > 0 {
            combined.push(',');
        }
        let _ = write!(combined, "\"{}\":{}", name, json);
    }
    combined.push('}');
    println!("{}", combined);
    Ok(())
}
