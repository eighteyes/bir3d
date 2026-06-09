# Vector System — Engine Blueprint

One-line: the shared engine that drives three separate WASM/WebGPU systems-games (Bird, Trees, Hexland) under one neon-vector aesthetic.

Document scope / responsibilities:
- Define the **module boundary** — what is shared engine vs bespoke per-game.
- Pin the **physical form** of the engine and the **WASM↔JS↔WebGPU** contract.
- Establish a **single unified frame budget** with honest, epistemically-tagged numbers.
- Name the **descopes** viability depends on, and the **profile-first** work that must precede any build.
- This is a **blueprint only**. No code is produced this round.

Status: DRAFT for review · 2026-06-08 · session `4f2f34f8` · see [context.md](file:///Users/god/projects/ai-jank/vector-system/.ai/plan/vector-engine-blueprint/context.md) for the decision trail and research provenance.

---

## 0. How to read this — epistemic tags

Every performance number carries one tag. **Do not treat an `[ESTIMATED]` number as a committed budget until it is `[MEASURED]`** (see §8, Validate-first).

```
[MEASURED]    profiled on real target hardware              — trust as budget
[DERIVED]     computed from a measured quantity + model     — trust the model, verify the inputs
[ESTIMATED]   hand-derived / midpoint / rule-of-thumb       — a hypothesis, not a budget
[FALSIFIED]   an earlier claim the adversarial pass refuted — do not build on it
```

**Honest headline (must survive into every summary of this doc):**
The engine is buildable at 60fps **only after the named descopes below.** Full-3D **domain-wide** fluid is **`[FALSIFIED]`** at 60fps — and worst on Apple silicon, where the base M-series has a ~150 GB/s **unified** bus shared by CPU and GPU. The viable wind model is a flock-following **moving window + a coarse 2.5D weather field**; the window's *internal* fidelity is the knob — **full-3D-in-window is reach** (discrete-GPU only, ~3–4 ms, borderline), and **2.5D stacked layers in the window is the realistic ship baseline** on both platforms (and mandatory on M-series). See the §6.1 fidelity ladder.

---

## 1. Thesis & physical form

**Thesis.** Three sims that share no math, no data structure, and no time-scale cannot share a compute core. They share an *aesthetic* and a *substrate*. So the engine is a **shared renderer + a thin substrate**; every simulation is bespoke and lives in its game.

The "Field primitive" unifies the **read** (`sample(pos,t)`), never the **update**. Wind is an Eulerian fluid; light is a visibility/occlusion query; demand is discrete jobs on a graph. Treating the Field as a shared *solver* is the central error this blueprint forbids.

**Physical form (the engine is three artifacts, consumed by three game crates):**

```
vs-core      Rust crate    scheduler, world/terrain substrate, WorldState I/O,
                           Field READ trait + grid containers, agent host glue,
                           the GPU-resource CONVENTION (not the kernels)
vs-shaders   WGSL library  the vector renderer (ribbon + segment-SDF + bloom),
                           MSDF text, shared compute plumbing (ping-pong, indirect)
vs-host      JS/TS         owns the GPUDevice, swapchain, pipelines; replays the
                           command descriptors WASM emits; runs the readback ring

bird / trees / hexland     three SEPARATE game crates, each its own build,
                           each linking vs-core + vs-shaders + vs-host
```

This resolves the open "one shell vs three apps" question: **three apps sharing crates.** `T`-thin was chosen — the shared GPU **plumbing** is in; the advect/project **kernels are Bird's**, not the engine's.

**Compute split (corrected from the doc's "WASM computes, WebGL renders"):**

```
WASM (CPU)       scheduler, game logic, AI/flock intent, economy, tree growth, pathfinding
GPU compute      Bird's fluid solver  AND  Bird's aero force-integration (see §6.1)
GPU render       always, all games — the shared vector renderer
```

---

## 2. Locked decisions (settled ground)

```
Deliverable      Blueprint / design doc only — no code this round
Architecture     One shared engine; three SEPARATE game builds
Cross-game        Data handoffs via the WorldState format — load-time only, never a live shared sim
The shared core   Renderer + GPU-resource convention + scheduler + terrain substrate + WorldState format
Compute split     WASM = default compute · GPU compute = Bird fluid + Bird aero · GPU render = always
Fluid             Moving-window solver + coarse 2.5D weather; window fidelity = full-3D (reach, discrete-only) → 2.5D-stacked (baseline) — §6.1 ladder
API               WebGPU for both render (all games) and compute (Bird) · Electron/Steam wrapper later
Players           Single-player throughout (AI competitors in Hexland; flocks are sim, not netcode)
Engine thickness  THIN spine — shared plumbing, bespoke kernels
Shell scope       Fold in the three important subsystems (tutorial, accessibility, perf-LOD); stub the rest
```

---

## 3. The unified frame budget — the spine

The #1 feasibility finding: **no unified budget existed** — each subsystem was sized against its own bandwidth assumption, and they do not share the same pool. This section is the correction, and it is the spine of the whole blueprint.

**Frame cost model:**
`frame_ms = max(GPU_timeline, CPU_timeline) + contention_derate`
NOT a sum — *unless* a synchronous GPU readback sneaks into the loop, which collapses pipelining into a sum. That readback is **forbidden** (hard rule, §4.2). `contention_derate ≈ 15–20%` on discrete; **on unified memory the bus IS the contention** — CPU traffic and GPU stencil/fill draw from one ~150 GB/s pool, so model M-series bandwidth as a single shared budget, not two timelines.

**Target: 1440p (2560×1440), 60fps = 16.6 ms.** Two platform columns because they fail differently:

```
                         RTX 3060 class            base M-series (M-Pro)
                         discrete, ~360 GB/s        unified, ~150 GB/s (CPU+GPU share)
-----------------------  ------------------------   --------------------------------
Fluid 2.5D mov-window    ~1.5–2.5 ms [ESTIMATED]    ~3.5–6 ms [ESTIMATED] (~2.4× worse) [DERIVED]
Fluid full-3D mov-window ~3–4 ms    [ESTIMATED]     OVER BUDGET [ESTIMATED] — not the M-series baseline
Fluid full-3D DOMAIN     [FALSIFIED] at 60fps       [FALSIFIED] at 60fps
Vector render (worst)    ~2–4 ms    [ESTIMATED]*    bus-contended, re-measure (TBDR may help overdraw)
Bloom (HDR)              ~0.3–0.6 ms [ESTIMATED]**   content-dependent, re-measure
Terrain render (CDLOD)   <2 ms      [ESTIMATED]      <2 ms [ESTIMATED]
Aero + flock GPU pass    <0.5 ms    [ESTIMATED]      <0.5 ms [ESTIMATED]
-----------------------  ------------------------   --------------------------------
GPU total @ 2.5D base    ~6–9 ms → fits             bandwidth-bound; 2.5D is the SHIP baseline here
CPU (WASM sched+agents)   ≤3 ms [ESTIMATED]          contends the shared bus — derate harder
```
*render worst-case = mature Trees canopy (~10k curves) zoomed; hinges on the 400px avg-curve-length `[ESTIMATED]` that ALL render budgets scale with — profile it first.
**bloom is NOT constant (the finder claimed it was); HDR blended fill scales with bright content `[FALSIFIED as constant]`.

**Named descopes viability depends on (not optional):**
1. Full-3D fluid is **moving-window only** (~256–512 m bubble), never domain-wide.
2. On M-series, **2.5D stacked layers** is the baseline; full-3D is desktop-discrete reach.
3. **fp16** velocity/scalars + **f32** pressure (a ~2× bandwidth swing vs all-f32 — pinned in §6.1).
4. Bird aero runs **on the GPU** (no per-bird readback — §6.1).
5. A **perf-LOD controller** (§7.3) measures per-pass GPU time and auto-selects fallback rungs; the ladders are real only because this controller exists.

> If profiling (§8) breaks an `[ESTIMATED]` row, the controller is what keeps the frame on budget — but a broken fluid row means the M-series ships a lower rung, full stop. Do not paper over it.

---

## 4. The shared engine (truly-shared modules)

### 4.1 Vector renderer core — the aesthetic spine

The single strongest true-share. Every content source lowers to **one** stroke primitive; the renderer never knows about wind, trees, or hexes.

**Submission API (one primitive, batched into one/few indirect draws):**

```
submitStroke({
  kind:      'polyline' | 'bezier_quad' | 'bezier_cubic' | 'catmull_rom',
  points:    Float32Array,   // xy, or xyz for Bird's 3D (projected in the prepass)
  closed:    bool,           // hex cells, tree rings
  width_px:  f32,            // core half-width, screen px (resolution-independent)
  cap, join: enum,           // butt|round|square, miter|round|bevel
  color:     vec3<f16>,      // HDR linear, may exceed 1.0 to drive bloom
  intensity: f32,            // phosphor brightness → HDR target
  glow_px:   f32,            // soft falloff radius beyond core (in-shader)
  layer:u16, z:f32, flags:u32 // order, depth, dashed/animated-flow/pulse
})
```

**Pipeline:** compute **flatten** (curve → segments at a ~0.2 px screen-space deviation tolerance) → instanced **ribbon quads** → per-fragment **segment-distance SDF** (crisp AA core + in-shader glow) → separable **bloom** on an **rgba16f** HDR target → tone/phosphor composite.

**The doc's "analytic per-pixel SDF is likely faster" is `[FALSIFIED]` at this scale.** Under a wide-glow aesthetic every line is fat, so **overdraw × per-fragment eval-cost** dominates. Segment-distance ≈ 12 FLOP/frag vs cubic-distance ≈ 150–300 FLOP/frag (`[ESTIMATED]`, ~16× ratio) makes analytic ALU-bound at ~5× fewer curves. Tessellation wins; flatten cubics in the prepass, never evaluate cubic distance per pixel.

**Per-game configuration at the edges (so it stays truly-shared at the core, not flatly):**
```
Projection feed   Bird/Trees = 3D project-then-flatten   ·  Hexland = 2.5D direct
Ordering          Bird = depth-test (terrain occlusion)  ·  Hexland/Trees = painter layers
Overlay           Hexland MAY add a hex-overlay pass on the shared pipeline
```

**Watch:** overdraw is THE cost — profile fill-rate first, not ALU; clamp `glow_px` and miter length; rgba16f is required (rgba8 bands the glow on dark fields); 3D widths need perspective scaling; animate dash/flow in-shader, don't re-submit geometry.

**Fallback ladder:** full → single-mip bloom → raise flatten tolerance 0.2→0.5–1.0 px → clamp `glow_px` (the dominant cost) → cull sub-pixel/merge dense branch clusters (protects mature-canopy Trees) → drop bloom, keep in-shader core glow (the always-crisp floor).

### 4.2 GPU resource convention + the runtime loop

**Wiring choice: JS owns `GPUDevice`; WASM is the CPU brain that emits flat command descriptors** over a shared `ArrayBuffer`, replayed by JS once per frame in bulk. `wgpu`-compiled-to-WASM is **rejected** — on the web it tunnels web-sys→JS→Dawn anyway, adding a second resource-tracking layer and boundary chatter (`[DERIVED]`, unprofiled — see §8).

```
[requestAnimationFrame — JS]
  JS writes real_dt → shared header → calls wasm.tick(real_dt)
    WASM scheduler advances due domains (CPU agent logic only),
      writes: per-domain command list (dispatch/draw descriptors, uniform bytes,
              buffer ids) + interpolation alphas + camera matrix
  JS reads command region ONCE (bulk), builds one encoder:
      replay compute passes (fluid step, aero+flock integrate, growth) — ping-pong
      replay render passes (vector ribbons via indirect draw) → bloom → composite
      queue.submit()
  JS kicks async mapAsync on the readback ring (non-blocking); a prior frame's
      resolved map copies AGGREGATES into shared memory for WASM next tick (2–3 frames stale, by design)
```

**State placement:**
```
GPU buffers (authoritative)   fluid grids, bird particle SoA (pos/vel), tree light buffers,
                              hex flow fields, ALL indirect/draw args
WASM linear memory (auth.)    scheduler, flock/AI intent, tree branch-graph arenas + succession,
                              Hexland economy/dispatch, AI competitors, PRNG, camera, input
shared ArrayBuffer (frame)    command descriptors, uniforms, interp alphas, camera matrices,
                              async readback aggregates
```

**Hard rules:** (1) **no synchronous readback in the frame loop** — `mapAsync` is async-only; an in-frame await stalls hard (lint it). (2) Cache pipelines/bind-groups JS-side keyed by id; per-frame payload is uniforms + dispatch sizes + buffer ids only (Dawn validation is ~1–5 µs/call and not pipelined). (3) Shared = the **convention** (ping-pong, indirect dispatch/draw, readback ring, command replay) — **not** the kernels.

### 4.3 Multi-rate scheduler

**One scheduler core; not one ECS.** A master clock plus per-domain *affine* sim-clocks: `sim_dt = slope · fixed_step`, so time-dilation is just a different slope — not a special case. A due-queue drains all due ticks per frame under a `MAX_SUBSTEPS` clamp, then emits one interpolation alpha per domain.

```
Domain            fixed_step      slope            effect
Bird flight       1/120 s         1                120 Hz sim, 60 Hz render-interp
Bird wind field   per render fr.  1                60 Hz, internal substep on CFL
Hexland traffic   1 sim-minute    ~600             ~1 tick / 100 ms wall
Hexland economy   event-driven    —                ~1–4 Hz wall
Trees growth      1 sim-day       ~1e7  EVENT-SKIP  never iterate raw decades — jump to next event
```

**Critical:** master clock is **f64** — f32's 24-bit mantissa cannot hold decade sim-time (3.15e8 s) alongside a 1/120 s step (Bird's step rounds to zero). **Trees runs as a separate, data-coupled build**, not live inside Bird's clock. High-slope domains use **event-skip** ("sleep until next light/competition event"), never brute substepping. `MAX_SUBSTEPS` + lag-the-clock (visible as slow-motion, never a hang) is the spiral-of-death guard.

### 4.4 Terrain substrate

One **bake-time-immutable** heightfield in one **local ENU** frame (East/North/Up, meters, **float32**), exposed through one read-only sampling API, wrapped in a **CDLOD** quadtree. Truly shared; every consumer derives from it.

```
World frame    local ENU, meters, f32, per-world geodetic origin
World cap      ~30 km (f32 keeps <1 cm error to ~40 km) — FAIL LOUD beyond; spawn a new world, never rebase
Heightfield    SRTM 1-arc-sec (~30 m land) + GEBCO/SRTM30_PLUS (bathymetry), gdalwarp→ENU at bake
               R32F tiles 512 m / 256² samples, 5 mip LODs (2/4/8/16/32 m); ~25–49 finest tiles resident (~6–12 MB)
API            sampleHeight / sampleNormal / sampleSlope / raycastGround  (texture-fetch cheap)
Consumers      render mesh (geomorph) · hex tiler ((q,r)→(E,N)→H projection) · fluid lower boundary
```

**Highest-risk seam: dual-consumer LOD coherence.** If render LOD and fluid-boundary LOD desync, wind deforms over a ridge the eye can't see → uncanny aero. The rule **"sim reads the node the renderer drew," one shared `lod_epoch`** is non-negotiable; ship a debug overlay that color-codes any cell where render-LOD ≠ boundary-LOD. **Hex height** = `sampleHeight` at hex center at a *fixed* LOD from a **slope-aware aggregate** (min/mean/max over the cell footprint), not a single point-sample, or zoning/roundabout geometry lies about the ground.

### 4.5 WorldState handoff format — the cross-game tie

A **versioned** file (CBOR or flatbuffer), read at load / written at export. **Load-time only — never a live protocol.** This is the *only* thing the three games exchange.

```
WorldState {
  schema_version, world_id
  coord_frame { origin_lat_lon | abstract, world_size_m, up_axis, units:'meters' }
  terrain     { seed, noise_params{octaves,lacunarity,gain,base_freq}, heightfield_ref?, sea_level_m }
  weather_timeline {                          // THE ORACLE PAYLOAD (coarse; baked by Bird's fluid, or a standalone coarse weather pass when Bird hasn't run)
      t0_epoch, dt_s, n_samples,
      grid{nx,ny,nz_or_1,cell_m},
      samples[] { t_offset_s, wind_vec, pressure, temp, precip }  // downsampled hard; cap ~1–10 MiB
  }                                           // a Tree derives "storm from NW in ~2h" by scanning samples near its xy
  agents { events[] { t_offset_s, kind:'bird_feed'|'tree_seed'|'flock_roost'|…, pos, payload } }  // append-only log
  ecology { grid{nx,ny,cell_m}, seed_density[], canopy_height[], feed_zone_id[] }  // one shared raster all 3 sample
  provenance { written_by, engine_ver, wall_clock }
}
```

**Cross-game links realized as pure data:** tree-as-oracle (bird queries a tree's baked `weather_timeline`) · bird feeding → `ecology.seed_density` → Trees spawns seedlings next load · Hexland renders `canopy_height` + events as ambient substrate, never stepping the bird/tree sim.

**Pinned, bit-exact contract (determinism trap).** Terrain is a **seed**, not a bitmap — so all three independently-built binaries on two hardware targets **must** regenerate identical terrain. Pin the **noise library version** inside `terrain.noise_params` and treat it as part of `schema_version`. Agree **one** `ecology.grid.cell_m` across games or seed-scatter/canopy reads misalign. (Cross-platform fBm bit-equality is a `[ESTIMATED]`/unverified assumption — see §8.)

---

## 5. Shared interfaces (bespoke bodies)

Thin contracts whose implementations are per-game. Listing these as "shared" without this caveat is the oversell the blueprint guards against.

```
Field READ trait      sample(pos,t)->T ; gradient(pos,t)->T          — signature only
Grid CONTAINER family dense voxel (wind) · sparse brick (light)      — TWO containers, not three
                      NO shared interpolator (trilinear ≠ hex-bary ≠ occlusion-lookup are bespoke)
                      NO shared solver kernel library (advect/project is Bird-only math)
                      Hexland demand does NOT implement Field — it's discrete graph state
Camera / input        shared projection/view MATH + raw event plumbing
                      controllers + key mappings are PER-GAME (3D free-fly ≠ hex city ≠ fixed look-up)
```

---

## 6. Per-game subsystems (the "behind the scenes varies")

### 6.1 Bird — earns detail (it sets the budget and exercises the shared plumbing hardest)

**Fluid (the fidelity target).** Two-level nested solver: a world-spanning **coarse 2.5D weather grid** (~256² × 8 layers, ticked 0.5–2 Hz — *this is the cross-game weather format*) feeds one-way Dirichlet boundary conditions into a **fine moving window** (~256–512 m bubble that follows the flock, solved per-frame). Semi-Lagrangian (Stam) advection for unconditional stability, **paired with vorticity confinement / additive curl-noise** to claw back the thermals/shear/eddies Stam's dissipation eats — without this the air feels like soup and the read-the-wind skill loop dies. **Multigrid** pressure projection (the biggest perf lever, but `[DERIVED]` ~2–4× not 5–10× on a masked moving domain; keep red-black Gauss-Seidel as the correctness fallback). **Precision pinned: fp16 velocity/scalars + f32 pressure.**

```
Fidelity ladder (perf-LOD controller selects the rung):
  R0 reach    full-3D moving window (64³–96×96×48) + multigrid + vort. confinement
              FEEL volumetric wind, rotor zones, spiral-able thermals · ~3–4 ms [ESTIMATED] · discrete-GPU only
  R1 BASELINE 2.5D stacked layers (8–16 horiz 2D Stam + vertical buoyancy) in the window + coarse 2.5D weather
              FEEL ridge lift, thermals, valley eddies, layered shear · ~1.5–2.5 ms [ESTIMATED] · the ship target
  R2 coarser  4–6 layers, ~6 m cells, fewer multigrid passes · ~0.8–1.2 ms [ESTIMATED]
  R3 bubble   tiny 2.5D Stam bubble (32²) only around the bird + analytic curl-noise elsewhere · ~0.3–0.6 ms
  R4 floor    pure terrain-modulated curl-noise, scripted thermal columns, no solver · <0.15 ms · runs anywhere
```

**Aero (corrected — runs on the GPU).** If WASM read GPU wind via the async ring it would be 33–50 ms stale and the bird would react to old air. So **aero force-integration is a GPU compute pass** (wind + gravity + lift(AoA) + drag(v²), F=ma, semi-implicit Euler), sampling the wind buffer **in-shader**. WASM keeps only high-level **flock intent**. **Flock** is GPU-resident; bird-bird drafting/wakes are **analytic vortex particles** superposed on neighbors' samples — **never injected into the grid** (a 3–4 m cell can't resolve a 1 m wingtip vortex; conflating these produces grid noise, not drafting).

**Consumes:** renderer, terrain sampling (fluid lower boundary), scheduler (120 Hz domain), WorldState (writes `weather_timeline`).

### 6.2 Trees — bespoke inventory

**Owns:** L-system / space-colonization growth geometry; **light = an occlusion/visibility query** (sun-view shadow or sparse-brick voxel occlusion, recomputed on growth/seasonal *events*, not per-frame — **not** a diffusion field); succession state machine; high-slope time-dilation via event-skip. **Consumes:** renderer, terrain sampling, scheduler (decade domain, separate build), WorldState (writes `ecology.seed_density`/`canopy_height`; bakes/reads `weather_timeline` as oracle). **Progression legible in-world:** a seedling growing into a canopy giant that shades its rivals — visible, not numeric.

### 6.3 Hexland — bespoke inventory

**Owns:** flat-top axial (q,r) hex topology + zoning-from-geometry (vertices=commercial, between=residential, centers=rural); economic/market tick (CPU/WASM, ~1–4 Hz); ambulance dispatch + A* routing on the hex graph; market-positioning ("SEO for ambulances"). **Demand is discrete jobs on a graph — NOT a field**; do not rasterize it into a diffusion grid (that adds cost and destroys the economic semantics that *are* the game). A smooth traffic visual, if wanted, is a **render-only heat overlay** derived from discrete state, never feeding the economic tick. **Consumes:** renderer (+ optional hex-overlay pass), terrain sampling (slope-aware per-hex aggregate), scheduler (minute domain), WorldState (reads canopy/ecology as substrate). **Progression legible in-world watch:** show fleet route lines, owned/lit hexes, territory shading — if dominance leaks into a numbers panel, the philosophy breaks.

---

## 7. The player-facing shell

The compute spine "can simulate and draw three worlds but cannot be heard, learned, tuned, saved mid-session, or shipped." Per `S`-important, the three serious subsystems are first-class here; the rest are named, scoped stubs.

### 7.1 Tutorialization (important — per-game, with a shared affordance primitive)

The locked philosophy ("inhabit, not visualize"; "no numbers"; "not perfectly solvable") **manufactures** a teachability problem: how is reading invisible wind, decade-scale tree competition, or "SEO for ambulances" taught without a stats panel? **Diegetic teaching only** — the world demonstrates: exaggerated streamline ribbons that over-communicate the field (the doc's "stylized wind, more legible than literal"), a mentor/lead bird to follow, ghost trails of good lines, time-lapse onboarding for Trees, a first contract that hand-holds dispatch. Per-game content; a **shared "hint/affordance overlay" stroke primitive** is the only engine-level piece.

### 7.2 Accessibility (important — engine-level, cross-cuts the renderer)

Bloom-heavy, pulsing neon on dark is a direct **photosensitivity/seizure** risk; hue-only line coding fails colorblind users; HDR bloom is a brightness-exposure risk. Engine-level requirements: **flash/bloom-intensity caps** (honor the 3-flashes-per-second guidance), **colorblind-safe palettes** (encode by more than hue — dash/width/motion), **motion-reduction mode** (calm streamlines, reduce pulse), **brightness/HDR ceiling**, **input remapping**. This is both an ethical floor and a **Steam store-compliance** gate. It cross-cuts §4.1 (the bloom caps live in the renderer config).

### 7.3 Perf-LOD controller (important — engine-level; makes every fallback ladder real)

Every ladder in this doc assumes a runtime arbiter that nobody had specced. It is: **GPU timestamp queries** per pass → a **frame-budget arbiter** that compares the measured timeline against the 16.6 ms target → **rung selection with hysteresis** (a dead-band so it doesn't thrash between rungs frame to frame). This is the subsystem that turns "auto-downgrade when over budget" from a hope into a mechanism. It is the runtime half of §3.

### 7.4 Scoped stubs (named, deferred — design constraints noted now)

```
Audio              candidate SHARED Web Audio / WASM-DSP node graph; wind/thermal SONIFICATION is a
                   real "read the invisible field" channel for soaring, not decoration. Scope before Bird ships.
Mid-session save   distinct from WorldState (which is load/export-only). GPU-resident AUTHORITATIVE state
                   (fluid grid, flock, clocks) makes this hard — treat "serializable sim state" as a
                   design constraint NOW even if built later, or it becomes impossible to retrofit.
Packaging          Electron/Steam: COOP/COEP headers (SharedArrayBuffer), Steam SDK (achievements/cloud/
                   overlay), macOS signing/notarization, auto-update. Three apps sharing a Rust crate.
Asset pipeline     build-time TOOLING: terrain bake (gdalwarp→tiled R32F + content-hash), MSDF glyph atlas,
                   WGSL shader pack, manifest + cache-invalidation. The handoff references a content-hash
                   nothing yet bakes.
Debug-viz          shared overlay framework (GPU-resident opaque state is near-impossible to inspect without
                   it; §4.4's LOD-coherence overlay and §6.1's field viz are the first customers).
Hot-reload         live tuning of feel-constants (aero AoA/lift/drag curves, vort-confinement gain, thermal
                   strength, growth/light thresholds, economic pricing) without rebuild. Tunables serialization.
Determinism/replay CPU-side only (fixed dt + seeded PRNG); GPU fluid is cross-vendor non-deterministic —
                   record the aggregate readback STREAM, never promise bit-exact fluid replay. Scope WHICH
                   games need replay (open).
```

---

## 8. Validate-first — profile before building

Viability depends on descopes **and** on numbers that are currently `[ESTIMATED]` or were `[FALSIFIED]`. **No `[ESTIMATED]` number becomes a committed budget until `[MEASURED]` on a real target.** Profile in this priority order — earlier items gate later design:

```
1  Fluid bandwidth on BOTH targets, esp. M-series 150 GB/s unified — THE make-or-break.
   Build a 2.5D moving-window microbenchmark FIRST, before any game code. Decides R0-vs-R1 baseline per platform.
2  Renderer overdraw/fill at worst case (mature Trees canopy, zoomed). MEASURE avg on-screen curve length —
   the 400 px [ESTIMATED] that every render budget scales with. Confirm tessellation > analytic at real scale.
3  Multigrid speedup on a masked, moving domain — [DERIVED] 2–4×, not the textbook 5–10×. Have Gauss-Seidel ready.
4  Bloom HDR fill cost — content-dependent, [FALSIFIED as constant]. Measure at worst-case bright coverage.
5  WASM↔JS boundary cost — JS-owns-device vs wgpu-on-wasm is [DERIVED], unprofiled. Microbench before committing.
6  Cross-platform terrain determinism — bit-exact fBm across 3 binaries × 2 targets. A known desync trap; verify early.
7  M-series sustained vs burst clocks (thermal throttling) — the 16.6 ms budget must hold under sustained load.
```

**Falsified claims carried forward so they don't creep back in:** "3060 is 15–30× an 8800 for fluid" (it's ~4× on the bandwidth axis that actually binds, and 128³ is 4× the cells → headroom flat); "M-series 200–360 GB/s" (base M-Pro is ~150, unified); "bloom is constant cost"; "analytic SDF is faster than tessellation at scale."

---

## 9. Build sequencing (when greenlit — not this round)

```
1  vs-shaders renderer core + submitStroke   ← the shared spine; nothing renders without it
2  vs-core terrain substrate + sampling API   ← second-most-depended-on
3  vs-core scheduler (f64 clock, due-queue)
4  Fluid 2.5D moving-window MICROBENCH         ← the budget driver; gates the whole fluid design (per §8.1)
5  Bird vertical slice (fluid + GPU aero + flock) on the engine
6  Freeze the shared interfaces the slice exercised  ← extract, don't pre-abstract
7  Trees      8  Hexland                       ← each proves a new engine demand; grow the spine only as forced
```
Pin **two things early and version them**: the **renderer submission API** and the **WorldState format** (bit-exact contract). Everything else can change cheaply; these two are expensive to change once three games depend on them.

---

## 10. Risk register (consolidated high-severity)

```
CRITICAL  Fluid budget on M-series unified memory — 2.5D baseline is mandatory there; full-3D is discrete-only
CRITICAL  Stale aero — bird must integrate on the GPU; any per-bird readback collapses the pipeline
HIGH      No synchronous readback in the frame loop — lint-enforced; the whole budget model assumes it
HIGH      Dual-consumer LOD coherence (render vs fluid boundary) — shared lod_epoch, debug overlay mandatory
HIGH      Terrain determinism across 3 binaries × 2 targets — pin noise lib version into schema_version
HIGH      Spiral of death — MAX_SUBSTEPS clamp + lag-the-clock is not optional
MED       Overdraw fill is the render cost — clamp glow_px/miter; profile fill, not ALU
MED       Trees decade-time needs f64 clock + event-skip + separate build — brute substepping freezes the game
MED       Field/kernel oversell — keep the shared Field to read-trait + containers + plumbing; kernels are Bird's
```

---

## 11. Open questions (carried, not blocking)

- Light recompute cadence for Trees — "near-zero amortized" holds only if growth events are genuinely sparse.
- Does Hexland need a bespoke hex-overlay render pass, or does the shared pipeline suffice?
- Which games actually need deterministic replay?
- Audio: shared node graph vs per-game — scope before Bird ships.
- WorldState `weather_timeline` size over a decades horizon — the ~1–10 MiB cap is an aspiration; verify downsampling holds.
