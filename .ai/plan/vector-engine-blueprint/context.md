# Vector System — Engine Blueprint · Context & Provenance

One-line: the decision trail, research provenance, and key-source pointers behind [blueprint.md](file:///Users/god/projects/ai-jank/vector-system/.ai/plan/vector-engine-blueprint/blueprint.md).

Responsibilities:
- Record how each locked decision was reached (so a reviewer can re-open one knowingly).
- Point to the research run and its raw output for audit.
- Capture the session id for conversation revisit.

Session id: `4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5` · Date: 2026-06-08

---

## Decision trail (the forks, in order, with the answer that was chosen)

```
1  Objective              → Blueprint / design doc only (no code this round)
2  World model            → Shared engine, 3 separate game builds; cross-game = data handoffs, not live sim
3  Sim fidelity           → Full hierarchical fluid (later refined to moving-window full-3D + 2.5D fallback)
4  Compute/render API     → WebGPU (compute + render); Electron/Steam wrapper later
   4a clarification        → "Same renderer, behind-the-scenes varies" — renderer is the shared core; sims bespoke
   4b clarification        → Only Bird has fluid — fluid is a per-game subsystem, not a shared field primitive
   4c clarification        → "GPU does the math, WebGL renders" reconciled: WASM = default compute,
                            GPU compute = Bird fluid + aero only, GPU render = always (WebGPU does both)
5  Fluid reinterpretation → ACCEPTED: full-3D in a flock-following moving window + coarse 2.5D weather;
                            2.5D-everywhere is the documented fallback / M-series baseline
6  Shell scope            → Fold in the 3 important subsystems (tutorial, accessibility, perf-LOD); stub the rest
7  Engine thickness       → THIN spine — shared plumbing, bespoke kernels (declined the shared kernel library)
```

## Why the spine is "renderer + thin substrate," not a shared compute core

The three sims share no math, data structure, time-scale, or agent-coupling topology:
- Wind  = Eulerian vector fluid (Navier-Stokes), per-frame, agents weakly perturb it.
- Light = visibility/occlusion query against canopy geometry, event-recomputed, agents ARE the occluders.
- Demand = discrete job arrival/depletion on a sparse hex graph, agents deplete it (no PDE).

So `sample(pos,t)` unifies the READ, never the UPDATE. The genuinely shared set is the vector renderer, the GPU-resource convention/plumbing, the multi-rate scheduler, the terrain substrate, and the WorldState handoff format — plus thin read-interfaces (Field trait + containers, camera/input math).

## Research provenance

```
Workflow run        wf_28b6bd2c-b80   (task id wm7dw6166)
Shape               6 parallel finders → 2 adversarial critics (feasibility skeptic + completeness critic)
Cost                8 agents · ~534k subagent tokens · 103 tool uses · ~42 min wall
Raw output (JSON)   /private/tmp/claude-501/-Users-god-projects-ai-jank-vector-system/4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5/tasks/wm7dw6166.output
Script              /Users/god/.claude/projects/-Users-god-projects-ai-jank-vector-system/4f2f34f8-ceb1-4a8e-ad37-2dfe8d0681f5/workflows/scripts/vector-engine-blueprint-research-wf_28b6bd2c-b80.js
```

## What the adversarial pass changed (so reviewers know the doc is post-critique)

- `[FALSIFIED]` the "3060 is 15–30× an old GPU → 128³ comfortable" reasoning — stencil fluid is bandwidth-bound (~4× real gain, 128³ is 4× cells → flat headroom).
- Corrected base M-series to ~150 GB/s **unified** memory (CPU+GPU share the bus) → full-3D domain-wide not viable at 60fps.
- Moved Bird aero to a GPU compute pass (async readback would feed F=ma 33–50 ms stale).
- Resolved 6 cross-finding classification contradictions (ECS, Field sampler, Hexland demand, input/camera, fp16-vs-f32, renderer edges).
- Surfaced 10 missing subsystems; promoted tutorialization, accessibility, and the perf-LOD controller to first-class.

## Constraints in force (from user global instructions)

- All working files live under `.ai/`. No `.md`/`.txt` outside `.ai/` unless explicitly requested.
- Tab-aligned columns, not piped markdown tables. Paths as `file://` links. No personality content in work files.
- Burden of proof on complexity; every shared component names the specific problem it solves.
