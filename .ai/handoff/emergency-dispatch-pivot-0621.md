---
timestamp: 2026-06-21T00:00:00Z
branch: worktree-emergency-glider
session_focus: "Built an emergency-service game, but built it ON the bird flight engine. User clarified at the end: they want a TOTALLY DIFFERENT game that only reuses the neon-line AESTHETIC — not the bird, not the flight sim. PIVOT to top-down neon dispatch."
user_goal: "the same neon line aesthetic but a totally different game" — an emergency-service dispatch game, TOP-DOWN, reusing ONLY the neon render look (EKG lines + bloom + dark additive palette + beam beacons). NOT a flight sim, NOT the bird.
status: blocked

# THE PIVOT (read this first)
# The current emergency game is architecturally WRONG. It IS the bird flight sim (Bird3D glider,
# physics, ChaseCamera, mouse-fly) with dispatch mechanics + HUD bolted on. The user called that
# "weird shit added to my bird game." The next session must build a SEPARATE top-down game that
# imports ONLY the render layer. The GOOD news: the pure game logic is view-agnostic and reusable.

reusable_assets:
  # all on branch worktree-emergency-glider (worktree may be removed — fetch from the branch)
  - path: src/host/game/sim.ts
    note: "KEEP. Pure-logic GameSim — incidents/relay/race/money/rival/transport-deadline. View-AGNOSTIC, no GPU/DOM. Player position is fed in as (x,z); for top-down that's a cursor/selected-unit, not a bird. Reuse nearly as-is."
  - path: tests/unit/sim.test.ts
    note: "KEEP. 9 tests incl. bounded-queue leak guard. Logic is solid."
  - path: tests/unit/balance.probe.test.ts
    note: "KEEP. Simulated-playtest balance ladder (idle 0% / decent ~38% / great ~88% win)."
  - path: src/host/gpu/beacons.ts + src/host/shaders/beacons.wgsl
    note: "MAYBE reuse — instanced colored light-beams. Works in 3D; for a true top-down view you may want 2D neon primitives instead."
  - path: "render aesthetic primitives (the actual 'vector system')"
    note: "TerrainEKG (gpu/terrain.ts — the signature stacked neon lines), Bloom (gpu/bloom.ts), SKY dark palette + additive blend. THIS is what the user means by 'vector system'. Reuse for the neon look."

scrap_or_rework:
  - path: src/host/emergency-main.ts
    note: "SCRAP/REPLACE. It is bird-main.ts + game overlays — imports Bird3D, flight physics, ChaseCamera, mouse-fly. The whole 'fly the glider to claim' player loop is the wrong form."
  - path: index-emergency.html
    note: "Keep the file, rewire <script> to a new top-down entry (e.g. dispatch-main.ts)."

files:
  created:
    - path: src/host/game/sim.ts
      purpose: "Pure game logic (REUSE)"
      status: complete
    - path: src/host/gpu/beacons.ts + src/host/shaders/beacons.wgsl
      purpose: "Instanced beam renderer (maybe reuse)"
      status: complete
    - path: src/host/emergency-main.ts
      purpose: "WRONG-FORM entry (bird flight + dispatch). To be replaced."
      status: needs_review
    - path: index-emergency.html
      purpose: "Entry html (rewire script tag)"
      status: needs_review
    - path: .ai/plan/emergency-glider/emergency-glider-design.md
      purpose: "Full design doc — game-design decisions still valid, the FORM (fly) is not"
      status: complete
    - path: .ai/plan/emergency-glider/review-2026-06-21.md
      purpose: "7-lens adversarial review record"
      status: complete
  modified:
    - path: src/host/bird-main.ts
      changes: "Restored byte-identical to da5de81 (the user's ORIGINAL bird game). DO NOT TOUCH IT. index-bird.html is the untouched bird game."
      status: complete

# Game-design decisions that SURVIVE the pivot (the theme + mechanics are good; only the form changes):
locked_decisions:
  - "Emergency service in a solarpunk post-car city; hexagons with central hills; competing providers."
  - "Dispatch board is the rival surface. FIRST ON SCENE wins a call, regardless of who dispatched. You can read rival crew count -> infer overextension -> raid their undefended area."
  - "RELAY: air leg (glider, fast) + ground leg (bike-wagon, hauls patient to hospital). Two crews per nasty call = why you hire / the finite-crew tension."
  - "ONE money. Reputation per-hex (future). No XP/time grind."
  - "Own the hill = first dibs on that hex's calls (future). Loseable."
  - "Persistent, trainable, talented team (future)."
  - "SPLIT PAYOUT: 35% on claim (the race = player skill), 65% on hospital delivery."
  - "TRANSPORT DEADLINE (deliverTtl): a claimed patient not delivered in time is LOST — bounds the queue + makes over-claiming cost you."
  - "Rival = kinematic ghost AI with finite glider units + wagons."
  - "ARC: pilot -> split-attention -> dispatcher. NOTE: top-down dispatch IS the dispatcher endgame — the pivot aligns with the original vision's destination."

next_actions:
  - task: "Confirm the exact top-down form with the user (see open_questions), then brainstorm/spec the top-down dispatch game in a fresh design doc. Do NOT reuse Bird3D / flight / chase cam."
    priority: high
    context: "User answered the form question as 'top down neon dispatch' (their note) then asked for this handoff before details were nailed."
    acceptance: "A spec for a top-down neon dispatch game exists; user approves before code."
  - task: "Stand up the neon TOP-DOWN render: an orthographic/overhead camera over the terrain (or a flat hex grid) using TerrainEKG-style neon lines + Bloom + the dark additive palette. Prove the LOOK first."
    priority: high
    context: "The 'vector system' the user wants is this aesthetic. Get a top-down neon view rendering before any gameplay."
    acceptance: "A top-down neon scene renders at 60fps; reads as the same visual universe as index-bird.html; no Bird3D import."
  - task: "Wire GameSim (src/host/game/sim.ts) into the top-down view: calls = beacons on the map, you dispatch units by clicking (not flying), units trace glowing paths, relay->pay, rival competes, scoreboard."
    priority: medium
    context: "The logic already works + is tested. The new work is the VIEW + the dispatch INPUT (click/command, not mouse-fly)."
    acceptance: "Playable top-down dispatch loop; sim tests still green."
  - task: "New entry: dispatch-main.ts + point index-emergency.html at it. Delete/retire emergency-main.ts. Keep index-bird.html + bird-main.ts UNTOUCHED."
    priority: medium
    context: "Clean separation; the bird game stays exactly as the user left it."
    acceptance: "Two independent games; bird-main.ts still byte-identical to da5de81."

open_questions:
  - "Top-down interaction model: do you click a CALL then assign a unit, or click a UNIT then a destination? (RTS-style vs board-style.)"
  - "Is there a literal hex GRID drawn, or just neon districts/positions on the existing terrain seen from above?"
  - "Do units move along roads/paths or straight lines? Any pathfinding, or kinematic straight-line like the current sim?"
  - "Any zoom-in to 'watch a run', or pure top-down always?"
  - "Keep the emergency-service theme (assumed yes — it was the user's whole pitch), just the FORM changes to top-down."

blockers:
  - issue: "Form is chosen (top-down neon dispatch) but the interaction model (how the player issues dispatch commands) and view specifics (hex grid? camera?) are unspecified."
    severity: high
    needs: "A short brainstorm/spec with the user on the top-down interaction + view, THEN build. Do not guess again."

conversation:
  user_tone: "Frustrated and blunt by end of session ('you fucking dolt'). Was patient during design, turned sharp when the deliverable was the wrong shape. Hates flailing and over-asking."
  user_preferences:
    - "Reuse the AESTHETIC (neon vector lines / bloom / dark palette), not the bird flight engine."
    - "The bird game (index-bird.html / bird-main.ts) is sacred — do NOT modify it."
    - "Wants a TOTALLY DIFFERENT game, not the bird-with-overlays."
    - "Act with certainty or ask ONE crisp question — don't gap-fill, don't over-question."
    - "Works in a git worktree; isolate before editing."
  important_constraints:
    - "DO NOT import or reuse Bird3D, the flight physics, ChaseCamera, or mouse-fly for the new game."
    - "DO NOT touch bird-main.ts or index-bird.html — the bird soaring demo must stay exactly as-is."
    - "Reuse ONLY: TerrainEKG neon-line render, Bloom, dark additive palette, (maybe) beacons.ts, and sim.ts game logic."
    - "Node is an nvm shell function in non-interactive shells — bypass with /opt/homebrew/bin/node; node_modules lives in the main checkout (symlink into worktrees)."
    - "WebGPU required (Chrome/Edge/Arc). The user hit a black screen testing — likely local browser/WebGPU or wrong dir; code renders fine headless."
---

## Notes

The original design (`.ai/plan/emergency-glider/emergency-glider-design.md`) and review are correct on
THEME and MECHANICS — keep them. They are wrong on FORM: they assumed the player pilots the bird glider.
The pivot to top-down dispatch actually *completes* the design's own stated arc ("pilot -> dispatcher");
the endgame view becomes the whole game.

Cleanest restart: new branch/worktree off the bird base; copy in `src/host/game/sim.ts` + its tests
(view-agnostic, already passing); build a top-down neon view from `TerrainEKG`/`Bloom`/palette; add a
click-to-dispatch input layer feeding `GameSim.update`. The flight code never enters the picture.

Branch `worktree-emergency-glider` holds all current code if you need to lift `sim.ts`/`beacons.ts`.
