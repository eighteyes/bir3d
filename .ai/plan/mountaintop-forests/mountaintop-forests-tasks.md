# Tasks

- [x] Explore render pipeline + integration points
- [x] Confirm form (recursive lines) + color (neon green) with user
- [x] Set up isolated worktree (other agent on core)
- [x] Write `trees.wgsl` (transform + fog + neon-green glow)
- [x] Write `trees.ts` (windowed placement, L-system, batched buffer, draw)
- [x] Wire into `bird-main.ts` (import, construct, draw call, `window.__trees`)
- [x] Typecheck clean
- [x] Perf harness — FPS A/B, tree count, rebuild cost
- [x] Fix WGSL uniform/struct name collision (pipeline was silently invalid)
- [x] GPU-readback proof trees render green (offscreen PNG + pixel counts)
- [x] Re-measure perf with working shader
- [x] Plan artifacts + HUMAN_REVIEW
- [ ] Human review on live app (localhost) — eyeball forest in the flying scene
- [ ] Merge worktree → build/foundation (after human review)
