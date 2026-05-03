---
name: Idempotency / restart-safety contract
description: mergedog must survive Ctrl-C at any point and resume cleanly without double-effects (no duplicate handoff comments, no spurious worktree thrash).
type: project
originSessionId: 99555959-243d-47e3-a3cf-e0d88b509da7
---
**Rule:** Ctrl-C anywhere → restart cleanly, no double-effects.
Restarts are *normal*, not exceptional.

**Past hits:**
- PR 169354: handoff comment almost got re-posted on restart.
- "You remove and recreate the worktree every time I restart" — reuse
  existing worktree at the right SHA.

**How to apply:**
- For any GitHub-mutating step (push/label/comment), ask: if I die
  after this and restart, what happens?
- Prefer "re-check GitHub state, then act" over "remember in local
  memory that we did X." GitHub is source of truth; local sqlite
  (`~/.mergedog/state.db`) is for trust assertions only.
