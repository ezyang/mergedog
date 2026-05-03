---
name: Mux / TUI UX preferences
description: Concrete UX rules ezyang has stated for the multi-shepherd mux: textual TUI, keyboard-first, no popups, generous space for the live log line.
type: project
originSessionId: 99555959-243d-47e3-a3cf-e0d88b509da7
---
**Provisional.** ezyang flagged 2026-05-03 that he's not happy with
the current mux UI but hasn't yet articulated why. Treat the rules
below as descriptive (what's there now) rather than prescriptive
(what should stay). Expect changes.

Stack: textual TUI, top display + bottom entry, "as bare bones as
possible." It's a status pane, not an app.

- Ctrl-C quits (don't catch SIGINT).
- No mouse capture (override textual's default — preserve copy/paste).
- No "started"/toast popups; just add the row.
- Auto-redraw on updates.
- One row per PR: PR number, status, last log line. Worktree column
  was dropped — give that width to the log.
- Auto-prune: when a PR is no longer open, drop the row and clean
  filesystem state.
- Mux input port accepts PR numbers same as the CLI (so old CLI runs
  can be Ctrl-C'd and re-fed into the mux).
- Per-PR HALT only puts that row in error; mux + other shepherds keep
  running.

**How to apply:** new UI elements justify their pixels; default answer
to "what goes here?" is "more log." No modals, confirmations, or
tooltips.
