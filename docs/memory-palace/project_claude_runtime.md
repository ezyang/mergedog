---
name: Claude-as-subprocess runtime preferences
description: Rules for how mergedog invokes the claude CLI and renders its output: not fast mode, stream tools, log filtering, path relativization.
type: project
originSessionId: 99555959-243d-47e3-a3cf-e0d88b509da7
---
mergedog shells into the `claude` CLI for agent steps.

- **Not fast mode.** All async, slow is OK; capability matters more.
- **Stream tool calls + assistant text.** Mux logs are the only signal
  ezyang has on a running shepherd.
- **Drop tool results from the log.** Too noisy.
- **Render newlines as newlines** (not literal `\n`).
- **Relativize paths** to worktree root.
- **No working PyTorch install.** Prompt must tell the agent it cannot
  run pytorch tests locally — fix CI from logs, not by reproducing.

**How to apply:** don't add a fast-mode toggle. Preserve the three
formatter rules (no results, newlines, relative paths) on any
refactor. CI failures requiring a local build = HALT, not "teach the
agent to build pytorch."
