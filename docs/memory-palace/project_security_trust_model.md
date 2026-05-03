---
name: Trust + prompt-injection model
description: How mergedog defines "trusted commit" and what it scrubs before letting an LLM see PR content; rationale for the squash+sidecar design.
type: project
originSessionId: 99555959-243d-47e3-a3cf-e0d88b509da7
---
**Trusted commit** = (a) GitHub says ezyang approved that SHA, or
(b) `[MERGEDOG]`-tagged commit produced by mergedog itself, recorded
in the local sqlite trust DB. Both halves needed.

**Threat model:** approval = "everything visible in GitHub UI is
clean." Invisible-to-UI content (intermediate commit messages, etc.)
is untrusted.

**Mitigations:**
1. Agent worktree is a single squashed commit (PR title + sanitized
   description). Agent never sees intermediate commits.
2. PR comments (UI-visible) can be exposed via a sidecar file.
3. Sanitizer (`mergedog/sanitize.py`) scrubs invisible/control content;
   tests on it are load-bearing.
4. Prompt instructs agent not to peek at git history.
5. Harness (not agent) replays the agent's commit onto the real
   sequence and pushes.

**How to apply:**
- New code paths that surface PR-author content to the agent need a
  sanitization story. Default: don't expose.
- A path letting post-approval pushes influence mergedog = security
  regression. Flag it.
- Sanitizer changes need new tests.
