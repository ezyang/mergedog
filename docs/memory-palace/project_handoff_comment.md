---
name: Handoff comment requirements
description: What the "mergedog handoff" GitHub comment must contain: bot-authored marker, commit hash, summary of every claude session and verdict.
type: project
originSessionId: 99555959-243d-47e3-a3cf-e0d88b509da7
---
"mergedog handoff" comment posted when CI is green and the PR is
ready for human `@pytorchbot merge`. Required content:

- Explicit "bot-authored" line + machine-parseable metadata (e.g.
  hidden HTML comment with structured fields).
- Commit SHA mergedog landed on (mergedog has no version number).
- Per-claude-session block: which CI failure prompted it, "before"
  SHA, verdict (fix-committed vs judged-spurious), failing jobs
  considered. Human must be able to audit why claude judged failures
  unrelated.
- Idempotent — see `project_idempotency.md`.

**How to apply:** don't trim the per-session block for length;
forensic detail is the point. Fix commits should also state the
(abridged) CI failure they address, so the handoff can cite by SHA.
Nontrivial merge resolutions should be reflected in both the
`[MERGEDOG] Merge main` commit title and the handoff.
