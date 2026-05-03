---
name: PyTorch CI specifics that mergedog encodes
description: PyTorch-specific CI behaviors mergedog has to know about — ciflow/trunk gating, action_required workflows, post-push race, pytorchbot merge handoff, allow-edits exception.
type: project
originSessionId: 99555959-243d-47e3-a3cf-e0d88b509da7
---
pytorch/pytorch's CI is bespoke. Generic "wait for required checks
green" logic does **not** work.

- **Two-stage:** wait for `pull` green → label `ciflow/trunk` → wait
  for trunk green → handoff.
- **Required-checks API is unreliable:** says "0 required checks"
  persistently. Enumerate all workflow runs for the head SHA instead.
  ezyang chose "do everything" over filtering to required-only.
- **40+ subjobs per run:** show done/all ratio, not a single flag.
- **`action_required` gates** on external-author PRs: approve via
  `gh api -X POST repos/pytorch/pytorch/actions/runs/<id>/approve`.
- **Post-push race:** after pushing, GitHub takes seconds to register
  new runs. CI may transiently look green because nothing triggered
  yet. Wait for new runs to appear before judging.
- **Terminal state = handoff, not merge.** Human triggers
  `@pytorchbot merge -i`; mergedog posts a "mergedog handoff" summary
  comment.
- **Fresh-enough merge heuristic:** if merge-base with main is recent,
  skip merging main (just churn). If old, merge main first; if it
  conflicts, hand to claude. Merge commit message states clean vs
  what-was-resolved.
- **Allow-edits-by-maintainers exception:** skip the HALT when the
  PR head ref is in pytorch/pytorch itself (ezyang has write access).

**How to apply:**
- New CI-aware step → consider post-push race + action_required.
- Don't rely on required-checks API. Don't hard-code workflow names
  (state+conclusion is stable, names rotate).
