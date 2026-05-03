---
name: Design pillars of mergedog
description: The four invariants (Adoption, Security, Autonomous, Synchronous) plus "smallest implementation" that govern every design decision in mergedog.
type: project
originSessionId: 99555959-243d-47e3-a3cf-e0d88b509da7
---
Pillars from `README.md`, in priority order: Adoption, Security,
Autonomous, Synchronous.

**Why each matters operationally:**
- Security ≻ ergonomics: post-approval contributor pushes are *actively
  harmful*. Default to security; explain any tradeoff.
- Autonomous ≠ no-HALT: prefer HALT-with-message over interactive
  prompts in the per-PR shepherd.
- Synchronous: per-PR process is foreground+sync internally even when
  the mux runs many in parallel.

Smallest-implementation is a complementary preference — see
`feedback_smallest_implementation.md`.
