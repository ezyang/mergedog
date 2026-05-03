---
name: Smallest implementation preference
description: ezyang explicitly optimizes for the smallest possible code change; avoid abstractions, frameworks, and gratuitous timeouts even when "more robust" patterns exist.
type: feedback
originSessionId: 99555959-243d-47e3-a3cf-e0d88b509da7
---
ezyang's direct preference: "smallest code implementation" / "bare
bones" / "no need to get fancy."

**Why:** personal tool, one user, prototyped fast. Over-engineering
costs real reading/maintenance time; under-engineering is cheap to
fix later.

**How to apply:**
- Propose the smaller of two implementations unless correctness
  requires bigger.
- No config knobs, plugin layers, or generic interfaces for
  hypothetical needs.
- Justify any timeout/retry/poll number you write; default short.
- Cleanup/refactor passes should *reduce* code, not relocate it.
- Skip error handling for paths that can't fail in practice — one
  user will see the trace and fix it.

Security goals (sanitizer, trust DB, idempotency) override this
default — never the other way.
