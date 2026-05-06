"""One-off interventions for known-transient CI failures.

Some CI failures are upstream hiccups, not bugs in the PR or trunk: the
GitHub GraphQL API 5xxs, a runner can't reach a mirror, etc. The right
remedy is to rerun the failed jobs, not to ask claude to "fix" anything.
This module keeps a small list of log patterns we recognize. The shepherd
matches each failing job's log against the list and, if a rule fires,
re-runs the failed jobs in that workflow run instead of falling through
to the claude fix path.

Adding a new intervention is intentionally a code change rather than a
config edit -- each one encodes a judgment that "this specific failure
shape is safe to retry without inspection."
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Intervention:
    name: str
    # Searched against the (already-trimmed) log excerpt the shepherd
    # passes to claude. The pattern should be specific enough that a
    # genuine PR bug can't accidentally match it.
    log_pattern: re.Pattern[str]


INTERVENTIONS: list[Intervention] = [
    Intervention(
        name="github graphql 5xx",
        # Hits like "Error fetching https://api.github.com/graphql HTTP
        # Error 504: Gateway Timeout" surfaced by trymerge's
        # ghstack-mergeability-check. Anchoring on the GraphQL URL keeps
        # us from retriggering on a 5xx that happened to come from
        # somewhere else (a user-facing API the PR is exercising, etc.).
        log_pattern=re.compile(
            r"https?://api\.github\.com/graphql.{0,200}HTTP Error 5\d\d",
            re.DOTALL,
        ),
    ),
]


def find_intervention(log: str) -> Intervention | None:
    """Return the first intervention whose pattern matches ``log``, else None."""
    for itv in INTERVENTIONS:
        if itv.log_pattern.search(log):
            return itv
    return None
