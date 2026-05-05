"""Shepherd a whole ghstack stack as a single foreground process.

Stub for now: resolves the stack and logs the membership, then exits.
The reactive scheduler (fix bottom-up, propagate via full ``ghstack
submit`` only when a green-stable parent has stale children, gate
ciflow/trunk on parent trunk-green) is still to be written.
"""
from __future__ import annotations

from mergedog import repo
from mergedog.log import log
from mergedog.stack import resolve_stack


def run_stack(
    pr: int,
    *,
    rebase: bool = False,
    accept_divergence: bool = False,
    ignore_sev: bool = False,
    reassess: bool = False,
) -> None:
    """Entry point for ``mergedog stack <pr>``.

    Flag semantics mirror the single-PR shepherd; the scheduler will
    apply them per-member once it lands.
    """
    repo.ensure_clone()
    repo.fetch_origin()

    members, _pr_data = resolve_stack(pr)
    log(f"resolved ghstack stack containing PR #{pr}: {len(members)} member(s)")
    for i, m in enumerate(members):
        log(f"  [{i}] PR #{m.pr}  head={m.head_ref}  orig={m.orig_ref}")
    log(
        "stack-shepherd scheduler is not yet implemented; exiting after "
        "stack resolution"
    )
