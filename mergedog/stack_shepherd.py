"""Shepherd a whole ghstack stack as a single foreground process.

For each member of the stack we mirror the per-PR ghstack setup that
:mod:`mergedog.shepherd` does for the single-PR case: validate the PR
is open and not a draft, seed trust from reviews, fetch /head and
/orig from origin, and stand up a worktree at /orig.

The reactive scheduler -- fix bottom-up via ``ghstack submit
--no-stack``, propagate via full ``ghstack submit`` only when a
green-stable parent has stale children, gate ciflow/trunk on parent
trunk-green -- is still to come. ``run_stack`` exits after setup so
each step of the buildup stays a small reviewable commit.
"""
from __future__ import annotations

import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path

from mergedog import github, repo
from mergedog.log import die, log
from mergedog.paths import REPO_SSH_URL
from mergedog.shepherd import EXIT_PR_NOT_ACTIONABLE, MERGEDOG_LABEL
from mergedog.stack import StackMember, resolve_stack
from mergedog.state import TrustDB
from mergedog.trust_seed import seed_trust_from_reviews


@dataclass
class _MemberCtx:
    """Operational state for one stack member during a run.

    Most fields are transient scheduler bookkeeping that the
    forthcoming tick loop will mutate; only ``trust`` is persisted to
    disk (via its own ``save()``).
    """

    member: StackMember
    pr_data: dict
    trust: TrustDB
    worktree: Path
    self_pr: bool
    head_sha: str
    orig_sha: str
    last_status: str | None = None
    stable_observation: tuple[str, int] | None = None
    stable_since: float = 0.0
    empty_log_defers: int = 0
    fix_commits_pushed: int = 0
    trunk_applied: bool = False
    spurious_check_names: set[str] = field(default_factory=set)
    run_state_cache: dict = field(default_factory=dict)


def _validate_member(pr_data: dict) -> None:
    pr = pr_data.get("number")
    state = pr_data.get("state")
    if state != "OPEN":
        # Use the prune exit code so the mux removes us automatically
        # when one member of the stack is no longer actionable. The
        # operator can re-add a smaller stack manually.
        die(
            f"PR #{pr} is not open (state={state}); "
            f"refusing to run stack mode",
            code=EXIT_PR_NOT_ACTIONABLE,
        )
    if pr_data.get("isDraft"):
        die(f"PR #{pr} is a draft")


def _setup_member(
    member: StackMember,
    pr_data: dict,
    *,
    accept_divergence: bool,
    viewer: str,
) -> _MemberCtx:
    """Per-member analogue of the ghstack init block in ``_shepherd_body``.

    ``viewer`` is passed in (rather than re-queried) so a stack run
    only hits ``gh api user`` once.
    """
    pr = member.pr
    _validate_member(pr_data)

    trust = TrustDB.load_or_create(pr)
    trust.head_branch = member.head_ref
    trust.head_repo_clone_url = REPO_SSH_URL
    trust.save()

    self_pr = github.is_self_pr(pr_data, viewer)
    if self_pr:
        log(f"PR #{pr} authored by current user ({viewer}); skipping approval gate")
    seed_trust_from_reviews(
        trust, pr, pr_data, accept_divergence, self_pr=self_pr
    )

    head_sha = pr_data["headRefOid"]
    origin_head_sha = repo.fetch_ghstack_head(member.head_ref)
    if origin_head_sha != head_sha:
        die(
            f"PR #{pr}: origin's {member.head_ref} "
            f"({origin_head_sha[:12]}) differs from the SHA GitHub "
            f"reports for the PR ({head_sha[:12]}); refusing to act"
        )
    orig_sha = repo.fetch_ghstack_orig(member.head_ref)
    worktree = repo.ensure_worktree(pr, orig_sha)

    return _MemberCtx(
        member=member,
        pr_data=pr_data,
        trust=trust,
        worktree=worktree,
        self_pr=self_pr,
        head_sha=head_sha,
        orig_sha=orig_sha,
    )


def _sigterm_to_systemexit(signum, frame) -> None:  # type: ignore[no-untyped-def]
    sys.exit(128 + signum)


def run_stack(
    pr: int,
    *,
    rebase: bool = False,
    accept_divergence: bool = False,
    ignore_sev: bool = False,
    reassess: bool = False,
) -> None:
    repo.ensure_clone()
    repo.fetch_origin()

    members, pr_data_by_pr = resolve_stack(pr)
    log(f"resolved ghstack stack containing PR #{pr}: {len(members)} member(s)")
    for i, m in enumerate(members):
        log(f"  [{i}] PR #{m.pr}  head={m.head_ref}  orig={m.orig_ref}")

    # Tag every member up front so other operators / mergedogs see the
    # whole stack is owned, then arrange to remove every label on any
    # exit path (success, halt, SIGTERM from ``mux cancel``, ctrl-c).
    # Track which labels we actually applied so the cleanup is precise
    # if some adds 502'd partway through.
    labelled: list[int] = []
    signal.signal(signal.SIGTERM, _sigterm_to_systemexit)
    try:
        for m in members:
            try:
                github.add_label(m.pr, MERGEDOG_LABEL)
                labelled.append(m.pr)
            except Exception as e:
                log(
                    f"WARNING: failed to add {MERGEDOG_LABEL} to "
                    f"PR #{m.pr}: {e}"
                )

        viewer = github.viewer_login()
        contexts: list[_MemberCtx] = []
        for m in members:
            ctx = _setup_member(
                m,
                pr_data_by_pr[m.pr],
                accept_divergence=accept_divergence,
                viewer=viewer,
            )
            contexts.append(ctx)
            log(
                f"  PR #{ctx.member.pr}: /head={ctx.head_sha[:12]} "
                f"/orig={ctx.orig_sha[:12]} worktree={ctx.worktree}"
            )

        log(
            "stack-shepherd setup complete; scheduler not yet wired -- "
            "exiting"
        )
    finally:
        for pr_num in labelled:
            try:
                github.remove_label(pr_num, MERGEDOG_LABEL)
            except Exception as e:
                log(
                    f"WARNING: failed to remove {MERGEDOG_LABEL} from "
                    f"PR #{pr_num}: {e}"
                )
