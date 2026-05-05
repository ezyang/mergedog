"""Shepherd a whole ghstack stack as a single foreground process.

For each member of the stack we mirror the per-PR ghstack setup that
:mod:`mergedog.shepherd` does for the single-PR case: validate the PR
is open and not a draft, seed trust from reviews, and snapshot the
current /head and /orig SHAs from origin.

Stack mode uses a single shared worktree (rooted at the bottom PR's
number) for the whole run -- we navigate its HEAD to whichever
member's ``/orig`` we're operating on rather than carrying N
worktrees. Per tick we batch-fetch every member's ``/head`` and
``/orig`` from origin so staleness checks are local.

Scheduler v0.1: per tick, every member is trust-checked, has its
pending workflow runs approved, and has its CI status inspected. Then
a bottom-up scan picks the lowest member with status="failed" and
actionable logs and invokes claude on it; on a clean fix the commit
is folded into /orig and re-published with ``ghstack submit
--no-stack`` (so siblings don't get hit with fresh CI). Propagation
of a parent fix to children, ciflow/trunk gating, and handoff are
still to come.

Cross-module helpers from :mod:`mergedog.shepherd` are imported with
their leading underscores; we'll promote them in a follow-up cleanup
once the stack work has settled.
"""
from __future__ import annotations

import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from mergedog import claude as claude_mod
from mergedog import context as context_mod
from mergedog import github, repo
from mergedog.handoff import ClaudeSession, utc_now_iso
from mergedog.log import die, log
from mergedog.paths import REPO_SSH_URL, context_file
from mergedog.prompts import render_fix_prompt
from mergedog.shepherd import (
    APPROVAL_SETTLE_SEC,
    CI_STABILITY_WINDOW_SEC,
    EXIT_PR_NOT_ACTIONABLE,
    MAX_EMPTY_LOG_DEFERS,
    MAX_FIX_COMMITS,
    MERGEDOG_LABEL,
    POLL_INTERVAL_SEC,
    _apply_spurious_overrides,
    _approve_pending_runs,
    _failed_logs_are_content_free,
    _publish_ghstack_fix,
    _record_claude_session,
    _wait_for_no_active_sev,
)
from mergedog.stack import StackMember, resolve_stack
from mergedog.state import TrustDB
from mergedog.trust_seed import seed_trust_from_reviews


@dataclass
class _MemberCtx:
    """Operational state for one stack member during a run.

    Most fields are transient scheduler bookkeeping that the tick loop
    mutates; only ``trust`` is persisted to disk (via its own
    ``save()``).

    ``head_sha`` is GitHub's view (used for trust checks); ``orig_sha``
    is origin's view (used for staleness). They're refreshed on
    different cadences -- ``head_sha`` per tick from ``gh pr view``,
    ``orig_sha`` from the per-tick batch git fetch.
    """

    member: StackMember
    pr_data: dict
    trust: TrustDB
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
    only hits ``gh api user`` once. Does not touch any worktree --
    that's owned by ``run_stack`` at the stack level.
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

    return _MemberCtx(
        member=member,
        pr_data=pr_data,
        trust=trust,
        self_pr=self_pr,
        head_sha=head_sha,
        orig_sha=orig_sha,
    )


def _refresh_stack_refs(contexts: list[_MemberCtx]) -> None:
    """Batch-fetch every member's /head and /orig and update ctx.orig_sha.

    ``ctx.head_sha`` is left to ``_refresh_member_head`` to set from
    GitHub's API view (which is what trust uses); we still fetch
    /head here to keep the local origin remote-tracking branch in
    sync with the remote, since later git-side tooling
    (``ghstack checkout`` / ``ghstack submit``) reads from there.
    """
    pairs = [(ctx.member.head_ref, ctx.member.orig_ref) for ctx in contexts]
    refs = repo.fetch_stack_refs(pairs)
    for ctx in contexts:
        ctx.orig_sha = refs[ctx.member.orig_ref]


def _refresh_member_head(ctx: _MemberCtx) -> None:
    """Re-read GitHub's view of the PR head and trust-check it."""
    current = github.get_pr_head_sha(ctx.member.pr)
    if ctx.self_pr:
        ctx.trust.trust(current)
    if not ctx.trust.is_trusted(current):
        subject = github.get_commit_subject(current)
        die(
            f"PR #{ctx.member.pr} head moved to untrusted commit "
            f"{current[:12]}: {subject!r}. Manual intervention required."
        )
    ctx.head_sha = current


def _refresh_member_pr_data(ctx: _MemberCtx) -> None:
    """Re-fetch PR metadata so per-tick decisions see fresh labels/state.

    A fresh ``gh pr view`` per tick per member is moderately chatty but
    correct -- labels/state can change underneath us (a maintainer
    could close, draft, or unlabel) and we'd rather catch it on the
    next tick than churn on stale data.
    """
    ctx.pr_data = github.get_pr(ctx.member.pr)


def _refresh_context_for(ctx: _MemberCtx) -> tuple[Path, list[dict]]:
    """Build the per-PR context sidecar (analogue of ``_refresh_context_file``)."""
    pr = ctx.member.pr
    comments = github.get_pr_comments(pr)
    text = context_mod.render_context(
        pr=pr,
        url=ctx.pr_data.get("url", ""),
        title=ctx.pr_data.get("title", ""),
        body=ctx.pr_data.get("body", "") or "",
        comments=comments,
    )
    path = context_file(pr)
    context_mod.write_context_file(path, text)
    return path, comments


def _inspect_member(ctx: _MemberCtx) -> str:
    """Update ctx with current CI status; return ``passed``/``failed``/``pending``."""
    checks = github.get_pr_checks_all(ctx.member.pr)
    effective = _apply_spurious_overrides(checks, ctx.spurious_check_names)
    status = github.evaluate_checks(effective)
    done = sum(1 for c in checks if c.get("bucket") not in {"pending", None})
    summary = f"{status} ({done}/{len(checks)} done)"
    if summary != ctx.last_status:
        log(f"PR #{ctx.member.pr} CI -> {summary}")
        ctx.last_status = summary
    observation = (status, len(checks))
    if ctx.stable_observation != observation:
        ctx.stable_observation = observation
        ctx.stable_since = time.time()
    if status != "failed":
        ctx.empty_log_defers = 0
    return status


def _try_fix(
    ctx: _MemberCtx,
    worktree: Path,
    earlier_in_stack: int,
    sessions: list[ClaudeSession],
    *,
    ignore_sev: bool,
) -> bool:
    """Attempt to fix or judge spurious for one member.

    Returns True if a state change happened (fix pushed or spurious
    judgment recorded). False means we deferred -- typically because
    GitHub hasn't published the failing job's logs yet -- and the
    caller should sleep before re-ticking.

    Navigates the shared worktree to this member's ``/orig`` before
    invoking claude. After a clean fix, the worktree's HEAD is left
    at the new ``/orig_i`` (the [MERGEDOG] commit folded into the
    contributor's commit); the next tick navigates elsewhere as needed.
    """
    pr = ctx.member.pr
    if ctx.fix_commits_pushed >= MAX_FIX_COMMITS:
        die(
            f"PR #{pr}: already pushed {ctx.fix_commits_pushed} fix "
            f"commits and CI is still failing; halting"
        )

    failed = github.get_failed_job_logs(pr)
    if (
        _failed_logs_are_content_free(failed)
        and ctx.empty_log_defers < MAX_EMPTY_LOG_DEFERS
    ):
        ctx.empty_log_defers += 1
        log(
            f"PR #{pr}: failed-job logs not yet available "
            f"(defer {ctx.empty_log_defers}/{MAX_EMPTY_LOG_DEFERS})"
        )
        return False
    ctx.empty_log_defers = 0

    ctx_path, comments = _refresh_context_for(ctx)
    checks = github.get_pr_checks_all(pr)
    failing_check_names = sorted(
        c.get("name", "")
        for c in checks
        if c.get("bucket") in {"fail", "cancel"} and c.get("name")
    )

    prompt = render_fix_prompt(
        url=ctx.pr_data.get("url", ""),
        branch=ctx.member.head_ref,
        context_path=str(ctx_path),
        failed_jobs=failed,
        failing_check_names=failing_check_names,
        is_ghstack=True,
        earlier_in_stack=earlier_in_stack,
        drci_summary=github.latest_drci_summary(comments, head_sha=ctx.head_sha),
    )
    session_failed_jobs = [name for name, _ in failed]

    # Position the shared worktree at this member's /orig before
    # claude touches it. Whatever HEAD was before (e.g., another
    # member's just-folded fix) is irrelevant -- we want claude
    # operating on a single-commit-on-/orig_{i-1} starting state.
    repo.set_worktree_to_sha(worktree, ctx.orig_sha)

    sha_before = ctx.head_sha
    started_at = utc_now_iso()
    ran_cleanly, new_sha, transcript = claude_mod.invoke_fixer(worktree, prompt)
    _record_claude_session(
        sessions,
        mode=f"fix-CI #{pr}",
        sha_before=sha_before,
        started_at=started_at,
        ran_cleanly=ran_cleanly,
        new_sha=new_sha,
        transcript=transcript,
        on_commit=f"pushed fix commit {{sha}} on PR #{pr}",
        on_clean_noop=f"judged failures on PR #{pr} spurious (no commit)",
        extra=(
            f" — failing jobs: {', '.join(session_failed_jobs)}"
            if session_failed_jobs
            else ""
        ),
    )
    if not ran_cleanly:
        die(f"PR #{pr}: claude exited abnormally or produced an invalid commit")

    if new_sha is None:
        # Spurious -- could be a true infra flake, or claude's signal
        # that the failure is actually parent-caused. Either way we
        # mark and wait. The mark sticks until we propagate (which
        # gives this member a new /head and we'll clear then).
        newly_spurious = {
            c.get("name")
            for c in checks
            if c.get("bucket") in {"fail", "cancel"} and c.get("name")
        }
        ctx.spurious_check_names |= newly_spurious
        ctx.trust.spurious_check_names = sorted(ctx.spurious_check_names)
        ctx.trust.save()
        log(
            f"PR #{pr}: claude judged {len(newly_spurious)} failure"
            f"{'' if len(newly_spurious) == 1 else 's'} spurious; continuing"
        )
        ctx.last_status = None  # force re-log on next inspect
        return True

    # Real fix: clear spurious -- the new /head is fresh CI.
    ctx.spurious_check_names.clear()
    ctx.trust.spurious_check_names = []
    ctx.trust.save()

    _publish_ghstack_fix(
        pr,
        worktree,
        ctx.member.head_ref,
        new_sha,
        ctx.trust,
        ignore_sev=ignore_sev,
    )
    ctx.fix_commits_pushed += 1
    ctx.last_status = None
    return True


def _propagation_needed(
    contexts: list[_MemberCtx], now: float
) -> bool:
    """Should we run a full ghstack submit to propagate parent fixes?

    Bottom-up scan over consecutive (parent, child) pairs. A pair is
    "stale" iff ``parent_of(child.orig_sha) != parent.orig_sha`` -- the
    parent has been updated on origin but the child's /orig still has
    the old parent's SHA in its parent pointer.

    Propagation only fires when *every* stale pair has a green-stable
    parent (status=passed for ``CI_STABILITY_WINDOW_SEC``). One full
    ghstack submit re-bases the entire stack in a single push, so we
    wait until it's safe at every layer rather than firing per-pair.
    A non-green parent at any depth blocks propagation -- we'd rather
    fix that parent first than rebase children onto code we know is
    failing.
    """
    found_stale = False
    for i in range(len(contexts) - 1):
        parent = contexts[i]
        child = contexts[i + 1]
        if repo.parent_sha(child.orig_sha) == parent.orig_sha:
            continue
        found_stale = True
        if parent.stable_observation is None:
            return False
        if parent.stable_observation[0] != "passed":
            return False
        if now - parent.stable_since < CI_STABILITY_WINDOW_SEC:
            return False
    return found_stale


def _propagate_stack(
    contexts: list[_MemberCtx],
    worktree: Path,
    *,
    ignore_sev: bool,
) -> None:
    """Run a full ghstack submit to push parent fixes down to children.

    ``ghstack checkout <top_pr>`` assembles the latest /orig branches
    into a local stack, cherry-picking upper commits onto whatever the
    current parent /orig is. ``ghstack submit HEAD`` then pushes every
    /head whose content differs from origin -- typically just the
    children that just got rebased.

    After the push: refresh /head + /orig for every member, trust the
    new /heads, clear ``spurious_check_names`` on members whose /head
    actually changed (fresh CI invalidates prior judgments), and reset
    stability so the trunk-promotion gate (future commit) waits the
    window again.
    """
    if len(contexts) < 2:
        return

    top = contexts[-1]
    log(
        f"propagating stack: ghstack checkout #{top.member.pr} + "
        f"submit (full)"
    )
    if _wait_for_no_active_sev(
        "running full ghstack submit to propagate parent fix",
        ignore_sev=ignore_sev,
    ):
        # SEV waited: /orig branches may have moved while we sat. Re-fetch
        # so ghstack checkout sees the latest.
        _refresh_stack_refs(contexts)

    pre_head = {ctx.member.pr: ctx.head_sha for ctx in contexts}

    repo.ghstack_checkout(worktree, top.member.pr)
    repo.ghstack_submit(
        worktree, "Propagate parent fix downstream", no_stack=False
    )

    # Refresh refs and update ctx state for any member whose /head moved.
    pairs = [(ctx.member.head_ref, ctx.member.orig_ref) for ctx in contexts]
    refs = repo.fetch_stack_refs(pairs)
    for ctx in contexts:
        new_head = refs[ctx.member.head_ref]
        new_orig = refs[ctx.member.orig_ref]
        ctx.orig_sha = new_orig
        if new_head != pre_head[ctx.member.pr]:
            ctx.trust.trust(new_head)
            ctx.spurious_check_names.clear()
            ctx.trust.spurious_check_names = []
            ctx.trust.save()
            ctx.head_sha = new_head
            ctx.last_status = None
            ctx.empty_log_defers = 0
            log(
                f"  PR #{ctx.member.pr}: /head -> {new_head[:12]} "
                f"(rebased; cleared spurious)"
            )
        # Reset stability either way -- a propagation push restarts the
        # clock for ciflow/trunk eligibility downstream.
        ctx.stable_observation = None


def _scheduler_tick(
    contexts: list[_MemberCtx],
    worktree: Path,
    sessions: list[ClaudeSession],
    *,
    ignore_sev: bool,
) -> bool:
    """One scheduler tick.

    Returns True if an action was taken (re-tick immediately); False
    if we should sleep before re-ticking.
    """
    # Refresh local origin refs (single git fetch for all members'
    # /head + /orig), then trust + metadata refresh per member.
    _refresh_stack_refs(contexts)
    for ctx in contexts:
        _refresh_member_head(ctx)
        _refresh_member_pr_data(ctx)

    # Approve any approval-pending workflow runs across the stack. If
    # we approved anything, re-tick after a short settle so the new
    # runs become visible before we evaluate status.
    approved_total = 0
    for ctx in contexts:
        approved_total += _approve_pending_runs(
            ctx.head_sha, ctx.run_state_cache
        )
    if approved_total > 0:
        time.sleep(APPROVAL_SETTLE_SEC)
        return True

    # Inspect every member. Bottom-first iteration order doesn't
    # matter for inspection itself, but it makes the log line ordering
    # match the natural stack reading order.
    member_status: list[tuple[_MemberCtx, str]] = []
    for ctx in contexts:
        member_status.append((ctx, _inspect_member(ctx)))

    # Bottom-up: take the lowest failing member and try to fix it. We
    # don't try to classify "child-only bug" up front -- the prompt
    # tells claude to no-commit if the failure looks parent-caused.
    # The fix path uses ghstack submit --no-stack so siblings aren't
    # disturbed; propagation (next step) is what eventually rebases
    # them onto the fixed parent.
    for i, (ctx, status) in enumerate(member_status):
        if status == "failed":
            took_action = _try_fix(
                ctx,
                worktree,
                earlier_in_stack=i,
                sessions=sessions,
                ignore_sev=ignore_sev,
            )
            if took_action:
                return True
            # Fix deferred (logs not ready). Don't fall through to
            # propagation -- a member is failing, parent fixes haven't
            # cleared, we should keep waiting on logs.
            return False

    # No failing members. Check whether the stack has stale children
    # below green-stable parents and, if so, propagate via a full
    # ghstack submit. This is the only path that updates /head on
    # multiple members at once.
    if _propagation_needed(contexts, time.time()):
        _propagate_stack(contexts, worktree, ignore_sev=ignore_sev)
        return True

    # All members passed/pending and no stale children. ciflow/trunk
    # gating + handoff exit come in later commits; for now, sleep.
    return False


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
            if reassess:
                ctx.spurious_check_names = set()
                ctx.trust.spurious_check_names = []
                ctx.trust.save()
            else:
                ctx.spurious_check_names = set(ctx.trust.spurious_check_names)
            contexts.append(ctx)
            log(
                f"  PR #{ctx.member.pr}: /head={ctx.head_sha[:12]} "
                f"/orig={ctx.orig_sha[:12]}"
            )

        # One worktree for the whole stack -- the bottom PR's number
        # gives the path. Initialize at the bottom member's /orig; the
        # tick loop navigates as needed for each operation.
        bottom = contexts[0]
        worktree = repo.ensure_stack_worktree(bottom.member.pr, bottom.orig_sha)
        log(f"  stack worktree: {worktree}")

        # Main scheduler loop. Spins forever until killed; propagation,
        # ciflow/trunk gating, and a real handoff exit are added in
        # follow-up commits.
        sessions: list[ClaudeSession] = []
        while True:
            took_action = _scheduler_tick(
                contexts, worktree, sessions, ignore_sev=ignore_sev
            )
            if not took_action:
                time.sleep(POLL_INTERVAL_SEC)
    finally:
        for pr_num in labelled:
            try:
                github.remove_label(pr_num, MERGEDOG_LABEL)
            except Exception as e:
                log(
                    f"WARNING: failed to remove {MERGEDOG_LABEL} from "
                    f"PR #{pr_num}: {e}"
                )
