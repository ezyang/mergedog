"""Shepherd a whole ghstack stack as a single foreground process.

GitHub's ``/orig`` branches are canonical. Stack membership is
discovered by walking the top PR's ``/orig`` commit ancestry and
extracting ``Pull-Request`` (or older ``Pull-Request-Resolved``)
trailers. The working tree for
any member is reconstructed on demand via ``ghstack cherry-pick
--no-fetch`` (one call per member, bottom to top) rather than
maintained as persistent local state.

Stack mode uses a single shared worktree (rooted at the bottom PR's
number) for the whole run. Per tick we batch-fetch every member's
``/head`` and ``/orig`` from origin so staleness checks are local.

Per tick, every member is trust-checked, has its pending workflow
runs approved, and has its CI status inspected. Then a bottom-up
scan picks the lowest member with status="failed" and actionable logs
and invokes claude on it; on a clean fix the commit is folded into
/orig and re-published with ``ghstack submit --no-stack`` (so siblings
don't get hit with fresh CI). Propagation of a parent fix to children
uses the same cherry-pick reconstruction followed by a full-stack
``ghstack submit``.
"""
from __future__ import annotations

import faulthandler
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from mergedog import claude as claude_mod
from mergedog import context as context_mod
from mergedog import github, repo
from mergedog.handoff import (
    ClaudeSession,
    is_merge_conflict_failure,
    latest_mergebot_failure_event,
    latest_mergebot_event,
    post_handoff_comment,
    utc_now_iso,
    watch_stack_post_handoff,
)
from mergedog.log import configure_log_file, die, log
from mergedog.paths import PUSHED_COMMITS_LOG, REPO_SSH_URL, ROOT, context_file
from mergedog.project import get_project_policy
from mergedog.prompts import render_fix_prompt, render_rebase_conflict_prompt
from mergedog.shepherd import (
    APPROVAL_SETTLE_SEC,
    CI_STABILITY_WINDOW_SEC,
    EXIT_PR_NOT_ACTIONABLE,
    MAX_EMPTY_LOG_DEFERS,
    MAX_FIX_COMMITS,
    MERGEDOG_LABEL,
    POLL_INTERVAL_SEC,
    TRUNK_LABEL,
    _apply_spurious_overrides,
    _approve_pending_runs,
    _failed_logs_are_content_free,
    _llm_halt_message,
    _llm_label,
    _record_claude_session,
    _spurious_check_names_from_checks,
    _wait_for_pr_head,
    _wait_for_no_active_sev,
    describe_log_state,
)
from mergedog.stack import StackMember, resolve_stack
from mergedog.state import TrustDB
from mergedog.trust_seed import seed_trust_from_reviews


_SHA_RE = re.compile(r"\b[0-9a-f]{40}\b")


@dataclass
class _StackLogState:
    """Tracks the last consolidated CI summary string for the whole stack.

    Holding it stack-level (rather than per-ctx) lets us emit one log
    line covering every member per tick instead of one line per member,
    which is what mux's per-process pane needs to stay readable.
    """

    last_summary: str | None = None


@dataclass
class _MemberCtx:
    """Operational state for one stack member during a run.

    GitHub's ``/orig`` branches are canonical. Two SHAs are tracked:

    - ``head_sha``: GitHub's view of the PR head (from ``gh pr view``).
      Used for trust checks.
    - ``orig_sha``: origin's view of the ``/orig`` branch (from a
      ``git fetch`` of remote-tracking branches). Used as the truth for
      "what has been pushed".

    The worktree is reconstructed on demand via ``ghstack cherry-pick
    --no-fetch`` rather than maintained as persistent local state.
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
    failing_check_names: list[str] = field(default_factory=list)


def _record_pushed_commit(kind: str, pr: int, sha: str, subject: str) -> None:
    """Append one line to ``~/.mergedog/pushed-commits.log``.

    A skim-friendly, append-only log of every commit a stack-mode shepherd
    publishes. Separate from the per-PR shepherd logs so the operator can
    review "what has mergedog pushed?" across the whole stack at a glance
    without grepping multiple files.
    """
    PUSHED_COMMITS_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    subject_line = subject.splitlines()[0] if subject else ""
    line = f"{ts}  {kind:<6}  PR#{pr}  {sha[:12]}  {subject_line}\n"
    with open(PUSHED_COMMITS_LOG, "a", encoding="utf-8") as f:
        f.write(line)


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
    origin_head_sha: str,
    origin_orig_sha: str,
    accept_divergence: bool,
    viewer: str,
) -> _MemberCtx:
    """Per-member analogue of the ghstack init block in ``_shepherd_body``.

    ``origin_head_sha`` / ``origin_orig_sha`` are read from a single
    batch ``git fetch`` done by ``run_stack`` before this loop runs --
    we used to fetch /head and /orig per-member here, which serialized
    8+ network round-trips and ate most of startup.

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
    if origin_head_sha != head_sha:
        die(
            f"PR #{pr}: origin's {member.head_ref} "
            f"({origin_head_sha[:12]}) differs from the SHA GitHub "
            f"reports for the PR ({head_sha[:12]}); refusing to act"
        )

    return _MemberCtx(
        member=member,
        pr_data=pr_data,
        trust=trust,
        self_pr=self_pr,
        head_sha=head_sha,
        orig_sha=origin_orig_sha,
    )


def _add_mergedog_labels_parallel(members: list[StackMember]) -> list[int]:
    """Apply the mergedog label to every member concurrently.

    Sequential ``gh pr edit ... --add-label`` calls were ~5s each on a
    real run -- 4 PRs took 20s of pure waiting. Threads are fine here
    since this is I/O-bound on a subprocess invocation. Failures are
    logged and the PR is left out of the returned list so the
    finally-cleanup only removes labels we actually added.
    """
    if not members:
        return []
    labelled: list[int] = []
    with ThreadPoolExecutor(max_workers=len(members)) as ex:
        futures = {
            ex.submit(github.add_label, m.pr, MERGEDOG_LABEL): m
            for m in members
        }
        for fut in as_completed(futures):
            m = futures[fut]
            try:
                fut.result()
                labelled.append(m.pr)
            except Exception as e:
                log(
                    f"WARNING: failed to add {MERGEDOG_LABEL} to "
                    f"PR #{m.pr}: {e}"
                )
    return sorted(labelled)


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


def _inspect_member(ctx: _MemberCtx) -> tuple[str, int, int]:
    """Update ctx with current CI status; return ``(status, done, total)``.

    Also caches the post-spurious-override failing check names on the ctx
    so that other members' fix prompts can reference them as the "earlier
    stack" status block without making another checks API call.

    Does not log; the consolidated stack-level summary is emitted by the
    caller in ``_scheduler_tick``.
    """
    checks = github.get_pr_checks_all(ctx.member.pr)
    effective = _apply_spurious_overrides(checks, ctx.spurious_check_names)
    status = github.evaluate_checks(effective)
    done = sum(1 for c in checks if c.get("bucket") not in {"pending", None})
    total = len(checks)
    ctx.last_status = f"{status} ({done}/{total} done)"
    observation = (status, total)
    if ctx.stable_observation != observation:
        ctx.stable_observation = observation
        ctx.stable_since = time.time()
    if status != "failed":
        ctx.empty_log_defers = 0
    ctx.failing_check_names = sorted(
        c.get("name", "")
        for c in effective
        if c.get("bucket") in {"fail", "cancel"} and c.get("name")
    )
    return status, done, total


def _format_stack_summary(
    inspections: list[tuple[_MemberCtx, str, int, int]],
) -> str:
    """Build one summary line covering the whole stack.

    Headline status is worst-of: failed > pending > passed. Per-member
    counts follow in stack order (bottom to top), since that matches the
    natural ghstack reading order.
    """
    statuses = {s for _, s, _, _ in inspections}
    if "failed" in statuses:
        overall = "failed"
    elif "pending" in statuses:
        overall = "pending"
    else:
        overall = "passed"
    counts = ", ".join(f"{done}/{total}" for _, _, done, total in inspections)
    return f"{overall} ({counts} done)"


def _reconstruct_stack_up_to(
    contexts: list[_MemberCtx], target_idx: int, worktree: Path
) -> None:
    """Reconstruct the stack via ``ghstack cherry-pick`` from bottom to ``target_idx``.

    Positions the worktree at the parent of the bottom member's ``/orig``
    (which sits on main), then cherry-picks each member's ``/orig`` one
    by one. This gives us the correct tree content for the target member
    without maintaining any persistent local state -- GitHub's ``/orig``
    branches are the single source of truth.
    """
    base = repo.parent_sha(contexts[0].orig_sha)
    repo.set_worktree_to_sha(worktree, base)
    for i in range(target_idx + 1):
        repo.ghstack_cherry_pick(worktree, contexts[i].member.pr)


def _publish_fix(
    ctx: _MemberCtx,
    worktree: Path,
    *,
    submit_message: str,
    ignore_sev: bool,
    force_ghstack: bool,
) -> None:
    """Push the worktree's HEAD to origin via ``ghstack submit --no-stack``.

    After a successful submit, refreshes the origin /head and /orig
    snapshots so ctx tracks what GitHub now has.
    """
    pr = ctx.member.pr
    _wait_for_no_active_sev(
        f"submitting PR #{pr} via ghstack", ignore_sev=ignore_sev
    )
    repo.ghstack_submit(worktree, submit_message, force=force_ghstack)
    new_head_sha = repo.fetch_ghstack_head(ctx.member.head_ref)
    new_orig_sha = repo.fetch_ghstack_orig(ctx.member.head_ref)
    ctx.trust.trust(new_head_sha)
    ctx.head_sha = new_head_sha
    ctx.orig_sha = new_orig_sha
    log(f"PR #{pr}: ghstack submitted; new /head = {new_head_sha[:12]}")


def _try_fix(
    ctx: _MemberCtx,
    contexts: list[_MemberCtx],
    target_idx: int,
    worktree: Path,
    sessions: list[ClaudeSession],
    *,
    ignore_sev: bool,
    force_ghstack: bool,
    extra_context: str | None = None,
) -> bool:
    """Attempt to fix or judge spurious for one member.

    Returns True if a state change happened (fix pushed or spurious
    judgment recorded). False means we deferred -- typically because
    GitHub hasn't published the failing job's logs yet -- and the
    caller should sleep before re-ticking.

    Reconstructs the stack up to the target member via
    ``ghstack cherry-pick --no-fetch``, invokes claude, folds the fix
    into the contributor's /orig commit, and pushes via
    ``ghstack submit --no-stack``.
    """
    pr = ctx.member.pr
    if ctx.fix_commits_pushed >= MAX_FIX_COMMITS:
        die(
            f"PR #{pr}: already pushed {ctx.fix_commits_pushed} fix "
            f"commits and CI is still failing; halting"
        )

    failed = [
        (name, text)
        for name, text in github.get_failed_job_logs(pr)
        if name not in ctx.spurious_check_names
    ]
    log_state = describe_log_state(failed, len(ctx.failing_check_names))
    if (
        _failed_logs_are_content_free(failed)
        and ctx.empty_log_defers < MAX_EMPTY_LOG_DEFERS
    ):
        ctx.empty_log_defers += 1
        log(
            f"PR #{pr}: failed-job logs not yet available "
            f"(defer {ctx.empty_log_defers}/{MAX_EMPTY_LOG_DEFERS}); "
            f"{log_state}"
        )
        return False
    ctx.empty_log_defers = 0
    log(f"PR #{pr}: invoking {_llm_label()} on failing CI ({log_state})")

    ctx_path, comments = _refresh_context_for(ctx)
    checks = github.get_pr_checks_all(pr)
    effective_checks = _apply_spurious_overrides(
        checks, ctx.spurious_check_names
    )
    failing_check_names = sorted(
        c.get("name", "")
        for c in effective_checks
        if c.get("bucket") in {"fail", "cancel"} and c.get("name")
    )

    earlier_members: list[dict] = []
    if target_idx > 0:
        for i, earlier in enumerate(contexts[:target_idx]):
            earlier_members.append(
                {
                    "pr": earlier.member.pr,
                    "head_offset": target_idx - i,
                    "status": earlier.last_status or "unknown",
                    "failing_checks": list(earlier.failing_check_names),
                    "fix_commits_pushed": earlier.fix_commits_pushed,
                }
            )

    prompt = render_fix_prompt(
        url=ctx.pr_data.get("url", ""),
        branch=ctx.member.head_ref,
        context_path=str(ctx_path),
        failed_jobs=failed,
        failing_check_names=failing_check_names,
        is_ghstack=True,
        earlier_in_stack=target_idx,
        earlier_members=earlier_members,
        drci_summary=github.latest_drci_summary(comments, head_sha=ctx.head_sha),
        extra_context=extra_context,
    )
    session_failed_jobs = [name for name, _ in failed]

    # Reconstruct the stack up to this member from GitHub's canonical
    # /orig branches. Each member is cherry-picked one by one.
    _reconstruct_stack_up_to(contexts, target_idx, worktree)

    sha_before = ctx.head_sha
    started_at = utc_now_iso()
    result = claude_mod.invoke_fixer(worktree, prompt)
    ran_cleanly, new_sha, transcript = result
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
        die(
            f"PR #{pr}: "
            + _llm_halt_message(
                result,
                f"{_llm_label()} exited abnormally or produced an invalid commit",
            )
        )

    if new_sha is None:
        newly_spurious = _spurious_check_names_from_checks(effective_checks)
        if not newly_spurious:
            die(
                f"PR #{pr}: {_llm_label()} made no commit, but mergedog "
                "could not map that no-op to any failed check; halting for "
                "human intervention"
            )
        ctx.spurious_check_names |= newly_spurious
        ctx.trust.spurious_check_names = sorted(ctx.spurious_check_names)
        ctx.trust.save()
        log(
            f"PR #{pr}: {_llm_label()} judged {len(newly_spurious)} failure"
            f"{'' if len(newly_spurious) == 1 else 's'} spurious; continuing"
        )
        ctx.last_status = None
        return True

    ctx.spurious_check_names.clear()
    ctx.trust.spurious_check_names = []
    ctx.trust.save()

    fix_message = repo.commit_message(worktree, new_sha)
    repo.fixup_into_parent(worktree)

    _publish_fix(
        ctx,
        worktree,
        submit_message=fix_message,
        ignore_sev=ignore_sev,
        force_ghstack=force_ghstack,
    )
    _record_pushed_commit("fix", pr, ctx.head_sha, fix_message)
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
        if not _is_green_stable(parent, now):
            return False
    return found_stale


def _propagate_stack(
    contexts: list[_MemberCtx],
    worktree: Path,
    *,
    ignore_sev: bool,
) -> None:
    """Reconstruct the full stack and push via ghstack submit.

    Cherry-picks each member's ``/orig`` one by one from bottom to top,
    then runs ``ghstack submit HEAD`` which pushes any ``/head`` whose
    content differs from origin -- typically just the children that got
    rebased onto the fixed parent.

    After the push: refresh /head + /orig for every member, trust the
    new /heads, clear ``spurious_check_names`` on members whose /head
    actually changed (fresh CI invalidates prior judgments), and reset
    stability so the trunk-promotion gate waits the window again.
    """
    if len(contexts) < 2:
        return

    top = contexts[-1]
    log(
        f"propagating stack: cherry-pick all {len(contexts)} members + "
        f"submit (full)"
    )
    if _wait_for_no_active_sev(
        "running full ghstack submit to propagate parent fix",
        ignore_sev=ignore_sev,
    ):
        _refresh_stack_refs(contexts)

    pre_head = {ctx.member.pr: ctx.head_sha for ctx in contexts}

    _reconstruct_stack_up_to(contexts, len(contexts) - 1, worktree)
    repo.ghstack_submit(
        worktree, "Propagate parent fix downstream", no_stack=False
    )

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
            _record_pushed_commit(
                "rebase", ctx.member.pr, new_head, "(propagated rebase)"
            )
        ctx.stable_observation = None


def _reset_after_head_change(ctx: _MemberCtx, new_head: str, new_orig: str) -> None:
    ctx.orig_sha = new_orig
    if new_head == ctx.head_sha:
        return
    ctx.trust.trust(new_head)
    ctx.spurious_check_names.clear()
    ctx.trust.spurious_check_names = []
    ctx.trust.save()
    ctx.head_sha = new_head
    ctx.last_status = None
    ctx.empty_log_defers = 0
    ctx.stable_observation = None


def _rebase_stack_prefix_onto_main(
    contexts: list[_MemberCtx],
    target_idx: int,
    worktree: Path,
    sessions: list[ClaudeSession],
    *,
    ignore_sev: bool,
    force_ghstack: bool,
) -> None:
    """Rebase the stack prefix ending at ``target_idx`` onto main.

    This is the stack-mode equivalent of the single-PR ``--rebase`` recovery:
    reconstruct the canonical /orig stack, rebase it onto the selected
    known-good main ref, resolve conflicts with Claude if necessary, and push
    the rebased stack with a full ghstack submit.
    """
    target_ctx = contexts[target_idx]
    if _wait_for_no_active_sev(
        "rebasing stack onto main", ignore_sev=ignore_sev
    ):
        repo.fetch_origin()

    target, reason = repo.select_rebase_target(worktree)
    log(f"rebase target: {reason}")
    _reconstruct_stack_up_to(contexts, target_idx, worktree)

    try:
        status, new_top_sha = repo.attempt_rebase_main(worktree, ref=target)
    except RuntimeError as e:
        die(str(e))

    if status == "noop":
        log("stack rebase produced no new commit (already at target)")
        return

    if status == "conflict":
        log(f"stack rebase produced conflicts; asking {_llm_label()} to resolve")
        ctx_path, _ = _refresh_context_for(target_ctx)
        prompt = render_rebase_conflict_prompt(
            url=target_ctx.pr_data.get("url", ""),
            branch=target_ctx.member.head_ref,
            context_path=str(ctx_path),
        )
        sha_before = repo.head_sha(worktree)
        started_at = utc_now_iso()
        result = claude_mod.invoke_rebase_resolver(
            worktree, prompt, allow_multiple_commits=True
        )
        ran_cleanly, new_top_sha, transcript = result
        _record_claude_session(
            sessions,
            mode=f"rebase-resolver #{target_ctx.member.pr}",
            sha_before=sha_before,
            started_at=started_at,
            ran_cleanly=ran_cleanly,
            new_sha=new_top_sha,
            transcript=transcript,
            on_commit="resolved stack rebase conflicts in commit {sha}",
            on_clean_noop="aborted the stack rebase",
        )
        if not ran_cleanly:
            die(
                _llm_halt_message(
                    result,
                    f"{_llm_label()} failed to resolve the stack rebase conflict cleanly",
                )
            )
        if new_top_sha is None:
            die(f"{_llm_label()} aborted the stack rebase; halting for human intervention")

    assert new_top_sha is not None
    log(
        f"rebased stack through PR #{target_ctx.member.pr} to "
        f"{new_top_sha[:12]}; re-publishing via ghstack"
    )
    _wait_for_no_active_sev(
        "re-publishing rebased stack via ghstack submit", ignore_sev=ignore_sev
    )
    repo.ghstack_submit(
        worktree,
        "Rebase stack onto origin/main",
        no_stack=False,
        force=force_ghstack,
    )

    refs = repo.fetch_stack_refs(
        [(ctx.member.head_ref, ctx.member.orig_ref) for ctx in contexts]
    )
    for ctx in contexts:
        old_head = ctx.head_sha
        new_head = refs[ctx.member.head_ref]
        new_orig = refs[ctx.member.orig_ref]
        _reset_after_head_change(ctx, new_head, new_orig)
        if new_head != old_head:
            log(f"  PR #{ctx.member.pr}: /head -> {new_head[:12]} (rebased)")
            _record_pushed_commit(
                "rebase", ctx.member.pr, new_head, "Rebase stack onto origin/main"
            )
            _wait_for_pr_head(ctx.member.pr, new_head)


def _latest_unhandled_stack_failure(
    contexts: list[_MemberCtx],
) -> tuple[_MemberCtx, str, str] | None:
    """Return the newest unhandled post-handoff mergebot failure in the stack.

    This covers stack shepherds that exited before they had a watch loop: the
    failure comment exists on GitHub, but ``last_observed_failure_*`` was never
    persisted locally. On restart we should recover from it instead of treating
    stale CI as the next thing to fix.
    """
    current_orig_shas = {ctx.orig_sha for ctx in contexts}
    failures: list[tuple[str, _MemberCtx, str]] = []
    for ctx in contexts:
        pr = ctx.member.pr
        event = latest_mergebot_failure_event(
            pr, ctx.trust.last_observed_failure_iso
        )
        if event is None:
            continue
        body = event[1]
        mentioned_shas = set(_SHA_RE.findall(body))
        if not (mentioned_shas & current_orig_shas):
            log(
                f"PR #{pr}: ignoring prior mergebot failure; no current "
                f"stack /orig SHA mentioned"
            )
            continue
        failures.append((event[0], ctx, body))
    if not failures:
        return None
    event_iso, ctx, body = max(failures, key=lambda item: item[0])
    return ctx, event_iso, body


def _is_green_stable(ctx: _MemberCtx, now: float) -> bool:
    """True iff ctx's CI verdict is ``passed`` and has held for the window.

    Used both for trunk-promotion eligibility and (via callers) for the
    propagation predicate. ``stable_observation is None`` means we
    haven't seen any inspection yet -- not stable.
    """
    if ctx.stable_observation is None:
        return False
    if ctx.stable_observation[0] != "passed":
        return False
    return now - ctx.stable_since >= CI_STABILITY_WINDOW_SEC


def _trunk_promotion_target(
    contexts: list[_MemberCtx], now: float
) -> _MemberCtx | None:
    """Lowest member eligible for ciflow/trunk promotion, or None.

    Eligibility:
      - ``trunk_applied`` is False (haven't promoted yet),
      - the member's current CI is green-stable,
      - parent (if any) is trunk-applied AND green-stable -- i.e., the
        parent's trunk-only CI has settled green.

    The "parent green-stable" check is effectively "parent's trunk-CI
    is green" because applying trunk on a member resets its
    ``stable_observation`` (new trunk-only checks appear), so the
    next time we see ``passed`` and stable, the trunk checks have
    settled.
    """
    if TRUNK_LABEL is None:
        return None
    for i, ctx in enumerate(contexts):
        if ctx.trunk_applied:
            continue
        if not _is_green_stable(ctx, now):
            return None
        if i > 0:
            prev = contexts[i - 1]
            if not prev.trunk_applied:
                return None
            if not _is_green_stable(prev, now):
                return None
        return ctx
    return None


def _apply_trunk(ctx: _MemberCtx, *, ignore_sev: bool) -> None:
    """Add the ciflow/trunk label to a stack member.

    Adding the label kicks off trunk-only workflow runs; we gate on a
    CI SEV first so we don't pile on broken trunk. Marks
    ``trunk_applied`` and resets stability so the next inspection
    starts the window over (because new checks will appear).
    """
    if TRUNK_LABEL is None:
        ctx.trunk_applied = True
        return
    pr = ctx.member.pr
    _wait_for_no_active_sev(
        f"applying {TRUNK_LABEL} to PR #{pr}", ignore_sev=ignore_sev
    )
    log(f"PR #{pr}: CI green; applying {TRUNK_LABEL} label")
    github.add_label(pr, TRUNK_LABEL)
    ctx.trunk_applied = True
    ctx.last_status = None
    ctx.stable_observation = None


def _all_trunk_green_stable(
    contexts: list[_MemberCtx], now: float
) -> bool:
    """True iff every member has had ciflow/trunk applied and is green-stable.

    The exit condition for ``run_stack``: nothing left to do but post
    the per-member handoff comments and let humans run
    ``@pytorchbot merge``.
    """
    for ctx in contexts:
        if TRUNK_LABEL is not None and not ctx.trunk_applied:
            return False
        if not _is_green_stable(ctx, now):
            return False
    return True


def _scheduler_tick(
    contexts: list[_MemberCtx],
    worktree: Path,
    sessions_by_pr: dict[int, list[ClaudeSession]],
    log_state: _StackLogState,
    *,
    ignore_sev: bool,
    force_ghstack: bool,
    extra_context: str | None = None,
) -> bool:
    """One scheduler tick.

    Returns True if an action was taken (re-tick immediately); False
    if we should sleep before re-ticking.
    """
    # Check GitHub for head SHA changes + fresh metadata per member.
    # Only git-fetch refs when a head actually moved -- avoids a
    # redundant ``git fetch origin`` every tick while CI is pending.
    prev_heads = {ctx.member.pr: ctx.head_sha for ctx in contexts}
    for ctx in contexts:
        _refresh_member_head(ctx)
        _refresh_member_pr_data(ctx)
    if any(ctx.head_sha != prev_heads[ctx.member.pr] for ctx in contexts):
        _refresh_stack_refs(contexts)

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

    # Inspect every member. Bottom-first iteration order is what the
    # consolidated summary line uses to lay out the per-member counts.
    inspections: list[tuple[_MemberCtx, str, int, int]] = []
    for ctx in contexts:
        status, done, total = _inspect_member(ctx)
        inspections.append((ctx, status, done, total))

    summary = _format_stack_summary(inspections)
    if summary != log_state.last_summary:
        log(f"stack CI -> {summary}")
        log_state.last_summary = summary

    member_status: list[tuple[_MemberCtx, str]] = [
        (ctx, status) for ctx, status, _, _ in inspections
    ]

    # Bottom-up: take the lowest failing member with actionable logs
    # and try to fix it. We don't try to classify "child-only bug" up
    # front -- the prompt tells claude to no-commit if the failure
    # looks parent-caused. The fix path uses ghstack submit --no-stack
    # so siblings aren't disturbed; propagation (next step) is what
    # eventually rebases them onto the fixed parent.
    #
    # If a lower failing member's logs aren't published yet, we defer
    # it but keep scanning -- a higher failing member may have
    # actionable logs *now*, and at worst its fix gets re-pushed when
    # the lower PR's fix later propagates. Better than sleeping a full
    # poll interval just because the bottom-most failure happened to
    # transition first.
    failing_count = 0
    for i, (ctx, status) in enumerate(member_status):
        if status != "failed":
            continue
        failing_count += 1
        took_action = _try_fix(
            ctx,
            contexts,
            target_idx=i,
            worktree=worktree,
            sessions=sessions_by_pr[ctx.member.pr],
            ignore_sev=ignore_sev,
            force_ghstack=force_ghstack,
            extra_context=extra_context,
        )
        if took_action:
            return True
        # Fix deferred for this member (logs not ready). Try higher
        # failing members in case one of them has logs we can act on.

    if failing_count:
        # All failures deferred. Don't fall through to propagation /
        # trunk -- a member is failing, parent fixes need to land first.
        log(
            f"tick: {failing_count} failing member(s) deferred this tick; "
            f"sleeping {POLL_INTERVAL_SEC}s"
        )
        return False

    now = time.time()

    # No failing members. Check whether the stack has stale children
    # below green-stable parents and, if so, propagate via a full
    # ghstack submit. This is the only path that updates /head on
    # multiple members at once.
    if _propagation_needed(contexts, now):
        _propagate_stack(contexts, worktree, ignore_sev=ignore_sev)
        return True

    # No fixes, no propagation. Try to advance the trunk frontier:
    # promote the lowest member that's green-stable on regular CI and
    # whose parent (if any) is already trunk-applied + green-stable.
    target = _trunk_promotion_target(contexts, now)
    if target is not None:
        _apply_trunk(target, ignore_sev=ignore_sev)
        return True

    # Everyone is in a stable state -- either green or waiting for the
    # window to elapse. Sleep; the outer loop checks the all-trunk-green
    # exit predicate after every tick.
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
    force_ghstack: bool = False,
    manage_mergedog_label: bool = False,
    extra_context: str | None = None,
) -> None:
    repo.ensure_clone()
    # Fetch just origin/main (needed for merge-base in stack discovery)
    # rather than a full fetch_origin which would pull all of pytorch.
    repo.fetch_main()

    members, pr_data_by_pr = resolve_stack(pr)
    bottom_pr = members[0].pr
    configure_log_file(ROOT / "logs" / f"stack-{bottom_pr}.log")
    log(f"resolved ghstack stack containing PR #{pr}: {len(members)} member(s)")
    for i, m in enumerate(members):
        log(f"  [{i}] PR #{m.pr}  head={m.head_ref}  orig={m.orig_ref}")

    # Optionally tag every member up front so other operators / mergedogs see
    # the whole stack is owned, then remove labels on any exit path.
    labelled: list[int] = []
    signal.signal(signal.SIGTERM, _sigterm_to_systemexit)
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1)
    try:
        if manage_mergedog_label:
            labelled = _add_mergedog_labels_parallel(members)

        # One batched ``git fetch`` for every member's /head + /orig.
        # _setup_member then reads SHAs out of this dict instead of
        # making per-member round-trips.
        ref_state = repo.fetch_stack_refs(
            [(m.head_ref, m.orig_ref) for m in members]
        )

        viewer = github.viewer_login()
        contexts: list[_MemberCtx] = []
        for m in members:
            ctx = _setup_member(
                m,
                pr_data_by_pr[m.pr],
                origin_head_sha=ref_state[m.head_ref],
                origin_orig_sha=ref_state[m.orig_ref],
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
            log(f"  PR #{ctx.member.pr}: /head={ctx.head_sha[:12]} /orig={ctx.orig_sha[:12]}")

        # One worktree for the whole stack -- the bottom PR's number
        # gives the path. Initialize at the bottom member's /orig; the
        # tick loop navigates as needed for each operation.
        bottom = contexts[0]
        worktree = repo.ensure_stack_worktree(bottom.member.pr, bottom.orig_sha)
        log(f"  stack worktree: {worktree}")

        # Main scheduler loop. Per-PR session lists keep claude
        # transcripts attributable so each member's handoff comment
        # only shows its own work.
        sessions_by_pr: dict[int, list[ClaudeSession]] = {
            ctx.member.pr: [] for ctx in contexts
        }
        if rebase:
            log("user requested upfront rebase of stack onto origin/main")
            _rebase_stack_prefix_onto_main(
                contexts,
                len(contexts) - 1,
                worktree,
                sessions_by_pr[contexts[-1].member.pr],
                ignore_sev=ignore_sev,
                force_ghstack=force_ghstack,
            )
            for ctx in contexts:
                ctx.last_status = None

        prior_conflict = next(
            (
                ctx
                for ctx in reversed(contexts)
                if ctx.trust.last_observed_failure_body
                and is_merge_conflict_failure(ctx.trust.last_observed_failure_body)
            ),
            None,
        )
        if prior_conflict is not None:
            log(
                f"prior merge-conflict failure detected on PR "
                f"#{prior_conflict.member.pr}; rebasing stack onto main"
            )
            _rebase_stack_prefix_onto_main(
                contexts,
                contexts.index(prior_conflict),
                worktree,
                sessions_by_pr[prior_conflict.member.pr],
                ignore_sev=ignore_sev,
                force_ghstack=force_ghstack,
            )
            prior_conflict.trust.last_observed_failure_body = ""
            prior_conflict.trust.save()

        unhandled_failure = _latest_unhandled_stack_failure(contexts)
        if unhandled_failure is not None:
            failed_ctx, event_iso, fail_body = unhandled_failure
            failed_ctx.trust.last_observed_failure_iso = event_iso
            failed_ctx.trust.last_observed_failure_body = fail_body
            failed_ctx.trust.save()
            if is_merge_conflict_failure(fail_body):
                log(
                    f"unhandled prior merge-conflict failure detected on PR "
                    f"#{failed_ctx.member.pr}; rebasing stack onto main"
                )
                repo.fetch_origin()
                _rebase_stack_prefix_onto_main(
                    contexts,
                    contexts.index(failed_ctx),
                    worktree,
                    sessions_by_pr[failed_ctx.member.pr],
                    ignore_sev=ignore_sev,
                    force_ghstack=force_ghstack,
                )
                failed_ctx.trust.last_observed_failure_body = ""
                failed_ctx.trust.save()

        log_state = _StackLogState()
        while True:
            try:
                took_action = _scheduler_tick(
                    contexts,
                    worktree,
                    sessions_by_pr,
                    log_state,
                    ignore_sev=ignore_sev,
                    force_ghstack=force_ghstack,
                    extra_context=extra_context,
                )
            except subprocess.CalledProcessError as e:
                # Transient gh/git subprocess failure (GraphQL 504, fetch
                # hiccup, etc.). Don't crash the whole shepherd -- log,
                # treat as "no action this tick", and re-tick after a
                # sleep. die()-style halts raise SystemExit, which
                # bypasses this handler.
                cmd = e.cmd[0] if isinstance(e.cmd, list) and e.cmd else "?"
                log(
                    f"WARNING: tick failed transiently ({cmd} exit "
                    f"{e.returncode}); will retry next tick"
                )
                took_action = False
            if _all_trunk_green_stable(contexts, time.time()):
                log(
                    "all members trunk-green; posting handoff "
                    "comments and exiting"
                )
                for ctx in contexts:
                    post_handoff_comment(
                        ctx.member.pr,
                        ctx.pr_data,
                        sessions_by_pr[ctx.member.pr],
                    )

                since_by_pr: dict[int, str] = {}
                for ctx in contexts:
                    handoff_iso = (
                        github.latest_mergedog_handoff_iso(ctx.member.pr)
                        or utc_now_iso()
                    )
                    since_by_pr[ctx.member.pr] = max(
                        handoff_iso, ctx.trust.last_observed_failure_iso
                    )
                project = get_project_policy()
                merge_instruction = (
                    f"comment `{project.merge_command}` on "
                    if project.merge_command
                    else "merge "
                )
                log(
                    "Hand off stack to a human reviewer; have them "
                    f"{merge_instruction}PR #{contexts[-1].member.pr}."
                )

                result, event_pr, event_iso, fail_body = watch_stack_post_handoff(
                    since_by_pr
                )
                if result == "closed":
                    die(
                        f"PR #{event_pr} is no longer open; pruning local "
                        "stack shepherd state",
                        code=EXIT_PR_NOT_ACTIONABLE,
                    )

                failed_ctx = next(
                    ctx for ctx in contexts if ctx.member.pr == event_pr
                )
                assert event_iso is not None
                failed_ctx.trust.last_observed_failure_iso = event_iso
                failed_ctx.trust.last_observed_failure_body = fail_body or ""
                failed_ctx.trust.save()

                if fail_body and is_merge_conflict_failure(fail_body):
                    log(
                        f"pytorchmergebot merge failed on PR #{event_pr} "
                        "due to merge conflict; rebasing stack onto main"
                    )
                    repo.fetch_origin()
                    _rebase_stack_prefix_onto_main(
                        contexts,
                        contexts.index(failed_ctx),
                        worktree,
                        sessions_by_pr[failed_ctx.member.pr],
                        ignore_sev=ignore_sev,
                        force_ghstack=force_ghstack,
                    )
                    failed_ctx.trust.last_observed_failure_body = ""
                    failed_ctx.trust.save()
                    log_state.last_summary = None
                    continue

                log(
                    "pytorchmergebot reported stack merge failure; "
                    "re-inspecting CI"
                )
                for ctx in contexts:
                    _refresh_member_pr_data(ctx)
                    ctx.last_status = None
                log_state.last_summary = None
                continue
            if not took_action:
                time.sleep(POLL_INTERVAL_SEC)
    finally:
        if labelled:
            with ThreadPoolExecutor(max_workers=len(labelled)) as ex:
                futures = {
                    ex.submit(github.remove_label, pr_num, MERGEDOG_LABEL): pr_num
                    for pr_num in labelled
                }
                for fut in as_completed(futures):
                    pr_num = futures[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        log(
                            f"WARNING: failed to remove {MERGEDOG_LABEL} from "
                            f"PR #{pr_num}: {e}"
                        )
