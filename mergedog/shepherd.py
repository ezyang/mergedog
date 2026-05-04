"""The main mergedog shepherding loop.

One process per PR. Synchronous. Halts on any sign of an untrusted change.
"""
from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

from mergedog import claude as claude_mod
from mergedog import context as context_mod
from mergedog import github, repo
from mergedog.handoff import (
    ClaudeSession,
    post_handoff_comment,
    utc_now_iso,
    watch_post_handoff,
)
from mergedog.log import die, log
from mergedog.paths import REPO_SLUG, REPO_SSH_URL, context_file
from mergedog.prompts import render_fix_prompt, render_merge_conflict_prompt
from mergedog.repo import MERGE_RESOLVED_SUBJECT
from mergedog.state import TrustDB
from mergedog.trust_seed import seed_trust_from_reviews


SEV_POLL_INTERVAL_SEC = 5 * 60  # SEVs are minutes-to-hours; don't spam ``gh``


def _wait_for_no_active_sev(reason: str, *, ignore_sev: bool) -> bool:
    """If pytorch CI has an open SEV, block until it clears.

    A CI SEV here is any open issue on pytorch/pytorch tagged
    ``ci: sev`` -- dev-infra's signal that trunk is degraded. Default
    behavior is to wait it out so we don't stampede broken CI with
    new pushes; ``ignore_sev`` (operator override via ``--ignore-sev``)
    skips the wait. Called only at "would trigger CI" critical spots,
    not in the inner poll, to keep the GH API call rate low.

    Returns True if it actually had to wait (i.e. a SEV was open at entry
    and has now cleared) -- callers can use this to discard work prepared
    against a stale view of trunk.
    """
    if ignore_sev:
        return False
    last_ids: tuple[int, ...] | None = None
    while True:
        sevs = github.list_active_ci_sevs()
        if not sevs:
            if last_ids is not None:
                log("CI SEV cleared; resuming")
                return True
            return False
        ids = tuple(sorted(s.get("number") for s in sevs if s.get("number")))
        if ids != last_ids:
            head = sevs[0]
            others = f" (+{len(sevs) - 1} more)" if len(sevs) > 1 else ""
            log(
                f"parked on ci: sev #{head.get('number')} "
                f"{head.get('title', '?')!r}{others}; "
                f"waiting before {reason}"
            )
            last_ids = ids
        time.sleep(SEV_POLL_INTERVAL_SEC)


POLL_INTERVAL_SEC = 60
APPROVAL_SETTLE_SEC = 15
PUSH_VISIBILITY_TIMEOUT_SEC = 90
# Distinct exit code shepherds use when the PR is no longer actionable
# (closed, merged, etc.). The mux watches for this and auto-prunes the
# session and on-disk state -- there's no point retrying.
EXIT_PR_NOT_ACTIONABLE = 42
# How long ``(status, check_count)`` must hold steady before we trust a
# "passed" verdict. Right after a push, GitHub registers the workflow runs
# over a span of seconds; without this window we'd see "1/1 done -> passed"
# before the other 10+ required workflows even exist, and slap the trunk
# label on prematurely.
CI_STABILITY_WINDOW_SEC = 60
TRUNK_LABEL = "ciflow/trunk"
# Marker label so humans can see at a glance which PRs already have a live
# mergedog shepherding them -- keeps two operators (or two mergedogs) from
# fighting over the same PR. Added after validation passes; removed on every
# exit path (success, HALT, SIGTERM from ``mux cancel``, ctrl-c).
MERGEDOG_LABEL = "mergedog"
MAX_FIX_COMMITS = 5  # safety cap; halt if claude keeps pushing fixes


def _is_ghstack(pr_data: dict) -> bool:
    branch = pr_data.get("headRefName", "") or ""
    body = pr_data.get("body", "") or ""
    if branch.startswith("gh/"):
        return True
    if "ghstack-source-id" in body:
        return True
    return False


def _is_fork_pr(pr_data: dict) -> bool:
    """True iff the PR head lives in a different repo than the base.

    GitHub only exposes a meaningful ``maintainerCanModify`` for fork PRs;
    for in-repo branches the flag is always false, but anyone with write
    access to the base repo can push to the branch directly.
    """
    owner = (pr_data.get("headRepositoryOwner") or {}).get("login")
    name = (pr_data.get("headRepository") or {}).get("name")
    if not owner or not name:
        return True  # be conservative if metadata is missing
    return f"{owner}/{name}" != REPO_SLUG


def _validate_pr(pr_data: dict) -> None:
    state = pr_data.get("state")
    if state != "OPEN":
        # Closed or merged: nothing to do here ever again. Use the prune
        # exit code so the mux removes us automatically.
        die(
            f"PR is not open (state={state}); pruning local shepherd state",
            code=EXIT_PR_NOT_ACTIONABLE,
        )
    if pr_data.get("isDraft"):
        die("PR is a draft")
    if _is_fork_pr(pr_data) and not pr_data.get("maintainerCanModify"):
        die(
            "'Allow edits by maintainers' is not enabled on this PR; "
            "mergedog cannot push fixes"
        )


def _fork_remote_name(pr_data: dict) -> str:
    """Use the contributor's GitHub login as the remote name.

    Each contributor gets one persistent remote, so ``git remote -v`` stays
    readable across many PRs from the same person.
    """
    owner = (pr_data.get("headRepositoryOwner") or {}).get("login")
    if not owner:
        die("PR head repository owner is missing; can't determine remote name")
    return owner


def _fork_ssh_url(pr_data: dict) -> str:
    owner = (pr_data.get("headRepositoryOwner") or {}).get("login")
    name = (pr_data.get("headRepository") or {}).get("name")
    if not owner or not name:
        die("PR head repository information is missing; can't push fixes")
    return f"git@github.com:{owner}/{name}.git"


def _refresh_context_file(pr_data: dict) -> Path:
    """Rebuild the per-PR sidecar from the latest title/body/comments.

    Refreshed before each claude invocation so that comments added partway
    through a shepherd run show up in the agent's context.
    """
    pr = pr_data["number"]
    comments = github.get_pr_comments(pr)
    text = context_mod.render_context(
        pr=pr,
        url=pr_data.get("url", ""),
        title=pr_data.get("title", ""),
        body=pr_data.get("body", "") or "",
        comments=comments,
    )
    path = context_file(pr)
    context_mod.write_context_file(path, text)
    return path


_APPROVAL_PENDING_STATUSES = {"action_required", "waiting"}


def _needs_approval(run: dict) -> bool:
    """Return True if a workflow run is awaiting maintainer approval.

    Two shapes:
      - ``status: action_required`` (or ``waiting``) -- the run is sitting
        idle, hasn't started.
      - ``status: completed, conclusion: action_required`` -- GitHub closes
        out the placeholder run with ``action_required`` as its conclusion
        and surfaces the "approve and run" button. Approving still moves it.
    """
    if run.get("status") in _APPROVAL_PENDING_STATUSES:
        return True
    if run.get("status") == "completed" and run.get("conclusion") == "action_required":
        return True
    return False


def _wait_for_pr_head(pr: int, expected_sha: str) -> None:
    """Block until ``gh pr view`` reports ``expected_sha`` as PR head.

    GitHub's PR-head ref and its derived APIs (``gh pr checks``,
    ``actions/runs?head_sha=``) lag a push by a few seconds. If we plough
    straight into the polling loop we end up querying the *old* SHA --
    which already has settled CI -- and miss any new approvals/checks
    triggered by the push.
    """
    start = time.time()
    while True:
        current = github.get_pr_head_sha(pr)
        if current == expected_sha:
            log(f"PR head is now {expected_sha[:12]} on GitHub")
            return
        if time.time() - start >= PUSH_VISIBILITY_TIMEOUT_SEC:
            log(
                f"WARNING: timed out waiting for PR head to become "
                f"{expected_sha[:12]} (still reads as {current[:12]}); "
                f"continuing anyway"
            )
            return
        log(f"waiting for PR head {expected_sha[:12]} (still {current[:12]})...")
        time.sleep(3)


def _safe_push(
    pr: int,
    worktree: Path,
    fork_remote: str,
    branch: str,
    new_sha: str,
    *,
    reason: str,
    ignore_sev: bool,
) -> None:
    """Push ``new_sha`` after gating on SEV, then wait for PR head to update.

    ``reason`` is the human-readable verb passed to the SEV-park log
    line, e.g. ``"pushing claude fix commit"``.
    """
    _wait_for_no_active_sev(reason, ignore_sev=ignore_sev)
    repo.push_to_fork(worktree, fork_remote, branch)
    _wait_for_pr_head(pr, new_sha)


def _publish_ghstack_fix(
    pr: int,
    worktree: Path,
    head_ref: str,
    fix_sha: str,
    trust: TrustDB,
    *,
    ignore_sev: bool,
) -> None:
    """Fold claude's [MERGEDOG] commit into /orig and re-publish via ghstack.

    Fixup (not squash): claude's commit message is dropped from the resulting
    /orig commit -- /orig keeps the contributor's original message -- and is
    instead passed to ``ghstack submit -m`` so it lands as the submit's audit
    message. After ghstack pushes, fetch the new synthetic /head SHA from
    origin and trust it before the polling loop sees it on GitHub's side.
    """
    # Capture claude's full [MERGEDOG] message before fixup discards it.
    fix_message = repo.commit_message(worktree, fix_sha)
    repo.fixup_into_parent(worktree)
    _wait_for_no_active_sev(
        "re-publishing via ghstack submit", ignore_sev=ignore_sev
    )
    repo.ghstack_submit(worktree, fix_message)
    new_head_sha = repo.fetch_ghstack_head(head_ref)
    trust.trust(new_head_sha)
    log(
        f"ghstack submitted; new {head_ref} = {new_head_sha[:12]}"
    )
    _wait_for_pr_head(pr, new_head_sha)


def _record_claude_session(
    sessions: list[ClaudeSession],
    *,
    mode: str,
    sha_before: str,
    started_at: str,
    ran_cleanly: bool,
    new_sha: str | None,
    transcript: list[str],
    on_commit: str,
    on_clean_noop: str,
    extra: str = "",
) -> None:
    """Append a :class:`ClaudeSession` summarizing one claude invocation.

    Both ``on_commit`` and ``on_clean_noop`` are the verdict strings shown
    in the handoff comment for this session. ``on_commit`` may contain a
    ``{sha}`` placeholder that gets the new commit's short SHA; the unclean
    case is constant. ``extra`` (if given) is appended to the verdict --
    used to tack on the failing-job names for fix-CI sessions.
    """
    if new_sha:
        verdict = on_commit.format(sha=new_sha[:12])
    elif ran_cleanly:
        verdict = on_clean_noop
    else:
        verdict = "exited with a contract violation"
    if extra:
        verdict += extra
    sessions.append(
        ClaudeSession(
            mode=mode,
            started_at=started_at,
            sha_before=sha_before,
            sha_after=new_sha,
            verdict=verdict,
            transcript=transcript,
        )
    )


def _approve_pending_runs(
    sha: str, run_state_cache: dict[int, tuple[str | None, str | None]]
) -> int:
    """Approve any approval-pending workflow runs.

    ``run_state_cache`` is mutated: it tracks per-run ``(status,
    conclusion)`` from the previous call, so we only log new runs and
    state transitions instead of dumping the full list every poll.
    """
    runs = github.list_workflow_runs_for_sha(sha)
    seen: set[int] = set()
    approved = 0
    for r in runs:
        run_id = r.get("id")
        if run_id is None:
            continue
        seen.add(run_id)
        name = r.get("name") or "?"
        status = r.get("status")
        conclusion = r.get("conclusion")
        prev = run_state_cache.get(run_id)
        cur = (status, conclusion)
        if prev is None:
            log(
                f"workflow run {run_id} {name!r}: status={status} "
                f"conclusion={conclusion}"
            )
        elif prev != cur:
            log(
                f"workflow run {run_id} {name!r}: "
                f"{prev[0]}/{prev[1]} -> {status}/{conclusion}"
            )
        run_state_cache[run_id] = cur
        if _needs_approval(r):
            ok, msg = github.approve_workflow_run(run_id)
            if ok:
                approved += 1
                log(f"  -> approved {run_id} {name!r}")
            else:
                log(f"  -> approve failed for {run_id} {name!r}: {msg}")
    # Forget runs that GitHub no longer reports (e.g. cancelled and dropped).
    for stale in [k for k in run_state_cache if k not in seen]:
        run_state_cache.pop(stale, None)
    return approved


def _merge_main_resolving_conflicts(
    worktree: Path,
    trust: TrustDB,
    branch: str,
    pr_data: dict,
    sessions: list[ClaudeSession],
    *,
    ignore_sev: bool,
) -> str | None:
    """Merge origin/main into HEAD, asking claude to resolve any conflicts.

    Returns the new head SHA if a merge commit was made, else None
    (already up to date). Trusts the new SHA. Caller is responsible
    for pushing.

    Park on a CI SEV BEFORE merging, not after. If we merged first and
    parked second, the merge's base would be hours stale when the SEV
    finally clears -- and likely *missing* the very fix that let trunk
    recover. Refetch origin if we waited so the merge picks up
    everything that landed during the SEV.
    """
    if _wait_for_no_active_sev("merging origin/main", ignore_sev=ignore_sev):
        repo.fetch_origin()

    try:
        status, new_sha = repo.attempt_merge_main(worktree)
    except RuntimeError as e:
        die(str(e))

    if status == "noop":
        log("merge produced no new commit (already up to date)")
        return None

    if status == "conflict":
        log("merge produced conflicts; asking claude to resolve")
        ctx_path = _refresh_context_file(pr_data)
        prompt = render_merge_conflict_prompt(
            url=pr_data.get("url", ""),
            branch=branch,
            context_path=str(ctx_path),
            merge_subject=MERGE_RESOLVED_SUBJECT,
        )
        sha_before = repo.head_sha(worktree)
        started_at = utc_now_iso()
        ran_cleanly, new_sha, transcript = claude_mod.invoke_merge_resolver(
            worktree, prompt
        )
        _record_claude_session(
            sessions,
            mode="merge-resolver",
            sha_before=sha_before,
            started_at=started_at,
            ran_cleanly=ran_cleanly,
            new_sha=new_sha,
            transcript=transcript,
            on_commit="resolved conflicts in commit {sha}",
            on_clean_noop="aborted the merge",
        )
        if not ran_cleanly:
            die("claude failed to resolve the merge conflict cleanly")
        if new_sha is None:
            die("claude aborted the merge; halting for human intervention")

    assert new_sha is not None
    trust.trust(new_sha)
    return new_sha


def _sigterm_to_systemexit(signum, frame) -> None:  # type: ignore[no-untyped-def]
    """Turn SIGTERM into SystemExit so the label-cleanup ``finally`` runs.

    ``mux cancel`` sends SIGTERM to the shepherd's process group; without a
    handler Python exits abruptly and the ``mergedog`` label sticks on the
    PR forever. Raising SystemExit lets the wrapper in ``shepherd`` clean up.
    """
    sys.exit(128 + signum)


def shepherd(
    pr: int,
    rebase: bool = False,
    accept_divergence: bool = False,
    ignore_sev: bool = False,
) -> None:
    repo.ensure_clone()
    repo.fetch_origin()

    pr_data = github.get_pr(pr)
    _validate_pr(pr_data)

    # Past validation: we're committed to running. Tag the PR so other
    # operators / mergedogs see it's already being handled, and arrange for
    # the tag to come off no matter how we exit.
    github.add_label(pr, MERGEDOG_LABEL)
    signal.signal(signal.SIGTERM, _sigterm_to_systemexit)
    try:
        _shepherd_body(pr, pr_data, rebase, accept_divergence, ignore_sev)
    finally:
        try:
            github.remove_label(pr, MERGEDOG_LABEL)
        except Exception as e:
            log(f"WARNING: failed to remove {MERGEDOG_LABEL} label: {e}")


def _shepherd_body(
    pr: int,
    pr_data: dict,
    rebase: bool,
    accept_divergence: bool,
    ignore_sev: bool,
) -> None:
    is_ghstack = _is_ghstack(pr_data)
    branch = pr_data["headRefName"]

    trust = TrustDB.load_or_create(pr)
    trust.head_branch = branch
    if is_ghstack:
        # ghstack PRs live in origin (pytorch/pytorch). The /head ref is the
        # synthetic GitHub-PR commit; the contributor's actual single-commit
        # change lives at the matching /orig ref. We work locally on /orig
        # and re-publish via ``ghstack submit --no-stack``.
        fork_url: str | None = None
        fork_remote: str | None = None
        trust.head_repo_clone_url = REPO_SSH_URL
    else:
        fork_url = _fork_ssh_url(pr_data)
        fork_remote = _fork_remote_name(pr_data)
        trust.head_repo_clone_url = fork_url
    trust.save()

    if not is_ghstack:
        assert fork_remote is not None and fork_url is not None
        repo.add_fork_remote(fork_remote, fork_url)

    viewer = github.viewer_login()
    self_pr = github.is_self_pr(pr_data, viewer)
    if self_pr:
        log(f"PR authored by current user ({viewer}); skipping approval gate")
    seed_trust_from_reviews(
        trust, pr, pr_data, accept_divergence, self_pr=self_pr
    )
    head_sha = pr_data["headRefOid"]

    if is_ghstack:
        # Verify origin's view of /head agrees with what gh reports for the
        # PR -- a sanity check analogous to the fork_sha != head_sha check
        # below. /orig is the actual checkout target.
        origin_head_sha = repo.fetch_ghstack_head(branch)
        if origin_head_sha != head_sha:
            die(
                f"origin's {branch} ({origin_head_sha[:12]}) differs from "
                f"the SHA GitHub reports for the PR ({head_sha[:12]}); "
                f"refusing to act"
            )
        orig_sha = repo.fetch_ghstack_orig(branch)
        worktree = repo.ensure_worktree(pr, orig_sha)
    else:
        assert fork_remote is not None
        fork_sha = repo.fetch_pr_branch(fork_remote, branch)
        if fork_sha != head_sha:
            die(
                f"contributor's fork HEAD ({fork_sha[:12]}) differs from the SHA "
                f"GitHub reports for the PR ({head_sha[:12]}); refusing to act"
            )
        worktree = repo.ensure_worktree(pr, head_sha, fork_remote, branch)

    log(f"shepherding PR #{pr}: {pr_data.get('title', '')}")
    log(f"  url:        {pr_data.get('url', '')}")
    log(f"  branch:     {branch}")
    if is_ghstack:
        log(f"  ghstack:    /head {head_sha[:12]} -> /orig {orig_sha[:12]}")
    else:
        log(f"  fork:       {fork_remote} -> {fork_url}")
    log(f"  worktree:   {worktree}")

    fix_commits_pushed = 0
    sessions: list[ClaudeSession] = []

    # User-requested upfront merge of origin/main. Default behavior
    # otherwise is to never auto-rebase based on age -- mergedog only
    # merges main when piggybacking on a fix push it was going to do
    # anyway, or when the operator explicitly asks via ``--rebase``.
    if rebase:
        if is_ghstack:
            # Auto-rebase for ghstack would mean rebasing /orig onto
            # origin/main and re-submitting. Not yet wired; for now the
            # operator handles base management with ``ghstack`` directly.
            log("--rebase ignored for ghstack PRs (not yet wired)")
        else:
            log("user requested upfront rebase onto origin/main")
            new_sha = _merge_main_resolving_conflicts(
                worktree, trust, branch, pr_data, sessions, ignore_sev=ignore_sev
            )
            if new_sha is not None:
                log(f"pushing merge commit {new_sha[:12]} to {fork_remote}/{branch}")
                _safe_push(
                    pr, worktree, fork_remote, branch, new_sha,
                    reason="pushing merge-main commit", ignore_sev=ignore_sev,
                )

    trunk_applied = github.has_label(pr_data, TRUNK_LABEL)
    run_state_cache: dict[int, tuple[str | None, str | None]] = {}
    last_status: str | None = None
    # (status, check_count) we last observed. Becomes the anchor for the
    # stability window: when it changes (new check arrives, status flips),
    # we reset the timer.
    stable_observation: tuple[str, int] | None = None
    stable_since: float = 0.0

    # Poll CI, fix or judge spurious until ready for handoff. Breaks out
    # (via the handoff path) when CI is green and the trunk label is on.
    while True:
        # 1. Verify the PR head is still trusted.
        current = github.get_pr_head_sha(pr)
        if self_pr:
            # On a self-authored PR, every push is implicitly approved by
            # the operator -- roll the trust forward instead of halting.
            trust.trust(current)
        if not trust.is_trusted(current):
            subject = github.get_commit_subject(current)
            die(
                f"PR head moved to untrusted commit {current[:12]}: "
                f"{subject!r}. Manual intervention required."
            )

        # 2. Approve any approval-pending workflow runs.
        approved = _approve_pending_runs(current, run_state_cache)
        if approved:
            stable_observation = None  # newly-approved runs invalidate stability
            time.sleep(APPROVAL_SETTLE_SEC)
            continue

        # 3. Read check status.
        checks = github.get_pr_checks_all(pr)
        status = github.evaluate_checks(checks)
        done = sum(
            1 for c in checks if c.get("bucket") not in {"pending", None}
        )
        summary = f"{status} ({done}/{len(checks)} done)"
        if summary != last_status:
            log(f"CI status -> {summary}")
            last_status = summary

        # Track stability: any change in (status, check_count) restarts the
        # quiescence timer. We only gate on it for the "passed" verdict, so
        # genuine pending/failed states proceed without artificial delay.
        observation = (status, len(checks))
        if stable_observation != observation:
            stable_observation = observation
            stable_since = time.time()

        if status == "pending":
            time.sleep(POLL_INTERVAL_SEC)
            continue

        if status == "failed":
            # Hand the failures to claude; only advance once claude is OK
            # with the situation (either pushed a fix or judged spurious).
            if fix_commits_pushed >= MAX_FIX_COMMITS:
                die(
                    f"already pushed {fix_commits_pushed} [MERGEDOG] fix "
                    f"commits and CI is still failing; halting for human "
                    f"intervention"
                )
            failed = github.get_failed_job_logs(pr)
            ctx_path = _refresh_context_file(pr_data)
            prompt = render_fix_prompt(
                url=pr_data.get("url", ""),
                branch=branch,
                context_path=str(ctx_path),
                failed_jobs=failed,
                is_ghstack=is_ghstack,
            )
            session_failed_jobs = [name for name, _ in failed]
            sha_before = current
            started_at = utc_now_iso()
            ran_cleanly, new_sha, transcript = claude_mod.invoke_fixer(
                worktree, prompt
            )
            _record_claude_session(
                sessions,
                mode="fix-CI",
                sha_before=sha_before,
                started_at=started_at,
                ran_cleanly=ran_cleanly,
                new_sha=new_sha,
                transcript=transcript,
                on_commit="pushed fix commit {sha}",
                on_clean_noop="judged failures spurious (no commit)",
                extra=(
                    f" — failing jobs: {', '.join(session_failed_jobs)}"
                    if session_failed_jobs
                    else ""
                ),
            )
            if not ran_cleanly:
                die("claude exited abnormally or produced an invalid commit")
            if new_sha is None:
                log("claude judged failures spurious; advancing")
                last_status = None  # force re-log on next pass
                # fall through to advance below
            elif is_ghstack:
                _publish_ghstack_fix(
                    pr, worktree, branch, new_sha, trust,
                    ignore_sev=ignore_sev,
                )
                fix_commits_pushed += 1
                last_status = None
                continue
            else:
                assert fork_remote is not None
                trust.trust(new_sha)
                # Piggyback: we're going to push and trigger fresh CI
                # anyway, so merge origin/main while we're at it. CI then
                # runs once on (PR + fix + main) instead of testing the
                # fix against a stale base.
                merge_sha = _merge_main_resolving_conflicts(
                    worktree, trust, branch, pr_data, sessions,
                    ignore_sev=ignore_sev,
                )
                final_sha = merge_sha if merge_sha is not None else new_sha
                log(
                    f"pushing {final_sha[:12]} to {fork_remote}/{branch}"
                )
                _safe_push(
                    pr, worktree, fork_remote, branch, final_sha,
                    reason="pushing claude fix commit",
                    ignore_sev=ignore_sev,
                )
                fix_commits_pushed += 1
                last_status = None
                continue

        # CI is "passed". Require it to have been passed continuously for
        # CI_STABILITY_WINDOW_SEC before we act, so that a freshly-pushed
        # commit can't trick us by reporting "1/1 done" while the rest of
        # the required workflows are still being created.
        if status == "passed":
            elapsed = time.time() - stable_since
            if elapsed < CI_STABILITY_WINDOW_SEC:
                remaining = int(CI_STABILITY_WINDOW_SEC - elapsed)
                log(
                    f"CI passed; waiting {remaining}s for stability "
                    f"(no new checks should appear)"
                )
                time.sleep(min(POLL_INTERVAL_SEC, remaining))
                continue

        # Either CI passed (and is stable), or claude said "spurious". Advance.
        if not trunk_applied:
            # Adding ciflow/trunk kicks off a fresh wave of trunk
            # workflows; gate on SEV so we don't pile on broken trunk.
            _wait_for_no_active_sev(
                f"applying {TRUNK_LABEL} label", ignore_sev=ignore_sev
            )
            log(f"required CI green; applying {TRUNK_LABEL} label")
            github.add_label(pr, TRUNK_LABEL)
            trunk_applied = True
            last_status = None
            time.sleep(APPROVAL_SETTLE_SEC)
            continue
        log("ALL CI GREEN.")
        break

    post_handoff_comment(pr, pr_data, sessions)
    # Anchor the watch loop on the actual handoff comment timestamp,
    # not "now": on restart this lets us notice a "Merge failed" that
    # already happened between the last handoff and our restart. But also
    # floor on any failure we've already halted on, so the next restart
    # doesn't re-react to the same stale comment.
    handoff_iso = github.latest_mergedog_handoff_iso(pr) or utc_now_iso()
    since_iso = max(handoff_iso, trust.last_observed_failure_iso)
    log(
        f"Hand off to a human reviewer; have them comment "
        f"`@pytorchbot merge` on {pr_data.get('url', f'PR #{pr}')}."
    )

    result, event_iso = watch_post_handoff(pr, since_iso)
    if result == "closed":
        die(
            "PR is no longer open; pruning local shepherd state",
            code=EXIT_PR_NOT_ACTIONABLE,
        )
    # result == "failed": pytorchmergebot rejected the merge. We don't
    # auto-remediate -- a merge failure could mean stale base, post-land
    # hooks, conflicts, or anything else, and a rebase is only one
    # possible response. Persist the failure timestamp so a future restart
    # won't re-fire on this same comment, then halt for human intervention.
    assert event_iso is not None
    trust.last_observed_failure_iso = event_iso
    trust.save()
    die(
        "pytorchmergebot reported merge failure; halting for human "
        "intervention (re-run with --rebase if a stale base is the cause)"
    )
