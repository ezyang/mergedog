"""The main mergedog shepherding loop.

One process per PR. Synchronous. Halts on any sign of an untrusted change.
"""
from __future__ import annotations

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
from mergedog.paths import REPO_SLUG, context_file
from mergedog.prompts import render_fix_prompt, render_merge_conflict_prompt
from mergedog.repo import MERGE_RESOLVED_SUBJECT
from mergedog.state import TrustDB
from mergedog.trust_seed import seed_trust_from_reviews


SEV_POLL_INTERVAL_SEC = 5 * 60  # SEVs are minutes-to-hours; don't spam ``gh``


def _wait_for_no_active_sev(reason: str, *, ignore_sev: bool) -> None:
    """If pytorch CI has an open SEV, block until it clears.

    A CI SEV here is any open issue on pytorch/pytorch tagged
    ``ci: sev`` -- dev-infra's signal that trunk is degraded. Default
    behavior is to wait it out so we don't stampede broken CI with
    new pushes; ``ignore_sev`` (operator override via ``--ignore-sev``)
    skips the wait. Called only at "would trigger CI" critical spots,
    not in the inner poll, to keep the GH API call rate low.
    """
    if ignore_sev:
        return
    last_ids: tuple[int, ...] | None = None
    while True:
        sevs = github.list_active_ci_sevs()
        if not sevs:
            if last_ids is not None:
                log("CI SEV cleared; resuming")
            return
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
MAX_FIX_COMMITS = 5  # safety cap; halt if claude keeps pushing fixes
DEFAULT_MAX_BASE_AGE_DAYS = 7


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
    if _is_ghstack(pr_data):
        die("ghstack PRs are not supported")
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


def _maybe_merge_main(
    worktree: Path,
    trust: TrustDB,
    fork_remote: str,
    branch: str,
    max_base_age_days: int,
    pr_data: dict,
    sessions: list[ClaudeSession],
    *,
    ignore_sev: bool,
) -> str | None:
    """If the PR's merge-base is older than the threshold, merge origin/main.

    On a clean merge, push it. On a conflict, hand off to claude to resolve;
    if claude can't, halt. Returns the new head SHA if a merge happened.
    Records any claude invocation into ``sessions`` for the handoff comment.
    """
    age_sec = repo.merge_base_age_seconds(worktree)
    age_days = age_sec / 86400.0
    log(f"merge-base with origin/main is {age_days:.1f} days old")
    if age_days <= max_base_age_days:
        log("PR is fresh enough; not merging main")
        return None
    log(f"merge-base older than {max_base_age_days} days; merging origin/main")

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
    log(f"pushing merge commit {new_sha[:12]} to {fork_remote}/{branch}")
    _safe_push(
        pr_data["number"],
        worktree,
        fork_remote,
        branch,
        new_sha,
        reason="pushing merge-main commit",
        ignore_sev=ignore_sev,
    )
    return new_sha


def shepherd(
    pr: int,
    max_base_age_days: int = DEFAULT_MAX_BASE_AGE_DAYS,
    accept_divergence: bool = False,
    ignore_sev: bool = False,
) -> None:
    repo.ensure_clone()
    repo.fetch_origin()

    pr_data = github.get_pr(pr)
    _validate_pr(pr_data)

    trust = TrustDB.load_or_create(pr)
    trust.head_branch = pr_data["headRefName"]
    fork_url = _fork_ssh_url(pr_data)
    trust.head_repo_clone_url = fork_url
    trust.save()

    fork_remote = _fork_remote_name(pr_data)
    repo.add_fork_remote(fork_remote, fork_url)

    seed_trust_from_reviews(trust, pr, pr_data, accept_divergence)
    head_sha = pr_data["headRefOid"]

    fork_sha = repo.fetch_pr_branch(fork_remote, pr_data["headRefName"])
    if fork_sha != head_sha:
        die(
            f"contributor's fork HEAD ({fork_sha[:12]}) differs from the SHA "
            f"GitHub reports for the PR ({head_sha[:12]}); refusing to act"
        )

    worktree = repo.ensure_worktree(
        pr, head_sha, fork_remote, pr_data["headRefName"]
    )

    log(f"shepherding PR #{pr}: {pr_data.get('title', '')}")
    log(f"  url:        {pr_data.get('url', '')}")
    log(f"  branch:     {pr_data['headRefName']}")
    log(f"  fork:       {fork_remote} -> {fork_url}")
    log(f"  worktree:   {worktree}")

    fix_commits_pushed = 0
    sessions: list[ClaudeSession] = []
    cycle = 0
    effective_max_base_age = max_base_age_days

    while True:  # outer cycle loop -- restarts after pytorchmergebot failure
        cycle += 1
        if cycle > 1:
            # Recovery cycle: re-fetch origin so the merge picks up any new
            # main, and re-validate the PR. If the contributor pushed an
            # untrusted commit during the wait, the head-trust check below
            # will halt us.
            repo.fetch_origin()
            pr_data = github.get_pr(pr)
            _validate_pr(pr_data)
            log(f"--- recovery cycle #{cycle} (forcing rebase onto main) ---")

        trunk_applied = github.has_label(pr_data, TRUNK_LABEL)
        main_merged = False
        run_state_cache: dict[int, tuple[str | None, str | None]] = {}
        last_status: str | None = None
        # (status, check_count) we last observed. Becomes the anchor for the
        # stability window: when it changes (new check arrives, status flips),
        # we reset the timer.
        stable_observation: tuple[str, int] | None = None
        stable_since: float = 0.0

        # Inner loop: poll CI, fix or judge spurious until ready for handoff.
        # Breaks out (via the handoff path) when CI is green and labels/merges
        # are settled.
        while True:
            # 1. Verify the PR head is still trusted.
            current = github.get_pr_head_sha(pr)
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
                    branch=pr_data["headRefName"],
                    context_path=str(ctx_path),
                    failed_jobs=failed,
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
                else:
                    trust.trust(new_sha)
                    log(
                        f"pushing {new_sha[:12]} to "
                        f"{fork_remote}/{pr_data['headRefName']}"
                    )
                    _safe_push(
                        pr,
                        worktree,
                        fork_remote,
                        pr_data["headRefName"],
                        new_sha,
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
            if not main_merged:
                new_sha = _maybe_merge_main(
                    worktree,
                    trust,
                    fork_remote,
                    pr_data["headRefName"],
                    effective_max_base_age,
                    pr_data,
                    sessions,
                    ignore_sev=ignore_sev,
                )
                main_merged = True
                if new_sha is not None:
                    # New CI will run on the merged commit; back to polling.
                    last_status = None
                    continue
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
            break  # leave the inner loop, post handoff and watch

        # Recovery cycles re-post so the new session blocks (merge-main
        # commit, follow-up claude judgments) are visible on the PR.
        post_handoff_comment(pr, pr_data, sessions, force=cycle > 1)
        # Anchor the watch loop on the actual handoff comment timestamp,
        # not "now": on restart this lets us notice a "Merge failed" that
        # already happened between the last handoff and our restart.
        handoff_iso = github.latest_mergedog_handoff_iso(pr) or utc_now_iso()
        log(
            f"Hand off to a human reviewer; have them comment "
            f"`@pytorchbot merge` on {pr_data.get('url', f'PR #{pr}')}."
        )

        result = watch_post_handoff(pr, handoff_iso)
        if result == "closed":
            die(
                "PR is no longer open; pruning local shepherd state",
                code=EXIT_PR_NOT_ACTIONABLE,
            )
        # result == "failed": pytorchmergebot rejected the merge. Force a
        # rebase onto current main (max-base-age 0 makes _maybe_merge_main
        # always merge) and run the cycle again.
        effective_max_base_age = 0
