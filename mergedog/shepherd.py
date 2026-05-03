"""The main mergedog shepherding loop.

One process per PR. Synchronous. Halts on any sign of an untrusted change.
"""
from __future__ import annotations

import time
from pathlib import Path

from mergedog import claude as claude_mod
from mergedog import github, repo
from mergedog.log import die, log
from mergedog.prompts import render_fix_prompt, render_merge_conflict_prompt
from mergedog.repo import MERGE_COMMIT_SUBJECT
from mergedog.state import TrustDB

POLL_INTERVAL_SEC = 60
APPROVAL_SETTLE_SEC = 15
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


def _validate_pr(pr_data: dict) -> None:
    if pr_data.get("state") != "OPEN":
        die(f"PR is not open (state={pr_data.get('state')})")
    if pr_data.get("isDraft"):
        die("PR is a draft")
    if _is_ghstack(pr_data):
        die("ghstack PRs are not supported")
    if not pr_data.get("maintainerCanModify"):
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


def _approve_pending_runs(sha: str) -> int:
    runs = github.list_workflow_runs_for_sha(sha)
    log(f"workflow runs for {sha[:12]}: {len(runs)}")
    approved = 0
    for r in runs:
        run_id = r.get("id")
        name = r.get("name") or "?"
        status = r.get("status")
        conclusion = r.get("conclusion")
        log(f"  run {run_id} {name!r} status={status} conclusion={conclusion}")
        if _needs_approval(r) and run_id is not None:
            ok, msg = github.approve_workflow_run(run_id)
            if ok:
                approved += 1
                log(f"    -> approved")
            else:
                log(f"    -> approve failed: {msg}")
    return approved


def _maybe_merge_main(
    worktree: Path,
    trust: TrustDB,
    fork_remote: str,
    branch: str,
    max_base_age_days: int,
    pr_data: dict,
) -> str | None:
    """If the PR's merge-base is older than the threshold, merge origin/main.

    On a clean merge, push it. On a conflict, hand off to claude to resolve;
    if claude can't, halt. Returns the new head SHA if a merge happened.
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
        prompt = render_merge_conflict_prompt(
            title=pr_data.get("title", ""),
            url=pr_data.get("url", ""),
            branch=branch,
            merge_subject=MERGE_COMMIT_SUBJECT,
        )
        ran_cleanly, new_sha = claude_mod.invoke_merge_resolver(worktree, prompt)
        if not ran_cleanly:
            die("claude failed to resolve the merge conflict cleanly")
        if new_sha is None:
            die("claude aborted the merge; halting for human intervention")

    assert new_sha is not None
    trust.trust(new_sha)
    log(f"pushing merge commit {new_sha[:12]} to {fork_remote}/{branch}")
    repo.push_to_fork(worktree, fork_remote, branch)
    return new_sha


def shepherd(pr: int, max_base_age_days: int = DEFAULT_MAX_BASE_AGE_DAYS) -> None:
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

    head_sha = pr_data["headRefOid"]
    trust.trust(head_sha)

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
    log(f"  url:           {pr_data.get('url', '')}")
    log(f"  approval SHA:  {head_sha}")
    log(f"  branch:        {pr_data['headRefName']}")
    log(f"  fork remote:   {fork_remote} -> {fork_url}")
    log(f"  worktree:      {worktree}")

    _maybe_merge_main(
        worktree,
        trust,
        fork_remote,
        pr_data["headRefName"],
        max_base_age_days,
        pr_data,
    )

    trunk_applied = github.has_label(pr_data, TRUNK_LABEL)
    fix_commits_pushed = 0

    while True:
        # 1. Verify the PR head is still trusted.
        current = github.get_pr_head_sha(pr)
        if not trust.is_trusted(current):
            subject = github.get_commit_subject(current)
            die(
                f"PR head moved to untrusted commit {current[:12]}: "
                f"{subject!r}. Manual intervention required."
            )

        # 2. Approve any first-time-contributor workflow runs.
        approved = _approve_pending_runs(current)
        if approved:
            time.sleep(APPROVAL_SETTLE_SEC)
            continue

        # 3. Read CI status (required checks only).
        checks = github.get_pr_checks(pr)
        status = github.evaluate_checks(checks)
        log(f"CI status: {status} ({len(checks)} required checks)")

        if status == "pending":
            time.sleep(POLL_INTERVAL_SEC)
            continue

        if status == "passed":
            if not trunk_applied:
                log(f"required CI green; applying {TRUNK_LABEL} label")
                github.add_label(pr, TRUNK_LABEL)
                trunk_applied = True
                time.sleep(APPROVAL_SETTLE_SEC)
                continue
            log("ALL CI GREEN.")
            log(
                f"Hand off to a human reviewer; have them comment "
                f"`@pytorchbot merge` on {pr_data.get('url', f'PR #{pr}')}."
            )
            return

        # status == "failed": invoke claude.
        if fix_commits_pushed >= MAX_FIX_COMMITS:
            die(
                f"already pushed {fix_commits_pushed} [MERGEDOG] fix commits "
                f"and CI is still failing; halting for human intervention"
            )

        failed = github.get_failed_job_logs(pr)
        prompt = render_fix_prompt(
            title=pr_data.get("title", ""),
            url=pr_data.get("url", ""),
            branch=pr_data["headRefName"],
            failed_jobs=failed,
        )
        ran_cleanly, new_sha = claude_mod.invoke_fixer(worktree, prompt)
        if not ran_cleanly:
            die("claude exited abnormally or produced an invalid commit")

        if new_sha is None:
            # Claude judged the failures spurious. Advance: either apply
            # ciflow/trunk if we haven't yet, or declare done.
            if not trunk_applied:
                log(
                    f"claude judged failures spurious; applying {TRUNK_LABEL} "
                    f"to advance"
                )
                github.add_label(pr, TRUNK_LABEL)
                trunk_applied = True
                time.sleep(APPROVAL_SETTLE_SEC)
                continue
            log("ALL CI EFFECTIVELY GREEN (claude judged remaining failures spurious).")
            log(
                f"Hand off to a human reviewer; have them comment "
                f"`@pytorchbot merge` on {pr_data.get('url', f'PR #{pr}')}."
            )
            return

        # Claude made a fix commit; trust it and push.
        trust.trust(new_sha)
        log(f"pushing {new_sha[:12]} to {fork_remote}/{pr_data['headRefName']}")
        repo.push_to_fork(worktree, fork_remote, pr_data["headRefName"])
        fix_commits_pushed += 1
        # New CI runs will fire on the next iteration.
        time.sleep(APPROVAL_SETTLE_SEC)
