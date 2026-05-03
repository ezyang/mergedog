"""The main mergedog shepherding loop.

One process per PR. Synchronous. Halts on any sign of an untrusted change.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from mergedog import claude as claude_mod
from mergedog import context as context_mod
from mergedog import github, repo
from mergedog.log import die, log
from mergedog.paths import context_file
from mergedog.prompts import render_fix_prompt, render_merge_conflict_prompt
from mergedog.repo import MERGE_RESOLVED_SUBJECT
from mergedog.state import TrustDB


@dataclass
class _ClaudeSession:
    """One claude invocation, captured for the handoff comment."""

    mode: str  # "fix-CI" or "merge-resolver"
    started_at: str  # UTC ISO 8601
    sha_before: str
    sha_after: str | None  # ``None`` when claude judged the situation a no-op
    verdict: str  # human-readable summary
    transcript: list[str] = field(default_factory=list)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# GitHub PR comments cap out at 65,536 characters. We leave headroom for
# our own framing and a "[truncated]" tail.
_MAX_COMMENT_LEN = 60000


def _format_handoff_comment(pr_data: dict, sessions: list["_ClaudeSession"]) -> str:
    """Build the markdown body posted on the PR at handoff."""
    n = len(sessions)
    head: list[str] = [
        "## mergedog handoff",
        "",
        "All required CI is green. Ready for human review and "
        "`@pytorchbot merge`.",
        "",
    ]
    if n == 0:
        head.append(
            "claude was not invoked during this run (CI was green from the "
            "start; no merge or fix needed)."
        )
        return "\n".join(head) + "\n"

    head.append(
        f"During shepherding, claude was invoked **{n}** time"
        f"{'' if n == 1 else 's'}. If any required CI shows red, claude "
        "judged that failure unrelated to this PR's changes — please verify "
        "below before merging."
    )
    head.append("")
    body = "\n".join(head)

    for i, s in enumerate(sessions, 1):
        section = [
            f"### Session {i} — {s.mode} ({s.started_at})",
            "",
            f"- **Before:** `{s.sha_before[:12]}`",
        ]
        if s.sha_after:
            section.append(f"- **After:** `{s.sha_after[:12]}`")
        section.append(f"- **Verdict:** {s.verdict}")
        section.append("")
        section.append("<details><summary>claude transcript</summary>")
        section.append("")
        section.append("```")
        section.extend(s.transcript)
        section.append("```")
        section.append("")
        section.append("</details>")
        section.append("")
        body += "\n" + "\n".join(section)

    if len(body) > _MAX_COMMENT_LEN:
        body = (
            body[:_MAX_COMMENT_LEN]
            + "\n\n_[truncated to fit GitHub's comment limit; full transcripts "
            "live in `~/.mergedog/logs/" + str(pr_data.get("number")) + ".log` "
            "on the operator's machine]_"
        )
    return body


def _post_handoff_comment(
    pr: int, pr_data: dict, sessions: list["_ClaudeSession"]
) -> None:
    body = _format_handoff_comment(pr_data, sessions)
    try:
        github.post_pr_comment(pr, body)
        log(f"posted handoff summary to PR #{pr}")
    except Exception as e:
        # Don't halt on comment failure -- shepherding is otherwise complete.
        log(f"WARNING: failed to post handoff comment: {e}")

POLL_INTERVAL_SEC = 60
APPROVAL_SETTLE_SEC = 15
PUSH_VISIBILITY_TIMEOUT_SEC = 90
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


def _seed_trust_from_reviews(
    trust: TrustDB, pr: int, pr_data: dict, accept_divergence: bool
) -> None:
    """Establish the initial trusted SHA from GitHub PR reviews.

    Queries the PR's reviews via GraphQL and finds the most recent
    APPROVED review by an author with a *trusted* association
    (MEMBER / COLLABORATOR / OWNER -- i.e. someone with push access,
    not a drive-by approval from any GitHub user). The ``commit_id`` of
    that approval is the trust seed.

    If the PR's current head differs from the approval SHA and the
    head isn't already in the trust DB (e.g. from a previous mergedog
    run that pushed [MERGEDOG] commits), we halt -- the contributor has
    pushed unblessed work after the approval. ``--accept-divergence``
    overrides this for cases where the operator has personally
    re-reviewed the new commits.
    """
    audit = github.get_pr_review_audit(pr)
    decision = audit.get("decision")
    log(f"PR review decision: {decision}")

    untrusted_approvals = [
        r for r in audit["reviews"]
        if r.get("state") == "APPROVED"
        and not github.is_trusted_association(r.get("association"))
    ]
    for r in untrusted_approvals:
        log(
            f"  ignoring APPROVED review by {r.get('login')!r} "
            f"(association={r.get('association')!r}, not a maintainer)"
        )

    trusted_approvals = [
        r for r in audit["reviews"]
        if r.get("state") == "APPROVED"
        and github.is_trusted_association(r.get("association"))
        and r.get("commit_id")
    ]
    if not trusted_approvals:
        die(
            "no APPROVED review from a maintainer "
            "(MEMBER/COLLABORATOR/OWNER) was found on this PR; "
            "halting. Get a real approval before running mergedog."
        )

    trusted_approvals.sort(key=lambda r: r.get("submitted_at") or "")
    latest = trusted_approvals[-1]
    approval_sha: str = latest["commit_id"]
    log(
        f"latest maintainer approval: {latest['login']} "
        f"({latest['association']}) at {approval_sha[:12]} "
        f"on {latest.get('submitted_at')}"
    )

    if decision != "APPROVED":
        die(
            f"PR review decision is {decision!r}, not APPROVED "
            f"(another reviewer may have requested changes after the "
            f"latest approval). Halting."
        )

    trust.trust(approval_sha)

    head_sha = pr_data["headRefOid"]
    if head_sha == approval_sha:
        log(f"current PR head matches the approval SHA")
    elif trust.is_trusted(head_sha):
        log(
            f"current PR head {head_sha[:12]} is already in our trust DB "
            f"(presumably a [MERGEDOG] commit from a previous run); "
            f"approval seed at {approval_sha[:12]}"
        )
    elif accept_divergence:
        log(
            f"WARNING: current PR head {head_sha[:12]} differs from the "
            f"latest maintainer approval at {approval_sha[:12]}. "
            f"--accept-divergence given; trusting head as well."
        )
        trust.trust(head_sha)
    else:
        die(
            f"current PR head {head_sha[:12]} differs from the latest "
            f"maintainer approval at {approval_sha[:12]} (by "
            f"{latest['login']}). The contributor pushed unreviewed "
            f"commits after approval. Rerun with --accept-divergence "
            f"after re-reviewing, or get a fresh approval."
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
    sessions: list[_ClaudeSession],
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
        started_at = _utc_now_iso()
        ran_cleanly, new_sha, transcript = claude_mod.invoke_merge_resolver(
            worktree, prompt
        )
        verdict = (
            f"resolved conflicts in commit {new_sha[:12]}"
            if new_sha
            else (
                "aborted the merge"
                if ran_cleanly
                else "exited with a contract violation"
            )
        )
        sessions.append(
            _ClaudeSession(
                mode="merge-resolver",
                started_at=started_at,
                sha_before=sha_before,
                sha_after=new_sha,
                verdict=verdict,
                transcript=transcript,
            )
        )
        if not ran_cleanly:
            die("claude failed to resolve the merge conflict cleanly")
        if new_sha is None:
            die("claude aborted the merge; halting for human intervention")

    assert new_sha is not None
    trust.trust(new_sha)
    log(f"pushing merge commit {new_sha[:12]} to {fork_remote}/{branch}")
    repo.push_to_fork(worktree, fork_remote, branch)
    _wait_for_pr_head(pr_data["number"], new_sha)
    return new_sha


def shepherd(
    pr: int,
    max_base_age_days: int = DEFAULT_MAX_BASE_AGE_DAYS,
    accept_divergence: bool = False,
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

    _seed_trust_from_reviews(trust, pr, pr_data, accept_divergence)
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

    trunk_applied = github.has_label(pr_data, TRUNK_LABEL)
    main_merged = False
    fix_commits_pushed = 0
    run_state_cache: dict[int, tuple[str | None, str | None]] = {}
    last_status: str | None = None
    # (status, check_count) we last observed. Becomes the anchor for the
    # stability window: when it changes (new check arrives, status flips),
    # we reset the timer.
    stable_observation: tuple[str, int] | None = None
    stable_since: float = 0.0
    sessions: list[_ClaudeSession] = []

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
            started_at = _utc_now_iso()
            ran_cleanly, new_sha, transcript = claude_mod.invoke_fixer(
                worktree, prompt
            )
            verdict = (
                f"pushed fix commit {new_sha[:12]}"
                if new_sha
                else (
                    "judged failures spurious (no commit)"
                    if ran_cleanly
                    else "exited with a contract violation"
                )
            )
            if session_failed_jobs:
                verdict += f" — failing jobs: {', '.join(session_failed_jobs)}"
            sessions.append(
                _ClaudeSession(
                    mode="fix-CI",
                    started_at=started_at,
                    sha_before=sha_before,
                    sha_after=new_sha,
                    verdict=verdict,
                    transcript=transcript,
                )
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
                repo.push_to_fork(worktree, fork_remote, pr_data["headRefName"])
                fix_commits_pushed += 1
                _wait_for_pr_head(pr, new_sha)
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
                max_base_age_days,
                pr_data,
                sessions,
            )
            main_merged = True
            if new_sha is not None:
                # New CI will run on the merged commit; back to polling.
                last_status = None
                continue
        if not trunk_applied:
            log(f"required CI green; applying {TRUNK_LABEL} label")
            github.add_label(pr, TRUNK_LABEL)
            trunk_applied = True
            last_status = None
            time.sleep(APPROVAL_SETTLE_SEC)
            continue
        log("ALL CI GREEN.")
        _post_handoff_comment(pr, pr_data, sessions)
        log(
            f"Hand off to a human reviewer; have them comment "
            f"`@pytorchbot merge` on {pr_data.get('url', f'PR #{pr}')}."
        )
        return
