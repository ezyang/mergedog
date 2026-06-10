"""The main mergedog shepherding loop.

One process per PR. Synchronous. Halts on any sign of an untrusted change.
"""
from __future__ import annotations

import faulthandler
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mergedog import claude as claude_mod
from mergedog import context as context_mod
from mergedog import github, interventions, labels, repo
from mergedog.config import get_llm_config
from mergedog.handoff import (
    ClaudeSession,
    PushedChange,
    is_merge_conflict_failure,
    is_retryable_merge_failure,
    mergebot_ignored_check_names,
    post_handoff_comment,
    suppression_drci_status_warning,
    watch_post_handoff,
)
from mergedog.head_trust import trust_mergebot_rebase_if_equivalent
from mergedog.log import (
    complete,
    configure_status_pr,
    die,
    log,
    set_approved,
    set_merging,
)
from mergedog.paths import REPO_SLUG, REPO_SSH_URL, context_file
from mergedog.project import get_project_policy
from mergedog.prompts import (
    render_cherry_pick_conflict_prompt,
    render_fix_prompt,
    render_merge_conflict_prompt,
    render_operator_fix_prompt,
    render_rebase_conflict_prompt,
)
from mergedog.repo import MERGE_RESOLVED_SUBJECT
from mergedog.state import TrustDB
from mergedog.status import utc_now_iso, write_status
from mergedog.trust_seed import seed_trust_from_reviews


SEV_POLL_INTERVAL_SEC = 5 * 60  # SEVs are minutes-to-hours; don't spam ``gh``


def _fetch_current_pr_head_for_trust(
    pr: int, trust: TrustDB, current_sha: str
) -> bool:
    if not trust.head_repo_clone_url or not trust.head_branch:
        return False
    local_ref = f"refs/remotes/mergedog-trust/{pr}"
    fetched = repo.fetch_branch_from_url(
        trust.head_repo_clone_url, trust.head_branch, local_ref
    )
    if fetched != current_sha:
        log(
            f"PR head moved again while validating mergebot rebase: "
            f"GitHub reported {current_sha[:12]}, fetched {fetched[:12]}"
        )
        return False
    return True


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
# (closed, merged, etc.). The mux keeps these completed sessions visible
# until the operator runs ``cleanup``.
EXIT_PR_NOT_ACTIONABLE = 42
# How long ``(status, check_count)`` must hold steady before we trust a
# "passed" verdict. Right after a push, GitHub registers the workflow runs
# over a span of seconds; without this window we'd see "1/1 done -> passed"
# before the other 10+ required workflows even exist, and slap the trunk
# label on prematurely.
CI_STABILITY_WINDOW_SEC = 60
PROJECT = get_project_policy()
TRUNK_LABEL = PROJECT.trunk_label
# Marker label so humans can see at a glance which PRs already have a live
# mergedog shepherding them -- keeps two operators (or two mergedogs) from
# fighting over the same PR. Added after validation passes; removed on every
# exit path (success, HALT, SIGTERM from ``mux cancel``, ctrl-c).
MERGEDOG_LABEL = "mergedog"
MAX_FIX_COMMITS = 5  # safety cap; halt if claude keeps pushing fixes
MAX_MERGE_AUTO_RETRIES = 3  # cap retries for infra-flake merge failures
# When CI flips to failed, ``gh run view --log-failed`` often returns nothing
# for the first few seconds because GitHub hasn't published the log yet.
# Calling claude on a content-free prompt nearly guarantees a "spurious"
# verdict, so we defer the invocation up to this many poll cycles waiting
# for logs to appear. Past the cap, we invoke claude anyway -- the trusted
# failing-check-name list in the prompt still gives it something to act on.
MAX_EMPTY_LOG_DEFERS = 3
# Below this many post-strip chars (across all failing jobs combined) the
# prompt's logs section is considered content-free.
MIN_USEFUL_LOG_CHARS = 200


@dataclass
class _GhstackParentDependency:
    parent_pr: int
    parent_head_ref: str
    parent_orig_ref: str
    child_head_ref: str
    child_orig_ref: str
    stable_observation: tuple[str, int] | None = None
    stable_since: float = 0.0
    last_log_state: str | None = None


@dataclass
class _GhstackParentStatus:
    stale: bool
    parent_ready: bool
    parent_status: str
    parent_done: int
    parent_total: int
    parent_orig_sha: str
    child_orig_sha: str
    child_parent_sha: str
    reason: str
    replay_base_sha: str | None = None


def _llm_label() -> str:
    return get_llm_config().provider


def _llm_halt_message(result: object, fallback: str) -> str:
    reason = getattr(result, "halt_reason", None)
    if isinstance(reason, str) and reason:
        return f"{_llm_label()} {reason}"
    return fallback


def _llm_signalled_inconclusive(result: object) -> bool:
    reason = getattr(result, "halt_reason", None)
    return isinstance(reason, str) and reason.startswith("signalled INCONCLUSIVE")


def _llm_requested_rebase(result: object) -> bool:
    reason = getattr(result, "halt_reason", None)
    return isinstance(reason, str) and reason.startswith("requested REBASE")


def _inconclusive_refresh_target(worktree: Path) -> tuple[bool, str]:
    target, reason = repo.select_rebase_target(worktree)
    return repo.rebase_target_advances(worktree, target), reason


def _is_ghstack_mergeability_failure(check_names: list[str]) -> bool:
    return any(
        name == "ghstack-mergeability-check"
        or ("ghstack" in name.lower() and "mergeability" in name.lower())
        for name in check_names
    )


def _failed_logs_are_content_free(
    failed: list[tuple[str, str]],
) -> bool:
    """True if no failing job has a substantive log excerpt yet.

    "<no log available>" placeholders and short stub bodies count as empty.
    Used to defer claude invocation when GitHub hasn't published logs yet
    for jobs that just transitioned to failed.
    """
    if not failed:
        return True
    return _useful_log_chars(failed) < MIN_USEFUL_LOG_CHARS


def _useful_log_chars(failed: list[tuple[str, str]]) -> int:
    total = 0
    for _, text in failed:
        stripped = (text or "").strip()
        if stripped == "<no log available>":
            continue
        total += len(stripped)
    return total


def _latest_completed_at(checks: list[dict]) -> float | None:
    """Latest ``completedAt`` across checks, as a unix timestamp.

    Used to anchor the CI stability window to actual GitHub completion
    time instead of "now we noticed". On a fresh mergedog start against
    a PR whose CI finished hours ago, anchoring to ``time.time()`` would
    needlessly burn a full ``CI_STABILITY_WINDOW_SEC`` before acting.
    Returns ``None`` if any check lacks a parseable completion timestamp
    (treat as "still moving"; fall back to ``time.time()``).
    """
    if not checks:
        return None
    timestamps: list[float] = []
    for c in checks:
        ts = c.get("completedAt") or ""
        if not ts or ts.startswith("0001-01-01"):
            return None
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
        timestamps.append(dt.timestamp())
    return max(timestamps) if timestamps else None


def describe_log_state(
    failed: list[tuple[str, str]], failing_check_count: int
) -> str:
    """Short diagnostic for the failed-job logs we got back from gh.

    Disambiguates the two ways ``_failed_logs_are_content_free`` can fire:
    failing checks with no Actions run link (status-only checks like Dr. CI
    -- nothing for ``gh run view`` to fetch) vs Actions runs whose logs
    haven't been published yet. Also used in the non-defer path so the
    "invoking claude" line records how much log content went into the
    prompt.
    """
    if not failed:
        return (
            f"0 of {failing_check_count} failing checks have Actions logs "
            f"(status-only checks?)"
        )
    chars = _useful_log_chars(failed)
    return f"{len(failed)} run(s), {chars} chars"


def _try_interventions(
    failed: list[tuple[str, str]],
    checks: list[dict],
    intervened_run_ids: set[int],
) -> bool:
    """Re-run failed jobs whose logs match a known-transient pattern.

    Iterates over ``failed`` (the trimmed log excerpts the shepherd would
    otherwise pass to claude), matches each against
    :data:`mergedog.interventions.INTERVENTIONS`, and on a hit calls
    ``gh run rerun --failed`` for the underlying workflow run. Records
    the run id in ``intervened_run_ids`` so we don't repeatedly rerun
    the same run -- if the failure persists past one rerun, claude
    takes over.

    Returns True if at least one rerun was successfully kicked off, so
    the caller can reset its status cache and resume polling instead of
    invoking claude.
    """
    triggered = False
    for name, log_text in failed:
        itv = interventions.find_intervention(log_text)
        if itv is None:
            continue
        check = next((c for c in checks if c.get("name") == name), None)
        if check is None:
            continue
        run_id = github.run_id_for_check(check)
        if run_id is None:
            continue
        if run_id in intervened_run_ids:
            log(
                f"intervention {itv.name!r} matched again on {name!r} "
                f"(run {run_id}); already retried this run, falling "
                f"through to claude"
            )
            continue
        log(
            f"intervention {itv.name!r} matched on {name!r}; "
            f"re-running failed jobs in run {run_id}"
        )
        ok, msg = github.rerun_failed_jobs(run_id)
        if ok:
            intervened_run_ids.add(run_id)
            triggered = True
        else:
            log(f"  -> rerun failed for run {run_id}: {msg}")
    return triggered


def _apply_spurious_overrides(
    checks: list[dict], spurious_names: set[str]
) -> list[dict]:
    """Flip checks whose names claude already judged spurious to ``skipping``.

    Used so the next status evaluation sees a green-on-the-non-spurious
    set without ignoring the rest. If any genuinely-pending checks are
    still outstanding, we keep waiting on them instead of handing off.
    """
    if not spurious_names:
        return checks
    out: list[dict] = []
    for c in checks:
        if (
            c.get("name") in spurious_names
            and c.get("bucket") in {"fail", "cancel"}
        ):
            c = {**c, "bucket": "skipping"}
        out.append(c)
    return out


def _spurious_check_names_from_checks(checks: list[dict]) -> set[str]:
    """Return concrete failed check names that can be suppressed as spurious."""
    return {
        c.get("name")
        for c in checks
        if c.get("bucket") in {"fail", "cancel"} and c.get("name")
    }


def _filter_spurious_failed_jobs(
    failed: list[tuple[str, str]], spurious_names: set[str]
) -> list[tuple[str, str]]:
    """Remove failed-job logs already classified as spurious."""
    if not spurious_names:
        return failed
    return [(name, text) for name, text in failed if name not in spurious_names]


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _actionable_lint_failure_names(failed: list[tuple[str, str]]) -> list[str]:
    """Return lint jobs whose logs contain concrete lintrunner diagnostics.

    These are not flaky/status-only signals: lintrunner identified a path and
    a rule violation. If an agent cannot fix one, it should signal TOO_HARD or
    INCONCLUSIVE rather than marking the check spurious.
    """
    names: list[str] = []
    for name, text in failed:
        if "lint" not in name.lower():
            continue
        clean = _ANSI_ESCAPE_RE.sub("", text or "")
        if "Lint failed!" in clean and ">>> Lint for " in clean:
            names.append(name)
    return names


def _write_status_best_effort(pr: int, **fields) -> None:
    try:
        write_status(pr, **fields)
    except Exception:
        pass


def _suppressed_failure_suffix(suppressed: int) -> str:
    if suppressed <= 0:
        return ""
    plural = "" if suppressed == 1 else "s"
    return f"; {suppressed} suppressed failure{plural}"


def _ci_status_message(
    status: str,
    done: int,
    total: int,
    failed: int,
    *,
    suppressed: int = 0,
) -> str:
    if status == "pending":
        return (
            f"waiting for CI: {done}/{total} checks done"
            f"{_suppressed_failure_suffix(suppressed)}"
        )
    if status == "failed":
        failures = (
            f"{failed} active failure"
            if failed == 1
            else f"{failed} active failures"
        )
        return (
            f"CI failed: {failures} ({done}/{total} checks done)"
            f"{_suppressed_failure_suffix(suppressed)}"
        )
    if status == "passed":
        if suppressed:
            plural = "" if suppressed == 1 else "s"
            return (
                f"CI passed with {suppressed} suppressed failure{plural}: "
                f"{done}/{total} checks done"
            )
        return f"CI passed: {done}/{total} checks done"
    return (
        f"CI {status}: {done}/{total} checks done"
        f"{_suppressed_failure_suffix(suppressed)}"
    )


def _fix_attempt_message(fix_attempts: int, max_fix_attempts: int) -> str:
    if max_fix_attempts == 0:
        return f"{fix_attempts} fixes pushed; fix cap disabled"
    return f"{fix_attempts}/{max_fix_attempts} fixes pushed"


def _fixing_ci_message(
    active_failed_count: int,
    fix_attempts: int,
    max_fix_attempts: int,
) -> str:
    failure_plural = "" if active_failed_count == 1 else "s"
    return (
        f"fixing CI: {active_failed_count} active failure{failure_plural}; "
        f"{_fix_attempt_message(fix_attempts, max_fix_attempts)}"
    )


def _intervention_suffix(intervention_count: int) -> str:
    plural = "" if intervention_count == 1 else "s"
    return (
        f"{intervention_count} mergedog intervention{plural} "
        "since last approval"
    )


def _status_with_interventions(message: str, intervention_count: int) -> str:
    return f"{message}; {_intervention_suffix(intervention_count)}"


def _latest_trusted_approval_sha(pr: int) -> str | None:
    audit = github.get_pr_review_audit(pr)
    trusted_approvals = [
        r for r in audit["reviews"]
        if r.get("state") == "APPROVED"
        and github.is_trusted_association(r.get("association"))
        and r.get("commit_id")
    ]
    if not trusted_approvals:
        return None
    trusted_approvals.sort(key=lambda r: r.get("submitted_at") or "")
    return trusted_approvals[-1]["commit_id"]


def _count_mergedog_interventions_since_ack(
    worktree: Path,
    human_ack_sha: str | None,
) -> int:
    if not human_ack_sha:
        return 0
    try:
        proc = subprocess.run(
            [
                "git",
                "rev-list",
                "--first-parent",
                "--grep=^\\[MERGEDOG\\]",
                "--count",
                f"{human_ack_sha}..HEAD",
            ],
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    if proc.returncode != 0:
        return 0
    try:
        return max(0, int(proc.stdout.strip() or "0"))
    except ValueError:
        return 0


def _refresh_status_prefix(pr: int) -> tuple[bool | None, bool | None]:
    """Toggle the [MERGING]/[APPROVED] log prefix based on the PR's state.

    Called once per main-poll iteration. The prefix is the only signal
    the mux has back from each shepherd that pytorchmergebot is actively
    merging the PR (or that the PR is approved and waiting) -- the mux
    only reads the last log line per shepherd, so we have to thread the
    state through the log itself.

    Failures are silently swallowed: a bad ``gh`` call here shouldn't HALT
    over a UI nicety.
    """
    try:
        labels, decision = github.get_pr_status_fields(pr)
    except Exception:
        return None, None
    merging = github.MERGING_LABEL in labels
    approved = (decision or "").upper() == "APPROVED"
    set_merging(merging)
    set_approved(approved)
    return approved, merging


def _is_ghstack(pr_data: dict) -> bool:
    branch = pr_data.get("headRefName", "") or ""
    body = pr_data.get("body", "") or ""
    if branch.startswith("gh/"):
        return True
    if "ghstack-source-id" in body:
        return True
    return False


def _resolve_ghstack_parent_dependency(
    pr: int, child_head_ref: str
) -> _GhstackParentDependency | None:
    """Return the immediate ghstack parent for ``pr``, if it has one.

    Single-PR mode remains best-effort for odd ghstack shapes: if stack
    discovery fails, keep shepherding the PR with the historical isolated
    behavior instead of refusing to run.
    """
    try:
        from mergedog.stack import resolve_stack

        members, _ = resolve_stack(pr)
    except SystemExit as e:
        log(
            f"WARNING: could not resolve ghstack parent for PR #{pr} "
            f"(exit {e.code}); continuing without decentralized stack "
            "propagation"
        )
        return None
    except Exception as e:
        log(
            f"WARNING: could not resolve ghstack parent for PR #{pr}: {e}; "
            "continuing without decentralized stack propagation"
        )
        return None

    idx = next((i for i, m in enumerate(members) if m.pr == pr), None)
    if idx is None:
        log(
            f"WARNING: PR #{pr} was not present in resolved ghstack; "
            "continuing without decentralized stack propagation"
        )
        return None
    if idx == 0:
        return None
    parent = members[idx - 1]
    child = members[idx]
    return _GhstackParentDependency(
        parent_pr=parent.pr,
        parent_head_ref=parent.head_ref,
        parent_orig_ref=parent.orig_ref,
        child_head_ref=child.head_ref or child_head_ref,
        child_orig_ref=child.orig_ref,
    )


def _check_trusted_ghstack_parent(
    dep: _GhstackParentDependency, current_sha: str
) -> bool:
    trust = TrustDB.load_or_create(dep.parent_pr)
    trust.head_branch = dep.parent_head_ref
    trust.head_repo_clone_url = REPO_SSH_URL
    trust.save()
    if trust.is_trusted(current_sha):
        return True
    trust_mergebot_rebase_if_equivalent(
        trust,
        current_sha,
        ensure_current_available=lambda: (
            repo.fetch_branch_from_url(
                REPO_SSH_URL,
                dep.parent_head_ref,
                f"refs/remotes/mergedog-trust/{dep.parent_pr}",
            )
            == current_sha
        ),
    )
    return trust.is_trusted(current_sha)


def _refresh_ghstack_parent_status(
    dep: _GhstackParentDependency,
) -> _GhstackParentStatus:
    refs = repo.fetch_stack_refs(
        [
            (dep.parent_head_ref, dep.parent_orig_ref),
            (dep.child_head_ref, dep.child_orig_ref),
        ]
    )
    parent_orig_sha = refs[dep.parent_orig_ref]
    child_orig_sha = refs[dep.child_orig_ref]
    child_parent_sha = repo.parent_sha(child_orig_sha)
    parent_orig_tree = repo.tree_sha(parent_orig_sha)
    child_parent_tree = repo.tree_sha(child_parent_sha)
    stale = child_parent_tree != parent_orig_tree

    checks = github.get_pr_checks_all(dep.parent_pr)
    parent_trust = _check_trusted_ghstack_parent(
        dep, github.get_pr_head_sha(dep.parent_pr)
    )
    if parent_trust:
        effective_checks = _apply_spurious_overrides(
            checks, set(TrustDB.load_or_create(dep.parent_pr).spurious_check_names)
        )
        parent_status = github.evaluate_checks(effective_checks)
    else:
        parent_status = "untrusted"

    done = sum(1 for c in checks if c.get("bucket") not in {"pending", None})
    total = len(checks)
    observation = (parent_status, total)
    if dep.stable_observation != observation:
        dep.stable_observation = observation
        anchor: float | None = None
        if parent_status == "passed":
            anchor = _latest_completed_at(checks)
        dep.stable_since = anchor if anchor is not None else time.time()
    parent_ready = (
        parent_status == "passed"
        and time.time() - dep.stable_since >= CI_STABILITY_WINDOW_SEC
    )
    if not parent_trust:
        reason = "parent head is not trusted"
    elif parent_status != "passed":
        reason = f"parent CI is {parent_status}"
    elif not parent_ready:
        remaining = int(CI_STABILITY_WINDOW_SEC - (time.time() - dep.stable_since))
        reason = f"parent CI is waiting {max(remaining, 0)}s for stability"
    else:
        reason = "parent is green-stable"
    return _GhstackParentStatus(
        stale=stale,
        parent_ready=parent_ready,
        parent_status=parent_status,
        parent_done=done,
        parent_total=total,
        parent_orig_sha=parent_orig_sha,
        child_orig_sha=child_orig_sha,
        child_parent_sha=child_parent_sha,
        reason=reason,
    )


def _maybe_use_landed_ghstack_parent_base(
    dep: _GhstackParentDependency, status: _GhstackParentStatus
) -> bool:
    """If the parent PR landed, replay the child onto the landed commit."""
    merge_sha = github.get_pr_merge_commit_sha(dep.parent_pr)
    if merge_sha is None:
        return False

    repo.fetch_origin()
    if not repo.is_ancestor(merge_sha, "origin/main"):
        status.parent_ready = False
        status.stale = True
        status.reason = (
            f"parent PR merged as {merge_sha[:12]}, but origin/main does not "
            "contain it yet"
        )
        return False

    landed_tree = repo.tree_sha(merge_sha)
    child_parent_tree = repo.tree_sha(status.child_parent_sha)
    status.stale = child_parent_tree != landed_tree
    status.parent_ready = True
    status.replay_base_sha = merge_sha
    if status.stale:
        status.reason = (
            f"parent PR merged as {merge_sha[:12]}; replaying child onto landed tree"
        )
    else:
        status.reason = (
            f"parent PR merged as {merge_sha[:12]}; child already has landed tree"
        )
    return True


def _publish_ghstack_parent_rebase(
    pr: int,
    worktree: Path,
    dep: _GhstackParentDependency,
    status: _GhstackParentStatus,
    trust: TrustDB,
    *,
    ignore_sev: bool,
    pr_data: dict | None = None,
    sessions: list[ClaudeSession] | None = None,
) -> bool:
    """Rebase this ghstack PR onto its current parent and submit only itself."""
    if _wait_for_no_active_sev(
        f"propagating stack parent PR #{dep.parent_pr} to PR #{pr}",
        ignore_sev=ignore_sev,
    ):
        return False

    refs = repo.fetch_stack_refs(
        [
            (dep.parent_head_ref, dep.parent_orig_ref),
            (dep.child_head_ref, dep.child_orig_ref),
        ]
    )
    if (
        refs[dep.parent_orig_ref] != status.parent_orig_sha
        or refs[dep.child_orig_ref] != status.child_orig_sha
    ):
        log(
            "stack refs changed while preparing parent propagation; "
            "rechecking before submitting"
        )
        return False

    base_sha = status.replay_base_sha or status.parent_orig_sha
    log(
        f"rebasing PR #{pr} onto updated stack parent PR #{dep.parent_pr} "
        f"({status.child_parent_sha[:12]} -> {base_sha[:12]})"
    )
    repo.set_worktree_to_sha(worktree, base_sha)
    try:
        repo.ghstack_cherry_pick(worktree, pr)
    except subprocess.CalledProcessError:
        if not repo.is_cherry_pick_in_progress(worktree):
            raise
        if pr_data is None or sessions is None:
            repo.run(
                ["git", "cherry-pick", "--abort"],
                cwd=worktree,
                check=False,
                loud=True,
            )
            die(
                "stack parent propagation produced conflicts; halting for "
                "human intervention (no pr_data/sessions available for "
                "LLM resolution)"
            )
        log(
            "stack parent replay produced conflicts; "
            f"asking {_llm_label()} to resolve"
        )
        ctx_path, _ = _refresh_context_file(pr_data, trusted=True)
        prompt = render_cherry_pick_conflict_prompt(
            url=pr_data.get("url", ""),
            branch=dep.child_head_ref,
            context_path=str(ctx_path),
        )
        sha_before = repo.head_sha(worktree)
        started_at = utc_now_iso()
        result = claude_mod.invoke_cherry_pick_resolver(worktree, prompt)
        ran_cleanly, new_orig_sha, transcript = result
        _record_claude_session(
            sessions,
            mode="cherry-pick-resolver",
            sha_before=sha_before,
            started_at=started_at,
            ran_cleanly=ran_cleanly,
            new_sha=new_orig_sha,
            transcript=transcript,
            on_commit="resolved stack parent replay conflicts in commit {sha}",
            on_clean_noop="aborted the cherry-pick",
        )
        if not ran_cleanly:
            die(
                _llm_halt_message(
                    result,
                    f"{_llm_label()} failed to resolve the stack parent "
                    "replay conflict cleanly",
                )
            )
        if new_orig_sha is None:
            die(
                f"{_llm_label()} aborted the cherry-pick; halting for "
                "human intervention"
            )
    repo.ghstack_submit(worktree, "Propagate parent update downstream")
    new_head_sha = repo.fetch_ghstack_head(dep.child_head_ref)
    trust.trust(new_head_sha)
    trust.spurious_check_names = []
    trust.save()
    log(f"ghstack submitted; new {dep.child_head_ref} = {new_head_sha[:12]}")
    _wait_for_pr_head(pr, new_head_sha)
    return True


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
        # Closed or merged: nothing to do here ever again. Use the
        # completed exit code so the mux can show the final row until
        # the operator explicitly cleans it up.
        complete(
            f"PR is not open (state={state}); shepherd complete",
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


def _refresh_context_file(
    pr_data: dict, *, trusted: bool = True
) -> tuple[Path, list[dict]]:
    """Rebuild the per-PR sidecar from the latest title/body/comments.

    Refreshed before each claude invocation so that comments added partway
    through a shepherd run show up in the agent's context. Returns both the
    sidecar path and the raw comments list so callers can pull out trusted
    snippets (e.g., dr. ci summary) without re-fetching.

    When *trusted* is False (external contributor PR), the description and
    non-bot comments are omitted to prevent prompt injection via PR text.
    """
    pr = pr_data["number"]
    comments = github.get_pr_comments(pr)
    text = context_mod.render_context(
        pr=pr,
        url=pr_data.get("url", ""),
        title=pr_data.get("title", ""),
        body=pr_data.get("body", "") or "",
        comments=comments,
        trusted=trusted,
    )
    path = context_file(pr)
    context_mod.write_context_file(path, text)
    return path, comments


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


def _run_operator_fix(
    *,
    pr: int,
    pr_data: dict,
    worktree: Path,
    branch: str,
    trust: TrustDB,
    operator_context: str,
    is_ghstack: bool,
    fork_remote: str | None,
    ignore_sev: bool,
    trusted_pr: bool,
    sessions: list[ClaudeSession],
    pushed_changes: list[PushedChange],
) -> bool:
    """Apply one trusted operator-requested follow-up to this PR."""
    request = operator_context.strip()
    if not request:
        die("operator fix requested with empty context")

    log(f"invoking {_llm_label()} for trusted operator fix")
    ctx_path, _comments = _refresh_context_file(pr_data, trusted=trusted_pr)
    prompt = render_operator_fix_prompt(
        url=pr_data.get("url", ""),
        branch=branch,
        context_path=str(ctx_path),
        operator_context=request,
        is_ghstack=is_ghstack,
    )

    sha_before = repo.head_sha(worktree)
    started_at = utc_now_iso()
    result = claude_mod.invoke_operator_fix(worktree, prompt)
    ran_cleanly, new_sha, transcript = result
    _record_claude_session(
        sessions,
        mode="operator-fix",
        sha_before=sha_before,
        started_at=started_at,
        ran_cleanly=ran_cleanly,
        new_sha=new_sha,
        transcript=transcript,
        on_commit="pushed operator fix commit {sha}",
        on_clean_noop="operator fix request already satisfied",
    )
    if not ran_cleanly:
        die(
            _llm_halt_message(
                result,
                f"{_llm_label()} exited abnormally or produced an invalid commit",
            )
        )
    if new_sha is None:
        log(f"{_llm_label()} made no operator-fix commit")
        return False

    trust.spurious_check_names = []
    if is_ghstack:
        _publish_ghstack_fix(
            pr, worktree, branch, new_sha, trust, ignore_sev=ignore_sev
        )
    else:
        assert fork_remote is not None
        trust.trust(new_sha)
        _safe_push(
            pr,
            worktree,
            fork_remote,
            branch,
            new_sha,
            reason=f"pushing {_llm_label()} operator fix commit",
            ignore_sev=ignore_sev,
        )
        _record_pushed_change(
            pushed_changes,
            worktree,
            new_sha,
            "pushed an LLM-authored operator fix",
            source=f"{_llm_label()} operator-fix",
        )
    trust.save()
    return True


def _rebase_ghstack_onto_main(
    pr: int,
    worktree: Path,
    head_ref: str,
    trust: TrustDB,
    *,
    ignore_sev: bool,
    pr_data: dict | None = None,
    sessions: list[ClaudeSession] | None = None,
    target_ref: str | None = None,
    target_reason: str | None = None,
) -> None:
    """Rebase /orig onto a known-good point on main and re-publish via ghstack.

    Target selection mirrors ``_merge_main_resolving_conflicts``: we pick
    viable/strict or a recent revert rather than raw trunk tip.
    """
    if _wait_for_no_active_sev(
        "rebasing /orig onto main", ignore_sev=ignore_sev
    ):
        repo.fetch_origin()

    if target_ref is None:
        target, reason = repo.select_rebase_target(worktree)
    else:
        target, reason = target_ref, target_reason or target_ref
    log(f"rebase target: {reason}")

    try:
        status, new_orig_sha = repo.attempt_rebase_main(worktree, ref=target)
    except RuntimeError as e:
        die(str(e))

    if status == "noop":
        log("rebase produced no new commit (already at target)")
        return

    if status == "conflict":
        if pr_data is None or sessions is None:
            repo.abort_rebase(worktree)
            die(
                "rebase produced conflicts; halting for human intervention "
                "(no pr_data/sessions available for LLM resolution)"
            )
        log(f"rebase produced conflicts; asking {_llm_label()} to resolve")
        ctx_path, _ = _refresh_context_file(pr_data, trusted=True)
        prompt = render_rebase_conflict_prompt(
            url=pr_data.get("url", ""),
            branch=head_ref,
            context_path=str(ctx_path),
        )
        sha_before = repo.head_sha(worktree)
        started_at = utc_now_iso()
        result = claude_mod.invoke_rebase_resolver(
            worktree,
            prompt,
            allow_multiple_commits=True,
        )
        ran_cleanly, new_orig_sha, transcript = result
        _record_claude_session(
            sessions,
            mode="rebase-resolver",
            sha_before=sha_before,
            started_at=started_at,
            ran_cleanly=ran_cleanly,
            new_sha=new_orig_sha,
            transcript=transcript,
            on_commit="resolved rebase conflicts in commit {sha}",
            on_clean_noop="aborted the rebase",
        )
        if not ran_cleanly:
            die(
                _llm_halt_message(
                    result,
                    f"{_llm_label()} failed to resolve the rebase conflict cleanly",
                )
            )
        if new_orig_sha is None:
            die(
                f"{_llm_label()} aborted the rebase; halting for human intervention"
            )

    assert new_orig_sha is not None
    log(f"rebased /orig to {new_orig_sha[:12]}; re-publishing via ghstack")
    _wait_for_no_active_sev(
        "re-publishing rebased /orig via ghstack submit", ignore_sev=ignore_sev
    )
    repo.ghstack_submit(worktree, "Rebase onto origin/main")
    new_head_sha = repo.fetch_ghstack_head(head_ref)
    trust.trust(new_head_sha)
    log(f"ghstack submitted; new {head_ref} = {new_head_sha[:12]}")
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


def _record_pushed_change(
    changes: list[PushedChange],
    worktree: Path,
    sha: str,
    summary: str,
    *,
    source: str | None = None,
) -> None:
    """Append a pushed-change summary for the handoff comment."""
    try:
        subject = repo.commit_message(worktree, sha).splitlines()[0]
    except Exception:
        subject = None
    changes.append(
        PushedChange(
            sha=sha,
            summary=summary,
            subject=subject,
            source=source,
        )
    )


def _latest_drci_summary_for_handoff(
    pr: int, head_sha: str, spurious_check_names: set[str]
) -> str | None:
    if not spurious_check_names:
        return None
    try:
        comments = github.get_pr_comments(pr)
        return github.latest_drci_summary(comments, head_sha=head_sha)
    except Exception as e:
        log(f"WARNING: could not fetch Dr. CI summary for handoff: {e}")
        return None


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
    trusted_pr: bool = True,
    target_ref: str | None = None,
    target_reason: str | None = None,
) -> str | None:
    """Merge a known-good point on main into HEAD, resolving conflicts via claude.

    Returns the new head SHA if a merge commit was made, else None
    (already up to date). Trusts the new SHA. Caller is responsible
    for pushing.

    Target selection: we never merge raw trunk tip. Instead we pick the
    best known-good ref via ``select_rebase_target`` -- viable/strict,
    a recent revert commit, or stay put if nothing is ahead of us.
    """
    if _wait_for_no_active_sev("merging main", ignore_sev=ignore_sev):
        repo.fetch_origin()

    if target_ref is None:
        target, reason = repo.select_rebase_target(worktree)
    else:
        target, reason = target_ref, target_reason or target_ref
    log(f"rebase target: {reason}")

    try:
        status, new_sha = repo.attempt_merge_main(worktree, ref=target)
    except RuntimeError as e:
        die(str(e))

    if status == "noop":
        log("merge produced no new commit (already up to date)")
        return None

    if status == "conflict":
        log(f"merge produced conflicts; asking {_llm_label()} to resolve")
        ctx_path, _ = _refresh_context_file(pr_data, trusted=trusted_pr)
        prompt = render_merge_conflict_prompt(
            url=pr_data.get("url", ""),
            branch=branch,
            context_path=str(ctx_path),
            merge_subject=MERGE_RESOLVED_SUBJECT,
        )
        sha_before = repo.head_sha(worktree)
        started_at = utc_now_iso()
        result = claude_mod.invoke_merge_resolver(worktree, prompt)
        ran_cleanly, new_sha, transcript = result
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
            die(
                _llm_halt_message(
                    result,
                    f"{_llm_label()} failed to resolve the merge conflict cleanly",
                )
            )
        if new_sha is None:
            die(f"{_llm_label()} aborted the merge; halting for human intervention")

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
    reassess: bool = False,
    max_fix_commits: int | None = None,
    extra_context: str | None = None,
    operator_fix_context: str | None = None,
    manage_mergedog_label: bool = False,
) -> None:
    configure_status_pr(pr)
    if max_fix_commits is None:
        max_fix_commits = MAX_FIX_COMMITS
    if max_fix_commits < 0:
        raise ValueError("--max-fix-commits must be >= 0")

    repo.ensure_clone()
    repo.fetch_origin()

    pr_data = github.get_pr(pr)
    _validate_pr(pr_data)

    # Optional coordination signal. Keep best-effort semantics so a transient
    # GitHub label failure does not abort the actual shepherding work.
    labelled = False
    if manage_mergedog_label:
        try:
            github.add_label(pr, MERGEDOG_LABEL)
            labelled = True
        except Exception as e:
            log(f"WARNING: failed to add {MERGEDOG_LABEL} label: {e}")
    signal.signal(signal.SIGTERM, _sigterm_to_systemexit)
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1)
    try:
        _shepherd_body(
            pr,
            pr_data,
            rebase,
            accept_divergence,
            ignore_sev,
            reassess,
            max_fix_commits,
            extra_context,
            operator_fix_context,
        )
    finally:
        if labelled:
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
    reassess: bool = False,
    max_fix_commits: int = MAX_FIX_COMMITS,
    extra_context: str | None = None,
    operator_fix_context: str | None = None,
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
    trusted_pr = self_pr or is_ghstack
    if self_pr:
        log(f"PR authored by current user ({viewer}); skipping approval gate")
    human_ack_sha = seed_trust_from_reviews(
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

    ghstack_parent = (
        _resolve_ghstack_parent_dependency(pr, branch) if is_ghstack else None
    )
    if ghstack_parent is not None:
        log(
            f"  stack parent: PR #{ghstack_parent.parent_pr} "
            f"({ghstack_parent.parent_orig_ref})"
        )

    labels.autolabel_if_needed(pr, pr_data)

    fix_commits_pushed = 0
    max_fix_attempts_status = max_fix_commits if max_fix_commits > 0 else 0
    if max_fix_commits == 0:
        log("  fix cap:    disabled")
    else:
        log(f"  fix cap:    {max_fix_commits} [MERGEDOG] commits")
    sessions: list[ClaudeSession] = []
    pushed_changes: list[PushedChange] = []
    recovery_attempts = 0
    last_approved: bool | None = None
    last_merging: bool | None = None
    intervention_count = _count_mergedog_interventions_since_ack(
        worktree, human_ack_sha
    )
    _write_status_best_effort(
        pr,
        phase="starting",
        category="action",
        action="starting",
        message=_status_with_interventions(
            "starting shepherd", intervention_count
        ),
        intervention_count=intervention_count,
        human_ack_sha=human_ack_sha,
        approved=last_approved,
        merging=last_merging,
        fix_attempts=fix_commits_pushed,
        max_fix_attempts=max_fix_attempts_status,
    )
    # Auto-retries for infra-flake merge failures (e.g. 504). Capped at
    # MAX_MERGE_AUTO_RETRIES to prevent runaway commenting during outages.
    auto_retries = 0

    # User-requested upfront merge of origin/main. Default behavior
    # otherwise is to never auto-rebase based on age -- mergedog only
    # merges main when piggybacking on a fix push it was going to do
    # anyway, or when the operator explicitly asks via ``--rebase``.
    if rebase:
        if is_ghstack:
            log("user requested upfront rebase of /orig onto origin/main")
            _rebase_ghstack_onto_main(
                pr, worktree, branch, trust, ignore_sev=ignore_sev,
                pr_data=pr_data, sessions=sessions,
            )
        else:
            log("user requested upfront rebase onto origin/main")
            new_sha = _merge_main_resolving_conflicts(
                worktree, trust, branch, pr_data, sessions,
                ignore_sev=ignore_sev, trusted_pr=trusted_pr,
            )
            if new_sha is not None:
                log(f"pushing merge commit {new_sha[:12]} to {fork_remote}/{branch}")
                _safe_push(
                    pr, worktree, fork_remote, branch, new_sha,
                    reason="pushing merge-main commit", ignore_sev=ignore_sev,
                )
                _record_pushed_change(
                    pushed_changes,
                    worktree,
                    new_sha,
                    "merged main into the PR branch",
                )

    # On restart, if the last observed failure was a merge conflict that
    # we never resolved (e.g. old code didn't have conflict handling),
    # proactively rebase now rather than waiting for a new failure.
    if trust.last_observed_failure_body and is_merge_conflict_failure(
        trust.last_observed_failure_body
    ):
        log(
            "prior merge-conflict failure detected on restart; "
            "rebasing onto main"
        )
        repo.fetch_origin()
        if is_ghstack:
            _rebase_ghstack_onto_main(
                pr, worktree, branch, trust, ignore_sev=ignore_sev,
                pr_data=pr_data, sessions=sessions,
            )
        else:
            assert fork_remote is not None
            new_sha = _merge_main_resolving_conflicts(
                worktree, trust, branch, pr_data, sessions,
                ignore_sev=ignore_sev, trusted_pr=trusted_pr,
            )
            if new_sha is not None:
                log(
                    f"pushing merge commit {new_sha[:12]} to "
                    f"{fork_remote}/{branch}"
                )
                _safe_push(
                    pr, worktree, fork_remote, branch, new_sha,
                    reason="pushing merge-main commit after conflict",
                    ignore_sev=ignore_sev,
                )
                _record_pushed_change(
                    pushed_changes,
                    worktree,
                    new_sha,
                    "merged main into the PR branch after merge failure",
                )
        trust.last_observed_failure_body = ""
        trust.save()

    if operator_fix_context is not None:
        if _run_operator_fix(
            pr=pr,
            pr_data=pr_data,
            worktree=worktree,
            branch=branch,
            trust=trust,
            operator_context=operator_fix_context,
            is_ghstack=is_ghstack,
            fork_remote=fork_remote,
            ignore_sev=ignore_sev,
            trusted_pr=trusted_pr,
            sessions=sessions,
            pushed_changes=pushed_changes,
        ):
            fix_commits_pushed += 1

    run_state_cache: dict[int, tuple[str | None, str | None]] = {}

    # Outer recovery loop: each iteration is one CI-inspect / claude-fix /
    # handoff / watch cycle. We re-enter when pytorchmergebot replies
    # "Merge failed" -- treated like CI going red, not a hard halt.
    # mergedog will not re-trigger ``@pytorchbot merge`` itself; a human
    # always owns the land decision (and the skip decision, when claude
    # judges spurious).
    while True:
        trunk_applied = (
            True
            if TRUNK_LABEL is None
            else github.has_label(pr_data, TRUNK_LABEL)
        )
        last_status: str | None = None
        # (status, check_count) we last observed. Becomes the anchor for
        # the stability window: when it changes (new check arrives,
        # status flips), we reset the timer.
        stable_observation: tuple[str, int] | None = None
        stable_since: float = 0.0
        # Names of failed checks that claude already judged spurious. We
        # keep these from re-triggering the fix loop and -- more
        # importantly -- treat them as if they were skipped so we keep
        # waiting for any *other* still-pending checks before handing
        # off. Cleared whenever we push a fix (fresh CI invalidates the
        # judgments). Seeded from the trust DB so that restarts don't
        # re-invoke claude for the same failures.
        if reassess:
            spurious_check_names: set[str] = set()
            trust.spurious_check_names = []
            trust.save()
        else:
            spurious_check_names = set(trust.spurious_check_names)
        # How many consecutive ``failed`` polls have come back with
        # content-free logs from gh. Reset whenever we either pull useful
        # logs or leave the failed branch.
        empty_log_defers = 0
        # Workflow run ids we've already triggered an intervention rerun
        # for in this recovery pass. Bounds the retry to one rerun per
        # run_id so a persistent (non-transient) failure that happens to
        # match an intervention pattern still falls through to claude.
        intervened_run_ids: set[int] = set()

        # Poll CI, fix or judge spurious until ready for handoff. Breaks
        # out (via the handoff path) when CI is green and the trunk
        # label is on.
        while True:
            approved, merging = _refresh_status_prefix(pr)
            if approved is not None:
                last_approved = approved
                if approved:
                    try:
                        refreshed_ack_sha = _latest_trusted_approval_sha(pr)
                    except Exception as e:
                        log(f"WARNING: could not refresh approval baseline: {e}")
                    else:
                        if refreshed_ack_sha:
                            human_ack_sha = refreshed_ack_sha
            if merging is not None:
                last_merging = merging
            # 1. Verify the PR head is still trusted.
            current = github.get_pr_head_sha(pr)
            if self_pr:
                # On a self-authored PR, every push is implicitly approved
                # by the operator -- roll the trust forward instead of
                # halting.
                trust.trust(current)
            if not trust.is_trusted(current):
                trust_mergebot_rebase_if_equivalent(
                    trust,
                    current,
                    ensure_current_available=lambda sha=current: (
                        _fetch_current_pr_head_for_trust(pr, trust, sha)
                    ),
                )
            if not trust.is_trusted(current):
                subject = github.get_commit_subject(current)
                die(
                    f"PR head moved to untrusted commit {current[:12]}: "
                    f"{subject!r}. Manual intervention required."
                )
            intervention_count = _count_mergedog_interventions_since_ack(
                worktree, human_ack_sha
            )

            # 2. Approve any approval-pending workflow runs.
            approved = _approve_pending_runs(current, run_state_cache)
            if approved:
                empty_log_defers = 0
                stable_observation = None  # newly-approved runs invalidate stability
                time.sleep(APPROVAL_SETTLE_SEC)
                continue

            # 3. Read check status. Failures claude already judged
            # spurious are flipped to "skipping" so we don't re-judge
            # them, and so the overall verdict reflects what's still
            # genuinely outstanding.
            checks = github.get_pr_checks_all(pr)
            effective_checks = _apply_spurious_overrides(
                checks, spurious_check_names
            )
            status = github.evaluate_checks(effective_checks)

            # Cross-check: gh pr checks (check-run API) can disagree
            # with the workflow-run API.  When a failed job is re-run
            # and the retry passes, gh pr checks shows the latest
            # (passing) result, but the workflow conclusion stays
            # "failure".  If any tracked workflow has conclusion=failure
            # but no individual check is failing, fetch logs directly
            # from the failed workflow runs instead.
            #
            # Skip this cross-check when the only reason status is
            # "passed" is that all failing checks were already judged
            # spurious -- the workflow conclusion is permanently stale
            # in that case and re-introducing it would loop forever.
            raw_failure_names = {
                c.get("name")
                for c in checks
                if c.get("bucket") in {"fail", "cancel"}
                and c.get("name")
            }
            all_failures_spurious = (
                bool(raw_failure_names)
                and raw_failure_names <= spurious_check_names
            )
            suppressed_failed_count = len(
                raw_failure_names & spurious_check_names
            )
            workflow_failed_run_ids: list[int] = []
            if status == "passed" and not all_failures_spurious:
                workflow_failed_run_ids = [
                    run_id
                    for run_id, (st, concl) in run_state_cache.items()
                    if concl == "failure"
                ]
                if workflow_failed_run_ids:
                    log(
                        f"gh pr checks says passed but workflow run(s) "
                        f"{workflow_failed_run_ids} have conclusion=failure; "
                        f"treating as failed"
                    )
                    status = "failed"

            done = sum(
                1 for c in checks if c.get("bucket") not in {"pending", None}
            )
            failed_count = sum(
                1 for c in checks if c.get("bucket") in {"fail", "cancel"}
            )
            active_failed_check_names = sorted(
                c.get("name", "")
                for c in effective_checks
                if c.get("bucket") in {"fail", "cancel"} and c.get("name")
            )
            active_failed_count = len(active_failed_check_names)
            if status == "failed" and not failed_count and workflow_failed_run_ids:
                failed_count = len(workflow_failed_run_ids)
            if (
                status == "failed"
                and not active_failed_count
                and workflow_failed_run_ids
            ):
                active_failed_count = len(workflow_failed_run_ids)
            _write_status_best_effort(
                pr,
                phase="polling_ci",
                category="waiting" if status == "pending" else "action",
                waiting_on="ci" if status == "pending" else None,
                action="inspecting_ci" if status != "pending" else None,
                message=_status_with_interventions(
                    _ci_status_message(
                        status,
                        done,
                        len(checks),
                        active_failed_count,
                        suppressed=suppressed_failed_count,
                    ),
                    intervention_count,
                ),
                intervention_count=intervention_count,
                human_ack_sha=human_ack_sha,
                approved=last_approved,
                merging=last_merging,
                ci_done=done,
                ci_total=len(checks),
                ci_failed=active_failed_count,
                fix_attempts=fix_commits_pushed,
                max_fix_attempts=max_fix_attempts_status,
            )
            summary = f"{status} ({done}/{len(checks)} done)"
            if summary != last_status:
                log(f"CI status -> {summary}")
                last_status = summary

            if ghstack_parent is not None:
                parent_status = _refresh_ghstack_parent_status(ghstack_parent)
                _maybe_use_landed_ghstack_parent_base(
                    ghstack_parent, parent_status
                )
                if parent_status.stale:
                    log_state_key = (
                        f"{parent_status.reason}; "
                        f"parent {parent_status.parent_done}/"
                        f"{parent_status.parent_total} done; "
                        f"child parent={parent_status.child_parent_sha[:12]} "
                        f"current parent={parent_status.parent_orig_sha[:12]}"
                    )
                    if log_state_key != ghstack_parent.last_log_state:
                        log(
                            f"stack parent PR #{ghstack_parent.parent_pr} "
                            f"advanced; {log_state_key}"
                        )
                        ghstack_parent.last_log_state = log_state_key
                    if not parent_status.parent_ready:
                        _write_status_best_effort(
                            pr,
                            phase="waiting_stack_parent",
                            category="waiting",
                            waiting_on="stack_parent",
                            message=_status_with_interventions(
                                (
                                    f"waiting for stack parent "
                                    f"#{ghstack_parent.parent_pr}: "
                                    f"{parent_status.reason}; parent "
                                    f"{parent_status.parent_done}/"
                                    f"{parent_status.parent_total} checks done"
                                ),
                                intervention_count,
                            ),
                            intervention_count=intervention_count,
                            human_ack_sha=human_ack_sha,
                            approved=last_approved,
                            merging=last_merging,
                            ci_done=done,
                            ci_total=len(checks),
                            ci_failed=active_failed_count,
                            fix_attempts=fix_commits_pushed,
                            max_fix_attempts=max_fix_attempts_status,
                        )
                        time.sleep(POLL_INTERVAL_SEC)
                        continue
                    if _publish_ghstack_parent_rebase(
                        pr,
                        worktree,
                        ghstack_parent,
                        parent_status,
                        trust,
                        ignore_sev=ignore_sev,
                        pr_data=pr_data,
                        sessions=sessions,
                    ):
                        spurious_check_names.clear()
                        stable_observation = None
                    last_status = None
                    ghstack_parent.last_log_state = None
                    continue
                ghstack_parent.last_log_state = None

            # Track stability: any change in (status, check_count) restarts
            # the quiescence timer. We only gate on it for the "passed"
            # verdict, so genuine pending/failed states proceed without
            # artificial delay. When we land on "passed", anchor to the
            # latest GitHub-reported completedAt rather than "now" -- on
            # a restart against long-finished CI, that lets us skip the
            # stability wait that's only meaningful right after a push.
            observation = (status, len(checks))
            if stable_observation != observation:
                stable_observation = observation
                anchor: float | None = None
                if status == "passed":
                    anchor = _latest_completed_at(checks)
                stable_since = anchor if anchor is not None else time.time()

            if status == "pending":
                empty_log_defers = 0
                time.sleep(POLL_INTERVAL_SEC)
                continue

            if status == "failed":
                # Hand the failures to claude; only advance once claude is
                # OK with the situation (either pushed a fix or judged
                # spurious).
                comments = github.get_pr_comments(pr)
                handoff_iso = github.latest_mergedog_handoff_iso_from_comments(
                    comments
                )
                ignore_since = handoff_iso or trust.last_observed_failure_iso
                newly_ignored = mergebot_ignored_check_names(
                    comments, checks, since_iso=ignore_since
                ) - spurious_check_names
                if newly_ignored:
                    spurious_check_names |= newly_ignored
                    trust.spurious_check_names = sorted(spurious_check_names)
                    trust.save()
                    log(
                        f"{PROJECT.mergebot_login} is ignoring "
                        f"{len(newly_ignored)} failed check"
                        f"{'' if len(newly_ignored) == 1 else 's'} "
                        "from merge -i; continuing with remaining CI"
                    )
                    last_status = None
                    continue
                if is_ghstack and _is_ghstack_mergeability_failure(
                    active_failed_check_names
                ):
                    log(
                        "ghstack mergeability check failed; rebasing /orig "
                        "onto origin/main before asking fix-CI"
                    )
                    _rebase_ghstack_onto_main(
                        pr,
                        worktree,
                        branch,
                        trust,
                        ignore_sev=ignore_sev,
                        pr_data=pr_data,
                        sessions=sessions,
                        target_ref="origin/main",
                        target_reason="origin/main (mergeability check)",
                    )
                    spurious_check_names.clear()
                    trust.spurious_check_names = []
                    trust.save()
                    last_status = None
                    stable_observation = None
                    continue
                failed = github.get_failed_job_logs(pr)
                if not failed and workflow_failed_run_ids:
                    failed = github.get_failed_job_logs_for_runs(
                        workflow_failed_run_ids
                    )
                failed = _filter_spurious_failed_jobs(
                    failed, spurious_check_names
                )
                if is_ghstack and _is_ghstack_mergeability_failure(
                    [name for name, _ in failed]
                ):
                    log(
                        "ghstack mergeability job failed; rebasing /orig "
                        "onto origin/main before asking fix-CI"
                    )
                    _rebase_ghstack_onto_main(
                        pr,
                        worktree,
                        branch,
                        trust,
                        ignore_sev=ignore_sev,
                        pr_data=pr_data,
                        sessions=sessions,
                        target_ref="origin/main",
                        target_reason="origin/main (mergeability job)",
                    )
                    spurious_check_names.clear()
                    trust.spurious_check_names = []
                    trust.save()
                    last_status = None
                    stable_observation = None
                    continue
                if max_fix_commits > 0 and fix_commits_pushed >= max_fix_commits:
                    die(
                        f"already pushed {fix_commits_pushed} [MERGEDOG] fix "
                        f"commits and CI is still failing; halting for human "
                        f"intervention"
                    )
                failing_check_count = active_failed_count
                if not failing_check_count and workflow_failed_run_ids:
                    failing_check_count = len(failed)
                log_state = describe_log_state(failed, failing_check_count)
                # GitHub publishes a job's log a few seconds after the job
                # transitions to failed. Calling claude on an empty logs
                # block (with a stale dr. ci that may still say "no
                # failures") almost always produces a "spurious" verdict;
                # defer a few cycles to let logs land.
                #
                # Skip deferral if the checks completed long enough ago
                # that logs would already be available (e.g. fresh start
                # against a PR whose CI failed hours ago).
                completed_at = _latest_completed_at(checks)
                logs_should_exist = (
                    completed_at is not None
                    and time.time() - completed_at
                    > MAX_EMPTY_LOG_DEFERS * POLL_INTERVAL_SEC
                )
                if (
                    _failed_logs_are_content_free(failed)
                    and empty_log_defers < MAX_EMPTY_LOG_DEFERS
                    and not logs_should_exist
                ):
                    empty_log_defers += 1
                    log(
                        f"failed-job logs not yet available from gh "
                        f"(defer {empty_log_defers}/{MAX_EMPTY_LOG_DEFERS}); "
                        f"{log_state}; "
                        f"waiting for logs to publish before invoking {_llm_label()}"
                    )
                    time.sleep(POLL_INTERVAL_SEC)
                    continue
                empty_log_defers = 0

                # Pre-claude pass: any failing job whose log matches a
                # known-transient pattern gets a single ``gh run rerun
                # --failed`` instead of being handed to claude. Capped at
                # one retry per run_id, so a persistent failure that
                # happens to match still falls through.
                if _try_interventions(
                    failed, checks, intervened_run_ids
                ):
                    last_status = None
                    stable_observation = None
                    time.sleep(APPROVAL_SETTLE_SEC)
                    continue

                log(f"invoking {_llm_label()} on failing CI ({log_state})")
                _write_status_best_effort(
                    pr,
                    phase="fixing_ci",
                    category="action",
                    action="fixing_ci",
                    message=_status_with_interventions(
                        _fixing_ci_message(
                            active_failed_count,
                            fix_commits_pushed,
                            max_fix_attempts_status,
                        ),
                        intervention_count,
                    ),
                    intervention_count=intervention_count,
                    human_ack_sha=human_ack_sha,
                    approved=last_approved,
                    merging=last_merging,
                    ci_done=done,
                    ci_total=len(checks),
                    ci_failed=active_failed_count,
                    fix_attempts=fix_commits_pushed,
                    max_fix_attempts=max_fix_attempts_status,
                )
                ctx_path, comments = _refresh_context_file(
                    pr_data, trusted=trusted_pr
                )
                trunk_ctx = repo.trunk_revert_context(worktree)
                effective_extra = extra_context or ""
                if trunk_ctx:
                    log(f"injecting trunk revert context into {_llm_label()} prompt")
                    effective_extra = (
                        f"{trunk_ctx}\n\n{effective_extra}" if effective_extra
                        else trunk_ctx
                    )
                prompt = render_fix_prompt(
                    url=pr_data.get("url", ""),
                    branch=branch,
                    context_path=str(ctx_path),
                    failed_jobs=failed,
                    failing_check_names=active_failed_check_names
                    or [name for name, _ in failed],
                    is_ghstack=is_ghstack,
                    drci_summary=github.latest_drci_summary(
                        comments, head_sha=current
                    ),
                    extra_context=effective_extra or None,
                )
                session_failed_jobs = [name for name, _ in failed]
                sha_before = current
                started_at = utc_now_iso()
                result = claude_mod.invoke_fixer(worktree, prompt)
                ran_cleanly, new_sha, transcript = result
                session_notes: list[str] = []
                if session_failed_jobs:
                    session_notes.append(
                        f"failing jobs: {', '.join(session_failed_jobs)}"
                    )
                if result.spurious_reason:
                    reason = " ".join(result.spurious_reason.split())
                    session_notes.append(f"rationale: {reason}")
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
                    extra=f" — {'; '.join(session_notes)}"
                    if session_notes
                    else "",
                )
                if not ran_cleanly:
                    if _llm_requested_rebase(result):
                        can_refresh, reason = _inconclusive_refresh_target(worktree)
                        if can_refresh:
                            log(
                                f"{_llm_label()} requested REBASE; "
                                f"refreshing stale base via {reason}"
                            )
                            if is_ghstack:
                                _rebase_ghstack_onto_main(
                                    pr,
                                    worktree,
                                    branch,
                                    trust,
                                    ignore_sev=ignore_sev,
                                    pr_data=pr_data,
                                    sessions=sessions,
                                )
                                spurious_check_names.clear()
                                trust.spurious_check_names = []
                                trust.save()
                            else:
                                assert fork_remote is not None
                                merge_sha = _merge_main_resolving_conflicts(
                                    worktree,
                                    trust,
                                    branch,
                                    pr_data,
                                    sessions,
                                    ignore_sev=ignore_sev,
                                    trusted_pr=trusted_pr,
                                )
                                if merge_sha is None:
                                    die(
                                        f"{_llm_label()} requested REBASE, "
                                        "but the selected rebase target no longer "
                                        "advanced the PR; halting for human review"
                                    )
                                log(
                                    f"pushing refreshed base {merge_sha[:12]} "
                                    f"to {fork_remote}/{branch}"
                                )
                                _safe_push(
                                    pr,
                                    worktree,
                                    fork_remote,
                                    branch,
                                    merge_sha,
                                    reason="pushing requested base refresh",
                                    ignore_sev=ignore_sev,
                                )
                                spurious_check_names.clear()
                                trust.spurious_check_names = []
                                trust.save()
                                _record_pushed_change(
                                    pushed_changes,
                                    worktree,
                                    merge_sha,
                                    "merged main into the PR branch after LLM requested rebase",
                                )
                            last_status = None
                            continue
                        die(
                            f"{_llm_label()} requested REBASE, but the selected "
                            "rebase target no longer advanced the PR; halting "
                            "for human review"
                        )
                    if _llm_signalled_inconclusive(result):
                        can_refresh, reason = _inconclusive_refresh_target(worktree)
                        if can_refresh:
                            log(
                                f"{_llm_label()} signalled INCONCLUSIVE; "
                                f"refreshing stale base via {reason}"
                            )
                            if is_ghstack:
                                _rebase_ghstack_onto_main(
                                    pr,
                                    worktree,
                                    branch,
                                    trust,
                                    ignore_sev=ignore_sev,
                                    pr_data=pr_data,
                                    sessions=sessions,
                                )
                                spurious_check_names.clear()
                                trust.spurious_check_names = []
                                trust.save()
                            else:
                                assert fork_remote is not None
                                merge_sha = _merge_main_resolving_conflicts(
                                    worktree,
                                    trust,
                                    branch,
                                    pr_data,
                                    sessions,
                                    ignore_sev=ignore_sev,
                                    trusted_pr=trusted_pr,
                                )
                                if merge_sha is None:
                                    die(
                                        f"{_llm_label()} signalled INCONCLUSIVE, "
                                        "but the selected rebase target no longer "
                                        "advanced the PR; halting for human review"
                                    )
                                log(
                                    f"pushing refreshed base {merge_sha[:12]} "
                                    f"to {fork_remote}/{branch}"
                                )
                                _safe_push(
                                    pr,
                                    worktree,
                                    fork_remote,
                                    branch,
                                    merge_sha,
                                    reason="pushing known-good base refresh",
                                    ignore_sev=ignore_sev,
                                )
                                spurious_check_names.clear()
                                trust.spurious_check_names = []
                                trust.save()
                                _record_pushed_change(
                                    pushed_changes,
                                    worktree,
                                    merge_sha,
                                    "merged main into the PR branch after inconclusive CI",
                                )
                            last_status = None
                            continue
                    die(
                        _llm_halt_message(
                            result,
                            f"{_llm_label()} exited abnormally or produced an "
                            "invalid commit",
                        )
                    )
                if new_sha is None:
                    # Mark the failed checks as spurious so we treat
                    # them as skipping in subsequent iterations. We
                    # still wait out any other pending checks before
                    # handing off -- a green-on-the-non-spurious-set
                    # verdict isn't a green-on-everything verdict.
                    actionable_lints = _actionable_lint_failure_names(failed)
                    if actionable_lints:
                        die(
                            f"{_llm_label()} signalled spurious for actionable "
                            "lint failure(s): "
                            f"{', '.join(actionable_lints)}; halting for "
                            "human review"
                        )
                    newly_spurious = _spurious_check_names_from_checks(
                        effective_checks
                    )
                    if not newly_spurious:
                        die(
                            f"{_llm_label()} made no commit, but mergedog "
                            "could not map that no-op to any failed check; "
                            "halting for human intervention"
                        )
                    spurious_check_names |= newly_spurious
                    trust.spurious_check_names = sorted(spurious_check_names)
                    trust.save()
                    log(
                        f"{_llm_label()} judged {len(newly_spurious)} failure"
                        f"{'' if len(newly_spurious) == 1 else 's'} spurious; "
                        "continuing to wait on remaining CI"
                    )
                    last_status = None  # force re-log on next pass
                    continue
                elif is_ghstack:
                    if repo.would_merge_conflict(worktree, "origin/main"):
                        log(
                            f"{_llm_label()} fix {new_sha[:12]} conflicts "
                            "with origin/main; discarding it and rebasing /orig"
                        )
                        repo.set_worktree_to_sha(worktree, sha_before)
                        spurious_check_names.clear()
                        trust.spurious_check_names = []
                        trust.save()
                        _rebase_ghstack_onto_main(
                            pr,
                            worktree,
                            branch,
                            trust,
                            ignore_sev=ignore_sev,
                            pr_data=pr_data,
                            sessions=sessions,
                            target_ref="origin/main",
                            target_reason="origin/main (discarded conflicting fix)",
                        )
                        last_status = None
                        stable_observation = None
                        continue
                    spurious_check_names.clear()
                    trust.spurious_check_names = []
                    trust.save()
                    _publish_ghstack_fix(
                        pr, worktree, branch, new_sha, trust,
                        ignore_sev=ignore_sev,
                    )
                    fix_commits_pushed += 1
                    last_status = None
                    continue
                else:
                    assert fork_remote is not None
                    if repo.would_merge_conflict(worktree, "origin/main"):
                        log(
                            f"{_llm_label()} fix {new_sha[:12]} conflicts "
                            "with origin/main; discarding it and refreshing base"
                        )
                        repo.set_worktree_to_sha(worktree, sha_before)
                        merge_sha = _merge_main_resolving_conflicts(
                            worktree,
                            trust,
                            branch,
                            pr_data,
                            sessions,
                            ignore_sev=ignore_sev,
                            trusted_pr=trusted_pr,
                            target_ref="origin/main",
                            target_reason="origin/main (discarded conflicting fix)",
                        )
                        if merge_sha is None:
                            die(
                                f"{_llm_label()} fix conflicted with origin/main, "
                                "but refreshing the original PR produced no new "
                                "commit; halting for human review"
                            )
                        log(
                            f"pushing refreshed base {merge_sha[:12]} "
                            f"to {fork_remote}/{branch}"
                        )
                        _safe_push(
                            pr,
                            worktree,
                            fork_remote,
                            branch,
                            merge_sha,
                            reason="pushing base refresh after discarding conflicting fix",
                            ignore_sev=ignore_sev,
                        )
                        spurious_check_names.clear()
                        trust.spurious_check_names = []
                        trust.save()
                        _record_pushed_change(
                            pushed_changes,
                            worktree,
                            merge_sha,
                            "merged main into the PR branch after discarding a conflicting LLM fix",
                        )
                        last_status = None
                        stable_observation = None
                        continue
                    spurious_check_names.clear()
                    trust.spurious_check_names = []
                    trust.trust(new_sha)
                    # Piggyback: we're going to push and trigger fresh CI
                    # anyway, so merge origin/main while we're at it. CI
                    # then runs once on (PR + fix + main) instead of
                    # testing the fix against a stale base.
                    merge_sha = _merge_main_resolving_conflicts(
                        worktree, trust, branch, pr_data, sessions,
                        ignore_sev=ignore_sev, trusted_pr=trusted_pr,
                    )
                    final_sha = merge_sha if merge_sha is not None else new_sha
                    log(
                        f"pushing {final_sha[:12]} to {fork_remote}/{branch}"
                    )
                    _safe_push(
                        pr, worktree, fork_remote, branch, final_sha,
                        reason=f"pushing {_llm_label()} fix commit",
                        ignore_sev=ignore_sev,
                    )
                    _record_pushed_change(
                        pushed_changes,
                        worktree,
                        new_sha,
                        "pushed an LLM-authored CI fix",
                        source=f"{_llm_label()} fix-CI",
                    )
                    if merge_sha is not None and merge_sha != new_sha:
                        _record_pushed_change(
                            pushed_changes,
                            worktree,
                            merge_sha,
                            "merged main into the PR branch after the fix",
                        )
                    fix_commits_pushed += 1
                    last_status = None
                    continue

            # CI is "passed". Require it to have been passed continuously
            # for CI_STABILITY_WINDOW_SEC before we act, so that a
            # freshly-pushed commit can't trick us by reporting "1/1 done"
            # while the rest of the workflows are still being created.
            if status == "passed":
                empty_log_defers = 0
                elapsed = time.time() - stable_since
                if elapsed < CI_STABILITY_WINDOW_SEC:
                    remaining = int(CI_STABILITY_WINDOW_SEC - elapsed)
                    passed_message = "CI passed"
                    if suppressed_failed_count:
                        plural = "" if suppressed_failed_count == 1 else "s"
                        passed_message = (
                            f"CI passed with {suppressed_failed_count} "
                            f"suppressed failure{plural}"
                        )
                    _write_status_best_effort(
                        pr,
                        phase="polling_ci",
                        category="waiting",
                        waiting_on="ci_stability",
                        message=_status_with_interventions(
                            f"{passed_message}; waiting {remaining}s for stability",
                            intervention_count,
                        ),
                        intervention_count=intervention_count,
                        human_ack_sha=human_ack_sha,
                        approved=last_approved,
                        merging=last_merging,
                        ci_done=done,
                        ci_total=len(checks),
                        ci_failed=0,
                        fix_attempts=fix_commits_pushed,
                        max_fix_attempts=max_fix_attempts_status,
                    )
                    log(
                        f"{passed_message}; waiting {remaining}s for stability "
                        f"(no new checks should appear)"
                    )
                    time.sleep(min(POLL_INTERVAL_SEC, remaining))
                    continue

            # Either CI passed (and is stable), or claude said "spurious".
            # Advance.
            # TODO: when trunk failures were judged spurious, assess whether
            # the PR's own critical signal actually ran before handing off.
            # A trunk failure that masks a job carrying the PR's signal is
            # worse than one on an unrelated job.
            if not trunk_applied and TRUNK_LABEL is not None:
                # Adding ciflow/trunk kicks off a fresh wave of trunk
                # workflows; gate on SEV so we don't pile on broken trunk.
                _wait_for_no_active_sev(
                    f"applying {TRUNK_LABEL} label", ignore_sev=ignore_sev
                )
                log(f"CI green; applying {TRUNK_LABEL} label")
                github.add_label(pr, TRUNK_LABEL, loud=False)
                log(
                    f"{TRUNK_LABEL} label applied; "
                    "waiting for trunk workflows to appear"
                )
                trunk_applied = True
                last_status = None
                time.sleep(APPROVAL_SETTLE_SEC)
                continue
            log("ALL CI GREEN.")
            ready_for_merge = last_approved is not False
            if suppressed_failed_count:
                plural = "" if suppressed_failed_count == 1 else "s"
                ci_green_phrase = (
                    f"CI is green except {suppressed_failed_count} "
                    f"suppressed failure{plural}"
                )
            else:
                ci_green_phrase = "CI is green"
            ready_message = (
                f"ready for human merge: {ci_green_phrase}"
                if ready_for_merge
                else f"waiting for maintainer approval: {ci_green_phrase}"
            )
            ready_user_action = (
                "review mergedog handoff and merge when satisfied"
                if ready_for_merge
                else "approve the PR after reviewing mergedog interventions"
            )
            _write_status_best_effort(
                pr,
                phase="ready",
                category="ready",
                user_action=ready_user_action,
                message=_status_with_interventions(
                    ready_message, intervention_count
                ),
                intervention_count=intervention_count,
                human_ack_sha=human_ack_sha,
                approved=last_approved,
                merging=last_merging,
                ci_done=done,
                ci_total=len(checks),
                ci_failed=failed_count,
                fix_attempts=fix_commits_pushed,
                max_fix_attempts=max_fix_attempts_status,
            )
            break

        handoff_started_iso = utc_now_iso()
        suppressed_failures = sorted(spurious_check_names)
        drci_summary = _latest_drci_summary_for_handoff(
            pr, current, spurious_check_names
        )
        suppression_warning = suppression_drci_status_warning(
            suppressed_failures, drci_summary
        )
        if suppression_warning:
            log(f"WARNING: {suppression_warning}")
        handoff_comment_ok = post_handoff_comment(
            pr,
            pr_data,
            sessions,
            pushed_changes=pushed_changes,
            suppressed_failures=suppressed_failures,
            drci_summary=drci_summary,
            recovering=recovery_attempts > 0,
        )
        # Anchor the watch loop on the actual handoff comment timestamp,
        # not "now": on restart this lets us notice a "Merge failed" that
        # already happened between the last handoff and our restart. But
        # also floor on any failure we've already halted on, so the next
        # restart doesn't re-react to the same stale comment.
        try:
            handoff_iso = (
                github.latest_mergedog_handoff_iso(pr) or handoff_started_iso
            )
        except Exception as e:
            log(f"WARNING: could not verify handoff comment timestamp: {e}")
            handoff_iso = handoff_started_iso
        since_iso = max(handoff_iso, trust.last_observed_failure_iso)
        merge_instruction = (
            f"comment `{PROJECT.merge_command}` on "
            if PROJECT.merge_command
            else "merge "
        )
        log(
            "Hand off to a human reviewer; have them "
            f"{merge_instruction}{pr_data.get('url', f'PR #{pr}')}."
        )
        if last_approved is False:
            handoff_category = "ready"
            handoff_waiting_on = "approval"
            handoff_user_action = (
                "approve after reviewing local mergedog log"
                if handoff_comment_ok is False
                else "approve the PR after reviewing mergedog interventions"
            )
            handoff_message = "waiting for maintainer approval"
        else:
            handoff_category = "waiting"
            handoff_waiting_on = "human_merge"
            handoff_user_action = (
                (
                    "review local mergedog log, then "
                    f"{merge_instruction}{pr_data.get('url', f'PR #{pr}')}"
                )
                if handoff_comment_ok is False
                else f"{merge_instruction}{pr_data.get('url', f'PR #{pr}')}"
            )
            handoff_message = (
                "waiting for human reviewer to "
                f"{merge_instruction}{pr_data.get('url', f'PR #{pr}')}"
            )
        handoff_warning_suffix = ""
        if suppression_warning:
            handoff_warning_suffix += f"; {suppression_warning}"
        if handoff_comment_ok is False:
            handoff_warning_suffix += "; handoff comment failed"
        _write_status_best_effort(
            pr,
            phase="watching_merge",
            category=handoff_category,
            waiting_on=handoff_waiting_on,
            user_action=handoff_user_action,
            message=_status_with_interventions(
                handoff_message + handoff_warning_suffix, intervention_count
            ),
            intervention_count=intervention_count,
            human_ack_sha=human_ack_sha,
            approved=last_approved,
            merging=last_merging,
            fix_attempts=fix_commits_pushed,
            max_fix_attempts=max_fix_attempts_status,
        )

        result, event_iso, fail_body = watch_post_handoff(
            pr,
            since_iso,
            intervention_count=intervention_count,
            human_ack_sha=human_ack_sha,
            suppressed_check_names=spurious_check_names,
            handoff_comment_ok=handoff_comment_ok,
            suppression_warning=suppression_warning,
        )
        if result == "closed":
            complete(
                "PR is no longer open; shepherd complete",
                code=EXIT_PR_NOT_ACTIONABLE,
            )
        if result == "conflict":
            recovery_attempts += 1
            log("post-handoff merge conflict detected; rebasing onto main")
            repo.fetch_origin()
            if is_ghstack:
                _rebase_ghstack_onto_main(
                    pr, worktree, branch, trust, ignore_sev=ignore_sev
                )
            else:
                assert fork_remote is not None
                new_sha = _merge_main_resolving_conflicts(
                    worktree, trust, branch, pr_data, sessions,
                    ignore_sev=ignore_sev, trusted_pr=trusted_pr,
                )
                if new_sha is not None:
                    log(
                        f"pushing merge commit {new_sha[:12]} to "
                        f"{fork_remote}/{branch}"
                    )
                    _safe_push(
                        pr, worktree, fork_remote, branch, new_sha,
                        reason="pushing merge-main commit after GitHub conflict",
                        ignore_sev=ignore_sev,
                    )
                    _record_pushed_change(
                        pushed_changes,
                        worktree,
                        new_sha,
                        "merged main into the PR branch after GitHub reported conflicts",
                    )
            last_status = None
            pr_data = github.get_pr(pr)
            continue
        if result == "ci_failed":
            recovery_attempts += 1
            log("post-handoff CI regression detected; re-entering CI inspection")
            last_status = None
            pr_data = github.get_pr(pr)
            continue
        # result == "failed": pytorchmergebot rejected the merge. Persist
        # the failure timestamp so we don't re-fire on this same comment,
        # then loop back to CI inspection -- claude can judge spurious or
        # push a fix.
        assert event_iso is not None
        trust.last_observed_failure_iso = event_iso
        trust.last_observed_failure_body = fail_body or ""
        trust.save()
        recovery_attempts += 1

        if fail_body and is_retryable_merge_failure(fail_body):
            if auto_retries < MAX_MERGE_AUTO_RETRIES:
                if PROJECT.merge_command is None:
                    die(
                        "merge failure was retryable, but this repo has no "
                        "configured merge command to auto-retry"
                    )
                auto_retries += 1
                log(
                    f"{PROJECT.mergebot_login} merge failure is retryable "
                    f"(infra flake); auto-retrying `{PROJECT.merge_command}` "
                    f"({auto_retries}/{MAX_MERGE_AUTO_RETRIES})"
                )
                github.post_pr_comment(pr, PROJECT.merge_command)
                pr_data = github.get_pr(pr)
                continue
            log(
                f"retryable merge failure but exhausted "
                f"{MAX_MERGE_AUTO_RETRIES} auto-retries; falling through "
                f"to manual recovery"
            )

        if fail_body and is_merge_conflict_failure(fail_body):
            log(
                "pytorchmergebot merge failed due to merge conflict; "
                "rebasing onto main"
            )
            repo.fetch_origin()
            if is_ghstack:
                _rebase_ghstack_onto_main(
                    pr, worktree, branch, trust, ignore_sev=ignore_sev
                )
            else:
                assert fork_remote is not None
                new_sha = _merge_main_resolving_conflicts(
                    worktree, trust, branch, pr_data, sessions,
                    ignore_sev=ignore_sev, trusted_pr=trusted_pr,
                )
                if new_sha is not None:
                    log(
                        f"pushing merge commit {new_sha[:12]} to "
                        f"{fork_remote}/{branch}"
                    )
                    _safe_push(
                        pr, worktree, fork_remote, branch, new_sha,
                        reason="pushing merge-main commit after conflict",
                        ignore_sev=ignore_sev,
                    )
                    _record_pushed_change(
                        pushed_changes,
                        worktree,
                        new_sha,
                        "merged main into the PR branch after merge failure",
                    )
            trust.last_observed_failure_body = ""
            trust.save()
            last_status = None
            pr_data = github.get_pr(pr)
            continue

        log(
            "pytorchmergebot reported merge failure; re-inspecting CI "
            "(mergedog will not retrigger merge -- a human owns the next "
            "`@pytorchbot merge`)"
        )
        # Refresh PR data: pytorchmergebot may have removed the merging
        # label, and labels/state generally are stale after the merge
        # attempt.
        pr_data = github.get_pr(pr)
