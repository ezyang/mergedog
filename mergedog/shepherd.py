"""The main mergedog shepherding loop.

One process per PR. Synchronous. Halts on any sign of an untrusted change.
"""
from __future__ import annotations

import faulthandler
import hashlib
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
from mergedog import github, injection, interventions, labels, repo
from mergedog.config import (
    format_ci_sev_ignored_numbers,
    get_ignored_ci_sev_numbers,
    get_llm_config,
)
from mergedog.handoff import (
    ClaudeSession,
    PushedChange,
    is_cla_merge_failure,
    is_merge_conflict_failure,
    is_retryable_merge_failure,
    latest_mergebot_event,
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
from mergedog.trust_seed import latest_trusted_approval, seed_trust_from_reviews


SEV_POLL_INTERVAL_SEC = 5 * 60  # SEVs are minutes-to-hours; don't spam ``gh``
SEV_CONFIG_POLL_INTERVAL_SEC = 15


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


def _write_ci_sev_status(pr: int, message: str) -> None:
    try:
        write_status(
            pr,
            phase="waiting_ci_sev",
            category="waiting",
            waiting_on="ci_sev",
            message=message,
        )
    except Exception:
        pass


def _ci_sev_number(sev: dict) -> int | None:
    raw = sev.get("number")
    if isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _split_configured_ignored_ci_sevs(
    sevs: list[dict],
) -> tuple[list[dict], list[dict]]:
    if not sevs:
        return [], []
    try:
        ignored_numbers = get_ignored_ci_sev_numbers()
    except ValueError as e:
        log(
            f"WARNING: could not read ci: sev ignore config: {e}; "
            "respecting all active ci: sev issues"
        )
        return sevs, []

    active: list[dict] = []
    ignored: list[dict] = []
    for sev in sevs:
        number = _ci_sev_number(sev)
        if number is not None and number in ignored_numbers:
            ignored.append(sev)
        else:
            active.append(sev)
    return active, ignored


def _ci_sev_refs(sevs: list[dict]) -> str:
    return format_ci_sev_ignored_numbers(
        {n for sev in sevs if (n := _ci_sev_number(sev)) is not None}
    )


def _sleep_until_next_sev_poll_or_config_ignore(active_sevs: list[dict]) -> bool:
    """Sleep until the next GitHub poll, unless local config unblocks us."""
    remaining = SEV_POLL_INTERVAL_SEC
    while remaining > 0:
        delay = min(SEV_CONFIG_POLL_INTERVAL_SEC, remaining)
        time.sleep(delay)
        remaining -= delay
        still_active, ignored = _split_configured_ignored_ci_sevs(active_sevs)
        if not still_active and ignored:
            return True
    return False


def _wait_for_no_active_sev(
    reason: str,
    *,
    ignore_sev: bool,
    pr: int | None = None,
) -> bool:
    """If pytorch CI has an open SEV, block until it clears.

    A CI SEV here is any open issue on pytorch/pytorch tagged
    ``ci: sev`` -- dev-infra's signal that trunk is degraded. Default
    behavior is to wait it out so we don't stampede broken CI with
    new pushes. ``ignore_sev`` (operator override via ``--ignore-sev``)
    skips the wait entirely. The persistent ci_sev.ignored config
    suppresses individual SEV issue numbers and is re-read while parked
    so already-parked shepherds can resume after the mux updates it.

    Returns True if it actually had to wait and then resumed (because the
    SEV cleared or became configured-ignored) -- callers can use this to
    discard work prepared against a stale view of trunk.
    """
    if ignore_sev:
        return False
    last_ids: tuple[int, ...] | None = None
    while True:
        sevs = github.list_active_ci_sevs()
        active_sevs, ignored_sevs = _split_configured_ignored_ci_sevs(sevs)
        if not active_sevs:
            if ignored_sevs:
                refs = _ci_sev_refs(ignored_sevs)
                if last_ids is not None:
                    log(f"ci: sev {refs} is configured ignored; resuming")
                    return True
                log(
                    f"ci: sev {refs} is configured ignored; "
                    f"continuing before {reason}"
                )
                return False
            if last_ids is not None:
                log("CI SEV cleared; resuming")
                return True
            return False
        ids = tuple(
            sorted(
                n
                for sev in active_sevs
                if (n := _ci_sev_number(sev)) is not None
            )
        )
        head = active_sevs[0]
        others = (
            f" (+{len(active_sevs) - 1} more)"
            if len(active_sevs) > 1
            else ""
        )
        message = (
            f"parked on ci: sev #{head.get('number')} "
            f"{head.get('title', '?')!r}{others}; "
            f"waiting before {reason}"
        )
        if pr is not None:
            _write_ci_sev_status(pr, message)
        if ids != last_ids:
            log(message)
            last_ids = ids
        if _sleep_until_next_sev_poll_or_config_ignore(active_sevs):
            # Refresh GitHub before resuming; a different unignored SEV may
            # have appeared since the last remote poll.
            continue


POLL_INTERVAL_SEC = 60
APPROVAL_SETTLE_SEC = 15
FULL_CHECK_REFRESH_SEC = 5 * 60
APPROVAL_REFRESH_INTERVAL_SEC = 10 * 60
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
# PyTorch PR CI should surface many checks before the trunk label is applied.
# A "green" verdict with only one or two checks usually means GitHub never
# created the pull-CI workflow runs for this head; refresh the base to trigger
# a real CI wave instead of treating the sparse result as green.
MIN_GREEN_CHECKS_TO_TRUST = 3
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
_INLINE_HUNK_MARKER_RE = re.compile(
    r"<!--\s*mergedog:inline-hunk\s+sha=([0-9a-f]{7,40})\s+key=([0-9a-f]+)\s*-->"
)


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


WorkflowFingerprint = tuple[tuple[int, str, str], ...]


@dataclass
class _TrunkCiGate:
    head_sha: str
    check_count: int
    workflow_fingerprint: WorkflowFingerprint


@dataclass
class _CiCheckPollCache:
    """Cached full check list used only for unchanged pending CI."""

    head_sha: str | None = None
    workflow_fingerprint: WorkflowFingerprint | None = None
    checks: list[dict] | None = None
    status: str | None = None
    fetched_at: float = 0.0

    def invalidate(self) -> None:
        self.head_sha = None
        self.workflow_fingerprint = None
        self.checks = None
        self.status = None
        self.fetched_at = 0.0

    def should_fetch(
        self,
        *,
        head_sha: str,
        workflow_fingerprint: WorkflowFingerprint,
        now: float,
    ) -> bool:
        if not self.checks:
            return True
        if self.head_sha != head_sha:
            return True
        if self.status != "pending":
            return True
        if self.workflow_fingerprint != workflow_fingerprint:
            return True
        return now - self.fetched_at >= FULL_CHECK_REFRESH_SEC

    def update(
        self,
        *,
        head_sha: str,
        workflow_fingerprint: WorkflowFingerprint,
        checks: list[dict],
        status: str,
        fetched_at: float,
    ) -> None:
        if not checks:
            self.invalidate()
            return
        self.head_sha = head_sha
        self.workflow_fingerprint = workflow_fingerprint
        self.checks = checks
        self.status = status
        self.fetched_at = fetched_at


def _workflow_state_fingerprint(
    run_state_cache: dict[int, tuple[str | None, str | None]],
) -> WorkflowFingerprint:
    return tuple(
        sorted(
            (run_id, status or "", conclusion or "")
            for run_id, (status, conclusion) in run_state_cache.items()
        )
    )


def _trunk_wave_has_started(
    gate: _TrunkCiGate,
    *,
    head_sha: str,
    check_count: int,
    workflow_fingerprint: WorkflowFingerprint,
) -> bool:
    """Return True once applying the trunk label has produced fresh CI evidence."""
    return (
        head_sha != gate.head_sha
        or check_count > gate.check_count
        or workflow_fingerprint != gate.workflow_fingerprint
    )


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


def _select_refresh_target(
    worktree: Path, target_ref: str | None, target_reason: str | None
) -> str:
    """Resolve the ref to merge/rebase onto, logging the choice."""
    if target_ref is None:
        target, reason = repo.select_rebase_target(worktree)
    else:
        target, reason = target_ref, target_reason or target_ref
    log(f"rebase target: {reason}")
    return target


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


_WORKFLOW_STATUSES_WAITING_FOR_MORE_CHECKS = {
    "action_required",
    "waiting",
    "requested",
    "queued",
    "pending",
    "in_progress",
}


def _has_workflow_gate_for_more_checks(
    run_state_cache: dict[int, tuple[str | None, str | None]],
) -> bool:
    """True when workflow-run state says more CI can still appear.

    ``gh pr checks`` and the workflow-run API can momentarily disagree. If a
    workflow is queued/in progress/action_required, a tiny passed check set is
    not wedged; we should keep waiting or approving rather than push a rebase.
    """
    for status, conclusion in run_state_cache.values():
        if status in _WORKFLOW_STATUSES_WAITING_FOR_MORE_CHECKS:
            return True
        if status == "completed" and conclusion == "action_required":
            return True
    return False


def _green_check_count_is_sparse(status: str, checks: list[dict]) -> bool:
    return (
        status == "passed"
        and TRUNK_LABEL is not None
        and len(checks) < MIN_GREEN_CHECKS_TO_TRUST
    )


def _sparse_green_needs_base_refresh(
    status: str,
    checks: list[dict],
    run_state_cache: dict[int, tuple[str | None, str | None]],
) -> bool:
    return _green_check_count_is_sparse(
        status, checks
    ) and not _has_workflow_gate_for_more_checks(run_state_cache)


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
    ``gh run rerun --failed`` for the underlying workflow run. Bounded to
    one rerun per run: ``intervened_run_ids`` catches repeats within this
    process, and GitHub's ``run_attempt`` catches reruns from a previous
    (restarted) shepherd -- if the failure persists past one rerun,
    claude takes over.

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
        attempt = github.workflow_run_attempt(run_id)
        if attempt is not None and attempt > 1:
            log(
                f"intervention {itv.name!r} matched on {name!r} "
                f"(run {run_id}), but the run is already on attempt "
                f"{attempt} (rerun before this shepherd started); "
                f"falling through to claude"
            )
            intervened_run_ids.add(run_id)
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


def _current_spurious_failure_names(
    checks: list[dict], spurious_names: set[str]
) -> set[str]:
    """Return remembered-spurious checks that are still failing now."""
    if not spurious_names:
        return set()
    return {
        c.get("name")
        for c in checks
        if c.get("bucket") in {"fail", "cancel"}
        and c.get("name") in spurious_names
    }


def _filter_spurious_failed_jobs(
    failed: list[tuple[str, str]], spurious_names: set[str]
) -> list[tuple[str, str]]:
    """Remove failed-job logs already classified as spurious."""
    if not spurious_names:
        return failed
    return [(name, text) for name, text in failed if name not in spurious_names]


def _apply_merge_i_ignored_checks(
    trust: TrustDB,
    comments: list[dict],
    checks: list[dict],
    spurious_check_names: set[str],
    *,
    since_iso: str,
    note: str = "from merge -i",
) -> set[str]:
    ignored = mergebot_ignored_check_names(
        comments, checks, since_iso=since_iso
    )
    # Keep lint active so the normal log/fixer path can inspect concrete
    # lintrunner diagnostics instead of inheriting mergebot's broad ignore.
    ignored_lint = {
        name
        for name in ignored - spurious_check_names
        if "lint" in name.lower()
    }
    if ignored_lint:
        log(
            f"{PROJECT.mergebot_login} / merge -i is ignoring lint check"
            f"{'' if len(ignored_lint) == 1 else 's'} "
            f"{note}, but mergedog will inspect/fix them: "
            f"{', '.join(sorted(ignored_lint))}"
        )
    newly_ignored = (ignored - ignored_lint) - spurious_check_names
    if not newly_ignored:
        return set()
    spurious_check_names |= newly_ignored
    trust.spurious_check_names = sorted(spurious_check_names)
    trust.save()
    log(
        f"{PROJECT.mergebot_login} / merge -i is ignoring "
        f"{len(newly_ignored)} failed check"
        f"{'' if len(newly_ignored) == 1 else 's'} "
        f"{note}; continuing with remaining CI"
    )
    return newly_ignored


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


def _ci_status_message(status: str, failed: int) -> str:
    """Qualitative CI state for the mux Status column.

    Progress counts (``done/total``) and suppressed-failure counts are
    intentionally omitted: the mux renders those in the dedicated CI-progress
    bar and Sup columns, so repeating them here would just be noise. Active
    failures stay in the text -- they have no dedicated column.
    """
    if status == "pending":
        return "waiting for CI"
    if status == "failed":
        failures = (
            f"{failed} active failure"
            if failed == 1
            else f"{failed} active failures"
        )
        return f"CI failed: {failures}"
    if status == "passed":
        return "CI passed"
    return f"CI {status}"


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


def _latest_trusted_approval_sha(pr: int) -> str | None:
    audit = github.get_pr_review_audit(pr)
    latest = latest_trusted_approval(audit["reviews"])
    return latest["commit_id"] if latest else None


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


def _refresh_status_prefix(
    pr: int,
) -> tuple[
    bool | None,
    bool | None,
    str | None,
    str | None,
    list[str] | None,
]:
    """Toggle the [MERGING]/[APPROVED] log prefix based on the PR's state.

    Called once per main-poll iteration. The prefix is the only signal
    the mux has back from each shepherd that pytorchmergebot is actively
    merging the PR (or that the PR is approved and waiting) -- the mux
    only reads the last log line per shepherd, so we have to thread the
    state through the log itself.

    Also returns the live label names from the same lightweight PR poll so
    the CI loop does not rely on a stale startup PR snapshot.

    Failures are silently swallowed: a bad ``gh`` call here shouldn't HALT
    over a UI nicety.
    """
    try:
        labels, decision, head_sha, merge_state = github.get_pr_poll_fields(pr)
    except Exception:
        return None, None, None, None, None
    merging = github.MERGING_LABEL in labels
    approved = (decision or "").upper() == "APPROVED"
    set_merging(merging)
    set_approved(approved)
    return approved, merging, head_sha, merge_state, labels


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
) -> tuple[bool, TrustDB]:
    trust = TrustDB.load_or_create(dep.parent_pr)
    if (
        trust.head_branch != dep.parent_head_ref
        or trust.head_repo_clone_url != REPO_SSH_URL
    ):
        # Write only when the metadata actually changes: the parent PR may
        # have its own live shepherd, and an unconditional load-modify-save
        # on every child poll can clobber state that shepherd persisted
        # between our load and save.
        trust.head_branch = dep.parent_head_ref
        trust.head_repo_clone_url = REPO_SSH_URL
        trust.save()
    if trust.is_trusted(current_sha):
        return True, trust
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
    return trust.is_trusted(current_sha), trust


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

    parent_head_sha = github.get_pr_head_sha(dep.parent_pr)
    checks = github.get_pr_checks_all(dep.parent_pr, head_sha=parent_head_sha)
    parent_trust, parent_trust_db = _check_trusted_ghstack_parent(
        dep, parent_head_sha
    )
    if parent_trust:
        effective_checks = _apply_spurious_overrides(
            checks, set(parent_trust_db.spurious_check_names)
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
        pr=pr,
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
    new_head_sha = _ghstack_submit_trusted(
        worktree,
        dep.child_head_ref,
        trust,
        "Propagate parent update downstream",
    )
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

    Defense in depth: even for trusted PRs, an LLM screen looks at the
    rendered sidecar; if it flags an injection attempt we degrade to the
    ``trusted=False`` rendering. Best-effort only -- the screen fails
    open (see injection.py) and the prompt's untrusted-data framing
    remains the primary defense.
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
    if trusted and injection.looks_like_injection(
        text, source=f"PR #{pr} sidecar"
    ):
        log(
            f"PR #{pr}: sidecar flagged by injection screen; "
            "degrading to bot-comments-only context"
        )
        text = context_mod.render_context(
            pr=pr,
            url=pr_data.get("url", ""),
            title=pr_data.get("title", ""),
            body=pr_data.get("body", "") or "",
            comments=comments,
            trusted=False,
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
    _wait_for_no_active_sev(reason, ignore_sev=ignore_sev, pr=pr)
    repo.push_to_fork(worktree, fork_remote, branch)
    _wait_for_pr_head(pr, new_sha)


def _ghstack_submit_trusted(
    worktree: Path, head_ref: str, trust: TrustDB, message: str
) -> str:
    """``ghstack submit`` and trust the resulting /head, restart-safely.

    We can't know the synthetic /head SHA until after ghstack pushes, so
    unlike regular pushes the new head can't be trusted up front. Record
    the local /orig commit first: a kill inside the push->fetch->trust
    window would otherwise leave the PR head permanently untrusted (the
    next run halts with "head moved to untrusted commit"). On restart,
    ``_recover_pending_ghstack_publish`` re-trusts a head whose /orig
    patch-id matches this record.
    """
    trust.pending_publish_orig_sha = repo.head_sha(worktree)
    trust.save()
    repo.ghstack_submit(worktree, message)
    new_head_sha = repo.fetch_ghstack_head(head_ref)
    trust.pending_publish_orig_sha = ""
    trust.trust(new_head_sha)
    trust.save()
    return new_head_sha


def _recover_pending_ghstack_publish(
    trust: TrustDB, pr_data: dict, head_ref: str
) -> None:
    """Re-trust a PR head left dangling by an interrupted ghstack submit.

    ghstack may rewrite the /orig commit on submit (source-id metadata),
    so the recovery matches by patch-id rather than SHA: if origin's
    current /orig carries the same diff we recorded before pushing, the
    publish was ours and the synthetic /head it produced is trusted.
    """
    pending = trust.pending_publish_orig_sha
    if not pending:
        return
    head_sha = str(pr_data.get("headRefOid") or "")
    if head_sha and not trust.is_trusted(head_sha):
        try:
            orig_sha = repo.fetch_ghstack_orig(head_ref)
        except Exception as e:
            # Leave the record in place: trust verification will halt
            # below anyway, and a later restart can still recover.
            log(
                f"WARNING: could not fetch /orig to verify interrupted "
                f"ghstack publish: {e}"
            )
            return
        if repo.patch_id_matches_any(orig_sha, [pending]):
            trust.trust(head_sha)
            log(
                f"recovered interrupted ghstack publish: /orig "
                f"{orig_sha[:12]} matches the commit recorded before the "
                f"submit; trusting head {head_sha[:12]}"
            )
        else:
            log(
                "interrupted ghstack publish record does not match "
                "origin's current /orig; leaving head untrusted"
            )
    trust.pending_publish_orig_sha = ""
    trust.save()


def _publish_ghstack_fix(
    pr: int,
    worktree: Path,
    head_ref: str,
    fix_sha: str,
    trust: TrustDB,
    *,
    ignore_sev: bool,
) -> str:
    """Fold claude's [MERGEDOG] commit into /orig and re-publish via ghstack.

    Fixup (not squash): claude's commit message is dropped from the resulting
    /orig commit -- /orig keeps the contributor's original message -- and is
    instead passed to ``ghstack submit -m`` so it lands as the submit's audit
    message. After ghstack pushes, fetch the new synthetic /head SHA from
    origin and trust it before the polling loop sees it on GitHub's side.
    """
    # Capture claude's full [MERGEDOG] message before fixup discards it.
    fix_message = repo.commit_message(worktree, fix_sha)
    audit_ref = f"refs/heads/mergedog/{pr}/{fix_sha}"
    _wait_for_no_active_sev(
        "pushing ghstack LLM audit commit", ignore_sev=ignore_sev, pr=pr
    )
    repo.push_ref(worktree, "origin", fix_sha, audit_ref)
    log(
        f"pushed ghstack LLM audit commit {fix_sha[:12]} to "
        f"origin/{audit_ref.removeprefix('refs/heads/')}"
    )
    repo.fixup_into_parent(worktree)
    _wait_for_no_active_sev(
        "re-publishing via ghstack submit", ignore_sev=ignore_sev, pr=pr
    )
    new_head_sha = _ghstack_submit_trusted(
        worktree, head_ref, trust, fix_message
    )
    log(
        f"ghstack submitted; new {head_ref} = {new_head_sha[:12]}"
    )
    _wait_for_pr_head(pr, new_head_sha)
    return new_head_sha


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
    viewer: str | None = None,
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
    _consume_fix_budget(trust)
    if is_ghstack:
        new_head_sha = _publish_ghstack_fix(
            pr, worktree, branch, new_sha, trust, ignore_sev=ignore_sev
        )
        _post_llm_hunk_comments(
            pr, worktree, new_sha, commit_id=new_head_sha, author=viewer
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
        _post_llm_hunk_comments(pr, worktree, new_sha, author=viewer)
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
) -> bool:
    """Rebase /orig onto main and re-publish via ghstack.

    By default, target selection mirrors
    ``_merge_main_resolving_conflicts``: we pick viable/strict or a recent
    revert rather than raw trunk tip. Callers handling GitHub mergeability
    failures can pass an explicit target such as ``origin/main``. Returns
    whether the rebase published a new head.
    """
    if _wait_for_no_active_sev(
        "rebasing /orig onto main", ignore_sev=ignore_sev, pr=pr
    ):
        repo.fetch_origin()

    target = _select_refresh_target(worktree, target_ref, target_reason)

    try:
        status, new_orig_sha = repo.attempt_rebase_main(worktree, ref=target)
    except RuntimeError as e:
        die(str(e))

    if status == "noop":
        log("rebase produced no new commit (already at target)")
        return False

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
        "re-publishing rebased /orig via ghstack submit",
        ignore_sev=ignore_sev,
        pr=pr,
    )
    new_head_sha = _ghstack_submit_trusted(
        worktree, head_ref, trust, "Rebase onto origin/main"
    )
    log(f"ghstack submitted; new {head_ref} = {new_head_sha[:12]}")
    _wait_for_pr_head(pr, new_head_sha)
    return True


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


def _inline_hunk_key(
    sha: str, target: repo.DiffHunkCommentTarget
) -> str:
    h = hashlib.sha256()
    h.update(sha.encode())
    h.update(b"\0")
    h.update(target.path.encode())
    h.update(b"\0")
    h.update(target.side.encode())
    h.update(b"\0")
    h.update(str(target.line).encode())
    return h.hexdigest()


def _inline_hunk_comment_body(sha: str, key: str) -> str:
    short = sha[:12]
    return (
        f"<!-- mergedog:inline-hunk sha={sha} key={key} -->\n"
        f"This hunk was LLM-generated by mergedog in "
        f"[commit `{short}`](https://github.com/{REPO_SLUG}/commit/{sha}). "
        "Review that commit for context and justification."
    )


def _post_llm_hunk_comments(
    pr: int,
    worktree: Path,
    sha: str,
    *,
    commit_id: str | None = None,
    author: str | None = None,
) -> None:
    try:
        targets = repo.diff_hunk_comment_targets(worktree, sha)
    except Exception as e:
        log(f"WARNING: failed to inspect LLM commit hunks for {sha[:12]}: {e}")
        return
    if not targets:
        return

    try:
        existing = github.get_pr_review_comments(pr)
    except Exception as e:
        log(
            f"WARNING: failed to fetch existing inline mergedog comments "
            f"for {sha[:12]}: {e}"
        )
        return

    # Only treat our own prior markers as dedup anchors; a spoofed marker
    # from another commenter must not suppress a real annotation.
    existing_keys = {
        match.group(2)
        for c in existing
        if github._is_own_comment(c, author)
        for match in [_INLINE_HUNK_MARKER_RE.search(c.get("body") or "")]
        if match is not None and match.group(1) == sha
    }
    posted = 0
    for target in targets:
        key = _inline_hunk_key(sha, target)
        if key in existing_keys:
            continue
        try:
            github.post_pr_review_comment(
                pr,
                body=_inline_hunk_comment_body(sha, key),
                commit_id=commit_id or sha,
                path=target.path,
                line=target.line,
                side=target.side,
            )
        except Exception as e:
            log(
                f"WARNING: failed to post inline mergedog marker for "
                f"{sha[:12]} {target.path}:{target.line}: {e}"
            )
            continue
        existing_keys.add(key)
        posted += 1
    if posted:
        log(f"posted {posted} inline mergedog hunk marker(s) for {sha[:12]}")


def _screen_failed_job_logs(
    pr: int, failed: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Withhold CI log excerpts the injection screen flags.

    Defense in depth, fail-open (see injection.py). The withheld notice
    keeps the job entry present so the fixer sees the check exists but
    lacks evidence to classify it -- the prompt directs it to signal
    INCONCLUSIVE in that case, which halts for human review.
    """
    out: list[tuple[str, str]] = []
    for name, text in failed:
        if injection.looks_like_injection(
            text, source=f"PR #{pr} CI log {name!r}"
        ):
            log(
                f"PR #{pr}: withholding CI log for {name!r} "
                "(flagged by injection screen)"
            )
            text = (
                "<log excerpt withheld: flagged as a possible "
                "prompt-injection attempt; treat this failure as lacking "
                "evidence>"
            )
        out.append((name, text))
    return out


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
    """Merge main into HEAD, resolving conflicts via claude.

    Returns the new head SHA if a merge commit was made, else None
    (already up to date). Trusts the new SHA. Caller is responsible
    for pushing.

    By default, target selection avoids raw trunk tip and picks the best
    known-good ref via ``select_rebase_target`` -- viable/strict, a recent
    revert commit, or staying put if nothing is ahead of us. Callers
    handling GitHub mergeability failures can pass an explicit target such
    as ``origin/main``.
    """
    if _wait_for_no_active_sev(
        "merging main", ignore_sev=ignore_sev, pr=pr_data["number"]
    ):
        repo.fetch_origin()

    target = _select_refresh_target(worktree, target_ref, target_reason)

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


def _sync_fix_budget(
    trust: TrustDB,
    human_ack_sha: str | None,
    *,
    reassess: bool = False,
    self_pr: bool = False,
) -> int:
    """Restore the persisted fix-commit count for the --max-fix-commits cap.

    The cap is a thrash guard; an in-memory count meant every restart
    granted a fresh budget. The count is scoped to the human approval
    baseline: a new maintainer approval (or an explicit --reassess)
    legitimately resets it. On self-authored PRs the "ack" is just the
    current head -- which mergedog's own pushes move -- so there the
    budget persists until --reassess.
    """
    if reassess:
        trust.fix_budget_ack_sha = human_ack_sha or ""
        trust.fix_commits_pushed = 0
        trust.save()
        return 0
    ack = human_ack_sha or ""
    if not self_pr and trust.fix_budget_ack_sha != ack:
        trust.fix_budget_ack_sha = ack
        trust.fix_commits_pushed = 0
        trust.save()
    elif trust.fix_commits_pushed:
        log(
            f"restored fix-commit count from previous run: "
            f"{trust.fix_commits_pushed}"
        )
    return trust.fix_commits_pushed


def _consume_fix_budget(trust: TrustDB) -> int:
    """Persist one unit of fix budget.

    Called *before* the push it pays for: if we're killed mid-push the
    budget is already spent, so a restart can't turn the cap into a
    fresh allowance (the safe failure mode for a runaway-thrash guard).
    """
    trust.fix_commits_pushed += 1
    trust.save()
    return trust.fix_commits_pushed


def _classify_failure_body(body: str) -> str:
    if not body:
        return "none"
    if is_cla_merge_failure(body):
        return "CLA"
    if is_merge_conflict_failure(body):
        return "merge conflict"
    if is_retryable_merge_failure(body):
        return "retryable infra flake"
    return "unclassified"


def _recover_from_merge_conflict(
    pr: int,
    worktree: Path,
    branch: str,
    trust: TrustDB,
    pr_data: dict,
    sessions: list[ClaudeSession],
    pushed_changes: list[PushedChange],
    *,
    is_ghstack: bool,
    fork_remote: str | None,
    ignore_sev: bool,
    trusted_pr: bool,
    change_summary: str,
) -> None:
    """Refresh the PR branch from main after a merge conflict.

    Shared by the post-handoff conflict watcher, the mergebot
    merge-conflict failure path, and the startup replay of a persisted
    conflict failure.

    Try the normal known-good refresh first. GitHub and pytorchmergebot
    report mergeability against the live base branch, so if known-good
    cannot advance the branch and a local probe confirms that live
    ``origin/main`` still conflicts, fall back to live main.
    """
    repo.fetch_origin()
    if is_ghstack:
        advanced = _rebase_ghstack_onto_main(
            pr, worktree, branch, trust, ignore_sev=ignore_sev,
            pr_data=pr_data, sessions=sessions,
        )
        if not advanced and repo.would_merge_conflict(worktree, "origin/main"):
            log(
                "known-good conflict recovery produced no new commit, "
                "but origin/main still conflicts; rebasing /orig onto "
                "origin/main"
            )
            _rebase_ghstack_onto_main(
                pr, worktree, branch, trust, ignore_sev=ignore_sev,
                pr_data=pr_data, sessions=sessions,
                target_ref="origin/main",
                target_reason="origin/main (merge conflict fallback)",
            )
    else:
        assert fork_remote is not None
        new_sha = _merge_main_resolving_conflicts(
            worktree, trust, branch, pr_data, sessions,
            ignore_sev=ignore_sev, trusted_pr=trusted_pr,
        )
        if new_sha is None and repo.would_merge_conflict(worktree, "origin/main"):
            log(
                "known-good conflict recovery produced no new commit, "
                "but origin/main still conflicts; merging origin/main"
            )
            new_sha = _merge_main_resolving_conflicts(
                worktree, trust, branch, pr_data, sessions,
                ignore_sev=ignore_sev, trusted_pr=trusted_pr,
                target_ref="origin/main",
                target_reason="origin/main (merge conflict fallback)",
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
                pushed_changes, worktree, new_sha, change_summary
            )


def _refresh_base_and_push(
    pr: int,
    worktree: Path,
    branch: str,
    trust: TrustDB,
    pr_data: dict,
    sessions: list[ClaudeSession],
    pushed_changes: list[PushedChange],
    spurious_check_names: set[str],
    *,
    is_ghstack: bool,
    fork_remote: str | None,
    ignore_sev: bool,
    trusted_pr: bool,
    no_advance_message: str,
    push_reason: str,
    change_summary: str,
    target_ref: str | None = None,
    target_reason: str | None = None,
) -> None:
    """Merge/rebase the PR onto a fresh base, push it, and clear spurious
    judgments (fresh CI invalidates them).

    Dies with ``no_advance_message`` if the refresh produced no new
    commit on the non-ghstack path.
    """
    if is_ghstack:
        _rebase_ghstack_onto_main(
            pr, worktree, branch, trust,
            ignore_sev=ignore_sev, pr_data=pr_data, sessions=sessions,
            target_ref=target_ref, target_reason=target_reason,
        )
    else:
        assert fork_remote is not None
        merge_sha = _merge_main_resolving_conflicts(
            worktree, trust, branch, pr_data, sessions,
            ignore_sev=ignore_sev, trusted_pr=trusted_pr,
            target_ref=target_ref, target_reason=target_reason,
        )
        if merge_sha is None:
            die(no_advance_message)
        log(
            f"pushing refreshed base {merge_sha[:12]} "
            f"to {fork_remote}/{branch}"
        )
        _safe_push(
            pr, worktree, fork_remote, branch, merge_sha,
            reason=push_reason, ignore_sev=ignore_sev,
        )
        _record_pushed_change(
            pushed_changes, worktree, merge_sha, change_summary
        )
    spurious_check_names.clear()
    trust.spurious_check_names = []
    trust.save()


def _log_restored_state(trust: TrustDB) -> None:
    """Surface persisted idempotency state that will steer this run.

    Restored state silently changing behavior is how restarts get
    "stuck": a spurious judgment or failure floor from a previous run
    suppresses work the operator expects to happen. Logging it up front
    makes the suppression visible and points at the escape hatch.
    """
    restored: list[str] = []
    if trust.trusted_shas:
        restored.append(f"  trusted SHAs: {len(trust.trusted_shas)}")
    if trust.spurious_check_names:
        names = ", ".join(trust.spurious_check_names)
        restored.append(
            f"  spurious judgments (reassess to clear): {names}"
        )
    if trust.last_observed_failure_iso:
        kind = _classify_failure_body(trust.last_observed_failure_body)
        restored.append(
            f"  last observed merge failure: "
            f"{trust.last_observed_failure_iso} ({kind}); older mergebot "
            f"comments will not re-trigger recovery"
        )
    if trust.merge_auto_retries:
        restored.append(
            f"  merge auto-retries used: {trust.merge_auto_retries}"
        )
    if trust.pending_publish_orig_sha:
        restored.append(
            f"  interrupted ghstack publish of /orig "
            f"{trust.pending_publish_orig_sha[:12]} (will verify and "
            f"re-trust)"
        )
    if restored:
        log("restored state from previous run:")
        for line in restored:
            log(line)


def _sigterm_to_systemexit(signum, frame) -> None:  # type: ignore[no-untyped-def]
    """Turn SIGTERM into SystemExit so ``finally`` blocks run on shutdown.

    ``mux cancel`` sends SIGTERM to the shepherd's process group; without a
    handler Python exits abruptly and in-flight cleanup (status writes,
    worktree locks) is skipped. Raising SystemExit lets normal unwinding run.
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

    # The ``mergedog`` label is owned by the mux: it marks a PR as tracked
    # long-term and is added/removed only as the PR joins or leaves mux. The
    # shepherd deliberately never touches it, so finishing -- or crashing --
    # leaves the label exactly as the mux set it.
    signal.signal(signal.SIGTERM, _sigterm_to_systemexit)
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1)
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
    _log_restored_state(trust)
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
    elif trust.pending_publish_orig_sha:
        # A previous run died between ``ghstack submit`` and trusting the
        # resulting /head. Re-establish trust before the approval seeding
        # and head-trust checks below, which would otherwise halt.
        _recover_pending_ghstack_publish(trust, pr_data, branch)

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

    fix_commits_pushed = _sync_fix_budget(
        trust, human_ack_sha, reassess=reassess, self_pr=self_pr
    )
    max_fix_attempts_status = max_fix_commits
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
        message="starting shepherd",
        intervention_count=intervention_count,
        human_ack_sha=human_ack_sha,
        approved=last_approved,
        merging=last_merging,
        fix_attempts=fix_commits_pushed,
        max_fix_attempts=max_fix_attempts_status,
    )
    # Auto-retries for infra-flake merge failures (e.g. 504). Capped at
    # MAX_MERGE_AUTO_RETRIES to prevent runaway commenting during outages.
    # Persisted in the trust DB: each retry posts a visible merge command,
    # so a restart must not refresh the budget mid-outage. --reassess (or
    # editing the state file) resets it.
    if reassess and trust.merge_auto_retries:
        trust.merge_auto_retries = 0
        trust.save()
    auto_retries = trust.merge_auto_retries
    if auto_retries:
        log(
            f"restored merge auto-retry count from previous run: "
            f"{auto_retries}/{MAX_MERGE_AUTO_RETRIES}"
        )

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

    # On restart, replay the last observed merge failure through the same
    # classification the post-handoff watcher uses. The failure timestamp
    # floor means the watcher will never re-react to this comment, so any
    # handling it deserves has to happen here -- including handling that
    # only exists because the code was updated since the failure was
    # recorded. Unclassified failures need no replay: the normal loop
    # re-inspects CI and claude judges from scratch.
    replay_kind = _classify_failure_body(trust.last_observed_failure_body)
    if replay_kind == "merge conflict":
        log(
            "prior merge-conflict failure detected on restart; "
            "rebasing onto main"
        )
        _recover_from_merge_conflict(
            pr, worktree, branch, trust, pr_data, sessions, pushed_changes,
            is_ghstack=is_ghstack, fork_remote=fork_remote,
            ignore_sev=ignore_sev, trusted_pr=trusted_pr,
            change_summary="merged main into the PR branch after merge failure",
        )
        trust.last_observed_failure_body = ""
        trust.save()
    elif replay_kind == "retryable infra flake":
        # Only retry if nothing has happened on the PR since the failure
        # -- a newer mergebot event means a human (or a previous
        # mergedog) already acted on it.
        newer_event = None
        try:
            newer_event = latest_mergebot_event(
                pr, trust.last_observed_failure_iso
            )
        except Exception as e:
            log(f"WARNING: could not check for newer mergebot events: {e}")
        if newer_event is not None:
            log(
                "prior retryable merge failure detected on restart, but "
                "there is newer mergebot activity; leaving it to the "
                "normal flow"
            )
        elif (
            PROJECT.merge_command is not None
            and auto_retries < MAX_MERGE_AUTO_RETRIES
        ):
            auto_retries += 1
            trust.merge_auto_retries = auto_retries
            trust.last_observed_failure_body = ""
            trust.save()
            log(
                f"prior merge failure was a retryable infra flake with no "
                f"activity since; auto-retrying `{PROJECT.merge_command}` "
                f"({auto_retries}/{MAX_MERGE_AUTO_RETRIES})"
            )
            github.post_pr_comment(pr, PROJECT.merge_command)
        else:
            log(
                "prior retryable merge failure detected on restart, but "
                "no auto-retry budget remains; falling through to manual "
                "recovery"
            )

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
            viewer=viewer,
        ):
            fix_commits_pushed = trust.fix_commits_pushed

    run_state_cache: dict[int, tuple[str | None, str | None]] = {}
    last_approval_refresh_at = time.time()

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
        check_poll_cache = _CiCheckPollCache()
        trunk_ci_gate: _TrunkCiGate | None = None

        # Poll CI, fix or judge spurious until ready for handoff. Breaks
        # out (via the handoff path) when CI is green and the trunk
        # label is on.
        while True:
            approved, merging, current, merge_state, observed_labels = (
                _refresh_status_prefix(pr)
            )
            if approved is not None:
                last_approved = approved
            if merging is not None:
                last_merging = merging
            if TRUNK_LABEL is not None and observed_labels is not None:
                observed_trunk_applied = TRUNK_LABEL in observed_labels
                if observed_trunk_applied != trunk_applied:
                    if observed_trunk_applied:
                        log(
                            f"{TRUNK_LABEL} label is already present; "
                            "continuing with trunk CI"
                        )
                    else:
                        log(
                            f"{TRUNK_LABEL} label is no longer present; "
                            "will request trunk CI again"
                        )
                        trunk_ci_gate = None
                        stable_observation = None
                    trunk_applied = observed_trunk_applied
            # 1. Verify the PR head is still trusted.
            if current is None:
                current = github.get_pr_head_sha(pr)
            if self_pr:
                # On a self-authored PR, every push is implicitly approved
                # by the operator -- roll the trust forward instead of
                # halting.
                trust.trust(current)
            if (
                approved
                and not self_pr
                and (
                    not trust.is_trusted(current)
                    or time.time() - last_approval_refresh_at
                    >= APPROVAL_REFRESH_INTERVAL_SEC
                )
            ):
                try:
                    refreshed_ack_sha = _latest_trusted_approval_sha(pr)
                except Exception as e:
                    log(f"WARNING: could not refresh approval baseline: {e}")
                else:
                    last_approval_refresh_at = time.time()
                    if refreshed_ack_sha:
                        trust.trust(refreshed_ack_sha)
                        human_ack_sha = refreshed_ack_sha
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
            if (
                merging is False
                and (merge_state or "").upper() == "DIRTY"
            ):
                recovery_attempts += 1
                log(
                    "GitHub reports branch conflicts before handoff; "
                    "rebasing onto main"
                )
                _recover_from_merge_conflict(
                    pr, worktree, branch, trust, pr_data, sessions,
                    pushed_changes,
                    is_ghstack=is_ghstack, fork_remote=fork_remote,
                    ignore_sev=ignore_sev, trusted_pr=trusted_pr,
                    change_summary=(
                        "merged main into the PR branch after GitHub "
                        "reported pre-handoff conflicts"
                    ),
                )
                last_status = None
                stable_observation = None
                pr_data = github.get_pr(pr)
                continue

            # 2. Approve any approval-pending workflow runs.
            approved = _approve_pending_runs(current, run_state_cache)
            if approved:
                empty_log_defers = 0
                stable_observation = None  # newly-approved runs invalidate stability
                check_poll_cache.invalidate()
                time.sleep(APPROVAL_SETTLE_SEC)
                continue

            # 3. Read check status. Failures claude already judged
            # spurious are flipped to "skipping" so we don't re-judge
            # them, and so the overall verdict reflects what's still
            # genuinely outstanding.
            workflow_fingerprint = _workflow_state_fingerprint(run_state_cache)
            check_poll_now = time.time()
            fetched_full_checks = check_poll_cache.should_fetch(
                head_sha=current,
                workflow_fingerprint=workflow_fingerprint,
                now=check_poll_now,
            )
            if fetched_full_checks:
                checks = github.get_pr_checks_all(pr, head_sha=current)
            else:
                checks = list(check_poll_cache.checks or [])
            effective_checks = _apply_spurious_overrides(
                checks, spurious_check_names
            )
            status = github.evaluate_checks(effective_checks)
            workflow_gate_for_more_checks = _has_workflow_gate_for_more_checks(
                run_state_cache
            )

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
            current_spurious_failures = _current_spurious_failure_names(
                checks, spurious_check_names
            )
            suppressed_failed_count = len(current_spurious_failures)
            workflow_failed_run_ids: list[int] = []
            if status == "passed" and not all_failures_spurious:
                workflow_failed_run_ids = [
                    run_id
                    for run_id, (_st, concl) in run_state_cache.items()
                    if concl == "failure"
                ]
                if workflow_failed_run_ids:
                    log(
                        f"gh pr checks says passed but workflow run(s) "
                        f"{workflow_failed_run_ids} have conclusion=failure; "
                        f"treating as failed"
                    )
                    status = "failed"

            if (
                _green_check_count_is_sparse(status, checks)
                and workflow_gate_for_more_checks
            ):
                status = "pending"
            waiting_for_trunk_wave = False
            if trunk_ci_gate is not None:
                if _trunk_wave_has_started(
                    trunk_ci_gate,
                    head_sha=current,
                    check_count=len(checks),
                    workflow_fingerprint=workflow_fingerprint,
                ):
                    log(f"{TRUNK_LABEL} workflows appeared; resuming CI wait")
                    trunk_ci_gate = None
                    stable_observation = None
                else:
                    # The trunk label was just applied, but GitHub is still
                    # reporting the pre-label check set. Do not let that old
                    # green observation satisfy handoff readiness.
                    status = "pending"
                    waiting_for_trunk_wave = True
            if fetched_full_checks:
                if waiting_for_trunk_wave:
                    check_poll_cache.invalidate()
                else:
                    check_poll_cache.update(
                        head_sha=current,
                        workflow_fingerprint=workflow_fingerprint,
                        checks=checks,
                        status=status,
                        fetched_at=check_poll_now,
                    )

            done = sum(
                1 for c in checks if c.get("bucket") not in {"pending", None}
            )
            failed_count = sum(
                1 for c in checks if c.get("bucket") in {"fail", "cancel"}
            )
            # Check names are workflow-defined (workflow YAML name/matrix in
            # the approved tree, or repo-authorized check apps like dr. ci),
            # not contributor-authored free text -- so they reach the prompt
            # untainted. This is why they aren't wrapped like ci_log excerpts;
            # the asymmetry is deliberate, not a missed source.
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
            ci_message = _ci_status_message(status, active_failed_count)
            if waiting_for_trunk_wave:
                ci_message = f"waiting for {TRUNK_LABEL} workflows to appear"
            _write_status_best_effort(
                pr,
                phase="polling_ci",
                category="waiting" if status == "pending" else "action",
                waiting_on="ci" if status == "pending" else None,
                action="inspecting_ci" if status != "pending" else None,
                message=ci_message,
                intervention_count=intervention_count,
                human_ack_sha=human_ack_sha,
                approved=last_approved,
                merging=last_merging,
                ci_done=done,
                ci_total=len(checks),
                ci_failed=active_failed_count,
                ci_suppressed=suppressed_failed_count,
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
                            message=(
                                f"waiting for stack parent "
                                f"#{ghstack_parent.parent_pr}: "
                                f"{parent_status.reason}; parent "
                                f"{parent_status.parent_done}/"
                                f"{parent_status.parent_total} checks done"
                            ),
                            intervention_count=intervention_count,
                            human_ack_sha=human_ack_sha,
                            approved=last_approved,
                            merging=last_merging,
                            ci_done=done,
                            ci_total=len(checks),
                            ci_failed=active_failed_count,
                            ci_suppressed=suppressed_failed_count,
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
                    comments, author=viewer
                )
                ignore_since = handoff_iso or trust.last_observed_failure_iso
                newly_ignored = _apply_merge_i_ignored_checks(
                    trust,
                    comments,
                    checks,
                    spurious_check_names,
                    since_iso=ignore_since,
                    note="from merge -i",
                )
                if newly_ignored:
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
                    message=_fixing_ci_message(
                        active_failed_count,
                        fix_commits_pushed,
                        max_fix_attempts_status,
                    ),
                    intervention_count=intervention_count,
                    human_ack_sha=human_ack_sha,
                    approved=last_approved,
                    merging=last_merging,
                    ci_done=done,
                    ci_total=len(checks),
                    ci_failed=active_failed_count,
                    ci_suppressed=suppressed_failed_count,
                    fix_attempts=fix_commits_pushed,
                    max_fix_attempts=max_fix_attempts_status,
                )
                ctx_path, comments = _refresh_context_file(
                    pr_data, trusted=trusted_pr
                )
                trunk_ctx = repo.trunk_revert_context(worktree)
                if trunk_ctx:
                    log(f"injecting trunk revert context into {_llm_label()} prompt")
                failed = _screen_failed_job_logs(pr, failed)
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
                    extra_context=extra_context or None,
                    trunk_revert_context=trunk_ctx,
                )
                session_failed_jobs = [name for name, _ in failed]
                sha_before = current
                started_at = utc_now_iso()
                result = claude_mod.invoke_fixer(worktree, prompt)
                ran_cleanly, new_sha, transcript = result
                newly_ignored_during_llm = _apply_merge_i_ignored_checks(
                    trust,
                    github.get_pr_comments(pr),
                    checks,
                    spurious_check_names,
                    since_iso=ignore_since,
                    note=f"while {_llm_label()} was running",
                )
                if newly_ignored_during_llm:
                    if new_sha is not None or not ran_cleanly:
                        repo.set_worktree_to_sha(worktree, sha_before)
                    _record_claude_session(
                        sessions,
                        mode="fix-CI",
                        sha_before=sha_before,
                        started_at=started_at,
                        ran_cleanly=True,
                        new_sha=None,
                        transcript=transcript,
                        on_commit="pushed fix commit {sha}",
                        on_clean_noop="superseded by merge -i ignore (no commit)",
                        extra=(
                            " — failing jobs: "
                            f"{', '.join(session_failed_jobs)}"
                            if session_failed_jobs
                            else ""
                        ),
                    )
                    last_status = None
                    continue
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
                            _refresh_base_and_push(
                                pr, worktree, branch, trust, pr_data,
                                sessions, pushed_changes,
                                spurious_check_names,
                                is_ghstack=is_ghstack,
                                fork_remote=fork_remote,
                                ignore_sev=ignore_sev,
                                trusted_pr=trusted_pr,
                                no_advance_message=(
                                    f"{_llm_label()} requested REBASE, "
                                    "but the selected rebase target no longer "
                                    "advanced the PR; halting for human review"
                                ),
                                push_reason="pushing requested base refresh",
                                change_summary="merged main into the PR branch after LLM requested rebase",
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
                            _refresh_base_and_push(
                                pr, worktree, branch, trust, pr_data,
                                sessions, pushed_changes,
                                spurious_check_names,
                                is_ghstack=is_ghstack,
                                fork_remote=fork_remote,
                                ignore_sev=ignore_sev,
                                trusted_pr=trusted_pr,
                                no_advance_message=(
                                    f"{_llm_label()} signalled INCONCLUSIVE, "
                                    "but the selected rebase target no longer "
                                    "advanced the PR; halting for human review"
                                ),
                                push_reason="pushing known-good base refresh",
                                change_summary="merged main into the PR branch after inconclusive CI",
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
                    fix_commits_pushed = _consume_fix_budget(trust)
                    new_head_sha = _publish_ghstack_fix(
                        pr, worktree, branch, new_sha, trust,
                        ignore_sev=ignore_sev,
                    )
                    _post_llm_hunk_comments(
                        pr, worktree, new_sha,
                        commit_id=new_head_sha, author=viewer,
                    )
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
                        _refresh_base_and_push(
                            pr, worktree, branch, trust, pr_data,
                            sessions, pushed_changes,
                            spurious_check_names,
                            is_ghstack=False,
                            fork_remote=fork_remote,
                            ignore_sev=ignore_sev,
                            trusted_pr=trusted_pr,
                            target_ref="origin/main",
                            target_reason="origin/main (discarded conflicting fix)",
                            no_advance_message=(
                                f"{_llm_label()} fix conflicted with origin/main, "
                                "but refreshing the original PR produced no new "
                                "commit; halting for human review"
                            ),
                            push_reason="pushing base refresh after discarding conflicting fix",
                            change_summary="merged main into the PR branch after discarding a conflicting LLM fix",
                        )
                        last_status = None
                        stable_observation = None
                        continue
                    spurious_check_names.clear()
                    trust.spurious_check_names = []
                    trust.trust(new_sha)
                    fix_commits_pushed = _consume_fix_budget(trust)
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
                    _post_llm_hunk_comments(
                        pr, worktree, new_sha,
                        commit_id=final_sha, author=viewer,
                    )
                    if merge_sha is not None and merge_sha != new_sha:
                        _record_pushed_change(
                            pushed_changes,
                            worktree,
                            merge_sha,
                            "merged main into the PR branch after the fix",
                        )
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
                        message=(
                            f"CI passed; waiting {remaining}s for stability"
                        ),
                        intervention_count=intervention_count,
                        human_ack_sha=human_ack_sha,
                        approved=last_approved,
                        merging=last_merging,
                        ci_done=done,
                        ci_total=len(checks),
                        ci_failed=0,
                        ci_suppressed=suppressed_failed_count,
                        fix_attempts=fix_commits_pushed,
                        max_fix_attempts=max_fix_attempts_status,
                    )
                    log(
                        f"{passed_message}; waiting {remaining}s for stability "
                        f"(no new checks should appear)"
                    )
                    time.sleep(min(POLL_INTERVAL_SEC, remaining))
                    continue

                if (
                    _sparse_green_needs_base_refresh(
                        status, checks, run_state_cache
                    )
                ):
                    log(
                        f"CI reported only {len(checks)} passed check"
                        f"{'' if len(checks) == 1 else 's'} and no pending "
                        "workflow gate; refreshing base to trigger CI"
                    )
                    _write_status_best_effort(
                        pr,
                        phase="refreshing_base",
                        category="action",
                        action="rebasing",
                        message=(
                            f"refreshing base to trigger CI: only "
                            f"{len(checks)} check"
                            f"{'' if len(checks) == 1 else 's'} reported"
                        ),
                        intervention_count=intervention_count,
                        human_ack_sha=human_ack_sha,
                        approved=last_approved,
                        merging=last_merging,
                        ci_done=done,
                        ci_total=len(checks),
                        ci_failed=0,
                        ci_suppressed=suppressed_failed_count,
                        fix_attempts=fix_commits_pushed,
                        max_fix_attempts=max_fix_attempts_status,
                    )
                    # Deliberately bypass the ci:sev gate for this recovery:
                    # the problem is that no real CI wave exists to wait on.
                    repo.fetch_origin()
                    _refresh_base_and_push(
                        pr, worktree, branch, trust, pr_data,
                        sessions, pushed_changes,
                        spurious_check_names,
                        is_ghstack=is_ghstack,
                        fork_remote=fork_remote,
                        ignore_sev=True,
                        trusted_pr=trusted_pr,
                        no_advance_message=(
                            f"CI reported only {len(checks)} passed check"
                            f"{'' if len(checks) == 1 else 's'} and no "
                            "pending workflow gate, but refreshing the "
                            "base produced no new commit; halting for "
                            "human intervention"
                        ),
                        push_reason="pushing base refresh to trigger missing CI",
                        change_summary="merged main into the PR branch to trigger missing CI",
                    )
                    last_status = None
                    stable_observation = None
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
                    f"applying {TRUNK_LABEL} label",
                    ignore_sev=ignore_sev,
                    pr=pr,
                )
                log(f"CI green; applying {TRUNK_LABEL} label")
                github.add_label(pr, TRUNK_LABEL, loud=False)
                log(
                    f"{TRUNK_LABEL} label applied; "
                    "waiting for trunk workflows to appear"
                )
                trunk_applied = True
                trunk_ci_gate = _TrunkCiGate(
                    head_sha=current,
                    check_count=len(checks),
                    workflow_fingerprint=workflow_fingerprint,
                )
                last_status = None
                stable_observation = None
                check_poll_cache.invalidate()
                time.sleep(APPROVAL_SETTLE_SEC)
                continue
            log("ALL CI GREEN.")
            ready_for_merge = last_approved is not False
            approval_actionable = not self_pr
            cla_blocked = is_cla_merge_failure(trust.last_observed_failure_body)
            # Suppressed-failure count lives in the mux Sup column, not the text.
            ci_green_phrase = "CI is green"
            if cla_blocked:
                ready_message = f"waiting for contributor CLA: {ci_green_phrase}"
                ready_user_action = None
                ready_category = "waiting"
                ready_waiting_on = "contributor"
            elif ready_for_merge:
                ready_message = f"ready for human merge: {ci_green_phrase}"
                ready_user_action = (
                    "review mergedog handoff and merge when satisfied"
                )
                ready_category = "ready"
                ready_waiting_on = None
            else:
                ready_message = (
                    f"waiting for maintainer approval: {ci_green_phrase}"
                )
                ready_user_action = (
                    "approve the PR after reviewing mergedog interventions"
                    if approval_actionable
                    else None
                )
                ready_category = "ready" if approval_actionable else "waiting"
                ready_waiting_on = "approval"
            _write_status_best_effort(
                pr,
                phase="ready",
                category=ready_category,
                waiting_on=ready_waiting_on,
                user_action=ready_user_action,
                message=ready_message,
                intervention_count=intervention_count,
                human_ack_sha=human_ack_sha,
                approved=last_approved,
                merging=last_merging,
                ci_done=done,
                ci_total=len(checks),
                ci_failed=failed_count,
                ci_suppressed=suppressed_failed_count,
                fix_attempts=fix_commits_pushed,
                max_fix_attempts=max_fix_attempts_status,
            )
            break

        handoff_started_iso = utc_now_iso()
        suppressed_failures = sorted(
            _current_spurious_failure_names(checks, spurious_check_names)
        )
        drci_summary = _latest_drci_summary_for_handoff(
            pr, current, set(suppressed_failures)
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
            author=viewer,
        )
        # Anchor the watch loop on the actual handoff comment timestamp,
        # not "now": on restart this lets us notice a "Merge failed" that
        # already happened between the last handoff and our restart. But
        # also floor on any failure we've already halted on, so the next
        # restart doesn't re-react to the same stale comment.
        try:
            handoff_iso = (
                github.latest_mergedog_handoff_iso(pr, author=viewer)
                or handoff_started_iso
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
        cla_blocked = is_cla_merge_failure(trust.last_observed_failure_body)
        if cla_blocked:
            log(
                "Hand off to a human reviewer after the contributor CLA is "
                f"complete: {pr_data.get('url', f'PR #{pr}')}."
            )
        else:
            log(
                "Hand off to a human reviewer; have them "
                f"{merge_instruction}{pr_data.get('url', f'PR #{pr}')}."
            )
        if cla_blocked:
            handoff_category = "waiting"
            handoff_waiting_on = "contributor"
            handoff_user_action = None
            handoff_message = "waiting for contributor CLA"
        elif last_approved is False:
            approval_actionable = not self_pr
            handoff_category = "ready" if approval_actionable else "waiting"
            handoff_waiting_on = "approval"
            handoff_user_action = (
                (
                    "approve after reviewing local mergedog log"
                    if handoff_comment_ok is False
                    else "approve the PR after reviewing mergedog interventions"
                )
                if approval_actionable
                else None
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
            message=handoff_message + handoff_warning_suffix,
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
            approval_actionable=not self_pr,
            cla_blocked=cla_blocked,
        )
        if result == "closed":
            complete(
                "PR is no longer open; shepherd complete",
                code=EXIT_PR_NOT_ACTIONABLE,
            )
        if result == "conflict":
            recovery_attempts += 1
            log("post-handoff merge conflict detected; rebasing onto main")
            _recover_from_merge_conflict(
                pr, worktree, branch, trust, pr_data, sessions,
                pushed_changes,
                is_ghstack=is_ghstack, fork_remote=fork_remote,
                ignore_sev=ignore_sev, trusted_pr=trusted_pr,
                change_summary=(
                    "merged main into the PR branch after GitHub "
                    "reported conflicts"
                ),
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
                # Persist before posting: if we're killed mid-post, the
                # budget reads as spent -- the safe direction for a guard
                # whose whole job is bounding comment spam.
                auto_retries += 1
                trust.merge_auto_retries = auto_retries
                trust.save()
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
            _recover_from_merge_conflict(
                pr, worktree, branch, trust, pr_data, sessions,
                pushed_changes,
                is_ghstack=is_ghstack, fork_remote=fork_remote,
                ignore_sev=ignore_sev, trusted_pr=trusted_pr,
                change_summary=(
                    "merged main into the PR branch after merge failure"
                ),
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
