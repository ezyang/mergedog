"""Post-handoff PR comment + watcher loop.

Once CI is green and the trunk label is on, the shepherd posts a summary
of what the configured LLM did during the run, then sits in
:func:`watch_post_handoff`
until either the PR closes/merges or pytorchmergebot reports a merge
failure (which kicks the shepherd into a recovery cycle).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from mergedog import github
from mergedog.log import log, set_approved, set_merging
from mergedog.paths import REPO_SLUG
from mergedog.project import get_project_policy
from mergedog.sanitize import sanitize_untrusted_text, strip_html_comments
from mergedog.status import write_status

PROJECT = get_project_policy()
_MERGEBOT_IGNORE_PHRASE = "will be merged while ignoring the following"


@dataclass
class ClaudeSession:
    """One LLM invocation, captured for the handoff comment."""

    mode: str  # "fix-CI" or "merge-resolver"
    started_at: str  # UTC ISO 8601
    sha_before: str
    sha_after: str | None  # ``None`` when the LLM judged the situation a no-op
    verdict: str  # human-readable summary
    transcript: list[str] = field(default_factory=list)


@dataclass
class PushedChange:
    """One mergedog-authored change pushed back to the PR branch."""

    sha: str
    summary: str
    subject: str | None = None
    source: str | None = None


# GitHub PR comments cap out at 65,536 characters. We leave headroom for
# our own framing and a "[truncated]" tail.
_MAX_COMMENT_LEN = 60000
_DRCI_HEADING_RE = re.compile(r"<b>([^<]+)</b>")
_DRCI_JOB_LINK_RE = re.compile(r"\* \[([^\]]+)\]\(")
_DRCI_UNRELATED_HEADINGS = {
    "BROKEN TRUNK",
    "FLAKY",
    "FLAKY FAILURES",
    "UNRELATED FAILURE",
    "UNRELATED FAILURES",
}
_GIT_SHA_RE = re.compile(r"\A[0-9a-fA-F]{7,40}\Z")


@dataclass(frozen=True)
class SuppressionDrciComparison:
    """How mergedog's suppressed-check list compares with Dr. CI."""

    not_listed: tuple[str, ...] = ()
    not_marked_unrelated: tuple[str, ...] = ()

    @property
    def has_warning(self) -> bool:
        return bool(self.not_listed or self.not_marked_unrelated)

    def status_warning(self) -> str | None:
        if not self.has_warning:
            return None
        parts: list[str] = []
        if self.not_listed:
            parts.append(f"{len(self.not_listed)} not listed")
        if self.not_marked_unrelated:
            parts.append(
                f"{len(self.not_marked_unrelated)} not marked unrelated"
            )
        detail = f" ({', '.join(parts)} by Dr. CI)" if parts else ""
        return f"suppression list differs from Dr. CI{detail}"


def _format_handoff_comment(
    pr_data: dict,
    sessions: list[ClaudeSession],
    *,
    pushed_changes: list[PushedChange] | None = None,
    suppressed_failures: list[str] | None = None,
    drci_summary: str | None = None,
    recovering: bool = False,
) -> str:
    """Build the markdown body posted on the PR at handoff.

    ``recovering=True`` reframes the comment for the case where
    pytorchmergebot already replied "Merge failed" once and mergedog
    re-inspected CI; the human still needs to re-run ``@pytorchbot
    merge`` themselves -- mergedog never auto-retriggers a land.
    """
    head_sha = str(pr_data.get("headRefOid") or "")
    marker = (
        f"<!-- mergedog:handoff head={head_sha} -->\n"
        if head_sha
        else "<!-- mergedog:handoff -->\n"
    )
    n = len(sessions)
    changes = list(pushed_changes or [])
    suppressed = sorted(set(suppressed_failures or []))
    seen_change_shas = {c.sha for c in changes}
    for s in sessions:
        if s.sha_after and s.sha_after not in seen_change_shas:
            changes.append(
                PushedChange(
                    sha=s.sha_after,
                    summary=s.verdict,
                    source=f"{s.mode} ({s.started_at})",
                )
            )
            seen_change_shas.add(s.sha_after)
    if recovering and PROJECT.has_pytorch_merge_bot:
        head: list[str] = [
            "## mergedog handoff (after merge failure)",
            "",
            f"{PROJECT.mergebot_login} reported `Merge failed`. "
            "mergedog re-inspected "
            "CI and is handing back off to you. **Please review the latest "
            f"session(s) below and re-run `{PROJECT.merge_command}` if you're "
            "happy** — mergedog will not retrigger the merge itself.",
            "",
        ]
    else:
        head = [
            "## mergedog handoff",
            "",
            "All CI is green (or skipped). Ready for human review and "
            + (
                f"`{PROJECT.merge_command}`."
                if PROJECT.merge_command
                else "merge."
            ),
            "",
        ]
    if n == 0:
        if head_sha:
            head.extend(["", f"Current PR head: `{head_sha}`."])
        if changes:
            head.extend(["", "### Autonomous changes pushed", ""])
            head.extend(_format_pushed_change_lines(changes))
        else:
            head.append(
                "No mergedog-authored commits were pushed during this run."
            )
        head.extend(_format_ci_notes(suppressed, drci_summary))
        head.extend(
            [
                "",
                "No LLM was invoked during this run.",
            ]
        )
        return marker + "\n".join(head) + "\n"

    if head_sha:
        head.append(f"Current PR head: `{head_sha}`.")
    head.append("")
    if changes:
        head.append("### Autonomous changes pushed")
        head.append("")
        head.extend(_format_pushed_change_lines(changes))
        head.append("")
    else:
        head.append("### Autonomous changes pushed")
        head.append("")
        head.append("- None.")
        head.append("")
    head.extend(_format_ci_notes(suppressed, drci_summary))
    head.append(
        f"During shepherding, the configured LLM was invoked **{n}** time"
        f"{'' if n == 1 else 's'}. If any CI shows red, it judged that "
        "failure unrelated to this PR's changes — please verify below before "
        "merging."
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
        section.append("<details><summary>LLM transcript</summary>")
        section.append("")
        section.append("```")
        section.extend(sanitize_untrusted_text(line) for line in s.transcript)
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
    return marker + body


def _format_ci_notes(
    suppressed_failures: list[str], drci_summary: str | None
) -> list[str]:
    if not suppressed_failures and not drci_summary:
        return []
    lines = ["### CI notes at handoff", ""]
    comparison = compare_suppressed_failures_with_drci(
        suppressed_failures, drci_summary
    )
    if suppressed_failures:
        lines.extend(
            [
                "mergedog is treating these still-failing checks as "
                "unrelated/spurious. Please verify before merging:",
                "",
            ]
        )
        lines.extend(
            f"- `{sanitize_untrusted_text(name)}`" for name in suppressed_failures
        )
        lines.append("")
    if comparison.has_warning:
        lines.extend(
            [
                "**Warning:** mergedog's suppressed check list differs from "
                "the latest Dr. CI summary.",
                "",
            ]
        )
        if comparison.not_listed:
            lines.append("Suppressed by mergedog but not listed by Dr. CI:")
            lines.append("")
            lines.extend(
                f"- `{sanitize_untrusted_text(name)}`"
                for name in comparison.not_listed
            )
            lines.append("")
        if comparison.not_marked_unrelated:
            lines.append(
                "Suppressed by mergedog but not marked unrelated by Dr. CI:"
            )
            lines.append("")
            lines.extend(
                f"- `{sanitize_untrusted_text(name)}`"
                for name in comparison.not_marked_unrelated
            )
            lines.append("")
    if drci_summary:
        lines.extend(
            [
                "<details><summary>Latest Dr. CI summary</summary>",
                "",
                "```markdown",
                _truncate_drci_summary(drci_summary),
                "```",
                "",
                "</details>",
                "",
            ]
        )
    return lines


def compare_suppressed_failures_with_drci(
    suppressed_failures: list[str] | set[str] | tuple[str, ...],
    drci_summary: str | None,
) -> SuppressionDrciComparison:
    """Compare mergedog suppressions with Dr. CI's unrelated-failure view."""
    suppressed = tuple(sorted(set(suppressed_failures)))
    if not suppressed or not drci_summary:
        return SuppressionDrciComparison()
    sections = _drci_failure_sections(drci_summary)
    all_labels = tuple(label for labels in sections.values() for label in labels)
    unrelated_labels = tuple(
        label
        for heading, labels in sections.items()
        if heading in _DRCI_UNRELATED_HEADINGS
        for label in labels
    )
    not_listed: list[str] = []
    not_marked_unrelated: list[str] = []
    for name in suppressed:
        if not _any_drci_label_matches(name, all_labels):
            not_listed.append(name)
        elif not _any_drci_label_matches(name, unrelated_labels):
            not_marked_unrelated.append(name)
    return SuppressionDrciComparison(
        not_listed=tuple(not_listed),
        not_marked_unrelated=tuple(not_marked_unrelated),
    )


def suppression_drci_status_warning(
    suppressed_failures: list[str] | set[str] | tuple[str, ...],
    drci_summary: str | None,
) -> str | None:
    return compare_suppressed_failures_with_drci(
        suppressed_failures, drci_summary
    ).status_warning()


def _drci_failure_sections(summary: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current_heading: str | None = None
    for line in summary.splitlines():
        heading_match = _DRCI_HEADING_RE.search(line)
        if heading_match:
            current_heading = heading_match.group(1).strip().upper()
            sections.setdefault(current_heading, [])
        if current_heading is None:
            continue
        sections[current_heading].extend(
            label.strip() for label in _DRCI_JOB_LINK_RE.findall(line)
        )
    return sections


def _any_drci_label_matches(check_name: str, labels: tuple[str, ...]) -> bool:
    return any(check_name in label or label in check_name for label in labels)


def _truncate_drci_summary(summary: str, limit: int = 8000) -> str:
    summary = strip_html_comments(sanitize_untrusted_text(summary)).strip()
    if len(summary) <= limit:
        return summary
    return (
        summary[:limit]
        + "\n\n_[truncated; see the PR's Dr. CI comment for the full summary]_"
    )


def _format_pushed_change_lines(changes: list[PushedChange]) -> list[str]:
    lines: list[str] = []
    for change in changes:
        sha = change.sha[:12]
        sha_text = f"`{sha}`"
        if _GIT_SHA_RE.fullmatch(change.sha):
            sha_text = (
                f"[`{sha}`](https://github.com/{REPO_SLUG}/commit/"
                f"{change.sha})"
            )
        line = f"- {sha_text} — {sanitize_untrusted_text(change.summary)}"
        if change.subject:
            line += f": {sanitize_untrusted_text(change.subject)}"
        if change.source:
            line += f" ({sanitize_untrusted_text(change.source)})"
        lines.append(line)
    return lines


def post_handoff_comment(
    pr: int,
    pr_data: dict,
    sessions: list[ClaudeSession],
    *,
    pushed_changes: list[PushedChange] | None = None,
    suppressed_failures: list[str] | None = None,
    drci_summary: str | None = None,
    force: bool = False,
    recovering: bool = False,
) -> bool:
    # Recovery handoffs always post a fresh comment (forcing past the
    # "already handed off" check) -- the prior comment said "ready to
    # merge", which is now stale, and the human needs the new framing.
    head_sha = str(pr_data.get("headRefOid") or "") or None
    if (
        not force
        and not recovering
        and github.has_mergedog_handoff_comment(pr, head_sha=head_sha)
    ):
        log(f"handoff comment already present on PR #{pr}; not re-posting")
        return True
    body = _format_handoff_comment(
        pr_data,
        sessions,
        pushed_changes=pushed_changes,
        suppressed_failures=suppressed_failures,
        drci_summary=drci_summary,
        recovering=recovering,
    )
    try:
        github.post_pr_comment(pr, body)
        log(f"posted handoff summary to PR #{pr}")
        return True
    except Exception as e:
        # Don't halt on comment failure -- shepherding is otherwise complete.
        log(f"WARNING: failed to post handoff comment: {e}")
        return False


PYTORCHMERGEBOT_LOGIN = PROJECT.mergebot_login


_RETRYABLE_FAILURE_PATTERNS = [
    "HTTP Error 504",
    "Gateway Timeout",
]


def is_cla_merge_failure(body: str) -> bool:
    """True when pytorchmergebot failed because EasyCLA is not satisfied."""
    return "EasyCLA" in body


def is_retryable_merge_failure(body: str) -> bool:
    """True when a pytorchmergebot "Merge failed" is an infra flake."""
    return any(pat in body for pat in _RETRYABLE_FAILURE_PATTERNS)


def is_merge_conflict_failure(body: str) -> bool:
    """True when pytorchmergebot failed because of a merge conflict with main."""
    return "CONFLICT" in body and "Merge conflict" in body


def _pr_has_merge_conflicts(pr_data: dict) -> bool:
    """True when GitHub's merge box reports branch conflicts."""
    return (pr_data.get("mergeStateStatus") or "").upper() == "DIRTY"


def latest_mergebot_event(
    pr: int, since_iso: str
) -> tuple[str, str, str] | None:
    """Classify pytorchmergebot's reply to a `@pytorchbot merge` request.

    Looks at pytorchmergebot comments newer than ``since_iso`` and returns
    ``(kind, created_at, body)`` for the most recent relevant comment, where
    ``kind`` is one of:
      - ``"failed"``  -- comment carries "Merge failed"
      - ``"started"`` -- comment carries "Merge started" (merge in progress)
      - ``"other"``   -- some other pytorchmergebot reply (e.g. rebase)
    Returns ``None`` if there are no pytorchmergebot comments after
    ``since_iso``.
    """
    if PYTORCHMERGEBOT_LOGIN is None:
        return None
    relevant = [
        c
        for c in github.get_pr_comments(pr)
        if c.get("author") == PYTORCHMERGEBOT_LOGIN
        and (c.get("created_at") or "") > since_iso
    ]
    if not relevant:
        return None
    latest = max(relevant, key=lambda c: c.get("created_at") or "")
    body = latest.get("body") or ""
    iso = latest.get("created_at") or ""
    if "Merge failed" in body:
        return "failed", iso, body
    if "Merge started" in body:
        return "started", iso, body
    return "other", iso, body


def mergebot_ignored_check_names(
    comments: list[dict], checks: list[dict], *, since_iso: str
) -> set[str]:
    """Return current failed checks explicitly ignored by pytorchmergebot.

    A trusted merge-bot reply to ``@pytorchbot merge -i`` names the checks
    a human authorized it to ignore. Treat only those named, currently-red
    checks as already handled so mergedog can focus on any new failures from
    the merge attempt.
    """
    if PROJECT.mergebot_login is None:
        return set()

    from mergedog.taint import untaint

    failed_names = {
        c.get("name")
        for c in checks
        if c.get("bucket") in {"fail", "cancel"} and c.get("name")
    }
    if not failed_names:
        return set()

    ignored: set[str] = set()
    for comment in comments:
        if comment.get("author") != PROJECT.mergebot_login:
            continue
        if (comment.get("created_at") or "") <= since_iso:
            continue
        body = comment.get("body") or ""
        # pytorchmergebot is trusted repo automation. We use this body only
        # as data to match against GitHub's current check names.
        clean_body = untaint(body)
        if _MERGEBOT_IGNORE_PHRASE not in clean_body.lower():
            continue
        ignored.update(name for name in failed_names if name in clean_body)
    return ignored


_POLL_INTERVAL_SEC = 60


def _intervention_suffix(intervention_count: int | None) -> str:
    if intervention_count is None:
        return ""
    plural = "" if intervention_count == 1 else "s"
    return (
        f"; {intervention_count} mergedog intervention{plural} "
        "since last approval"
    )


def _suppressed_failure_suffix(suppressed_failure_count: int) -> str:
    if suppressed_failure_count <= 0:
        return ""
    plural = "" if suppressed_failure_count == 1 else "s"
    return f"; {suppressed_failure_count} suppressed failure{plural}"


def _handoff_warning_suffix(
    *,
    handoff_comment_ok: bool | None = None,
    suppression_warning: str | None = None,
) -> str:
    suffix = ""
    if suppression_warning:
        suffix += f"; {suppression_warning}"
    if handoff_comment_ok is False:
        suffix += "; handoff comment failed"
    return suffix


def _write_status_with_handoff_fields(pr: int, **fields: object) -> None:
    handoff_comment_ok = fields.pop("handoff_comment_ok", None)
    suppression_warning = fields.pop("suppression_warning", None)
    if handoff_comment_ok is not None:
        fields["handoff_comment_ok"] = handoff_comment_ok
    if suppression_warning:
        fields["suppression_warning"] = suppression_warning
    write_status(pr, **fields)  # type: ignore[arg-type]


def _apply_suppressed_overrides(
    checks: list[dict], suppressed_check_names: set[str] | None
) -> list[dict]:
    """Treat known-suppressed failed checks as skipped during handoff watch."""
    if not suppressed_check_names:
        return checks
    out: list[dict] = []
    for c in checks:
        if (
            c.get("name") in suppressed_check_names
            and c.get("bucket") in {"fail", "cancel"}
        ):
            c = {**c, "bucket": "skipping"}
        out.append(c)
    return out


def _post_handoff_ci_status(
    pr: int, *, suppressed_check_names: set[str] | None = None
) -> tuple[str, int, int, int, int] | None:
    try:
        checks = github.get_pr_checks_all(pr)
    except Exception:
        return None
    effective_checks = _apply_suppressed_overrides(
        checks, suppressed_check_names
    )
    total = len(checks)
    done = sum(1 for c in checks if c.get("bucket") not in {"pending", None})
    failed = sum(
        1 for c in effective_checks if c.get("bucket") in {"fail", "cancel"}
    )
    suppressed = sum(
        1
        for c in checks
        if c.get("name") in (suppressed_check_names or set())
        and c.get("bucket") in {"fail", "cancel"}
    )
    return github.evaluate_checks(effective_checks), done, total, failed, suppressed


def _write_post_handoff_ci_status(
    pr: int,
    *,
    status: str,
    done: int,
    total: int,
    failed: int,
    approved: bool,
    merging: bool,
    intervention_count: int | None,
    human_ack_sha: str | None,
    suppressed_failure_count: int = 0,
    handoff_comment_ok: bool | None = None,
    suppression_warning: str | None = None,
) -> None:
    if status == "failed":
        failure_plural = "" if failed == 1 else "s"
        message = (
            f"CI regressed after handoff: {failed} active "
            f"failure{failure_plural} ({done}/{total} checks done)"
        )
        category = "action"
        waiting_on = None
        action = "inspecting_ci"
    else:
        message = f"waiting for CI after handoff: {done}/{total} checks done"
        category = "waiting"
        waiting_on = "ci"
        action = None
    message += _suppressed_failure_suffix(suppressed_failure_count)
    message += _handoff_warning_suffix(
        handoff_comment_ok=handoff_comment_ok,
        suppression_warning=suppression_warning,
    )
    try:
        _write_status_with_handoff_fields(
            pr,
            phase="polling_ci",
            category=category,
            waiting_on=waiting_on,
            action=action,
            message=message + _intervention_suffix(intervention_count),
            intervention_count=intervention_count,
            human_ack_sha=human_ack_sha,
            approved=approved,
            merging=merging,
            ci_done=done,
            ci_total=total,
            ci_failed=failed,
            handoff_comment_ok=handoff_comment_ok,
            suppression_warning=suppression_warning,
        )
    except Exception:
        pass


def _write_handoff_status(
    pr: int,
    *,
    approved: bool,
    merging: bool,
    intervention_count: int | None = None,
    human_ack_sha: str | None = None,
    suppressed_failure_count: int = 0,
    handoff_comment_ok: bool | None = None,
    suppression_warning: str | None = None,
    approval_actionable: bool = True,
    cla_blocked: bool = False,
) -> None:
    if merging:
        category = "waiting"
        waiting_on = "mergebot"
        user_action = None
        message = "mergebot picked up the merge; waiting for outcome"
    elif cla_blocked:
        category = "waiting"
        waiting_on = "contributor"
        user_action = None
        message = "waiting for contributor CLA"
    elif approved:
        category = "ready"
        waiting_on = None
        user_action = (
            "review local mergedog log and merge when satisfied"
            if handoff_comment_ok is False
            else "merge when satisfied"
        )
        message = "ready for human merge"
    else:
        category = "ready" if approval_actionable else "waiting"
        waiting_on = "approval"
        user_action = (
            (
                "approve after reviewing local mergedog log"
                if handoff_comment_ok is False
                else "approve the PR after reviewing mergedog interventions"
            )
            if approval_actionable
            else None
        )
        message = "waiting for maintainer approval"
    try:
        _write_status_with_handoff_fields(
            pr,
            phase="watching_merge" if merging else "ready",
            category=category,
            waiting_on=waiting_on,
            user_action=user_action,
            message=(
                message
                + _suppressed_failure_suffix(suppressed_failure_count)
                + _handoff_warning_suffix(
                    handoff_comment_ok=handoff_comment_ok,
                    suppression_warning=suppression_warning,
                )
                + _intervention_suffix(intervention_count)
            ),
            intervention_count=intervention_count,
            human_ack_sha=human_ack_sha,
            approved=approved,
            merging=merging,
            handoff_comment_ok=handoff_comment_ok,
            suppression_warning=suppression_warning,
        )
    except Exception:
        pass


def _merging_progress_line(pr: int) -> str:
    """Build the body of a [MERGING]-phase log line: CI progress + failures.

    Replaces the verbose "handed off; awaiting @pytorchbot merge" message
    once pytorchmergebot has actually picked up the merge -- the [MERGING]
    log prefix already says what state we're in, so the body shows how
    much of pytorchmergebot's rebased CI is done and how many checks it's
    waving past as failed (== "ignoring").
    """
    try:
        checks = github.get_pr_checks_all(pr)
    except Exception:
        return "waiting for merge"
    total = len(checks)
    done = sum(1 for c in checks if c.get("bucket") not in {"pending", None})
    failed = sum(
        1 for c in checks if c.get("bucket") in {"fail", "cancel"}
    )
    body = f"waiting for merge; CI {done}/{total} done"
    if failed:
        body += f", {failed} failed"
    return body


def watch_post_handoff(
    pr: int,
    since_iso: str,
    *,
    intervention_count: int | None = None,
    human_ack_sha: str | None = None,
    suppressed_check_names: set[str] | None = None,
    handoff_comment_ok: bool | None = None,
    suppression_warning: str | None = None,
    approval_actionable: bool = True,
    cla_blocked: bool = False,
) -> tuple[str, str | None, str | None]:
    """Block after handoff, returning when there's something to react to.

    ``since_iso`` is the floor for "what counts as new pytorchmergebot
    activity": typically ``max(handoff_comment_iso, last_observed_failure_iso)``
    so a restart doesn't re-fire on a failure we already halted on.

    Returns ``(kind, event_iso, body)``:
      - ``("closed", None, None)``     -- PR is no longer ``OPEN`` (merged or
        closed by hand); caller should exit as completed.
      - ``("failed", event_iso, body)`` -- pytorchmergebot reported a merge
        failure; caller should persist ``event_iso`` so a future restart
        won't re-react to the same comment.
      - ``("conflict", None, None)``   -- GitHub reports the PR branch has
        conflicts with the base branch before mergebot produced a failure
        comment; caller should refresh the branch.
      - ``("ci_failed", None, None)``  -- CI regressed after handoff; caller
        should re-enter the normal CI inspection loop.
    """
    last_state: str | None = None
    last_merging_msg: str | None = None
    while True:
        pr_data = github.get_pr(pr, log_context="watching post-handoff")
        merging = github.has_label(pr_data, github.MERGING_LABEL)
        approved = (pr_data.get("reviewDecision") or "").upper() == "APPROVED"
        set_merging(merging)
        set_approved(approved)
        if pr_data.get("state") != "OPEN":
            return "closed", None, None
        if not merging and _pr_has_merge_conflicts(pr_data):
            log("GitHub reports branch conflicts after handoff; rebasing onto main")
            return "conflict", None, None
        event = latest_mergebot_event(pr, since_iso)
        kind = event[0] if event else None
        if kind == "failed":
            log(f"{PROJECT.mergebot_login} reported merge failure; halting")
            return "failed", event[1], event[2]
        if merging:
            _write_handoff_status(
                pr,
                approved=approved,
                merging=merging,
                intervention_count=intervention_count,
                human_ack_sha=human_ack_sha,
                handoff_comment_ok=handoff_comment_ok,
                suppression_warning=suppression_warning,
                approval_actionable=approval_actionable,
                cla_blocked=cla_blocked,
            )
            msg = _merging_progress_line(pr)
            if msg != last_merging_msg:
                log(msg)
                last_merging_msg = msg
            # Force a re-log of the awaiting message if the label disappears
            # later (shouldn't happen in normal flow, but cheap to handle).
            last_state = "merging"
        else:
            ci = _post_handoff_ci_status(
                pr, suppressed_check_names=suppressed_check_names
            )
            if ci is not None:
                ci_status, done, total, failed, suppressed = ci
                if ci_status in {"failed", "pending"}:
                    _write_post_handoff_ci_status(
                        pr,
                        status=ci_status,
                        done=done,
                        total=total,
                        failed=failed,
                        approved=approved,
                        merging=merging,
                        intervention_count=intervention_count,
                        human_ack_sha=human_ack_sha,
                        suppressed_failure_count=suppressed,
                        handoff_comment_ok=handoff_comment_ok,
                        suppression_warning=suppression_warning,
                    )
                    if ci_status == "failed":
                        log(
                            "CI regressed after handoff; returning to CI "
                            "inspection"
                        )
                        return "ci_failed", None, None
                    time.sleep(_POLL_INTERVAL_SEC)
                    continue
            _write_handoff_status(
                pr,
                approved=approved,
                merging=merging,
                intervention_count=intervention_count,
                human_ack_sha=human_ack_sha,
                suppressed_failure_count=suppressed if ci is not None else 0,
                handoff_comment_ok=handoff_comment_ok,
                suppression_warning=suppression_warning,
                approval_actionable=approval_actionable,
                cla_blocked=cla_blocked,
            )
            last_merging_msg = None
            if kind == "started":
                new_state = "started"
            elif approved:
                new_state = "awaiting_merge"
            else:
                new_state = "awaiting_approval"
            if new_state != last_state:
                if new_state == "started":
                    log(
                        f"{PROJECT.mergebot_login} picked up the merge; "
                        "waiting for outcome"
                    )
                elif new_state == "awaiting_merge":
                    if PROJECT.has_pytorch_merge_bot:
                        if cla_blocked:
                            log(
                                "handed off; waiting for contributor CLA "
                                f"before `{PROJECT.merge_command}`. "
                                "Will recover on a Merge failed reply, or "
                                "exit completed on close/merge."
                            )
                        else:
                            log(
                                f"handed off; awaiting `{PROJECT.merge_command}`. "
                                "Will recover on a Merge failed reply, or "
                                "exit completed on close/merge."
                            )
                    else:
                        log("handed off; awaiting human merge.")
                else:
                    if PROJECT.has_pytorch_merge_bot:
                        log(
                            "handed off; awaiting approval. Will recover "
                            "on a Merge failed reply, or exit completed "
                            "on close/merge."
                        )
                    else:
                        log("handed off; awaiting approval.")
                last_state = new_state
        time.sleep(_POLL_INTERVAL_SEC)


def watch_stack_post_handoff(
    prs_since_iso: dict[int, str],
) -> tuple[str, int, str | None, str | None]:
    """Stack analogue of :func:`watch_post_handoff`.

    Watches every stack member because pytorchmergebot comments land on the PR
    where the human typed ``@pytorchbot merge`` -- usually the stack top, but
    not guaranteed.

    Returns ``(kind, pr, event_iso, body)``:
      - ``("closed", pr, None, None)`` -- one member is no longer open.
      - ``("failed", pr, event_iso, body)`` -- mergebot failed that member.
    """
    last_state: str | None = None
    last_merging_msg: str | None = None
    prs = sorted(prs_since_iso)
    while True:
        any_merging = False
        any_approved = False
        started_prs: list[int] = []
        merging_prs: list[int] = []
        for pr in prs:
            pr_data = github.get_pr(pr, log_context="watching post-handoff stack")
            if pr_data.get("state") != "OPEN":
                set_merging(False)
                set_approved(False)
                return "closed", pr, None, None

            merging = github.has_label(pr_data, github.MERGING_LABEL)
            approved = (
                (pr_data.get("reviewDecision") or "").upper() == "APPROVED"
            )
            any_merging = any_merging or merging
            any_approved = any_approved or approved
            if merging:
                merging_prs.append(pr)
            if not merging and _pr_has_merge_conflicts(pr_data):
                set_merging(any_merging)
                set_approved(any_approved)
                log(f"GitHub reports branch conflicts after handoff on PR #{pr}")
                return "conflict", pr, None, None

            event = latest_mergebot_event(pr, prs_since_iso[pr])
            kind = event[0] if event else None
            if kind == "failed":
                set_merging(any_merging)
                set_approved(any_approved)
                log(f"{PROJECT.mergebot_login} reported merge failure on PR #{pr}")
                return "failed", pr, event[1], event[2]
            if kind == "started":
                started_prs.append(pr)

        set_merging(any_merging)
        set_approved(any_approved)
        if merging_prs:
            msg = "waiting for stack merge on " + ", ".join(
                f"PR #{pr}" for pr in merging_prs
            )
            if len(merging_prs) == 1:
                msg += f"; {_merging_progress_line(merging_prs[0])}"
            if msg != last_merging_msg:
                log(msg)
                last_merging_msg = msg
            last_state = "merging"
        else:
            last_merging_msg = None
            if started_prs:
                new_state = "started"
            elif any_approved:
                new_state = "awaiting_merge"
            else:
                new_state = "awaiting_approval"
            if new_state != last_state:
                if new_state == "started":
                    log(
                        f"{PROJECT.mergebot_login} picked up a stack merge; "
                        "waiting for outcome"
                    )
                elif new_state == "awaiting_merge":
                    if PROJECT.has_pytorch_merge_bot:
                        log(
                            f"handed off stack; awaiting `{PROJECT.merge_command}`. "
                            "Will recover on a Merge failed reply."
                        )
                    else:
                        log("handed off stack; awaiting human merge.")
                else:
                    if PROJECT.has_pytorch_merge_bot:
                        log(
                            "handed off stack; awaiting approval. Will recover "
                            "on a Merge failed reply."
                        )
                    else:
                        log("handed off stack; awaiting approval.")
                last_state = new_state
        time.sleep(_POLL_INTERVAL_SEC)
