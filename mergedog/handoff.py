"""Post-handoff PR comment + watcher loop.

Once CI is green and the trunk label is on, the shepherd posts a summary
of what claude did during the run, then sits in :func:`watch_post_handoff`
until either the PR closes/merges or pytorchmergebot reports a merge
failure (which kicks the shepherd into a recovery cycle).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from mergedog import github
from mergedog.log import log, set_approved, set_merging


@dataclass
class ClaudeSession:
    """One claude invocation, captured for the handoff comment."""

    mode: str  # "fix-CI" or "merge-resolver"
    started_at: str  # UTC ISO 8601
    sha_before: str
    sha_after: str | None  # ``None`` when claude judged the situation a no-op
    verdict: str  # human-readable summary
    transcript: list[str] = field(default_factory=list)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# GitHub PR comments cap out at 65,536 characters. We leave headroom for
# our own framing and a "[truncated]" tail.
_MAX_COMMENT_LEN = 60000


def _format_handoff_comment(
    pr_data: dict,
    sessions: list[ClaudeSession],
    *,
    recovering: bool = False,
) -> str:
    """Build the markdown body posted on the PR at handoff.

    ``recovering=True`` reframes the comment for the case where
    pytorchmergebot already replied "Merge failed" once and mergedog
    re-inspected CI; the human still needs to re-run ``@pytorchbot
    merge`` themselves -- mergedog never auto-retriggers a land.
    """
    marker = "<!-- mergedog:handoff -->\n"
    n = len(sessions)
    if recovering:
        head: list[str] = [
            "## mergedog handoff (after merge failure)",
            "",
            "pytorchmergebot reported `Merge failed`. mergedog re-inspected "
            "CI and is handing back off to you. **Please review the latest "
            "session(s) below and re-run `@pytorchbot merge` if you're "
            "happy** — mergedog will not retrigger the merge itself.",
            "",
        ]
    else:
        head = [
            "## mergedog handoff",
            "",
            "All CI is green (or skipped). Ready for human review and "
            "`@pytorchbot merge`.",
            "",
        ]
    if n == 0:
        head.append(
            "claude was not invoked during this run (CI was green from the "
            "start; no merge or fix needed)."
        )
        return marker + "\n".join(head) + "\n"

    head.append(
        f"During shepherding, claude was invoked **{n}** time"
        f"{'' if n == 1 else 's'}. If any CI shows red, claude judged that "
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
    return marker + body


def post_handoff_comment(
    pr: int,
    pr_data: dict,
    sessions: list[ClaudeSession],
    *,
    force: bool = False,
    recovering: bool = False,
) -> None:
    # Recovery handoffs always post a fresh comment (forcing past the
    # "already handed off" check) -- the prior comment said "ready to
    # merge", which is now stale, and the human needs the new framing.
    if not force and not recovering and github.has_mergedog_handoff_comment(pr):
        log(f"handoff comment already present on PR #{pr}; not re-posting")
        return
    body = _format_handoff_comment(pr_data, sessions, recovering=recovering)
    try:
        github.post_pr_comment(pr, body)
        log(f"posted handoff summary to PR #{pr}")
    except Exception as e:
        # Don't halt on comment failure -- shepherding is otherwise complete.
        log(f"WARNING: failed to post handoff comment: {e}")


PYTORCHMERGEBOT_LOGIN = "pytorchmergebot"


_RETRYABLE_FAILURE_PATTERNS = [
    "HTTP Error 504",
    "Gateway Timeout",
]


def is_retryable_merge_failure(body: str) -> bool:
    """True when a pytorchmergebot "Merge failed" is an infra flake."""
    return any(pat in body for pat in _RETRYABLE_FAILURE_PATTERNS)


def is_merge_conflict_failure(body: str) -> bool:
    """True when pytorchmergebot failed because of a merge conflict with main."""
    return "CONFLICT" in body and "Merge conflict" in body


def _latest_mergebot_event(
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


_POLL_INTERVAL_SEC = 60


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
    pr: int, since_iso: str
) -> tuple[str, str | None, str | None]:
    """Block after handoff, returning when there's something to react to.

    ``since_iso`` is the floor for "what counts as new pytorchmergebot
    activity": typically ``max(handoff_comment_iso, last_observed_failure_iso)``
    so a restart doesn't re-fire on a failure we already halted on.

    Returns ``(kind, event_iso, body)``:
      - ``("closed", None, None)``     -- PR is no longer ``OPEN`` (merged or
        closed by hand); caller should auto-prune.
      - ``("failed", event_iso, body)`` -- pytorchmergebot reported a merge
        failure; caller should persist ``event_iso`` so a future restart
        won't re-react to the same comment.
    """
    last_state: str | None = None
    last_merging_msg: str | None = None
    while True:
        pr_data = github.get_pr(pr)
        merging = github.has_label(pr_data, github.MERGING_LABEL)
        approved = (pr_data.get("reviewDecision") or "").upper() == "APPROVED"
        set_merging(merging)
        set_approved(approved)
        if pr_data.get("state") != "OPEN":
            return "closed", None, None
        event = _latest_mergebot_event(pr, since_iso)
        kind = event[0] if event else None
        if kind == "failed":
            log("pytorchmergebot reported merge failure; halting")
            return "failed", event[1], event[2]
        if merging:
            msg = _merging_progress_line(pr)
            if msg != last_merging_msg:
                log(msg)
                last_merging_msg = msg
            # Force a re-log of the awaiting message if the label disappears
            # later (shouldn't happen in normal flow, but cheap to handle).
            last_state = "merging"
        else:
            last_merging_msg = None
            if kind == "started":
                new_state = "started"
            elif approved:
                new_state = "awaiting_merge"
            else:
                new_state = "awaiting_approval"
            if new_state != last_state:
                if new_state == "started":
                    log("pytorchmergebot picked up the merge; waiting for outcome")
                elif new_state == "awaiting_merge":
                    log(
                        "handed off; awaiting `@pytorchbot merge`. Will recover "
                        "on a Merge failed reply, or auto-prune on close/merge."
                    )
                else:
                    log(
                        "handed off; awaiting approval. Will recover "
                        "on a Merge failed reply, or auto-prune on close/merge."
                    )
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
            pr_data = github.get_pr(pr)
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

            event = _latest_mergebot_event(pr, prs_since_iso[pr])
            kind = event[0] if event else None
            if kind == "failed":
                set_merging(any_merging)
                set_approved(any_approved)
                log(f"pytorchmergebot reported merge failure on PR #{pr}")
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
                        "pytorchmergebot picked up a stack merge; "
                        "waiting for outcome"
                    )
                elif new_state == "awaiting_merge":
                    log(
                        "handed off stack; awaiting `@pytorchbot merge`. "
                        "Will recover on a Merge failed reply."
                    )
                else:
                    log(
                        "handed off stack; awaiting approval. Will recover "
                        "on a Merge failed reply."
                    )
                last_state = new_state
        time.sleep(_POLL_INTERVAL_SEC)
