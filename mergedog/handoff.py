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
from mergedog.log import log, set_merging


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


def _format_handoff_comment(pr_data: dict, sessions: list[ClaudeSession]) -> str:
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


def post_handoff_comment(
    pr: int,
    pr_data: dict,
    sessions: list[ClaudeSession],
    *,
    force: bool = False,
) -> None:
    if not force and github.has_mergedog_handoff_comment(pr):
        log(f"handoff comment already present on PR #{pr}; not re-posting")
        return
    body = _format_handoff_comment(pr_data, sessions)
    try:
        github.post_pr_comment(pr, body)
        log(f"posted handoff summary to PR #{pr}")
    except Exception as e:
        # Don't halt on comment failure -- shepherding is otherwise complete.
        log(f"WARNING: failed to post handoff comment: {e}")


PYTORCHMERGEBOT_LOGIN = "pytorchmergebot"


def _latest_mergebot_event(
    pr: int, since_iso: str
) -> tuple[str, str] | None:
    """Classify pytorchmergebot's reply to a `@pytorchbot merge` request.

    Looks at pytorchmergebot comments newer than ``since_iso`` and returns
    ``(kind, created_at)`` for the most recent relevant comment, where
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
        return "failed", iso
    if "Merge started" in body:
        return "started", iso
    return "other", iso


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


def watch_post_handoff(pr: int, since_iso: str) -> tuple[str, str | None]:
    """Block after handoff, returning when there's something to react to.

    ``since_iso`` is the floor for "what counts as new pytorchmergebot
    activity": typically ``max(handoff_comment_iso, last_observed_failure_iso)``
    so a restart doesn't re-fire on a failure we already halted on.

    Returns ``(kind, event_iso)``:
      - ``("closed", None)``     -- PR is no longer ``OPEN`` (merged or
        closed by hand); caller should auto-prune.
      - ``("failed", event_iso)`` -- pytorchmergebot reported a merge
        failure; caller should persist ``event_iso`` so a future restart
        won't re-react to the same comment.
    """
    last_state: str | None = None
    last_merging_msg: str | None = None
    while True:
        pr_data = github.get_pr(pr)
        merging = github.has_label(pr_data, github.MERGING_LABEL)
        set_merging(merging)
        if pr_data.get("state") != "OPEN":
            return "closed", None
        event = _latest_mergebot_event(pr, since_iso)
        kind = event[0] if event else None
        if kind == "failed":
            log("pytorchmergebot reported merge failure; halting")
            return "failed", event[1]
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
            new_state = kind or "watching"
            if new_state != last_state:
                if new_state == "started":
                    log("pytorchmergebot picked up the merge; waiting for outcome")
                else:
                    log(
                        "handed off; awaiting `@pytorchbot merge`. Will recover "
                        "on a Merge failed reply, or auto-prune on close/merge."
                    )
                last_state = new_state
        time.sleep(_POLL_INTERVAL_SEC)
