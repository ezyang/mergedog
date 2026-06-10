"""Tiny logging helper. Prints to stderr with a timestamp prefix."""
from __future__ import annotations

import sys
import time

# Module-level signifiers the shepherd toggles based on PR state.
# Surfaced into every log line so the mux's last-line-of-the-log readout
# is enough to tell at a glance which PR pytorchmergebot is actively
# merging, or which PRs are approved and waiting -- there's no other
# channel back to the parent process. ``[MERGING]`` takes precedence
# over ``[APPROVED]`` (a PR being merged is also approved).
_merging = False
_approved = False
_outcome: str | None = None
_status_pr: int | None = None


def configure_status_pr(pr: int | None) -> None:
    """Also write terminal status updates for ``die``/``complete``."""
    global _status_pr
    _status_pr = pr


def set_merging(value: bool) -> None:
    global _merging
    _merging = value


def set_approved(value: bool) -> None:
    global _approved
    _approved = value


def set_outcome(value: str | None) -> None:
    global _outcome
    _outcome = value


def log(msg: str) -> None:
    from mergedog.sanitize import sanitize_untrusted_text

    msg = sanitize_untrusted_text(str(msg))
    ts = time.strftime("%H:%M:%S")
    if _outcome:
        prefix = f"[{_outcome}] "
    elif _merging:
        prefix = "[MERGING] "
    elif _approved:
        prefix = "[APPROVED] "
    else:
        prefix = ""
    line = f"[{ts}] {prefix}{msg}"
    print(line, file=sys.stderr, flush=True)


def die(msg: str, code: int = 1) -> None:
    from mergedog import notify

    notify.notify_halt(msg)
    if _status_pr is not None:
        try:
            from mergedog.status import write_status

            write_status(
                _status_pr,
                phase="halted",
                category="blocked",
                message=f"HALT: {msg}",
                user_action=msg,
            )
        except Exception:
            pass
    log(f"HALT: {msg}")
    sys.exit(code)


def complete(msg: str, *, code: int = 0, outcome: str = "DONE") -> None:
    set_outcome(outcome)
    if _status_pr is not None:
        try:
            from mergedog.status import write_status

            write_status(
                _status_pr,
                phase="complete",
                category="done",
                message=msg,
            )
        except Exception:
            pass
    log(msg)
    sys.exit(code)
