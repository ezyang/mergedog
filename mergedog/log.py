"""Tiny logging helper. Prints to stderr with a timestamp prefix."""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Module-level signifiers the shepherd toggles based on PR state.
# Surfaced into every log line so the mux's last-line-of-the-log readout
# is enough to tell at a glance which PR pytorchmergebot is actively
# merging, or which PRs are approved and waiting -- there's no other
# channel back to the parent process. ``[MERGING]`` takes precedence
# over ``[APPROVED]`` (a PR being merged is also approved).
_merging = False
_approved = False
_outcome: str | None = None
_log_file: Path | None = None


def configure_log_file(path: Path) -> None:
    """Also append future log lines to ``path``."""
    global _log_file
    path.parent.mkdir(parents=True, exist_ok=True)
    _log_file = path
    with open(path, "a", encoding="utf-8") as f:
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n=== mergedog start at {started} ===\n")


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
    if _log_file is not None:
        try:
            with open(_log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


def die(msg: str, code: int = 1) -> None:
    from mergedog import notify

    notify.notify_halt(msg)
    log(f"HALT: {msg}")
    sys.exit(code)


def complete(msg: str, *, code: int = 0, outcome: str = "DONE") -> None:
    set_outcome(outcome)
    log(msg)
    sys.exit(code)
