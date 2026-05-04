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


def set_merging(value: bool) -> None:
    global _merging
    _merging = value


def set_approved(value: bool) -> None:
    global _approved
    _approved = value


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    if _merging:
        prefix = "[MERGING] "
    elif _approved:
        prefix = "[APPROVED] "
    else:
        prefix = ""
    print(f"[{ts}] {prefix}{msg}", file=sys.stderr, flush=True)


def die(msg: str, code: int = 1) -> None:
    log(f"HALT: {msg}")
    sys.exit(code)
