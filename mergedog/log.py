"""Tiny logging helper. Prints to stderr with a timestamp prefix."""
from __future__ import annotations

import sys
import time

# Module-level signifier the shepherd toggles when the PR carries the
# ``merging`` label. Surfaced into every log line so the mux's
# last-line-of-the-log readout is enough to tell at a glance which PR
# pytorchmergebot is actively merging -- there's no other channel back
# to the parent process.
_merging = False


def set_merging(value: bool) -> None:
    global _merging
    _merging = value


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    prefix = "[MERGING] " if _merging else ""
    print(f"[{ts}] {prefix}{msg}", file=sys.stderr, flush=True)


def die(msg: str, code: int = 1) -> None:
    log(f"HALT: {msg}")
    sys.exit(code)
