"""Tiny logging helper. Prints to stderr with a timestamp prefix."""
from __future__ import annotations

import sys
import time


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def die(msg: str, code: int = 1) -> None:
    log(f"HALT: {msg}")
    sys.exit(code)
