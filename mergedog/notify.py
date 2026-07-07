"""Optional Google Chat notification via the ``meta`` CLI.

Sends a DM when a shepherd HALTs and needs human attention.  The ``meta``
tool is not assumed to exist -- if it's missing or the send fails, the
shepherd logs a warning and continues to exit normally.
"""
from __future__ import annotations

import shutil
import subprocess
import sys

from mergedog.paths import REPO_SLUG

_gchat_to: str | None = None
_pr: int | None = None


def configure(*, pr: int | None = None, gchat_to: str | None = None) -> None:
    global _gchat_to, _pr
    if gchat_to is not None:
        _gchat_to = gchat_to
    if pr is not None:
        _pr = pr


def notify_halt(msg: str) -> None:
    """Send a gchat DM about a HALT.  Never raises."""
    if not _gchat_to:
        return
    meta = shutil.which("meta")
    if meta is None:
        print(
            "[notify] 'meta' CLI not found; skipping gchat notification",
            file=sys.stderr,
            flush=True,
        )
        return
    pr_tag = f"PR #{_pr}" if _pr else "unknown PR"
    url = f"https://github.com/{REPO_SLUG}/pull/{_pr}" if _pr else ""
    text = f"[mergedog] {pr_tag} HALT: {msg}"
    if url:
        text += f"\n{url}"
    try:
        proc = subprocess.run(
            [
                meta,
                "google.chat.message",
                "send",
                f"--to={_gchat_to}",
                f"--text={text}",
            ],
            capture_output=True,
            timeout=30,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode(errors="replace").strip()
            print(
                f"[notify] gchat send failed: exit {proc.returncode}: {stderr}",
                file=sys.stderr,
                flush=True,
            )
    except Exception as exc:
        print(
            f"[notify] gchat send failed: {exc}",
            file=sys.stderr,
            flush=True,
        )
