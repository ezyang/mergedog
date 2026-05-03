"""Thin subprocess wrappers."""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from mergedog.log import log


def _build_env(extra: Mapping[str, str] | None) -> dict[str, str] | None:
    if not extra:
        return None
    env = os.environ.copy()
    env.update(extra)
    return env


def _format_cmd(cmd: Sequence[str], cwd: str | Path | None) -> str:
    parts: list[str] = []
    if cwd is not None:
        parts.append(f"cd {shlex.quote(str(cwd))} &&")
    parts.extend(shlex.quote(c) for c in cmd)
    return " ".join(parts)


def run(
    cmd: Sequence[str],
    *,
    cwd: str | Path | None = None,
    check: bool = True,
    capture: bool = True,
    input_text: str | None = None,
    env_extra: Mapping[str, str] | None = None,
    loud: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a command. Defaults to capturing stdout/stderr and raising on non-zero exit.

    Pass ``loud=True`` to echo the command (with ``cd <dir> &&`` prefix when
    relevant) before executing. Used for the small set of mutating /
    interesting commands the operator probably wants to see.
    """
    if loud:
        log(f"$ {_format_cmd(cmd, cwd)}")
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        check=check,
        capture_output=capture,
        text=True,
        input=input_text,
        env=_build_env(env_extra),
    )


def run_streamed(
    cmd: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env_extra: Mapping[str, str] | None = None,
    loud: bool = True,
) -> int:
    """Run a command and stream its stdout/stderr to the parent's.

    Returns exit code. ``loud`` defaults to True here because by the time
    we're handing off the terminal to a subprocess, the operator should
    know what's about to take it over.
    """
    if loud:
        log(f"$ {_format_cmd(cmd, cwd)}")
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        env=_build_env(env_extra),
    )
    return proc.returncode
