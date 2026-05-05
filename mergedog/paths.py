"""Filesystem layout used by mergedog.

The root directory defaults to ``~/.mergedog`` but can be overridden via
the ``MERGEDOG_ROOT`` environment variable. This is honoured at import
time, so any caller that wants to redirect mergedog at a different root
should set the env var (or use ``--root`` on the entry-point CLIs, which
sets it for you) before importing anything from this module.
"""
from __future__ import annotations

import os
from pathlib import Path


def _resolve_root() -> Path:
    env = os.environ.get("MERGEDOG_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".mergedog"


ROOT = _resolve_root()
REPO_DIR = ROOT / "repo"
WORKTREES_DIR = ROOT / "worktrees"
STATE_DIR = ROOT / "state"
CONTEXTS_DIR = ROOT / "contexts"
LINTRUNNER_VENV = ROOT / "lintrunner-venv"

REPO_SSH_URL = "git@github.com:pytorch/pytorch.git"
REPO_SLUG = "pytorch/pytorch"


def worktree_dir(pr: int) -> Path:
    return WORKTREES_DIR / str(pr)


def stack_worktree_dir(bottom_pr: int) -> Path:
    """Worktree path for a stack-mode shepherd, namespaced by bottom PR.

    The whole stack uses one worktree (we navigate its HEAD to whichever
    member's /orig we're operating on). Naming by the bottom PR keeps it
    distinct from any single-PR worktree that may have once existed for
    the same number, and makes ``ls ~/.mergedog/worktrees`` skim-readable.
    """
    return WORKTREES_DIR / f"stack-{bottom_pr}"


def state_file(pr: int) -> Path:
    return STATE_DIR / f"{pr}.json"


def context_file(pr: int) -> Path:
    return CONTEXTS_DIR / f"{pr}.md"


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    WORKTREES_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CONTEXTS_DIR.mkdir(parents=True, exist_ok=True)
