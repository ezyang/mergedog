"""Filesystem layout used by mergedog."""
from __future__ import annotations

from pathlib import Path

ROOT = Path.home() / ".mergedog"
REPO_DIR = ROOT / "repo"
WORKTREES_DIR = ROOT / "worktrees"
STATE_DIR = ROOT / "state"
CONTEXTS_DIR = ROOT / "contexts"

REPO_SSH_URL = "git@github.com:pytorch/pytorch.git"
REPO_SLUG = "pytorch/pytorch"


def worktree_dir(pr: int) -> Path:
    return WORKTREES_DIR / str(pr)


def state_file(pr: int) -> Path:
    return STATE_DIR / f"{pr}.json"


def context_file(pr: int) -> Path:
    return CONTEXTS_DIR / f"{pr}.md"


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    WORKTREES_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CONTEXTS_DIR.mkdir(parents=True, exist_ok=True)
