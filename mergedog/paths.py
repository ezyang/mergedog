"""Filesystem layout used by mergedog.

The root directory defaults to ``~/.mergedog`` but can be overridden via
the ``MERGEDOG_ROOT`` environment variable. The repo defaults to
``pytorch/pytorch`` but can be overridden via ``MERGEDOG_REPO`` or
``MERGEDOG_REPO_SLUG``. These are honoured at import time, so callers that
want to redirect mergedog should set env vars (or use entry-point CLI flags,
which set them for you) before importing anything from this module.
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
STATUS_DIR = ROOT / "status"
CONTEXTS_DIR = ROOT / "contexts"
CI_LOGS_DIR = ROOT / "ci-logs"
LINTRUNNER_VENV = ROOT / "lintrunner-venv"
PUSHED_COMMITS_LOG = ROOT / "pushed-commits.log"
CONFIG_FILE = ROOT / "config.json"
# Curated list of regular PRs the mux should resume. Distinct from STATE_DIR --
# the latter is per-PR shepherd state authored by the shepherd itself, and
# includes PRs the mux has since dropped. ``MUX_JOBS_FILE`` is the newer source
# of truth for ``--resume-known``; this file remains for older tools.
MUX_PRS_FILE = ROOT / "mux-prs.json"
MUX_JOBS_FILE = ROOT / "mux-jobs.json"
MUX_LOCK_FILE = ROOT / "mux.lock"
MUX_SOCKET = ROOT / "mux.sock"

REPO_SLUG = (
    os.environ.get("MERGEDOG_REPO_SLUG")
    or os.environ.get("MERGEDOG_REPO")
    or "pytorch/pytorch"
)
REPO_SSH_URL = os.environ.get("MERGEDOG_REPO_SSH_URL") or (
    f"git@github.com:{REPO_SLUG}.git"
)


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


def status_file(pr: int) -> Path:
    return STATUS_DIR / f"{pr}.json"


def context_file(pr: int) -> Path:
    return CONTEXTS_DIR / f"{pr}.md"


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    WORKTREES_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    CONTEXTS_DIR.mkdir(parents=True, exist_ok=True)
    CI_LOGS_DIR.mkdir(parents=True, exist_ok=True)
