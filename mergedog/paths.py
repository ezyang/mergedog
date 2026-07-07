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
LOGS_DIR = ROOT / "logs"
CI_LOGS_DIR = ROOT / "ci-logs"
LINTRUNNER_VENV = ROOT / "lintrunner-venv"
GH_API_CALLS_LOG = ROOT / "gh-api-calls.jsonl"
GH_API_GOVERNOR_LOCK = ROOT / "gh-api-governor.lock"
GH_API_GOVERNOR_STATE = ROOT / "gh-api-governor.json"
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


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tempfile + rename).

    Atomic so a SIGKILL mid-write can't leave a truncated or empty file.
    The temp name is pid-suffixed so concurrent writers can't clobber
    each other's in-flight temp file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def worktree_dir(pr: int) -> Path:
    return WORKTREES_DIR / str(pr)


def state_file(pr: int) -> Path:
    return STATE_DIR / f"{pr}.json"


def status_file(pr: int) -> Path:
    return STATUS_DIR / f"{pr}.json"


def context_file(pr: int) -> Path:
    return CONTEXTS_DIR / f"{pr}.md"


def log_file(pr: int) -> Path:
    return LOGS_DIR / f"{pr}.log"


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    WORKTREES_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    CONTEXTS_DIR.mkdir(parents=True, exist_ok=True)
    CI_LOGS_DIR.mkdir(parents=True, exist_ok=True)
