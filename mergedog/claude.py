"""Invoke claude as a subprocess to investigate and (maybe) fix CI failures."""
from __future__ import annotations

from pathlib import Path

from mergedog.log import log
from mergedog.process import run, run_streamed
from mergedog import repo as repo_mod
from mergedog.repo import head_sha, head_subject

MERGEDOG_PREFIX = "[MERGEDOG]"


def _is_clean(worktree: Path) -> bool:
    proc = run(["git", "status", "--porcelain"], cwd=worktree)
    return proc.stdout.strip() == ""


def _commits_between(worktree: Path, before: str, after: str) -> int:
    proc = run(
        ["git", "rev-list", "--count", f"{before}..{after}"],
        cwd=worktree,
    )
    return int(proc.stdout.strip() or "0")


def invoke_fixer(worktree: Path, prompt: str) -> tuple[bool, str | None]:
    """Run claude in the worktree to fix CI failures.

    Returns ``(ran_cleanly, new_sha)``:
    - ``ran_cleanly`` is False if claude exited non-zero, left a dirty
      working tree, made multiple commits, or violated the commit-message
      contract -- the harness should halt in any of those cases.
    - ``new_sha`` is the SHA of the new ``[MERGEDOG]`` commit if claude made
      one, else None. ``(True, None)`` means "claude judged it spurious;
      advance the PR".
    """
    before = head_sha(worktree)
    cmd = [
        "claude",
        "-p",
        prompt,
        "--permission-mode",
        "bypassPermissions",
    ]
    name, email = repo_mod.get_mergedog_identity()
    log("invoking claude...")
    rc = run_streamed(cmd, cwd=worktree, env_extra=repo_mod.author_env(name, email))
    if rc != 0:
        log(f"claude exited with code {rc}")
        return False, None

    if not _is_clean(worktree):
        log("claude left an uncommitted working tree; refusing to push")
        return False, None

    after = head_sha(worktree)
    if after == before:
        log("claude made no commit (treating as: failures are spurious / no-op)")
        return True, None

    n = _commits_between(worktree, before, after)
    if n != 1:
        log(f"claude produced {n} commits but mergedog only allows one per pass")
        return False, None

    subject = head_subject(worktree)
    if not subject.startswith(MERGEDOG_PREFIX):
        log(
            f"claude produced a commit ({after[:12]}) but the subject does "
            f"not start with {MERGEDOG_PREFIX!r}: {subject!r}"
        )
        return False, None

    log(f"claude produced fix commit {after[:12]}: {subject}")
    return True, after


def invoke_merge_resolver(worktree: Path, prompt: str) -> tuple[bool, str | None]:
    """Run claude in a mid-merge worktree to resolve conflicts.

    Returns ``(ran_cleanly, new_sha)``:
    - ``ran_cleanly`` is False on any contract violation (non-zero exit,
      dirty tree after, more than one new commit, wrong subject, not a
      merge commit) -- caller should halt.
    - ``new_sha=None`` with ``ran_cleanly=True`` means claude aborted the
      merge cleanly and chose to give up -- caller should also halt.
    """
    before = head_sha(worktree)
    cmd = [
        "claude",
        "-p",
        prompt,
        "--permission-mode",
        "bypassPermissions",
    ]
    name, email = repo_mod.get_mergedog_identity()
    log("invoking claude to resolve merge conflicts...")
    rc = run_streamed(cmd, cwd=worktree, env_extra=repo_mod.author_env(name, email))
    if rc != 0:
        log(f"claude exited with code {rc}")
        return False, None

    if repo_mod.is_merge_in_progress(worktree):
        log("claude exited but the merge is still in progress; refusing to push")
        return False, None

    if not _is_clean(worktree):
        log("claude left an uncommitted working tree; refusing to push")
        return False, None

    after = head_sha(worktree)
    if after == before:
        # Claude aborted the merge.
        log("claude aborted the merge without committing")
        return True, None

    n = _commits_between(worktree, before, after)
    if n != 1:
        log(f"claude produced {n} commits but a merge resolution should be exactly one")
        return False, None

    if repo_mod.parent_count(worktree, after) != 2:
        log(
            f"claude's commit {after[:12]} is not a merge commit "
            f"(expected 2 parents)"
        )
        return False, None

    subject = head_subject(worktree)
    if not subject.startswith(MERGEDOG_PREFIX):
        log(
            f"claude's merge commit {after[:12]} subject does not start "
            f"with {MERGEDOG_PREFIX!r}: {subject!r}"
        )
        return False, None

    log(f"claude resolved the merge: {after[:12]}: {subject}")
    return True, after
