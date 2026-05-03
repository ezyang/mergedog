"""Manage the local pytorch clone and per-PR worktrees.

mergedog owns its own SSH clone of pytorch/pytorch under ``~/.mergedog/repo``.
Per-PR work happens in disposable worktrees under ``~/.mergedog/worktrees/<pr>``.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Sequence

from mergedog.log import die, log
from mergedog.paths import REPO_DIR, REPO_SSH_URL, ensure_dirs, worktree_dir
from mergedog.process import run, run_streamed


_GIT_LOCK_TIMEOUT_SEC = 2.0
_GIT_LOCK_MAX_BACKOFF_SEC = 0.5


def _git_write_with_retry(
    args: Sequence[str],
    *,
    cwd: Path,
    loud: bool = False,
):
    """Run a git command that mutates ``.git/config`` or refs, retrying on lock contention.

    When the mux launches several shepherd subprocesses at once, they all
    hammer the same ``~/.mergedog/repo/.git/config`` and a few collide on
    git's coarse "could not lock config file" error (exit 255). The
    operations are idempotent, so we back off and retry.

    Hard time budget: ``_GIT_LOCK_TIMEOUT_SEC`` total wall-clock. Anything
    longer is not lock contention -- something is broken (stale lock file,
    permission issue) and we should surface the original error rather than
    sit silently for minutes.
    """
    deadline = time.monotonic() + _GIT_LOCK_TIMEOUT_SEC
    sleep = 0.05
    attempts = 0
    while True:
        attempts += 1
        proc = run(list(args), cwd=cwd, check=False, loud=loud)
        if proc.returncode == 0:
            if attempts > 1:
                log(f"  (recovered after {attempts} attempts)")
            return proc
        stderr = (proc.stderr or "").lower()
        is_lock_error = proc.returncode in (255, 128) and (
            "could not lock" in stderr
            or "unable to create" in stderr
            or "another git process seems to be running" in stderr
        )
        remaining = deadline - time.monotonic()
        if not is_lock_error or remaining <= 0:
            proc.check_returncode()
        log(f"  git lock contention; retrying in {sleep:.2f}s")
        time.sleep(min(sleep, remaining))
        sleep = min(sleep * 1.5, _GIT_LOCK_MAX_BACKOFF_SEC)


_AUTHOR_SUFFIX = " via mergedog"


def get_mergedog_identity() -> tuple[str, str]:
    """Return ``(name, email)`` to use for mergedog-authored commits.

    Uses the operator's configured ``user.name``/``user.email`` so that
    GitHub's CLA tool still recognizes the author. The name has
    " via mergedog" appended so it's clear in ``git log`` who actually
    produced the commit.
    """
    name = run(["git", "config", "user.name"], check=False).stdout.strip()
    email = run(["git", "config", "user.email"], check=False).stdout.strip()
    if not name or not email:
        die(
            "git user.name / user.email is not set; mergedog needs them to "
            "author commits in a CLA-friendly way"
        )
    if not name.endswith(_AUTHOR_SUFFIX):
        name = name + _AUTHOR_SUFFIX
    return name, email


def author_env(name: str, email: str) -> dict[str, str]:
    return {
        "GIT_AUTHOR_NAME": name,
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": name,
        "GIT_COMMITTER_EMAIL": email,
    }


def ensure_clone() -> None:
    """Clone pytorch/pytorch over SSH if we don't already have it."""
    ensure_dirs()
    git_dir = REPO_DIR / ".git"
    if not git_dir.exists():
        if REPO_DIR.exists() and any(REPO_DIR.iterdir()):
            die(f"{REPO_DIR} exists and is not a git checkout; refusing to clobber it")
        log(f"cloning {REPO_SSH_URL} -> {REPO_DIR} (this takes a while)")
        REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        rc = run_streamed(["git", "clone", REPO_SSH_URL, str(REPO_DIR)])
        if rc != 0:
            die(f"git clone failed (exit {rc})")
    # Make ``git push`` (no args) follow each branch's configured upstream.
    # That's how we make per-worktree manual pushes Just Work without the
    # operator having to remember a fork remote name. Read first to skip
    # the contended write in steady state.
    current = run(
        ["git", "config", "--get", "push.default"], cwd=REPO_DIR, check=False
    ).stdout.strip()
    if current != "upstream":
        _git_write_with_retry(
            ["git", "config", "push.default", "upstream"], cwd=REPO_DIR
        )


def fetch_origin() -> None:
    run(["git", "fetch", "--prune", "origin"], cwd=REPO_DIR, capture=False, loud=True)


def add_fork_remote(name: str, ssh_url: str) -> None:
    """Add (or update) a remote that points at the contributor's fork over SSH."""
    proc = run(["git", "remote", "get-url", name], cwd=REPO_DIR, check=False)
    if proc.returncode == 0:
        existing = proc.stdout.strip()
        if existing != ssh_url:
            _git_write_with_retry(
                ["git", "remote", "set-url", name, ssh_url],
                cwd=REPO_DIR,
                loud=True,
            )
    else:
        _git_write_with_retry(
            ["git", "remote", "add", name, ssh_url], cwd=REPO_DIR, loud=True
        )


def fetch_pr_branch(remote: str, branch: str) -> str:
    """Fetch the contributor's branch and return the fetched SHA."""
    run(
        ["git", "fetch", remote, f"+refs/heads/{branch}:refs/remotes/{remote}/{branch}"],
        cwd=REPO_DIR,
        capture=False,
        loud=True,
    )
    sha = run(
        ["git", "rev-parse", f"refs/remotes/{remote}/{branch}"],
        cwd=REPO_DIR,
    ).stdout.strip()
    return sha


def _worktree_alive(wt: Path) -> bool:
    if not wt.exists():
        return False
    proc = run(["git", "rev-parse", "HEAD"], cwd=wt, check=False)
    return proc.returncode == 0


def _wipe_worktree(pr: int) -> None:
    wt = worktree_dir(pr)
    if wt.exists():
        run(
            ["git", "worktree", "remove", "--force", str(wt)],
            cwd=REPO_DIR,
            check=False,
            loud=True,
        )
        if wt.exists():
            shutil.rmtree(wt, ignore_errors=True)
    run(["git", "worktree", "prune"], cwd=REPO_DIR, check=False)


def ensure_worktree(pr: int, sha: str, fork_remote: str, fork_branch: str) -> Path:
    """Get a worktree for ``pr`` at ``sha`` -- reusing the existing one when possible.

    Decision tree:
      - No worktree dir, or git can't see it as a worktree: create from scratch.
      - Worktree exists, points at ``sha``, working tree clean, no merge in
        progress: leave it alone (no I/O at all).
      - Worktree exists but is dirty / mid-merge / on a different SHA: abort
        any merge and ``git reset --hard <sha>`` rather than re-checking out
        the entire 22k-file tree.
    """
    wt = worktree_dir(pr)
    wt.parent.mkdir(parents=True, exist_ok=True)
    local_branch = f"mergedog/{pr}"

    if not _worktree_alive(wt):
        _wipe_worktree(pr)
        run(["git", "branch", "-D", local_branch], cwd=REPO_DIR, check=False)
        run(
            ["git", "worktree", "add", "-B", local_branch, str(wt), sha],
            cwd=REPO_DIR,
            capture=False,
            loud=True,
        )
    else:
        head = run(["git", "rev-parse", "HEAD"], cwd=wt).stdout.strip()
        merge_in_progress = is_merge_in_progress(wt)
        clean = run(["git", "status", "--porcelain"], cwd=wt).stdout.strip() == ""
        current_branch = run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=wt
        ).stdout.strip()
        if (
            head == sha
            and not merge_in_progress
            and clean
            and current_branch == local_branch
        ):
            log(f"reusing worktree at {wt} (already at {sha[:12]}, clean)")
        else:
            log(
                f"resetting worktree at {wt} -> {sha[:12]} "
                f"(was {head[:12]} on {current_branch!r}, "
                f"clean={clean}, merge_in_progress={merge_in_progress})"
            )
            if merge_in_progress:
                run(["git", "merge", "--abort"], cwd=wt, check=False, loud=True)
            if current_branch != local_branch:
                run(
                    ["git", "checkout", "-B", local_branch],
                    cwd=wt,
                    loud=True,
                )
            run(["git", "reset", "--hard", sha], cwd=wt, loud=True)

    # Idempotent; safe even when reusing.
    run(
        [
            "git",
            "branch",
            f"--set-upstream-to={fork_remote}/{fork_branch}",
            local_branch,
        ],
        cwd=wt,
        loud=True,
    )
    return wt


def push_to_fork(worktree: Path, remote: str | None = None, branch: str | None = None) -> None:
    """Push the worktree's HEAD.

    When ``remote`` and ``branch`` are both provided, push explicitly to
    ``<remote>/<branch>``. Otherwise rely on the branch's configured
    upstream (the default for our worktrees, so ``git push`` Just Works
    when run by hand).
    """
    if remote and branch:
        run(
            ["git", "push", remote, f"HEAD:refs/heads/{branch}"],
            cwd=worktree,
            capture=False,
            loud=True,
        )
    else:
        run(["git", "push"], cwd=worktree, capture=False, loud=True)


def head_sha(worktree: Path) -> str:
    return run(["git", "rev-parse", "HEAD"], cwd=worktree).stdout.strip()


def head_subject(worktree: Path) -> str:
    return run(["git", "log", "-1", "--pretty=%s"], cwd=worktree).stdout.strip()


def merge_base_age_seconds(worktree: Path, ref: str = "origin/main") -> int:
    """Return the age in seconds of the merge-base between HEAD and ``ref``.

    The merge-base is the commit on ``ref`` from which the PR branch
    diverged. Its age is a good proxy for "how stale is this PR's view of
    main": even if the contributor pushed yesterday, if they branched off
    main two months ago, the merge-base is two months old.
    """
    base = run(["git", "merge-base", "HEAD", ref], cwd=worktree).stdout.strip()
    ts = run(["git", "show", "-s", "--format=%ct", base], cwd=worktree).stdout.strip()
    import time as _time

    return int(_time.time()) - int(ts)


MERGE_COMMIT_SUBJECT = "[MERGEDOG] Merge main into PR branch"
CLEAN_MERGE_BODY = "Clean merge with origin/main; no conflicts."


def attempt_merge_main(worktree: Path, ref: str = "origin/main") -> tuple[str, str | None]:
    """Try to merge ``ref`` into HEAD.

    Returns ``(status, sha)`` where status is one of:
      - ``"noop"``    : already up to date, no commit made.
      - ``"ok"``      : merge succeeded cleanly; ``sha`` is the new HEAD.
      - ``"conflict"``: merge is in progress with conflicts. The worktree is
                        left in the conflicted state for someone else (claude)
                        to resolve.
    """
    name, email = get_mergedog_identity()
    before = head_sha(worktree)
    proc = run(
        [
            "git",
            "merge",
            "--no-ff",
            "--no-edit",
            "-m",
            MERGE_COMMIT_SUBJECT,
            "-m",
            CLEAN_MERGE_BODY,
            ref,
        ],
        cwd=worktree,
        check=False,
        env_extra=author_env(name, email),
        loud=True,
    )
    if proc.returncode == 0:
        after = head_sha(worktree)
        if after == before:
            return "noop", None
        return "ok", after
    # Distinguish conflict from other failures by checking for MERGE_HEAD.
    merge_head = worktree / ".git" / "MERGE_HEAD"
    if not merge_head.exists():
        # Some worktree layouts put .git as a file; ask git directly.
        proc2 = run(
            ["git", "rev-parse", "--verify", "MERGE_HEAD"],
            cwd=worktree,
            check=False,
        )
        if proc2.returncode != 0:
            run(["git", "merge", "--abort"], cwd=worktree, check=False)
            raise RuntimeError(
                f"git merge {ref} failed for a non-conflict reason:\n"
                f"{proc.stdout}\n{proc.stderr}"
            )
    return "conflict", None


def abort_merge(worktree: Path) -> None:
    run(["git", "merge", "--abort"], cwd=worktree, check=False, loud=True)


def is_merge_in_progress(worktree: Path) -> bool:
    proc = run(
        ["git", "rev-parse", "--verify", "MERGE_HEAD"],
        cwd=worktree,
        check=False,
    )
    return proc.returncode == 0


def parent_count(worktree: Path, sha: str) -> int:
    out = run(
        ["git", "rev-list", "--parents", "-n", "1", sha],
        cwd=worktree,
    ).stdout.strip()
    # "<sha> <parent1> [<parent2> ...]"
    return max(0, len(out.split()) - 1)
