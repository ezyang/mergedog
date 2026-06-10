"""Manage the local pytorch clone and per-PR worktrees.

mergedog owns its own SSH clone of pytorch/pytorch under ``~/.mergedog/repo``.
Per-PR work happens in disposable worktrees under ``~/.mergedog/worktrees/<pr>``.
"""
from __future__ import annotations

import contextlib
import fcntl
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterator, Sequence

from mergedog.log import die, log
from mergedog.net import github_api_env_extra, is_transient_network_error
from mergedog.paths import (
    REPO_DIR,
    REPO_SSH_URL,
    ensure_dirs,
    worktree_dir,
)
from mergedog.process import run, run_streamed
from mergedog.project import get_project_policy


_GIT_LOCK_TIMEOUT_SEC = 2.0
_GIT_LOCK_MAX_BACKOFF_SEC = 0.5
_GHSTACK_MAX_RETRIES = 3
_GHSTACK_RETRY_DELAY_SEC = 5


def _log_completed_process_output(proc: subprocess.CompletedProcess[str]) -> None:
    for text in (proc.stdout, proc.stderr):
        text = (text or "").rstrip()
        if not text:
            continue
        for line in text.splitlines():
            log(line)


def _is_transient_ghstack_failure(proc: subprocess.CompletedProcess[str]) -> bool:
    if proc.returncode == 0:
        return False
    return is_transient_network_error(f"{proc.stdout or ''}\n{proc.stderr or ''}")


def _run_ghstack(args: Sequence[str], *, cwd: Path) -> None:
    """Run ghstack, retrying transient GitHub/proxy failures.

    ghstack uses the GitHub API even for ``cherry-pick --no-fetch`` to resolve
    the PR's head ref. Local proxy hiccups should not kill the shepherd.
    """
    for attempt in range(_GHSTACK_MAX_RETRIES):
        proc = run(
            args,
            cwd=cwd,
            check=False,
            capture=True,
            env_extra=github_api_env_extra(),
            loud=(attempt == 0),
        )
        _log_completed_process_output(proc)
        if proc.returncode == 0:
            if attempt > 0:
                log("  ghstack recovered after transient failure")
            return
        if not _is_transient_ghstack_failure(proc):
            proc.check_returncode()
        if attempt + 1 == _GHSTACK_MAX_RETRIES:
            proc.check_returncode()
        log(
            "  ! ghstack transient failure "
            f"(attempt {attempt + 1}/{_GHSTACK_MAX_RETRIES}), "
            f"retrying in {_GHSTACK_RETRY_DELAY_SEC}s"
        )
        time.sleep(_GHSTACK_RETRY_DELAY_SEC)


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
    with _clone_lock():
        git_dir = REPO_DIR / ".git"
        if not git_dir.exists():
            if REPO_DIR.exists() and any(REPO_DIR.iterdir()):
                die(
                    f"{REPO_DIR} exists and is not a git checkout; "
                    "refusing to clobber it"
                )
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


@contextlib.contextmanager
def _clone_lock() -> Iterator[None]:
    """Serialize initial creation of the shared base checkout."""
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    lock_path = REPO_DIR.parent / ".mergedog-clone.lock"
    with open(lock_path, "w") as fp:
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("waiting for shared clone lock")
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _fetch_lock(activity: str | None = None) -> Iterator[None]:
    """Serialize fetches across concurrent shepherds.

    The mux launches shepherds in parallel and they all hit ``fetch_origin``
    on the same ``~/.mergedog/repo`` near-simultaneously. ``git fetch`` takes
    per-ref locks (``refs/.../X.lock``) and a colliding fetch fails outright
    with exit 1 rather than waiting -- which we then surface as a halt.
    A coarse OS-level file lock here makes the second-and-Nth shepherds
    queue up; once the first fetch lands the rest are near-no-ops.
    """
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = REPO_DIR / ".mergedog-fetch.lock"
    with open(lock_path, "w") as fp:
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if activity:
                log(f"waiting for shared fetch lock: {activity}")
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


def fetch_origin() -> None:
    # ``capture=True`` (the default) so any future failure surfaces git's
    # stderr through ``process.run``'s error path instead of bare ``rc=1``.
    with _fetch_lock():
        run(["git", "fetch", "--prune", "origin"], cwd=REPO_DIR, loud=True)


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


def fetch_branch_from_url(url: str, branch: str, local_ref: str) -> str:
    """Fetch ``branch`` from ``url`` into ``local_ref`` and return its SHA."""
    with _fetch_lock():
        run(
            ["git", "fetch", url, f"+refs/heads/{branch}:{local_ref}"],
            cwd=REPO_DIR,
            capture=False,
            loud=True,
        )
    return run(["git", "rev-parse", local_ref], cwd=REPO_DIR).stdout.strip()


def commit_patch_id(sha: str) -> str | None:
    """Return git's stable patch-id for a single-parent commit, if available."""
    show = run(
        [
            "git",
            "show",
            "--format=",
            "--no-ext-diff",
            "--no-renames",
            "--binary",
            sha,
        ],
        cwd=REPO_DIR,
        check=False,
    )
    if show.returncode != 0 or not show.stdout.strip():
        return None
    proc = subprocess.run(
        ["git", "patch-id", "--stable"],
        cwd=REPO_DIR,
        input=show.stdout,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    first = proc.stdout.split()
    return first[0] if first else None


def patch_id_matches_any(sha: str, candidates: Sequence[str]) -> bool:
    """True if ``sha`` has the same patch-id as any candidate commit."""
    patch_id = commit_patch_id(sha)
    if patch_id is None:
        return False
    return any(commit_patch_id(candidate) == patch_id for candidate in candidates)


def _worktree_alive(wt: Path) -> bool:
    if not wt.exists():
        return False
    proc = run(["git", "rev-parse", "HEAD"], cwd=wt, check=False)
    return proc.returncode == 0


def wipe_worktree(pr: int) -> None:
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


def ensure_worktree(
    pr: int,
    sha: str,
    fork_remote: str | None = None,
    fork_branch: str | None = None,
) -> Path:
    """Get a worktree for ``pr`` at ``sha`` -- reusing the existing one when possible.

    Decision tree:
      - No worktree dir, or git can't see it as a worktree: create from scratch.
      - Worktree exists, points at ``sha``, working tree clean, no merge in
        progress: leave it alone (no I/O at all).
      - Worktree exists but is dirty / mid-merge / on a different SHA: abort
        any merge and ``git reset --hard <sha>`` rather than re-checking out
        the entire 22k-file tree.

    When ``fork_remote`` and ``fork_branch`` are given, the local branch is
    set to track them so a bare ``git push`` from the worktree Just Works.
    For ghstack PRs we leave the upstream unset -- ``ghstack submit`` works
    from commit metadata, not from a tracked branch.
    """
    wt = worktree_dir(pr)
    wt.parent.mkdir(parents=True, exist_ok=True)
    local_branch = f"mergedog/{pr}"

    if not _worktree_alive(wt):
        wipe_worktree(pr)
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
        rebase_in_progress = is_rebase_in_progress(wt)
        clean = run(["git", "status", "--porcelain"], cwd=wt).stdout.strip() == ""
        current_branch = run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=wt
        ).stdout.strip()
        if (
            head == sha
            and not merge_in_progress
            and not rebase_in_progress
            and clean
            and current_branch == local_branch
        ):
            log(f"reusing worktree at {wt} (already at {sha[:12]}, clean)")
        else:
            log(
                f"resetting worktree at {wt} -> {sha[:12]} "
                f"(was {head[:12]} on {current_branch!r}, "
                f"clean={clean}, merge_in_progress={merge_in_progress}, "
                f"rebase_in_progress={rebase_in_progress})"
            )
            if merge_in_progress:
                run(["git", "merge", "--abort"], cwd=wt, check=False, loud=True)
            if rebase_in_progress:
                run(["git", "rebase", "--abort"], cwd=wt, check=False, loud=True)
            if current_branch != local_branch:
                run(
                    ["git", "checkout", "-B", local_branch],
                    cwd=wt,
                    loud=True,
                )
            run(["git", "reset", "--hard", sha], cwd=wt, loud=True)

    if fork_remote and fork_branch:
        # Idempotent; safe even when reusing. Goes through the retry helper
        # because ``--set-upstream-to`` writes to ``.git/config`` and races
        # with concurrent shepherds spawned by the mux.
        _git_write_with_retry(
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


def set_worktree_to_sha(worktree: Path, sha: str) -> None:
    """Move ``worktree``'s HEAD to ``sha``, aborting any in-flight op.

    Used when a ghstack PR needs to reconstruct itself on top of a parent
    ``/orig`` commit. Stays on whichever local branch is current; the
    branch just gets fast-moved to ``sha``.
    """
    if is_merge_in_progress(worktree):
        run(["git", "merge", "--abort"], cwd=worktree, check=False, loud=True)
    if is_rebase_in_progress(worktree):
        run(["git", "rebase", "--abort"], cwd=worktree, check=False, loud=True)
    if is_cherry_pick_in_progress(worktree):
        run(["git", "cherry-pick", "--abort"], cwd=worktree, check=False, loud=True)
    run(["git", "reset", "--hard", sha], cwd=worktree, loud=True)


def is_cherry_pick_in_progress(worktree: Path) -> bool:
    """True if a cherry-pick is mid-flight in this worktree.

    git records this by writing ``.git/CHERRY_PICK_HEAD``; the equivalent
    of ``MERGE_HEAD`` for a cherry-pick.
    """
    proc = run(
        ["git", "rev-parse", "--verify", "CHERRY_PICK_HEAD"],
        cwd=worktree,
        check=False,
    )
    return proc.returncode == 0





def fetch_stack_refs(
    head_orig_pairs: list[tuple[str, str]],
) -> dict[str, str]:
    """Batch-fetch a set of (head_ref, orig_ref) pairs from origin.

    Returns a mapping from ref name (``gh/<user>/<id>/{head,orig}``) to
    the current SHA. One ``git fetch`` for everything keeps per-tick
    refresh cheap on a stack of size N -- there's a single network
    round-trip rather than 2N.
    """
    refspecs: list[str] = []
    refs: list[str] = []
    for head_ref, orig_ref in head_orig_pairs:
        for ref in (head_ref, orig_ref):
            refspecs.append(f"+refs/heads/{ref}:refs/remotes/origin/{ref}")
            refs.append(ref)
    activity = f"fetch {len(refs)} stack refs from origin"
    with _fetch_lock(activity):
        log(f"fetching {len(refs)} stack refs from origin")
        run(
            ["git", "fetch", "origin", *refspecs],
            cwd=REPO_DIR,
            capture=False,
        )
    log(f"fetched {len(refs)} stack refs from origin")
    out: dict[str, str] = {}
    for ref in refs:
        out[ref] = run(
            ["git", "rev-parse", f"refs/remotes/origin/{ref}"],
            cwd=REPO_DIR,
        ).stdout.strip()
    return out


def parent_sha(sha: str) -> str:
    """Return the first parent of ``sha`` from the local object database.
    """
    return run(
        ["git", "rev-parse", f"{sha}^"], cwd=REPO_DIR
    ).stdout.strip()


def tree_sha(sha: str) -> str:
    """Return the tree object for ``sha`` from the local object database."""
    return run(
        ["git", "rev-parse", f"{sha}^{{tree}}"], cwd=REPO_DIR
    ).stdout.strip()


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


def commit_message(worktree: Path, ref: str = "HEAD") -> str:
    """Return the full commit message (subject + body) of ``ref``."""
    return run(
        ["git", "log", "-1", "--pretty=%B", ref], cwd=worktree
    ).stdout.rstrip("\n")


VIABLE_STRICT_REF = "origin/viable/strict"


def _resolve_ref(ref: str) -> str | None:
    """Return the SHA for ``ref``, or None if it doesn't exist locally."""
    proc = run(["git", "rev-parse", "--verify", ref], cwd=REPO_DIR, check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    proc = run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO_DIR,
        check=False,
    )
    return proc.returncode == 0


def is_ancestor(ancestor: str, descendant: str) -> bool:
    return _is_ancestor(ancestor, descendant)


def _find_latest_revert(since: str, until: str) -> str | None:
    """Find the most recent revert commit on ``until`` after ``since``.

    Looks for commits whose subject starts with "Revert " in the
    ``since..until`` range, returns the SHA of the newest one.
    """
    proc = run(
        ["git", "log", "--oneline", "--format=%H %s", f"{since}..{until}"],
        cwd=REPO_DIR,
        check=False,
    )
    if proc.returncode != 0:
        return None
    for line in proc.stdout.strip().splitlines():
        if not line:
            continue
        sha, _, subject = line.partition(" ")
        if subject.startswith("Revert "):
            return sha
    return None


def select_rebase_target(worktree: Path) -> tuple[str, str]:
    """Pick the best ref to merge/rebase onto.

    Returns ``(ref, reason)`` where ``ref`` is a git ref or SHA and
    ``reason`` is a human-readable explanation for logging.

    Policy:
      - Never target raw trunk tip; only move to known-good points.
      - Prefer viable/strict as the default safe target.
      - If a revert commit exists on main ahead of viable/strict, prefer
        it (reverts restore trunk to a known-good state and are closer
        to tip, reducing land-race risk).
      - If neither viable/strict nor a revert is ahead of us, stay put
        by returning the current merge-base (which will noop the merge).
    """
    merge_base = run(
        ["git", "merge-base", "HEAD", "origin/main"], cwd=worktree
    ).stdout.strip()

    known_good_ref = get_project_policy().known_good_ref
    if known_good_ref is None:
        main = _resolve_ref("origin/main")
        if (
            main is not None
            and _is_ancestor(merge_base, main)
            and merge_base != main
        ):
            return "origin/main", "origin/main"
        return merge_base, "already at origin/main (staying put)"

    viable = _resolve_ref(known_good_ref)
    viable_ahead = viable is not None and _is_ancestor(merge_base, viable) and merge_base != viable

    revert = _find_latest_revert(merge_base, "origin/main")
    revert_ahead_of_viable = (
        revert is not None
        and viable is not None
        and _is_ancestor(viable, revert)
        and viable != revert
    )

    if revert is not None and revert_ahead_of_viable:
        return revert, f"revert commit {revert[:12]} (ahead of {known_good_ref})"
    if viable_ahead:
        return known_good_ref, known_good_ref.removeprefix("origin/")
    if revert is not None and _is_ancestor(merge_base, revert) and merge_base != revert:
        return revert, f"revert commit {revert[:12]}"
    return merge_base, "already at best known-good point (staying put)"


def rebase_target_advances(worktree: Path, target: str) -> bool:
    """True if rebasing/merging ``target`` would refresh HEAD's main base."""
    merge_base = run(
        ["git", "merge-base", "HEAD", "origin/main"], cwd=worktree
    ).stdout.strip()
    target_sha = run(
        ["git", "rev-parse", "--verify", target], cwd=worktree
    ).stdout.strip()
    return target_sha != merge_base


def trunk_revert_context(worktree: Path) -> str | None:
    """If trunk has recent reverts ahead of the PR's base, describe them.

    Used to inject "known trunk failures" context into Claude's prompt
    when we choose NOT to rebase (CI in flight). This is only a clue:
    a PR can expose a real failure in the same area as a trunk revert.
    """
    merge_base = run(
        ["git", "merge-base", "HEAD", "origin/main"], cwd=worktree
    ).stdout.strip()
    proc = run(
        ["git", "log", "--oneline", "--format=%H %s", f"{merge_base}..origin/main"],
        cwd=REPO_DIR,
        check=False,
    )
    if proc.returncode != 0:
        return None
    reverts = []
    for line in proc.stdout.strip().splitlines():
        if not line:
            continue
        sha, _, subject = line.partition(" ")
        if subject.startswith("Revert "):
            reverts.append(subject)
    if not reverts:
        return None
    header = (
        "The following commits were recently reverted on trunk (main). "
        "Use this only as diagnostic context: a matching failure may be "
        "unrelated trunk breakage, but it may also be a real incompatibility "
        "between this PR and current trunk. Do not treat a revert-area match "
        "as sufficient reason to choose spurious. If the failing test or "
        "build error is plausibly related to this PR's changes, choose "
        "INCONCLUSIVE instead of spurious unless the logs clearly prove the "
        "failure is unrelated."
    )
    body = "\n".join(f"- {s}" for s in reverts)
    return f"{header}\n\n{body}"


# Commit-subject prefix marking commits mergedog itself authored; the
# trust DB and audit trail key off it.
MERGEDOG_PREFIX = "[MERGEDOG]"
MERGE_COMMIT_SUBJECT = f"{MERGEDOG_PREFIX} Merge main into PR branch"
# Used by claude when it had to resolve conflicts; the suffix is what we
# tell claude to put on the commit. The harness only validates the
# ``[MERGEDOG]`` prefix, but a distinct subject makes ``git log`` and the
# handoff comment immediately show whether human-style judgment was
# applied during the merge.
MERGE_RESOLVED_SUBJECT = f"{MERGEDOG_PREFIX} Merge main into PR branch (resolved conflicts)"
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


def would_merge_conflict(worktree: Path, ref: str = "origin/main") -> bool:
    """Return whether merging ``ref`` into HEAD would conflict.

    This is a non-mutating probe used before publishing an LLM-authored fix:
    GitHub mergeability is against the base branch, so a fix that cannot
    merge with ``origin/main`` should be discarded in favor of rebasing the
    original PR.
    """
    proc = run(
        ["git", "merge-tree", "--write-tree", "HEAD", ref],
        cwd=worktree,
        check=False,
    )
    if proc.returncode == 0:
        return False
    if proc.returncode == 1:
        return True
    raise RuntimeError(
        f"git merge-tree HEAD {ref} failed unexpectedly:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


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


def head_orig_ref(head_ref: str) -> str:
    """Map a ghstack head ref to its companion orig ref.

    Mirrors what ``ghstack checkout`` does. ``gh/foo/123/head`` → ``gh/foo/123/orig``.
    """
    if not head_ref.endswith("/head"):
        die(f"head ref {head_ref!r} doesn't look like a ghstack ref")
    return head_ref[: -len("/head")] + "/orig"


def fetch_ghstack_orig(head_ref: str) -> str:
    """Fetch the orig branch for a ghstack PR; return its SHA.

    The orig branch (``gh/<user>/<n>/orig``) lives in the base repo (origin)
    next to the synthetic ``/head``. It's the "real" commit ghstack manages;
    we modify and re-submit it.
    """
    orig_ref = head_orig_ref(head_ref)
    return _fetch_origin_branch(orig_ref)


def fetch_ghstack_head(head_ref: str) -> str:
    """Fetch the synthetic /head branch for a ghstack PR; return its SHA.

    Used both at startup to verify origin's view of /head matches gh's, and
    after ``ghstack submit`` to learn the new /head SHA so we can trust it
    before the polling loop sees it on the GitHub side.
    """
    return _fetch_origin_branch(head_ref)


def _fetch_origin_branch(branch: str) -> str:
    """Fetch a single branch from origin and return its current SHA."""
    run(
        [
            "git",
            "fetch",
            "origin",
            f"+refs/heads/{branch}:refs/remotes/origin/{branch}",
        ],
        cwd=REPO_DIR,
        capture=False,
        loud=True,
    )
    return run(
        ["git", "rev-parse", f"refs/remotes/origin/{branch}"],
        cwd=REPO_DIR,
    ).stdout.strip()


def fixup_into_parent(worktree: Path) -> str:
    """Fold HEAD into HEAD~1 with fixup (not squash) semantics.

    The parent's commit message is kept verbatim; HEAD's tree is folded in;
    HEAD's *commit message* is discarded. Returns the SHA of the new amended
    commit. Used by the ghstack fix-CI flow: claude's ``[MERGEDOG]`` fix
    becomes part of the contributor's original commit, and the message that
    described the fix is propagated separately into ``ghstack submit -m``
    (so it shows up as the submit's audit message rather than rewriting the
    contributor's commit message on /orig).
    """
    name, email = get_mergedog_identity()
    run(["git", "reset", "--soft", "HEAD~1"], cwd=worktree, capture=False, loud=True)
    run(
        ["git", "commit", "--amend", "--no-edit"],
        cwd=worktree,
        env_extra=author_env(name, email),
        capture=False,
        loud=True,
    )
    return head_sha(worktree)


def ghstack_submit(
    worktree: Path,
    message: str,
    *,
    no_stack: bool = True,
    force: bool = False,
) -> None:
    """Run ``ghstack submit ... -m <message> HEAD`` in the worktree.

    Default ``no_stack=True`` re-uploads only the commit at HEAD's PR so
    siblings don't get hit with fresh CI for an unrelated parent fix.

    Pass ``no_stack=False`` to make ghstack walk
    every commit reachable from HEAD down to the merge-base with main
    and pushes any /head whose contents differ from origin.

    ``force=True`` adds ghstack's ``--force`` flag, bypassing its
    "cowardly refusing to push" anti-clobber check.

    The submit message is prefixed with ``[MERGEDOG] `` so the audit
    line on Phabricator is unambiguously from this harness; idempotent
    when the caller already prefixed it (e.g. claude's fix-CI message).
    """
    if not message.startswith(MERGEDOG_PREFIX):
        message = f"{MERGEDOG_PREFIX} {message}"
    args = ["ghstack", "submit", "-m", message, "HEAD"]
    if no_stack:
        args.insert(2, "--no-stack")
    if force:
        args.insert(2, "--force")
    _run_ghstack(args, cwd=worktree)


def ghstack_cherry_pick(worktree: Path, pr: int, *, no_fetch: bool = True) -> None:
    """Cherry-pick a single PR's ``/orig`` commit onto the worktree's HEAD.

    Used to reconstruct the stack one member at a time: start from the
    merge-base with main, then cherry-pick each member bottom-to-top.
    ``--no-fetch`` skips the per-PR remote fetch since we batch-fetch
    all refs up front.
    """
    args = ["ghstack", "cherry-pick", str(pr)]
    if no_fetch:
        args.append("--no-fetch")
    _run_ghstack(args, cwd=worktree)



_PR_TRAILER_RE = re.compile(
    r"Pull[- ]Request(?:[- ][Rr]esolved)?:\s*https://github\.com/[^/]+/[^/]+/pull/(\d+)"
)


def walk_orig_stack(orig_ref: str) -> list[int]:
    """Walk the commit ancestry of ``orig_ref`` and return PR numbers bottom-up.

    Each commit on a ghstack ``/orig`` branch carries a
    ``Pull-Request:`` (or older ``Pull-Request-Resolved:``) trailer. We
    walk from the tip back to the merge-base with ``origin/main`` and
    extract the PR number from each commit. The result is bottom-first
    (parents before children).
    """
    merge_base = run(
        ["git", "merge-base", f"refs/remotes/origin/{orig_ref}", "origin/main"],
        cwd=REPO_DIR,
    ).stdout.strip()
    log_output = run(
        [
            "git", "log", "--reverse", "--format=%B---END---",
            f"{merge_base}..refs/remotes/origin/{orig_ref}",
        ],
        cwd=REPO_DIR,
    ).stdout
    prs: list[int] = []
    for chunk in log_output.split("---END---"):
        m = _PR_TRAILER_RE.search(chunk)
        if m:
            prs.append(int(m.group(1)))
    return prs


REBASE_BODY_HINT = "Rebase onto origin/main to refresh stale base."


def attempt_rebase_main(worktree: Path, ref: str = "origin/main") -> tuple[str, str | None]:
    """Try to rebase HEAD onto ``ref``.

    Returns ``(status, sha)`` where status is one of:
      - ``"noop"``    : already on top of ``ref``; HEAD unchanged.
      - ``"ok"``      : rebase succeeded cleanly; ``sha`` is the new HEAD.
      - ``"conflict"``: rebase is in progress with conflicts. The worktree
                        is left in that state for claude to resolve.
    """
    name, email = get_mergedog_identity()
    before = head_sha(worktree)
    proc = run(
        ["git", "rebase", ref],
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
    if is_rebase_in_progress(worktree):
        return "conflict", None
    raise RuntimeError(
        f"git rebase {ref} failed for a non-conflict reason:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def is_rebase_in_progress(worktree: Path) -> bool:
    """True if a rebase is mid-flight in this worktree.

    git stores rebase state in either ``rebase-merge`` (interactive / merge
    backend) or ``rebase-apply`` (am backend) under ``.git`` — we have to
    ask git for the directory rather than guess at the layout.
    """
    git_dir = run(
        ["git", "rev-parse", "--git-dir"], cwd=worktree
    ).stdout.strip()
    git_dir_path = Path(git_dir)
    if not git_dir_path.is_absolute():
        git_dir_path = worktree / git_dir_path
    return (git_dir_path / "rebase-merge").exists() or (
        git_dir_path / "rebase-apply"
    ).exists()


def abort_rebase(worktree: Path) -> None:
    run(["git", "rebase", "--abort"], cwd=worktree, check=False, loud=True)
