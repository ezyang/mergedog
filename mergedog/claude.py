"""Invoke claude as a subprocess to investigate and (maybe) fix CI failures."""
from __future__ import annotations

import fcntl
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Iterator, Mapping

from mergedog.log import log
from mergedog.paths import LINTRUNNER_VENV, REPO_DIR
from mergedog.process import run
from mergedog import repo as repo_mod
from mergedog.repo import head_sha, head_subject

MERGEDOG_PREFIX = "[MERGEDOG]"

# Pin the model so an operator's local default (or "fast mode" on an older
# Opus) can't downgrade us. mergedog runs async; slow is fine, capable is
# not optional.
_CLAUDE_MODEL = "opus"


def _relativize(text: str, worktree: Path | None) -> str:
    """Replace the worktree absolute path with ``./`` so log lines aren't
    dominated by the same 50-char prefix on every entry."""
    if not worktree or not text:
        return text
    s = str(worktree)
    return text.replace(s + "/", "./").replace(s, ".")


def _yield_multiline(prefix: str, text: str, *, max_chars: int = 800) -> Iterator[str]:
    """Yield log lines for a possibly-multi-line value.

    The first line carries ``prefix``; continuation lines are indented to
    line up under the content (so ``claude → Bash: foo`` is followed by
    space-aligned continuations rather than being mashed onto one line
    with ⏎ separators).
    """
    if not text:
        return
    if len(text) > max_chars:
        text = text[:max_chars] + " …"
    lines = text.splitlines() or [""]
    yield prefix + lines[0]
    pad = " " * len(prefix)
    for line in lines[1:]:
        yield pad + line


def _summarize_tool_input(tool: str, inp: dict, worktree: Path | None) -> str:
    """Render a tool's ``input`` blob, with paths relativized to the worktree."""
    if tool == "Bash":
        return _relativize(inp.get("command") or "?", worktree)
    if tool in ("Read", "Edit", "Write", "NotebookEdit"):
        path = str(inp.get("file_path") or inp.get("notebook_path") or "?")
        return _relativize(path, worktree)
    if tool == "Grep":
        bits = [f"pattern={inp.get('pattern')!r}"]
        if inp.get("path"):
            bits.append(f"path={_relativize(str(inp['path']), worktree)}")
        return " ".join(bits)
    if tool == "Glob":
        return _relativize(str(inp.get("pattern") or "?"), worktree)
    return _relativize(json.dumps(inp, default=str), worktree)


def _summarize_event(ev: dict, worktree: Path | None = None) -> Iterator[str]:
    """Render a stream-json event into zero or more human-readable log lines.

    Tool-result events are intentionally dropped: the operator can read the
    actual result by looking at what claude does next (or by checking out
    the worktree). The intermediate dump was almost always either too short
    to be useful or too long to read.
    """
    t = ev.get("type")
    if t == "system" and ev.get("subtype") == "init":
        sid = (ev.get("session_id") or "")[:8]
        if sid:
            yield f"claude session {sid}"
        return
    if t == "assistant":
        msg = ev.get("message") or {}
        for block in msg.get("content") or []:
            kind = block.get("type")
            if kind == "text":
                txt = (block.get("text") or "").strip()
                for line in txt.splitlines():
                    if line.strip():
                        yield f"claude: {line}"
            elif kind == "tool_use":
                tool = block.get("name") or "?"
                summary = _summarize_tool_input(
                    tool, block.get("input") or {}, worktree
                )
                yield from _yield_multiline(f"claude → {tool}: ", summary)
            elif kind == "thinking":
                txt = (block.get("thinking") or "").strip()
                if txt:
                    yield from _yield_multiline("claude 💭 ", txt)
        return
    if t == "result":
        cost = ev.get("total_cost_usd")
        sub = ev.get("subtype") or "?"
        if isinstance(cost, (int, float)):
            yield f"claude finished: {sub} (cost ${cost:.4f})"
        else:
            yield f"claude finished: {sub}"


def _run_claude_streaming(
    prompt: str, cwd: Path, env_extra: Mapping[str, str]
) -> tuple[int, list[str]]:
    """Run claude in stream-json mode and pretty-print events as they arrive.

    Returns ``(returncode, transcript_lines)``: the same human-readable
    summaries we wrote to the log, kept in order, so the shepherd can
    later quote them in a handoff comment.
    """
    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        _CLAUDE_MODEL,
        "--permission-mode",
        "bypassPermissions",
        "--output-format",
        "stream-json",
        "--verbose",
    ]
    # Quote everything except the prompt itself, which is huge and would
    # bury the rest of the line.
    redacted = [c if c is not prompt else "<prompt>" for c in cmd]
    log("$ " + " ".join(shlex.quote(c) for c in redacted))
    env = os.environ.copy()
    env.update(env_extra)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    transcript: list[str] = []
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            log(f"claude (raw): {line[:300]}")
            continue
        for summary in _summarize_event(ev, worktree=cwd):
            log(summary)
            transcript.append(summary)
    return proc.wait(), transcript


def _is_clean(worktree: Path) -> bool:
    proc = run(["git", "status", "--porcelain"], cwd=worktree)
    return proc.stdout.strip() == ""


def _ensure_lintrunner_setup() -> Path | None:
    """Ensure mergedog's shared lintrunner venv + ``.lintbin`` exist.

    mergedog manages its own lintrunner via uv: a single venv at
    ``<ROOT>/lintrunner-venv`` and a single ``.lintbin`` populated once
    in ``<REPO_DIR>``. Both are reused across all worktrees -- each
    worktree symlinks ``.lintbin`` rather than re-initing.

    Held under a file lock so the mux's concurrent shepherds cooperate:
    only one runs ``uv venv`` / ``lintrunner init``; the rest see the
    artifacts and skip. Returns the lintrunner binary, or None if setup
    fails (no uv, install error) -- a missing lintrunner is a warning,
    not a halt; formatting drift would still be caught by CI.
    """
    binary = LINTRUNNER_VENV / "bin" / "lintrunner"
    lintbin = REPO_DIR / ".lintbin"
    if binary.exists() and lintbin.is_dir():
        return binary
    if shutil.which("uv") is None:
        log("WARNING: uv not found on PATH; skipping lintrunner")
        return None
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = REPO_DIR / ".mergedog-lintrunner-setup.lock"
    with open(lock_path, "w") as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        # Re-check under the lock; another shepherd may have set up while
        # we were blocked.
        if not binary.exists():
            log(f"creating shared lintrunner venv at {LINTRUNNER_VENV}")
            try:
                if not LINTRUNNER_VENV.exists():
                    run(["uv", "venv", str(LINTRUNNER_VENV)], loud=True)
                run(
                    [
                        "uv", "pip", "install",
                        "--python", str(LINTRUNNER_VENV / "bin" / "python"),
                        "lintrunner",
                    ],
                    loud=True,
                )
            except subprocess.CalledProcessError as e:
                log(f"WARNING: lintrunner venv setup failed: {e}; skipping")
                return None
            if not binary.exists():
                return None
        if not lintbin.is_dir():
            env = os.environ.copy()
            env["PATH"] = f"{binary.parent}{os.pathsep}{env.get('PATH', '')}"
            log("running lintrunner init (one-time; downloads pinned binaries)")
            proc = subprocess.run(
                [str(binary), "init"],
                cwd=str(REPO_DIR),
                env=env,
                capture_output=True,
                text=True,
            )
            for line in (proc.stdout or "").splitlines()[-20:]:
                log(f"  lintrunner init: {line}")
            if proc.returncode != 0:
                for line in (proc.stderr or "").splitlines()[-10:]:
                    log(f"  lintrunner init stderr: {line}")
                log(
                    f"WARNING: lintrunner init exited {proc.returncode}; "
                    ".lintbin may be incomplete"
                )
    return binary


def _run_lintrunner_amend(worktree: Path) -> str | None:
    """Run ``lintrunner -a`` in ``worktree``; amend any auto-fixes into HEAD.

    Run after every claude commit so an autoformat miss (clang-format,
    ruff format, ...) gets folded into the same ``[MERGEDOG]`` commit
    instead of triggering a second fix-CI cycle. Returns the new HEAD
    SHA if the amend changed anything, else None.

    A non-zero exit from lintrunner is fine: it just means there are
    still unfixable lints on top of whatever it auto-applied. We fold in
    the auto-fixes regardless, and CI will surface anything else.

    Scope: only the files claude changed in HEAD (``-r HEAD~1``). The
    rest of the PR was lint-clean before we started shepherding; running
    lintrunner across the whole diff against main would slow us down
    without finding anything new.
    """
    lintrunner = _ensure_lintrunner_setup()
    if lintrunner is None:
        return None
    # lintrunner shells out to its sibling tools (clang-format, ruff, ...);
    # they live next to lintrunner inside the venv, so put that bin dir at
    # the front of PATH.
    venv_bin = lintrunner.parent
    env = os.environ.copy()
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    # pytorch's ``.lintrunner.toml`` references native lint binaries by the
    # relative path ``.lintbin/<tool>`` (clang-format, clang-tidy,
    # actionlint, ...). ``_ensure_lintrunner_setup`` populated ``.lintbin``
    # in the main clone via ``lintrunner init``; symlink that dir into the
    # worktree so lintrunner finds them without us having to re-init per
    # worktree.
    main_lintbin = REPO_DIR / ".lintbin"
    worktree_lintbin = worktree / ".lintbin"
    if main_lintbin.is_dir() and not worktree_lintbin.exists():
        try:
            worktree_lintbin.symlink_to(main_lintbin)
        except OSError as e:
            log(f"WARNING: could not symlink .lintbin into worktree: {e}")
    # CLANGTIDY (and the EXECUTORCH variant) want a populated ``build/``
    # directory -- we don't build PyTorch, so skip them. CI runs clang-tidy
    # itself; we only care about auto-fixable lints here.
    skip = "CLANGTIDY,CLANGTIDY_EXECUTORCH_COMPATIBILITY"
    log(f"$ {lintrunner} -a -r HEAD~1 --skip {skip}  (post-claude autoformat)")
    proc = subprocess.run(
        [str(lintrunner), "-a", "-r", "HEAD~1", "--skip", skip],
        cwd=str(worktree),
        env=env,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "").rstrip()
    err = (proc.stderr or "").rstrip()
    # Tail-bias the log: lintrunner's full output is verbose per-file
    # status, but the summary at the end carries the actionable bits.
    for line in out.splitlines()[-20:]:
        log(f"  lintrunner: {line}")
    if proc.returncode != 0 and err:
        for line in err.splitlines()[-10:]:
            log(f"  lintrunner stderr: {line}")
    if _is_clean(worktree):
        log("lintrunner: no auto-fixes applied")
        return None
    name, email = repo_mod.get_mergedog_identity()
    run(["git", "add", "-A"], cwd=worktree, loud=True)
    run(
        ["git", "commit", "--amend", "--no-edit"],
        cwd=worktree,
        env_extra=repo_mod.author_env(name, email),
        loud=True,
    )
    new_sha = head_sha(worktree)
    log(f"lintrunner: amended auto-fixes into {new_sha[:12]}")
    return new_sha


def _commits_between(worktree: Path, before: str, after: str) -> int:
    """How many commits did claude add on top of ``before`` to reach ``after``?

    Uses ``--first-parent`` so that a merge commit counts as a single
    commit (the merge itself), rather than ``1 + everything brought in
    from the merged side``. Without this a merge resolution that pulls
    in months of main reads as thousands of commits.
    """
    proc = run(
        ["git", "rev-list", "--count", "--first-parent", f"{before}..{after}"],
        cwd=worktree,
    )
    return int(proc.stdout.strip() or "0")


def _invoke(
    worktree: Path,
    prompt: str,
    *,
    mode: str,
    expect_merge_commit: bool,
    expect_rebase_resolution: bool = False,
) -> tuple[bool, str | None, list[str]]:
    """Run claude in ``worktree`` and validate that its output meets the contract.

    Shared by ``invoke_fixer``, ``invoke_merge_resolver``, and
    ``invoke_rebase_resolver``. The modes differ in post-run checks:
    ``expect_merge_commit`` rejects leftover mid-merge state and requires
    two parents; ``expect_rebase_resolution`` rejects leftover mid-rebase
    state.

    Returns ``(ran_cleanly, new_sha, transcript)``:
    - ``ran_cleanly`` is False if claude exited non-zero, left a dirty
      working tree (or unfinished merge), made multiple commits, or
      violated the commit-message contract -- the harness should halt
      in any of those cases.
    - ``new_sha`` is the SHA of the new ``[MERGEDOG]`` commit if claude
      made one, else None. ``(True, None, ...)`` means "claude judged
      it a no-op" (spurious failures, or merge aborted).
    - ``transcript`` is the streamed event lines (same as the log) so
      the shepherd can include claude's reasoning in a handoff comment.
    """
    before = head_sha(worktree)
    name, email = repo_mod.get_mergedog_identity()
    log(f"invoking claude ({mode} mode)...")
    rc, transcript = _run_claude_streaming(
        prompt, cwd=worktree, env_extra=repo_mod.author_env(name, email)
    )
    if rc != 0:
        log(f"claude exited with code {rc}")
        return False, None, transcript

    if expect_merge_commit and repo_mod.is_merge_in_progress(worktree):
        log("claude exited but the merge is still in progress; refusing to push")
        return False, None, transcript

    if expect_rebase_resolution and repo_mod.is_rebase_in_progress(worktree):
        log("claude exited but the rebase is still in progress; refusing to push")
        return False, None, transcript

    inconclusive_path = worktree / ".mergedog-inconclusive"
    inconclusive = inconclusive_path.exists()
    if inconclusive:
        inconclusive_path.unlink()

    if not _is_clean(worktree):
        log("claude left an uncommitted working tree; refusing to push")
        return False, None, transcript

    after = head_sha(worktree)
    if after == before:
        if inconclusive:
            log("claude signalled INCONCLUSIVE; halting for human review")
            return False, None, transcript
        if expect_merge_commit:
            log("claude aborted the merge without committing")
        elif expect_rebase_resolution:
            log("claude aborted the rebase without committing")
        else:
            log("claude made no commit (treating as: failures are spurious / no-op)")
        return True, None, transcript

    n = _commits_between(worktree, before, after)
    if n != 1:
        expected = (
            "a merge resolution should be exactly one"
            if expect_merge_commit
            else "mergedog only allows one per pass"
        )
        log(f"claude produced {n} commits but {expected}")
        return False, None, transcript

    if expect_merge_commit and repo_mod.parent_count(worktree, after) != 2:
        log(
            f"claude's commit {after[:12]} is not a merge commit "
            f"(expected 2 parents)"
        )
        return False, None, transcript

    subject = head_subject(worktree)
    if not subject.startswith(MERGEDOG_PREFIX):
        log(
            f"claude's commit {after[:12]} subject does not start "
            f"with {MERGEDOG_PREFIX!r}: {subject!r}"
        )
        return False, None, transcript

    # Run lintrunner -a and fold any auto-fixes into claude's commit.
    # Catches autoformat misses (clang-format, ruff format, ...) before
    # they'd cause a wasted fix-CI cycle on a follow-up lint failure.
    # Skipped for merge resolver: HEAD~1 there is just one parent of the
    # merge, so lintrunner's "what changed" view would include everything
    # brought in from main -- too broad and too slow to be useful.
    if not expect_merge_commit:
        amended = _run_lintrunner_amend(worktree)
        if amended is not None:
            after = amended
            subject = head_subject(worktree)

    if expect_merge_commit:
        log(f"claude resolved the merge: {after[:12]}: {subject}")
    else:
        log(f"claude produced fix commit {after[:12]}: {subject}")
    return True, after, transcript


def invoke_fixer(worktree: Path, prompt: str) -> tuple[bool, str | None, list[str]]:
    """Run claude in the worktree to fix CI failures."""
    return _invoke(worktree, prompt, mode="fix-CI", expect_merge_commit=False)


def invoke_merge_resolver(
    worktree: Path, prompt: str
) -> tuple[bool, str | None, list[str]]:
    """Run claude in a mid-merge worktree to resolve conflicts."""
    return _invoke(worktree, prompt, mode="merge-resolver", expect_merge_commit=True)


def invoke_rebase_resolver(
    worktree: Path, prompt: str
) -> tuple[bool, str | None, list[str]]:
    """Run claude in a mid-rebase worktree to resolve conflicts."""
    return _invoke(
        worktree, prompt, mode="rebase-resolver",
        expect_merge_commit=False, expect_rebase_resolution=True,
    )
