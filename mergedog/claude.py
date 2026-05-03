"""Invoke claude as a subprocess to investigate and (maybe) fix CI failures."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterator, Mapping

from mergedog.log import log
from mergedog.process import run
from mergedog import repo as repo_mod
from mergedog.repo import head_sha, head_subject

MERGEDOG_PREFIX = "[MERGEDOG]"

# Pin the model so an operator's local default (or "fast mode" on an older
# Opus) can't downgrade us. mergedog runs async; slow is fine, capable is
# not optional.
_CLAUDE_MODEL = "opus"


def _summarize_tool_input(tool: str, inp: dict) -> str:
    """Compress a tool's ``input`` blob into one informative line."""
    if tool == "Bash":
        return (inp.get("command") or "?")[:300]
    if tool in ("Read", "Edit", "Write", "NotebookEdit"):
        return str(inp.get("file_path") or inp.get("notebook_path") or "?")
    if tool == "Grep":
        bits = [f"pattern={inp.get('pattern')!r}"]
        if inp.get("path"):
            bits.append(f"path={inp['path']}")
        return " ".join(bits)
    if tool == "Glob":
        return str(inp.get("pattern") or "?")
    return json.dumps(inp, default=str)[:300]


def _summarize_event(ev: dict) -> Iterator[str]:
    """Render a stream-json event into zero or more human-readable log lines."""
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
                yield f"claude → {tool}: {_summarize_tool_input(tool, block.get('input') or {})}"
            elif kind == "thinking":
                txt = (block.get("thinking") or "").strip()
                if txt:
                    yield f"claude 💭 {txt[:300]}"
        return
    if t == "user":
        msg = ev.get("message") or {}
        for block in msg.get("content") or []:
            if block.get("type") == "tool_result":
                content = block.get("content", "")
                if isinstance(content, list):
                    content = "".join(
                        c.get("text", "") for c in content if isinstance(c, dict)
                    )
                text = str(content).replace("\n", " ⏎ ")[:200]
                yield f"claude ← {text}"
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
) -> int:
    """Run claude in stream-json mode and pretty-print events as they arrive."""
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
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            log(f"claude (raw): {line[:300]}")
            continue
        for summary in _summarize_event(ev):
            log(summary)
    return proc.wait()


def _is_clean(worktree: Path) -> bool:
    proc = run(["git", "status", "--porcelain"], cwd=worktree)
    return proc.stdout.strip() == ""


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
    name, email = repo_mod.get_mergedog_identity()
    log("invoking claude (fix-CI mode)...")
    rc = _run_claude_streaming(
        prompt, cwd=worktree, env_extra=repo_mod.author_env(name, email)
    )
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
    name, email = repo_mod.get_mergedog_identity()
    log("invoking claude (merge-resolver mode)...")
    rc = _run_claude_streaming(
        prompt, cwd=worktree, env_extra=repo_mod.author_env(name, email)
    )
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
