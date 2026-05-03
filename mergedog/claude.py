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


def invoke_fixer(worktree: Path, prompt: str) -> tuple[bool, str | None, list[str]]:
    """Run claude in the worktree to fix CI failures.

    Returns ``(ran_cleanly, new_sha, transcript)``:
    - ``ran_cleanly`` is False if claude exited non-zero, left a dirty
      working tree, made multiple commits, or violated the commit-message
      contract -- the harness should halt in any of those cases.
    - ``new_sha`` is the SHA of the new ``[MERGEDOG]`` commit if claude made
      one, else None. ``(True, None, ...)`` means "claude judged it
      spurious; advance the PR".
    - ``transcript`` is the streamed event lines (same as the log) so the
      shepherd can include claude's reasoning in a handoff comment.
    """
    before = head_sha(worktree)
    name, email = repo_mod.get_mergedog_identity()
    log("invoking claude (fix-CI mode)...")
    rc, transcript = _run_claude_streaming(
        prompt, cwd=worktree, env_extra=repo_mod.author_env(name, email)
    )
    if rc != 0:
        log(f"claude exited with code {rc}")
        return False, None, transcript

    if not _is_clean(worktree):
        log("claude left an uncommitted working tree; refusing to push")
        return False, None, transcript

    after = head_sha(worktree)
    if after == before:
        log("claude made no commit (treating as: failures are spurious / no-op)")
        return True, None, transcript

    n = _commits_between(worktree, before, after)
    if n != 1:
        log(f"claude produced {n} commits but mergedog only allows one per pass")
        return False, None, transcript

    subject = head_subject(worktree)
    if not subject.startswith(MERGEDOG_PREFIX):
        log(
            f"claude produced a commit ({after[:12]}) but the subject does "
            f"not start with {MERGEDOG_PREFIX!r}: {subject!r}"
        )
        return False, None, transcript

    log(f"claude produced fix commit {after[:12]}: {subject}")
    return True, after, transcript


def invoke_merge_resolver(
    worktree: Path, prompt: str
) -> tuple[bool, str | None, list[str]]:
    """Run claude in a mid-merge worktree to resolve conflicts.

    Returns ``(ran_cleanly, new_sha, transcript)``: see ``invoke_fixer``.
    """
    before = head_sha(worktree)
    name, email = repo_mod.get_mergedog_identity()
    log("invoking claude (merge-resolver mode)...")
    rc, transcript = _run_claude_streaming(
        prompt, cwd=worktree, env_extra=repo_mod.author_env(name, email)
    )
    if rc != 0:
        log(f"claude exited with code {rc}")
        return False, None, transcript

    if repo_mod.is_merge_in_progress(worktree):
        log("claude exited but the merge is still in progress; refusing to push")
        return False, None, transcript

    if not _is_clean(worktree):
        log("claude left an uncommitted working tree; refusing to push")
        return False, None, transcript

    after = head_sha(worktree)
    if after == before:
        # Claude aborted the merge.
        log("claude aborted the merge without committing")
        return True, None, transcript

    n = _commits_between(worktree, before, after)
    if n != 1:
        log(f"claude produced {n} commits but a merge resolution should be exactly one")
        return False, None, transcript

    if repo_mod.parent_count(worktree, after) != 2:
        log(
            f"claude's commit {after[:12]} is not a merge commit "
            f"(expected 2 parents)"
        )
        return False, None, transcript

    subject = head_subject(worktree)
    if not subject.startswith(MERGEDOG_PREFIX):
        log(
            f"claude's merge commit {after[:12]} subject does not start "
            f"with {MERGEDOG_PREFIX!r}: {subject!r}"
        )
        return False, None, transcript

    log(f"claude resolved the merge: {after[:12]}: {subject}")
    return True, after, transcript
