"""Invoke a local LLM CLI to investigate and (maybe) fix CI failures."""
from __future__ import annotations

import fcntl
import json
import os
import shlex
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator, Mapping

from mergedog.config import LLMConfig, get_llm_config
from mergedog.log import log
from mergedog.paths import LINTRUNNER_VENV, REPO_DIR
from mergedog.process import run
from mergedog import repo as repo_mod
from mergedog.repo import MERGEDOG_PREFIX, head_sha, head_subject
from mergedog.sanitize import sanitize_untrusted_text

SPURIOUS_MARKER = ".mergedog-spurious"


@dataclass(frozen=True)
class _LLMInvocation:
    provider: str
    cmd: list[str]
    event_label: str
    stdin_input: str | None = None


@dataclass(frozen=True)
class LLMResult:
    ran_cleanly: bool
    new_sha: str | None
    transcript: list[str]
    halt_reason: str | None = None
    spurious_reason: str | None = None

    def __iter__(self) -> Iterator[object]:
        # Preserve the old ``ran_cleanly, new_sha, transcript = invoke_*()``
        # calling convention while letting callers inspect ``halt_reason``.
        yield self.ran_cleanly
        yield self.new_sha
        yield self.transcript


def _escape_embedded_nuls(text: str) -> str:
    return sanitize_untrusted_text(text)


def _build_llm_invocation(
    prompt: str, cwd: Path, config: LLMConfig
) -> _LLMInvocation:
    prompt = _escape_embedded_nuls(prompt)
    model = config.effective_model
    if config.provider == "claude":
        cmd = [
            "claude",
            "-p",
            "--permission-mode",
            "bypassPermissions",
            "--output-format",
            "stream-json",
            "--verbose",
        ]
        if model:
            cmd.extend(["--model", model])
        return _LLMInvocation(config.provider, cmd, "claude", stdin_input=prompt)
    if config.provider == "codex":
        cmd = [
            "codex",
            "exec",
            "--json",
            "--color",
            "never",
            "--dangerously-bypass-approvals-and-sandbox",
            "-C",
            str(cwd),
        ]
        if model:
            cmd.extend(["--model", model])
        return _LLMInvocation(config.provider, cmd, "codex", stdin_input=prompt)
    if config.provider == "metacode":
        cmd = [
            "metacode",
            "run",
            "--yolo",
            "--format",
            "json",
            "--dir",
            str(cwd),
        ]
        if model:
            cmd.extend(["--model", model])
        cmd.append(prompt)
        return _LLMInvocation(config.provider, cmd, "metacode")
    raise AssertionError(f"unknown LLM provider: {config.provider}")


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


def _summarize_claude_event(ev: dict, worktree: Path | None = None) -> Iterator[str]:
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


def _string_from_block(block: object) -> str | None:
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return None
    for key in ("text", "message", "summary", "content", "delta"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _summarize_generic_event(
    provider: str, ev: dict, worktree: Path | None = None
) -> Iterator[str]:
    """Best-effort rendering for JSON event streams that are not Claude's."""
    t = ev.get("type") or ev.get("event") or ev.get("kind")
    if t in ("session.created", "session", "init", "started", "start"):
        sid = ev.get("session_id") or ev.get("sessionId") or ev.get("id")
        if isinstance(sid, str) and sid:
            yield f"{provider} session {sid[:8]}"
        return
    if t in ("exec_command", "command", "tool_call", "tool_use"):
        command = ev.get("command") or ev.get("cmd")
        if isinstance(command, list):
            command = shlex.join(str(x) for x in command)
        if isinstance(command, str) and command.strip():
            yield from _yield_multiline(
                f"{provider} → command: ", _relativize(command, worktree)
            )
            return
    for key in ("message", "text", "content", "output", "result", "delta"):
        value = ev.get(key)
        text = _string_from_block(value)
        if text is None and isinstance(value, list):
            parts = [_string_from_block(block) for block in value]
            text = "\n".join(part for part in parts if part)
        if text:
            for line in _relativize(text, worktree).splitlines():
                if line.strip():
                    yield f"{provider}: {line}"
            return
    if t in ("done", "finished", "result", "completed"):
        yield f"{provider} finished"


def _summarize_event(
    provider: str, ev: dict, worktree: Path | None = None
) -> Iterator[str]:
    if provider == "claude":
        yield from _summarize_claude_event(ev, worktree)
    else:
        yield from _summarize_generic_event(provider, ev, worktree)


def _run_llm_streaming(
    prompt: str, cwd: Path, env_extra: Mapping[str, str], config: LLMConfig
) -> tuple[int, list[str]]:
    """Run the configured LLM and pretty-print events as they arrive.

    Returns ``(returncode, transcript_lines)``: the same human-readable
    summaries we wrote to the log, kept in order, so the shepherd can
    later quote them in a handoff comment.
    """
    nul_count = prompt.count("\x00")
    if nul_count:
        log(f"WARNING: replacing {nul_count} embedded NUL byte(s) in LLM prompt")
    invocation = _build_llm_invocation(prompt, cwd, config)
    # Quote everything except argv-passed prompts, which are huge and would
    # bury the rest of the line. Stdin prompts are noted explicitly.
    redacted = []
    for i, c in enumerate(invocation.cmd):
        if (
            invocation.provider == "metacode"
            and invocation.stdin_input is None
            and i == len(invocation.cmd) - 1
        ):
            redacted.append("<prompt>")
        else:
            redacted.append(c)
    suffix = " < <prompt>" if invocation.stdin_input is not None else ""
    log("$ " + " ".join(shlex.quote(c) for c in redacted) + suffix)
    env = os.environ.copy()
    env.update({k: _escape_embedded_nuls(v) for k, v in env_extra.items()})
    try:
        proc = subprocess.Popen(
            invocation.cmd,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.PIPE if invocation.stdin_input else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as e:
        summary = f"{invocation.event_label} failed to start: {e}"
        log(summary)
        return 127, [summary]
    assert proc.stdout is not None
    if invocation.stdin_input:
        assert proc.stdin is not None
        def _feed(stdin: object, data: str) -> None:
            stdin.write(data)
            stdin.close()
        threading.Thread(
            target=_feed, args=(proc.stdin, invocation.stdin_input), daemon=True,
        ).start()
    transcript: list[str] = []
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            log(f"{invocation.event_label} (raw): {line[:300]}")
            continue
        for summary in _summarize_event(invocation.event_label, ev, worktree=cwd):
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
    log(f"$ {lintrunner} -a -r HEAD~1 --skip {skip}  (post-LLM autoformat)")
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
    """How many commits did the LLM add on top of ``before`` to reach ``after``?

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


def _read_marker_reason(path: Path, *, max_chars: int = 1200) -> str | None:
    try:
        text = sanitize_untrusted_text(path.read_text()).strip()
    except OSError:
        return None
    if not text:
        return None
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + " [truncated]"
    return text


def _invoke(
    worktree: Path,
    prompt: str,
    *,
    mode: str,
    expect_merge_commit: bool,
    expect_rebase_resolution: bool = False,
    expect_cherry_pick_resolution: bool = False,
    allow_multiple_commits: bool = False,
) -> LLMResult:
    """Run the configured LLM and validate that its output meets the contract.

    Shared by ``invoke_fixer``, ``invoke_merge_resolver``, and
    ``invoke_rebase_resolver``. The modes differ in post-run checks:
    ``expect_merge_commit`` rejects leftover mid-merge state and requires
    two parents; ``expect_rebase_resolution`` rejects leftover mid-rebase
    state; ``expect_cherry_pick_resolution`` rejects leftover mid-cherry-pick
    state.

    Returns an :class:`LLMResult`, which can still be unpacked as
    ``(ran_cleanly, new_sha, transcript)``:
    - ``ran_cleanly`` is False if the LLM exited non-zero, left a dirty
      working tree (or unfinished merge), made an unexpected number of
      commits, or violated the commit-message contract -- the harness
      should halt in any of those cases.
    - ``new_sha`` is the SHA of the new ``[MERGEDOG]`` commit if the LLM
      made one, else None. ``(True, None, ...)`` means "the LLM judged
      it a no-op" (explicitly marked spurious failures, an already-satisfied
      operator request, or merge/rebase/cherry-pick aborted).
    - ``transcript`` is the streamed event lines (same as the log) so
      the shepherd can include the LLM's reasoning in a handoff comment.
    - ``halt_reason`` is a specific operator-facing reason when
      ``ran_cleanly`` is false.
    """
    before = head_sha(worktree)
    name, email = repo_mod.get_mergedog_identity()
    llm_config = get_llm_config()
    agent = llm_config.provider
    # A stale spurious marker from a killed prior invocation must not let a
    # silent no-op suppress fresh CI failures.
    spurious_path = worktree / SPURIOUS_MARKER
    if spurious_path.exists():
        spurious_path.unlink()
    log(f"invoking {agent} ({mode} mode)...")
    rc, transcript = _run_llm_streaming(
        prompt, cwd=worktree, env_extra=repo_mod.author_env(name, email),
        config=llm_config,
    )
    if rc != 0:
        start_prefix = f"{agent} failed to start: "
        if transcript and transcript[-1].startswith(start_prefix):
            reason = transcript[-1][len(f"{agent} ") :]
        else:
            reason = f"exited with code {rc}"
            log(f"{agent} {reason}")
        return LLMResult(False, None, transcript, reason)

    if expect_merge_commit and repo_mod.is_merge_in_progress(worktree):
        reason = "exited but the merge is still in progress; refusing to push"
        log(f"{agent} {reason}")
        return LLMResult(False, None, transcript, reason)

    if expect_rebase_resolution and repo_mod.is_rebase_in_progress(worktree):
        reason = "exited but the rebase is still in progress; refusing to push"
        log(f"{agent} {reason}")
        return LLMResult(False, None, transcript, reason)

    if (
        expect_cherry_pick_resolution
        and repo_mod.is_cherry_pick_in_progress(worktree)
    ):
        reason = (
            "exited but the cherry-pick is still in progress; refusing to push"
        )
        log(f"{agent} {reason}")
        return LLMResult(False, None, transcript, reason)

    too_hard_path = worktree / ".mergedog-too-hard"
    too_hard = too_hard_path.exists()
    if too_hard:
        too_hard_path.unlink()

    inconclusive_path = worktree / ".mergedog-inconclusive"
    inconclusive = inconclusive_path.exists()
    if inconclusive:
        inconclusive_path.unlink()

    rebase_path = worktree / ".mergedog-rebase"
    rebase = rebase_path.exists()
    if rebase:
        rebase_path.unlink()

    spurious = spurious_path.exists()
    spurious_reason = _read_marker_reason(spurious_path) if spurious else None
    if spurious:
        spurious_path.unlink()

    if not _is_clean(worktree):
        reason = "left an uncommitted working tree; refusing to push"
        log(f"{agent} {reason}")
        return LLMResult(False, None, transcript, reason)

    after = head_sha(worktree)
    if after == before:
        if too_hard:
            reason = "reported a real PR-related failure that is too hard to fix safely"
            log(f"{agent} {reason}; halting for human review")
            return LLMResult(False, None, transcript, reason)
        if inconclusive:
            reason = "signalled INCONCLUSIVE; halting for human review"
            log(f"{agent} {reason}")
            return LLMResult(False, None, transcript, reason)
        if rebase:
            reason = "requested REBASE; refreshing stale base"
            log(f"{agent} {reason}")
            return LLMResult(False, None, transcript, reason)
        if expect_merge_commit:
            log(f"{agent} aborted the merge without committing")
        elif expect_rebase_resolution:
            log(f"{agent} aborted the rebase without committing")
        elif expect_cherry_pick_resolution:
            log(f"{agent} aborted the cherry-pick without committing")
        elif mode == "fix-CI":
            if not spurious:
                reason = (
                    f"made no commit without signalling {SPURIOUS_MARKER}; "
                    "refusing to mark CI failures spurious"
                )
                log(f"{agent} {reason}")
                return LLMResult(False, None, transcript, reason)
            log(f"{agent} signalled spurious failures (no commit)")
        else:
            log(f"{agent} made no commit (treating as no-op)")
        return LLMResult(
            True, None, transcript, spurious_reason=spurious_reason
        )

    n = _commits_between(worktree, before, after)
    if n != 1 and not allow_multiple_commits:
        expected = (
            "a merge resolution should be exactly one"
            if expect_merge_commit
            else "mergedog only allows one per pass"
        )
        reason = f"produced {n} commits but {expected}"
        log(f"{agent} {reason}")
        return LLMResult(False, None, transcript, reason)
    if n < 1:
        reason = "produced no commits but HEAD moved unexpectedly"
        log(f"{agent} {reason}")
        return LLMResult(False, None, transcript, reason)

    if expect_merge_commit and repo_mod.parent_count(worktree, after) != 2:
        reason = (
            f"produced commit {after[:12]} that is not a merge commit "
            "(expected 2 parents)"
        )
        log(f"{agent} {reason}")
        return LLMResult(False, None, transcript, reason)

    subject = head_subject(worktree)
    # Rebase/cherry-pick resolution preserves the original commit message --
    # don't require the [MERGEDOG] prefix.
    if (
        not expect_rebase_resolution
        and not expect_cherry_pick_resolution
        and not subject.startswith(MERGEDOG_PREFIX)
    ):
        reason = (
            f"produced commit {after[:12]} whose subject does not start "
            f"with {MERGEDOG_PREFIX!r}: {subject!r}"
        )
        log(f"{agent} {reason}")
        return LLMResult(False, None, transcript, reason)

    # Run lintrunner -a and fold any auto-fixes into claude's commit.
    # Catches autoformat misses (clang-format, ruff format, ...) before
    # they'd cause a wasted fix-CI cycle on a follow-up lint failure.
    # Skipped for merge/rebase/cherry-pick resolver: the diff view would
    # include everything brought in from main/parent -- too broad and too slow.
    if (
        not expect_merge_commit
        and not expect_rebase_resolution
        and not expect_cherry_pick_resolution
    ):
        amended = _run_lintrunner_amend(worktree)
        if amended is not None:
            after = amended
            subject = head_subject(worktree)

    if expect_merge_commit:
        log(f"{agent} resolved the merge: {after[:12]}: {subject}")
    elif expect_rebase_resolution:
        log(f"{agent} resolved the rebase: {after[:12]}: {subject}")
    elif expect_cherry_pick_resolution:
        log(f"{agent} resolved the cherry-pick: {after[:12]}: {subject}")
    else:
        log(f"{agent} produced fix commit {after[:12]}: {subject}")
    return LLMResult(True, after, transcript)


def invoke_fixer(worktree: Path, prompt: str) -> LLMResult:
    """Run the configured LLM in the worktree to fix CI failures."""
    return _invoke(worktree, prompt, mode="fix-CI", expect_merge_commit=False)


def invoke_operator_fix(worktree: Path, prompt: str) -> LLMResult:
    """Run the configured LLM in the worktree for a trusted operator request."""
    return _invoke(worktree, prompt, mode="operator-fix", expect_merge_commit=False)


def invoke_merge_resolver(worktree: Path, prompt: str) -> LLMResult:
    """Run the configured LLM in a mid-merge worktree to resolve conflicts."""
    return _invoke(worktree, prompt, mode="merge-resolver", expect_merge_commit=True)


def invoke_rebase_resolver(
    worktree: Path, prompt: str, *, allow_multiple_commits: bool = False
) -> LLMResult:
    """Run the configured LLM in a mid-rebase worktree to resolve conflicts."""
    return _invoke(
        worktree, prompt, mode="rebase-resolver",
        expect_merge_commit=False, expect_rebase_resolution=True,
        allow_multiple_commits=allow_multiple_commits,
    )


def invoke_cherry_pick_resolver(worktree: Path, prompt: str) -> LLMResult:
    """Run the configured LLM in a mid-cherry-pick worktree."""
    return _invoke(
        worktree,
        prompt,
        mode="cherry-pick-resolver",
        expect_merge_commit=False,
        expect_cherry_pick_resolution=True,
    )
