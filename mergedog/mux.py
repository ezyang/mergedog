"""Tiny multi-PR supervisor with a Textual TUI.

Top: a DataTable with one row per shepherd subprocess in this mux session.
Bottom: an Input field that takes commands.

Run via::

    python -m mergedog.mux [<pr>...] [--resume-known|--no-resume-known]

Commands typed at the bottom (enter to submit):

    add <pr> [extra mergedog flags]   start a shepherd
    <pr>                              shorthand for ``add <pr>``
    fix <pr> <trusted request>        make an operator-requested follow-up
    restart <pr>                      kill and re-spawn a shepherd
    restart all                       kill and re-spawn every session job
    restart dead                      re-spawn only crashed shepherds
    rebase <pr>                       start/restart a shepherd with --rebase
    rebase all                        re-run every session job with --rebase
    mark-spurious <pr>                mark current failed/cancelled checks
                                      spurious and restart the shepherd
    cancel <pr>                       SIGTERM a shepherd (keeps state)
    cleanup | clean                   forget successful completed shepherds
    remove <pr>                       SIGTERM and forget (wipes worktree)
    log <pr>                          show the path to its log file
    help                              show phase meanings and commands
    ignore-sev [on|off]               toggle (or show) the mux-wide
                                      ``--ignore-sev`` default applied to
                                      every shepherd spawn
    ignore-sev add <issue>            persistently ignore one ci: sev issue;
    ignore-sev remove <issue>         parked shepherds re-read this each poll
    fix-cap [N|off|default]           set/show the mux-wide
                                      ``--max-fix-commits`` default. ``off``
                                      disables the cap for future spawns
    migrate                            print commands to resume on another
                                      server, then quit
    quit                              terminate everything and exit

Each shepherd's stdout/stderr is piped to ``~/.mergedog/logs/<pr>.log``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from mergedog.bootstrap import promote_early_env

promote_early_env(sys.argv[1:])

from rich.text import Text  # noqa: E402
from textual import work  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.suggester import SuggestFromList  # noqa: E402
from textual.widgets import DataTable, Input, Static  # noqa: E402


COMMAND_SUGGESTIONS = [
    "add ",
    "cancel ",
    "cleanup",
    "cleanup all",
    "clean",
    "clean all",
    "fix ",
    "fix-cap ",
    "help",
    "ignore-sev ",
    "ignore-sev add ",
    "ignore-sev remove ",
    "ignore-sev list",
    "ignore-sev clear",
    "log ",
    "mark-spurious ",
    "migrate",
    "quit",
    "reassess ",
    "rebase ",
    "rebase all",
    "remove ",
    "restart ",
    "restart all",
    "restart dead",
    "status",
]

PHASE_NO_ACTION = "🟢"
PHASE_YOUR_EASY_ACTION = "🟡"
PHASE_YOUR_REVIEW_ACTION = "🟠"
PHASE_EXTERNAL_ACTION = "🔵"
PHASE_HALTED = "🔴"
CLEANUP_HINT = "Run 'cleanup' to remove finished mergedogs"


class HistoryInput(Input):
    """Input widget with rlwrap-style command history (Up/Down arrows)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        self._history_index: int = 0
        self._saved_line: str = ""

    def _on_key(self, event) -> None:
        if event.key == "up":
            event.prevent_default()
            event.stop()
            if not self._history:
                return
            if self._history_index == len(self._history):
                self._saved_line = self.value
            if self._history_index > 0:
                self._history_index -= 1
                self.value = self._history[self._history_index]
                self.cursor_position = len(self.value)
        elif event.key == "down":
            event.prevent_default()
            event.stop()
            if self._history_index < len(self._history) - 1:
                self._history_index += 1
                self.value = self._history[self._history_index]
                self.cursor_position = len(self.value)
            elif self._history_index == len(self._history) - 1:
                self._history_index = len(self._history)
                self.value = self._saved_line
                self.cursor_position = len(self.value)

    def record(self, line: str) -> None:
        if line and (not self._history or self._history[-1] != line):
            self._history.append(line)
        self._history_index = len(self._history)
        self._saved_line = ""

from mergedog import github  # noqa: E402
from mergedog.cli import _parse_pr  # noqa: E402
from mergedog.config import (  # noqa: E402
    add_ignored_ci_sev,
    clear_ignored_ci_sevs,
    format_ci_sev_ignored_numbers,
    get_ci_sev_config,
    parse_ci_sev_number,
    remove_ignored_ci_sev,
)
from mergedog.handoff import is_cla_merge_failure  # noqa: E402
from mergedog.ipc import acquire_lock, release_lock  # noqa: E402
from mergedog.paths import (  # noqa: E402
    LOGS_DIR,
    MUX_JOBS_FILE,
    MUX_PRS_FILE,
    MUX_SOCKET,
    REPO_SLUG,
    atomic_write_text,
    context_file,
    ensure_dirs,
    log_file,
    state_file,
    status_file,
    worktree_dir,
)
from mergedog.shepherd import (  # noqa: E402
    EXIT_PR_NOT_ACTIONABLE,
    MAX_FIX_COMMITS,
    MERGEDOG_LABEL,
)
from mergedog.status import read_status  # noqa: E402
from mergedog.state import TrustDB  # noqa: E402

TITLE_TRUNC = 20
JobKey = tuple[str, int]
PR_JOB = "pr"
_STACK_PARENT_RE = re.compile(r"\bstack parent(?: PR)? #(\d+)\b")


def _pr_job(pr: int) -> JobKey:
    return (PR_JOB, pr)


def _coerce_job(job: JobKey | int) -> JobKey:
    if isinstance(job, tuple):
        return job
    return _pr_job(job)


def _job_label(job: JobKey | int) -> str:
    _, pr = _coerce_job(job)
    return str(pr)


def _job_log_file(job: JobKey | int) -> Path:
    _, pr = _coerce_job(job)
    return log_file(pr)


def _read_pr_context_title(pr: int) -> str:
    """Read the PR title from the sidecar populated from GitHub metadata."""
    try:
        lines = context_file(pr).read_text().splitlines()
    except OSError:
        return ""
    try:
        title_index = lines.index("[TITLE]") + 1
    except ValueError:
        return ""
    if title_index >= len(lines):
        return ""
    title = lines[title_index].strip()
    return title


def _read_pr_worktree_title(pr: int) -> str:
    """Best-effort title fallback from the worktree's HEAD subject.

    ``--invert-grep`` skips ``[MERGEDOG]`` commits so an in-flight
    merge-main doesn't replace the actual contributor title with the
    merge commit's subject. ``--first-parent`` keeps the fallback on
    the PR branch side of a merge-main commit, rather than walking into
    unrelated upstream commits from ``main``.
    """
    wt = worktree_dir(pr)
    if not wt.exists():
        return ""
    try:
        proc = subprocess.run(
            [
                "git", "log", "-1", "--pretty=%s",
                "--first-parent",
                "--invert-grep", "--grep=^\\[MERGEDOG\\]", "HEAD",
            ],
            cwd=wt,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _read_pr_title(pr: int) -> str:
    """Best-effort PR title for mux display."""
    return _read_pr_context_title(pr) or _read_pr_worktree_title(pr)


def _read_pr_commit_parent(pr: int) -> tuple[str, str] | None:
    """Best-effort contributor commit and first parent for a tracked PR.

    Independently shepherded ghstack PRs each check out a branch whose
    contributor commit is parented by the previous PR's contributor commit.
    This gives mux enough local information to group rows cosmetically
    without polling GitHub from the refresh loop.
    """
    wt = worktree_dir(pr)
    if not wt.exists():
        return None
    try:
        proc = subprocess.run(
            [
                "git", "log", "-1", "--format=%H%x00%P",
                "--first-parent",
                "--invert-grep", "--grep=^\\[MERGEDOG\\]", "HEAD",
            ],
            cwd=wt,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    line = proc.stdout.strip()
    if not line:
        return None
    sha, _, parents = line.partition("\x00")
    parent = parents.split()[0] if parents.split() else ""
    if not sha or not parent:
        return None
    return sha, parent


def _read_stack_parent_pr_from_log(path: Path) -> int | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fp:
            lines = fp.readlines()
    except OSError:
        return None
    for line in reversed(lines):
        m = _STACK_PARENT_RE.search(line)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    return None


def _stack_display_layout(
    jobs: list[JobKey],
    parent_hints: dict[JobKey, JobKey] | None = None,
) -> tuple[list[JobKey], dict[JobKey, int]]:
    """Return jobs sorted with local PR commit stacks grouped bottom-up."""
    base_order = sorted(jobs)
    base_index = {job: i for i, job in enumerate(base_order)}
    job_set = set(base_order)

    sha_by_job: dict[JobKey, str] = {}
    parent_sha_by_job: dict[JobKey, str] = {}
    for job in base_order:
        kind, pr = job
        if kind != PR_JOB:
            continue
        info = _read_pr_commit_parent(pr)
        if info is None:
            continue
        sha, parent_sha = info
        sha_by_job[job] = sha
        parent_sha_by_job[job] = parent_sha

    jobs_by_sha: dict[str, list[JobKey]] = {}
    for job, sha in sha_by_job.items():
        jobs_by_sha.setdefault(sha, []).append(job)
    unique_job_by_sha = {
        sha: sha_jobs[0]
        for sha, sha_jobs in jobs_by_sha.items()
        if len(sha_jobs) == 1
    }

    parent_of: dict[JobKey, JobKey] = {}
    children: dict[JobKey, list[JobKey]] = {job: [] for job in base_order}
    for job, parent in (parent_hints or {}).items():
        if job not in job_set or parent not in job_set or parent == job:
            continue
        parent_of[job] = parent

    for job, parent_sha in parent_sha_by_job.items():
        if job in parent_of:
            continue
        parent = unique_job_by_sha.get(parent_sha)
        if parent is None or parent == job:
            continue
        parent_of[job] = parent
    for job, parent in parent_of.items():
        children[parent].append(job)
    for child_jobs in children.values():
        child_jobs.sort(key=lambda job: base_index[job])

    ordered: list[JobKey] = []
    depths: dict[JobKey, int] = {}
    seen: set[JobKey] = set()

    def emit(job: JobKey, depth: int) -> None:
        if job in seen:
            return
        seen.add(job)
        ordered.append(job)
        depths[job] = depth
        for child in children[job]:
            emit(child, depth + 1)

    for job in base_order:
        if job in seen:
            continue
        root = job
        ancestors: set[JobKey] = set()
        while root in parent_of and root not in ancestors:
            ancestors.add(root)
            root = parent_of[root]
        emit(root, 0)

    return ordered, depths


def _truncate_title(title: str, n: int = TITLE_TRUNC) -> str:
    if len(title) <= n:
        return title
    return title[: n - 1] + "…"


def _last_log_line(path: Path) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fp:
            tail = fp.readlines()[-1] if fp.readable() else ""
    except OSError:
        return ""
    return tail.rstrip()


def _crash_summary_from_log(path: Path) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fp:
            lines = [line.rstrip() for line in fp.readlines()[-200:]]
    except OSError:
        return ""
    traceback_start = None
    for i, line in enumerate(lines):
        if line == "Traceback (most recent call last):":
            traceback_start = i
    if traceback_start is None:
        return ""
    for line in reversed(lines[traceback_start + 1 :]):
        line = line.strip()
        if not line:
            continue
        return line[:300]
    return ""


def _status_updated_ts(structured: dict | None) -> float | None:
    if structured is None:
        return None
    updated_at = structured.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at:
        return None
    try:
        return datetime.fromisoformat(
            updated_at.replace("Z", "+00:00")
        ).timestamp()
    except ValueError:
        return None


def _status_is_stale(
    structured: dict | None,
    *,
    started_at: float | None,
    rc: int | None = None,
) -> bool:
    if structured is None:
        return False
    if started_at is not None:
        updated_at = _status_updated_ts(structured)
        if updated_at is None:
            return True
        if updated_at < started_at:
            return True
    if rc is not None and rc not in (0, EXIT_PR_NOT_ACTIONABLE):
        return structured.get("phase") != "halted"
    return False


def _record_job_started(app: object, job: JobKey, started_at: float) -> None:
    started = getattr(app, "_job_started_at", None)
    if started is None:
        started = {}
        setattr(app, "_job_started_at", started)
    started[job] = started_at


def _cleanup_job_files(job: JobKey | int) -> None:
    """Remove per-PR on-disk state for a forgotten mux job."""
    _, pr = _coerce_job(job)
    for path in (state_file(pr), context_file(pr), status_file(pr)):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
    try:
        from mergedog import repo as repo_mod

        repo_mod.wipe_worktree(pr)
    except Exception:
        pass


def _status_message(
    structured: dict | None,
    *,
    rc: int | None,
    last_log: str,
    crash_summary: str = "",
    stale: bool = False,
) -> str:
    if stale:
        if rc is not None and rc not in (0, EXIT_PR_NOT_ACTIONABLE):
            if crash_summary:
                return f"HALT: shepherd crashed: {crash_summary}"
            if last_log:
                return f"HALT: shepherd exited: {last_log}"
            return "HALT: shepherd exited; ignoring stale status"
        return "starting; ignoring stale status from previous shepherd"
    if structured is not None:
        msg = structured.get("message")
        if isinstance(msg, str) and msg:
            return msg
        phase = structured.get("phase")
        if phase == "polling_ci":
            done = structured.get("ci_done", "?")
            total = structured.get("ci_total", "?")
            failed = structured.get("ci_failed")
            if failed:
                return (
                    f"CI failed: {failed} active failures "
                    f"({done}/{total} checks done)"
                )
            return f"waiting for CI: {done}/{total} checks done"
        if isinstance(phase, str) and phase:
            return phase.replace("_", " ")
    if rc is None:
        return "starting; waiting for status"
    if rc in (0, EXIT_PR_NOT_ACTIONABLE):
        return last_log or "complete"
    return "HALT: see log"


def _phase_label(
    structured: dict | None,
    *,
    rc: int | None,
    stale: bool = False,
    cla_blocked: bool = False,
) -> str:
    if rc is not None and rc not in (0, EXIT_PR_NOT_ACTIONABLE):
        return PHASE_HALTED
    if stale:
        return PHASE_NO_ACTION
    if structured is not None:
        category = structured.get("category")
        phase = structured.get("phase")
        if category == "done" or phase == "complete":
            return ""
        if category == "blocked":
            return PHASE_HALTED
        if phase == "halted":
            return PHASE_HALTED
        if cla_blocked:
            return PHASE_EXTERNAL_ACTION
        if _has_user_action(structured):
            if _user_action_needs_review(structured):
                return PHASE_YOUR_REVIEW_ACTION
            return PHASE_YOUR_EASY_ACTION
        if _waiting_on_external_human(structured):
            return PHASE_EXTERNAL_ACTION
        if category == "ready":
            return PHASE_YOUR_EASY_ACTION
        if category == "waiting":
            return PHASE_NO_ACTION
        if category == "action":
            return PHASE_NO_ACTION
        if phase == "ready":
            return PHASE_YOUR_EASY_ACTION
    if rc is None:
        return PHASE_NO_ACTION
    return ""


def _status_text_field(structured: dict, key: str) -> str:
    value = structured.get(key)
    return value.strip() if isinstance(value, str) else ""


def _status_int_field(structured: dict, key: str) -> int:
    value = structured.get(key)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _has_user_action(structured: dict) -> bool:
    return bool(_status_text_field(structured, "user_action"))


def _user_action_needs_review(structured: dict) -> bool:
    if _status_text_field(structured, "waiting_on") == "approval":
        return True
    if structured.get("handoff_comment_ok") is False:
        return True
    return _status_int_field(structured, "intervention_count") > 0


def _waiting_on_external_human(structured: dict) -> bool:
    waiting_on = _status_text_field(structured, "waiting_on")
    return waiting_on in {
        "approval",
        "contributor",
        "human_merge",
        "reviewer",
        "maintainer",
    }


def _read_last_observed_failure_body(pr: int) -> str:
    try:
        data = json.loads(state_file(pr).read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    body = data.get("last_observed_failure_body")
    return body if isinstance(body, str) else ""


def _status_has_cla_blocker(pr: int, structured: dict | None) -> bool:
    if structured is None or structured.get("merging") is True:
        return False
    phase = _status_text_field(structured, "phase")
    category = _status_text_field(structured, "category")
    waiting_on = _status_text_field(structured, "waiting_on")
    if waiting_on == "contributor":
        return True
    if phase not in {"ready", "watching_merge"}:
        return False
    if category not in {"ready", "waiting"}:
        return False
    return is_cla_merge_failure(_read_last_observed_failure_body(pr))


def _cla_blocked_status_message(status: str) -> str:
    prefix = "ready for human merge"
    if status.startswith(prefix):
        return "waiting for contributor CLA" + status[len(prefix):]
    return status or "waiting for contributor CLA"


def _read_mux_prs() -> list[int]:
    """Curated list of PRs the mux is tracking.

    Stored at ``MUX_PRS_FILE`` so a restart only resumes PRs the operator
    explicitly added -- not every stale ``state/<pr>.json`` left behind
    by a long-since-merged shepherd.
    """
    if not MUX_PRS_FILE.exists():
        return []
    try:
        data = json.loads(MUX_PRS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    out: set[int] = set()
    for x in data:
        try:
            out.add(int(x))
        except (TypeError, ValueError):
            continue
    return sorted(out)


def _write_mux_prs(prs: list[int]) -> None:
    atomic_write_text(MUX_PRS_FILE, json.dumps(sorted(set(prs))))


def _read_mux_job_records() -> dict[JobKey, dict]:
    """Curated mux jobs with their persisted metadata.

    ``mux-prs.json`` remains as the backwards-compatible regular-PR list
    for older tools. Newer mux instances persist jobs here. Each record
    may carry ``rc``: the exit code the shepherd had when the previous
    mux shut down, so a restart can tell halted jobs from running ones.
    """
    if not MUX_JOBS_FILE.exists():
        return {_pr_job(pr): {} for pr in _read_mux_prs()}
    try:
        data = json.loads(MUX_JOBS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[JobKey, dict] = {}
    for item in data:
        try:
            if isinstance(item, int):
                out[_pr_job(item)] = {}
            elif isinstance(item, str):
                out[_pr_job(_parse_pr(item))] = {}
            elif isinstance(item, dict):
                kind = str(item.get("kind", PR_JOB))
                pr = int(item["pr"])
                if kind == PR_JOB:
                    record = {}
                    if isinstance(item.get("rc"), int):
                        record["rc"] = item["rc"]
                    out[_pr_job(pr)] = record
        except (TypeError, ValueError, KeyError):
            continue
    return out


def _read_mux_jobs() -> list[JobKey]:
    return sorted(_read_mux_job_records())


def _write_mux_job_records(records: dict[JobKey, dict]) -> None:
    items = []
    for kind, pr in sorted(records):
        item: dict = {"kind": kind, "pr": pr}
        rc = records[(kind, pr)].get("rc")
        if isinstance(rc, int):
            item["rc"] = rc
        items.append(item)
    atomic_write_text(MUX_JOBS_FILE, json.dumps(items))
    _write_mux_prs([pr for _, pr in records])


def _write_mux_jobs(jobs: list[JobKey]) -> None:
    _write_mux_job_records({job: {} for job in jobs})


def _add_mux_job(job: JobKey) -> bool:
    """Persist ``job`` as a tracked mux member.

    Returns ``True`` only when the job was not previously tracked at all --
    i.e. it is genuinely joining the mux. Re-adding an already-tracked job
    (restart/rebase/reassess) or un-parking a halted one returns ``False``,
    so callers can apply the ``mergedog`` join label exactly once.
    """
    records = _read_mux_job_records()
    newly_joined = job not in records
    # Always rewrite the record: a (re)spawned job sheds any recorded
    # exit code, so a later resume treats it as live rather than parked.
    if records.get(job) == {}:
        return False
    records[job] = {}
    _write_mux_job_records(records)
    return newly_joined


def _remove_mux_job(job: JobKey) -> None:
    records = _read_mux_job_records()
    if job not in records:
        return
    records.pop(job)
    _write_mux_job_records(records)


def _resolve_initial_jobs(
    raw_prs: list[str],
    *,
    resume_known: bool,
) -> tuple[
    list[JobKey],
    dict[JobKey, int],
    list[tuple[str, argparse.ArgumentTypeError]],
]:
    """Split resumable jobs from parked (previously halted) ones.

    A job whose record carries an exit code halted before the previous
    mux shut down. Respawning it would just repeat the halt -- and
    re-fire its halt notification -- so it is parked: shown in the table
    as HALT, only respawned on an explicit ``restart``/``add``. An
    explicitly listed PR on the command line *is* that explicit ask, so
    it overrides parking.
    """
    initial: list[JobKey] = []
    parked: dict[JobKey, int] = {}
    if resume_known:
        for job, record in sorted(_read_mux_job_records().items()):
            rc = record.get("rc")
            if isinstance(rc, int):
                parked[job] = rc
            else:
                initial.append(job)

    skipped: list[tuple[str, argparse.ArgumentTypeError]] = []
    for raw in raw_prs:
        try:
            initial.append(_pr_job(_parse_pr(raw)))
        except argparse.ArgumentTypeError as e:
            skipped.append((raw, e))

    seen: set[JobKey] = set()
    initial = [job for job in initial if not (job in seen or seen.add(job))]
    for job in initial:
        parked.pop(job, None)
    return initial, parked, skipped


def _spawn(
    job: JobKey | int,
    extra: list[str],
    *,
    spawn_pr: int | None = None,
) -> tuple[subprocess.Popen, object, Path]:
    job = _coerce_job(job)
    _, pr = job
    arg_pr = spawn_pr if spawn_pr is not None else pr
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _job_log_file(job)
    f = open(log_path, "a", buffering=1, encoding="utf-8")
    f.write(f"\n=== mergedog start at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    command = [sys.executable, "-m", "mergedog"]
    command.extend([str(arg_pr), *extra])
    p = subprocess.Popen(
        command,
        stdout=f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        # Make each shepherd its own session/process-group leader. Without
        # this, ``cancel`` would SIGTERM only the python interpreter and
        # leave any in-flight ``claude`` / ``gh`` / ``git`` subprocess
        # running orphaned -- claude in particular can keep editing the
        # worktree after we thought the shepherd was dead.
        start_new_session=True,
    )
    return (p, f, log_path)


def _terminate_group(p: subprocess.Popen, *, grace: float = 5.0) -> None:
    """SIGTERM the process group; SIGKILL it after ``grace`` seconds if needed."""
    if p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        p.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            p.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass


class MuxApp(App):
    CSS = """
    DataTable { height: 1fr; }
    #cleanup-hint {
        display: none;
        padding: 0 1;
        color: $text-muted;
    }
    Input { dock: bottom; }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(
        self,
        initial: list[JobKey | int],
        *,
        parked: dict[JobKey, int] | None = None,
        ignore_sev: bool = False,
        max_fix_commits: int = MAX_FIX_COMMITS,
        gchat_to: str | None = None,
        repo_slug: str = REPO_SLUG,
        lock_fd: int = -1,
    ) -> None:
        super().__init__()
        self.procs: dict[JobKey, tuple[subprocess.Popen, object, Path]] = {}
        self._job_started_at: dict[JobKey, float] = {}
        self._pr_titles: dict[JobKey, str] = {}
        self._pr_status: dict[JobKey, dict] = {}
        self._cleanup_jobs: set[JobKey] = set()
        self._cleanup_status: dict[JobKey, str] = {}
        # Jobs that halted before a previous mux shut down. Kept visible
        # as HALT rows but not respawned: rerunning them would repeat the
        # halt (and its notification) on every mux restart. An explicit
        # ``restart``/``add`` unparks them.
        self._parked_jobs: dict[JobKey, int] = dict(parked or {})
        self._initial = [_coerce_job(job) for job in initial]
        self.ignore_sev = ignore_sev
        self.max_fix_commits = max_fix_commits
        self.gchat_to = gchat_to
        self.repo_slug = repo_slug
        self._lock_fd = lock_fd
        self._ipc_server: asyncio.AbstractServer | None = None
        self._unresumable_jobs: set[JobKey] = set()

    def compose(self) -> ComposeResult:
        yield DataTable()
        yield Static("", id="cleanup-hint")
        yield HistoryInput(
            placeholder=(
                "<pr> | add <pr> | restart <pr|all|dead> | rebase <pr|all> | reassess <pr> | "
                "fix <pr> | mark-spurious <pr> | cancel <pr> | cleanup | clean | "
                "remove <pr> | log <pr> | fix-cap | "
                "help | migrate | quit"
            ),
            suggester=SuggestFromList(COMMAND_SUGGESTIONS),
        )

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("PR", "Title", "Phase", "Status")
        for job in self._initial:
            self._do_add_job(job, [])
        self._refresh()
        self.set_interval(2.0, self._refresh)
        self.query_one(HistoryInput).focus()
        self._start_ipc_server()

    def _shepherd_args(self, extra: list[str]) -> list[str]:
        """Apply mux-wide defaults to a shepherd argv tail."""
        out = list(extra)
        if self.ignore_sev and "--ignore-sev" not in out:
            out = ["--ignore-sev", *out]
        max_fix_commits = getattr(self, "max_fix_commits", MAX_FIX_COMMITS)
        if max_fix_commits != MAX_FIX_COMMITS and not any(
            a == "--max-fix-commits" or a.startswith("--max-fix-commits=")
            for a in out
        ):
            out = [f"--max-fix-commits={max_fix_commits}", *out]
        if self.gchat_to and not any(a.startswith("--gchat-to") for a in out):
            out = [f"--gchat-to={self.gchat_to}", *out]
        if self.repo_slug and not any(
            a == "--repo" or a.startswith("--repo=") for a in out
        ):
            out = [f"--repo={self.repo_slug}", *out]
        return out

    def _do_add_job(
        self,
        job: JobKey | int,
        extra: list[str],
        *,
        spawn_pr: int | None = None,
        alias_job: JobKey | None = None,
    ) -> str:
        job = _coerce_job(job)
        label = _job_label(job)
        if job in getattr(self, "_cleanup_jobs", set()):
            return f"[{label}] cleanup in progress"
        if job in self.procs and self.procs[job][0].poll() is None:
            return f"[{label}] already running"
        getattr(self, "_parked_jobs", {}).pop(job, None)
        started_at = time.time()
        try:
            self.procs[job] = _spawn(
                job, self._shepherd_args(extra), spawn_pr=spawn_pr
            )
        except Exception as e:
            return f"[{label}] spawn failed: {e}"
        _record_job_started(self, job, started_at)
        self._pr_titles.pop(job, None)
        self._unresumable_jobs.discard(job)
        if alias_job is not None and alias_job != job:
            _remove_mux_job(alias_job)
        if _add_mux_job(job):
            # First time this PR joins the mux: stamp the ``mergedog`` label
            # so it is visible on GitHub as long-term tracked. Restarts and
            # rebases re-enter _do_add_job but _add_mux_job returns False for
            # an already-tracked job, so we don't re-hit the API each time.
            self._set_mergedog_label(job, present=True)
        return f"[{label}] started"

    def _set_mergedog_label(self, job: JobKey | int, *, present: bool) -> None:
        """Best-effort add/remove of the ``mergedog`` mux-membership label.

        The label tracks mux membership only: added when a PR joins, removed
        only on an explicit ``remove``. A failure here never blocks the
        command -- it is surfaced as a notification and otherwise ignored.
        """
        _, pr = _coerce_job(job)
        try:
            if present:
                github.add_label(pr, MERGEDOG_LABEL, loud=False)
            else:
                github.remove_label(pr, MERGEDOG_LABEL)
        except Exception as e:
            self.notify(
                f"[{pr}] mergedog label {'add' if present else 'remove'} "
                f"failed: {e}",
                severity="warning",
            )

    def _do_add(self, pr: int, extra: list[str]) -> str:
        return self._do_add_job(_pr_job(pr), extra)

    def _operator_fix_args(self, args: list[str]) -> list[str] | str:
        if not args:
            return "missing operator fix request"
        request = " ".join(args).strip()
        if not request:
            return "missing operator fix request"
        return [f"--operator-fix-context={request}"]

    def _dead_jobs(self) -> list[JobKey]:
        dead = {
            job
            for job, (p, _, _) in self.procs.items()
            if p.poll() not in (None, 0, EXIT_PR_NOT_ACTIONABLE)
        }
        dead.update(
            job
            for job in getattr(self, "_parked_jobs", {})
            if job not in self.procs
        )
        return sorted(dead)

    def _completed_jobs(self) -> list[JobKey]:
        cleanup_jobs = getattr(self, "_cleanup_jobs", set())
        return sorted(
            job
            for job, (p, _, _) in self.procs.items()
            if job not in cleanup_jobs and p.poll() in (0, EXIT_PR_NOT_ACTIONABLE)
        )

    def _refresh_cleanup_hint(self) -> None:
        hint = self.query_one("#cleanup-hint", Static)
        completed = bool(self._completed_jobs())
        hint.update(CLEANUP_HINT if completed else "")
        hint.display = completed

    def _restart_jobs(
        self,
        jobs: list[JobKey | int],
        extra: list[str],
        action: str,
        empty: str,
    ) -> None:
        # Runs in a worker thread so the input bar / table stay responsive
        # while shepherds tear down. ``self.notify`` is thread-safe in
        # Textual (it posts a message to the app loop).
        cleanup_jobs = getattr(self, "_cleanup_jobs", set())
        parked_jobs = getattr(self, "_parked_jobs", {})
        jobs = [_coerce_job(job) for job in jobs]
        jobs = [
            job
            for job in jobs
            if (job in self.procs or job in parked_jobs)
            and job not in cleanup_jobs
        ]
        if not jobs:
            self.notify(empty, severity="warning")
            return
        # Signal every shepherd up front so they all wind down in
        # parallel. Without this, each ``_terminate_group`` below would
        # block up to ``grace`` seconds *per PR* in series before the
        # next SIGTERM was even sent.
        for job in jobs:
            if job not in self.procs:
                continue
            p = self.procs[job][0]
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        # Now reap. ``_terminate_group`` is a no-op for already-exited
        # processes, and the second SIGTERM it sends to stragglers is
        # harmless.
        for job in jobs:
            if job in self.procs:
                _terminate_group(self.procs[job][0])
        shepherd_args = self._shepherd_args(extra)
        for job in jobs:
            parked_jobs.pop(job, None)
            started_at = time.time()
            try:
                self.procs[job] = _spawn(job, shepherd_args)
            except Exception as e:
                self.notify(f"[{_job_label(job)}] failed: {e}", severity="error")
            else:
                _record_job_started(self, job, started_at)
            self._pr_titles.pop(job, None)
        self.notify(f"{action} {len(jobs)} job(s)")

    def _all_jobs(self) -> list[JobKey]:
        return sorted(set(self.procs) | set(getattr(self, "_parked_jobs", {})))

    @work(thread=True, exclusive=True, group="restart-all")
    def _do_restart_all(self, extra: list[str] | None = None) -> None:
        self._restart_jobs(
            self._all_jobs(),
            extra or [],
            "restarting",
            "no PRs to restart",
        )

    @work(thread=True, exclusive=True, group="restart-all")
    def _do_restart_dead(
        self,
        jobs: list[JobKey | int] | None = None,
        extra: list[str] | None = None,
    ) -> None:
        self._restart_jobs(
            jobs if jobs is not None else self._dead_jobs(),
            extra or [],
            "restarting dead",
            "no dead PRs to restart",
        )

    @work(thread=True, exclusive=True, group="restart-all")
    def _do_rebase_all(self) -> None:
        self._restart_jobs(
            self._all_jobs(),
            ["--rebase"],
            "rebasing",
            "no PRs to rebase",
        )

    def _do_ignore_sev(self, rest: list[str]) -> str:
        if not rest:
            state = "on" if self.ignore_sev else "off"
            cfg = get_ci_sev_config()
            ignored = format_ci_sev_ignored_numbers(cfg.ignored_numbers)
            return f"ignore-sev is {state}; ignored ci: sev: {ignored}"
        arg = rest[0].lower()
        if arg in ("on", "true", "1", "yes"):
            new = True
        elif arg in ("off", "false", "0", "no"):
            new = False
        elif arg == "toggle":
            new = not self.ignore_sev
        elif arg in ("list", "show"):
            if len(rest) != 1:
                return "usage: ignore-sev list"
            cfg = get_ci_sev_config()
            ignored = format_ci_sev_ignored_numbers(cfg.ignored_numbers)
            return f"ignored ci: sev: {ignored}"
        elif arg in ("add", "ignore"):
            if len(rest) != 2:
                return "usage: ignore-sev add <issue>"
            number = parse_ci_sev_number(rest[1])
            cfg = add_ignored_ci_sev(number)
            ignored = format_ci_sev_ignored_numbers(cfg.ignored_numbers)
            return (
                f"ignored ci: sev #{number} (persistent; "
                f"ignored ci: sev: {ignored})"
            )
        elif arg in ("remove", "rm", "del", "delete", "unignore"):
            if len(rest) != 2:
                return "usage: ignore-sev remove <issue>"
            number = parse_ci_sev_number(rest[1])
            cfg = remove_ignored_ci_sev(number)
            ignored = format_ci_sev_ignored_numbers(cfg.ignored_numbers)
            return (
                f"respecting ci: sev #{number} (persistent; "
                f"ignored ci: sev: {ignored})"
            )
        elif arg == "clear":
            if len(rest) != 1:
                return "usage: ignore-sev clear"
            clear_ignored_ci_sevs()
            return "cleared ignored ci: sev list"
        else:
            try:
                number = parse_ci_sev_number(arg)
            except ValueError:
                return (
                    "usage: ignore-sev "
                    "[on|off|toggle|add <issue>|remove <issue>|clear|list]"
                )
            cfg = add_ignored_ci_sev(number)
            ignored = format_ci_sev_ignored_numbers(cfg.ignored_numbers)
            return (
                f"ignored ci: sev #{number} (persistent; "
                f"ignored ci: sev: {ignored})"
            )
        self.ignore_sev = new
        state = "on" if new else "off"
        return (
            f"ignore-sev {state} (applies to future spawns; "
            "per-SEV ignores unchanged; "
            f"use `rebase all` to apply to running PRs)"
        )

    def _do_fix_cap(self, rest: list[str]) -> str:
        current = getattr(self, "max_fix_commits", MAX_FIX_COMMITS)
        if not rest:
            state = "off" if current == 0 else str(current)
            return f"fix-cap is {state}"
        arg = rest[0].lower()
        if arg in ("off", "none", "unlimited", "disable", "disabled"):
            new = 0
        elif arg in ("default", "on"):
            new = MAX_FIX_COMMITS
        elif arg == "toggle":
            new = MAX_FIX_COMMITS if current == 0 else 0
        else:
            try:
                new = int(arg)
            except ValueError:
                return "usage: fix-cap [N|off|default|toggle]"
            if new < 0:
                return "usage: fix-cap [N|off|default|toggle]"
        self.max_fix_commits = new
        state = "off" if new == 0 else str(new)
        return (
            f"fix-cap {state} (applies to future spawns; "
            f"use `restart <pr>` or `restart all` to apply to running PRs)"
        )

    def _do_cancel_job(
        self,
        job: JobKey | int,
        *,
        keep_resumable: bool = False,
    ) -> str:
        job = _coerce_job(job)
        label = _job_label(job)
        if job in getattr(self, "_cleanup_jobs", set()):
            return f"[{label}] cleanup in progress"
        entry = self.procs.get(job)
        if entry is None:
            return f"[{label}] unknown"
        _terminate_group(entry[0])
        if not keep_resumable:
            self._unresumable_jobs.add(job)
            _remove_mux_job(job)
        return f"[{label}] terminated"

    def _do_cancel(self, pr: int) -> str:
        return self._do_cancel_job(_pr_job(pr))

    def _do_remove_job(self, job: JobKey | int) -> str:
        job = _coerce_job(job)
        label = _job_label(job)
        if job in getattr(self, "_cleanup_jobs", set()):
            return f"[{label}] cleanup in progress"
        entry = self.procs.get(job)
        if entry is None:
            if getattr(self, "_parked_jobs", {}).pop(job, None) is not None:
                self._prune_job(job)
                # Explicit removal from mux is the only thing that strips the
                # ``mergedog`` label -- completion and crashes leave it on.
                self._set_mergedog_label(job, present=False)
                return f"[{label}] removed"
            return f"[{label}] unknown"
        if entry[0].poll() is None:
            _terminate_group(entry[0])
        self._prune_job(job)
        # Explicit removal from mux is the only thing that strips the
        # ``mergedog`` label -- completion and crashes leave it on.
        self._set_mergedog_label(job, present=False)
        return f"[{label}] removed"

    def _do_remove(self, pr: int) -> str:
        return self._do_remove_job(_pr_job(pr))

    def _do_cleanup(self, rest: list[str]) -> str:
        if rest and rest != ["all"]:
            return "usage: cleanup [all]"
        jobs = self._completed_jobs()
        if not jobs:
            if getattr(self, "_cleanup_jobs", set()):
                return "cleanup already in progress"
            return "no completed jobs to cleanup"
        self._begin_cleanup_jobs(jobs)
        self._cleanup_completed_jobs(jobs)
        return f"cleaning up {len(jobs)} completed job(s)"

    def _begin_cleanup_jobs(self, jobs: list[JobKey]) -> None:
        cleanup_jobs = getattr(self, "_cleanup_jobs", None)
        if cleanup_jobs is None:
            cleanup_jobs = set()
            self._cleanup_jobs = cleanup_jobs
        cleanup_status = getattr(self, "_cleanup_status", None)
        if cleanup_status is None:
            cleanup_status = {}
            self._cleanup_status = cleanup_status
        total = len(jobs)
        for index, job in enumerate(jobs, start=1):
            cleanup_jobs.add(job)
            cleanup_status[job] = f"cleanup: queued ({index}/{total})"

    @work(thread=True, group="cleanup")
    def _cleanup_completed_jobs(self, jobs: list[JobKey]) -> None:
        total = len(jobs)
        for index, job in enumerate(jobs, start=1):
            label = _job_label(job)
            self.call_from_thread(
                self._set_cleanup_status,
                job,
                f"cleanup: removing worktree for [{label}] ({index}/{total})",
            )
            _cleanup_job_files(job)
            self.call_from_thread(self._finish_cleanup_job, job)

    def _set_cleanup_status(self, job: JobKey, status: str) -> None:
        cleanup_status = getattr(self, "_cleanup_status", None)
        if cleanup_status is None:
            cleanup_status = {}
            self._cleanup_status = cleanup_status
        cleanup_status[job] = status
        self._refresh()

    def _finish_cleanup_job(self, job: JobKey) -> None:
        self._forget_job_record(job)
        getattr(self, "_cleanup_jobs", set()).discard(job)
        getattr(self, "_cleanup_status", {}).pop(job, None)
        self._refresh()

    def _do_mark_spurious(self, pr: int) -> str:
        checks = github.get_pr_checks_all(pr)
        names = sorted(
            {
                c.get("name")
                for c in checks
                if c.get("bucket") in {"fail", "cancel"} and c.get("name")
            }
        )
        if not names:
            return f"[{pr}] no current failed/cancelled checks to mark spurious"

        trust = TrustDB.load_or_create(pr)
        before = set(trust.spurious_check_names)
        merged = sorted(before | set(names))
        trust.spurious_check_names = merged
        trust.save()

        if _pr_job(pr) in self.procs:
            self._do_cancel_job(_pr_job(pr), keep_resumable=True)
            started = self._do_add(pr, [])
            suffix = f"; {started}"
        else:
            suffix = ""

        added = len(set(names) - before)
        return (
            f"[{pr}] marked {len(names)} current failed/cancelled check"
            f"{'' if len(names) == 1 else 's'} spurious "
            f"({added} new){suffix}"
        )

    def _refresh(self) -> None:
        table = self.query_one(DataTable)
        table.clear()
        self._refresh_cleanup_hint()
        parent_hints: dict[JobKey, JobKey] = {}
        for job, (_, _, log_path) in self.procs.items():
            parent_pr = _read_stack_parent_pr_from_log(log_path)
            if parent_pr is not None:
                parent_hints[job] = _pr_job(parent_pr)
        jobs, depths = _stack_display_layout(list(self.procs), parent_hints)
        cleanup_jobs = getattr(self, "_cleanup_jobs", set())
        cleanup_status = getattr(self, "_cleanup_status", {})
        for job in jobs:
            _, pr = job
            p, _, log_path = self.procs[job]
            rc = p.poll()
            last = _last_log_line(log_path)
            if job in cleanup_jobs:
                structured = None
                stale = False
                self._pr_status.pop(job, None)
                phase = PHASE_NO_ACTION
                status = cleanup_status.get(job, "cleanup: queued")
            else:
                structured = read_status(pr)
                started_at = getattr(self, "_job_started_at", {}).get(job)
                stale = _status_is_stale(structured, started_at=started_at, rc=rc)
                if structured is None:
                    self._pr_status.pop(job, None)
                else:
                    self._pr_status[job] = structured
                cla_blocked = (
                    not stale and _status_has_cla_blocker(pr, structured)
                )
                phase = _phase_label(
                    structured, rc=rc, stale=stale, cla_blocked=cla_blocked
                )
                status = _status_message(
                    structured,
                    rc=rc,
                    last_log=last,
                    crash_summary=_crash_summary_from_log(log_path)
                    if stale and rc is not None
                    else "",
                    stale=stale,
                )
                if cla_blocked:
                    status = _cla_blocked_status_message(status)
            title = self._pr_titles.get(job, "")
            if not title:
                title = _read_pr_title(pr)
                if title:
                    self._pr_titles[job] = title
            # OSC-8 hyperlink so cmd/ctrl-click on the PR opens the PR;
            # the worktree path is omitted from the table entirely now,
            # but it's still ``~/.mergedog/worktrees/<pr>/`` if you need
            # it from the shell.
            indent = "  " * depths.get(job, 0)
            pr_cell = Text(
                indent + _job_label(job),
                style=f"link https://github.com/{REPO_SLUG}/pull/{pr}",
            )
            table.add_row(pr_cell, Text(_truncate_title(title)), phase, status)
        for job in sorted(getattr(self, "_parked_jobs", {})):
            if job in self.procs:
                continue
            _, pr = job
            title = self._pr_titles.get(job, "") or _read_pr_title(pr)
            if title:
                self._pr_titles[job] = title
            pr_cell = Text(
                _job_label(job),
                style=f"link https://github.com/{REPO_SLUG}/pull/{pr}",
            )
            structured = read_status(pr)
            detail = (structured or {}).get("message")
            status = (
                detail
                if isinstance(detail, str) and detail
                else "halted before mux restart"
            )
            table.add_row(
                pr_cell,
                Text(_truncate_title(title)),
                PHASE_HALTED,
                f"{status} (not resumed; `restart {pr}` to re-run)",
            )

    def _prune_job(self, job: JobKey | int) -> None:
        """Forget a shepherd and clean up its on-disk state.

        The log file is kept so the operator can audit why the shepherd quit.
        """
        job = _coerce_job(job)
        _cleanup_job_files(job)
        self._forget_job_record(job)
        getattr(self, "_cleanup_jobs", set()).discard(job)
        getattr(self, "_cleanup_status", {}).pop(job, None)

    def _forget_job_record(self, job: JobKey | int) -> None:
        job = _coerce_job(job)
        entry = self.procs.pop(job, None)
        if entry is not None:
            try:
                entry[1].close()  # type: ignore[attr-defined]
            except Exception:
                pass
        _remove_mux_job(job)
        self._pr_titles.pop(job, None)
        getattr(self, "_job_started_at", {}).pop(job, None)
        getattr(self, "_unresumable_jobs", set()).discard(job)
        getattr(self, "_parked_jobs", {}).pop(job, None)

    def _prune_pr(self, pr: int) -> None:
        self._prune_job(_pr_job(pr))

    # ------------------------------------------------------------------
    # Command dispatch (shared by TUI input and IPC server)
    # ------------------------------------------------------------------

    def _format_help(self) -> str:
        lines = [
            (
                f"phases: {PHASE_NO_ACTION} no action; "
                f"{PHASE_YOUR_EASY_ACTION} you can merge; "
                f"{PHASE_YOUR_REVIEW_ACTION} review/approve first; "
                f"{PHASE_EXTERNAL_ACTION} waiting on someone else; "
                f"{PHASE_HALTED} halted/crashed"
            ),
            (
                "commands: add <pr> | restart <pr|all|dead> | "
                "rebase <pr|all> | reassess <pr> | "
                "fix <pr> <trusted request>"
            ),
            (
                "commands: mark-spurious <pr> | cancel <pr> | cleanup | "
                "remove <pr> | log <pr> | status | migrate | quit"
            ),
            (
                "commands: ignore-sev [on|off] | ignore-sev add <issue> | "
                "ignore-sev remove <issue> | ignore-sev clear"
            ),
        ]
        return "\n".join(lines)

    def _dispatch_command(self, line: str) -> str:
        """Parse and execute a mux command.  Returns a response string."""
        if not line.strip():
            return ""
        try:
            args = shlex.split(line)
        except ValueError as e:
            return f"parse error: {e}"
        cmd, rest = args[0], args[1:]
        if cmd.isdigit() or "/pull/" in cmd:
            cmd, rest = "add", args
        try:
            if cmd in ("add", "a"):
                if not rest:
                    return "usage: add <pr> [flags]"
                return self._do_add(_parse_pr(rest[0]), rest[1:])
            elif cmd == "rebase":
                if not rest:
                    return "usage: rebase <pr> | rebase all"
                if rest[0] == "all":
                    n = len(
                        [
                            job
                            for job in self._all_jobs()
                            if job not in getattr(self, "_cleanup_jobs", set())
                        ]
                    )
                    if not n:
                        return "no PRs to rebase"
                    self._do_rebase_all()
                    return f"rebasing {n} job(s)"
                pr = _parse_pr(rest[0])
                self._do_cancel_job(_pr_job(pr), keep_resumable=True)
                return self._do_add(pr, ["--rebase", *rest[1:]])
            elif cmd in ("restart", "r"):
                if not rest:
                    return "usage: restart <pr> | restart all | restart dead"
                if rest[0] == "all":
                    n = len(
                        [
                            job
                            for job in self._all_jobs()
                            if job not in getattr(self, "_cleanup_jobs", set())
                        ]
                    )
                    if not n:
                        return "no PRs to restart"
                    self._do_restart_all(rest[1:])
                    return f"restarting {n} job(s)"
                if rest[0] == "dead":
                    jobs = self._dead_jobs()
                    if not jobs:
                        return "no dead PRs to restart"
                    self._do_restart_dead(jobs, rest[1:])
                    return f"restarting dead {len(jobs)} job(s)"
                pr = _parse_pr(rest[0])
                self._do_cancel_job(_pr_job(pr), keep_resumable=True)
                return self._do_add(pr, rest[1:])
            elif cmd == "reassess":
                if not rest:
                    return "usage: reassess <pr>"
                pr = _parse_pr(rest[0])
                self._do_cancel_job(_pr_job(pr), keep_resumable=True)
                return self._do_add(pr, ["--reassess", *rest[1:]])
            elif cmd == "fix":
                if len(rest) < 2:
                    return "usage: fix <pr> <trusted request>"
                pr = _parse_pr(rest[0])
                fix_args = self._operator_fix_args(rest[1:])
                if isinstance(fix_args, str):
                    return fix_args
                self._do_cancel_job(_pr_job(pr), keep_resumable=True)
                return self._do_add(pr, fix_args)
            elif cmd in ("mark-spurious", "spurious", "ignore-failures"):
                if not rest:
                    return "usage: mark-spurious <pr>"
                return self._do_mark_spurious(_parse_pr(rest[0]))
            elif cmd in ("cancel", "c", "kill"):
                if not rest:
                    return "usage: cancel <pr>"
                return self._do_cancel(_parse_pr(rest[0]))
            elif cmd in ("cleanup", "clean"):
                return self._do_cleanup(rest)
            elif cmd in ("remove", "rm", "forget"):
                if not rest:
                    return "usage: remove <pr>"
                return self._do_remove(_parse_pr(rest[0]))
            elif cmd == "log":
                if not rest:
                    return "usage: log <pr>"
                pr = _parse_pr(rest[0])
                entry = self.procs.get(_pr_job(pr))
                if entry is not None:
                    return str(entry[2])
                if _pr_job(pr) in getattr(self, "_parked_jobs", {}):
                    return str(_job_log_file(_pr_job(pr)))
                return f"[{pr}] unknown"
            elif cmd == "migrate":
                return self._format_migrate()
            elif cmd in ("ignore-sev", "ignore_sev"):
                return self._do_ignore_sev(rest)
            elif cmd in ("fix-cap", "fix_cap"):
                return self._do_fix_cap(rest)
            elif cmd == "status":
                return self._format_status()
            elif cmd in ("help", "h", "?"):
                return self._format_help()
            elif cmd in ("quit", "q", "exit"):
                return "use the TUI to quit the mux"
            else:
                return f"unknown command: {cmd!r}"
        except Exception as e:
            return f"error: {e}"

    def _format_status(self) -> str:
        """JSON status of all tracked jobs (consumed by MCP server)."""
        rows = []
        cleanup_jobs = getattr(self, "_cleanup_jobs", set())
        cleanup_status = getattr(self, "_cleanup_status", {})
        for job in sorted(self.procs):
            kind, pr = job
            p, _, log_path = self.procs[job]
            rc = p.poll()
            if job in cleanup_jobs:
                state = "cleaning"
            elif rc is None:
                state = "running"
            elif rc == 0:
                state = "exited_ok"
            elif rc == EXIT_PR_NOT_ACTIONABLE:
                state = "completed"
            else:
                state = "exited_error"
            last = _last_log_line(log_path)
            title = self._pr_titles.get(job, "") or _read_pr_title(pr)
            if job in cleanup_jobs:
                structured = None
                stale = False
                phase = PHASE_NO_ACTION
                status_message = cleanup_status.get(job, "cleanup: queued")
            else:
                structured = read_status(pr)
                started_at = getattr(self, "_job_started_at", {}).get(job)
                stale = _status_is_stale(
                    structured, started_at=started_at, rc=rc
                )
                cla_blocked = (
                    not stale and _status_has_cla_blocker(pr, structured)
                )
                phase = _phase_label(
                    structured, rc=rc, stale=stale, cla_blocked=cla_blocked
                )
                status_message = _status_message(
                    structured,
                    rc=rc,
                    last_log=last,
                    crash_summary=_crash_summary_from_log(log_path)
                    if stale and rc is not None
                    else "",
                    stale=stale,
                )
                if cla_blocked:
                    status_message = _cla_blocked_status_message(
                        status_message
                    )
            rows.append({
                "kind": kind,
                "pr": pr,
                "title": title,
                "state": state,
                "phase": phase,
                "status": status_message,
                "last_log": last,
                "shepherd_status_stale": stale,
                "shepherd_status": structured,
            })
        for job in sorted(getattr(self, "_parked_jobs", {})):
            if job in self.procs:
                continue
            kind, pr = job
            log_path = _job_log_file(job)
            rows.append({
                "kind": kind,
                "pr": pr,
                "title": self._pr_titles.get(job, "") or _read_pr_title(pr),
                "state": "parked",
                "phase": PHASE_HALTED,
                "status": (
                    "halted before mux restart; not resumed "
                    f"(`restart {pr}` to re-run)"
                ),
                "last_log": _last_log_line(log_path),
                "shepherd_status_stale": True,
                "shepherd_status": read_status(pr),
            })
        return json.dumps(rows, indent=2)

    def _format_migrate(self) -> str:
        jobs = sorted(self.procs)
        if not jobs:
            return "no PRs tracked"
        prs = [pr for _, pr in jobs]
        state_files = [
            str(state_file(pr)) for _, pr in jobs if state_file(pr).exists()
        ]
        repo_arg = f"--repo {shlex.quote(self.repo_slug)} "
        pr_args = " ".join(str(pr) for pr in prs)
        lines = [
            "# Copy state files to the new server:",
            f"scp {shlex.join(state_files)} NEW_HOST:~/.mergedog/state/",
            "# Then start the mux there:",
            f"python -m mergedog.mux {repo_arg}{pr_args}",
        ]
        return "\n".join(lines)

    def on_input_submitted(self, message: Input.Submitted) -> None:
        line = message.value.strip()
        message.input.value = ""
        if line:
            message.input.record(line)
        if not line:
            return
        # TUI-only: quit and migrate have side effects beyond a response
        try:
            first = shlex.split(line)[0]
        except (ValueError, IndexError):
            first = ""
        if first in ("quit", "q", "exit"):
            self.exit()
            return
        if first == "migrate":
            text = self._format_migrate()
            if text == "no PRs tracked":
                self.notify(text, severity="warning")
            else:
                self._migrate_output = text
                self.exit()
            return
        result = self._dispatch_command(line)
        if result:
            self.notify(result)
        self._refresh()

    # ------------------------------------------------------------------
    # IPC server (Unix socket, same commands as TUI input)
    # ------------------------------------------------------------------

    @work(thread=False)
    async def _start_ipc_server(self) -> None:
        sock_path = str(MUX_SOCKET)
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass
        server = await asyncio.start_unix_server(
            self._handle_ipc_connection, path=sock_path,
        )
        self._ipc_server = server
        async with server:
            await server.serve_forever()

    async def _handle_ipc_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not raw:
                return
            command = raw.decode().strip()
            result = self._dispatch_command(command)
            payload = json.dumps({"ok": True, "message": result})
            writer.write(payload.encode() + b"\n")
            await writer.drain()
        except Exception as e:
            try:
                payload = json.dumps({"ok": False, "message": str(e)})
                writer.write(payload.encode() + b"\n")
                await writer.drain()
            except Exception:
                pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    def on_unmount(self) -> None:
        unresumable_jobs = getattr(self, "_unresumable_jobs", set())
        records: dict[JobKey, dict] = {
            job: {"rc": rc}
            for job, rc in getattr(self, "_parked_jobs", {}).items()
        }
        for job, (p, _f, _) in self.procs.items():
            rc = p.poll()
            if job in unresumable_jobs or rc in (0, EXIT_PR_NOT_ACTIONABLE):
                continue
            # Live jobs resume on the next mux start; jobs that halted
            # carry their exit code so the next mux parks them instead
            # of re-running (and re-notifying) a deterministic halt.
            records[job] = {} if rc is None else {"rc": rc}
        _write_mux_job_records(records)
        if self._ipc_server is not None:
            self._ipc_server.close()
        if self._lock_fd >= 0:
            release_lock(self._lock_fd)
        for _job, (p, _f, _) in self.procs.items():
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for _job, (p, f, _) in self.procs.items():
            _terminate_group(p, grace=2.0)
            try:
                f.close()  # type: ignore[attr-defined]
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="mergedog.mux",
        description="Supervise multiple mergedog shepherds in one TUI process.",
    )
    parser.add_argument(
        "prs",
        nargs="*",
        help="PR numbers or URLs to start shepherding immediately.",
    )
    parser.add_argument(
        "--resume-known",
        action="store_true",
        default=None,
        help=(
            "Start a shepherd for every job in the mux-tracked list "
            f"({MUX_JOBS_FILE}, falling back to {MUX_PRS_FILE}). This is "
            "the default when no PRs are provided."
        ),
    )
    parser.add_argument(
        "--no-resume-known",
        action="store_false",
        dest="resume_known",
        help=(
            "Start only the PRs provided on the command line, or no jobs if "
            "none are provided."
        ),
    )
    parser.add_argument(
        "--ignore-sev",
        action="store_true",
        help=(
            "Mux-wide default: pass --ignore-sev to every spawned "
            "shepherd so they don't park on open ``ci: sev`` issues. "
            "Toggle at runtime with the ``ignore-sev on|off`` command."
        ),
    )
    parser.add_argument(
        "--max-fix-commits",
        type=int,
        default=MAX_FIX_COMMITS,
        help=(
            "Mux-wide default: pass --max-fix-commits to every spawned "
            "shepherd. Defaults to 5; use 0 to disable the cap. Toggle at "
            "runtime with the ``fix-cap N|off|default`` command."
        ),
    )
    parser.add_argument(
        "--root",
        metavar="DIR",
        help=(
            "Override the on-disk root (default: ``~/.mergedog`` or the "
            "``MERGEDOG_ROOT`` env var). Use this to run a second mux "
            "against a disjoint set of PRs -- a separate clone, "
            "worktrees, state, logs, and trust DB -- so destructive "
            "commands like ``rebase all`` don't touch the default "
            "install. The chosen root is exported to ``MERGEDOG_ROOT`` "
            "before any spawn, so every shepherd inherits it."
        ),
    )
    parser.add_argument(
        "--repo",
        metavar="OWNER/NAME",
        default=REPO_SLUG,
        help=(
            "GitHub repository to shepherd (default: MERGEDOG_REPO, "
            "MERGEDOG_REPO_SLUG, or pytorch/pytorch). Passed to every "
            "spawned shepherd."
        ),
    )
    parser.add_argument(
        "--gchat-to",
        metavar="USER",
        help=(
            "Send a Google Chat DM to USER (unixname) whenever a "
            "shepherd HALTs and needs human intervention. Passed to "
            "every spawned shepherd. Requires the ``meta`` CLI; "
            "silently skipped if ``meta`` is not installed."
        ),
    )
    args = parser.parse_args()
    if args.max_fix_commits < 0:
        parser.error("--max-fix-commits must be >= 0")

    ensure_dirs()

    try:
        lock_fd = acquire_lock()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    resume_known = args.resume_known
    if resume_known is None:
        resume_known = not args.prs
    initial, parked, skipped = _resolve_initial_jobs(
        args.prs,
        resume_known=resume_known,
    )
    for raw, e in skipped:
        print(f"skipping {raw!r}: {e}", file=sys.stderr)

    app = MuxApp(
        initial,
        parked=parked,
        ignore_sev=args.ignore_sev,
        max_fix_commits=args.max_fix_commits,
        gchat_to=args.gchat_to,
        repo_slug=args.repo,
        lock_fd=lock_fd,
    )
    app.run(mouse=False)
    if hasattr(app, "_migrate_output"):
        print(app._migrate_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
