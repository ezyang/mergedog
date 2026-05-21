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
    rebase <pr>                       shorthand for ``add <pr> --rebase``
    rebase all                        re-run every session job with --rebase
    mark-spurious <pr>                mark current failed/cancelled checks
                                      spurious and restart the shepherd
    cancel <pr>                       SIGTERM a shepherd (keeps state)
    cleanup | clean                   forget successful completed shepherds
    remove <pr>                       SIGTERM and forget (wipes worktree)
    log <pr>                          show the path to its log file
    ignore-sev [on|off]               toggle (or show) the mux-wide
                                      ``--ignore-sev`` default applied to
                                      every shepherd spawn
    mergedog-label [on|off]           toggle (or show) the mux-wide
                                      ``--manage-mergedog-label`` default
                                      applied to every shepherd spawn
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
from pathlib import Path

from mergedog.bootstrap import promote_early_env

promote_early_env(sys.argv[1:])

from rich.text import Text  # noqa: E402
from textual import work  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.suggester import SuggestFromList  # noqa: E402
from textual.widgets import DataTable, Input  # noqa: E402


COMMAND_SUGGESTIONS = [
    "add ",
    "cancel ",
    "cleanup",
    "cleanup all",
    "clean",
    "clean all",
    "fix ",
    "fix-cap ",
    "ignore-sev ",
    "log ",
    "mark-spurious ",
    "mergedog-label ",
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
from mergedog.ipc import acquire_lock, release_lock  # noqa: E402
from mergedog.paths import (  # noqa: E402
    MUX_JOBS_FILE,
    MUX_PRS_FILE,
    MUX_SOCKET,
    REPO_SLUG,
    ROOT,
    context_file,
    ensure_dirs,
    state_file,
    status_file,
    worktree_dir,
)
from mergedog.shepherd import EXIT_PR_NOT_ACTIONABLE, MAX_FIX_COMMITS  # noqa: E402
from mergedog.status import read_status  # noqa: E402
from mergedog.state import TrustDB  # noqa: E402

LOG_DIR = ROOT / "logs"

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


def _job_log_name(job: JobKey | int) -> str:
    _, pr = _coerce_job(job)
    return f"{pr}.log"


def _read_pr_title(pr: int) -> str:
    """Best-effort PR title from the worktree's HEAD subject.

    ``--invert-grep`` skips ``[MERGEDOG]`` commits so an in-flight
    merge-main doesn't replace the actual contributor title with the
    merge commit's subject. Returns "" if the worktree doesn't exist
    yet (shepherd just spawned) or git fails -- the caller will retry.
    """
    wt = worktree_dir(pr)
    if not wt.exists():
        return ""
    try:
        proc = subprocess.run(
            [
                "git", "log", "-1", "--pretty=%s",
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
    MUX_PRS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = MUX_PRS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sorted(set(prs))))
    os.replace(tmp, MUX_PRS_FILE)


def _read_mux_jobs() -> list[JobKey]:
    """Curated mux jobs.

    ``mux-prs.json`` remains as the backwards-compatible regular-PR list
    for older tools. Newer mux instances persist jobs here.
    """
    if not MUX_JOBS_FILE.exists():
        return [_pr_job(pr) for pr in _read_mux_prs()]
    try:
        data = json.loads(MUX_JOBS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    out: set[JobKey] = set()
    for item in data:
        try:
            if isinstance(item, int):
                out.add(_pr_job(item))
            elif isinstance(item, str):
                out.add(_pr_job(_parse_pr(item)))
            elif isinstance(item, dict):
                kind = str(item.get("kind", PR_JOB))
                pr = int(item["pr"])
                if kind == PR_JOB:
                    out.add(_pr_job(pr))
        except (TypeError, ValueError, KeyError):
            continue
    return sorted(out)


def _write_mux_jobs(jobs: list[JobKey]) -> None:
    jobs = sorted(set(jobs))
    MUX_JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = MUX_JOBS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps([{"kind": kind, "pr": pr} for kind, pr in jobs]))
    os.replace(tmp, MUX_JOBS_FILE)
    _write_mux_prs([pr for _, pr in jobs])


def _add_mux_job(job: JobKey) -> None:
    jobs = set(_read_mux_jobs())
    if job in jobs:
        return
    jobs.add(job)
    _write_mux_jobs(sorted(jobs))


def _remove_mux_job(job: JobKey) -> None:
    jobs = set(_read_mux_jobs())
    if job not in jobs:
        return
    jobs.discard(job)
    _write_mux_jobs(sorted(jobs))


def _add_mux_pr(pr: int) -> None:
    _add_mux_job(_pr_job(pr))


def _remove_mux_pr(pr: int) -> None:
    _remove_mux_job(_pr_job(pr))


def _resolve_initial_jobs(
    raw_prs: list[str],
    *,
    resume_known: bool,
) -> tuple[list[JobKey], list[tuple[str, argparse.ArgumentTypeError]]]:
    initial: list[JobKey] = []
    if resume_known:
        initial.extend(_read_mux_jobs())

    skipped: list[tuple[str, argparse.ArgumentTypeError]] = []
    for raw in raw_prs:
        try:
            initial.append(_pr_job(_parse_pr(raw)))
        except argparse.ArgumentTypeError as e:
            skipped.append((raw, e))

    seen: set[JobKey] = set()
    initial = [job for job in initial if not (job in seen or seen.add(job))]
    return initial, skipped


def _spawn(
    job: JobKey | int,
    extra: list[str],
    *,
    spawn_pr: int | None = None,
) -> tuple[subprocess.Popen, object, Path]:
    job = _coerce_job(job)
    _, pr = job
    arg_pr = spawn_pr if spawn_pr is not None else pr
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / _job_log_name(job)
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
    Input { dock: bottom; }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(
        self,
        initial: list[JobKey | int],
        *,
        ignore_sev: bool = False,
        manage_mergedog_label: bool = False,
        max_fix_commits: int = MAX_FIX_COMMITS,
        gchat_to: str | None = None,
        repo_slug: str = REPO_SLUG,
        lock_fd: int = -1,
    ) -> None:
        super().__init__()
        self.procs: dict[JobKey, tuple[subprocess.Popen, object, Path]] = {}
        self._pr_titles: dict[JobKey, str] = {}
        self._pr_status: dict[JobKey, dict] = {}
        self._initial = [_coerce_job(job) for job in initial]
        self.ignore_sev = ignore_sev
        self.manage_mergedog_label = manage_mergedog_label
        self.max_fix_commits = max_fix_commits
        self.gchat_to = gchat_to
        self.repo_slug = repo_slug
        self._lock_fd = lock_fd
        self._ipc_server: asyncio.AbstractServer | None = None
        self._unresumable_jobs: set[JobKey] = set()

    def compose(self) -> ComposeResult:
        yield DataTable()
        yield HistoryInput(
            placeholder=(
                "<pr> | add <pr> | restart <pr|all|dead> | rebase <pr|all> | reassess <pr> | "
                "fix <pr> | mark-spurious <pr> | cancel <pr> | cleanup | clean | "
                "remove <pr> | log <pr> | fix-cap | mergedog-label | "
                "migrate | quit"
            ),
            suggester=SuggestFromList(COMMAND_SUGGESTIONS),
        )

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("PR", "Title", "", "Last")
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
        if (
            self.manage_mergedog_label
            and "--manage-mergedog-label" not in out
        ):
            out = ["--manage-mergedog-label", *out]
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
        if job in self.procs and self.procs[job][0].poll() is None:
            return f"[{label}] already running"
        try:
            self.procs[job] = _spawn(
                job, self._shepherd_args(extra), spawn_pr=spawn_pr
            )
        except Exception as e:
            return f"[{label}] spawn failed: {e}"
        self._pr_titles.pop(job, None)
        self._unresumable_jobs.discard(job)
        if alias_job is not None and alias_job != job:
            _remove_mux_job(alias_job)
        _add_mux_job(job)
        return f"[{label}] started"

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
        return sorted(
            job
            for job, (p, _, _) in self.procs.items()
            if p.poll() not in (None, 0, EXIT_PR_NOT_ACTIONABLE)
        )

    def _completed_jobs(self) -> list[JobKey]:
        return sorted(
            job
            for job, (p, _, _) in self.procs.items()
            if p.poll() in (0, EXIT_PR_NOT_ACTIONABLE)
        )

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
        jobs = [_coerce_job(job) for job in jobs if _coerce_job(job) in self.procs]
        if not jobs:
            self.notify(empty, severity="warning")
            return
        # Signal every shepherd up front so they all wind down in
        # parallel. Without this, each ``_terminate_group`` below would
        # block up to ``grace`` seconds *per PR* in series before the
        # next SIGTERM was even sent.
        for job in jobs:
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
            _terminate_group(self.procs[job][0])
        shepherd_args = self._shepherd_args(extra)
        for job in jobs:
            try:
                self.procs[job] = _spawn(job, shepherd_args)
            except Exception as e:
                self.notify(f"[{_job_label(job)}] failed: {e}", severity="error")
            self._pr_titles.pop(job, None)
        self.notify(f"{action} {len(jobs)} job(s)")

    @work(thread=True, exclusive=True, group="restart-all")
    def _do_restart_all(self, extra: list[str] | None = None) -> None:
        self._restart_jobs(
            sorted(self.procs),
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
            sorted(self.procs),
            ["--rebase"],
            "rebasing",
            "no PRs to rebase",
        )

    def _do_ignore_sev(self, rest: list[str]) -> str:
        if not rest:
            state = "on" if self.ignore_sev else "off"
            return f"ignore-sev is {state}"
        arg = rest[0].lower()
        if arg in ("on", "true", "1", "yes"):
            new = True
        elif arg in ("off", "false", "0", "no"):
            new = False
        elif arg == "toggle":
            new = not self.ignore_sev
        else:
            return "usage: ignore-sev [on|off|toggle]"
        self.ignore_sev = new
        state = "on" if new else "off"
        return (
            f"ignore-sev {state} (applies to future spawns; "
            f"use `rebase all` to apply to running PRs)"
        )

    def _do_mergedog_label(self, rest: list[str]) -> str:
        if not rest:
            state = "on" if self.manage_mergedog_label else "off"
            return f"mergedog-label is {state}"
        arg = rest[0].lower()
        if arg in ("on", "true", "1", "yes"):
            new = True
        elif arg in ("off", "false", "0", "no"):
            new = False
        elif arg == "toggle":
            new = not self.manage_mergedog_label
        else:
            return "usage: mergedog-label [on|off|toggle]"
        self.manage_mergedog_label = new
        state = "on" if new else "off"
        return (
            f"mergedog-label {state} (applies to future spawns; "
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
        entry = self.procs.get(job)
        if entry is None:
            return f"[{label}] unknown"
        if entry[0].poll() is None:
            _terminate_group(entry[0])
        self._prune_job(job)
        return f"[{label}] removed"

    def _do_remove(self, pr: int) -> str:
        return self._do_remove_job(_pr_job(pr))

    def _do_cleanup(self, rest: list[str]) -> str:
        if rest and rest != ["all"]:
            return "usage: cleanup [all]"
        jobs = self._completed_jobs()
        if not jobs:
            return "no completed jobs to cleanup"
        for job in jobs:
            self._prune_job(job)
        return f"cleaned up {len(jobs)} completed job(s)"

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
        parent_hints: dict[JobKey, JobKey] = {}
        for job, (_, _, log_path) in self.procs.items():
            parent_pr = _read_stack_parent_pr_from_log(log_path)
            if parent_pr is not None:
                parent_hints[job] = _pr_job(parent_pr)
        jobs, depths = _stack_display_layout(list(self.procs), parent_hints)
        for job in jobs:
            _, pr = job
            p, _, log_path = self.procs[job]
            rc = p.poll()
            if rc is None:
                state = "🟢"
            elif rc in (0, EXIT_PR_NOT_ACTIONABLE):
                state = ""
            else:
                state = "🔴"
            last = _last_log_line(log_path)
            structured = read_status(pr)
            if structured is None:
                self._pr_status.pop(job, None)
            else:
                self._pr_status[job] = structured
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
            table.add_row(pr_cell, Text(_truncate_title(title)), state, last)

    def _prune_job(self, job: JobKey | int) -> None:
        """Forget a shepherd and clean up its on-disk state.

        The log file is kept so the operator can audit why the shepherd quit.
        """
        job = _coerce_job(job)
        _, pr = job
        entry = self.procs.pop(job, None)
        if entry is not None:
            try:
                entry[1].close()  # type: ignore[attr-defined]
            except Exception:
                pass
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
        _remove_mux_job(job)
        self._pr_titles.pop(job, None)
        self._unresumable_jobs.discard(job)

    def _prune_pr(self, pr: int) -> None:
        self._prune_job(_pr_job(pr))

    # ------------------------------------------------------------------
    # Command dispatch (shared by TUI input and IPC server)
    # ------------------------------------------------------------------

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
                    n = len(self.procs)
                    if not n:
                        return "no PRs to rebase"
                    self._do_rebase_all()
                    return f"rebasing {n} job(s)"
                return self._do_add(_parse_pr(rest[0]), ["--rebase", *rest[1:]])
            elif cmd in ("restart", "r"):
                if not rest:
                    return "usage: restart <pr> | restart all | restart dead"
                if rest[0] == "all":
                    n = len(self.procs)
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
                return f"[{pr}] unknown"
            elif cmd == "migrate":
                return self._format_migrate()
            elif cmd in ("ignore-sev", "ignore_sev"):
                return self._do_ignore_sev(rest)
            elif cmd in ("mergedog-label", "mergedog_label"):
                return self._do_mergedog_label(rest)
            elif cmd in ("fix-cap", "fix_cap"):
                return self._do_fix_cap(rest)
            elif cmd == "status":
                return self._format_status()
            elif cmd in ("quit", "q", "exit"):
                return "use the TUI to quit the mux"
            else:
                return f"unknown command: {cmd!r}"
        except Exception as e:
            return f"error: {e}"

    def _format_status(self) -> str:
        """JSON status of all tracked jobs (consumed by MCP server)."""
        rows = []
        for job in sorted(self.procs):
            kind, pr = job
            p, _, log_path = self.procs[job]
            rc = p.poll()
            if rc is None:
                state = "running"
            elif rc == 0:
                state = "exited_ok"
            elif rc == EXIT_PR_NOT_ACTIONABLE:
                state = "completed"
            else:
                state = "exited_error"
            last = _last_log_line(log_path)
            title = self._pr_titles.get(job, "") or _read_pr_title(pr)
            structured = read_status(pr)
            rows.append({
                "kind": kind,
                "pr": pr,
                "title": title,
                "state": state,
                "last_log": last,
                "shepherd_status": structured,
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
        _write_mux_jobs(
            sorted(
                job
                for job, (p, _f, _) in self.procs.items()
                if job not in unresumable_jobs
                and p.poll() not in (0, EXIT_PR_NOT_ACTIONABLE)
            )
        )
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
        "--manage-mergedog-label",
        action="store_true",
        help=(
            "Mux-wide default: pass --manage-mergedog-label to every spawned "
            "shepherd so it adds the ``mergedog`` label at startup and "
            "removes it on exit. Toggle at runtime with the "
            "``mergedog-label on|off`` command."
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
    initial, skipped = _resolve_initial_jobs(
        args.prs,
        resume_known=resume_known,
    )
    for raw, e in skipped:
        print(f"skipping {raw!r}: {e}", file=sys.stderr)

    app = MuxApp(
        initial,
        ignore_sev=args.ignore_sev,
        manage_mergedog_label=args.manage_mergedog_label,
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
