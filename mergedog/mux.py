"""Tiny multi-PR supervisor with a Textual TUI.

Top: a DataTable with one row per active shepherd subprocess.
Bottom: an Input field that takes commands.

Run via::

    python -m mergedog.mux [<pr>...] [--resume-known]

Commands typed at the bottom (enter to submit):

    add <pr> [extra mergedog flags]   start a shepherd
    <pr>                              shorthand for ``add <pr>``
    restart <pr>                      kill and re-spawn a shepherd
    rebase <pr>                       shorthand for ``add <pr> --rebase``
    rebase all                        re-run every tracked PR with --rebase
    cancel <pr>                       SIGTERM a shepherd (keeps state)
    remove <pr>                       SIGTERM and forget (wipes worktree)
    log <pr>                          show the path to its log file
    ignore-sev [on|off]               toggle (or show) the mux-wide
                                      ``--ignore-sev`` default applied to
                                      every shepherd spawn
    mergedog-label [on|off]           toggle (or show) the mux-wide
                                      ``--manage-mergedog-label`` default
                                      applied to every shepherd spawn
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
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


def _preparse_root(argv: list[str]) -> None:
    """Promote ``--root`` to ``MERGEDOG_ROOT`` before mergedog imports.

    ``mergedog.paths`` resolves the root at import time, so we seed the
    env var here -- before the ``from mergedog...`` block below -- and
    leave canonical argparse handling to ``main()`` for help text. Env
    inheritance then carries the same root into every spawned shepherd.
    """
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--root" and i + 1 < len(argv):
            os.environ["MERGEDOG_ROOT"] = str(Path(argv[i + 1]).expanduser().resolve())
            return
        if a.startswith("--root="):
            value = a.split("=", 1)[1]
            os.environ["MERGEDOG_ROOT"] = str(Path(value).expanduser().resolve())
            return
        i += 1


_preparse_root(sys.argv[1:])

from rich.text import Text  # noqa: E402
from textual import work  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.widgets import DataTable, Input  # noqa: E402


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

from mergedog import repo as repo_mod  # noqa: E402
from mergedog.cli import _parse_pr  # noqa: E402
from mergedog.ipc import acquire_lock, release_lock  # noqa: E402
from mergedog.paths import (  # noqa: E402
    MUX_PRS_FILE,
    MUX_SOCKET,
    ROOT,
    context_file,
    ensure_dirs,
    state_file,
    worktree_dir,
)
from mergedog.shepherd import EXIT_PR_NOT_ACTIONABLE  # noqa: E402

LOG_DIR = ROOT / "logs"

TITLE_TRUNC = 20


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


def _add_mux_pr(pr: int) -> None:
    prs = set(_read_mux_prs())
    if pr in prs:
        return
    prs.add(pr)
    _write_mux_prs(sorted(prs))


def _remove_mux_pr(pr: int) -> None:
    prs = set(_read_mux_prs())
    if pr not in prs:
        return
    prs.discard(pr)
    _write_mux_prs(sorted(prs))


def _spawn(pr: int, extra: list[str]) -> tuple[subprocess.Popen, object, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{pr}.log"
    f = open(log_path, "a", buffering=1, encoding="utf-8")
    f.write(f"\n=== mergedog start at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    p = subprocess.Popen(
        [sys.executable, "-m", "mergedog", str(pr), *extra],
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
        initial: list[int],
        *,
        ignore_sev: bool = False,
        manage_mergedog_label: bool = False,
        gchat_to: str | None = None,
        lock_fd: int = -1,
    ) -> None:
        super().__init__()
        self.procs: dict[int, tuple[subprocess.Popen, object, Path]] = {}
        self._pr_titles: dict[int, str] = {}
        self._initial = initial
        self.ignore_sev = ignore_sev
        self.manage_mergedog_label = manage_mergedog_label
        self.gchat_to = gchat_to
        self._lock_fd = lock_fd
        self._ipc_server: asyncio.AbstractServer | None = None

    def compose(self) -> ComposeResult:
        yield DataTable()
        yield HistoryInput(
            placeholder=(
                "<pr> | add <pr> | restart <pr> | rebase <pr|all> | reassess <pr> | "
                "cancel <pr> | remove <pr> | log <pr> | mergedog-label | migrate | quit"
            )
        )

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("PR", "Title", "", "Last")
        for pr in self._initial:
            self._do_add(pr, [])
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
        if self.gchat_to and not any(a.startswith("--gchat-to") for a in out):
            out = [f"--gchat-to={self.gchat_to}", *out]
        return out

    def _do_add(self, pr: int, extra: list[str]) -> str:
        if pr in self.procs and self.procs[pr][0].poll() is None:
            return f"[{pr}] already running"
        try:
            self.procs[pr] = _spawn(pr, self._shepherd_args(extra))
        except Exception as e:
            return f"[{pr}] spawn failed: {e}"
        self._pr_titles.pop(pr, None)
        _add_mux_pr(pr)
        return f"[{pr}] started"

    @work(thread=True, exclusive=True, group="rebase-all")
    def _do_rebase_all(self) -> None:
        # Runs in a worker thread so the input bar / table stay responsive
        # while shepherds tear down. ``self.notify`` is thread-safe in
        # Textual (it posts a message to the app loop).
        prs = sorted(self.procs)
        if not prs:
            self.notify("no PRs to rebase", severity="warning")
            return
        # Signal every shepherd up front so they all wind down in
        # parallel. Without this, each ``_terminate_group`` below would
        # block up to ``grace`` seconds *per PR* in series before the
        # next SIGTERM was even sent.
        for pr in prs:
            p = self.procs[pr][0]
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        # Now reap. ``_terminate_group`` is a no-op for already-exited
        # processes, and the second SIGTERM it sends to stragglers is
        # harmless.
        for pr in prs:
            _terminate_group(self.procs[pr][0])
        rebase_args = self._shepherd_args(["--rebase"])
        for pr in prs:
            try:
                self.procs[pr] = _spawn(pr, rebase_args)
            except Exception as e:
                self.notify(f"[{pr}] failed: {e}", severity="error")
            self._pr_titles.pop(pr, None)
        self.notify(f"rebasing {len(prs)} PR(s)")

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

    def _do_cancel(self, pr: int) -> str:
        entry = self.procs.get(pr)
        if entry is None:
            return f"[{pr}] unknown"
        _terminate_group(entry[0])
        return f"[{pr}] terminated"

    def _do_remove(self, pr: int) -> str:
        entry = self.procs.get(pr)
        if entry is None:
            return f"[{pr}] unknown"
        if entry[0].poll() is None:
            _terminate_group(entry[0])
        self._prune_pr(pr)
        return f"[{pr}] removed"

    def _refresh(self) -> None:
        # Detect any shepherds that exited with the "PR not actionable"
        # code and prune them before redrawing the table. We do this here
        # (in the periodic refresh) rather than only on user input so the
        # auto-prune happens even if the operator is just watching.
        for pr in list(self.procs):
            p = self.procs[pr][0]
            if p.poll() == EXIT_PR_NOT_ACTIONABLE:
                self._prune_pr(pr)
                self.notify(f"[{pr}] pruned (PR no longer open)")

        table = self.query_one(DataTable)
        table.clear()
        for pr in sorted(self.procs):
            p, _, log_path = self.procs[pr]
            rc = p.poll()
            if rc is None:
                state = "🟢"
            elif rc == 0:
                state = "✅"
            else:
                state = "🔴"
            last = _last_log_line(log_path)
            title = self._pr_titles.get(pr, "")
            if not title:
                title = _read_pr_title(pr)
                if title:
                    self._pr_titles[pr] = title
            # OSC-8 hyperlink so cmd/ctrl-click on the PR opens the PR;
            # the worktree path is omitted from the table entirely now,
            # but it's still ``~/.mergedog/worktrees/<pr>/`` if you need
            # it from the shell.
            pr_cell = Text(
                str(pr),
                style=f"link https://github.com/pytorch/pytorch/pull/{pr}",
            )
            table.add_row(pr_cell, Text(_truncate_title(title)), state, last)

    def _prune_pr(self, pr: int) -> None:
        """Forget a shepherd and clean up its on-disk state.

        The log file is kept so the operator can audit why the shepherd quit.
        """
        entry = self.procs.pop(pr, None)
        if entry is not None:
            try:
                entry[1].close()  # type: ignore[attr-defined]
            except Exception:
                pass
        for path in (state_file(pr), context_file(pr)):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            repo_mod.wipe_worktree(pr)
        except Exception:
            pass
        _remove_mux_pr(pr)
        self._pr_titles.pop(pr, None)

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
                    return f"rebasing {n} PR(s)"
                return self._do_add(_parse_pr(rest[0]), ["--rebase", *rest[1:]])
            elif cmd in ("restart", "r"):
                if not rest:
                    return "usage: restart <pr>"
                pr = _parse_pr(rest[0])
                self._do_cancel(pr)
                return self._do_add(pr, rest[1:])
            elif cmd == "reassess":
                if not rest:
                    return "usage: reassess <pr>"
                pr = _parse_pr(rest[0])
                self._do_cancel(pr)
                return self._do_add(pr, ["--reassess", *rest[1:]])
            elif cmd in ("cancel", "c", "kill"):
                if not rest:
                    return "usage: cancel <pr>"
                return self._do_cancel(_parse_pr(rest[0]))
            elif cmd in ("remove", "rm", "forget"):
                if not rest:
                    return "usage: remove <pr>"
                return self._do_remove(_parse_pr(rest[0]))
            elif cmd == "log":
                if not rest:
                    return "usage: log <pr>"
                pr = _parse_pr(rest[0])
                entry = self.procs.get(pr)
                if entry is not None:
                    return str(entry[2])
                return f"[{pr}] unknown"
            elif cmd == "migrate":
                return self._format_migrate()
            elif cmd in ("ignore-sev", "ignore_sev"):
                return self._do_ignore_sev(rest)
            elif cmd in ("mergedog-label", "mergedog_label"):
                return self._do_mergedog_label(rest)
            elif cmd == "status":
                return self._format_status()
            elif cmd in ("quit", "q", "exit"):
                return "use the TUI to quit the mux"
            else:
                return f"unknown command: {cmd!r}"
        except Exception as e:
            return f"error: {e}"

    def _format_status(self) -> str:
        """JSON status of all tracked PRs (consumed by MCP server)."""
        rows = []
        for pr in sorted(self.procs):
            p, _, log_path = self.procs[pr]
            rc = p.poll()
            if rc is None:
                state = "running"
            elif rc == 0:
                state = "exited_ok"
            elif rc == EXIT_PR_NOT_ACTIONABLE:
                state = "prunable"
            else:
                state = "exited_error"
            last = _last_log_line(log_path)
            title = self._pr_titles.get(pr, "") or _read_pr_title(pr)
            rows.append({
                "pr": pr,
                "title": title,
                "state": state,
                "last_log": last,
            })
        return json.dumps(rows, indent=2)

    def _format_migrate(self) -> str:
        prs = sorted(self.procs)
        if not prs:
            return "no PRs tracked"
        state_files = [
            str(state_file(pr)) for pr in prs if state_file(pr).exists()
        ]
        pr_args = " ".join(str(pr) for pr in prs)
        lines = [
            "# Copy state files to the new server:",
            f"scp {shlex.join(state_files)} NEW_HOST:~/.mergedog/state/",
            "# Then start the mux there:",
            f"python -m mergedog.mux {pr_args}",
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
        if self._ipc_server is not None:
            self._ipc_server.close()
        if self._lock_fd >= 0:
            release_lock(self._lock_fd)
        for _pr, (p, _f, _) in self.procs.items():
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for _pr, (p, f, _) in self.procs.items():
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
        help=(
            "Start a shepherd for every PR in the mux-tracked list "
            f"({MUX_PRS_FILE})."
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

    ensure_dirs()

    try:
        lock_fd = acquire_lock()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    initial: list[int] = []
    if args.resume_known:
        initial.extend(_read_mux_prs())
    for raw in args.prs:
        try:
            initial.append(_parse_pr(raw))
        except argparse.ArgumentTypeError as e:
            print(f"skipping {raw!r}: {e}", file=sys.stderr)
    seen: set[int] = set()
    initial = [pr for pr in initial if not (pr in seen or seen.add(pr))]

    app = MuxApp(
        initial,
        ignore_sev=args.ignore_sev,
        manage_mergedog_label=args.manage_mergedog_label,
        gchat_to=args.gchat_to,
        lock_fd=lock_fd,
    )
    app.run(mouse=False)
    if hasattr(app, "_migrate_output"):
        print(app._migrate_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
