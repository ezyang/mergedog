"""Tiny multi-PR supervisor with a Textual TUI.

Top: a DataTable with one row per active shepherd subprocess.
Bottom: an Input field that takes commands.

Run via::

    python -m mergedog.mux [<pr>...] [--resume-known]

Commands typed at the bottom (enter to submit):

    add <pr> [extra mergedog flags]   start a shepherd
    <pr>                              shorthand for ``add <pr>``
    rebase <pr>                       shorthand for ``add <pr> --rebase``
    rebase all                        re-run every tracked PR with --rebase
    cancel <pr>                       SIGTERM a shepherd
    log <pr>                          show the path to its log file
    ignore-sev [on|off]               toggle (or show) the mux-wide
                                      ``--ignore-sev`` default applied to
                                      every shepherd spawn
    quit                              terminate everything and exit

Each shepherd's stdout/stderr is piped to ``~/.mergedog/logs/<pr>.log``.
"""
from __future__ import annotations

import argparse
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

from mergedog import repo as repo_mod  # noqa: E402
from mergedog.cli import _parse_pr  # noqa: E402
from mergedog.paths import (  # noqa: E402
    MUX_PRS_FILE,
    ROOT,
    context_file,
    ensure_dirs,
    state_file,
)
from mergedog.shepherd import EXIT_PR_NOT_ACTIONABLE  # noqa: E402

LOG_DIR = ROOT / "logs"


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
    except ProcessLookupError:
        pass
    try:
        p.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
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

    def __init__(self, initial: list[int], *, ignore_sev: bool = False) -> None:
        super().__init__()
        self.procs: dict[int, tuple[subprocess.Popen, object, Path]] = {}
        self._initial = initial
        # Mux-wide default for ``--ignore-sev``. When True, every shepherd
        # spawn gets the flag injected so newly added (or rebase-respawned)
        # PRs don't park on an open CI SEV. Existing parked shepherds are
        # NOT auto-restarted on toggle -- use ``rebase all`` (or cancel +
        # add) to apply the new default to running PRs.
        self.ignore_sev = ignore_sev

    def compose(self) -> ComposeResult:
        yield DataTable()
        yield Input(
            placeholder=(
                "<pr> | add <pr> | rebase <pr|all> | reassess <pr> | "
                "cancel <pr> | log <pr> | ignore-sev [on|off] | quit"
            )
        )

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("PR", "", "Last")
        for pr in self._initial:
            self._do_add(pr, [])
        self._refresh()
        self.set_interval(2.0, self._refresh)
        self.query_one(Input).focus()

    def _shepherd_args(self, extra: list[str]) -> list[str]:
        """Apply mux-wide defaults to a shepherd argv tail."""
        if self.ignore_sev and "--ignore-sev" not in extra:
            return ["--ignore-sev", *extra]
        return list(extra)

    def _do_add(self, pr: int, extra: list[str]) -> None:
        if pr in self.procs and self.procs[pr][0].poll() is None:
            self.notify(f"[{pr}] already running", severity="warning")
            return
        try:
            self.procs[pr] = _spawn(pr, self._shepherd_args(extra))
        except Exception as e:
            self.notify(f"[{pr}] failed: {e}", severity="error")
            return
        _add_mux_pr(pr)

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
        self.notify(f"rebasing {len(prs)} PR(s)")

    def _do_ignore_sev(self, rest: list[str]) -> None:
        if not rest:
            state = "on" if self.ignore_sev else "off"
            self.notify(f"ignore-sev is {state}")
            return
        arg = rest[0].lower()
        if arg in ("on", "true", "1", "yes"):
            new = True
        elif arg in ("off", "false", "0", "no"):
            new = False
        elif arg == "toggle":
            new = not self.ignore_sev
        else:
            self.notify(f"usage: ignore-sev [on|off|toggle]", severity="warning")
            return
        self.ignore_sev = new
        state = "on" if new else "off"
        self.notify(
            f"ignore-sev {state} (applies to future spawns; "
            f"use `rebase all` to apply to running PRs)"
        )

    def _do_cancel(self, pr: int) -> None:
        entry = self.procs.get(pr)
        if entry is None:
            self.notify(f"[{pr}] unknown", severity="warning")
            return
        _terminate_group(entry[0])
        self.notify(f"[{pr}] terminated")

    def _refresh(self) -> None:
        # Detect any shepherds that exited with the "PR not actionable"
        # code and prune them before redrawing the table. We do this here
        # (in the periodic refresh) rather than only on user input so the
        # auto-prune happens even if the operator is just watching.
        for pr in list(self.procs):
            p = self.procs[pr][0]
            if p.poll() == EXIT_PR_NOT_ACTIONABLE:
                self._prune_pr(pr)

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
            # OSC-8 hyperlink so cmd/ctrl-click on the PR opens the PR;
            # the worktree path is omitted from the table entirely now,
            # but it's still ``~/.mergedog/worktrees/<pr>/`` if you need
            # it from the shell.
            pr_cell = Text(
                str(pr),
                style=f"link https://github.com/pytorch/pytorch/pull/{pr}",
            )
            table.add_row(pr_cell, state, last)

    def _prune_pr(self, pr: int) -> None:
        """Forget a shepherd and clean up its on-disk state.

        Used when the shepherd exits with ``EXIT_PR_NOT_ACTIONABLE`` (PR
        closed/merged) -- there's no recovery, no reason to keep the
        worktree, trust DB, or context file around. The log file is kept
        so the operator can audit why the shepherd quit.
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
        self.notify(f"[{pr}] pruned (PR no longer open)")

    def on_input_submitted(self, message: Input.Submitted) -> None:
        line = message.value.strip()
        message.input.value = ""
        if not line:
            return
        try:
            args = shlex.split(line)
        except ValueError as e:
            self.notify(f"parse error: {e}", severity="error")
            return
        cmd, rest = args[0], args[1:]
        # Bare PR number or PR URL → treat as ``add <pr>``.
        if cmd.isdigit() or "/pull/" in cmd:
            cmd, rest = "add", args
        try:
            if cmd in ("add", "a"):
                if not rest:
                    self.notify("usage: add <pr> [flags]", severity="warning")
                else:
                    self._do_add(_parse_pr(rest[0]), rest[1:])
            elif cmd == "rebase":
                if not rest:
                    self.notify("usage: rebase <pr> | rebase all", severity="warning")
                elif rest[0] == "all":
                    self._do_rebase_all()
                else:
                    self._do_add(_parse_pr(rest[0]), ["--rebase", *rest[1:]])
            elif cmd == "reassess":
                if not rest:
                    self.notify("usage: reassess <pr>", severity="warning")
                else:
                    self._do_add(_parse_pr(rest[0]), ["--reassess", *rest[1:]])
            elif cmd in ("cancel", "c", "kill", "rm"):
                if not rest:
                    self.notify("usage: cancel <pr>", severity="warning")
                else:
                    self._do_cancel(_parse_pr(rest[0]))
            elif cmd == "log":
                if not rest:
                    self.notify("usage: log <pr>", severity="warning")
                else:
                    pr = _parse_pr(rest[0])
                    entry = self.procs.get(pr)
                    if entry is not None:
                        self.notify(str(entry[2]))
                    else:
                        self.notify(f"[{pr}] unknown", severity="warning")
            elif cmd in ("ignore-sev", "ignore_sev"):
                self._do_ignore_sev(rest)
            elif cmd in ("quit", "q", "exit"):
                self.exit()
                return
            else:
                self.notify(f"unknown command: {cmd!r}", severity="warning")
        except Exception as e:
            self.notify(f"error: {e}", severity="error")
        self._refresh()

    def on_unmount(self) -> None:
        # Fan out SIGTERM first so the per-PR grace windows in
        # ``_terminate_group`` overlap instead of serializing -- otherwise
        # quitting with N tracked PRs blocks for up to ~4*N seconds.
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
    args = parser.parse_args()

    ensure_dirs()

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

    # ``mouse=False`` keeps the terminal's native selection / right-click
    # paste working. We don't actually click anything in the table.
    MuxApp(initial, ignore_sev=args.ignore_sev).run(mouse=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
