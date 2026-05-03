"""Tiny multi-PR supervisor with a Textual TUI.

Top: a DataTable with one row per active shepherd subprocess.
Bottom: an Input field that takes commands.

Run via::

    python -m mergedog.mux [<pr>...] [--resume-known]

Commands typed at the bottom (enter to submit):

    add <pr> [extra mergedog flags]   start a shepherd
    cancel <pr>                       SIGTERM a shepherd
    log <pr>                          show the path to its log file
    quit                              terminate everything and exit

Each shepherd's stdout/stderr is piped to ``~/.mergedog/logs/<pr>.log``.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import DataTable, Input

from mergedog.cli import _parse_pr
from mergedog.paths import ROOT, STATE_DIR, ensure_dirs, worktree_dir

LOG_DIR = ROOT / "logs"


def _last_log_line(path: Path) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fp:
            tail = fp.readlines()[-1] if fp.readable() else ""
    except OSError:
        return ""
    return tail.rstrip()


def _known_prs() -> list[int]:
    if not STATE_DIR.exists():
        return []
    out: list[int] = []
    for p in STATE_DIR.glob("*.json"):
        try:
            out.append(int(p.stem))
        except ValueError:
            continue
    return sorted(out)


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
    )
    return (p, f, log_path)


class MuxApp(App):
    CSS = """
    DataTable { height: 1fr; }
    Input { dock: bottom; }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self, initial: list[int]) -> None:
        super().__init__()
        self.procs: dict[int, tuple[subprocess.Popen, object, Path]] = {}
        self._initial = initial

    def compose(self) -> ComposeResult:
        yield DataTable()
        yield Input(placeholder="add <pr> | cancel <pr> | log <pr> | quit")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("PR", "State", "Worktree", "Last")
        for pr in self._initial:
            self._do_add(pr, [])
        self._refresh()
        self.set_interval(2.0, self._refresh)
        self.query_one(Input).focus()

    def _do_add(self, pr: int, extra: list[str]) -> None:
        if pr in self.procs and self.procs[pr][0].poll() is None:
            self.notify(f"[{pr}] already running", severity="warning")
            return
        try:
            self.procs[pr] = _spawn(pr, extra)
            self.notify(f"[{pr}] started")
        except Exception as e:
            self.notify(f"[{pr}] failed: {e}", severity="error")

    def _do_cancel(self, pr: int) -> None:
        entry = self.procs.get(pr)
        if entry is None:
            self.notify(f"[{pr}] unknown", severity="warning")
            return
        p = entry[0]
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        self.notify(f"[{pr}] terminated")

    def _refresh(self) -> None:
        table = self.query_one(DataTable)
        table.clear()
        for pr in sorted(self.procs):
            p, _, log_path = self.procs[pr]
            rc = p.poll()
            if rc is None:
                state = "RUNNING"
            elif rc == 0:
                state = "DONE"
            else:
                state = f"HALTED rc={rc}"
            last = _last_log_line(log_path)[:200]
            wt = str(worktree_dir(pr))
            table.add_row(str(pr), state, wt, last)

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
        try:
            if cmd in ("add", "a"):
                if not rest:
                    self.notify("usage: add <pr> [flags]", severity="warning")
                else:
                    self._do_add(_parse_pr(rest[0]), rest[1:])
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
            elif cmd in ("quit", "q", "exit"):
                self.exit()
                return
            else:
                self.notify(f"unknown command: {cmd!r}", severity="warning")
        except Exception as e:
            self.notify(f"error: {e}", severity="error")
        self._refresh()

    def on_unmount(self) -> None:
        for _pr, (p, f, _) in self.procs.items():
            if p.poll() is None:
                p.terminate()
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
        help="Start a shepherd for every PR with state on disk.",
    )
    args = parser.parse_args()

    ensure_dirs()

    initial: list[int] = []
    if args.resume_known:
        initial.extend(_known_prs())
    for raw in args.prs:
        try:
            initial.append(_parse_pr(raw))
        except argparse.ArgumentTypeError as e:
            print(f"skipping {raw!r}: {e}", file=sys.stderr)
    seen: set[int] = set()
    initial = [pr for pr in initial if not (pr in seen or seen.add(pr))]

    MuxApp(initial).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
