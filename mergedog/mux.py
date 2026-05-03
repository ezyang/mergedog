"""Tiny multi-PR supervisor.

Runs each shepherd as a separate ``python -m mergedog <pr>`` subprocess
with its stdout/stderr piped to ``~/.mergedog/logs/<pr>.log``. The mux
process itself is a single-line REPL that knows which subprocesses it
spawned.

Run via:

    python -m mergedog.mux

Commands:

    add <pr> [extra mergedog flags]   start a shepherd
    status                            list active shepherds + last log line
    cancel <pr>                       SIGTERM a shepherd
    log <pr>                          print path to its log file
    quit                              terminate everything and exit
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

from mergedog.cli import _parse_pr
from mergedog.paths import ROOT, STATE_DIR, ensure_dirs, worktree_dir

LOG_DIR = ROOT / "logs"


def _add(procs: dict, pr_arg: str, extra: list[str]) -> None:
    pr = _parse_pr(pr_arg)
    if pr in procs and procs[pr][0].poll() is None:
        print(f"[{pr}] already running")
        return
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
    procs[pr] = (p, f, log_path)
    print(f"[{pr}] started (log: {log_path})")


def _cancel(procs: dict, pr_arg: str) -> None:
    pr = _parse_pr(pr_arg)
    entry = procs.get(pr)
    if entry is None:
        print(f"[{pr}] unknown")
        return
    p = entry[0]
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    print(f"[{pr}] terminated")


def _last_log_line(path: Path) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fp:
            tail = fp.readlines()[-1] if fp.readable() else ""
    except OSError:
        return ""
    return tail.rstrip()


def _status(procs: dict) -> None:
    if not procs:
        print("(no shepherds)")
        return
    for pr in sorted(procs):
        p, _, log_path = procs[pr]
        rc = p.poll()
        state = "RUNNING" if rc is None else f"EXIT={rc}"
        last = _last_log_line(log_path)[:120]
        print(f"[{pr:>7}] {state:>9}  wt={worktree_dir(pr)}  :: {last}")


def _known_prs() -> list[int]:
    """Every PR that has a persisted trust DB on disk, sorted."""
    if not STATE_DIR.exists():
        return []
    out = []
    for p in STATE_DIR.glob("*.json"):
        try:
            out.append(int(p.stem))
        except ValueError:
            continue
    return sorted(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="mergedog.mux",
        description="Supervise multiple mergedog shepherds in one process.",
    )
    parser.add_argument(
        "prs",
        nargs="*",
        help=(
            "PR numbers (or URLs) to start shepherding immediately. "
            "Pass --resume-known to start every PR with state on disk."
        ),
    )
    parser.add_argument(
        "--resume-known",
        action="store_true",
        help=(
            "Start a shepherd for every PR with a state file in "
            "~/.mergedog/state/. Useful after ctrl-c'ing a batch of "
            "old-style ``mergedog <pr>`` sessions."
        ),
    )
    args = parser.parse_args()

    ensure_dirs()
    procs: dict[int, tuple[subprocess.Popen, object, Path]] = {}

    initial: list[int] = []
    if args.resume_known:
        initial.extend(_known_prs())
    for raw in args.prs:
        try:
            initial.append(_parse_pr(raw))
        except argparse.ArgumentTypeError as e:
            print(f"skipping {raw!r}: {e}")
    # Dedup, preserve order.
    seen = set()
    for pr in initial:
        if pr in seen:
            continue
        seen.add(pr)
        try:
            _add(procs, str(pr), [])
        except Exception as e:
            print(f"[{pr}] failed to start: {e}")

    print("mergedog mux. commands: add <pr>, status, cancel <pr>, log <pr>, quit")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            _status(procs)
            continue
        try:
            args = shlex.split(line)
        except ValueError as e:
            print(f"parse error: {e}")
            continue
        cmd, rest = args[0], args[1:]
        try:
            if cmd in ("add", "a"):
                if not rest:
                    print("usage: add <pr> [mergedog flags]")
                else:
                    _add(procs, rest[0], rest[1:])
            elif cmd in ("status", "s", "ls"):
                _status(procs)
            elif cmd in ("cancel", "c", "rm", "kill"):
                if not rest:
                    print("usage: cancel <pr>")
                else:
                    _cancel(procs, rest[0])
            elif cmd == "log":
                if not rest:
                    print("usage: log <pr>")
                elif (entry := procs.get(_parse_pr(rest[0]))):
                    print(entry[2])
                else:
                    print("unknown PR")
            elif cmd in ("quit", "q", "exit"):
                break
            else:
                print(f"unknown command: {cmd!r}")
        except Exception as e:
            print(f"error: {e}")
    for pr, (p, f, _) in procs.items():
        if p.poll() is None:
            p.terminate()
        try:
            f.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
