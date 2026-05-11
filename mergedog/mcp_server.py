"""MCP server exposing mergedog mux interaction as Claude Code tools.

Run via::

    python -m mergedog.mcp_server

Configure in Claude Code's MCP settings to connect via stdio.

Tools:

    mergedog_command   Send any mux command (add, cancel, rebase, etc.)
    mergedog_status    Get live status of all tracked PRs
    mergedog_log       Read a PR's shepherd log (tail)
    mergedog_state     Read a PR's trust-DB state
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _preparse_root() -> None:
    """Honor ``MERGEDOG_ROOT`` so paths resolve correctly."""
    for i, a in enumerate(sys.argv[1:]):
        if a == "--root" and i + 1 < len(sys.argv) - 1:
            os.environ["MERGEDOG_ROOT"] = str(
                Path(sys.argv[i + 2]).expanduser().resolve()
            )
            return
        if a.startswith("--root="):
            os.environ["MERGEDOG_ROOT"] = str(
                Path(a.split("=", 1)[1]).expanduser().resolve()
            )
            return


_preparse_root()

from mcp.server import FastMCP  # noqa: E402

from mergedog.ipc import discover_mux, send_command  # noqa: E402
from mergedog.paths import ROOT, MUX_PRS_FILE  # noqa: E402

LOG_DIR = ROOT / "logs"
STATE_DIR = ROOT / "state"

mcp = FastMCP(
    "mergedog",
    instructions=(
        "Mergedog manages pytorch CI shepherding.  Use these tools to "
        "interact with a running mergedog mux instance — query PR status, "
        "read logs, and send commands."
    ),
)


@mcp.tool(
    description=(
        "Send a command to the running mergedog mux.  Accepts the same "
        "text commands as the mux TUI input bar.\n\n"
        "Commands:\n"
        "  add <pr>           — start shepherding a PR\n"
        "  cancel <pr>        — stop shepherding (keeps state)\n"
        "  remove <pr>        — stop and forget (wipes worktree)\n"
        "  restart <pr>       — cancel + add\n"
        "  restart all        — restart every tracked PR\n"
        "  rebase <pr>        — restart with --rebase\n"
        "  rebase all         — rebase every tracked PR\n"
        "  reassess <pr>      — restart with --reassess\n"
        "  ignore-sev [on|off] — toggle mux-wide --ignore-sev\n"
        "  mergedog-label [on|off] — toggle mux-wide --manage-mergedog-label\n"
        "  log <pr>           — show log file path\n"
        "  status             — JSON status of all PRs\n"
    ),
)
async def mergedog_command(command: str) -> str:
    """Send a command to the mux and return its response."""
    try:
        return await send_command(command)
    except RuntimeError as e:
        return f"error: {e}"


@mcp.tool(
    description=(
        "Get the current status of all PRs tracked by the mux.  Returns "
        "JSON with pr, title, state (running/exited_ok/exited_error), and "
        "last_log for each PR."
    ),
)
async def mergedog_status() -> str:
    """Get live status of all tracked PRs from the mux."""
    try:
        return await send_command("status")
    except RuntimeError as e:
        return f"error: {e}"


@mcp.tool(
    description=(
        "Read the tail of a PR's shepherd log.  Reads directly from disk, "
        "does not require the mux to be running."
    ),
)
async def mergedog_log(pr: int, lines: int = 100) -> str:
    """Read the last N lines of a PR's log file."""
    log_path = LOG_DIR / f"{pr}.log"
    if not log_path.exists():
        return f"No log file for PR {pr}"
    try:
        all_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return "\n".join(tail)
    except OSError as e:
        return f"error reading log: {e}"


@mcp.tool(
    description=(
        "Read a PR's trust-DB state (trusted SHAs, branch, failure history). "
        "Reads directly from disk."
    ),
)
async def mergedog_state(pr: int) -> str:
    """Read a PR's persisted state JSON."""
    state_path = STATE_DIR / f"{pr}.json"
    if not state_path.exists():
        return f"No state file for PR {pr}"
    try:
        data = json.loads(state_path.read_text())
        return json.dumps(data, indent=2)
    except (OSError, json.JSONDecodeError) as e:
        return f"error reading state: {e}"


@mcp.tool(
    description=(
        "List all PR numbers the mux is tracking (from the durable "
        "subscription list).  Works even when the mux is not running."
    ),
)
async def mergedog_list_prs() -> str:
    """List tracked PR numbers from the mux subscription file."""
    if not MUX_PRS_FILE.exists():
        return "No tracked PRs (mux-prs.json not found)"
    try:
        data = json.loads(MUX_PRS_FILE.read_text())
        return json.dumps(data, indent=2)
    except (OSError, json.JSONDecodeError) as e:
        return f"error reading mux-prs.json: {e}"


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
