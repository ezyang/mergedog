"""MCP server exposing mergedog mux interaction as Claude Code tools.

Run via::

    python -m mergedog.mcp_server

Configure in Claude Code's MCP settings to connect via stdio.

Tools:

    mergedog_command   Send any mux command (add, cancel, rebase, etc.)
    mergedog_status    Get live status of all tracked PR jobs
    mergedog_log       Read a PR's shepherd log (tail)
    mergedog_state     Read a PR's trust-DB state
"""
from __future__ import annotations

import json
import sys

from mergedog.bootstrap import promote_early_env


promote_early_env(sys.argv[1:])

from mcp.server import FastMCP  # noqa: E402

from mergedog.ipc import send_command  # noqa: E402
from mergedog.paths import ROOT, MUX_JOBS_FILE, MUX_PRS_FILE  # noqa: E402

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
        "  fix <pr> <request> — make a trusted operator-requested "
        "[MERGEDOG] follow-up commit\n"
        "  cancel <pr>        — stop shepherding; keep state\n"
        "  cleanup | clean    — forget successful completed shepherds\n"
        "  remove <pr>        — stop and forget (wipes worktree)\n"
        "  restart <pr>       — cancel + add\n"
        "  restart all        — restart every current mux-session job\n"
        "  restart dead       — restart only crashed shepherds\n"
        "  rebase <pr>        — restart with --rebase\n"
        "  rebase all         — rebase every current mux-session job\n"
        "  reassess <pr>      — restart with --reassess\n"
        "  fix <pr> <request> — restart with --operator-fix-context\n"
        "  mark-spurious <pr> — snapshot current failed/cancelled checks "
        "as spurious and restart\n"
        "  ignore-sev [on|off] — toggle mux-wide --ignore-sev\n"
        "  mergedog-label [on|off] — toggle mux-wide --manage-mergedog-label\n"
        "  fix-cap [N|off|default] — set mux-wide --max-fix-commits\n"
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
        "Get the current status of all PR jobs tracked by the mux. "
        "Returns JSON with kind, pr, title, state "
        "(running/exited_ok/exited_error), phase, status, last_log, "
        "and shepherd_status for each job."
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
        "Read the tail of a PR shepherd log.  Reads directly from "
        "disk, does not require the mux to be running."
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
        "List all PR jobs the mux is tracking (from the durable subscription "
        "list).  Works even when the mux is not running."
    ),
)
async def mergedog_list_prs() -> str:
    """List tracked mux PR jobs from the subscription file."""
    path = MUX_JOBS_FILE if MUX_JOBS_FILE.exists() else MUX_PRS_FILE
    if not path.exists():
        return "No tracked PRs (mux subscription file not found)"
    try:
        data = json.loads(path.read_text())
        return json.dumps(data, indent=2)
    except (OSError, json.JSONDecodeError) as e:
        return f"error reading {path.name}: {e}"


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
