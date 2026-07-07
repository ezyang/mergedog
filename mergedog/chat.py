"""Launch Claude Code with the mergedog MCP server pre-configured.

Usage::

    python -m mergedog.chat [--root DIR] [--repo OWNER/NAME]

This starts ``claude`` with the mergedog MCP wired up via
``--mcp-config`` so the user doesn't need to edit any settings files.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from mergedog.bootstrap import promote_early_env


def main() -> int:
    # Resolve the python interpreter inside our venv so the MCP server
    # runs with the same environment regardless of how the user invoked us.
    python = sys.executable

    # Forward --root/--repo to the MCP server
    promote_early_env(sys.argv[1:])
    mergedog_root = os.environ.get("MERGEDOG_ROOT", "")
    mergedog_repo = os.environ.get("MERGEDOG_REPO_SLUG") or os.environ.get(
        "MERGEDOG_REPO", ""
    )

    mcp_args = ["-m", "mergedog.mcp_server"]
    if mergedog_root:
        mcp_args.extend(["--root", mergedog_root])
    if mergedog_repo:
        mcp_args.extend(["--repo", mergedog_repo])

    mcp_config = json.dumps({
        "mcpServers": {
            "mergedog": {
                "command": python,
                "args": mcp_args,
            }
        }
    })

    prompt_file = Path(__file__).resolve().parent.parent / "docs" / "mcp-prompt.md"

    prompt = prompt_file.read_text() if prompt_file.exists() else ""

    cmd = [
        "claude",
        "--mcp-config", mcp_config,
    ]
    if prompt:
        cmd.extend(["--append-system-prompt", prompt])

    os.execvp("claude", cmd)


if __name__ == "__main__":
    sys.exit(main())
