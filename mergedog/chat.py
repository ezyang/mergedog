"""Launch Claude Code with the mergedog MCP server pre-configured.

Usage::

    python -m mergedog.chat [--root DIR]

This starts ``claude`` with the mergedog MCP wired up via
``--mcp-config`` so the user doesn't need to edit any settings files.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    # Resolve the python interpreter inside our venv so the MCP server
    # runs with the same environment regardless of how the user invoked us.
    python = sys.executable

    root_args: list[str] = []
    mergedog_root = os.environ.get("MERGEDOG_ROOT", "")
    # Forward --root to the MCP server
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--root" and i + 1 < len(sys.argv):
            mergedog_root = str(Path(sys.argv[i + 1]).expanduser().resolve())
            i += 2
            continue
        if sys.argv[i].startswith("--root="):
            mergedog_root = str(
                Path(sys.argv[i].split("=", 1)[1]).expanduser().resolve()
            )
            i += 1
            continue
        i += 1

    mcp_args = ["-m", "mergedog.mcp_server"]
    if mergedog_root:
        mcp_args.extend(["--root", mergedog_root])
        root_args = ["--root", mergedog_root]

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
