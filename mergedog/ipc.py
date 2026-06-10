"""IPC primitives for mux <-> MCP server communication.

The mux holds an flock on ``MUX_LOCK_FILE`` for its lifetime.  When the
process exits (cleanly or via crash), the OS releases the lock.  The MCP
server discovers the mux by attempting a non-blocking flock: if it
succeeds, no mux is running; if it fails (EWOULDBLOCK), the mux is alive
and the lock file contains the socket path.
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
from pathlib import Path

from mergedog.paths import MUX_LOCK_FILE, MUX_SOCKET


def acquire_lock() -> int:
    """Acquire the mux lock.  Returns the fd (caller must keep it open).

    Raises ``RuntimeError`` if another mux already holds the lock.
    """
    MUX_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(MUX_LOCK_FILE), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        try:
            info = json.loads(MUX_LOCK_FILE.read_text())
            pid = info.get("pid", "?")
        except Exception:
            pid = "?"
        raise RuntimeError(
            f"Another mux is already running (pid {pid}).  "
            f"Kill it first or use --root to run a separate instance."
        ) from None
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(
        fd,
        json.dumps({"pid": os.getpid(), "socket": str(MUX_SOCKET)}).encode(),
    )
    return fd


def release_lock(fd: int) -> None:
    """Release the mux lock and clean up socket + lock files."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        os.close(fd)
    except Exception:
        pass
    MUX_SOCKET.unlink(missing_ok=True)
    MUX_LOCK_FILE.unlink(missing_ok=True)


def discover_mux() -> Path:
    """Find a running mux.  Returns the socket path.

    Raises ``RuntimeError`` if no mux is running.
    """
    if not MUX_LOCK_FILE.exists():
        raise RuntimeError(
            "No mux running (lock file not found).  "
            "Start one with: python -m mergedog.mux"
        )
    fd = os.open(str(MUX_LOCK_FILE), os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # We got the lock — means no mux holds it (stale file).
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        raise RuntimeError(
            "Mux lock file exists but no mux is running (stale lock).  "
            "Start one with: python -m mergedog.mux"
        )
    except BlockingIOError:
        # Good — mux is alive and holds the lock.
        os.close(fd)
    try:
        info = json.loads(MUX_LOCK_FILE.read_text())
        sock = Path(info["socket"])
    except Exception as e:
        raise RuntimeError(f"Cannot read mux lock file: {e}") from e
    if not sock.exists():
        raise RuntimeError(
            f"Mux socket {sock} not found (mux may still be starting up)"
        )
    return sock


async def send_command(command: str, *, timeout: float = 30.0) -> str:
    """Send a command to the running mux and return the response."""
    sock = discover_mux()
    reader, writer = await asyncio.open_unix_connection(str(sock))
    try:
        writer.write((command + "\n").encode())
        await writer.drain()
        raw = await asyncio.wait_for(reader.readline(), timeout=timeout)
        data = json.loads(raw.decode())
        return data.get("message", "")
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
