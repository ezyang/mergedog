"""Structured shepherd status sidecar.

The human-readable log remains the audit trail. This module maintains the
small machine-readable JSON file the mux can poll for richer state.
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mergedog.paths import status_file

SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def write_status(
    pr: int,
    *,
    phase: str,
    message: str | None = None,
    category: str | None = None,
    waiting_on: str | None = None,
    action: str | None = None,
    user_action: str | None = None,
    approved: bool | None = None,
    merging: bool | None = None,
    ci_done: int | None = None,
    ci_total: int | None = None,
    ci_failed: int | None = None,
    fix_attempts: int | None = None,
    max_fix_attempts: int | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Atomically write one PR's structured shepherd status.

    Returns the payload for callers/tests that want to inspect the exact JSON.
    Optional fields with ``None`` values are omitted to keep the sidecar small.
    """
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _now_iso(),
        "phase": phase,
    }
    optional = {
        "message": message,
        "category": category,
        "waiting_on": waiting_on,
        "action": action,
        "user_action": user_action,
        "approved": approved,
        "merging": merging,
        "ci_done": ci_done,
        "ci_total": ci_total,
        "ci_failed": ci_failed,
        "fix_attempts": fix_attempts,
        "max_fix_attempts": max_fix_attempts,
    }
    payload.update({k: v for k, v in optional.items() if v is not None})

    dst = path or status_file(pr)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(tmp, dst)
    return payload


def read_status(pr: int, *, path: Path | None = None) -> dict[str, Any] | None:
    """Read a structured status sidecar, tolerating absent/corrupt files."""
    src = path or status_file(pr)
    try:
        data = json.loads(src.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if not isinstance(data.get("phase"), str):
        return None
    return data
