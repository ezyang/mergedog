"""Network environment helpers for subprocesses that talk to GitHub."""
from __future__ import annotations

import os
import subprocess
from functools import lru_cache


_PROXY_ENV_NAMES = (
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
)


def _env_has_proxy() -> bool:
    return any(os.environ.get(name) for name in _PROXY_ENV_NAMES)


@lru_cache(maxsize=1)
def _git_configured_proxy() -> str | None:
    for key in ("https.proxy", "http.proxy"):
        try:
            proc = subprocess.run(
                ["git", "config", "--get", key],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        proxy = proc.stdout.strip()
        if proc.returncode == 0 and proxy:
            return proxy
    return None


def github_api_env_extra() -> dict[str, str] | None:
    """Return proxy env for GitHub API tools when only Git config has it."""
    if _env_has_proxy():
        return None
    proxy = _git_configured_proxy()
    if not proxy:
        return None
    return {
        "HTTPS_PROXY": proxy,
        "HTTP_PROXY": proxy,
    }
