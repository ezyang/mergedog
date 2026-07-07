"""Network environment helpers for subprocesses that talk to GitHub."""
from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache


# Output that indicates a transient network/proxy failure worth retrying.
# Shared by the gh and ghstack retry loops so the lists can't drift.
TRANSIENT_HTTP_CODES = ("502", "503", "504")
# Match the codes as standalone numbers only, not as digit runs inside
# larger numbers (run IDs, job IDs) like "runs/9502312345".
_TRANSIENT_HTTP_CODE_RE = re.compile(
    r"(?<!\d)(?:" + "|".join(TRANSIENT_HTTP_CODES) + r")(?!\d)"
)
TRANSIENT_MESSAGES = (
    "connection refused",
    "connection reset",
    "connection timed out",
    "error connecting to api.github.com",
    "failed to establish a new connection",
    "i/o timeout",
    "max retries exceeded",
    "newconnectionerror",
    "proxyerror",
    "temporary failure",
    "tls handshake timeout",
    "unable to connect to proxy",
    "unexpected eof",
)


def is_transient_network_error(text: str) -> bool:
    lower = text.lower()
    return _TRANSIENT_HTTP_CODE_RE.search(text) is not None or any(
        msg in lower for msg in TRANSIENT_MESSAGES
    )


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
