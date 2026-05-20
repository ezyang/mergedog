"""Collect and paste diagnostics for a mergedog PR."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from mergedog.paths import ROOT


_SECRET_KEY_RE = re.compile(
    r"""(?ix)
    (
      (?:
        ["']? [A-Z0-9_.-]*
        (?:token|secret|password|passwd|api[_-]?key|access[_-]?key|
           credential|cookie|session|authorization|private[_-]?key)
        [A-Z0-9_.-]* ["']?
      )
      \s* [:=] \s*
    )
    (["']?)
    [^"',\s}]+
    (["']?)
    """
)
_AUTH_HEADER_RE = re.compile(
    r"(?im)^(\s*(?:authorization|proxy-authorization)\s*:\s*)\S+.*$"
)
_URL_CREDENTIAL_RE = re.compile(r"(?i)\b(https?://)([^/\s:@]+):([^@\s/]+)@")
_SSH_WITH_TOKEN_RE = re.compile(r"(?i)\b(x-access-token:)[^@\s/]+@")
_GITHUB_TOKEN_RE = re.compile(
    r"\b(?:github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9_]{20,})\b"
)
_SLACK_TOKEN_RE = re.compile(r"\bxox[aboprs]-[A-Za-z0-9-]{10,}\b")
_AWS_ACCESS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_PASTE_URL_RE = re.compile(r"https?://\S+")
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)


def redact_secrets(text: str) -> str:
    """Best-effort credential scrubber for diagnostics before paste upload."""
    text = _PRIVATE_KEY_RE.sub("<REDACTED_PRIVATE_KEY>", text)
    text = _URL_CREDENTIAL_RE.sub(r"\1\2:<REDACTED>@", text)
    text = _SSH_WITH_TOKEN_RE.sub(r"\1<REDACTED>@", text)
    text = _AUTH_HEADER_RE.sub(r"\1<REDACTED>", text)
    text = _GITHUB_TOKEN_RE.sub("<REDACTED_GITHUB_TOKEN>", text)
    text = _SLACK_TOKEN_RE.sub("<REDACTED_SLACK_TOKEN>", text)
    text = _AWS_ACCESS_KEY_RE.sub("<REDACTED_AWS_ACCESS_KEY>", text)
    return _SECRET_KEY_RE.sub(r"\1\2<REDACTED>\3", text)


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _section(title: str, path: Path, body: str | None) -> str:
    lines = [f"## {title}", "", f"Path: `{path}`", ""]
    if body is None:
        lines.append("(missing)")
    else:
        lines.extend(["```", body.rstrip(), "```"])
    return "\n".join(lines)


def _git_head() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _git_worktree_summary(path: Path) -> str | None:
    if not path.exists():
        return None
    commands = [
        ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        ("head", ["git", "rev-parse", "HEAD"]),
        ("status", ["git", "status", "--short", "--branch"]),
    ]
    lines = []
    for label, cmd in commands:
        try:
            proc = subprocess.run(
                cmd,
                cwd=path,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as e:
            lines.append(f"{label}: error running git: {e}")
            continue
        if proc.returncode == 0:
            lines.append(f"{label}:\n{proc.stdout.rstrip()}")
        else:
            err = (proc.stderr or proc.stdout or "").strip()
            lines.append(f"{label}: git exited {proc.returncode}: {err}")
    return "\n\n".join(lines)


def _pushed_commits_for(prs: list[int], pushed_commits_path: Path) -> str | None:
    pushed_commits = _read_text(pushed_commits_path)
    if pushed_commits is None:
        return None
    pattern = re.compile(r"\bPR#(?:" + "|".join(str(pr) for pr in prs) + r")\b")
    return "\n".join(
        line for line in pushed_commits.splitlines() if pattern.search(line)
    )


def build_report(pr: int, *, root: Path = ROOT) -> str:
    """Build a redacted markdown diagnostics bundle for ``pr``."""
    root = root.expanduser().resolve()
    log_path = root / "logs" / f"{pr}.log"
    state_path = root / "state" / f"{pr}.json"
    context_path = root / "contexts" / f"{pr}.md"
    mux_prs_path = root / "mux-prs.json"
    pushed_commits_path = root / "pushed-commits.log"
    worktree_path = root / "worktrees" / str(pr)
    pushed_commits = _pushed_commits_for([pr], pushed_commits_path)

    parts = [
        f"# mergedog rage PR #{pr}",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"Root: `{root}`",
        f"mergedog git HEAD: `{_git_head()}`",
        "",
        _section("shepherd log", log_path, _read_text(log_path)),
        _section("trust/state JSON", state_path, _read_text(state_path)),
        _section("context sidecar", context_path, _read_text(context_path)),
        _section("mux tracked PRs", mux_prs_path, _read_text(mux_prs_path)),
        _section("pushed commits for PR", pushed_commits_path, pushed_commits),
        _section(
            "worktree git summary",
            worktree_path,
            _git_worktree_summary(worktree_path),
        ),
    ]
    return redact_secrets("\n\n".join(parts) + "\n")


def create_paste(content: str, *, title: str) -> str:
    """Upload ``content`` to the local paste tool and return its output."""
    pastebin = shutil.which("pastebin")
    if pastebin is not None:
        proc = subprocess.run(
            [pastebin, "--md", "--private", "--title", title],
            input=content,
            check=False,
            capture_output=True,
            text=True,
        )
    else:
        arc = shutil.which("arc")
        if arc is None:
            raise RuntimeError("no paste tool found; install pastebin or arc")
        proc = subprocess.run(
            [arc, "paste", "--title", title, "--lang", "md"],
            input=content,
            check=False,
            capture_output=True,
            text=True,
        )

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"paste creation failed: {err}")

    text = (proc.stdout or proc.stderr or "").strip()
    try:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError:
        rows = []
    for row in rows:
        for key in ("url", "uri", "link", "paste_url"):
            value = row.get(key)
            if value:
                return str(value)
    match = _PASTE_URL_RE.search(text)
    if match is not None:
        return match.group(0)
    return text


def rage(pr: int, *, root: Path = ROOT) -> str:
    report = build_report(pr, root=root)
    title = f"mergedog rage PR #{pr}"
    return create_paste(report, title=title)
