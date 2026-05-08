"""Collect and paste diagnostics for a mergedog PR."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

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


def _stack_worktree_candidates(root: Path, pr: int) -> str | None:
    worktrees = root / "worktrees"
    if not worktrees.exists():
        return None
    paths = sorted(p for p in worktrees.glob("stack-*") if p.is_dir())
    if not paths:
        return None
    exact = worktrees / f"stack-{pr}"
    lines = []
    if exact in paths:
        lines.append(
            f"`{exact}` exists; this PR is the bottom PR for a stack worktree."
        )
    else:
        lines.append(
            "No `stack-<this-pr>` worktree exists. If this PR is a non-bottom "
            "stack member, the shared worktree is named after the bottom PR."
        )
    lines.append("")
    lines.append("Existing stack worktrees:")
    lines.extend(f"- `{path}`" for path in paths)
    return "\n".join(lines)


def _pushed_commits_for(
    prs: Sequence[int], pushed_commits_path: Path
) -> str | None:
    pushed_commits = _read_text(pushed_commits_path)
    if pushed_commits is None:
        return None
    pattern = re.compile(r"\bPR#(?:" + "|".join(str(pr) for pr in prs) + r")\b")
    return "\n".join(
        line for line in pushed_commits.splitlines() if pattern.search(line)
    )


def _resolve_stack_prs(pr: int) -> list[int]:
    from mergedog.stack import resolve_stack

    members, _ = resolve_stack(pr)
    return [member.pr for member in members]


def _stack_log_path(root: Path, bottom_pr: int) -> Path:
    # Stack mode is one process for the whole stack, so rage uses one log
    # keyed by the bottom PR, matching the shared stack worktree name.
    stack_path = root / "logs" / f"stack-{bottom_pr}.log"
    if stack_path.exists():
        return stack_path
    legacy_path = root / "logs" / f"{bottom_pr}.log"
    if legacy_path.exists():
        return legacy_path
    return stack_path


def build_report(pr: int, *, root: Path = ROOT) -> str:
    """Build a redacted markdown diagnostics bundle for ``pr``."""
    root = root.expanduser().resolve()
    log_path = root / "logs" / f"{pr}.log"
    state_path = root / "state" / f"{pr}.json"
    context_path = root / "contexts" / f"{pr}.md"
    mux_prs_path = root / "mux-prs.json"
    pushed_commits_path = root / "pushed-commits.log"
    worktree_path = root / "worktrees" / str(pr)
    stack_worktree_path = root / "worktrees" / f"stack-{pr}"
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
        _section(
            "stack worktree git summary",
            stack_worktree_path,
            _git_worktree_summary(stack_worktree_path),
        ),
        _section(
            "stack worktree candidates",
            root / "worktrees",
            _stack_worktree_candidates(root, pr),
        ),
    ]
    return redact_secrets("\n\n".join(parts) + "\n")


def build_stack_report(
    pr: int, *, root: Path = ROOT, members: Sequence[int] | None = None
) -> str:
    """Build a redacted markdown diagnostics bundle for a whole stack."""
    root = root.expanduser().resolve()
    member_prs = list(members) if members is not None else _resolve_stack_prs(pr)
    if not member_prs:
        member_prs = [pr]
    bottom_pr = member_prs[0]
    log_path = _stack_log_path(root, bottom_pr)
    pushed_commits_path = root / "pushed-commits.log"
    stack_worktree_path = root / "worktrees" / f"stack-{bottom_pr}"

    parts = [
        f"# mergedog rage stack containing PR #{pr}",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"Root: `{root}`",
        f"mergedog git HEAD: `{_git_head()}`",
        "",
        _section(
            "stack members",
            root / "state",
            "\n".join(f"- PR #{member_pr}" for member_pr in member_prs),
        ),
        _section("stack shepherd log", log_path, _read_text(log_path)),
        _section(
            "stack worktree git summary",
            stack_worktree_path,
            _git_worktree_summary(stack_worktree_path),
        ),
        _section(
            "pushed commits for stack",
            pushed_commits_path,
            _pushed_commits_for(member_prs, pushed_commits_path),
        ),
        _section(
            "mux tracked PRs",
            root / "mux-prs.json",
            _read_text(root / "mux-prs.json"),
        ),
    ]
    for member_pr in member_prs:
        parts.extend(
            [
                _section(
                    f"PR #{member_pr} trust/state JSON",
                    root / "state" / f"{member_pr}.json",
                    _read_text(root / "state" / f"{member_pr}.json"),
                ),
                _section(
                    f"PR #{member_pr} context sidecar",
                    root / "contexts" / f"{member_pr}.md",
                    _read_text(root / "contexts" / f"{member_pr}.md"),
                ),
            ]
        )
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


def rage(pr: int, *, root: Path = ROOT, stack: bool = False) -> str:
    if stack:
        report = build_stack_report(pr, root=root)
        title = f"mergedog rage stack containing PR #{pr}"
    else:
        report = build_report(pr, root=root)
        title = f"mergedog rage PR #{pr}"
    return create_paste(report, title=title)
