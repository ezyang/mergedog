"""Build the per-PR sidecar file the agent reads for context.

The sidecar holds untrusted text written by the PR author and commenters
(title, description, conversation comments). It is sanitized for GitHub
UI-hidden injection vectors before being written; the agent prompt frames
the file as untrusted data.

Section markers in the sidecar are best-effort. They can be spoofed in the
PR body, but the prompt tells the agent to treat the entire file as data
regardless of what it claims to be.
"""
from __future__ import annotations

from pathlib import Path

from mergedog.paths import atomic_write_text
from mergedog.sanitize import sanitize_untrusted_markdown, sanitize_untrusted_text
from mergedog.taint import untaint


_TRUSTED_COMMENT_AUTHORS = frozenset(
    {
        "pytorch-bot",
        "pytorch-bot[bot]",
        "pytorchmergebot",
        "facebook-github-bot",
    }
)


def render_context(
    *,
    pr: int,
    url: str,
    title: str,
    body: str,
    comments: list[dict],
    trusted: bool = True,
) -> str:
    parts: list[str] = [
        f"PR #{pr}",
        f"URL: {url}",
        "",
        "[TITLE]",
        untaint(sanitize_untrusted_markdown(title or "")),
    ]
    if trusted:
        parts.extend(
            [
                "",
                "[DESCRIPTION]",
                untaint(sanitize_untrusted_markdown(body)) if body else "(no description)",
            ]
        )
    for c in comments:
        author = c.get("author", "?")
        if not trusted and author not in _TRUSTED_COMMENT_AUTHORS:
            continue
        created = sanitize_untrusted_text(c.get("created_at", ""))
        parts.extend(
            [
                "",
                "[COMMENT by "
                f"{sanitize_untrusted_text(untaint(author))} at {created}]",
                untaint(sanitize_untrusted_markdown(c.get("body", ""))),
            ]
        )
    return "\n".join(parts) + "\n"


def write_context_file(path: Path, text: str) -> None:
    # Atomic so a SIGKILL mid-write can't leave a truncated sidecar that
    # would confuse claude on the next run.
    atomic_write_text(path, text)
