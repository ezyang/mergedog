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

from mergedog.sanitize import sanitize_untrusted_markdown


def render_context(
    *,
    pr: int,
    url: str,
    title: str,
    body: str,
    comments: list[dict],
) -> str:
    parts: list[str] = [
        f"PR #{pr}",
        f"URL: {url}",
        "",
        "[TITLE]",
        sanitize_untrusted_markdown(title or ""),
        "",
        "[DESCRIPTION]",
        sanitize_untrusted_markdown(body) if body else "(no description)",
    ]
    for c in comments:
        author = c.get("author", "?")
        created = c.get("created_at", "")
        parts.extend(
            [
                "",
                f"[COMMENT by {author} at {created}]",
                sanitize_untrusted_markdown(c.get("body", "")),
            ]
        )
    return "\n".join(parts) + "\n"


def write_context_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
