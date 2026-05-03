"""Sanitize untrusted markdown from PR descriptions and comments.

Threat model: a maintainer approves a PR by attesting to what they see in
the GitHub UI. Anything *rendered to nothing* or *visually hidden* in that
UI is an injection vector -- instructions the maintainer didn't read.

We strip the rendered-but-hidden surfaces before feeding text to the agent.
"""
from __future__ import annotations

import re


# Codepoint ranges that render to nothing (or to misleading visual order)
# in the GitHub UI. Built from numeric ranges so the source is reviewable.
_INVISIBLE_RANGES: tuple[tuple[int, int], ...] = (
    (0x200B, 0x200D),    # zero-width space / non-joiner / joiner
    (0x2060, 0x2060),    # word joiner
    (0xFEFF, 0xFEFF),    # zero-width no-break space / BOM
    (0x202A, 0x202E),    # bidi embedding / override
    (0x2066, 0x2069),    # bidi isolates
    (0xE0000, 0xE007F),  # tag block (used in invisible-prompt attacks)
)


def _build_invisible_class() -> str:
    parts = []
    for lo, hi in _INVISIBLE_RANGES:
        if lo == hi:
            parts.append(re.escape(chr(lo)))
        else:
            parts.append(f"{re.escape(chr(lo))}-{re.escape(chr(hi))}")
    return "[" + "".join(parts) + "]"


_INVISIBLE_RE = re.compile(_build_invisible_class())

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

_DETAILS_TAGS_RE = re.compile(
    r"</?(details|summary)(\s[^>]*)?>", re.IGNORECASE
)


def strip_html_comments(text: str) -> str:
    return _HTML_COMMENT_RE.sub("", text)


def strip_invisible_unicode(text: str) -> str:
    return _INVISIBLE_RE.sub("", text)


def unwrap_details(text: str) -> str:
    return _DETAILS_TAGS_RE.sub("", text)


def sanitize_untrusted_markdown(text: str) -> str:
    text = strip_html_comments(text)
    text = unwrap_details(text)
    text = strip_invisible_unicode(text)
    return text
