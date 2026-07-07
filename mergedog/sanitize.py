"""Sanitize untrusted markdown from PR descriptions and comments.

Threat model: a maintainer approves a PR by attesting to what they see in
the GitHub UI. Anything *rendered to nothing* or *visually hidden* in that
UI is an injection vector -- instructions the maintainer didn't read.

We strip the rendered-but-hidden surfaces before feeding text to the agent.

Sanitizing is not declassifying: every function here preserves TaintedStr,
so tainted input yields tainted output and stripping taint still requires
an explicit, justified untaint() call at the consumer.
"""
from __future__ import annotations

import re
import unicodedata

from mergedog.taint import TaintedStr


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


def _retaint_like(original: str, result: str) -> str:
    """Carry the input's taint over to a derived string.

    re.sub / str-building lose the TaintedStr subclass; sanitizers must not
    silently launder taint that way.
    """
    if isinstance(original, TaintedStr) and not isinstance(result, TaintedStr):
        return TaintedStr(result, source=original.source)
    return result


def _escape_control(ch: str) -> str:
    code = ord(ch)
    if code <= 0xFF:
        return f"\\x{code:02x}"
    if code <= 0xFFFF:
        return f"\\u{code:04x}"
    return f"\\U{code:08x}"


def sanitize_untrusted_text(text: str) -> str:
    """Canonicalize untrusted text before it reaches prompts or logs.

    Keep ordinary Unicode content, but remove characters whose display is
    invisible or layout-dependent, normalize whitespace to ASCII forms, and
    render process-control bytes visibly.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    for ch in normalized:
        if ch == "\n":
            out.append(ch)
            continue
        if ch == "\t":
            out.append(ch)
            continue
        category = unicodedata.category(ch)
        if category == "Zl" or category == "Zp":
            out.append("\n")
        elif category == "Zs":
            out.append(" ")
        elif category == "Cf":
            continue
        elif category in {"Cc", "Cs"}:
            out.append(_escape_control(ch))
        else:
            out.append(ch)
    return _retaint_like(text, "".join(out))


def strip_html_comments(text: str) -> str:
    return _retaint_like(text, _HTML_COMMENT_RE.sub("", text))


def strip_invisible_unicode(text: str) -> str:
    return _retaint_like(text, _INVISIBLE_RE.sub("", text))


def unwrap_details(text: str) -> str:
    return _retaint_like(text, _DETAILS_TAGS_RE.sub("", text))


def sanitize_untrusted_markdown(text: str) -> str:
    text = sanitize_untrusted_text(text)
    text = strip_html_comments(text)
    text = unwrap_details(text)
    text = strip_invisible_unicode(text)
    return text
