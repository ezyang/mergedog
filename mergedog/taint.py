"""Dynamic information flow control for untrusted strings.

Strings from external sources (GitHub API responses) are wrapped in
TaintedStr. Prompt construction sites call assert_untainted() to crash
if tainted data leaks through without explicit declassification via
untaint().

Architecture
============

There are three roles in the system:

  Sources (TCB):  github.py wraps API return values in taint().
  Sinks:          Prompt construction (prompts.py, labels.py) calls
                  assert_untainted() on every interpolated value.
  Declassifiers:  Specific call sites that strip taint via untaint(),
                  each with a documented justification for why it's safe.

The *only* code you need to audit for correctness is: (1) the sources
in github.py (did we label everything that came from an external user?)
and (2) the untaint() call sites (is the justification still valid?).

Limitations
===========

TaintedStr is a str subclass, so any operation *on a plain str* that
merely consumes a tainted argument returns a plain str: f-strings,
``"template".format(tainted)``, ``", ".join(tainted_list)``, ``"%s" %
tainted``, and ``encode().decode()`` all silently launder taint. Only
methods called on the tainted object itself propagate it.

Two mitigations keep this from being exploitable:

  - Prompt sinks render templates via format_untainted(), which calls
    assert_untainted() on every string argument before formatting, so a
    tainted value that reaches a sink via laundering-prone plumbing
    still crashes instead of leaking.
  - The sanitize.py helpers explicitly re-apply taint to their output
    (sanitizing is not declassifying), so the common
    sanitize-then-interpolate path cannot launder by itself.

When adding a new prompt or template, use format_untainted() rather
than str.format or an f-string for the final assembly.

Handling TaintError
===================

If you hit a TaintError, it means untrusted data from a GitHub API
response is reaching a prompt without being deliberately declassified.
This is the system working as intended — DO NOT just add untaint() to
silence the error.

Instead:

  1. Trace where the tainted value originates (the error message
     includes the ``source`` label, e.g. "pr_comment", "ci_log").

  2. Decide how the value should reach the prompt:

     a) Through the sidecar file — the existing pattern for PR
        title/body/comments. Pass it through
        sanitize_untrusted_markdown(), then untaint(), then write it
        to the sidecar. The prompt already tells Claude to treat the
        sidecar as untrusted data.

     b) Direct interpolation with output constraints — the pattern
        used by autolabel (labels.py). The LLM output is validated
        against a known set, so injection can only cause a wrong
        label, not arbitrary actions. Document this justification
        in a comment at the untaint() call.

     c) Direct interpolation as data, not instructions — the pattern
        used for CI log excerpts. The prompt frames them as log
        output to analyze, not as directives. Document this.

     d) Don't interpolate it at all — if there's no safe way to
        include the data, restructure the code so it doesn't reach
        the prompt.

  3. Add untaint() at the appropriate boundary with a comment
     explaining which pattern (a/b/c) applies and why.

  4. If none of the above fit, the right fix is probably to route
     the data through the sidecar (pattern a).

Adding a bare ``untaint()`` without justification defeats the purpose
of the entire system.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class TaintError(Exception):
    """Raised when tainted data reaches a prompt construction site."""


class TaintedStr(str):
    """A str subclass that tracks untrusted provenance.

    All str operations that produce a new string preserve the taint.
    """

    source: str

    def __new__(cls, value: str = "", *, source: str = "unknown") -> TaintedStr:
        obj = super().__new__(cls, value)
        obj.source = source
        return obj

    def _wrap(self, value: str) -> TaintedStr:
        return TaintedStr(value, source=self.source)

    def __repr__(self) -> str:
        return f"TaintedStr({super().__repr__()}, source={self.source!r})"

    def __add__(self, other: str) -> TaintedStr:
        return self._wrap(super().__add__(other))

    def __radd__(self, other: str) -> TaintedStr:
        return self._wrap(other.__add__(self))

    def __mul__(self, n: int) -> TaintedStr:
        return self._wrap(super().__mul__(n))

    def __rmul__(self, n: int) -> TaintedStr:
        return self._wrap(super().__mul__(n))

    def __mod__(self, args: Any) -> TaintedStr:
        return self._wrap(super().__mod__(args))

    def __getitem__(self, key: Any) -> TaintedStr:
        return self._wrap(super().__getitem__(key))

    def capitalize(self) -> TaintedStr:
        return self._wrap(super().capitalize())

    def casefold(self) -> TaintedStr:
        return self._wrap(super().casefold())

    def center(self, width: int, fillchar: str = " ") -> TaintedStr:
        return self._wrap(super().center(width, fillchar))

    def expandtabs(self, tabsize: int = 8) -> TaintedStr:
        return self._wrap(super().expandtabs(tabsize))

    def format(self, *args: Any, **kwargs: Any) -> TaintedStr:
        return self._wrap(super().format(*args, **kwargs))

    def format_map(self, mapping: Any) -> TaintedStr:
        return self._wrap(super().format_map(mapping))

    def join(self, iterable: Iterable[str]) -> TaintedStr:
        return self._wrap(super().join(iterable))

    def ljust(self, width: int, fillchar: str = " ") -> TaintedStr:
        return self._wrap(super().ljust(width, fillchar))

    def lower(self) -> TaintedStr:
        return self._wrap(super().lower())

    def lstrip(self, chars: str | None = None) -> TaintedStr:
        return self._wrap(super().lstrip(chars))

    def partition(self, sep: str) -> tuple[TaintedStr, TaintedStr, TaintedStr]:
        a, b, c = super().partition(sep)
        return self._wrap(a), self._wrap(b), self._wrap(c)

    def removeprefix(self, prefix: str) -> TaintedStr:
        return self._wrap(super().removeprefix(prefix))

    def removesuffix(self, suffix: str) -> TaintedStr:
        return self._wrap(super().removesuffix(suffix))

    def replace(self, old: str, new: str, count: int = -1) -> TaintedStr:
        return self._wrap(super().replace(old, new, count))

    def rjust(self, width: int, fillchar: str = " ") -> TaintedStr:
        return self._wrap(super().rjust(width, fillchar))

    def rpartition(self, sep: str) -> tuple[TaintedStr, TaintedStr, TaintedStr]:
        a, b, c = super().rpartition(sep)
        return self._wrap(a), self._wrap(b), self._wrap(c)

    def rstrip(self, chars: str | None = None) -> TaintedStr:
        return self._wrap(super().rstrip(chars))

    def split(self, sep: str | None = None, maxsplit: int = -1) -> list[TaintedStr]:
        return [self._wrap(s) for s in super().split(sep, maxsplit)]

    def rsplit(self, sep: str | None = None, maxsplit: int = -1) -> list[TaintedStr]:
        return [self._wrap(s) for s in super().rsplit(sep, maxsplit)]

    def splitlines(self, keepends: bool = False) -> list[TaintedStr]:
        return [self._wrap(s) for s in super().splitlines(keepends)]

    def strip(self, chars: str | None = None) -> TaintedStr:
        return self._wrap(super().strip(chars))

    def swapcase(self) -> TaintedStr:
        return self._wrap(super().swapcase())

    def title(self) -> TaintedStr:
        return self._wrap(super().title())

    def translate(self, table: Any) -> TaintedStr:
        return self._wrap(super().translate(table))

    def upper(self) -> TaintedStr:
        return self._wrap(super().upper())

    def zfill(self, width: int) -> TaintedStr:
        return self._wrap(super().zfill(width))


def taint(value: str, source: str) -> TaintedStr:
    """Mark a string as untrusted."""
    if not isinstance(value, str):
        raise TypeError(f"taint() expects str, got {type(value).__name__}")
    return TaintedStr(value, source=source)


def taint_dict(d: dict, source: str, keys: Iterable[str]) -> dict:
    """Return a copy of *d* with the named string-valued keys tainted."""
    out = dict(d)
    for k in keys:
        v = out.get(k)
        if isinstance(v, str):
            out[k] = taint(v, source)
    return out


def untaint(value: str) -> str:
    """Explicitly declassify a tainted string.

    Every call site is part of the security-critical surface — it must
    have a justification for why the taint can be safely stripped.
    """
    return str(value)


def assert_untainted(*values: str) -> None:
    """Raise TaintError if any argument is tainted."""
    for v in values:
        if isinstance(v, TaintedStr):
            raise TaintError(
                f"tainted string (source={v.source!r}) reached a prompt "
                f"construction site without declassification: {v[:80]!r}..."
            )


def format_untainted(template: str, **kwargs: Any) -> str:
    """``template.format(**kwargs)`` that refuses tainted arguments.

    ``str.format`` on a plain-str template returns a plain str, silently
    laundering any TaintedStr argument (same for f-strings and ``join``).
    Prompt sinks must use this wrapper so a missed untaint() crashes with
    a TaintError instead of leaking undeclared untrusted text.
    """
    assert_untainted(*(v for v in kwargs.values() if isinstance(v, str)))
    return template.format(**kwargs)
