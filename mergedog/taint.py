"""Dynamic information flow control for untrusted strings.

Strings from external sources (GitHub API responses) are wrapped in
TaintedStr. Prompt construction sites call assert_untainted() to crash
if tainted data leaks through without explicit declassification via
untaint().
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
