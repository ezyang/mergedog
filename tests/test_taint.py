"""Tests for the dynamic IFC taint-tracking system."""
from __future__ import annotations

import pytest

from mergedog.taint import (
    TaintedStr,
    TaintError,
    assert_untainted,
    taint,
    taint_dict,
    untaint,
)


class TestTaintedStrPropagation:
    def test_add(self):
        t = taint("hello", "test")
        assert isinstance(t + " world", TaintedStr)

    def test_radd(self):
        t = taint("hello", "test")
        assert isinstance("prefix " + t, TaintedStr)

    def test_mul(self):
        t = taint("ab", "test")
        assert isinstance(t * 3, TaintedStr)
        assert t * 3 == "ababab"

    def test_getitem_slice(self):
        t = taint("hello", "test")
        assert isinstance(t[1:3], TaintedStr)
        assert t[1:3] == "el"

    def test_strip(self):
        t = taint("  hello  ", "test")
        assert isinstance(t.strip(), TaintedStr)
        assert t.strip() == "hello"

    def test_split(self):
        t = taint("a,b,c", "test")
        parts = t.split(",")
        assert all(isinstance(p, TaintedStr) for p in parts)
        assert parts == ["a", "b", "c"]

    def test_splitlines(self):
        t = taint("line1\nline2", "test")
        lines = t.splitlines()
        assert all(isinstance(ln, TaintedStr) for ln in lines)

    def test_replace(self):
        t = taint("hello", "test")
        r = t.replace("l", "r")
        assert isinstance(r, TaintedStr)
        assert r == "herro"

    def test_upper_lower(self):
        t = taint("Hello", "test")
        assert isinstance(t.upper(), TaintedStr)
        assert isinstance(t.lower(), TaintedStr)

    def test_join(self):
        t = taint(",", "test")
        result = t.join(["a", "b", "c"])
        assert isinstance(result, TaintedStr)
        assert result == "a,b,c"

    def test_format(self):
        t = taint("hello {}", "test")
        assert isinstance(t.format("world"), TaintedStr)

    def test_partition(self):
        t = taint("a:b", "test")
        a, sep, b = t.partition(":")
        assert all(isinstance(x, TaintedStr) for x in (a, sep, b))

    def test_source_preserved(self):
        t = taint("hello", "github_api")
        assert t.source == "github_api"
        assert t.upper().source == "github_api"
        assert (t + " world").source == "github_api"


class TestHelpers:
    def test_taint_basic(self):
        t = taint("value", "src")
        assert isinstance(t, TaintedStr)
        assert t == "value"
        assert t.source == "src"

    def test_taint_rejects_non_str(self):
        with pytest.raises(TypeError):
            taint(42, "src")  # type: ignore[arg-type]

    def test_taint_dict(self):
        d = {"title": "my pr", "body": "description", "number": 123}
        result = taint_dict(d, "pr", ["title", "body"])
        assert isinstance(result["title"], TaintedStr)
        assert isinstance(result["body"], TaintedStr)
        assert not isinstance(result["number"], TaintedStr)
        assert d is not result  # returns a copy

    def test_taint_dict_missing_key(self):
        d = {"title": "my pr"}
        result = taint_dict(d, "pr", ["title", "nonexistent"])
        assert isinstance(result["title"], TaintedStr)
        assert "nonexistent" not in result

    def test_untaint(self):
        t = taint("hello", "test")
        clean = untaint(t)
        assert not isinstance(clean, TaintedStr)
        assert isinstance(clean, str)
        assert clean == "hello"

    def test_untaint_plain_str(self):
        clean = untaint("already clean")
        assert clean == "already clean"


class TestAssertUntainted:
    def test_passes_for_clean_strings(self):
        assert_untainted("a", "b", "c")

    def test_raises_for_tainted(self):
        t = taint("injected", "attacker")
        with pytest.raises(TaintError, match="attacker"):
            assert_untainted("clean", t)

    def test_error_message_includes_source(self):
        t = taint("payload", "pr_comment")
        with pytest.raises(TaintError) as exc_info:
            assert_untainted(t)
        assert "pr_comment" in str(exc_info.value)


class TestStrInterop:
    def test_equality_with_plain_str(self):
        t = taint("hello", "test")
        assert t == "hello"
        assert "hello" == t

    def test_hash_matches_plain_str(self):
        t = taint("hello", "test")
        assert hash(t) == hash("hello")

    def test_in_frozenset(self):
        t = taint("pytorch-bot", "test")
        s = frozenset({"pytorch-bot", "other"})
        assert t in s

    def test_in_check_with_str(self):
        t = taint("hello world", "test")
        assert "world" in t

    def test_bool(self):
        assert bool(taint("nonempty", "test"))
        assert not bool(taint("", "test"))

    def test_len(self):
        assert len(taint("hello", "test")) == 5
