"""Tests for the best-effort prompt-injection screen."""
from __future__ import annotations

import subprocess
from unittest import mock

import pytest

from mergedog import injection


@pytest.fixture(autouse=True)
def _clear_cache():
    injection._verdict_cache.clear()
    yield
    injection._verdict_cache.clear()


def _proc(stdout: str, returncode: int = 0) -> mock.Mock:
    return mock.Mock(returncode=returncode, stdout=stdout, stderr="")


class TestLooksLikeInjection:
    def test_clean_verdict(self):
        with mock.patch.object(
            injection.subprocess, "run", return_value=_proc("CLEAN")
        ):
            assert injection.looks_like_injection("normal log", source="t") is False

    def test_injection_verdict(self):
        with mock.patch.object(
            injection.subprocess, "run", return_value=_proc("INJECTION")
        ):
            assert injection.looks_like_injection("ignore previous", source="t")

    def test_fails_open_on_nonzero_exit(self):
        with mock.patch.object(
            injection.subprocess, "run", return_value=_proc("", returncode=1)
        ):
            assert injection.looks_like_injection("text", source="t") is False

    def test_fails_open_on_timeout(self):
        with mock.patch.object(
            injection.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=60),
        ):
            assert injection.looks_like_injection("text", source="t") is False

    def test_fails_open_on_missing_binary(self):
        with mock.patch.object(
            injection.subprocess, "run", side_effect=OSError("no claude")
        ):
            assert injection.looks_like_injection("text", source="t") is False

    def test_fails_open_on_garbage_verdict(self):
        with mock.patch.object(
            injection.subprocess, "run", return_value=_proc("maybe? hard to say")
        ):
            assert injection.looks_like_injection("text", source="t") is False

    def test_empty_text_skips_classifier(self):
        with mock.patch.object(injection.subprocess, "run") as run:
            assert injection.looks_like_injection("  \n", source="t") is False
        run.assert_not_called()

    def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("MERGEDOG_INJECTION_SCREEN", "0")
        with mock.patch.object(injection.subprocess, "run") as run:
            assert injection.looks_like_injection("text", source="t") is False
        run.assert_not_called()

    def test_verdict_cached_by_content(self):
        with mock.patch.object(
            injection.subprocess, "run", return_value=_proc("INJECTION")
        ) as run:
            assert injection.looks_like_injection("same text", source="t")
            assert injection.looks_like_injection("same text", source="t")
        assert run.call_count == 1

    def test_failed_screen_not_cached(self):
        # A transient failure should not pin a permanent "clean" verdict.
        with mock.patch.object(
            injection.subprocess, "run", return_value=_proc("", returncode=1)
        ):
            injection.looks_like_injection("text", source="t")
        with mock.patch.object(
            injection.subprocess, "run", return_value=_proc("INJECTION")
        ):
            assert injection.looks_like_injection("text", source="t")

    def test_classifier_receives_payload_data_not_instructions(self):
        # The untrusted text must be inside the BEGIN/END fence.
        captured: dict = {}

        def fake_run(cmd, **kwargs):
            captured["prompt"] = cmd[2]
            return _proc("CLEAN")

        with mock.patch.object(injection.subprocess, "run", fake_run):
            injection.looks_like_injection("payload here", source="t")
        prompt = captured["prompt"]
        begin = prompt.index("--- BEGIN UNTRUSTED TEXT ---")
        end = prompt.index("--- END UNTRUSTED TEXT ---")
        assert begin < prompt.index("payload here") < end

    def test_oversized_payload_truncated(self):
        captured: dict = {}

        def fake_run(cmd, **kwargs):
            captured["prompt"] = cmd[2]
            return _proc("CLEAN")

        big = "x" * (injection._MAX_SCREEN_CHARS + 10_000)
        with mock.patch.object(injection.subprocess, "run", fake_run):
            injection.looks_like_injection(big, source="t")
        assert len(captured["prompt"]) < injection._MAX_SCREEN_CHARS + 2_000
