import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mergedog import claude
from mergedog.config import LLMConfig


class TestInvoke(unittest.TestCase):
    def test_real_failure_marker_returns_specific_halt_reason(self):
        with tempfile.TemporaryDirectory() as d:
            worktree = Path(d)
            marker = worktree / ".mergedog-real-failure"
            marker.touch()

            with (
                mock.patch.object(claude, "head_sha", return_value="a" * 40),
                mock.patch.object(
                    claude.repo_mod,
                    "get_mergedog_identity",
                    return_value=("mergedog", "mergedog@example.com"),
                ),
                mock.patch.object(
                    claude.repo_mod,
                    "author_env",
                    return_value={},
                ),
                mock.patch.object(
                    claude,
                    "get_llm_config",
                    return_value=LLMConfig("claude"),
                ),
                mock.patch.object(claude, "_run_llm_streaming", return_value=(0, [])),
                mock.patch.object(claude, "_is_clean", return_value=True),
            ):
                result = claude._invoke(
                    worktree,
                    "prompt",
                    mode="fix-CI",
                    expect_merge_commit=False,
                )

        self.assertFalse(result.ran_cleanly)
        self.assertEqual(
            result.halt_reason,
            "reported a real PR-related failure it could not safely fix",
        )
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
