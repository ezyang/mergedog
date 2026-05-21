import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mergedog import claude
from mergedog.config import LLMConfig


class TestInvoke(unittest.TestCase):
    def test_fix_ci_codex_uses_xhigh_reasoning_effort(self):
        with tempfile.TemporaryDirectory() as d:
            worktree = Path(d)

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
                    return_value=LLMConfig("codex"),
                ),
                mock.patch.object(
                    claude, "_run_llm_streaming", return_value=(0, [])
                ) as run_llm,
                mock.patch.object(claude, "_is_clean", return_value=True),
            ):
                result = claude._invoke(
                    worktree,
                    "prompt",
                    mode="fix-CI",
                    expect_merge_commit=False,
                )

        self.assertTrue(result.ran_cleanly)
        self.assertEqual(run_llm.call_args.kwargs["reasoning_effort"], "xhigh")

    def test_merge_resolver_codex_uses_default_reasoning_effort(self):
        with tempfile.TemporaryDirectory() as d:
            worktree = Path(d)

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
                    return_value=LLMConfig("codex"),
                ),
                mock.patch.object(
                    claude, "_run_llm_streaming", return_value=(0, [])
                ) as run_llm,
                mock.patch.object(
                    claude.repo_mod,
                    "is_merge_in_progress",
                    return_value=False,
                ),
                mock.patch.object(claude, "_is_clean", return_value=True),
            ):
                result = claude._invoke(
                    worktree,
                    "prompt",
                    mode="merge-resolver",
                    expect_merge_commit=True,
                )

        self.assertTrue(result.ran_cleanly)
        self.assertIsNone(run_llm.call_args.kwargs["reasoning_effort"])

    def test_too_hard_marker_returns_specific_halt_reason(self):
        with tempfile.TemporaryDirectory() as d:
            worktree = Path(d)
            marker = worktree / ".mergedog-too-hard"
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
            "reported a real PR-related failure that is too hard to fix safely",
        )
        self.assertFalse(marker.exists())

    def test_rebase_marker_returns_specific_halt_reason(self):
        with tempfile.TemporaryDirectory() as d:
            worktree = Path(d)
            marker = worktree / ".mergedog-rebase"
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
            "requested REBASE; refreshing stale base",
        )
        self.assertFalse(marker.exists())

    def test_rebase_resolution_allows_multiple_replayed_commits(self):
        with tempfile.TemporaryDirectory() as d:
            worktree = Path(d)

            with (
                mock.patch.object(
                    claude,
                    "head_sha",
                    side_effect=["a" * 40, "b" * 40],
                ),
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
                mock.patch.object(
                    claude.repo_mod,
                    "is_rebase_in_progress",
                    return_value=False,
                ),
                mock.patch.object(claude, "_is_clean", return_value=True),
                mock.patch.object(claude, "_commits_between", return_value=4),
                mock.patch.object(
                    claude,
                    "head_subject",
                    return_value="[MERGEDOG] Propagate parent update downstream",
                ),
            ):
                result = claude.invoke_rebase_resolver(
                    worktree,
                    "prompt",
                    allow_multiple_commits=True,
                )

        self.assertTrue(result.ran_cleanly)
        self.assertEqual(result.new_sha, "b" * 40)

    def test_cherry_pick_resolution_allows_original_commit_subject(self):
        with tempfile.TemporaryDirectory() as d:
            worktree = Path(d)

            with (
                mock.patch.object(
                    claude,
                    "head_sha",
                    side_effect=["a" * 40, "b" * 40],
                ),
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
                mock.patch.object(
                    claude.repo_mod,
                    "is_cherry_pick_in_progress",
                    return_value=False,
                ),
                mock.patch.object(claude, "_is_clean", return_value=True),
                mock.patch.object(claude, "_commits_between", return_value=1),
                mock.patch.object(
                    claude,
                    "head_subject",
                    return_value="Use PyTorch Min/Max throughout Inductor",
                ),
            ):
                result = claude.invoke_cherry_pick_resolver(worktree, "prompt")

        self.assertTrue(result.ran_cleanly)
        self.assertEqual(result.new_sha, "b" * 40)

    def test_cherry_pick_resolution_rejects_unfinished_cherry_pick(self):
        with tempfile.TemporaryDirectory() as d:
            worktree = Path(d)

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
                mock.patch.object(
                    claude.repo_mod,
                    "is_cherry_pick_in_progress",
                    return_value=True,
                ),
            ):
                result = claude.invoke_cherry_pick_resolver(worktree, "prompt")

        self.assertFalse(result.ran_cleanly)
        self.assertEqual(
            result.halt_reason,
            "exited but the cherry-pick is still in progress; refusing to push",
        )


if __name__ == "__main__":
    unittest.main()
