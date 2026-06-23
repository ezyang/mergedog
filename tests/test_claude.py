import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mergedog import claude
from mergedog.config import LLMConfig


class TestInvoke(unittest.TestCase):
    def test_codex_prompt_is_passed_on_stdin(self):
        with tempfile.TemporaryDirectory() as d:
            prompt = "x" * 200_000

            invocation = claude._build_llm_invocation(
                prompt, Path(d), LLMConfig("codex")
            )

        self.assertEqual(invocation.provider, "codex")
        self.assertEqual(invocation.stdin_input, prompt)
        self.assertNotIn(prompt, invocation.cmd)

    def test_llm_start_failure_is_returned_without_traceback(self):
        err = OSError(7, "Argument list too long", "codex")
        with tempfile.TemporaryDirectory() as d, mock.patch.object(
            claude.subprocess, "Popen", side_effect=err
        ):
            rc, transcript = claude._run_llm_streaming(
                "prompt", Path(d), {}, LLMConfig("codex")
            )

        self.assertEqual(rc, 127)
        self.assertEqual(
            transcript,
            ["codex failed to start: [Errno 7] Argument list too long: 'codex'"],
        )

    def test_start_failure_uses_specific_halt_reason(self):
        with tempfile.TemporaryDirectory() as d:
            worktree = Path(d)

            with (
                mock.patch.object(claude, "head_sha", return_value="a" * 40),
                mock.patch.object(
                    claude.repo_mod,
                    "get_mergedog_identity",
                    return_value=("mergedog", "mergedog@example.com"),
                ),
                mock.patch.object(claude.repo_mod, "author_env", return_value={}),
                mock.patch.object(
                    claude, "get_llm_config", return_value=LLMConfig("codex")
                ),
                mock.patch.object(
                    claude,
                    "_run_llm_streaming",
                    return_value=(
                        127,
                        [
                            "codex failed to start: [Errno 7] "
                            "Argument list too long: 'codex'"
                        ],
                    ),
                ),
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
            "failed to start: [Errno 7] Argument list too long: 'codex'",
        )

    def test_fix_ci_no_commit_without_spurious_marker_is_contract_violation(self):
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
            "made no commit without signalling .mergedog-spurious; "
            "refusing to mark CI failures spurious",
        )

    def test_stale_spurious_marker_does_not_allow_silent_noop(self):
        with tempfile.TemporaryDirectory() as d:
            worktree = Path(d)
            marker = worktree / claude.SPURIOUS_MARKER
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
                    return_value=LLMConfig("codex"),
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
        self.assertFalse(marker.exists())

    def test_fix_ci_spurious_marker_allows_no_commit(self):
        with tempfile.TemporaryDirectory() as d:
            worktree = Path(d)
            marker = worktree / claude.SPURIOUS_MARKER

            def run_llm(*args, **kwargs):
                marker.write_text("lint is unrelated to this PR\n")
                return 0, []

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
                mock.patch.object(claude, "_run_llm_streaming", side_effect=run_llm),
                mock.patch.object(claude, "_is_clean", return_value=True),
            ):
                result = claude._invoke(
                    worktree,
                    "prompt",
                    mode="fix-CI",
                    expect_merge_commit=False,
                )

        self.assertTrue(result.ran_cleanly)
        self.assertIsNone(result.new_sha)
        self.assertEqual(result.spurious_reason, "lint is unrelated to this PR")
        self.assertFalse(marker.exists())

    def test_operator_fix_still_allows_already_satisfied_noop(self):
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
                mock.patch.object(claude, "_run_llm_streaming", return_value=(0, [])),
                mock.patch.object(claude, "_is_clean", return_value=True),
            ):
                result = claude._invoke(
                    worktree,
                    "prompt",
                    mode="operator-fix",
                    expect_merge_commit=False,
                )

        self.assertTrue(result.ran_cleanly)
        self.assertIsNone(result.new_sha)

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
