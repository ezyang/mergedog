import subprocess
import unittest
from unittest import mock

from mergedog import github


class TestGhRetries(unittest.TestCase):
    def test_connection_error_is_transient(self):
        proc = subprocess.CompletedProcess(
            ["gh"], 1, "", "error connecting to api.github.com\n"
        )

        self.assertTrue(github._is_transient_gh_failure(proc))

    def test_logs_recovery_after_retry_success(self):
        calls = [
            subprocess.CompletedProcess(["gh"], 1, "", "HTTP 503"),
            subprocess.CompletedProcess(["gh"], 0, "{}", ""),
        ]

        with mock.patch.object(github, "run", side_effect=calls), mock.patch.object(
            github.time, "sleep"
        ), mock.patch.object(github, "log") as log:
            proc = github._gh(["pr", "view", "1"])

        self.assertEqual(proc.returncode, 0)
        log.assert_any_call(
            "  ! gh transient failure (attempt 1/3), retrying in 5s"
        )
        log.assert_any_call("  gh recovered after transient failure")

    def test_logs_recovery_context_after_retry_success(self):
        calls = [
            subprocess.CompletedProcess(["gh"], 1, "", "HTTP 503"),
            subprocess.CompletedProcess(["gh"], 0, "{}", ""),
        ]

        with mock.patch.object(github, "run", side_effect=calls), mock.patch.object(
            github.time, "sleep"
        ), mock.patch.object(github, "log") as log:
            proc = github._gh(["pr", "view", "1"], log_context="watching post-handoff")

        self.assertEqual(proc.returncode, 0)
        log.assert_any_call(
            "  gh recovered after transient failure while watching post-handoff"
        )


class TestPrMergeCommit(unittest.TestCase):
    def test_returns_merge_commit_for_merged_pr(self):
        with mock.patch.object(
            github,
            "_gh_json",
            return_value={"state": "MERGED", "mergeCommit": {"oid": "abc123"}},
        ):
            self.assertEqual(github.get_pr_merge_commit_sha(1), "abc123")

    def test_returns_none_for_open_pr(self):
        with mock.patch.object(
            github,
            "_gh_json",
            return_value={"state": "OPEN", "mergeCommit": None},
        ):
            self.assertIsNone(github.get_pr_merge_commit_sha(1))


if __name__ == "__main__":
    unittest.main()
