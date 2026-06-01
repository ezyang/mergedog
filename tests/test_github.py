import subprocess
import unittest
from unittest import mock

from mergedog import github


class TestGhRetries(unittest.TestCase):
    def setUp(self):
        self.proxy_patcher = mock.patch.object(
            github, "github_api_env_extra", return_value=None
        )
        self.proxy_patcher.start()
        self.addCleanup(self.proxy_patcher.stop)

    def test_connection_error_is_transient(self):
        proc = subprocess.CompletedProcess(
            ["gh"], 1, "", "error connecting to api.github.com\n"
        )

        self.assertTrue(github._is_transient_gh_failure(proc))

    def test_unexpected_eof_is_transient(self):
        proc = subprocess.CompletedProcess(
            ["gh"],
            1,
            "",
            'Post "https://api.github.com/graphql": unexpected EOF\n',
        )

        self.assertTrue(github._is_transient_gh_failure(proc))

    def test_go_runtime_startup_crash_is_transient(self):
        proc = subprocess.CompletedProcess(
            ["gh"],
            2,
            "",
            "runtime: lfstack.push invalid packing\nfatal error: lfstack.push\n",
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

    def test_logs_stderr_after_transient_retries_exhausted(self):
        calls = [
            subprocess.CompletedProcess(
                ["gh"],
                1,
                "",
                "error connecting to api.github.com\ncheck your internet connection\n",
            )
            for _ in range(github._GH_MAX_RETRIES)
        ]

        with mock.patch.object(github, "run", side_effect=calls), mock.patch.object(
            github.time, "sleep"
        ) as sleep, mock.patch.object(github, "log") as log:
            with self.assertRaises(subprocess.CalledProcessError):
                github._gh(["pr", "view", "1"])

        self.assertEqual(sleep.call_count, github._GH_MAX_RETRIES - 1)
        log.assert_any_call("  ! gh transient failure after 3 attempts")
        log.assert_any_call("  ! gh pr view 1")
        log.assert_any_call("    stderr: error connecting to api.github.com")
        log.assert_any_call("    stderr: check your internet connection")

    def test_uses_working_alternate_after_startup_crash(self):
        old_command = github._GH_COMMAND
        github._GH_COMMAND = ["gh"]
        calls = [
            subprocess.CompletedProcess(
                ["gh"],
                2,
                "",
                "runtime: lfstack.push invalid packing\nfatal error: lfstack.push\n",
            ),
            subprocess.CompletedProcess(["/usr/local/bin/gh"], 0, "{}", ""),
        ]

        try:
            with mock.patch.object(
                github, "run", side_effect=calls
            ) as run, mock.patch.object(
                github, "_find_working_gh_executable", return_value="/usr/local/bin/gh"
            ), mock.patch.object(
                github.shutil, "which", return_value="/broken/gh"
            ), mock.patch.object(
                github, "log"
            ) as log:
                proc = github._gh(["pr", "view", "1"])

            self.assertEqual(github._GH_COMMAND, ["/usr/local/bin/gh"])
        finally:
            github._GH_COMMAND = old_command

        self.assertEqual(proc.returncode, 0)
        run.assert_has_calls(
            [
                mock.call(["gh", "pr", "view", "1"], check=False, loud=False),
                mock.call(
                    ["/usr/local/bin/gh", "pr", "view", "1"],
                    check=False,
                    loud=False,
                ),
            ]
        )
        log.assert_any_call(
            "  ! gh startup crash from /broken/gh; "
            "retrying with /usr/local/bin/gh"
        )

    def test_passes_proxy_env_to_gh_when_configured(self):
        env_extra = {
            "HTTPS_PROXY": "http://proxy.example",
            "HTTP_PROXY": "http://proxy.example",
        }
        with mock.patch.object(
            github, "github_api_env_extra", return_value=env_extra
        ), mock.patch.object(
            github,
            "run",
            return_value=subprocess.CompletedProcess(["gh"], 0, "{}", ""),
        ) as run:
            github._gh(["pr", "view", "1"])

        run.assert_called_once_with(
            ["gh", "pr", "view", "1"],
            check=False,
            env_extra=env_extra,
            loud=False,
        )

    def test_post_pr_comment_uses_retrying_gh_wrapper_with_stdin(self):
        with mock.patch.object(github, "_gh") as gh:
            github.post_pr_comment(123, "body text")

        gh.assert_called_once_with(
            ["pr", "comment", "123", "--repo", github.REPO, "--body-file", "-"],
            input_text="body text",
            log_context="posting PR comment",
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
