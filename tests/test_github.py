import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mergedog import github


class TestGhRetries(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.api_log = Path(self.tmp.name) / "gh-api-calls.jsonl"
        self.api_log_patcher = mock.patch.object(
            github, "GH_API_CALLS_LOG", self.api_log
        )
        self.api_log_patcher.start()
        self.addCleanup(self.api_log_patcher.stop)
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

    def test_records_gh_attempt_attribution(self):
        with mock.patch.object(
            github,
            "run",
            return_value=subprocess.CompletedProcess(["gh"], 0, "{}", ""),
        ):
            github._gh(
                [
                    "api",
                    f"repos/{github.REPO}/commits/abc/check-runs"
                    "?per_page=100&page=2",
                ]
            )

        rows = [json.loads(line) for line in self.api_log.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["operation"], "api:check-runs")
        self.assertEqual(row["repo"], github.REPO)
        self.assertEqual(row["sha"], "abc")
        self.assertEqual(row["page"], 2)
        self.assertEqual(row["attempt"], 1)
        self.assertEqual(row["exit_code"], 0)
        self.assertIsNone(row["github_function"])
        self.assertIn("tests.test_github", row["caller"])

    def test_classifies_graphql_without_logging_query_body(self):
        fields = github._classify_gh_args(
            [
                "api",
                "graphql",
                "-f",
                "query=query($pr:Int!){ viewer { login } }",
                "-F",
                "pr=123",
            ]
        )

        self.assertEqual(fields["operation"], "api:graphql")
        self.assertEqual(fields["pr"], 123)
        self.assertIn("query=<redacted>", fields["args"])

    def test_post_pr_comment_uses_retrying_gh_wrapper_with_stdin(self):
        with mock.patch.object(github, "_gh") as gh:
            github.post_pr_comment(123, "body text")

        gh.assert_called_once_with(
            ["pr", "comment", "123", "--repo", github.REPO, "--body-file", "-"],
            input_text="body text",
            log_context="posting PR comment",
        )

    def test_get_pr_review_comments_uses_rest_endpoint(self):
        with mock.patch.object(
            github,
            "_gh_json",
            side_effect=[
                [
                    {
                        "body": "marker",
                        "commit_id": "abc",
                        "path": "foo.py",
                        "line": 7,
                        "side": "RIGHT",
                    }
                ],
                [],
            ],
        ) as gh_json:
            comments = github.get_pr_review_comments(123, per_page=1)

        self.assertEqual(
            comments,
            [
                {
                    "body": "marker",
                    "commit_id": "abc",
                    "path": "foo.py",
                    "line": 7,
                    "side": "RIGHT",
                }
            ],
        )
        self.assertEqual(
            gh_json.call_args_list[0].args[0],
            [
                "api",
                f"repos/{github.REPO}/pulls/123/comments?per_page=1&page=1",
            ],
        )

    def test_post_pr_review_comment_uses_rest_endpoint(self):
        with mock.patch.object(github, "_gh") as gh:
            github.post_pr_review_comment(
                123,
                body="body text",
                commit_id="abc",
                path="foo.py",
                line=7,
                side="RIGHT",
            )

        gh.assert_called_once()
        args, kwargs = gh.call_args
        self.assertEqual(
            args[0],
            [
                "api",
                "-X",
                "POST",
                f"repos/{github.REPO}/pulls/123/comments",
                "--input",
                "-",
            ],
        )
        self.assertEqual(
            json.loads(kwargs["input_text"]),
            {
                "body": "body text",
                "commit_id": "abc",
                "path": "foo.py",
                "line": 7,
                "side": "RIGHT",
            },
        )
        self.assertEqual(kwargs["log_context"], "posting PR review comment")

    def test_add_label_uses_rest_issue_endpoint(self):
        with mock.patch.object(github, "_gh") as gh:
            github.add_label(123, "topic: bug fixes", loud=False)

        gh.assert_called_once_with(
            [
                "api",
                "-X",
                "POST",
                f"repos/{github.REPO}/issues/123/labels",
                "--input",
                "-",
            ],
            input_text='{"labels": ["topic: bug fixes"]}',
            loud=False,
        )

    def test_remove_label_uses_rest_issue_endpoint(self):
        with mock.patch.object(github, "_gh") as gh:
            github.remove_label(123, "topic: bug fixes")

        gh.assert_called_once_with(
            [
                "api",
                "-X",
                "DELETE",
                f"repos/{github.REPO}/issues/123/labels/topic%3A%20bug%20fixes",
            ],
            loud=True,
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


class TestPrPollFields(unittest.TestCase):
    def test_combines_status_and_head_fields(self):
        with mock.patch.object(
            github,
            "_gh_json",
            return_value={
                "labels": [{"name": "ciflow/trunk"}],
                "reviewDecision": "APPROVED",
                "headRefOid": "abc",
            },
        ) as gh_json:
            self.assertEqual(
                github.get_pr_poll_fields(123),
                (["ciflow/trunk"], "APPROVED", "abc"),
            )

        self.assertEqual(
            gh_json.call_args.args[0],
            [
                "pr",
                "view",
                "123",
                "--repo",
                github.REPO,
                "--json",
                "labels,reviewDecision,headRefOid",
            ],
        )


class TestPrChecksFallback(unittest.TestCase):
    def test_uses_gh_pr_checks_when_present(self):
        checks = [{"name": "pull", "bucket": "pass"}]

        with mock.patch.object(
            github, "_gh_pr_checks_json", return_value=checks
        ), mock.patch.object(github, "get_pr_head_sha") as head:
            self.assertIs(github.get_pr_checks_all(123), checks)

        head.assert_not_called()

    def test_falls_back_to_rest_check_runs(self):
        check_runs = [
            {
                "name": "linux",
                "status": "completed",
                "conclusion": "failure",
                "html_url": "https://github.com/pytorch/pytorch/actions/runs/1/job/2",
                "completed_at": "2026-06-01T01:02:03Z",
            },
            {
                "name": "macos",
                "status": "in_progress",
                "conclusion": None,
                "html_url": "https://github.com/pytorch/pytorch/actions/runs/3/job/4",
                "completed_at": None,
            },
        ]

        with (
            mock.patch.object(github, "_gh_pr_checks_json", return_value=[]),
            mock.patch.object(github, "get_pr_head_sha", return_value="abc"),
            mock.patch.object(
                github, "list_check_runs_for_sha", return_value=check_runs
            ),
            mock.patch.object(github, "list_workflow_runs_for_sha") as workflows,
            mock.patch.object(github, "log"),
        ):
            self.assertEqual(
                github.get_pr_checks_all(123),
                [
                    {
                        "name": "linux",
                        "state": "FAILURE",
                        "workflow": "",
                        "link": (
                            "https://github.com/pytorch/pytorch/actions/runs/1/job/2"
                        ),
                        "bucket": "fail",
                        "completedAt": "2026-06-01T01:02:03Z",
                    },
                    {
                        "name": "macos",
                        "state": "PENDING",
                        "workflow": "",
                        "link": (
                            "https://github.com/pytorch/pytorch/actions/runs/3/job/4"
                        ),
                        "bucket": "pending",
                        "completedAt": "",
                    },
                ],
            )

        workflows.assert_not_called()

    def test_falls_back_to_workflow_runs_when_check_runs_empty(self):
        workflow_runs = [
            {
                "id": 10,
                "name": "trunk",
                "status": "completed",
                "conclusion": "cancelled",
                "html_url": "https://github.com/pytorch/pytorch/actions/runs/10",
                "updated_at": "2026-06-01T01:02:03Z",
            }
        ]

        with (
            mock.patch.object(github, "_gh_pr_checks_json", return_value=[]),
            mock.patch.object(github, "get_pr_head_sha", return_value="abc"),
            mock.patch.object(github, "list_check_runs_for_sha", return_value=[]),
            mock.patch.object(
                github, "list_workflow_runs_for_sha", return_value=workflow_runs
            ),
            mock.patch.object(github, "log"),
        ):
            self.assertEqual(
                github.get_pr_checks_all(123),
                [
                    {
                        "name": "trunk",
                        "state": "CANCELLED",
                        "workflow": "trunk",
                        "link": "https://github.com/pytorch/pytorch/actions/runs/10",
                        "bucket": "cancel",
                        "completedAt": "2026-06-01T01:02:03Z",
                    }
                ],
            )

    def test_fallback_reuses_supplied_head_sha(self):
        with (
            mock.patch.object(github, "_gh_pr_checks_json", return_value=[]),
            mock.patch.object(github, "get_pr_head_sha") as head,
            mock.patch.object(github, "list_check_runs_for_sha", return_value=[]),
            mock.patch.object(github, "list_workflow_runs_for_sha", return_value=[]),
        ):
            self.assertEqual(github.get_pr_checks_all(123, head_sha="abc"), [])

        head.assert_not_called()


if __name__ == "__main__":
    unittest.main()
