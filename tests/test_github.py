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


if __name__ == "__main__":
    unittest.main()
