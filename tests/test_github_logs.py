import threading
import unittest
from unittest import mock

from mergedog import github
from mergedog.github import (
    _strip_gh_log_prefix,
    _trim_log_for_prompt,
    latest_drci_summary,
)


class TestStripPrefix(unittest.TestCase):
    def test_strips_job_step_timestamp(self):
        line = (
            "linux-jammy-py3.10-clang18-asan / build\tUNKNOWN STEP\t"
            "2026-05-04T18:20:18.5493257Z FAILED: foo.o"
        )
        self.assertEqual(_strip_gh_log_prefix(line), "FAILED: foo.o")

    def test_no_prefix_passthrough(self):
        self.assertEqual(_strip_gh_log_prefix("plain line"), "plain line")

    def test_no_timestamp_keeps_content(self):
        line = "job\tstep\tno timestamp here"
        self.assertEqual(_strip_gh_log_prefix(line), "no timestamp here")

    def test_two_tabs_only(self):
        # Only two tabs (job + content, no step) — shouldn't crash.
        self.assertEqual(_strip_gh_log_prefix("a\tb"), "a\tb")


def _make_log(prefix_lines: int, error_block: str, tail_lines: int) -> str:
    """Build a synthetic ``gh run view --log`` shaped string."""
    job_prefix = "job-x / build\tUNKNOWN STEP\t2026-05-04T18:00:00.0000000Z "
    head = "\n".join(
        f"{job_prefix}preamble line {i}" for i in range(prefix_lines)
    )
    err = "\n".join(f"{job_prefix}{line}" for line in error_block.splitlines())
    tail = "\n".join(
        f"{job_prefix}post-job cleanup noise {i}" for i in range(tail_lines)
    )
    return "\n".join([head, err, tail])


class TestTrimLog(unittest.TestCase):
    def test_short_log_passthrough_minus_prefix(self):
        text = "job\tstep\t2026-05-04T18:20:18Z FAILED: build broken\n"
        out = _trim_log_for_prompt(text, max_chars=10_000)
        self.assertIn("FAILED: build broken", out)
        # The job/step/timestamp prefix should be gone.
        self.assertNotIn("UNKNOWN STEP", out)
        self.assertNotIn("2026-05-04T18:20:18Z", out)

    def test_window_around_late_marker(self):
        # Mimics the 182115 shape: real error well before tons of cleanup.
        # Preamble has to be big enough to push the marker past the
        # before-the-marker budget so we see head_truncated.
        log = _make_log(
            prefix_lines=10_000,
            error_block=(
                "FAILED: caffe2/init.cpp.o\n"
                "init.cpp:4046:40: error: 'c10d::ProcessGroupNCCL' has not been declared\n"
                "ninja: build stopped: subcommand failed."
            ),
            tail_lines=40_000,
        )
        out = _trim_log_for_prompt(log, max_chars=4_000)
        self.assertLessEqual(len(out), 4_500)
        self.assertIn("error:", out)
        self.assertIn("ProcessGroupNCCL", out)
        # We skipped the bulk of both preamble and tail.
        self.assertIn("[head truncated]", out)
        self.assertIn("[tail truncated]", out)

    def test_pytest_failures_section_extracted(self):
        prefix = "job\tstep\t2026-05-04T18:00:00Z "
        preamble = "\n".join(f"{prefix}PASS test_{i}" for i in range(5_000))
        failures = "\n".join(
            f"{prefix}{line}"
            for line in [
                "= FAILURES =",
                "___ test_redispatch_as_strided ___",
                "    def test_redispatch_as_strided(self):",
                "        x = torch.randn(4)",
                ">       self.assertEqual(expect, actual)",
                "E       AssertionError: tensor not close",
                "",
                "= short test summary info =",
                "FAILED test/test_overrides.py::test_redispatch_as_strided",
                "= 1 failed, 200 passed =",
            ]
        )
        log = f"{preamble}\n{failures}"
        out = _trim_log_for_prompt(log, max_chars=4_000)
        self.assertIn("= FAILURES =", out)
        self.assertIn("AssertionError: tensor not close", out)
        self.assertIn("1 failed", out)
        self.assertIn("[head truncated]", out)
        # Preamble PASS lines should be gone.
        self.assertNotIn("PASS test_4999", out)

    def test_no_marker_falls_back_to_head_and_tail(self):
        # No failure markers at all — just lots of unrelated lines.
        prefix = "job\tstep\t2026-05-04T18:00:00Z "
        body = "\n".join(f"{prefix}line {i}" for i in range(20_000))
        out = _trim_log_for_prompt(body, max_chars=2_000)
        self.assertIn("[middle truncated]", out)
        # Should contain both an early line and a late line.
        self.assertIn("line 0", out)
        self.assertIn("line 19999", out)


class TestFetchFailedJobLogs(unittest.TestCase):
    def test_fetches_multiple_failed_logs_in_parallel(self):
        checks = [
            {
                "name": "job one",
                "state": "FAILURE",
                "link": "https://github.com/pytorch/pytorch/actions/runs/1/job/101",
            },
            {
                "name": "job two",
                "state": "FAILURE",
                "link": "https://github.com/pytorch/pytorch/actions/runs/2/job/102",
            },
        ]
        second_started = threading.Event()
        lock = threading.Lock()
        started: list[int | None] = []

        def fake_fetch(run_id: int, job_id: int | None) -> str:
            with lock:
                started.append(job_id)
                is_first_fetch = len(started) == 1
                if len(started) == 2:
                    second_started.set()
            if is_first_fetch:
                self.assertTrue(
                    second_started.wait(1.0),
                    "second log fetch did not start while first was running",
                )
            return f"FAILED: job {job_id}"

        with (
            mock.patch.object(github, "get_pr_checks_all", return_value=checks),
            mock.patch.object(github, "_fetch_job_log", side_effect=fake_fetch),
            mock.patch.object(github, "log"),
        ):
            out = github.get_failed_job_logs(123, max_jobs=2, max_chars=1000)

        self.assertEqual(
            [(str(name), str(text)) for name, text in out],
            [("job one", "FAILED: job 101"), ("job two", "FAILED: job 102")],
        )


class TestLatestDrciSummary(unittest.TestCase):
    def test_picks_latest_pytorch_bot_with_trailer(self):
        comments = [
            {"author": "pytorch-bot", "body": "old\n\nThis comment was automatically generated by Dr. CI and updates every 15 minutes."},
            {"author": "ezyang", "body": "lgtm"},
            {"author": "pytorch-bot", "body": "newer\n\nThis comment was automatically generated by Dr. CI and updates every 15 minutes."},
        ]
        out = __import__("mergedog.github", fromlist=["latest_drci_summary"]).latest_drci_summary(comments)
        self.assertIsNotNone(out)
        self.assertIn("newer", out)

    def test_returns_none_when_no_drci(self):
        self.assertIsNone(latest_drci_summary([]))
        self.assertIsNone(
            latest_drci_summary([{"author": "ezyang", "body": "hi"}])
        )

    def test_ignores_pytorch_bot_without_trailer(self):
        # pytorch-bot can in principle post non-dr.ci comments. Without the
        # trailer we don't trust it as a failure summary.
        comments = [
            {"author": "pytorch-bot", "body": "some unrelated bot comment"},
        ]
        self.assertIsNone(latest_drci_summary(comments))

    def test_drops_summary_when_sha_does_not_match(self):
        # dr. ci updates every ~15 min, so a freshly-pushed head can have
        # a stale "no failures" summary referring to the previous head.
        body = (
            "## :hourglass_flowing_sand: No Failures, 72 Pending\n"
            "As of commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa with merge base bbbb...\n"
            "\nThis comment was automatically generated by Dr. CI and updates every 15 minutes."
        )
        comments = [{"author": "pytorch-bot", "body": body}]
        # Different head -> dropped.
        self.assertIsNone(
            latest_drci_summary(comments, head_sha="cccccccccccccccccccccccccccccccccccccccc")
        )
        # Matching head -> returned. We accept any prefix match so that
        # an abbreviated SHA in the body still resolves against a full
        # 40-char SHA from gh.
        self.assertIsNotNone(
            latest_drci_summary(comments, head_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        )

    def test_returns_summary_with_no_sha_block(self):
        # If we can't find an "As of commit" line at all, treat the body
        # as untrusted (drop it) when a head_sha was supplied.
        body = (
            "no commit reference here\n"
            "\nThis comment was automatically generated by Dr. CI and updates every 15 minutes."
        )
        comments = [{"author": "pytorch-bot", "body": body}]
        self.assertIsNone(latest_drci_summary(comments, head_sha="aaaa"))
        # Without a head_sha, we keep the old behavior (return whatever's there).
        self.assertIsNotNone(latest_drci_summary(comments))


if __name__ == "__main__":
    unittest.main()
