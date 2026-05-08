import unittest
from datetime import datetime, timezone
from unittest import mock

from mergedog import shepherd
from mergedog.shepherd import (
    MIN_USEFUL_LOG_CHARS,
    _failed_logs_are_content_free,
    _latest_completed_at,
    _spurious_check_names_from_checks,
    describe_log_state,
)


class TestMergedogLabelManagement(unittest.TestCase):
    def _run_shepherd_wrapper(self, **kwargs):
        pr_data = {"number": 123, "labels": [], "isDraft": False, "state": "OPEN"}
        with mock.patch.object(shepherd.repo, "ensure_clone"), mock.patch.object(
            shepherd.repo, "fetch_origin"
        ), mock.patch.object(
            shepherd.github, "get_pr", return_value=pr_data
        ), mock.patch.object(
            shepherd, "_validate_pr"
        ), mock.patch.object(
            shepherd, "_shepherd_body"
        ), mock.patch.object(
            shepherd.signal, "signal"
        ), mock.patch.object(
            shepherd.faulthandler, "enable"
        ), mock.patch.object(
            shepherd.faulthandler, "register"
        ), mock.patch.object(
            shepherd.github, "add_label"
        ) as add_label, mock.patch.object(
            shepherd.github, "remove_label"
        ) as remove_label:
            shepherd.shepherd(123, **kwargs)
        return add_label, remove_label

    def test_default_does_not_touch_mergedog_label(self):
        add_label, remove_label = self._run_shepherd_wrapper()
        add_label.assert_not_called()
        remove_label.assert_not_called()

    def test_explicit_flag_adds_and_removes_mergedog_label(self):
        add_label, remove_label = self._run_shepherd_wrapper(
            manage_mergedog_label=True
        )
        add_label.assert_called_once_with(123, shepherd.MERGEDOG_LABEL)
        remove_label.assert_called_once_with(123, shepherd.MERGEDOG_LABEL)


class TestFailedLogsAreContentFree(unittest.TestCase):
    def test_empty_list_is_content_free(self):
        self.assertTrue(_failed_logs_are_content_free([]))

    def test_no_log_available_placeholder_is_content_free(self):
        self.assertTrue(
            _failed_logs_are_content_free(
                [
                    ("a", "<no log available>"),
                    ("b", "<no log available>"),
                ]
            )
        )

    def test_long_log_is_substantial(self):
        self.assertFalse(
            _failed_logs_are_content_free(
                [("a", "x" * (MIN_USEFUL_LOG_CHARS + 1))]
            )
        )

    def test_short_stub_below_threshold(self):
        # A few bytes of whitespace-padded noise still counts as content-free.
        self.assertTrue(
            _failed_logs_are_content_free([("a", "   error\n   ")])
        )

    def test_mixed_one_real_one_empty_is_substantial(self):
        # If even one job has real logs, claude has something to act on.
        self.assertFalse(
            _failed_logs_are_content_free(
                [
                    ("a", "<no log available>"),
                    ("b", "y" * (MIN_USEFUL_LOG_CHARS + 1)),
                ]
            )
        )


class TestDescribeLogState(unittest.TestCase):
    def test_empty_failed_list_calls_out_status_only_checks(self):
        self.assertIn(
            "0 of 3 failing checks have Actions logs",
            describe_log_state([], failing_check_count=3),
        )

    def test_non_empty_reports_run_count_and_chars(self):
        self.assertEqual(
            describe_log_state(
                [("a", "abcde"), ("b", "fghij")], failing_check_count=2
            ),
            "2 run(s), 10 chars",
        )

    def test_no_log_available_placeholder_doesnt_count_chars(self):
        self.assertEqual(
            describe_log_state(
                [("a", "<no log available>"), ("b", "real")],
                failing_check_count=2,
            ),
            "2 run(s), 4 chars",
        )


class TestLatestCompletedAt(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(_latest_completed_at([]))

    def test_picks_max_timestamp(self):
        result = _latest_completed_at(
            [
                {"completedAt": "2026-05-06T10:00:00Z"},
                {"completedAt": "2026-05-06T11:30:00Z"},
                {"completedAt": "2026-05-06T09:15:00Z"},
            ]
        )
        expected = datetime(2026, 5, 6, 11, 30, tzinfo=timezone.utc).timestamp()
        self.assertEqual(result, expected)

    def test_missing_completed_at_returns_none(self):
        self.assertIsNone(
            _latest_completed_at(
                [
                    {"completedAt": "2026-05-06T10:00:00Z"},
                    {"completedAt": ""},
                ]
            )
        )

    def test_zero_placeholder_returns_none(self):
        self.assertIsNone(
            _latest_completed_at(
                [{"completedAt": "0001-01-01T00:00:00Z"}]
            )
        )

    def test_unparseable_returns_none(self):
        self.assertIsNone(
            _latest_completed_at([{"completedAt": "not-a-date"}])
        )


class TestSpuriousCheckNames(unittest.TestCase):
    def test_collects_only_named_failed_checks(self):
        self.assertEqual(
            _spurious_check_names_from_checks(
                [
                    {"name": "pull / linux", "bucket": "fail"},
                    {"name": "lint", "bucket": "cancel"},
                    {"name": "docs", "bucket": "pass"},
                    {"name": "", "bucket": "fail"},
                    {"bucket": "fail"},
                ]
            ),
            {"pull / linux", "lint"},
        )

    def test_workflow_only_failure_has_no_check_to_mark(self):
        self.assertEqual(_spurious_check_names_from_checks([]), set())


if __name__ == "__main__":
    unittest.main()
