import unittest
from datetime import datetime, timezone

from mergedog.shepherd import (
    MIN_USEFUL_LOG_CHARS,
    _failed_logs_are_content_free,
    _latest_completed_at,
    describe_log_state,
)


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


if __name__ == "__main__":
    unittest.main()
