import contextlib
import io
import unittest
from unittest import mock

from mergedog import log as log_mod


class TestLog(unittest.TestCase):
    def tearDown(self):
        log_mod.set_outcome(None)
        log_mod.set_merging(False)
        log_mod.set_approved(False)
        log_mod.configure_status_pr(None)

    def test_complete_uses_done_prefix_without_halt(self):
        log_mod.set_approved(True)
        stderr = io.StringIO()

        with (
            contextlib.redirect_stderr(stderr),
            mock.patch("mergedog.notify.notify_halt") as notify_halt,
            self.assertRaises(SystemExit) as raised,
        ):
            log_mod.complete(
                "PR is no longer open; shepherd complete",
                code=42,
            )

        self.assertEqual(raised.exception.code, 42)
        self.assertIn(
            "[DONE] PR is no longer open; shepherd complete",
            stderr.getvalue(),
        )
        self.assertNotIn("HALT", stderr.getvalue())
        notify_halt.assert_not_called()

    def test_die_writes_halted_status(self):
        log_mod.configure_status_pr(123)
        stderr = io.StringIO()

        with (
            contextlib.redirect_stderr(stderr),
            mock.patch("mergedog.notify.notify_halt"),
            mock.patch("mergedog.status.write_status") as write_status,
            self.assertRaises(SystemExit),
        ):
            log_mod.die("manual intervention required")

        write_status.assert_called_once_with(
            123,
            phase="halted",
            category="blocked",
            message="HALT: manual intervention required",
            user_action="manual intervention required",
        )

    def test_log_escapes_control_characters(self):
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            log_mod.log("bad\x00\x1b[31m")

        self.assertIn("bad\\x00\\x1b[31m", stderr.getvalue())
        self.assertNotIn("\x00", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
