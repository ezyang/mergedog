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

    def test_complete_uses_outcome_prefix_without_halt(self):
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
                outcome="MERGED",
            )

        self.assertEqual(raised.exception.code, 42)
        self.assertIn(
            "[MERGED] PR is no longer open; shepherd complete",
            stderr.getvalue(),
        )
        self.assertNotIn("HALT", stderr.getvalue())
        notify_halt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
