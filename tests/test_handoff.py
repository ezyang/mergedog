import unittest
from unittest import mock

from mergedog import handoff


class TestWatchStackPostHandoff(unittest.TestCase):
    def test_returns_failed_member(self):
        comments = {
            101: [],
            102: [
                {
                    "author": "pytorchmergebot",
                    "created_at": "2026-05-08T14:00:00Z",
                    "body": "## Merge failed\nCONFLICT (content): Merge conflict",
                }
            ],
        }

        def get_pr(pr):
            return {
                "number": pr,
                "state": "OPEN",
                "reviewDecision": "APPROVED",
                "labels": [],
            }

        with mock.patch.object(
            handoff.github, "get_pr", side_effect=get_pr
        ), mock.patch.object(
            handoff.github,
            "get_pr_comments",
            side_effect=lambda pr: comments[pr],
        ):
            result = handoff.watch_stack_post_handoff(
                {
                    101: "2026-05-08T13:00:00Z",
                    102: "2026-05-08T13:00:00Z",
                }
            )

        self.assertEqual(result[0], "failed")
        self.assertEqual(result[1], 102)
        self.assertEqual(result[2], "2026-05-08T14:00:00Z")
        self.assertIn("Merge failed", result[3])

    def test_returns_closed_member(self):
        def get_pr(pr):
            return {
                "number": pr,
                "state": "CLOSED" if pr == 101 else "OPEN",
                "reviewDecision": "APPROVED",
                "labels": [],
            }

        with mock.patch.object(handoff.github, "get_pr", side_effect=get_pr):
            result = handoff.watch_stack_post_handoff(
                {
                    101: "2026-05-08T13:00:00Z",
                    102: "2026-05-08T13:00:00Z",
                }
            )

        self.assertEqual(result, ("closed", 101, None, None))


if __name__ == "__main__":
    unittest.main()
