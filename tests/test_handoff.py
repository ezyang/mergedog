import unittest
from unittest import mock

from mergedog import github
from mergedog import handoff


class TestHandoffComments(unittest.TestCase):
    def test_handoff_comment_records_current_head(self):
        body = handoff._format_handoff_comment(
            {
                "number": 101,
                "headRefOid": "a" * 40,
            },
            [],
        )

        self.assertIn(f"<!-- mergedog:handoff head={'a' * 40} -->", body)
        self.assertIn(f"Current PR head: `{'a' * 40}`.", body)

    def test_handoff_comment_idempotency_is_scoped_to_head(self):
        comments = [
            {
                "author": "mergedog",
                "created_at": "2026-05-08T13:00:00Z",
                "body": f"<!-- mergedog:handoff head={'a' * 40} -->",
            },
            {
                "author": "mergedog",
                "created_at": "2026-05-08T14:00:00Z",
                "body": "<!-- mergedog:handoff -->",
            },
        ]

        with mock.patch.object(github, "get_pr_comments", return_value=comments):
            self.assertTrue(github.has_mergedog_handoff_comment(101))
            self.assertTrue(
                github.has_mergedog_handoff_comment(
                    101, head_sha="a" * 40
                )
            )
            self.assertFalse(
                github.has_mergedog_handoff_comment(
                    101, head_sha="b" * 40
                )
            )
            self.assertEqual(
                github.latest_mergedog_handoff_iso(101),
                "2026-05-08T14:00:00Z",
            )


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
