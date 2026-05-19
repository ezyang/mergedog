import unittest
from types import SimpleNamespace
from unittest import mock

from mergedog import head_trust


class _Trust:
    def __init__(self):
        self.trusted_shas = ["trusted"]
        self.newly_trusted = []

    def trust(self, sha):
        self.newly_trusted.append(sha)


class TestMergebotRebaseTrust(unittest.TestCase):
    def test_trusts_mergebot_committed_equivalent_patch(self):
        trust = _Trust()
        ensure = mock.Mock(return_value=True)
        with mock.patch.object(
            head_trust, "PROJECT", SimpleNamespace(mergebot_login="pytorchmergebot")
        ), mock.patch.object(
            head_trust.github,
            "get_commit_actor_logins",
            return_value=("author", "pytorchmergebot"),
        ), mock.patch.object(
            head_trust.repo, "patch_id_matches_any", return_value=True
        ) as matches, mock.patch.object(head_trust, "log"):
            self.assertTrue(
                head_trust.trust_mergebot_rebase_if_equivalent(
                    trust, "newsha", ensure_current_available=ensure
                )
            )

        ensure.assert_called_once_with()
        matches.assert_called_once_with("newsha", ["trusted"])
        self.assertEqual(trust.newly_trusted, ["newsha"])

    def test_rejects_author_only_bot_commit(self):
        trust = _Trust()
        ensure = mock.Mock(return_value=True)
        with mock.patch.object(
            head_trust, "PROJECT", SimpleNamespace(mergebot_login="pytorchmergebot")
        ), mock.patch.object(
            head_trust.github,
            "get_commit_actor_logins",
            return_value=("pytorchmergebot", "contributor"),
        ), mock.patch.object(head_trust.repo, "patch_id_matches_any") as matches:
            self.assertFalse(
                head_trust.trust_mergebot_rebase_if_equivalent(
                    trust, "newsha", ensure_current_available=ensure
                )
            )

        ensure.assert_not_called()
        matches.assert_not_called()
        self.assertEqual(trust.newly_trusted, [])

    def test_rejects_mergebot_commit_with_different_patch(self):
        trust = _Trust()
        with mock.patch.object(
            head_trust, "PROJECT", SimpleNamespace(mergebot_login="pytorchmergebot")
        ), mock.patch.object(
            head_trust.github,
            "get_commit_actor_logins",
            return_value=("author", "pytorchmergebot"),
        ), mock.patch.object(
            head_trust.repo, "patch_id_matches_any", return_value=False
        ):
            self.assertFalse(
                head_trust.trust_mergebot_rebase_if_equivalent(
                    trust, "newsha", ensure_current_available=lambda: True
                )
            )

        self.assertEqual(trust.newly_trusted, [])

    def test_rejects_when_current_head_cannot_be_verified_locally(self):
        trust = _Trust()
        with mock.patch.object(
            head_trust, "PROJECT", SimpleNamespace(mergebot_login="pytorchmergebot")
        ), mock.patch.object(
            head_trust.github,
            "get_commit_actor_logins",
            return_value=("author", "pytorchmergebot"),
        ), mock.patch.object(head_trust.repo, "patch_id_matches_any") as matches:
            self.assertFalse(
                head_trust.trust_mergebot_rebase_if_equivalent(
                    trust, "newsha", ensure_current_available=lambda: False
                )
            )

        matches.assert_not_called()
        self.assertEqual(trust.newly_trusted, [])


if __name__ == "__main__":
    unittest.main()
