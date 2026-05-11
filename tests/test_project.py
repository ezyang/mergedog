import unittest
from unittest import mock

from mergedog.bootstrap import promote_early_env
from mergedog import project


class TestProjectPolicy(unittest.TestCase):
    def test_pytorch_policy_keeps_existing_conventions(self):
        with mock.patch.object(project, "REPO_SLUG", "pytorch/pytorch"):
            with mock.patch.object(
                project, "REPO_SSH_URL", "git@github.com:pytorch/pytorch.git"
            ):
                policy = project.get_project_policy()

        self.assertEqual(policy.trunk_label, "ciflow/trunk")
        self.assertEqual(policy.ci_sev_label, "ci: sev")
        self.assertEqual(policy.mergebot_login, "pytorchmergebot")
        self.assertEqual(policy.merge_command, "@pytorchbot merge")
        self.assertEqual(policy.known_good_ref, "origin/viable/strict")

    def test_generic_policy_disables_pytorch_only_hooks(self):
        with mock.patch.object(project, "REPO_SLUG", "owner/repo"):
            with mock.patch.object(
                project, "REPO_SSH_URL", "git@github.com:owner/repo.git"
            ):
                policy = project.get_project_policy()

        self.assertEqual(policy.repo_slug, "owner/repo")
        self.assertIsNone(policy.trunk_label)
        self.assertIsNone(policy.ci_sev_label)
        self.assertIsNone(policy.mergebot_login)
        self.assertIsNone(policy.merge_command)
        self.assertIsNone(policy.known_good_ref)


class TestEarlyEnv(unittest.TestCase):
    def test_repo_flag_promotes_to_env(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            promote_early_env(["--repo", "owner/repo"])

            import os

            self.assertEqual(os.environ["MERGEDOG_REPO_SLUG"], "owner/repo")

    def test_repo_equals_flag_promotes_to_env(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            promote_early_env(["--repo=owner/repo"])

            import os

            self.assertEqual(os.environ["MERGEDOG_REPO_SLUG"], "owner/repo")


if __name__ == "__main__":
    unittest.main()
