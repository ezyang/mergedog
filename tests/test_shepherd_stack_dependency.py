import time
import unittest
from pathlib import Path
from unittest import mock

from mergedog import shepherd


class _Trust:
    def __init__(self, trusted=True):
        self.trusted = trusted
        self.spurious_check_names = []
        self.trusted_shas = []
        self.head_branch = None
        self.head_repo_clone_url = None

    def is_trusted(self, sha):
        return self.trusted or sha in self.trusted_shas

    def trust(self, sha):
        self.trusted_shas.append(sha)

    def save(self):
        pass


def _dep():
    return shepherd._GhstackParentDependency(
        parent_pr=100,
        parent_head_ref="gh/u/100/head",
        parent_orig_ref="gh/u/100/orig",
        child_head_ref="gh/u/101/head",
        child_orig_ref="gh/u/101/orig",
    )


class TestGhstackParentStatus(unittest.TestCase):
    def test_stale_parent_not_ready_while_pending(self):
        dep = _dep()
        refs = {
            "gh/u/100/head": "P_HEAD",
            "gh/u/100/orig": "P_NEW",
            "gh/u/101/head": "C_HEAD",
            "gh/u/101/orig": "C_ORIG",
        }
        checks = [{"name": "linux", "bucket": "pending"}]
        with mock.patch.object(
            shepherd.repo, "fetch_stack_refs", return_value=refs
        ), mock.patch.object(
            shepherd.repo, "parent_sha", return_value="P_OLD"
        ), mock.patch.object(
            shepherd.github, "get_pr_checks_all", return_value=checks
        ), mock.patch.object(
            shepherd.github, "get_pr_head_sha", return_value="P_HEAD"
        ), mock.patch.object(
            shepherd.github, "evaluate_checks", return_value="pending"
        ), mock.patch.object(
            shepherd.TrustDB, "load_or_create", return_value=_Trust()
        ):
            status = shepherd._refresh_ghstack_parent_status(dep)

        self.assertTrue(status.stale)
        self.assertFalse(status.parent_ready)
        self.assertEqual(status.reason, "parent CI is pending")

    def test_stale_parent_ready_when_green_stable(self):
        dep = _dep()
        old = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(time.time() - shepherd.CI_STABILITY_WINDOW_SEC - 5),
        )
        refs = {
            "gh/u/100/head": "P_HEAD",
            "gh/u/100/orig": "P_NEW",
            "gh/u/101/head": "C_HEAD",
            "gh/u/101/orig": "C_ORIG",
        }
        checks = [{"name": "linux", "bucket": "pass", "completedAt": old}]
        with mock.patch.object(
            shepherd.repo, "fetch_stack_refs", return_value=refs
        ), mock.patch.object(
            shepherd.repo, "parent_sha", return_value="P_OLD"
        ), mock.patch.object(
            shepherd.github, "get_pr_checks_all", return_value=checks
        ), mock.patch.object(
            shepherd.github, "get_pr_head_sha", return_value="P_HEAD"
        ), mock.patch.object(
            shepherd.github, "evaluate_checks", return_value="passed"
        ), mock.patch.object(
            shepherd.TrustDB, "load_or_create", return_value=_Trust()
        ):
            status = shepherd._refresh_ghstack_parent_status(dep)

        self.assertTrue(status.stale)
        self.assertTrue(status.parent_ready)
        self.assertEqual(status.reason, "parent is green-stable")

    def test_untrusted_parent_blocks(self):
        dep = _dep()
        refs = {
            "gh/u/100/head": "P_HEAD",
            "gh/u/100/orig": "P_NEW",
            "gh/u/101/head": "C_HEAD",
            "gh/u/101/orig": "C_ORIG",
        }
        with mock.patch.object(
            shepherd.repo, "fetch_stack_refs", return_value=refs
        ), mock.patch.object(
            shepherd.repo, "parent_sha", return_value="P_OLD"
        ), mock.patch.object(
            shepherd.github, "get_pr_checks_all", return_value=[]
        ), mock.patch.object(
            shepherd.github, "get_pr_head_sha", return_value="P_HEAD"
        ), mock.patch.object(
            shepherd.TrustDB, "load_or_create", return_value=_Trust(trusted=False)
        ), mock.patch.object(
            shepherd, "trust_mergebot_rebase_if_equivalent", return_value=False
        ):
            status = shepherd._refresh_ghstack_parent_status(dep)

        self.assertTrue(status.stale)
        self.assertFalse(status.parent_ready)
        self.assertEqual(status.reason, "parent head is not trusted")


class TestResolveGhstackParentDependency(unittest.TestCase):
    def test_stack_resolution_halt_falls_back_to_isolated_mode(self):
        with mock.patch(
            "mergedog.stack.resolve_stack", side_effect=SystemExit(1)
        ), mock.patch.object(shepherd, "log") as log:
            dep = shepherd._resolve_ghstack_parent_dependency(
                101, "gh/u/101/head"
            )

        self.assertIsNone(dep)
        log.assert_called_once()


class TestPublishGhstackParentRebase(unittest.TestCase):
    def test_rebases_child_onto_parent_and_submits_only_child(self):
        dep = _dep()
        status = shepherd._GhstackParentStatus(
            stale=True,
            parent_ready=True,
            parent_status="passed",
            parent_done=1,
            parent_total=1,
            parent_orig_sha="P_NEW",
            child_orig_sha="C_OLD",
            child_parent_sha="P_OLD",
            reason="parent is green-stable",
        )
        refs = {
            "gh/u/100/head": "P_HEAD",
            "gh/u/100/orig": "P_NEW",
            "gh/u/101/head": "C_HEAD",
            "gh/u/101/orig": "C_OLD",
        }
        trust = _Trust()
        worktree = Path("/tmp/worktree")
        with mock.patch.object(
            shepherd, "_wait_for_no_active_sev", return_value=False
        ), mock.patch.object(
            shepherd.repo, "fetch_stack_refs", return_value=refs
        ), mock.patch.object(
            shepherd.repo, "set_worktree_to_sha"
        ) as set_worktree, mock.patch.object(
            shepherd.repo, "ghstack_cherry_pick"
        ) as cherry_pick, mock.patch.object(
            shepherd.repo, "ghstack_submit"
        ) as submit, mock.patch.object(
            shepherd.repo, "fetch_ghstack_head", return_value="C_NEW_HEAD"
        ), mock.patch.object(
            shepherd, "_wait_for_pr_head"
        ) as wait_head:
            changed = shepherd._publish_ghstack_parent_rebase(
                101,
                worktree,
                dep,
                status,
                trust,
                ignore_sev=False,
            )

        self.assertTrue(changed)
        set_worktree.assert_called_once_with(worktree, "P_NEW")
        cherry_pick.assert_called_once_with(worktree, 101)
        submit.assert_called_once_with(
            worktree, "Propagate parent update downstream"
        )
        self.assertEqual(trust.trusted_shas, ["C_NEW_HEAD"])
        self.assertEqual(trust.spurious_check_names, [])
        wait_head.assert_called_once_with(101, "C_NEW_HEAD")

    def test_ref_change_aborts_before_submit(self):
        dep = _dep()
        status = shepherd._GhstackParentStatus(
            stale=True,
            parent_ready=True,
            parent_status="passed",
            parent_done=1,
            parent_total=1,
            parent_orig_sha="P_NEW",
            child_orig_sha="C_OLD",
            child_parent_sha="P_OLD",
            reason="parent is green-stable",
        )
        refs = {
            "gh/u/100/head": "P_HEAD",
            "gh/u/100/orig": "P_NEWER",
            "gh/u/101/head": "C_HEAD",
            "gh/u/101/orig": "C_OLD",
        }
        with mock.patch.object(
            shepherd, "_wait_for_no_active_sev", return_value=False
        ), mock.patch.object(
            shepherd.repo, "fetch_stack_refs", return_value=refs
        ), mock.patch.object(
            shepherd.repo, "ghstack_submit"
        ) as submit:
            changed = shepherd._publish_ghstack_parent_rebase(
                101,
                Path("/tmp/worktree"),
                dep,
                status,
                _Trust(),
                ignore_sev=False,
            )

        self.assertFalse(changed)
        submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
