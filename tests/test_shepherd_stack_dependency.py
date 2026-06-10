import subprocess
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
        self.pending_publish_orig_sha = ""

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


def _tree_sha(tree_map):
    return lambda sha: tree_map[sha]


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
            shepherd.repo,
            "tree_sha",
            side_effect=_tree_sha(
                {"P_NEW": "P_NEW_TREE", "P_OLD": "P_OLD_TREE"}
            ),
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

    def test_equivalent_parent_rewrite_is_not_stale(self):
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
            shepherd.repo,
            "tree_sha",
            side_effect=_tree_sha(
                {"P_NEW": "SAME_TREE", "P_OLD": "SAME_TREE"}
            ),
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

        self.assertFalse(status.stale)
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
            shepherd.repo,
            "tree_sha",
            side_effect=_tree_sha(
                {"P_NEW": "P_NEW_TREE", "P_OLD": "P_OLD_TREE"}
            ),
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
            shepherd.repo,
            "tree_sha",
            side_effect=_tree_sha(
                {"P_NEW": "P_NEW_TREE", "P_OLD": "P_OLD_TREE"}
            ),
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
    def test_landed_parent_base_marks_parent_ready(self):
        dep = _dep()
        status = shepherd._GhstackParentStatus(
            stale=True,
            parent_ready=False,
            parent_status="pending",
            parent_done=0,
            parent_total=1,
            parent_orig_sha="P_NEW",
            child_orig_sha="C_OLD",
            child_parent_sha="P_OLD",
            reason="parent CI is pending",
        )

        with mock.patch.object(
            shepherd.github, "get_pr_merge_commit_sha", return_value="P_LANDED"
        ), mock.patch.object(
            shepherd.repo, "fetch_origin"
        ) as fetch_origin, mock.patch.object(
            shepherd.repo, "is_ancestor", return_value=True
        ) as is_ancestor, mock.patch.object(
            shepherd.repo,
            "tree_sha",
            side_effect=_tree_sha(
                {"P_LANDED": "P_LANDED_TREE", "P_OLD": "P_OLD_TREE"}
            ),
        ):
            changed = shepherd._maybe_use_landed_ghstack_parent_base(dep, status)

        self.assertTrue(changed)
        self.assertTrue(status.stale)
        self.assertTrue(status.parent_ready)
        self.assertEqual(status.replay_base_sha, "P_LANDED")
        self.assertEqual(
            status.reason,
            "parent PR merged as P_LANDED; replaying child onto landed tree",
        )
        fetch_origin.assert_called_once()
        is_ancestor.assert_called_once_with("P_LANDED", "origin/main")

    def test_landed_parent_base_clears_stale_when_child_already_matches_landed_tree(self):
        dep = _dep()
        status = shepherd._GhstackParentStatus(
            stale=True,
            parent_ready=False,
            parent_status="pending",
            parent_done=0,
            parent_total=1,
            parent_orig_sha="P_NEW",
            child_orig_sha="C_OLD",
            child_parent_sha="P_OLD",
            reason="parent CI is pending",
        )

        with mock.patch.object(
            shepherd.github, "get_pr_merge_commit_sha", return_value="P_LANDED"
        ), mock.patch.object(
            shepherd.repo, "fetch_origin"
        ), mock.patch.object(
            shepherd.repo, "is_ancestor", return_value=True
        ), mock.patch.object(
            shepherd.repo,
            "tree_sha",
            side_effect=_tree_sha(
                {"P_LANDED": "SAME_TREE", "P_OLD": "SAME_TREE"}
            ),
        ):
            changed = shepherd._maybe_use_landed_ghstack_parent_base(dep, status)

        self.assertTrue(changed)
        self.assertFalse(status.stale)
        self.assertTrue(status.parent_ready)
        self.assertEqual(status.replay_base_sha, "P_LANDED")

    def test_landed_parent_base_waits_until_origin_main_contains_merge(self):
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

        with mock.patch.object(
            shepherd.github, "get_pr_merge_commit_sha", return_value="P_LANDED"
        ), mock.patch.object(
            shepherd.repo, "fetch_origin"
        ), mock.patch.object(
            shepherd.repo, "is_ancestor", return_value=False
        ):
            changed = shepherd._maybe_use_landed_ghstack_parent_base(dep, status)

        self.assertFalse(changed)
        self.assertFalse(status.parent_ready)
        self.assertIsNone(status.replay_base_sha)
        self.assertEqual(
            status.reason,
            "parent PR merged as P_LANDED, but origin/main does not contain it yet",
        )

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
            shepherd.repo, "head_sha", return_value="C_NEW_ORIG"
        ), mock.patch.object(
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
        self.assertEqual(trust.pending_publish_orig_sha, "")
        wait_head.assert_called_once_with(101, "C_NEW_HEAD")

    def test_landed_parent_replay_uses_merge_commit_base(self):
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
            reason="parent PR merged as P_LANDED; replaying child onto landed tree",
            replay_base_sha="P_LANDED",
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
        ), mock.patch.object(
            shepherd.repo, "head_sha", return_value="C_NEW_ORIG"
        ), mock.patch.object(
            shepherd.repo, "ghstack_submit"
        ), mock.patch.object(
            shepherd.repo, "fetch_ghstack_head", return_value="C_NEW_HEAD"
        ), mock.patch.object(
            shepherd, "_wait_for_pr_head"
        ):
            changed = shepherd._publish_ghstack_parent_rebase(
                101,
                worktree,
                dep,
                status,
                trust,
                ignore_sev=False,
            )

        self.assertTrue(changed)
        set_worktree.assert_called_once_with(worktree, "P_LANDED")

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

    def test_conflicted_child_replay_invokes_resolver_then_submits(self):
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
        pr_data = {
            "number": 101,
            "url": "https://github.com/pytorch/pytorch/pull/101",
        }
        trust = _Trust()
        sessions = []
        worktree = Path("/tmp/worktree")
        error = subprocess.CalledProcessError(
            1, ["ghstack", "cherry-pick", "101"]
        )

        with mock.patch.object(
            shepherd, "_wait_for_no_active_sev", return_value=False
        ), mock.patch.object(
            shepherd.repo, "fetch_stack_refs", return_value=refs
        ), mock.patch.object(
            shepherd.repo, "set_worktree_to_sha"
        ), mock.patch.object(
            shepherd.repo, "ghstack_cherry_pick", side_effect=error
        ), mock.patch.object(
            shepherd.repo, "is_cherry_pick_in_progress", return_value=True
        ), mock.patch.object(
            shepherd, "_refresh_context_file", return_value=(Path("/tmp/ctx"), [])
        ), mock.patch.object(
            shepherd.repo, "head_sha", return_value="P_NEW"
        ), mock.patch.object(
            shepherd.claude_mod,
            "invoke_cherry_pick_resolver",
            return_value=(True, "C_NEW_ORIG", []),
        ) as resolver, mock.patch.object(
            shepherd.repo, "ghstack_submit"
        ) as submit, mock.patch.object(
            shepherd.repo, "fetch_ghstack_head", return_value="C_NEW_HEAD"
        ), mock.patch.object(
            shepherd, "_wait_for_pr_head"
        ):
            changed = shepherd._publish_ghstack_parent_rebase(
                101,
                worktree,
                dep,
                status,
                trust,
                ignore_sev=False,
                pr_data=pr_data,
                sessions=sessions,
            )

        self.assertTrue(changed)
        resolver.assert_called_once()
        submit.assert_called_once_with(
            worktree, "Propagate parent update downstream"
        )
        self.assertEqual(trust.trusted_shas, ["C_NEW_HEAD"])
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].mode, "cherry-pick-resolver")


if __name__ == "__main__":
    unittest.main()
