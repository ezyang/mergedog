import unittest

from mergedog.prompts import (
    render_fix_prompt,
    render_cherry_pick_conflict_prompt,
    render_operator_fix_prompt,
    render_rebase_conflict_prompt,
)


_BASE_KWARGS = dict(
    url="https://github.com/pytorch/pytorch/pull/1",
    branch="gh/u/1/head",
    context_path="/tmp/ctx.txt",
    failed_jobs=[("pull / linux", "BOOM")],
)


class TestEarlierStackSection(unittest.TestCase):
    def test_fix_prompt_separates_too_hard_from_inconclusive(self):
        prompt = render_fix_prompt(**_BASE_KWARGS)

        self.assertIn("touch .mergedog-too-hard", prompt)
        self.assertIn("real and PR-related", prompt)
        self.assertIn("cannot safely fix it in one commit", prompt)
        self.assertIn("touch .mergedog-spurious", prompt)
        self.assertIn("touch .mergedog-inconclusive", prompt)
        self.assertIn("Choose this only if you genuinely", prompt)

    def test_fix_prompt_warns_against_xpass_marker_removal(self):
        prompt = render_fix_prompt(**_BASE_KWARGS)

        self.assertIn("Caution on XPASS / unexpected-success failures", prompt)
        self.assertIn("test-policy change", prompt)
        self.assertIn("directly exercises", prompt)
        self.assertIn("causal link", prompt)
        self.assertIn("rather than unrelated trunk drift", prompt)
        self.assertIn("standalone test-policy cleanup", prompt)

    def test_section_omitted_when_no_earlier_members(self):
        prompt = render_fix_prompt(**_BASE_KWARGS)
        self.assertNotIn("earlier-stack status", prompt)

    def test_section_omitted_when_earlier_members_empty(self):
        prompt = render_fix_prompt(**_BASE_KWARGS, earlier_members=[])
        self.assertNotIn("earlier-stack status", prompt)

    def test_section_renders_passing_earlier_member(self):
        prompt = render_fix_prompt(
            **_BASE_KWARGS,
            is_ghstack=True,
            earlier_in_stack=1,
            earlier_members=[
                {
                    "pr": 100,
                    "head_offset": 1,
                    "status": "passed (191/191 done)",
                    "failing_checks": [],
                    "fix_commits_pushed": 0,
                }
            ],
        )
        self.assertIn("--- begin earlier-stack status ---", prompt)
        self.assertIn("PR #100 (HEAD~1): passed (191/191 done)", prompt)
        self.assertNotIn("failing checks:", prompt)
        self.assertNotIn("fix commit", prompt)

    def test_section_renders_failing_earlier_member_with_checks_and_fixes(self):
        prompt = render_fix_prompt(
            **_BASE_KWARGS,
            is_ghstack=True,
            earlier_in_stack=2,
            earlier_members=[
                {
                    "pr": 99,
                    "head_offset": 2,
                    "status": "failed (262/273 done)",
                    "failing_checks": ["pull / linux", "lint"],
                    "fix_commits_pushed": 1,
                },
                {
                    "pr": 100,
                    "head_offset": 1,
                    "status": "passed (191/191 done)",
                    "failing_checks": [],
                    "fix_commits_pushed": 0,
                },
            ],
        )
        self.assertIn("PR #99 (HEAD~2): failed (262/273 done)", prompt)
        self.assertIn("    failing checks:", prompt)
        self.assertIn("      - pull / linux", prompt)
        self.assertIn("      - lint", prompt)
        self.assertIn("pushed 1 fix commit on this PR", prompt)
        self.assertIn("PR #100 (HEAD~1): passed (191/191 done)", prompt)

    def test_section_pluralizes_multiple_fix_commits(self):
        prompt = render_fix_prompt(
            **_BASE_KWARGS,
            is_ghstack=True,
            earlier_in_stack=1,
            earlier_members=[
                {
                    "pr": 99,
                    "head_offset": 1,
                    "status": "failed",
                    "failing_checks": [],
                    "fix_commits_pushed": 3,
                }
            ],
        )
        self.assertIn("pushed 3 fix commits on this PR", prompt)

    def test_stack_member_hint_references_section_when_earlier_in_stack(self):
        prompt = render_fix_prompt(
            **_BASE_KWARGS,
            is_ghstack=True,
            earlier_in_stack=1,
            earlier_members=[
                {
                    "pr": 99,
                    "head_offset": 1,
                    "status": "failed",
                    "failing_checks": ["pull / linux"],
                    "fix_commits_pushed": 0,
                }
            ],
        )
        self.assertIn('"earlier-stack status" section above', prompt)
        self.assertIn("share a root cause", prompt)

    def test_operator_fix_prompt_uses_trusted_context_without_ci_logs(self):
        prompt = render_operator_fix_prompt(
            url="https://github.com/pytorch/pytorch/pull/1",
            branch="gh/u/1/head",
            context_path="/tmp/ctx.txt",
            operator_context="Return type should be a TypeGuard.",
            is_ghstack=True,
        )

        self.assertIn("trusted mergedog operator", prompt)
        self.assertIn("--- begin operator context ---", prompt)
        self.assertIn("Return type should be a TypeGuard.", prompt)
        self.assertIn("Do NOT push", prompt)
        self.assertNotIn("Failed CI jobs", prompt)

    def test_rebase_prompt_allows_replaying_existing_commits_only(self):
        prompt = render_rebase_conflict_prompt(
            url="https://github.com/pytorch/pytorch/pull/1",
            branch="gh/u/1/head",
            context_path="/tmp/ctx.txt",
        )

        self.assertIn("Do not make standalone commits", prompt)
        self.assertIn("may replay multiple existing PR", prompt)
        self.assertIn("mergedog commits; that is expected", prompt)
        self.assertIn("let git continue the in-progress rebase", prompt)

    def test_cherry_pick_prompt_describes_stack_parent_replay(self):
        prompt = render_cherry_pick_conflict_prompt(
            url="https://github.com/pytorch/pytorch/pull/1",
            branch="gh/u/1/head",
            context_path="/tmp/ctx.txt",
        )

        self.assertIn("ghstack parent PR advanced", prompt)
        self.assertIn(".git/CHERRY_PICK_HEAD", prompt)
        self.assertIn("git cherry-pick --continue", prompt)
        self.assertIn("git cherry-pick --abort", prompt)
        self.assertIn("Do NOT push", prompt)


if __name__ == "__main__":
    unittest.main()
