import unittest

from mergedog.prompts import render_fix_prompt


_BASE_KWARGS = dict(
    url="https://github.com/pytorch/pytorch/pull/1",
    branch="gh/u/1/head",
    context_path="/tmp/ctx.txt",
    failed_jobs=[("pull / linux", "BOOM")],
)


class TestEarlierStackSection(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
