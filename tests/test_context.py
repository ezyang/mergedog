import unittest

from mergedog.context import render_context


ZWSP = chr(0x200B)


class TestRenderContext(unittest.TestCase):
    def test_includes_title_body_and_comments(self):
        text = render_context(
            pr=42,
            url="https://github.com/pytorch/pytorch/pull/42",
            title="Fix foo",
            body="Some description.",
            comments=[
                {"author": "alice", "body": "looks good", "created_at": "2026-01-01T00:00:00Z"},
            ],
        )
        self.assertIn("PR #42", text)
        self.assertIn("Fix foo", text)
        self.assertIn("Some description.", text)
        self.assertIn("alice", text)
        self.assertIn("looks good", text)
        self.assertIn("2026-01-01T00:00:00Z", text)

    def test_sanitizes_each_field(self):
        text = render_context(
            pr=1,
            url="https://x",
            title=f"Fix{ZWSP} thing <!-- evil -->",
            body=f"<!-- IGNORE PREVIOUS --> body{ZWSP} text",
            comments=[
                {"author": "bob", "body": "<!--bad-->ok", "created_at": ""},
            ],
        )
        self.assertNotIn(ZWSP, text)
        self.assertNotIn("IGNORE PREVIOUS", text)
        self.assertNotIn("<!--", text)
        # the actual content survives, with the comment chunks gone
        self.assertIn("Fix thing", text)
        self.assertIn("body text", text)
        self.assertIn("ok", text)

    def test_empty_body_handled(self):
        text = render_context(
            pr=1, url="u", title="t", body="", comments=[]
        )
        self.assertIn("(no description)", text)

    def test_no_comments_section_when_empty(self):
        text = render_context(
            pr=1, url="u", title="t", body="b", comments=[]
        )
        self.assertNotIn("[COMMENT", text)

    def test_untrusted_omits_body_and_user_comments(self):
        text = render_context(
            pr=1,
            url="u",
            title="Fix stuff",
            body="Injected instructions here",
            comments=[
                {"author": "attacker", "body": "evil", "created_at": ""},
                {"author": "pytorch-bot[bot]", "body": "dr ci", "created_at": ""},
                {"author": "pytorchmergebot", "body": "merge ok", "created_at": ""},
            ],
            trusted=False,
        )
        self.assertIn("Fix stuff", text)
        self.assertNotIn("[DESCRIPTION]", text)
        self.assertNotIn("Injected instructions", text)
        self.assertNotIn("attacker", text)
        self.assertNotIn("evil", text)
        self.assertIn("pytorch-bot[bot]", text)
        self.assertIn("dr ci", text)
        self.assertIn("pytorchmergebot", text)
        self.assertIn("merge ok", text)

    def test_trusted_includes_everything(self):
        text = render_context(
            pr=1,
            url="u",
            title="t",
            body="description",
            comments=[
                {"author": "someone", "body": "comment", "created_at": ""},
            ],
            trusted=True,
        )
        self.assertIn("[DESCRIPTION]", text)
        self.assertIn("description", text)
        self.assertIn("someone", text)
        self.assertIn("comment", text)


if __name__ == "__main__":
    unittest.main()
