import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mergedog import rage


class TestRageReport(unittest.TestCase):
    def test_build_report_includes_relevant_pr_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "logs").mkdir()
            (root / "state").mkdir()
            (root / "contexts").mkdir()
            (root / "logs" / "123.log").write_text("HALT: bad\n")
            (root / "state" / "123.json").write_text('{"trusted_shas": ["abc"]}')
            (root / "contexts" / "123.md").write_text("PR context")
            (root / "mux-prs.json").write_text("[123, 456]")
            (root / "pushed-commits.log").write_text(
                "2026-01-01T00:00:00  fix     PR#123  abc  subject\n"
                "2026-01-01T00:00:00  fix     PR#456  def  other\n"
            )

            report = rage.build_report(123, root=root)

        self.assertIn("HALT: bad", report)
        self.assertIn('"trusted_shas": ["abc"]', report)
        self.assertIn("PR context", report)
        self.assertIn("[123, 456]", report)
        self.assertIn("PR#123", report)
        self.assertNotIn("PR#456", report)

    def test_build_report_redacts_credentials(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "logs").mkdir()
            (root / "state").mkdir()
            (root / "logs" / "1.log").write_text(
                "Authorization: Bearer ghp_abcdefghijklmnopqrstuvwxyz\n"
                "url=https://user:pass@example.com/repo.git\n"
            )
            (root / "state" / "1.json").write_text(
                '{"github_token": "github_pat_abcdefghijklmnopqrstuvwxyz"}'
            )

            report = rage.build_report(1, root=root)

        self.assertNotIn("Bearer ghp_", report)
        self.assertNotIn("user:pass@", report)
        self.assertNotIn("github_pat_", report)
        self.assertIn("Authorization: <REDACTED>", report)
        self.assertIn("https://user:<REDACTED>@example.com/repo.git", report)
        self.assertIn('"github_token": "<REDACTED>"', report)


class TestCreatePaste(unittest.TestCase):
    @mock.patch("shutil.which")
    @mock.patch("subprocess.run")
    def test_uses_private_markdown_pastebin(self, run, which):
        which.side_effect = lambda name: "/bin/pastebin" if name == "pastebin" else None
        run.return_value = subprocess.CompletedProcess(
            ["/bin/pastebin"], 0, stdout="https://paste/P123\n", stderr=""
        )

        out = rage.create_paste("body", title="title")

        self.assertEqual(out, "https://paste/P123")
        run.assert_called_once_with(
            ["/bin/pastebin", "--md", "--private", "--title", "title"],
            input="body",
            check=False,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
