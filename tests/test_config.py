import json
import tempfile
import unittest
from pathlib import Path

from mergedog.config import (
    add_ignored_ci_sev,
    clear_ignored_ci_sevs,
    format_ci_sev_ignored_numbers,
    get_ci_sev_config,
    LLMConfig,
    parse_ci_sev_number,
    remove_ignored_ci_sev,
    get_llm_config,
    set_llm_config,
)
from mergedog.claude import _build_llm_invocation


class TestLLMConfig(unittest.TestCase):
    def test_default_is_codex(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = get_llm_config(Path(d) / "config.json")

        self.assertEqual(cfg.provider, "codex")
        self.assertIsNone(cfg.model)
        self.assertIsNone(cfg.effective_model)

    def test_set_provider_and_model_preserves_other_config(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.json"
            path.write_text(json.dumps({"other": {"keep": True}}))

            cfg = set_llm_config("codex", model="gpt-5.4", path=path)

            self.assertEqual(cfg.provider, "codex")
            self.assertEqual(cfg.effective_model, "gpt-5.4")
            self.assertEqual(json.loads(path.read_text())["other"], {"keep": True})

    def test_rejects_unknown_provider(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.json"

            with self.assertRaisesRegex(ValueError, "provider must be one of"):
                set_llm_config("bogus", path=path)


class TestCiSevConfig(unittest.TestCase):
    def test_default_ignored_sevs_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = get_ci_sev_config(Path(d) / "config.json")

        self.assertEqual(cfg.ignored_numbers, ())

    def test_add_and_remove_ignored_sev_preserves_other_config(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.json"
            path.write_text(json.dumps({"other": {"keep": True}}))

            cfg = add_ignored_ci_sev(187193, path=path)
            self.assertEqual(cfg.ignored_numbers, (187193,))
            cfg = add_ignored_ci_sev(187193, path=path)
            self.assertEqual(cfg.ignored_numbers, (187193,))
            cfg = add_ignored_ci_sev(188122, path=path)
            self.assertEqual(cfg.ignored_numbers, (187193, 188122))
            self.assertEqual(json.loads(path.read_text())["other"], {"keep": True})

            cfg = remove_ignored_ci_sev(187193, path=path)
            self.assertEqual(cfg.ignored_numbers, (188122,))
            cfg = clear_ignored_ci_sevs(path=path)
            self.assertEqual(cfg.ignored_numbers, ())

    def test_parse_and_format_ci_sev_numbers(self):
        self.assertEqual(parse_ci_sev_number("#187193"), 187193)
        self.assertEqual(parse_ci_sev_number(" 187193 "), 187193)
        self.assertEqual(
            format_ci_sev_ignored_numbers({188122, 187193}),
            "#187193, #188122",
        )

    def test_rejects_invalid_ignored_sev_config(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.json"
            path.write_text(json.dumps({"ci_sev": {"ignored": ["not-a-number"]}}))

            with self.assertRaisesRegex(ValueError, "invalid issue"):
                get_ci_sev_config(path)


class TestLLMInvocation(unittest.TestCase):
    def test_builds_claude_command_with_default_model(self):
        inv = _build_llm_invocation("fix it", Path("/tmp/wt"), LLMConfig("claude"))

        self.assertEqual(inv.cmd[:2], ["claude", "-p"])
        self.assertNotIn("fix it", inv.cmd)
        self.assertEqual(inv.stdin_input, "fix it")
        self.assertIn("--model", inv.cmd)
        self.assertEqual(inv.cmd[inv.cmd.index("--model") + 1], "opus")

    def test_builds_codex_command(self):
        inv = _build_llm_invocation(
            "fix it", Path("/tmp/wt"), LLMConfig("codex", "gpt-5.4")
        )

        self.assertEqual(inv.cmd[:2], ["codex", "exec"])
        self.assertIn("--json", inv.cmd)
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", inv.cmd)
        self.assertEqual(inv.cmd[inv.cmd.index("-C") + 1], "/tmp/wt")
        self.assertEqual(inv.stdin_input, "fix it")
        self.assertNotIn("fix it", inv.cmd)

    def test_escapes_embedded_nuls_in_codex_stdin_prompt(self):
        inv = _build_llm_invocation("fix\x00it", Path("/tmp/wt"), LLMConfig("codex"))

        self.assertEqual(inv.stdin_input, "fix\\x00it")
        self.assertNotIn("\x00", inv.stdin_input)

    def test_escapes_embedded_nuls_in_stdin_prompt(self):
        inv = _build_llm_invocation("fix\x00it", Path("/tmp/wt"), LLMConfig("claude"))

        self.assertEqual(inv.stdin_input, "fix\\x00it")
        self.assertNotIn("\x00", inv.stdin_input)

    def test_builds_metacode_command(self):
        inv = _build_llm_invocation(
            "fix it", Path("/tmp/wt"), LLMConfig("metacode", "provider/model")
        )

        self.assertEqual(inv.cmd[:2], ["metacode", "run"])
        self.assertIn("--yolo", inv.cmd)
        self.assertIn("--format", inv.cmd)
        self.assertEqual(inv.cmd[inv.cmd.index("--dir") + 1], "/tmp/wt")
        self.assertEqual(inv.cmd[-1], "fix it")


if __name__ == "__main__":
    unittest.main()
