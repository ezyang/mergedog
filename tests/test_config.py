import json
import tempfile
import unittest
from pathlib import Path

from mergedog.config import (
    LLMConfig,
    get_llm_config,
    set_llm_config,
)
from mergedog.claude import _build_llm_invocation


class TestLLMConfig(unittest.TestCase):
    def test_default_is_claude_opus(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = get_llm_config(Path(d) / "config.json")

        self.assertEqual(cfg.provider, "claude")
        self.assertIsNone(cfg.model)
        self.assertEqual(cfg.effective_model, "opus")

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
        self.assertEqual(inv.cmd[-1], "fix it")

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
