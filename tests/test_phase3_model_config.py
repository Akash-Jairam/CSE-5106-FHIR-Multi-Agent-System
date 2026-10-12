"""CC-28: one configured base model shared by every system and entry point.

No model API is called; LiteLLM's completion call is mocked.
"""
import ast
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_agent as runner
from scripts import run_experiments as experiments
from utils import core_utils as core
from utils import model_config

ROOT = Path(__file__).resolve().parent.parent


def response():
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15, prompt_tokens_details=None)
    return SimpleNamespace(usage=usage, _hidden_params={"response_cost": 0.0},
                           choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="ok"))])


class ResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Path(self.temp.name, "config.yml")
        for patcher in (patch.object(model_config, "CONFIG_PATH", self.config), patch.dict(os.environ, {"MODEL": ""})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_environment_then_config_then_default(self):
        self.assertEqual(model_config.get_default_model(), "gpt-6-luna")
        self.config.write_text('MODEL: "gpt-5-nano"\nOPENAI_API_KEY: "your-api-key"\n')
        self.assertEqual(model_config.get_default_model(), "gpt-5-nano")
        with patch.dict(os.environ, {"MODEL": "o4-mini"}):
            self.assertEqual(model_config.get_default_model(), "o4-mini")

    def test_placeholders_and_blanks_fall_through(self):
        self.config.write_text('MODEL: "your-model"\n')
        self.assertEqual(model_config.get_default_model(), "gpt-6-luna")
        with patch.dict(os.environ, {"MODEL": "   "}):
            self.assertEqual(model_config.get_default_model(), "gpt-6-luna")

    def test_unreadable_config_is_an_error_not_a_silent_default(self):
        self.config.write_text("MODEL: [unclosed\n")
        with self.assertRaisesRegex(ValueError, "Could not read the model"):
            model_config.get_default_model()

    def test_script_prints_the_model_for_shell_launchers(self):
        result = subprocess.run([sys.executable, "utils/model_config.py"], cwd=ROOT, capture_output=True, text=True,
                                env={**os.environ, "MODEL": "gpt-6-luna-test"}, check=True)
        self.assertEqual(result.stdout.strip(), "gpt-6-luna-test")


class EntryPointTests(unittest.TestCase):
    def test_every_entry_point_defaults_to_the_configured_model(self):
        with patch.dict(os.environ, {"MODEL": "shared-test-model"}):
            self.assertEqual(runner.build_parser().parse_args(["--agent_strategy", "direct_llm"]).model,
                             "shared-test-model")
            args = experiments.build_parser().parse_args(["plan"])
        self.assertEqual(args.model, "shared-test-model")
        # The demo and setup check parse arguments inside their scripts; check their --model defaults directly
        for path in ("demo/app.py", "scripts/check_setup.py", "evaluation_metrics.py"):
            with self.subTest(entry_point=path):
                tree = ast.parse((ROOT / path).read_text())
                defaults = [kw.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                            and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--model"
                            for kw in node.keywords if kw.arg == "default"]
                self.assertEqual([ast.unparse(d) for d in defaults], ["get_default_model()"])

    def test_every_experiment_arm_runs_the_same_model(self):
        args = experiments.build_parser().parse_args(["run", "--model", "gpt-6-luna"])
        models = {runner.build_parser().parse_args(experiments.runner_argv(arm, args)).model for arm in experiments.ARMS}
        self.assertEqual(models, {"gpt-6-luna"})

    def test_launchers_take_the_model_from_the_shared_setting(self):
        launchers = sorted(ROOT.glob("scripts/run_*.sh"))
        self.assertTrue(launchers)
        for path in launchers:
            text = path.read_text()
            if "--model" not in text or "run_demo.sh" in path.name:
                continue
            with self.subTest(launcher=path.name):
                self.assertIn('model="$(python utils/model_config.py)"', text)
                self.assertIsNone(re.search(r'^model="(?!\$\(python utils/model_config\.py\))', text, re.M))


class ReasoningModelTests(unittest.TestCase):
    def test_openai_reasoning_models_are_detected(self):
        for model, expected in (("gpt-6-luna", True), ("o4-mini", True), ("gpt-5-nano", True),
                                ("gpt-4.1-nano", False), ("gemini/gemini-2.5-flash", False),
                                ("qwen/qwen3", False), ("test", False)):
            with self.subTest(model=model):
                self.assertEqual(core.is_reasoning_llm(model), expected)

    def test_gpt_6_luna_requests_omit_temperature_and_cap_completion_tokens(self):
        with patch.object(core.litellm, "completion", return_value=response()) as call:
            _, error, _ = core.safe_llm_call("gpt-6-luna", [{"role": "user", "content": "q"}], max_completion_tokens=50)
        self.assertIsNone(error)
        kwargs = call.call_args.kwargs
        self.assertIsNone(kwargs["temperature"])
        self.assertEqual(kwargs["max_completion_tokens"], 50)
        self.assertNotIn("max_tokens", kwargs)

    def test_non_reasoning_models_keep_temperature_zero_and_max_tokens(self):
        with patch.object(core.litellm, "completion", return_value=response()) as call:
            core.safe_llm_call("gpt-4.1-nano", [{"role": "user", "content": "q"}], max_completion_tokens=50)
        kwargs = call.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertEqual(kwargs["max_tokens"], 50)


if __name__ == "__main__":
    unittest.main()
