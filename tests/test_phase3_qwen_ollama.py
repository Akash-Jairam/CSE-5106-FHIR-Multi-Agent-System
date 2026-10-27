"""Qwen3 (e.g. qwen3:4b) served locally by Ollama's OpenAI-compatible endpoint.

No model API is called; LiteLLM's completion call is mocked.
"""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_agent as runner
from utils import core_utils as core
from utils import model_config

MESSAGES = [{"role": "user", "content": "q"}]


def response(content=None, tool_calls=None, finish_reason="stop"):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15, prompt_tokens_details=None)
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(usage=usage, _hidden_params={"response_cost": None},
                           choices=[SimpleNamespace(finish_reason=finish_reason, message=message)])


def tool_call(name="get_resources", arguments='{"resource_type": "Patient"}'):
    return SimpleNamespace(id="call_1", type="function", function=SimpleNamespace(name=name, arguments=arguments))


class BaseUrlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Path(self.temp.name, "config.yml")
        for patcher in (patch.object(model_config, "CONFIG_PATH", self.config),
                        patch.dict(os.environ, {"BASE_URL": ""})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_qwen_defaults_to_local_ollama_and_other_models_to_none(self):
        self.assertEqual(model_config.get_default_base_url("qwen3:4b"), "http://localhost:11434/v1")
        self.assertIsNone(model_config.get_default_base_url("gpt-6-luna"))

    def test_environment_then_config_then_ollama(self):
        self.config.write_text('BASE_URL: "http://gpu-box:8000/v1"\n')
        self.assertEqual(model_config.get_default_base_url("qwen3:4b"), "http://gpu-box:8000/v1")
        with patch.dict(os.environ, {"BASE_URL": "http://env:11434/v1"}):
            self.assertEqual(model_config.get_default_base_url("qwen3:4b"), "http://env:11434/v1")
        self.config.write_text('BASE_URL: "your-base-url"\n')
        self.assertEqual(model_config.get_default_base_url("qwen3:4b"), "http://localhost:11434/v1")

    def test_run_agent_resolves_the_default_for_the_chosen_model(self):
        with patch.object(runner, "load_inputs", side_effect=SystemExit) as load:
            with self.assertRaises(SystemExit):
                runner.main(["--agent_strategy", "direct_llm", "--model", "qwen3:4b",
                             "--output", os.path.join(self.temp.name, "out.json")])
        self.assertTrue(load.called)
        args = runner.build_parser().parse_args(["--agent_strategy", "direct_llm", "--model", "qwen3:4b"])
        self.assertIsNone(args.base_url)  # resolved in main(), after --model is known


class QwenTransportTests(unittest.TestCase):
    def call(self, *replies, **kwargs):
        with patch.object(core.litellm, "completion", side_effect=list(replies)) as completion, \
                patch.object(core.time, "sleep"):
            result = core.safe_llm_call("qwen3:4b", MESSAGES, base_url="http://localhost:11434/v1", **kwargs)
        return result, completion

    def test_request_uses_the_openai_transport_without_thinking(self):
        (_, error, _), completion = self.call(response("ok"), max_completion_tokens=50)
        self.assertIsNone(error)
        kwargs = completion.call_args.kwargs
        self.assertEqual(kwargs["custom_llm_provider"], "openai")
        self.assertEqual(kwargs["base_url"], "http://localhost:11434/v1")
        self.assertEqual(kwargs["extra_body"], {"think": False})
        self.assertEqual(kwargs["api_key"], "ollama")
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertEqual(kwargs["max_tokens"], 50)

    def test_native_tool_calls_are_kept(self):
        calls = [tool_call()]
        (message, error, _), _ = self.call(response("", calls, finish_reason="tool_calls"))
        self.assertIsNone(error)
        self.assertIs(message.tool_calls, calls)

    def test_raw_tool_call_text_is_still_parsed(self):
        text = 'Looking it up.\n<tool_call>\n{"name": "get_resources", "arguments": {"resource_type": "Patient"}}\n</tool_call>'
        (message, error, _), _ = self.call(response(text))
        self.assertIsNone(error)
        self.assertEqual(message.content, "Looking it up.")
        self.assertEqual(message.tool_calls[0]["function"]["name"], "get_resources")

    def test_think_blocks_are_stripped(self):
        (message, _, _), _ = self.call(response('<think>\nreasoning\n</think>\n\n{"answer": 1}'))
        self.assertEqual(message.content, '{"answer": 1}')

    def test_a_reply_that_is_only_thinking_is_retried(self):
        (message, error, usage), completion = self.call(response("<think>still going"), response("done"))
        self.assertIsNone(error)
        self.assertEqual(message.content, "done")
        self.assertEqual(completion.call_count, 2)
        self.assertEqual(usage["llm_calls"], 2)


if __name__ == "__main__":
    unittest.main()
