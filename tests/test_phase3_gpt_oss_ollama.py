"""gpt-oss (e.g. gpt-oss:120b) served locally by Ollama's OpenAI-compatible endpoint.

No model API is called; LiteLLM's completion call is mocked.
"""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from utils import core_utils as core
from utils import model_config

MESSAGES = [{"role": "user", "content": "q"}]


def response(content="ok"):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15, prompt_tokens_details=None)
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(usage=usage, _hidden_params={"response_cost": None},
                           choices=[SimpleNamespace(finish_reason="stop", message=message)])


class GptOssRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        for patcher in (patch.object(model_config, "CONFIG_PATH", Path(self.temp.name, "config.yml")),
                        patch.dict(os.environ, {"BASE_URL": ""})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_gpt_oss_is_a_local_model_not_openai(self):
        self.assertEqual(core.get_model_provider("gpt-oss:120b"), "gpt-oss")
        self.assertEqual(model_config.get_default_base_url("gpt-oss:120b"), "http://localhost:11434/v1")
        self.assertFalse(core.is_reasoning_llm("gpt-oss:120b"))
        self.assertEqual(core.get_model_provider("gpt-6-luna"), "openai")

    def test_setup_needs_an_endpoint_not_an_api_key(self):
        with patch.object(core, "get_fhir_config", return_value={"PROJECT_ID": "p"}):
            self.assertTrue(core.validate_config("gpt-oss:120b", "http://localhost:11434/v1"))
            self.assertFalse(core.validate_config("gpt-oss:120b", None))


class GptOssTransportTests(unittest.TestCase):
    def test_request_uses_ollama_with_a_fixed_reasoning_effort(self):
        with patch.object(core.litellm, "completion", side_effect=[response()]) as completion, \
                patch.object(core.time, "sleep"):
            message, error, _ = core.safe_llm_call("gpt-oss:120b", MESSAGES, base_url="http://localhost:11434/v1")
        self.assertIsNone(error)
        self.assertEqual(message.content, "ok")
        kwargs = completion.call_args.kwargs
        self.assertEqual(kwargs["custom_llm_provider"], "openai")
        self.assertEqual(kwargs["extra_body"], {"reasoning_effort": core.GPT_OSS_REASONING_EFFORT})
        self.assertEqual(kwargs["api_key"], "ollama")

    def test_a_rejected_tool_call_is_sent_back_for_correction(self):
        # Ollama 0.40.13 answers gpt-oss's malformed tool-call JSON with HTTP 500
        rejected = Exception("litellm.InternalServerError: OpenAIException - error parsing tool call: "
                             "raw='{\"code\":\"answer = 1\"], \"timeout\": 10}', "
                             "err=invalid character ']' after object key:value pair")
        with patch.object(core.litellm, "completion", side_effect=[rejected, response()]) as completion, \
                patch.object(core.time, "sleep") as sleep:
            message, error, usage = core.safe_llm_call("gpt-oss:120b", MESSAGES, base_url="http://localhost:11434/v1")
        self.assertIsNone(error)
        self.assertEqual(message.content, "ok")
        self.assertEqual(usage["llm_calls"], 2)
        sleep.assert_not_called()
        retry = completion.call_args_list[1].kwargs["messages"]
        self.assertEqual(retry[:-1], MESSAGES)
        self.assertIn("invalid character ']' after object key:value pair", retry[-1]["content"])
        self.assertEqual(MESSAGES, [{"role": "user", "content": "q"}])  # the caller's history is untouched

    def test_other_errors_resend_the_same_request(self):
        with patch.object(core.litellm, "completion", side_effect=[Exception("timeout"), response()]) as completion, \
                patch.object(core.time, "sleep"):
            core.safe_llm_call("gpt-oss:120b", MESSAGES, base_url="http://localhost:11434/v1")
        self.assertEqual(completion.call_args_list[1].kwargs["messages"], MESSAGES)


if __name__ == "__main__":
    unittest.main()
