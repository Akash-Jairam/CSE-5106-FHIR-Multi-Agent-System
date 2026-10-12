"""Exercise provider parameter mapping through a real SDK and offline transport."""
import json
import os
from pathlib import Path
import runpy
import socket
import sys
import unittest
from unittest.mock import patch

import httpx
import openai
from litellm.llms.openai.openai import OpenAIChatCompletion

from utils import core_utils as core


TOOLS = [{"type": "function", "function": {
    "name": "inspect_record", "description": "Inspect a synthetic record.",
    "parameters": {"type": "object", "properties": {}},
}}]


class LunaTransportTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        sdk_httpx = getattr(openai._base_client, "httpx2", httpx)

        def respond(request):
            self.requests.append(json.loads(request.content))
            return sdk_httpx.Response(200, json={
                "id": "offline-completion", "object": "chat.completion",
                "created": 1, "model": "gpt-6-luna",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                          "total_tokens": 15},
            })

        transport = sdk_httpx.MockTransport(respond)
        http_client = sdk_httpx.Client(transport=transport)
        self.addCleanup(http_client.close)
        self.client = openai.OpenAI(api_key="offline-test-key-not-real",
                                    base_url="https://offline.invalid/v1",
                                    http_client=http_client, max_retries=0)
        self.addCleanup(self.client.close)
        for patcher in (
                patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test-key-not-real"}),
                patch.object(core.litellm, "telemetry", False),
                patch.object(OpenAIChatCompletion, "_get_openai_client", return_value=self.client),
                patch.object(socket, "socket", side_effect=AssertionError("Network is forbidden"))):
            patcher.start()
            self.addCleanup(patcher.stop)

    def call(self, model, tools=None, cap=50):
        response, error, usage = core.safe_llm_call(
            model, [{"role": "user", "content": "Synthetic transport test."}],
            tools=tools, max_completion_tokens=cap, max_retries=1)
        self.assertIsNone(error)
        self.assertEqual(response.content, "ok")
        self.assertEqual(usage["total_tokens"], 15)
        self.assertEqual(len(self.requests), 1)
        return self.requests[0]

    def assert_luna_payload(self, payload, cap=50):
        self.assertEqual(payload["model"], "gpt-6-luna")
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["max_completion_tokens"], cap)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("max_tokens", payload)

    def test_luna_tool_request_survives_litellm_and_sdk_mapping(self):
        payload = self.call("gpt-6-luna", tools=TOOLS)
        self.assert_luna_payload(payload)
        self.assertEqual(payload["tools"], TOOLS)
        self.assertTrue(payload["parallel_tool_calls"])

    def test_luna_without_tools_uses_the_same_reasoning_mode(self):
        payload = self.call("gpt-6-luna")
        self.assert_luna_payload(payload)
        self.assertNotIn("tools", payload)
        self.assertNotIn("parallel_tool_calls", payload)

    def test_luna_unbudgeted_request_still_uses_none_reasoning(self):
        payload = self.call("gpt-6-luna", cap=None)
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertNotIn("temperature", payload)
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("max_completion_tokens", payload)

    def test_actual_secondary_judge_uses_the_shared_luna_transport(self):
        argv = ["evaluation_metrics.py", "--input", "unused", "--skip_llm_judge"]
        script = Path(__file__).resolve().parent.parent / "evaluation_metrics.py"
        with patch.object(sys, "argv", argv), patch.dict(os.environ, {"MODEL": "gpt-6-luna"}):
            namespace = runpy.run_path(str(script))
        result = namespace["call_model"]("gpt-6-luna", "Synthetic judge transport test.")
        self.assertEqual(result, "ok")
        self.assertEqual(len(self.requests), 1)
        payload = self.requests[0]
        self.assertEqual(payload["model"], "gpt-6-luna")
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertNotIn("temperature", payload)
        self.assertNotIn("max_tokens", payload)

    def test_explicit_openai_luna_prefix_is_normalized_by_provider(self):
        payload = self.call("openai/gpt-6-luna", tools=TOOLS, cap=75)
        self.assert_luna_payload(payload, cap=75)

    def test_luna_does_not_depend_on_new_litellm_reasoning_metadata(self):
        with patch.object(core.litellm, "supports_reasoning", return_value=False):
            payload = self.call("gpt-6-luna", tools=TOOLS)
        self.assert_luna_payload(payload)

    def test_luna_metadata_lookup_failure_does_not_change_request(self):
        with patch.object(core.litellm, "supports_reasoning", side_effect=KeyError("unknown model")):
            payload = self.call("gpt-6-luna", tools=TOOLS)
        self.assert_luna_payload(payload)

    def test_non_reasoning_openai_model_preserves_its_request_parameters(self):
        payload = self.call("gpt-4.1-nano", tools=TOOLS)
        self.assertEqual(payload["model"], "gpt-4.1-nano")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["max_tokens"], 50)
        self.assertNotIn("max_completion_tokens", payload)
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(payload["tools"], TOOLS)


if __name__ == "__main__":
    unittest.main()
