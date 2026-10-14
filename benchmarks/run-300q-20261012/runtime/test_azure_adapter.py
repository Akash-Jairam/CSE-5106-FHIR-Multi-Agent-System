"""Offline integration checks: all sockets are disabled; Azure is simulated."""
from dataclasses import replace
import json
import multiprocessing as mp
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from urllib.parse import urlsplit
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / ".local-setup"
sys.path[:0] = [str(LOCAL), str(ROOT)]
os.environ["TIKTOKEN_CACHE_DIR"] = str(ROOT / ".cache/tiktoken")

import azure_adapter as adapter
from openai import _base_client
HTTP = _base_client.httpx2


def worker_probe(queue):
    import utils
    queue.put((utils.get_model_provider("gpt-6-luna"),
               utils.validate_model_api_key("gpt-6-luna")))


class AzureIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.block_network = patch("socket.socket.connect", side_effect=AssertionError("Network disabled in offline tests"))
        cls.block_network.start()
        cls.settings = adapter.AzureSettings.load(ROOT / "config.yml")

    @classmethod
    def tearDownClass(cls):
        cls.block_network.stop()

    def setUp(self):
        self.requests = []
        adapter.install(replace(self.settings, api_key="offline-test-key-not-real"))
        self.http = HTTP.Client(transport=HTTP.MockTransport(self.respond))
        adapter._client = adapter.make_client(adapter._settings, self.http)

    def tearDown(self):
        adapter._client.close()

    def respond(self, request):
        body = json.loads(request.content)
        self.requests.append((request, body))
        message = {"role": "assistant", "content": "5"}
        finish = "stop"
        if body.get("tools"):
            message = {"role": "assistant", "content": None,
                       "tool_calls": [{"id": "call-offline", "type": "function",
                                       "function": {"name": "lookup", "arguments": "{}"}}]}
            finish = "tool_calls"
        return HTTP.Response(200, json={
            "id": "chatcmpl-offline", "object": "chat.completion", "created": 1,
            "model": "gpt-6-luna", "choices": [{"index": 0, "finish_reason": finish,
                                                 "message": message}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}})

    def test_real_sdk_uses_supplied_azure_route_and_parameters(self):
        import utils
        message, error, usage = utils.safe_llm_call("gpt-6-luna", [{"role": "user", "content": "offline fixture"}])
        self.assertIsNone(error)
        self.assertEqual(message.content, "5")
        self.assertEqual(usage["total_tokens"], 13)
        request, body = self.requests[0]
        expected_url = (self.settings.endpoint.rstrip("/") + "/openai/deployments/" + self.settings.deployment
                        + "/chat/completions?api-version=" + self.settings.api_version)
        self.assertEqual(str(request.url), expected_url)
        self.assertEqual(request.headers["api-key"], "offline-test-key-not-real")
        self.assertEqual(body["model"], "gpt-6-luna")
        self.assertEqual(body["max_completion_tokens"], self.settings.max_completion_tokens)
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertNotIn("temperature", body)

    def test_agent_tool_calls_and_usage_preserve_unknown_cost(self):
        import utils
        from agent.base_agent import BaseAgent
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]
        message, error, usage = utils.safe_llm_call("gpt-6-luna", [{"role": "user", "content": "offline fixture"}], tools=tools)
        self.assertIsNone(error)
        self.assertEqual(message.tool_calls[0].function.name, "lookup")
        self.assertEqual(message.to_dict()["tool_calls"][0]["id"], "call-offline")
        agent = BaseAgent("gpt-6-luna")
        agent._update_usage(usage)
        self.assertIsNone(agent.total_usage["cost"])
        self.assertEqual(agent.total_usage["cost_status"], "not_estimated")
        self.assertEqual(agent.total_usage["total_tokens"], 13)
        self.assertEqual(agent.total_usage["provider"], "azure")

    def test_evaluator_uses_same_azure_client(self):
        with patch.object(sys, "argv", ["evaluation_metrics.py", "--input", "unused.json", "--model", "gpt-6-luna", "--num_processes", "1"]):
            evaluator = runpy.run_path(str(ROOT / "evaluation_metrics.py"), run_name="offline_evaluator")
        evaluator["setup_environment"]()
        self.assertEqual(evaluator["call_model"]("gpt-6-luna", "offline fixture"), "5")
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0][0].url.host, urlsplit(self.settings.endpoint).hostname)

    def test_explicit_completion_cap_reaches_azure(self):
        adapter.azure_completion(model=self.settings.model,
                                 messages=[{"role": "user", "content": "offline fixture"}],
                                 max_completion_tokens=37)
        self.assertEqual(self.requests[0][1]["max_completion_tokens"], 37)

    def test_unconfigured_model_is_rejected_without_request(self):
        with self.assertRaisesRegex(ValueError, "configured deployment"):
            adapter.azure_completion(model="o4-mini", messages=[])
        self.assertEqual(self.requests, [])

    def test_missing_key_does_not_create_client(self):
        with self.assertRaisesRegex(ValueError, "AZURE_OPENAI.API_KEY"):
            adapter.make_client(replace(self.settings, api_key=""))
        self.assertEqual(self.requests, [])

    def test_spawned_workers_load_adapter_without_key(self):
        import yaml
        config = {"AZURE_OPENAI": {"API_KEY": "", "ENDPOINT": self.settings.endpoint,
                  "API_VERSION": self.settings.api_version, "DEPLOYMENT": self.settings.deployment,
                  "MODEL": self.settings.model, "MAX_COMPLETION_TOKENS": 16384,
                  "REASONING_EFFORT": "none"}}
        with tempfile.TemporaryDirectory(dir=LOCAL) as directory:
            fixture = Path(directory) / "config.yml"
            fixture.write_text(yaml.safe_dump(config))
            env = os.environ.copy()
            env.pop("AZURE_OPENAI_API_KEY", None)
            env.pop("AZURE_API_KEY", None)
            env["FHIR_LOCAL_AZURE_CONFIG"] = str(fixture)
            env["PYTHONPATH"] = os.pathsep.join([str(LOCAL), str(ROOT)])
            with patch.dict(os.environ, env, clear=True):
                context = mp.get_context("spawn")
                queue = context.Queue()
                process = context.Process(target=worker_probe, args=(queue,))
                process.start()
                try:
                    self.assertEqual(queue.get(timeout=30), ("azure", False))
                    process.join(timeout=10)
                    self.assertEqual(process.exitcode, 0)
                finally:
                    if process.is_alive():
                        process.terminate()
                        process.join()
                    queue.close()
                    queue.join_thread()


if __name__ == "__main__":
    unittest.main(verbosity=2)
