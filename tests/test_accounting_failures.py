"""Synthetic SDK→bridge→usage→agent→worker→scorer accounting regressions.

No private configuration, patient data, model request or network connection.
The installed native SDK/default transport parses in-memory send responses.
"""
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

import litellm
from openai.types.chat import ChatCompletion

from agent.base_agent import BaseAgent, BUDGET_EXCEEDED, PROVIDER_ACCOUNTING_ERROR
from agent.single_agent import SingleAgent
from scripts.variability import is_failed, score_row
from utils import core_utils as core


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / ".local-setup/azure_adapter.py"
if ADAPTER_PATH.is_file():
    spec = importlib.util.spec_from_file_location("accounting_fixture_adapter", ADAPTER_PATH)
    adapter = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {spec.name: adapter}):
        spec.loader.exec_module(adapter)

ABSENT = object()
PATIENT = "00000000-0000-0000-0000-000000000001"
QUESTION = "Question: Synthetic existence question?\nContext:\nPatient FHIR ID is " + PATIENT + "."
EMPTY_REFERENCE = {"true_answer": "[]", "question": "Synthetic existence question?", "template": ""}


class SharedAccountingTests(unittest.TestCase):
    """The shared error contract also runs without the optional personal adapter."""
    def setUp(self):
        with patch("agent.base_agent.setup_api_keys"):
            self.agent = BaseAgent("synthetic")
        self.agent.set_budget(max_total_tokens=100)

        def one_call(_question):
            _message, error, _usage = self.agent._llm_call([])
            return {"answer": None, "error": error}

        self.agent._run = one_call

    def run_usage(self, usage):
        with patch("agent.base_agent.count_tokens_in_messages", return_value=1), patch(
                "agent.base_agent.safe_llm_call", return_value=(None, None, usage)):
            return core.run_agent_safe(self.agent, "Synthetic question")

    def test_missing_usage_after_admission_is_terminal_in_shared_base(self):
        row = self.run_usage({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "cost": None, "llm_calls": 1, "per_call": [{"prompt": 0, "completion": 0, "usage_available": False}]})
        self.assertIn(PROVIDER_ACCOUNTING_ERROR, row["error"])
        self.assertTrue(is_failed(row))
        self.assertEqual(score_row(row, EMPTY_REFERENCE), 0)
        self.assertFalse(row["tokens"]["per_call"][0]["usage_available"])

    def test_actual_overshoot_is_terminal_in_shared_base(self):
        row = self.run_usage({"prompt_tokens": 60, "completion_tokens": 41, "total_tokens": 101,
            "llm_calls": 1, "per_call": [{"prompt": 60, "completion": 41, "usage_available": True}]})
        self.assertIn(PROVIDER_ACCOUNTING_ERROR, row["error"])
        self.assertEqual(score_row(row, EMPTY_REFERENCE), 0)
        self.assertEqual(row["tokens"]["total"], 101)

    def test_preflight_refusal_is_unchanged_in_shared_base(self):
        self.agent.set_budget(max_total_tokens=0)
        with patch("agent.base_agent.count_tokens_in_messages", return_value=1), patch(
                "agent.base_agent.safe_llm_call") as provider:
            row = core.run_agent_safe(self.agent, "Synthetic question")
        provider.assert_not_called()
        self.assertIsNone(row["error"])
        self.assertEqual(score_row(row, EMPTY_REFERENCE), 1)


@unittest.skipUnless(ADAPTER_PATH.is_file(), "Optional personal Azure adapter is unavailable")
class AccountingFailureTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target in ((adapter.AzureSettings, "load"), (socket, "getaddrinfo"), (socket.socket, "connect")):
            self.stack.enter_context(patch.object(*target, side_effect=AssertionError("Configuration/network denied")))
        for name in ("read_text", "read_bytes", "open"):
            original = getattr(Path, name)

            def safe_read(path, *args, _original=original, **kwargs):
                if path.name.startswith("config") and path.suffix in {".yml", ".yaml"}:
                    raise AssertionError("Private configuration must not be read")
                return _original(path, *args, **kwargs)

            self.stack.enter_context(patch.object(Path, name, safe_read))
        self.stack.enter_context(patch.dict(os.environ, {"FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR": ""}))
        self.stack.enter_context(patch("agent.base_agent.setup_api_keys"))
        self.stack.enter_context(patch("agent.base_agent.count_tokens_in_messages", return_value=1))
        self.stack.enter_context(patch("agent.single_agent.get_starting_packet", return_value={
            "resources": {"Patient": [{"id": PATIENT}]}, "text": "Synthetic starting packet"}))
        self.stack.enter_context(patch.object(SingleAgent, "_check", return_value={"issues": [], "results": []}))
        self.settings = adapter.AzureSettings(
            endpoint="https://synthetic.invalid", api_version="synthetic-version", deployment="synthetic-deployment",
            model="gpt-6-luna", max_completion_tokens=1234, reasoning_effort="none", api_key="synthetic-unused-key")
        self.client = adapter.make_client(self.settings)
        self.addCleanup(self.client.close)
        self.stack.enter_context(patch.object(adapter, "_settings", self.settings))
        self.stack.enter_context(patch.object(adapter, "_client", self.client))
        self.stack.enter_context(patch.object(core.litellm, "completion", side_effect=adapter.azure_completion))
        self.requests, self.native_responses = [], []
        native_create = self.client.chat.completions.create

        def retain_native(**kwargs):
            response = native_create(**kwargs)
            self.native_responses.append(response)
            return response

        self.stack.enter_context(patch.object(self.client.chat.completions, "create", side_effect=retain_native))

    def send(self, usage=ABSENT):
        import httpx2 as httpx

        def send(request, **_kwargs):
            self.requests.append(json.loads(request.content))
            payload = {"id": "synthetic-completion", "object": "chat.completion", "created": 0,
                "model": self.settings.model, "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": '{"answer":null,"evidence":[],"limitation":null}'}}]}
            if usage is not ABSENT:
                payload["usage"] = usage
            return httpx.Response(200, json=payload, request=request)

        return patch.object(self.client._client, "send", side_effect=send)

    def run_single(self, usage=ABSENT, ceiling=200):
        agent = SingleAgent(self.settings.model)
        agent.set_budget(max_total_tokens=ceiling, max_tool_calls=4)
        with self.send(usage):
            row = core.run_agent_safe(agent, QUESTION, include_full_audit=True)
        return row

    def assert_accounting_failure(self, row):
        self.assertIsNone(row["agent_answer"])
        self.assertEqual(row["evidence"], [])
        self.assertIn(PROVIDER_ACCOUNTING_ERROR, row["error"])
        self.assertNotIn(BUDGET_EXCEEDED, row["error"])
        self.assertTrue(is_failed(row))
        self.assertEqual(score_row(row, EMPTY_REFERENCE), 0)
        self.assertEqual(row["tokens"]["llm_calls"], 1)

    def test_native_missing_usage_survives_bridge_and_is_terminal_not_empty_credit(self):
        row = self.run_single()
        self.assertEqual(len(self.native_responses), 1)
        self.assertIsInstance(self.native_responses[0], ChatCompletion)
        self.assertIsNone(self.native_responses[0].usage)
        self.assert_accounting_failure(row)
        self.assertFalse(row["tokens"]["per_call"][0]["usage_available"])
        self.assertIsNone(row["tokens"]["cost"])
        self.assertEqual(row["tokens"]["total"], 0)  # explicitly unavailable, not known/free

    def test_native_explicit_null_usage_is_also_unavailable(self):
        row = self.run_single(None)
        self.assertIsNone(self.native_responses[0].usage)
        self.assert_accounting_failure(row)
        self.assertFalse(row["tokens"]["per_call"][0]["usage_available"])

    def test_bridge_preserves_none_before_real_safe_llm_normalization(self):
        with self.send():
            output = adapter.azure_completion(model=self.settings.model, messages=[])
        self.assertIsInstance(output, litellm.ModelResponse)
        self.assertIsNone(output.usage)

    def test_explicit_reported_zero_usage_remains_known_and_null_can_be_correct(self):
        row = self.run_single({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        self.assertIsNotNone(self.native_responses[0].usage)
        self.assertIsNone(row["error"])
        self.assertFalse(is_failed(row))
        self.assertEqual(score_row(row, EMPTY_REFERENCE), 1)
        self.assertTrue(row["tokens"]["per_call"][0]["usage_available"])
        self.assertEqual((row["tokens"]["prompt"], row["tokens"]["completion"], row["tokens"]["total"]), (0, 0, 0))

    def test_reported_known_usage_is_unchanged_and_default_sdk_options_remain(self):
        row = self.run_single({"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30,
                              "prompt_tokens_details": {"cached_tokens": 5}})
        self.assertIsNone(row["error"])
        self.assertEqual(score_row(row, EMPTY_REFERENCE), 1)
        self.assertEqual((row["tokens"]["prompt"], row["tokens"]["completion"], row["tokens"]["total"],
                          row["tokens"]["cached_prompt"]), (20, 10, 30, 5))
        self.assertTrue(row["tokens"]["per_call"][0]["usage_available"])
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual(request["max_completion_tokens"], 135)
        self.assertEqual(request["reasoning_effort"], "none")
        self.assertEqual(request["model"], "synthetic-deployment")
        self.assertNotIn("temperature", request)
        self.assertEqual(self.client.max_retries, 0)
        self.assertEqual(self.client.timeout, 60.0)
        self.assertEqual(type(self.client._client._transport).__module__ + "." +
                         type(self.client._client._transport).__name__, "httpx2.HTTPTransport")

    def test_known_preflight_refusal_remains_nonerror_budget_abstention(self):
        with patch.object(core.litellm, "completion") as model:
            row = self.run_single(ceiling=0)
        model.assert_not_called()
        self.assertEqual(self.requests, [])
        self.assertIsNone(row["error"])
        self.assertIsNone(row["agent_answer"])
        self.assertIn("token budget", row["budget"]["stopped"])
        self.assertEqual(score_row(row, EMPTY_REFERENCE), 1)
        self.assertEqual(row["tokens"]["llm_calls"], 0)

    def test_actual_provider_overshoot_is_terminal_and_retains_reported_counters(self):
        row = self.run_single({"prompt_tokens": 150, "completion_tokens": 100, "total_tokens": 250})
        self.assert_accounting_failure(row)
        self.assertIn("provider usage exceeded", row["error"])
        self.assertEqual((row["tokens"]["prompt"], row["tokens"]["completion"], row["tokens"]["total"]), (150, 100, 250))
        self.assertTrue(row["tokens"]["per_call"][0]["usage_available"])

    def test_existing_provider_exception_is_already_terminal_and_still_retained(self):
        import httpx2 as httpx
        with patch.object(self.client._client, "send", side_effect=httpx.ReadTimeout("Synthetic timeout")):
            agent = SingleAgent(self.settings.model)
            agent.set_budget(max_total_tokens=200, max_tool_calls=4)
            row = core.run_agent_safe(agent, QUESTION)
        self.assertTrue(is_failed(row))
        self.assertEqual(score_row(row, EMPTY_REFERENCE), 0)
        self.assertIn("Max retries exceeded", row["error"])
        self.assertFalse(row["tokens"]["per_call"][0]["usage_available"])
        self.assertEqual(row["tokens"]["llm_calls"], 1)


if __name__ == "__main__":
    unittest.main()
