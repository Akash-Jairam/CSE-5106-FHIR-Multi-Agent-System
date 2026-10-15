"""V4 actual SDK JSON encoding and worker audit round trip, with network mocked."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import litellm
from openai import AzureOpenAI

import run_agent as runner
from agent.plan_conditioned_agent import PlanLedgerSolver
from agent.controller import FixedController
from agent.query_contract_agent import HostBoundaryMessage
from scripts.audit_store import externalize_result_audit, restore_result_audit
from tests import test_query_contract_agent as query_fixture
from tests.test_query_contract_agent import (
    BARE, FixtureMessage, QUESTION, CODE, fetch, final_call, reply,
)
from utils import core_utils


def advisory_plan():
    quote = "greater than 70"
    start = BARE.index(quote)
    return {"status": "READY", "steps": [
        {"id": "p1", "operator": "filter", "inputs": [], "quote": quote,
         "span": [start, start + len(quote)], "requirement": "strictly greater than 70"}
    ], "unknowns": []}


class V4TransportTests(unittest.TestCase):
    def test_boundary_is_plain_api_mapping_with_lossless_nested_audit(self):
        # The inherited append path needs only the two histories for this host
        # message. No model or credential setup is needed for the pure boundary.
        solver = PlanLedgerSolver.__new__(PlanLedgerSolver)
        solver.messages, solver.model_messages = [], []
        boundary = HostBoundaryMessage()
        solver._append_message(boundary)
        self.assertEqual(solver.model_messages, [boundary.model_dump()])
        self.assertEqual(solver.messages, [boundary.to_dict()])
        self.assertEqual(type(solver.model_messages[0]), dict)
        self.assertNotIn("model_call", solver.model_messages[0])
        self.assertNotIn("phase", solver.model_messages[0])
        json.dumps({"messages": solver.model_messages}, allow_nan=False)
        nested = {"trace": deepcopy(solver.messages), "latency": 1,
                  "revision": {"messages": deepcopy(solver.messages),
                               "primary_revision": {"messages": deepcopy(solver.messages),
                                                    "nested_trace": [deepcopy(solver.messages)]}}}
        with tempfile.TemporaryDirectory() as directory:
            compact = externalize_result_audit(nested, directory, question_id="synthetic-v4")
            json.dumps(compact, allow_nan=False)
            restored = restore_result_audit(compact, directory, question_id="synthetic-v4")
        self.assertEqual(restored, nested)
        self.assertFalse(restored["revision"]["primary_revision"]["messages"][0]["model_call"])
        self.assertEqual(restored["revision"]["primary_revision"]["messages"][0]["phase"], "solver_boundary")

    def test_actual_sdk_advice_source_feedback_dispatch_and_worker_gzip(self):
        self._exercise_actual_sdk(native_default=False)

    def test_native_default_httpx2_sdk_source_feedback_and_worker_gzip(self):
        self._exercise_actual_sdk(native_default=True)

    def _exercise_actual_sdk(self, native_default):
        fixture = query_fixture.QueryContractIntegrationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        wrong = CODE.replace("'value': r['valueQuantity']['value']", "'value': 999")
        responses = iter([
            reply(json.dumps(advisory_plan())), fetch(),
            final_call(wrong, "wrong-citation"), final_call(CODE, "corrected-citation"),
        ])
        requests = []

        wire = httpx
        if native_default:
            try:
                import httpx2 as wire
            except ImportError:
                self.skipTest("Installed SDK native HTTPX2 dependency is unavailable")

        def handle(request, **_kwargs):
            body = json.loads(request.content)
            requests.append(body)
            response = FixtureMessage(**vars(next(responses))).model_dump()
            return wire.Response(200, request=request, json={
                "id": "offline-v4-response", "object": "chat.completion", "created": 0,
                "model": "gpt-6-luna", "choices": [{"index": 0, "message": response,
                    "finish_reason": "tool_calls" if response.get("tool_calls") else "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            })

        settings = {"azure_endpoint":"https://offline-fixture.invalid",
                    "api_version":"2024-12-01-preview",
                    "api_key":"non-secret-unit-test-placeholder",
                    "timeout":60.0,"max_retries":0}
        if native_default:
            # Construct the actual SDK default client without http_client or a
            # replacement transport. Intercept its send in memory; no sockets,
            # HTTPX2 pool behavior, network or provider availability are tested.
            client = AzureOpenAI(**settings)
            self.assertIsInstance(client._client, wire.Client)
            self.assertIsInstance(client._client._transport, wire.HTTPTransport)
            self.assertIn("httpx2.Client", [c.__module__+"."+c.__qualname__
                                          for c in type(client._client).__mro__])
            self.assertEqual(type(client._client).__module__+"."+type(client._client).__name__,
                             "openai._base_client.SyncHttpxClientWrapper")
            send = patch.object(client._client,"send",side_effect=handle)
            send.start()
            self.addCleanup(send.stop)
        else:
            http_client = httpx.Client(transport=httpx.MockTransport(handle))
            self.addCleanup(http_client.close)
            client = AzureOpenAI(**settings,http_client=http_client)
        self.addCleanup(client.close)
        self.assertEqual(client.max_retries,0)

        def sdk_completion(**kwargs):
            # Pass actual histories unchanged to the SDK JSON encoder. A
            # model_dump fallback here would conceal a boundary transport bug.
            parameters = {"model": "offline-deployment", "messages": kwargs["messages"],
                          "max_completion_tokens": min(kwargs.get("max_completion_tokens", 16384), 16384),
                          "reasoning_effort":"none"}
            if kwargs.get("tools"):
                parameters["tools"] = kwargs["tools"]
            response = client.chat.completions.create(**parameters)
            return litellm.ModelResponse(**response.model_dump(exclude_none=True))

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(core_utils.litellm, "completion", side_effect=sdk_completion):
            job = runner.prepare_multiprocessing_inputs(
                [QUESTION], "multi_agent_v4", "gpt-6-luna", False, None, False,
                {"max_total_tokens": 200000, "max_tool_calls": 30},
                include_full_audit=True,
                audit_contexts=[{"output_dir": directory, "question_id": "synthetic-v4"}],
            )[0]
            compact = runner.dispatch_agent_job(job)
            json.dumps(compact, allow_nan=False)
            restored = restore_result_audit(compact, directory, question_id="synthetic-v4")
            json.dumps(restored, allow_nan=False)
            self.assertTrue((Path(directory) / compact["audit_artifacts"]["path"]).is_file())

        self.assertIsNone(restored["error"], restored.get("error"))
        self.assertEqual(restored["agent_answer"], 2)
        self.assertEqual(len(requests), 4)
        self.assertEqual(restored["tokens"]["llm_calls"], 4)
        self.assertEqual(restored["budget"]["tool_calls_run"], 5)
        primary_revision = restored["revision"]["primary_revision"]
        self.assertEqual(primary_revision["final_check_issues"], [])
        self.assertTrue(primary_revision["revised"])
        self.assertIsNone(primary_revision["revision_error"])

        self.assertFalse(requests[0].get("tools"))
        planner_input = "\n".join(message.get("content") or "" for message in requests[0]["messages"])
        self.assertNotIn(CODE, planner_input)
        self.assertNotIn(wrong, planner_input)
        self.assertNotIn('"o1"', planner_input)
        self.assertNotIn('"o2"', planner_input)
        planner_context = json.loads([message["content"] for message in requests[0]["messages"]
                                      if message.get("role") == "user"][0])
        self.assertEqual(planner_context, FixedController.question_context(QUESTION))
        first_solver_input = "\n".join(message.get("content") or "" for message in requests[1]["messages"])
        self.assertIn("strictly greater than 70", first_solver_input)
        self.assertIn("p1", first_solver_input)
        self.assertIn(BARE, first_solver_input)
        expected_tools = {"get_resources_by_patient_fhir_id", "get_resources_by_resource_id",
                          "execute_python_code", "check_fields"}
        for request in requests[1:]:
            self.assertEqual({tool["function"]["name"] for tool in request["tools"]}, expected_tools)
            self.assertTrue(all(type(message) is dict for message in request["messages"]))
        self.assertTrue(all(request["reasoning_effort"]=="none" and request["max_completion_tokens"]<=16384
                            for request in requests))

        wire_boundaries = [message for message in requests[3]["messages"]
                           if message.get("content") == HostBoundaryMessage.content]
        self.assertEqual(wire_boundaries, [HostBoundaryMessage().model_dump()])
        feedback = [message for message in restored["trace"]
                    if message.get("role") == "user"
                    and isinstance(message.get("content"), str)
                    and message["content"].startswith("The field check found problems")]
        self.assertEqual(len(feedback), 1)
        audit_boundaries = [message for message in restored["trace"]
                            if message.get("phase") == "solver_boundary"]
        self.assertEqual(len(audit_boundaries), 2)
        self.assertTrue(all(message["model_call"] is False for message in audit_boundaries))


if __name__ == "__main__":
    unittest.main()
