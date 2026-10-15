"""V7 real SDK serialization, isolated worker and owned gzip; synthetic data only."""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx2
from openai import AzureOpenAI

import run_agent as runner
from scripts.audit_store import restore_result_audit
from tests.test_independent_recompute_aux import Message, native
from tools import get_tool as original_tool
from utils import core_utils


PATIENT = "11111111-1111-1111-1111-111111111111"
ROWS = [{"resourceType": "Observation", "id": name,
         "subject": {"reference": "Patient/" + PATIENT},
         "valueQuantity": {"value": value}}
        for name, value in (("o1", 70), ("o2", 71))]
PACKET = {"text": "One synthetic numeric observation is initially held.",
          "resources": {"Observation": [ROWS[0]]}}
QUESTION = ("Question: What is the largest recorded synthetic numeric value?\n"
            f"Context:\nPatient FHIR ID is {PATIENT}.")


def program(label, *, defect=False):
    if defect:
        return "# B_RUNTIME_ERROR\nraise ValueError('synthetic recoverable failure')"
    return f"""# {label}_PRIVATE_PROGRAM_SENTINEL
r = max(retrieved_resources['Observation'], key=lambda item: item['valueQuantity']['value'])
answer = r['valueQuantity']['value']
evidence = [{{'resource_id': 'Observation/' + r['id'], 'field_path': 'valueQuantity.value', 'value': answer}}]
contributing_ids = ['Observation/' + r['id']]
operation = 'Find the largest numeric value among all held observations'
limitation = None
"""


def retrieval():
    return Message(calls=[{"id": "B-original-retrieval", "type": "function", "function": {
        "name": "get_resources_by_patient_fhir_id", "arguments": json.dumps({
            "resource_type": "Observation", "patient_fhir_id": PATIENT})}}])


def selection(*, choice="B", a_status="contradicted", b_status="supported"):
    return Message(json.dumps({"requirements": [{
        "requirement": "Return the largest recorded numeric value for the question patient",
        "A": {"status": a_status, "basis": "A returns70, but original Observation/o2 has71"},
        "B": {"status": b_status, "basis": "B fetched both observations, computed max and cited o2=71"}}],
        "choice": choice, "reason": "B resolves the concrete missing larger observation"}))


class ResidualTrajectoryTransportTests(unittest.TestCase):
    def exercise(self, *, feedback=False, missing_usage_at=None, choice="B",
                 a_status="contradicted", b_status="supported", agree=False):
        adapter = Path(__file__).resolve().parents[1] / ".local-setup/azure_adapter.py"
        if not adapter.is_file():
            self.skipTest("Optional public local Azure bridge unavailable")
        spec = importlib.util.spec_from_file_location("offline_v7_bridge", adapter)
        bridge = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: bridge}):
            spec.loader.exec_module(bridge)
        # No config discovery/load/install. The only transport is mocked below.
        bridge._settings = bridge.AzureSettings("https://offline-fixture.invalid",
            "2024-12-01-preview", "offline-deployment", "gpt-6-luna", 16384,
            "none", "non-secret-unit-test-placeholder")
        client = AzureOpenAI(azure_endpoint=bridge._settings.endpoint,
            api_version=bridge._settings.api_version, api_key=bridge._settings.api_key,
            timeout=60.0, max_retries=0)
        self.addCleanup(client.close)
        bridge._client = client
        self.assertEqual(client.max_retries, 0)
        self.assertIsInstance(client._client._transport, httpx2.HTTPTransport)

        # A's native output is followed by its natural final model turn.
        responses = [native(program("A")), Message('{"answer":999,"evidence":[],"limitation":null}')]
        if not agree:
            responses.append(retrieval())
        if feedback:
            responses.append(native(program("B", defect=True)))
        responses += [native(program("B")), Message('{"answer":999,"evidence":[],"limitation":null}')]
        if not agree:
            responses.append(selection(choice=choice, a_status=a_status, b_status=b_status))
        # Native SDK tool-call identities are distinct within a trajectory.
        # The reused fixture helper otherwise gives both B executions one ID,
        # correctly making the final execution/source binding ambiguous.
        for index, response in enumerate(responses):
            for call_index, call in enumerate(response.tool_calls or []):
                call["id"] = f"v7-response-{index}-call-{call_index}"
        pending = iter(responses)
        requests, source_reads = [], []

        def send(request, **_):
            body = json.loads(request.content)
            requests.append(body)
            message = next(pending).model_dump()
            data = {"id": "offline-v7", "object": "chat.completion", "created": 0,
                "model": "gpt-6-luna", "choices": [{"index": 0, "message": message,
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}]}
            if len(requests) != missing_usage_at:
                data["usage"] = {"prompt_tokens": 100, "completion_tokens": 80, "total_tokens": 180}
            return httpx2.Response(200, request=request, json=data)

        def source_fetch(refs):
            source_reads.append(deepcopy(refs))
            return {r["id"]: deepcopy(r) for r in ROWS if ("Observation", r["id"]) in refs}

        packet = Mock(side_effect=lambda *_a, **_kw: deepcopy(PACKET))
        fetch = Mock(return_value={"Observation": deepcopy(ROWS)})
        def dispatch(name):
            return fetch if name == "get_resources_by_patient_fhir_id" else original_tool(name)

        with ExitStack() as stack:
            stack.enter_context(patch("agent.base_agent.setup_api_keys"))
            stack.enter_context(patch("agent.single_agent.get_starting_packet", packet))
            stack.enter_context(patch("tools.field_check.fetch_from_store", side_effect=source_fetch))
            stack.enter_context(patch("agent.multi_turn_code_resource_agent.get_tool", side_effect=dispatch))
            stack.enter_context(patch.object(client._client, "send", side_effect=send))
            stack.enter_context(patch.object(core_utils.litellm, "completion", side_effect=bridge.azure_completion))
            stack.enter_context(patch.dict(os.environ, {"FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR": ""}))
            with tempfile.TemporaryDirectory() as directory:
                job = runner.prepare_multiprocessing_inputs([QUESTION], "multi_agent_v7", "gpt-6-luna",
                    False, None, False, {"max_total_tokens": 200000, "max_tool_calls": 30},
                    include_full_audit=True, audit_contexts=[{
                        "output_dir": directory, "question_id": "synthetic-v7-production"}])[0]
                compact = runner.dispatch_agent_job(job)
                self.assertEqual(compact["trace"], [])
                self.assertEqual(compact["revision"]["messages"], [])
                self.assertNotIn("retrieved_fhir_resources", compact)
                restored = restore_result_audit(compact, directory, question_id="synthetic-v7-production")
                json.dumps(restored, allow_nan=False)
                with self.assertRaises(ValueError):
                    restore_result_audit(compact, directory, question_id="another-owner")
        return restored, requests, source_reads, packet.call_count, fetch.call_count

    def test_fresh_retrieval_real_execution_selection_and_two_owned_snapshots(self):
        result, requests, source_reads, initializations, fetches = self.exercise()
        self.assertIsNone(result["error"], result)
        self.assertEqual(result["agent_answer"], 71)
        self.assertEqual(initializations, 2)
        self.assertEqual(fetches, 1)
        self.assertEqual(len(requests), 6)
        self.assertEqual(result["tokens"]["llm_calls"], 6)
        self.assertEqual(result["tokens"]["total"], 1080)
        # Two A operations; extra B initialization; B retrieval, execution, check.
        self.assertEqual(result["budget"]["tool_calls_run"], 6)
        self.assertEqual(len(source_reads), 2)
        self.assertIn("A_PRIVATE_PROGRAM_SENTINEL", json.dumps(requests[1]))
        self.assertNotIn("A_PRIVATE_PROGRAM_SENTINEL", json.dumps(requests[2:5]))
        self.assertEqual(requests[0]["messages"], requests[2]["messages"])
        self.assertEqual(requests[0]["tools"], requests[2]["tools"])
        self.assertNotIn("tools", requests[-1])
        self.assertEqual(requests[-1]["max_completion_tokens"], 2000)
        self.assertIn("A_PRIVATE_PROGRAM_SENTINEL", json.dumps(requests[-1]))
        self.assertIn("B_PRIVATE_PROGRAM_SENTINEL", json.dumps(requests[-1]))
        self.assertEqual({r["id"] for r in result["retrieved_fhir_resources"]["Observation"]}, {"o1", "o2"})
        snapshots = [m for m in result["revision"]["messages"] if m.get("kind") == "candidate_snapshot"]
        self.assertEqual(len(snapshots), 2)
        text = json.dumps(snapshots)
        self.assertIn("A_PRIVATE_PROGRAM_SENTINEL", text)
        self.assertIn("B_PRIVATE_PROGRAM_SENTINEL", text)

    def test_second_agent_receives_its_own_runtime_error_and_repairs_before_selection(self):
        result, requests, _, initializations, _ = self.exercise(feedback=True)
        self.assertIsNone(result["error"], result)
        self.assertEqual(result["agent_answer"], 71)
        self.assertEqual(initializations, 2)
        self.assertEqual(len(requests), 7)
        self.assertEqual(result["budget"]["tool_calls_run"], 7)
        self.assertIn("synthetic recoverable failure", json.dumps(requests[4]["messages"]))
        self.assertNotIn("A_PRIVATE_PROGRAM_SENTINEL", json.dumps(requests[4]["messages"]))

    def test_exact_agreement_keeps_whole_primary_without_selector(self):
        result, requests, _, initializations, fetches = self.exercise(agree=True)
        self.assertIsNone(result["error"], result)
        self.assertEqual(result["agent_answer"], 70)
        self.assertEqual(len(requests), 4)
        self.assertEqual(initializations, 2)
        self.assertEqual(fetches, 0)
        self.assertTrue(all(request.get("tools") for request in requests))
        self.assertEqual(result["budget"]["tool_calls_run"], 5)

    def test_missing_A_evidence_alone_does_not_authorize_B_selection(self):
        result, requests, _, _, _ = self.exercise(a_status="not_found")
        self.assertIsNone(result["error"], result)
        self.assertEqual(result["agent_answer"], 70)
        self.assertEqual(len(requests), 6)
        self.assertEqual([r["id"] for r in result["retrieved_fhir_resources"]["Observation"]], ["o1"])

    def test_unresolved_B_requirement_keeps_primary(self):
        result, _, _, _, _ = self.exercise(b_status="not_found")
        self.assertIsNone(result["error"], result)
        self.assertEqual(result["agent_answer"], 70)

    def test_unknown_admitted_B_usage_is_global_fatal_with_primary_snapshot_retained(self):
        result, requests, source_reads, initializations, _ = self.exercise(missing_usage_at=3)
        self.assertIsNone(result["agent_answer"])
        self.assertIsNotNone(result["error"])
        self.assertEqual(len(requests), 3)
        self.assertEqual(result["tokens"]["llm_calls"], 3)
        self.assertFalse(result["tokens"]["per_call"][-1]["usage_available"])
        self.assertIsNone(result["tokens"]["cost"])
        self.assertEqual(initializations, 2)
        self.assertEqual(len(source_reads), 1)
        self.assertIn("A_PRIVATE_PROGRAM_SENTINEL", json.dumps(result["revision"]["messages"]))

    def test_unknown_selector_usage_is_global_fatal_after_both_bound_candidates(self):
        result, requests, source_reads, _, _ = self.exercise(missing_usage_at=6)
        self.assertIsNone(result["agent_answer"])
        self.assertIsNotNone(result["error"])
        self.assertEqual(len(requests), 6)
        self.assertEqual(len(source_reads), 2)
        self.assertFalse(result["tokens"]["per_call"][-1]["usage_available"])


if __name__ == "__main__":
    unittest.main()
