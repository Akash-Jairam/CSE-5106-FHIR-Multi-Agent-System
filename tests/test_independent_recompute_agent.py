"""Actual Single→native SDK→auxiliary→worker/gzip, synthetic records only."""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx2
from openai import AzureOpenAI

import run_agent as runner
from scripts.audit_store import restore_result_audit
from tests.test_independent_recompute_aux import (CITATION, CODE, CONTEXT, Message,
    WORKSPACE, candidate, native, same, select)
from utils import core_utils

PATIENT = "11111111-1111-1111-1111-111111111111"
WORKSPACE = deepcopy(WORKSPACE)
WORKSPACE["Observation"][0]["subject"]["reference"] = "Patient/" + PATIENT
CONTEXT = {**CONTEXT, "patient_fhir_id": PATIENT}
QUESTION = ("Question: " + CONTEXT["question"] + "\nContext:\nPatient FHIR ID is " + PATIENT + ".\n"
            "Assume current time is " + CONTEXT["current_time"] + ".")
PRIMARY_CODE = CODE.replace("answer = r['valueQuantity']['value']", "answer = r['valueQuantity']['value'] - 1")
PRIMARY_CODE = PRIMARY_CODE.replace("'value':answer", "'value':r['valueQuantity']['value']")
PRIMARY_CODE = "# PRIMARY_PRIVATE_PROGRAM_SENTINEL\n" + PRIMARY_CODE
PRIMARY_CODE = PRIMARY_CODE.replace("Select the recorded weight.", "PRIMARY_PRIVATE_OPERATION_SENTINEL")


class V6ProductionPathTests(unittest.TestCase):
    def exercise(self, auxiliary="select_B", *, prose_primary=False):
        adapter_path = Path(__file__).resolve().parents[1] / ".local-setup" / "azure_adapter.py"
        if not adapter_path.is_file():
            self.skipTest("Optional local public Azure bridge unavailable")
        spec = importlib.util.spec_from_file_location("offline_v6_azure_bridge", adapter_path)
        bridge = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: bridge}):
            spec.loader.exec_module(bridge)
        # Explicit placeholder settings only: AzureSettings.load/install and all
        # credential/config discovery paths are deliberately never called.
        bridge._settings = bridge.AzureSettings("https://offline-fixture.invalid",
            "2024-12-01-preview", "offline-deployment", "gpt-6-luna", 16384,
            "none", "non-secret-unit-test-placeholder")
        client = AzureOpenAI(azure_endpoint=bridge._settings.endpoint,
            api_version=bridge._settings.api_version, api_key=bridge._settings.api_key,
            timeout=60.0, max_retries=0)
        self.addCleanup(client.close)
        bridge._client = client
        self.assertIsInstance(client._client, httpx2.Client)
        self.assertIsInstance(client._client._transport, httpx2.HTTPTransport)
        self.assertEqual(client.max_retries, 0)
        a_call = native(PRIMARY_CODE)
        a_call.tool_calls[0]["id"] = "actual-primary-first-final"
        # A's native program is followed by a real natural model turn. Its prose
        # cannot replace the executed artifact, and no host early handoff occurs.
        a_prose = Message(json.dumps({"answer": 999, "evidence": [], "limitation": None}))
        first_prose = Message(json.dumps({key: value for key, value in candidate(70).items()
                                         if key in ("answer", "evidence", "limitation")}))
        responses = [first_prose] if prose_primary else [a_call, a_prose]
        if not prose_primary:
            if auxiliary == "invalid_B":
                responses.append(Message("No native program"))
            elif auxiliary == "missing_usage":
                responses.append(native())
            else:
                responses += [native(), select()]
        pending = iter(responses)
        requests, source_reads = [], []
        def send(request, **_):
            body = json.loads(request.content)
            requests.append(body)
            response = next(pending).model_dump()
            payload = {"id": "offline-v6-worker", "object": "chat.completion", "created": 0,
                "model": "gpt-6-luna", "choices": [{"index": 0, "message": response,
                    "finish_reason": "tool_calls" if response.get("tool_calls") else "stop"}]}
            if not (auxiliary == "missing_usage" and len(requests) == 3):
                payload["usage"] = {"prompt_tokens": 100, "completion_tokens": 80, "total_tokens": 180}
            return httpx2.Response(200, request=request, json=payload)
        def fetch(refs):
            source_reads.append(deepcopy(refs))
            return {"o1": deepcopy(WORKSPACE["Observation"][0])} if ("Observation", "o1") in refs else {}

        with ExitStack() as stack:
            stack.enter_context(patch("agent.base_agent.setup_api_keys"))
            stack.enter_context(patch("agent.single_agent.get_starting_packet", return_value={
                "resources": deepcopy(WORKSPACE), "text": "One synthetic Observation is held."}))
            stack.enter_context(patch("tools.field_check.fetch_from_store", side_effect=fetch))
            stack.enter_context(patch.object(client._client, "send", side_effect=send))
            stack.enter_context(patch.object(core_utils.litellm, "completion", side_effect=bridge.azure_completion))
            stack.enter_context(patch.dict(os.environ, {"FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR": ""}))
            with tempfile.TemporaryDirectory() as directory:
                job = runner.prepare_multiprocessing_inputs([QUESTION], "multi_agent_v6", "gpt-6-luna",
                    False, None, False, {"max_total_tokens": 200000, "max_tool_calls": 30},
                    include_full_audit=True, audit_contexts=[{
                        "output_dir": directory, "question_id": "synthetic-v6-production"}])[0]
                compact = runner.dispatch_agent_job(job)
                json.dumps(compact, allow_nan=False)
                restored = restore_result_audit(compact, directory, question_id="synthetic-v6-production")
                json.dumps(restored, allow_nan=False)
                self.assertTrue((Path(directory) / compact["audit_artifacts"]["path"]).is_file())
                with self.assertRaises(ValueError):
                    restore_result_audit(compact, directory, question_id="different-owner")
        return restored, requests, source_reads

    def test_four_actual_sdk_calls_natural_primary_selection_and_gzip_worker(self):
        result, requests, source_reads = self.exercise()
        self.assertIsNone(result["error"], result)
        self.assertEqual(result["agent_answer"], 71)
        self.assertEqual(result["tokens"]["llm_calls"], 4)
        self.assertEqual(result["tokens"]["total"], 720)
        self.assertEqual(result["budget"]["tool_calls_run"], 4)
        self.assertEqual(len(requests), 4)
        self.assertEqual(len(source_reads), 2)
        revision = result["revision"]
        self.assertTrue(revision["independent_candidate_selected"])
        self.assertEqual(revision["final_check_issues"], [])
        self.assertEqual(revision["primary"]["answer"], 70)
        self.assertEqual(revision["final"]["answer"], 71)
        self.assertEqual(revision["final"]["evidence"], [CITATION])
        self.assertEqual(revision["primary_diagnostics"]["tokens"], 360)
        self.assertEqual(revision["primary_diagnostics"]["max_iterations"], 30)
        self.assertEqual(revision["primary_diagnostics"]["model_call_attempts_recorded"], 2)
        self.assertEqual(revision["recompute"]["phase"]["tokens"], 360)
        self.assertEqual(revision["recompute"]["model_calls"], 2)
        self.assertEqual(revision["recompute"]["tool_calls"], 2)
        # Natural turn2 actually observes the primary execution, before B exists.
        self.assertIn("PRIMARY_PRIVATE_PROGRAM_SENTINEL", json.dumps(requests[1]["messages"]))
        self.assertTrue(any(message.get("role") == "tool" for message in requests[1]["messages"]))
        self.assertTrue(requests[1].get("tools"))
        b_packet = json.loads(requests[2]["messages"][1]["content"])
        self.assertEqual(set(b_packet), {"context", "workspace_sha256", "held_inventory", "retrieval_allowed"})
        self.assertNotIn("PRIMARY_PRIVATE_PROGRAM_SENTINEL", json.dumps(requests[2]))
        self.assertNotIn("PRIMARY_PRIVATE_OPERATION_SENTINEL", json.dumps(requests[2]))
        self.assertEqual(len(requests[2]["messages"]), 2)
        self.assertEqual([item["function"]["name"] for item in requests[2]["tools"]], ["execute_python_code"])
        self.assertEqual([requests[i]["max_completion_tokens"] for i in (2, 3)], [4000, 1800])
        self.assertNotIn("tools", requests[3])
        selector = json.loads(requests[3]["messages"][1]["content"])
        self.assertEqual(selector["A"]["program"], PRIMARY_CODE)
        self.assertEqual(selector["B"]["program"], CODE)
        self.assertTrue(same(selector["B"]["artifact"], revision["final"]))
        self.assertEqual(revision["recompute"]["B_source_check"]["n_valid"], 1)
        self.assertEqual(revision["primary_revision"]["final_check_issues"], [])
        self.assertFalse(any(message.get("phase") == "solver_boundary" for message in result["trace"]))

    def test_invalid_B_preserves_actual_A_without_hidden_selection_or_free_tools(self):
        result, requests, source_reads = self.exercise("invalid_B")
        self.assertIsNone(result["error"], result)
        self.assertEqual(result["agent_answer"], 70)
        self.assertEqual(len(requests), 3)
        self.assertEqual(result["tokens"]["llm_calls"], 3)
        self.assertEqual(result["budget"]["tool_calls_run"], 2)
        self.assertEqual(len(source_reads), 1)
        revision = result["revision"]
        self.assertFalse(revision["independent_candidate_selected"])
        self.assertFalse(revision["recompute"]["fatal"])
        self.assertEqual(revision["final"]["answer"], revision["primary"]["answer"])

    def test_prose_only_actual_primary_is_preserved_and_auxiliary_not_invented(self):
        result, requests, source_reads = self.exercise(prose_primary=True)
        self.assertIsNone(result["error"], result)
        self.assertEqual(result["agent_answer"], 70)
        self.assertEqual(len(requests), 1)
        self.assertEqual(result["tokens"]["llm_calls"], 1)
        self.assertEqual(result["budget"]["tool_calls_run"], 1)
        self.assertEqual(len(source_reads), 1)
        self.assertFalse(result["revision"]["primary_diagnostics"]["eligible"])
        self.assertNotIn("recompute", result["revision"])

    def test_actual_bridge_missing_native_usage_is_fatal_not_synthetic_free_B(self):
        result, requests, source_reads = self.exercise("missing_usage")
        self.assertIsNone(result["agent_answer"])
        self.assertIsNotNone(result["error"])
        self.assertEqual(len(requests), 3)
        self.assertEqual(len(source_reads), 1)
        self.assertEqual(result["tokens"]["llm_calls"], 3)
        self.assertEqual(result["budget"]["tool_calls_run"], 2)
        self.assertTrue(result["revision"]["recompute"]["fatal"])
        self.assertEqual(result["revision"]["release_policy"], "fatal_execution_or_accounting_failure")
        self.assertTrue(result["revision"]["recompute"]["phase"]["fatal"])


if __name__ == "__main__":
    unittest.main()
