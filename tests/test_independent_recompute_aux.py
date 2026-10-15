"""Synthetic native-tool, accounting and frozen-candidate boundaries; no network."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from agent.base_agent import BaseAgent
from agent.single_agent import SingleAgent
from agent.independent_recompute_aux import (AuxPhase, LIMITS, TOKEN_CAP, TOOL_CAP,
    fingerprint, run_independent_recompute, same)

WORKSPACE = {"Observation": [{"resourceType": "Observation", "id": "o1",
    "subject": {"reference": "Patient/p1"}, "effectiveDateTime": "2130-01-01T01:00:00Z",
    "code": {"coding": [{"code": "weight", "display": "Weight"}]},
    "valueQuantity": {"value": 71, "unit": "kg"}}]}
CONTEXT = {"question": "What is the recorded weight?", "assumptions": "",
           "patient_fhir_id": "p1", "current_time": "2131-01-01T00:00:00Z"}
CITATION = {"resource_id": "Observation/o1", "field_path": "valueQuantity.value", "value": 71}
CODE = """r = retrieved_resources['Observation'][0]
answer = r['valueQuantity']['value']
evidence = [{'resource_id':'Observation/'+r['id'], 'field_path':'valueQuantity.value', 'value':answer}]
contributing_ids = ['Observation/'+r['id']]
operation = 'Select the recorded weight.'
limitation = None
"""


class Message:
    def __init__(self, content=None, calls=None):
        self.role, self.content, self.tool_calls = "assistant", content, calls

    def model_dump(self, **_):
        return {"role": "assistant", "content": self.content, "tool_calls": deepcopy(self.tool_calls)}


def native(code=CODE, **arguments):
    return Message(calls=[{"id": "call-independent", "type": "function", "function": {
        "name": "execute_python_code", "arguments": json.dumps({"code": code, "mode": "final", **arguments})}}])


def select(choice="B", **extra):
    return Message(json.dumps({"choice": choice, "reason": "The actual program matches the requested projection.", **extra}))


def usage(completion=80, prompt=100):
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
            "llm_calls": 1, "cost": 0.0, "per_call": [{"prompt": prompt, "completion": completion, "usage_available": True}]}


def candidate(answer=71, **updates):
    return {"answer": answer, "evidence": [deepcopy(CITATION)], "operation": "Select recorded weight.",
            "contributing_ids": ["Observation/o1"], "limitation": None, **updates}


def check(citations, **_):
    return {"issues": [], "n_valid": len(citations), "results": [
        {"citation": deepcopy(citation), "status": "valid", "warnings": []} for citation in citations]}


class Host(BaseAgent):
    def __init__(self, responses=()):
        # Do not invoke BaseAgent.__init__/setup_api_keys in offline fixtures.
        self.model, self.verbose, self.base_url = "test-model", False, None
        self._reset_usage()
        self.messages = []
        self.set_budget(max_total_tokens=200000, max_tool_calls=30)
        self.total_usage.update(total_tokens=1000, prompt_tokens=900, completion_tokens=100, llm_calls=1)
        self.call_usage = [{"prompt": 900, "completion": 100, "usage_available": True}]
        self.tool_calls_made = 5
        self.responses, self.requests = iter(responses), []

    def _llm_call(self, messages, tools=None):
        self.requests.append({"messages": deepcopy(messages), "tools": deepcopy(tools), "ceiling": self.max_total_tokens})
        item = next(self.responses)
        if isinstance(item, Exception):
            raise item
        response, error, raw = item if isinstance(item, tuple) else (item, None, usage())
        if raw is not None:
            self._update_usage(raw)
        return response, error, raw


class AuxiliaryTests(unittest.TestCase):
    def run_aux(self, responses, *, a=None, b=None, executor=None, source_checker=check, workspace=None,
                context=None, host=None, source=None, artifact_update=None):
        a = candidate(answer=70) if a is None else deepcopy(a)
        b = candidate() if b is None else deepcopy(b)
        workspace = deepcopy(WORKSPACE if workspace is None else workspace)
        context = deepcopy(CONTEXT if context is None else context)
        artifact = {**deepcopy(a), "code": "# ORIGINAL_A_CODE_SENTINEL\nanswer=70", "mode": "final",
                    "workspace_sha256": SingleAgent._workspace_sha256(workspace)}
        artifact.update(artifact_update or {})
        host = Host(responses) if host is None else host
        executions = []
        def execute(code, **kwargs):
            executions.append((code, deepcopy(kwargs)))
            return {**deepcopy(b), "mode": "final"}
        released, diagnostic = run_independent_recompute(host, a, artifact, workspace, context,
            check(a["evidence"]) if source is None else source, executor=executor or execute, source_checker=source_checker)
        return host, released, diagnostic, executions

    def test_disagreement_selects_only_existing_B_two_calls_two_tools(self):
        host, released, d, executions = self.run_aux([native(), select()])
        self.assertEqual(released, candidate())
        self.assertEqual((d["model_calls"], d["tool_calls"]), (2, 2))
        self.assertEqual(len(executions), 1)
        self.assertTrue(d["switched"])
        self.assertEqual(d["selected"], "B")
        self.assertFalse(d["fatal"])
        self.assertEqual(host.max_total_tokens, 200000)
        self.assertEqual(d["phase"]["token_cap"], TOKEN_CAP)
        self.assertEqual(d["phase"]["tool_cap"], TOOL_CAP)
        self.assertEqual(executions[0][1]["mode"], "final")
        self.assertEqual(executions[0][1]["global_vars"], {"retrieved_resources": WORKSPACE})
        self.assertIsNone(host.requests[1]["tools"])

    def test_generation_is_blind_selector_is_fresh_intact_and_native_schema_counted(self):
        host, _, d, _ = self.run_aux([native(), select()])
        first, second = host.requests
        serialized = json.dumps(first["messages"])
        self.assertNotIn("ORIGINAL_A_CODE_SENTINEL", serialized)
        self.assertNotIn("Select recorded weight.", serialized)
        packet = json.loads(first["messages"][1]["content"])
        self.assertEqual(set(packet), {"context", "workspace_sha256", "held_inventory", "retrieval_allowed"})
        self.assertEqual(packet["context"], CONTEXT)
        self.assertEqual([tool["function"]["name"] for tool in first["tools"]], ["execute_python_code"])
        from utils.core_utils import count_tokens_in_messages
        self.assertEqual(d["generation_prompt_tokens"], count_tokens_in_messages(first["messages"], first["tools"]))
        self.assertGreater(d["generation_prompt_tokens"], count_tokens_in_messages(first["messages"]))
        self.assertEqual(len(second["messages"]), 2)
        selector_packet = json.loads(second["messages"][1]["content"])
        self.assertIn("ORIGINAL_A_CODE_SENTINEL", selector_packet["A"]["program"])
        self.assertEqual(selector_packet["B"]["program"], CODE)
        self.assertEqual(selector_packet["B"]["artifact"], candidate())
        self.assertEqual(selector_packet["A"]["artifact"], candidate(answer=70))

    def test_exact_agreement_keeps_A_evidence_operation_and_skips_selector(self):
        a = candidate(operation="Original operation preserved")
        host, released, d, _ = self.run_aux([native()], a=a)
        self.assertTrue(same(released, a))
        self.assertEqual(d["status"], "agreement_preserved")
        self.assertEqual((d["model_calls"], d["tool_calls"]), (1, 2))
        self.assertEqual(len(host.requests), 1)

    def test_bool_number_string_list_and_precision_are_not_scorer_agreement(self):
        for left, right in [(False, 0), (True, 1), (1, 1.0), (71, "71"), ([71], 71), ([1, 2], [2, 1])]:
            self.assertFalse(same(left, right))
            _, _, d, _ = self.run_aux([native(), select("A")], a=candidate(left), b=candidate(right))
            self.assertFalse(d["answer_agreement"])
            self.assertEqual(d["model_calls"], 2)

    def test_unknown_A_selection_invalid_schema_or_tool_request_preserve_A(self):
        for response in [select("A"), select("UNKNOWN"), select(answer=999), Message('{"choice":"B","choice":"A","reason":"x"}'), native()]:
            with self.subTest(response=response.content):
                _, released, d, _ = self.run_aux([native(), response])
                self.assertEqual(released, candidate(answer=70))
                self.assertFalse(d["switched"])
                self.assertFalse(d["fatal"])

    def test_program_generation_must_be_one_native_final_call(self):
        bad = [Message(CODE), native(mode="inspect"), native(global_vars={}), native(timeout=True)]
        two = native(); two.tool_calls *= 2; bad.append(two)
        wrong = native(); wrong.tool_calls[0]["function"]["name"] = "get_resources_by_patient_fhir_id"; bad.append(wrong)
        for response in bad:
            with self.subTest(response=response.content):
                _, released, d, executions = self.run_aux([response])
                self.assertEqual(released, candidate(answer=70))
                self.assertEqual(executions, [])
                self.assertEqual(d["tool_calls"], 0)

    def test_original_source_check_once_with_original_context_and_exact_B_citations(self):
        observed = []
        def source(citations, **kwargs):
            observed.append((deepcopy(citations), deepcopy(kwargs)))
            return check(citations)
        _, _, d, _ = self.run_aux([native(), select()], source_checker=source)
        self.assertEqual(observed, [(candidate()["evidence"], {"question": CONTEXT["question"],
            "current_time": CONTEXT["current_time"], "patient_fhir_id": CONTEXT["patient_fhir_id"]})])
        self.assertTrue(d["B_eligible"])

    def test_source_error_warnings_bad_status_or_mismatched_citation_preserve_A(self):
        variants = []
        value = check([CITATION]); value["error"] = "bad checker"; variants.append(value)
        value = check([CITATION]); value["issues"] = ["patient: wrong"]; variants.append(value)
        value = check([CITATION]); value["results"][0]["status"] = "value_mismatch"; variants.append(value)
        value = check([CITATION]); value["results"][0]["warnings"] = ["current_time: future"]; variants.append(value)
        value = check([CITATION]); value["results"][0]["citation"]["value"] = True; variants.append(value)
        value = check([CITATION]); value["n_valid"] = True; variants.append(value)
        for value in variants:
            _, released, d, _ = self.run_aux([native()], source_checker=lambda *a, **k: deepcopy(value))
            self.assertEqual(released, candidate(answer=70))
            self.assertEqual(d["tool_calls"], 2)
            self.assertEqual(d["model_calls"], 1)
            self.assertFalse(d["fatal"])

    def test_empty_and_null_candidates_are_not_preferred_or_rejected(self):
        for answer in [None, False, 0, []]:
            b = candidate(answer, evidence=[], contributing_ids=[])
            _, released, d, _ = self.run_aux([native(), select()], b=b)
            self.assertTrue(same(released, b))
            self.assertTrue(d["switched"])
        _, released, d, _ = self.run_aux([native()], b=candidate(5, evidence=[], contributing_ids=[]))
        self.assertEqual(released["answer"], 70)
        self.assertFalse(d["switched"])

    def test_b_execution_schema_limitation_and_nonfinite_fail_without_selection(self):
        for b in [candidate(limitation="missing records"), candidate(contributing_ids=[]), candidate(float("nan"))]:
            _, released, d, _ = self.run_aux([native()], b=b)
            self.assertEqual(released["answer"], 70)
            self.assertEqual(d["model_calls"], 1)
            self.assertFalse(d["switched"])

    def test_fresh_worker_copy_mutation_does_not_change_A_workspace(self):
        def execute(code, **kwargs):
            kwargs["global_vars"]["retrieved_resources"]["Observation"][0]["valueQuantity"]["value"] = 999
            return {**candidate(), "mode": "final"}
        _, _, d, _ = self.run_aux([native(), select()], executor=execute)
        self.assertEqual(WORKSPACE["Observation"][0]["valueQuantity"]["value"], 71)
        self.assertTrue(d["switched"])

    def test_primary_binding_failures_skip_before_call_and_do_not_mutate_A(self):
        for update in [{"workspace_sha256": "wrong"}, {"answer": 999}, {"mode": "inspect"}]:
            host, released, d, _ = self.run_aux([], artifact_update=update)
            self.assertEqual(released["answer"], 70)
            self.assertEqual(host.requests, [])
            self.assertFalse(d["packet_admitted"])
        source = check([CITATION]); source["results"][0]["warnings"] = ["bad"]
        host, _, _, _ = self.run_aux([], source=source)
        self.assertEqual(host.requests, [])

    def test_no_essential_truncation_or_call_when_two_packet_reservation_cannot_fit(self):
        host = Host([]); host.max_total_tokens = host.total_usage["total_tokens"] + 100
        host, _, d, _ = self.run_aux([], host=host)
        self.assertEqual(d["status"], "coverage_skip")
        self.assertEqual(host.requests, [])
        self.assertFalse(d["packet_admitted"])
        host, _, d, _ = self.run_aux([], artifact_update={"code": "x" * 100000})
        self.assertEqual(d["status"], "coverage_skip")
        self.assertEqual(host.requests, [])

    def test_B_bounds_and_tool_allowance_preserve_A(self):
        _, _, d, executions = self.run_aux([native("x" * (LIMITS["program_chars"] + 1))])
        self.assertEqual(executions, [])
        self.assertFalse(d["switched"])
        b = candidate(operation="x" * LIMITS["candidate_entry_bytes"])
        _, _, d, _ = self.run_aux([native()], b=b)
        self.assertFalse(d["selector_admitted"])
        self.assertFalse(d["switched"])
        self.assertEqual(d["tool_calls"], 1)
        host = Host([]); host.max_tool_calls = host.tool_calls_made + 1
        host, _, d, _ = self.run_aux([], host=host)
        self.assertEqual(d["status"], "coverage_skip")
        self.assertEqual(host.requests, [])

    def test_admitted_exception_or_unknown_usage_is_fatal_not_free_fallback(self):
        for item in [RuntimeError("synthetic"), (None, "provider unavailable", None),
                     (None, "provider unavailable", {**usage(), "per_call": []})]:
            host, released, d, _ = self.run_aux([item])
            self.assertTrue(d["fatal"])
            self.assertEqual(d["status"], "accounting_failed")
            self.assertEqual(released["answer"], 70)  # Caller must withhold it.
            self.assertFalse(host.call_usage[-1]["usage_available"])
            json.dumps(host.call_usage, allow_nan=False)

    def test_verified_error_preserves_A_nonfatally(self):
        _, released, d, _ = self.run_aux([(None, "provider error", usage())])
        self.assertEqual(released["answer"], 70)
        self.assertFalse(d["fatal"])
        self.assertEqual(d["model_calls"], 1)

    def test_completion_subcaps_are_fatal_for_first_and_second_call(self):
        for replies in [[(native(), None, usage(completion=4001))],
                        [native(), (select(), None, usage(completion=1801))]]:
            _, _, d, _ = self.run_aux(replies)
            self.assertTrue(d["fatal"])
            self.assertEqual(d["status"], "accounting_failed")

    def test_malformed_cached_cost_and_aggregate_are_fatal_and_serializable(self):
        variants = []
        for key, value in [("cost", float("nan")), ("total_tokens", 999), ("cached_prompt_tokens", True)]:
            raw = usage(); raw[key] = value; variants.append(raw)
        raw = usage(); raw["per_call"][0]["cached"] = object(); variants.append(raw)
        for raw in variants:
            host, _, d, _ = self.run_aux([(native(), None, raw)])
            self.assertTrue(d["fatal"])
            json.dumps({"usage": host.total_usage, "calls": host.call_usage, "diagnostics": d}, allow_nan=False)

    def test_phase_and_global_overshoot_fatal_and_caps_restore(self):
        host = Host([(native(), None, usage(prompt=26000))])
        host, _, d, _ = self.run_aux([], host=host)
        self.assertTrue(d["fatal"])
        self.assertEqual(host.max_total_tokens, 200000)

    def test_cumulative_cost_overflow_is_fatal_and_quarantined(self):
        raw = usage(); raw["cost"] = 1e308
        host = Host([(native(), None, raw)]); host.total_usage["cost"] = 1e308
        host, _, d, _ = self.run_aux([], host=host)
        self.assertTrue(d["fatal"])
        self.assertIsNone(host.total_usage["cost"])
        json.dumps({"usage": host.total_usage, "diagnostic": d}, allow_nan=False)


class RealWorkerTests(unittest.TestCase):
    run_aux = AuxiliaryTests.run_aux

    def test_real_worker_and_original_field_check(self):
        from tools.python_executor import execute_python_code
        from tools.field_check import check_fields
        def fetch(refs):
            return {"o1": deepcopy(WORKSPACE["Observation"][0])} if ("Observation", "o1") in refs else {}
        with patch("tools.field_check.fetch_from_store", side_effect=fetch):
            _, released, d, _ = self.run_aux([native(), select()], executor=execute_python_code, source_checker=check_fields)
        self.assertTrue(d["switched"], d)
        self.assertEqual(released, candidate(operation="Select the recorded weight."))
        self.assertEqual(d["tool_calls"], 2)

    def test_native_default_sdk_json_encoding_exact_subcaps_and_gzip_restore(self):
        import tempfile
        import litellm
        import httpx2
        from openai import AzureOpenAI
        from scripts.audit_store import externalize_result_audit, restore_result_audit
        from utils import core_utils

        pending = iter([native(), select()])
        requests = []
        def send(request, **_):
            requests.append(json.loads(request.content))
            message = next(pending).model_dump()
            return httpx2.Response(200, request=request, json={"id": "offline-v6",
                "object": "chat.completion", "created": 0, "model": "gpt-6-luna",
                "choices": [{"index": 0, "message": message,
                             "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 80, "total_tokens": 180}})
        client = AzureOpenAI(azure_endpoint="https://offline-fixture.invalid",
            api_version="2024-12-01-preview", api_key="non-secret-unit-test-placeholder",
            timeout=60.0, max_retries=0)
        self.addCleanup(client.close)
        self.assertIsInstance(client._client, httpx2.Client)
        self.assertIsInstance(client._client._transport, httpx2.HTTPTransport)
        def sdk(**kwargs):
            parameters = {"model": "offline-deployment", "messages": kwargs["messages"],
                "max_completion_tokens": min(kwargs.get("max_completion_tokens", 16384), 16384),
                "reasoning_effort": "none"}
            if kwargs.get("tools"):
                parameters["tools"] = kwargs["tools"]
            return litellm.ModelResponse(**client.chat.completions.create(**parameters).model_dump(exclude_none=True))
        class SDKHost(Host):
            _llm_call = BaseAgent._llm_call
        host = SDKHost(); host.model = "gpt-6-luna"
        with patch.object(client._client, "send", side_effect=send), \
                patch.object(core_utils.litellm, "completion", side_effect=sdk):
            host, released, diagnostic, _ = self.run_aux([], host=host)
        self.assertTrue(diagnostic["switched"], diagnostic)
        self.assertEqual([request["max_completion_tokens"] for request in requests], [4000, 1800])
        self.assertEqual([len(request["messages"]) for request in requests], [2, 2])
        self.assertTrue(requests[0]["tools"])
        self.assertNotIn("tools", requests[1])
        raw = {**released, "latency": 1.0, "trace": host.messages,
               "revision": {"independent_recompute": diagnostic}}
        with tempfile.TemporaryDirectory() as directory:
            compact = externalize_result_audit(raw, directory, question_id="synthetic-v6")
            restored = restore_result_audit(compact, directory, question_id="synthetic-v6")
            self.assertTrue(same(restored, raw))
        self.assertEqual(diagnostic["model_calls"], 2)
        self.assertEqual(host.total_usage["llm_calls"], 3)
        self.assertEqual(diagnostic["tool_calls"], 2)


if __name__ == "__main__":
    unittest.main()
