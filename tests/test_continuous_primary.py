"""Actual SingleAgent loop with synthetic tools/provider; no configuration or network."""
from contextlib import ExitStack
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent.base_agent import BaseAgent
from agent.continuous_primary import run_continuous_primary, PRIMARY_ITERATIONS
from agent.single_agent import SingleAgent


PATIENT = "11111111-1111-1111-1111-111111111111"
RESOURCE = {"resourceType": "Observation", "id": "o1", "subject": {"reference": "Patient/" + PATIENT},
            "valueQuantity": {"value": 7}}
PACKET = {"text": "Synthetic observation", "resources": {"Observation": [RESOURCE]}}
QUESTION = f"Question: What is the synthetic value?\nContext:\nPatient FHIR ID is {PATIENT}."


def usage(prompt=10, completion=2, known=True, cost=.01):
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
            "cost": cost, "llm_calls": 1,
            "per_call": [{"prompt": prompt, "completion": completion, "cached": 0, "usage_available": known}]}


def computation(answer=7, **changes):
    return {"answer": answer, "evidence": [{"resource_id": "Observation/o1", "field_path": "valueQuantity.value", "value": answer}],
            "operation": "Read the synthetic value", "contributing_ids": ["Observation/o1"], "limitation": None,
            "mode": "final", **changes}


def tool(identifier, name="execute_python_code", **arguments):
    return SimpleNamespace(id=identifier, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def reply(content=None, *calls):
    return SimpleNamespace(role="assistant", content=content, tool_calls=list(calls) or None)


def final(identifier="final", value=7):
    return reply(None, tool(identifier, code=f"answer={value}", mode="final"))


def prose(value=7):
    value = computation(value)
    return reply(json.dumps({key: value[key] for key in ("answer", "evidence", "limitation")}))


def source_check(citations, **_):
    results = [{"citation": deepcopy(citation), "status": "valid" if citation.get("value") == 7 else "value_mismatch", "warnings": []}
               for citation in citations]
    return {"results": results, "issues": ["value_mismatch" for result in results if result["status"] != "valid"],
            "n_valid": sum(result["status"] == "valid" for result in results)}


class ContinuousPrimaryTests(unittest.TestCase):
    def environment(self, responses, executions=(), usages=None, check=source_check):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch("agent.base_agent.setup_api_keys"))
        stack.enter_context(patch("agent.single_agent.get_starting_packet", return_value=deepcopy(PACKET)))
        stack.enter_context(patch("agent.single_agent.check_fields", side_effect=check))
        stack.enter_context(patch("agent.base_agent.count_tokens_in_messages", return_value=10))
        execute = Mock(side_effect=deepcopy(list(executions)))
        fetch = Mock(return_value={"Observation": [deepcopy(RESOURCE)]})
        stack.enter_context(patch("agent.multi_turn_code_resource_agent.get_tool", side_effect=lambda name: execute if name == "execute_python_code" else fetch))
        requests = []
        values = iter(responses)
        budgets = iter(usages) if usages is not None else None

        def provider(**kwargs):
            requests.append(deepcopy(kwargs))
            return next(values), None, deepcopy(next(budgets) if budgets is not None else usage())

        stack.enter_context(patch("agent.base_agent.safe_llm_call", side_effect=provider))
        host = BaseAgent("synthetic")
        host.set_budget(max_total_tokens=200_000, max_tool_calls=30)
        return host, requests, execute, fetch

    def test_natural_post_final_inspect_and_second_final_are_not_short_circuited(self):
        responses = [final(), reply(None, tool("inspect", code="answer={'count':1}", mode="inspect")),
                     final("second"), reply("Malformed final prose is harmless")]
        host, requests, _, _ = self.environment(responses, [computation(), {"mode": "inspect", "answer": {"count": 1}}, computation(operation="Second natural computation")])
        result = run_continuous_primary(host, QUESTION)
        self.assertEqual(len(requests), 4)
        self.assertEqual(result.released["answer"], 7)
        self.assertTrue(result.diagnostics["eligible"])
        self.assertEqual(result.artifact["operation"], "Second natural computation")
        self.assertEqual(result.diagnostics["binding"]["execution_tool_call_id"], "second")
        self.assertEqual(result.diagnostics["binding"]["source_check_tool_call_id"], "automatic_field_check_1")
        self.assertEqual(result.source_check, source_check(computation()["evidence"]))
        for request in requests:
            text = json.dumps(request["messages"], default=lambda value: vars(value))
            self.assertNotIn("Solver allocation remaining", text)
            self.assertNotIn("HostBoundary", text)
        self.assertEqual(result.diagnostics["max_iterations"], PRIMARY_ITERATIONS)

    def test_request_trajectory_matches_independent_actual_single_same_budget(self):
        responses = [final(), reply(None, tool("inspect", code="answer=1", mode="inspect")), prose()]
        outputs = [computation(), {"mode": "inspect", "answer": 1}]
        host, requests, _, _ = self.environment(responses, outputs)
        result = run_continuous_primary(host, QUESTION)
        observed = deepcopy(requests)
        _, control_requests, _, _ = self.environment(responses, outputs)
        control = SingleAgent("synthetic", max_iterations=30)
        control.set_budget(max_total_tokens=175_000, max_tool_calls=27)
        expected = control.run(QUESTION)
        self.assertEqual(observed, control_requests)
        for key in ("answer", "evidence", "limitation", "tokens", "tool_calls", "trace", "revision", "budget", "error"):
            self.assertEqual(result.released[key], expected[key])

    def test_source_revision_finishes_naturally_and_binds_second_source_check(self):
        host, requests, _, _ = self.environment([final(value=99), prose(99), final("corrected"), prose()], [computation(99), computation()])
        result = run_continuous_primary(host, QUESTION)
        self.assertEqual(len(requests), 4)
        self.assertEqual(result.released["revision"]["draft"]["answer"], 99)
        self.assertEqual(result.released["answer"], 7)
        self.assertTrue(result.released["revision"]["revised"])
        self.assertEqual(result.diagnostics["binding"]["source_check_tool_call_id"], "automatic_field_check_2")
        self.assertEqual(host.tool_calls_made, 4)

    def test_later_retrieval_requires_new_actual_computation(self):
        host, requests, _, fetch = self.environment([final(), reply(None, tool("fetch", "get_resources_by_patient_fhir_id", resource_type="Observation", patient_fhir_id=PATIENT)),
            final("new_final"), prose()], [computation(), computation()])
        result = run_continuous_primary(host, QUESTION)
        self.assertEqual(len(requests), 4)
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(result.diagnostics["binding"]["execution_tool_call_id"], "new_final")

    def test_prose_only_primary_is_preserved_but_not_invented_into_program(self):
        host, _, _, _ = self.environment([prose()])
        result = run_continuous_primary(host, QUESTION)
        self.assertEqual(result.released["answer"], 7)
        self.assertIsNone(result.artifact)
        self.assertIsNone(result.source_check)
        self.assertFalse(result.diagnostics["fatal"])
        self.assertEqual(result.released["revision"]["final_check_issues"], [])

    def test_failed_source_primary_is_preserved_not_promoted_from_executed_candidate(self):
        host, _, _, _ = self.environment([final(value=99), prose(99), final("again", 99), prose(99)], [computation(99), computation(99)])
        result = run_continuous_primary(host, QUESTION)
        self.assertIsNone(result.released["answer"])
        self.assertEqual(result.released["revision"]["final"]["answer"], 99)
        self.assertIn("Evidence could not be verified", result.released["limitation"])
        self.assertIsNone(result.artifact)
        self.assertFalse(result.diagnostics["fatal"])

    def test_inclusive_cap_refusals_preserve_local_stop_without_global_stop(self):
        host, requests, _, fetch = self.environment([final(), reply(None, tool("refused", "get_resources_by_patient_fhir_id", resource_type="Observation", patient_fhir_id=PATIENT)), prose()], [computation()])
        host.set_budget(max_total_tokens=100_000, max_tool_calls=1)
        result = run_continuous_primary(host, QUESTION)
        self.assertEqual(fetch.call_count, 0)
        self.assertEqual(len(requests), 3)
        self.assertEqual(result.diagnostics["token_cap"], 100_000)
        self.assertEqual(result.diagnostics["tool_cap"], 1)
        self.assertEqual(host.tool_calls_made, 1)
        self.assertTrue(result.diagnostics["primary_budget_stop"])
        self.assertIsNone(host.budget_stop)
        self.assertIsNone(result.artifact)
        self.assertFalse(result.diagnostics["fatal"])

    def test_zero_token_global_cap_makes_no_sdk_attempt_and_remains_local(self):
        host, requests, _, _ = self.environment([])
        host.set_budget(max_total_tokens=0, max_tool_calls=0)
        result = run_continuous_primary(host, QUESTION)
        self.assertEqual(requests, [])
        self.assertEqual(result.diagnostics["token_cap"], 0)
        self.assertEqual(host.total_usage["llm_calls"], 0)
        self.assertIsNone(host.budget_stop)
        self.assertFalse(result.diagnostics["fatal"])

    def test_raw_usage_is_adopted_exactly_once_and_unknown_usage_is_fatal(self):
        reported = usage(0, 0, known=False, cost=None)
        host, requests, _, _ = self.environment([prose()], usages=[reported])
        result = run_continuous_primary(host, QUESTION)
        self.assertEqual(host.call_usage, reported["per_call"])
        self.assertEqual(host.total_usage, {key: reported[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost", "llm_calls")})
        self.assertIsNone(host.total_usage["cost"])
        self.assertTrue(result.diagnostics["fatal"])
        self.assertFalse(result.diagnostics["usage_known"])
        self.assertIsNone(result.artifact)
        with self.assertRaisesRegex(ValueError, "clean"):
            run_continuous_primary(host, QUESTION)
        self.assertEqual(len(requests), 1)

    def test_known_usage_with_unknown_price_is_not_mistaken_for_unknown_tokens(self):
        host, _, _, _ = self.environment([final(), prose()], [computation()], [usage(cost=None), usage(cost=None)])
        result = run_continuous_primary(host, QUESTION)
        self.assertIsNone(host.total_usage["cost"])
        self.assertTrue(result.diagnostics["usage_known"])
        self.assertFalse(result.diagnostics["fatal"])
        self.assertIsNotNone(result.artifact)

    def test_reported_token_overshoot_is_whole_fatal(self):
        host, _, _, _ = self.environment([prose()], usages=[usage(110, 10)])
        host.set_budget(max_total_tokens=100, max_tool_calls=30)
        result = run_continuous_primary(host, QUESTION)
        self.assertEqual(host.total_usage["total_tokens"], 120)
        self.assertTrue(result.diagnostics["fatal"])
        self.assertIn("primary reported token ceiling overshoot", result.diagnostics["errors"])
        self.assertIsNone(host.budget_stop)

    def test_malformed_finite_accounting_is_quarantined_and_tagged(self):
        malformed = usage()
        malformed["per_call"][0]["prompt"] = "bad"
        host, _, _, _ = self.environment([prose()], usages=[malformed])
        result = run_continuous_primary(host, QUESTION)
        self.assertTrue(result.diagnostics["fatal"])
        self.assertEqual(result.diagnostics["raw_accounting"]["call_usage"][0]["prompt"], "bad")
        self.assertFalse(host.call_usage[0]["usage_available"])
        self.assertIsNone(host.total_usage["cost"])
        json.dumps(result.diagnostics, allow_nan=False)

    def test_actual_common_formatter_exception_retains_trace_and_unknown_accounting(self):
        malformed = usage()
        malformed["per_call"][0]["cached"] = "bad"
        host, _, _, _ = self.environment([prose()], usages=[malformed])
        result = run_continuous_primary(host, QUESTION)
        self.assertTrue(result.diagnostics["fatal"])
        self.assertFalse(result.diagnostics["released_available"])
        self.assertTrue(result.released["trace"])
        self.assertFalse(host.call_usage[0]["usage_available"])
        self.assertIsNone(result.artifact)
        json.dumps(result.diagnostics, allow_nan=False)

    def test_nonfinite_cost_is_tagged_without_crashing_later_json(self):
        host, _, _, _ = self.environment([prose()], usages=[usage(cost=float("nan"))])
        result = run_continuous_primary(host, QUESTION)
        self.assertTrue(result.diagnostics["fatal"])
        self.assertEqual(result.diagnostics["raw_accounting"]["total_usage"]["cost"], {"nonfinite_float": "nan"})
        self.assertIsNone(host.total_usage["cost"])
        json.dumps(result.diagnostics, allow_nan=False)

    def test_defensive_snapshots_do_not_allow_auxiliary_to_mutate_saved_primary(self):
        host, _, _, _ = self.environment([final(), prose()], [computation()])
        result = run_continuous_primary(host, QUESTION)
        result.workspace["Observation"][0]["valueQuantity"]["value"] = 99
        result.artifact["answer"] = 99
        result.source_check["issues"].append("bad")
        result.released["answer"] = 99
        self.assertEqual(result.workspace["Observation"][0]["valueQuantity"]["value"], 7)
        self.assertEqual(result.artifact["answer"], 7)
        self.assertEqual(result.source_check["issues"], [])
        self.assertEqual(result.released["answer"], 7)

    def test_exact_bool_vs_number_source_binding_mismatch_skips_without_changing_A(self):
        def wrong_check(citations, **kwargs):
            altered = deepcopy(citations[0])
            altered["value"] = 0  # Python equality would incorrectly match False.
            return {"issues": [], "n_valid": 1,
                    "results": [{"citation": altered, "status": "valid", "warnings": []}]}
        artifact = computation(False)
        artifact["evidence"][0]["value"] = False
        host, _, _, _ = self.environment([final(value=False), reply("Finished")], [artifact], check=wrong_check)
        result = run_continuous_primary(host, QUESTION)
        self.assertIs(result.released["answer"], False)
        self.assertIsNone(result.artifact)
        self.assertFalse(result.diagnostics["fatal"])

    def test_new_parent_run_reset_allows_a_new_exactly_once_adoption(self):
        host, _, _, _ = self.environment([prose(), prose()])
        first = run_continuous_primary(host, QUESTION)
        host._reset_usage()
        host.messages = []
        second = run_continuous_primary(host, QUESTION)
        self.assertEqual(first.released["answer"], second.released["answer"])
        self.assertEqual(host.total_usage["llm_calls"], 1)

    def test_actual_executed_empty_results_remain_eligible_without_invented_evidence(self):
        for answer in (None, False, 0, []):
            with self.subTest(answer=answer):
                artifact = computation(answer, evidence=[], contributing_ids=[], operation="Completed synthetic search; no matches")
                host, _, _, _ = self.environment([reply(None, tool("empty", code=f"answer={answer!r}", mode="final")), reply("Finished")], [artifact])
                result = run_continuous_primary(host, QUESTION)
                self.assertIsNotNone(result.artifact)
                self.assertEqual(json.dumps(result.released["answer"]), json.dumps(answer))
                self.assertEqual(result.source_check, {"results": [], "issues": [], "n_valid": 0})

    def test_inconsistent_numeric_ledger_is_fatal_and_preserves_raw_known_counters(self):
        inconsistent = usage()
        inconsistent["per_call"][0]["prompt"] = 9
        host, _, _, _ = self.environment([prose()], usages=[inconsistent])
        result = run_continuous_primary(host, QUESTION)
        self.assertTrue(result.diagnostics["fatal"])
        self.assertEqual(result.diagnostics["raw_accounting"]["total_usage"]["prompt_tokens"], 10)
        self.assertEqual(result.diagnostics["raw_accounting"]["call_usage"][0]["prompt"], 9)
        self.assertEqual(host.total_usage["prompt_tokens"], 10)
        self.assertFalse(host.call_usage[0]["usage_available"])
        self.assertIsNone(host.total_usage["cost"])

    def test_unknown_usage_cannot_claim_a_known_zero_cost(self):
        host, _, _, _ = self.environment([prose()], usages=[usage(0, 0, known=False, cost=0)])
        result = run_continuous_primary(host, QUESTION)
        self.assertTrue(result.diagnostics["fatal"])
        self.assertIsNone(host.total_usage["cost"])
        self.assertEqual(result.diagnostics["raw_accounting"]["total_usage"]["cost"], 0)

    def test_reported_model_call_overshoot_is_fatal_without_claiming_journal_send_counts(self):
        reported = usage(0, 0, cost=None)
        reported["llm_calls"] = 31
        reported["per_call"] *= 31
        host, requests, _, _ = self.environment([prose()], usages=[reported])
        result = run_continuous_primary(host, QUESTION)
        self.assertEqual(len(requests), 1)  # Synthetic reporting fault, not31 observed SDK sends.
        self.assertTrue(result.diagnostics["fatal"])
        self.assertIn("primary reported model-call ceiling overshoot", result.diagnostics["errors"])
        self.assertEqual(host.total_usage["llm_calls"], 31)


if __name__ == "__main__":
    unittest.main()
