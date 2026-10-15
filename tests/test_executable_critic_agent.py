"""Generic V5 interface/accounting tests: mocked model/store, real sandbox."""
from copy import deepcopy
import json
from unittest.mock import patch
import unittest

from agent.executable_critic_agent import (ExecutableCriticAgent, V5PhaseHost,
    PHASE_CAPS, LIMITS, parse_first, parse_patch, validate_anchor, validate_fact,
    validate_probe, _same)
from agent.controller import FixedController
from tests import test_query_contract_agent as fixture
from tests.test_query_contract_agent import BARE, QUESTION, CODE, FixtureMessage, fetch, final_call, reply

CONTEXT = FixedController.question_context(QUESTION)
WORKSPACE = {"Observation": deepcopy(fixture.STORE)}
FACT = {"resource_id": "Observation/o1", "field_path": "valueQuantity.value", "value": 70}
PROBE = {"facts": [FACT], "observations": [{"name": "boundary_included", "value": True}]}
INSPECT_CODE = """r = retrieved_resources['Observation'][0]
answer = {'facts': [{'resource_id':'Observation/'+r['id'], 'field_path':'valueQuantity.value', 'value':r['valueQuantity']['value']}],
          'observations':[{'name':'boundary_included','value':r['valueQuantity']['value'] >= 70}]}
"""


def first(code=INSPECT_CODE):
    return {"decision": "INSPECT", "anchor": {"channel": "question", "quote": "greater than 70"},
            "claim": "The executed filter includes an explicitly excluded boundary.", "inspect_code": code}


def proposal(old="r['valueQuantity']['value'] >= 70", new="r['valueQuantity']['value'] > 70"):
    return {"decision": "PATCH", "anchor": first()["anchor"],
            "observed_discrepancy": "The boundary is present in the current count.",
            "expected_obligation": "Use the original question's strict relation.",
            "preserved_obligations": ["Keep population, source values and count aggregation."],
            "witnesses": [{"fact": deepcopy(FACT), "observation_name": "boundary_included",
                           "observed_value": True, "relevance": "The boundary inclusion differs."}],
            "patch": {"old_text": old, "new_text": new}}


class GenericSchemaTests(unittest.TestCase):
    def parse(self, value):
        return parse_patch(json.dumps(value), CONTEXT, first(), PROBE, CODE, WORKSPACE)

    def test_unique_anchor_derives_span_without_indices(self):
        result = validate_anchor(first()["anchor"], CONTEXT)
        self.assertEqual(result["span"], [BARE.index("greater than 70"), BARE.index("greater than 70")+15])
        self.assertNotIn("span", first()["anchor"])

    def test_missing_ambiguous_overlapping_and_supplied_span_anchors_rejected(self):
        for text, quote in [(BARE, "absent"), ("aaa", "aa"), ("x x", "x")]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_anchor({"channel": "question", "quote": quote}, {**CONTEXT, "question": text})
        with self.assertRaises(ValueError):
            validate_anchor({**first()["anchor"], "span": [0, 1]}, CONTEXT)

    def test_duplicate_nonfinite_and_unexpected_json_rejected(self):
        for value in ['{"decision":"KEEP","decision":"UNKNOWN","reason":"x"}',
                      '{"decision":"KEEP","reason":NaN}',
                      '{"decision":"KEEP","reason":"x","answer":5}']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_first(value, CONTEXT)

    def test_quote_switch_to_other_valid_clause_rejected(self):
        value = proposal()
        value["anchor"]["quote"] = "weight measurements"
        with self.assertRaisesRegex(ValueError, "switched"):
            self.parse(value)

    def test_source_boolean_is_not_integer_and_missing_is_not_null(self):
        self.assertFalse(_same(True, 1))
        value = deepcopy(FACT); value["value"] = True
        with self.assertRaises(ValueError):
            validate_fact(value, WORKSPACE)
        for path in ["missing", "valueQuantity.absent"]:
            with self.assertRaises(ValueError):
                validate_fact({**FACT, "field_path": path, "value": None}, WORKSPACE)
        records = deepcopy(WORKSPACE); records["Observation"][0]["nullable"] = None
        validate_fact({**FACT, "field_path": "nullable", "value": None}, records)

    def test_unindexed_array_path_is_not_concrete(self):
        records = deepcopy(WORKSPACE); records["Observation"][0]["items"] = [{"v": 7}]
        with self.assertRaises(ValueError):
            validate_fact({**FACT, "field_path": "items.v", "value": 7}, records)
        validate_fact({**FACT, "field_path": "items[0].v", "value": 7}, records)

    def test_fabricated_original_and_unobserved_witnesses_rejected(self):
        for field, replacement in [("value", 999), ("resource_id", "Observation/absent")]:
            value = proposal(); value["witnesses"][0]["fact"][field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError): self.parse(value)
        value = proposal(); value["witnesses"][0]["fact"] = {**FACT, "resource_id": "Observation/o2", "value": 71}
        with self.assertRaisesRegex(ValueError, "not observed"): self.parse(value)

    def test_observation_fabrication_and_boolean_integer_coercion_rejected(self):
        for replacement in [False, 1, "True"]:
            value = proposal(); value["witnesses"][0]["observed_value"] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(ValueError): self.parse(value)

    def test_repeated_and_overlapping_patch_regions_rejected(self):
        value = proposal(old="r[", new="changed")
        with self.assertRaises(ValueError): self.parse(value)
        value = proposal(old="aa", new="b")
        with self.assertRaises(ValueError):
            parse_patch(json.dumps(value), CONTEXT, first(), PROBE, "aaa", WORKSPACE)

    def test_arbitrary_generic_region_not_fixed_predicate_repair_grammar(self):
        value = proposal(old="answer = len(rows)", new="answer = sum(1 for row in rows)")
        parsed, code = self.parse(value)
        self.assertIn("sum(1 for row in rows)", code)
        self.assertEqual(parsed["decision"], "PATCH")

    def test_probe_rejects_fabricated_fact_even_with_successful_execution(self):
        value = deepcopy(PROBE); value["facts"][0]["value"] = 999
        with self.assertRaises(ValueError): validate_probe({"answer": value}, WORKSPACE)

    def test_limits_and_phase_caps_are_explicit(self):
        self.assertEqual(PHASE_CAPS, {"solver": (175000, 27), "critic": (25000, 3)})
        self.assertEqual(LIMITS["second_completion"], 4000)


class CriticPipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.QueryContractIntegrationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.requests = []

    def run_agent(self, responses, tokens=200000, tools=30):
        pending = iter(responses)
        def llm(**kwargs):
            self.requests.append(deepcopy(kwargs))
            item = next(pending)
            if isinstance(item, Exception): raise item
            if isinstance(item, tuple): return item
            usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
                     "llm_calls": 1, "cost": 0,
                     "per_call": [{"prompt": 100, "completion": 50, "usage_available": True}]}
            return FixtureMessage(**vars(item)), None, usage
        with patch("agent.base_agent.safe_llm_call", side_effect=llm):
            host = ExecutableCriticAgent("test-model")
            host.set_budget(max_total_tokens=tokens, max_tool_calls=tools)
            result = host.run(QUESTION)
        return host, result

    def responses(self, first_value=None, second_value=None, code=CODE):
        return [fetch(), final_call(code), reply(json.dumps(first_value or first())),
                reply(json.dumps(second_value or proposal()))]

    def test_full_real_sandbox_candidate_has_two_critic_calls_three_tools(self):
        host, result = self.run_agent(self.responses())
        self.assertIsNone(result["error"], result)
        self.assertEqual(result["answer"], 1)
        self.assertEqual(result["revision"]["primary"]["answer"], 2)
        self.assertTrue(result["revision"]["critic_revised"])
        self.assertEqual(host.tool_calls_made, 6)
        self.assertEqual(host.total_usage["llm_calls"], 4)
        diagnostic = result["revision"]["critic"]
        self.assertEqual((diagnostic["model_calls"], diagnostic["tool_calls"]), (2, 3))
        self.assertEqual(diagnostic["assurance"], "fallible_model_semantics_not_global_proof")
        self.assertEqual(result["revision"]["release_policy"], "executed_source_checked_critic_proposal")
        self.assertLessEqual(self.requests[2]["max_completion_tokens"], 1800)
        self.assertLessEqual(self.requests[3]["max_completion_tokens"], 4000)
        self.assertIsNone(self.requests[2]["tools"])
        self.assertIsNone(self.requests[3]["tools"])

    def test_packet_preflight_exception_preserves_valid_primary(self):
        with patch("agent.executable_critic_agent.critic_packet", side_effect=TypeError("metadata failure")):
            host, result = self.run_agent([fetch(), final_call()])
        self.assertEqual(result["answer"], 2)
        self.assertEqual(host.total_usage["llm_calls"], 2)
        self.assertIn("preflight", result["revision"]["critic"]["reason"])

    def test_tokenizer_preflight_exception_preserves_valid_primary(self):
        with patch("agent.executable_critic_agent.count_tokens_in_messages", side_effect=ValueError("tokenizer failure")):
            host, result = self.run_agent([fetch(), final_call()])
        self.assertEqual(result["answer"], 2)
        self.assertEqual(host.total_usage["llm_calls"], 2)
        self.assertIn("preflight", result["revision"]["critic"]["reason"])

    def test_keep_stops_with_one_critic_call_zero_tools(self):
        host, result = self.run_agent(self.responses({"decision":"KEEP","reason":"No discrepancy established."})[:3])
        self.assertEqual(result["answer"], 2)
        self.assertEqual(host.total_usage["llm_calls"], 3)
        self.assertEqual(host.tool_calls_made, 3)
        self.assertFalse(result["revision"]["critic_revised"])

    def test_inspection_unknown_preserves_primary_without_free_synthesis(self):
        host, result = self.run_agent(self.responses(second_value={"decision":"UNKNOWN","reason":"Ambiguous intended mapping."}))
        self.assertEqual(result["answer"], 2)
        self.assertEqual(host.tool_calls_made, 4)
        self.assertEqual(host.total_usage["llm_calls"], 4)
        self.assertEqual(result["revision"]["final_check_issues"], [])

    def test_failed_probe_preserves_primary_and_is_audited(self):
        host, result = self.run_agent(self.responses(first_value=first("raise ValueError('probe failed')"))[:3])
        self.assertEqual(result["answer"], 2)
        self.assertEqual(host.tool_calls_made, 4)
        self.assertIn("error", result["revision"]["critic"]["probe_output"])
        self.assertEqual(result["revision"]["final_check_issues"], [])

    def test_probe_mutation_does_not_change_primary_or_immutable_witness(self):
        code = "retrieved_resources['Observation'][0]['valueQuantity']['value'] = 999\n" + INSPECT_CODE
        host, result = self.run_agent(self.responses(first_value=first(code))[:3])
        self.assertEqual(result["answer"], 2)
        self.assertEqual(result["retrieved_fhir_resources"]["Observation"][0]["valueQuantity"]["value"], 70)
        self.assertIn("immutable", result["revision"]["critic"]["reason"])
        self.assertEqual(host.tool_calls_made, 4)

    def test_first_native_tool_call_rejected_without_execution(self):
        native = fetch()
        host, result = self.run_agent([fetch(), final_call(), native])
        self.assertEqual(result["answer"], 2)
        self.assertEqual(host.tool_calls_made, 3)
        self.assertIn("JSON-only", result["revision"]["critic"]["reason"])

    def test_second_native_tool_call_rejected_without_execution(self):
        host, result = self.run_agent([fetch(), final_call(), reply(json.dumps(first())), fetch()])
        self.assertEqual(result["answer"], 2)
        self.assertEqual(host.tool_calls_made, 4)

    def test_failed_patch_execution_gets_no_metadata_correction(self):
        value = proposal(new="1 / 0")
        host, result = self.run_agent(self.responses(second_value=value))
        self.assertEqual(result["answer"], 2)
        self.assertEqual(host.tool_calls_made, 5)
        self.assertEqual(host.total_usage["llm_calls"], 4)
        self.assertIn("alternate_execution", result["revision"]["critic"])

    def test_substantive_same_contributors_projection_change_allowed(self):
        value = proposal(old="answer = len(rows)", new="answer = str(len(rows))")
        host, result = self.run_agent(self.responses(second_value=value))
        self.assertEqual(result["answer"], "2")
        self.assertEqual(result["revision"]["primary"]["contributing_ids"], result["revision"]["final"]["contributing_ids"])
        self.assertEqual(host.tool_calls_made, 6)

    def test_operation_only_change_is_no_effect_and_preserves_primary(self):
        value = proposal(old="count selected weight measurements", new="count the same selected records")
        host, result = self.run_agent(self.responses(second_value=value))
        self.assertEqual(result["answer"], 2)
        self.assertFalse(result["revision"]["critic_revised"])
        self.assertEqual(host.tool_calls_made, 5)
        self.assertIn("no substantive", result["revision"]["critic"]["reason"])

    def test_candidate_specific_established_empty_uses_new_artifact(self):
        value = proposal(new="r['valueQuantity']['value'] > 100")
        host, result = self.run_agent(self.responses(second_value=value))
        self.assertEqual(result["answer"], 0)
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["revision"]["primary"]["answer"], 2)
        self.assertEqual(result["revision"]["critic"]["candidate_issues"], [])
        self.assertEqual(host.tool_calls_made, 6)
        # Intentionally fallible semantics: the host proves source/execution,
        # not the correctness of the model's strict-threshold interpretation.
        self.assertEqual(result["revision"]["critic"]["proposal"], value)

    def test_raw_global_source_error_preserves_primary_and_is_audited(self):
        with patch("agent.executable_critic_agent.check_fields", return_value={"issues":["global failure"],"results":[
                {"citation":{"resource_id":"Observation/o2","field_path":"valueQuantity.value","value":71},
                 "status":"valid","warnings":[]}]}):
            host, result = self.run_agent(self.responses())
        self.assertEqual(result["answer"], 2)
        self.assertEqual(host.tool_calls_made, 6)
        self.assertIn("global failure", result["revision"]["critic"]["candidate_issues"])
        self.assertEqual(result["revision"]["final_check_issues"], [])

    def test_contradictory_raw_error_envelope_rejected(self):
        check = {"issues":[], "error":"failed", "results":[
            {"citation":{"resource_id":"Observation/o2","field_path":"valueQuantity.value","value":71},
             "status":"valid","warnings":[]}]}
        with patch("agent.executable_critic_agent.check_fields", return_value=check):
            host, result = self.run_agent(self.responses())
        self.assertEqual(result["answer"], 2)
        self.assertEqual(host.tool_calls_made, 6)
        self.assertIn("malformed", result["revision"]["critic"]["reason"])

    def test_escaping_admitted_model_exception_is_fatal_and_counted(self):
        host, result = self.run_agent([fetch(), final_call(), RuntimeError("provider escaped")])
        self.assertIsNone(result["answer"])
        self.assertEqual(result["revision"]["release_policy"], "aggregate_accounting_failed")
        self.assertEqual(result["revision"]["critic"]["model_calls"], 1)
        self.assertEqual(host.max_total_tokens, 200000)

    def test_incomplete_or_exception_source_checker_preserves_primary(self):
        for effect in [None, RuntimeError("store unavailable")]:
            kwargs = {"return_value":{"issues":[],"results":[]}} if effect is None else {"side_effect":effect}
            with self.subTest(effect=effect), patch("agent.executable_critic_agent.check_fields", **kwargs):
                host, result = self.run_agent(self.responses())
            self.assertEqual(result["answer"], 2)
            checks = [m for m in result["trace"] if isinstance(m,dict) and m.get("tool_call_id")=="v5-critic-source-check"]
            self.assertEqual(len(checks), 1)
            self.assertEqual(host.tool_calls_made, 6)

    def test_two_call_budget_reservation_skips_before_any_critic_call(self):
        host, result = self.run_agent([fetch(), final_call()], tokens=10000)
        self.assertEqual(result["answer"], 2)
        self.assertEqual(host.total_usage["llm_calls"], 2)
        self.assertEqual(result["revision"]["critic"]["status"], "coverage_skip")

    def test_intact_oversized_code_packet_skips_without_truncation(self):
        code = CODE + "\n# " + "important_literal " * 8000
        host, result = self.run_agent([fetch(), final_call(code)])
        self.assertEqual(result["answer"], 2)
        self.assertEqual(host.total_usage["llm_calls"], 2)
        self.assertEqual(result["revision"]["critic"]["status"], "coverage_skip")

    def test_unknown_provider_usage_is_globally_fatal_despite_valid_primary(self):
        usage = {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,"llm_calls":1,"cost":None,
                 "per_call":[{"prompt":0,"completion":0,"usage_available":False}]}
        host, result = self.run_agent([fetch(), final_call(), (FixtureMessage(**vars(reply('{}'))), None, usage)])
        self.assertIsNone(result["answer"])
        self.assertEqual(result["revision"]["release_policy"], "aggregate_accounting_failed")
        self.assertEqual(result["revision"]["critic"]["status"], "accounting_failed")
        self.assertTrue(result["revision"]["phases"][-1]["fatal"])

    def test_provider_error_without_usage_is_globally_fatal(self):
        _, result = self.run_agent([fetch(), final_call(), (None, "transport failed", None)])
        self.assertIsNone(result["answer"])
        self.assertEqual(result["revision"]["release_policy"], "aggregate_accounting_failed")

    def test_response_without_usage_is_globally_fatal(self):
        host, result = self.run_agent([fetch(), final_call(), (FixtureMessage(**vars(reply('{}'))), None, None)])
        self.assertIsNone(result["answer"])
        self.assertEqual(result["revision"]["release_policy"], "aggregate_accounting_failed")
        self.assertEqual(host.total_usage["llm_calls"], 3)
        self.assertFalse(host.call_usage[-1]["usage_available"])

    def test_empty_usage_envelope_is_globally_fatal(self):
        host, result = self.run_agent([fetch(), final_call(), (FixtureMessage(**vars(reply('{}'))), None, {})])
        self.assertIsNone(result["answer"])
        self.assertEqual(host.total_usage["llm_calls"], 3)

    def test_first_completion_subcap_overshoot_is_fatal_below_aggregate_ceiling(self):
        usage = {"prompt_tokens":100,"completion_tokens":1801,"total_tokens":1901,"llm_calls":1,"cost":0,
                 "per_call":[{"prompt":100,"completion":1801,"usage_available":True}]}
        host, result = self.run_agent([fetch(), final_call(), (FixtureMessage(**vars(reply('{}'))), None, usage)])
        self.assertIsNone(result["answer"])
        self.assertTrue(result["revision"]["phases"][-1]["fatal"])
        self.assertIn("subcap", result["revision"]["critic"]["reason"])
        self.assertLess(host.total_usage["total_tokens"], 25000)

    def test_second_completion_subcap_overshoot_is_fatal_below_aggregate_ceiling(self):
        usage = {"prompt_tokens":100,"completion_tokens":4001,"total_tokens":4101,"llm_calls":1,"cost":0,
                 "per_call":[{"prompt":100,"completion":4001,"usage_available":True}]}
        host, result = self.run_agent([fetch(), final_call(), reply(json.dumps(first())),
            (FixtureMessage(**vars(reply(json.dumps(proposal())))), None, usage)])
        self.assertIsNone(result["answer"])
        self.assertTrue(result["revision"]["phases"][-1]["fatal"])
        self.assertEqual(host.tool_calls_made, 4)
        self.assertEqual(result["revision"]["critic"]["model_calls"], 2)

    def test_malformed_per_call_usage_is_fatal_without_formatter_crash(self):
        for entry in [7, {"prompt":True,"completion":50,"usage_available":True},
                      {"prompt":100,"completion":"bad","usage_available":True}]:
            usage = {"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,"llm_calls":1,"cost":0,
                     "per_call":[entry]}
            with self.subTest(entry=entry):
                host, result = self.run_agent([fetch(), final_call(), (FixtureMessage(**vars(reply('{}'))), None, usage)])
            self.assertIsNone(result["answer"])
            self.assertEqual(result["revision"]["release_policy"], "aggregate_accounting_failed")
            self.assertEqual(host.total_usage["total_tokens"], 450)
            self.assertFalse(host.call_usage[-1]["usage_available"])

    def test_malformed_optional_cached_usage_is_fatal_without_formatter_crash(self):
        for cached in ["bad", True, -1, float("nan")]:
            usage = {"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,"llm_calls":1,"cost":0,
                     "per_call":[{"prompt":100,"completion":50,"cached":cached,"usage_available":True}]}
            with self.subTest(cached=cached):
                host, result = self.run_agent([fetch(), final_call(), (FixtureMessage(**vars(reply('{}'))), None, usage)])
            self.assertIsNone(result["answer"])
            self.assertEqual(result["revision"]["release_policy"], "aggregate_accounting_failed")
            self.assertEqual(host.total_usage["total_tokens"], 450)
            self.assertEqual(host.call_usage[-1]["cached"], 0)
            self.assertFalse(host.call_usage[-1]["usage_available"])

    def test_malformed_aggregate_counts_preserve_known_totals_and_fatal_audit(self):
        for bad in [float("inf"), -1000, "bad"]:
            usage = {"prompt_tokens":bad,"completion_tokens":50,"total_tokens":150,"llm_calls":1,"cost":0,
                     "per_call":[{"prompt":100,"completion":50,"cached":0,"usage_available":True}]}
            with self.subTest(bad=bad):
                host, result = self.run_agent([fetch(), final_call(), (FixtureMessage(**vars(reply('{}'))), None, usage)])
            self.assertIsNone(result["answer"])
            self.assertEqual(result["revision"]["release_policy"], "aggregate_accounting_failed")
            self.assertEqual(host.total_usage["prompt_tokens"], 200)
            self.assertEqual(host.total_usage["llm_calls"], 3)
            self.assertFalse(host.call_usage[-1]["usage_available"])

    def test_nonfinite_cost_and_extra_per_call_fields_cannot_escape_serialization(self):
        values = []
        for cost in [float("inf"), float("nan"), -1, True]:
            values.append({"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,"llm_calls":1,"cost":cost,
                     "per_call":[{"prompt":100,"completion":50,"cached":0,"usage_available":True}]})
        values.append({"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,"llm_calls":1,"cost":0,
                     "per_call":[{"prompt":100,"completion":50,"cached":0,"usage_available":True,"extra":float("nan")}]})
        for usage in values:
            with self.subTest(usage=usage):
                host, result = self.run_agent([fetch(), final_call(), (FixtureMessage(**vars(reply('{}'))), None, usage)])
            self.assertIsNone(result["answer"])
            self.assertEqual(result["revision"]["release_policy"], "aggregate_accounting_failed")
            json.dumps(result["tokens"], allow_nan=False)
            self.assertEqual(host.total_usage["total_tokens"], 450)
            self.assertFalse(host.call_usage[-1]["usage_available"])
            self.assertNotIn("extra", host.call_usage[-1])

    def test_reported_completion_overshoot_is_fatal(self):
        usage = {"prompt_tokens":20000,"completion_tokens":10000,"total_tokens":30000,"llm_calls":1,"cost":0,
                 "per_call":[{"prompt":20000,"completion":10000,"usage_available":True}]}
        _, result = self.run_agent([fetch(), final_call(), (FixtureMessage(**vars(reply('{}'))), None, usage)])
        self.assertIsNone(result["answer"])
        self.assertEqual(result["revision"]["release_policy"], "aggregate_accounting_failed")


class CriticEntryTests(unittest.TestCase):
    def test_additive_dispatch_and_legacy_forwarding(self):
        import run_agent
        job = ("question", "multi_agent_v5", "same-model", False, None, False, {}, None)
        with patch("agent.executable_critic_agent.run_v5_job", return_value="v5") as worker:
            self.assertEqual(run_agent.dispatch_agent_job(job), "v5")
        worker.assert_called_once_with(job)
        legacy = ("question", "single_agent", "same-model", False, None, False, {}, None)
        with patch("run_agent.run_agent_with_input", return_value="single") as worker:
            self.assertEqual(run_agent.dispatch_agent_job(legacy), "single")
        worker.assert_called_once_with(legacy)

    def test_subset_full_parser_forwards_actual_team_and_gzip(self):
        from pathlib import Path
        from types import SimpleNamespace
        from scripts.run_experiments import Arm, runner_argv
        from scripts.full_benchmark import arm_command
        import run_agent
        args = SimpleNamespace(team_strategy="multi_agent_v5", model="same-model", input="synthetic.csv",
            output_dir="/private/tmp/v5-output", split_manifest="frozen.json", split="dev",
            num_processes=1, save_interval=1, base_url=None, max_total_tokens=200000,
            max_tool_calls=30, limit=300, verbose=False, audit_storage="gzip")
        parsed = run_agent.build_parser().parse_args(runner_argv(Arm("ordinary", "multi_agent"), args))
        self.assertEqual(parsed.agent_strategy, "multi_agent_v5")
        self.assertEqual(parsed.audit_storage, "gzip")
        control = runner_argv(Arm("ordinary", "single_agent"), args)
        self.assertEqual(control[control.index("--agent_strategy") + 1], "single_agent")
        args.input, args.output_dir = Path(args.input), Path(args.output_dir)
        command = arm_command("multi_agent", args)
        self.assertEqual(command[command.index("--agent_strategy") + 1], "multi_agent_v5")

    def test_worker_common_budget_and_lossless_externalization_contract(self):
        from agent.executable_critic_agent import run_v5_job
        budget = {"max_total_tokens":200000,"max_tool_calls":30}
        job = ("input", "multi_agent_v5", "same-model", False, None, False, budget,
               None, True, {"output_dir":"/private/tmp/v5-output","question_id":"synthetic"})
        with patch("agent.executable_critic_agent.ExecutableCriticAgent") as cls, \
             patch("utils.run_agent_safe", return_value={"audit":"carrier"}) as safe, \
             patch("scripts.audit_store.externalize_result_audit", return_value={"saved":True}) as save:
            self.assertEqual(run_v5_job(job), {"saved":True})
        cls.return_value.set_budget.assert_called_once_with(**budget)
        cls.return_value.set_perturbation.assert_called_once_with(None)
        safe.assert_called_once_with(cls.return_value, "input", include_full_audit=True)
        save.assert_called_once_with({"audit":"carrier"}, "/private/tmp/v5-output", question_id="synthetic")


if __name__ == "__main__":
    unittest.main()
