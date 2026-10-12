import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.direct_llm_agent import DirectLLMAgent
from agent.multi_turn_code_resource_agent import MultiTurnCodeResourceAgent
from agent.output_schema import parse_final_message
from agent.single_agent import SingleAgent
from tools import cache, resource_tools
from tools.field_check import check_fields


PATIENT = "11111111-1111-1111-1111-111111111111"
CONTEXT = f"Question: What was the weight in kg?\nContext:\nPatient FHIR ID is {PATIENT}."
OBSERVATION = {
    "resourceType": "Observation", "id": "o1", "subject": {"reference": f"Patient/{PATIENT}"},
    "effectiveDateTime": "2186-09-15T18:00:00-04:00", "issued": "2186-09-15T18:21:00-04:00",
    "valueQuantity": {"value": 7, "unit": "kg"},
}


def citation(value=7, path="valueQuantity.value", rid="Observation/o1"):
    return {"resource_id": rid, "field_path": path, "value": value}


def message(answer, evidence=None):
    return SimpleNamespace(role="assistant", tool_calls=None, content=json.dumps({
        "answer": answer, "evidence": evidence if evidence is not None else [citation(answer)],
        "limitation": None,
    }))


def tool_call(name, **arguments):
    return SimpleNamespace(id="call1", function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def packet():
    visible = {"id": "o1", "valueQuantity": {"value": 7, "unit": "kg"}}
    return {"text": "## Observation\n" + json.dumps(visible), "resources": {"Observation": [visible]},
            "resource_ids": {"Observation": ["o1"]}, "n_tokens": 20, "truncated": False}


class SingleAgentTests(unittest.TestCase):
    def run_agent(self, responses, budget=None, source=None, context=CONTEXT):
        with patch("agent.base_agent.setup_api_keys"), patch("agent.single_agent.get_starting_packet", return_value=packet()), \
                patch("tools.field_check.fetch_from_store", return_value=source or {"o1": deepcopy(OBSERVATION)}):
            agent = SingleAgent("test-model")
            if budget is not None:
                agent.set_budget(max_tool_calls=budget)
            agent._llm_call = Mock(side_effect=[(response, error, None) for response, error in responses])
            return agent, agent.run(context)

    def test_revised_citations_are_checked_and_invalid_answer_abstains(self):
        agent, output = self.run_agent([(message(99), None), (message(8), None)])
        self.assertIsNone(output["answer"])
        self.assertIn("value_mismatch", output["limitation"])
        self.assertEqual(output["revision"]["draft"]["answer"], 99)
        self.assertEqual(output["revision"]["final"]["answer"], 8)
        self.assertTrue(output["revision"]["revised"])
        self.assertEqual(agent._llm_call.call_count, 2)
        self.assertEqual(output["budget"]["tool_calls_run"], 2)
        self.assertEqual([c["name"] for c in output["tool_calls"]], ["check_fields", "check_fields"])

    def test_one_repair_can_release_a_checked_answer(self):
        _, output = self.run_agent([(message(99), None), (message(7), None)])
        self.assertEqual(output["answer"], 7)
        self.assertEqual(output["revision"]["final_check_issues"], [])

    def test_revision_failure_preserves_draft_and_partial_trace(self):
        _, output = self.run_agent([(message(99), None), (None, "provider unavailable")])
        self.assertIsNone(output["answer"])
        self.assertEqual(output["revision"]["draft"]["answer"], 99)
        self.assertEqual(output["revision"]["revision_error"], "provider unavailable")
        self.assertEqual(output["error"], "provider unavailable")
        self.assertIn("Revision stopped", output["limitation"])

    def test_auto_check_cannot_bypass_zero_tool_budget(self):
        agent, output = self.run_agent([(message(7), None)], budget=0)
        self.assertIsNone(output["answer"])
        self.assertEqual(output["budget"]["tool_calls_run"], 0)
        self.assertIn("did not allow", output["limitation"])
        self.assertEqual(agent._llm_call.call_count, 1)

    def test_exhausted_budget_cannot_release_an_unchecked_revision(self):
        _, output = self.run_agent([(message(99), None), (message(7), None)], budget=1)
        self.assertIsNone(output["answer"])
        self.assertEqual(output["budget"]["tool_calls_run"], 1)
        self.assertIn("did not allow", output["limitation"])

    def test_preload_contains_only_packet_fields(self):
        _, output = self.run_agent([(message(7), None)])
        loaded = output["retrieved_fhir_resources"]["Observation"][0]
        self.assertEqual(loaded, {**packet()["resources"]["Observation"][0], "resourceType": "Observation"})
        self.assertNotIn("issued", loaded)
        self.assertNotIn("subject", loaded)

    def test_source_checker_cannot_trust_mutated_working_resources(self):
        call = tool_call("execute_python_code", code="mutate the working copy")
        first = SimpleNamespace(role="assistant", content=None, tool_calls=[call])

        def corrupt_working_copy(code, global_vars):
            global_vars["retrieved_resources"]["Observation"][0]["valueQuantity"]["value"] = 99
            return {"stdout": "99"}

        with patch("agent.multi_turn_code_resource_agent.get_tool", return_value=corrupt_working_copy):
            _, output = self.run_agent([(first, None), (message(99), None), (message(7), None)])
        self.assertEqual(output["answer"], 7)
        self.assertIn("value_mismatch", output["revision"]["check_issues"][0])

    def test_other_patient_search_is_rejected(self):
        with patch("agent.base_agent.setup_api_keys"):
            agent = SingleAgent("test-model")
        agent._patient_fhir_id = PATIENT
        call = tool_call("get_resources_by_patient_fhir_id", resource_type="Observation", patient_fhir_id="other")
        with patch("agent.multi_turn_code_resource_agent.get_tool") as get_tool:
            response = agent._handle_tool_call(call, {})
        self.assertIn("Retrieve only", response["content"])
        get_tool.assert_not_called()

    def test_direct_agent_does_one_call_without_tools(self):
        with patch("agent.base_agent.setup_api_keys"), patch("agent.direct_llm_agent.get_starting_packet", return_value=packet()):
            agent = DirectLLMAgent("test-model")
            agent._llm_call = Mock(return_value=(message(7), None, None))
            output = agent.run(CONTEXT)
        self.assertEqual(output["answer"], 7)
        self.assertEqual(output["tool_calls"], [])
        self.assertEqual(agent._llm_call.call_args.args[0][1]["content"], f"{CONTEXT}\n\n{packet()['text']}")
        agent._llm_call.assert_called_once()

    def test_converted_answer_keeps_exact_source_value_without_forced_revision(self):
        converted = 7 / 0.45359237
        _, output = self.run_agent([(message(converted, [citation(7)]), None)],
                                   context=CONTEXT.replace("in kg", "in pounds"))
        self.assertEqual(output["answer"], converted)
        self.assertFalse(output["revision"]["revised"])
        self.assertEqual(output["evidence"][0]["value"], 7)

    def test_direct_agent_truncated_json_is_an_explicit_failed_output(self):
        response = SimpleNamespace(role="assistant", tool_calls=None, content='{"answer": 7')
        with patch("agent.base_agent.setup_api_keys"), patch("agent.direct_llm_agent.get_starting_packet", return_value=packet()):
            agent = DirectLLMAgent("test-model")
            agent._llm_call = Mock(return_value=(response, None, None))
            output = agent.run(CONTEXT)
        self.assertIsNone(output["answer"])
        self.assertIn("complete JSON", output["limitation"])
        self.assertIn("Invalid final output", output["error"])
        agent._llm_call.assert_called_once()


class FieldCheckTests(unittest.TestCase):
    def test_correct_value_from_wrong_patient_is_flagged(self):
        observation = deepcopy(OBSERVATION)
        observation["subject"]["reference"] = "Patient/other"
        result = check_fields([citation()], resources={"o1": observation}, patient_fhir_id=PATIENT)
        self.assertIn("patient_scope", result["issues"][0])

    def test_condition_in_future_encounter_is_flagged(self):
        condition = {"resourceType": "Condition", "id": "c1", "encounter": {"reference": "Encounter/e1"},
                     "subject": {"reference": f"Patient/{PATIENT}"}, "code": {"text": "condition"}}
        encounter = {"resourceType": "Encounter", "id": "e1", "period": {"start": "2187-01-01T00:00:00-04:00"}}
        result = check_fields([citation("condition", "code.text", "Condition/c1")],
                              resources={"c1": condition, "e1": encounter}, current_time="2186-12-31 23:59:00")
        self.assertIn("current_time", result["issues"][0])

    def test_encounter_reference_is_normalized_and_icu_child_allowed(self):
        observation = deepcopy(OBSERVATION)
        observation["encounter"] = {"reference": "Encounter/icu"}
        child = {"resourceType": "Encounter", "id": "icu", "partOf": {"reference": "Encounter/hospital"}}
        result = check_fields([citation()], resources={"o1": observation, "icu": child},
                              encounter_id="Encounter/hospital")
        self.assertEqual(result["issues"], [])

    def test_latest_hospital_scope_uses_original_encounters_and_cutoff(self):
        observation = deepcopy(OBSERVATION)
        observation["encounter"] = {"reference": "Encounter/old"}
        encounters = [{"resourceType": "Encounter", "id": rid, "subject": {"reference": f"Patient/{PATIENT}"},
                       "identifier": [{"system": "http://example/encounter-hosp"}], "period": {"start": start, "end": end}}
                      for rid, start, end in [("old", "2185-01-01", "2185-01-03"),
                                              ("latest", "2186-01-01", "2186-01-03"),
                                              ("future", "2187-01-01", "2187-01-03")]]
        client = Mock()
        client.search_with_pagination.return_value = encounters
        with patch("tools.field_check.fetch_from_store", return_value={"o1": observation}), \
                patch("fhir_client.get_fhir_client", return_value=client):
            result = check_fields([citation()], question="Weight during the latest hospital encounter",
                                  patient_fhir_id=PATIENT, current_time="2186-12-31 23:59:00")
        self.assertTrue(any("scoped Encounter/latest" in issue for issue in result["issues"]))

    def test_measurement_time_and_unit_mistakes_are_flagged(self):
        result = check_fields([citation(OBSERVATION["issued"], "issued"), citation()],
                              resources={"o1": OBSERVATION}, question="What was the weight in ml?")
        self.assertTrue(any("time_field" in issue for issue in result["issues"]))
        self.assertTrue(any("unit:" in issue for issue in result["issues"]))

    def test_compatible_unit_conversion_is_informational(self):
        result = check_fields([citation()], resources={"o1": OBSERVATION}, question="What was the weight in pounds?",
                              patient_fhir_id=PATIENT)
        self.assertEqual(result["issues"], [])
        self.assertAlmostEqual(result["results"][0]["conversion"]["factor"], 1 / 0.45359237)


class ToolDispatchTests(unittest.TestCase):
    def setUp(self):
        with patch("agent.base_agent.setup_api_keys"):
            self.agent = MultiTurnCodeResourceAgent("test-model")

    def test_point_fetch_retains_other_resources_and_replaces_same_id(self):
        loaded = {"Observation": [{"id": "keep"}, {"id": "replace", "value": 1}]}
        fetched = {"Observation": [{"id": "replace", "value": 2}]}
        with patch("agent.multi_turn_code_resource_agent.get_tool", return_value=lambda **args: fetched):
            self.agent._handle_tool_call(tool_call("get_resources_by_resource_id", resource_type="Observation", resource_id="replace"), loaded)
        self.assertEqual(loaded, {"Observation": [{"id": "keep"}, {"id": "replace", "value": 2}]})

    def test_empty_patient_fetch_replaces_partial_packet(self):
        loaded = {"Observation": [{"id": "old"}]}
        with patch("agent.multi_turn_code_resource_agent.get_tool", return_value=lambda **args: {}):
            self.agent._handle_tool_call(tool_call("get_resources_by_patient_fhir_id", resource_type="Observation", patient_fhir_id=PATIENT), loaded)
        self.assertEqual(loaded, {"Observation": []})

    def test_retrieval_error_is_reported_and_does_not_erase_loaded_data(self):
        loaded = {"Observation": [{"id": "keep"}]}
        with patch("agent.multi_turn_code_resource_agent.get_tool", return_value=lambda **args: {"error": "store down"}):
            result = self.agent._handle_tool_call(tool_call("get_resources_by_resource_id", resource_type="Observation", resource_id="o1"), loaded)
        self.assertEqual(json.loads(result["content"]), {"error": "store down"})
        self.assertEqual(loaded, {"Observation": [{"id": "keep"}]})

    def test_bad_tool_arguments_return_a_recoverable_tool_message(self):
        call = tool_call("get_resources_by_resource_id")
        call.function.arguments = "{broken"
        result = self.agent._handle_tool_call(call, {})
        self.assertIn("error", json.loads(result["content"]))

    def test_empty_and_injected_ids_cannot_search_entire_store(self):
        with patch.object(cache, "CACHE_ENABLED", False), patch("tools.resource_tools.get_fhir_client") as get_client:
            for identifier in ("", "o1&_count=1000"):
                result = resource_tools.get_resources_by_resource_id("Observation", identifier)
                self.assertIn("error", result)
            get_client.assert_not_called()

    def test_transient_failures_are_not_saved_or_reused_from_cache(self):
        result = {"error": "temporary failure"}
        function = Mock(side_effect=[result, {"Observation": []}])
        function.__name__ = "test_retrieval"
        decorated = cache.cached_tool(function)
        with patch.object(cache, "CACHE_ENABLED", True), patch.object(cache, "load_from_cache", return_value=result), \
                patch.object(cache, "save_to_cache") as save:
            self.assertEqual(decorated(), result)
            save.assert_not_called()
            self.assertEqual(decorated(), {"Observation": []})
            save.assert_called_once()

    def test_nested_answer_key_cannot_override_final_output(self):
        final = {"answer": "correct", "evidence": [{"value": {"answer": "nested"}}], "limitation": None}
        answer, _, _ = parse_final_message(json.dumps(final))
        self.assertEqual(answer, "correct")


if __name__ == "__main__":
    unittest.main()
