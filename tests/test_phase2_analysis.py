"""P2-14: the analysis role computes in the Python sandbox over the shared
records and returns a draft with evidence, operation, and contributing IDs."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.analysis_agent import AnalysisAgent
from agent.base_agent import BaseAgent
from agent.controller import FixedController
from agent.retrieval_agent import RetrievalAgent
from tools import cache

PATIENT = "11111111-1111-1111-1111-111111111111"
NOW = "2153-12-31 23:59:00"
CONTEXT = {"question": "What was the lowest weight of the patient?", "patient_fhir_id": PATIENT,
           "current_time": NOW, "assumptions": f"Patient FHIR ID is {PATIENT}.\nAssume current time is {NOW}."}
QUESTION = f"Question: {CONTEXT['question']}\nContext:\n{CONTEXT['assumptions']}"
RECORDS = {"resource_types": ["Observation"],
           "scope": {"patient_fhir_id": PATIENT, "current_time": NOW, "encounter_ids": []},
           "limitation": None, "record_ids": {"Observation": ["Observation/o1", "Observation/o2"]}}
MIN_CODE = ("lowest = min(retrieved_resources['Observation'], key=lambda r: r['valueQuantity']['value'])\n"
            "answer = {'id': lowest['id'], 'value': lowest['valueQuantity']['value']}")


def observation(rid, value):
    return {"resourceType": "Observation", "id": rid, "subject": {"reference": f"Patient/{PATIENT}"},
            "effectiveDateTime": "2150-01-02T08:00:00", "valueQuantity": {"value": value, "unit": "kg"}}


def run_python(code=MIN_CODE):
    return SimpleNamespace(role="assistant", content=None, tool_calls=[SimpleNamespace(
        id="call1", function=SimpleNamespace(name="execute_python_code", arguments=json.dumps({"code": code})))])


def reply(content):
    return SimpleNamespace(role="assistant", content=content, tool_calls=None)


def draft(**overrides):
    body = {"answer": 68, "evidence": [{"resource_id": "Observation/o2", "field_path": "valueQuantity.value",
                                        "value": 68}],
            "operation": "min of valueQuantity.value over the patient's weight Observations",
            "contributing_ids": ["Observation/o1", "Observation/o2"], "limitation": None}
    body.update(overrides)
    return reply("Done.\n" + json.dumps(body))


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("agent.base_agent.setup_api_keys")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.host = BaseAgent("test-model")
        self.workspace = {"Observation": [observation("o1", 70), observation("o2", 68)]}
        self.agent = AnalysisAgent(self.host, CONTEXT, self.workspace)

    def analyze(self, *responses, task=None):
        self.host._llm_call = Mock(side_effect=[(r, None, None) for r in responses])
        return self.agent(task or {"context": CONTEXT, "records": RECORDS})

    def test_analyst_can_supplement_retrieval_and_execute_in_the_sandbox(self):
        self.assertEqual(self.agent.role.toolbox.names, ("get_resources_by_patient_fhir_id",
                                                       "get_resources_by_resource_id", "execute_python_code"))

    def test_computes_in_the_sandbox_and_returns_the_draft(self):
        result = self.analyze(run_python(), draft())
        python_reply = json.loads(self.host.messages[3]["content"])
        self.assertEqual(python_reply, {"answer": {"id": "o2", "value": 68}, "mode": "auto"})
        self.assertEqual(result, json.loads(draft().content.split("\n", 1)[1]))

    def test_prompt_shares_scope_and_previews_the_records(self):
        self.analyze(draft())
        prompt = self.host._llm_call.call_args.args[0][1]["content"]
        self.assertIn(CONTEXT["question"], prompt)
        self.assertIn('"encounter_ids": []', prompt)
        self.assertIn("- Observation: 2 records, e.g. {", prompt)
        self.assertNotIn("subject", prompt)

    def test_contributing_ids_are_explicit_and_deduplicated(self):
        result = self.analyze(draft(contributing_ids=["Observation/o2", "Observation/o2"], evidence=[
            {"resource_id": "Observation/o2", "field_path": "valueQuantity.value", "value": 68},
            {"resource_id": "Observation/o2", "field_path": "valueQuantity.unit", "value": "kg"}]))
        self.assertEqual(result["contributing_ids"], ["Observation/o2"])

    def test_abstention_is_a_valid_draft(self):
        result = self.analyze(draft(answer=None, evidence=[], contributing_ids=[],
                                    limitation="No weight Observations were retrieved"))
        self.assertIsNone(result["answer"])
        self.assertEqual(result["limitation"], "No weight Observations were retrieved")

    def test_malformed_drafts_are_errors(self):
        cases = [
            (reply("The lowest weight is 68 kg."), "no JSON draft"),
            (reply('{"answer": NaN, "evidence": []}'), "draft answer must be"),
            (draft(evidence=[{"resource_id": "Observation/o2", "value": 68}]), "draft evidence must be"),
            (draft(evidence="Observation/o2"), "draft evidence must be"),
            (draft(contributing_ids=["o2"]), "contributing_ids must be"),
        ]
        for response, problem in cases:
            self.assertIn(problem, self.analyze(response)["error"])

    def test_repair_shows_previous_draft_issues_and_updated_records(self):
        self.workspace["Observation"].append(observation("o3", 65))
        issue = {"type": "missing_evidence", "record": None, "field": None, "responsible_role": "retrieval",
                 "action": "include all weights"}
        result = self.analyze(run_python(), draft(answer=65), task={
            "context": CONTEXT, "records": RECORDS, "issues": [issue], "previous_draft": {"answer": 68}})
        prompt = self.host._llm_call.call_args_list[0].args[0][1]["content"]
        self.assertIn("missing_evidence", prompt)
        self.assertIn('{"answer": 68}', prompt)
        self.assertIn("- Observation: 3 records", prompt)
        self.assertEqual(json.loads(self.host.messages[3]["content"]),
                         {"answer": {"id": "o3", "value": 65}, "mode": "auto"})
        self.assertEqual(result["answer"], 65)

    def test_model_error_is_reported(self):
        self.host._llm_call = Mock(return_value=(None, "provider down", None))
        self.assertEqual(self.agent({"context": CONTEXT, "records": RECORDS}), {"error": "provider down"})


class PipelineTests(unittest.TestCase):
    def test_retrieval_then_sandboxed_analysis_through_the_controller(self):
        packet = {"text": "Starting packet (no weights)", "resources": {}, "resource_ids": {}}
        store = Mock(search_with_pagination=Mock(return_value=[observation("o1", 70), observation("o2", 68)]))
        verified = {}

        class System(FixedController):
            def build_roles(self, context, workspace):
                def verification(task):
                    verified["draft"] = task["draft"]
                    return {"issues": []}
                return {"retrieval": RetrievalAgent(self, context, workspace),
                        "analysis": AnalysisAgent(self, context, workspace), "verification": verification}

        fetch = SimpleNamespace(role="assistant", content=None, tool_calls=[SimpleNamespace(
            id="fetch1", function=SimpleNamespace(name="get_resources_by_patient_fhir_id", arguments=json.dumps(
                {"resource_type": "Observation", "patient_fhir_id": PATIENT})))])
        scope = reply(json.dumps({"resource_types": ["Observation"], "encounter_ids": [], "limitation": None}))
        with patch("agent.base_agent.setup_api_keys"), \
                patch("agent.retrieval_agent.get_starting_packet", return_value=packet), \
                patch.object(cache, "CACHE_ENABLED", False), \
                patch("tools.resource_tools.get_fhir_client", return_value=store):
            system = System("test-model")
            system._llm_call = Mock(side_effect=[(r, None, None) for r in (fetch, scope, run_python(), draft())])
            output = system.run(QUESTION)
        self.assertIsNone(output["error"])
        self.assertEqual(output["answer"], 68)
        self.assertEqual(verified["draft"]["contributing_ids"], ["Observation/o1", "Observation/o2"])
        self.assertIn("min of valueQuantity.value", verified["draft"]["operation"])
        self.assertEqual([c["name"] for c in output["tool_calls"]],
                         ["get_resources_by_patient_fhir_id", "execute_python_code"])


if __name__ == "__main__":
    unittest.main()
