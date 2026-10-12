"""P2-13: the retrieval role starts from the starting packet, fetches only
within the question's patient and current time, and reports records and scope."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.base_agent import BaseAgent
from agent.controller import FixedController
from agent.framework import RETRIEVAL_TOOLS
from agent.retrieval_agent import RetrievalAgent
from tools import cache

PATIENT = "11111111-1111-1111-1111-111111111111"
OTHER = "22222222-2222-2222-2222-222222222222"
NOW = "2153-12-31 23:59:00"
CONTEXT = {"question": "What was the last weight of the patient during the first hospital visit?",
           "patient_fhir_id": PATIENT, "current_time": NOW,
           "assumptions": f"Patient FHIR ID is {PATIENT}.\nAssume current time is {NOW}."}
QUESTION = f"Question: {CONTEXT['question']}\nContext:\n{CONTEXT['assumptions']}"
ENCOUNTER = {"id": "e1", "identifier": [{"system": "http://mimic/encounter-hosp", "value": "1"}],
             "period": {"start": "2150-01-01T00:00:00"}}


def packet():
    return {"text": "Starting packet\n## Encounter\n" + json.dumps(ENCOUNTER),
            "resources": {"Encounter": [ENCOUNTER]}, "resource_ids": {"Encounter": ["e1"]}}


def observation(rid, when="2150-01-02T08:00:00", patient=PATIENT):
    return {"resourceType": "Observation", "id": rid, "subject": {"reference": f"Patient/{patient}"},
            "effectiveDateTime": when, "valueQuantity": {"value": 70, "unit": "kg"}}


STORE = [observation("o1"), observation("o2", when="2154-01-01T00:00:00"), observation("o3", patient=OTHER)]


def fetch(name="get_resources_by_patient_fhir_id", call_id="call1", **arguments):
    arguments = arguments or {"resource_type": "Observation", "patient_fhir_id": PATIENT}
    return SimpleNamespace(role="assistant", content=None, tool_calls=[SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))])


def reply(content):
    return SimpleNamespace(role="assistant", content=content, tool_calls=None)


def summary(types=("Observation", "Encounter"), encounters=("e1",), limitation=None):
    return reply(json.dumps({"resource_types": list(types), "encounter_ids": list(encounters),
                             "limitation": limitation}))


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        for patcher in (patch("agent.base_agent.setup_api_keys"),
                        patch("agent.retrieval_agent.get_starting_packet", return_value=packet()),
                        patch.object(cache, "CACHE_ENABLED", False),
                        patch("tools.resource_tools.get_fhir_client",
                              return_value=Mock(search_with_pagination=Mock(return_value=STORE)))):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.host = BaseAgent("test-model")
        self.workspace = {}
        self.agent = RetrievalAgent(self.host, CONTEXT, self.workspace)

    def answer(self, *responses, task=None):
        self.host._llm_call = Mock(side_effect=[(r, None, None) for r in responses])
        return self.agent(task or {"context": CONTEXT})

    def test_only_read_only_searches_are_available(self):
        self.assertEqual(self.agent.role.toolbox.names, RETRIEVAL_TOOLS)

    def test_starts_from_the_packet_and_fetches_within_patient_and_current_time(self):
        result = self.answer(fetch(), summary())
        prompt = self.host._llm_call.call_args_list[0].args[0][1]["content"]
        self.assertIn("Starting packet", prompt)
        self.assertIn(CONTEXT["question"], prompt)
        self.assertEqual(self.workspace["Encounter"][0]["resourceType"], "Encounter")
        self.assertEqual([r["id"] for r in self.workspace["Observation"]], ["o1"])
        tool_reply = json.loads(self.host.messages[3]["content"])
        self.assertEqual(tool_reply["excluded"], {"after_current_time": 1, "other_patient": 1})
        self.assertEqual(result, {
            "record_ids": {"Encounter": ["Encounter/e1"], "Observation": ["Observation/o1"]},
            "resource_types": ["Observation", "Encounter"],
            "scope": {"patient_fhir_id": PATIENT, "current_time": NOW, "encounter_ids": ["e1"]},
            "limitation": None})

    def test_point_fetch_of_another_patients_record_is_left_out(self):
        store = Mock(search_with_pagination=Mock(return_value=[observation("o3", patient=OTHER)]))
        with patch("tools.resource_tools.get_fhir_client", return_value=store):
            self.answer(fetch("get_resources_by_resource_id", resource_type="Observation", resource_id="o3"),
                        summary(types=("Encounter",), encounters=()))
        store.search_with_pagination.assert_called_once_with("Observation?_id=o3")
        self.assertEqual(self.workspace.get("Observation", []), [])
        self.assertIn('"other_patient": 1', self.host.messages[3]["content"])

    def test_unknown_encounter_scope_is_dropped_and_reported(self):
        result = self.answer(summary(encounters=("Encounter/e1", "e9"), limitation="Weight unit unclear"))
        self.assertEqual(result["scope"]["encounter_ids"], ["e1"])
        self.assertEqual(result["limitation"],
                         "Weight unit unclear. Encounter scope not among the retrieved encounters: e9")
        self.assertEqual(result["resource_types"], ["Encounter"])

    def test_unstructured_reply_still_returns_the_records(self):
        result = self.answer(reply("I fetched everything."))
        self.assertEqual(result["record_ids"], {"Encounter": ["Encounter/e1"]})
        self.assertEqual(result["resource_types"], ["Encounter"])
        self.assertEqual(result["scope"]["encounter_ids"], [])
        self.assertIn("did not report the question's scope", result["limitation"])

    def test_repair_keeps_the_workspace_and_shows_the_issues(self):
        self.answer(summary(encounters=()))
        issue = {"type": "missing_evidence", "record": None, "field": None,
                 "responsible_role": "retrieval", "action": "fetch Observations"}
        with patch("agent.retrieval_agent.get_starting_packet") as reload:
            result = self.answer(fetch(), summary(), task={"context": CONTEXT, "issues": [issue]})
        reload.assert_not_called()
        prompt = self.host._llm_call.call_args_list[0].args[0][1]["content"]
        self.assertIn("missing_evidence", prompt)
        self.assertIn('{"Encounter": 1}', prompt)
        self.assertEqual(result["record_ids"]["Observation"], ["Observation/o1"])

    def test_missing_patient_or_model_error_is_reported(self):
        no_patient = RetrievalAgent(self.host, {**CONTEXT, "patient_fhir_id": None}, {})
        self.assertIn("patient FHIR ID", no_patient({"context": CONTEXT})["error"])
        self.host._llm_call = Mock(return_value=(None, "provider down", None))
        self.assertEqual(self.agent({"context": CONTEXT}), {"error": "provider down"})


class ControllerIntegrationTests(unittest.TestCase):
    def test_retrieved_records_reach_analysis_through_the_controller(self):
        seen = {}

        class System(FixedController):
            def build_roles(self, context, workspace):
                def analysis(task):
                    seen["records"] = task["records"]
                    seen["workspace"] = {t: [r["id"] for r in items] for t, items in workspace.items()}
                    return {"answer": 70, "evidence": [{"resource_id": "Observation/o1",
                                                        "field_path": "valueQuantity.value", "value": 70}],
                            "limitation": None}
                return {"retrieval": RetrievalAgent(self, context, workspace), "analysis": analysis,
                        "verification": lambda task: {"issues": []}}

        with patch("agent.base_agent.setup_api_keys"), \
                patch("agent.retrieval_agent.get_starting_packet", return_value=packet()), \
                patch.object(cache, "CACHE_ENABLED", False), \
                patch("tools.resource_tools.get_fhir_client",
                      return_value=Mock(search_with_pagination=Mock(return_value=STORE))):
            system = System("test-model")
            system._llm_call = Mock(side_effect=[(fetch(), None, None), (summary(), None, None)])
            output = system.run(QUESTION)
        self.assertIsNone(output["error"])
        self.assertEqual(output["answer"], 70)
        self.assertEqual(seen["records"]["scope"]["encounter_ids"], ["e1"])
        self.assertEqual(seen["workspace"], {"Encounter": ["e1"], "Observation": ["o1"]})
        self.assertEqual([c["name"] for c in output["tool_calls"]], ["get_resources_by_patient_fhir_id"])
        self.assertEqual(output["budget"]["tool_calls_run"], 1)


if __name__ == "__main__":
    unittest.main()
