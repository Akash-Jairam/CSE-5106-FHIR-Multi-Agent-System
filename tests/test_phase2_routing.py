"""P2-17: repairs are routed by issue type (missing evidence to retrieval,
wrong field or calculation to analysis), then re-checked before release."""
import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.analysis_agent import AnalysisAgent
from agent.controller import FixedController
from agent.retrieval_agent import RetrievalAgent
from agent.verification_agent import VerificationAgent
from tools import cache

PATIENT = "11111111-1111-1111-1111-111111111111"
QUESTION = (f"Question: What was the lowest weight of the patient?\nContext:\nPatient FHIR ID is {PATIENT}.\n"
            "Assume current time is 2153-12-31 23:59:00.")
CITATION = {"resource_id": "Observation/o2", "field_path": "valueQuantity.value", "value": 68}


def draft(answer=68, evidence=(CITATION,), limitation=None):
    return {"answer": answer, "evidence": list(evidence), "limitation": limitation}


def issue(kind, role=None, record=None):
    return {"type": kind, "record": record, "field": None, "responsible_role": role, "action": "fix it"}


class Scripted(FixedController):
    """Roles replay scripted results; retrieval leaves Observation/o1 in the workspace."""

    def __init__(self, analysis, verification, retrieval=2):
        with patch("agent.base_agent.setup_api_keys"):
            super().__init__("test-model")
        self.scripts = {"retrieval": [{"record_ids": []}] * retrieval, "analysis": list(analysis),
                        "verification": list(verification)}
        self.calls = {role: 0 for role in self.scripts}

    def build_roles(self, context, workspace):
        workspace["Observation"] = [{"id": "o1"}]

        def handler(role):
            def handle(task):
                self.calls[role] += 1
                return self.scripts[role].pop(0)
            return handle
        return {role: handler(role) for role in self.scripts}


class RoutingTests(unittest.TestCase):
    def repair_role(self, issues, first=None):
        system = Scripted([first or draft(), draft()], [{"issues": issues}, {"issues": []}])
        output = system.run(QUESTION)
        return output["revision"]["repair_role"], system, output

    def test_wrong_field_or_calculation_goes_to_analysis_whatever_the_verifier_says(self):
        for kind in ("time_window", "value_mismatch", "unit", "recompute_mismatch", "wrong_calculation"):
            role, system, output = self.repair_role([issue(kind, role="retrieval")])
            self.assertEqual(role, "analysis", kind)
            self.assertEqual(system.calls["retrieval"], 1)
            self.assertEqual(output["answer"], 68)

    def test_missing_evidence_goes_to_retrieval_whatever_the_verifier_says(self):
        role, system, output = self.repair_role([issue("missing_evidence", role="analysis")])
        self.assertEqual(role, "retrieval")
        self.assertEqual((system.calls["retrieval"], system.calls["analysis"]), (2, 2))
        self.assertEqual(output["answer"], 68)

    def test_abstained_draft_the_verifier_could_answer_goes_to_retrieval(self):
        role, _, output = self.repair_role([issue("recompute_mismatch", role="analysis")],
                                           first=draft(answer=None, evidence=(), limitation="No weights found"))
        self.assertEqual(role, "retrieval")
        self.assertIn("records may be missing", output["revision"]["routing"][0]["reason"])

    def test_unknown_types_route_by_record_then_by_the_verifier(self):
        self.assertEqual(self.repair_role([issue("stale_value", "analysis", "Observation/o9")])[0], "retrieval")
        self.assertEqual(self.repair_role([issue("stale_value", "analysis", "Observation/o1")])[0], "analysis")
        self.assertEqual(self.repair_role([issue("stale_value", "retrieval")])[0], "retrieval")

    def test_issues_needing_both_roles_start_from_retrieval(self):
        role, system, _ = self.repair_role([issue("time_window"), issue("missing_evidence")])
        self.assertEqual(role, "retrieval")
        self.assertEqual(system.calls["analysis"], 2)

    def test_unrepairable_issues_abstain_without_a_repair(self):
        for issues in ([issue("source_check")], [issue("ambiguous_question", role="verification")]):
            system = Scripted([draft()], [{"issues": issues}], retrieval=1)
            output = system.run(QUESTION)
            self.assertIsNone(output["answer"])
            self.assertIsNone(output["revision"]["repair_role"])
            self.assertIn("No role could repair", output["limitation"])
            self.assertEqual(system.calls["analysis"], 1)

    def test_failed_recheck_releases_a_limitation_not_the_answer(self):
        system = Scripted([draft(70), draft(69)], [{"issues": [issue("wrong_calculation")]},
                                                   {"issues": [issue("wrong_calculation")]}])
        output = system.run(QUESTION)
        self.assertIsNone(output["answer"])
        self.assertIn("Evidence could not be verified: wrong_calculation", output["limitation"])
        self.assertEqual(system.calls["verification"], 2)

    def test_routing_decisions_are_recorded(self):
        _, _, output = self.repair_role([issue("time_window", record="Observation/o1"), issue("missing_evidence")])
        self.assertEqual(output["revision"]["routing"], [
            {"type": "time_window", "record": "Observation/o1", "role": "analysis",
             "reason": "wrong field or calculation"},
            {"type": "missing_evidence", "record": None, "role": "retrieval", "reason": "missing evidence"}])
        routed = [m for m in output["revision"]["messages"] if m["kind"] == "routing"]
        self.assertEqual(routed[0]["payload"]["role"], "retrieval")


def observation(rid, value):
    return {"resourceType": "Observation", "id": rid, "subject": {"reference": f"Patient/{PATIENT}"},
            "effectiveDateTime": "2150-01-02T08:00:00", "valueQuantity": {"value": value, "unit": "kg"}}


def reply(content):
    return SimpleNamespace(role="assistant", content=content if isinstance(content, str) else json.dumps(content),
                           tool_calls=None)


def call(name, call_id, **arguments):
    return SimpleNamespace(role="assistant", content=None, tool_calls=[SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))])


class PipelineTests(unittest.TestCase):
    def test_missed_records_are_fetched_on_repair_and_the_answer_released(self):
        store_records = [observation("o1", 70), observation("o2", 68)]
        min_code = "answer = min(r['valueQuantity']['value'] for r in retrieved_resources['Observation'])"
        fetch = {"resource_type": "Observation", "patient_fhir_id": PATIENT}
        scope = {"resource_types": ["Encounter"], "encounter_ids": [], "limitation": None}
        responses = [
            reply(scope),                                                            # retrieval misses weights
            reply(draft(answer=None, evidence=(), limitation="No weight records")),  # analysis abstains
            call("get_resources_by_patient_fhir_id", "v1", **fetch),                 # verifier re-searches
            call("execute_python_code", "v2", code=min_code),
            reply({"recomputed_answer": 68, "issues": []}),                          # only a recompute mismatch
            call("get_resources_by_patient_fhir_id", "r1", **fetch),                 # retrieval repair
            reply({**scope, "resource_types": ["Observation"]}),
            call("execute_python_code", "a1", code=min_code),                        # analysis reruns
            reply({**draft(), "operation": "min weight", "contributing_ids": ["Observation/o1", "Observation/o2"]}),
            reply({"recomputed_answer": 68, "issues": []}),                          # clean re-check
        ]

        class System(FixedController):
            def build_roles(self, context, workspace):
                return {"retrieval": RetrievalAgent(self, context, workspace),
                        "analysis": AnalysisAgent(self, context, workspace),
                        "verification": VerificationAgent(self, context)}

        by_id = {r["id"]: r for r in store_records}
        packet = {"text": "Starting packet", "resources": {}, "resource_ids": {}}
        with patch("agent.base_agent.setup_api_keys"), \
                patch("agent.retrieval_agent.get_starting_packet", return_value=packet), \
                patch.object(cache, "CACHE_ENABLED", False), \
                patch("tools.resource_tools.get_fhir_client",
                      return_value=Mock(search_with_pagination=Mock(side_effect=lambda q: deepcopy(store_records)))), \
                patch("tools.field_check.fetch_from_store",
                      side_effect=lambda refs: {i: deepcopy(by_id[i]) for _, i in refs if i in by_id}):
            system = System("test-model")
            system._llm_call = Mock(side_effect=[(r, None, None) for r in responses])
            output = system.run(QUESTION)

        self.assertIsNone(output["error"])
        self.assertEqual(output["answer"], 68)
        revision = output["revision"]
        self.assertEqual([i["type"] for i in revision["check_issues"]], ["recompute_mismatch"])
        self.assertEqual(revision["repair_role"], "retrieval")
        self.assertEqual(revision["final_check_issues"], [])
        self.assertEqual(sorted(r["id"] for r in output["retrieved_fhir_resources"]["Observation"]), ["o1", "o2"])


if __name__ == "__main__":
    unittest.main()
