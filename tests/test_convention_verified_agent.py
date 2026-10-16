"""V8 team: deterministic convention verification with one routed repair (mocked LLM, real sandbox/source checks)."""
from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch

from agent.convention_verified_agent import ConventionVerifiedAgent, PHASE_CAPS_V8
from tests.test_phase2_verification import PATIENT, NOW, call, reply
from tests.test_query_contract_agent import FixtureMessage
from tools import cache

BARE = "What was the first drug prescribed to the patient?"
QUESTION = f"Question: {BARE}\nContext:\nPatient FHIR ID is {PATIENT}.\nAssume current time is {NOW}."
PACKET = {"resources": {}, "text": "Starting packet is empty; fetch required records."}


def request(rid, name, start):
    return {"resourceType": "MedicationRequest", "id": rid, "subject": {"reference": f"Patient/{PATIENT}"},
            "medicationCodeableConcept": {"coding": [{"display": name}]},
            "dispenseRequest": {"validityPeriod": {"start": start}}}


STORE = [request("rx1", "Heparin", "2150-01-02T08:00:00"), request("rx2", "Insulin", "2150-01-02T08:00:00"),
         request("rx3", "Aspirin", "2150-01-03T08:00:00")]

ONE_CODE = """rows = sorted(retrieved_resources['MedicationRequest'], key=lambda r: r['dispenseRequest']['validityPeriod']['start'])
first = rows[0]
answer = first['medicationCodeableConcept']['coding'][0]['display']
evidence = [{'resource_id': 'MedicationRequest/' + first['id'], 'field_path': 'medicationCodeableConcept.coding[0].display', 'value': answer}]
contributing_ids = ['MedicationRequest/' + first['id']]
operation = 'first prescription by start'
limitation = None
"""
TIE_CODE = """rows = sorted(retrieved_resources['MedicationRequest'], key=lambda r: r['dispenseRequest']['validityPeriod']['start'])
start = rows[0]['dispenseRequest']['validityPeriod']['start']
tied = [r for r in rows if r['dispenseRequest']['validityPeriod']['start'] == start]
answer = sorted(r['medicationCodeableConcept']['coding'][0]['display'] for r in tied)
evidence = [{'resource_id': 'MedicationRequest/' + r['id'], 'field_path': 'medicationCodeableConcept.coding[0].display', 'value': r['medicationCodeableConcept']['coding'][0]['display']} for r in tied]
contributing_ids = ['MedicationRequest/' + r['id'] for r in tied]
operation = 'all prescriptions sharing the first start'
limitation = None
"""
NULL_CODE = "answer = None\nevidence = []\ncontributing_ids = []\noperation = 'abstain'\nlimitation = 'unsure'\n"
CONTRACT = {"requirements": [{"id": "c1", "kind": "ordinal", "quote": "first drug", "span": [13, 23],
                              "expected": "earliest prescription start"}], "unknowns": []}
AUDIT_PASS = {"checks": [{"requirement_id": "c1", "verdict": "PASS", "reason": "sorted by start"}], "repair": None}


def fetch_requests(call_id="fetch1"):
    return call("get_resources_by_patient_fhir_id", call_id, resource_type="MedicationRequest", patient_fhir_id=PATIENT)


def final(code, identifier):
    return call("execute_python_code", identifier, code=code, mode="final")


class ConventionVerifiedAgentTests(unittest.TestCase):
    def setUp(self):
        self.store = deepcopy(STORE)
        self.requests = []
        patches = [patch("agent.base_agent.setup_api_keys"),
                   patch.object(cache, "CACHE_ENABLED", False),
                   patch("agent.single_agent.get_starting_packet", return_value=deepcopy(PACKET)),
                   patch("tools.resource_tools.get_fhir_client", return_value=Mock(
                       search_with_pagination=Mock(side_effect=lambda *_a, **_kw: deepcopy(self.store)))),
                   patch("tools.field_check.fetch_from_store", side_effect=lambda refs: {
                       record["id"]: deepcopy(record) for record in self.store
                       if any(identifier == record["id"] for _, identifier in refs)})]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_agent(self, responses, verifier_model=None):
        pending = iter(responses)

        def llm(**kwargs):
            self.requests.append(deepcopy(kwargs))
            item = FixtureMessage(**vars(next(pending)))
            usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "llm_calls": 1, "cost": 0,
                     "per_call": [{"prompt": 100, "completion": 50, "usage_available": True}]}
            return item, None, usage

        with patch("agent.base_agent.safe_llm_call", side_effect=llm):
            agent = ConventionVerifiedAgent("base-model", verifier_model=verifier_model)
            agent.set_budget(max_total_tokens=200_000, max_tool_calls=30)
            return agent, agent.run(QUESTION)

    def base_responses(self):
        return [reply(json.dumps(CONTRACT)), fetch_requests(), final(ONE_CODE, "primary-final"), reply(json.dumps(AUDIT_PASS))]

    def test_tie_completeness_repair_is_routed_to_analysis_and_released(self):
        agent, output = self.run_agent(self.base_responses() + [final(TIE_CODE, "repair-final")])
        self.assertIsNone(output["error"], output)
        self.assertEqual(output["answer"], ["Heparin", "Insulin"])
        revision = output["revision"]
        self.assertEqual(revision["protocol"], "convention-verified-v8")
        self.assertEqual(revision["draft"]["answer"], "Heparin")
        self.assertEqual(revision["release_policy"], "convention_repair_released")
        self.assertEqual(revision["routing"]["role"], "analysis")
        self.assertEqual([i["check"] for i in revision["convention_issues"]], ["tie_completeness"])
        self.assertEqual(revision["convention_repair"]["status"], "released")
        self.assertEqual(revision["convention_repair"]["remaining_convention_issues"], [])
        self.assertTrue(revision["revised"] and revision["answer_changed"])
        repair_prompt = json.dumps(self.requests[-1]["messages"], default=lambda o: vars(o))
        self.assertIn("tie_completeness -> analysis", repair_prompt)
        self.assertIn("MedicationRequest/rx2", repair_prompt)
        self.assertEqual([p["phase"] for p in revision["phases"]], ["contract", "solver", "audit", "repair"])
        self.assertLessEqual(revision["phases"][-1]["tool_calls"], PHASE_CAPS_V8["repair"][1])

    def test_repair_to_null_or_still_failing_keeps_the_source_valid_draft(self):
        _, output = self.run_agent(self.base_responses() + [final(NULL_CODE, "repair-final")])
        self.assertEqual(output["answer"], "Heparin")
        self.assertEqual(output["revision"]["release_policy"], "source_valid_primary_preserved")
        self.assertEqual(output["revision"]["convention_repair"]["status"], "rejected")
        self.assertFalse(output["revision"]["answer_changed"])

        self.requests = []
        _, output = self.run_agent(self.base_responses() + [final(ONE_CODE, "repair-final")])
        self.assertEqual(output["answer"], "Heparin")
        self.assertIn("still fire", output["revision"]["convention_repair"]["reason"])

    def test_prose_repair_without_executed_final_is_rejected(self):
        _, output = self.run_agent(self.base_responses() + [
            reply('{"answer": ["Heparin", "Insulin"], "evidence": [], "limitation": null}'),
            reply('{"answer": ["Heparin", "Insulin"], "evidence": [], "limitation": null}')])
        self.assertEqual(output["answer"], "Heparin")
        self.assertIn(output["revision"]["convention_repair"]["status"], ("rejected", "failed"))

    def test_clean_draft_runs_no_repair_and_matches_v3_release(self):
        responses = [reply(json.dumps(CONTRACT)), fetch_requests(), final(TIE_CODE, "primary-final"), reply(json.dumps(AUDIT_PASS))]
        agent, output = self.run_agent(responses)
        self.assertEqual(output["answer"], ["Heparin", "Insulin"])
        self.assertEqual(output["revision"]["convention_issues"], [])
        self.assertIsNone(output["revision"]["convention_repair"])
        self.assertEqual(agent.total_usage["llm_calls"], 4)

    def null_release_responses(self, repair_code):
        # primary final cites a value its record does not hold; the solver's one revision repeats it
        bad = ONE_CODE.replace("'value': answer}]", "'value': 'Warfarin'}]")
        return [reply(json.dumps(CONTRACT)), fetch_requests(), final(bad, "primary-final"),
                final(bad, "primary-revise"), final(repair_code, "repair-final")]

    def test_source_failed_final_gets_one_routed_repair_instead_of_abstaining(self):
        _, output = self.run_agent(self.null_release_responses(TIE_CODE))
        revision = output["revision"]
        self.assertEqual(output["answer"], ["Heparin", "Insulin"])
        self.assertEqual(revision["release_policy"], "null_release_repair_released")
        self.assertEqual(revision["routing"]["role"], "analysis")
        self.assertEqual([i["check"] for i in revision["convention_issues"]], ["source_check", "tie_completeness"])
        self.assertEqual(revision["convention_repair"]["status"], "released")
        self.assertEqual(revision["final_check_issues"], [])

    def test_repair_of_a_null_release_that_fails_the_source_check_stays_null(self):
        bad = ONE_CODE.replace("'value': answer}]", "'value': 'Warfarin'}]")
        _, output = self.run_agent(self.null_release_responses(bad))
        self.assertIsNone(output["answer"])
        self.assertEqual(output["revision"]["release_policy"], "primary_failed_execution_or_source_contract")
        self.assertEqual(output["revision"]["convention_repair"]["status"], "rejected")

    def test_heterogeneous_verifier_runs_only_the_audit_on_the_other_model(self):
        _, output = self.run_agent(self.base_responses() + [final(TIE_CODE, "repair-final")], verifier_model="other-family")
        models = [r["model"] for r in self.requests]
        self.assertEqual(models, ["base-model", "base-model", "base-model", "other-family", "base-model"])
        self.assertEqual(output["revision"]["verifier_model"], "other-family")


if __name__ == "__main__":
    unittest.main()
