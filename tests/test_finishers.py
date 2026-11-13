"""V10 finishers: routing, anchors, replacement conditions, evidence verification, no nulls."""
import unittest
from copy import deepcopy
from unittest import mock

from agent import finishers
from agent.finishers import Fetcher, count_window, run_finisher

PATIENT = "p1"
NOW = "2150-12-31 23:59:00"
NAME = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-medication-name"
TEST = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-microbiology-test"
ORG = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-microbiology-organism"


def medication(mid, name):
    return {"resourceType": "Medication", "id": mid, "identifier": [{"system": NAME, "value": name}]}


def request(rid, mid, start, route="IV", authored=None, encounter="Encounter/h1"):
    record = {"resourceType": "MedicationRequest", "id": rid, "subject": {"reference": f"Patient/{PATIENT}"},
              "encounter": {"reference": encounter}, "medicationReference": {"reference": f"Medication/{mid}"},
              "authoredOn": authored or start, "dosageInstruction": [{"route": {"coding": [{"code": route}]}}],
              "dispenseRequest": {"validityPeriod": {"start": start}}}
    if encounter is None:
        del record["encounter"]
    return record


def observation(oid, code, system, when, **links):
    return {"resourceType": "Observation", "id": oid, "subject": {"reference": f"Patient/{PATIENT}"},
            "code": {"coding": [{"system": system, "display": code}]}, "effectiveDateTime": when, **links}


STORE = {
    "MedicationRequest": [
        request("rx1", "m1", "2150-01-02T08:00:00-05:00", authored="2150-01-02T06:12:00-05:00"),
        request("rx2", "m2", "2150-01-02T08:00:00-05:00"),
        request("rx3", "m3", "2150-01-03T08:00:00-05:00", route="PO"),
        request("rx4", "m1", "2150-02-03T08:00:00-05:00", route="PO"),
        request("rx5", "m4", "2150-01-02T08:00:00-05:00", route="PO"),
        request("rx9", "m5", "2151-06-01T08:00:00-05:00"),  # after the current time
    ],
    "Medication": [medication("m1", "Heparin"), medication("m2", "Insulin"), medication("m3", "Aspirin"),
                   medication("m4", "Senna"), medication("m5", "Future drug")],
    "Observation": [
        observation("o1", "BLOOD CULTURE", TEST, "2150-01-05T10:30:00-05:00", specimen={"reference": "Specimen/s1"},
                    hasMember=[{"reference": "Observation/g1"}, {"reference": "Observation/g2"}]),
        observation("g1", "STAPH AUREUS COAG +", ORG, "2150-01-05T10:30:00-05:00",
                    derivedFrom=[{"reference": "Observation/o1"}]),
        observation("g2", "ESCHERICHIA COLI", ORG, "2150-01-06T09:00:00-05:00",
                    derivedFrom=[{"reference": "Observation/o1"}]),
    ],
    "Specimen": [{"resourceType": "Specimen", "id": "s1", "subject": {"reference": f"Patient/{PATIENT}"},
                  "type": {"coding": [{"display": "BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)"}]},
                  "collection": {"collectedDateTime": "2150-01-05T10:30:00-05:00"}}],
}


def fake_tool(store, calls):
    def get_resources_by_patient_fhir_id(resource_type, patient_fhir_id):
        calls.append(resource_type)
        records = deepcopy(store.get(resource_type, []))
        if resource_type == "Medication":
            return {"Medication": records, "retrieval_metadata": {"complete": True}}
        return {resource_type: records}
    return lambda name: get_resources_by_patient_fhir_id


def candidate(answer, *ids):
    return {"answer": answer, "contributing_ids": list(ids), "limitation": None,
            "evidence": [{"resource_id": rid, "field_path": "id", "value": rid.split("/")[1]} for rid in ids]}


class FinisherTest(unittest.TestCase):
    def setUp(self):
        self.calls, self.charged = [], []
        patcher = mock.patch("agent.framework.get_tool", fake_tool(STORE, self.calls))
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_one(self, question, cand, budget=True):
        fetcher = Fetcher(lambda: self.charged.append(1) is None and budget, PATIENT, NOW)
        context = {"question": question, "patient_fhir_id": PATIENT, "current_time": NOW}
        return run_finisher(context, cand, fetcher)

    # T051 / T052 -----------------------------------------------------------
    def test_tie_member_is_replaced_by_the_complete_verified_tie_set(self):
        record = self.run_one("What was the name of the drug that patient 10001 was first prescribed?",
                              candidate("Heparin", "MedicationRequest/rx1"))
        self.assertEqual(record["template"], "T051")
        self.assertEqual(record["finisher"], "replaced")
        self.assertEqual(sorted(record["result"]), ["Heparin", "Insulin", "Senna"])
        self.assertEqual(self.calls, ["MedicationRequest", "Medication"])
        self.assertEqual(len(self.charged), 2)
        paths = {c["field_path"] for c in record["evidence"]}
        self.assertIn("dispenseRequest.validityPeriod.start", paths)
        self.assertIn("identifier[0].value", paths)

    def test_route_question_ties_only_within_the_cited_route(self):
        record = self.run_one("What was the first drug that patient 10001 was prescribed via the iv route?",
                              candidate("Heparin", "MedicationRequest/rx1"))
        self.assertEqual(record["template"], "T052")
        self.assertEqual(sorted(record["result"]), ["Heparin", "Insulin"])

    def test_untied_selection_and_non_member_answers_keep_the_candidate(self):
        record = self.run_one("What was the name of the drug that patient 10001 was last prescribed?",
                              candidate("Aspirin", "MedicationRequest/rx3"))
        self.assertEqual((record["finisher"], record["reason"]), ("confirmed_primary", "result_equals_primary"))
        record = self.run_one("What was the name of the drug that patient 10001 was first prescribed?",
                              candidate("Warfarin", "MedicationRequest/rx1"))
        self.assertEqual((record["finisher"], record["reason"]), ("confirmed_primary", "condition_not_met"))
        self.assertIsNone(record["result"])

    # T054 ------------------------------------------------------------------
    def test_specimen_follows_the_cited_test_to_specimen_type(self):
        record = self.run_one("What was the name of the specimen that patient 10001 was first tested?",
                              candidate("BLOOD CULTURE", "Observation/o1"))
        self.assertEqual(record["template"], "T054")
        self.assertEqual(record["finisher"], "replaced")
        self.assertEqual(record["result"], "BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)")
        self.assertEqual(self.calls, ["Observation", "Specimen"])

    def test_specimen_answer_citing_the_specimen_is_kept(self):
        record = self.run_one("What was the name of the specimen that patient 10001 was first tested?",
                              candidate("Urine", "Observation/o1", "Specimen/s1"))
        self.assertEqual((record["finisher"], record["reason"]), ("confirmed_primary", "condition_not_met"))

    # T055 ------------------------------------------------------------------
    def test_organisms_are_the_cited_tests_hasmember_children(self):
        question = "What was the organism found in the first blood culture test of patient 10001?"
        record = self.run_one(question, candidate("STAPH AUREUS COAG +", "Observation/g1"))
        self.assertEqual(record["template"], "T055")
        self.assertEqual(record["finisher"], "replaced")
        self.assertEqual(sorted(record["result"]), ["ESCHERICHIA COLI", "STAPH AUREUS COAG +"])
        record = self.run_one(question, candidate(["ESCHERICHIA COLI", "STAPH AUREUS COAG +"], "Observation/o1"))
        self.assertEqual(record["reason"], "result_equals_primary")

    # T001 ------------------------------------------------------------------
    def test_distinct_count_uses_the_complete_closure_within_the_window(self):
        question = "How many distinct drugs were prescribed to patient 10001 in 01/2150?"
        record = self.run_one(question, candidate(1, "MedicationRequest/rx1"))
        self.assertEqual(record["template"], "T001")
        self.assertEqual(record["finisher"], "replaced")
        self.assertEqual(record["result"], 4)  # Heparin, Insulin, Aspirin, Senna; February and 2151 excluded
        self.assertEqual(record["detail"]["window_source"], "question_window")

    def test_unrestricted_count_and_unsupported_scope(self):
        record = self.run_one("What is the number of unique drugs patient 10001 was prescribed?",
                              candidate(4, "MedicationRequest/rx1"))
        self.assertEqual(record["reason"], "result_equals_primary")
        record = self.run_one("What is the number of unique drugs patient 10001 was prescribed?",
                              candidate(5, "MedicationRequest/rx1"))
        self.assertEqual(record["result"], 4)  # four distinct names; the 2151 request is out of scope
        self.calls.clear()
        record = self.run_one("How many distinct drugs were prescribed to patient 10001 during their first "
                              "hospital visit?", candidate(3, "MedicationRequest/rx1"))
        self.assertEqual(record["reason"], "encounter_scope_unsupported")
        self.assertEqual(self.calls, [])

    def test_count_window_sources(self):
        self.assertEqual(count_window("How many distinct drugs in 2150?", NOW)[2], "question_window")
        self.assertEqual(count_window("How many distinct drugs?", NOW)[2], "question_unrestricted")
        with self.assertRaises(finishers.FinisherStop):
            count_window("How many distinct drugs?", None)

    def test_undated_prescriptions_that_would_change_a_windowed_count_skip_it(self):
        store = deepcopy(STORE)
        store["MedicationRequest"].append(request("rx6", "m5", "2150-01-20T08:00:00-05:00"))
        del store["MedicationRequest"][-1]["dispenseRequest"]  # a new name, dated only by authoredOn
        store["MedicationRequest"].append(request("rx7", "m1", "2150-01-21T08:00:00-05:00"))
        del store["MedicationRequest"][-1]["dispenseRequest"]  # an undated name already counted
        question = "How many distinct drugs were prescribed to patient 10001 in 01/2150?"
        with mock.patch("agent.framework.get_tool", fake_tool(store, self.calls)):
            record = self.run_one(question, candidate(1, "MedicationRequest/rx1"))
        self.assertEqual(record["reason"], "undated_prescriptions_in_window")
        del store["MedicationRequest"][-2]
        with mock.patch("agent.framework.get_tool", fake_tool(store, self.calls)):
            record = self.run_one(question, candidate(1, "MedicationRequest/rx1"))
        self.assertEqual(record["result"], 4)

    # T064 / T073 -----------------------------------------------------------
    def test_date_only_or_child_time_becomes_the_parent_test_time(self):
        question = "When was the first microbiology test of patient 10001?"
        record = self.run_one(question, candidate("2150-01-05", "Observation/o1"))
        self.assertEqual(record["template"], "T064")
        self.assertEqual((record["finisher"], record["result"]), ("replaced", "2150-01-05T10:30:00-05:00"))
        record = self.run_one(question, candidate("2150-01-06T09:00:00-05:00", "Observation/g2"))
        self.assertEqual(record["detail"]["source"], "child_finding")
        self.assertEqual(record["finisher"], "replaced")
        record = self.run_one(question, candidate("2150-01-05 10:30:00", "Observation/o1"))
        self.assertEqual(record["reason"], "result_equals_primary")
        record = self.run_one(question, candidate("2150-03-01 10:30:00", "Observation/o1"))
        self.assertEqual(record["reason"], "answer_source_unknown")

    def test_authored_on_becomes_the_validity_start(self):
        question = "When was the first time that patient 10001 was prescribed a medication via the iv route?"
        record = self.run_one(question, candidate("2150-01-02T06:12:00-05:00", "MedicationRequest/rx1"))
        self.assertEqual(record["template"], "T073")
        self.assertEqual((record["finisher"], record["result"]), ("replaced", "2150-01-02T08:00:00-05:00"))
        self.assertEqual(record["detail"]["source"], "authoredOn")

    # Shared rules ----------------------------------------------------------
    def test_no_anchor_and_unrouted_questions_spend_nothing(self):
        record = self.run_one("What was the name of the drug that patient 10001 was first prescribed?",
                              candidate("Heparin", "Observation/o1"))
        self.assertEqual(record["finisher"], "skipped_no_anchor")
        record = self.run_one("What was the dose of heparin that patient 10001 was first prescribed?",
                              candidate("5000", "MedicationRequest/rx1"))
        self.assertEqual(record["finisher"], "none")
        self.assertEqual((self.calls, self.charged), ([], []))

    def test_exhausted_tool_budget_keeps_the_candidate(self):
        record = self.run_one("What was the name of the drug that patient 10001 was first prescribed?",
                              candidate("Heparin", "MedicationRequest/rx1"), budget=False)
        self.assertEqual((record["finisher"], record["reason"]), ("confirmed_primary", "tool_budget"))
        self.assertEqual(self.calls, [])

    def test_unverified_evidence_is_never_released(self):
        with mock.patch.object(finishers, "check_citation", return_value="value_mismatch"):
            record = self.run_one("What was the name of the drug that patient 10001 was first prescribed?",
                                  candidate("Heparin", "MedicationRequest/rx1"))
        self.assertEqual((record["finisher"], record["reason"]), ("confirmed_primary", "evidence_unverified"))

    def test_fetch_failure_and_anchor_outside_scope(self):
        def broken(name):
            def call(resource_type, patient_fhir_id):
                return {"error": "store unavailable"}
            return call
        with mock.patch("agent.framework.get_tool", broken):
            record = self.run_one("What was the name of the drug that patient 10001 was first prescribed?",
                                  candidate("Heparin", "MedicationRequest/rx1"))
        self.assertEqual(record["reason"], "fetch_failed")
        record = self.run_one("What was the name of the drug that patient 10001 was first prescribed?",
                              candidate("Future drug", "MedicationRequest/rx9"))
        self.assertEqual(record["reason"], "anchor_out_of_scope")

    def test_fetch_limit_is_two(self):
        fetcher = Fetcher(lambda: True, PATIENT, NOW)
        fetcher.patient("MedicationRequest")
        fetcher.patient("Medication")
        with self.assertRaises(finishers.FinisherStop):
            fetcher.patient("Observation")
        self.assertTrue(fetcher.complete("MedicationRequest"))


if __name__ == "__main__":
    unittest.main()
