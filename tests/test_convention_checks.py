"""V8 convention checks: deterministic, gold-free, routed to the responsible role."""
import unittest

from agent.controller import FixedController
from agent.convention_checks import convention_issues, repair_message

PATIENT = "p1"


def request(rid, name, start, route="PO", encounter="Encounter/h1"):
    return {"resourceType": "MedicationRequest", "id": rid, "subject": {"reference": f"Patient/{PATIENT}"},
            "encounter": {"reference": encounter}, "authoredOn": start[:10],
            "medicationCodeableConcept": {"coding": [{"display": name}]},
            "dosageInstruction": [{"route": {"coding": [{"display": route}]}}],
            "dispenseRequest": {"validityPeriod": {"start": start}}}


def draft(answer, *ids, field="medicationCodeableConcept.coding[0].display", value="x"):
    return {"answer": answer, "contributing_ids": list(ids), "limitation": None,
            "evidence": [{"resource_id": rid, "field_path": field, "value": value} for rid in ids]}


def route(issues):
    router = FixedController.__new__(FixedController)
    router._workspace = {}
    return router.route(issues, {"answer": 1})[0]


class TieCompletenessTest(unittest.TestCase):
    workspace = {"MedicationRequest": [request("rx1", "Heparin", "2150-01-02T08:00:00"),
                                       request("rx2", "Insulin", "2150-01-02T08:00:00"),
                                       request("rx3", "Aspirin", "2150-01-03T08:00:00")]}

    def test_single_value_at_a_tied_start_time_is_flagged_for_analysis(self):
        issues = convention_issues("What was the first drug prescribed to the patient?",
                                   draft("Heparin", "MedicationRequest/rx1"), self.workspace)
        self.assertEqual([i["check"] for i in issues], ["tie_completeness"])
        self.assertEqual(issues[0]["type"], "wrong_filter")
        self.assertEqual(route(issues), "analysis")
        self.assertIn("2 distinct drugs", issues[0]["action"])

    def test_complete_tie_set_passes(self):
        issues = convention_issues("What was the first drug prescribed to the patient?",
                                   draft(["Heparin", "insulin"], "MedicationRequest/rx1", "MedicationRequest/rx2"),
                                   self.workspace)
        self.assertEqual(issues, [])

    def test_untied_selection_and_when_questions_do_not_fire(self):
        self.assertEqual(convention_issues("What was the last drug prescribed to the patient?",
                                           draft("Aspirin", "MedicationRequest/rx3"), self.workspace), [])
        self.assertEqual(convention_issues("When was the first drug prescribed?",
                                           draft("2150-01-02 08:00:00", "MedicationRequest/rx1"), self.workspace), [])

    def test_route_question_ties_only_within_the_cited_route(self):
        workspace = {"MedicationRequest": [request("rx1", "Heparin", "2150-01-02T08:00:00", route="IV"),
                                           request("rx2", "Insulin", "2150-01-02T08:00:00", route="SC")]}
        self.assertEqual(convention_issues("What was the first drug prescribed via the iv route?",
                                           draft("Heparin", "MedicationRequest/rx1"), workspace), [])

    def test_dose_count_and_time_answers_are_not_tie_selections(self):
        for question, answer in (("What was the last dose of heparin prescribed?", 5000),
                                 ("How much heparin was prescribed the last time?", "5000"),
                                 ("And what was the last time the patient was prescribed a drug?", "2150-01-02 08:00:00")):
            self.assertEqual(convention_issues(question, draft(answer, "MedicationRequest/rx1"), self.workspace), [],
                             question)

    def test_mixture_components_count_as_distinct_drugs(self):
        workspace = {"MedicationRequest": [
            {**request("rx1", "", "2150-01-02T08:00:00"), "medicationReference": {"reference": "Medication/mix"}},
            request("rx2", "Insulin", "2150-01-02T08:00:00")],
            "Medication": [{"resourceType": "Medication", "id": "mix", "ingredient": [
                {"itemReference": {"reference": "Medication/a"}}, {"itemReference": {"reference": "Medication/b"}}]},
                {"resourceType": "Medication", "id": "a", "code": {"coding": [{"display": "Dextrose"}]}},
                {"resourceType": "Medication", "id": "b", "code": {"coding": [{"display": "Potassium"}]}}]}
        issues = convention_issues("What was the first drug prescribed?", draft(["Insulin", "Dextrose"],
                                   "MedicationRequest/rx1", "MedicationRequest/rx2"), workspace)
        self.assertEqual(len(issues), 1)
        self.assertIn("3 distinct drugs", issues[0]["action"])


class MicrobiologyChecksTest(unittest.TestCase):
    test = {"resourceType": "Observation", "id": "t1", "subject": {"reference": f"Patient/{PATIENT}"},
            "code": {"coding": [{"system": "x/mimic-microbiology-test", "display": "BLOOD CULTURE"}]},
            "effectiveDateTime": "2150-01-02T08:30:00", "specimen": {"reference": "Specimen/s1"},
            "hasMember": [{"reference": "Observation/org1"}]}
    specimen = {"resourceType": "Specimen", "id": "s1", "type": {"coding": [{"display": "BLOOD CULTURE"}]}}

    def test_specimen_question_without_specimen_resources_routes_to_retrieval(self):
        issues = convention_issues("What was the name of the specimen of the first microbiology test?",
                                   draft("Blood culture", "Observation/t1", field="code.coding[0].display"),
                                   {"Observation": [self.test]})
        self.assertEqual(issues[0]["type"], "missing_evidence")
        self.assertEqual(issues[0]["record"], "Specimen/s1")
        self.assertEqual(route(issues), "retrieval")

    def test_specimen_answer_matching_specimen_type_passes(self):
        issues = convention_issues("What was the name of the specimen of the first microbiology test?",
                                   draft("blood culture", "Observation/t1", "Specimen/s1", field="type.coding[0].display"),
                                   {"Observation": [self.test], "Specimen": [self.specimen]})
        self.assertEqual(issues, [])

    def test_unretrieved_organism_children_are_missing_evidence(self):
        issues = convention_issues("What organisms were found in the first blood culture?",
                                   draft(["BLOOD CULTURE"], "Observation/t1", field="code.coding[0].display"),
                                   {"Observation": [self.test]})
        self.assertEqual([(i["check"], i["type"], i["record"]) for i in issues],
                         [("organism_children", "missing_evidence", "Observation/org1")])

    def test_when_answer_must_be_the_parent_test_time(self):
        child = {"resourceType": "Observation", "id": "org1", "subject": {"reference": f"Patient/{PATIENT}"},
                 "code": {"coding": [{"system": "x/mimic-microbiology-organism", "display": "E. COLI"}]},
                 "effectiveDateTime": "2150-01-02T10:15:00", "derivedFrom": [{"reference": "Observation/t1"}]}
        workspace = {"Observation": [self.test, child]}
        issues = convention_issues("When was the first microbiology test performed?",
                                   draft("2150-01-02 10:15:00", "Observation/org1", field="effectiveDateTime"), workspace)
        self.assertEqual([i["check"] for i in issues], ["timestamp_field"])
        self.assertEqual(issues[0]["type"], "wrong_field")
        self.assertEqual(convention_issues("When was the first microbiology test performed?",
                                           draft("2150-01-02 08:30:00", "Observation/t1", field="effectiveDateTime"),
                                           workspace), [])


class CountAndCodedFieldChecksTest(unittest.TestCase):
    def test_distinct_drug_count_requires_a_complete_patient_wide_search(self):
        workspace = {"MedicationRequest": [request(f"rx{i}", f"d{i}", "2150-01-02T08:00:00") for i in range(5)]}
        question = "How many different drugs were prescribed to the patient this year?"
        issues = convention_issues(question, draft(5, *[f"MedicationRequest/rx{i}" for i in range(5)]), workspace)
        self.assertEqual(issues[0]["type"], "missing_evidence")
        self.assertEqual(route(issues), "retrieval")
        self.assertEqual(convention_issues(question, draft(5, "MedicationRequest/rx0"), workspace,
                                           complete_types={"MedicationRequest"}), [])

    def test_prescription_when_answer_is_the_validity_start(self):
        workspace = {"MedicationRequest": [request("rx1", "Heparin", "2150-01-02T08:00:00")]}
        bad = convention_issues("When was the patient first prescribed heparin?",
                                draft("2150-01-02 00:00:00", "MedicationRequest/rx1", field="authoredOn"), workspace)
        self.assertEqual([i["check"] for i in bad], ["timestamp_field"])
        good = convention_issues("When was the patient first prescribed heparin?",
                                 draft("2150-01-02 08:00:00", "MedicationRequest/rx1",
                                       field="dispenseRequest.validityPeriod.start"), workspace)
        self.assertEqual(good, [])

    def test_distinct_count_gate_covers_count_and_number_phrasings(self):
        workspace = {"MedicationRequest": [request("rx0", "d0", "2150-01-02T08:00:00")]}
        for question in ("Tell me the count of distinct drugs the patient was prescribed.",
                         "Find out the number of unique medicines prescribed to the patient."):
            issues = convention_issues(question, draft(1, "MedicationRequest/rx0"), workspace)
            self.assertEqual([i["check"] for i in issues], ["distinct_count_scope"], question)

    def test_prescription_without_validity_start_is_not_timestamp_checked(self):
        rx = request("rx1", "Heparin", "2150-01-02T08:00:00")
        rx["dispenseRequest"] = {}
        self.assertEqual(convention_issues("When was the patient first prescribed heparin?",
                                           draft("2150-01-02 13:00:00", "MedicationRequest/rx1"),
                                           {"MedicationRequest": [rx]}), [])

    def test_admission_type_is_not_checked(self):
        encounter = {"resourceType": "Encounter", "id": "h1", "class": {"code": "EMER", "display": "emergency"},
                     "priority": {"coding": [{"code": "UR", "display": "urgent"}]}}
        self.assertEqual(convention_issues("What was the admission type of the last hospital visit?",
                                           draft("emergency", "Encounter/h1", field="class.display"),
                                           {"Encounter": [encounter]}), [])

    def test_null_drafts_and_unrelated_questions_never_fire(self):
        workspace = {"MedicationRequest": [request("rx1", "Heparin", "2150-01-02T08:00:00"),
                                           request("rx2", "Insulin", "2150-01-02T08:00:00")]}
        self.assertEqual(convention_issues("What was the first drug prescribed?", draft(None), workspace), [])
        self.assertEqual(convention_issues("What was the lowest weight of the patient?", draft(70.5, "Observation/o1"),
                                           {"Observation": []}), [])

    def test_repair_message_names_check_role_and_record(self):
        issues = convention_issues("What was the first drug prescribed to the patient?", draft("Heparin", "MedicationRequest/rx1"),
                                   {"MedicationRequest": [request("rx1", "Heparin", "2150-01-02T08:00:00"),
                                                          request("rx2", "Insulin", "2150-01-02T08:00:00")]})
        text = repair_message(issues)
        self.assertIn("[tie_completeness -> analysis]", text)
        self.assertIn("MedicationRequest/rx2", text)
        self.assertIn("mode=final", text)


if __name__ == "__main__":
    unittest.main()
