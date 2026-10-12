"""P2-16: the shared field check flags evidence outside the question's time
window, using the dataset's SQL semantics for each phrasing."""
import unittest

from agent.verification_agent import field_check_issues
from tools.field_check import check_fields, question_encounter_scope, question_time_window, resource_event_time

PATIENT = "11111111-1111-1111-1111-111111111111"


def window(question, now="2137-12-31 23:59:00"):
    found = question_time_window(question, now)
    return found and found[:2]


class TimeWindowParserTests(unittest.TestCase):
    def test_absolute_windows(self):
        self.assertEqual(window("Any drug prescribed since 05/2137?"), ("2137-05-01 00:00:00", None))
        self.assertEqual(window("Total output since 08/11/2137."), ("2137-08-11 00:00:00", None))
        self.assertEqual(window("Was heparin prescribed since 2137 to patient 10036156?"),
                         ("2137-01-01 00:00:00", None))
        self.assertEqual(window("Any lab testing in 11/2136?"), ("2136-11-01 00:00:00", "2136-12-01 00:00:00"))
        self.assertEqual(window("Any catheterization procedure in 2142?"),
                         ("2142-01-01 00:00:00", "2143-01-01 00:00:00"))
        self.assertEqual(window("Total output on 12/06/2183?"), ("2183-12-06 00:00:00", "2183-12-07 00:00:00"))

    def test_relative_windows_count_back_from_midnight_like_sqlite(self):
        self.assertEqual(window("Weight since 2 months ago?", "2110-12-31 23:59:00"), ("2110-10-31 00:00:00", None))
        # SQLite rolls 11-31 over to 12-01
        self.assertEqual(window("Weight since 1 month ago?", "2110-12-31 23:59:00"), ("2110-12-01 00:00:00", None))
        self.assertEqual(window("Been to the er since 1 year ago?", "2131-12-31 23:59:00"),
                         ("2130-12-31 00:00:00", None))
        self.assertEqual(window("Max systolic blood pressure since 92 days ago?"), ("2137-09-30 00:00:00", None))

    def test_calendar_windows_relative_to_the_current_time(self):
        self.assertEqual(window("Milk of magnesia prescribed in 09/this year?", "2128-12-31 23:59:00"),
                         ("2128-09-01 00:00:00", "2128-10-01 00:00:00"))
        self.assertEqual(window("Mean blood pressure less than 79.0 on 03/07/this year?", "2125-12-31 23:59:00"),
                         ("2125-03-07 00:00:00", "2125-03-08 00:00:00"))
        self.assertEqual(window("Any drugs in 04/last year?", "2182-12-31 23:59:00"),
                         ("2181-04-01 00:00:00", "2181-05-01 00:00:00"))
        self.assertEqual(window("Hospital visits during this year."), ("2137-01-01 00:00:00", "2138-01-01 00:00:00"))
        self.assertEqual(window("On this month/21, the minimum respiratory rate?", "2116-12-31 23:59:00"),
                         ("2116-12-21 00:00:00", "2116-12-22 00:00:00"))
        self.assertEqual(window("Unique medicines during the last month."), ("2137-11-01 00:00:00", "2137-12-01 00:00:00"))
        self.assertEqual(window("Heart rate on last month/14?", "2138-01-15 10:00:00"),
                         ("2137-12-14 00:00:00", "2137-12-15 00:00:00"))
        self.assertEqual(window("Maximum heart rate yesterday?"), ("2137-12-30 00:00:00", "2137-12-31 00:00:00"))
        self.assertEqual(window("Respiratory rate below 23.0 today?"), ("2137-12-31 00:00:00", "2138-01-01 00:00:00"))

    def test_phrases_that_are_not_windows(self):
        for question in ["How many days since patient 10021487's hospital admission?",
                         "Hours since the last time on this hospital visit the patient had a creatinine test?",
                         "Days since last time that patient 10004733 stayed in careunit neurology?",
                         "Weight during the last hospital visit?",
                         "Was the volume ever greater than 1000 in total?"]:
            self.assertIsNone(question_time_window(question, "2137-12-31 23:59:00"), question)

    def test_relative_window_needs_a_current_time_and_dates_must_exist(self):
        self.assertIsNone(question_time_window("Any drugs this year?"))
        self.assertEqual(question_time_window("Any drugs in 03/2196?")[:2], ("2196-03-01 00:00:00", "2196-04-01 00:00:00"))
        self.assertIsNone(question_time_window("Output on 02/30/2183?"))
        self.assertIsNone(question_time_window("Any drugs this year?", "not a time"))

    def test_current_time_accepts_fhir_offsets_dates_and_minute_precision(self):
        for now in ["2137-12-31T23:59:00-05:00", "2137-12-31 23:59", "2137-12-31"]:
            self.assertEqual(window("Minimum weight yesterday?", now),
                             ("2137-12-30 00:00:00", "2137-12-31 00:00:00"))
        self.assertIsNone(window("Weight since 999999999999 days ago?"))


def observation(rid, when):
    return {"resourceType": "Observation", "id": rid, "subject": {"reference": f"Patient/{PATIENT}"},
            "effectiveDateTime": when, "valueQuantity": {"value": 7, "unit": "kg"}}


def cite(rid, path="valueQuantity.value", value=7, rtype="Observation"):
    return {"resource_id": f"{rtype}/{rid}", "field_path": path, "value": value}


class TimeWindowCheckTests(unittest.TestCase):
    QUESTION = "What was the minimum weight in 11/2136?"

    def check(self, citations, resources, question=QUESTION, now="2136-12-31 23:59:00"):
        result = check_fields(citations, question=question, current_time=now, resources=resources,
                              patient_fhir_id=PATIENT)
        return [w for r in result["results"] for w in r["warnings"]]

    def test_window_start_is_inclusive_and_end_exclusive_on_wall_clock_time(self):
        resources = {"in": observation("in", "2136-11-01T00:00:00-05:00"),
                     "last": observation("last", "2136-11-30T23:59:59-05:00"),
                     "out": observation("out", "2136-12-01T00:00:00-05:00"),
                     "before": observation("before", "2136-10-31T23:59:00-04:00")}
        warnings = self.check([cite("in"), cite("last"), cite("out"), cite("before")], resources)
        self.assertEqual(len(warnings), 2)
        self.assertIn('dated 2136-12-01 00:00:00, outside the question\'s window "in 11/2136" '
                      "(2136-11-01 00:00:00 to 2136-12-01 00:00:00)", warnings[0])
        self.assertIn("2136-10-31 23:59:00", warnings[1])

    def test_since_window_runs_to_the_current_time(self):
        resources = {"old": observation("old", "2136-04-30T12:00:00"), "new": observation("new", "2136-06-01T12:00:00")}
        warnings = self.check([cite("old"), cite("new")], resources, question="Weight since 05/2136?")
        self.assertEqual(len(warnings), 1)
        self.assertIn("to the current time", warnings[0])

    def test_condition_is_dated_by_its_encounter(self):
        resources = {"e1": {"resourceType": "Encounter", "id": "e1", "subject": {"reference": f"Patient/{PATIENT}"},
                            "period": {"start": "2135-03-01T10:00:00"}},
                     "c1": {"resourceType": "Condition", "id": "c1", "subject": {"reference": f"Patient/{PATIENT}"},
                            "encounter": {"reference": "Encounter/e1"}, "code": {"text": "Sepsis"}}}
        warnings = self.check([cite("c1", "code.text", "Sepsis", "Condition")], resources,
                              question="Was the patient diagnosed with sepsis in 2136?")
        self.assertEqual(len(warnings), 1)
        self.assertTrue(warnings[0].startswith("time_window: the resource is dated 2135-03-01 10:00:00"))

    def test_prescription_without_a_start_time_is_not_dated_by_its_order(self):
        order_only = {"resourceType": "MedicationRequest", "id": "m1", "subject": {"reference": f"Patient/{PATIENT}"},
                      "authoredOn": "2136-10-27T07:59:59-05:00"}
        started = {**order_only, "id": "m2", "dispenseRequest": {"validityPeriod": {"start": "2136-10-27T08:00:00"}}}
        warnings = self.check([cite("m1", "id", "m1", "MedicationRequest"), cite("m2", "id", "m2", "MedicationRequest")],
                              {"m1": order_only, "m2": started}, question="Any drug prescribed in 11/2136?")
        self.assertEqual(len(warnings), 2)
        self.assertIn("no event timestamp", warnings[0])
        self.assertIn("2136-10-27 08:00:00", warnings[1])

    def test_supporting_encounter_metadata_is_not_a_measurement_window_event(self):
        resources = {"o1": observation("o1", "2136-11-01T01:00:00"),
                     "e1": {"resourceType": "Encounter", "id": "e1",
                            "subject": {"reference": f"Patient/{PATIENT}"},
                            "period": {"start": "2136-10-31T23:00:00"}}}
        self.assertEqual(self.check([cite("o1"), cite("e1", "period.start", "2136-10-31T23:00:00", "Encounter")],
                                    resources), [])

    def test_counted_encounter_is_still_checked_when_other_evidence_is_present(self):
        resources = {"o1": observation("o1", "2136-11-01T01:00:00"),
                     "e1": {"resourceType": "Encounter", "id": "e1",
                            "subject": {"reference": f"Patient/{PATIENT}"},
                            "period": {"start": "2136-10-31T23:00:00"}}}
        warnings = self.check([cite("o1"), cite("e1", "period.start", "2136-10-31T23:00:00", "Encounter")],
                              resources, question="How many hospital visits in 11/2136?")
        self.assertEqual(len(warnings), 1)
        self.assertIn("2136-10-31", warnings[0])

    def test_linked_specimen_metadata_is_not_filtered_by_its_collection_date(self):
        resources = {"o1": observation("o1", "2136-11-01T01:00:00"),
                     "s1": {"resourceType": "Specimen", "id": "s1",
                            "subject": {"reference": f"Patient/{PATIENT}"},
                            "collection": {"collectedDateTime": "2136-10-31T23:00:00"},
                            "type": {"text": "Blood"}}}
        self.assertEqual(self.check([cite("o1"), cite("s1", "type.text", "Blood", "Specimen")], resources), [])

    def test_release_time_does_not_establish_missing_measurement_time(self):
        resource = observation("o1", "2136-11-01T01:00:00")
        resource["issued"] = resource.pop("effectiveDateTime")
        warnings = self.check([cite("o1")], {"o1": resource})
        self.assertEqual(len(warnings), 1)
        self.assertIn("no event timestamp", warnings[0])

    def test_missing_condition_encounter_date_cannot_establish_a_window(self):
        condition = {"resourceType": "Condition", "id": "c1", "subject": {"reference": f"Patient/{PATIENT}"},
                     "code": {"text": "Sepsis"}}
        warnings = self.check([cite("c1", "code.text", "Sepsis", "Condition")], {"c1": condition},
                              question="Was the patient diagnosed with sepsis in 2136?")
        self.assertEqual(len(warnings), 1)
        self.assertIn("no event timestamp", warnings[0])

    def test_future_cited_discharge_is_not_hidden_by_an_earlier_admission(self):
        resource = {"resourceType": "Encounter", "id": "e1", "subject": {"reference": f"Patient/{PATIENT}"},
                    "period": {"start": "2136-11-01T01:00:00", "end": "2137-01-01T00:00:00"}}
        warnings = self.check([cite("e1", "period.end", "2137-01-01T00:00:00", "Encounter")],
                              {"e1": resource}, question="When was the patient discharged?")
        self.assertEqual(len(warnings), 1)
        self.assertTrue(warnings[0].startswith("current_time:"))

    def test_minute_precision_current_time_includes_the_exact_second(self):
        resources = {"o1": observation("o1", "2136-12-31T23:59:00-05:00")}
        self.assertEqual(self.check([cite("o1")], resources, question="Minimum weight today?",
                                    now="2136-12-31T23:59-05:00"), [])

    def test_questions_without_a_window_are_unaffected(self):
        resources = {"o1": observation("o1", "2100-01-01T00:00:00")}
        self.assertEqual(self.check([cite("o1")], resources, question="What was the minimum weight?"), [])

    def test_verifier_turns_the_warning_into_an_analysis_issue(self):
        resources = {"out": observation("out", "2136-12-01T00:00:00")}
        check = check_fields([cite("out")], question=self.QUESTION, current_time="2136-12-31 23:59:00",
                             resources=resources, patient_fhir_id=PATIENT)
        issue = field_check_issues({"answer": 7, "evidence": [cite("out")]}, check)[0]
        self.assertEqual((issue["type"], issue["record"], issue["responsible_role"]),
                         ("time_window", "Observation/out", "analysis"))


class EncounterScopeTests(unittest.TestCase):
    NOW = "2136-12-31 23:59:00"

    def encounter(self, rid, start, end=None, kind="hosp", **extra):
        period = {"start": start}
        if end is not None:
            period["end"] = end
        return {"resourceType": "Encounter", "id": rid, "subject": {"reference": f"Patient/{PATIENT}"},
                "identifier": [{"system": f"http://example.org/encounter-{kind}"}], "period": period, **extra}

    def select(self, question, *records, now=NOW):
        return question_encounter_scope(question, PATIENT, {r["id"]: r for r in records}, now)

    def test_last_visit_is_latest_completed_not_ongoing_or_future(self):
        completed = self.encounter("done", "2136-10-01", "2136-10-03")
        ongoing = self.encounter("ongoing", "2136-12-30")
        ends_later = self.encounter("future-end", "2136-12-31", "2137-01-02")
        future = self.encounter("future", "2137-01-03", "2137-01-04")
        for wording in ("last hospital visit", "latest hospital encounter", "last completed hospital stay"):
            self.assertEqual(self.select(f"Minimum weight during the {wording}?", completed, ongoing, ends_later, future),
                             ("done", []))

    def test_current_visit_is_ongoing_at_assumed_time_not_latest_completed(self):
        completed = self.encounter("done", "2136-10-01", "2136-10-03")
        ongoing = self.encounter("ongoing", "2136-12-30", "2137-01-02")
        future = self.encounter("future", "2137-01-03")
        for wording in ("current hospital visit", "this hospital encounter"):
            self.assertEqual(self.select(f"Minimum weight during the {wording}?", completed, ongoing, future),
                             ("ongoing", []))

    def test_first_visit_filters_the_patient_and_encounter_kind(self):
        first = self.encounter("first", "2136-10-01", "2136-10-03")
        later = self.encounter("later", "2136-12-30")
        other = self.encounter("other", "2135-01-01", subject={"reference": "Patient/other"})
        icu = self.encounter("icu", "2135-01-01", "2135-01-02", kind="icu")
        self.assertEqual(self.select("First hospital visit?", later, other, icu, first), ("first", []))
        self.assertEqual(self.select("First ICU stay?", later, other, icu, first), ("icu", []))

    def test_exact_discharge_boundary_is_completed_not_current(self):
        encounter = self.encounter("e1", "2136-12-30", self.NOW)
        self.assertEqual(self.select("Last hospital visit?", encounter), ("e1", []))
        self.assertTrue(self.select("This hospital visit?", encounter)[1])

    def test_missing_current_time_ambiguous_current_and_invalid_end_fail_scope(self):
        a = self.encounter("a", "2136-12-30")
        b = self.encounter("b", "2136-12-31")
        invalid = self.encounter("bad", "2136-12-30", "not a timestamp")
        self.assertTrue(self.select("This hospital visit?", a, now=None)[1])
        self.assertIn("ambiguous", self.select("This hospital visit?", a, b)[1][0])
        self.assertTrue(self.select("This hospital visit?", invalid)[1])

    def test_claimed_encounter_cannot_override_question_derived_scope(self):
        completed = self.encounter("done", "2136-10-01", "2136-10-03")
        ongoing = self.encounter("ongoing", "2136-12-30")
        result = check_fields([], question="Minimum weight on the last hospital visit?", current_time=self.NOW,
                              patient_fhir_id=PATIENT, encounter_id="ongoing",
                              resources={"done": completed, "ongoing": ongoing})
        self.assertIn("selects Encounter/done, not Encounter/ongoing", result["issues"][0])

    def test_pure_event_time_helper_does_not_substitute_order_time(self):
        self.assertEqual(resource_event_time({"resourceType": "MedicationRequest", "authoredOn": self.NOW}), "")


if __name__ == "__main__":
    unittest.main()
