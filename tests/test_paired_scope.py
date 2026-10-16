"""Isolated, synthetic source-scope tests. No model, gold label or FHIR calls."""
import unittest
from unittest.mock import patch
from tools import field_check as proposed

PATIENT = "patient-a"
NOW = "2199-12-31 23:59:00"
ISO_PAIR = ("What was the change in respiratory rate measured at 2199-06-01 10:00:00 "
            "compared to the value measured at 2199-06-01 09:00:00?")


def encounter(rid, start, end, kind="hosp", parent=None):
    resource = {"resourceType": "Encounter", "id": rid,
                "subject": {"reference": f"Patient/{PATIENT}"},
                "identifier": [{"system": f"http://example/encounter-{kind}"}],
                "period": {"start": start}}
    if end:
        resource["period"]["end"] = end
    if parent:
        resource["partOf"] = {"reference": f"Encounter/{parent}"}
    return resource


def observation(rid, when, visit=None, patient=PATIENT):
    resource = {"resourceType": "Observation", "id": rid,
                "subject": {"reference": f"Patient/{patient}"},
                "effectiveDateTime": when, "issued": when,
                "valueQuantity": {"value": 7, "unit": "kg"}}
    if visit:
        resource["encounter"] = {"reference": f"Encounter/{visit}"}
    return resource


def citation(rid, path="valueQuantity.value", value=7, scope=None):
    result = {"resource_id": f"Observation/{rid}", "field_path": path, "value": value}
    if scope is not None:
        result["scope_index"] = scope
    return result


def index(*resources):
    return {f"{r['resourceType']}/{r['id']}": r for r in resources}


class PairedScopeTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(proposed, "fetch_from_store", side_effect=AssertionError("network forbidden"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.earlier = observation("earlier", "2199-06-01T09:00:00-05:00")
        self.later = observation("later", "2199-06-01T10:00:00-05:00")
        self.outside = observation("outside", "2199-06-01T08:00:00-05:00")
        self.records = index(self.earlier, self.later, self.outside)

    def check(self, citations, question=ISO_PAIR, records=None, now=NOW, **kwargs):
        return proposed.check_fields(citations, question=question, current_time=now,
                                     patient_fhir_id=PATIENT,
                                     resources=self.records if records is None else records, **kwargs)

    def test_two_explicit_times_both_accepted_without_annotations(self):
        result = self.check([citation("later"), citation("earlier")])
        self.assertEqual(result["issues"], [])
        self.assertEqual([r["matched_scopes"] for r in result["results"]], [[0], [1]])
        self.assertEqual(result["question_scopes"][0]["time_window"][:2],
                         ("2199-06-01 10:00:00", "2199-06-01 10:00:01"))
        self.assertTrue(result["scope_membership_only"])

    def test_outside_both_requested_times_is_rejected(self):
        result = self.check([citation("outside")])
        self.assertTrue(result["issues"])
        self.assertEqual(result["results"][0]["matched_scopes"], [])
        self.assertIn("paired_scope", result["issues"][0])

    def test_explicit_index_cannot_switch_to_other_operand_or_widen_scope(self):
        self.assertEqual(self.check([citation("later", scope=0), citation("earlier", scope=1)])["issues"], [])
        wrong = self.check([citation("earlier", scope=0)])
        self.assertTrue(wrong["issues"])
        self.assertEqual(wrong["results"][0]["matched_scopes"], [])
        for invalid in (True, -1, 2, "0"):
            with self.subTest(invalid=invalid):
                bad = self.check([citation("later", scope=invalid)])
                self.assertIn("scope_index must be", " ".join(bad["issues"]))

    def test_inverted_compared_to_sentence_preserves_textual_scope_indices(self):
        question = ("Compared to measured at 2199-06-01 10:00:00, is the respiratory rate "
                    "greater than measured at 2199-06-01 09:00:00?")
        self.assertEqual(self.check([citation("later", scope=0), citation("earlier", scope=1)], question)["issues"], [])

    def test_inverted_comparison_with_trailing_greater_and_no_than(self):
        visit = encounter("visit", "2199-05-01T00:00:00", "2199-06-10T00:00:00", "icu")
        a = observation("a", "2199-06-01T09:00:00", "visit")
        b = observation("b", "2199-06-01T10:00:00", "visit")
        question = "Compared to first measured on the first icu visit, is the spo2 value last measured on the first icu visit greater?"
        result = self.check([citation("a", scope=0), citation("b", scope=1)], question, index(visit, a, b))
        self.assertEqual(result["issues"], [])
        self.assertEqual([s["encounter_id"] for s in result["question_scopes"]], ["visit", "visit"])

    def test_wrong_patient_bad_field_bad_value_and_wrong_time_field_remain_blocked(self):
        other = observation("other", "2199-06-01T10:00:00", patient="patient-b")
        result = self.check([citation("other")], records=index(other))
        self.assertIn("patient_scope", " ".join(result["issues"]))
        self.assertIn("bad_path", " ".join(self.check([citation("later", path="nonexistent")])["issues"]))
        self.assertIn("value_mismatch", " ".join(self.check([citation("later", value=99)])["issues"]))
        wrong_time = self.check([citation("later", path="issued", value=self.later["issued"])])
        self.assertIn("time_field", " ".join(wrong_time["issues"]))

    def test_current_time_is_enforced_even_if_question_requests_future_operand(self):
        result = self.check([citation("later")], now="2199-06-01 09:30:00")
        self.assertIn("current_time", " ".join(result["issues"]))
        self.assertEqual(result["results"][0]["matched_scopes"], [])

    def test_two_calendar_windows_are_a_union_not_a_global_widening(self):
        question = "What is the difference in weight last measured in 05/2199 compared to first measured in 07/2199?"
        may = observation("may", "2199-05-05T00:00:00")
        july = observation("july", "2199-07-05T00:00:00")
        june = observation("june", "2199-06-05T00:00:00")
        records = index(may, july, june)
        okay = self.check([citation("may"), citation("july")], question, records)
        self.assertEqual(okay["issues"], [])
        self.assertEqual([r["matched_scopes"] for r in okay["results"]], [[0], [1]])
        self.assertTrue(self.check([citation("june")], question, records)["issues"])

    def test_two_encounter_scopes_and_child_parent_links(self):
        first = encounter("first", "2199-01-01T00:00:00", "2199-01-10T00:00:00")
        middle = encounter("middle", "2199-02-01T00:00:00", "2199-02-10T00:00:00")
        last = encounter("last", "2199-03-01T00:00:00", "2199-03-10T00:00:00")
        child = encounter("child", "2199-01-02T00:00:00", "2199-01-05T00:00:00", "icu", "first")
        a = observation("a", "2199-01-03T00:00:00", "child")
        b = observation("b", "2199-03-03T00:00:00", "last")
        c = observation("c", "2199-02-03T00:00:00", "middle")
        records = index(first, middle, last, child, a, b, c)
        question = "What is the change from first measured on the first hospital visit compared to last measured on the last hospital visit?"
        result = self.check([citation("a", scope=0), citation("b", scope=1)], question, records)
        self.assertEqual(result["issues"], [])
        self.assertEqual([s["encounter_id"] for s in result["question_scopes"]], ["first", "last"])
        self.assertTrue(self.check([citation("c")], question, records)["issues"])
        self.assertTrue(self.check([citation("b", scope=0)], question, records)["issues"])
        self.assertTrue(self.check([citation("a")], question, records, encounter_id="middle")["issues"])

    def test_shared_final_icu_qualifier_on_explicit_from_to_form(self):
        first = encounter("first", "2199-01-01T00:00:00", "2199-01-10T00:00:00", "icu")
        last = encounter("last", "2199-03-01T00:00:00", "2199-03-10T00:00:00", "icu")
        a = observation("a", "2199-03-03T00:00:00", "last")
        b = observation("b", "2199-03-04T00:00:00", "last")
        c = observation("c", "2199-01-03T00:00:00", "first")
        records = index(first, last, a, b, c)
        question = "How much was the difference in temperature from the second-to-last value measured to the last value measured during the final icu visit?"
        result = self.check([citation("a", scope=0), citation("b", scope=1)], question, records)
        self.assertEqual(result["issues"], [])
        self.assertEqual([s["encounter_id"] for s in result["question_scopes"]], ["last", "last"])
        self.assertTrue(self.check([citation("c")], question, records)["issues"])

    def test_invalid_requested_timestamp_or_unresolved_relative_window_fails_closed(self):
        question = ISO_PAIR.replace("2199-06-01 10:00:00", "2199-02-30 10:00:00")
        self.assertIn("invalid explicitly requested", " ".join(self.check([citation("earlier")], question)["issues"]))
        question = "What is the difference from last measured yesterday compared to first measured today?"
        self.assertIn("cannot be resolved", " ".join(self.check([], question, now=None)["issues"]))

    def test_threshold_is_not_a_paired_question_and_legacy_checks_remain_scoped(self):
        questions = ["Was weight measured less than 8 in 05/2199?", "What was the minimum weight in 05/2199?",
                     "What was the weight last measured on the first hospital visit?"]
        visit = encounter("visit", "2199-05-01T00:00:00", "2199-05-10T00:00:00")
        record = observation("record", "2199-05-02T00:00:00", "visit")
        records = index(visit, record)
        for question in questions:
            with self.subTest(question=question):
                self.assertIsNone(proposed.question_evidence_scopes(question, NOW))
                args = {"question": question, "current_time": NOW, "patient_fhir_id": PATIENT,
                        "resources": records}
                result = proposed.check_fields([citation("record")], **args)
                self.assertNotIn("question_scopes", result)
                self.assertEqual(result["issues"], [])
                other = encounter("outside", "2199-06-01T00:00:00", "2199-06-10T00:00:00")
                wrong = {**args, "resources": index(visit, other,
                                                    observation("record", "2199-06-02T00:00:00", "outside"))}
                self.assertTrue(proposed.check_fields([citation("record")], **wrong)["issues"])


if __name__ == "__main__":
    unittest.main()
