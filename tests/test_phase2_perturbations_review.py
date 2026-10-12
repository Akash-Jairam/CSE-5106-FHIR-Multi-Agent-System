"""Synthetic development cases prove actual visible-query answer changes."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_perturbations import evaluate
from scripts.packet_query import PacketQuery, UnsupportedQuery
from tools.perturbation import get_perturbed_starting_packet, load_perturbations, omit_from_packet, text_sha256
from tools.starting_packet import PACKET_VERSION, build_starting_packet

PATIENT = "dev-patient"


def observation(rid, value, day, label="Glucose", itemid="1"):
    return {"resourceType": "Observation", "id": rid, "subject": {"reference": f"Patient/{PATIENT}"},
            "encounter": {"reference": "Encounter/h1"}, "effectiveDateTime": f"2150-01-{day:02}T08:00:00",
            "code": {"coding": [{"code": itemid, "system": "x/mimic-d-labitems", "display": label}]},
            "valueQuantity": {"value": value, "unit": "mg/dL"}}


def record(*observations):
    return {"Observation": list(observations), "Encounter": [{"resourceType": "Encounter", "id": "h1",
            "subject": {"reference": f"Patient/{PATIENT}"},
            "identifier": [{"system": "x/encounter-hosp", "value": "10"}],
            "period": {"start": "2150-01-01T00:00:00", "end": "2150-02-01T00:00:00"}}]}


def question(sql, answer, gold=("a",), text="What was the maximum Glucose value?"):
    return {"question_id": "synthetic-dev", "question": text, "patient_fhir_id": PATIENT,
            "sql_query": sql, "true_answer": f"[[{answer}]]", "true_fhir_ids": repr({"Observation": list(gold)}),
            "val_dict": {"val_placeholder": {"patient_id": 1}}, "main_table_name": "labevents", "template_id": "dev"}


def frozen(q, source, max_tokens=8000):
    packet = build_starting_packet(q["question"], source, PATIENT, max_tokens)
    row = {"resource_ids": packet["resource_ids"], "packet_text_sha256": text_sha256(packet["text"]),
           "current_time": None, "max_tokens": max_tokens, "fhir_source_sha256": "synthetic-dev"}
    return packet, row


class AnswerChangeReviewTests(unittest.TestCase):
    def evaluate(self, q, source, family="numerical_aggregation", template="What was the [agg_function] Glucose value?"):
        _, row = frozen(q, source)
        return evaluate(q, row, source, family, template)

    def test_incomplete_gold_cannot_make_redundant_existence_eligible(self):
        q = question("SELECT COUNT(*)>0 FROM labevents", 1, text="Has the patient received a Glucose lab test?")
        entry, reason = self.evaluate(q, record(observation("a", 90, 1), observation("b", 90, 2)),
                                      "lookup_counting", "Has_verb patient received a Glucose lab test?")
        self.assertIsNone(entry)
        self.assertIn("no one-resource omission", reason)

    def test_operation_is_the_query_operation_not_a_matching_gold_value(self):
        q = question("SELECT AVG(valuenum) FROM labevents", 120, gold=("b",))
        entry, reason = self.evaluate(q, record(observation("a", 90, 1), observation("b", 120, 2), observation("c", 150, 3)))
        self.assertIsNone(reason)
        self.assertEqual(entry["rule"], "aggregate:avg")
        self.assertNotEqual(entry["proof"]["original_result_sha256"], entry["proof"]["perturbed_result_sha256"])

    def test_query_predicate_ignores_unrelated_visible_labels(self):
        q = question("SELECT MAX(valuenum) FROM labevents WHERE itemid IN "
                     "(SELECT itemid FROM d_labitems WHERE label='Glucose')", 120, gold=("b",))
        source = record(observation("a", 90, 1), observation("b", 120, 2), observation("other", 1000, 3, "Sodium", "2"))
        entry, reason = self.evaluate(q, source)
        self.assertIsNone(reason)
        self.assertEqual(entry["removed"]["id"], "b")

    def test_tied_maximum_is_not_answer_changing(self):
        q = question("SELECT MAX(valuenum) FROM labevents", 120, gold=("a",))
        entry, reason = self.evaluate(q, record(observation("a", 120, 1), observation("b", 120, 2)))
        self.assertIsNone(entry)
        self.assertIn("no one-resource omission", reason)

    def test_count_uses_all_visible_records_not_gold_cardinality(self):
        q = question("SELECT COUNT(*) FROM labevents", 3, gold=("a",))
        entry, reason = self.evaluate(q, record(observation("a", 90, 1), observation("b", 120, 2), observation("c", 150, 3)),
                                      "lookup_counting", "Count the number of Glucose tests")
        self.assertIsNone(reason)
        self.assertEqual(entry["rule"], "count")

    def test_original_visible_query_must_reproduce_reference(self):
        q = question("SELECT MAX(valuenum) FROM labevents", 999, gold=("a",))
        entry, reason = self.evaluate(q, record(observation("a", 90, 1)))
        self.assertIsNone(entry)
        self.assertEqual(reason, "visible query does not reproduce reference")

    def test_date_and_encounter_filters_are_executed(self):
        q = question("SELECT MAX(valuenum) FROM labevents WHERE hadm_id IN "
                     "(SELECT hadm_id FROM admissions WHERE subject_id=1 AND dischtime IS NOT NULL) "
                     "AND strftime('%d',charttime)='02'", 120, gold=("b",))
        entry, reason = self.evaluate(q, record(observation("a", 200, 1), observation("b", 120, 2)))
        self.assertIsNone(reason)
        self.assertEqual(entry["removed"]["id"], "b")

    def test_discharge_after_reference_time_is_an_ongoing_visit(self):
        q = question("SELECT MAX(valuenum) FROM labevents WHERE hadm_id IN "
                     "(SELECT hadm_id FROM admissions WHERE subject_id=1 AND dischtime IS NULL)", 120, gold=("b",))
        q["assumption"] = "Assume the current time is 2150-01-10 23:59:00."
        entry, reason = self.evaluate(q, record(observation("a", 90, 1), observation("b", 120, 2)))
        self.assertIsNone(reason)
        self.assertEqual(entry["removed"]["id"], "b")

    def test_unsupported_query_is_excluded_without_guessing(self):
        q = question("SELECT MAX(imaginary_value) FROM labevents", 90)
        entry, reason = self.evaluate(q, record(observation("a", 90, 1)))
        self.assertIsNone(entry)
        self.assertIn("visible packet", reason)
        with self.assertRaises(UnsupportedQuery):
            PacketQuery({**q, "sql_query": "DELETE FROM labevents"}, {})

    def test_change_within_existing_scorer_tolerance_is_not_eligible(self):
        q = question("SELECT AVG(valuenum) FROM labevents", 100, gold=("a",))
        entry, reason = self.evaluate(q, record(observation("a", 99.9, 1), observation("b", 100.1, 2)))
        self.assertIsNone(entry)
        self.assertIn("no one-resource omission", reason)

    def test_retained_drug_metadata_cannot_supply_the_removed_name(self):
        q = question("SELECT drug FROM prescriptions ORDER BY starttime LIMIT 1", "'Glucagon'",
                     text="What was the name of the first medication prescribed?")
        q.update(main_table_name="prescriptions", true_fhir_ids=repr({"MedicationRequest": ["rx"]}))
        source = record()
        source["Medication"] = [{"resourceType": "Medication", "id": "m1", "identifier": [
            {"system": "x/mimic-medication-name", "value": "Glucagon"}]}]
        source["MedicationRequest"] = [{"resourceType": "MedicationRequest", "id": "rx",
            "subject": {"reference": f"Patient/{PATIENT}"}, "encounter": {"reference": "Encounter/h1"},
            "medicationReference": {"reference": "Medication/m1"},
            "dispenseRequest": {"validityPeriod": {"start": "2150-01-02T08:00:00"}}}]
        entry, reason = self.evaluate(q, source, "temporal_selection", "What was the [time_filter_exact1] drug name?")
        self.assertIsNone(entry)
        self.assertIn("no one-resource omission", reason)

    def test_incoming_clinical_reference_cannot_be_hidden_to_create_change(self):
        q = question("SELECT MAX(valuenum) FROM labevents", 120, gold=("b",))
        source = record(observation("a", 90, 1), observation("b", 120, 2))
        source["Observation"][0]["derivedFrom"] = [{"reference": "Observation/b"}]
        entry, reason = self.evaluate(q, source)
        self.assertIsNone(entry)
        self.assertIn("no one-resource omission", reason)


class StrictOmissionReviewTests(unittest.TestCase):
    def test_omission_does_not_refill_a_truncated_packet(self):
        q = question("SELECT MAX(valuenum) FROM labevents", 999)
        source = record(*(observation(f"o{i}", i, 1) for i in range(40)))
        packet, _ = frozen(q, source, max_tokens=800)
        held = packet["resource_ids"]["Observation"]
        self.assertLess(len(held), 40)
        omitted = omit_from_packet(packet, [("Observation", held[0])])
        self.assertEqual(omitted["resource_ids"]["Observation"], held[1:])
        self.assertEqual(len(packet["resource_ids"]["Observation"]), len(held))

    def test_reference_redaction_preserves_unrelated_null_and_clinical_values(self):
        q = question("SELECT MAX(valuenum) FROM labevents", 120)
        source = record(observation("a", 90, 1), observation("b", 120, 2))
        source["Observation"][0].update({"derivedFrom": [{"reference": "Observation/b"}], "valueString": None})
        packet, _ = frozen(q, source)
        omitted = omit_from_packet(packet, [("Observation", "b")])
        self.assertNotIn("Observation/b", omitted["text"])
        self.assertEqual(omitted["resources"]["Observation"][0]["valueQuantity"]["value"], 90)
        self.assertIsNone(omitted["resources"]["Observation"][0]["valueString"])

    def test_runtime_requires_frozen_context_and_rejects_live_packet_drift(self):
        q = question("SELECT MAX(valuenum) FROM labevents", 120)
        source = record(observation("a", 90, 1), observation("b", 120, 2))
        entry, _ = AnswerChangeReviewTests().evaluate(q, source)
        with self.assertRaisesRegex(ValueError, "frozen perturbation specification"):
            get_perturbed_starting_packet(q["question"], PATIENT, [("Observation", "b")])
        with self.assertRaisesRegex(ValueError, "context"):
            get_perturbed_starting_packet("another question", PATIENT, entry)
        with patch("tools.resource_tools.get_all_resources_by_patient_fhir_id", return_value=source):
            packet = get_perturbed_starting_packet(q["question"], PATIENT, entry)
            self.assertNotIn("b", packet["resource_ids"]["Observation"])
        source["Observation"][0]["valueQuantity"]["value"] = 999
        with patch("tools.resource_tools.get_all_resources_by_patient_fhir_id", return_value=source):
            with self.assertRaisesRegex(ValueError, "frozen original"):
                get_perturbed_starting_packet(q["question"], PATIENT, entry)

    def test_loader_refuses_stale_or_changed_freezes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"version": 1}))
            with self.assertRaisesRegex(ValueError, "Rebuild perturbations"):
                load_perturbations(path)
            path.write_text(json.dumps({"version": 2, "packet_version": PACKET_VERSION,
                                        "split": "held_out", "size": 50,
                                        "source_sha256": "tampered"}))
            with self.assertRaisesRegex(ValueError, "artifact mismatch"):
                load_perturbations(path)


if __name__ == "__main__":
    unittest.main()
