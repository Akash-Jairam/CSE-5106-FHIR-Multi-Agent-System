"""P2-18: missing-evidence perturbations remove one answer-changing resource
from held-out starting packets and leave the store intact."""
import json
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from scripts.build_perturbations import (DEFAULT_FHIR_DIR, carries, event_gold, leaves, redundant,
                                         sample)
from scripts.packet_query import PacketQuery
from scripts.build_split import file_sha256
from tools.perturbation import load_perturbations, omit_from_packet, without_resources
from tools.starting_packet import build_starting_packet

ROOT = Path(__file__).resolve().parent.parent
PATIENT = "11111111-1111-1111-1111-111111111111"


def observation(rid, value, when):
    return {"resourceType": "Observation", "id": rid, "subject": {"reference": f"Patient/{PATIENT}"},
            "code": {"coding": [{"system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-d-labitems",
                                 "display": "Glucose"}]}, "effectiveDateTime": when,
            "valueQuantity": {"value": value, "unit": "mg/dL"}}


def by_id(*resources):
    return {f"{r['resourceType']}/{r['id']}": r for r in resources}


class RuleTests(unittest.TestCase):
    def setUp(self):
        self.records = by_id(observation("a", 90, "2150-01-01T08:00:00"), observation("b", 120, "2150-01-02T08:00:00"),
                             observation("c", 150, "2150-01-03T08:00:00"), observation("z", 0, "2150-01-04T08:00:00"))

    def test_event_gold_ignores_shared_resources(self):
        gold = {"Observation": ["b", "a"], "Medication": ["m"], "Patient": ["p"], "Location": ["l"]}
        self.assertEqual(event_gold(gold), [("Observation", "a"), ("Observation", "b")])

    def test_operations_come_from_the_actual_sql_query(self):
        source = {"Observation": [self.records[f"Observation/{rid}"] for rid in ("a", "b", "c")]}
        packet = build_starting_packet("Glucose lab value", source, PATIENT)
        q = {"val_dict": {"val_placeholder": {"patient_id": 1}}, "main_table_name": "labevents"}
        for operation, result in (("MIN", 90), ("MAX", 150), ("SUM", 360), ("AVG", 120), ("COUNT", 3)):
            with self.subTest(operation=operation):
                column = "*" if operation == "COUNT" else "valuenum"
                oracle = PacketQuery({**q, "sql_query": f"SELECT {operation}({column}) FROM labevents"}, self.records)
                self.assertEqual(oracle.run(packet), [result])


class RedundancyTests(unittest.TestCase):
    def test_answer_values_match_by_kind(self):
        self.assertTrue(carries("2150-01-02 08:00:00", "2150-01-02T08:00:00-05:00"))
        self.assertFalse(carries("2150-01-02 08:00:00", "2150-01-02T08:01:00"))
        self.assertTrue(carries("95.", 95))
        self.assertFalse(carries("95", True))
        self.assertTrue(carries("EW EMER.", "ew emer."))
        self.assertFalse(carries("Heparin", "Heparin Sodium"))

    def test_names_are_resolved_through_references(self):
        medication = {"resourceType": "Medication", "id": "m1",
                      "identifier": [{"system": "x/mimic-medication-name", "value": "Glucagon"}]}
        first = {"resourceType": "MedicationRequest", "id": "r1", "medicationReference": {"reference": "Medication/m1"},
                 "subject": {"reference": f"Patient/{PATIENT}"}}
        second = {**first, "id": "r2"}
        records = by_id(medication, first, second)
        self.assertIn("Glucagon", leaves(second, records))
        self.assertTrue(redundant(["Glucagon"], ("MedicationRequest", "r1"), {"MedicationRequest": ["r2"]}, records))
        self.assertFalse(redundant(["Heparin"], ("MedicationRequest", "r1"), {"MedicationRequest": ["r2"]}, records))

    def test_another_record_with_the_answer_value_is_redundant(self):
        records = by_id(observation("a", 95, "2150-01-01T08:00:00"), observation("b", 95, "2150-01-02T08:00:00"))
        self.assertTrue(redundant([95.0], ("Observation", "b"), {"Observation": ["a"]}, records))
        self.assertFalse(redundant(["2150-01-02 08:00:00"], ("Observation", "b"), {"Observation": ["a"]}, records))


class SampleTests(unittest.TestCase):
    def test_proportional_seeded_sample(self):
        eligible = ([{"question_id": f"t{i:02}", "family": "temporal_selection"} for i in range(45)]
                    + [{"question_id": f"l{i:02}", "family": "lookup_counting"} for i in range(28)]
                    + [{"question_id": f"n{i:02}", "family": "numerical_aggregation"} for i in range(10)])
        chosen = sample(eligible, 50, 5106)
        self.assertEqual(Counter(e["family"] for e in chosen),
                         {"temporal_selection": 27, "lookup_counting": 17, "numerical_aggregation": 6})
        self.assertEqual(chosen, sample(list(reversed(eligible)), 50, 5106))
        with self.assertRaises(ValueError):
            sample(eligible[:10], 50, 5106)


class RuntimeTests(unittest.TestCase):
    def test_removed_resource_is_left_out_of_the_packet_only(self):
        record = {"Observation": [observation("a", 90, "2150-01-01T08:00:00"),
                                  observation("b", 120, "2150-01-02T08:00:00")],
                  "Patient": [{"resourceType": "Patient", "id": PATIENT}]}
        self.assertEqual([r["id"] for r in without_resources(record, [("Observation", "b")])["Observation"]], ["a"])
        self.assertEqual(len(record["Observation"]), 2)
        original = build_starting_packet("What was the last glucose value?", record, PATIENT)
        packet = omit_from_packet(original, [("Observation", "b")])
        self.assertEqual(packet["resource_ids"]["Observation"], ["a"])
        self.assertIn("(1 of 1)", packet["text"])  # looks complete: the record is absent, not truncated


class FrozenManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((ROOT / "manifests/perturbations.json").read_text())
        self.split = json.loads((ROOT / "manifests/split.json").read_text())
        self.packets = {p["question_id"]: p for p in map(json.loads,
                        (ROOT / "manifests/starting_packets.jsonl").read_text().splitlines())}

    def test_manifest_matches_current_sources(self):
        m = self.manifest
        self.assertEqual((m["split"], m["seed"], m["size"]), ("held_out", self.split["seed"], 50))
        self.assertEqual(m["source_sha256"], file_sha256(ROOT / "final_dataset/questions_answers_sql_fhir.csv"))
        self.assertEqual(m["split_sha256"], file_sha256(ROOT / "manifests/split.json"))
        self.assertEqual(m["families_sha256"], file_sha256(ROOT / "manifests/question_families.csv"))
        self.assertEqual(m["packets_sha256"], file_sha256(ROOT / "manifests/starting_packets.jsonl"))
        self.assertEqual(m["builder_sha256"], file_sha256(ROOT / "scripts/build_perturbations.py"))

    def test_fifty_distinct_held_out_questions_each_missing_one_packet_resource(self):
        held_out = set(self.split["splits"]["held_out"]["question_ids"])
        entries = self.manifest["perturbations"]
        self.assertEqual(len({e["question_id"] for e in entries}), 50)
        self.assertLessEqual({e["question_id"] for e in entries}, held_out)
        self.assertEqual(sum(self.manifest["selected_by_family"].values()), 50)
        for e in entries:
            original = self.packets[e["question_id"]]
            removed = e["removed"]
            self.assertIn(removed["id"], original["resource_ids"][removed["resource_type"]])
            self.assertNotIn(removed["id"], e["perturbed_resource_ids"].get(removed["resource_type"], []))
            self.assertEqual(e["original_packet_sha256"], original["packet_text_sha256"])
            self.assertNotEqual(e["perturbed_packet_sha256"], e["original_packet_sha256"])
            self.assertLessEqual(e["perturbed_n_tokens"], 8000)
        self.assertEqual(load_perturbations(ROOT / "manifests/perturbations.json")[entries[0]["question_id"]], entries[0])

    @unittest.skipUnless(Path(DEFAULT_FHIR_DIR).is_dir(), "local MIMIC-IV FHIR demo NDJSON not downloaded")
    def test_perturbed_packets_rebuild_from_the_source_records(self):
        import pandas as pd
        from scripts.build_starting_packets import index_patient_lines, load_record
        from tools.starting_packet import build_starting_packet
        import hashlib

        entries = self.manifest["perturbations"]
        patient = entries[0]["patient_fhir_id"]
        mine = [e for e in entries if e["patient_fhir_id"] == patient]
        questions = pd.read_csv(ROOT / "final_dataset/questions_answers_sql_fhir.csv",
                                usecols=["question_id", "question"]).set_index("question_id").question
        shared, offsets = index_patient_lines(DEFAULT_FHIR_DIR, {patient})
        record = load_record(shared, offsets[patient])
        for e in mine:
            removed = [(e["removed"]["resource_type"], e["removed"]["id"])]
            original = build_starting_packet(questions[e["question_id"]], record, patient, 8000, e["current_time"])
            packet = omit_from_packet(original, removed)
            self.assertEqual(hashlib.sha256(packet["text"].encode()).hexdigest(), e["perturbed_packet_sha256"])


if __name__ == "__main__":
    unittest.main()
