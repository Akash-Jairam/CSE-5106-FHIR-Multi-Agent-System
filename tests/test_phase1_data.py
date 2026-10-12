"""Offline regressions for proposal scope, held-out integrity, and packets."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from scripts.build_question_families import build_manifest, classify_template
from scripts.build_split import build_split, check_disjoint, file_sha256, subject_id
from scripts.build_starting_packets import index_patient_lines, load_record
from tools.starting_packet import PACKET_VERSION, build_starting_packet, count_tokens

ROOT = Path(__file__).resolve().parents[1]


def patient(identifier="p"):
    return {"resourceType": "Patient", "id": identifier, "gender": "female"}


def encounter(identifier, start, **extra):
    return {"resourceType": "Encounter", "id": identifier,
            "subject": {"reference": "Patient/p"}, "period": {"start": start},
            "identifier": [{"system": "https://example/encounter-hosp", "value": identifier}], **extra}


def observation(identifier, time, visit="e1", **extra):
    return {"resourceType": "Observation", "id": identifier,
            "subject": {"reference": "Patient/p"}, "encounter": {"reference": f"Encounter/{visit}"},
            "code": {"coding": [{"system": "https://example/chartevents-d-items", "display": "Heart Rate", "code": "hr"}]},
            "effectiveDateTime": time, "valueQuantity": {"value": 80, "unit": "bpm"}, **extra}


class ScopeAndSplitTests(unittest.TestCase):
    def test_primary_operation_classification(self):
        cases = {
            "Count hospital visits [time_filter_global1]": "lookup_counting",
            "What was [agg_function] value [time_filter_global1]": "numerical_aggregation",
            "What was total output [time_filter_global1]": "numerical_aggregation",
            "When was the [time_filter_exact1] rate [comparison] than 20?": "temporal_selection",
            "What is the change in rate from [time_filter_exact2] compared to [time_filter_exact1]?": "out_of_scope",
            "Is the rate [time_filter_exact2] [comparison] than [time_filter_exact1]?": "out_of_scope",
            "How many [unit_count] have passed since admission?": "out_of_scope",
            "What was the [time_filter_exact1] length of ICU stay?": "out_of_scope",
        }
        for template, expected in cases.items():
            with self.subTest(template=template):
                self.assertEqual(classify_template(template), expected)

    def test_real_split_excludes_exposed_patients_and_operations(self):
        source = pd.read_csv(ROOT / "final_dataset/questions_answers_sql_fhir.csv")
        manifest = build_manifest(source)
        self.assertEqual(len(manifest), 2931)
        self.assertEqual(manifest.answer_kind.value_counts()["empty"], 387)
        self.assertEqual(manifest.answer_kind.value_counts()["null_value"], 146)
        self.assertTrue(manifest.loc[manifest.family == "out_of_scope", "excluded_reason"].ne("").all())
        source["subject_id"] = source.val_dict.map(subject_id)
        df = source[["question_id", "patient_fhir_id", "subject_id"]].merge(
            manifest[["question_id", "family", "answer_kind"]], on="question_id", validate="one_to_one")
        exposure = json.loads((ROOT / "manifests/development_exposure.json").read_text())
        exposed = {p for item in exposure["exposures"] for p in item["subject_ids"]}
        splits = build_split(df, 5106, exposed)
        check_disjoint(splits)
        self.assertEqual(splits, build_split(df.iloc[::-1], 5106, exposed))
        self.assertEqual(exposed, {p["subject_id"] for p in splits["dev"]["patients"]})
        by_question = df.set_index("question_id")
        for name, total in (("pilot", 20), ("dev", 50), ("held_out", 300)):
            self.assertEqual(len(splits[name]["question_ids"]), total)
            families = set(by_question.loc[splits[name]["question_ids"], "family"])
            self.assertEqual(families, {"lookup_counting", "temporal_selection", "numerical_aggregation"})
        self.assertGreater(splits["held_out"]["answer_kind_counts"]["empty"], 0)
        self.assertGreater(splits["held_out"]["answer_kind_counts"]["null_value"], 0)
        self.assertEqual(splits, json.loads((ROOT / "manifests/split.json").read_text())["splits"])


class StartingPacketTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "Patient": [patient(), patient("other")],
            "Encounter": [encounter("e1", "2150-01-01T00:00:00-05:00")],
            "Observation": [observation("o1", "2150-01-01T03:00:00-05:00"),
                            observation("o2", "2150-01-02T03:00:00-05:00"),
                            observation("other", "2150-01-01T03:00:00-05:00",
                                        subject={"reference": "Patient/other"})],
        }

    def test_fetch_order_and_current_time_filter_are_deterministic(self):
        expected = build_starting_packet("first heart rate", self.record, "p", 2000, "2150-01-01 23:59:00")
        self.assertEqual(expected["resource_ids"]["Observation"], ["o1"])
        self.assertEqual(expected["resource_ids"]["Patient"], ["p"])
        self.assertEqual(expected["after_current_time"], 1)
        reversed_record = {key: list(reversed(self.record[key])) for key in reversed(self.record)}
        self.assertEqual(expected, build_starting_packet("first heart rate", reversed_record, "p", 2000,
                                                        "2150-01-01 23:59:00"))

    def test_selected_evidence_brings_encounter_omitted_from_context(self):
        record = copy.deepcopy(self.record)
        record["Encounter"].append(encounter("bulky", "2151-01-01T00:00:00",
                                             hospitalization={"origin": {"display": "large " * 1000}}))
        packet = build_starting_packet("first heart rate", record, "p", 900)
        self.assertIn("o1", packet["resource_ids"]["Observation"])
        self.assertIn("e1", packet["resource_ids"]["Encounter"])
        self.assertNotIn("bulky", packet["resource_ids"]["Encounter"])

    def test_encounter_admission_type_survives_pruning(self):
        self.record["Encounter"][0]["priority"] = {"coding": [{"code": "UR", "display": "urgent", "system": "x"}]}
        packet = build_starting_packet("first hospital admission type", self.record, "p", 1000)
        self.assertEqual(packet["resources"]["Encounter"][0]["priority"]["coding"][0]["code"], "UR")

    def test_linked_resources_resolve_ids_within_resource_type(self):
        record = copy.deepcopy(self.record)
        record["Encounter"] = [encounter("same", "2150-01-01T00:00:00")]
        record["Medication"] = [{"resourceType": "Medication", "id": "same",
                                  "code": {"coding": [{"display": "Heparin"}]}}]
        record["MedicationRequest"] = [{"resourceType": "MedicationRequest", "id": "request",
            "subject": {"reference": "Patient/p"}, "encounter": {"reference": "Encounter/same"},
            "medicationReference": {"reference": "Medication/same"}, "authoredOn": "2150-01-01"}]
        packet = build_starting_packet("Heparin prescribed", record, "p", 2000)
        self.assertEqual(packet["resource_ids"]["Medication"], ["same"])
        self.assertEqual(packet["resource_ids"]["Encounter"], ["same"])
        self.assertIn("Heparin", packet["text"])

    def test_microbiology_dependencies_and_patient_scope(self):
        organism = observation("organism", "2150-01-01T01:00:00", derivedFrom=[{"reference": "Observation/test"}],
                               code={"coding": [{"system": "https://example/microbiology", "display": "Organism"}]})
        test = observation("test", "2150-01-01T01:00:00", specimen={"reference": "Specimen/blood"},
                           code={"coding": [{"system": "https://example/microbiology", "display": "Culture"}]})
        specimen = {"resourceType": "Specimen", "id": "blood", "subject": {"reference": "Patient/p"},
                    "type": {"coding": [{"display": "Blood"}]}}
        record = {"Patient": [patient()], "Encounter": self.record["Encounter"],
                  "Observation": [organism, test], "Specimen": [specimen]}
        packet = build_starting_packet("organism blood culture", record, "p", 2000)
        self.assertIn("blood", packet["resource_ids"]["Specimen"])
        self.assertEqual(set(packet["resource_ids"]["Observation"]), {"test", "organism"})
        self.assertEqual(len(packet["resource_ids"]["Observation"]), 2)
        by_id = {r["id"]: r for r in packet["resources"]["Observation"]}
        self.assertEqual(by_id["organism"]["code"]["coding"][0]["display"], "Organism")
        record["Encounter"][0]["subject"] = {"reference": "Patient/other"}
        packet = build_starting_packet("organism blood culture", record, "p", 2000)
        self.assertNotIn("Encounter", packet["resource_ids"])

    def test_resources_are_exactly_the_rendered_json_with_valid_paths(self):
        self.record["Observation"][0]["component"] = [
            {"code": {"coding": [{"code": "a", "system": "x"}]}, "valueString": "first"},
            {"code": {"coding": [{"code": "b", "system": "x"}]}, "valueString": "second"}]
        packet = build_starting_packet("heart rate", self.record, "p", 2000)
        shown = [json.loads(line) for line in packet["text"].splitlines() if line.startswith("{")]
        actual = [r for resources in packet["resources"].values() for r in resources]
        self.assertCountEqual(actual, shown)
        original = {(r["resourceType"], r["id"]): r for rows in self.record.values() for r in rows}

        def validate_paths(value, full):
            if isinstance(value, dict):
                for key, child in value.items():
                    self.assertIn(key, full)
                    validate_paths(child, full[key])
            elif isinstance(value, list):
                self.assertEqual(len(value), len(full))
                for item, full_item in zip(value, full):
                    validate_paths(item, full_item)
            else:
                self.assertEqual(value, full)

        for kind, rows in packet["resources"].items():
            for row in rows:
                validate_paths(row, original[(kind, row["id"])])

    def test_exact_token_ceiling_including_headers_and_dependencies(self):
        for limit in (100, 150, 250, 350, 500, 1000):
            with self.subTest(limit=limit):
                packet = build_starting_packet("heart rate", self.record, "p", limit)
                self.assertEqual(packet["n_tokens"], count_tokens(packet["text"]))
                self.assertLessEqual(packet["n_tokens"], limit)
        with self.assertRaises(ValueError):
            build_starting_packet("heart rate", self.record, "p", 1)

    def test_compact_and_gzipped_ndjson_match_runtime_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "MimicPatient.ndjson").write_text(json.dumps(patient(), separators=(",", ":")) + "\n")
            with gzip.open(path / "MimicObservationChartevents.ndjson.gz", "wt") as handle:
                for item in self.record["Observation"]:
                    handle.write(json.dumps(item, separators=(",", ":")) + "\n")
            (path / "MimicEncounter.ndjson").write_text(json.dumps(self.record["Encounter"][0]) + "\n")
            shared, offsets = index_patient_lines(directory, {"p"})
            indexed = load_record(shared, offsets["p"])
            self.assertEqual(build_starting_packet("heart rate", indexed, "p", 2000),
                             build_starting_packet("heart rate", self.record, "p", 2000))

    def test_generation_and_sweep_do_not_require_gold_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "MimicPatient.ndjson").write_text(json.dumps(patient()) + "\n")
            questions = pd.DataFrame([
                {"question_id": name, "question": "patient gender", "assumption": "", "patient_fhir_id": "p"}
                for name in ("pilot", "dev", "held_out")])
            questions.to_csv(path / "questions.csv", index=False)
            pd.DataFrame([{"question_id": name, "family": "lookup_counting"}
                          for name in ("pilot", "dev", "held_out")]).to_csv(path / "families.csv", index=False)
            (path / "split.json").write_text(json.dumps({"splits": {
                name: {"question_ids": [name]} for name in ("pilot", "dev", "held_out")}}))
            output = path / "packets.jsonl"
            command = [sys.executable, str(ROOT / "scripts/build_starting_packets.py"),
                       "--input", str(path / "questions.csv"), "--families", str(path / "families.csv"),
                       "--split", str(path / "split.json"), "--fhir-dir", directory, "--output", str(output)]
            generated = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertNotIn("gold recall", generated.stdout)
            frozen = output.read_bytes()
            packets = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual({packet["split"] for packet in packets}, {"pilot", "dev", "held_out"})
            expected = build_starting_packet("patient gender", {"Patient": [patient()]}, "p")
            expected_hash = hashlib.sha256(expected["text"].encode("utf-8")).hexdigest()
            self.assertTrue(all(packet["packet_text_sha256"] == expected_hash for packet in packets))
            sweep = subprocess.run(command + ["--sweep", "400", "800"], capture_output=True, text=True, check=True)
            self.assertNotIn("gold recall", sweep.stdout)
            self.assertNotIn("[held_out", sweep.stdout)
            self.assertEqual(output.read_bytes(), frozen)
            questions["true_fhir_ids"] = ["{'Patient': ['p']}", "{'Patient': ['p']}", "must not parse held-out labels"]
            questions.to_csv(path / "questions.csv", index=False)
            subprocess.run(command, capture_output=True, text=True, check=True)
            frozen = output.read_bytes()
            diagnostics = subprocess.run(command + ["--gold-diagnostics"], capture_output=True, text=True, check=True)
            self.assertIn("[pilot] frozen gold-ID recall", diagnostics.stdout)
            self.assertIn("[dev] frozen gold-ID recall", diagnostics.stdout)
            self.assertNotIn("[held_out", diagnostics.stdout)
            self.assertEqual(output.read_bytes(), frozen)

    def test_frozen_manifest_matches_current_split_and_sources(self):
        split = json.loads((ROOT / "manifests/split.json").read_text())
        packets = [json.loads(line) for line in (ROOT / "manifests/starting_packets.jsonl").read_text().splitlines()]
        provenance = json.loads((ROOT / "manifests/starting_packets.source.json").read_text())
        fhir_hash = hashlib.sha256(json.dumps(provenance["files"], sort_keys=True).encode()).hexdigest()
        assignments = {q: name for name, details in split["splits"].items() for q in details["question_ids"]}
        patients = {name: {p["patient_fhir_id"] for p in details["patients"]}
                    for name, details in split["splits"].items()}
        self.assertEqual(len(packets), 370)
        self.assertEqual({packet["question_id"] for packet in packets}, set(assignments))
        source_hash = file_sha256(ROOT / "final_dataset/questions_answers_sql_fhir.csv")
        split_hash = file_sha256(ROOT / "manifests/split.json")
        retriever_hash = file_sha256(ROOT / "tools/starting_packet.py")
        for packet in packets:
            self.assertEqual(packet["split"], assignments[packet["question_id"]])
            self.assertIn(packet["patient_fhir_id"], patients[packet["split"]])
            self.assertEqual(packet["packet_version"], PACKET_VERSION)
            self.assertEqual(packet["max_tokens"], 8000)
            self.assertRegex(packet["packet_text_sha256"], r"^[0-9a-f]{64}$")
            self.assertLessEqual(packet["n_tokens"], 8000)
            self.assertEqual(packet["source_sha256"], source_hash)
            self.assertEqual(packet["split_sha256"], split_hash)
            self.assertEqual(packet["retriever_sha256"], retriever_hash)
            self.assertEqual(packet["fhir_source_sha256"], fhir_hash)


if __name__ == "__main__":
    unittest.main()
