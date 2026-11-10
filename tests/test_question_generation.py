"""New question sources: unsampled pools, the full-record oracle, generated
questions, finisher paraphrases, multi-record perturbations and graded demo questions."""
import ast
import json
import os
import random
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import run_agent
from agent.template_router import classify_template
from demo import backend
from scripts import revisions
from scripts import run_experiments as experiments
from scripts.build_paraphrases import FINISHER_TEMPLATES, paraphrase_rows, rewrite
from scripts.build_question_pools import POOLS, per_template, pool_manifest, stratified
from scripts.build_split import file_sha256
from scripts.fhir_oracle import RecordQuery, format_answer, full_packet, release
from scripts.generate_fhir_questions import instantiate, is_negative, slot_assignments, sql_literal

from tools.perturbation import load_perturbations, omit_from_packet, removed_pairs, text_sha256, verified_omission
from tools.starting_packet import build_starting_packet

ROOT = Path(__file__).resolve().parent.parent
PATIENT = "11111111-1111-1111-1111-111111111111"
LAB = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-d-labitems"


def lab(rid, label, value, when, encounter="h1"):
    return {"resourceType": "Observation", "id": rid, "subject": {"reference": f"Patient/{PATIENT}"},
            "encounter": {"reference": f"Encounter/{encounter}"},
            "code": {"coding": [{"system": LAB, "code": label[:3], "display": label}]},
            "effectiveDateTime": when, "valueQuantity": {"value": value, "unit": "mg/dL"}}


def record():
    visits = [{"resourceType": "Encounter", "id": hid, "subject": {"reference": f"Patient/{PATIENT}"},
               "identifier": [{"system": "http://mimic/encounter-hosp", "value": str(i)}],
               "period": {"start": start, "end": end}}
              for i, (hid, start, end) in enumerate((("h1", "2150-01-01T00:00:00", "2150-01-05T00:00:00"),
                                                     ("h2", "2150-03-01T00:00:00", "2150-03-04T00:00:00")), 1)]
    labs = [lab("a", "Glucose", 90, "2150-01-01T08:00:00"), lab("b", "Glucose", 150, "2150-01-02T08:00:00"),
            lab("c", "Sodium", 140, "2150-01-03T08:00:00"), lab("d", "Glucose", 120, "2150-03-02T08:00:00", "h2")]
    return {"Patient": [{"resourceType": "Patient", "id": PATIENT, "gender": "female"},
                        {"resourceType": "Patient", "id": "someone-else"}],
            "Encounter": visits, "Observation": labs}


FIRST_VISIT = ("labevents.hadm_id IN ( SELECT admissions.hadm_id FROM admissions WHERE admissions.subject_id = 1 "
               "AND admissions.dischtime IS NOT NULL ORDER BY admissions.admittime ASC LIMIT 1 )")
GLUCOSE = "labevents.itemid IN ( SELECT d_labitems.itemid FROM d_labitems WHERE d_labitems.label = 'Glucose' )"


def question(sql, **extra):
    return {"sql_query": sql, "main_table_name": "labevents", "assumption": None,
            "val_dict": repr({"val_placeholder": {"patient_id": 1, "lab_name": "glucose"}, "op_placeholder": {},
                              "time_placeholder": {}}), **extra}


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.packet = full_packet(record(), PATIENT)

    def tearDown(self):
        release(self.packet)

    def run_sql(self, sql):
        return RecordQuery(question(sql), self.packet).run_traced()

    def test_full_packet_keeps_only_the_patients_own_patient_resource(self):
        self.assertEqual([p["id"] for p in self.packet["resources"]["Patient"]], [PATIENT])

    def test_counts_and_aggregates_cite_every_reduced_row(self):
        answer, cited = self.run_sql(f"SELECT COUNT(*) FROM labevents WHERE {FIRST_VISIT} AND {GLUCOSE}")
        self.assertEqual(answer, [2])
        self.assertEqual(sorted(cited["Observation"]), ["a", "b"])
        answer, cited = self.run_sql(f"SELECT MAX(labevents.valuenum) FROM labevents WHERE {GLUCOSE}")
        self.assertEqual(answer, [150.0])
        self.assertEqual(sorted(cited["Observation"]), ["a", "b", "d"])

    def test_selections_cite_the_selected_row(self):
        answer, cited = self.run_sql(f"SELECT labevents.charttime FROM labevents WHERE {GLUCOSE} "
                                     "ORDER BY labevents.charttime DESC LIMIT 1")
        self.assertEqual(answer, ["2150-03-02 08:00:00"])
        self.assertEqual(cited, {"Observation": ["d"]})

    def test_paired_differences_cite_each_subquery(self):
        last = f"SELECT labevents.valuenum FROM labevents WHERE {GLUCOSE} ORDER BY labevents.charttime DESC LIMIT 1"
        first = f"SELECT labevents.valuenum FROM labevents WHERE {GLUCOSE} ORDER BY labevents.charttime ASC LIMIT 1"
        answer, cited = self.run_sql(f"SELECT ( {last} ) - ( {first} )")
        self.assertEqual(answer, [30.0])
        self.assertEqual(sorted(cited["Observation"]), ["a", "d"])

    def test_answers_use_the_released_reference_format(self):
        self.assertEqual(format_answer([0.1 + 0.2]), "[[0.3]]")
        self.assertEqual(format_answer(["a", "b"]), "[['a'], ['b']]")
        self.assertEqual(format_answer([]), "[]")


class GeneratorTests(unittest.TestCase):
    SKELETON = {
        "question_id": "q1", "patient_fhir_id": "source",
        "question": "What was the last glucose value of patient 10001 this month?",
        "sql_query": "SELECT labevents.valuenum FROM labevents WHERE labevents.subject_id = 10001 AND "
                     "d_labitems.label = 'Glucose' AND datetime(labevents.charttime,'start of month') = "
                     "datetime('2150-12-31 23:59:00','start of month','-0 month')",
        "assumption": "Assume the current time is 2150-12-31 23:59:00.",
        "val_dict": repr({"val_placeholder": {"patient_id": 10001, "lab_name": "glucose"}, "op_placeholder": {},
                          "time_placeholder": {"time_filter_global1": {
                              "nlq": "this month", "type": "rel-month-this",
                              "sql": "datetime('2150-12-31 23:59:00','start of month','-0 month')"}}}),
    }

    def test_instantiate_swaps_patient_slot_and_current_time(self):
        row = instantiate(self.SKELETON, {"patient_fhir_id": PATIENT, "subject_id": 10002}, "2160-06-30 23:59:00",
                          {"lab_name": "Sodium, Whole Blood"})
        self.assertEqual(row["question"], "What was the last sodium, whole blood value of patient 10002 this month?")
        self.assertIn("'Sodium, Whole Blood'", row["sql_query"])
        self.assertIn("subject_id = 10002", row["sql_query"])
        self.assertIn("2160-06-30", row["sql_query"])
        self.assertNotIn("2150-12-31", row["sql_query"])
        self.assertEqual(row["assumption"], "Assume the current time is 2160-06-30 23:59:00.")
        values = ast.literal_eval(row["val_dict"])
        self.assertEqual(values["val_placeholder"], {"patient_id": 10002, "lab_name": "sodium, whole blood"})
        self.assertIn("2160-06-30", values["time_placeholder"]["time_filter_global1"]["sql"])

    def test_a_time_dependent_skeleton_needs_the_target_patients_anchor(self):
        self.assertIsNone(instantiate(self.SKELETON, {"patient_fhir_id": PATIENT, "subject_id": 10002}, None, {}))

    def test_slot_literals_and_negative_answers(self):
        self.assertEqual(sql_literal("WHERE label = 'MRSA SCREEN'", "mrsa screen"), "MRSA SCREEN")
        self.assertIsNone(sql_literal("WHERE label = 'Glucose'", "sodium"))
        self.assertTrue(all(is_negative(v) for v in ([], [None], [0], [0.0])))
        self.assertFalse(any(is_negative(v) for v in ([1], ["Heparin"], [0, 1])))

    def test_multi_slot_assignments_use_distinct_record_values(self):
        skeleton = {"val_dict": repr({"val_placeholder": {"patient_id": 1, "drug_name1": "a", "drug_name2": "b"}})}
        choices = slot_assignments(skeleton, {"drug_name": ["X", "Y", "Z"]}, random.Random(0))
        self.assertTrue(choices)
        self.assertTrue(all(c["drug_name1"] != c["drug_name2"] for c in choices))
        unknown = {"val_dict": repr({"val_placeholder": {"patient_id": 1, "diagnosis_name": "x"}})}
        self.assertEqual(slot_assignments(unknown, {}, random.Random(0)), [])

    def test_generated_bank_is_runnable_when_built(self):
        path = ROOT / "manifests" / "generated" / "fhir_native.split.json"
        if not path.exists():
            self.skipTest("run scripts/generate_fhir_questions.py first")
        manifest = json.loads(path.read_text())
        frame = run_agent.load_inputs(str(ROOT / manifest["source"]), split_manifest=str(path), split="held_out")
        self.assertEqual(len(frame), len(manifest["splits"]["held_out"]["question_ids"]))
        released = pd.read_csv(ROOT / "final_dataset" / "questions_answers_sql_fhir.csv", usecols=["question_id"])
        self.assertFalse(set(frame.question_id) & set(released.question_id))


class PoolTests(unittest.TestCase):
    def test_stratified_sample_moves_quota_from_small_families(self):
        pool = pd.DataFrame({"question_id": range(30), "family": ["a"] * 2 + ["b"] * 28})
        mix = pd.Series({"a": 50, "b": 50})
        chosen = stratified(pool, 20, mix, 1)
        self.assertEqual(chosen.family.value_counts().to_dict(), {"b": 18, "a": 2})
        self.assertEqual(len(stratified(pool, 40, mix, 1)), 30)

    def test_per_template_caps_each_template(self):
        pool = pd.DataFrame({"question_id": range(9), "template_id": ["T1"] * 7 + ["T2"] * 2})
        self.assertEqual(per_template(pool, 3, 0).template_id.value_counts().to_dict(), {"T1": 3, "T2": 2})

    def test_built_pools_are_fresh_held_out_questions(self):
        split_path = ROOT / "manifests" / "split.json"
        split = json.loads(split_path.read_text())
        asked = set(split["splits"]["held_out"]["question_ids"])
        found = [name for name in POOLS if (ROOT / "manifests" / "pools" / f"{name}.json").exists()]
        if not found:
            self.skipTest("run scripts/build_question_pools.py first")
        for name in found:
            with self.subTest(pool=name):
                path = ROOT / "manifests" / "pools" / f"{name}.json"
                manifest = json.loads(path.read_text())
                self.assertEqual(manifest["pool"]["base_split_sha256"], file_sha256(split_path))
                frame = run_agent.load_inputs(str(ROOT / manifest["source"]), split_manifest=str(path), split="held_out")
                self.assertFalse(set(frame.question_id) & asked)
                self.assertEqual(manifest["splits"]["dev"], split["splits"]["dev"])

    def test_pool_manifest_replaces_only_held_out_questions(self):
        split_path = ROOT / "manifests" / "split.json"
        split = json.loads(split_path.read_text())
        chosen = pd.DataFrame({"question_id": ["x"], "family": ["out_of_scope"], "answer_kind": ["value"],
                               "patient_fhir_id": ["p"], "template_id": ["T023"]})
        manifest = pool_manifest(split, str(split_path), "test", chosen, "criteria")
        self.assertEqual(manifest["splits"]["held_out"]["question_ids"], ["x"])
        self.assertEqual(manifest["splits"]["pilot"], split["splits"]["pilot"])
        self.assertEqual(manifest["in_scope_families"], ["out_of_scope"])

    def test_size_check_applies_only_to_the_frozen_split(self):
        args = experiments.build_parser().parse_args(["status", "--split_manifest", "manifests/pools/held_out_v2.json"])
        statuses = [{"group": "ordinary", "questions": 1}]
        experiments.check_sizes(statuses, experiments.ARMS, args)  # no error for a pool of another size
        default = experiments.build_parser().parse_args(["status"])
        with self.assertRaises(ValueError):
            experiments.check_sizes(statuses, experiments.ARMS, default)


class ParaphraseTests(unittest.TestCase):
    def test_routed_rewrites_keep_their_finisher_and_unrouted_rewrites_lose_it(self):
        df = pd.read_csv(ROOT / "final_dataset" / "questions_answers_sql_fhir.csv")
        families = pd.read_csv(ROOT / "manifests" / "question_families.csv", usecols=["question_id", "template_id"])
        df = df.merge(families, on="question_id")
        finisher = df[df.template_id.isin(FINISHER_TEMPLATES)]
        rows = paraphrase_rows(finisher)
        self.assertEqual(len(rows), 2 * len(finisher))
        for row in rows.itertuples():
            expected = row.template_id if row.variant == "routed" else None
            self.assertEqual(classify_template(row.question), expected, row.question)
        self.assertTrue((rows.true_answer.values == finisher.set_index("question_id").loc[
            rows.source_question_id, "true_answer"].values).all())

    def test_unrouted_scopes_avoid_first_and_last(self):
        values = {"val_placeholder": {"patient_id": 7, "drug_route": "iv"}, "time_placeholder": {
            "time_filter_exact1": {"type": "exact-last"},
            "time_filter_global1": {"nlq": "on the first hospital visit"}}}
        text = rewrite("T052", values, "unrouted")
        self.assertEqual(text, "Which medication(s) given by the iv route to patient 7 have the newest prescription "
                               "start time on the oldest hospital visit?")
        self.assertIsNone(rewrite("T051", {"val_placeholder": {"patient_id": 7}, "time_placeholder": {}}, "routed"))


class MultiPerturbationTests(unittest.TestCase):
    def test_frozen_entries_remove_only_their_records(self):
        path = ROOT / "manifests" / "perturbations_multi.json"
        if not path.exists():
            self.skipTest("run scripts/build_multi_perturbations.py first")
        manifest = json.loads(path.read_text())
        self.assertEqual(manifest["version"], 3)
        packets = {p["question_id"]: p for p in map(json.loads, (ROOT / "manifests" / "starting_packets.jsonl")
                                                    .read_text().splitlines())}
        held_out = set(json.loads((ROOT / "manifests" / "split.json").read_text())["splits"]["held_out"]["question_ids"])
        for entry in manifest["perturbations"]:
            with self.subTest(question=entry["question_id"], kind=entry["kind"]):
                self.assertIn(entry["question_id"], held_out)
                removed = {(r["resource_type"], r["id"]) for r in entry["removed"]}
                self.assertEqual(len(removed), len(entry["removed"]))
                if entry["kind"] != "link_target":
                    self.assertGreaterEqual(len(removed), 1 if entry["kind"] == "tie_peer" else 2)
                original = packets[entry["question_id"]]["resource_ids"]
                self.assertTrue(all(rid in original.get(rtype, []) for rtype, rid in removed))
                expected = {t: [i for i in ids if (t, i) not in removed] for t, ids in original.items()}
                if entry["kind"] != "link_target":  # link omissions also redact references
                    self.assertEqual(entry["perturbed_resource_ids"], {t: ids for t, ids in expected.items() if ids})
                self.assertNotEqual(entry["proof"]["original_result_sha256"], entry["proof"]["perturbed_result_sha256"])


class MultiRuntimeTests(unittest.TestCase):
    SPEC = {"removed": [{"resource_type": "Observation", "id": "a"}, {"resource_type": "Observation", "id": "b"}]}

    def test_removed_pairs_read_both_protocols(self):
        self.assertEqual(removed_pairs({"removed": {"resource_type": "Observation", "id": "a"}}), [("Observation", "a")])
        self.assertEqual(removed_pairs(self.SPEC), [("Observation", "a"), ("Observation", "b")])

    def test_verified_omission_leaves_out_every_listed_record(self):
        packet = build_starting_packet("Glucose lab value", record(), PATIENT)
        omitted = omit_from_packet(packet, removed_pairs(self.SPEC))
        spec = {**self.SPEC, "original_packet_sha256": text_sha256(packet["text"]),
                "perturbed_packet_sha256": text_sha256(omitted["text"]),
                "perturbed_resource_ids": omitted["resource_ids"], "perturbed_n_tokens": omitted["n_tokens"]}
        result = verified_omission(packet, spec)
        self.assertNotIn("a", result["resource_ids"]["Observation"])
        self.assertNotIn("b", result["resource_ids"]["Observation"])
        self.assertIn("d", result["resource_ids"]["Observation"])

    def test_recovery_flags_need_every_removed_record(self):
        reference = {"question_id": "q1", "question": "Q", "true_answer": "[[1]]", "template": "",
                     "family": "lookup_counting"}
        base = {"agent_answer": "1", "evidence": [], "revision": None, "error": None}
        partial = revisions.question_record("arm", {**base, "agent_fhir_resources": {"Observation": ["a"]}},
                                            reference, self.SPEC)
        self.assertEqual(partial["omitted_resource"], "Observation/a,Observation/b")
        self.assertFalse(partial["omitted_retrieved"])
        both = revisions.question_record("arm", {**base, "agent_fhir_resources": {"Observation": ["a", "b"]}},
                                         reference, self.SPEC)
        self.assertTrue(both["omitted_retrieved"])

    def test_the_frozen_multi_record_manifest_loads(self):
        path = ROOT / "manifests" / "perturbations_multi.json"
        if not path.exists():
            self.skipTest("run scripts/build_multi_perturbations.py first")
        entries = load_perturbations(str(path))
        self.assertEqual(len(entries), json.loads(path.read_text())["size"])
        self.assertEqual(len(load_perturbations(str(ROOT / "manifests" / "perturbations.json"))), 50)


class DemoGeneratedTests(unittest.TestCase):
    def test_generated_questions_are_graded_but_never_for_held_out_patients(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "generated.csv")
            pd.DataFrame([{"question_id": "g1", "question": "How many glucose tests did patient 1 have?",
                           "assumption": "Assume the current time is 2150-12-31 23:59:00.", "patient_fhir_id": "p1",
                           "true_answer": "[[3]]", "template": "Count the number of times patient {patient_id} ..."}]
                         ).to_csv(path, index=False)
            patient = {"patient_fhir_id": "p1", "split": "dev"}
            questions = backend.generated_questions(patient, path)
            self.assertEqual(questions, [{"question_id": "g1", "question": "How many glucose tests did patient 1 have?",
                                          "current_time": "2150-12-31 23:59:00"}])
            self.assertNotIn("true_answer", questions[0])
            self.assertTrue(backend.grade("g1", patient, 3, path)["correct"])
            self.assertFalse(backend.grade("g1", patient, 4, path)["correct"])
            held_out = {**patient, "split": "held_out"}
            self.assertEqual(backend.generated_questions(held_out, path), [])
            with self.assertRaises(ValueError):
                backend.grade("g1", held_out, 3, path)
            with self.assertRaises(ValueError):
                backend.grade("g1", {"patient_fhir_id": "p2", "split": "dev"}, 3, path)


if __name__ == "__main__":
    unittest.main()
