"""Synthetic full cohorts only: never read real benchmark questions or answers."""
import argparse
import ast
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Import the public scorer/analysis without utils' provider/configuration bootstrap.
utils_stub = types.ModuleType("utils")
utils_stub.__path__ = [str(ROOT / "utils")]
for node in ast.parse((ROOT / "utils/core_utils.py").read_text()).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RESULT_COLUMNS" for t in node.targets):
        utils_stub.RESULT_COLUMNS = ast.literal_eval(node.value)
prior_utils = sys.modules.get("utils")
sys.modules["utils"] = utils_stub
try:
    from scripts import compare_runs as m
    from scripts import evaluation_strata as strata
finally:
    if prior_utils is None:
        sys.modules.pop("utils", None)
    else:
        sys.modules["utils"] = prior_utils


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.input = self.root / "final_dataset/questions_answers_sql_fhir.csv"
        self.families = self.root / "manifests/question_families.csv"
        self.split = self.root / "manifests/split.json"
        self.exposure = self.root / "manifests/development_exposure.json"
        self.manifest_path = self.root / "manifests/evaluation.json"
        self.ids = ["a", "c1", "c2", "c3", "c4", "d", "p"]
        self.patients = {"d": "dev", "p": "pilot", "a": "A", "c1": "A", "c2": "A", "c3": "A", "c4": "B"}
        self.input.parent.mkdir(parents=True)
        self.families.parent.mkdir(parents=True)
        with self.input.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["question_id", "patient_fhir_id", "question", "true_answer"])
            writer.writeheader()
            for q in self.ids:
                writer.writerow({"question_id": q, "patient_fhir_id": self.patients[q],
                                 "question": "How many synthetic events?", "true_answer": "[[1]]"})
        with self.families.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["question_id", "family", "template"])
            writer.writeheader()
            for q in self.ids:
                writer.writerow({"question_id": q, "family": "out_of_scope" if q == "c4" else "lookup_counting",
                                 "template": "Count synthetic events"})
        write_json(self.split, {"in_scope_families": ["lookup_counting", "numerical_aggregation", "temporal_selection"],
            "splits": {"dev": {"patients": [{"patient_fhir_id": "dev", "subject_id": 1}], "question_ids": ["d"]},
                       "pilot": {"patients": [{"patient_fhir_id": "pilot", "subject_id": 2}], "question_ids": ["p"]},
                       "held_out": {"patients": [{"patient_fhir_id": "A", "subject_id": 3},
                                                  {"patient_fhir_id": "B", "subject_id": 4}], "question_ids": ["a"]}}})
        write_json(self.exposure, {"exposures": [{"subject_ids": [1]}]})
        def entry(ids):
            return {"question_ids": ids, "questions": len(ids), "patients": len({self.patients[q] for q in ids})}
        self.manifest = {"version": 1, "dataset_sha256": sha(self.input), "families_sha256": sha(self.families),
            "split_manifest_sha256": sha(self.split), "development_exposure_sha256": sha(self.exposure),
            "upstream_sha256": {"manifests/split.json": sha(self.split),
                                "manifests/development_exposure.json": sha(self.exposure)},
            "full_question_ids": self.ids,
            "splits": {"development_patients": entry(["d"]), "pilot_patients": entry(["p"]),
                       "adaptive_questions": entry(["a"]), "question_confirmation": entry(["c1", "c2", "c3", "c4"])},
            "views": {"confirmation_in_scope": entry(["c1", "c2", "c3"]),
                      "confirmation_other": entry(["c4"]), "nominal_new_patient": entry(["c4"])}}
        self.rows, self.paths = {}, {}
        for name in ("single", "team"):
            rows = []
            for q in self.ids:
                answer = 0 if (name == "single" and q in ("c1", "c2", "c3")) or (name == "team" and q == "c4") else 1
                rows.append({"question_id": q, "patient_fhir_id": self.patients[q], "agent_answer": answer,
                    "latency": 1.0, "error": None, "tokens": {"total": 10, "cost": None},
                    "budget": {"stopped": None}, "attempt_history": []})
            self.rows[name] = rows
            self.paths[name] = self.root / "output" / (name + ".json")
            write_json(self.paths[name], rows)
            write_json(Path(str(self.paths[name]) + ".run.json"),
                       {"model": "synthetic-model", "max_total_tokens": 200000, "max_tool_calls": 30,
                        "input_hash": "a" * 64, "split": None, "condition": None,
                        "subset_hash": None, "repeat": None, "code_hash": name,
                        "agent_strategy": name})
        self.save_manifest()

    def save_manifest(self):
        write_json(self.manifest_path, self.manifest)

    def args(self, enabled=True):
        return argparse.Namespace(input=str(self.input), families=str(self.families), split_manifest=str(self.split),
            split="held_out", full=True, arm=[n + "=" + str(p) for n, p in self.paths.items()], bootstrap=10000,
            evaluation_strata=str(self.manifest_path) if enabled else None,
            evaluation_strata_sha256=sha(self.manifest_path) if enabled else None,
            full_terminal_confirmed=enabled)

    def compare(self, args=None):
        with patch.object(m, "ROOT", self.root):
            return m.compare(args or self.args())

    def save_rows(self, name):
        write_json(self.paths[name], self.rows[name])


class FullEvaluationStrataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-full-strata-")
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(self.temp.name)

    def no_scoring(self, pattern, args=None):
        with patch.object(m, "arm_records", side_effect=AssertionError("scorer invoked before validation")) as called:
            with self.assertRaisesRegex(ValueError, pattern):
                self.f.compare(args)
            called.assert_not_called()

    def test_default_whole_cohort_report_unchanged(self):
        baseline = self.f.compare(self.f.args(False))
        with_strata = self.f.compare()
        with_strata.pop("evaluation_strata")
        self.assertEqual(with_strata, baseline)

    def test_exact_partition_and_question_weighted_patient_block_statistics(self):
        r = self.f.compare()["evaluation_strata"]
        self.assertEqual(r["manifest_sha256"], sha(self.f.manifest_path))
        c = r["reports"]["question_confirmation"]
        self.assertEqual(c["role"], "primary_stratum")
        self.assertEqual((c["arms"]["team"]["correct"], c["arms"]["single"]["correct"]), (3, 1))
        self.assertEqual(c["arms"]["team"]["n"], 4)
        p = c["comparisons"][0]
        self.assertEqual((p["wins"], p["losses"], p["estimate"], p["patients"]), (3, 1, .5, 2))
        # Patient A has three +1 differences; B has one -1. Question weighting is +.5, not0.
        expected = m.paired_difference([1, 1, 1, 0], [0, 0, 0, 1], ["A", "A", "A", "B"], n_boot=10000, seed=5106)
        self.assertEqual({k: p[k] for k in expected}, expected)

    def test_one_patient_view_has_unavailable_ci(self):
        v = self.f.compare()["evaluation_strata"]["reports"]["nominal_new_patient"]
        self.assertEqual(v["role"], "descriptive")
        for r in [*v["arms"].values(), *v["comparisons"]]:
            self.assertEqual(r["patients"], 1)
            self.assertIsNone(r["ci_low"])
            self.assertIsNone(r["ci_high"])

    def test_terminal_error_with_correct_answer_stays_wrong_and_in_denominator(self):
        self.f.rows["team"][1]["error"] = "synthetic timeout"
        self.f.save_rows("team")
        r = self.f.compare()["evaluation_strata"]["reports"]["question_confirmation"]["arms"]["team"]
        self.assertEqual((r["correct"], r["terminal_errors"], r["n"]), (2, 1, 4))

    def test_failed_first_attempt_is_not_replaced_by_later_correct_retry(self):
        row = self.f.rows["team"][1]
        row["attempt_history"] = [{"agent_answer": None, "error": "first attempt timeout", "latency": 1,
                                   "tokens": {"total": 5}, "budget": {"stopped": None}}]
        self.f.save_rows("team")
        r = self.f.compare()["evaluation_strata"]["reports"]["question_confirmation"]["arms"]["team"]
        self.assertEqual((r["correct"], r["terminal_errors"], r["n"]), (2, 1, 4))
        self.assertEqual(r["recorded_later_attempts"], 1)

    def test_partial_second_arm_fails_before_scoring_first_arm(self):
        self.f.rows["team"].pop()
        self.f.save_rows("team")
        self.no_scoring("wrong size")

    def test_unattempted_second_arm_fails_before_any_scoring(self):
        self.f.rows["team"][1]["latency"] = None
        self.f.save_rows("team")
        self.no_scoring("first attempt")

    def test_wrong_identity_fails_before_any_scoring(self):
        path = Path(str(self.f.paths["team"]) + ".run.json")
        d = json.loads(path.read_text()); d["model"] = "different-model"; write_json(path, d)
        self.no_scoring("identity differs")

    def test_absent_or_invalid_shared_identity_bindings_fail_before_scoring(self):
        for name in self.f.paths:
            p = Path(str(self.f.paths[name]) + ".run.json")
            d = json.loads(p.read_text())
            for key in ("model", "max_total_tokens", "max_tool_calls", "input_hash"): d.pop(key)
            write_json(p, d)
        self.no_scoring("explicit full-run identity")

    def test_invalid_full_identity_types_and_nonfull_scope_fail_closed(self):
        p = Path(str(self.f.paths["team"]) + ".run.json")
        original = json.loads(p.read_text())
        for key, value in (("max_tool_calls", True), ("model", ""), ("input_hash", None), ("split", "held_out")):
            write_json(p, dict(original, **{key: value}))
            self.no_scoring("identity|ordinary run")

    def test_invalid_second_arm_latency_or_usage_fails_before_scoring(self):
        row = self.f.rows["team"][1]
        for value in (-1, "1.0", True):
            row["latency"] = value; self.f.save_rows("team")
            self.no_scoring("first-attempt latency")
        row["latency"] = float("nan")
        self.f.paths["team"].write_text(json.dumps(self.f.rows["team"]))
        self.no_scoring("first-attempt latency")
        row["latency"] = 1.0
        row["tokens"]["total"] = -1; self.f.save_rows("team")
        self.no_scoring("token/cost")
        row["tokens"] = []; self.f.save_rows("team")
        self.no_scoring("usage/budget")

    def test_empty_family_label_fails_before_scoring(self):
        data = self.f.families.read_text().replace("c1,lookup_counting", "c1,")
        self.f.families.write_text(data)
        self.f.manifest["families_sha256"] = sha(self.f.families)
        self.f.save_manifest(); self.no_scoring("nonempty labels")

    def test_wrong_or_missing_second_arm_patient_fails_before_scoring(self):
        self.f.rows["team"][1]["patient_fhir_id"] = "wrong-patient"
        self.f.save_rows("team"); self.no_scoring("patient IDs differ")
        self.f.rows["team"][1].pop("patient_fhir_id")
        self.f.save_rows("team"); self.no_scoring("patient IDs differ")

    def test_duplicate_arm_name_rejected_before_scoring(self):
        args = self.f.args(); args.arm.append(args.arm[0])
        self.no_scoring("unique name", args)

    def test_wrong_manifest_digest_rejected(self):
        args = self.f.args(); args.evaluation_strata_sha256 = "0" * 64
        self.no_scoring("manifest digest differs", args)

    def test_manifest_requires_full_and_digest_pair(self):
        args = self.f.args(); args.full = False; self.no_scoring("together with --full", args)
        args = self.f.args(); args.evaluation_strata_sha256 = None; self.no_scoring("together with --full", args)

    def test_primary_overlap_and_equal_size_mislabeling_rejected(self):
        x = self.f.manifest["splits"]
        x["pilot_patients"]["question_ids"] = ["d"]
        self.f.save_manifest(); self.no_scoring("overlap|membership")
        self.f.manifest["splits"] = deepcopy(Fixture(self.f.root / "other").manifest["splits"])
        x = self.f.manifest["splits"]
        x["development_patients"]["question_ids"], x["pilot_patients"]["question_ids"] = ["p"], ["d"]
        self.f.save_manifest(); self.no_scoring("membership differs")

    def test_missing_or_duplicate_question_ids_rejected(self):
        self.f.manifest["splits"]["question_confirmation"]["question_ids"].pop()
        self.f.save_manifest(); self.no_scoring("declared questions|membership")
        self.f.manifest["splits"]["question_confirmation"]["question_ids"] = ["c1", "c1", "c2", "c3"]
        self.f.save_manifest(); self.no_scoring("duplicate question")

    def test_wrong_dataset_and_family_bindings_rejected(self):
        self.f.manifest["dataset_sha256"] = "0" * 64
        self.f.save_manifest(); self.no_scoring("dataset digest")
        self.f.manifest["dataset_sha256"] = sha(self.f.input)
        self.f.manifest["families_sha256"] = "0" * 64
        self.f.save_manifest(); self.no_scoring("families digest")

    def test_wrong_canonical_split_or_exposure_dependency_rejected(self):
        self.f.manifest["upstream_sha256"]["manifests/split.json"] = "0" * 64
        self.f.save_manifest(); self.no_scoring("dependency differs")
        self.f.manifest["upstream_sha256"]["manifests/split.json"] = sha(self.f.split)
        write_json(self.f.exposure, {"exposures": [{"subject_ids": [99]}]})
        self.f.manifest["development_exposure_sha256"] = sha(self.f.exposure)
        self.f.manifest["upstream_sha256"]["manifests/development_exposure.json"] = sha(self.f.exposure)
        self.f.save_manifest(); self.no_scoring("escaped development")

    def test_wrong_nested_scope_view_rejected(self):
        self.f.manifest["views"]["confirmation_in_scope"]["question_ids"] = ["c1", "c2", "c4"]
        self.f.manifest["views"]["confirmation_in_scope"]["patients"] = 2
        self.f.save_manifest(); self.no_scoring("scope views differ")

    def test_wrong_nominal_patient_view_rejected(self):
        self.f.manifest["views"]["nominal_new_patient"]["question_ids"] = ["c1"]
        self.f.save_manifest(); self.no_scoring("single metadata candidate")

    def test_views_optional_but_scope_pair_required(self):
        self.f.manifest.pop("views"); self.f.save_manifest()
        r = self.f.compare()["evaluation_strata"]
        self.assertEqual(len(r["reports"]), 4)
        self.f.manifest["views"] = {"confirmation_in_scope": {"question_ids": ["c1", "c2", "c3"]}}
        self.f.save_manifest(); self.no_scoring("required together")

    def test_bool_count_and_nonfinite_or_duplicate_json_rejected(self):
        self.f.manifest["splits"]["pilot_patients"]["questions"] = True
        self.f.save_manifest(); self.no_scoring("declared questions")
        self.f.manifest_path.write_text('{"version":1,"version":1}')
        self.no_scoring("Duplicate JSON")
        self.f.manifest_path.write_text('{"version":NaN}')
        self.no_scoring("Non-finite JSON")

    def test_source_and_families_duplicate_ids_rejected(self):
        with self.f.input.open("a") as f: f.write('c1,A,How many synthetic events?,[[1]]\n')
        self.no_scoring("unique question IDs")

    def test_operational_terminal_confirmation_required_before_scoring(self):
        args = self.f.args(); args.full_terminal_confirmed = False
        self.no_scoring("terminal/source/audit guard", args)

    def test_source_refusal_keeps_denominator_and_cannot_gain_credit(self):
        row = self.f.rows["team"][1]
        row.update(agent_answer=None, revision={"final_check_issues": ["source_scope"]})
        self.f.save_rows("team")
        r = self.f.compare()["evaluation_strata"]["reports"]["question_confirmation"]["arms"]["team"]
        self.assertEqual((r["correct"], r["terminal_errors"], r["n"]), (2, 0, 4))

    def test_confirmation_bootstrap_count_is_prespecified(self):
        args = self.f.args(); args.bootstrap = 10
        self.no_scoring("10000", args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
