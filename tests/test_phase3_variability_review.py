"""Synthetic regressions for fixed-subset sampling and repeat denominators."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import build_variability_subset as subset
from scripts import variability
from scripts import run_experiments as experiments


def row(answer, error=None, tokens=10, latency=1):
    return {"agent_answer": answer, "error": error, "tokens": {"total": tokens}, "latency": latency}


class SubsetReviewTests(unittest.TestCase):
    def setUp(self):
        self.split = {"seed": 5106, "splits": {"held_out": {"question_ids": ["a", "b", "c", "d"]}}}
        self.families = pd.DataFrame({"question_id": ["a", "b", "c", "d"],
                                      "family": ["lookup_counting", "lookup_counting",
                                                 "numerical_aggregation", "temporal_selection"]})

    def test_sampling_is_unique_and_deterministic(self):
        first = subset.build_subset(self.split, self.families, size=3)
        self.assertEqual(first, subset.build_subset(self.split, self.families.iloc[::-1], size=3))
        self.assertEqual(len(first[0]), len(set(first[0])))
        self.assertEqual(sum(first[1].values()), 3)

    def test_missing_or_duplicate_family_rows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            subset.build_subset(self.split, self.families.iloc[:-1], size=2)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            subset.build_subset(self.split, pd.concat([self.families, self.families.iloc[:1]]), size=2)

    def test_invalid_families_and_duplicate_split_ids_are_rejected(self):
        for family in (None, "out_of_scope"):
            invalid = self.families.copy()
            invalid.loc[0, "family"] = family
            with self.subTest(family=family), self.assertRaisesRegex(ValueError, "in-scope"):
                subset.build_subset(self.split, invalid, size=2)
        self.split["splits"]["held_out"]["question_ids"].append("a")
        with self.assertRaisesRegex(ValueError, "unique"):
            subset.build_subset(self.split, self.families, size=2)

    def test_invalid_sample_sizes_are_rejected(self):
        for size in (0, -1, True, 1.5, 5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                subset.build_subset(self.split, self.families, size=size)

    def test_cli_rejects_a_family_manifest_changed_after_split_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            families = path / "families.csv"
            self.families.to_csv(families, index=False)
            self.split["family_manifest_sha256"] = "different"
            split = path / "split.json"
            split.write_text(json.dumps(self.split), encoding="utf-8")
            with patch("sys.argv", ["build_subset", "--split", str(split), "--families", str(families),
                                    "--output", str(path / "subset.json")]), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "different family"):
                subset.main()
            self.assertFalse((path / "subset.json").exists())


class VariabilityReviewTests(unittest.TestCase):
    def test_report_keeps_fully_attempted_repeats_with_errors(self):
        arms = [arm for arm in experiments.ARMS if arm.group == "variability" and arm.strategy == "single_agent"][:1]
        frame = pd.DataFrame([{"question_id": "a", "question": "Synthetic count", "true_answer": "[[1]]",
                               "patient_fhir_id": "synthetic-patient"}])
        families = pd.DataFrame([{"question_id": "a", "template": "count", "family": "lookup_counting"}])
        results = {"a": row(None, error="provider failure")}
        with patch.object(experiments.run_agent, "read_input_data", return_value=frame), \
                patch("pandas.read_csv", return_value=families), \
                patch.object(experiments, "arm_results",
                             return_value=({"state": "in progress", "analyzable": True}, results)):
            report = experiments.variability_report(arms, type("Args", (), {
                "subset": "synthetic.json", "input": "synthetic.csv", "families": "synthetic-families.csv"})(),
                                                   {"variability": ["a"]})
        self.assertEqual(report["systems"]["single_agent"]["repeats"], 1)
        self.assertEqual(report["systems"]["single_agent"]["accuracy"]["mean"], 0)
        self.assertEqual(report["systems"]["single_agent"]["error_rate"]["mean"], 1)

    def test_errors_and_intentional_abstentions_have_separate_denominators(self):
        references = {qid: {"question": "How many synthetic visits?", "true_answer": "[]"}
                      for qid in ("error", "null", "text")}
        repeated = {"error": row(None, error="provider failure"), "null": row(None),
                    "text": row("Unable to determine")}
        summary = variability.summarize([repeated], references)
        self.assertEqual(summary["accuracy"]["mean"], 2 / 3)
        self.assertEqual(summary["abstention_rate"]["mean"], 2 / 3)
        self.assertEqual(summary["error_rate"]["mean"], 1 / 3)
        self.assertEqual(summary["identical_answers"], 2 / 3)

    def test_missing_usage_does_not_silently_drop_a_question_or_repeat(self):
        references = {qid: {"question": "Synthetic count", "true_answer": "[[1]]"} for qid in ("a", "b")}
        summary = variability.summarize([
            {"a": row(1), "b": row(1, tokens=None, latency=None)},
            {"a": row(1), "b": row(1, tokens=30, latency=3)}], references)
        self.assertIsNone(summary["per_repeat"][0]["mean_tokens"])
        self.assertIsNone(summary["per_repeat"][0]["median_latency"])
        self.assertEqual(summary["mean_tokens"]["n"], 1)
        self.assertEqual(summary["median_latency"]["n"], 1)
        self.assertEqual(summary["accuracy"]["n"], 2)

    def test_invalid_measures_and_empty_repeats_are_rejected(self):
        reference = {"a": {"question": "Synthetic count", "true_answer": "[[1]]"}}
        for value in (-1, float("inf"), float("nan"), True, "10"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite non-negative"):
                variability.summarize([{"a": row(1, tokens=value)}], reference)
        with self.assertRaisesRegex(ValueError, "at least one"):
            variability.summarize([{}], {})

    def test_failed_answers_do_not_agree_with_successful_abstentions(self):
        references = {"a": {"question": "Synthetic count", "true_answer": "[[1]]"}}
        summary = variability.summarize([{"a": row(None, error="failure")}, {"a": row(None)}], references)
        self.assertEqual(summary["identical_answers"], 0)
        self.assertEqual(summary["never_correct"], 1)
        self.assertEqual(summary["correctness_agreement"], 1)

    def test_retry_success_does_not_replace_a_first_attempt_failure(self):
        references = {"a": {"question": "Synthetic count", "true_answer": "[[1]]"}}
        saved = {**row(1, tokens=5, latency=2), "question_id": "a", "patient_fhir_id": "synthetic-patient",
                 "attempt_history": [row(None, error="failure", tokens=10, latency=3)]}
        summary = variability.summarize([{"a": saved}], references)
        self.assertEqual(variability.first_attempt(saved)["patient_fhir_id"], "synthetic-patient")
        self.assertEqual(summary["accuracy"]["mean"], 0)
        self.assertEqual(summary["error_rate"]["mean"], 1)
        self.assertEqual(summary["mean_tokens"]["mean"], 10)
        self.assertEqual(summary["median_latency"]["mean"], 3)
        self.assertEqual(summary["retry_inclusive_tokens_total"]["mean"], 15)
        self.assertEqual(summary["retry_inclusive_latency_total"]["mean"], 5)
        self.assertEqual(summary["retry_count"]["mean"], 1)
        self.assertEqual(summary["per_repeat"][0]["failed_attempts"], 1)

    def test_unknown_first_attempt_usage_is_not_filled_by_retry_usage(self):
        saved = {**row(1), "attempt_history": [{"error": "failure"}]}
        first = variability.first_attempt(saved)
        self.assertIsNone(first["tokens"])
        self.assertIsNone(first["latency"])
        self.assertIsNone(variability.cumulative_measure(saved, "tokens"))
        self.assertIsNone(variability.cumulative_measure(saved, "latency"))
        with self.assertRaisesRegex(ValueError, "history"):
            variability.first_attempt({"attempt_history": "malformed"})


if __name__ == "__main__":
    unittest.main()
