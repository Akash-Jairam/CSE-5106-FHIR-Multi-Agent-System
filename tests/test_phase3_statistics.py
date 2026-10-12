"""P3-21: patient-clustered bootstrap CIs, paired differences, latency, and the accuracy-vs-tokens chart.

No model or FHIR store is called, and no answer is written or printed.
"""
import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_agent as runner
from scripts import analysis
from scripts import run_experiments as experiments
from scripts.experiment_plots import accuracy_vs_tokens_svg
from scripts.experiment_stats import clustered_mean, latency_summary, paired_difference


class BootstrapTests(unittest.TestCase):
    def test_estimate_interval_and_reproducibility(self):
        values = [1, 0, 1, 1, 0, 1, 0, 1]
        patients = ["a", "a", "b", "b", "c", "c", "d", "d"]
        result = clustered_mean(values, patients, n_boot=2000, seed=7)
        self.assertEqual((result["n"], result["patients"], result["estimate"]), (8, 4, 0.625))
        self.assertLessEqual(result["ci_low"], result["estimate"])
        self.assertGreaterEqual(result["ci_high"], result["estimate"])
        self.assertEqual(result, clustered_mean(values, patients, n_boot=2000, seed=7))

    def test_clustering_widens_the_interval_for_correlated_patients(self):
        # Same 50% accuracy: 10 patients whose 10 questions all agree, vs 100 independent patients
        clustered = clustered_mean([p % 2 for p in range(10) for _ in range(10)],
                                   [p for p in range(10) for _ in range(10)], n_boot=4000, seed=1)
        independent = clustered_mean([q % 2 for q in range(100)], list(range(100)), n_boot=4000, seed=1)
        width = lambda r: r["ci_high"] - r["ci_low"]
        self.assertGreater(width(clustered), 2 * width(independent))

    def test_paired_difference(self):
        patients = [p for p in range(20) for _ in range(3)]
        same = paired_difference([1, 0, 1] * 20, [1, 0, 1] * 20, patients, n_boot=500)
        self.assertEqual((same["estimate"], same["ci_low"], same["ci_high"]), (0, 0, 0))
        better = paired_difference([1] * 60, [1, 0, 0] * 20, patients, n_boot=500)
        self.assertAlmostEqual(better["estimate"], 2 / 3)
        self.assertGreater(better["ci_low"], 0)
        self.assertNotIn("p_bootstrap", better)
        # The former empirical tail was 0 even for two constant patient effects.
        tiny = paired_difference([1, 1], [0, 0], ["a", "b"], n_boot=200)
        self.assertEqual((tiny["patients"], tiny["ci_low"], tiny["ci_high"]), (2, 1, 1))
        self.assertNotIn("p_bootstrap", tiny)
        with self.assertRaisesRegex(ValueError, "same questions"):
            paired_difference([1, 0], [1], ["a", "a"])

    def test_empty_inputs_and_latency(self):
        self.assertIsNone(clustered_mean([], [])["estimate"])
        self.assertEqual(latency_summary([]), {"n": 0, "median": None, "p95": None})
        summary = latency_summary(list(range(1, 101)) + [None])
        self.assertEqual((summary["n"], summary["median"]), (100, 50.5))
        self.assertAlmostEqual(summary["p95"], 95.05)


    def test_unequal_clusters_preserve_the_question_weighted_estimand(self):
        result = clustered_mean([1] * 9 + [0], ["a"] * 9 + ["b"], n_boot=2000, seed=7)
        self.assertEqual((result["estimate"], result["ci_low"], result["ci_high"]), (0.9, 0, 1))

    def test_one_patient_has_an_estimate_but_no_between_patient_interval(self):
        result = clustered_mean([1, 0, 1], ["a"] * 3, n_boot=200)
        self.assertEqual((result["n"], result["patients"], result["estimate"]), (3, 1, 2 / 3))
        self.assertIsNone(result["ci_low"])
        self.assertIsNone(result["ci_high"])

    def test_invalid_observations_and_resample_counts_are_rejected(self):
        for values, patients in (([1], []), ([], ["a"]), ([[1]], ["a"]), ([np.nan], ["a"]),
                                 ([np.inf], ["a"]), ([1], [None]), ([1], [""]), ([1], [np.nan])):
            with self.subTest(values=values, patients=patients), self.assertRaises(ValueError):
                clustered_mean(values, patients)
        for n_boot in (0, -1, 1.5, True):
            with self.subTest(n_boot=n_boot), self.assertRaisesRegex(ValueError, "positive integer"):
                clustered_mean([1, 0], ["a", "b"], n_boot=n_boot)
        with self.assertRaisesRegex(ValueError, "finite"):
            paired_difference([1, 0], [1, np.nan], ["a", "b"])
        for latency in ([np.nan], [np.inf], [-1], [[1]]):
            with self.subTest(latency=latency), self.assertRaisesRegex(ValueError, "latencies"):
                latency_summary(latency)


def point(accuracy, tokens):
    ci = lambda v, spread: {"n": 300, "patients": 58, "estimate": v, "ci_low": v - spread, "ci_high": v + spread}
    return {"accuracy": ci(accuracy, 0.05), "tokens": ci(tokens, tokens / 10)}


class ChartTests(unittest.TestCase):
    def test_every_system_is_labeled_with_a_tooltip_and_fixed_color(self):
        svg = accuracy_vs_tokens_svg({"direct_llm": point(0.45, 6000), "single_agent": point(0.57, 50000),
                                      "multi_agent": point(0.69, 95000)}, subtitle="model & split")
        for label in ("Direct LLM 45%", "Single agent 57%", "Multi-agent 69%"):
            self.assertIn(label, svg)
        self.assertEqual(svg.count("<title>"), 3)
        self.assertIn("model &amp; split", svg)
        self.assertIn("prefers-color-scheme: dark", svg)
        # Color follows the system: a lone multi-agent dot still uses its own slot
        lone = accuracy_vs_tokens_svg({"multi_agent": point(0.69, 95000)})
        self.assertIn('fill="var(--multi_agent)"', lone)
        self.assertNotIn('fill="var(--direct_llm)"', lone)

    def test_no_points(self):
        self.assertIn("No complete arms to plot yet", accuracy_vs_tokens_svg({}))


    def test_zero_token_cost_and_zero_lower_bound_are_not_dropped(self):
        zero = {"accuracy": clustered_mean([0, 0], ["a", "b"], n_boot=200),
                "tokens": clustered_mean([0, 0], ["a", "b"], n_boot=200)}
        svg = accuracy_vs_tokens_svg({"direct_llm": zero, "multi_agent": point(0.69, 95000)})
        self.assertIn("Direct LLM 0%", svg)
        self.assertIn("linear scale", svg)
        self.assertEqual(svg.count("<title>"), 2)

    def test_one_patient_is_plotted_without_fake_whiskers(self):
        single = {"accuracy": clustered_mean([1], ["a"], n_boot=200),
                  "tokens": clustered_mean([100], ["a"], n_boot=200)}
        svg = accuracy_vs_tokens_svg({"direct_llm": single})
        self.assertIn("Direct LLM 100%", svg)
        self.assertIn("95% CI unavailable", svg)
        self.assertNotIn('stroke-width="2" stroke-linecap="round"', svg)


class PairingTests(unittest.TestCase):
    def test_pairing_requires_unique_ids_and_the_exact_question_set(self):
        a = {"question_id": "a", "patient": "p", "family": "lookup_counting"}
        b = {**a, "question_id": "b"}
        for left, right in (([a, a], [a]), ([a, b], [a]), ([a], [b])):
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                analysis._paired_records(left, right)
        self.assertEqual(analysis._paired_records([a, b], [b, a]), [(a, a), (b, b)])

    def test_pairing_rejects_patient_and_family_changes(self):
        a = {"question_id": "a", "patient": "p", "family": "lookup_counting"}
        for changed in ({**a, "patient": "other"}, {**a, "family": "temporal_selection"}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "patient and family"):
                analysis._paired_records([a], [changed])


class AnalyzeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        ids = [f"q{i}" for i in range(6)]
        patients = [f"p{i // 2}" for i in range(6)]
        frame = pd.DataFrame({"question_id": ids, "patient_fhir_id": patients, "question": ["Synthetic Q"] * 6,
                              "assumption": [None] * 6, "true_answer": ["[[1]]"] * 6})
        family_frame = pd.DataFrame({"question_id": ids, "family": ["lookup_counting", "temporal_selection",
                                                                        "numerical_aggregation"] * 2,
                                    "template": ["Synthetic Q"] * 6})
        paths = {name: os.path.join(self.temp.name, name + ".json") for name in ("split", "perturbations")}
        with open(paths["split"], "w") as stream:
            json.dump({"seed": 7, "splits": {"held_out": {"question_ids": ids,
                      "patients": [{"patient_fhir_id": p} for p in sorted(set(patients))]}}}, stream)
        with open(paths["perturbations"], "w") as stream:
            json.dump({"version": 2}, stream)
        input_path = os.path.join(self.temp.name, "input.csv")
        families_path = os.path.join(self.temp.name, "families.csv")
        frame.to_csv(input_path, index=False)
        family_frame.to_csv(families_path, index=False)
        perturbations = patch("tools.perturbation.load_perturbations", return_value={qid: {} for qid in ids})
        perturbations.start()
        self.addCleanup(perturbations.stop)
        self.cli = ["--model", "test-model", "--output_dir", self.temp.name, "--limit", "6", "--bootstrap", "200",
                    "--input", input_path, "--families", families_path, "--split_manifest", paths["split"],
                    "--perturbations", paths["perturbations"]]

    def write_arm(self, arm, answers):
        args = experiments.build_parser().parse_args(["status", *self.cli])
        settings = runner.build_parser().parse_args(experiments.runner_argv(arm, args))
        frame = experiments.arm_inputs(arm, settings, experiments.load_question_sets([arm], args))
        identity = runner.run_identity(settings, frame)
        results = runner.prepare_results(frame, settings.output, identity)
        for index, answer in zip(results.index, answers):
            results.at[index, "agent_answer"] = answer
            results.at[index, "latency"] = 2.0 + index
            results.at[index, "tokens"] = {"total": 1000 * (index + 1)}
        with contextlib.redirect_stdout(io.StringIO()):
            runner.save_results_json(results, settings.output)
        runner._write_json(identity, settings.output + ".run.json")

    def test_analysis_tables_comparisons_and_chart_without_answers(self):
        arms = {arm.name: arm for arm in experiments.ARMS}
        self.write_arm(arms["ordinary_multi_agent"], ["SECRET-1"] * 6)
        self.write_arm(arms["ordinary_single_agent"], ["SECRET-1", "SECRET-0"] * 3)
        self.write_arm(arms["recovery_multi_agent_intact"], ["SECRET-1"] * 6)
        self.write_arm(arms["recovery_multi_agent_omitted"], ["SECRET-0"] * 6)
        self.write_arm(arms["ordinary_direct_llm"], ["SECRET-1"] * 2)  # incomplete: skipped
        with patch("scripts.analysis.score_row", side_effect=lambda row, ref: int(row["agent_answer"] == "SECRET-1")), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(experiments.main(["analyze", *self.cli]), 0)
        directory = os.path.join(self.temp.name, "analysis")
        written = "".join(Path(directory, name).read_text() for name in os.listdir(directory))
        self.assertNotIn("SECRET", out.getvalue() + written)

        with open(os.path.join(directory, "summary.json")) as stream:
            report = json.load(stream)
        self.assertEqual(report["analyzed"], ["ordinary_multi_agent", "ordinary_single_agent",
                                              "recovery_multi_agent_intact", "recovery_multi_agent_omitted"])
        self.assertIn("ordinary_direct_llm", report["skipped"])
        overall = {(r["comparison"]): r for r in report["comparisons"] if r["family"] == "all"}
        self.assertEqual(set(overall), {"multi_agent - single_agent", "multi_agent: omitted - intact"})
        self.assertAlmostEqual(overall["multi_agent - single_agent"]["estimate"], 0.5)
        self.assertEqual(overall["multi_agent: omitted - intact"]["estimate"], -1.0)
        with open(os.path.join(directory, "accuracy.csv")) as stream:
            rows = list(csv.DictReader(stream))
        self.assertIn(("ordinary_single_agent", "all", "0.5"), {(r["arm"], r["family"], r["estimate"]) for r in rows})
        self.assertTrue({r["family"] for r in rows} > {"all"})
        with open(os.path.join(directory, "comparisons.csv")) as stream:
            self.assertNotIn("p_bootstrap", next(csv.reader(stream)))
        with open(os.path.join(directory, "latency.csv")) as stream:
            latency = {r["arm"]: r for r in csv.DictReader(stream)}
        self.assertEqual(float(latency["ordinary_multi_agent"]["median"]), 4.5)
        self.assertEqual(float(latency["ordinary_multi_agent"]["mean_tokens"]), 3500)
        with open(os.path.join(directory, "accuracy_vs_tokens.svg")) as stream:
            self.assertIn("Multi-agent 100%", stream.read())

    def test_bootstrap_must_be_positive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            experiments.main(["analyze", "--model", "m", "--bootstrap", "0"])


    def synthetic_report(self, rows, state="complete", attempted=6):
        arm = next(a for a in experiments.ARMS if a.name == "ordinary_single_agent")
        args = experiments.build_parser().parse_args(["analyze", *self.cli, "--arms", arm.name])
        references = {qid: {"patient": f"p{i // 2}", "family": "lookup_counting", "question": "Synthetic Q",
                             "true_answer": "[[1]]", "template": ""} for i, qid in enumerate(rows)}
        status = {"state": state, "questions": 6, "attempted": attempted, "failed": 0,
                  "analyzable": attempted == 6}
        with patch.object(experiments, "arm_results", return_value=(status, rows)), \
                patch.object(experiments, "load_references", return_value=references):
            return analysis.analyze([arm], args, {})

    def result_rows(self):
        return {f"q{i}": {"question_id": f"q{i}", "patient_fhir_id": f"p{i // 2}", "agent_answer": "1",
                           "error": None, "tokens": {"total": 10}, "latency": 1.0} for i in range(6)}

    def test_every_attempted_failure_stays_in_the_denominator_and_unknown_usage_is_visible(self):
        rows = self.result_rows()
        rows["q0"].update({"agent_answer": None, "error": "synthetic provider failure", "tokens": None, "latency": None})
        report = self.synthetic_report(rows, state="in progress")
        overall = next(r for r in report["accuracy"] if r["family"] == "all")
        self.assertEqual((overall["n"], overall["estimate"]), (6, 5 / 6))
        self.assertEqual(report["failed_attempts"], {"ordinary_single_agent": 1})
        cost = report["latency"][0]
        self.assertEqual((cost["questions"], cost["n"], cost["missing_latency"], cost["tokens_n"],
                          cost["missing_tokens"]), (6, 5, 1, 5, 1))
        self.assertIsNone(cost["mean_tokens"])
        self.assertIsNone(cost["retry_inclusive_tokens_total"])
        self.assertIsNone(cost["retry_inclusive_latency_total"])
        self.assertEqual(self.synthetic_report(rows, state="in progress", attempted=5)["analyzed"], [])

    def test_retries_preserve_first_attempt_outcomes_and_report_extra_cost_separately(self):
        rows = self.result_rows()
        rows["q0"].update({"tokens": {"total": 100}, "latency": 1.0, "attempt_history": [
            {"agent_answer": None, "error": "synthetic failure", "tokens": {"total": 20}, "latency": 2.0}]})
        report = self.synthetic_report(rows)
        overall = next(r for r in report["accuracy"] if r["family"] == "all")
        self.assertEqual(overall["estimate"], 5 / 6)
        self.assertEqual(report["failed_attempts"]["ordinary_single_agent"], 1)
        cost = report["latency"][0]
        self.assertEqual((cost["mean_tokens"], cost["retry_count"], cost["retry_inclusive_tokens_total"],
                          cost["retry_inclusive_latency_total"]), (70 / 6, 1, 170, 8))
        self.assertEqual(report["plot_points"]["single_agent"]["tokens"]["estimate"], 70 / 6)

    def test_saved_patient_mismatch_and_missing_references_are_rejected(self):
        rows = self.result_rows()
        rows["q0"]["patient_fhir_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "patients"):
            self.synthetic_report(rows)
        with self.assertRaisesRegex(ValueError, "authoritative reference"):
            analysis.arm_records({"q": {}}, {})


if __name__ == "__main__":
    unittest.main()
