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

import run_agent as runner
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
        self.assertEqual((same["estimate"], same["ci_low"], same["ci_high"], same["p_bootstrap"]), (0, 0, 0, 1))
        better = paired_difference([1] * 60, [1, 0, 0] * 20, patients, n_boot=500)
        self.assertAlmostEqual(better["estimate"], 2 / 3)
        self.assertGreater(better["ci_low"], 0)
        self.assertEqual(better["p_bootstrap"], 0)
        with self.assertRaisesRegex(ValueError, "same questions"):
            paired_difference([1, 0], [1], ["a", "a"])

    def test_empty_inputs_and_latency(self):
        self.assertIsNone(clustered_mean([], [])["estimate"])
        self.assertEqual(latency_summary([]), {"n": 0, "median": None, "p95": None})
        summary = latency_summary(list(range(1, 101)) + [None])
        self.assertEqual((summary["n"], summary["median"]), (100, 50.5))
        self.assertAlmostEqual(summary["p95"], 95.05)


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


class AnalyzeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cli = ["--model", "test-model", "--output_dir", self.temp.name, "--limit", "6", "--bootstrap", "200"]

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
        with open(os.path.join(directory, "latency.csv")) as stream:
            latency = {r["arm"]: r for r in csv.DictReader(stream)}
        self.assertEqual(float(latency["ordinary_multi_agent"]["median"]), 4.5)
        self.assertEqual(float(latency["ordinary_multi_agent"]["mean_tokens"]), 3500)
        with open(os.path.join(directory, "accuracy_vs_tokens.svg")) as stream:
            self.assertIn("Multi-agent 100%", stream.read())

    def test_bootstrap_must_be_positive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            experiments.main(["analyze", "--model", "m", "--bootstrap", "0"])


if __name__ == "__main__":
    unittest.main()
