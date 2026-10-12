"""P3-20: a fixed held-out subset, repeated N times per system, and its variance report.

No model or FHIR store is called, and no reference answer is printed.
"""
import contextlib
import io
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import run_agent as runner
from scripts import run_experiments as experiments
from scripts import variability
from scripts.build_split import file_sha256
from scripts.build_variability_subset import REPEATS, SUBSET_SIZE, build_subset

SUBSET = experiments.DEFAULT_SUBSET
SPLIT = experiments.DEFAULT_SPLIT


class SubsetTests(unittest.TestCase):
    def test_frozen_subset_rebuilds_from_the_current_split(self):
        with open(SUBSET) as stream:
            manifest = json.load(stream)
        with open(SPLIT) as stream:
            split = json.load(stream)
        families = pd.read_csv(experiments.DEFAULT_FAMILIES, usecols=["question_id", "family"])
        question_ids, family_counts = build_subset(split, families)
        self.assertEqual(manifest["question_ids"], question_ids)
        self.assertEqual(manifest["family_counts"], family_counts)
        self.assertEqual((manifest["size"], manifest["repeats"]), (SUBSET_SIZE, REPEATS))
        self.assertEqual(REPEATS, experiments.REPEATS)
        self.assertEqual(manifest["split_sha256"], file_sha256(SPLIT))
        self.assertTrue(set(question_ids) <= set(split["splits"]["held_out"]["question_ids"]))
        self.assertEqual(set(family_counts), {"lookup_counting", "numerical_aggregation", "temporal_selection"})

    def test_subset_must_match_its_split(self):
        self.assertEqual(len(runner.load_subset(SUBSET, SPLIT, "held_out")), SUBSET_SIZE)
        with self.assertRaisesRegex(ValueError, "not pilot"):
            runner.load_subset(SUBSET, SPLIT, "pilot")
        with tempfile.TemporaryDirectory() as directory:
            other_split = os.path.join(directory, "split.json")
            with open(other_split, "w") as stream:
                json.dump({}, stream)
            with self.assertRaisesRegex(ValueError, "different split manifest"):
                runner.load_subset(SUBSET, other_split, "held_out")
            malformed = os.path.join(directory, "subset.json")
            with open(malformed, "w") as stream:
                json.dump({"version": 1, "size": 3, "question_ids": ["q1"]}, stream)
            with self.assertRaisesRegex(ValueError, "malformed"):
                runner.load_subset(malformed, SPLIT, "held_out")


class RunnerTests(unittest.TestCase):
    def test_subset_and_repeat_options_are_validated(self):
        base = ["--agent_strategy", "direct_llm", "--model", "m"]
        split = ["--split_manifest", SPLIT, "--split", "held_out"]
        for extra in (["--subset", SUBSET], ["--repeat", "1"], [*split, "--repeat", "1"],
                      [*split, "--subset", SUBSET, "--repeat", "0"],
                      [*split, "--subset", SUBSET, "--perturbations", "p.json", "--condition", "intact"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main(base + extra)

    def test_repeat_and_subset_are_part_of_the_resume_identity(self):
        frame = pd.DataFrame([{"question_id": "q1", "question": "Q", "true_answer": "1", "assumption": None,
                               "patient_fhir_id": "p"}])
        base = {"agent_strategy": "direct_llm", "model": "m", "base_url": None, "max_total_tokens": None,
                "max_tool_calls": None, "add_patient_fhir_id": True, "enable_cache": True}
        identities = [runner.run_identity(SimpleNamespace(**base, **extra), frame) for extra in (
            {}, {"subset": SUBSET, "repeat": 1}, {"subset": SUBSET, "repeat": 2})]
        self.assertEqual(len({json.dumps(i, sort_keys=True) for i in identities}), 3)


def row(answer, tokens=10, latency=1.0):
    return {"agent_answer": answer, "error": None, "tokens": {"total": tokens}, "latency": latency}


class StatisticsTests(unittest.TestCase):
    def test_spread_and_agreement_over_repeats(self):
        references = {q: {"question": q, "true_answer": "[[1]]", "template": ""} for q in ("a", "b", "c", "d")}
        # a: always right; b: never right; c: right in 2 of 3 repeats; d: right in 1 of 3
        repeats = [{"a": row(1), "b": row(0), "c": row(1), "d": row(1, tokens=40)},
                   {"a": row(1), "b": row(0), "c": row(1), "d": row(None, tokens=20)},
                   {"a": row(1), "b": row(0), "c": row(0), "d": row(None, tokens=20)}]
        with patch.object(variability, "score_answer",
                          side_effect=lambda answer, *_: {"typed_correct": int(answer == 1)}):
            summary = variability.summarize(repeats, references)
        self.assertEqual(summary["repeats"], 3)
        self.assertEqual([r["accuracy"] for r in summary["per_repeat"]], [0.75, 0.5, 0.25])
        self.assertAlmostEqual(summary["accuracy"]["mean"], 0.5)
        self.assertAlmostEqual(summary["accuracy"]["sd"], 0.25)
        self.assertEqual((summary["accuracy"]["min"], summary["accuracy"]["max"]), (0.25, 0.75))
        self.assertEqual((summary["always_correct"], summary["never_correct"], summary["unstable"]), (1, 1, 2))
        # pairwise agreement: a 1, b 1, c 1/3, d 1/3
        self.assertAlmostEqual(summary["correctness_agreement"], (1 + 1 + 1 / 3 + 1 / 3) / 4)
        self.assertEqual(summary["identical_answers"], 0.5)
        self.assertEqual([r["abstention_rate"] for r in summary["per_repeat"]], [0.0, 0.25, 0.25])
        self.assertEqual([r["mean_tokens"] for r in summary["per_repeat"]], [17.5, 12.5, 12.5])

    def test_single_repeat_has_no_spread_and_mismatched_repeats_are_rejected(self):
        references = {"a": {"question": "a", "true_answer": "[[1]]"}}
        with patch.object(variability, "score_answer", return_value={"typed_correct": 1}):
            summary = variability.summarize([{"a": row(1)}], references)
            self.assertIsNone(summary["accuracy"]["sd"])
            self.assertIsNone(summary["correctness_agreement"])
            with self.assertRaisesRegex(ValueError, "exactly the subset"):
                variability.summarize([{"a": row(1)}, {}], references)

    def test_errored_rows_score_as_wrong(self):
        self.assertEqual(variability.score_row({"agent_answer": 1, "error": "boom"}, {"true_answer": "[[1]]",
                                                                                        "question": "Q"}), 0)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cli = ["--model", "test-model", "--output_dir", self.temp.name, "--limit", "3"]

    def write_repeat(self, arm, answers):
        args = experiments.build_parser().parse_args(["status", *self.cli])
        settings = runner.build_parser().parse_args(experiments.runner_argv(arm, args))
        question_sets = experiments.load_question_sets([arm], args)
        frame = experiments.arm_inputs(arm, settings, question_sets)
        identity = runner.run_identity(settings, frame)
        results = runner.prepare_results(frame, settings.output, identity)
        for index, answer in enumerate(answers):
            results.at[index, "agent_answer"] = answer
            results.at[index, "latency"] = 1.0
            results.at[index, "tokens"] = {"total": 100}
        with contextlib.redirect_stdout(io.StringIO()):
            runner.save_results_json(results, settings.output)
        runner._write_json(identity, settings.output + ".run.json")

    def test_report_uses_complete_repeats_and_never_prints_answers(self):
        arms = {arm.name: arm for arm in experiments.ARMS}
        self.write_repeat(arms["variability_single_agent_r1"], ["SECRET-1", "SECRET-1", "SECRET-1"])
        self.write_repeat(arms["variability_single_agent_r2"], ["SECRET-1", "SECRET-0", "SECRET-0"])
        self.write_repeat(arms["variability_single_agent_r3"], ["SECRET-1"])  # incomplete: skipped
        with patch("scripts.variability.score_answer",
                   side_effect=lambda answer, *_: {"typed_correct": int(answer == "SECRET-1")}), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(experiments.main(["variability", *self.cli]), 0)
        printed = out.getvalue()
        self.assertNotIn("SECRET", printed)
        self.assertIn("single_agent: 2/5 repeats x 3 questions", printed)
        self.assertIn("variability_single_agent_r3", printed)
        self.assertNotIn("multi_agent:", printed)
        with open(os.path.join(self.temp.name, "variability_report.json")) as stream:
            report = json.load(stream)
        self.assertNotIn("SECRET", json.dumps(report))
        single = report["systems"]["single_agent"]
        self.assertEqual([r["accuracy"] for r in single["per_repeat"]], [1.0, 1 / 3])
        self.assertEqual((single["always_correct"], single["never_correct"], single["unstable"]), (1, 0, 2))

    def test_variability_arms_need_held_out(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            experiments.main(["plan", "--model", "m", "--split", "pilot", "--arms", "variability"])


if __name__ == "__main__":
    unittest.main()
