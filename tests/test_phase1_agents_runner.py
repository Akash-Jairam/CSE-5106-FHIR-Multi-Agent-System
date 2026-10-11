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


def inputs(ids=("q1", "q2")):
    return pd.DataFrame([{ "question_id": identifier, "question": "Question " + identifier,
                           "true_answer": "7", "assumption": None, "patient_fhir_id": "patient"}
                         for identifier in ids])


def settings(**overrides):
    values = {"agent_strategy": "single_agent", "model": "test-model", "base_url": None,
              "max_total_tokens": 1000, "max_tool_calls": 4, "add_patient_fhir_id": True,
              "enable_cache": True}
    return SimpleNamespace(**{**values, **overrides})


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = os.path.join(self.temp.name, "results.json")
        self.inputs = inputs()
        self.identity = runner.run_identity(settings(), self.inputs)

    def write_previous(self, results):
        runner.save_results_json(results, self.output)
        runner._write_json(self.identity, self.output + ".run.json")

    def test_failure_with_latency_is_retryable_but_abstention_is_done(self):
        self.assertFalse(runner.is_done({"agent_answer": None, "latency": 1.2, "error": "failed"}))
        self.assertTrue(runner.is_done({"agent_answer": None, "latency": 1.2, "error": None}))
        self.assertFalse(runner.is_done({"agent_answer": None, "latency": None, "error": None}))

    def test_resume_maps_saved_results_by_question_id_even_at_same_size(self):
        previous = runner.prepare_results(self.inputs, self.output, self.identity)
        previous["agent_answer"] = [11, 22]
        self.write_previous(previous.iloc[::-1])
        loaded = runner.prepare_results(self.inputs, self.output, self.identity)
        self.assertEqual(loaded["question_id"].tolist(), ["q1", "q2"])
        self.assertEqual(loaded["agent_answer"].tolist(), [11, 22])

    def test_resume_rejects_model_strategy_budget_or_inputs_change(self):
        previous = runner.prepare_results(self.inputs, self.output, self.identity)
        self.write_previous(previous)
        for changes in ({"model": "other-model"}, {"agent_strategy": "direct_llm"}, {"max_total_tokens": 2000},
                        {"max_tool_calls": 5}, {"base_url": "http://another-deployment"}):
            identity = runner.run_identity(settings(**changes), self.inputs)
            with self.assertRaisesRegex(ValueError, "different inputs or run settings"):
                runner.prepare_results(self.inputs, self.output, identity)
        altered = inputs(("q3", "q4"))
        with self.assertRaisesRegex(ValueError, "different inputs or run settings"):
            runner.prepare_results(altered, self.output, runner.run_identity(settings(), altered))

    def test_resume_rejects_changed_agent_or_tool_code(self):
        previous = runner.prepare_results(self.inputs, self.output, self.identity)
        self.write_previous(previous)
        with patch.object(runner, "code_fingerprint", return_value="changed-code"):
            identity = runner.run_identity(settings(), self.inputs)
        with self.assertRaisesRegex(ValueError, "different inputs or run settings"):
            runner.prepare_results(self.inputs, self.output, identity)

    def test_legacy_results_without_identity_are_not_silently_reused(self):
        runner.save_results_json(self.inputs, self.output)
        with self.assertRaisesRegex(ValueError, "no run identity"):
            runner.prepare_results(self.inputs, self.output, self.identity)

    def test_prior_failed_attempt_is_preserved_when_retry_succeeds(self):
        results = runner.prepare_results(self.inputs, self.output, self.identity)
        first = {"agent_answer": None, "error": "failed", "latency": 1.2,
                 "tokens": {"total": 99, "cost": None}, "trace": [{"role": "assistant", "content": "draft"}]}
        second = {"agent_answer": 7, "error": None, "latency": 2.3, "tokens": {"total": 50, "cost": 0.1}}
        runner.store_attempt(results, 0, first)
        runner.store_attempt(results, 0, second)
        runner.save_results_json(results, self.output)
        with open(self.output, encoding="utf-8") as stream:
            row = json.load(stream)[0]
        self.assertEqual(row["agent_answer"], 7)
        self.assertEqual(row["attempt_history"][0]["tokens"]["total"], 99)
        self.assertEqual(row["attempt_history"][0]["error"], "failed")
        self.assertEqual(row["attempt_history"][0]["trace"][0]["content"], "draft")

    def test_all_done_resume_avoids_credentials_and_zero_process_pool(self):
        identity = runner.run_identity(settings(max_total_tokens=None, max_tool_calls=None), self.inputs)
        results = runner.prepare_results(self.inputs, self.output, identity)
        results["latency"] = [1.2, 1.3]
        runner.save_results_json(results, self.output)
        runner._write_json(identity, self.output + ".run.json")
        with patch.object(runner, "load_inputs", return_value=self.inputs), patch.object(runner, "validate_config") as validate, \
                patch.object(runner, "check_tool_credentials") as credentials, patch.object(runner.mp, "Pool") as pool:
            self.assertEqual(runner.main(["--agent_strategy", "single_agent", "--model", "test-model", "--output", self.output]), 0)
        validate.assert_not_called()
        credentials.assert_not_called()
        pool.assert_not_called()

    def test_unknown_cost_and_executed_tool_counts_are_reported(self):
        results = pd.DataFrame([{ "tokens": {"total": 10, "prompt": 8, "completion": 2, "llm_calls": 1, "cost": None},
                                 "budget": {"tool_calls_run": 1}, "tool_calls": ["executed", "rejected"]}])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            runner.print_usage_summary(results)
        self.assertIn("cost unavailable", output.getvalue())
        self.assertIn("Tool calls executed: 1", output.getvalue())

    def test_boolean_false_is_parsed_as_false(self):
        args = runner.build_parser().parse_args(["--agent_strategy", "single_agent", "--enable_cache", "false",
                                               "--add_patient_fhir_id", "false"])
        self.assertFalse(args.enable_cache)
        self.assertFalse(args.add_patient_fhir_id)

    def test_split_selection_is_frozen_and_limit_applied_after_selection(self):
        frame = inputs(("outside", "second", "first"))
        frame["patient_fhir_id"] = ["outside-patient", "pilot-patient", "pilot-patient"]
        manifest = {"splits": {
            "pilot": {"question_ids": ["first", "second"], "patients": [{"patient_fhir_id": "pilot-patient"}],
                      "family_counts": {"lookup_counting": 2}},
            "held_out": {"question_ids": ["outside"], "patients": [{"patient_fhir_id": "outside-patient"}]},
        }}
        manifest_file = os.path.join(self.temp.name, "split.json")
        runner._write_json(manifest, manifest_file)
        with patch.object(runner, "read_input_data", return_value=frame):
            selected = runner.load_inputs("unused.csv", limit=1, split_manifest=manifest_file, split="pilot")
        self.assertEqual(selected["question_id"].tolist(), ["first"])

    def test_invalid_split_counts_ids_and_patient_overlap_are_rejected(self):
        for manifest in (
            {"splits": {"pilot": {"question_ids": ["missing"], "patients": [{"patient_fhir_id": "patient"}]}}},
            {"splits": {"pilot": {"question_ids": ["q1"], "patients": [{"patient_fhir_id": "other"}]}}},
            {"splits": {"pilot": {"question_ids": ["q1"], "patients": [{"patient_fhir_id": "patient"}], "family_counts": {"count": 2}}}},
            {"splits": {"pilot": {"question_ids": ["q1"], "patients": [{"patient_fhir_id": "patient"}]},
                        "held_out": {"question_ids": ["q2"], "patients": [{"patient_fhir_id": "patient"}]}}},
        ):
            with self.assertRaises(ValueError):
                runner.select_split(self.inputs, manifest, "pilot")

    def test_empty_input_creates_valid_results_without_calling_model(self):
        with patch.object(runner, "load_inputs", return_value=self.inputs.head(0)), \
                patch.object(runner, "validate_config") as validate, patch.object(runner.mp, "Pool") as pool:
            self.assertEqual(runner.main(["--agent_strategy", "single_agent", "--model", "test-model", "--output", self.output]), 0)
        validate.assert_not_called()
        pool.assert_not_called()
        with open(self.output, encoding="utf-8") as stream:
            self.assertEqual(json.load(stream), [])


if __name__ == "__main__":
    unittest.main()
