"""P3-23: the demo's patients, questions, evaluated-pipeline call, verification status, and app.

No model or FHIR store is called.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

import run_agent as runner
from demo import backend
from scripts import run_experiments as experiments
from utils import curate_input_dataset

INPUT, SPLIT = experiments.DEFAULT_INPUT, experiments.DEFAULT_SPLIT
APP = os.path.join(backend.ROOT, "demo", "app.py")


def output(answer=72.5, evidence=None, revision=None, error=None, stopped=None):
    return {"agent_answer": answer, "evidence": evidence if evidence is not None else [
                {"resource_id": "Observation/o1", "field_path": "valueQuantity.value", "value": 72.5},
                {"resource_id": "Observation/o2", "field_path": "effectiveDateTime", "value": "2150-03-02T08:00:00"}],
            "limitation": None, "revision": revision, "error": error, "budget": {"stopped": stopped},
            "tokens": {"total": 100, "llm_calls": 2, "cost": None}, "latency": 1.5, "tool_calls": []}


def multi_revision(final_issues, revised=True):
    return {"draft": {"answer": 70}, "revised": revised, "repair_role": "analysis" if revised else None,
            "check_issues": [{"type": "recompute_mismatch", "record": None, "field": None}],
            "final_check_issues": final_issues}


class PatientTests(unittest.TestCase):
    def test_examples_come_from_non_held_out_patients_without_answers(self):
        patients = backend.load_patients(INPUT, SPLIT)
        with open(SPLIT) as stream:
            splits = json.load(stream)["splits"]
        held_out = {p["patient_fhir_id"] for p in splits["held_out"]["patients"]}
        self.assertTrue(patients)
        self.assertFalse(held_out & {p["patient_fhir_id"] for p in patients})
        self.assertEqual({p["split"] for p in patients}, {"pilot", "dev"})
        self.assertEqual(sum(len(p["questions"]) for p in patients), 70)
        self.assertNotIn("true_answer", json.dumps(patients))
        self.assertTrue(any(q["current_time"] for p in patients for q in p["questions"]))

    def test_built_input_matches_the_runner_context_for_a_dataset_row(self):
        frame = runner.read_input_data(INPUT)
        row = frame[frame["assumption"].notna()].iloc[0]
        expected = curate_input_dataset(pd.DataFrame([row]), True)[0]
        built = backend.build_input(row["patient_fhir_id"], row["question"], backend.current_time_of(row["assumption"]))
        self.assertEqual(built, expected)
        self.assertNotIn("current time", backend.build_input("p", "Q?"))
        for question, now in (("", None), ("Q?", "next tuesday")):
            with self.assertRaises(ValueError):
                backend.build_input("p", question, now)

    def test_ask_uses_the_runner_job_with_cache_and_budget(self):
        settings = {"model": "m", "base_url": None, "max_total_tokens": 9, "max_tool_calls": 3}
        with patch.object(backend, "run_agent_with_input", return_value={"ok": 1}) as run:
            self.assertEqual(backend.ask("multi_agent", "input", settings), {"ok": 1})
        run.assert_called_once_with(("input", "multi_agent", "m", False, None, True,
                                     {"max_total_tokens": 9, "max_tool_calls": 3}, None))
        with self.assertRaises(ValueError):
            backend.ask("other", "input", settings)


class StatusTests(unittest.TestCase):
    def test_states(self):
        status = lambda system, out: backend.verification_status(system, out)["state"]
        self.assertEqual(status("multi_agent", output(revision=multi_revision([]))), "verified")
        self.assertIn("after one repair (from analysis)",
                      backend.verification_status("multi_agent", output(revision=multi_revision([])))["detail"])
        unverified = backend.verification_status("multi_agent", output(answer=None, revision=multi_revision(
            [{"type": "value_mismatch", "record": "Observation/o1", "field": "valueQuantity.value"}])))
        self.assertEqual((unverified["state"], unverified["issues"][0]["type"]), ("unverified", "value_mismatch"))
        single = backend.verification_status("single_agent", output(answer=None, revision={
            "draft": {"answer": 1}, "revised": False, "check_issues": ["unit: cited kg, record has lb"],
            "final_check_issues": None}))
        self.assertEqual((single["state"], single["issues"][0]["type"]), ("unverified", "unit"))
        self.assertEqual(status("direct_llm", output()), "unchecked")
        self.assertEqual(status("multi_agent", output(error="boom")), "error")
        stopped = backend.verification_status("single_agent", output(answer=None, stopped="token budget"))
        self.assertEqual((stopped["state"], stopped["detail"]), ("abstained", "Stopped by the token budget."))
        self.assertEqual(status("multi_agent", output(answer=None, revision=multi_revision([], False))), "abstained")

    def test_cited_fields_flag_only_the_issue_field_or_whole_record(self):
        field_issue = output(answer=None, revision=multi_revision(
            [{"type": "value_mismatch", "record": "Observation/o2", "field": "valueQuantity.value"}]))
        self.assertEqual([r["check"] for r in backend.cited_fields(field_issue)], ["ok", "ok"])
        record_issue = output(answer=None, revision=multi_revision(
            [{"type": "patient_scope", "record": "Observation/o2", "field": None}]))
        rows = backend.cited_fields(record_issue)
        self.assertEqual([r["check"] for r in rows], ["ok", "patient_scope"])
        self.assertEqual([r["value"] for r in rows], ["72.5", "2150-03-02T08:00:00"])


class EvaluatedRunTests(unittest.TestCase):
    def test_matches_the_evaluated_ordinary_arm(self):
        settings = {"model": "test-model", "base_url": None, "max_total_tokens": None, "max_tool_calls": 30}
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(backend.evaluated_run("multi_agent", settings, directory)["found"])
            args = experiments.build_parser().parse_args(
                ["status", "--model", "test-model", "--max_tool_calls", "30", "--output_dir", directory, "--limit", "1"])
            arm = next(a for a in experiments.ARMS if a.name == "ordinary_multi_agent")
            run_settings = runner.build_parser().parse_args(experiments.runner_argv(arm, args))
            runner._write_json(runner.run_identity(run_settings, experiments.arm_inputs(arm, run_settings, {})),
                               run_settings.output + ".run.json")
            self.assertTrue(backend.evaluated_run("multi_agent", settings, directory)["all_match"])
            other = backend.evaluated_run("multi_agent", {**settings, "max_tool_calls": 5}, directory)
            self.assertEqual([k for k, ok in other["matches"].items() if not ok], ["max_tool_calls"])


class AppTests(unittest.TestCase):
    def run_app(self):
        app = AppTest.from_file(APP, default_timeout=60)
        app.run()
        self.assertFalse(app.exception)
        return app

    def test_ask_shows_answer_cited_fields_and_status(self):
        with patch.object(backend, "ask", return_value=output(revision=multi_revision([]))) as ask, \
                patch("utils.validate_config", return_value=True):
            app = self.run_app()
            self.assertIn("Choose a patient", app.info[0].value)
            app.button[0].click().run()
        self.assertFalse(app.exception)
        system, input_str, settings = ask.call_args.args
        self.assertEqual(system, "multi_agent")
        self.assertTrue(input_str.startswith("Question: ") and "Patient FHIR ID is" in input_str)
        self.assertIn("Verified", app.success[0].value)
        self.assertEqual(app.metric[0].value, "72.5")
        self.assertEqual(len(app.dataframe[0].value), 2)

    def test_custom_question_and_missing_configuration(self):
        with patch.object(backend, "ask") as ask, patch("utils.validate_config", return_value=False):
            app = self.run_app()
            next(r for r in app.radio if r.label == "Question").set_value("Custom").run()
            app.button[0].click().run()
            self.assertIn("Enter a question", app.error[0].value)
            app.text_area[0].input("What was the last weight?").run()
            app.button[0].click().run()
            self.assertIn("not configured", app.error[0].value)
        ask.assert_not_called()


if __name__ == "__main__":
    unittest.main()
