"""Lossless optional driver I/O is independent of model execution and scoring."""
from copy import deepcopy
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import run_agent as runner
from scripts.audit_store import restore_result_audit
from utils.core_utils import RESULT_COLUMNS, parse_outputs


def agent_output(answer=7, error=None):
    return {"answer": answer, "evidence": [{"resource_id": "Observation/o1", "field_path": "valueQuantity.value",
            "value": 7, "scope_index": 0}], "limitation": None,
            "tokens": {"prompt": 10, "completion": 2, "total": 12, "cost": None, "llm_calls": 1},
            "tool_calls": [{"name": "execute_python_code", "arguments": {"code": "answer=7", "mode": "final"}}],
            "latency": 1, "error": error, "budget": {"max_total_tokens": 200000, "max_tool_calls": 30,
                "tool_calls_run": 1, "stopped": None},
            "trace": [{"role": "tool", "content": "exact Unicode 測 code/evidence " * 10000}],
            "retrieved_fhir_resources": {"Observation": [{"resourceType": "Observation", "id": "o1",
                "valueQuantity": {"value": 7}, "code": {"text": "source label " * 10000}}]},
            "revision": {"draft": {"answer": answer}, "final": {"answer": answer}, "revised": False,
                "check_issues": [], "messages": [{"payload": {"scope_index": 0, "metadata": "full bus " * 10000}}]}}


class DriverAuditIOTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.output_file = str(self.directory / "results.json")
        self.inputs = pd.DataFrame([{"question_id": "q1", "question": "Fixture", "true_answer": "[[7]]",
                                    "assumption": "", "patient_fhir_id": "patient"}])
        self.identity = {"audit_storage": "gzip", "audit_artifact_schema": 2}

    def frame(self):
        return runner.prepare_results(self.inputs, self.output_file, self.identity)

    def test_full_resources_are_transient_only_when_requested(self):
        raw = agent_output()
        inline = parse_outputs(raw)
        gzip_carrier = parse_outputs(raw, include_full_audit=True)
        self.assertNotIn("retrieved_fhir_resources", inline)
        self.assertEqual(inline["agent_fhir_resources"], {"Observation": ["o1"]})
        self.assertEqual(gzip_carrier["retrieved_fhir_resources"], raw["retrieved_fhir_resources"])
        for key in inline:
            self.assertEqual(inline[key], gzip_carrier[key])

    def test_externalization_happens_before_dataframe_storage_and_exact_roundtrip(self):
        original = parse_outputs(agent_output(), include_full_audit=True)
        snapshot = deepcopy(original)
        frame = self.frame()
        runner.store_attempt(frame, 0, original, audit_storage="gzip", output_file=self.output_file)
        saved = {key: frame.at[0, key] for key in RESULT_COLUMNS}
        self.assertEqual(frame.at[0, "trace"], [])
        self.assertNotIn("retrieved_fhir_resources", frame.columns)
        self.assertEqual(frame.at[0, "revision"]["messages"], [])
        self.assertEqual(restore_result_audit(saved, self.directory, question_id="q1"), snapshot)
        self.assertEqual(original, snapshot)
        self.assertLess(len(json.dumps(saved)), len(json.dumps(snapshot)) // 100)

    def test_failed_attempt_history_keeps_both_immutable_artifacts_on_resume(self):
        first = parse_outputs(agent_output(error="fixture first failure"), include_full_audit=True)
        second = parse_outputs(agent_output(answer=8), include_full_audit=True)
        second["trace"].append({"role": "assistant", "content": '{"answer":8}'})
        frame = self.frame()
        runner.store_attempt(frame, 0, first, audit_storage="gzip", output_file=self.output_file)
        runner.store_attempt(frame, 0, second, audit_storage="gzip", output_file=self.output_file)
        runner._write_json(self.identity, self.output_file + ".run.json")
        runner.save_results_json(frame, self.output_file)
        resumed = runner.prepare_results(self.inputs, self.output_file, self.identity)
        history = resumed.at[0, "attempt_history"]
        self.assertEqual(len(history), 1)
        self.assertEqual(restore_result_audit(history[0], self.directory, question_id="q1"), first)
        self.assertNotEqual(history[0]["audit_artifacts"]["sha256"], resumed.at[0, "audit_artifacts"]["sha256"])

    def test_missing_or_corrupt_audit_refuses_resume_before_any_model_call(self):
        frame = self.frame()
        runner.store_attempt(frame, 0, parse_outputs(agent_output(), include_full_audit=True),
                             audit_storage="gzip", output_file=self.output_file)
        runner._write_json(self.identity, self.output_file + ".run.json")
        runner.save_results_json(frame, self.output_file)
        reference = frame.at[0, "audit_artifacts"]
        (self.directory / reference["path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            runner.prepare_results(self.inputs, self.output_file, self.identity)

    def test_worker_capture_flag_changes_only_transport_not_model_creation_or_budget(self):
        from utils import core_utils as core
        class FixtureAgent:
            def set_budget(self, **budget):
                self.budget = budget
            def set_perturbation(self, value):
                self.perturbation = value
            def run(self, question):
                return agent_output()
        agent = FixtureAgent()
        budget = {"max_total_tokens": 200000, "max_tool_calls": 30}
        job = runner.prepare_multiprocessing_inputs(["question"], "single_agent", "fixture", False,
                    None, True, budget, include_full_audit=True)[0]
        with patch.object(core, "create_agent", return_value=agent) as create:
            carrier = core.run_agent_with_input(job)
        create.assert_called_once_with("single_agent", "fixture", False, None)
        self.assertEqual(agent.budget, budget)
        self.assertIsNone(agent.perturbation)
        self.assertIn("retrieved_fhir_resources", carrier)

    def test_inline_default_and_gzip_identity_are_explicit(self):
        with patch.object(runner, "get_default_model", return_value="fixture"):
            inline = runner.build_parser().parse_args(["--agent_strategy", "single_agent", "--model", "fixture"])
            compressed = runner.build_parser().parse_args(["--agent_strategy", "single_agent", "--model", "fixture",
                                                          "--audit_storage", "gzip"])
        a, b = runner.run_identity(inline, self.inputs), runner.run_identity(compressed, self.inputs)
        self.assertEqual(a["audit_storage"], "inline")
        self.assertEqual(b["audit_artifact_schema"], 2)
        for key in ("input_hash", "model", "max_total_tokens", "max_tool_calls", "enable_cache"):
            self.assertEqual(a[key], b[key])

    def test_subset_launcher_forwards_explicit_storage_to_each_selected_arm(self):
        from scripts import run_experiments as experiments
        args = experiments.build_parser().parse_args([
            "plan", "--model", "fixture", "--output_dir", str(self.directory),
            "--max_total_tokens", "200000", "--max_tool_calls", "30", "--audit_storage", "gzip"])
        for arm in experiments.ARMS:
            with self.subTest(arm=arm.name):
                settings = runner.build_parser().parse_args(experiments.runner_argv(arm, args))
                self.assertEqual(settings.audit_storage, "gzip")
                self.assertEqual((settings.max_total_tokens, settings.max_tool_calls), (200000, 30))

    def test_subset_team_strategy_and_gzip_flags_are_independent(self):
        from scripts import run_experiments as experiments
        args = experiments.build_parser().parse_args([
            "plan", "--model", "fixture", "--output_dir", str(self.directory),
            "--team_strategy", "multi_agent_v3", "--audit_storage", "gzip",
            "--max_total_tokens", "200000", "--max_tool_calls", "30"])
        for arm in experiments.ARMS[:3]:
            settings = runner.build_parser().parse_args(experiments.runner_argv(arm, args))
            expected_strategy = "multi_agent_v3" if arm.strategy == "multi_agent" else arm.strategy
            self.assertEqual(settings.agent_strategy, expected_strategy)
            self.assertEqual(settings.audit_storage, "gzip")
            self.assertTrue(settings.output.endswith(arm.name + ".json"))

    def test_v3_worker_dispatch_externalizes_before_ipc_with_common_execution_contract(self):
        raw = agent_output()
        expected = parse_outputs(raw, include_full_audit=True)
        budget = {"max_total_tokens": 200000, "max_tool_calls": 30}
        class FixtureAgent:
            def set_budget(self, **value):
                self.budget = value
            def set_perturbation(self, value):
                self.perturbation = value
            def run(self, question):
                self.question = question
                return raw
        agent = FixtureAgent()
        job = runner.prepare_multiprocessing_inputs(["question"], "multi_agent_v3", "fixture", False,
                    None, True, budget, include_full_audit=True,
                    audit_contexts=[{"output_dir": str(self.directory), "question_id": "q1"}])[0]
        with patch("agent.query_contract_agent.QueryContractAgent", return_value=agent) as factory:
            returned = runner.dispatch_agent_job(job)
        factory.assert_called_once_with("fixture", verbose=False, base_url=None)
        self.assertEqual((agent.question, agent.budget, agent.perturbation), ("question", budget, None))
        self.assertEqual(returned["trace"], [])
        self.assertNotIn("retrieved_fhir_resources", returned)
        self.assertLess(len(pickle.dumps(returned)), len(pickle.dumps(expected)) // 100)
        self.assertEqual(restore_result_audit(returned, self.directory, question_id="q1"), expected)

    def test_worker_externalizes_before_ipc_and_parent_dataframe_accumulation(self):
        from utils import core_utils as core
        raw = agent_output()
        carrier_before = parse_outputs(raw, include_full_audit=True)
        expected = deepcopy(carrier_before)
        class FixtureAgent:
            def set_budget(self, **budget):
                pass
            def set_perturbation(self, value):
                pass
            def run(self, question):
                return deepcopy(raw)
        job = runner.prepare_multiprocessing_inputs(["question"], "single_agent", "fixture", False,
                    None, True, {"max_total_tokens": 200000, "max_tool_calls": 30},
                    include_full_audit=True,
                    audit_contexts=[{"output_dir": str(self.directory), "question_id": "q1"}])[0]
        with patch.object(core, "create_agent", return_value=FixtureAgent()):
            returned = core.run_agent_with_input(job)
        self.assertEqual(returned["trace"], [])
        self.assertNotIn("retrieved_fhir_resources", returned)
        self.assertLess(len(pickle.dumps(returned)), len(pickle.dumps(carrier_before)) // 100)
        frame = self.frame()
        runner.store_attempt(frame, 0, returned, audit_storage="gzip", output_file=self.output_file)
        self.assertEqual(frame.at[0, "audit_artifacts"], returned["audit_artifacts"])
        restored = restore_result_audit({key: frame.at[0, key] for key in RESULT_COLUMNS},
                                       self.directory, question_id="q1")
        self.assertEqual(restored, expected)

    def test_multiple_compact_attempts_do_not_accumulate_raw_audits_in_dataframe(self):
        count = 8
        inputs = pd.concat([self.inputs.assign(question_id=f"q{index}") for index in range(count)], ignore_index=True)
        frame = runner.prepare_results(inputs, self.output_file, self.identity)
        for index in range(count):
            runner.store_attempt(frame, index, parse_outputs(agent_output(), include_full_audit=True),
                                 audit_storage="gzip", output_file=self.output_file)
        inline = frame[RESULT_COLUMNS].to_dict("records")
        raw_one = parse_outputs(agent_output(), include_full_audit=True)
        self.assertLess(len(json.dumps(inline)), len(json.dumps(raw_one)) * count // 100)
        self.assertTrue(all(row["trace"] == [] and row["revision"]["messages"] == [] for row in inline))
        self.assertNotIn("retrieved_fhir_resources", frame.columns)
        for index, row in enumerate(inline):
            self.assertEqual(restore_result_audit(row, self.directory, question_id=f"q{index}"), raw_one)

    def test_complete_driver_gzip_and_inline_have_identical_scoring_and_execution_fields(self):
        from utils import core_utils as core
        self.inputs.to_csv(self.directory / "input.csv", index=False)
        recorded = []
        class FixtureAgent:
            def set_budget(self, **budget):
                self.budget = budget
            def set_perturbation(self, value):
                self.perturbation = value
            def run(self, question):
                recorded.append((question, self.budget, self.perturbation))
                return agent_output()
        paths = {}
        with patch.object(runner, "get_default_model", return_value="fixture"), \
                patch.object(runner, "validate_config", return_value=True), \
                patch.object(runner, "check_tool_credentials"), \
                patch.object(core, "create_agent", side_effect=lambda *args: FixtureAgent()):
            for mode in ("inline", "gzip"):
                path = self.directory / f"{mode}.json"
                code = runner.main(["--model", "fixture", "--agent_strategy", "single_agent",
                        "--input", str(self.directory / "input.csv"), "--output", str(path),
                        "--audit_storage", mode, "--num_processes", "1", "--save_interval", "1",
                        "--max_total_tokens", "200000", "--max_tool_calls", "30", "--quiet"])
                self.assertEqual(code, 0)
                paths[mode] = json.loads(path.read_text())[0]
        self.assertEqual(recorded[0], recorded[1])
        restored = restore_result_audit(paths["gzip"], self.directory)
        for field in RESULT_COLUMNS:
            self.assertEqual(restored[field], paths["inline"][field])
        self.assertEqual(restored["retrieved_fhir_resources"], agent_output()["retrieved_fhir_resources"])

    def test_result_array_validation_streams_instead_of_reloading_the_whole_file(self):
        rows = [{"question_id": "q1", "agent_answer": 7, "trace": []},
                {"question_id": "q2", "agent_answer": None, "trace": []}]
        with patch.object(runner.json, "load", side_effect=AssertionError("Whole-file load is forbidden")):
            runner._write_json(rows, self.output_file)
        self.assertEqual(json.loads(Path(self.output_file).read_text()), rows)


if __name__ == "__main__":
    unittest.main()
