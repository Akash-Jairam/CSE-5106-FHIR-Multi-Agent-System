"""Full-run scheduling and first-attempt health require no model or FHIR requests."""
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import full_benchmark as full


class FullBenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input = self.root / "input.csv"
        self.output = self.root / "results"
        self.output.mkdir()
        self.families = self.root / "families.csv"
        with self.input.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("question_id", "patient_fhir_id", "question", "true_answer", "assumption"))
            writer.writeheader()
            writer.writerows({"question_id": f"q{index}", "patient_fhir_id": f"patient{index % 10}",
                              "question": "Synthetic fixture", "true_answer": "[]", "assumption": ""}
                             for index in range(full.QUESTIONS))
        self.families.write_text("question_id,family,template\n")
        self.patients = full.dataset_cohort(self.input)
        self.args = full.build_parser().parse_args(["run", "--output_dir", str(self.output),
                "--input", str(self.input), "--families", str(self.families), "--model", "fixture"])
        self.identity = {"model": "fixture", "max_total_tokens": 200000, "max_tool_calls": 30,
                         "agent_strategy": "single_agent", "input_hash": "fixture", "code_hash": "fixture"}

    def write_results(self, error=False, retry=False):
        rows = [{"question_id": question_id, "patient_fhir_id": patient,
                 "latency": 1.0, "agent_answer": None, "error": "fixture failure" if error and index == 0 else None,
                 "tokens": {"total": 10, "cost": None, "per_call": [{"usage_available": True}]},
                 "budget": {"stopped": None}, "attempt_history": [{"latency": 1}] if retry and index == 0 else None}
                for index, (question_id, patient) in enumerate(self.patients.items())]
        (self.output / "full_single_agent.json").write_text(json.dumps(rows))
        (self.output / "full_single_agent.json.run.json").write_text(json.dumps(self.identity))

    def test_all_three_commands_share_full_cohort_model_budget_cache_and_workers(self):
        commands = [full.arm_command(strategy, self.args) for strategy in full.SYSTEMS]
        for strategy, command in zip(full.SYSTEMS, commands):
            settings = dict(zip(command[2::2], command[3::2]))
            # --quiet is the one terminal flag rather than an option/value pair.
            self.assertEqual(settings["--agent_strategy"], strategy)
            self.assertEqual(settings["--limit"], "2931")
            self.assertEqual(settings["--model"], "fixture")
            self.assertEqual(settings["--max_total_tokens"], "200000")
            self.assertEqual(settings["--max_tool_calls"], "30")
            self.assertEqual(settings["--num_processes"], "2")
            self.assertEqual(settings["--save_interval"], "10")
            self.assertEqual(settings["--enable_cache"], "true")
            self.assertNotIn("--split", command)
            self.assertNotIn("--split_manifest", command)

    def test_v3_gzip_manifest_pins_actual_team_and_shared_storage_settings(self):
        self.args.team_strategy = "multi_agent_v3"
        self.args.audit_storage = "gzip"
        manifest = full.build_manifest(self.args, self.patients)
        self.assertEqual(manifest["settings"]["team_strategy"], "multi_agent_v3")
        self.assertEqual(manifest["settings"]["audit_storage"], "gzip")
        for strategy, identity in manifest["run_identities"].items():
            actual = "multi_agent_v3" if strategy == "multi_agent" else strategy
            self.assertEqual(identity["agent_strategy"], actual)
            self.assertEqual((identity["audit_storage"], identity["audit_artifact_schema"]), ("gzip", 2))
            self.assertEqual((identity["max_total_tokens"], identity["max_tool_calls"]), (200000, 30))
        self.assertEqual(manifest["expected_first_attempts"], 8793)

    def test_wrong_size_or_duplicate_dataset_is_rejected(self):
        with self.input.open("a") as stream:
            stream.write("q0,patient0,Synthetic fixture,[],\n")
        with self.assertRaisesRegex(ValueError, "unique"):
            full.dataset_cohort(self.input)
        self.input.write_text("question_id,patient_fhir_id\nq0,patient0\n")
        with self.assertRaisesRegex(ValueError, "2931"):
            full.dataset_cohort(self.input)

    def test_streamed_array_handles_chunk_boundaries_and_rejects_corruption(self):
        path = self.root / "stream.json"
        rows = [{"text": "Unicode 測 \"quoted\" brackets ], {", "nested": [1, {"x": True}]},
                {"text": "x" * 10000, "value": None}]
        path.write_text(json.dumps(rows, ensure_ascii=False))
        self.assertEqual(list(full.iter_json_array(path, chunk_chars=7)), rows)
        for malformed in ('[{},]', '[{"a":', '[{}] trailing', '{}'):
            path.write_text(malformed)
            with self.assertRaises(ValueError):
                list(full.iter_json_array(path, chunk_chars=3))

    def test_existing_state_identity_or_results_never_resume_silently(self):
        for filename in ("supervisor_state.json", "full_manifest.json", "full_multi_agent.json",
                         "full_single_agent.json.run.json"):
            path = self.output / filename
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "No automatic resume"):
                full.require_fresh_output(self.output)
            path.unlink()

    def test_terminal_errors_remain_completed_first_attempts_and_cost_unknown(self):
        self.write_results(error=True)
        health = full.arm_health("single_agent", self.output, self.patients, self.identity)
        self.assertEqual(health["attempted"], full.QUESTIONS)
        self.assertEqual(health["completed"], full.QUESTIONS - 1)
        self.assertEqual(health["errors"], 1)
        self.assertEqual(health["cost_unknown_attempts"], full.QUESTIONS)
        self.assertIsNone(health["cost_total"])
        self.assertTrue(health["identity_verified"])
        self.assertEqual(health["problems"], [])

    def test_retry_or_identity_mismatch_requires_attention(self):
        self.write_results(retry=True)
        health = full.arm_health("single_agent", self.output, self.patients, self.identity)
        self.assertIn("unexpected_retry_history", health["problems"])
        changed = {**self.identity, "max_total_tokens": 999}
        health = full.arm_health("single_agent", self.output, self.patients, changed)
        self.assertIn("run_identity_mismatch", health["problems"])

    def test_public_manifest_binds_all_arm_identities_and_comparison(self):
        with patch("run_agent.get_default_model", return_value="fixture"):
            manifest = full.build_manifest(self.args, self.patients)
        self.assertEqual(manifest["expected_first_attempts"], 8793)
        self.assertFalse(manifest["automatic_retries"])
        identities = manifest["run_identities"]
        self.assertEqual(set(identities), set(full.SYSTEMS))
        shared_keys = ("model", "max_total_tokens", "max_tool_calls", "input_hash", "code_hash")
        for key in shared_keys:
            self.assertEqual(len({identity[key] for identity in identities.values()}), 1)
        self.assertIn("--full", manifest["comparison_command"])
        self.assertEqual(manifest["comparison_command"].count("--arm"), 3)
        self.assertNotIn("config.yml", manifest["source_sha256"])
        self.assertEqual(manifest["settings"]["input_guard_tokens"], 32000)

    def test_supervisor_starts_each_arm_once_and_analyzes_terminal_errors_without_retry(self):
        self.args.audit_storage = "inline"  # This scheduling fixture has no real audit payloads.
        with patch("run_agent.get_default_model", return_value="fixture"):
            manifest = full.build_manifest(self.args, self.patients)
        launches = []

        class FinishedChild:
            pid = 999999

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        def launch(command, **kwargs):
            launches.append(list(command))
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(kwargs["env"]["FHIR_FULL_TEST_BOOTSTRAP"], "fixture")
            if "--agent_strategy" in command:
                strategy = command[command.index("--agent_strategy") + 1]
                rows = [{"question_id": question_id, "patient_fhir_id": patient, "latency": 1,
                         "agent_answer": None, "error": "fixture" if strategy == "single_agent" and index == 0 else None,
                         "tokens": {"total": 10, "cost": None}, "budget": {"stopped": None}}
                        for index, (question_id, patient) in enumerate(self.patients.items())]
                (self.output / f"full_{strategy}.json").write_text(json.dumps(rows))
                (self.output / f"full_{strategy}.json.run.json").write_text(json.dumps(manifest["run_identities"][strategy]))
            else:
                (self.output / "full_comparison.json").write_text(json.dumps({
                    "arms": {strategy: {"accuracy": {"all": {"correct": 1, "n": full.QUESTIONS}}}
                             for strategy in full.SYSTEMS}, "comparisons": []}))
            return FinishedChild()

        with patch.object(full.subprocess, "Popen", side_effect=launch), \
                patch.object(full.shutil, "which", return_value=None), \
                patch.object(full.signal, "signal"), patch.object(full, "source_unchanged", return_value=True), \
                patch.dict(os.environ, {"FHIR_FULL_TEST_BOOTSTRAP": "fixture"}), patch("builtins.print"):
            code = full.run_supervisor(self.args, manifest, self.patients)
        state = json.loads((self.output / "supervisor_state.json").read_text())
        self.assertEqual(code, 0)
        self.assertEqual(state["phase"], "complete_with_errors")
        self.assertTrue(state["started_verified"])
        self.assertTrue(state["all_first_attempts_complete"])
        self.assertFalse(state["automatic_retries"])
        self.assertEqual(len(launches), 4)
        self.assertEqual([command[command.index("--agent_strategy") + 1] for command in launches[:3]], list(full.SYSTEMS))
        self.assertEqual(state["arms"]["single_agent"]["errors"], 1)
        with self.assertRaises(ValueError):
            full.run_supervisor(self.args, manifest, self.patients)

    def test_gzip_health_requires_owned_existing_artifacts(self):
        from scripts.audit_store import externalize_result_audit
        self.identity["audit_storage"] = "gzip"
        self.identity["audit_artifact_schema"] = 2
        rows = [{"question_id": question_id, "patient_fhir_id": patient, "latency": None, "error": None}
                for question_id, patient in self.patients.items()]
        rows[0].update(latency=1, tokens={"total": 10, "cost": None}, trace=[], retrieved_fhir_resources={})
        rows[0] = externalize_result_audit(rows[0], self.output)
        path = self.output / "full_single_agent.json"
        path.write_text(json.dumps(rows))
        (self.output / "full_single_agent.json.run.json").write_text(json.dumps(self.identity))
        good = full.arm_health("single_agent", self.output, self.patients, self.identity)
        self.assertEqual(good["audit_artifact_rows"], 1)
        self.assertEqual(good["missing_audit_artifacts"], 0)
        (self.output / rows[0]["audit_artifacts"]["path"]).unlink()
        bad = full.arm_health("single_agent", self.output, self.patients, self.identity)
        self.assertEqual(bad["missing_audit_artifacts"], 1)
        self.assertIn("missing_or_invalid_audit_artifact", bad["problems"])


if __name__ == "__main__":
    unittest.main()
