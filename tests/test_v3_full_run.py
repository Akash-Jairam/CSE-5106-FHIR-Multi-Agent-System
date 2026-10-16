"""Network-free tests of answer-blind infrastructure recovery and provenance."""
import csv
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import v3_full_run as runner
from scripts.audit_store import externalize_result_audit, restore_result_audit


def row(answer="wrong", error=None, trace=None, revision=None):
    return {"question_id": "q0", "patient_fhir_id": "p0", "agent_answer": answer,
            "evidence": [], "latency": 7.5, "error": error,
            "tokens": {"total": 25, "prompt": 20, "completion": 5, "cached_prompt": 2,
                       "cost": None, "per_call": [{"prompt": 20, "completion": 5, "cached": 2, "usage_available": True}]},
            "budget": {"stopped": None}, "trace": trace or [], "revision": revision}


class ClassifierTests(unittest.TestCase):
    def classify(self, value):
        return runner.classify_attempt(value)["status"]

    def test_completed_wrong_and_null_answers_are_final_without_gold(self):
        self.assertEqual(self.classify(row("wrong")), "accepted")
        self.assertEqual(self.classify(row(None)), "accepted")

    def test_provider_network_http_and_auxiliary_timeouts_retry(self):
        for error in ("Max retries exceeded: Request timed out.", "Error code: 408 - service timeout",
                      "Error code: 429", "HTTP 503", "APIConnectionError", "SSLError: certificate verify failed",
                      "Cannot send a request, as the client has been closed.",
                      "502 Server Error: Bad Gateway for url: fixture", "Max retries exceeded: EXCEPTION: Empty or None output."):
            with self.subTest(error=error):
                self.assertEqual(self.classify(row(error=error)), "runtime")
                self.assertEqual(self.classify(row(revision={"phases": [{"errors": [error]}]})), "runtime")

    def test_host_targeted_timeout_and_ambiguous_external_http(self):
        self.assertEqual(self.classify(row(revision={"targeted_proof": {"reason": "targeted check failed: ConnectionError"}})), "runtime")
        self.assertEqual(self.classify(row(revision={"targeted_proof": {"reason": "targeted check failed: HTTPError"}})), "attention")

    def test_host_source_check_issue_lists_retry_only_runtime(self):
        error = "source_check: could not verify the original records (ReadTimeout: endpoint timed out)"
        self.assertEqual(self.classify(row(revision={"final_check_issues": [error]})), "runtime")
        for detail in ("503 Server Error: Service Unavailable", "503 Server Error: Fault", "502 Server Error: Fault"):
            host_error = f"source_check: could not verify the original records ({detail})"
            self.assertEqual(self.classify(row(revision={"final_check_issues": [host_error]})), "runtime")
        trace = [{"role": "tool", "name": "check_fields", "content": json.dumps({"issues": [error]})}]
        self.assertEqual(self.classify(row(trace=trace)), "runtime")
        self.assertEqual(self.classify(row(revision={"final_check_issues": ["source_check: invalid contributing field"]})), "accepted")

    def test_retrieval_json_tool_failure_is_runtime(self):
        trace = [{"role": "tool", "name": "get_resources_by_patient_fhir_id",
                  "content": json.dumps({"error": "ReadTimeout: endpoint timed out"})}]
        self.assertEqual(self.classify(row(trace=trace)), "runtime")
        trace[0]["content"] = json.dumps({"error": "Unknown external retrieval condition"})
        self.assertEqual(self.classify(row(trace=trace)), "attention")

    def test_host_python_timeouts_retry_but_model_code_strings_do_not(self):
        for error in ("Python execution exceeded 60 seconds", "Python sandbox worker failed",
                      "Python sandbox unavailable: requires sandbox-exec", "Python execution failed: OSError"):
            trace = [{"role": "tool", "name": "execute_python_code", "content": json.dumps({"error": error})}]
            self.assertEqual(self.classify(row(trace=trace)), "runtime")
        for error in ("KeyError: '408'", "ValueError: '503'", "RuntimeError: timeout", "timeout must be positive"):
            trace = [{"role": "tool", "name": "execute_python_code", "content": json.dumps({"error": error})}]
            self.assertEqual(self.classify(row(trace=trace)), "accepted")

    def test_intrinsic_terminal_limits_and_invalid_output_never_retry(self):
        for error in ("Solver did not finalize within its 20 model-call allocation", "Invalid output: schema wrong",
                      "Input tokens exceeded: 40000 > 32000", "NameError: x not defined"):
            self.assertEqual(self.classify(row(error=error)), "accepted")
        value = row(None)
        value["budget"]["stopped"] = "token budget"
        self.assertEqual(self.classify(value), "accepted")

    def test_natural_language_and_executable_code_are_not_diagnostics(self):
        value = row("timeout 408")
        value["question"] = "What does timeout 429 mean?"
        value["revision"] = {"code": "raise RuntimeError('timeout')", "audit": {"reason": "ConnectionError is a term"}}
        value["trace"] = [{"role": "assistant", "content": "timeout"}, {"role": "user", "content": "408"}]
        self.assertEqual(self.classify(value), "accepted")

    def test_unknown_and_accounting_failures_require_attention(self):
        self.assertEqual(self.classify(row(error="Unexpected internal event")), "attention")
        self.assertEqual(self.classify(row(error="Provider accounting failure: unknown usage")), "attention")
        value = row()
        value["tokens"]["per_call"][0]["usage_available"] = False
        self.assertEqual(self.classify(value), "attention")
        value["latency"] = float("nan")
        self.assertEqual(self.classify(value), "attention")

    def test_classification_restores_owned_gzip_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = row(trace=[{"role": "tool", "name": "execute_python_code", "content": json.dumps({"error": "Python execution exceeded 60 seconds"})}])
            compact = externalize_result_audit(source, temporary, question_id="q0")
            self.assertEqual(compact["trace"], [])
            self.assertEqual(runner.classify_attempt(compact, temporary)["status"], "runtime")


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "run"
        self.directory.mkdir()
        self.input = self.root / "dataset.csv"
        with self.input.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("question_id", "patient_fhir_id", "question", "true_answer", "assumption"))
            writer.writeheader()
            writer.writerows({"question_id": f"q{i}", "patient_fhir_id": f"p{i}", "question": "Fixture", "true_answer": "gold", "assumption": ""} for i in range(2))
        self.families = self.root / "families.csv"
        self.families.write_text("question_id,family,template\nq0,fixture,fixture\nq1,fixture,fixture\n")
        args = runner.build_parser().parse_args(["run", "--run-dir", str(self.directory), "--input", str(self.input), "--families", str(self.families), "--min-free-gb", "0", "--retry-delay-seconds", "0", "--batch-size", "2"])
        with patch("scripts.full_benchmark.QUESTIONS", 2):
            self.manifest = runner.manifest_for(args)
        self.manifest["analysis_command"] = ["fixture-analysis"]
        runner.atomic_json(self.directory / "run_manifest.json", self.manifest)
        self.patients = {"q0": "p0", "q1": "p1"}

    def write_batch_results(self, batch, runtime=False):
        output = Path(batch["output"])
        records = []
        for qid in batch["question_ids"]:
            value = row("wrong", "APIConnectionError" if runtime and qid == "q0" else None,
                        trace=[{"role": "assistant", "content": "Fixture audit"}])
            value.update(question_id=qid, patient_fhir_id=self.patients[qid])
            records.append(externalize_result_audit(value, output.parent, question_id=qid))
        runner.atomic_json(output, records)
        runner.atomic_json(str(output) + ".run.json", batch["run_identity"])

    def test_sealed_round_selects_first_clean_and_keeps_wrong(self):
        first_path, first = runner.create_batch(self.directory, "multi_agent", set(self.patients), self.manifest, 1)
        self.write_batch_results(first, runtime=True)
        runner.seal_batch(first_path, 0, "fixture")
        accepted, choices, retry, attention, counts = runner.collect_accepted(self.directory, self.manifest, self.patients)
        self.assertEqual(set(accepted["multi_agent"]), {"q1"})
        self.assertEqual(accepted["multi_agent"]["q1"]["agent_answer"], "wrong")
        second_path, second = runner.create_batch(self.directory, "multi_agent", {"q0"}, self.manifest, 2)
        self.write_batch_results(second)
        runner.seal_batch(second_path, 0, "fixture")
        accepted, choices, retry, attention, counts = runner.collect_accepted(self.directory, self.manifest, self.patients)
        self.assertEqual(set(accepted["multi_agent"]), set(self.patients))
        self.assertEqual(choices["multi_agent"]["q0"]["round"], 2)
        self.assertEqual(choices["multi_agent"]["q1"]["round"], 1)
        self.assertEqual(counts["multi_agent"]["runtime_failures"], 1)
        self.assertEqual(attention, [])
        self.assertNotIn("attempt_history", accepted["multi_agent"]["q0"])
        self.assertTrue(restore_result_audit(accepted["multi_agent"]["q0"], self.directory)["trace"])

    def test_sealed_mutation_and_accepted_rerun_are_rejected(self):
        path, batch = runner.create_batch(self.directory, "multi_agent", set(self.patients), self.manifest, 1)
        self.write_batch_results(batch)
        runner.seal_batch(path, 0, "fixture")
        second_path, second = runner.create_batch(self.directory, "multi_agent", {"q0"}, self.manifest, 2)
        self.write_batch_results(second)
        runner.seal_batch(second_path, 0, "fixture")
        with self.assertRaisesRegex(ValueError, "was rerun"):
            runner.collect_accepted(self.directory, self.manifest, self.patients)
        Path(batch["output"]).write_text("[]")
        with self.assertRaisesRegex(ValueError, "Sealed attempt"):
            runner.collect_accepted(self.directory, self.manifest, self.patients)

    def test_resume_seals_partial_without_replacing_it(self):
        path, batch = runner.create_batch(self.directory, "multi_agent", set(self.patients), self.manifest, 1)
        self.write_batch_results(batch)
        runner.recover_unsealed(self.directory)
        self.assertTrue(path.with_suffix(".seal.json").exists())
        accepted, *_ = runner.collect_accepted(self.directory, self.manifest, self.patients)
        self.assertEqual(len(accepted["multi_agent"]), 2)

    def test_supervisor_finishes_team_before_controls_and_retries_only_runtime(self):
        launches = []
        owner = self
        class Child:
            returncode = 0
            pid = 999999
            def poll(self):
                return 0
            def wait(self, timeout=None):
                return 0
        def launch(command, **kwargs):
            self.assertTrue(kwargs["start_new_session"])
            if command == ["fixture-analysis"]:
                launches.append("analysis")
                return Child()
            strategy = command[command.index("--agent_strategy") + 1]
            output = Path(command[command.index("--output") + 1])
            batch = json.loads(output.with_suffix(".batch.json").read_text())
            owner.write_batch_results(batch, runtime=strategy == "multi_agent_v3" and batch["round"] == 1)
            launches.append(strategy)
            return Child()
        with patch.object(runner, "validate_manifest"), patch.object(runner.shutil, "which", return_value=None), \
             patch.object(runner.subprocess, "Popen", side_effect=launch), patch.object(runner.time, "sleep"):
            result = runner.run_supervisor(self.directory, self.manifest, self.patients)
        self.assertEqual(result, 0)
        self.assertEqual(launches[:2], ["multi_agent_v3", "multi_agent_v3"])
        self.assertEqual(set(launches[2:4]), {"single_agent", "direct_llm"})
        self.assertEqual(launches[-1], "analysis")
        ledger = json.loads((self.directory / "acceptance.json").read_text())
        self.assertTrue(ledger["complete"])
        self.assertEqual(ledger["arms"]["multi_agent"]["q0"]["round"], 2)
        self.assertEqual(ledger["arms"]["multi_agent"]["q1"]["round"], 1)
        self.assertEqual(json.loads((self.directory / "supervisor_state.json").read_text())["phase"], "complete")

    def test_injected_unknown_classification_halts_without_scoring_wrong(self):
        path, batch = runner.create_batch(self.directory, "multi_agent", set(self.patients), self.manifest, 1)
        self.write_batch_results(batch)
        runner.seal_batch(path, 0, "fixture")
        classifier = lambda row, root: {"status": "attention", "reason": "fixture_unknown"}
        accepted, _, _, attention, _ = runner.collect_accepted(self.directory, self.manifest, self.patients, classifier)
        self.assertEqual(accepted["multi_agent"], {})
        self.assertEqual(len(attention), 2)


if __name__ == "__main__":
    unittest.main()
