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

    def test_actual_parser_preserved_model_fields_cannot_trigger_runtime_retry(self):
        from agent.query_contract_agent import parse_contract, parse_audit
        question = "How many measurements?"
        content = {"requirements": [{"id": "c1", "kind": "aggregation", "quote": "How many",
                    "span": [0, 8], "expected": "count measurements", "extra": {"error": "timeout"}}],
                   "error": "APIConnectionError"}
        contract, contract_error = parse_contract(json.dumps(content), question)
        self.assertIsNone(contract_error)
        self.assertEqual(contract["error"], "APIConnectionError")
        self.assertEqual(contract["requirements"][0]["extra"]["error"], "timeout")
        audit, audit_error = parse_audit(json.dumps({"checks": [{"requirement_id": "c1", "verdict": "PASS",
                     "reason": "fixture", "errors": ["HTTP 503"]}], "repair": None,
                     "extra": {"errors": ["ReadTimeout"]}}), contract)
        self.assertIsNone(audit_error)
        self.assertEqual(audit["extra"]["errors"], ["ReadTimeout"])
        self.assertEqual(self.classify(row(revision={"contract": contract})), "accepted")
        self.assertEqual(self.classify(row(revision={"audit": audit})), "accepted")
        self.assertEqual(self.classify(row(revision={"contract": contract, "audit": audit})), "accepted")

    def test_host_failures_still_retry_when_model_subtrees_are_excluded(self):
        model_fields = {"contract": {"error": "timeout"}, "audit": {"errors": ["HTTP 503"]}}
        host_cases = [{"contract_error": "APIConnectionError"}, {"audit_error": "Request timed out"},
                      {"phases": [{"errors": ["HTTP 503"]}]},
                      {"final_check_issues": ["source_check: could not verify the original records (503 Server Error: Fault)"]},
                      {"targeted_proof": {"reason": "targeted check failed: ConnectionError"}}]
        for host in host_cases:
            with self.subTest(host=host):
                decision = runner.classify_attempt(row(revision={**model_fields, **host}))
                self.assertEqual(decision["status"], "runtime")
                self.assertTrue(decision["diagnostic_paths"])
                self.assertFalse(any(path.startswith(("revision.contract.", "revision.audit."))
                                     for path in decision["diagnostic_paths"]))

    def test_actual_toolbox_schema_errors_echoing_timeout_are_intrinsic(self):
        from types import SimpleNamespace
        from agent.framework import Toolbox
        def call(name, args):
            return SimpleNamespace(id="fixture", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
        box = Toolbox(("get_resources_by_patient_fhir_id",))
        unknown = box.call(call("get_resources_by_patient_fhir_id",
                               {"resource_type": "Observation", "patient_fhir_id": "fixture", "timeout": 1}))
        denied = box.call(call("timeout", {}))
        self.assertEqual(json.loads(unknown["content"])["error"], "Unknown arguments: timeout")
        self.assertEqual(json.loads(denied["content"])["error"], "Tool 'timeout' is not available to this role")
        for reply in (unknown, denied):
            with self.subTest(reply=reply):
                self.assertEqual(self.classify(row(trace=[reply])), "accepted")
                value = json.loads(reply["content"])["error"]
                self.assertEqual(self.classify(row(revision={"retrieval_ledger": [{"error": value}]})), "accepted")

    def test_actual_baseline_unexpected_timeout_keyword_is_intrinsic(self):
        from types import SimpleNamespace
        from agent.multi_turn_code_resource_agent import MultiTurnCodeResourceAgent
        agent = MultiTurnCodeResourceAgent.__new__(MultiTurnCodeResourceAgent)
        agent.verbose = False
        from tools.resource_tools import get_resources_by_patient_fhir_id as registered_tool
        get_resources_by_patient_fhir_id = registered_tool.__wrapped__
        call = SimpleNamespace(id="fixture", function=SimpleNamespace(name="get_resources_by_patient_fhir_id",
                               arguments=json.dumps({"resource_type": "Observation", "patient_fhir_id": "fixture", "timeout": 1})))
        with patch("agent.multi_turn_code_resource_agent.get_tool", return_value=get_resources_by_patient_fhir_id):
            reply = agent._handle_tool_call(call, {})
        self.assertIn("got an unexpected keyword argument 'timeout'", json.loads(reply["content"])["error"])
        self.assertEqual(self.classify(row(trace=[reply])), "accepted")

    def test_actual_resource_validation_and_network_timeout_remain_distinct(self):
        from types import SimpleNamespace
        from agent.framework import Toolbox
        from tools.resource_tools import _validate_search
        box = Toolbox(("get_resources_by_patient_fhir_id",))
        call = SimpleNamespace(id="fixture", function=SimpleNamespace(name="get_resources_by_patient_fhir_id",
                               arguments=json.dumps({"resource_type": "timeout", "patient_fhir_id": "fixture"})))
        def validate_only(name, args):
            _validate_search(args["resource_type"], args["patient_fhir_id"])
        with patch.object(box, "_retrieve", side_effect=validate_only):
            invalid_type = box.call(call)
        self.assertEqual(json.loads(invalid_type["content"])["error"], "Unsupported FHIR resource type")
        self.assertEqual(self.classify(row(trace=[invalid_type])), "accepted")
        with patch.object(box, "_retrieve", side_effect=TimeoutError("Actual retrieval timed out")):
            genuine_timeout = box.call(call)
        self.assertEqual(self.classify(row(trace=[genuine_timeout])), "runtime")
        for diagnostic in ("Request timed out", "Unknown arguments: timeout; request actually timed out"):
            # Only the fixed leading validation template is intrinsic. A real
            # provider message with different leading context remains runtime.
            expected = "accepted" if diagnostic.startswith("Unknown arguments:") else "runtime"
            self.assertEqual(self.classify(row(trace=[{"role": "tool", "name": "get_resources_by_patient_fhir_id",
                             "content": json.dumps({"error": diagnostic})}])), expected)

    def test_fixed_json_argument_and_scope_validation_are_intrinsic(self):
        from types import SimpleNamespace
        from agent.framework import Toolbox
        box = Toolbox(("get_resources_by_patient_fhir_id",))
        for malformed in ("[]", "{timeout}", '{"timeout":', '{"timeout":"unterminated}'):
            call = SimpleNamespace(id="fixture", function=SimpleNamespace(name="get_resources_by_patient_fhir_id", arguments=malformed))
            reply = box.call(call)
            self.assertEqual(self.classify(row(trace=[reply])), "accepted")
        for error in ("A non-empty FHIR resource ID is required", "Retrieve only the patient identified in this question"):
            self.assertEqual(self.classify(row(trace=[{"role": "tool", "name": "get_resources_by_patient_fhir_id",
                             "content": json.dumps({"error": error})}])), "accepted")
        for error in ("SDKClient() got an unexpected keyword argument 'timeout'",
                      "Max retries exceeded: SDKClient() got an unexpected keyword argument 'timeout'",
                      "TypeError: SDKClient() got an unexpected keyword argument 'timeout'"):
            self.assertEqual(self.classify(row(error=error)), "attention")
        self.assertEqual(self.classify(row(error="Max retries exceeded: Request timed out")), "runtime")

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

    def test_resume_preserves_original_start_and_records_resumed_at(self):
        original_start = "2026-10-16T00:00:00+00:00"
        resumed_at = "2026-10-16T01:00:00+00:00"
        runner.atomic_json(self.directory / "supervisor_state.json",
                           {"supervisor_pid": 0, "started_at": original_start, "phase": "interrupted", "active": {}})
        class Child:
            returncode = 0
            pid = 999999
            def poll(self):
                return 0
            def wait(self, timeout=None):
                return 0
        def launch(command, **kwargs):
            if command != ["fixture-analysis"]:
                output = Path(command[command.index("--output") + 1])
                self.write_batch_results(json.loads(output.with_suffix(".batch.json").read_text()))
            return Child()
        with patch.object(runner, "validate_manifest"), patch.object(runner.shutil, "which", return_value=None), \
             patch.object(runner.subprocess, "Popen", side_effect=launch), patch.object(runner.time, "sleep"), \
             patch.object(runner, "utc_now", return_value=resumed_at):
            result = runner.run_supervisor(self.directory, self.manifest, self.patients, resume=True)
        self.assertEqual(result, 0)
        state = json.loads((self.directory / "supervisor_state.json").read_text())
        self.assertEqual(state["started_at"], original_start)
        self.assertEqual(state["resumed_at"], resumed_at)
        self.assertEqual(runner.compact_state(state)["resumed_at"], resumed_at)
        self.assertEqual(state["phase"], "complete")

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
