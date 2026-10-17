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

    def prepare_selected_resume(self):
        path, batch = runner.create_batch(self.directory, "multi_agent", set(self.patients), self.manifest, 1)
        self.write_batch_results(batch)
        runner.seal_batch(path, 0, "fixture")
        accepted, choices, _, _, counts = runner.collect_accepted(self.directory, self.manifest, self.patients)
        runner.save_accepted(self.directory, self.manifest, accepted, choices, counts)
        self.manifest["selected_arms"] = ["single_agent"]
        runner.atomic_json(self.directory / "run_manifest.json", self.manifest)
        runner.atomic_json(self.directory / "supervisor_state.json", {
            "supervisor_pid": 0, "started_at": "2026-10-16T04:16:57+00:00",
            "phase": "attention", "active": {}, "issues": [{"arm": "direct_llm"}]})

    def parked_evidence(self):
        files = {name: (self.directory / name).read_bytes() for arm in ("multi_agent", "direct_llm")
                 for name in (f"full_{arm}.json", f"full_{arm}.json.run.json")}
        ledger = json.loads((self.directory / "acceptance.json").read_text())
        slices = {section: {arm: ledger[section][arm] for arm in ("multi_agent", "direct_llm")}
                  for section in ("arms", "counts")}
        return files, slices

    def selected_run(self, launch, **kwargs):
        with patch.object(runner, "validate_manifest"), patch.object(runner.shutil, "which", return_value=None), \
             patch.object(runner.subprocess, "Popen", side_effect=launch), patch.object(runner.time, "sleep"):
            return runner.run_supervisor(self.directory, self.manifest, self.patients, resume=True,
                                         only_arms=["single_agent"], **kwargs)

    def test_selected_scope_requires_exact_pin_explicit_flag_and_resume(self):
        self.assertEqual(runner.execution_arms(self.manifest), tuple(runner.ARMS))
        for pin, flags, resume in ((None, ["single_agent"], True), (["single_agent"], None, True),
                                   (["single_agent"], ["single_agent"], False),
                                   (["single_agent", "direct_llm"], ["single_agent"], True),
                                   (["single_agent"], ["single_agent", "single_agent"], True),
                                   (["direct_llm"], ["direct_llm"], True), ([], None, True)):
            value = deepcopy(self.manifest)
            if pin is not None:
                value["selected_arms"] = pin
            with self.subTest(pin=pin, flags=flags, resume=resume), self.assertRaisesRegex(ValueError, "Selected arms"):
                runner.execution_arms(value, flags, resume)
        for command in ("plan", "run", "status"):
            with self.subTest(command=command), patch.object(runner, "manifest_for") as make, \
                 self.assertRaisesRegex(ValueError, "only for resume"):
                runner.main([command, "--run-dir", str(self.directory), "--only-arm", "single_agent"])
            make.assert_not_called()
        self.manifest["selected_arms"] = ["single_agent"]
        runner.atomic_json(self.directory / "run_manifest.json", self.manifest)
        original = {path.name: path.read_bytes() for path in self.directory.iterdir() if path.is_file()}
        with patch.object(runner, "validate_manifest") as validate, patch.object(runner, "run_supervisor") as run, \
             self.assertRaisesRegex(ValueError, "Selected arms"):
            runner.main(["resume", "--run-dir", str(self.directory)])
        validate.assert_not_called()
        run.assert_not_called()
        self.assertEqual(original, {path.name: path.read_bytes() for path in self.directory.iterdir() if path.is_file()})

    def test_selected_resume_reuses_clean_wrong_ignores_deferred_attention_and_stops(self):
        self.prepare_selected_resume()
        # A sealed deferred attention row must remain unclassified, and an
        # unrelated unsealed batch must remain unsealed during scoped recovery.
        direct_path, direct = runner.create_batch(self.directory, "direct_llm", set(self.patients), self.manifest, 1)
        self.write_batch_results(direct)
        values = json.loads(Path(direct["output"]).read_text())
        values[0]["error"] = "Unknown provider rejection"
        runner.atomic_json(direct["output"], values)
        runner.seal_batch(direct_path, 0, "fixture_attention")
        unsealed, _ = runner.create_batch(self.directory, "direct_llm", {"q0"}, self.manifest, 2)
        single_path, single = runner.create_batch(self.directory, "single_agent", set(self.patients), self.manifest, 1)
        self.write_batch_results(single, runtime=True)
        original_raw = Path(single["output"]).read_bytes()
        parked = self.parked_evidence()
        launches = []
        class Child:
            returncode = 0
            pid = 999999
            def poll(self):
                return 0
            def wait(self, timeout=None):
                return 0
        def launch(command, **kwargs):
            strategy = command[command.index("--agent_strategy") + 1]
            self.assertEqual(strategy, "single_agent")
            batch = json.loads(Path(command[command.index("--output") + 1]).with_suffix(".batch.json").read_text())
            launches.append(batch["question_ids"])
            self.write_batch_results(batch)
            return Child()
        self.assertEqual(self.selected_run(launch), 0)
        self.assertEqual(launches, [["q0"]])
        self.assertEqual(self.parked_evidence(), parked)
        self.assertFalse(unsealed.with_suffix(".seal.json").exists())
        self.assertTrue(single_path.with_suffix(".seal.json").exists())
        self.assertEqual(Path(single["output"]).read_bytes(), original_raw)
        ledger = json.loads((self.directory / "acceptance.json").read_text())
        self.assertFalse(ledger["complete"])
        self.assertEqual(ledger["arms"]["single_agent"]["q1"]["round"], 1)
        self.assertEqual(ledger["arms"]["single_agent"]["q0"]["round"], 2)
        self.assertEqual(ledger["counts"]["single_agent"]["runtime_failures"], 1)
        self.assertEqual({value["agent_answer"] for value in json.loads((self.directory / "full_single_agent.json").read_text())}, {"wrong"})
        state = json.loads((self.directory / "supervisor_state.json").read_text())
        self.assertEqual(state["phase"], "complete_selected_arms")
        self.assertEqual(state["selected_arms"], ["single_agent"])
        self.assertEqual(state["accepted"], {"multi_agent": 2, "single_agent": 2, "direct_llm": 0})
        self.assertTrue(state["selected_scope_complete"])
        self.assertTrue(state["pending_scoped_export"])
        self.assertFalse(state["benchmark_complete"])
        self.assertEqual(state["analysis_status"], "pending_scoped_export")
        self.assertNotIn("analysis_return_code", state)
        self.assertEqual(state["started_at"], "2026-10-16T04:16:57+00:00")
        self.assertFalse((self.directory / "analysis.log").exists())
        # Repeating a completed selected resume cannot launch anything, and
        # keeps the original selected wall-time origin for operational analysis.
        with patch.object(runner.subprocess, "Popen") as child:
            self.assertEqual(self.selected_run(child), 0)
        child.assert_not_called()
        after = json.loads((self.directory / "supervisor_state.json").read_text())
        self.assertEqual(after["scope_started_at"], state["scope_started_at"])
        self.assertEqual(self.parked_evidence(), parked)

    def test_selected_resume_rejects_incomplete_or_changed_v3_before_launch_or_recovery(self):
        self.prepare_selected_resume()
        unsealed, _ = runner.create_batch(self.directory, "single_agent", {"q0"}, self.manifest, 1)
        original = (self.directory / "full_multi_agent.json").read_bytes()
        cases = ([], [dict(row(), patient_fhir_id="other"), dict(row(), question_id="q1", patient_fhir_id="p1")])
        for records in cases:
            runner.atomic_json(self.directory / "full_multi_agent.json", records)
            with self.subTest(records=records), patch.object(runner.subprocess, "Popen") as child:
                self.assertEqual(self.selected_run(child), 2)
            child.assert_not_called()
            self.assertFalse(unsealed.with_suffix(".seal.json").exists())
        (self.directory / "full_multi_agent.json").write_bytes(original)
        runner.atomic_json(self.directory / "full_multi_agent.json.run.json", {"different": True})
        with patch.object(runner.subprocess, "Popen") as child:
            self.assertEqual(self.selected_run(child), 2)
        child.assert_not_called()
        self.assertFalse(unsealed.with_suffix(".seal.json").exists())

    def test_selected_retry_limit_and_unknown_attention_never_launch_or_change_parked_arms(self):
        for failure in ("APIConnectionError", "Unknown provider rejection"):
            with self.subTest(failure=failure):
                # Each case uses a fresh fixture so a prior terminal decision
                # cannot affect the second branch of the guard.
                fixture = SupervisorTests()
                fixture.setUp()
                try:
                    fixture.prepare_selected_resume()
                    fixture.manifest["settings"]["max_runtime_retries"] = 0
                    runner.atomic_json(fixture.directory / "run_manifest.json", fixture.manifest)
                    path, batch = runner.create_batch(fixture.directory, "single_agent", set(fixture.patients), fixture.manifest, 1)
                    fixture.write_batch_results(batch)
                    records = json.loads(Path(batch["output"]).read_text())
                    records[0]["error"] = failure
                    runner.atomic_json(batch["output"], records)
                    runner.seal_batch(path, 0, "fixture_failure")
                    parked = fixture.parked_evidence()
                    with patch.object(runner.subprocess, "Popen") as child:
                        self.assertEqual(fixture.selected_run(child), 2)
                    child.assert_not_called()
                    self.assertEqual(fixture.parked_evidence(), parked)
                    state = json.loads((fixture.directory / "supervisor_state.json").read_text())
                    self.assertEqual(state["reason"], "runtime_retry_limit" if failure == "APIConnectionError" else "unclassified_or_invalid_attempt")
                    self.assertFalse(state["selected_scope_complete"])
                    ledger = json.loads((fixture.directory / "acceptance.json").read_text())
                    self.assertEqual(set(ledger["arms"]["single_agent"]), {"q1"})
                    self.assertFalse(ledger["complete"])
                finally:
                    fixture.doCleanups()

    def test_selected_completed_question_rerun_remains_rejected(self):
        self.prepare_selected_resume()
        for number in (1, 2):
            path, batch = runner.create_batch(self.directory, "single_agent", {"q0"}, self.manifest, number)
            self.write_batch_results(batch)
            runner.seal_batch(path, 0, "fixture")
        parked = self.parked_evidence()
        with patch.object(runner.subprocess, "Popen") as child:
            self.assertEqual(self.selected_run(child), 2)
        child.assert_not_called()
        self.assertEqual(self.parked_evidence(), parked)
        state = json.loads((self.directory / "supervisor_state.json").read_text())
        self.assertIn("Completed accepted question was rerun", state["safe_message"])

    def test_selected_new_attention_batch_persists_its_clean_rows_before_stopping(self):
        self.prepare_selected_resume()
        parked = self.parked_evidence()
        launches = []
        class Child:
            returncode = 0
            pid = 999999
            def poll(self):
                return 0
            def wait(self, timeout=None):
                return 0
        def launch(command, **kwargs):
            self.assertEqual(command[command.index("--agent_strategy") + 1], "single_agent")
            output = Path(command[command.index("--output") + 1])
            batch = json.loads(output.with_suffix(".batch.json").read_text())
            self.write_batch_results(batch)
            records = json.loads(output.read_text())
            records[0]["error"] = "Unknown provider rejection"
            runner.atomic_json(output, records)
            launches.append(batch["question_ids"])
            return Child()
        self.assertEqual(self.selected_run(launch), 2)
        self.assertEqual(launches, [["q0", "q1"]])
        self.assertEqual(self.parked_evidence(), parked)
        ledger = json.loads((self.directory / "acceptance.json").read_text())
        self.assertEqual(set(ledger["arms"]["single_agent"]), {"q1"})
        self.assertFalse(ledger["complete"])
        state = json.loads((self.directory / "supervisor_state.json").read_text())
        self.assertEqual(state["reason"], "unclassified_or_invalid_attempt")
        self.assertEqual(state["accepted"]["single_agent"], 1)
        self.assertEqual(state["counts"]["single_agent"]["raw_attempts"], 2)
        self.assertFalse(state["selected_scope_complete"])


if __name__ == "__main__":
    unittest.main()
