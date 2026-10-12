"""P3-22: revision transitions, unsupported claims, empty-answer correctness, and omitted-resource recovery.

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

import pandas as pd

import run_agent as runner
from scripts import revisions
from scripts import run_experiments as experiments

REFERENCE = {"question_id": "q1", "question": "Q", "true_answer": "[[1]]", "template": "", "family": "lookup_counting"}
EMPTY = {**REFERENCE, "question_id": "q2", "true_answer": "[]"}
SPEC = {"removed": {"resource_type": "Observation", "id": "o9"}}


def fake_score(answer, true_answer, *_):
    if true_answer == "[]":
        return {"typed_correct": int(answer is None), "answer_type": "empty"}
    return {"typed_correct": int(answer == "RIGHT"), "answer_type": "category"}


def cite(rid="Observation/o9", value=7):
    return {"resource_id": rid, "field_path": "valueQuantity.value", "value": value}


def row(answer, draft=None, revised=False, evidence=(), error=None, fetched=None, repair_role=None):
    revision = None if draft is None else {"draft": {"answer": draft}, "revised": revised, "repair_role": repair_role}
    return {"agent_answer": answer, "evidence": list(evidence), "revision": revision, "error": error,
            "agent_fhir_resources": fetched or {}}


def verified_row(answer, contributing=(), fetched=None, trace=True):
    result = row(answer, fetched=fetched)
    candidate = {"answer": answer, "evidence": [], "contributing_ids": list(contributing)}
    result["revision"] = {"draft": candidate, "final": candidate, "final_check_issues": [], "messages": [
        {"sender": "analysis", "recipient": "controller", "kind": "draft", "payload": candidate},
        {"sender": "verification", "recipient": "controller", "kind": "verdict",
         "payload": {"issues": [], "recomputed_answer": answer}}]}
    if trace:
        result["trace"] = [
            {"role": "system", "content": "You are the verification agent in a team."},
            {"role": "tool", "name": "get_resources_by_patient_fhir_id", "content": '{"ids": []}'},
            {"role": "tool", "name": "execute_python_code", "content": json.dumps({"answer": answer})}]
    return result


class RecordTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(revisions, "score_answer", side_effect=fake_score)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_transitions_count_a_withheld_correct_draft_as_right_to_wrong(self):
        withheld = revisions.question_record("arm", row(None, draft="RIGHT"), REFERENCE)
        self.assertEqual((withheld["transition"], withheld["answered"], withheld["unsupported"]),
                         ("right_to_wrong", False, None))
        repaired = revisions.question_record("arm", row("RIGHT", draft="WRONG", revised=True, evidence=[cite()],
                                                        repair_role="retrieval"), REFERENCE)
        self.assertEqual((repaired["transition"], repaired["revised"], repaired["repair_role"]),
                         ("wrong_to_right", True, "retrieval"))
        self.assertIsNone(revisions.question_record("arm", row("RIGHT"), REFERENCE)["transition"])

    def test_unsupported_claims_and_errors(self):
        self.assertTrue(revisions.question_record("arm", row("RIGHT"), REFERENCE)["unsupported"])
        resources = {"Observation/o9": {"resourceType": "Observation", "id": "o9", "valueQuantity": {"value": 7}}}
        valid = revisions.question_record("arm", row("RIGHT", evidence=[cite()]), REFERENCE, resources=resources)
        self.assertEqual((valid["evidence_invalid"], valid["unsupported"]), (0, False))
        wrong = revisions.question_record("arm", row("RIGHT", evidence=[cite(value=8)]), REFERENCE, resources=resources)
        self.assertEqual((wrong["evidence_invalid"], wrong["unsupported"]), (1, True))
        errored = revisions.question_record("arm", row("RIGHT", error="boom"), REFERENCE)
        self.assertEqual((errored["correct"], errored["answered"]), (0, False))

    def test_failed_attempts_are_wrong_without_false_transitions_or_abstentions(self):
        records = [revisions.question_record("arm", row("RIGHT", draft="RIGHT", revised=True, error="provider down"), REFERENCE),
                   revisions.question_record("arm", row("RIGHT", draft="WRONG", revised=True, error="provider down"), EMPTY)]
        self.assertEqual([(r["correct"], r["transition"], r["abstained"]) for r in records],
                         [(0, None, False), (0, None, False)])
        summary = revisions.summarize_arm(records, False)
        self.assertEqual(summary["accuracy"], {"n": 2, "count": 0, "rate": 0.0})
        self.assertEqual(summary["empty_reference"], {"n": 1, "count": 0, "rate": 0.0})
        self.assertEqual(summary["false_abstention"]["count"], 0)
        self.assertEqual(summary["runs_scored_transitions"], 0)
        self.assertEqual(summary["transition_excluded_errors"], 2)
        self.assertEqual(summary["transitions"]["right_to_wrong"], 0)

    def test_retry_success_does_not_replace_first_attempt_outcome(self):
        saved = row("RIGHT", draft="WRONG", revised=True, evidence=[cite()])
        saved["attempt_history"] = [row(None, draft="RIGHT", error="provider down")]
        record = revisions.question_record("arm", saved, REFERENCE)
        self.assertEqual((record["correct"], record["error"], record["transition"]), (0, True, None))
        self.assertEqual((record["attempt_count"], record["failed_attempts"]), (2, 1))

    def test_verified_empty_result_is_supported_without_nonexistent_citations(self):
        for answer in (0, False, [], None):
            with self.subTest(answer=answer):
                record = revisions.question_record("arm", verified_row(answer), EMPTY)
                self.assertEqual((record["empty_verified"], record["answered"], record["abstained"], record["unsupported"]),
                                 (True, True, False, False))
        unknown = revisions.question_record("arm", verified_row(0, trace=False), EMPTY)
        self.assertIsNone(unknown["empty_verified"])
        self.assertIsNone(unknown["unsupported"])
        unverified = revisions.question_record("arm", row(0), EMPTY)
        self.assertTrue(unverified["unsupported"])

    def test_stale_or_failed_verifier_tools_do_not_establish_empty_result(self):
        saved = verified_row(0)
        saved["trace"].append({"role": "tool", "name": "get_resources_by_patient_fhir_id",
                               "content": '{"error": "source down"}'})
        record = revisions.question_record("arm", saved, EMPTY)
        self.assertIsNone(record["empty_verified"])
        saved = verified_row(0)
        saved["trace"][-1]["content"] = '{"answer": 1}'
        record = revisions.question_record("arm", saved, EMPTY)
        self.assertFalse(record["empty_verified"])
        self.assertTrue(record["unsupported"])
        saved = verified_row(0)
        saved["trace"].append({"role": "system", "content": "You are the verification agent in a fresh check."})
        self.assertIsNone(revisions.question_record("arm", saved, EMPTY)["empty_verified"])

    def test_malformed_citation_is_unsupported_without_source_lookup(self):
        record = revisions.question_record("arm", row("RIGHT", evidence=[{}]), REFERENCE)
        self.assertTrue(record["unsupported"])
        valid = revisions.question_record("arm", row("RIGHT", evidence=[cite()]), REFERENCE)
        summary = revisions.summarize_arm([valid], False)
        self.assertFalse(summary["unsupported"]["support_completeness_checked"])
        self.assertTrue(summary["unsupported"]["lower_bound"])

    def test_omitted_resource_retrieved_and_cited(self):
        record = revisions.question_record("arm", row("RIGHT", evidence=[cite("o9")],
                                                      fetched={"Observation": ["o1", "o9"]}), REFERENCE, SPEC)
        self.assertEqual((record["omitted_resource"], record["omitted_retrieved"], record["omitted_cited"]),
                         ("Observation/o9", True, True))
        missed = revisions.question_record("arm", row("WRONG", evidence=[cite("Encounter/o9")]), REFERENCE, SPEC)
        self.assertEqual((missed["omitted_retrieved"], missed["omitted_cited"]), (False, False))

    def test_recovery_requires_answer_use_and_keeps_verifier_fetch_separate(self):
        saved = verified_row("RIGHT", contributing=["Observation/o9"], fetched={"Observation": ["o9"]})
        recovered = revisions.question_record("arm", saved, REFERENCE, SPEC)
        self.assertEqual((recovered["omitted_fetched"], recovered["omitted_retrieved"], recovered["omitted_used"],
                          recovered["omitted_verifier_only"], recovered["omitted_recovered"]),
                         (True, True, True, False, True))
        saved = verified_row("WRONG")
        saved["trace"][1]["content"] = '{"ids": ["Observation/o9"]}'
        verifier_only = revisions.question_record("arm", saved, REFERENCE, SPEC)
        self.assertEqual((verifier_only["omitted_fetched"], verifier_only["omitted_retrieved"],
                          verifier_only["omitted_used"], verifier_only["omitted_verifier_only"],
                          verifier_only["omitted_recovered"]), (True, False, False, True, False))
        no_use_metadata = revisions.question_record("arm", row("RIGHT", fetched={"Observation": ["o9"]}), REFERENCE, SPEC)
        self.assertIsNone(no_use_metadata["omitted_used"])
        self.assertIsNone(no_use_metadata["omitted_recovered"])
        summary = revisions.summarize_recovery([recovered, verifier_only, no_use_metadata])
        self.assertEqual(summary["recovered"], {"n": 2, "count": 1, "rate": 0.5})
        self.assertEqual(summary["unknown"]["used"], 1)

    def test_bare_citation_with_ambiguous_resource_type_is_unknown(self):
        record = revisions.question_record("arm", row("RIGHT", evidence=[cite("o9")],
                                                      fetched={"Observation": ["o9"], "Encounter": ["o9"]}), REFERENCE, SPEC)
        self.assertIsNone(record["omitted_cited"])

    def test_arm_and_recovery_summaries(self):
        records = [revisions.question_record("arm", r, ref) for r, ref in (
            (row("RIGHT", draft="WRONG", revised=True, evidence=[cite()]), REFERENCE),
            (row(None, draft="RIGHT"), {**REFERENCE, "question_id": "q3"}),
            (row(None, draft=None), EMPTY),
            (row("RIGHT", draft="RIGHT"), {**REFERENCE, "question_id": "q4"}))]
        summary = revisions.summarize_arm(records, citations_checked=False)
        self.assertEqual(summary["transitions"], {"right_to_right": 1, "right_to_wrong": 1, "wrong_to_right": 1,
                                                  "wrong_to_wrong": 0, "net": 0})
        self.assertEqual(summary["transitions_revised"]["wrong_to_right"], 1)
        self.assertEqual((summary["runs_with_draft"], summary["runs_revised"]), (3, 1))
        self.assertEqual((summary["unsupported"]["count"], summary["unsupported"]["n"]), (1, 2))
        self.assertIsNone(summary["unsupported"]["invalid_citation"])
        self.assertEqual(summary["empty_reference"], {"n": 1, "count": 1, "rate": 1.0})
        self.assertEqual(summary["false_abstention"]["count"], 1)

        omitted = [revisions.question_record("o", row("RIGHT", fetched={"Observation": ["o9"]}), REFERENCE, SPEC),
                   revisions.question_record("o", row("WRONG"), {**REFERENCE, "question_id": "q3"}, SPEC)]
        intact = [revisions.question_record("i", row("RIGHT"), REFERENCE),
                  revisions.question_record("i", row("RIGHT"), {**REFERENCE, "question_id": "q3"})]
        recovery = revisions.summarize_recovery(omitted, intact)
        self.assertEqual(recovery["retrieved"]["rate"], 0.5)
        self.assertEqual(recovery["correct_when_retrieved"]["rate"], 1.0)
        self.assertEqual(recovery["correct_given_intact_correct"], {"n": 2, "count": 1, "rate": 0.5})
        self.assertIsNone(revisions.summarize_recovery(omitted)["correct_given_intact_correct"])


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cli = ["--model", "test-model", "--output_dir", self.temp.name, "--limit", "4"]
        self.frame = pd.DataFrame([{ "question_id": f"q{index}", "question": "Q", "true_answer": "[[1]]",
                                     "assumption": "", "patient_fhir_id": "synthetic-patient"}
                                   for index in range(1, 5)])
        self.references = {qid: {**REFERENCE, "question_id": qid, "patient": "synthetic-patient"}
                           for qid in self.frame["question_id"]}
        self.specs = {qid: SPEC for qid in self.frame["question_id"]}
        for patcher in (
                patch.object(experiments, "arm_inputs", side_effect=lambda *_: self.frame.copy()),
                patch.object(experiments, "load_references", return_value=self.references),
                patch.object(experiments, "load_question_sets", return_value={"recovery": list(self.references)}),
                patch("tools.perturbation.load_perturbations", return_value=self.specs)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_arm(self, arm, make_row):
        args = experiments.build_parser().parse_args(["status", *self.cli])
        settings = runner.build_parser().parse_args(experiments.runner_argv(arm, args))
        frame = experiments.arm_inputs(arm, settings, experiments.load_question_sets([arm], args))
        identity = runner.run_identity(settings, frame)
        results = runner.prepare_results(frame, settings.output, identity)
        for index, qid in zip(results.index, results["question_id"]):
            for key, value in {**make_row(qid), "latency": 1.0}.items():
                results.at[index, key] = value
        with contextlib.redirect_stdout(io.StringIO()):
            runner.save_results_json(results, settings.output)
        runner._write_json(identity, settings.output + ".run.json")

    def test_revisions_command_logs_flags_without_answers(self):
        from tools.perturbation import load_perturbations
        specs = load_perturbations()
        arms = {arm.name: arm for arm in experiments.ARMS}
        self.write_arm(arms["ordinary_single_agent"],
                       lambda qid: row("SECRET-RIGHT", draft="SECRET-WRONG", revised=True, evidence=[cite()]))
        self.write_arm(arms["recovery_multi_agent_intact"], lambda qid: row("SECRET-RIGHT"))
        self.write_arm(arms["recovery_multi_agent_omitted"], lambda qid: row(
            "SECRET-RIGHT", fetched={specs[qid]["removed"]["resource_type"]: [specs[qid]["removed"]["id"]]}))
        score = lambda answer, *_: {"typed_correct": int(answer == "SECRET-RIGHT"), "answer_type": "category"}
        with patch.object(revisions, "score_answer", side_effect=score), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(experiments.main(["revisions", *self.cli]), 0)
        directory = Path(self.temp.name, "revisions")
        written = "".join(path.read_text() for path in directory.iterdir())
        self.assertNotIn("SECRET", out.getvalue() + written)
        report = json.loads((directory / "revisions.json").read_text())
        single = report["arms"]["ordinary_single_agent"]
        self.assertEqual((single["transitions"]["wrong_to_right"], single["transitions"]["net"]), (4, 4))
        self.assertEqual(single["unsupported"]["count"], 0)
        recovery = report["recovery"]["multi_agent"]
        self.assertEqual((recovery["retrieved"]["rate"], recovery["correct_given_intact_correct"]["rate"]), (1.0, 1.0))
        self.assertIn("ordinary_direct_llm", report["skipped"])
        with open(directory / "revision_log.csv") as stream:
            log = list(csv.DictReader(stream))
        self.assertEqual(len(log), 12)
        self.assertEqual({r["transition"] for r in log if r["arm"] == "ordinary_single_agent"}, {"wrong_to_right"})

    def test_ndjson_evidence_needs_a_directory(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            experiments.main(["revisions", "--model", "m", "--evidence_source", "ndjson"])

    def test_stale_arm_never_contributes_transitions_or_recovery(self):
        arm = next(arm for arm in experiments.ARMS if arm.name == "ordinary_single_agent")
        self.write_arm(arm, lambda _: row("RIGHT", draft="WRONG", revised=True))
        identity_path = Path(self.temp.name, f"{arm.name}.json.run.json")
        identity = json.loads(identity_path.read_text())
        identity["model"] = "different-model"
        identity_path.write_text(json.dumps(identity))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(experiments.main(["revisions", *self.cli]), 0)
        report = json.loads(Path(self.temp.name, "revisions", "revisions.json").read_text())
        self.assertEqual(report["arms"], {})
        self.assertEqual(report["recovery"], {})
        self.assertIn("stale", report["skipped"][arm.name])


class TypedRecordTests(unittest.TestCase):
    def test_zero_is_a_real_answer_and_failures_do_not_get_empty_reference_credit(self):
        count = {**REFERENCE, "question": "How many matching records?", "true_answer": "[[0]]"}
        verified = revisions.question_record("arm", verified_row(0), count)
        self.assertEqual((verified["correct"], verified["reference_empty"], verified["abstained"]), (1, False, False))
        empty = {**count, "true_answer": "[]"}
        failed = revisions.question_record("arm", row(None, error="provider down"), empty)
        self.assertEqual((failed["correct"], failed["reference_empty"], failed["abstained"]), (0, True, False))

    def test_actual_runtime_verification_trace_establishes_empty_result(self):
        import test_phase2_verification_runtime as runtime
        from utils import parse_outputs

        for answer in (0, False, [], None):
            with self.subTest(answer=answer):
                candidate = {"answer": answer, "evidence": [], "contributing_ids": [],
                             "operation": "inspect matching records", "limitation": None}
                code = f"answer = {answer!r}"
                responses = [runtime.fetch("r-fetch"), runtime.VerificationRuntimeTests.report(),
                             runtime.call(runtime.PYTHON_TOOL, "a-python", code=code),
                             runtime.reply(json.dumps(candidate)), runtime.fetch("v-fetch"),
                             runtime.call(runtime.PYTHON_TOOL, "v-python", code=code),
                             runtime.verdict(recomputed=answer)]
                _, output, _, _ = runtime.VerificationRuntimeTests().run_team(responses, records=[])
                record = revisions.question_record("arm", parse_outputs(output), EMPTY)
                self.assertEqual((record["empty_verified"], record["answered"], record["unsupported"]),
                                 (True, True, False))


if __name__ == "__main__":
    unittest.main()
