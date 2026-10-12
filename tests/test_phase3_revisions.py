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

    def test_omitted_resource_retrieved_and_cited(self):
        record = revisions.question_record("arm", row("RIGHT", evidence=[cite("o9")],
                                                      fetched={"Observation": ["o1", "o9"]}), REFERENCE, SPEC)
        self.assertEqual((record["omitted_resource"], record["omitted_retrieved"], record["omitted_cited"]),
                         ("Observation/o9", True, True))
        missed = revisions.question_record("arm", row("WRONG", evidence=[cite("Encounter/o9")]), REFERENCE, SPEC)
        self.assertEqual((missed["omitted_retrieved"], missed["omitted_cited"]), (False, False))

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


if __name__ == "__main__":
    unittest.main()
