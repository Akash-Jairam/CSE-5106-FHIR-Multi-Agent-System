"""Accepted-only reports are offline and do not charge or launch model requests."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts import v3_full_analysis as analysis


class V3FullAnalysisTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = self.directory / "source.csv"
        self.families = self.directory / "families.csv"
        self.write_csv(self.source, [
            {"question_id": "q0", "patient_fhir_id": "p0", "question": "How many?", "true_answer": "[[2]]", "true_fhir_ids": "{'Observation': ['o0']}"},
            {"question_id": "q1", "patient_fhir_id": "p1", "question": "What value?", "true_answer": "[]", "true_fhir_ids": "{}"},
            {"question_id": "q2", "patient_fhir_id": "p1", "question": "How many?", "true_answer": "[[3]]", "true_fhir_ids": "{}"}])
        self.write_csv(self.families, [{"question_id": f"q{i}", "family": "counting", "template": "Count", "template_id": "T0"} for i in range(3)])
        self.identities = {arm: {"model": "fixture", "agent_strategy": arm, "code_hash": "frozen"} for arm in analysis.ARMS}
        self.write_json("run_manifest.json", {"input": {"path": str(self.source), "sha256": analysis.sha256(self.source)},
                                              "families": {"path": str(self.families)}, "run_identities": self.identities})
        self.ledger = {"version": 1, "protocol": "v3-first-runtime-clean-terminal", "selection": "first_runtime_clean_terminal",
                       "complete": True, "arms": {}}
        self.write_json("split.json", {"splits": {"dev": {"patients": [{"patient_fhir_id": "p0"}], "question_ids": ["q0"]},
                                                  "held_out": {"patients": [{"patient_fhir_id": "p1"}], "question_ids": ["q1", "q2"]}}})
        self.pricing = {"label": "Official OpenAI schedule; estimated", "per_million_tokens": {"input": 0.10, "cached_input": 0.01, "output": 0.50}}
        self.write_json("pricing.json", self.pricing)
        self.rows = [{"question_id": f"q{i}", "patient_fhir_id": f"p{int(i > 0)}", "agent_answer": answer,
                      "latency": i + 1, "error": None, "evidence": [], "limitation": None,
                      "tokens": {"prompt": 100, "completion": 10, "total": 110, "cached_prompt": 20,
                                 "repeated_prompt": 0, "llm_calls": 1, "cost": None,
                                 "per_call": [{"prompt": 100, "completion": 10, "cached": 20, "usage_available": True}]},
                      "budget": {"tool_calls_run": 0, "stopped": None}, "tool_calls": []}
                     for i, answer in enumerate((2, None, 9))]
        for arm in analysis.ARMS:
            self.write_json(f"full_{arm}.json", self.rows)
            self.write_json(f"full_{arm}.json.run.json", self.identities[arm])
            raw = self.directory / "attempts" / arm / "round-0001.json"
            raw.parent.mkdir(parents=True)
            raw.write_text(json.dumps(self.rows))
            self.ledger["arms"][arm] = {row["question_id"]: {"attempt_file": str(raw.relative_to(self.directory)),
                "round": 1, "selection": "runtime_clean_terminal", "runtime_clean": True} for row in self.rows}
        self.write_json("acceptance.json", self.ledger)

    def write_csv(self, path, rows):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def write_json(self, name, value):
        (self.directory / name).write_text(json.dumps(value))

    def analyze(self, **kwargs):
        return analysis.analyze(self.directory, n_boot=30, expected_questions=3, **kwargs)

    def test_null_answer_is_scored_as_abstention_and_correct_for_empty_reference(self):
        report, records = self.analyze()
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["arms"]["multi_agent"]["correct"], 2)
        empty = next(row for row in records if row["arm"] == "multi_agent" and row["question_id"] == "q1")
        self.assertIsNone(empty["final_answer"])
        self.assertTrue(empty["correct"])
        self.assertTrue(empty["abstained"])

    def test_aggregate_totals_means_and_sample_standard_deviation(self):
        report, _ = self.analyze()
        metrics = report["arms"]["direct_llm"]["metrics"]
        self.assertEqual(metrics["latency_seconds"]["total"], 6)
        self.assertEqual(metrics["latency_seconds"]["mean"], 2)
        self.assertEqual(metrics["latency_seconds"]["median"], 2)
        self.assertEqual(metrics["latency_seconds"]["stdev"], 1)
        self.assertAlmostEqual(metrics["latency_seconds"]["p95"], 2.9)
        self.assertEqual(metrics["total_tokens"]["total"], 330)
        self.assertEqual(metrics["total_tokens"]["mean"], 110)
        expected = ((100 - 20) * 0.10 + 20 * 0.01 + 10 * 0.50) / 1_000_000
        self.assertAlmostEqual(metrics["estimated_cost_usd"]["total"], expected * 3)
        self.assertAlmostEqual(metrics["estimated_cost_usd"]["mean"], expected)
        self.assertIsNone(metrics["reported_cost_usd"]["total"])

    def test_retry_attempts_are_not_loaded_or_counted(self):
        raw = self.directory / "attempts" / "multi_agent"
        raw.mkdir(parents=True, exist_ok=True)
        (raw / "failed.json").write_text(json.dumps([{**self.rows[0], "latency": 9999,
                                                       "error": "timeout", "tokens": {"total": 999999}}]))
        report, _ = self.analyze()
        self.assertEqual(report["arms"]["multi_agent"]["metrics"]["latency_seconds"]["total"], 6)
        self.assertEqual(report["arms"]["multi_agent"]["metrics"]["total_tokens"]["total"], 330)
        self.assertEqual(report["arms"]["multi_agent"]["intrinsic_errors"], 0)
        rows = [{**row, "attempt_history": [self.rows[0]]} for row in self.rows]
        self.write_json("full_multi_agent.json", rows)
        with self.assertRaisesRegex(ValueError, "attempt_history"):
            self.analyze()

    def test_missing_cost_or_unknown_provider_usage_is_never_silently_zero(self):
        rows = json.loads(json.dumps(self.rows))
        rows[0]["tokens"]["cached_prompt"] = None
        rows[1]["tokens"]["per_call"][0]["usage_available"] = False
        self.write_json("full_direct_llm.json", rows)
        report, _ = self.analyze()
        cost = report["arms"]["direct_llm"]["metrics"]["estimated_cost_usd"]
        self.assertEqual(cost["missing_n"], 2)
        self.assertIsNone(cost["total"])
        self.assertIsNone(cost["mean"])
        self.assertGreater(cost["observed_only"]["total"], 0)
        self.assertEqual(report["arms"]["direct_llm"]["metrics"]["total_tokens"]["missing_n"], 1)
        self.pricing["per_million_tokens"]["input"] = None
        self.write_json("pricing.json", self.pricing)
        report, _ = self.analyze()
        self.assertEqual(report["arms"]["single_agent"]["metrics"]["estimated_cost_usd"]["missing_n"], 3)

    def test_incomplete_full_report_refuses_and_partial_reports_missing_denominators(self):
        self.write_json("full_multi_agent.json", self.rows[:1])
        with self.assertRaisesRegex(ValueError, "missing"):
            self.analyze()
        report, records = self.analyze(partial=True)
        self.assertEqual(report["status"], "PARTIAL_PROGRESS_DO_NOT_REPORT_AS_FINAL")
        self.assertEqual(report["completeness"]["multi_agent"]["missing"], 2)
        comparison = next(row for row in report["comparisons"] if row["right"] == "multi_agent")
        self.assertEqual(comparison["matched_questions"], 1)
        self.assertEqual(len(records), 7)

    def test_intrinsic_failure_cannot_receive_empty_reference_credit(self):
        rows = [dict(row) for row in self.rows]
        rows[1]["error"] = "No explicit final computation"
        self.write_json("full_multi_agent.json", rows)
        report, records = self.analyze()
        self.assertEqual(report["arms"]["multi_agent"]["correct"], 1)
        row = next(row for row in records if row["arm"] == "multi_agent" and row["question_id"] == "q1")
        self.assertTrue(row["typed_answer_matches"])
        self.assertFalse(row["correct"])

    def test_revision_transitions_preserve_null_drafts_and_targeted_effects(self):
        rows = [dict(row) for row in self.rows]
        rows[0]["revision"] = {"draft": {"answer": None}, "primary": {"answer": 9},
                               "revised": True, "targeted_revised": True, "final_check_issues": []}
        self.write_json("full_multi_agent.json", rows)
        report, _ = self.analyze()
        mechanisms = report["mechanisms"]["multi_agent"]
        self.assertEqual(mechanisms["drafts"], 1)
        self.assertEqual(mechanisms["wrong_to_right"], 1)
        self.assertEqual(mechanisms["targeted_primary_wrong_to_final_right"], 1)
        self.assertEqual(mechanisms["source_checks_clean"], 1)

    def test_local_evidence_validation_and_outputs_are_json_safe(self):
        fhir = self.directory / "fhir"
        fhir.mkdir()
        (fhir / "Observation.ndjson").write_text(json.dumps({"resourceType": "Observation", "id": "o0", "valueQuantity": {"value": 2}}) + "\n")
        rows = [dict(row) for row in self.rows]
        rows[0]["evidence"] = [{"resource_id": "Observation/o0", "field_path": "valueQuantity.value", "value": 2}]
        self.write_json("full_multi_agent.json", rows)
        report, records = self.analyze(fhir_dir=fhir)
        row = next(row for row in records if row["arm"] == "multi_agent" and row["question_id"] == "q0")
        self.assertEqual(row["evidence_valid"], 1)
        self.assertEqual(row["evidence_precision"], 1)
        self.assertEqual(row["evidence_recall"], 1)
        empty = next(row for row in records if row["question_id"] == "q1")
        self.assertIsNone(empty["valid_evidence_fraction"])
        destination = self.directory / "analysis"
        analysis.write_report(report, records, destination)
        saved = json.loads((destination / "summary.json").read_text())
        self.assertEqual(saved["status"], "COMPLETE")
        self.assertEqual(len((destination / "per_question.jsonl").read_text().splitlines()), 9)
        self.assertIn("Accepted runtime-clean", (destination / "report.txt").read_text())
        self.assertTrue((destination / "by_patient_fhir_id.csv").exists())

    def test_mismatched_input_hash_and_duplicate_cohort_are_rejected(self):
        self.write_json("full_direct_llm.json", [*self.rows, self.rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.analyze()
        self.write_json("full_direct_llm.json", self.rows)
        self.source.write_text(self.source.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "source hash"):
            self.analyze()

    def test_final_analysis_requires_frozen_identities_and_complete_clean_selection(self):
        identity_path = self.directory / "full_multi_agent.json.run.json"
        identity_path.unlink()
        with self.assertRaisesRegex(ValueError, "identities"):
            self.analyze()
        self.write_json(identity_path.name, {**self.identities["multi_agent"], "agent_strategy": "foreign"})
        with self.assertRaisesRegex(ValueError, "frozen manifest"):
            self.analyze()
        self.write_json(identity_path.name, self.identities["multi_agent"])
        self.ledger["complete"] = False
        self.write_json("acceptance.json", self.ledger)
        with self.assertRaisesRegex(ValueError, "complete first-runtime-clean"):
            self.analyze()
        self.ledger["complete"] = True
        self.ledger["arms"]["multi_agent"]["q0"]["runtime_clean"] = False
        self.write_json("acceptance.json", self.ledger)
        with self.assertRaisesRegex(ValueError, "runtime-clean selection"):
            self.analyze()

    def test_grand_totals_and_operational_wall_clocks_are_separate(self):
        self.write_json("supervisor_state.json", {"started_at": "2026-10-15T10:00:00+00:00", "finished_at": "2026-10-15T11:00:00+00:00"})
        events = [{"at": "2026-10-15T10:00:00+00:00", "event": "supervisor_started"},
                  {"at": "2026-10-15T10:05:00+00:00", "event": "batch_started", "arm": "multi_agent"},
                  {"at": "2026-10-15T10:25:00+00:00", "event": "batch_sealed", "arm": "multi_agent"}]
        (self.directory / "operations.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")
        report, _ = self.analyze()
        self.assertEqual(report["all_arms_accepted_metrics"]["latency_seconds"]["total"], 18)
        self.assertEqual(report["operational_context"]["wall_clock_seconds"], 3600)
        self.assertEqual(report["operational_context"]["per_arm"]["multi_agent"]["wall_clock_seconds"], 1200)


if __name__ == "__main__":
    unittest.main()
