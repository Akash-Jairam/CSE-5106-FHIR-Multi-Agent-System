"""Synthetic reporting admission and first-attempt source selection."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts import run_experiments as experiments


class RevisionReportingTests(unittest.TestCase):
    def test_errors_remain_in_report_and_source_refs_describe_first_attempt(self):
        arm = experiments.ARMS[0]
        citation = lambda rid: {"resource_id": rid, "field_path": "valueQuantity.value", "value": 1}
        rows = {
            "a": {"agent_answer": None, "latency": 1, "error": "provider failed"},
            "b": {"agent_answer": 2, "latency": 1, "error": None, "evidence": [citation("Observation/retry")],
                  "attempt_history": [{"agent_answer": 1, "latency": 1, "error": None,
                                       "evidence": [citation("Observation/first")]}]},
        }
        references = {qid: {"question": "Synthetic count", "true_answer": "[[1]]", "template": "",
                            "family": "lookup_counting"} for qid in rows}
        args = SimpleNamespace(model="test", split="dev", limit=2, evidence_source="ndjson", fhir_dir="unused")
        with patch.object(experiments, "arm_results",
                          return_value=({"state": "in progress", "analyzable": True}, rows)), \
                patch.object(experiments, "load_references", return_value=references), \
                patch("utils.evidence_scorer.fetch_from_ndjson", return_value={}) as fetch:
            report, records = experiments.revision_tracking([arm], args, {})
        self.assertEqual(report["arms"][arm.name]["accuracy"]["n"], 2)
        self.assertTrue(next(record for record in records if record["question_id"] == "a")["error"])
        self.assertEqual(fetch.call_args.args[0], {("Observation", "first")})


if __name__ == "__main__":
    unittest.main()
