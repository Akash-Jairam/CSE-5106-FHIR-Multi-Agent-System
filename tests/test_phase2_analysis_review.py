"""Regression tests for the analysis role's mandatory audit provenance."""
import json
import unittest

from agent.analysis_agent import DEFAULT_OPERATION, AnalysisAgent


def draft(**overrides):
    value = {"answer": 2, "evidence": [], "operation": "count qualifying Observations",
             "contributing_ids": ["Observation/o1", "Observation/o2"], "limitation": None}
    value.update(overrides)
    return json.dumps(value)


class AnalysisProvenanceReviewTests(unittest.TestCase):
    """A draft's metadata is repaired or dropped; only a missing or invalid answer is an error."""

    def test_a_missing_operation_is_filled_in_for_an_answer(self):
        for overrides in ({"operation": None}, {"operation": ""}, {"operation": 42}):
            with self.subTest(overrides=overrides):
                self.assertEqual(AnalysisAgent._draft(draft(**overrides))["operation"], DEFAULT_OPERATION)
        self.assertIsNone(AnalysisAgent._draft(draft(answer=None, operation=None))["operation"])

    def test_contributors_are_completed_from_the_citations(self):
        cited = {"resource_id": "Observation/o3", "field_path": "valueQuantity.value", "value": 2}
        result = AnalysisAgent._draft(draft(contributing_ids=None, evidence=[cited]))
        self.assertEqual(result["contributing_ids"], ["Observation/o3"])

    def test_nonfinite_citation_values_are_dropped(self):
        for value in (float("nan"), float("inf"), {"nested": float("nan")}):
            with self.subTest(value=value):
                self.assertEqual(AnalysisAgent._draft(draft(evidence=[
                    {"resource_id": "Observation/o1", "field_path": "valueQuantity.value", "value": value}]))["evidence"], [])

    def test_falsey_evidence_values_become_no_citations(self):
        for evidence in (False, 0, "", {}):
            with self.subTest(evidence=evidence):
                self.assertEqual(AnalysisAgent._draft(draft(evidence=evidence))["evidence"], [])

    def test_untyped_or_unsupported_ids_are_dropped(self):
        for rid in ("Observation/", "/o1", "Observation/o1/extra", "Invented/o1", "Observation/has space"):
            with self.subTest(rid=rid):
                self.assertEqual(AnalysisAgent._draft(draft(contributing_ids=[rid]))["contributing_ids"], [])

    def test_contributors_that_were_not_retrieved_are_dropped(self):
        result = AnalysisAgent._draft(draft(), {"Observation": [{"id": "o1"}]})
        self.assertEqual((result["answer"], result["contributing_ids"]), (2, ["Observation/o1"]))

    def test_an_invalid_answer_is_still_an_error(self):
        self.assertIn("error", AnalysisAgent._draft(draft(answer={"value": 2})))
        self.assertIn("error", AnalysisAgent._draft("no JSON here"))

    def test_known_lookup_needs_no_extra_python_call(self):
        result = AnalysisAgent._draft(draft(answer="High", operation="read Observation/o1 interpretation",
                                            contributing_ids=["Observation/o1"]),
                                      {"Observation": [{"id": "o1"}]})
        self.assertEqual(result["answer"], "High")

    def test_zero_count_with_explicit_empty_contributors_is_valid(self):
        result = AnalysisAgent._draft(draft(answer=0, contributing_ids=[]), {})
        self.assertEqual(result["answer"], 0)


if __name__ == "__main__":
    unittest.main()
