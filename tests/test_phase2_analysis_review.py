"""Regression tests for the analysis role's mandatory audit provenance."""
import json
import unittest

from agent.analysis_agent import AnalysisAgent


def draft(**overrides):
    value = {"answer": 2, "evidence": [], "operation": "count qualifying Observations",
             "contributing_ids": ["Observation/o1", "Observation/o2"], "limitation": None}
    value.update(overrides)
    return json.dumps(value)


class AnalysisProvenanceReviewTests(unittest.TestCase):
    def test_aggregate_requires_an_operation_and_explicit_contributors(self):
        for overrides in ({"operation": None}, {"operation": ""}, {"operation": 42},
                          {"contributing_ids": None}):
            with self.subTest(overrides=overrides):
                self.assertIn("error", AnalysisAgent._draft(draft(**overrides)))

    def test_nonfinite_citation_values_are_not_structured_evidence(self):
        for value in (float("nan"), float("inf"), {"nested": float("nan")}):
            with self.subTest(value=value):
                self.assertIn("error", AnalysisAgent._draft(draft(evidence=[
                    {"resource_id": "Observation/o1", "field_path": "valueQuantity.value", "value": value}])))

    def test_falsey_evidence_values_are_not_coerced_to_an_empty_list(self):
        for evidence in (False, 0, "", {}):
            with self.subTest(evidence=evidence):
                self.assertIn("error", AnalysisAgent._draft(draft(evidence=evidence)))

    def test_typed_ids_must_identify_supported_resources(self):
        for rid in ("Observation/", "/o1", "Observation/o1/extra", "Invented/o1", "Observation/has space"):
            with self.subTest(rid=rid):
                self.assertIn("error", AnalysisAgent._draft(draft(contributing_ids=[rid])))

    def test_citation_ids_must_be_typed_and_part_of_contributors(self):
        for rid in ("o1", "Observation/", "Observation/missing"):
            with self.subTest(rid=rid):
                self.assertIn("error", AnalysisAgent._draft(draft(evidence=[
                    {"resource_id": rid, "field_path": "valueQuantity.value", "value": 2}])))

    def test_contributors_must_have_been_retrieved(self):
        self.assertIn("not among the retrieved", AnalysisAgent._draft(
            draft(), {"Observation": [{"id": "o1"}]})["error"])

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
