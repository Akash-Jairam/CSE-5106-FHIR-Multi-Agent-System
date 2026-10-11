import contextlib
import io
import sys
import unittest
from unittest.mock import Mock, patch

from agent.output_schema import AgentOutput, Tokens, validate_output
from scripts import check_setup


def output():
    value = {key: None for key in AgentOutput.__annotations__}
    value.update({"answer": 7, "evidence": [], "tool_calls": [], "latency": 1.2,
                  "trace": [], "retrieved_fhir_resources": {}, "error": None,
                  "tokens": {key: [] if key == "per_call" else None if key == "cost" else 0
                             for key in Tokens.__annotations__}})
    return value


class SetupCheckTests(unittest.TestCase):
    def run_check(self, patient_count):
        client = Mock()
        client.get_resources_by_resource_ids.return_value = [{"id": check_setup.SMOKE_PATIENT_FHIR_ID}]
        counts = lambda _, resource_type: patient_count if resource_type == "Patient" else 1
        with patch.object(check_setup, "validate_config", return_value=True), \
                patch.object(check_setup, "count_resources", side_effect=counts), \
                patch("fhir_client.get_fhir_client", return_value=client), \
                patch.object(sys, "argv", ["check_setup.py"]), contextlib.redirect_stdout(io.StringIO()) as log:
            try:
                check_setup.main()
                status = 0
            except SystemExit as exc:
                status = exc.code
        return status, log.getvalue(), client

    def test_partial_or_unexpected_patient_import_fails_before_smoke_lookup(self):
        for count in (0, 1, 99, 101):
            with self.subTest(count=count):
                status, log, client = self.run_check(count)
                self.assertEqual(status, 1)
                self.assertIn("expected 100", log)
                self.assertNotIn("Setup looks good", log)
                client.get_resources_by_resource_ids.assert_not_called()

    def test_expected_demo_patient_count_can_pass_readiness_checks(self):
        status, log, client = self.run_check(100)
        self.assertEqual(status, 0)
        self.assertIn("Setup looks good", log)
        client.get_resources_by_resource_ids.assert_called_once()


class OutputSchemaTests(unittest.TestCase):
    def test_non_list_evidence_returns_validation_problem_without_crashing(self):
        for evidence in (42, "not a list", {"resource_id": "Observation/o1"}):
            with self.subTest(evidence=evidence):
                value = output()
                value["evidence"] = evidence
                self.assertIn("evidence must be a list", validate_output(value))

    def test_invalid_answer_shapes_and_non_finite_numbers_are_rejected(self):
        for answer in ({"unexpected": "object"}, [1, {"unexpected": 2}], float("nan"), float("inf")):
            with self.subTest(answer=answer):
                value = output()
                value["answer"] = answer
                self.assertTrue(any(problem.startswith("answer must") for problem in validate_output(value)))

    def test_supported_answers_and_unknown_cost_are_valid(self):
        for answer in (None, False, 0, 7.1, "Heparin", ["a", "b"], [[1], [2]]):
            with self.subTest(answer=answer):
                value = output()
                value["answer"] = answer
                self.assertEqual(validate_output(value), [])

    def test_malformed_citation_fields_are_rejected(self):
        value = output()
        value["evidence"] = [{"resource_id": 1, "field_path": " ", "value": 7}]
        self.assertTrue(any("non-empty strings" in problem for problem in validate_output(value)))

    def test_malformed_output_and_token_objects_return_problems(self):
        self.assertEqual(validate_output([]), ["output must be an object"])
        value = output()
        value["tokens"] = 42
        self.assertTrue(any(problem.startswith("tokens must") for problem in validate_output(value)))


if __name__ == "__main__":
    unittest.main()
