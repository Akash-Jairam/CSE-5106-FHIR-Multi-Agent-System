"""Real sandbox tests use harmless sentinels, never project credentials."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.python_executor import execute_python_code
from tools.python_worker import DiagnosticOutput, STDOUT_CHAR_LIMIT

FINAL_RESOURCES = {"Observation": [{"resourceType": "Observation", "id": "o1", "valueQuantity": {"value": 20}},
                                    {"resourceType": "Observation", "id": "o2", "valueQuantity": {"value": 18}}]}
FINAL_METADATA = ("\nevidence=[{'resource_id':'Observation/o2','field_path':'valueQuantity.value','value':18}]"
                  "\ncontributing_ids=['Observation/o1','Observation/o2']\noperation='Selected scoped measurements and computed result'"
                  "\nlimitation=None")


class ExecutorTests(unittest.TestCase):
    def test_calculation_and_resource_alias(self):
        records = {"Observation": [{"valueQuantity": {"value": 20}}, {"valueQuantity": {"value": 18}}]}
        result = execute_python_code("answer = min(r['valueQuantity']['value'] for r in retrieved_fhir_resources['Observation'])",
                                     global_vars={"retrieved_resources": records})
        self.assertEqual(result, {"answer": 18, "mode": "auto"})

    def test_system_exit_does_not_kill_parent(self):
        self.assertIn("SystemExit", execute_python_code("raise SystemExit")["error"])

    def test_explicit_null_is_a_result_but_unset_answer_is_an_error(self):
        self.assertEqual(execute_python_code("answer = None"), {"answer": None, "mode": "auto"})
        self.assertIn("no answer variable set", execute_python_code("pass")["error"])

    def test_timeout_terminates_worker(self):
        self.assertIn("exceeded", execute_python_code("while True: pass", timeout=0.1)["error"])

    def test_generated_code_cannot_read_workspace(self):
        with tempfile.NamedTemporaryFile(dir=Path(__file__).resolve().parent, mode="w") as sentinel:
            sentinel.write("test sentinel")
            sentinel.flush()
            result = execute_python_code(f"answer = open({sentinel.name!r}).read()")
        self.assertIn("error", result)
        self.assertNotIn("answer", result)

    def test_generated_code_cannot_read_host_environment(self):
        with patch.dict(os.environ, {"FHIR_TEST_SENTINEL": "test sentinel"}):
            result = execute_python_code("import os; answer = os.environ.get('FHIR_TEST_SENTINEL', 'absent')")
        self.assertEqual(result, {"answer": "absent", "mode": "auto"})

    def test_final_computation_and_source_metadata_come_from_the_same_worker(self):
        code = ("answer = 18\nevidence = [{'resource_id':'Observation/o2','field_path':'valueQuantity.value','value':18}]\n"
                "operation = 'minimum'\ncontributing_ids = ['Observation/o1','Observation/o2']\nlimitation=None")
        result = execute_python_code(code, mode="final", global_vars={"retrieved_resources": FINAL_RESOURCES})
        self.assertEqual(result, {"answer": 18, "mode": "final", "operation": "minimum",
                                 "contributing_ids": ["Observation/o1", "Observation/o2"],
                                 "evidence": [{"resource_id": "Observation/o2", "field_path": "valueQuantity.value", "value": 18}],
                                 "limitation": None})
        self.assertIn("metadata", execute_python_code("answer = 1\nevidence = 5", mode="final")["error"])

    def test_invalid_mode_is_rejected_before_running_code(self):
        with patch("tools.python_executor._sandbox_command") as sandbox:
            result = execute_python_code("answer = 1", mode="unsupported")
        sandbox.assert_not_called()
        self.assertIn("mode", result["error"])

    def test_diagnostics_are_inspectable_but_rejected_as_final_computations(self):
        for value in ({"count": 1}, [{"id": "o1"}], float("nan")):
            code = "answer = " + ("float('nan')" if isinstance(value, float) else repr(value))
            self.assertIn("Final answer", execute_python_code(code, mode="final")["error"])
        diagnostics = {"count": 1, "sample": [{"id": "o1"}]}
        self.assertEqual(execute_python_code(f"answer = {diagnostics!r}", mode="inspect"),
                         {"answer": diagnostics, "mode": "inspect"})
        nested = execute_python_code("answer = [[1, None], 'yes', False]" + FINAL_METADATA, mode="final",
                                     global_vars={"retrieved_resources": FINAL_RESOURCES})
        self.assertEqual(nested["answer"], [[1, None], "yes", False])

    def test_exact_final_citation_contract_is_recoverable(self):
        malformed = ["evidence=[{'resource':'Observation/o2','field':'valueQuantity.value','value':18}]",
                     "evidence=[{'resource_id':'Observation/o2','field_path':'valueQuantity.value'}]",
                     "evidence=[{'resource_id':'Observation/o2','field_path':'valueQuantity.value / unit','value':18}]",
                     "contributing_ids=['o2']", "contributing_ids=['Observation/o2','Observation/o2']",
                     "contributing_ids=['Observation/unknown']", "limitation=5", "operation=''", "evidence=[]"]
        for replacement in malformed:
            result = execute_python_code("answer=18" + FINAL_METADATA + "\n" + replacement, mode="final",
                                         global_vars={"retrieved_resources": FINAL_RESOURCES})
            self.assertEqual(result["error_type"], "final_contract")
            self.assertIn("resource_id", result["error"])
            self.assertNotIn("answer", result)
        missing = execute_python_code("answer=18", mode="final")
        self.assertEqual(missing["error_type"], "final_contract")
        self.assertIn("missing variables", missing["error"])

    def test_completed_empty_calculation_may_have_no_citations(self):
        for value in (None, 0, False, []):
            code = f"answer={value!r}\nevidence=[]\ncontributing_ids=[]\noperation='Completed scoped search; no matches'\nlimitation=None"
            result = execute_python_code(code, mode="final")
            self.assertEqual(result["answer"], value)
            self.assertEqual(result["evidence"], [])

    def test_stdout_supports_inspection_but_cannot_certify_a_final(self):
        for mode in ("inspect", "auto"):
            result = execute_python_code("print('actual diagnostic data')", mode=mode)
            self.assertEqual(result["stdout"], "actual diagnostic data\n")
            self.assertTrue(result["diagnostic"])
            self.assertNotIn("answer", result)
            self.assertFalse(result["stdout_truncated"])
        final = execute_python_code("print(18)", mode="final")
        self.assertEqual(final["error_type"], "final_contract")
        self.assertNotIn("answer", final)
        self.assertEqual(final["stdout"], "18\n")
        structured = execute_python_code("print('checking'); answer=None", mode="inspect")
        self.assertIsNone(structured["answer"])
        self.assertEqual(structured["stdout"], "checking\n")

    def test_stdout_prefix_is_bounded_and_truncation_is_honest(self):
        writer = DiagnosticOutput(limit=16)
        for _ in range(1000):
            writer.write("")
        self.assertEqual(writer.parts, [])
        writer.write("a" * 1000)
        self.assertEqual(writer.kept_chars, 16)
        self.assertEqual(writer.fields()["stdout_chars"], 1000)
        result = execute_python_code("print('x' * 100000)", mode="inspect")
        self.assertEqual(len(result["stdout"]), STDOUT_CHAR_LIMIT)
        self.assertEqual(result["stdout_chars"], 100001)
        self.assertTrue(result["stdout_truncated"])
        final = execute_python_code("print('x' * 100000); answer=18" + FINAL_METADATA, mode="final",
                                    global_vars={"retrieved_resources": FINAL_RESOURCES})
        self.assertEqual(final["answer"], 18)
        self.assertEqual(final["evidence"][0]["value"], 18)
        self.assertTrue(final["stdout_truncated"])

    def test_copied_resources_cannot_mutate_parent(self):
        data = {"Observation": [{"id": "original"}]}
        execute_python_code("retrieved_resources['Observation'][0]['id'] = 'changed'; answer = 1",
                            global_vars={"retrieved_resources": data})
        self.assertEqual(data["Observation"][0]["id"], "original")

    def test_fail_closed_when_sandbox_unavailable(self):
        with patch("tools.python_executor._sandbox_command", return_value=None):
            result = execute_python_code("answer = 1")
        self.assertIn("unavailable", result["error"])

    def test_network_is_denied(self):
        result = execute_python_code("import socket; s = socket.socket(); s.connect(('127.0.0.1', 9)); answer = 1")
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
