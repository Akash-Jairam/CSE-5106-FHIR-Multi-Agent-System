"""Real sandbox tests use harmless sentinels, never project credentials."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.python_executor import execute_python_code


class ExecutorTests(unittest.TestCase):
    def test_calculation_and_resource_alias(self):
        records = {"Observation": [{"valueQuantity": {"value": 20}}, {"valueQuantity": {"value": 18}}]}
        result = execute_python_code("answer = min(r['valueQuantity']['value'] for r in retrieved_fhir_resources['Observation'])",
                                     global_vars={"retrieved_resources": records})
        self.assertEqual(result, {"answer": 18})

    def test_system_exit_does_not_kill_parent(self):
        self.assertIn("SystemExit", execute_python_code("raise SystemExit")["error"])

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
        self.assertEqual(result, {"answer": "absent"})

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
