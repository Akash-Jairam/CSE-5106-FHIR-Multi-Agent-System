"""Paired operand bindings survive the executable final-answer boundary."""
import unittest

from tools.python_executor import execute_python_code
from tools.field_check import check_fields


class PairedComputationTests(unittest.TestCase):
    def test_bound_operand_citation_is_preserved_and_checked_against_question(self):
        records = {"Observation": [{"resourceType": "Observation", "id": "o1",
                                   "subject": {"reference": "Patient/p1"},
                                   "effectiveDateTime": "2199-06-01T09:00:00",
                                   "valueQuantity": {"value": 7}}]}
        code = """r = retrieved_resources['Observation'][0]
answer = r['valueQuantity']['value']
evidence = [{'resource_id':'Observation/o1','field_path':'valueQuantity.value','value':answer,'scope_index':1}]
contributing_ids = ['Observation/o1']
operation = 'Selected the value for the second textual operand'
limitation = None
"""
        result = execute_python_code(code, mode="final", global_vars={"retrieved_resources": records})
        self.assertNotIn("error", result)
        self.assertEqual(result["evidence"][0]["scope_index"], 1)
        question = ("What was the difference in weight measured at 2199-06-01 10:00:00 "
                    "compared to weight measured at 2199-06-01 09:00:00?")
        indexed = {"Observation/o1": records["Observation"][0]}
        check = check_fields(result["evidence"], question, "2199-12-31 23:59:00",
                             resources=indexed, patient_fhir_id="p1")
        self.assertEqual(check["issues"], [])
        result["evidence"][0]["scope_index"] = 0
        check = check_fields(result["evidence"], question, "2199-12-31 23:59:00",
                             resources=indexed, patient_fhir_id="p1")
        self.assertTrue(check["issues"])


if __name__ == "__main__":
    unittest.main()
