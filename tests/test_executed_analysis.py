"""The execution artifact, rather than prose copying, determines a team draft."""
import json
import unittest
from unittest.mock import Mock, patch

from agent.analysis_agent import AnalysisAgent
from agent.base_agent import BaseAgent
from agent.framework import RoleResult


class ExecutedAnalysisTests(unittest.TestCase):
    def setUp(self):
        with patch("agent.base_agent.setup_api_keys"):
            self.host = BaseAgent("test-model")
        self.resources = {"Observation": [{"resourceType": "Observation", "id": "o1",
                                           "valueQuantity": {"value": 12.5}}]}
        self.agent = AnalysisAgent(self.host, {"question": "What is the maximum value?",
                                               "assumptions": ""}, self.resources)
        self.computation = {"mode": "final", "answer": 12.5,
                            "evidence": [{"resource_id": "Observation/o1",
                                          "field_path": "valueQuantity.value", "value": 12.5}],
                            "contributing_ids": ["Observation/o1"],
                            "operation": "max(valueQuantity.value)", "limitation": None}

    def test_malformed_prose_cannot_erase_valid_executed_answer(self):
        def execute(_):
            self.agent.toolbox.final_computation = self.computation
            return RoleResult(content="done")
        self.agent.role.run = Mock(side_effect=execute)
        result = self.agent({"records": {}})
        self.assertEqual(result["answer"], 12.5)
        self.assertEqual(result["evidence"], self.computation["evidence"])

    def test_model_cannot_change_execution_result_when_copying(self):
        def execute(_):
            self.agent.toolbox.final_computation = self.computation
            return RoleResult(content=json.dumps({"answer": 99}))
        self.agent.role.run = Mock(side_effect=execute)
        self.assertEqual(self.agent({"records": {}})["answer"], 12.5)

    def test_provider_failure_still_prevents_release(self):
        self.agent.toolbox.final_computation = self.computation
        self.agent.role.run = Mock(return_value=RoleResult(error="provider unavailable"))
        self.assertEqual(self.agent({"records": {}}), {"error": "provider unavailable"})

    def test_unknown_contributors_are_not_accepted_from_execution(self):
        def execute(_):
            self.agent.toolbox.final_computation = {**self.computation,
                                                    "contributing_ids": ["Observation/unknown"]}
            return RoleResult(content="done")
        self.agent.role.run = Mock(side_effect=execute)
        self.assertIn("error", self.agent({"records": {}}))


if __name__ == "__main__":
    unittest.main()
