"""An unrepairable verification failure must not consume a repair on another issue."""
import unittest
from unittest.mock import patch

from agent.controller import FixedController


class UnrepairableRoutingTests(unittest.TestCase):
    def setUp(self):
        with patch("agent.base_agent.setup_api_keys"):
            self.controller = FixedController("test-model")
        self.controller._workspace = {}

    def test_source_check_blocks_a_mixed_analysis_repair(self):
        role, decisions = self.controller.route([
            {"type": "source_check", "responsible_role": None},
            {"type": "wrong_calculation", "responsible_role": "analysis"}], {"answer": 7})
        self.assertIsNone(role)
        self.assertEqual([d["role"] for d in decisions], [None, "analysis"])

    def test_unroutable_issue_blocks_a_mixed_retrieval_repair(self):
        role, _ = self.controller.route([
            {"type": "unknown_issue", "responsible_role": None},
            {"type": "missing_evidence", "responsible_role": "retrieval"}], {"answer": None})
        self.assertIsNone(role)


if __name__ == "__main__":
    unittest.main()
