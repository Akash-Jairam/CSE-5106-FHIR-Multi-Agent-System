"""Regression tests for aggregate ceilings and accounting, without API calls."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.base_agent import BaseAgent
from utils import core_utils as core


def response(content="done", prompt=20, completion=10):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                              total_tokens=prompt + completion, prompt_tokens_details=None),
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))],
        _hidden_params={"response_cost": 0.00001})


class BudgetTests(unittest.TestCase):
    def setUp(self):
        with patch("agent.base_agent.setup_api_keys"):
            self.agent = BaseAgent("test")

    def test_completion_is_capped_by_remaining_budget(self):
        self.agent.set_budget(max_total_tokens=200)
        with patch("agent.base_agent.count_tokens_in_messages", return_value=100), patch(
                "agent.base_agent.safe_llm_call", return_value=("answer", None, None)) as call:
            self.agent._llm_call([{"role": "user", "content": "q"}])
        self.assertEqual(call.call_args.kwargs["max_completion_tokens"], 36)
        self.assertEqual(call.call_args.kwargs["max_retries"], 1)

    def test_second_prompt_consumes_remaining_budget(self):
        self.agent.set_budget(max_total_tokens=200)
        self.agent.total_usage["total_tokens"] = 150
        with patch("agent.base_agent.count_tokens_in_messages", return_value=10), patch(
                "agent.base_agent.safe_llm_call") as call:
            _, error, _ = self.agent._llm_call([])
        self.assertIn("Budget exceeded", error)
        call.assert_not_called()

    def test_tool_schema_and_tool_arguments_count(self):
        messages = [{"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "query", "arguments": "patient " * 100}}]}]
        self.assertGreater(core.count_tokens_in_messages(messages), core.count_tokens_in_messages([]))
        self.assertGreater(core.count_tokens_in_messages([], [{"description": "query " * 100}]),
                           core.count_tokens_in_messages([]))

    def test_api_receives_output_limit(self):
        with patch.object(core.litellm, "completion", return_value=response()) as call:
            _, error, usage = core.safe_llm_call("test", [], max_completion_tokens=37)
        self.assertIsNone(error)
        self.assertEqual(call.call_args.kwargs["max_tokens"], 37)
        self.assertEqual(usage["total_tokens"], 30)

    def test_empty_response_retry_keeps_both_calls_and_cost(self):
        with patch.object(core.litellm, "completion", side_effect=[response(None), response()]), patch.object(core.time, "sleep"):
            _, error, usage = core.safe_llm_call("test", [], max_retries=2)
        self.assertIsNone(error)
        self.agent._update_usage(usage)
        self.assertEqual(self.agent.total_usage["total_tokens"], 60)
        self.assertEqual(self.agent.total_usage["llm_calls"], 2)
        self.assertEqual(len(self.agent.call_usage), 2)
        self.assertAlmostEqual(self.agent.total_usage["cost"], 0.00002)

    def test_usage_survives_exception_after_model_call(self):
        def crash(_):
            self.agent._update_usage({"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13})
            raise ValueError("invalid tool result")
        self.agent._run = crash
        result = self.agent.run("q")
        self.assertIsNone(result["answer"])
        self.assertIn("invalid tool", result["error"])
        self.assertEqual(result["tokens"]["total"], 13)

    def test_unknown_cost_remains_unknown(self):
        self.agent._update_usage({"cost": None})
        self.agent._update_usage({"cost": 0.25})
        self.assertIsNone(self.agent.total_usage["cost"])

    def test_rejected_provider_overrun_is_recorded(self):
        self.agent.set_budget(max_total_tokens=100)
        with patch("agent.base_agent.count_tokens_in_messages", return_value=1), patch(
                "agent.base_agent.safe_llm_call", return_value=("answer", None, {"total_tokens": 101})):
            message, error, _ = self.agent._llm_call([])
        self.assertIsNone(message)
        self.assertIn("Budget exceeded", error)
        self.assertEqual(self.agent.total_usage["total_tokens"], 101)

    def test_no_automatic_whole_question_retry(self):
        class Crashed:
            calls = 0
            def run(self, question):
                self.calls += 1
                raise ValueError("failure")
        agent = Crashed()
        self.assertEqual(core.run_agent_safe(agent, "q")["error"], "failure")
        self.assertEqual(agent.calls, 1)

    def test_full_trace_is_preserved_without_mutation(self):
        trace = [{"role": "tool", "content": '{"Observation": [1, 2, 3]}'}]
        saved = core.process_trace(trace, max_items=1)
        self.assertEqual(saved, trace)
        saved[0]["content"] = "different"
        self.assertNotEqual(saved, trace)

    def test_single_resource_id_is_not_split_into_characters(self):
        self.assertEqual(core.return_resource_ids({"Patient": {"id": "abc"}}), {"Patient": ["abc"]})

    def test_budget_validation_and_zero_tool_ceiling(self):
        for invalid in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                self.agent.set_budget(max_total_tokens=invalid)
        self.agent.set_budget(max_tool_calls=0)
        self.assertFalse(self.agent._take_tool_call())

    def test_current_gpt_generation_routes_to_openai(self):
        self.assertEqual(core.get_model_provider("gpt-6-luna"), "openai")


if __name__ == "__main__":
    unittest.main()
