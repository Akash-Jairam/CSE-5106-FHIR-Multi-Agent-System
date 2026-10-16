"""Regression tests for aggregate ceilings and accounting, without API calls."""
import unittest
import litellm
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

    def test_completion_is_capped_by_model_output_limit(self):
        self.agent.model = "gpt-6-luna"
        self.agent.set_budget(max_total_tokens=200_000)
        with patch("agent.base_agent.count_tokens_in_messages", return_value=100), patch(
                "agent.base_agent.safe_llm_call", return_value=("answer", None, None)) as call:
            self.agent._llm_call([{"role": "user", "content": "q"}])
        self.assertEqual(call.call_args.kwargs["max_completion_tokens"], 128_000)

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

    def test_real_sdk_messages_include_serialized_tool_arguments(self):
        message = litellm.Message(role="assistant", content=None, tool_calls=[{
            "id": "search1", "type": "function", "function": {
                "name": "search", "arguments": '{"query": "' + "patient " * 100 + '"}'}}])
        self.assertEqual(core.count_tokens_in_messages([message]),
                         core.count_tokens_in_messages([message.model_dump(exclude_none=True)]))
        self.assertGreater(core.count_tokens_in_messages([message]),
                           core.count_tokens_in_messages([{"role": "assistant", "content": None}]))

    def test_second_budgeted_model_turn_accepts_real_sdk_message(self):
        self.agent.set_budget(max_total_tokens=2000)
        messages = [{"role": "user", "content": "Question"}]
        usage = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        reply = litellm.Message(role="assistant", content="Inspect the evidence")
        with patch("agent.base_agent.safe_llm_call", return_value=(reply, None, usage)) as call:
            message, error, _ = self.agent._llm_call(messages)
            self.assertIsNone(error)
            messages.append(message)
            _, error, _ = self.agent._llm_call(messages)
        self.assertIsNone(error)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(self.agent.total_usage["total_tokens"], 60)

    def test_api_receives_output_limit(self):
        with patch.object(core.litellm, "completion", return_value=response()) as call:
            _, error, usage = core.safe_llm_call("test", [], max_completion_tokens=37)
        self.assertIsNone(error)
        self.assertEqual(call.call_args.kwargs["max_tokens"], 37)
        self.assertEqual(call.call_args.kwargs["num_retries"], 0)
        self.assertEqual(usage["total_tokens"], 30)

    def test_missing_provider_pricing_is_unknown_not_free(self):
        result = response()
        result._hidden_params = {}
        with patch.object(core.litellm, "completion", return_value=result):
            _, error, usage = core.safe_llm_call("test", [], max_retries=1)
        self.assertIsNone(error)
        self.assertIsNone(usage["cost"])

    def test_missing_usage_refuses_answer_under_aggregate_ceiling(self):
        result = response()
        result.usage = None
        self.agent.set_budget(max_total_tokens=200)
        with patch.object(core.litellm, "completion", return_value=result), patch(
                "agent.base_agent.count_tokens_in_messages", return_value=1):
            message, error, usage = self.agent._llm_call([])
        self.assertIsNone(message)
        self.assertIn("cannot be verified", error)
        self.assertFalse(usage["per_call"][0]["usage_available"])

    def test_failed_request_keeps_attempt_and_unknown_usage(self):
        with patch.object(core.litellm, "completion", side_effect=RuntimeError("unavailable")):
            _, error, usage = core.safe_llm_call("test", [], max_retries=1)
        self.assertIn("unavailable", error)
        self.assertEqual(usage["llm_calls"], 1)
        self.assertIsNone(usage["cost"])
        self.assertFalse(usage["per_call"][0]["usage_available"])

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

    def test_legacy_agent_api_error_is_a_failed_run(self):
        def legacy(_):
            _, error, _ = self.agent._llm_call([])
            return {"final_answer": f"Error: {error}"}
        self.agent._run = legacy
        with patch("agent.base_agent.safe_llm_call", return_value=(None, "API unavailable", None)):
            result = self.agent.run("q")
        self.assertIsNone(result["answer"])
        self.assertEqual(result["error"], "API unavailable")

    def test_iteration_failure_is_not_an_empty_answer(self):
        self.agent._run = lambda _: {"final_answer": "No final answer reached within iteration limit."}
        result = self.agent.run("q")
        self.assertIsNone(result["answer"])
        self.assertIn("iteration limit", result["error"])

    def test_startup_failure_does_not_retain_previous_patient_trace(self):
        self.agent.messages = [{"role": "user", "content": "previous patient"}]
        def fail(_):
            raise ValueError("packet unavailable")
        self.agent._run = fail
        self.assertEqual(self.agent.run("new patient")["trace"], [])

    def test_runtime_rejects_malformed_answer_and_evidence(self):
        for result in ({"answer": {"value": 7}}, {"answer": 7, "evidence": 42}):
            with self.subTest(result=result):
                self.agent._run = lambda _: result
                output = self.agent.run("q")
                self.assertIsNone(output["answer"])
                self.assertEqual(output["evidence"], [])
                self.assertIn("Invalid output", output["error"])

    def test_non_object_agent_result_keeps_usage(self):
        def invalid(_):
            self.agent._update_usage({"total_tokens": 13})
            return ["invalid"]
        self.agent._run = invalid
        output = self.agent.run("q")
        self.assertIsNone(output["answer"])
        self.assertEqual(output["tokens"]["total"], 13)
        self.assertIn("must be an object", output["error"])

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
