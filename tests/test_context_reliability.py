"""Model previews and deterministic history management preserve authoritative results."""
import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import litellm

from agent.base_agent import BaseAgent
from agent.framework import MODEL_RETRIEVAL_CHARS, Role, Toolbox, _model_tool_reply
from utils.core_utils import compact_model_messages, count_tokens_in_messages


def call(name="execute_python_code", **arguments):
    return SimpleNamespace(id="tool1", function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def response(tool_calls=None, content=None):
    return SimpleNamespace(role="assistant", tool_calls=tool_calls, content=content)


class ComputationTrackingTests(unittest.TestCase):
    def setUp(self):
        self.resources = {"Observation": [{"resourceType": "Observation", "id": "o1", "valueQuantity": {"value": 3}}]}
        self.box = Toolbox(["execute_python_code", "get_resources_by_resource_id"], self.resources)

    def compute(self, mode, answer=3):
        worker = {"answer": answer, "mode": mode, "evidence": [{"resource_id": "Observation/o1",
                  "field_path": "valueQuantity.value", "value": 3}], "operation": "lookup",
                  "contributing_ids": ["Observation/o1"]}
        with patch("agent.framework.get_tool", return_value=Mock(return_value=worker)):
            self.box.call(call(code="answer = 3", mode=mode))
        return worker

    def test_final_metadata_is_copied_from_worker_and_survives_diagnostics(self):
        original = self.compute("final")
        final = deepcopy(self.box.final_computation)
        original["evidence"].clear()
        self.compute("inspect", {"diagnostic": list(range(1000))})
        self.assertEqual(self.box.final_computation, final)
        self.assertEqual(self.box.last_execution["mode"], "inspect")
        self.assertEqual(final["answer"], 3)
        self.assertEqual(final["operation"], "lookup")
        self.assertEqual(final["contributing_ids"], ["Observation/o1"])
        self.assertEqual(final["workspace_sha256"], self.box.workspace_sha256())

    def test_mutation_and_legacy_recalculation_invalidate_prior_final(self):
        self.compute("final")
        self.resources["Observation"][0]["valueQuantity"]["value"] = 4
        self.compute("inspect")
        self.assertIsNone(self.box.final_computation)
        self.compute("final")
        self.compute("auto", {"diagnostic": 1})
        self.assertIsNone(self.box.final_computation)

    def test_failed_fetch_even_malformed_arguments_invalidates_computation(self):
        for fetch in (call("get_resources_by_resource_id", global_vars={}),
                      call("get_resources_by_resource_id", resource_type="Observation", resource_id="o1")):
            self.compute("final")
            with patch("agent.framework.get_tool", return_value=Mock(return_value={"error": "unavailable"})):
                message = self.box.call(fetch)
            self.assertIn("error", json.loads(message["content"]))
            self.assertIsNone(self.box.final_computation)
            self.assertIsNone(self.box.last_execution)

    def test_failed_final_does_not_leave_an_older_success_authoritative(self):
        self.compute("final")
        with patch("agent.framework.get_tool", return_value=Mock(return_value={"error": "bad calculation"})):
            self.box.call(call(code="raise ValueError()", mode="final"))
        self.assertIsNone(self.box.final_computation)
        self.assertIsNone(self.box.last_execution)


class ModelContextTests(unittest.TestCase):
    def test_every_tool_output_is_bounded_and_full_input_unchanged(self):
        for name, payload in (("execute_python_code", {"answer": [{"text": "unicode 測" * 10_000}] * 100}),
                              ("check_fields", {"issues": ["long issue " * 10_000] * 100, "n_valid": 123}),
                              ("other_tool", "huge output " * 100_000)):
            original = {"role": "tool", "tool_call_id": "t", "name": name,
                        "content": json.dumps(payload)}
            copied = deepcopy(original)
            preview = _model_tool_reply(original)
            self.assertEqual(original, copied)
            self.assertLessEqual(len(preview["content"]), MODEL_RETRIEVAL_CHARS)
            self.assertTrue(json.loads(preview["content"])["preview_truncated"])

    def test_compaction_preserves_question_and_complete_recent_sdk_tool_exchange(self):
        task = {"role": "user", "content": "What was the last weight? Patient scope fixture."}
        messages = [{"role": "system", "content": "Use retrieved records."}, task]
        for index in range(30):
            messages.extend([litellm.Message(role="assistant", content=None, tool_calls=[{
                "id": f"t{index}", "type": "function", "function": {"name": "execute_python_code",
                "arguments": json.dumps({"code": "answer = 1\n# " + "history word " * 600})}}]),
                {"role": "tool", "name": "execute_python_code", "tool_call_id": f"t{index}",
                 "content": json.dumps({"answer": index, "mode": "inspect"})}])
        original_count = len(messages)
        compacted, report = compact_model_messages(messages)
        self.assertEqual(len(messages), original_count)
        self.assertIs(compacted[1], task)
        self.assertIsNotNone(report)
        self.assertLess(count_tokens_in_messages(compacted), 32_000)
        self.assertEqual(compacted[-1]["tool_call_id"], "t29")
        known = set()
        for message in compacted:
            item = message if isinstance(message, dict) else message.model_dump(exclude_none=True)
            for tool in item.get("tool_calls") or []:
                known.add(tool["id"])
            if item["role"] == "tool":
                self.assertIn(item["tool_call_id"], known)

    def test_local_role_allocation_preserves_host_budget_for_next_role(self):
        with patch("agent.base_agent.setup_api_keys"):
            host = BaseAgent("fixture")
        host.set_budget(max_tool_calls=30)
        host._llm_call = Mock(side_effect=[(response([call(code="answer=1"), call(code="answer=2")]), None, None),
                                          (response(content="final JSON"), None, None)])
        role = Role("analysis", "Compute", Toolbox(["execute_python_code"]), host, max_tool_calls=1)
        with patch("agent.framework.get_tool", return_value=Mock(return_value={"answer": 1, "mode": "auto"})) as tool:
            result = role.run("question")
        self.assertEqual(result.content, "final JSON")
        self.assertEqual(tool.return_value.call_count, 1)
        self.assertEqual(host.tool_calls_made, 1)
        self.assertIsNone(host.budget_stop)
        self.assertIsNone(host._llm_call.call_args.kwargs["tools"])

    def test_single_agent_model_preview_keeps_full_audit(self):
        from agent.single_agent import SingleAgent
        with patch("agent.base_agent.setup_api_keys"):
            host = SingleAgent("fixture")
        host.messages, host.model_messages = [], []
        raw = {"role": "tool", "tool_call_id": "t", "name": "execute_python_code",
               "content": json.dumps({"answer": list(range(20_000)), "mode": "inspect"})}
        host._append_message(raw)
        self.assertEqual(host.messages[0]["content"], raw["content"])
        self.assertLessEqual(len(host.model_messages[0]["content"]), MODEL_RETRIEVAL_CHARS)
        self.assertEqual(host.messages[0]["model_content"], host.model_messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
