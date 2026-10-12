"""Bound model retrieval context while preserving complete records and audit replies."""
import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from openai.types.chat import ChatCompletionMessage

from agent.base_agent import BaseAgent
from agent.framework import (MODEL_RETRIEVAL_CHARS, MODEL_RETRIEVAL_ITEMS, PYTHON_TOOL,
                             Role, Toolbox, _model_tool_reply)
from utils.core_utils import count_tokens_in_messages

PATIENT = "11111111-1111-1111-1111-111111111111"


def tool_call(resource_type):
    return {"id": "fetch", "type": "function", "function": {
        "name": "get_resources_by_patient_fhir_id", "arguments": json.dumps({
            "resource_type": resource_type, "patient_fhir_id": PATIENT})}}


def message(payload, name="get_resources_by_patient_fhir_id"):
    return {"role": "tool", "tool_call_id": "fetch", "name": name, "content": json.dumps(payload)}


def records(resource_type, size):
    return [{"resourceType": resource_type, "id": f"record-{i:06d}-{i * 7919:032x}",
             "subject": {"reference": f"Patient/{PATIENT}"},
             "period": {"start": "2150-01-01T00:00:00"},
             "identifier": [{"system": "http://synthetic/encounter-hosp", "value": str(i)}],
             "valueQuantity": {"value": i}} for i in range(size)]


class RetrievalPreviewTests(unittest.TestCase):
    def run_role(self, resource_type, source):
        with patch("agent.base_agent.setup_api_keys"):
            host = BaseAgent("synthetic-model")
        workspace = {}
        toolbox = Toolbox(["get_resources_by_patient_fhir_id"], workspace, patient_fhir_id=PATIENT)
        responses = [ChatCompletionMessage(role="assistant", content=None, tool_calls=[tool_call(resource_type)]),
                     ChatCompletionMessage(role="assistant", content="Fetched.")]
        prompts = []

        def completion(**kwargs):
            prompts.append(deepcopy(kwargs["messages"]))
            response = responses[len(prompts) - 1]
            return SimpleNamespace(choices=[SimpleNamespace(message=response,
                                   finish_reason="tool_calls" if response.tool_calls else "stop")],
                                   usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                                   _hidden_params={"response_cost": 0})

        with patch("agent.framework.get_tool", return_value=lambda **_: {resource_type: source}), \
                patch("utils.core_utils.litellm.completion", side_effect=completion) as provider:
            result = Role("retrieval", "Fetch the synthetic records.", toolbox, host).run("Synthetic question")
        self.assertIsNone(result.error)
        self.assertEqual(provider.call_count, 2)
        self.assertLess(count_tokens_in_messages(prompts[1], toolbox.definitions), 32_000)
        model_reply = next(m for m in prompts[1] if isinstance(m, dict) and m.get("role") == "tool")
        audit_reply = next(m for m in host.messages if isinstance(m, dict) and m.get("role") == "tool")
        self.assertNotIn("model_content", model_reply)
        if model_reply["content"] != audit_reply["content"]:
            self.assertEqual(audit_reply["model_content"], model_reply["content"])
        else:
            self.assertNotIn("model_content", audit_reply)
        self.assertEqual(workspace[resource_type], source)
        self.assertEqual(json.loads(audit_reply["content"])["ids"],
                         [f"{resource_type}/{r['id']}" for r in source])
        return workspace, json.loads(model_reply["content"]), json.loads(audit_reply["content"]), model_reply

    def test_large_id_reply_fits_next_safe_call_and_keeps_the_full_workspace_and_audit(self):
        source = records("Observation", 10_000)
        workspace, preview, audit, model_reply = self.run_role("Observation", source)
        self.assertGreater(count_tokens_in_messages([message(audit)]), 32_000)
        self.assertEqual(len(preview["ids"]), MODEL_RETRIEVAL_ITEMS)
        self.assertEqual((preview["ids_count"], preview["items"], preview["ids_truncated"]), (10_000, 10_000, True))
        self.assertLessEqual(len(model_reply["content"]), MODEL_RETRIEVAL_CHARS)

        def calculate(code, global_vars):
            return {"answer": len(global_vars["retrieved_resources"]["Observation"])}

        call = SimpleNamespace(id="calculate", function=SimpleNamespace(name=PYTHON_TOOL,
                                                                       arguments=json.dumps({"code": "answer = 1"})))
        with patch("agent.framework.get_tool", return_value=calculate):
            result = Toolbox([PYTHON_TOOL], workspace).call(call)
        self.assertEqual(json.loads(result["content"]), {"answer": 10_000})

    def test_many_encounters_have_an_explicit_bounded_preview_and_complete_audit(self):
        source = records("Encounter", 1_000)
        _, preview, audit, model_reply = self.run_role("Encounter", source)
        self.assertEqual(len(audit["records"]), len(source))
        self.assertEqual(preview["records_count"], len(source))
        self.assertTrue(preview["records_truncated"])
        self.assertLessEqual(len(preview["records"]), MODEL_RETRIEVAL_ITEMS)
        self.assertLessEqual(len(model_reply["content"]), MODEL_RETRIEVAL_CHARS)
        self.assertEqual(preview["records"], audit["records"][:len(preview["records"])])

    def test_one_oversized_encounter_is_omitted_whole_without_losing_its_source(self):
        source = records("Encounter", 1)
        source[0]["identifier"][0]["value"] = "\u6d4b" * 30_000
        _, preview, audit, model_reply = self.run_role("Encounter", source)
        self.assertEqual(audit["records"][0]["identifier"][0]["value"], source[0]["identifier"][0]["value"])
        self.assertEqual((preview["records"], preview["records_count"], preview["records_truncated"]), ([], 1, True))
        self.assertFalse(preview["ids_truncated"])
        self.assertLessEqual(len(model_reply["content"]), MODEL_RETRIEVAL_CHARS)

    def test_small_retrieval_replies_remain_byte_for_byte_unchanged(self):
        for payload in ({"ids": [f"Observation/o{i}" for i in range(MODEL_RETRIEVAL_ITEMS)], "items": 32},
                        {"ids": ["Encounter/e"], "records": [{"id": "e", "period": {"start": "2150-01-01"}}]},
                        {"ids": [], "items": 0, "excluded": {"after_current_time": 10}}):
            original = message(payload)
            with self.subTest(payload=payload):
                self.assertEqual(_model_tool_reply(original), original)
                self.assertIsNot(_model_tool_reply(original), original)
        _, preview, audit, _ = self.run_role("Observation", records("Observation", 2))
        self.assertEqual(preview, audit)

    def test_truncation_count_is_returned_ids_and_preserves_workspace_count_and_exclusions(self):
        original = message({"ids": [f"Observation/o{i}" for i in range(33)], "items": 1000,
                            "excluded": {"after_current_time": 7}, "stored_as": "retrieved_resources['Observation']"})
        preview = json.loads(_model_tool_reply(original)["content"])
        self.assertEqual((preview["ids_count"], preview["items"], preview["ids_truncated"]), (33, 1000, True))
        self.assertEqual(preview["excluded"], {"after_current_time": 7})
        self.assertEqual(preview["stored_as"], "retrieved_resources['Observation']")
        self.assertEqual(len(json.loads(original["content"])["ids"]), 33)

    def test_error_budget_and_non_retrieval_replies_are_unchanged(self):
        replies = [message({"error": "synthetic failure"}), message(["unexpected"]),
                   message({"answer": list(range(100))}, name=PYTHON_TOOL),
                   {"role": "tool", "name": "get_resources_by_patient_fhir_id", "content": "Not run: budget"}]
        for original in replies:
            with self.subTest(original=original):
                self.assertEqual(_model_tool_reply(original), original)


if __name__ == "__main__":
    unittest.main()
