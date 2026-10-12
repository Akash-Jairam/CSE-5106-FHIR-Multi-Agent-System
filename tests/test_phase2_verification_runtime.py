"""Verifier integration on its original PR stack, without later routing changes.

Use the actual BaseAgent budgets, roles, controller, and Python sandbox with
mocked model/source boundaries. Dictionary-compatible scripted messages keep
SDK serialization coverage in the separate framework correction, rather than
making this verifier PR depend on that fix to run its own regressions.
"""
import json
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from agent.analysis_agent import AnalysisAgent
from agent.controller import FixedController
from agent.framework import PYTHON_TOOL
from agent.retrieval_agent import RetrievalAgent
from agent.verification_agent import VerificationAgent
from tools import cache

from test_phase2_verification import CONTEXT, MIN_CODE, QUESTION, STORE, call, draft, fetch, reply, verdict


class ScriptedMessage(dict):
    """A finite scripted model message supporting the host's mapping interface."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def as_message(response):
    calls = None if not response.tool_calls else [ScriptedMessage({
        "id": item.id, "type": "function", "function": ScriptedMessage({
            "name": item.function.name, "arguments": item.function.arguments})}) for item in response.tool_calls]
    return ScriptedMessage(role=response.role, content=response.content, tool_calls=calls)


class Team(FixedController):
    def build_roles(self, context, workspace):
        return {"retrieval": RetrievalAgent(self, context, workspace),
                "analysis": AnalysisAgent(self, context, workspace),
                "verification": VerificationAgent(self, context)}


class VerificationRuntimeTests(unittest.TestCase):
    def run_team(self, responses, records=None, question=QUESTION, max_total_tokens=64000, max_tool_calls=24):
        records = STORE if records is None else records
        source = {r["id"]: deepcopy(r) for r in records}
        scripts, requests = iter(responses), []

        def provider(**kwargs):
            requests.append(deepcopy(kwargs))
            usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10,
                     "cost": None, "llm_calls": 1,
                     "per_call": [{"prompt": 7, "completion": 3, "cached": 0, "usage_available": True}]}
            return as_message(next(scripts)), None, usage

        packet = {"text": "Starting packet", "resources": {}, "resource_ids": {}}
        client = Mock(search_with_pagination=lambda _: deepcopy(list(source.values())))
        with patch("agent.base_agent.setup_api_keys"), patch.object(cache, "CACHE_ENABLED", False), \
                patch("agent.retrieval_agent.get_starting_packet", return_value=packet), \
                patch("tools.resource_tools.get_fhir_client", return_value=client), \
                patch("tools.field_check.fetch_from_store", side_effect=lambda refs: {
                    rid: deepcopy(source[rid]) for _, rid in refs if rid in source}), \
                patch("agent.base_agent.safe_llm_call", side_effect=provider):
            team = Team("test-model")
            team.set_budget(max_total_tokens=max_total_tokens, max_tool_calls=max_tool_calls)
            output = team.run(question)
        return team, output, requests, source

    @staticmethod
    def report():
        return reply(json.dumps({"resource_types": ["Observation"], "encounter_ids": [], "limitation": None}))

    def clean_responses(self):
        return [fetch("r-fetch"), self.report(), call(PYTHON_TOOL, "a-python", code=MIN_CODE),
                reply(json.dumps(draft())), fetch("v-fetch"), call(PYTHON_TOOL, "v-python", code=MIN_CODE), verdict()]

    def test_targeted_repair_is_rechecked_in_a_fresh_context_under_shared_budgets(self):
        code = "# ANALYSIS_PRIVATE_MARKER\nanswer = max(r['valueQuantity']['value'] for r in retrieved_resources['Observation'])"
        wrong = draft(answer=70, evidence=[{"resource_id": "Observation/o1", "field_path": "valueQuantity.value", "value": 70}])
        responses = [fetch("r-fetch"), self.report(), call(PYTHON_TOOL, "a-wrong", code=code), reply(json.dumps(wrong)),
                     fetch("v-fetch"), call(PYTHON_TOOL, "v-python", code=MIN_CODE), verdict(),
                     call(PYTHON_TOOL, "a-repair", code=MIN_CODE), reply(json.dumps(draft())),
                     fetch("v-refetch"), call(PYTHON_TOOL, "v-recompute", code=MIN_CODE), verdict()]
        _, output, requests, _ = self.run_team(responses)
        self.assertIsNone(output["error"])
        self.assertEqual(output["answer"], 68)
        self.assertEqual(output["revision"]["repair_role"], "analysis")
        self.assertTrue(output["revision"]["answer_changed"])
        self.assertEqual(output["revision"]["final_check_issues"], [])
        self.assertEqual(output["budget"]["tool_calls_run"], 9)
        self.assertEqual(output["tokens"]["total"], len(requests) * 10)
        self.assertEqual(output["tokens"]["llm_calls"], len(requests))
        for index, request in enumerate(requests):
            self.assertEqual(request["max_retries"], 1)
            self.assertLessEqual(request["max_completion_tokens"], 64000 - index * 10)
        verifier = [r for r in requests if "You are the verification agent" in r["messages"][0]["content"]]
        self.assertEqual(sum(len(r["messages"]) == 2 for r in verifier), 2)
        self.assertNotIn("ANALYSIS_PRIVATE_MARKER", json.dumps(verifier))

    def test_original_fields_are_rechecked_after_analysis_mutates_its_python_copy(self):
        mutation = "r = retrieved_resources['Observation'][1]\nr['valueQuantity']['value'] = 99\nanswer = r['valueQuantity']['value']"
        responses = [fetch("r-fetch"), self.report(), call(PYTHON_TOOL, "a-mutation", code=mutation),
                     reply(json.dumps(draft(answer=99, evidence=[{"resource_id": "Observation/o2",
                                                               "field_path": "valueQuantity.value", "value": 99}]))),
                     fetch("v-fetch"), call(PYTHON_TOOL, "v-python", code=MIN_CODE), verdict(),
                     call(PYTHON_TOOL, "a-repair", code=MIN_CODE), reply(json.dumps(draft())),
                     fetch("v-refetch"), call(PYTHON_TOOL, "v-recompute", code=MIN_CODE), verdict()]
        team, output, _, source = self.run_team(responses)
        self.assertEqual(output["answer"], 68)
        self.assertEqual({i["type"] for i in output["revision"]["check_issues"]},
                         {"value_mismatch", "recompute_mismatch"})
        self.assertEqual(source["o2"]["valueQuantity"]["value"], 68)
        self.assertEqual(team._roles["analysis"].workspace["Observation"][1]["valueQuantity"]["value"], 68)
        self.assertIsNot(team._roles["verification"].workspace, team._roles["analysis"].workspace)

    def test_empty_answers_require_successful_independent_search_and_calculation(self):
        cases = [(0, "How many weight readings were recorded?", "answer = len(retrieved_resources['Observation'])"),
                 (False, "Were any weight readings recorded?", "answer = bool(retrieved_resources['Observation'])"),
                 ([], "Which weight values were recorded?", "answer = [r['id'] for r in retrieved_resources['Observation']]"),
                 (None, "What was the latest weight?", "answer = next(iter(retrieved_resources['Observation']), None)")]
        for answer, text, code in cases:
            with self.subTest(answer=answer):
                empty = {"answer": answer, "evidence": [], "contributing_ids": [], "operation": "inspect matching records", "limitation": None}
                responses = [fetch("r-fetch"), self.report(), call(PYTHON_TOOL, "a-python", code=code), reply(json.dumps(empty)),
                             fetch("v-fetch"), call(PYTHON_TOOL, "v-python", code=code), verdict(recomputed=answer)]
                prompt = f"Question: {text}\nContext:\n{CONTEXT['assumptions']}"
                _, output, _, _ = self.run_team(responses, records=[], question=prompt)
                self.assertIsNone(output["error"])
                self.assertEqual(output["answer"], answer)
                self.assertEqual(output["revision"]["final_check_issues"], [])

    def test_shared_tool_ceiling_does_not_skip_the_automatic_source_check(self):
        _, output, requests, _ = self.run_team(self.clean_responses(), max_tool_calls=2)
        self.assertIsNone(output["answer"])
        self.assertIsNone(output["error"])
        self.assertEqual(output["budget"]["tool_calls_run"], 2)
        self.assertEqual(output["revision"]["final_check_issues"][0]["type"], "source_check")
        self.assertEqual(len(requests), 4)
        self.assertIsNone(requests[-1]["tools"])

    def test_a_budget_refused_later_fetch_cannot_reuse_a_successful_computation(self):
        responses = self.clean_responses()
        responses[-2].tool_calls.append(fetch("v-extra-fetch").tool_calls[0])
        _, output, requests, _ = self.run_team(responses, max_tool_calls=5)
        self.assertIsNone(output["answer"])
        self.assertEqual(output["budget"]["tool_calls_run"], 5)
        self.assertIn("tool-call budget", output["budget"]["stopped"])
        self.assertEqual(output["revision"]["final_check_issues"][0]["type"], "source_check")
        self.assertIsNone(requests[-1]["tools"])

    def test_zero_token_ceiling_prevents_all_model_requests(self):
        _, output, requests, _ = self.run_team([], max_total_tokens=0)
        self.assertEqual(requests, [])
        self.assertIsNone(output["answer"])
        self.assertIsNone(output["error"])
        self.assertEqual(output["tokens"]["total"], 0)


if __name__ == "__main__":
    unittest.main()
