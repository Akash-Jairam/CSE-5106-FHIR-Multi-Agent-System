"""Independent actual-loop A/control parity, including continued solving and release gates."""
from contextlib import ExitStack
from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch

from litellm import Message
from agent.residual_trajectory_agent import ResidualTrajectoryAgent
from agent.residual_trajectory_phase import message_dict
from agent.single_agent import SingleAgent
from tests.test_continuous_primary import PACKET, PATIENT, QUESTION, RESOURCE, computation, source_check, usage


def action(identifier, name="execute_python_code", **arguments):
    return Message(role="assistant", tool_calls=[{"id": identifier, "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)}}])


def final(identifier="final", value=7):
    return action(identifier, code=f"answer={value}", mode="final")


def finish(value=7):
    return Message(role="assistant", content=json.dumps({"answer": value,
        "evidence": computation()["evidence"], "limitation": None}))


class ActualPrimaryParityTests(unittest.TestCase):
    def compare(self, responses, outputs=(), bills=None):
        def run(team):
            requests = []
            pending = iter(deepcopy(responses))
            reported = iter(deepcopy(bills)) if bills is not None else None
            executor = Mock(side_effect=deepcopy(outputs))
            fetch = Mock(return_value={"Observation": [deepcopy(RESOURCE)]})
            def provider(**kwargs):
                requests.append(deepcopy(kwargs))
                return next(pending), None, next(reported) if reported is not None else usage()
            with ExitStack() as stack:
                stack.enter_context(patch("agent.base_agent.setup_api_keys"))
                stack.enter_context(patch("agent.base_agent.count_tokens_in_messages", return_value=10))
                stack.enter_context(patch("agent.base_agent.safe_llm_call", side_effect=provider))
                stack.enter_context(patch("agent.single_agent.get_starting_packet", side_effect=lambda *_a, **_k: deepcopy(PACKET)))
                stack.enter_context(patch("agent.single_agent.check_fields", side_effect=source_check))
                stack.enter_context(patch("agent.multi_turn_code_resource_agent.get_tool",
                    side_effect=lambda name: executor if name == "execute_python_code" else fetch))
                # A deterministic coverage refusal AFTER A completes; no alteration
                # of the primary constructor, loop, budget, prompt or responses.
                stack.enter_context(patch("agent.residual_trajectory_agent.ENTRY_BYTES", 0))
                agent = ResidualTrajectoryAgent("independent-fixture") if team else SingleAgent("independent-fixture")
                agent.set_budget(max_total_tokens=200000, max_tool_calls=30)
                result = agent.run(QUESTION)
            return result, requests, executor.call_count, fetch.call_count
        control, expected_requests, expected_executions, expected_fetches = run(False)
        team, observed_requests, executions, fetches = run(True)
        self.assertEqual(observed_requests, expected_requests)
        self.assertEqual((executions, fetches), (expected_executions, expected_fetches))
        self.assertIsNone(team["error"], team)
        for key in ("answer", "evidence", "limitation", "error", "tokens"):
            self.assertEqual(team[key], control[key], key)
        # The outer budget describes the aggregate controller. The original
        # primary's local stop is preserved exactly in its actual snapshot.
        for key in ("max_total_tokens", "max_tool_calls", "tool_calls_run"):
            self.assertEqual(team["budget"][key], control["budget"][key])
        self.assertIsNone(team["budget"]["stopped"])
        snapshot = team["revision"]["messages"][0]
        for key in ("answer", "evidence", "limitation", "error", "budget", "revision"):
            self.assertEqual(snapshot["released"][key], control[key], key)
        expected_trace = [message_dict(m) for m in control["trace"]]
        actual_trace = [{k: v for k, v in m.items() if k != "phase"} for m in team["trace"]]
        self.assertEqual(actual_trace, expected_trace)
        self.assertEqual(len(team["revision"]["messages"]), 1)
        return team, control, observed_requests

    def test_post_final_inspect_retrieval_and_new_final_match_standalone_single(self):
        responses = [final(), action("inspect", code="answer={'count':1}", mode="inspect"),
            action("fetch", "get_resources_by_patient_fhir_id", resource_type="Observation", patient_fhir_id=PATIENT),
            final("later-final"), finish(999)]
        team, _, requests = self.compare(responses, [computation(), {"mode": "inspect", "answer": {"count": 1}},
            computation(operation="Natural final after a later retrieval")])
        self.assertEqual(len(requests), 5)
        self.assertEqual(team["answer"], 7)
        self.assertTrue(team["revision"]["messages"][0]["artifact"])

    def test_actual_source_feedback_and_distinct_revision_match_control(self):
        team, _, requests = self.compare([final(value=99), finish(99), final("corrected"), finish()],
            [computation(99), computation()])
        self.assertEqual(len(requests), 4)
        self.assertEqual(team["answer"], 7)
        self.assertTrue(team["revision"]["messages"][0]["released"]["revision"]["revised"])

    def test_computed_correctness_is_not_substituted_for_source_withheld_release(self):
        team, _, _ = self.compare([final(value=99), finish(99), final("still-unsupported", 99), finish(99)],
            [computation(99), computation(99)])
        self.assertIsNone(team["answer"])
        snapshot = team["revision"]["messages"][0]
        self.assertEqual(snapshot["released"]["revision"]["final"]["answer"], 99)
        self.assertIsNone(snapshot["artifact"])

    def test_actual_prose_primary_does_not_acquire_an_invented_program(self):
        team, _, requests = self.compare([finish()])
        self.assertEqual(len(requests), 1)
        self.assertEqual(team["answer"], 7)
        self.assertIsNone(team["revision"]["messages"][0]["artifact"])

    def test_known_primary_token_refusal_keeps_original_nonerror_abstention(self):
        team, _, requests = self.compare([final()], [computation()], [usage(199990, 0)])
        self.assertEqual(len(requests), 1)
        self.assertIsNone(team["answer"])
        self.assertTrue(team["revision"]["messages"][0]["released"]["budget"]["stopped"])


if __name__ == "__main__":
    unittest.main()
