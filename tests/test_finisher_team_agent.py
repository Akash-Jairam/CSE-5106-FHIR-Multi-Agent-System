"""V10 team: k-draw vote, residual scheme, finisher dispatch, narrowed adjudication (mocked LLM)."""
from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch

from agent.dual_solver_agent import ADJUDICATION_RESERVE
from agent.finisher_team_agent import (
    FINISHER_RESERVE, MIN_DRAW, PROTOCOL_V10, RELEASE_POLICIES, THIRD_DRAW_MIN, VOTES, FinisherTeamAgent,
    adjudication_pair, draw_allowance, merge_candidates, third_draw_allowed, vote)
from tests.test_convention_verified_agent import (
    NULL_CODE, ONE_CODE, PACKET, QUESTION, STORE, TIE_CODE, fetch_requests, final)
from tests.test_dual_solver_agent import FINAL_JSON, adjudication, label_of
from tests.test_query_contract_agent import FixtureMessage
from tools import cache

LAST_CODE = ONE_CODE.replace("rows[0]", "rows[-1]").replace("first by start", "last by start")


def fake_finisher_tool(store):
    def get_resources_by_patient_fhir_id(resource_type, patient_fhir_id):
        if resource_type == "Medication":
            return {"Medication": [], "retrieval_metadata": {"complete": True}}
        return {resource_type: deepcopy([r for r in store if r["resourceType"] == resource_type])}
    return lambda name: get_resources_by_patient_fhir_id


def no_finisher(context, candidate, fetcher):
    return {"template": None, "finisher": "none", "reason": None, "anchors": [], "result": None, "evidence": [],
            "fetches": [], "detail": {}}


class ControllerRuleTests(unittest.TestCase):
    def test_vote_outcomes(self):
        self.assertEqual(vote(["Heparin"]), ("single_draw", [0]))
        self.assertEqual(vote(["Heparin", "heparin "]), ("agree2", [0, 1]))
        self.assertEqual(vote(["Heparin", "Insulin"]), ("split", [0]))
        self.assertEqual(vote(["Heparin", "Insulin", "insulin"]), ("majority3", [1, 2]))
        self.assertEqual(vote(["Heparin", "Insulin", "Aspirin"]), ("split", [0]))
        self.assertEqual(vote([None, None]), ("agree2", [0, 1]))
        self.assertEqual(set(VOTES), {"agree2", "majority3", "split", "single_draw"})

    def test_draw_allowance_holds_the_reserve(self):
        allowance = draw_allowance(200_000, 30, 50_000, 8)
        self.assertEqual((allowance["tokens"], allowance["tools"]),
                         (150_000 - FINISHER_RESERVE[0], 22 - FINISHER_RESERVE[1]))
        self.assertIsNone(allowance["skip"])
        self.assertIn("tokens", draw_allowance(200_000, 30, 200_000 - FINISHER_RESERVE[0] - MIN_DRAW[0] + 1, 8)["skip"])
        self.assertIn("tools", draw_allowance(200_000, 30, 0, 30 - FINISHER_RESERVE[1] - MIN_DRAW[1] + 1)["skip"])
        self.assertIsNone(draw_allowance(None, None, 10**6, 100)["skip"])
        self.assertTrue(third_draw_allowed(200_000, 30, 200_000 - THIRD_DRAW_MIN[0], 30 - THIRD_DRAW_MIN[1]))
        self.assertFalse(third_draw_allowed(200_000, 30, 200_000 - THIRD_DRAW_MIN[0] + 1, 0))
        self.assertFalse(third_draw_allowed(200_000, 30, 0, 30 - THIRD_DRAW_MIN[1] + 1))

    def test_merge_and_adjudication_pair(self):
        a = {"answer": "x", "evidence": [{"resource_id": "A/1", "field_path": "f", "value": 1}],
             "contributing_ids": ["A/1"], "limitation": None}
        b = {"answer": "X", "evidence": [{"resource_id": "A/1", "field_path": "f", "value": 1},
                                         {"resource_id": "B/2", "field_path": "g", "value": 2}],
             "contributing_ids": ["B/2"], "limitation": "y"}
        merged = merge_candidates([a, b])
        self.assertEqual(merged["answer"], "x")
        self.assertEqual(len(merged["evidence"]), 2)
        self.assertEqual(merged["contributing_ids"], ["A/1", "B/2"])
        self.assertEqual(adjudication_pair(["x", "y", "z"]), (0, 1))
        self.assertEqual(adjudication_pair(["x", None, "z"]), (0, 2))
        self.assertIsNone(adjudication_pair([None, "y", "z"]))


class FinisherTeamAgentTests(unittest.TestCase):
    def setUp(self):
        self.store = deepcopy(STORE)
        self.requests = []
        patches = [patch("agent.base_agent.setup_api_keys"),
                   patch.object(cache, "CACHE_ENABLED", False),
                   patch("agent.single_agent.get_starting_packet", return_value=deepcopy(PACKET)),
                   patch("tools.resource_tools.get_fhir_client", return_value=Mock(
                       search_with_pagination=Mock(side_effect=lambda *_a, **_kw: deepcopy(self.store)))),
                   patch("tools.field_check.fetch_from_store", side_effect=lambda refs: {
                       record["id"]: deepcopy(record) for record in self.store
                       if any(identifier == record["id"] for _, identifier in refs)}),
                   patch("agent.framework.get_tool", fake_finisher_tool(self.store))]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_agent(self, responses, max_total_tokens=200_000, max_tool_calls=30):
        pending = iter(responses)

        def llm(**kwargs):
            self.requests.append(deepcopy(kwargs))
            item = FixtureMessage(**vars(next(pending)))
            return item, None, {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "llm_calls": 1,
                                "cost": 0, "per_call": [{"prompt": 100, "completion": 50, "usage_available": True}]}

        with patch("agent.base_agent.safe_llm_call", side_effect=llm):
            agent = FinisherTeamAgent("base-model")
            agent.set_budget(max_total_tokens=max_total_tokens, max_tool_calls=max_tool_calls)
            return agent, agent.run(QUESTION)

    @staticmethod
    def draw(code, tag):
        return [fetch_requests(f"{tag}-fetch"), final(code, f"{tag}-final"), FINAL_JSON]

    def test_agreeing_draws_and_a_tie_finisher_replace_a_bare_member(self):
        agent, output = self.run_agent(self.draw(ONE_CODE, "d1") + self.draw(ONE_CODE, "d2"))
        revision = output["revision"]
        self.assertIsNone(output["error"], output)
        self.assertEqual(revision["protocol"], PROTOCOL_V10)
        self.assertEqual(revision["vote"], "agree2")
        self.assertEqual(revision["vote_group"], [1, 2])
        self.assertEqual(revision["finisher"]["template"], "T051")
        self.assertEqual(revision["finisher"]["finisher"], "replaced")
        self.assertEqual(revision["release_policy"], "finisher_replaced")
        self.assertIn(revision["release_policy"], RELEASE_POLICIES)
        self.assertEqual(sorted(output["answer"]), ["Heparin", "Insulin"])
        self.assertEqual(revision["draft"]["answer"], "Heparin")
        self.assertTrue(revision["revised"] and revision["answer_changed"])
        self.assertEqual([p["phase"] for p in revision["phases"]], ["draw1", "draw2", "finisher"])
        self.assertEqual(agent.total_usage["llm_calls"], 6)
        # three tools per draw (fetch, final, automatic check) plus two finisher fetches
        self.assertEqual(output["budget"]["tool_calls_run"], 8)
        self.assertIn("finisher_fetch_1", json.dumps(output["trace"], default=str))

    def test_draws_are_independent_trajectories(self):
        self.run_agent(self.draw(ONE_CODE, "d1") + self.draw(ONE_CODE, "d2"))
        second = json.dumps(self.requests[3]["messages"], default=lambda o: vars(o))
        self.assertEqual(len(self.requests[3]["messages"]), 2)
        self.assertNotIn("d1-final", second)
        self.assertEqual(self.requests[0]["messages"][:2], self.requests[3]["messages"][:2])

    def test_finisher_that_recomputes_the_same_answer_keeps_the_candidate(self):
        _, output = self.run_agent(self.draw(TIE_CODE, "d1") + self.draw(TIE_CODE, "d2"))
        revision = output["revision"]
        self.assertEqual(output["answer"], ["Heparin", "Insulin"])
        self.assertEqual(revision["release_policy"], "vote_candidate")
        self.assertEqual((revision["finisher"]["finisher"], revision["finisher"]["reason"]),
                         ("confirmed_primary", "result_equals_primary"))
        self.assertFalse(revision["answer_changed"])

    def test_disagreement_runs_a_third_draw_and_takes_the_majority(self):
        _, output = self.run_agent(self.draw(TIE_CODE, "d1") + self.draw(ONE_CODE, "d2")
                                   + self.draw(TIE_CODE, "d3"))
        revision = output["revision"]
        self.assertEqual(revision["vote"], "majority3")
        self.assertEqual(revision["vote_group"], [1, 3])
        self.assertEqual(output["answer"], ["Heparin", "Insulin"])
        draw3 = revision["phases"][2]
        self.assertEqual(draw3["phase"], "draw3")
        self.assertEqual(draw3["allowance"]["tools"],
                         30 - 6 - FINISHER_RESERVE[1] - ADJUDICATION_RESERVE[1])

    def test_split_without_a_deciding_finisher_goes_to_adjudication(self):
        responses = (self.draw(ONE_CODE, "d1") + self.draw(TIE_CODE, "d2") + self.draw(LAST_CODE, "d3")
                     + [adjudication(label_of("recompute"))])
        with patch("agent.finisher_team_agent.run_finisher", side_effect=no_finisher):
            _, output = self.run_agent(responses)
        revision = output["revision"]
        self.assertEqual(revision["vote"], "split")
        self.assertEqual(revision["release_policy"], "adjudicated_other")
        self.assertEqual(revision["adjudication"]["other_draw"], 2)
        self.assertEqual(output["answer"], ["Heparin", "Insulin"])
        self.assertEqual(revision["phases"][-1]["phase"], "adjudicate")

    def test_split_with_a_deciding_finisher_skips_adjudication(self):
        responses = self.draw(ONE_CODE, "d1") + self.draw(TIE_CODE, "d2") + self.draw(LAST_CODE, "d3")
        agent, output = self.run_agent(responses)
        revision = output["revision"]
        self.assertEqual(revision["vote"], "split")
        self.assertEqual(revision["release_policy"], "finisher_replaced")
        self.assertIsNone(revision["adjudication"])
        self.assertEqual(agent.total_usage["llm_calls"], 9)

    def test_small_residual_skips_draw_two(self):
        agent, output = self.run_agent(self.draw(ONE_CODE, "d1"), max_total_tokens=30_000)
        revision = output["revision"]
        self.assertEqual(revision["vote"], "single_draw")
        self.assertIn("tokens", revision["budget_plan"]["draw2"]["skip"])
        self.assertEqual(agent.total_usage["llm_calls"], 3)
        self.assertEqual(revision["release_policy"], "finisher_replaced")

    def test_no_third_draw_below_the_threshold(self):
        _, output = self.run_agent(self.draw(TIE_CODE, "d1") + self.draw(ONE_CODE, "d2"), max_total_tokens=40_000)
        revision = output["revision"]
        self.assertEqual(revision["vote"], "split")
        self.assertIn("residual", revision["budget_plan"]["draw3"]["skip"])
        self.assertEqual(output["answer"], ["Heparin", "Insulin"])  # draw 1, confirmed by the finisher
        self.assertLessEqual(output["tokens"]["total"], 40_000)

    def test_draw_two_stays_inside_its_capped_allowance(self):
        _, output = self.run_agent(self.draw(ONE_CODE, "d1") + self.draw(ONE_CODE, "d2"), max_tool_calls=12)
        draw2 = output["revision"]["phases"][1]
        self.assertEqual(draw2["allowance"]["tools"], 12 - 3 - FINISHER_RESERVE[1])
        self.assertLessEqual(draw2["tool_calls"], draw2["allowance"]["tools"])
        self.assertLessEqual(output["budget"]["tool_calls_run"], 12)

    def test_null_draws_release_null_without_a_finisher_answer(self):
        _, output = self.run_agent(self.draw(NULL_CODE, "d1") + self.draw(NULL_CODE, "d2"))
        revision = output["revision"]
        self.assertIsNone(output["answer"])
        self.assertEqual(revision["vote"], "agree2")
        self.assertEqual(revision["finisher"]["finisher"], "skipped_no_anchor")
        self.assertEqual(revision["release_policy"], "vote_candidate")


if __name__ == "__main__":
    unittest.main()
