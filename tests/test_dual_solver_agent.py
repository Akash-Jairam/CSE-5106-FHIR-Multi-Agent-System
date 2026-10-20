"""V9 dual-solver team: controller rules, budget split, adjudication (mocked LLM, real sandbox/source checks)."""
from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch

from agent.dual_solver_agent import (
    ADJUDICATION_RESERVE, PROTOCOL_V9, RELEASE_POLICIES, SCOPE_ISSUE_PREFIX, DualSolverAgent, answers_agree,
    budget_plan, candidate_order, decide, parse_adjudication, record_index, source_valid, verify_check)
from agent.query_contract_agent import PHASE_CAPS
from tests.test_convention_verified_agent import (
    BARE, NULL_CODE, ONE_CODE, PACKET, QUESTION, STORE, TIE_CODE, fetch_requests, final)
from tests.test_phase2_verification import reply
from tests.test_query_contract_agent import FixtureMessage
from tools import cache

FINAL_JSON = reply('{"answer": "see executed final", "evidence": [], "limitation": null}')


def adjudication(pick, resource_id="MedicationRequest/rx2", field_path="dispenseRequest.validityPeriod.start",
                 observed="2150-01-02T08:00:00"):
    return reply(json.dumps({"pick": pick, "check": {"resource_id": resource_id, "field_path": field_path,
                                                      "rule": "a first-drug question returns every tied start",
                                                      "observed_value": observed}}))


def label_of(name):
    return "AB"[candidate_order(BARE).index(name)]


class ControllerRuleTests(unittest.TestCase):
    def test_agreement_uses_typed_normalisation(self):
        self.assertTrue(answers_agree("Heparin", ["heparin "]))
        self.assertTrue(answers_agree(3, 3.0))
        self.assertTrue(answers_agree(["b", "a"], ["a", "b"]))
        self.assertTrue(answers_agree("2150-01-02T08:00:30", "2150-01-02 08:00:00", "At what time (minute)?"))
        self.assertFalse(answers_agree(True, 1))
        self.assertFalse(answers_agree(None, 0))
        self.assertTrue(answers_agree(None, None))

    def test_decide_covers_agreement_null_rule_and_adjudication(self):
        self.assertEqual(decide("x", None), "recompute_skipped_budget")
        self.assertEqual(decide("x", {"answer": "X", "source_valid": False}), "agree_primary")
        self.assertEqual(decide(None, {"answer": None, "source_valid": False}), "agree_primary")
        self.assertEqual(decide("x", {"answer": None, "source_valid": False}), "null_rule_primary")
        self.assertEqual(decide(None, {"answer": "y", "source_valid": True}), "null_rule_recompute")
        self.assertEqual(decide(None, {"answer": "y", "source_valid": False}), "null_rule_primary")
        # flags are informational: an invalid recompute still reaches adjudication
        self.assertEqual(decide("x", {"answer": "y", "source_valid": False}), "adjudicate")
        self.assertEqual(decide(0, {"answer": 2, "source_valid": True}), "adjudicate")

    def test_source_validity_ignores_reference_only_and_uncited_scope_notes(self):
        note = SCOPE_ISSUE_PREFIX + "Medication/m1=unlinked_metadata"
        self.assertTrue(source_valid([], {}, {"MedicationRequest/rx1"}))
        self.assertTrue(source_valid([note], {"Medication/m1": "unlinked_metadata"}, {"Medication/m1"}))
        self.assertTrue(source_valid([note], None, {"Medication/m1"}))  # parsed from the issue text
        uncited = SCOPE_ISSUE_PREFIX + "Observation/o9=unknown_time"
        self.assertTrue(source_valid([uncited], {"Observation/o9": "unknown_time"}, {"Observation/o1"}))
        self.assertFalse(source_valid([uncited], {"Observation/o9": "unknown_time"}, {"Observation/o9"}))
        self.assertFalse(source_valid(["evidence[0] value_mismatch"], {}, set()))
        self.assertFalse(source_valid([note, "no evidence cited for a non-null answer"], {}, set()))
        self.assertFalse(source_valid(None))

    def test_budget_plan_holds_the_reserve_and_skips_small_residuals(self):
        plan = budget_plan(200_000, 30, 60_000, 8)
        self.assertEqual(plan["recompute_tokens"], min(PHASE_CAPS["solver"][0], 140_000 - ADJUDICATION_RESERVE[0]))
        self.assertEqual(plan["recompute_tools"], min(PHASE_CAPS["solver"][1], 22 - ADJUDICATION_RESERVE[1]))
        self.assertIsNone(plan["skip"])
        self.assertEqual(budget_plan(200_000, 30, 0, 0)["recompute_tokens"], PHASE_CAPS["solver"][0])
        self.assertIn("residual", budget_plan(200_000, 30, 176_000, 8)["skip"])
        self.assertIsNone(budget_plan(200_000, 30, 175_000, 8)["skip"])
        self.assertIn("tools", budget_plan(200_000, 30, 60_000, 25)["skip"])
        self.assertIsNone(budget_plan(200_000, 30, 60_000, 24)["skip"])
        self.assertIsNone(budget_plan(None, None, 10**6, 100)["skip"])

    def test_adjudication_schema_is_strict(self):
        good = {"pick": "A", "check": {"resource_id": "Specimen/s1", "field_path": "type.coding[0].display",
                                       "rule": "specimen name is Specimen.type", "observed_value": "Blood"}}
        verdict, error = parse_adjudication("Answer:\n" + json.dumps(good))
        self.assertIsNone(error)
        self.assertEqual(verdict["pick"], "A")
        for broken in ({**good, "pick": "primary"}, {**good, "check": None},
                       {**good, "check": {**good["check"], "resource_id": "s1"}},
                       {**good, "check": {**good["check"], "field_path": ""}},
                       {**good, "check": {**good["check"], "rule": " "}},
                       {**good, "check": {k: v for k, v in good["check"].items() if k != "observed_value"}}):
            self.assertIsNone(parse_adjudication(json.dumps(broken))[0], broken)
        self.assertIsNone(parse_adjudication("no json here")[0])
        self.assertIsNone(parse_adjudication(None)[0])

    def test_check_verification_needs_the_held_record_and_exact_value(self):
        index = record_index({"MedicationRequest": deepcopy(STORE)})
        check = {"resource_id": "MedicationRequest/rx2", "field_path": "dispenseRequest.validityPeriod.start",
                 "observed_value": "2150-01-02T08:00:00"}
        self.assertEqual(verify_check(check, index), "valid")
        self.assertEqual(verify_check({**check, "observed_value": "2150-01-05T08:00:00"}, index), "value_mismatch")
        self.assertEqual(verify_check({**check, "field_path": "authoredOn"}, index), "bad_path")
        self.assertEqual(verify_check({**check, "resource_id": "MedicationRequest/rx9"}, index), "missing_resource")

    def test_candidate_order_is_fixed_per_question(self):
        self.assertEqual(candidate_order(BARE), candidate_order(BARE))
        self.assertEqual(set(candidate_order(BARE)), {"primary", "recompute"})
        orders = {candidate_order(f"question {i}") for i in range(40)}
        self.assertEqual(len(orders), 2)


class DualSolverAgentTests(unittest.TestCase):
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
                       if any(identifier == record["id"] for _, identifier in refs)})]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_agent(self, responses, max_total_tokens=200_000, max_tool_calls=30, usage=None):
        pending = iter(responses)
        usage = usage or {}

        def llm(**kwargs):
            index = len(self.requests)
            self.requests.append(deepcopy(kwargs))
            item = FixtureMessage(**vars(next(pending)))
            total = usage.get(index, 150)
            return item, None, {"prompt_tokens": total - 50, "completion_tokens": 50, "total_tokens": total,
                                "llm_calls": 1, "cost": 0,
                                "per_call": [{"prompt": total - 50, "completion": 50, "usage_available": True}]}

        with patch("agent.base_agent.safe_llm_call", side_effect=llm):
            agent = DualSolverAgent("base-model")
            agent.set_budget(max_total_tokens=max_total_tokens, max_tool_calls=max_tool_calls)
            return agent, agent.run(QUESTION)

    def primary(self, code=ONE_CODE):
        return [fetch_requests("p-fetch"), final(code, "p-final"), FINAL_JSON]

    def recompute(self, code=ONE_CODE):
        # LedgerSolver ends at the host boundary after its executed final; no extra model call
        return [fetch_requests("r-fetch"), final(code, "r-final")]

    def test_agreement_releases_the_primary(self):
        agent, output = self.run_agent(self.primary() + self.recompute())
        revision = output["revision"]
        self.assertIsNone(output["error"], output)
        self.assertEqual(output["answer"], "Heparin")
        self.assertEqual(revision["protocol"], PROTOCOL_V9)
        self.assertEqual(revision["release_policy"], "agree_primary")
        self.assertIn(revision["release_policy"], RELEASE_POLICIES)
        self.assertEqual(revision["draft"]["answer"], "Heparin")
        self.assertFalse(revision["revised"] or revision["answer_changed"])
        self.assertTrue(revision["recompute"]["source_valid"])
        self.assertIsNone(revision["adjudication"])
        self.assertEqual([p["phase"] for p in revision["phases"]], ["primary", "recompute"])
        self.assertEqual(agent.total_usage["llm_calls"], 5)
        self.assertLessEqual(output["budget"]["tool_calls_run"], 30)

    def test_recompute_does_not_see_the_primary(self):
        self.run_agent(self.primary() + self.recompute())
        recompute_prompt = json.dumps(self.requests[3]["messages"], default=lambda o: vars(o))
        self.assertNotIn("p-final", recompute_prompt)
        self.assertNotIn("see executed final", recompute_prompt)

    def test_null_rule_releases_a_source_valid_recompute(self):
        _, output = self.run_agent(self.primary(NULL_CODE) + self.recompute())
        revision = output["revision"]
        self.assertEqual(output["answer"], "Heparin")
        self.assertEqual(revision["release_policy"], "null_rule_recompute")
        self.assertIsNone(revision["draft"]["answer"])
        self.assertTrue(revision["revised"] and revision["answer_changed"])
        self.assertEqual(output["evidence"][0]["resource_id"], "MedicationRequest/rx1")

    def test_null_rule_keeps_the_primary_when_only_the_primary_answers(self):
        _, output = self.run_agent(self.primary() + self.recompute(NULL_CODE))
        self.assertEqual(output["answer"], "Heparin")
        self.assertEqual(output["revision"]["release_policy"], "null_rule_primary")

    def test_null_rule_rejects_an_unverified_recompute(self):
        bad = ONE_CODE.replace("'value': answer}]", "'value': 'Warfarin'}]")
        # the recompute's one source revision repeats the bad citation
        responses = self.primary(NULL_CODE) + self.recompute(bad) + [final(bad, "r-final-2")]
        _, output = self.run_agent(responses)
        self.assertIsNone(output["answer"])
        self.assertEqual(output["revision"]["release_policy"], "null_rule_primary")
        self.assertFalse(output["revision"]["recompute"]["source_valid"])

    def test_verified_adjudication_can_release_the_recompute(self):
        responses = self.primary() + self.recompute(TIE_CODE) + [adjudication(label_of("recompute"))]
        _, output = self.run_agent(responses)
        revision = output["revision"]
        self.assertEqual(output["answer"], ["Heparin", "Insulin"])
        self.assertEqual(revision["release_policy"], "adjudicated_recompute")
        self.assertEqual(revision["adjudication"]["check_status"], "valid")
        self.assertEqual(revision["adjudication"]["pick"], "recompute")
        self.assertEqual(revision["phases"][-1]["phase"], "adjudicate")
        self.assertLessEqual(revision["phases"][-1]["tokens"], ADJUDICATION_RESERVE[0])
        prompt = json.dumps(self.requests[-1]["messages"], default=lambda o: vars(o))
        self.assertNotIn("primary", json.loads(self.requests[-1]["messages"][1]["content"])["candidates"])
        self.assertIn("executed_final_program", prompt)

    def test_verified_adjudication_can_keep_the_primary(self):
        responses = self.primary() + self.recompute(TIE_CODE) + [adjudication(
            label_of("primary"), "MedicationRequest/rx1", "medicationCodeableConcept.coding[0].display", "Heparin")]
        _, output = self.run_agent(responses)
        self.assertEqual(output["answer"], "Heparin")
        self.assertEqual(output["revision"]["release_policy"], "adjudicated_primary")

    def test_unverifiable_or_malformed_adjudication_defaults_to_the_primary(self):
        for verdict in (adjudication(label_of("recompute"), observed="2150-02-01T00:00:00"),
                        adjudication(label_of("recompute"), resource_id="MedicationRequest/rx9"),
                        reply('{"pick": "recompute"}'), reply("I think B.")):
            self.requests = []
            _, output = self.run_agent(self.primary() + self.recompute(TIE_CODE) + [verdict])
            self.assertEqual(output["answer"], "Heparin", verdict)
            self.assertEqual(output["revision"]["release_policy"], "adjudication_defaulted")
            self.assertIsNone(output["error"])

    def test_adjudication_over_its_reserve_defaults_to_the_primary(self):
        responses = self.primary() + self.recompute(TIE_CODE) + [adjudication(label_of("recompute"))]
        _, output = self.run_agent(responses, usage={5: ADJUDICATION_RESERVE[0] + 1000})
        revision = output["revision"]
        self.assertEqual(output["answer"], "Heparin")
        self.assertEqual(revision["release_policy"], "adjudication_defaulted")
        self.assertTrue(revision["adjudication"]["phase"]["fatal"])

    def test_small_residual_skips_the_recompute(self):
        agent, output = self.run_agent(self.primary(), max_total_tokens=20_000)
        revision = output["revision"]
        self.assertEqual(output["answer"], "Heparin")
        self.assertEqual(revision["release_policy"], "recompute_skipped_budget")
        self.assertIsNone(revision["recompute"])
        self.assertIn("residual", revision["budget_plan"]["skip"])
        self.assertEqual(agent.total_usage["llm_calls"], 3)

    def test_recompute_stays_inside_the_residual_tool_allowance(self):
        _, output = self.run_agent(self.primary() + self.recompute(), max_tool_calls=10)
        plan = output["revision"]["budget_plan"]
        self.assertEqual(plan["recompute_tools"], 10 - 3 - ADJUDICATION_RESERVE[1])
        self.assertLessEqual(output["revision"]["phases"][1]["tool_calls"], plan["recompute_tools"])
        self.assertLessEqual(output["budget"]["tool_calls_run"], 10)


if __name__ == "__main__":
    unittest.main()
