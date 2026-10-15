"""V4 selected-candidate interface/accounting: mock model/store, real sandbox checks."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.plan_conditioned_agent import (
    PlanConditionedAgent, PlanLedgerSolver, V4PhaseHost, PHASE_CAPS, parse_plan,
)
from tests import test_query_contract_agent as query_fixture
from tests.test_query_contract_agent import (
    FixtureMessage, BARE, QUESTION, CODE,
    fetch, final_call, reply,
)


def plan(question=BARE, quote="greater than 70"):
    start = question.index(quote)
    return {"status": "READY", "steps": [{"id": "p1", "operator": "filter", "inputs": [],
             "quote": quote, "span": [start, start + len(quote)],
             "requirement": "strictly greater than 70"}], "unknowns": []}


class PlanInterfaceTests(unittest.TestCase):
    def test_unique_quote_normalizes_span_and_freezes_original(self):
        raw = plan()
        raw["steps"][0]["span"] = [0, 1]
        parsed, error = parse_plan(json.dumps(raw), BARE)
        self.assertIsNone(error)
        step = parsed["steps"][0]
        self.assertEqual(step["span"], [BARE.index("greater than 70"), BARE.index("greater than 70")+15])
        self.assertEqual(step["supplied_span"], [0, 1])
        self.assertEqual(step["span_normalization"], "host_unique_quote")
        self.assertEqual(raw["steps"][0]["span"], [0, 1])

    def test_missing_or_multiple_occurrences_including_overlapping_repetitions_are_ignored(self):
        for text in ["How many weights greater than 70 or heights greater than 70?", "missing", "aaa"]:
            raw = plan()
            if text == "aaa":
                raw["steps"][0]["quote"] = "aa"
            result, error = parse_plan(json.dumps(raw), text)
            self.assertIsNone(result)
            self.assertIn("missing or ambiguous", error)

    def test_distinct_uniquely_anchored_steps_may_overlap(self):
        raw = plan()
        quote = "greater than 70"
        raw["steps"].append({"id":"p2","operator":"project","inputs":["p1"],
            "quote":"70","span":[BARE.index("70"),BARE.index("70")+2],
            "requirement":"Keep the cited threshold unchanged."})
        parsed, error = parse_plan(json.dumps(raw), BARE)
        self.assertIsNone(error)
        self.assertEqual(len(parsed["steps"]),2)
        self.assertLess(parsed["steps"][1]["span"][0],parsed["steps"][0]["span"][1])

    def test_invalid_unknown_or_answer_bearing_plan_is_ignored(self):
        bad = [None, "not json", "null"]
        for key, value in [("status", []), ("status", "UNKNOWN"), ("answer", 5), ("code", "answer=5")]:
            raw = plan()
            raw[key] = value
            bad.append(json.dumps(raw))
        raw = plan()
        raw["steps"][0]["operator"] = []
        bad.append(json.dumps(raw))
        raw = plan()
        raw["steps"][0]["inputs"] = ["p1"]
        bad.append(json.dumps(raw))
        raw = plan()
        raw["unknowns"] = [float("nan")]
        bad.append(json.dumps(raw))
        for text in bad:
            with self.subTest(text=text):
                result, error = parse_plan(text, BARE)
                self.assertIsNone(result)
                self.assertTrue(error)

    def test_deep_json_is_ignored_without_aborting_the_solver(self):
        nested = '{"status":"READY","steps":[],"unknowns":' + '[' * 1200 + '0' + ']' * 1200 + '}'
        result, error = parse_plan(nested, BARE)
        self.assertIsNone(result)
        self.assertTrue(error)
        # Dependency runtimes can replace the JSON decoder. Exercise the
        # explicit recursion failure path as well as the actual nested input.
        with patch("agent.plan_conditioned_agent.json.loads", side_effect=RecursionError("deep JSON")):
            result, error = parse_plan(nested, BARE)
        self.assertIsNone(result)
        self.assertIn("finite JSON", error)


class PlanPipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = query_fixture.QueryContractIntegrationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.requests = []

    def run_agent(self, responses, question=QUESTION, tokens=200000, tools=30):
        pending = iter(responses)
        def llm(**kwargs):
            self.requests.append(deepcopy(kwargs))
            item = next(pending)
            if isinstance(item, tuple):
                return item
            usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
                     "llm_calls": 1, "cost": 0,
                     "per_call": [{"prompt": 100, "completion": 50, "usage_available": True}]}
            return FixtureMessage(**vars(item)), None, usage
        with patch("agent.base_agent.safe_llm_call", side_effect=llm):
            host = PlanConditionedAgent("test-model")
            host.set_budget(max_total_tokens=tokens, max_tool_calls=tools)
            result = host.run(question)
        return host, result

    def test_plan_question_only_visible_first_solver_call_all_tools_no_audit(self):
        host, output = self.run_agent([reply(json.dumps(plan())), fetch(), final_call()])
        initial = json.dumps(self.requests[0]["messages"], default=lambda obj: vars(obj))
        self.assertIn(BARE, initial)
        self.assertNotIn(CODE, initial)
        self.assertNotIn("Observation/o1", initial)
        self.assertIsNone(self.requests[0]["tools"])
        solver_input = self.requests[1]["messages"][0]["content"]
        self.assertIn("Frozen advisory plan JSON", solver_input)
        self.assertIn("strictly greater than 70", solver_input)
        self.assertIn(BARE, self.requests[1]["messages"][1]["content"])
        names = {tool["function"]["name"] for tool in self.requests[1]["tools"]}
        self.assertTrue({"get_resources_by_patient_fhir_id", "get_resources_by_resource_id",
                         "execute_python_code", "check_fields"}.issubset(names))
        self.assertEqual(output["answer"], 2)  # Advice cannot impose a model repair.
        self.assertEqual(host.total_usage["llm_calls"], 3)
        self.assertEqual(host.tool_calls_made, 3)
        self.assertEqual([item["phase"] for item in output["revision"]["phases"]], ["planner", "solver"])
        self.assertEqual([(item["token_cap"], item["tool_cap"]) for item in output["revision"]["phases"]],
                         [(10000, 0), (190000, 30)])
        self.assertEqual(output["revision"]["final_check_issues"], [])
        self.assertFalse(output["revision"]["targeted_revised"])

    def test_format_unknown_and_known_usage_failure_do_not_veto_primary(self):
        nested = '{"status":"READY","steps":[],"unknowns":' + '[' * 1200 + '0' + ']' * 1200 + '}'
        for first in [reply("malformed"), reply(nested), reply(json.dumps({"status":"UNKNOWN","steps":[],"unknowns":[]})),
                      (None, "injected failure with verified usage", {"prompt_tokens":100,"completion_tokens":50,
                       "total_tokens":150,"llm_calls":1,"cost":0,
                       "per_call":[{"prompt":100,"completion":50,"usage_available":True}]})]:
            self.requests.clear()
            _, output = self.run_agent([first, fetch(), final_call()])
            self.assertEqual(output["answer"], 2)
            self.assertIsNone(output["error"])
            self.assertFalse(output["revision"]["plan_visible"])
            self.assertNotIn("Frozen advisory plan JSON", self.requests[1]["messages"][0]["content"])
            self.assertTrue(output["revision"]["plan_error"])

    def test_unverified_planner_usage_stops_before_any_solver_call(self):
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                 "llm_calls": 1, "per_call": [{"usage_available": False}]}
        _, output = self.run_agent([(reply(json.dumps(plan())), None, usage)])
        self.assertIsNone(output["answer"])
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(output["revision"]["release_policy"], "aggregate_accounting_failed")

    def test_provider_error_with_unknown_usage_is_fatal_before_solver(self):
        # Zero accounted tokens here are placeholders, not known zero usage.
        usage = {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,
                 "llm_calls":1,"cost":None,"per_call":[{"usage_available":False}]}
        host, output = self.run_agent([(None,"injected transport timeout",usage)])
        self.assertEqual(len(self.requests),1)
        self.assertIsNone(output["answer"])
        self.assertTrue(output["error"])
        self.assertEqual(output["revision"]["release_policy"],"aggregate_accounting_failed")
        self.assertTrue(output["revision"]["phases"][0]["fatal"])
        self.assertFalse(output["revision"]["plan_visible"])
        self.assertEqual(host.total_usage["llm_calls"],1)

    def test_planner_tools_never_run_and_are_not_exposed_as_advice(self):
        _, output = self.run_agent([fetch("forbidden-plan-tool"), fetch(), final_call()])
        self.assertEqual(output["answer"], 2)
        self.assertFalse(output["revision"]["plan_visible"])
        self.assertIn("requested tools", output["revision"]["plan_error"])
        self.assertEqual(output["budget"]["tool_calls_run"], 3)

    def test_source_feedback_is_only_revision_and_failures_still_block(self):
        bad = CODE + "\nevidence[0]['value'] = 999\n"
        _, fixed = self.run_agent([reply(json.dumps(plan())), fetch(), final_call(bad), final_call()])
        self.assertEqual(fixed["answer"], 2)
        self.assertTrue(fixed["revision"]["solver_source_revised"])
        self.assertEqual(fixed["revision"]["final_check_issues"], [])
        self.requests.clear()
        _, failed = self.run_agent([reply(json.dumps(plan())), fetch(), final_call(bad), final_call(bad)])
        self.assertIsNone(failed["answer"])
        self.assertTrue(failed["revision"]["final_check_issues"])
        self.assertEqual(failed["revision"]["release_policy"], "primary_failed_execution_or_source_contract")

    def test_twentieth_solver_call_final_after_source_rejection_gets_second_check(self):
        from agent.query_contract_agent import HostBoundaryMessage
        bad = CODE + "\nevidence[0]['value'] = 999\n"
        responses = [reply(json.dumps(plan())), fetch(), final_call(bad)]
        responses += [reply("Continue reasoning before the corrected calculation.")] * 17
        responses += [final_call()]
        host, output = self.run_agent(responses)
        self.assertEqual(output["answer"], 2)
        self.assertEqual(len(self.requests), 21)  # one planner plus20 solver calls
        self.assertEqual(host.total_usage["llm_calls"], 21)
        self.assertEqual(host.tool_calls_made, 5)
        self.assertEqual(output["revision"]["final_check_issues"], [])
        self.assertTrue(output["revision"]["solver_source_revised"])
        boundaries = [item for item in output["trace"] if isinstance(item, dict)
                      and item.get("content") == HostBoundaryMessage.content]
        self.assertEqual(len(boundaries), 2)
        checks = [item for item in output["trace"] if isinstance(item, dict)
                  and item.get("role") == "tool" and item.get("name") == "check_fields"]
        self.assertEqual(len(checks), 2)

    def test_extra_host_slot_never_adds_a_twenty_first_solver_invocation(self):
        responses = [reply(json.dumps(plan()))] + [reply("No computation yet.")] * 20
        host, output = self.run_agent(responses)
        self.assertIsNone(output["answer"])
        self.assertEqual(len(self.requests), 21)  # one planner plus20 solver calls
        self.assertEqual(host.total_usage["llm_calls"], 21)
        self.assertEqual(host.tool_calls_made, 0)
        self.assertIn("20 model-call", output["error"])
        self.assertEqual(output["revision"]["release_policy"], "primary_failed_execution_or_source_contract")

    def test_positive_and_empty_results_with_unresolved_source_are_blocked(self):
        from tests.test_query_contract_agent import STORE
        self.fixture.store = [deepcopy(STORE[1]), deepcopy(STORE[0])]
        self.fixture.store[1].pop("effectiveDateTime", None)
        self.fixture.store[1].pop("issued", None)
        for code in [CODE, CODE.replace(">= 70", "> 100")]:
            self.requests.clear()
            _, output = self.run_agent([reply(json.dumps(plan())), fetch(), final_call(code), final_call(code)])
            self.assertIsNone(output["answer"])
            self.assertIn("Observation/o1", output["revision"]["unresolved_scope"])
            self.assertTrue(output["revision"]["final_check_issues"])

    def test_same_batch_retrieval_and_early_prose_do_not_bypass_final(self):
        batch = final_call()
        batch.tool_calls += fetch("later-fetch").tool_calls
        early = reply('{"answer":999,"evidence":[],"limitation":null}')
        _, output = self.run_agent([reply(json.dumps(plan())), early, fetch(), batch, final_call()])
        self.assertEqual(output["answer"], 2)
        self.assertEqual(len(self.requests), 5)
        self.assertTrue(any("No valid final computation" in item.get("content", "")
                            for item in output["trace"] if isinstance(item, dict)))


class PlanBudgetTests(unittest.TestCase):
    def make_host(self):
        with patch("agent.base_agent.setup_api_keys"):
            host = PlanConditionedAgent("test-model")
        host.set_budget(max_total_tokens=200000, max_tool_calls=30)
        return host

    def test_solver_reserves_two_source_calls_from_30_without_changing_v3(self):
        host = self.make_host()
        with patch("agent.base_agent.setup_api_keys"):
            solver = PlanLedgerSolver(host, V4PhaseHost(host, "solver"))
        for _ in range(28):
            self.assertTrue(solver._solver_take_tool_call())
        self.assertFalse(solver._solver_take_tool_call())
        solver._source_check_active = True
        self.assertTrue(solver._solver_take_tool_call())
        self.assertTrue(solver._solver_take_tool_call())
        self.assertFalse(solver._solver_take_tool_call())
        self.assertEqual(host.tool_calls_made, 30)
        from agent.query_contract_agent import PHASE_CAPS as v3_caps
        self.assertEqual(v3_caps["solver"], (175000, 27))

    def test_planner_local_refusal_restores_global_and_solver_reserve(self):
        host = self.make_host()
        planner = V4PhaseHost(host, "planner")
        host.total_usage["total_tokens"] = 9999
        with patch("agent.base_agent.safe_llm_call") as provider:
            _, error, _ = planner._llm_call([{"role":"user","content":"next prompt"}])
        provider.assert_not_called()
        self.assertIn("Budget exceeded", error)
        self.assertEqual(host.max_total_tokens, 200000)
        self.assertIsNone(host.last_llm_error)
        solver = V4PhaseHost(host, "solver")
        with patch("agent.base_agent.safe_llm_call", return_value=(reply("{}"),None,None)) as provider:
            solver._llm_call([{"role":"user","content":"short prompt"}])
        self.assertGreater(provider.call_args.kwargs["max_completion_tokens"], 189000)
        self.assertEqual(host.max_total_tokens, 200000)

    def test_unknown_usage_and_phase_overshoot_remain_fatal(self):
        for usage in [
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
             "per_call": [{"usage_available": False}]},
            {"prompt_tokens": 100, "completion_tokens": 9901, "total_tokens": 10001,
             "per_call": [{"usage_available": True}]}]:
            host = self.make_host()
            phase = V4PhaseHost(host, "planner")
            with patch("agent.base_agent.safe_llm_call", return_value=(reply("{}"),None,usage)):
                phase._llm_call([{"role":"user","content":"short prompt"}])
            self.assertTrue(phase.fatal)
            self.assertEqual(host.max_total_tokens, 200000)


class PlanEntryTests(unittest.TestCase):
    def test_explicit_runner_flags_forward_v4_and_gzip_without_changing_control(self):
        from pathlib import Path
        from scripts.run_experiments import Arm, runner_argv
        from scripts.full_benchmark import arm_command
        import run_agent
        args = SimpleNamespace(team_strategy="multi_agent_v4", model="same-model", input="synthetic.csv",
            output_dir="/private/tmp/v4-output", split_manifest="frozen.json", split="dev",
            num_processes=1, save_interval=1, base_url=None, max_total_tokens=200000,
            max_tool_calls=30, limit=300, verbose=False, audit_storage="gzip")
        argv = runner_argv(Arm("ordinary", "multi_agent"), args)
        parsed = run_agent.build_parser().parse_args(argv)
        self.assertEqual(parsed.agent_strategy, "multi_agent_v4")
        self.assertEqual(parsed.audit_storage, "gzip")
        control = runner_argv(Arm("ordinary", "single_agent"), args)
        self.assertEqual(control[control.index("--agent_strategy") + 1], "single_agent")
        args.input, args.output_dir = Path(args.input), Path(args.output_dir)
        command = arm_command("multi_agent", args)
        self.assertEqual(command[command.index("--agent_strategy") + 1], "multi_agent_v4")
        self.assertEqual(command[command.index("--audit_storage") + 1], "gzip")

    def test_v3_dispatch_and_legacy_job_forwarding_are_unchanged(self):
        import run_agent
        job = ("question", "multi_agent_v3", "same-model", False, None, False, {}, None)
        with patch("agent.query_contract_agent.run_v3_job", return_value="v3") as worker:
            self.assertEqual(run_agent.dispatch_agent_job(job), "v3")
        worker.assert_called_once_with(job)
        job = ("question", "single_agent", "same-model", False, None, False, {}, None)
        with patch("run_agent.run_agent_with_input", return_value="single") as worker:
            self.assertEqual(run_agent.dispatch_agent_job(job), "single")
        worker.assert_called_once_with(job)


if __name__ == "__main__":
    unittest.main()
