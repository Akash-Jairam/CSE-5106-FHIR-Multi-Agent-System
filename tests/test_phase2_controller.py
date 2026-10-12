"""P2-12: the fixed controller shares question context, routes structured
messages, and allows at most one repair."""
import unittest
from unittest.mock import patch

from agent.base_agent import BUDGET_EXCEEDED
from agent.controller import FixedController
from agent.output_schema import validate_output

PATIENT = "11111111-1111-1111-1111-111111111111"
QUESTION = (f"Question: What was the last weight of patient 10001?\nContext:\nPatient FHIR ID is {PATIENT}.\n"
            "Assume current time is 2153-12-31 23:59:00.")
CITATION = {"resource_id": "Observation/o1", "field_path": "valueQuantity.value", "value": 7}


def draft(answer=7, evidence=(CITATION,), limitation=None):
    return {"answer": answer, "evidence": list(evidence), "limitation": limitation}


def issue(role, kind="wrong_field"):
    return {"type": kind, "record": "Observation/o1", "field": "issued", "responsible_role": role,
            "action": "cite effectiveDateTime"}


class StubController(FixedController):
    """Roles replay scripted results and record the tasks they were given."""

    def __init__(self, retrieval=(), analysis=(), verification=()):
        with patch("agent.base_agent.setup_api_keys"):
            super().__init__("test-model")
        self.scripts = {"retrieval": list(retrieval), "analysis": list(analysis),
                        "verification": list(verification)}
        self.tasks = {role: [] for role in self.scripts}

    def build_roles(self, context, workspace):
        workspace["Observation"] = [{"id": "o1"}]

        def handler(role):
            def handle(task):
                self.tasks[role].append(task)
                result = self.scripts[role].pop(0)
                if isinstance(result, Exception):
                    raise result
                return result
            return handle
        return {role: handler(role) for role in self.scripts}


def records(*ids):
    return {"record_ids": list(ids or ["Observation/o1"])}


class ControllerTests(unittest.TestCase):
    def test_clean_draft_is_released_and_every_role_gets_the_shared_context(self):
        system = StubController([records()], [draft()], [{"issues": []}])
        output = system.run(QUESTION)
        self.assertEqual(validate_output(output), [])
        self.assertIsNone(output["error"])
        self.assertEqual((output["answer"], output["evidence"]), (7, [CITATION]))
        context = {"question": "What was the last weight of patient 10001?", "patient_fhir_id": PATIENT,
                   "current_time": "2153-12-31 23:59:00",
                   "assumptions": f"Patient FHIR ID is {PATIENT}.\nAssume current time is 2153-12-31 23:59:00."}
        for role_tasks in system.tasks.values():
            self.assertEqual([task["context"] for task in role_tasks], [context])
        self.assertEqual(system.tasks["analysis"][0]["records"], records())
        self.assertEqual(system.tasks["verification"][0]["draft"], draft())
        log = output["revision"]["messages"]
        self.assertEqual([(m["sender"], m["recipient"], m["kind"]) for m in log], [
            ("controller", "retrieval", "task"), ("retrieval", "controller", "records"),
            ("controller", "analysis", "task"), ("analysis", "controller", "draft"),
            ("controller", "verification", "task"), ("verification", "controller", "verdict"),
            ("controller", "user", "final")])
        self.assertIsNone(output["revision"]["repair_role"])
        self.assertFalse(output["revision"]["revised"])
        self.assertEqual(output["retrieved_fhir_resources"], {"Observation": [{"id": "o1"}]})

    def test_analysis_issue_repairs_analysis_only(self):
        fixed = draft(evidence=[{**CITATION, "field_path": "valueQuantity.value"}])
        system = StubController([records()], [draft(), fixed], [{"issues": [issue("analysis")]}, {"issues": []}])
        output = system.run(QUESTION)
        self.assertEqual(output["answer"], 7)
        self.assertEqual(len(system.tasks["retrieval"]), 1)
        repair = system.tasks["analysis"][1]
        self.assertEqual((repair["issues"], repair["previous_draft"]), ([issue("analysis")], draft()))
        self.assertEqual(output["revision"]["repair_role"], "analysis")
        self.assertTrue(output["revision"]["revised"])
        self.assertEqual(output["revision"]["final_check_issues"], [])

    def test_retrieval_issue_reruns_retrieval_then_analysis(self):
        system = StubController([records(), records("Observation/o1", "Observation/o2")], [draft(), draft(8)],
                                [{"issues": [issue("analysis"), issue("retrieval", "missing_evidence")]},
                                 {"issues": []}])
        output = system.run(QUESTION)
        self.assertEqual(output["answer"], 8)
        self.assertTrue(output["revision"]["answer_changed"])
        self.assertEqual(output["revision"]["repair_role"], "retrieval")
        self.assertEqual(system.tasks["retrieval"][1]["previous_records"], records())
        self.assertEqual(system.tasks["analysis"][1]["records"], records("Observation/o1", "Observation/o2"))

    def test_only_one_repair_and_unresolved_issues_abstain(self):
        system = StubController([records()], [draft(), draft(9), draft(10)],
                                [{"issues": [issue("analysis")]}, {"issues": [issue("analysis")]}, {"issues": []}])
        output = system.run(QUESTION)
        self.assertIsNone(output["answer"])
        self.assertIsNone(output["error"])
        self.assertIn("Evidence could not be verified: wrong_field Observation/o1 issued", output["limitation"])
        self.assertEqual((len(system.tasks["analysis"]), len(system.tasks["verification"])), (2, 2))
        self.assertEqual(output["revision"]["final"]["answer"], 9)

    def test_issue_no_role_can_repair_abstains_without_a_repair(self):
        system = StubController([records()], [draft(limitation="Units unclear")],
                                [{"issues": [issue("verification", "unsupported")]}])
        output = system.run(QUESTION)
        self.assertIsNone(output["answer"])
        self.assertTrue(output["limitation"].startswith("Units unclear. Evidence could not be verified"))
        self.assertIn("No role could repair", output["limitation"])
        self.assertEqual(len(system.tasks["analysis"]), 1)

    def test_any_unrepairable_issue_blocks_a_mixed_repair_request(self):
        for responsible in (None, "verification", "unknown"):
            with self.subTest(responsible=responsible):
                issues = [issue("analysis"), issue(responsible, "source_check")]
                system = StubController([records()], [draft(), draft(8)],
                                        [{"issues": issues}, {"issues": []}])
                output = system.run(QUESTION)
                self.assertIsNone(output["answer"])
                self.assertIsNone(output["error"])
                self.assertIsNone(output["revision"]["repair_role"])
                self.assertIn("No role could repair", output["limitation"])
                self.assertEqual(len(system.tasks["analysis"]), 1)
                self.assertEqual(len(system.tasks["verification"]), 1)

    def test_role_failure_before_a_draft_is_a_retryable_error(self):
        system = StubController([RuntimeError("store unavailable")])
        output = system.run(QUESTION)
        self.assertIsNone(output["answer"])
        self.assertEqual(output["error"], "retrieval: RuntimeError: store unavailable")
        self.assertEqual(output["revision"]["messages"][-2]["kind"], "error")
        self.assertEqual(system.tasks["analysis"], [])

    def test_failed_verification_never_releases_the_draft(self):
        system = StubController([records()], [draft()], [{"error": "provider down"}])
        output = system.run(QUESTION)
        self.assertIsNone(output["answer"])
        self.assertEqual(output["error"], "verification: provider down")
        self.assertEqual(output["revision"]["revision_error"], "verification: provider down")

    def test_budget_stop_during_repair_is_an_abstention_not_an_error(self):
        system = StubController([records()], [draft(), {"error": f"{BUDGET_EXCEEDED}: token budget"}],
                                [{"issues": [issue("analysis")]}])
        output = system.run(QUESTION)
        self.assertIsNone(output["answer"])
        self.assertIsNone(output["error"])
        self.assertIn("Evidence could not be verified", output["limitation"])
        self.assertIn(BUDGET_EXCEEDED, output["limitation"])

    def test_results_that_break_the_message_contract_stop_the_run(self):
        for analysis, verification, problem in [
                ([{"evidence": []}], [], "a draft must have an answer"),
                ([draft()], [{"issues": "none"}], "issues must be a list"),
                ([draft(answer=float("nan"))], [], "finite JSON")]:
            output = StubController([records()], analysis, verification).run(QUESTION)
            self.assertIsNone(output["answer"])
            self.assertIn(problem, output["error"])

    def test_roles_cannot_alter_logged_messages(self):
        class Mutating(StubController):
            def build_roles(self, context, workspace):
                roles = super().build_roles(context, workspace)
                analysis = roles["analysis"]

                def mutate(task):
                    task["records"]["record_ids"].append("Observation/forged")
                    return analysis(task)
                return {**roles, "analysis": mutate}

        output = Mutating([records()], [draft()], [{"issues": []}]).run(QUESTION)
        logged = output["revision"]["messages"][2]["payload"]["records"]
        self.assertEqual(logged, records())

    def test_missing_role_is_an_error(self):
        class Incomplete(StubController):
            def build_roles(self, context, workspace):
                return {"retrieval": lambda task: {}}

        output = Incomplete().run(QUESTION)
        self.assertIn("needs handlers for: analysis, verification", output["error"])

    def test_every_role_and_repair_execute_as_nodes_in_the_real_graph(self):
        system = StubController([records(), records()], [draft(), draft(8)],
                                [{"issues": [issue("retrieval", "missing_evidence")]}, {"issues": []}])
        output = system.run(QUESTION)
        self.assertEqual(output["answer"], 8)
        self.assertEqual(set(system._workflow.graph.nodes),
                         {"__start__", "retrieval", "analysis", "verification", "repair"})
        self.assertIsNone(system._workflow.graph.checkpointer)
        self.assertEqual(system._workflow.graph.retry_policy, ())
        self.assertTrue(output["revision"]["revised"])

    def test_failed_repair_does_not_mark_the_unchanged_draft_as_revised(self):
        system = StubController([records(), RuntimeError("store unavailable")], [draft()],
                                [{"issues": [issue("retrieval", "missing_evidence")]}])
        output = system.run(QUESTION)
        self.assertIsNone(output["answer"])
        self.assertFalse(output["revision"]["revised"])
        self.assertEqual(output["revision"]["final"]["answer"], 7)
        self.assertEqual(len(system.tasks["retrieval"]), 2)
        self.assertEqual(len(system.tasks["analysis"]), 1)

    def test_graph_preserves_the_original_scalar_repair_route_api(self):
        class ScalarRoute(StubController):
            def route(self, issues):
                self.routed_issues = issues
                return "analysis"

        system = ScalarRoute([records()], [draft(), draft(8)],
                             [{"issues": [issue("analysis")]}, {"issues": []}])
        output = system.run(QUESTION)
        self.assertEqual(output["answer"], 8)
        self.assertEqual(system.routed_issues, [issue("analysis")])
        self.assertNotIn("routing", output["revision"])


if __name__ == "__main__":
    unittest.main()
