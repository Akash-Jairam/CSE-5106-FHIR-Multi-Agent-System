"""P2-15: the verification role re-checks cited fields, recomputes from a
fresh context, and reports structured issues."""
import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.analysis_agent import AnalysisAgent
from agent.base_agent import BaseAgent
from agent.controller import FixedController
from agent.framework import PYTHON_TOOL, RETRIEVAL_TOOLS
from agent.retrieval_agent import RetrievalAgent
from agent.verification_agent import VerificationAgent, _canonical, field_check_issues
from tools import cache

PATIENT = "11111111-1111-1111-1111-111111111111"
NOW = "2153-12-31 23:59:00"
CONTEXT = {"question": "What was the lowest weight of the patient?", "patient_fhir_id": PATIENT,
           "current_time": NOW, "assumptions": f"Patient FHIR ID is {PATIENT}.\nAssume current time is {NOW}."}
QUESTION = f"Question: {CONTEXT['question']}\nContext:\n{CONTEXT['assumptions']}"


def observation(rid, value):
    return {"resourceType": "Observation", "id": rid, "subject": {"reference": f"Patient/{PATIENT}"},
            "effectiveDateTime": "2150-01-02T08:00:00", "issued": "2150-01-02T09:00:00",
            "valueQuantity": {"value": value, "unit": "kg"}}


STORE = [observation("o1", 70), observation("o2", 68)]
MIN_CODE = "answer = min(r['valueQuantity']['value'] for r in retrieved_resources['Observation'])"


def cite(rid="Observation/o2", path="valueQuantity.value", value=68):
    return {"resource_id": rid, "field_path": path, "value": value}


def draft(answer=68, evidence=None):
    return {"answer": answer, "evidence": [cite()] if evidence is None else evidence,
            "operation": "min of valueQuantity.value", "contributing_ids": ["Observation/o1", "Observation/o2"],
            "limitation": None}


def call(name, call_id="call1", **arguments):
    return SimpleNamespace(role="assistant", content=None, tool_calls=[SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))])


def fetch(call_id="fetch1"):
    return call("get_resources_by_patient_fhir_id", call_id, resource_type="Observation", patient_fhir_id=PATIENT)


def reply(content):
    return SimpleNamespace(role="assistant", content=content, tool_calls=None)


def verdict(recomputed=68, issues=()):
    return reply(json.dumps({"recomputed_answer": recomputed, "issues": list(issues)}))


def store_patches():
    by_id = {r["id"]: deepcopy(r) for r in STORE}
    return [patch.object(cache, "CACHE_ENABLED", False),
            patch("tools.resource_tools.get_fhir_client",
                  return_value=Mock(search_with_pagination=Mock(return_value=deepcopy(STORE)))),
            patch("tools.field_check.fetch_from_store", side_effect=lambda refs: {
                i: by_id[i] for _, i in refs if i in by_id})]


class VerificationTests(unittest.TestCase):
    def setUp(self):
        for patcher in [patch("agent.base_agent.setup_api_keys")] + store_patches():
            patcher.start()
            self.addCleanup(patcher.stop)
        self.host = BaseAgent("test-model")
        self.agent = VerificationAgent(self.host, CONTEXT)

    def verify(self, *responses, checked=None):
        self.host._llm_call = Mock(side_effect=[(r, None, None) for r in responses])
        return self.agent({"context": CONTEXT, "draft": checked or draft()})

    def test_tools_are_search_and_sandbox_over_its_own_workspace(self):
        self.assertEqual(self.agent.role.toolbox.names, RETRIEVAL_TOOLS + (PYTHON_TOOL,))
        self.verify(fetch(), verdict())
        self.assertEqual([r["id"] for r in self.agent.workspace["Observation"]], ["o1", "o2"])
        self.verify(verdict())
        self.assertEqual(self.agent.workspace, {})

    def test_clean_draft_is_rechecked_recomputed_and_passes(self):
        result = self.verify(fetch(), call(PYTHON_TOOL, "py1", code=MIN_CODE), verdict())
        self.assertEqual(result, {"issues": [], "recomputed_answer": 68})
        check = self.host.messages[0]["tool_calls"][0]["function"]
        self.assertEqual((check["name"], json.loads(check["arguments"])), ("check_fields", {"citations": [cite()]}))
        self.assertEqual(json.loads(self.host.messages[1]["content"])["n_valid"], 1)
        python_reply = next(m for m in self.host.messages if isinstance(m, dict) and m.get("name") == PYTHON_TOOL)
        self.assertEqual(json.loads(python_reply["content"]), {"answer": 68})
        self.assertEqual(self.host.tool_calls_made, 3)

    def test_prompt_is_a_fresh_context_with_the_draft_and_field_check(self):
        self.verify(verdict(), checked=draft(evidence=[cite(path="issued", value="2150-01-02T09:00:00")]))
        messages = self.host._llm_call.call_args.args[0]
        self.assertEqual([m["role"] for m in messages[:2]], ["system", "user"])
        prompt = messages[1]["content"]
        self.assertIn('"operation": "min of valueQuantity.value"', prompt)
        self.assertIn('"contributing_ids": ["Observation/o1", "Observation/o2"]', prompt)
        self.assertIn('"type": "time_field"', prompt)

    def test_field_check_problems_become_structured_analysis_issues(self):
        result = self.verify(fetch(), call(PYTHON_TOOL, code=MIN_CODE), verdict(), checked=draft(evidence=[
            cite(value=99), cite(path="issued", value="2150-01-02T09:00:00"), cite(rid="Observation/o9")]))
        by_type = {i["type"]: i for i in result["issues"]}
        self.assertEqual(set(by_type), {"value_mismatch", "time_field", "missing_resource"})
        self.assertEqual(by_type["value_mismatch"], {
            "type": "value_mismatch", "record": "Observation/o2", "field": "valueQuantity.value",
            "responsible_role": "analysis",
            "action": "Cite a field that exists in the record with its exact value. The value there is [68]."})
        self.assertIn("effectiveDateTime", by_type["time_field"]["action"])
        self.assertEqual(by_type["missing_resource"]["record"], "Observation/o9")

    def test_answer_without_evidence_is_an_issue(self):
        result = self.verify(fetch(), call(PYTHON_TOOL, code=MIN_CODE), verdict(), checked=draft(evidence=[]))
        self.assertEqual([i["type"] for i in result["issues"]], ["no_evidence"])

    def test_encounter_scope_issue_is_structured(self):
        check = {"issues": ["encounter_scope: the question selects Encounter/e1, not Encounter/e2"], "results": []}
        self.assertEqual(field_check_issues(draft(answer=None, evidence=[]), check), [{
            "type": "encounter_scope", "record": None, "field": None, "responsible_role": "analysis",
            "action": "the question selects Encounter/e1, not Encounter/e2"}])

    def test_model_issues_keep_their_role_and_unknown_roles_are_unroutable(self):
        result = self.verify(fetch(), call(PYTHON_TOOL, code=MIN_CODE), verdict(issues=[
            {"type": "missing_evidence", "record": None, "field": None, "responsible_role": "retrieval",
             "action": "fetch weights from the second visit"},
            {"type": "ambiguous_question", "responsible_role": "verification", "action": "none"}]))
        self.assertEqual([(i["type"], i["responsible_role"]) for i in result["issues"]],
                         [("missing_evidence", "retrieval"), ("ambiguous_question", None)])

    def test_recomputed_answer_that_differs_is_flagged(self):
        result = self.verify(fetch(), call(PYTHON_TOOL, code=MIN_CODE), verdict(), checked=draft(answer=65))
        self.assertEqual(result["issues"][0]["type"], "recompute_mismatch")
        self.assertEqual(result["issues"][0]["responsible_role"], "analysis")
        self.assertIn("recomputed 68", result["issues"][0]["action"])
        self.assertEqual(self.verify(verdict(recomputed=None))["issues"][0]["type"], "independent_check")

    def test_equivalent_answer_forms_are_not_mismatches(self):
        for a, b in [(68, "68.0"), (["Heparin"], "heparin"), ("2150-01-02T08:00:00", "2150-01-02 08:00:00"),
                     (["a", "b"], ["B", "A"]), (True, True)]:
            self.assertEqual(_canonical(a), _canonical(b), (a, b))
        for a, b in [("Tylenol", " ylenol"), (True, 1), (68, 68.5), (["a", "b"], ["a"])]:
            self.assertNotEqual(_canonical(a), _canonical(b), (a, b))

    def test_malformed_verdicts_and_model_errors_are_errors(self):
        self.assertIn("no JSON verdict", self.verify(reply("Looks right to me."))["issues"][-1]["action"])
        self.assertIn("list of objects with a type", self.verify(verdict(issues=["wrong unit"]))["issues"][-1]["action"])
        self.host._llm_call = Mock(return_value=(None, "provider down", None))
        result = self.agent({"context": CONTEXT, "draft": draft()})
        self.assertEqual((result["issues"][-1]["type"], result["issues"][-1]["responsible_role"],
                          result["issues"][-1]["action"]), ("verification_failure", None, "provider down"))

    def test_a_claimed_answer_or_null_without_tools_cannot_verify(self):
        for recomputed in (68, None):
            result = self.verify(verdict(recomputed=recomputed))
            self.assertEqual([(i["type"], i["responsible_role"]) for i in result["issues"]],
                             [("independent_check", None)])

    def test_reported_answer_must_match_the_successful_python_result(self):
        result = self.verify(fetch(), call(PYTHON_TOOL, code=MIN_CODE), verdict(recomputed=70))
        self.assertEqual(result["issues"][0]["type"], "independent_check")

    def test_failed_python_or_fetch_does_not_count_as_recomputation(self):
        for responses in [(fetch(), call(PYTHON_TOOL, code="raise ValueError('broken')"), verdict()),
                          (call("get_resources_by_patient_fhir_id", resource_type="Observation",
                                patient_fhir_id="different-patient"), call(PYTHON_TOOL, code="answer = 68"), verdict())]:
            result = self.verify(*responses)
            self.assertEqual(result["issues"][-1]["type"], "independent_check")

    def test_new_fetch_invalidates_a_calculation_over_previous_records(self):
        result = self.verify(fetch(), call(PYTHON_TOOL, code=MIN_CODE), fetch("later-fetch"), verdict())
        self.assertEqual(result["issues"][-1]["type"], "independent_check")

    def test_null_recomputation_is_a_mismatch_for_a_nonempty_draft(self):
        result = self.verify(fetch(), call(PYTHON_TOOL, code="answer = None"), verdict(recomputed=None))
        self.assertEqual(result["issues"][0]["type"], "recompute_mismatch")

    def test_verified_empty_results_do_not_need_nonexistent_citations(self):
        with patch("tools.resource_tools.get_fhir_client",
                   return_value=Mock(search_with_pagination=Mock(return_value=[]))):
            for answer, code in [(0, "answer = len(retrieved_resources['Observation'])"),
                                 (False, "answer = bool(retrieved_resources['Observation'])"),
                                 ([], "answer = [r['id'] for r in retrieved_resources['Observation']]"),
                                 (None, "answer = next(iter(retrieved_resources['Observation']), None)")]:
                result = self.verify(fetch(), call(PYTHON_TOOL, code=code), verdict(recomputed=answer),
                                     checked=draft(answer=answer, evidence=[]))
                self.assertEqual(result["issues"], [], answer)
                self.assertEqual(result["recomputed_answer"], answer)

    def test_unverified_empty_result_is_not_treated_as_established(self):
        result = self.verify(verdict(recomputed=0), checked=draft(answer=0, evidence=[]))
        self.assertEqual({i["type"] for i in result["issues"]}, {"no_evidence", "independent_check"})

    def test_a_model_duplicate_cannot_reroute_a_deterministic_issue(self):
        duplicate = {"type": "value_mismatch", "record": "Observation/o2", "field": "valueQuantity.value",
                     "responsible_role": "retrieval", "action": "wrong suggestion"}
        result = self.verify(fetch(), call(PYTHON_TOOL, code=MIN_CODE), verdict(issues=[duplicate]),
                             checked=draft(evidence=[cite(value=99)]))
        self.assertEqual(result["issues"][0]["responsible_role"], "analysis")
        self.assertIn("exact value", result["issues"][0]["action"])

    def test_role_failures_keep_deterministic_findings_and_are_unrepairable(self):
        result = self.verify(reply("not JSON"), checked=draft(evidence=[cite(value=99)]))
        self.assertEqual([i["type"] for i in result["issues"]], ["value_mismatch", "verification_failure"])
        self.assertIsNone(result["issues"][-1]["responsible_role"])

    def test_original_source_failure_is_logged_and_cannot_be_approved(self):
        with patch("agent.verification_agent.check_fields", side_effect=RuntimeError("source down")):
            result = self.verify(verdict())
        self.assertEqual(result["issues"][0]["type"], "source_check")
        self.assertIsNone(result["issues"][0]["responsible_role"])
        self.host._llm_call.assert_not_called()
        self.assertIn("error", json.loads(self.host.messages[-1]["content"]))

    def test_recomputation_must_be_present_and_finite(self):
        for text in ['{"issues": []}', '{"recomputed_answer": NaN, "issues": []}',
                     '{"recomputed_answer": {"value": 68}, "issues": []}']:
            result = self.verify(reply(text))
            self.assertEqual(result["issues"][-1]["type"], "verification_failure")

    def test_no_tool_budget_for_the_field_check_is_unrepairable(self):
        self.host.set_budget(max_tool_calls=0)
        self.host._llm_call = Mock()
        result = self.agent({"context": CONTEXT, "draft": draft()})
        self.assertEqual([(i["type"], i["responsible_role"]) for i in result["issues"]], [("source_check", None)])
        self.host._llm_call.assert_not_called()


class PipelineTests(unittest.TestCase):
    def test_verifier_catches_a_wrong_draft_and_the_repair_is_released(self):
        class System(FixedController):
            def build_roles(self, context, workspace):
                return {"retrieval": RetrievalAgent(self, context, workspace),
                        "analysis": AnalysisAgent(self, context, workspace),
                        "verification": VerificationAgent(self, context)}

        max_code = "answer = max(r['valueQuantity']['value'] for r in retrieved_resources['Observation'])"
        wrong = reply(json.dumps({**draft(answer=70, evidence=[cite("Observation/o1", value=70)]),
                                  "operation": "max of valueQuantity.value"}))
        responses = [
            fetch(), reply(json.dumps({"resource_types": ["Observation"], "encounter_ids": [], "limitation": None})),
            call(PYTHON_TOOL, "py1", code=max_code), wrong,                      # analysis: wrong aggregation
            fetch("fetch2"), call(PYTHON_TOOL, "py2", code=MIN_CODE), verdict(),  # verifier recomputes 68
            call(PYTHON_TOOL, "py3", code=MIN_CODE), reply(json.dumps(draft())),  # analysis repair
            fetch("fetch3"), call(PYTHON_TOOL, "py4", code=MIN_CODE), verdict(),   # second independent check
        ]
        packet = {"text": "Starting packet", "resources": {}, "resource_ids": {}}
        patches = [patch("agent.base_agent.setup_api_keys"),
                   patch("agent.retrieval_agent.get_starting_packet", return_value=packet)] + store_patches()
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        system = System("test-model")
        system._llm_call = Mock(side_effect=[(r, None, None) for r in responses])
        output = system.run(QUESTION)

        self.assertIsNone(output["error"])
        self.assertEqual(output["answer"], 68)
        revision = output["revision"]
        self.assertEqual([i["type"] for i in revision["check_issues"]], ["recompute_mismatch"])
        self.assertEqual(revision["repair_role"], "analysis")
        self.assertTrue(revision["answer_changed"])
        self.assertEqual(revision["final_check_issues"], [])
        self.assertEqual([c["name"] for c in output["tool_calls"]], [
            "get_resources_by_patient_fhir_id", PYTHON_TOOL, "check_fields", "get_resources_by_patient_fhir_id",
            PYTHON_TOOL, PYTHON_TOOL, "check_fields", "get_resources_by_patient_fhir_id", PYTHON_TOOL])
        self.assertEqual(output["budget"]["tool_calls_run"], 9)


if __name__ == "__main__":
    unittest.main()
