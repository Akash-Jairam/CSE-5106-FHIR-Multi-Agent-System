"""Grounded blind verification and stable rechecking of repaired candidates."""
import json
import unittest
from unittest.mock import Mock, patch

from agent.base_agent import BaseAgent
from agent.framework import PYTHON_TOOL
from agent.verification_agent import VerificationAgent
from tests.test_phase2_verification import CONTEXT, MIN_CODE, PATIENT, call, cite, draft, fetch, reply, store_patches


FINAL_CODE = (MIN_CODE + '\nevidence = [{"resource_id": "Observation/o2", '
              '"field_path": "valueQuantity.value", "value": answer}]\n'
              'operation = "minimum of the patient weight readings"\n'
              'contributing_ids = ["Observation/" + r["id"] for r in retrieved_resources["Observation"]]\n'
              'limitation = None')


class BlindVerificationTests(unittest.TestCase):
    def setUp(self):
        for patcher in [patch("agent.base_agent.setup_api_keys")] + store_patches():
            patcher.start()
            self.addCleanup(patcher.stop)
        self.host = BaseAgent("test-model")
        self.agent = VerificationAgent(self.host, CONTEXT)

    def verify(self, responses, candidate=None):
        self.host._llm_call = Mock(side_effect=[(r, None, None) for r in responses])
        return self.agent({"draft": candidate or draft()})

    def final(self, code=FINAL_CODE):
        return call(PYTHON_TOOL, "final-calculation", code=code, mode="final")

    def test_independent_prompt_excludes_candidate_and_source_findings(self):
        candidate = draft(answer="CANDIDATE_ANSWER_MARKER", evidence=[cite(value="CITATION_VALUE_MARKER")])
        candidate["operation"] = "CANDIDATE_OPERATION_MARKER"
        self.verify([fetch(), self.final(), reply("executed")], candidate)
        requests = self.host._llm_call.call_args_list
        rendered = json.dumps(requests[0].args[0], default=lambda obj: vars(obj))
        for marker in ["CANDIDATE_ANSWER_MARKER", "CITATION_VALUE_MARKER", "CANDIDATE_OPERATION_MARKER",
                       "value_mismatch", "Draft to verify"]:
            self.assertNotIn(marker, rendered)
        self.assertIn(CONTEXT["question"], rendered)
        self.assertIn(PATIENT, rendered)

    def test_executed_final_survives_malformed_summary(self):
        result = self.verify([fetch(), self.final(), reply("The calculation finished.")])
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["recomputed_answer"], 68)
        self.assertEqual(result["independent_candidate"]["evidence"], [cite()])
        self.assertTrue(result["independent_computation"]["workspace_sha256"])

    def test_read_only_inspection_preserves_explicit_final(self):
        inspect = call(PYTHON_TOOL, "inspect", code="answer = {'record_count': len(retrieved_resources['Observation'])}",
                       mode="inspect")
        result = self.verify([fetch(), self.final(), inspect, reply("done")])
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["recomputed_answer"], 68)

    def test_inspection_scalar_is_not_an_independent_final(self):
        inspect = call(PYTHON_TOOL, "inspect", code=MIN_CODE, mode="inspect")
        result = self.verify([fetch(), inspect, reply('{"recomputed_answer": 68, "issues": []}')])
        self.assertIn("independent_check", [issue["type"] for issue in result["issues"]])

    def test_production_verifier_cannot_use_legacy_auto_diagnostic_as_certification(self):
        self.agent = VerificationAgent(self.host, CONTEXT, require_explicit_final=True)
        auto = call(PYTHON_TOOL, "auto", code=MIN_CODE)
        result = self.verify([fetch(), auto, reply('{"recomputed_answer":68,"issues":[]}')])
        self.assertIn("verification_failure", [issue["type"] for issue in result["issues"]])
        self.assertIsNone(self.agent._independent_cache)

    def test_later_fetch_invalidates_explicit_final_even_when_fetch_fails(self):
        for later in [fetch("later"), call("get_resources_by_patient_fhir_id", "wrong-patient",
                                          resource_type="Observation", patient_fhir_id="different")]:
            with self.subTest(later=later):
                self.agent = VerificationAgent(self.host, CONTEXT)
                result = self.verify([fetch(), self.final(), later,
                                      reply('{"recomputed_answer": 68, "issues": []}')])
                self.assertIn("independent_check", [issue["type"] for issue in result["issues"]])

    def test_repair_reuses_frozen_independent_computation_and_rechecks_citations(self):
        result = self.verify([fetch(), self.final(), reply("done")], draft(answer=70))
        self.assertEqual([issue["type"] for issue in result["issues"]], ["recompute_mismatch"])
        calls_before = self.host.tool_calls_made
        self.host._llm_call.reset_mock()
        repaired = self.agent({"draft": draft()})
        self.assertEqual(repaired["issues"], [])
        self.assertTrue(repaired["independent_check_reused"])
        self.host._llm_call.assert_not_called()
        self.assertEqual(self.host.tool_calls_made, calls_before + 1)
        bad = self.agent({"draft": draft(evidence=[cite(value=999)])})
        self.assertIn("value_mismatch", [issue["type"] for issue in bad["issues"]])

    def test_cached_computation_cannot_cross_changed_workspace_or_context(self):
        for change in ("workspace", "context"):
            with self.subTest(change=change):
                self.agent = VerificationAgent(self.host, dict(CONTEXT))
                self.verify([fetch(), self.final(), reply("done")])
                if change == "workspace":
                    self.agent.workspace["Observation"][0]["valueQuantity"]["value"] = 1
                else:
                    self.agent.context["question"] = "What was the highest weight?"
                self.host._llm_call.reset_mock()
                result = self.agent({"draft": draft()})
                self.assertIn("verification_failure", [issue["type"] for issue in result["issues"]])
                self.host._llm_call.assert_not_called()

    def test_cached_recheck_still_requires_source_availability_and_budget(self):
        self.verify([fetch(), self.final(), reply("done")])
        with patch("agent.verification_agent.check_fields", side_effect=RuntimeError("source down")):
            failed = self.agent({"draft": draft()})
        self.assertIn("source_check", [issue["type"] for issue in failed["issues"]])
        self.host.set_budget(max_tool_calls=self.host.tool_calls_made)
        failed = self.agent({"draft": draft()})
        self.assertIn("source_check", [issue["type"] for issue in failed["issues"]])

    def test_diagnostic_object_cannot_be_final_answer(self):
        result = self.verify([fetch(), self.final("answer = {'record_count': 2}"),
                              reply('{"recomputed_answer": 68, "issues": []}'), reply("No correction possible")])
        self.assertTrue(result["issues"])

    def test_explicit_final_does_not_discard_a_valid_independent_warning(self):
        warning = {"type": "missing_evidence", "responsible_role": "retrieval",
                   "action": "A needed record could not be located"}
        result = self.verify([fetch(), self.final(), reply(json.dumps({
            "recomputed_answer": 999, "issues": [warning]}))])
        self.assertEqual(result["recomputed_answer"], 68)
        self.assertEqual([issue["type"] for issue in result["issues"]], ["missing_evidence"])
        self.assertIsNone(self.agent._independent_cache)

    def test_executed_limitation_blocks_independent_certification(self):
        result = self.verify([fetch(), self.final(FINAL_CODE + '\nlimitation = "The requested scope is ambiguous"'),
                              reply("done")])
        self.assertIn("independent_check", [issue["type"] for issue in result["issues"]])
        self.assertIsNone(self.agent._independent_cache)

    def test_explicit_empty_search_and_computation_need_no_invented_citations(self):
        with patch("tools.resource_tools.get_fhir_client", return_value=Mock(
                search_with_pagination=Mock(return_value=[]))):
            for answer, code in [(None, "answer = next(iter(retrieved_resources['Observation']), None)"),
                                 (0, "answer = len(retrieved_resources['Observation'])"),
                                 (False, "answer = bool(retrieved_resources['Observation'])")]:
                with self.subTest(answer=answer):
                    self.agent = VerificationAgent(self.host, CONTEXT)
                    complete = code + '\nevidence = []\ncontributing_ids = []\noperation = "search scoped observations"\nlimitation = None'
                    result = self.verify([fetch(), self.final(complete), reply("done")],
                                         draft(answer=answer, evidence=[]))
                    self.assertEqual(result["issues"], [])
                    self.assertEqual(result["recomputed_answer"], answer)

    def test_explicit_final_requires_complete_valid_execution_metadata(self):
        invalid = [MIN_CODE,
                   MIN_CODE + '\nevidence = []\ncontributing_ids = []\noperation = "constant answer"',
                   FINAL_CODE + '\nevidence = {"not": "a citation list"}',
                   FINAL_CODE + '\ncontributing_ids = ["Observation/not-held"]',
                   FINAL_CODE + '\noperation = None',
                   FINAL_CODE + '\nevidence[0]["field_path"] = "missing.path"',
                   FINAL_CODE + '\nevidence[0]["value"] = 999']
        for code in invalid:
            with self.subTest(code=code):
                self.agent = VerificationAgent(self.host, CONTEXT)
                result = self.verify([fetch(), self.final(code), reply("done"), reply("No correction possible")])
                self.assertTrue(result["issues"])
                self.assertIsNone(self.agent._independent_cache)

    def test_independent_evidence_source_failure_cannot_certify_candidate(self):
        from agent.verification_agent import check_fields
        calls = 0

        def check(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("independent source unavailable")
            return check_fields(*args, **kwargs)

        with patch("agent.verification_agent.check_fields", side_effect=check):
            result = self.verify([fetch(), self.final(), reply("done")])
        self.assertIn("independent_evidence", [issue["type"] for issue in result["issues"]])
        self.assertIsNone(self.agent._independent_cache)

    def test_one_counted_blind_metadata_recalculation_can_recover(self):
        malformed = FINAL_CODE.replace('"resource_id"', '"resource"')
        candidate = draft()
        candidate["operation"] = "PRIVATE_CANDIDATE_OPERATION"
        result = self.verify([fetch(), self.final(malformed), reply("Independent calculation finished"),
                              self.final(), reply("Corrected metadata")], candidate)
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["recomputed_answer"], 68)
        self.assertEqual(self.host.tool_calls_made, 5)  # both source checks, fetch, bad final, corrected final
        correction_request = self.host._llm_call.call_args_list[-1]
        encoded = json.dumps(correction_request.args[0], default=lambda obj: vars(obj))
        self.assertNotIn("PRIVATE_CANDIDATE_OPERATION", encoded)
        self.assertIn("ONE counted final Python correction", encoded)

    def test_failed_metadata_correction_is_bounded_and_cannot_certify(self):
        malformed = FINAL_CODE.replace('"resource_id"', '"resource"')
        result = self.verify([fetch(), self.final(malformed), reply("done"),
                              self.final(malformed), reply("done")])
        self.assertTrue(result["issues"])
        self.assertEqual(self.host.tool_calls_made, 4)
        self.assertEqual(self.host._llm_call.call_count, 5)
        self.assertIsNone(self.agent._independent_cache)

    def test_metadata_correction_cannot_exceed_role_allocation(self):
        self.agent.role.max_tool_calls = 2
        malformed = FINAL_CODE.replace('"resource_id"', '"resource"')
        result = self.verify([fetch(), self.final(malformed), reply("done")])
        self.assertTrue(result["issues"])
        self.assertEqual(self.host.tool_calls_made, 3)
        self.assertEqual(self.host._llm_call.call_count, 3)


if __name__ == "__main__":
    unittest.main()
