"""The multi-agent team's release: a vote among the analysis, packet-reading and
verifier answers; abstentions do not vote; a later role failure keeps the draft."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.multi_agent import MultiAgentSystem
from agent.output_schema import validate_output

PATIENT = "11111111-1111-1111-1111-111111111111"
QUESTION = (f"Question: What was the last weight of patient 10001?\nContext:\nPatient FHIR ID is {PATIENT}.\n"
            "Assume current time is 2153-12-31 23:59:00.")
CITATION = {"resource_id": "Observation/o1", "field_path": "valueQuantity.value", "value": 7}


def draft(answer=7, evidence=(CITATION,)):
    return {"answer": answer, "evidence": list(evidence), "limitation": None}


def verdict(recomputed, issues=()):
    return {"issues": list(issues), "recomputed_answer": recomputed}


MISMATCH = {"type": "recompute_mismatch", "record": None, "field": None, "responsible_role": "analysis",
            "action": "recheck"}


class StubTeam(MultiAgentSystem):
    def __init__(self, direct=None, analysis=(), verification=()):
        with patch("agent.base_agent.setup_api_keys"):
            super().__init__("test-model")
        self.direct = direct
        self.scripts = {"analysis": list(analysis), "verification": list(verification)}

    def build_roles(self, context, workspace):
        self._recomputed = []
        self._retrieval = SimpleNamespace(direct_candidate=self.direct)

        def handler(role):
            def handle(task):
                result = self.scripts[role].pop(0)
                if isinstance(result, Exception):
                    raise result
                return result
            return handle
        return {"retrieval": lambda task: {"record_ids": {}}, "analysis": handler("analysis"),
                "verification": handler("verification")}


class ReleaseTests(unittest.TestCase):
    def run_team(self, team):
        output = team.run(QUESTION)
        self.assertEqual(validate_output(output), [])
        return output

    def test_agreeing_draft_and_verifier_release_the_draft_with_its_citations(self):
        output = self.run_team(StubTeam(direct=draft(5, ()), analysis=[draft()], verification=[verdict(7)]))
        self.assertEqual((output["answer"], output["evidence"], output["error"]), (7, [CITATION], None))
        self.assertEqual(output["revision"]["release"]["winner"], "draft")

    def test_a_repair_the_verifier_agrees_with_is_released(self):
        output = self.run_team(StubTeam(analysis=[draft(), draft(5)],
                                        verification=[verdict(5, [MISMATCH]), verdict(5)]))
        self.assertEqual((output["answer"], output["revision"]["release"]["winner"]), (5, "repaired"))

    def test_packet_reading_and_verifier_outvote_the_analysis(self):
        output = self.run_team(StubTeam(direct=draft(5, ()), analysis=[draft(), draft(9)],
                                        verification=[verdict(5, [MISMATCH]), verdict(5, [MISMATCH])]))
        self.assertEqual((output["answer"], output["revision"]["release"]["winner"]), (5, "direct"))

    def test_open_issues_do_not_withhold_the_answer(self):
        output = self.run_team(StubTeam(analysis=[draft(), draft()],
                                        verification=[verdict(9, [MISMATCH]), verdict(9, [MISMATCH])]))
        # A tie between the analysis and the verifier goes to the analysis
        self.assertEqual(output["answer"], 7)

    def test_an_abstaining_draft_is_rescued_by_the_packet_reading(self):
        output = self.run_team(StubTeam(direct=draft(5), analysis=[draft(None, ())],
                                        verification=[verdict(None)]))
        self.assertEqual(output["answer"], 5)

    def test_a_verifier_failure_after_a_draft_keeps_the_draft(self):
        output = self.run_team(StubTeam(analysis=[draft()], verification=[{"error": "no final reply"}]))
        self.assertEqual((output["answer"], output["error"]), (7, None))
        self.assertIn("verification", output["revision"]["revision_error"])

    def test_an_analysis_failure_falls_back_to_the_packet_reading(self):
        output = self.run_team(StubTeam(direct=draft(5), analysis=[{"error": "no final reply"}]))
        self.assertEqual((output["answer"], output["error"]), (5, None))

    def test_no_answer_at_all_is_a_retryable_error(self):
        output = self.run_team(StubTeam(analysis=[{"error": "provider down"}]))
        self.assertIsNone(output["answer"])
        self.assertIn("provider down", output["error"])


if __name__ == "__main__":
    unittest.main()
