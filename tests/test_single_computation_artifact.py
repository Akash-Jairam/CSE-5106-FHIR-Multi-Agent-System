"""The refreshed single context gets the same executed candidate contract as the team."""
import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.single_agent import SingleAgent


PATIENT = "11111111-1111-1111-1111-111111111111"
RESOURCE = {"resourceType": "Observation", "id": "o1", "valueQuantity": {"value": 7}}
QUESTION = f"Question: What was the weight?\nContext:\nPatient FHIR ID is {PATIENT}."
PACKET = {"text": "Observation o1 weight 7", "resources": {"Observation": [RESOURCE]}}


def computation(answer=7, **changes):
    return {"answer": answer, "evidence": [{"resource_id": "Observation/o1",
            "field_path": "valueQuantity.value", "value": answer}], "contributing_ids": ["Observation/o1"],
            "operation": "lookup weight", "limitation": None, "mode": "final", **changes}


def tool(name="execute_python_code", **arguments):
    return SimpleNamespace(id="t", function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def assistant(content=None, *calls):
    return SimpleNamespace(role="assistant", content=content, tool_calls=list(calls) or None)


class SingleComputationArtifactTests(unittest.TestCase):
    def run_single(self, responses, outputs):
        with patch("agent.base_agent.setup_api_keys"), \
                patch("agent.single_agent.get_starting_packet", return_value=deepcopy(PACKET)), \
                patch("agent.multi_turn_code_resource_agent.get_tool", return_value=Mock(side_effect=outputs)), \
                patch("agent.single_agent.check_fields", side_effect=lambda citations, **_: {
                    "issues": [] if all(citation.get("value") == 7 for citation in citations) else ["value_mismatch"],
                    "n_valid": len(citations)}):
            agent = SingleAgent("fixture", max_iterations=len(responses))
            agent._llm_call = Mock(side_effect=[(item, None, None) for item in responses])
            return agent, agent.run(QUESTION)

    def test_executed_answer_wins_over_malformed_or_changed_prose(self):
        for prose in ('{"answer": 99', json.dumps({"answer": 99, "evidence": [], "limitation": None})):
            _, result = self.run_single([
                assistant(None, tool(code="answer=7", mode="final")), assistant(prose)], [computation()])
            self.assertEqual(result["answer"], 7)
            self.assertEqual(result["evidence"], computation()["evidence"])
            self.assertIsNone(result["error"])

    def test_diagnostic_inspection_preserves_full_final_artifact(self):
        agent, result = self.run_single([
            assistant(None, tool(code="answer=7", mode="final")),
            assistant(None, tool(code="answer={'diagnostic': 1}", mode="inspect")),
            assistant("Finished.")], [computation(), {"answer": {"diagnostic": 1}, "mode": "inspect"}])
        self.assertEqual(result["answer"], 7)
        self.assertEqual(agent.last_execution["mode"], "inspect")
        self.assertEqual(agent.final_computation["answer"], 7)

    def test_invalid_metadata_returns_tool_error_and_can_be_corrected(self):
        _, result = self.run_single([
            assistant(None, tool(code="answer=7", mode="final")),
            assistant(None, tool(code="answer=7", mode="final")), assistant("Finished.")],
            [computation(contributing_ids=["Observation/not-held"]), computation()])
        replies = [item for item in result["trace"] if isinstance(item, dict) and item.get("name") == "execute_python_code"]
        self.assertIn("Invalid final computation", json.loads(replies[0]["content"])["error"])
        self.assertEqual(result["answer"], 7)

    def test_invalid_final_artifact_cannot_fall_back_to_copied_prose(self):
        _, result = self.run_single([
            assistant(None, tool(code="answer=7", mode="final")),
            assistant(json.dumps({"answer": 7, "evidence": computation()["evidence"], "limitation": None}))],
            [computation(operation="")])
        self.assertIsNone(result["answer"])

    def test_source_evidence_revision_recomputes_once(self):
        _, result = self.run_single([
            assistant(None, tool(code="answer=99", mode="final")), assistant("Finished."),
            assistant(None, tool(code="answer=7", mode="final")), assistant("Finished.")],
            [computation(99), computation()])
        self.assertEqual(result["answer"], 7)
        self.assertEqual(result["revision"]["draft"]["answer"], 99)
        self.assertEqual(result["revision"]["final"]["answer"], 7)
        self.assertTrue(result["revision"]["revised"])
        self.assertEqual(sum(item["name"] == "check_fields" for item in result["tool_calls"]), 2)

    def test_failed_retrieval_and_changed_workspace_invalidate_authority(self):
        with patch("agent.base_agent.setup_api_keys"):
            agent = SingleAgent("fixture")
        agent._patient_fhir_id = PATIENT
        resources = {"Observation": [deepcopy(RESOURCE)]}
        with patch("agent.multi_turn_code_resource_agent.get_tool", return_value=Mock(return_value=computation())):
            agent._handle_tool_call(tool(code="answer=7", mode="final"), resources)
        resources["Observation"][0]["valueQuantity"]["value"] = 8
        self.assertIsNone(agent._computed_candidate(resources))
        with patch("agent.multi_turn_code_resource_agent.get_tool", return_value=Mock(return_value=computation())):
            agent._handle_tool_call(tool(code="answer=7", mode="final"), resources)
        message = agent._handle_tool_call(tool("get_resources_by_patient_fhir_id", patient_fhir_id="other"), resources)
        self.assertIn("Retrieve only", message["content"])
        self.assertIsNone(agent.final_computation)
        self.assertIsNone(agent.last_execution)

    def test_failed_auto_or_final_clears_stale_final(self):
        for mode in ("auto", "final"):
            with patch("agent.base_agent.setup_api_keys"):
                agent = SingleAgent("fixture")
            resources = {"Observation": [deepcopy(RESOURCE)]}
            with patch("agent.multi_turn_code_resource_agent.get_tool", return_value=Mock(side_effect=[
                    computation(), {"error": "failed computation"}])):
                agent._handle_tool_call(tool(code="answer=7", mode="final"), resources)
                agent._handle_tool_call(tool(code="raise ValueError()", mode=mode), resources)
            self.assertIsNone(agent.final_computation)
            self.assertIsNone(agent.last_execution)

    def test_validated_empty_search_can_release_zero_false_or_empty_list_without_citations(self):
        for value in (0, False, []):
            artifact = computation(value, evidence=[], contributing_ids=[], operation="Completed scoped search; no matches")
            _, result = self.run_single([
                assistant(None, tool(code=f"answer={value!r}", mode="final")), assistant("Finished.")], [artifact])
            self.assertEqual(result["answer"], value)
            self.assertEqual(result["revision"]["final_check_issues"], [])
        _, legacy = self.run_single([assistant(json.dumps({"answer": 0, "evidence": [], "limitation": None}))], [])
        self.assertIsNone(legacy["answer"])
        self.assertIn("no evidence", legacy["limitation"])


if __name__ == "__main__":
    unittest.main()
