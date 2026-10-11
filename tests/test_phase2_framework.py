"""P2-11 spike: roles built on the in-house framework use the existing tool
registry and Python sandbox under one shared budget."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.base_agent import BaseAgent
from agent.framework import Bus, Role, Toolbox
from agent.output_schema import parse_final_message
from tools import cache

PATIENT = "11111111-1111-1111-1111-111111111111"
OBSERVATIONS = [
    {"resourceType": "Observation", "id": "o1", "valueQuantity": {"value": 20}},
    {"resourceType": "Observation", "id": "o2", "valueQuantity": {"value": 18}},
]


def tool_call(name, call_id="call1", **arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def calls(*tool_calls):
    return SimpleNamespace(role="assistant", content=None, tool_calls=list(tool_calls))


def reply(content):
    return SimpleNamespace(role="assistant", content=content, tool_calls=None)


def fake_client(resources=OBSERVATIONS):
    return Mock(search_with_pagination=Mock(return_value=resources))


class PipelineAgent(BaseAgent):
    """Two-role retrieval -> analysis pipeline, the shape the fixed controller (P2-12) will take."""

    def _run(self, question):
        bus, records = Bus(), {}
        retrieval = Role("retrieval", "Fetch records.", Toolbox(
            ["get_resources_by_patient_fhir_id"], records, patient_fhir_id=PATIENT), self)
        analysis = Role("analysis", "Compute the answer.", Toolbox(["execute_python_code"], records), self)
        bus.send("controller", "retrieval", "task", {"question": question})
        fetched = retrieval.run(question)
        if fetched.error:
            return {"answer": None, "error": fetched.error, "trace": self.messages}
        bus.send("retrieval", "analysis", "records",
                 {"ids": [f"Observation/{r['id']}" for r in records.get("Observation", [])]})
        draft = analysis.run(question)
        if draft.error:
            return {"answer": None, "error": draft.error, "trace": self.messages}
        answer, evidence, limitation = parse_final_message(draft.content)
        bus.send("analysis", "controller", "draft", {"answer": answer, "evidence": evidence})
        self.bus = bus
        return {"answer": answer, "evidence": evidence, "limitation": limitation,
                "trace": self.messages, "retrieved_fhir_resources": records}


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("agent.base_agent.setup_api_keys")
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_pipeline(self, responses, max_tool_calls=None):
        agent = PipelineAgent("test-model")
        agent.set_budget(max_tool_calls=max_tool_calls)
        agent._llm_call = Mock(side_effect=[(r, None, None) for r in responses])
        with patch.object(cache, "CACHE_ENABLED", False), \
                patch("tools.resource_tools.get_fhir_client", return_value=fake_client()):
            return agent, agent.run("What was the lowest value?")

    def test_toolbox_exposes_only_allow_listed_registry_tools(self):
        box = Toolbox(["execute_python_code", "check_fields"])
        self.assertEqual([d["function"]["name"] for d in box.definitions], ["execute_python_code", "check_fields"])
        with self.assertRaises(KeyError):
            Toolbox(["execute_python_code", "delete_patient"])

    def test_tool_outside_the_allow_list_is_refused_without_running(self):
        with patch("agent.framework.get_tool") as get_tool:
            message = Toolbox(["execute_python_code"]).call(tool_call("fhir_request_get", query_string="Patient"))
        get_tool.assert_not_called()
        self.assertIn("not available", json.loads(message["content"])["error"])

    def test_model_cannot_supply_undeclared_arguments(self):
        box = Toolbox(["execute_python_code"], {"Observation": OBSERVATIONS})
        message = box.call(tool_call("execute_python_code", code="answer = 1",
                                     global_vars={"retrieved_resources": {}}))
        self.assertIn("Unknown arguments: global_vars", message["content"])

    def test_bound_arguments_override_the_model(self):
        with patch("agent.framework.get_tool") as get_tool:
            get_tool.return_value = Mock(return_value={"issues": []})
            Toolbox(["check_fields"], bound={"check_fields": {"patient_fhir_id": PATIENT, "encounter_id": "e1"}}).call(
                tool_call("check_fields", citations=[], encounter_id="e2"))
        get_tool.return_value.assert_called_once_with(citations=[], encounter_id="e1", patient_fhir_id=PATIENT)

    def test_retrieval_is_limited_to_the_question_patient(self):
        box = Toolbox(["get_resources_by_patient_fhir_id"], patient_fhir_id=PATIENT)
        message = box.call(tool_call("get_resources_by_patient_fhir_id", resource_type="Observation",
                                     patient_fhir_id="22222222-2222-2222-2222-222222222222"))
        self.assertIn("only the patient", message["content"])
        self.assertEqual(box.resources, {})

    def test_sandboxed_analysis_reads_records_another_role_retrieved(self):
        agent, output = self.run_pipeline([
            calls(tool_call("get_resources_by_patient_fhir_id", resource_type="Observation", patient_fhir_id=PATIENT)),
            reply("Fetched."),
            calls(tool_call("execute_python_code",
                            code="answer = min(r['valueQuantity']['value'] for r in retrieved_resources['Observation'])")),
            reply(json.dumps({"answer": 18, "evidence": [
                {"resource_id": "Observation/o2", "field_path": "valueQuantity.value", "value": 18}],
                "limitation": None})),
        ])
        self.assertIsNone(output["error"])
        self.assertEqual(output["answer"], 18)
        python_reply = next(m for m in output["trace"] if isinstance(m, dict) and m.get("name") == "execute_python_code")
        self.assertEqual(json.loads(python_reply["content"]), {"answer": 18})
        self.assertEqual([c["name"] for c in output["tool_calls"]],
                         ["get_resources_by_patient_fhir_id", "execute_python_code"])
        self.assertEqual(output["budget"]["tool_calls_run"], 2)
        self.assertEqual([m["kind"] for m in agent.bus.to_list()], ["task", "records", "draft"])

    def test_tool_budget_is_shared_across_roles(self):
        agent, output = self.run_pipeline([
            calls(tool_call("get_resources_by_patient_fhir_id", resource_type="Observation", patient_fhir_id=PATIENT)),
            reply("Fetched."),
            calls(tool_call("execute_python_code", code="answer = 1")),
            reply(json.dumps({"answer": None, "evidence": [], "limitation": "No calculation run"})),
        ], max_tool_calls=1)
        self.assertEqual(output["budget"]["tool_calls_run"], 1)
        self.assertIn("tool-call budget", output["budget"]["stopped"])
        refused = [m for m in output["trace"] if isinstance(m, dict) and m.get("name") == "execute_python_code"]
        self.assertIn("Not run", refused[0]["content"])

    def test_each_role_run_starts_from_a_fresh_context(self):
        agent = PipelineAgent("test-model")
        agent.messages = []
        agent._llm_call = Mock(side_effect=[(reply("first"), None, None), (reply("second"), None, None)])
        role = Role("verification", "Check.", Toolbox([]), agent)
        role.run("task one")
        role.run("task two")
        second_prompt = agent._llm_call.call_args_list[1].args[0]
        # The mock keeps the role's live context: system prompt, the new task, and its reply
        self.assertEqual([m["content"] for m in second_prompt[:2]], ["Check.", "task two"])
        self.assertEqual(len(second_prompt), 3)

    def test_role_error_and_turn_limit_are_reported(self):
        agent = PipelineAgent("test-model")
        agent.messages = []
        agent._llm_call = Mock(return_value=(None, "provider down", None))
        self.assertEqual(Role("analysis", "", Toolbox([]), agent).run("x").error, "provider down")
        agent._llm_call = Mock(return_value=(calls(tool_call("execute_python_code", code="answer = 1")), None, None))
        role = Role("analysis", "", Toolbox(["execute_python_code"]), agent, max_turns=2)
        with patch("agent.framework.get_tool", return_value=Mock(return_value={"answer": 1})):
            self.assertIn("no final reply within 2 turns", role.run("x").error)

    def test_bus_copies_payloads_and_rejects_non_json(self):
        bus, payload = Bus(), {"ids": ["Observation/o1"]}
        bus.send("retrieval", "analysis", "records", payload)
        payload["ids"].append("Observation/o2")
        self.assertEqual(bus.log[0].payload, {"ids": ["Observation/o1"]})
        for bad in ({"value": float("nan")}, {"value": object()}, ["not", "an", "object"]):
            with self.assertRaises((TypeError, ValueError)):
                bus.send("a", "b", "draft", bad)


if __name__ == "__main__":
    unittest.main()
