"""V10 shared retrieval: get_linked_resources follows one record's references, for every arm."""
from copy import deepcopy
from types import SimpleNamespace
import json
import unittest
from unittest.mock import Mock, patch

from agent.single_agent import LINK_TOOL, SingleAgent
from tools import cache
from tools.resource_tools import get_linked_resources

TEST = {"resourceType": "Observation", "id": "o1", "subject": {"reference": "Patient/p1"},
        "specimen": {"reference": "Specimen/s1"},
        "hasMember": [{"reference": "Observation/g1"}, {"reference": "Observation/g2"}]}
RECORDS = [TEST,
           {"resourceType": "Specimen", "id": "s1", "subject": {"reference": "Patient/p1"}},
           {"resourceType": "Observation", "id": "g1", "subject": {"reference": "Patient/p1"}},
           {"resourceType": "Observation", "id": "g2", "subject": {"reference": "Patient/p2"}}]


def search(query):
    rtype, _, params = query.partition("?_id=")
    ids = params.split("&")[0].split(",")
    return [deepcopy(r) for r in RECORDS if r["resourceType"] == rtype and r["id"] in ids]


def call(**arguments):
    return SimpleNamespace(id="c1", function=SimpleNamespace(name=LINK_TOOL, arguments=json.dumps(arguments)))


class LinkedResourcesTest(unittest.TestCase):
    def setUp(self):
        for patcher in (patch.object(cache, "CACHE_ENABLED", False),
                        patch("tools.resource_tools.get_fhir_client",
                              return_value=Mock(search_with_pagination=Mock(side_effect=search)))):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_follows_specimen_and_children(self):
        out = get_linked_resources("Observation", "o1", "specimen")
        self.assertEqual([r["id"] for r in out["Specimen"]], ["s1"])
        self.assertEqual(out["linked_ids"], ["Specimen/s1"])
        out = get_linked_resources("Observation", "o1", "hasMember")
        self.assertEqual([r["id"] for r in out["Observation"]], ["o1", "g1", "g2"])

    def test_rejects_unknown_fields_and_missing_records(self):
        self.assertIn("error", get_linked_resources("Observation", "o1", "subject"))
        self.assertIn("error", get_linked_resources("Encounter", "e1", "partOf"))
        self.assertIn("error", get_linked_resources("Observation", "o9", "specimen"))

    def test_single_agent_merges_only_the_patients_records(self):
        with patch("agent.base_agent.setup_api_keys"):
            agent = SingleAgent("base-model")
        self.assertIn(LINK_TOOL, [t["function"]["name"] for t in agent.tools])
        agent._patient_fhir_id = "p1"
        workspace = {"Observation": [{"resourceType": "Observation", "id": "g1", "stale": True}]}
        reply = agent._handle_tool_call(call(resource_type="Observation", resource_id="o1", field="hasMember"),
                                        workspace)
        content = json.loads(reply["content"])
        self.assertEqual(content["linked_ids"], ["Observation/g1"])  # g2 belongs to another patient
        self.assertEqual(sorted(r["id"] for r in workspace["Observation"]), ["g1", "o1"])
        self.assertNotIn("stale", next(r for r in workspace["Observation"] if r["id"] == "g1"))
        agent._patient_fhir_id = "p2"
        reply = agent._handle_tool_call(call(resource_type="Observation", resource_id="o1", field="specimen"), {})
        self.assertIn("Retrieve only the patient", reply["content"])

    def test_the_ledger_solver_does_not_get_the_link_tool(self):
        from agent.query_contract_agent import LedgerSolver
        parent = SimpleNamespace(model="base-model", verbose=False, base_url=None, perturbation=None)
        with patch("agent.base_agent.setup_api_keys"):
            solver = LedgerSolver(parent, phase=None)
        self.assertNotIn(LINK_TOOL, [t["function"]["name"] for t in solver.tools])


if __name__ == "__main__":
    unittest.main()
