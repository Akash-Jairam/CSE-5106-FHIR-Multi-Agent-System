"""Regression tests for source-scoped retrieval, without live FHIR/model calls."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.framework import RETRIEVAL_TOOLS, Toolbox
from agent.retrieval_agent import RetrievalAgent

PATIENT = "patient1"
NOW = "2153-12-31 23:59:00"


def record(rtype, rid, **fields):
    return {"resourceType": rtype, "id": rid, **fields}


def call(rtype, rid=None):
    name = RETRIEVAL_TOOLS[1 if rid else 0]
    args = {"resource_type": rtype,
            "resource_id" if rid else "patient_fhir_id": rid or PATIENT}
    return SimpleNamespace(id="fetch", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class RetrievalScopeReviewTests(unittest.TestCase):
    def fetch(self, workspace, rtype, records, rid=None):
        box = Toolbox(RETRIEVAL_TOOLS, workspace, patient_fhir_id=PATIENT, current_time=NOW)
        with patch("agent.framework.get_tool", return_value=Mock(return_value={rtype: records})):
            return json.loads(box.call(call(rtype, rid))["content"])

    def test_ownerless_patient_record_is_not_accepted_by_id(self):
        workspace = {}
        reply = self.fetch(workspace, "Observation", [record("Observation", "unknown", valueString="secret")],
                           rid="unknown")
        self.assertEqual(workspace.get("Observation", []), [])
        self.assertEqual(reply["excluded"]["unknown_patient"], 1)

    def test_unrelated_global_medication_is_not_in_patient_workspace(self):
        workspace = {"MedicationRequest": [record("MedicationRequest", "rx", subject={"reference": "Patient/patient1"},
                                                  medicationReference={"reference": "Medication/m1"})]}
        reply = self.fetch(workspace, "Medication", [record("Medication", "m1"), record("Medication", "m2")])
        self.assertEqual([r["id"] for r in workspace["Medication"]], ["m1"])
        self.assertEqual(reply["excluded"]["unlinked_metadata"], 1)

    def test_prescription_start_governs_current_time_not_order_time(self):
        workspace = {}
        prescription = record("MedicationRequest", "rx", subject={"reference": "Patient/patient1"},
                              authoredOn="2150-01-01T00:00:00",
                              dispenseRequest={"validityPeriod": {"start": "2154-01-01T00:00:00"}})
        reply = self.fetch(workspace, "MedicationRequest", [prescription])
        self.assertEqual(workspace.get("MedicationRequest", []), [])
        self.assertEqual(reply["excluded"]["after_current_time"], 1)

    def test_unknown_encounter_time_is_not_assumed_in_scope(self):
        workspace = {}
        condition = record("Condition", "c1", subject={"reference": "Patient/patient1"},
                           encounter={"reference": "Encounter/future"})
        reply = self.fetch(workspace, "Condition", [condition])
        self.assertEqual(workspace.get("Condition", []), [])
        self.assertEqual(reply["excluded"]["unknown_time"], 1)

    def test_patient_refresh_does_not_erase_a_point_fetched_record(self):
        old = record("Observation", "old", subject={"reference": "Patient/patient1"},
                     effectiveDateTime="2150-01-01T00:00:00")
        new = record("Observation", "new", subject={"reference": "Patient/patient1"},
                     effectiveDateTime="2151-01-01T00:00:00")
        workspace = {"Observation": [old]}
        self.fetch(workspace, "Observation", [new])
        self.assertEqual({r["id"] for r in workspace["Observation"]}, {"old", "new"})

    def test_encounter_fetch_returns_scope_fields_not_only_opaque_ids(self):
        encounter = record("Encounter", "e1", subject={"reference": "Patient/patient1"},
                           identifier=[{"system": "http://mimic/encounter-hosp", "value": "1"}],
                           period={"start": "2150-01-01T00:00:00"})
        reply = self.fetch({}, "Encounter", [encounter])
        self.assertEqual(reply["records"][0]["period"], encounter["period"])

    def test_wrong_type_or_malformed_records_are_refused(self):
        for records in ([record("Patient", PATIENT)], [None], [record("Observation", "")]):
            with self.subTest(records=records):
                reply = self.fetch({}, "Observation", records)
                self.assertIn("error", reply)

    def test_condition_uses_its_known_encounter_time(self):
        condition = record("Condition", "c1", subject={"reference": "Patient/patient1"},
                           encounter={"reference": "Encounter/e1"})
        workspace = {"Encounter": [record("Encounter", "e1", period={"start": "2150-01-01T00:00:00"})]}
        self.fetch(workspace, "Condition", [condition])
        self.assertEqual([r["id"] for r in workspace["Condition"]], ["c1"])

    def test_location_metadata_must_be_linked_to_a_retrieved_encounter(self):
        workspace = {"Encounter": [record("Encounter", "e1", location=[
            {"location": {"reference": "Location/l1"}}])]}
        self.fetch(workspace, "Location", [record("Location", "l1"), record("Location", "l2")])
        self.assertEqual([r["id"] for r in workspace["Location"]], ["l1"])

    def test_subject_of_another_resource_type_does_not_establish_patient(self):
        workspace = {}
        self.fetch(workspace, "Observation", [record("Observation", "o1", subject={"reference": "Group/patient1"},
                                                   effectiveDateTime="2150-01-01T00:00:00")])
        self.assertEqual(workspace.get("Observation", []), [])

    def test_point_fetch_requires_the_requested_id(self):
        reply = self.fetch({}, "Observation", [record("Observation", "other")], rid="requested")
        self.assertIn("requested ID", reply["error"])

    def test_unknown_time_is_retained_as_limitation_until_resolved(self):
        workspace = {}
        box = Toolbox(RETRIEVAL_TOOLS, workspace, patient_fhir_id=PATIENT, current_time=NOW)
        condition = record("Condition", "c1", subject={"reference": "Patient/patient1"},
                           encounter={"reference": "Encounter/e1"})
        with patch("agent.framework.get_tool", return_value=Mock(return_value={"Condition": [condition]})):
            box.call(call("Condition"))
            self.assertEqual(box.scope_failures, {"Condition/c1": "unknown_time"})
            workspace["Encounter"] = [record("Encounter", "e1", period={"start": "2150-01-01T00:00:00"})]
            box.call(call("Condition"))
            self.assertEqual(box.scope_failures, {})


class RetrievalSummaryReviewTests(unittest.TestCase):
    def test_invalid_structured_scope_does_not_silently_become_unscoped(self):
        host = SimpleNamespace(messages=[], _llm_call=Mock(return_value=(
            SimpleNamespace(tool_calls=None, content='{"resource_types": [], "encounter_ids": [42]}'),
            None, None)))
        agent = RetrievalAgent(host, {"question": "q", "patient_fhir_id": PATIENT}, {})
        agent.packet = {"resources": {}, "text": "Starting packet"}
        self.assertIn("error", agent({}))


if __name__ == "__main__":
    unittest.main()
