"""Patient-rooted batched Medication closure and strict metadata reachability."""
import json
import tempfile
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from agent.framework import Toolbox
from fhir_client import FHIRClient
from tools import cache
from tools.resource_tools import get_resources_by_patient_fhir_id, get_all_resources_by_patient_fhir_id


PATIENT = "p1"


def event(rid="request", patient=PATIENT, medication="mix", rtype="MedicationRequest"):
    return {"resourceType": rtype, "id": rid, "subject": {"reference": "Patient/" + patient},
            "dispenseRequest": {"validityPeriod": {"start": "2150-01-01T10:00:00"}},
            "effectiveDateTime": "2150-01-01T10:00:00", "medicationReference": {"reference": "Medication/" + medication}}


def medication(rid, ingredients=(), name=None):
    result = {"resourceType": "Medication", "id": rid,
              "ingredient": [{"itemReference": {"reference": "Medication/" + target}} for target in ingredients]}
    if name:
        result["code"] = {"coding": [{"display": name}]}
    return result


class Store:
    def __init__(self, requests=(), administrations=(), medications=()):
        self.requests = list(requests)
        self.administrations = list(administrations)
        self.medications = {record["id"]: record for record in medications}
        self.queries = []
        self.fhir_store_url = "https://example.invalid/fhir"

    def search_with_pagination(self, query):
        self.queries.append(query)
        if query.startswith("MedicationRequest?"):
            result = deepcopy(self.requests)
        elif query.startswith("MedicationAdministration?"):
            result = deepcopy(self.administrations)
        elif query.startswith("Medication?_id="):
            ids = query.split("_id=", 1)[1].split("&", 1)[0].split(",")
            result = [deepcopy(self.medications[rid]) for rid in ids if rid in self.medications]
        else:
            result = []
        self.last_search_stats = {"physical_requests": 1, "pages": 1, "returned_records": len(result),
                                  "response_bytes": len(json.dumps(result)), "complete": True}
        return result


class LinkedMedicationTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(cache, "CACHE_ENABLED", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def fetch(self, store):
        with patch("tools.resource_tools.get_fhir_client", return_value=store):
            return get_resources_by_patient_fhir_id("Medication", PATIENT)

    def test_batched_transitive_components_include_names_and_stop_cycles(self):
        store = Store([event()], medications=[medication("mix", ["a", "b"]), medication("a", name="Drug A"),
                                              medication("b", ["mix"], name="Drug B"), medication("unrelated")])
        result = self.fetch(store)
        self.assertEqual([record["id"] for record in result["Medication"]], ["a", "b", "mix"])
        self.assertEqual(store.queries[-1], "Medication?_id=a,b&_count=100")
        self.assertEqual(result["retrieval_metadata"]["physical_queries"], 4)
        self.assertEqual(result["retrieval_metadata"]["physical_requests"], 4)
        self.assertTrue(result["retrieval_metadata"]["complete"])
        self.assertFalse(any(query.startswith("Medication?patient=") for query in store.queries))

    def test_roots_from_administrations_are_included_and_other_patients_excluded(self):
        store = Store([event(patient="other", medication="unrelated")], [event(rtype="MedicationAdministration")],
                      [medication("mix"), medication("unrelated")])
        result = self.fetch(store)
        self.assertEqual([record["id"] for record in result["Medication"]], ["mix"])
        self.assertEqual(result["retrieval_metadata"]["excluded_other_patient_roots"], 1)

    def test_missing_dependency_is_incomplete_and_never_empty(self):
        store = Store([event()], medications=[medication("mix", ["missing"])])
        result = self.fetch(store)
        self.assertIn("error", result)
        self.assertFalse(result["retrieval_metadata"]["complete"])
        self.assertEqual(result["retrieval_metadata"]["unresolved_ids"], ["missing"])
        self.assertNotIn("Medication", result)

    def test_invalid_owner_id_external_reference_and_missing_subject_fail(self):
        malformed = [event(medication="bad,id"), {**event(), "subject": {}},
                     {**event(), "medicationReference": {"reference": "https://other.invalid/Medication/mix"}}]
        for root in malformed:
            with self.subTest(root=root):
                result = self.fetch(Store([root], medications=[medication("mix")]))
                self.assertIn("error", result)
                self.assertFalse(result["retrieval_metadata"]["complete"])

    def test_id_batch_is_bounded_and_never_fetches_unrequested_metadata(self):
        roots = [event(str(i), medication=f"m{i}") for i in range(205)]
        store = Store(roots, medications=[medication(f"m{i}") for i in range(205)])
        result = self.fetch(store)
        batches = [query for query in store.queries if query.startswith("Medication?_id=")]
        self.assertEqual(len(batches), 3)
        self.assertTrue(all(len(query.split("_id=")[1].split("&")[0].split(",")) <= 100 for query in batches))
        self.assertEqual(len(result["Medication"]), 205)

    def test_closure_bound_reports_unresolved_records(self):
        store = Store([event()], medications=[medication("mix", ["a"]), medication("a")])
        with patch("tools.resource_tools.MEDICATION_MAX_DEPTH", 1):
            result = self.fetch(store)
        self.assertIn("error", result)
        self.assertEqual(result["retrieval_metadata"]["unresolved_ids"], ["a"])

    def test_empty_patient_events_are_a_complete_empty_medication_search(self):
        store = Store()
        result = self.fetch(store)
        self.assertEqual(result["Medication"], [])
        self.assertTrue(result["retrieval_metadata"]["complete"])
        self.assertEqual(len(store.queries), 2)

    def test_get_all_delegates_medication_to_dependency_queries(self):
        store = Store([event()], medications=[medication("mix")])
        with patch("tools.resource_tools.get_fhir_client", return_value=store):
            result = get_all_resources_by_patient_fhir_id(PATIENT)
        self.assertEqual([r["id"] for r in result["Medication"]], ["mix"])
        self.assertFalse(any(query.startswith("Medication?patient=") for query in store.queries))

    def test_cached_result_reports_no_new_physical_requests(self):
        store = Store([event()], medications=[medication("mix")])
        with tempfile.TemporaryDirectory() as directory, patch.object(cache, "CACHE_ENABLED", True), \
                patch.object(cache, "CACHE_DIR", directory), patch("tools.resource_tools.get_fhir_client", return_value=store):
            first = get_resources_by_patient_fhir_id("Medication", PATIENT)
            second = get_resources_by_patient_fhir_id("Medication", PATIENT)
        self.assertEqual(first["retrieval_metadata"]["physical_requests"], 3)
        self.assertTrue(second["retrieval_metadata"]["cache_hit"])
        self.assertEqual(second["retrieval_metadata"]["physical_requests"], 0)
        self.assertEqual(second["retrieval_metadata"]["origin_physical_requests"], 3)


class ReachabilityTests(unittest.TestCase):
    def test_only_patient_rooted_transitive_ingredients_are_in_scope(self):
        held = {"MedicationRequest": [event()], "Medication": [medication("unrooted", ["evil"])]}
        toolbox = Toolbox(["get_resources_by_patient_fhir_id"], held, patient_fhir_id=PATIENT,
                          current_time="2150-12-31 23:59:00")
        rows = [medication("mix", ["a"]), medication("a", ["b"]), medication("b"),
                medication("unrelated"), medication("evil")]
        kept, excluded = toolbox._in_scope(rows)
        self.assertEqual([r["id"] for r in kept], ["mix", "a", "b"])
        self.assertEqual(excluded, {"unlinked_metadata": 2})

    def test_other_patient_and_future_held_roots_cannot_authorize_metadata(self):
        future = event("future", medication="future-med")
        future["effectiveDateTime"] = "2151-01-01T00:00:00"
        future["dispenseRequest"]["validityPeriod"]["start"] = "2151-01-01T00:00:00"
        group = event("group", medication="group-med")
        group["subject"] = {"reference": "Group/" + PATIENT}
        toolbox = Toolbox(["get_resources_by_patient_fhir_id"], {"MedicationRequest": [
            event(patient="other", medication="other-med"), future, group]}, patient_fhir_id=PATIENT,
            current_time="2150-12-31 23:59:00")
        kept, excluded = toolbox._in_scope([medication("other-med"), medication("future-med"), medication("group-med")])
        self.assertEqual(kept, [])
        self.assertEqual(excluded, {"unlinked_metadata": 3})

    def test_incomplete_metadata_reply_does_not_materialize_records(self):
        toolbox = Toolbox(["get_resources_by_patient_fhir_id"], {"MedicationRequest": [event()]},
                          patient_fhir_id=PATIENT)
        call = SimpleNamespace(id="fetch", function=SimpleNamespace(name="get_resources_by_patient_fhir_id",
                arguments=json.dumps({"resource_type": "Medication", "patient_fhir_id": PATIENT})))
        output = {"Medication": [medication("mix")], "retrieval_metadata": {"complete": False}}
        with patch("agent.framework.get_tool", return_value=lambda **_: output):
            result = json.loads(toolbox.call(call)["content"])
        self.assertIn("error", result)
        self.assertFalse(result["retrieval_metadata"]["complete"])
        self.assertNotIn("Medication", toolbox.resources)


class PhysicalSearchTests(unittest.TestCase):
    def test_http_pages_and_returned_resources_are_counted_without_urls(self):
        responses = [SimpleNamespace(raise_for_status=lambda: None, content=b"page-one",
                     json=lambda: {"entry": [{"resource": medication("a")}],
                                   "link": [{"relation": "next", "url": "https://example.invalid/next"}]}),
                     SimpleNamespace(raise_for_status=lambda: None, content=b"page-two",
                     json=lambda: {"entry": [{"resource": medication("b")}]} )]
        class Session:
            def get(self, url):
                return responses.pop(0)
        client = FHIRClient("test", "test", "test", "test", creds=Session())
        result = client.search_with_pagination("Medication?_id=a,b")
        self.assertEqual(len(result), 2)
        self.assertEqual(client.last_search_stats, {"physical_requests": 2, "pages": 2,
                         "returned_records": 2, "response_bytes": 16, "complete": True})


if __name__ == "__main__":
    unittest.main()
