"""Shared packet regressions discovered during the Phase 2 retrieval review."""
import unittest

from tools.starting_packet import _resource_time, build_starting_packet


def prescription(identifier, ordered, started):
    return {"resourceType": "MedicationRequest", "id": identifier,
            "subject": {"reference": "Patient/p"}, "authoredOn": ordered,
            "dispenseRequest": {"validityPeriod": {"start": started}},
            "medicationCodeableConcept": {"coding": [{"display": "Heparin"}]}}


class PrescriptionPacketTests(unittest.TestCase):
    def packet(self, *records):
        return build_starting_packet("When was heparin prescribed?", {
            "Patient": [{"resourceType": "Patient", "id": "p"}],
            "MedicationRequest": list(records)}, "p", current_time="2150-12-31 23:59:00")

    def test_prescription_clock_prefers_start_over_order_time(self):
        record = prescription("a", "2150-01-01T08:00:00", "2150-02-01T08:00:00")
        self.assertEqual(_resource_time(record), "2150-02-01T08:00:00")

    def test_future_prescription_start_is_excluded(self):
        packet = self.packet(prescription("a", "2150-01-01T08:00:00", "2151-01-01T08:00:00"))
        self.assertNotIn("MedicationRequest", packet["resource_ids"])
        self.assertEqual(packet["after_current_time"], 1)

    def test_prescriptions_are_rendered_in_actual_start_order(self):
        packet = self.packet(
            prescription("later", "2150-01-01T08:00:00", "2150-02-01T08:00:00"),
            prescription("earlier", "2150-01-02T08:00:00", "2150-01-03T08:00:00"))
        self.assertEqual(packet["resource_ids"]["MedicationRequest"], ["earlier", "later"])


if __name__ == "__main__":
    unittest.main()
