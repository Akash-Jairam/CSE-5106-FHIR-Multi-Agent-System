"""Missing-evidence perturbations (P2-18): starting packets with one resource held back.

manifests/perturbations.json lists, for each perturbed question, one
(resourceType, id) to leave out of its starting packet. The FHIR store is
not changed, so a system can recover the evidence by fetching it.
"""
import json

from .starting_packet import PACKET_MAX_TOKENS, build_starting_packet

DEFAULT_PATH = "manifests/perturbations.json"


def without_resources(record, removed):
    """`record` (resource type -> resources) minus the (resourceType, id) pairs in `removed`."""
    removed = {tuple(pair) for pair in removed}
    return {rtype: [r for r in items if (r.get("resourceType", rtype), r.get("id")) not in removed]
            for rtype, items in record.items()}


def load_perturbations(path=DEFAULT_PATH):
    """{question_id: [(resourceType, id), ...]} from a perturbation manifest."""
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    return {p["question_id"]: [(p["removed"]["resource_type"], p["removed"]["id"])]
            for p in manifest["perturbations"]}


def get_perturbed_starting_packet(question, patient_fhir_id, removed, max_tokens=PACKET_MAX_TOKENS,
                                  current_time=None):
    """The starting packet built as if the `removed` resources were not in the patient's record."""
    from .resource_tools import get_all_resources_by_patient_fhir_id

    record = get_all_resources_by_patient_fhir_id(patient_fhir_id=patient_fhir_id)
    if "error" in record:
        raise RuntimeError(f"Could not fetch record for {patient_fhir_id}: {record['error']}")
    return build_starting_packet(question, without_resources(record, removed), patient_fhir_id,
                                 max_tokens, current_time)
