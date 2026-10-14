import json
import re
from copy import deepcopy
from .registry import tool_registry
from .cache import cached_tool
from fhir_client import get_fhir_client

supported_types = [
    "Patient", "Encounter", "Condition", "MedicationRequest", 
    "Procedure", "Observation", "MedicationAdministration",
    "Location", "Specimen", "Medication"
]
FHIR_ID_RE = re.compile(r"^[A-Za-z0-9.-]{1,64}$")
MEDICATION_BATCH_SIZE = 100
MEDICATION_MAX_RESOURCES = 10_000
MEDICATION_MAX_DEPTH = 16


class _LinkedMedicationFailure(ValueError):
    def __init__(self, message, metadata):
        super().__init__(message)
        self.metadata = {**metadata, "complete": False}


def _linked_medications(client, patient_fhir_id, source_roots=None):
    """Resolve only Medication dependencies of patient-owned events, including mix ingredients.

    Medication has no patient ownership/search parameter. Google FHIR supports comma-separated
    _id values as OR; MIMIC medication mixes point to components through ingredient.itemReference.
    Missing or invalid dependencies are explicit incomplete retrievals, never empty evidence.
    """
    metadata = {"strategy": "patient_medication_dependency_closure_v3", "cache_hit": False,
                "physical_queries": 0, "physical_requests": 0, "physical_requests_known": True,
                "pages": 0, "response_bytes": 0, "returned_records": 0, "root_records": 0,
                "excluded_other_patient_roots": 0, "closure_depth": 0,
                "requested_medication_ids": 0, "unresolved_ids": [], "complete": False}

    def fail(message, unresolved=()):
        metadata["unresolved_ids"] = sorted(set(unresolved))
        raise _LinkedMedicationFailure(message, metadata)

    def search(query):
        metadata["physical_queries"] += 1
        try:
            records = client.search_with_pagination(query)
        except Exception as exc:
            stats = getattr(client, "last_search_stats", None)
            if isinstance(stats, dict):
                for key in ("physical_requests", "pages", "response_bytes"):
                    metadata[key] += stats.get(key, 0)
            fail(f"Linked Medication search failed ({type(exc).__name__})")
        stats = getattr(client, "last_search_stats", None)
        if isinstance(stats, dict) and isinstance(stats.get("physical_requests"), int):
            for key in ("physical_requests", "pages", "response_bytes"):
                metadata[key] += stats.get(key, 0)
            if stats.get("complete") is False:
                fail("Linked Medication search pagination was incomplete")
        else:
            metadata["physical_requests_known"] = False
        if not isinstance(records, list):
            fail("Linked Medication search returned an invalid resource list")
        metadata["returned_records"] += len(records)
        return records

    def reference_id(value, resource_type, label):
        reference = value.get("reference") if isinstance(value, dict) else None
        if not isinstance(reference, str):
            fail(f"Invalid {label} reference")
        local = f"{resource_type}/"
        base = getattr(client, "fhir_store_url", None)
        if reference.startswith(local):
            rid = reference[len(local):]
        elif isinstance(base, str) and reference.startswith(base.rstrip("/") + "/" + local):
            rid = reference[len(base.rstrip("/") + "/" + local):]
        else:
            fail(f"Unsupported or external {label} reference")
        if not FHIR_ID_RE.fullmatch(rid):
            fail(f"Invalid {label} resource ID")
        return rid

    roots = set()
    for resource_type in ("MedicationRequest", "MedicationAdministration"):
        events = (source_roots[resource_type] if source_roots is not None
                  else search(_patient_query(resource_type, patient_fhir_id)))
        for record in events:
            if (not isinstance(record, dict) or record.get("resourceType") != resource_type
                    or not isinstance(record.get("id"), str) or not FHIR_ID_RE.fullmatch(record["id"])):
                fail("Linked Medication search returned an invalid patient event")
            owner = reference_id(record.get("subject") or record.get("patient"), "Patient", "patient owner")
            if owner != patient_fhir_id:
                metadata["excluded_other_patient_roots"] += 1
                continue
            metadata["root_records"] += 1
            if "medicationReference" in record:
                roots.add(reference_id(record["medicationReference"], "Medication", "medication"))
    held = {}
    pending = roots
    seen = set()
    for depth in range(MEDICATION_MAX_DEPTH):
        if not pending:
            break
        if len(seen | pending) > MEDICATION_MAX_RESOURCES:
            fail("Linked Medication dependency closure exceeds the resource bound", pending)
        metadata["closure_depth"] = depth + 1
        requested = sorted(pending)
        metadata["requested_medication_ids"] += len(requested)
        seen.update(pending)
        pending = set()
        for offset in range(0, len(requested), MEDICATION_BATCH_SIZE):
            batch = requested[offset:offset + MEDICATION_BATCH_SIZE]
            returned = search("Medication?_id=" + ",".join(batch) + f"&_count={MEDICATION_BATCH_SIZE}")
            found = set()
            for record in returned:
                if (not isinstance(record, dict) or record.get("resourceType") != "Medication"
                        or not isinstance(record.get("id"), str) or record["id"] not in batch):
                    fail("Linked Medication _id search returned an unrequested or invalid resource", batch)
                rid = record["id"]
                if rid in held and held[rid] != record:
                    fail("Linked Medication search returned conflicting resource versions", [rid])
                held[rid] = deepcopy(record)
                found.add(rid)
            missing = set(batch) - found
            if missing:
                fail("Linked Medication dependency records are missing", missing)
        for rid in requested:
            ingredients = held[rid].get("ingredient", [])
            if not isinstance(ingredients, list):
                fail("Linked Medication has malformed ingredient metadata", [rid])
            for ingredient in ingredients:
                if not isinstance(ingredient, dict):
                    fail("Linked Medication has malformed ingredient metadata", [rid])
                if "itemReference" in ingredient:
                    target = reference_id(ingredient["itemReference"], "Medication", "ingredient")
                    if target not in seen:
                        pending.add(target)
    if pending:
        fail("Linked Medication dependency closure exceeds the depth bound", pending)
    metadata["complete"] = True
    metadata["medication_records"] = len(held)
    return {"Medication": [held[rid] for rid in sorted(held)], "retrieval_metadata": metadata}


def _validate_search(resource_type, identifier):
    if resource_type not in supported_types:
        raise ValueError("Unsupported FHIR resource type")
    if not isinstance(identifier, str) or not FHIR_ID_RE.fullmatch(identifier):
        raise ValueError("A non-empty FHIR resource ID is required")

def _patient_query(resource_type: str, patient_fhir_id: str) -> str:
    """Build a search for one patient's resources of a given type."""
    if not patient_fhir_id:
        return resource_type
    # Patient has no "patient" search parameter; the Cloud Healthcare API silently
    # ignores it and returns every patient, so filter by _id instead.
    if resource_type == "Patient":
        return f"Patient?_id={patient_fhir_id}"
    return f"{resource_type}?patient={patient_fhir_id}"


@cached_tool(version=3)  # Invalidate old broad searches and unvalidated IDs.
def get_resources_by_resource_id(resource_type: str, resource_id: str) -> dict:
    """Get FHIR resources by resource ID."""
    try:
        _validate_search(resource_type, resource_id)
        client = get_fhir_client()
        # "id" is not a FHIR search parameter; the Cloud Healthcare API ignores it and returns every
        # resource of the type, so search by _id
        query = f"{resource_type}?_id={resource_id}"
        resources = client.search_with_pagination(query)
        
        resources_by_type = {}
        for resource in resources:
            rt = resource.get('resourceType', 'Unknown')
            if rt not in resources_by_type:
                resources_by_type[rt] = []
            resources_by_type[rt].append(resource)
        
        return resources_by_type
    except Exception as e:
        return {"error": str(e)}

@cached_tool(version=3)
def get_resources_by_patient_fhir_id(resource_type: str, patient_fhir_id: str) -> dict:
    """Get FHIR resources by patient FHIR ID."""
    try:
        _validate_search(resource_type, patient_fhir_id)
        client = get_fhir_client()
        if resource_type == "Medication":
            return _linked_medications(client, patient_fhir_id)
        query = _patient_query(resource_type, patient_fhir_id)
        resources = client.search_with_pagination(query)
        
        resources_by_type = {}
        for resource in resources:
            rt = resource.get('resourceType', 'Unknown')
            if rt not in resources_by_type:
                resources_by_type[rt] = []
            resources_by_type[rt].append(resource)
        
        return resources_by_type
    except _LinkedMedicationFailure as e:
        return {"error": str(e), "retrieval_metadata": e.metadata}
    except Exception as e:
        return {"error": str(e)}

@cached_tool(version=3)
def get_all_resources_by_patient_fhir_id(patient_fhir_id: str) -> dict:
    """Get all FHIR resources by patient FHIR ID."""
    try:
        _validate_search("Patient", patient_fhir_id)
        client = get_fhir_client()
        all_resources = []        
        roots = {}
        prefix_stats = {"physical_queries": 0, "physical_requests": 0, "pages": 0,
                        "response_bytes": 0, "returned_records": 0, "physical_requests_known": True}
        for resource_type in supported_types:
            if resource_type == "Medication":
                medication = _linked_medications(client, patient_fhir_id, source_roots=roots)
                for key in ("physical_queries", "physical_requests", "pages", "response_bytes", "returned_records"):
                    medication["retrieval_metadata"][key] += prefix_stats[key]
                medication["retrieval_metadata"]["physical_requests_known"] &= prefix_stats["physical_requests_known"]
                all_resources.extend(medication["Medication"])
                continue
            query = _patient_query(resource_type, patient_fhir_id)
            resources = client.search_with_pagination(query)
            prefix_stats["physical_queries"] += 1
            prefix_stats["returned_records"] += len(resources)
            stats = getattr(client, "last_search_stats", None)
            if isinstance(stats, dict) and isinstance(stats.get("physical_requests"), int):
                for key in ("physical_requests", "pages", "response_bytes"):
                    prefix_stats[key] += stats.get(key, 0)
            else:
                prefix_stats["physical_requests_known"] = False
            if resource_type in ("MedicationRequest", "MedicationAdministration"):
                roots[resource_type] = resources
            all_resources.extend(resources)
        
        # Group by resource type
        resources_by_type = {}
        for resource in all_resources:
            rt = resource.get('resourceType', 'Unknown')
            if rt not in resources_by_type:
                resources_by_type[rt] = []
            resources_by_type[rt].append(resource)
        
        return {**resources_by_type, "retrieval_metadata": medication["retrieval_metadata"]}
    except _LinkedMedicationFailure as e:
        return {"error": str(e), "retrieval_metadata": e.metadata}
    except Exception as e:
        return {"error": str(e)}


tool_registry.register_tool("get_resources_by_resource_id", get_resources_by_resource_id, {
    "type": "function",
    "function": {
        "name": "get_resources_by_resource_id",
        "description": "Get FHIR resources by resource ID",
        "parameters": {
            "type": "object",
            "properties": {
                "resource_type": {"type": "string", "description": "FHIR resource type"},
                "resource_id": {"type": "string", "description": "FHIR resource ID"}
            },
            "required": ["resource_type", "resource_id"]
        }
    }
})

tool_registry.register_tool("get_resources_by_patient_fhir_id", get_resources_by_patient_fhir_id, {
    "type": "function",
    "function": {
        "name": "get_resources_by_patient_fhir_id",
        "description": "Get a patient's FHIR resources of one type. Medication resolves that patient's MedicationRequest/MedicationAdministration references and all ingredient component Medication metadata in batched searches; it does not search all store medications. A complete Medication result includes all linked mix components in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "resource_type": {"type": "string", "description": "FHIR resource type"},
                "patient_fhir_id": {"type": "string", "description": "Patient FHIR ID"}
            },
            "required": ["resource_type", "patient_fhir_id"]
        }
    }
})

tool_registry.register_tool("get_all_resources_by_patient_fhir_id", get_all_resources_by_patient_fhir_id, {
    "type": "function",
    "function": {
        "name": "get_all_resources_by_patient_fhir_id",
        "description": "Get all FHIR resources by patient FHIR ID",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_fhir_id": {"type": "string", "description": "Patient FHIR ID"}
            },
            "required": ["patient_fhir_id"]
        }
    }
})
