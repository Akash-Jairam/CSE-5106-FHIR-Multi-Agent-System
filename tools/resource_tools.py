import json
import re
from .registry import tool_registry
from .cache import cached_tool
from fhir_client import get_fhir_client

supported_types = [
    "Patient", "Encounter", "Condition", "MedicationRequest", 
    "Procedure", "Observation", "MedicationAdministration",
    "Location", "Specimen", "Medication"
]
FHIR_ID_RE = re.compile(r"^[A-Za-z0-9.-]{1,64}$")


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

@cached_tool(version=2)
def get_resources_by_patient_fhir_id(resource_type: str, patient_fhir_id: str) -> dict:
    """Get FHIR resources by patient FHIR ID."""
    try:
        _validate_search(resource_type, patient_fhir_id)
        client = get_fhir_client()
        query = _patient_query(resource_type, patient_fhir_id)
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

@cached_tool(version=2)
def get_all_resources_by_patient_fhir_id(patient_fhir_id: str) -> dict:
    """Get all FHIR resources by patient FHIR ID."""
    try:
        _validate_search("Patient", patient_fhir_id)
        client = get_fhir_client()
        all_resources = []        
        for resource_type in supported_types:
            query = _patient_query(resource_type, patient_fhir_id)
            resources = client.search_with_pagination(query)
            all_resources.extend(resources)
        
        # Group by resource type
        resources_by_type = {}
        for resource in all_resources:
            rt = resource.get('resourceType', 'Unknown')
            if rt not in resources_by_type:
                resources_by_type[rt] = []
            resources_by_type[rt].append(resource)
        
        return resources_by_type
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
        "description": "Get FHIR resources by patient FHIR ID",
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
