"""Deterministic field check for cited FHIR evidence.

Shared by the tool-equipped single agent and, later, the verification agent.
For each citation {"resource_id", "field_path", "value"} it reports whether
the citation is valid (same rules as the evidence scorer: the resource
exists, the path resolves, the value matches) and flags common field
mistakes:

- time field: Observation `issued` is when a result was released, not when it
  was measured (`effectiveDateTime`); MedicationRequest `authoredOn` is the
  order time, not the prescription start (`dispenseRequest.validityPeriod.start`)
- unit: the unit of a cited `valueQuantity.value`, flagged if the question asks
  for a different unit
- encounter link: if the answer is scoped to an encounter, the cited resource
  must belong to it (or to an ICU stay that is part of it)
- current time: the cited resource is dated after the question's assumed
  current time
- patient: patient-owned evidence must link to the question's patient
"""
import re

from utils.answer_scorer import UNITS
from utils.evidence_scorer import check_citation, fetch_from_store, get_cited_resource, parse_citation, resolve_path
from .registry import tool_registry

QUESTION_UNIT_RE = re.compile(r"\bin\s+(" + "|".join(sorted(map(re.escape, UNITS), key=len, reverse=True)) + r")\b", re.I)
ENCOUNTER_SCOPE_RE = re.compile(r"\b(first|last|latest)\s+(hospital|icu)\s+(?:visit|encounter|stay)\b", re.I)


def _ref_id(reference):
    if isinstance(reference, str):
        return reference.rstrip("/").split("/")[-1]
    return (reference or {}).get("reference", "").rstrip("/").split("/")[-1]


def _wall_clock(value):
    return value[:19].replace("T", " ") if isinstance(value, str) else ""


def _resource_time(resource):
    for path in ("effectiveDateTime", "dispenseRequest.validityPeriod.start", "performedDateTime",
                 "performedPeriod.start", "period.start", "issued", "authoredOn"):
        values = resolve_path(resource, path)
        if values:
            return _wall_clock(values[0])
    return ""


def _unit_key(unit):
    return (unit or "").strip().lower().rstrip(".")


def _question_encounter(question, patient_fhir_id, resources, from_store, current_time):
    match = ENCOUNTER_SCOPE_RE.search(question or "")
    if not match or not patient_fhir_id:
        return None, []
    order, kind = match.groups()
    if from_store:
        from fhir_client import get_fhir_client
        encounters = get_fhir_client().search_with_pagination(f"Encounter?patient={patient_fhir_id}")
        resources.update({f"Encounter/{r['id']}": r for r in encounters})
    else:
        encounters = [r for r in resources.values() if r.get("resourceType") == "Encounter"]
    suffix = "encounter-hosp" if kind.lower() == "hospital" else "encounter-icu"
    encounters = [r for r in encounters
                  if _ref_id(r.get("subject")) == patient_fhir_id
                  and any(i.get("system", "").endswith(suffix) for i in r.get("identifier", []))
                  and _resource_time(r)
                  and (not current_time or _resource_time(r) <= current_time)]
    if not encounters:
        return None, [f"encounter_scope: no {kind.lower()} encounter establishes the requested scope"]
    encounters.sort(key=lambda r: (_resource_time(r), r["id"]))
    return encounters[0 if order.lower() == "first" else -1]["id"], []


def _check_one(citation, resources, question, current_time, encounter_id, patient_fhir_id):
    status = check_citation(citation, resources)
    result = {"citation": citation, "status": status, "warnings": []}
    parsed = parse_citation(citation)
    if not parsed or status in ("malformed", "missing_resource"):
        return result
    resource_type, rid, path, _ = parsed
    resource = get_cited_resource(resources, resource_type, rid)
    if resource is None:
        return result
    rtype = resource["resourceType"]
    if status == "value_mismatch":
        result["actual"] = resolve_path(resource, path)

    if patient_fhir_id and rtype not in ("Medication", "Location"):
        patient = rid if rtype == "Patient" else _ref_id(resource.get("subject") or resource.get("patient"))
        if patient != patient_fhir_id:
            result["warnings"].append(
                f"patient_scope: the resource belongs to Patient/{patient or '?'}, not Patient/{patient_fhir_id}")

    if rtype == "Observation" and path == "issued" and "effectiveDateTime" in resource:
        result["warnings"].append("time_field: `issued` is when the result was released; the measurement time is "
                                  f"`effectiveDateTime` ({resource['effectiveDateTime']})")
    start = resolve_path(resource, "dispenseRequest.validityPeriod.start")
    if rtype == "MedicationRequest" and path == "authoredOn" and start:
        result["warnings"].append("time_field: `authoredOn` is the order time; the prescription start time is "
                                  f"`dispenseRequest.validityPeriod.start` ({start[0]})")

    quantity_path = path.rsplit(".", 1)[0] if path.endswith((".value", ".unit", ".code")) else path
    if quantity_path.endswith("valueQuantity"):
        quantities = resolve_path(resource, quantity_path) or []
        quantity = next((q for q in quantities if isinstance(q, dict)), {})
        unit = quantity.get("unit") or quantity.get("code")
        result["unit"] = unit
        asked = QUESTION_UNIT_RE.search(question or "")
        if asked and unit:
            asked_key, cited_key = _unit_key(asked.group(1)), _unit_key(unit)
            # Evidence retains the original unit even when the answer converts it.
            same = asked_key == cited_key or (cited_key in UNITS and UNITS[asked_key] == UNITS[cited_key])
            if not same and cited_key in UNITS and UNITS[asked_key][0] == UNITS[cited_key][0]:
                result["conversion"] = {"source_unit": unit, "target_unit": asked.group(1),
                                        "factor": UNITS[cited_key][1] / UNITS[asked_key][1]}
            elif not same:
                result["warnings"].append(f"unit: the value is in {unit}, but the question asks for {asked.group(1)}")

    if encounter_id and rtype not in ("Patient", "Medication", "Location"):
        linked = _ref_id(resource.get("encounter") or resource.get("context"))
        parent = _ref_id((get_cited_resource(resources, "Encounter", linked) or {}).get("partOf"))
        if rtype == "Encounter":
            linked, parent = rid, _ref_id(resource.get("partOf"))
        if encounter_id not in (linked, parent):
            result["warnings"].append(f"encounter_link: the resource belongs to Encounter/{linked or '?'}, "
                                      f"not the scoped Encounter/{encounter_id}")

    time = _resource_time(resource)
    if not time:
        linked = _ref_id(resource.get("encounter") or resource.get("context"))
        time = _resource_time(get_cited_resource(resources, "Encounter", linked) or {})
    if current_time and time and time > current_time:
        result["warnings"].append(f"current_time: the resource is dated {time}, after the assumed current time {current_time}")
    return result


def check_fields(citations, question="", current_time=None, encounter_id=None, resources=None,
                 patient_fhir_id=None):
    """Check cited evidence against supplied original resources or the store.

    Patient scope and recognizable first/last encounter scopes are checked
    when question context is supplied. Compatible unit conversions are
    reported as metadata; they do not establish answer arithmetic.
    """
    citations = list(citations or [])
    from_store = resources is None
    encounter_id = _ref_id(encounter_id) if encounter_id else None
    patient_fhir_id = _ref_id(patient_fhir_id) if patient_fhir_id else None
    current_time = _wall_clock(current_time) if current_time else None
    resources = dict(resources or {})
    parsed = [p for p in map(parse_citation, citations) if p]
    refs = {p[:2] for p in parsed if get_cited_resource(resources, p[0], p[1]) is None}
    if refs:
        resources.update(fetch_from_store(refs))
    scoped, scope_issues = _question_encounter(question, patient_fhir_id, resources, from_store, current_time)
    if scoped:
        if encounter_id and encounter_id != scoped:
            scope_issues.append(f"encounter_scope: the question selects Encounter/{scoped}, not Encounter/{encounter_id}")
        encounter_id = scoped
    # Encounters also establish a date for resources such as Conditions.
    if encounter_id or current_time:
        cited = [r for p in parsed if (r := get_cited_resource(resources, p[0], p[1])) is not None]
        linked = {_ref_id(r.get("encounter") or r.get("context")) for r in cited}
        missing = {("Encounter", i) for i in linked if i and get_cited_resource(resources, "Encounter", i) is None}
        if missing:
            resources.update(fetch_from_store(missing))

    results = [_check_one(c, resources, question, current_time, encounter_id, patient_fhir_id)
               for c in citations]
    issues = scope_issues
    for i, r in enumerate(results):
        if r["status"] != "valid":
            issues.append(f"evidence[{i}] {r['status']}" + (f" (actual value: {r['actual']})" if "actual" in r else ""))
        issues.extend(f"evidence[{i}] {w}" for w in r["warnings"])
    return {"results": results, "issues": issues, "n_valid": sum(r["status"] == "valid" for r in results)}


tool_registry.register_tool("check_fields", check_fields, {
    "type": "function",
    "function": {
        "name": "check_fields",
        "description": ("Deterministically check cited FHIR evidence: whether each resource exists, the field path "
                        "resolves, and the value matches; also flags the wrong time field (issued vs "
                        "effectiveDateTime, authoredOn vs dispenseRequest.validityPeriod.start), unit "
                        "mismatches, resources outside a scoped encounter, and resources after the current time."),
        "parameters": {
            "type": "object",
            "properties": {
                "citations": {
                    "type": "array",
                    "description": "Evidence to check",
                    "items": {
                        "type": "object",
                        "properties": {
                            "resource_id": {"type": "string", "description": "ResourceType/id"},
                            "field_path": {"type": "string", "description": "e.g. valueQuantity.value"},
                            "value": {"description": "The value you read at field_path"}
                        },
                        "required": ["resource_id", "field_path", "value"]
                    }
                },
                "encounter_id": {"type": "string", "description": "Optional: the Encounter id the answer is scoped to"}
            },
            "required": ["citations"]
        }
    }
})
