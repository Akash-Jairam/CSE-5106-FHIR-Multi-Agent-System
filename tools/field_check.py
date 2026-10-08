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
"""
import re

from utils.answer_scorer import UNITS
from utils.evidence_scorer import check_citation, fetch_from_store, parse_citation, resolve_path
from .registry import tool_registry

QUESTION_UNIT_RE = re.compile(r"\bin\s+(" + "|".join(sorted(map(re.escape, UNITS), key=len, reverse=True)) + r")\b", re.I)


def _ref_id(reference):
    return (reference or {}).get("reference", "").split("/")[-1]


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


def _check_one(citation, resources, question, current_time, encounter_id):
    status = check_citation(citation, resources)
    result = {"citation": citation, "status": status, "warnings": []}
    parsed = parse_citation(citation)
    if not parsed or status in ("malformed", "missing_resource"):
        return result
    _, rid, path, _ = parsed
    resource = resources[rid]
    rtype = resource["resourceType"]
    if status == "value_mismatch":
        result["actual"] = resolve_path(resource, path)

    if rtype == "Observation" and path == "issued" and "effectiveDateTime" in resource:
        result["warnings"].append("time_field: `issued` is when the result was released; the measurement time is "
                                  f"`effectiveDateTime` ({resource['effectiveDateTime']})")
    start = resolve_path(resource, "dispenseRequest.validityPeriod.start")
    if rtype == "MedicationRequest" and path == "authoredOn" and start:
        result["warnings"].append("time_field: `authoredOn` is the order time; the prescription start time is "
                                  f"`dispenseRequest.validityPeriod.start` ({start[0]})")

    if path.startswith("valueQuantity"):
        unit = (resource.get("valueQuantity") or {}).get("unit")
        result["unit"] = unit
        asked = QUESTION_UNIT_RE.search(question or "")
        if asked and unit:
            asked_key, cited_key = _unit_key(asked.group(1)), _unit_key(unit)
            # Same unit, or a synonym with the same dimension and factor ("kg" / "kilograms")
            same = asked_key == cited_key or (cited_key in UNITS and UNITS[asked_key] == UNITS[cited_key])
            if not same:
                result["warnings"].append(f"unit: the value is in {unit}, but the question asks for {asked.group(1)}")

    if encounter_id and rtype not in ("Patient", "Medication", "Location"):
        linked = _ref_id(resource.get("encounter") or resource.get("context"))
        parent = _ref_id((resources.get(linked) or {}).get("partOf"))
        if rtype == "Encounter":
            linked, parent = rid, _ref_id(resource.get("partOf"))
        if encounter_id not in (linked, parent):
            result["warnings"].append(f"encounter_link: the resource belongs to Encounter/{linked or '?'}, "
                                      f"not the scoped Encounter/{encounter_id}")

    time = _resource_time(resource)
    if current_time and time and time > current_time:
        result["warnings"].append(f"current_time: the resource is dated {time}, after the assumed current time {current_time}")
    return result


def check_fields(citations, question="", current_time=None, encounter_id=None, resources=None):
    """Check cited evidence. `resources` ({id: resource}) is used first; anything
    missing is fetched from the FHIR store. Returns per-citation results and a
    list of `issues` (invalid citations and warnings)."""
    citations = list(citations or [])
    resources = dict(resources or {})
    parsed = [p for p in map(parse_citation, citations) if p]
    refs = {p[:2] for p in parsed if p[1] not in resources}
    if refs:
        resources.update(fetch_from_store(refs))
    # Encounters the cited resources point to, for the encounter-link check
    if encounter_id:
        cited = [resources[p[1]] for p in parsed if p[1] in resources]
        linked = {_ref_id(r.get("encounter") or r.get("context")) for r in cited}
        missing = {("Encounter", i) for i in linked if i and i not in resources}
        if missing:
            resources.update(fetch_from_store(missing))

    results = [_check_one(c, resources, question, current_time, encounter_id) for c in citations]
    issues = []
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
