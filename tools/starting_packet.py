"""Deterministic starting-packet retriever.

Builds a token-bounded excerpt of one patient's FHIR record for a question,
with no LLM calls and no access to gold answers:

1. Context: the Patient resource, the patient's hospital and ICU Encounters,
   and the Locations they reference. Encounters use at most CONTEXT_SHARE of
   the budget.
2. Candidates: resources from the sources the questions are about (lab,
   chart, output, and microbiology Observations; MedicationRequests; ICD
   Procedures; Conditions), grouped by label (e.g. the lab name "Hemoglobin"
   or the drug name "Heparin"). Each label is scored by IDF-weighted keyword
   overlap with the question, plus a bonus when the question names the
   label's kind ("lab", "prescribed", "procedure", ...). Labels with score 0
   are left out; ties go to the label with the most recent resource.
3. Truncation: labels are added best-first. Within a label (and among
   Encounters) resources are taken latest, earliest, second latest, ..., so
   both ends of the timeline survive; MedicationRequests bring their
   Medication along. Filling stops at the first resource that would exceed
   `max_tokens`, and what was taken is shown in time order.

Resources are rendered as pruned JSON: keys that carry no answer (meta,
subject, code-system URLs, ...) and keys repeated in the group header (code,
category, ...) are dropped, but list items never are, so a field path in the
packet (e.g. "valueQuantity.value") is valid on the full resource in the store.
"""
import json
import math
import re
from collections import defaultdict

import tiktoken

from .resource_tools import get_all_resources_by_patient_fhir_id

# Chosen on the dev split: mean gold recall 73% at 4k, 90% at 8k, 97% at 16k
PACKET_MAX_TOKENS = 8000
CONTEXT_SHARE = 0.5
KIND_BONUS = 1.0

ENCODING = tiktoken.get_encoding("cl100k_base")

CANDIDATE_KINDS = ("lab", "vital", "output", "micro", "medication", "procedure", "condition")

# Words that name a kind of resource rather than a specific label
KIND_HINTS = {
    "medication": ("prescri", "medication", "drug", "dose", "route", "medicine"),
    "lab": ("lab",),
    "vital": ("vital",),
    "output": ("output",),
    "micro": ("microbio", "specimen", "organism", "culture"),
    "procedure": ("procedure", "undergo", "underwent"),
    "condition": ("diagnos",),
}

STOPWORDS = set("""
a an and any are as at be been by can could did do does during each ever for
from had has have how i in is it its last first second third me of on or
patient patients s since than that the their them there this time times to
was were what when which who with within year years month months day days
today week tell show list give get number many much value values measured
hospital visit visits encounter icu stay current currently ago
after before until between also find total maximum minimum average max min
avg mean change difference compared greater less more higher lower same
length out take go went done observed found name type received receive
""".split())

DROP_KEYS = {"meta", "text", "subject", "managingOrganization", "assigner",
             "referenceRange", "extension", "communication"}
# Already stated by the group header, so dropped from each grouped resource
GROUPED_DROP_KEYS = ("resourceType", "code", "category", "status", "intent")
# Encounters are always in the packet, so keep only the fields questions use
ENCOUNTER_KEYS = ("resourceType", "id", "identifier", "class", "period", "partOf",
                  "hospitalization", "location")


def _is_hint(word):
    return any(word.startswith(h) for hints in KIND_HINTS.values() for h in hints)


def _tokens(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w
            for w in words if not w.isdigit() and w not in STOPWORDS]


def count_tokens(text):
    return len(ENCODING.encode(text))


def _display(codeable):
    codings = (codeable or {}).get("coding", [])
    return next((c.get("display") or c.get("code") for c in codings if c.get("display") or c.get("code")), "")


def _code_system(resource):
    codings = resource.get("code", {}).get("coding", [])
    return codings[0].get("system", "") if codings else ""


def _ref_id(reference):
    return (reference or {}).get("reference", "").split("/")[-1]


def _resource_kind(resource):
    rtype = resource["resourceType"]
    if rtype == "MedicationRequest":
        return "medication"
    if rtype == "Procedure":
        return "procedure" if "procedure-icd" in _code_system(resource) else "other"
    if rtype != "Observation":
        return rtype.lower()
    system = _code_system(resource)
    category = _display(resource.get("category", [{}])[0])
    if system.endswith("mimic-d-labitems"):
        return "lab"
    if "microbiology" in system:
        return "micro"
    if category == "Output":
        return "output"
    if system.endswith("chartevents-d-items"):
        return "vital"
    return "other"


def _medication_name(medication):
    for ident in medication.get("identifier", []):
        if ident.get("system", "").endswith(("mimic-medication-name", "medication-mix")):
            return ident.get("value", "")
    return _display(medication.get("code"))


def _resource_time(resource):
    for path in (("effectiveDateTime",), ("authoredOn",), ("performedDateTime",),
                 ("performedPeriod", "start"), ("issued",), ("period", "start")):
        value = resource
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value:
            return value
    return ""


def _time_order(resource):
    # Ties broken by id so the order does not depend on how the record was fetched
    return _resource_time(resource), resource["id"]


def _label(resource, by_id):
    """Human-readable label used for keyword matching and grouping."""
    rtype = resource["resourceType"]
    if rtype == "MedicationRequest":
        medication = by_id.get(_ref_id(resource.get("medicationReference")), {})
        name = _medication_name(medication) or _display(resource.get("medicationCodeableConcept"))
        routes = {_display(d.get("route")) for d in resource.get("dosageInstruction", [])} - {""}
        return " / ".join([name] + sorted(routes))
    label = _display(resource.get("code"))
    if _resource_kind(resource) == "micro":
        # Micro tests are asked about by specimen type ("blood culture"); organisms
        # reach their specimen through the test they were derived from
        source = resource
        if "specimen" not in source and source.get("derivedFrom"):
            source = by_id.get(_ref_id(source["derivedFrom"][0]), {})
        specimen = by_id.get(_ref_id(source.get("specimen")), {})
        label = f"{label} / {_display(specimen.get('type'))}".rstrip(" /")
    return label


def _prune(value, in_identifier=False):
    if isinstance(value, dict):
        # Code-system URLs are noise, but an identifier's system says what it
        # identifies (e.g. a hospital vs ICU encounter), so it is kept
        coded = "code" in value and not in_identifier
        return {k: _prune(v, in_identifier or k == "identifier") for k, v in value.items()
                if k not in DROP_KEYS and not (coded and k == "system")}
    if isinstance(value, list):
        return [_prune(v, in_identifier) for v in value]
    return value


def render_resource(resource, grouped=False):
    pruned = _prune(resource)
    if grouped:
        for key in GROUPED_DROP_KEYS:
            pruned.pop(key, None)
    if resource["resourceType"] == "Encounter":
        pruned = {k: pruned[k] for k in ENCOUNTER_KEYS if k in pruned}
        for ident in pruned.get("identifier", []):
            ident.pop("use", None)
    elif resource["resourceType"] != "Medication":
        pruned.pop("identifier", None)
    # Sorted keys so packets are identical whether the record came from the store or NDJSON
    return json.dumps(pruned, separators=(",", ":"), ensure_ascii=False, sort_keys=True)


def _is_ed_encounter(encounter):
    # ED stays come from the ED module, not the admissions/icustays tables questions are built on
    return any(i.get("system", "").endswith("encounter-ed") for i in encounter.get("identifier", []))


def _patient_resources(record, patient_fhir_id):
    """Split `record` into the patient's Patient resource, their own resources,
    and an id lookup over everything (used to resolve Medication/Location refs).

    The store's patient search can return other patients' Patient, Medication,
    and Location resources (those types have no `patient` search parameter).
    """
    resources = [r for rtype, items in record.items() if isinstance(items, list) for r in items]
    by_id = {r["id"]: r for r in resources}
    own = [r for r in resources
           if r["resourceType"] not in ("Patient", "Medication", "Location")
           and _ref_id(r.get("subject")) == patient_fhir_id]
    patient = [r for r in resources if r["resourceType"] == "Patient" and r["id"] == patient_fhir_id]
    return patient, own, by_id


def score_labels(question, labels):
    """Score each (kind, label) by IDF-weighted overlap with the question."""
    # Kind words ("lab", "medications") only trigger the kind bonus; matching
    # them against labels would pull in e.g. "All Medications Tolerated"
    q_terms = {t for t in _tokens(question) if not _is_hint(t)}
    q_text = question.lower()
    hinted = {kind for kind, hints in KIND_HINTS.items() if any(h in q_text for h in hints)}
    label_terms = {key: set(_tokens(key[1])) for key in labels}
    df = defaultdict(int)
    for terms in label_terms.values():
        for t in terms:
            df[t] += 1
    n = len(labels)
    scores = {}
    for key, terms in label_terms.items():
        score = sum(math.log(1 + n / df[t]) for t in q_terms & terms)
        if key[0] in hinted:
            score += KIND_BONUS
        scores[key] = score
    return scores


def _ends_first(items):
    """Order items latest, earliest, second latest, second earliest, ...

    Questions ask about both the first and the last events in a scope, so a
    truncated group keeps both ends and loses the middle.
    """
    ordered = sorted(items, key=_time_order)
    result = []
    while ordered:
        result.append(ordered.pop())
        if ordered:
            result.append(ordered.pop(0))
    return result


def _linked(resource, by_id):
    """Resources needed to read `resource`: an Encounter's Locations, or a
    MedicationRequest's Medication (plus its ingredients if it is a mix)."""
    if resource["resourceType"] == "Encounter":
        refs = [loc.get("location") for loc in resource.get("location", [])]
    else:
        medication = by_id.get(_ref_id(resource.get("medicationReference")))
        if not medication:
            return []
        refs = [resource["medicationReference"]] + [i.get("itemReference") for i in medication.get("ingredient", [])]
    ids = dict.fromkeys(_ref_id(r) for r in refs)
    return [by_id[i] for i in ids if i in by_id]


def build_starting_packet(question, record, patient_fhir_id, max_tokens=PACKET_MAX_TOKENS):
    """Build the starting packet for `question` from one patient's `record`.

    `record` maps resource type -> list of resources, as returned by
    `get_all_resources_by_patient_fhir_id`. Returns a dict with the packet
    `text`, the included `resource_ids` (by type, in packet order),
    `n_tokens`, whether any matching resource was `truncated`, and the
    number of `omitted_encounters`.
    """
    patient, own, by_id = _patient_resources(record, patient_fhir_id)
    encounters = [r for r in own if r["resourceType"] == "Encounter" and not _is_ed_encounter(r)]

    groups = defaultdict(list)
    for r in own:
        kind = _resource_kind(r)
        if kind in CANDIDATE_KINDS:
            groups[(kind, _label(r, by_id))].append(r)
    scores = score_labels(question, groups)
    latest = {k: max(_resource_time(r) for r in items) for k, items in groups.items()}
    ranked = sorted((k for k in groups if scores[k] > 0),
                    key=lambda k: (scores[k], latest[k], k), reverse=True)

    lines = [f"Starting packet for patient {patient_fhir_id}. Resources are pruned JSON; "
             "field paths match the full resources in the FHIR store."]
    resource_ids = defaultdict(list)
    used = count_tokens(lines[0])

    def add_section(header, candidates, limit, grouped):
        """Take `candidates` in order, with the resources they link to, until the
        next one would push the packet past `limit` tokens; render the taken ones
        in time order. Returns how many candidates were taken."""
        nonlocal used
        spent = count_tokens(header(len(candidates))) + 1
        included = {i for ids in resource_ids.values() for i in ids}
        chosen, linked = [], []
        for r in candidates:
            new = [x for x in _linked(r, by_id) if x["id"] not in included]
            cost = count_tokens(render_resource(r, grouped)) + 1 + sum(count_tokens(render_resource(x)) + 1 for x in new)
            if used + spent + cost > limit:
                break
            chosen.append(r)
            linked.extend(new)
            included.update(x["id"] for x in new)
            spent += cost
        if chosen:
            lines.append(header(len(chosen)))
            for x in linked:
                lines.append(render_resource(x))
                resource_ids[x["resourceType"]].append(x["id"])
            for r in sorted(chosen, key=_time_order):
                lines.append(render_resource(r, grouped))
                resource_ids[r["resourceType"]].append(r["id"])
            used += spent
        return len(chosen)

    add_section(lambda n: "## Patient", patient, max_tokens, grouped=False)
    shown_encounters = add_section(
        lambda n: f"## Hospital and ICU encounters ({n} of {len(encounters)}) and their Locations",
        _ends_first(encounters), min(max_tokens, used + int(max_tokens * CONTEXT_SHARE)), grouped=False)

    truncated = False
    for kind, label in ranked:
        items = groups[(kind, label)]
        def header(n, items=items, kind=kind, label=label):
            kept = "" if n == len(items) else "; earliest and latest kept"
            return f"## {items[0]['resourceType']} [{kind}] {label} ({n} of {len(items)}{kept})"

        if add_section(header, _ends_first(items), max_tokens, grouped=True) < len(items):
            truncated = True
            break

    text = "\n".join(lines)
    return {
        "text": text,
        "resource_ids": dict(resource_ids),
        "n_tokens": count_tokens(text),
        "truncated": truncated,
        "omitted_encounters": len(encounters) - shown_encounters,
    }


def get_starting_packet(question, patient_fhir_id, max_tokens=PACKET_MAX_TOKENS):
    """Fetch the patient's record from the FHIR store and build the starting packet."""
    record = get_all_resources_by_patient_fhir_id(patient_fhir_id=patient_fhir_id)
    if "error" in record:
        raise RuntimeError(f"Could not fetch record for {patient_fhir_id}: {record['error']}")
    return build_starting_packet(question, record, patient_fhir_id, max_tokens)
