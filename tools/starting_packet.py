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
If the question assumes a current time ("Assume the current time is
2133-12-31 23:59:00."), resources and encounters dated after it are left out
first: the record continues past that time, and keeping the latest readings
would otherwise keep future ones.

3. Truncation: labels are added best-first. Within a label (and among
   Encounters) resources are taken latest, earliest, second latest, ..., so
   both ends of the timeline survive; MedicationRequests bring their
   Medication along. Filling stops at the first resource that would exceed
   `max_tokens`, and what was taken is shown in time order.

Resources are rendered as pruned JSON: keys that carry no answer (meta,
subject, code-system URLs, ...) and repeated group metadata (category,
status, ...) are dropped. Codes remain because organism names can be answer
fields. List items never are dropped, so a field path in the
packet (e.g. "valueQuantity.value") is valid on the full resource in the store.
"""
import json
import math
import re
from collections import defaultdict

import tiktoken

# Fixed input cap for the reviewed protocol; do not optimize it using gold IDs.
PACKET_MAX_TOKENS = 8000
PACKET_VERSION = 3
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
GROUPED_DROP_KEYS = ("resourceType", "category", "status", "intent")
# Encounters are always in the packet, so keep only the fields questions use
ENCOUNTER_KEYS = ("resourceType", "id", "identifier", "class", "period", "partOf",
                  "priority", "hospitalization", "location")


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


def _reference_key(resource):
    return f"{resource['resourceType']}/{resource['id']}"


def _resolve(reference, by_id):
    # IDs are unique within a FHIR type, not across the entire store.
    reference = (reference or {}).get("reference", "")
    return by_id.get("/".join(reference.split("/")[-2:]), {})


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
    if resource.get("resourceType") == "MedicationRequest":
        start = (resource.get("dispenseRequest") or {}).get("validityPeriod", {}).get("start")
        if start:
            return start
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
        medication = _resolve(resource.get("medicationReference"), by_id)
        name = _medication_name(medication) or _display(resource.get("medicationCodeableConcept"))
        routes = {_display(d.get("route")) for d in resource.get("dosageInstruction", [])} - {""}
        return " / ".join([name] + sorted(routes))
    label = _display(resource.get("code"))
    if _resource_kind(resource) == "micro":
        # Micro tests are asked about by specimen type ("blood culture"); organisms
        # reach their specimen through the test they were derived from
        source = resource
        if "specimen" not in source and source.get("derivedFrom"):
            source = _resolve(source["derivedFrom"][0], by_id)
        specimen = _resolve(source.get("specimen"), by_id)
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


CURRENT_TIME_RE = re.compile(r"current time is (\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?)")


def parse_current_time(text):
    """The assumed current time in a question's assumption/context, as "YYYY-MM-DD HH:MM:SS"."""
    match = CURRENT_TIME_RE.search(text or "")
    if not match:
        return None
    value = match.group(1)
    return value if len(value) == 19 else value + ":00"


def _wall_clock(resource, by_id):
    """Local wall-clock time of a resource ("YYYY-MM-DD HH:MM:SS"); Conditions use their encounter's start."""
    time = _resource_time(resource)
    if not time and resource.get("encounter"):
        time = _resource_time(_resolve(resource["encounter"], by_id))
    return time[:19].replace("T", " ")


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
    by_id = {_reference_key(r): r for r in resources}
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
        score = sum(math.log(1 + n / df[t]) for t in sorted(q_terms & terms))
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
    """Encounter/location, specimen, and medication dependencies of evidence."""
    # Retain parent visits and microbiology tests/specimens even when their
    # up-front context was truncated. Guard against cyclic record references.
    linked, pending = {}, [resource]
    visited = {_reference_key(resource)}
    patient_id = _ref_id(resource.get("subject"))
    while pending:
        source = pending.pop()
        refs = [source.get("encounter"), source.get("specimen"), source.get("partOf"),
                source.get("medicationReference")]
        refs.extend(source.get("derivedFrom", []))
        if source["resourceType"] == "Encounter":
            refs.extend(loc.get("location") for loc in source.get("location", []))
        if source["resourceType"] == "Medication":
            refs.extend(i.get("itemReference") for i in source.get("ingredient", []))
        for ref in refs:
            resolved = _resolve(ref, by_id)
            if not resolved or _reference_key(resolved) in visited:
                continue
            subject = _ref_id(resolved.get("subject"))
            if patient_id and subject and subject != patient_id:
                continue
            key = _reference_key(resolved)
            visited.add(key)
            linked[key] = resolved
            pending.append(resolved)
    return [linked[key] for key in sorted(linked)]


def build_starting_packet(question, record, patient_fhir_id, max_tokens=PACKET_MAX_TOKENS, current_time=None):
    """Build the starting packet for `question` from one patient's `record`.

    `record` maps resource type -> list of resources, as returned by
    `get_all_resources_by_patient_fhir_id`. Returns a dict with the packet
    `text`, the included `resource_ids` and exactly rendered `resources`
    (by type, in packet order),
    `n_tokens`, whether any matching resource was `truncated`, the number
    of `omitted_encounters`, and how many resources were `after_current_time`.
    `current_time` ("YYYY-MM-DD HH:MM:SS", see parse_current_time) excludes
    everything dated after it.
    """
    patient, own, by_id = _patient_resources(record, patient_fhir_id)
    n_own = len(own)
    if current_time:
        own = [r for r in own if not _wall_clock(r, by_id) > current_time]
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
             "field paths match the full resources in the FHIR store."
             + (f" Resources dated after the current time ({current_time}) are not included." if current_time else "")]
    resource_ids = defaultdict(list)
    rendered_resources = defaultdict(list)
    used = count_tokens(lines[0])
    if max_tokens < used:
        raise ValueError(f"Packet budget must fit its header ({used} tokens)")

    def render_lines(chosen, linked, header, grouped):
        return [header(len(chosen))] + [render_resource(r) for r in linked] + [
            render_resource(r, grouped) for r in sorted(chosen, key=_time_order)]

    def add_section(header, candidates, limit, grouped):
        """Take `candidates` in order, with the resources they link to, until the
        next one would push the packet past `limit` tokens; render the taken ones
        in time order. Returns how many candidates were taken."""
        nonlocal used
        included = {f"{rtype}/{i}" for rtype, ids in resource_ids.items() for i in ids}
        chosen, linked, already_shown = [], [], 0
        for r in candidates:
            if _reference_key(r) in included:
                already_shown += 1
                continue
            new = [x for x in _linked(r, by_id) if _reference_key(x) not in included]
            tentative = render_lines(chosen + [r], linked + new, header, grouped)
            if count_tokens("\n".join(lines + tentative)) > limit:
                break
            chosen.append(r)
            linked.extend(new)
            included.update(_reference_key(x) for x in [r] + new)
        if chosen:
            lines.append(header(len(chosen)))
            for x in linked:
                lines.append(render_resource(x))
                resource_ids[x["resourceType"]].append(x["id"])
                rendered_resources[x["resourceType"]].append(json.loads(lines[-1]))
            for r in sorted(chosen, key=_time_order):
                lines.append(render_resource(r, grouped))
                resource_ids[r["resourceType"]].append(r["id"])
                rendered_resources[r["resourceType"]].append(json.loads(lines[-1]))
            used = count_tokens("\n".join(lines))
        return len(chosen) + already_shown

    add_section(lambda n: "## Patient", patient, max_tokens, grouped=False)
    add_section(
        lambda n: f"## Hospital and ICU encounters ({n} of {len(encounters)}) and their Locations",
        _ends_first(encounters), min(max_tokens, used + int(max_tokens * CONTEXT_SHARE)), grouped=False)

    truncated = False
    for kind, label in ranked:
        items = groups[(kind, label)]
        def header(n, items=items, kind=kind, label=label):
            kept = "" if n == len(items) else "; truncated"
            return f"## {items[0]['resourceType']} [{kind}] {label} ({n} of {len(items)}{kept})"

        if add_section(header, _ends_first(items), max_tokens, grouped=True) < len(items):
            truncated = True
            break

    text = "\n".join(lines)
    return {
        "packet_version": PACKET_VERSION,
        "max_tokens": max_tokens,
        "text": text,
        "resource_ids": dict(resource_ids),
        "resources": dict(rendered_resources),
        "n_tokens": count_tokens(text),
        "truncated": truncated,
        "omitted_encounters": len({r["id"] for r in encounters} - set(resource_ids.get("Encounter", []))),
        "after_current_time": n_own - len(own),
    }


def get_starting_packet(question, patient_fhir_id, max_tokens=PACKET_MAX_TOKENS, current_time=None):
    """Fetch the patient's record from the FHIR store and build the starting packet."""
    from .resource_tools import get_all_resources_by_patient_fhir_id

    record = get_all_resources_by_patient_fhir_id(patient_fhir_id=patient_fhir_id)
    if "error" in record:
        raise RuntimeError(f"Could not fetch record for {patient_fhir_id}: {record['error']}")
    return build_starting_packet(question, record, patient_fhir_id, max_tokens, current_time)
