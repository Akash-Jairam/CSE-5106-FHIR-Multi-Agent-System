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
- time window: the cited resource is dated outside the window the question
  restricts events to ("since 05/2137", "in 11/2136", "09/this year",
  "last month/21", "since 2 months ago", "yesterday", ...). The windows
  follow the dataset's SQL: relative windows count back from midnight of
  the current date with SQLite month arithmetic, and "since" windows end at
  the current time, which the current-time check enforces.
- patient: patient-owned evidence must link to the question's patient
"""
import calendar
import re
from datetime import datetime, timedelta

from utils.answer_scorer import UNITS
from utils.evidence_scorer import check_citation, fetch_from_store, get_cited_resource, parse_citation, resolve_path
from .registry import tool_registry

QUESTION_UNIT_RE = re.compile(r"\bin\s+(" + "|".join(sorted(map(re.escape, UNITS), key=len, reverse=True)) + r")\b", re.I)
ENCOUNTER_SCOPE_RE = re.compile(r"\b(first|last|latest|current|this)\s+(?:completed\s+)?(hospital|icu)\s+(?:visit|encounter|stay)\b", re.I)
ENCOUNTER_EVENT_RE = re.compile(
    r"\b(?:how many|number of|count(?: of)?|any|which)\s+(?:different\s+|total\s+|unique\s+)?"
    r"(?:hospital|icu|er|emergency(?: department)?)\s+(?:visits?|encounters?|stays?|admissions?)\b"
    r"|\b(?:been to|admitted to)\s+(?:the\s+)?(?:hospital|er|icu)\b|\b(?:hospital|icu)\s+admissions?\b", re.I)
EVENT_TYPES = {"Observation", "MedicationRequest", "Procedure", "Condition"}
# MIMIC-IV dates are shifted into 2100-2299; anything else is not a date window
YEAR = r"2[12]\d{2}"
SINCE_AGO_RE = re.compile(r"\bsince (\d+) (day|month|year)s? ago\b", re.I)
SINCE_DATE_RE = re.compile(rf"\bsince ((?:\d{{2}}/){{0,2}}{YEAR})\b", re.I)
ON_DATE_RE = re.compile(rf"\b(?:in|on) ((?:\d{{2}}/){{0,2}}{YEAR})\b", re.I)
RELATIVE_YEAR_RE = re.compile(r"((?:\d{2}/){0,2})\b(this|last) year\b", re.I)
RELATIVE_MONTH_RE = re.compile(r"\b(this|last) month(?:/(\d{2}))?", re.I)
RELATIVE_DAY_RE = re.compile(r"\b(today|yesterday)\b", re.I)
WALL_CLOCK = "%Y-%m-%d %H:%M:%S"

def _ref_id(reference):
    if isinstance(reference, str):
        return reference.rstrip("/").split("/")[-1]
    return (reference or {}).get("reference", "").rstrip("/").split("/")[-1]


def _wall_clock(value):
    if not isinstance(value, str):
        return ""
    try:
        # The benchmark SQL compares deidentified local clock values, not UTC instants.
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None).strftime(WALL_CLOCK)
    except ValueError:
        return ""


def _months_before(day, months):
    """`day` minus `months` as SQLite computes it: keep the day of month and
    roll an overflow into the next month (12-31 minus 1 month is 12-01)."""
    year, month = divmod(day.year * 12 + day.month - 1 - months, 12)
    month += 1
    last = calendar.monthrange(year, month)[1]
    return datetime(year, month, min(day.day, last)) + timedelta(days=max(0, day.day - last))


def _period(year, month=None, day=None):
    """[start, end) of a calendar year, month, or day."""
    if day is not None:
        start = datetime(year, month, day)
        return start, start + timedelta(days=1)
    if month is not None:
        start = datetime(year, month, 1)
        return start, _months_before(start, -1)
    return datetime(year, 1, 1), datetime(year + 1, 1, 1)


def _date_parts(text):
    """(year, month, day) from "MM/DD/YYYY", "MM/YYYY", or "YYYY"."""
    parts = [int(p) for p in text.split("/")]
    year = parts.pop()
    return (year, *(parts + [None, None])[:2])


def question_time_window(question, current_time=None):
    """The window a question restricts events to, as (start, end, phrase).

    start and end are wall-clock strings; end is None for "since" windows,
    which run to the current time. Returns None if the question has no
    recognized window, or has a relative one and no current time.
    """
    question = question or ""
    try:
        now = datetime.strptime(_wall_clock(current_time), WALL_CLOCK) if current_time else None
        today = datetime(now.year, now.month, now.day) if now else None
        if (match := SINCE_AGO_RE.search(question)) and today:
            n, unit = int(match.group(1)), match.group(2).lower()
            start = today - timedelta(days=n) if unit == "day" else _months_before(today, n * (12 if unit == "year" else 1))
            window = start, None
        elif match := SINCE_DATE_RE.search(question):
            window = _period(*_date_parts(match.group(1)))[0], None
        elif match := ON_DATE_RE.search(question):
            window = _period(*_date_parts(match.group(1)))
        elif (match := RELATIVE_YEAR_RE.search(question)) and today:
            year = today.year - (match.group(2).lower() == "last")
            prefix = [int(p) for p in match.group(1).split("/") if p]
            window = _period(year, *(prefix + [None, None])[:2])
        elif (match := RELATIVE_MONTH_RE.search(question)) and today:
            first = _months_before(datetime(today.year, today.month, 1), match.group(1).lower() == "last")
            window = _period(first.year, first.month, int(match.group(2)) if match.group(2) else None)
        elif (match := RELATIVE_DAY_RE.search(question)) and today:
            day = today - timedelta(days=match.group(1).lower() == "yesterday")
            window = _period(day.year, day.month, day.day)
        else:
            return None
    except (ValueError, OverflowError):  # impossible dates or out-of-range relative offsets
        return None
    start, end = window
    return start.strftime(WALL_CLOCK), end.strftime(WALL_CLOCK) if end else None, match.group(0)


# Paired detection deliberately requires two explicitly mentioned measurements and
# a relation between them. An ordinary numeric threshold has only one measurement
# and keeps the exact legacy scope path below. Indices follow textual order, not
# the sign/direction of an arithmetic operation.
PAIRED_MEASUREMENT_RE = re.compile(r"\b(?:measured|measurements?)\b", re.I)
PAIRED_CONNECTOR_RE = re.compile(
    r"\bcompared\s+(?:to|with)\b|\b(?:greater|less|higher|lower|smaller)\s+than\b"
    r"|\bversus\b|\bvs\.?", re.I)
PAIRED_CHANGE_RE = re.compile(r"\b(?:difference|change|changes|differ|differs|shifts?)\b", re.I)
EXACT_MEASUREMENT_TIME_RE = re.compile(
    rf"\bat\s+({YEAR}-\d{{2}}-\d{{2}}[ T]\d{{2}}:\d{{2}}(?::\d{{2}})?"
    r"(?:Z|[+-]\d{2}:\d{2})?)\b", re.I)
FINAL_VISIT_RE = re.compile(r"\bfinal(?=\s+(?:hospital|icu)\s+(?:visit|encounter|stay)\b)", re.I)


def question_evidence_scopes(question, current_time=None):
    """Return two gold-free operand scopes for an explicit paired question, or None.

    The accepted structure contains two measurement mentions and a comparison
    connector between them. The explicit ``from ... measured to ... measured
    during <visit>`` difference form inherits its trailing common visit/window.
    Scope values always come from the question; a citation may select only an
    index, never supply arbitrary window boundaries or encounter selectors.

    This source checker does not establish concept matching, measurement ordinal
    selection, subtraction direction, or operand completeness. Those remain the
    solver/independent computation's responsibility.
    """
    question = question or ""
    measurements = list(PAIRED_MEASUREMENT_RE.finditer(question))
    if len(measurements) != 2:
        return None
    first, second = measurements
    between = question[first.end():second.start()]
    connector = PAIRED_CONNECTOR_RE.search(between)
    shared_trailing = False
    if (connector is None and re.match(r"\s*compared\s+(?:to|with)\b", question, re.I)
            and re.search(r"\b(?:greater|less|higher|lower|smaller)\b", question[second.end():], re.I)):
        # Inverted comparison: ``Compared to <first measurement>, is <second
        # measurement> greater?``. The comma separates the two explicit operands.
        connector = re.search(r"[,;]", between)
    if connector is None and PAIRED_CHANGE_RE.search(question):
        # The release includes ``from second-to-last ... measured to last ...
        # measured during final ICU visit``. This recognizes its general syntax.
        connector = re.search(r"\bto\b", between, re.I)
        shared_trailing = connector is not None
    if connector is None:
        return None
    boundary_start = first.end() + connector.start()
    boundary_end = first.end() + connector.end()
    clauses = [question[:boundary_start].strip(), question[boundary_end:].strip()]
    scopes = []
    for index, clause in enumerate(clauses):
        normalized = FINAL_VISIT_RE.sub("last", clause)
        exact_matches = list(EXACT_MEASUREMENT_TIME_RE.finditer(normalized))
        exact = exact_matches[0] if exact_matches else None
        window = question_time_window(normalized, current_time)
        parse_issues = []
        if len(exact_matches) > 1:
            parse_issues.append("multiple exact measurement timestamps within one paired operand")
        if exact:
            when = _wall_clock(exact.group(1))
            if not when:
                parse_issues.append("invalid explicitly requested measurement timestamp")
            else:
                start = datetime.strptime(when, WALL_CLOCK)
                width = 1 if re.search(r"\d{2}:\d{2}:\d{2}", exact.group(1)) else 60
                window = (when, (start + timedelta(seconds=width)).strftime(WALL_CLOCK), exact.group(0))
        elif window is None and any(pattern.search(normalized) for pattern in
                                    (SINCE_AGO_RE, SINCE_DATE_RE, ON_DATE_RE, RELATIVE_YEAR_RE,
                                     RELATIVE_MONTH_RE, RELATIVE_DAY_RE)):
            parse_issues.append("the requested calendar window cannot be resolved from the question and reference time")
        encounter_phrases = list(ENCOUNTER_SCOPE_RE.finditer(normalized))
        if len(encounter_phrases) > 1:
            parse_issues.append("multiple encounter selectors within one paired operand")
        scopes.append({"index": index, "text": clause, "scope_question": normalized,
                       "time_window": window,
                       "encounter_phrase": encounter_phrases[0].group(0) if encounter_phrases else None,
                       "exact_measurement_time": exact.group(1) if exact else None,
                       "parse_issues": parse_issues})
    if shared_trailing and not scopes[0]["time_window"] and not scopes[0]["encounter_phrase"]:
        # A single final qualifier is inherited only for this explicit from/to
        # construction, not for arbitrary asymmetric compared-to clauses.
        scopes[0]["time_window"] = scopes[1]["time_window"]
        scopes[0]["encounter_phrase"] = scopes[1]["encounter_phrase"]
        scopes[0]["scope_question"] = scopes[1]["scope_question"]
        scopes[0]["parse_issues"] = list(scopes[1]["parse_issues"])
        scopes[0]["shared_trailing_scope"] = True
    return scopes


def _resource_time(resource):
    for path in ("effectiveDateTime", "dispenseRequest.validityPeriod.start", "performedDateTime",
                 "performedPeriod.start", "period.start", "issued", "authoredOn"):
        values = resolve_path(resource, path)
        if values:
            return _wall_clock(values[0])
    return ""


def resource_event_time(resource, resources=None):
    """Dataset event time, excluding release/order timestamps and metadata dates."""
    rtype = resource.get("resourceType")
    paths = {"Observation": ("effectiveDateTime", "effectivePeriod.start"),
             "MedicationRequest": ("dispenseRequest.validityPeriod.start",),
             "Procedure": ("performedDateTime", "performedPeriod.start"),
             "Encounter": ("period.start",)}.get(rtype, ())
    for path in paths:
        values = resolve_path(resource, path)
        if values and (moment := _wall_clock(values[0])):
            return moment
    if rtype == "Condition":
        linked = _ref_id(resource.get("encounter") or resource.get("context"))
        encounter = get_cited_resource(resources or {}, "Encounter", linked) or {}
        return resource_event_time(encounter)
    return ""


def _unit_key(unit):
    return (unit or "").strip().lower().rstrip(".")


def question_encounter_scope(question, patient_fhir_id, resources, current_time=None):
    """Pure first/completed-last/ongoing-current visit selection from original records."""
    match = ENCOUNTER_SCOPE_RE.search(question or "")
    if not match or not patient_fhir_id:
        return None, []
    order, kind = match.groups()
    order, kind = order.lower(), kind.lower()
    patient_fhir_id = _ref_id(patient_fhir_id)
    current_time = _wall_clock(current_time) if current_time else None
    if order in ("current", "this") and not current_time:
        return None, [f"encounter_scope: the current {kind} encounter requires an assumed current time"]
    # The evidence index retains both typed and legacy bare keys for one resource.
    encounters = list({r["id"]: r for r in resources.values()
                       if r.get("resourceType") == "Encounter" and r.get("id")}.values())
    suffix = "encounter-hosp" if kind.lower() == "hospital" else "encounter-icu"
    encounters = [r for r in encounters
                  if _ref_id(r.get("subject")) == patient_fhir_id
                  and any(i.get("system", "").endswith(suffix) for i in r.get("identifier", []))
                  and resource_event_time(r)
                  and (not current_time or resource_event_time(r) <= current_time)]
    if order in ("last", "latest"):
        encounters = [r for r in encounters if (end := _wall_clock(r.get("period", {}).get("end")))
                      and end >= resource_event_time(r) and (not current_time or end <= current_time)]
    elif order in ("current", "this"):
        encounters = [r for r in encounters if not r.get("period", {}).get("end")
                      or _wall_clock(r["period"]["end"]) > current_time]
        if len(encounters) > 1:
            return None, [f"encounter_scope: multiple ongoing {kind} encounters make the requested scope ambiguous"]
    if not encounters:
        return None, [f"encounter_scope: no {kind.lower()} encounter establishes the requested scope"]
    encounters.sort(key=lambda r: (resource_event_time(r), r["id"]))
    return encounters[0 if order == "first" else -1]["id"], []


def _question_encounter(question, patient_fhir_id, resources, from_store, current_time):
    if from_store and ENCOUNTER_SCOPE_RE.search(question or "") and patient_fhir_id:
        from fhir_client import get_fhir_client
        encounters = get_fhir_client().search_with_pagination(f"Encounter?patient={patient_fhir_id}")
        resources.update({f"Encounter/{r['id']}": r for r in encounters})
    return question_encounter_scope(question, patient_fhir_id, resources, current_time)


def _check_one(citation, resources, question, current_time, encounter_id, patient_fhir_id, window=None,
               encounter_window=True):
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

    time = resource_event_time(resource, resources)
    availability_time = time or _resource_time(resource)
    cited_times = [_wall_clock(value) for value in resolve_path(resource, path) or []]
    future = max([availability_time, *cited_times])
    if current_time and future and future > current_time:
        result["warnings"].append(f"current_time: the resource or cited field is dated {future}, after the assumed current time {current_time}")
    # Encounters can be supporting context for an event inside a later window.
    if window and (rtype in EVENT_TYPES or (rtype == "Encounter" and encounter_window)):
        window_start, window_end, phrase = window
        if not time:
            result["warnings"].append(f"time_window: no event timestamp establishes whether the resource is inside "
                                      f"the question's window \"{phrase}\"")
        elif time < window_start or (window_end and time >= window_end):
            result["warnings"].append(f"time_window: the resource is dated {time}, outside the question's window "
                                      f"\"{phrase}\" ({window_start} to {window_end or 'the current time'})")
    return result


def _check_paired_fields(citations, scopes, question, current_time, encounter_id, resources,
                         patient_fhir_id):
    """Check source integrity once, then temporal/visit membership per requested operand.

    Unannotated citations may match the requested union. An explicit scope_index
    must match that operand, so it cannot widen an original question constraint.
    The metadata describes scope membership only, never arithmetic correctness
    or proof that the two selected measurements are first/last as requested.
    """
    citations = list(citations or [])
    from_store = resources is None
    patient_fhir_id = _ref_id(patient_fhir_id) if patient_fhir_id else None
    current_time = _wall_clock(current_time) if current_time else None
    supplied_encounter = _ref_id(encounter_id) if encounter_id else None
    resources = dict(resources or {})
    parsed = [p for p in map(parse_citation, citations) if p]
    refs = {p[:2] for p in parsed if get_cited_resource(resources, p[0], p[1]) is None}
    if refs:
        resources.update(fetch_from_store(refs))

    # Fetch complete encounter context once. Looking only at cited visits cannot
    # establish first/latest-completed/current scope.
    if from_store and patient_fhir_id and any(scope["encounter_phrase"] for scope in scopes):
        from fhir_client import get_fhir_client
        encounters = get_fhir_client().search_with_pagination(f"Encounter?patient={patient_fhir_id}")
        resources.update({f"Encounter/{r['id']}": r for r in encounters})
    resolved = []
    scope_issues = []
    for scope in scopes:
        scoped, issues = question_encounter_scope(scope["scope_question"], patient_fhir_id,
                                                   resources, current_time)
        own_issues = [f"paired_scope: operand {scope['index']}: {issue}"
                      for issue in scope["parse_issues"]]
        own_issues.extend(issues)
        resolved.append({**scope, "encounter_id": scoped, "scope_issues": own_issues})
        scope_issues.extend(own_issues)
    selected_encounters = {scope["encounter_id"] for scope in resolved if scope["encounter_id"]}
    if supplied_encounter and supplied_encounter not in selected_encounters:
        scope_issues.append("encounter_scope: the supplied encounter is not one of the question-derived paired scopes")

    # Preserve parent links and event availability checks from the legacy path.
    cited = [r for p in parsed if (r := get_cited_resource(resources, p[0], p[1])) is not None]
    linked = {_ref_id(r.get("encounter") or r.get("context")) for r in cited}
    missing = {("Encounter", i) for i in linked if i and get_cited_resource(resources, "Encounter", i) is None}
    if missing:
        resources.update(fetch_from_store(missing))
    event_citations = any(p[0] in EVENT_TYPES for p in parsed)
    encounter_window = not event_citations or bool(ENCOUNTER_EVENT_RE.search(question or ""))

    results = []
    for citation in citations:
        base = _check_one(citation, resources, question, current_time, None, patient_fhir_id,
                          window=None, encounter_window=encounter_window)
        requested = citation.get("scope_index") if isinstance(citation, dict) else None
        indexed = isinstance(citation, dict) and "scope_index" in citation
        valid_index = not indexed or (type(requested) is int and requested in (0, 1))
        chosen = resolved if not indexed else [scope for scope in resolved if scope["index"] == requested]
        base["matched_scopes"] = []
        base["scope_checks"] = []
        if indexed:
            base["requested_scope_index"] = requested
        if not valid_index:
            base["warnings"].append("paired_scope: scope_index must be the integer 0 or 1 from the question-derived scopes")
        else:
            for scope in chosen:
                checked = _check_one(citation, resources, question, current_time, scope["encounter_id"],
                                     patient_fhir_id, scope["time_window"], encounter_window)
                warnings = [w for w in checked["warnings"] if w not in base["warnings"]]
                warnings.extend(scope["scope_issues"])
                base["scope_checks"].append({"index": scope["index"], "warnings": warnings})
                if base["status"] == "valid" and not base["warnings"] and not warnings:
                    base["matched_scopes"].append(scope["index"])
            if base["status"] == "valid" and not base["warnings"] and not base["matched_scopes"]:
                prefix = f"operand {requested}" if indexed else "either operand"
                base["warnings"].append(f"paired_scope: citation does not match {prefix}'s question-derived time/visit scope")
        results.append(base)
    issues = list(dict.fromkeys(scope_issues))
    for index, result in enumerate(results):
        if result["status"] != "valid":
            issues.append(f"evidence[{index}] {result['status']}" +
                          (f" (actual value: {result['actual']})" if "actual" in result else ""))
        issues.extend(f"evidence[{index}] {warning}" for warning in result["warnings"])
    return {"results": results, "issues": issues,
            "n_valid": sum(result["status"] == "valid" for result in results),
            "question_scopes": [{key: scope[key] for key in
                                 ("index", "text", "time_window", "encounter_phrase", "encounter_id",
                                  "exact_measurement_time", "scope_issues")} for scope in resolved],
            "scope_membership_only": True}


def check_fields(citations, question="", current_time=None, encounter_id=None, resources=None,
                 patient_fhir_id=None):
    """Check cited evidence against supplied original resources or the store.

    Patient scope and recognizable first/last encounter scopes are checked
    when question context is supplied. Compatible unit conversions are
    reported as metadata; they do not establish answer arithmetic.
    """
    scopes = question_evidence_scopes(question, current_time)
    if scopes is not None:
        return _check_paired_fields(citations, scopes, question, current_time, encounter_id, resources,
                                    patient_fhir_id)
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
    window = question_time_window(question, current_time)
    event_citations = any(p[0] in EVENT_TYPES for p in parsed)
    encounter_window = not event_citations or bool(ENCOUNTER_EVENT_RE.search(question or ""))
    # Encounters also establish a date for resources such as Conditions.
    if encounter_id or current_time or window:
        cited = [r for p in parsed if (r := get_cited_resource(resources, p[0], p[1])) is not None]
        linked = {_ref_id(r.get("encounter") or r.get("context")) for r in cited}
        missing = {("Encounter", i) for i in linked if i and get_cited_resource(resources, "Encounter", i) is None}
        if missing:
            resources.update(fetch_from_store(missing))

    results = [_check_one(c, resources, question, current_time, encounter_id, patient_fhir_id, window,
                          encounter_window)
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
                        "mismatches, resources outside a scoped encounter, resources outside the question's time "
                        "window (e.g. 'since 05/2137', 'last month'), and resources after the current time."),
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
                            "value": {"description": "The value you read at field_path"},
                            "scope_index": {"type": "integer", "enum": [0, 1],
                                            "description": "Optional for explicit paired measurements: bind this citation to operand 0 or 1 in question textual order; boundaries are derived from the question, never supplied by the model."}
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
