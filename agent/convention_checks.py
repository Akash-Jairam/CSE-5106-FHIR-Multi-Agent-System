"""Deterministic convention checks for the V8 verifier (docs/diagnosis_v8/decision.md D2).

Each check inspects the analyst's source-valid draft against the retrieved
workspace and the shared data-representation conventions
(agent.ehr_semantics.CONVENTION_GUIDANCE). No model call is made and no
reference answer is consulted: a check can only point at a field, a record
or a missing retrieval. Issues use the fixed controller's vocabulary so the
one permitted repair is routed to the responsible role:

- missing_evidence -> retrieval (a needed record/type was never fetched)
- wrong_field / wrong_filter -> analysis (data present, convention not applied)
"""
import re

from tools.field_check import _wall_clock
from tools.starting_packet import _display, _medication_name

FIRST_LAST_RE = re.compile(r"\b(first|last|earliest|latest)\b", re.I)
DRUG_RE = re.compile(r"\b(drug|medication|medicine|prescri\w*)\b", re.I)
ROUTE_RE = re.compile(r"\broute\b", re.I)
# live calibration: "And what was the last time ..." is a when-question too
WHEN_RE = re.compile(r"^\s*(?:and\s+)?(when|what time|at what time|what was the (?:\w+\s+)?(?:date|time))\b", re.I)
SPECIMEN_RE = re.compile(r"\bspecimen\b", re.I)
ORGANISM_RE = re.compile(r"\borganism", re.I)
MICRO_TEST_RE = re.compile(r"\b(microbiolog\w*|culture|specimen)\b", re.I)
# T001 is mostly phrased "count of distinct" / "number of unique", not "how many distinct"
DISTINCT_COUNT_RE = re.compile(r"\b(how many|count|number|total)\b.*\b(different|distinct|unique)\b", re.I)
DOSE_RE = re.compile(r"\b(doses?|how much|amount|total|volume)\b", re.I)


def _minute(value):
    """Wall-clock string at minute precision ('YYYY-MM-DD HH:MM'), or None."""
    moment = _wall_clock(value) if value else ""
    if not moment:
        # date-only values are not timestamps of a record
        return None
    return moment[:16]


def _by_id(workspace):
    return {f"{rtype}/{r['id']}": r for rtype, items in workspace.items() for r in items if r.get("id")}


def _ref(value):
    reference = (value or {}).get("reference", "")
    return "/".join(reference.rstrip("/").split("/")[-2:])


def _answer_items(answer):
    if answer is None:
        return None
    items = answer if isinstance(answer, list) else [answer]
    return [str(item).strip() for item in items]


def _norm(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _request_start(request):
    return ((request.get("dispenseRequest") or {}).get("validityPeriod") or {}).get("start")


def _request_names(request, by_id, seen=None):
    """Released drug name strings of one prescription (mixtures expand to components)."""
    medication = by_id.get(_ref(request.get("medicationReference")))
    if medication is None:
        name = _display(request.get("medicationCodeableConcept"))
        return [name] if name else []
    seen = set() if seen is None else seen
    if medication["id"] in seen:
        return []
    seen.add(medication["id"])
    ingredients = medication.get("ingredient") or []
    if ingredients:
        names = []
        for ingredient in ingredients:
            component = by_id.get(_ref(ingredient.get("itemReference")))
            if component is not None:
                names.extend(_request_names({"medicationReference": {"reference": f"Medication/{component['id']}"}}, by_id, seen))
        if names:
            return names
    name = _medication_name(medication)
    return [name] if name else []


def _request_route(request):
    dosages = request.get("dosageInstruction") or [{}]
    return _display((dosages[0] or {}).get("route"))


def _cited(draft, workspace, rtype):
    by_id = _by_id(workspace)
    ids = list(draft.get("contributing_ids") or []) + [c.get("resource_id") for c in draft.get("evidence") or []]
    seen, records = set(), []
    for rid in ids:
        if isinstance(rid, str) and rid.startswith(rtype + "/") and rid not in seen and rid in by_id:
            seen.add(rid)
            records.append(by_id[rid])
    return records


def _issue(check, kind, role, action, record=None, field=None):
    return {"type": kind, "check": check, "responsible_role": role, "action": action,
            "record": record, "field": field}


def check_tie_completeness(question, draft, workspace, **_):
    """first/last drug or route: every distinct value at the selected start time."""
    if not (FIRST_LAST_RE.search(question) and (DRUG_RE.search(question) or ROUTE_RE.search(question))
            and not WHEN_RE.match(question) and not DOSE_RE.search(question)):
        return []
    items = _answer_items(draft.get("answer"))
    if not items or isinstance(draft.get("answer"), (int, float, bool)) or all(_minute(i) for i in items):
        # a dose, count or time answer is not a drug/route selection
        return []
    by_id = _by_id(workspace)
    cited = _cited(draft, workspace, "MedicationRequest")
    starts = {_minute(_request_start(r)) for r in cited} - {None}
    if len(starts) != 1:
        return []
    selected = next(iter(starts))
    encounters = {_ref(r.get("encounter")) for r in cited}
    by_route = ROUTE_RE.search(question) and not DRUG_RE.search(question)
    # "drug ... via the iv route": the tie set is restricted to the cited route string
    routes = {_norm(_request_route(r)) for r in cited} if ROUTE_RE.search(question) and not by_route else None
    tied = {}
    for request in workspace.get("MedicationRequest", []):
        if _minute(_request_start(request)) != selected:
            continue
        if encounters and _ref(request.get("encounter")) not in encounters:
            continue
        if routes is not None and _norm(_request_route(request)) not in routes:
            continue
        values = [_request_route(request)] if by_route else _request_names(request, by_id)
        for value in values:
            if value:
                tied.setdefault(_norm(value), f"MedicationRequest/{request['id']}")
    missing = sorted(set(tied) - {_norm(i) for i in items})
    if not missing:
        return []
    what = "routes" if by_route else "drugs"
    return [_issue("tie_completeness", "wrong_filter", "analysis",
                   f"{len(tied)} distinct {what} share the selected start time {selected} but the answer "
                   f"lists {len(items)}; a first/last selection returns every tied value",
                   record=tied[missing[0]], field="dispenseRequest.validityPeriod.start")]


def check_specimen_field(question, draft, workspace, **_):
    """Specimen name is Specimen.type display, not the test code."""
    if not SPECIMEN_RE.search(question) or ORGANISM_RE.search(question) or WHEN_RE.match(question):
        return []
    items = _answer_items(draft.get("answer"))
    if items is None or DISTINCT_COUNT_RE.search(question) or re.match(r"\s*how many", question, re.I):
        return []
    specimens = workspace.get("Specimen", [])
    if not specimens:
        tests = _cited(draft, workspace, "Observation")
        refs = sorted({_ref(t.get("specimen")) for t in tests if t.get("specimen")})
        return [_issue("specimen_field", "missing_evidence", "retrieval",
                       "the specimen name is the Specimen.type display, but no Specimen resource was retrieved"
                       + (f"; cited tests reference {', '.join(refs[:3])}" if refs else ""),
                       record=refs[0] if refs else "Specimen", field="type")]
    displays = {_norm(_display(s.get("type"))) for s in specimens} - {""}
    if all(_norm(item) in displays for item in items):
        return []
    return [_issue("specimen_field", "wrong_field", "analysis",
                   "the answer is not a Specimen.type display string of a retrieved Specimen; the specimen name is "
                   "the specimen's type, not the test name in Observation.code",
                   record=f"Specimen/{specimens[0]['id']}", field="type")]


def check_organism_children(question, draft, workspace, **_):
    """Organisms are child Observations reachable through hasMember."""
    if not ORGANISM_RE.search(question):
        return []
    by_id = _by_id(workspace)
    missing = []
    for test in _cited(draft, workspace, "Observation"):
        for member in test.get("hasMember") or []:
            key = _ref(member)
            if key and key not in by_id:
                missing.append(key)
    if not missing:
        return []
    return [_issue("organism_children", "missing_evidence", "retrieval",
                   f"{len(missing)} organism/child observations linked to the cited test through hasMember were "
                   "never retrieved", record=missing[0], field="hasMember")]


def check_distinct_count_scope(question, draft, workspace, complete_types=(), **_):
    """A distinct drug count needs the complete scoped prescription set."""
    if not DISTINCT_COUNT_RE.search(question) or not DRUG_RE.search(question):
        return []
    if not isinstance(draft.get("answer"), (int, float)) or isinstance(draft.get("answer"), bool):
        return []
    if "MedicationRequest" in set(complete_types):
        return []
    return [_issue("distinct_count_scope", "missing_evidence", "retrieval",
                   f"the answer counts {len(workspace.get('MedicationRequest', []))} held prescriptions, but no complete "
                   "patient-wide MedicationRequest search was recorded; distinct counts need the full scoped set",
                   record="MedicationRequest", field="medicationReference")]


def check_timestamp_field(question, draft, workspace, **_):
    """A 'when' answer equals the convention's timestamp field of a cited record."""
    if not WHEN_RE.match(question):
        return []
    answer = draft.get("answer")
    if not isinstance(answer, str):
        return []
    moment = _minute(answer)
    if moment is None:
        return []
    if DRUG_RE.search(question):
        cited = _cited(draft, workspace, "MedicationRequest")
        starts = {_minute(_request_start(r)) for r in cited} - {None}
        # some released prescriptions carry no validityPeriod.start (authoredOn is hours off); nothing to check
        if starts and moment not in starts:
            return [_issue("timestamp_field", "wrong_field", "analysis",
                           "the answered time is not the dispenseRequest.validityPeriod.start of any cited prescription",
                           record=f"MedicationRequest/{cited[0]['id']}", field="dispenseRequest.validityPeriod.start")]
        return []
    if MICRO_TEST_RE.search(question):
        cited = _cited(draft, workspace, "Observation")
        by_id = _by_id(workspace)
        parents = []
        for record in cited:
            parent = record
            # child organism/susceptibility findings carry the parent test through derivedFrom
            while parent.get("derivedFrom") and _ref(parent["derivedFrom"][0]) in by_id:
                parent = by_id[_ref(parent["derivedFrom"][0])]
            parents.append(parent)
        times = {_minute(p.get("effectiveDateTime")) for p in parents} - {None}
        if parents and moment not in times:
            return [_issue("timestamp_field", "wrong_field", "analysis",
                           "the answered time is not the effectiveDateTime of a cited microbiology test (the parent "
                           "test, not a child finding or a date-only value)",
                           record=f"Observation/{parents[0]['id']}", field="effectiveDateTime")]
    return []


# No admission-type check: live calibration found the reference admission_type nowhere in the
# released Encounter (label_issue), so a repair could only spend budget (exposure.md).
CHECKS = (check_tie_completeness, check_specimen_field, check_organism_children,
          check_distinct_count_scope, check_timestamp_field)


def convention_issues(question, draft, workspace, complete_types=()):
    """All convention issues of a draft; an empty list means no convention check fired."""
    if not isinstance(draft, dict) or draft.get("answer") is None:
        return []
    issues = []
    for check in CHECKS:
        try:
            issues.extend(check(question, draft, workspace, complete_types=complete_types))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            issues.append(_issue(check.__name__.removeprefix("check_"), "check_error", None, f"{type(exc).__name__}"))
    return issues


def repair_message(issues):
    """The one repair instruction, addressed to the routed role."""
    lines = []
    for issue in issues:
        where = f" ({issue['record']}.{issue['field']})" if issue.get("record") and issue.get("field") else ""
        lines.append(f"- [{issue['check']} -> {issue['responsible_role']}] {issue['action']}{where}")
    return ("Convention verification of your source-valid final found these problems:\n" + "\n".join(lines) +
            "\nYou may repair once. If a problem is a missing retrieval, fetch the named records first; otherwise "
            "recompute with the stated convention. Then call execute_python_code with mode=final again. "
            "The repaired final is released only if its evidence passes the same source check as before: cite "
            "field_paths that exist on the cited record with their exact values, and cite only records from the "
            "encounter and time scope the question selects. "
            "If, after checking the actual records, the original answer is correct, re-execute it unchanged.")
