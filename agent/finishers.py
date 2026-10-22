"""Template-routed deterministic finishers for the V10 team (docs/diagnosis_v8/design_v10.txt §2).

The V8 convention checks (agent.convention_checks) detected a convention error
and asked the model to repair it. A finisher recomputes the convention-sensitive
last step in code instead, from the anchor record the candidate cited:

  T051/T052  first/last drug [via route]: every distinct linked drug name of the
             patient's prescriptions that share the cited start minute (T052:
             within the cited route). Replaces a candidate that lists a strict
             subset of that tie set.
  T054       specimen name: the cited test's Observation.specimen -> Specimen.type
             display. Replaces a candidate that answered the test's code display
             or cited no Specimen.
  T055       organism: the distinct code displays of the cited test's hasMember
             children. Replaces a null or single-string candidate.
  T001       distinct drug count: distinct linked drug names over the complete
             patient MedicationRequest -> Medication closure within the question
             window. Replaces a different count.
  T064/T073  timestamp field: the parent test's effectiveDateTime (T064) or the
             prescription's dispenseRequest.validityPeriod.start (T073) of the
             cited record, substituted when the candidate's time came from a
             date-only value, a child finding or authoredOn.

Rules shared by every finisher: routing is by `classify_template` on the
question text, never by a question id or a draft flag; a finisher runs only if
the candidate cites a record of the expected type; it makes at most FETCH_LIMIT
patient-scoped retrievals, each charged as one tool call to the shared
accountant; it never releases a null; and its result is released only if the
replacement condition holds and every citation it builds verifies against the
records it fetched. Every other path releases the candidate unchanged. No model
is called and no reference answer is consulted.
"""
import json
import re

from tools.field_check import (ENCOUNTER_SCOPE_RE, ON_DATE_RE, RELATIVE_DAY_RE, RELATIVE_MONTH_RE,
                               RELATIVE_YEAR_RE, SINCE_AGO_RE, SINCE_DATE_RE, _wall_clock, question_time_window)
from tools.starting_packet import _display
from utils.evidence_scorer import check_citation
from .convention_checks import (_answer_items, _by_id, _minute, _norm, _ref, _request_names, _request_start,
                                tie_set)
from .framework import RETRIEVAL_TOOLS, Toolbox
from .template_router import classify_template
from .verification_agent import _canonical

FINISHER_OUTCOMES = ("none", "skipped_no_anchor", "confirmed_primary", "replaced")
FETCH_LIMIT = 2
ANCHOR_TYPE = {"T001": "MedicationRequest", "T051": "MedicationRequest", "T052": "MedicationRequest",
               "T073": "MedicationRequest", "T054": "Observation", "T055": "Observation", "T064": "Observation"}
DATE_ONLY_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s*$")
WINDOW_PATTERNS = (SINCE_AGO_RE, SINCE_DATE_RE, ON_DATE_RE, RELATIVE_YEAR_RE, RELATIVE_MONTH_RE, RELATIVE_DAY_RE)


class FinisherStop(Exception):
    """The finisher cannot produce a verified result; the candidate is released."""


class Fetcher:
    """At most `limit` patient-wide retrievals through the shared scope validator.

    Each retrieval is one get_resources_by_patient_fhir_id call charged through
    `take_tool_call` (the team's aggregate accountant). Records owned by another
    patient or dated after the current time are left out exactly as for every
    other role (agent.framework.Toolbox); Medication is accepted only when a held
    prescription links it, so MedicationRequest is fetched first.
    """

    def __init__(self, take_tool_call, patient_fhir_id, current_time, limit=FETCH_LIMIT):
        self._take_tool_call = take_tool_call
        self.toolbox = Toolbox(RETRIEVAL_TOOLS, resources={}, patient_fhir_id=patient_fhir_id,
                               current_time=current_time)
        self.limit = limit
        self.ledger = []

    @property
    def workspace(self):
        return self.toolbox.resources

    def patient(self, resource_type):
        """The patient's in-scope records of one type."""
        if len(self.ledger) >= self.limit:
            raise FinisherStop("fetch_limit")
        if not self._take_tool_call():
            raise FinisherStop("tool_budget")
        entry = {"resource_type": resource_type, "success": False, "complete": False}
        self.ledger.append(entry)
        try:
            output = json.loads(self.toolbox._retrieve("get_resources_by_patient_fhir_id", {
                "resource_type": resource_type, "patient_fhir_id": self.toolbox.patient_fhir_id}))
        except Exception as exc:
            entry["error"] = type(exc).__name__
            raise FinisherStop("fetch_failed") from exc
        if output.get("error"):
            entry["error"] = "incomplete retrieval"
            raise FinisherStop("fetch_failed")
        unresolved = [key for key in self.toolbox.scope_failures if key.startswith(resource_type + "/")]
        entry.update(success=True, complete=not unresolved, items=len(self.workspace.get(resource_type, [])),
                     excluded=output.get("excluded") or {})
        return self.workspace.get(resource_type, [])

    def complete(self, resource_type):
        return any(e["resource_type"] == resource_type and e["complete"] for e in self.ledger)


def cited_ids(candidate):
    """Cited "Type/id" strings of a candidate, in citation order."""
    candidate = candidate or {}
    ids = [c.get("resource_id") for c in candidate.get("evidence") or [] if isinstance(c, dict)]
    ids += list(candidate.get("contributing_ids") or [])
    return list(dict.fromkeys(rid for rid in ids if isinstance(rid, str)))


def _find_path(node, value, path=""):
    """The first dot/index path in `node` whose string value equals `value`, or None."""
    if isinstance(node, dict):
        for key, item in node.items():
            found = _find_path(item, value, f"{path}.{key}" if path else key)
            if found:
                return found
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found = _find_path(item, value, f"{path}[{index}]")
            if found:
                return found
    elif isinstance(node, str) and node == value:
        return path
    return None


def _cite(record, value, field=None):
    """A citation of `value` on `record` (at `field`, else wherever the value is stored)."""
    path = field or _find_path(record, value)
    if path is None:
        raise FinisherStop("value_not_in_record")
    return {"resource_id": f"{record['resourceType']}/{record['id']}", "field_path": path, "value": value}


def _as_answer(values):
    return values[0] if len(values) == 1 else list(values)


def _single_string(answer):
    if isinstance(answer, str):
        return answer
    if isinstance(answer, list) and len(answer) == 1 and isinstance(answer[0], str):
        return answer[0]
    return None


def _anchors(by_id, anchor_ids):
    records = [by_id[rid] for rid in anchor_ids if rid in by_id]
    if not records:
        raise FinisherStop("anchor_out_of_scope")
    return records


def _test_of(record, by_id):
    """The top-level microbiology test of a cited test, organism or susceptibility Observation."""
    seen = set()
    while record.get("derivedFrom") and _ref(record["derivedFrom"][0]) in by_id and record["id"] not in seen:
        seen.add(record["id"])
        record = by_id[_ref(record["derivedFrom"][0])]
    return record


def _tests(by_id, anchor_ids):
    tests = list({t["id"]: t for t in (_test_of(r, by_id) for r in _anchors(by_id, anchor_ids))}.values())
    if len({t.get("effectiveDateTime") for t in tests}) != 1:
        raise FinisherStop("anchor_time_ambiguous")
    return tests


# ---------------------------------------------------------------------------
# The finishers: (result, evidence, replace?, detail) or FinisherStop
# ---------------------------------------------------------------------------

def finish_tie(template, context, candidate, anchor_ids, fetcher):
    fetcher.patient("MedicationRequest")
    fetcher.patient("Medication")
    workspace = fetcher.workspace
    by_id = _by_id(workspace)
    cited = _anchors(by_id, anchor_ids)
    tie = tie_set(cited, workspace, same_route=template == "T052")
    if tie is None:
        raise FinisherStop("anchor_start_ambiguous")
    _, tied = tie
    if not tied:
        raise FinisherStop("empty_result")
    evidence = []
    for value, rid in tied.values():
        request = by_id[rid]
        evidence.append(_cite(request, _request_start(request), "dispenseRequest.validityPeriod.start"))
        evidence.append(_cite(_name_source(request, value, by_id), value))
    items = _answer_items(candidate.get("answer")) or []
    keys = {_norm(item) for item in items}
    replace = len(tied) > 1 and bool(keys) and keys < set(tied)
    return (_as_answer([value for value, _ in tied.values()]), evidence, replace,
            {"tie_size": len(tied), "candidate_items": len(items), "candidate_in_tie": bool(keys) and keys <= set(tied)})


def _name_source(request, name, by_id):
    """The record that holds a released drug name of `request` (its Medication, a mix component, or itself)."""
    pending, seen = [_ref(request.get("medicationReference"))], set()
    while pending:
        key = pending.pop(0)
        medication = by_id.get(key)
        if medication is None or key in seen:
            continue
        seen.add(key)
        if _find_path(medication, name):
            return medication
        pending.extend(_ref(i.get("itemReference")) for i in medication.get("ingredient") or [])
    return request


def finish_specimen(template, context, candidate, anchor_ids, fetcher):
    fetcher.patient("Observation")
    fetcher.patient("Specimen")
    by_id = _by_id(fetcher.workspace)
    tests = _tests(by_id, anchor_ids)
    displays, evidence = {}, []
    for test in tests:
        specimen = by_id.get(_ref(test.get("specimen")))
        if specimen is None:
            raise FinisherStop("specimen_missing")
        display = _display(specimen.get("type"))
        if not display:
            raise FinisherStop("specimen_type_missing")
        displays.setdefault(_norm(display), display)
        evidence += [_cite(test, test["specimen"]["reference"], "specimen.reference"), _cite(specimen, display)]
    answer = _single_string(candidate.get("answer"))
    test_codes = {_norm(_display(t.get("code"))) for t in tests} - {""}
    cited_specimen = any(rid.startswith("Specimen/") for rid in cited_ids(candidate))
    replace = (answer is not None and _norm(answer) in test_codes) or not cited_specimen
    return (_as_answer(list(displays.values())), evidence, replace,
            {"tests": len(tests), "candidate_is_test_code": answer is not None and _norm(answer) in test_codes,
             "candidate_cited_specimen": cited_specimen})


def finish_organism(template, context, candidate, anchor_ids, fetcher):
    fetcher.patient("Observation")
    by_id = _by_id(fetcher.workspace)
    tests = _tests(by_id, anchor_ids)
    organisms, evidence = {}, []
    for test in tests:
        for index, member in enumerate(test.get("hasMember") or []):
            child = by_id.get(_ref(member))
            display = _display((child or {}).get("code"))
            if not display:
                continue
            organisms.setdefault(_norm(display), display)
            evidence += [_cite(test, member["reference"], f"hasMember[{index}].reference"), _cite(child, display)]
    if not organisms:
        raise FinisherStop("no_children")
    answer = candidate.get("answer")
    replace = answer is None or _single_string(answer) is not None
    return _as_answer(list(organisms.values())), evidence, replace, {"tests": len(tests), "organisms": len(organisms)}


def count_window(question, current_time):
    """(start, end, source) of a distinct-count question's prescription window; end is exclusive."""
    if ENCOUNTER_SCOPE_RE.search(question):
        raise FinisherStop("encounter_scope_unsupported")
    if not _wall_clock(current_time):
        # live calibration: without a current time the closure holds prescriptions the reference does not
        raise FinisherStop("no_current_time")
    window = question_time_window(question, current_time)
    if window is not None:
        start, end, _ = window
        return start, end, "question_window"
    if any(pattern.search(question) for pattern in WINDOW_PATTERNS):
        raise FinisherStop("window_unparsed")
    return None, None, "question_unrestricted"


def finish_count(template, context, candidate, anchor_ids, fetcher):
    answer = candidate.get("answer")
    if isinstance(answer, bool) or not isinstance(answer, (int, float)):
        raise FinisherStop("candidate_not_a_count")
    start, end, source = count_window(context["question"], context.get("current_time"))
    fetcher.patient("MedicationRequest")
    fetcher.patient("Medication")
    if not (fetcher.complete("MedicationRequest") and fetcher.complete("Medication")):
        raise FinisherStop("closure_incomplete")
    workspace = fetcher.workspace
    by_id = _by_id(workspace)
    names, undated, evidence = {}, set(), []
    for request in workspace.get("MedicationRequest", []):
        if not request.get("encounter"):
            continue  # the reference counts prescriptions of a hospital admission
        moment = _wall_clock(_request_start(request))
        if start is not None and not moment:
            # live calibration: the reference counts a prescription without validityPeriod.start in
            # some windows and not in others, so an undated name inside the window is ambiguous
            authored = _wall_clock(request.get("authoredOn"))
            if authored and start <= authored and (end is None or authored < end):
                undated.update(_request_names(request, by_id))
            continue
        if start is not None and not (start <= moment and (end is None or moment < end)):
            continue
        for name in _request_names(request, by_id):
            if name and name not in names:
                names[name] = request
                evidence.append(_cite(request, _request_start(request), "dispenseRequest.validityPeriod.start")
                                if moment else _cite(request, request["id"], "id"))
                evidence.append(_cite(_name_source(request, name, by_id), name))
    if undated - set(names):
        raise FinisherStop("undated_prescriptions_in_window")
    if not names:
        raise FinisherStop("zero_count")
    return len(names), evidence, len(names) != answer, {"window_source": source,
                                                        "prescriptions": len(workspace.get("MedicationRequest", []))}


def finish_test_time(template, context, candidate, anchor_ids, fetcher):
    answer = candidate.get("answer")
    if not isinstance(answer, str):
        raise FinisherStop("candidate_not_a_time")
    fetcher.patient("Observation")
    by_id = _by_id(fetcher.workspace)
    cited = _anchors(by_id, anchor_ids)
    tests = _tests(by_id, anchor_ids)
    target = tests[0].get("effectiveDateTime")
    if not _minute(target):
        raise FinisherStop("no_full_timestamp")
    if _minute(answer) == _minute(target):
        return target, [_cite(tests[0], target, "effectiveDateTime")], False, {"source": "convention_field"}
    child_times = {_minute(r.get("effectiveDateTime")) for r in cited if r not in tests} - {None}
    if DATE_ONLY_RE.match(answer) and answer.strip() == _wall_clock(target)[:10]:
        source = "date_only"
    elif _minute(answer) and _minute(answer) in child_times:
        source = "child_finding"
    else:
        raise FinisherStop("answer_source_unknown")
    return target, [_cite(tests[0], target, "effectiveDateTime")], True, {"source": source}


def finish_prescription_time(template, context, candidate, anchor_ids, fetcher):
    answer = candidate.get("answer")
    if not isinstance(answer, str):
        raise FinisherStop("candidate_not_a_time")
    fetcher.patient("MedicationRequest")
    by_id = _by_id(fetcher.workspace)
    cited = [r for r in _anchors(by_id, anchor_ids) if _minute(_request_start(r))]
    if not cited:
        raise FinisherStop("no_validity_start")
    if any(_minute(_request_start(r)) == _minute(answer) for r in cited):
        request = next(r for r in cited if _minute(_request_start(r)) == _minute(answer))
        return (_request_start(request), [_cite(request, _request_start(request), "dispenseRequest.validityPeriod.start")],
                False, {"source": "convention_field"})
    if DATE_ONLY_RE.match(answer):
        matches, source = [r for r in cited if _wall_clock(_request_start(r))[:10] == answer.strip()], "date_only"
    else:
        matches, source = [r for r in cited if _minute(answer) and _minute(r.get("authoredOn")) == _minute(answer)], "authoredOn"
    starts = {_request_start(r) for r in matches}
    if len(starts) != 1:
        raise FinisherStop("answer_source_unknown" if not starts else "anchor_start_ambiguous")
    request = matches[0]
    return (_request_start(request), [_cite(request, _request_start(request), "dispenseRequest.validityPeriod.start")],
            True, {"source": source})


FINISHERS = {"T051": finish_tie, "T052": finish_tie, "T054": finish_specimen, "T055": finish_organism,
             "T001": finish_count, "T064": finish_test_time, "T073": finish_prescription_time}


def run_finisher(context, candidate, fetcher):
    """Route, run and verify one finisher. Returns the finisher record (`finisher` in FINISHER_OUTCOMES).

    `result`/`evidence` are set only when the outcome is "replaced".
    """
    template = classify_template(context["question"])
    record = {"template": template, "finisher": "none", "reason": None, "anchors": [], "result": None,
              "evidence": [], "fetches": fetcher.ledger, "detail": {}}
    if template is None:
        return record
    anchors = [rid for rid in cited_ids(candidate) if rid.startswith(ANCHOR_TYPE[template] + "/")]
    record["anchors"] = anchors
    if not anchors:
        record["finisher"] = "skipped_no_anchor"
        return record
    record["finisher"] = "confirmed_primary"
    try:
        result, evidence, replace, detail = FINISHERS[template](template, context, candidate, anchors, fetcher)
    except FinisherStop as stop:
        record["reason"] = str(stop)
        return record
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        record["reason"] = f"finisher_error: {type(exc).__name__}"
        return record
    record["detail"] = detail
    held = {f"{rtype}/{r['id']}": r for rtype, items in fetcher.workspace.items() for r in items}
    if result is None:
        record["reason"] = "null_result"
    elif not evidence or any(check_citation(c, held) != "valid" for c in evidence):
        record["reason"] = "evidence_unverified"
    elif _canonical(result, context["question"]) == _canonical(candidate.get("answer"), context["question"]):
        record["reason"] = "result_equals_primary"
    elif not replace:
        record["reason"] = "condition_not_met"
    else:
        record.update(finisher="replaced", result=result, evidence=evidence)
    return record
