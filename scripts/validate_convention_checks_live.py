"""Calibrate the V8 convention checks on live FHIR resources (hand-off step 3a).

For every question in docs/diagnosis_v8/root_causes.csv (development patients,
exposure already recorded in exposure.md) this pulls the patient's resources
through the same retrieval tools the solver uses, builds the solver-shaped
workspace, and runs agent.convention_checks on two drafts:

  gold   the reference answer citing the reference FHIR ids (true_fhir_ids)
  wrong  the typical wrong answer the audit note describes for the template
         (test name for T054, one drug of a tie set for T051/T052, class for
         T043, organisms without hasMember children for T055, a child or
         date-only time for T064, authoredOn for T073, a preview count for T001)

A check firing on `gold` is a false positive (it would cost a repair and risk a
right->wrong flip); a check staying silent on `wrong` is a miss. Per-template
probes settle the open questions in docs/diagnosis_v8/handoff.md §3a.

No model is called. Reference answers are used only for comparisons and are
never printed or written: the report holds check names, booleans and counts.

Usage:
    python scripts/validate_convention_checks_live.py [--question_ids id1,id2] [--limit N]
                                                      [--output output/diagnosis/convention_checks_live.json]
"""
import argparse
import ast
import collections
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import convention_checks as cc
from tools.resource_tools import get_resources_by_patient_fhir_id, supported_types
from tools.starting_packet import _display, _medication_name
from utils.answer_scorer import parse_reference

ROOT_CAUSES = "docs/diagnosis_v8/root_causes.csv"
DATASET = "final_dataset/questions_answers_sql_fhir.csv"
PREVIEW_IDS = 32

# The check each template's audited failure should trigger, and its question gate.
EXPECTED_CHECK = {
    "T054": "specimen_field", "T051": "tie_completeness", "T052": "tie_completeness",
    "T001": "distinct_count_scope", "T055": "organism_children",
    "T064": "timestamp_field", "T073": "timestamp_field",
}
GATES = {
    "tie_completeness": lambda q: bool(cc.FIRST_LAST_RE.search(q) and (cc.DRUG_RE.search(q) or cc.ROUTE_RE.search(q))
                                       and not cc.WHEN_RE.match(q)),
    "specimen_field": lambda q: bool(cc.SPECIMEN_RE.search(q) and not cc.ORGANISM_RE.search(q) and not cc.WHEN_RE.match(q)),
    "organism_children": lambda q: bool(cc.ORGANISM_RE.search(q)),
    "distinct_count_scope": lambda q: bool(cc.DISTINCT_COUNT_RE.search(q) and cc.DRUG_RE.search(q)),
    "timestamp_field": lambda q: bool(cc.WHEN_RE.match(q)),
}


def read_csv(path):
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def fetch_workspace(patient_fhir_id):
    """Patient-wide retrieval of every supported type, as complete get_resources_by_patient_fhir_id calls."""
    workspace, failed = {}, []
    for rtype in supported_types:
        result = get_resources_by_patient_fhir_id(rtype, patient_fhir_id)
        if not isinstance(result, dict) or "error" in result:
            failed.append(rtype)
            continue
        for key, items in result.items():
            if isinstance(items, list):
                workspace.setdefault(key, []).extend(items)
    return workspace, failed


def gold_ids(row):
    try:
        ids = ast.literal_eval(row["true_fhir_ids"]) if row.get("true_fhir_ids") else {}
    except (ValueError, SyntaxError):
        ids = {}
    return [f"{rtype}/{rid}" for rtype, items in ids.items() for rid in items]


def as_answer(values, template_id):
    if template_id == "T001" and values:
        values = [int(float(v)) for v in values]
    if not values:
        return None
    return values[0] if len(values) == 1 else list(values)


def run_checks(question, draft, workspace, complete_types):
    """Fired check names per check function (issue text can carry record values, so it is not kept)."""
    fired = []
    for issue in cc.convention_issues(question, draft, workspace, complete_types=complete_types):
        fired.append(issue["check"] if issue["type"] != "check_error" else f"{issue['check']}:error")
    return sorted(set(fired))


def records(workspace, keys, rtype):
    by_id = cc._by_id(workspace)
    return [by_id[k] for k in keys if k.startswith(rtype + "/") and k in by_id]


def without(workspace, drop):
    return {rtype: [r for r in items if f"{rtype}/{r.get('id')}" not in drop] for rtype, items in workspace.items()}


def wrong_draft(template_id, gold, workspace, cited):
    """(draft, workspace) for the audit's typical wrong answer, or (None, None) when the template has none."""
    by_id = cc._by_id(workspace)
    if template_id == "T054":
        tests = records(workspace, cited, "Observation")
        return ({"answer": _display(tests[0].get("code")), "contributing_ids": cited}, workspace) if tests else (None, None)
    if template_id in ("T051", "T052"):
        requests = [k for k in cited if k.startswith("MedicationRequest/")]
        items = gold["answer"] if isinstance(gold["answer"], list) else [gold["answer"]]
        return ({"answer": items[0], "contributing_ids": requests[:1]}, workspace) if requests else (None, None)
    if template_id == "T001":
        return {"answer": PREVIEW_IDS, "contributing_ids": []}, workspace
    if template_id == "T043":
        encounters = records(workspace, cited, "Encounter")
        if not encounters:
            return None, None
        klass = encounters[0].get("class") or {}
        return {"answer": klass.get("display") or klass.get("code"), "contributing_ids": cited}, workspace
    if template_id == "T055":
        children = {cc._ref(m) for t in records(workspace, cited, "Observation") for m in t.get("hasMember") or []}
        parents = [k for k in cited if k not in children]
        return {**gold, "contributing_ids": parents}, without(workspace, children)
    if template_id == "T064":
        tests = records(workspace, cited, "Observation")
        if not tests:
            return None, None
        children = [by_id[cc._ref(m)] for m in tests[0].get("hasMember") or [] if cc._ref(m) in by_id]
        timed = [c for c in children if c.get("effectiveDateTime") and c.get("effectiveDateTime") != tests[0].get("effectiveDateTime")]
        if timed:
            return {"answer": timed[0]["effectiveDateTime"], "contributing_ids": [f"Observation/{timed[0]['id']}"]}, workspace
        return {"answer": str(gold["answer"])[:10], "contributing_ids": cited}, workspace
    if template_id == "T073":
        requests = records(workspace, cited, "MedicationRequest")
        authored = requests[0].get("authoredOn") if requests else None
        return ({"answer": authored, "contributing_ids": cited}, workspace) if authored else (None, None)
    return None, None


def probes(template_id, values, workspace, cited):
    """Booleans/counts answering the hand-off §3a questions; never the values themselves."""
    by_id = cc._by_id(workspace)
    norm_gold = {cc._norm(v) for v in values}
    out = {}
    if template_id == "T054":
        tests = records(workspace, cited, "Observation")
        specimens = [by_id.get(cc._ref(t.get("specimen"))) for t in tests if t.get("specimen")]
        specimens = [s for s in specimens if s]
        first = [cc._norm(((s.get("type") or {}).get("coding") or [{}])[0].get("display", "")) for s in specimens]
        out.update(cited_test_has_specimen_ref=any(t.get("specimen") for t in tests),
                   specimen_resolved=bool(specimens),
                   gold_is_cited_specimen_type_coding0_display=bool(first) and norm_gold <= set(first),
                   gold_is_any_specimen_type_display=norm_gold <= {cc._norm(_display(s.get("type"))) for s in workspace.get("Specimen", [])},
                   gold_is_cited_test_code_display=bool(tests) and norm_gold <= {cc._norm(_display(t.get("code"))) for t in tests})
    elif template_id == "T055":
        tests = records(workspace, cited, "Observation")
        members = {cc._ref(m) for t in tests for m in t.get("hasMember") or []}
        derived = {k for k in cited if any(cc._ref(d) in cited for d in (by_id.get(k) or {}).get("derivedFrom") or [])}
        child_names = set()
        for key in members:
            child = by_id.get(key) or {}
            for field in ("code", "valueCodeableConcept"):
                if child.get(field):
                    child_names.add(cc._norm(_display(child[field])))
        out.update(cited_tests_with_hasMember=sum(1 for t in tests if t.get("hasMember")),
                   gold_children_via_hasMember=len(members & set(cited)),
                   gold_children_via_derivedFrom=len(derived),
                   gold_organisms_found_in_hasMember_children=bool(norm_gold) and norm_gold <= child_names)
    elif template_id == "T043":
        encounters = records(workspace, cited, "Encounter")
        priority = {cc._norm(c.get(k)) for e in encounters for c in (e.get("priority") or {}).get("coding") or []
                    for k in ("display", "code") if c.get(k)}
        paths = sorted({p for e in encounters for p in _paths_with(e, norm_gold)})
        out.update(gold_in_priority_coding=bool(priority) and norm_gold <= priority,
                   gold_string_paths_in_encounter=paths)
    elif template_id in ("T051", "T052"):
        requests = records(workspace, cited, "MedicationRequest")
        minutes = {cc._minute(cc._request_start(r)) for r in requests} - {None}
        encounters = {cc._ref(r.get("encounter")) for r in requests}
        by_route = template_id == "T052"
        same = [r for r in workspace.get("MedicationRequest", []) if cc._minute(cc._request_start(r)) in minutes]
        scoped = [r for r in same if cc._ref(r.get("encounter")) in encounters]
        names = lambda rs: {cc._norm(n) for r in rs for n in cc._request_names(r, by_id)}
        # the check expands mixtures into components; the gold may name the mixture itself
        direct = lambda rs: {cc._norm(_medication_name(by_id[cc._ref(r.get("medicationReference"))]))
                             if cc._ref(r.get("medicationReference")) in by_id else cc._norm(_display(r.get("medicationCodeableConcept")))
                             for r in rs}
        out.update(gold_distinct_start_minutes=len(minutes), gold_requests=len(requests),
                   tie_requests_same_minute=len(same), tie_requests_same_minute_and_encounter=len(scoped),
                   gold_set_equals_tie_names_encounter_scoped=bool(scoped) and names(scoped) == norm_gold,
                   gold_set_equals_tie_names_unscoped=bool(same) and names(same) == norm_gold,
                   gold_set_equals_tie_unexpanded_names=bool(scoped) and direct(scoped) == norm_gold,
                   gold_set_within_expanded_tie_names=bool(scoped) and norm_gold <= names(scoped),
                   gold_set_size=len(norm_gold), expanded_tie_name_count=len(names(scoped)),
                   gold_names_covered_expanded_or_direct=len(norm_gold & (names(scoped) | direct(scoped))),
                   gold_names_covered_by_gold_requests=len(norm_gold & (names(requests) | direct(requests))),
                   gold_requests_unique=len({r['id'] for r in requests}))
        if by_route:
            out["route_filter_note"] = "tie names include every route; gold filters one route string"
    elif template_id == "T001":
        requests = workspace.get("MedicationRequest", [])
        all_names = {cc._norm(n) for r in requests for n in cc._request_names(r, by_id)}
        gold_reqs = records(workspace, cited, "MedicationRequest")
        gold_names = {cc._norm(n) for r in gold_reqs for n in cc._request_names(r, by_id)}
        count = int(float(values[0])) if values else None
        out.update(patient_requests=len(requests), patient_requests_exceed_preview=len(requests) > PREVIEW_IDS,
                   distinct_names_full_closure=len(all_names),
                   gold_ids_distinct_names_equals_gold=count is not None and len(gold_names) == count,
                   gold_count_exceeds_preview=count is not None and count > PREVIEW_IDS)
    elif template_id in ("T064", "T073"):
        if template_id == "T064":
            fields = {cc._minute(r.get("effectiveDateTime")) for r in records(workspace, cited, "Observation")}
        else:
            fields = {cc._minute(cc._request_start(r)) for r in records(workspace, cited, "MedicationRequest")}
        out["gold_minute_equals_cited_field"] = bool(values) and cc._minute(str(values[0])) in fields
    return out


def _paths_with(node, targets, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _paths_with(value, targets, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for value in node:
            yield from _paths_with(value, targets, path + "[]")
    elif isinstance(node, str) and cc._norm(node) in targets:
        yield path


def validate(question_ids=None, limit=None):
    audit = read_csv(ROOT_CAUSES)
    if question_ids:
        audit = [r for r in audit if r["question_id"] in question_ids]
    if limit:
        audit = audit[:limit]
    dataset = {r["question_id"]: r for r in read_csv(DATASET) if r["question_id"] in {a["question_id"] for a in audit}}
    workspaces, rows = {}, []
    for entry in audit:
        qid, template_id = entry["question_id"], entry["template_id"]
        row = dataset[qid]
        patient = row["patient_fhir_id"]
        if patient not in workspaces:
            workspaces[patient] = fetch_workspace(patient)
        workspace, failed = workspaces[patient]
        question = row["question"]
        values = parse_reference(row["true_answer"])
        cited = gold_ids(row)
        gold = {"answer": as_answer(values, template_id), "contributing_ids": cited}
        complete = set(supported_types) - set(failed)
        expected = EXPECTED_CHECK.get(template_id)
        record = {
            "question_id": qid, "template_id": template_id, "set": entry["set"], "root_cause": entry["root_cause"],
            "retrieval_failures": failed, "gold_ids_resolved": sum(k in cc._by_id(workspace) for k in cited),
            "gold_ids": len(cited), "expected_check": expected,
            "expected_gate_matches": GATES[expected](question) if expected else None,
            "fires_on_gold": run_checks(question, gold, workspace, complete) if gold["answer"] is not None else [],
            "fires_on_gold_without_complete_types": run_checks(question, gold, workspace, ())
            if gold["answer"] is not None else [],
        }
        draft, wrong_ws = wrong_draft(template_id, gold, workspace, cited) if gold["answer"] is not None else (None, None)
        record["wrong_constructed"] = draft is not None
        record["fires_on_wrong"] = run_checks(question, draft, wrong_ws, () if template_id == "T001" else complete) \
            if draft is not None else None
        record["probes"] = probes(template_id, values, workspace, cited)
        rows.append(record)
        print(f"{qid} {template_id:5s} gate={record['expected_gate_matches']!s:5s} "
              f"gold={','.join(record['fires_on_gold']) or '-':28s} "
              f"wrong={','.join(record['fires_on_wrong'] or []) or ('-' if draft else 'n/a')}", flush=True)
    return rows


def summarize(rows):
    checks = collections.defaultdict(lambda: {"fires_on_gold": 0, "expected_rows": 0, "gate_matches": 0,
                                              "fires_on_wrong": 0, "wrong_constructed": 0})
    for r in rows:
        for name in r["fires_on_gold"]:
            checks[name]["fires_on_gold"] += 1
        expected = r["expected_check"]
        if expected:
            entry = checks[expected]
            entry["expected_rows"] += 1
            entry["gate_matches"] += bool(r["expected_gate_matches"])
            if r["wrong_constructed"]:
                entry["wrong_constructed"] += 1
                entry["fires_on_wrong"] += expected in (r["fires_on_wrong"] or [])
    templates = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        for key, value in r["probes"].items():
            if isinstance(value, bool):
                templates[r["template_id"]][f"{key}={value}"] += 1
    return {"questions": len(rows), "checks": dict(checks),
            "probe_counts": {t: dict(c) for t, c in sorted(templates.items())},
            "check_errors": sorted({n for r in rows for n in r["fires_on_gold"] + (r["fires_on_wrong"] or [])
                                    if n.endswith(":error")})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--question_ids", default="", help="comma-separated subset of root_causes.csv ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default="output/diagnosis/convention_checks_live.json")
    args = parser.parse_args(argv)
    ids = {i.strip() for i in args.question_ids.split(",") if i.strip()} or None
    rows = validate(ids, args.limit)
    summary = summarize(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump({"summary": summary, "rows": rows}, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
