"""Revision tracking (P3-22): what revisions changed, unsupported claims, empty answers, recovery.

Run through `python scripts/run_experiments.py revisions ...`, which locates
and checks each arm's result file; only current arms with every question
attempted are used.
Answers are scored with the typed scorer and never written or printed: the
per-question log holds only question IDs and flags.

- Revisions. Each tool-equipped run saves its first draft in `revision`.
  Scoring the draft and the released answer gives right->right, right->wrong,
  wrong->right, and wrong->wrong. A released answer is what is evaluated, so
  a correct draft that is withheld as unverified counts as right->wrong.
  Counts use the first execution attempt, excluding failed attempts from
  transitions. Failed attempts still count as wrong in accuracy.
- Unsupported claims. A released answer with no evidence, or with
  a citation that does not match its source record. Citations are checked
  only when an evidence source (FHIR store or local NDJSON) is given. An
  independently established empty result needs no nonexistent citation.
  Citation checks do not prove completeness of the supporting evidence.
- Empty answers. Questions whose reference is empty ([] or [[None]]): the
  share answered correctly (abstained, or said none/zero). False abstentions
  are abstentions on questions that do have a reference answer.
- Recovery. On the omitted packet, whether the removed record was fetched
  back, reached the answer workspace, was cited or reported as used, and
  whether the answer was right, also among questions the
  same system got right from the intact packet.
"""
import csv
import json
import os

from utils.answer_scorer import is_abstention, score_answer
from utils.evidence_scorer import check_citation, get_cited_resource, parse_citation
from scripts.variability import attempts, first_attempt, is_failed

TRANSITIONS = ("right_to_right", "right_to_wrong", "wrong_to_right", "wrong_to_wrong")
LOG_COLUMNS = ["arm", "question_id", "family", "attempt_count", "failed_attempts", "error", "answered", "abstained",
               "empty_verified", "correct", "reference_empty", "has_draft", "draft_correct",
               "transition", "revised", "repair_role", "evidence_cited", "evidence_invalid", "unsupported",
               "omitted_resource", "omitted_fetched", "omitted_retrieved", "omitted_cited", "omitted_used",
               "omitted_verifier_only", "omitted_recovered"]


def _score(answer, reference):
    return score_answer(answer, reference["true_answer"], reference["question"], reference.get("template", ""))


def _cites(evidence, rtype, rid, workspace, resources):
    unknown = False
    for citation in evidence or []:
        parsed = parse_citation(citation)
        if not parsed or parsed[1] != rid:
            continue
        if parsed[0] == rtype:
            return True
        if parsed[0] is None:
            source = get_cited_resource(resources, None, rid) if resources is not None else None
            types = {kind for kind, ids in workspace.items() if isinstance(ids, list) and rid in ids}
            resolved = ((source or {}).get("resourceType") if resources is not None
                        else next(iter(types)) if len(types) == 1 else None)
            if resolved == rtype:
                return True
            unknown |= resolved is None
    return None if unknown else False


def _empty_value(value):
    return value is None or value is False or value == [] or (
        isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0)


def _bus_payloads(revision, sender, kind):
    messages = revision.get("messages")
    return [message["payload"] for message in messages or []
            if isinstance(message, dict) and message.get("sender") == sender
            and message.get("recipient") == "controller" and message.get("kind") == kind
            and isinstance(message.get("payload"), dict)] if isinstance(messages, list) else []


def _trace_checks(row):
    """Recover successful fetches and the final verifier's actual tool result."""
    fetched, verifier_fetched = set(), set()
    role, saw_verifier, id_trace_complete = None, False, True
    independently_fetched, computation = False, None
    trace = row.get("trace")
    if not isinstance(trace, list):
        return fetched, None, None, False
    for message in trace:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "system":
            text = message.get("content") or ""
            role = "verification" if isinstance(text, str) and text.startswith("You are the verification agent ") else None
            independently_fetched, computation = False, None
            if role:
                saw_verifier = True
            continue
        if message.get("role") != "tool":
            continue
        name = message.get("name")
        retrieval = name in ("get_resources_by_patient_fhir_id", "get_resources_by_resource_id")
        if role and (retrieval or name == "execute_python_code"):
            computation = None
            if retrieval:
                independently_fetched = False
        try:
            output = json.loads(message.get("content") or "")
        except (ValueError, TypeError):
            if retrieval:
                id_trace_complete = False
            continue
        if not isinstance(output, dict) or "error" in output:
            continue
        if retrieval and isinstance(output.get("ids"), list):
            ids = {rid for rid in output["ids"] if isinstance(rid, str)}
            fetched.update(ids)
            if role:
                verifier_fetched.update(ids)
                independently_fetched = True
        elif retrieval:
            id_trace_complete = False
        elif role and name == "execute_python_code" and independently_fetched and "answer" in output:
            computation = output
    return fetched, verifier_fetched if saw_verifier else None, computation, id_trace_complete


def _empty_verified(answer, revision, computation):
    if not _empty_value(answer):
        return False
    verdicts = _bus_payloads(revision, "verification", "verdict")
    if "final_check_issues" in revision and revision["final_check_issues"] != []:
        return False
    if not verdicts or revision.get("final_check_issues") != []:
        return None
    verdict = verdicts[-1]
    if verdict.get("issues") != [] or "recomputed_answer" not in verdict:
        return False
    if computation is None:
        return None
    return (_empty_value(verdict["recomputed_answer"]) and answer == verdict["recomputed_answer"]
            and answer == computation["answer"])


def _used_ids(row, revision):
    candidates = _bus_payloads(revision, "analysis", "draft")
    candidate = candidates[-1] if candidates else revision.get("final")
    if not isinstance(candidate, dict) or candidate.get("answer") != row.get("agent_answer"):
        return None
    ids = candidate.get("contributing_ids")
    return set(ids) if isinstance(ids, list) and all(isinstance(rid, str) for rid in ids) else None


def _joint(*flags):
    return False if False in flags else None if None in flags else True


def question_record(arm, row, reference, perturbation=None, resources=None):
    """Flags for the first execution attempt of one question of a validated arm."""
    history = attempts(row)
    row = first_attempt(row)
    failed = is_failed(row)
    answer = None if failed else row.get("agent_answer")
    final = _score(answer, reference)
    correct = 0 if failed else final["typed_correct"]
    evidence = row.get("evidence") if isinstance(row.get("evidence"), list) else []
    revision = row.get("revision") if isinstance(row.get("revision"), dict) else {}
    draft = revision.get("draft") if isinstance(revision.get("draft"), dict) else None
    draft = draft if draft is not None and "answer" in draft and not is_failed(draft) else None
    fetched, verifier_fetched, computation, id_trace_complete = _trace_checks(row)
    verified_empty = False if failed else _empty_verified(answer, revision, computation)
    abstained = not failed and is_abstention(answer) and verified_empty is not True
    answered = not failed and (not is_abstention(answer) or verified_empty is True)
    record = {"arm": arm, "question_id": reference["question_id"], "family": reference.get("family"),
              "attempt_count": len(history), "failed_attempts": sum(is_failed(attempt) for attempt in history),
              "error": failed, "answered": answered, "abstained": abstained, "empty_verified": verified_empty,
              "correct": correct,
              "reference_empty": final["answer_type"] == "empty", "has_draft": draft is not None,
              "draft_correct": None, "transition": None, "revised": bool(revision.get("revised")),
              "repair_role": revision.get("repair_role"), "evidence_cited": len(evidence),
              "evidence_invalid": None, "unsupported": None,
              "omitted_resource": None, "omitted_fetched": None, "omitted_retrieved": None,
              "omitted_cited": None, "omitted_used": None, "omitted_verifier_only": None,
              "omitted_recovered": None}
    if draft is not None:
        record["draft_correct"] = _score(draft.get("answer"), reference)["typed_correct"]
        if not failed:
            record["transition"] = (f"{'right' if record['draft_correct'] else 'wrong'}_to_"
                                    f"{'right' if correct else 'wrong'}")
    record["evidence_invalid"] = sum(check_citation(c, resources) != "valid" if resources is not None
                                     else parse_citation(c) is None for c in evidence)
    if record["answered"]:
        record["unsupported"] = (not evidence and verified_empty is not True) or bool(record["evidence_invalid"])
        if not evidence and verified_empty is None and _bus_payloads(revision, "verification", "verdict"):
            record["unsupported"] = None
    if perturbation is not None:
        rtype, rid = perturbation["removed"]["resource_type"], perturbation["removed"]["id"]
        workspace = row.get("agent_fhir_resources") if isinstance(row.get("agent_fhir_resources"), dict) else None
        record["omitted_resource"] = f"{rtype}/{rid}"
        workspace_ids = workspace.get(rtype, []) if workspace is not None else None
        record["omitted_retrieved"] = rid in workspace_ids if isinstance(workspace_ids, list) else None
        record["omitted_fetched"] = (True if record["omitted_retrieved"] or f"{rtype}/{rid}" in fetched
                                     else False if id_trace_complete and isinstance(row.get("trace"), list) else None)
        record["omitted_cited"] = False if failed else _cites(evidence, rtype, rid, workspace or {}, resources)
        used = _used_ids(row, revision)
        record["omitted_used"] = False if not answered else f"{rtype}/{rid}" in used if used is not None else None
        record["omitted_verifier_only"] = (f"{rtype}/{rid}" in verifier_fetched and not record["omitted_retrieved"]
                                            if verifier_fetched is not None and workspace is not None else None)
        record["omitted_recovered"] = _joint(record["omitted_retrieved"], record["omitted_used"], bool(correct))
    return record


def _rate(numerator, denominator):
    return {"n": denominator, "count": numerator, "rate": numerator / denominator if denominator else None}


def _flag_rate(records, key):
    known = [record[key] for record in records if record[key] is not None]
    return _rate(sum(bool(flag) for flag in known), len(known))


def summarize_arm(records, citations_checked):
    with_draft = [r for r in records if r["has_draft"]]
    transitions = [r for r in with_draft if r["transition"] is not None]
    revised = [r for r in transitions if r["revised"]]
    answered = [r for r in records if r["answered"]]
    empty = [r for r in records if r["reference_empty"]]
    non_empty = [r for r in records if not r["reference_empty"]]
    summary = {
        "questions": len(records),
        "failed_first_attempts": sum(r["error"] for r in records),
        "attempts": sum(r["attempt_count"] for r in records),
        "failed_attempts": sum(r["failed_attempts"] for r in records),
        "accuracy": _rate(sum(r["correct"] for r in records), len(records)),
        "transitions": {t: sum(r["transition"] == t for r in transitions) for t in TRANSITIONS} if with_draft else None,
        "transitions_revised": {t: sum(r["transition"] == t for r in revised) for t in TRANSITIONS}
        if with_draft else None,
        "runs_with_draft": len(with_draft),
        "runs_scored_transitions": len(transitions),
        "transition_excluded_errors": sum(r["error"] for r in with_draft),
        "runs_revised": len(revised),
        "unsupported": {**_flag_rate(answered, "unsupported"),
                        "unknown": sum(r["unsupported"] is None for r in answered),
                        "no_evidence": sum(r["evidence_cited"] == 0 and r["empty_verified"] is not True for r in answered),
                        "verified_empty": sum(r["empty_verified"] is True for r in records),
                        "invalid_citation": (sum(bool(r["evidence_invalid"]) for r in answered)
                                             if citations_checked else None),
                        "citations_checked": citations_checked,
                        "support_completeness_checked": False,
                        "lower_bound": True},
        "empty_reference": _rate(sum(r["correct"] for r in empty), len(empty)),
        "empty_reference_verified": _rate(sum(r["correct"] and r["empty_verified"] is True for r in empty), len(empty)),
        "empty_verification_unknown": sum(r["empty_verified"] is None for r in records),
        "false_abstention": _rate(sum(r["abstained"] for r in non_empty), len(non_empty)),
    }
    for kind in ("transitions", "transitions_revised"):
        if summary[kind]:
            summary[kind]["net"] = summary[kind]["wrong_to_right"] - summary[kind]["right_to_wrong"]
    return summary


def summarize_recovery(omitted, intact=None):
    """Recovery on the omitted packet; `intact` is the same system's intact-arm records, if complete."""
    summary = {"questions": len(omitted),
               "fetched": _flag_rate(omitted, "omitted_fetched"),
               "retrieved": _flag_rate(omitted, "omitted_retrieved"),
               "cited": _flag_rate(omitted, "omitted_cited"),
               "used": _flag_rate(omitted, "omitted_used"),
               "verifier_only": _flag_rate(omitted, "omitted_verifier_only"),
               "recovered": _flag_rate(omitted, "omitted_recovered"),
               "unknown": {key: sum(r[f"omitted_{key}"] is None for r in omitted)
                           for key in ("fetched", "retrieved", "cited", "used", "verifier_only", "recovered")},
               "correct": _rate(sum(r["correct"] for r in omitted), len(omitted)),
               "correct_when_retrieved": _rate(sum(r["correct"] for r in omitted if r["omitted_retrieved"]),
                                               sum(bool(r["omitted_retrieved"]) for r in omitted)),
               "correct_given_intact_correct": None}
    if intact is not None:
        right = {r["question_id"] for r in intact if r["correct"]}
        kept = [r for r in omitted if r["question_id"] in right]
        summary["correct_given_intact_correct"] = _rate(sum(r["correct"] for r in kept), len(kept))
    return summary


def write_revisions(report, records, directory):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "revisions.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1, allow_nan=False)
        stream.write("\n")
    with open(os.path.join(directory, "revision_log.csv"), "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def _pct(rate):
    return "n/a" if rate is None or rate["rate"] is None else f"{rate['count']}/{rate['n']} ({rate['rate']:.0%})"


def print_revisions(report):
    if not report["arms"]:
        print("No analyzable arms yet.")
    for name, s in report["arms"].items():
        print(f"\n{name}: accuracy {_pct(s['accuracy'])}")
        print(f"  first attempts failed {s['failed_first_attempts']}; {s['attempts']} execution attempts total")
        if s["transitions"]:
            t, r = s["transitions"], s["transitions_revised"]
            print(f"  draft -> released: right->wrong {t['right_to_wrong']}, wrong->right {t['wrong_to_right']} "
                  f"(net {t['net']:+d}) over {s['runs_scored_transitions']} scored drafts "
                  f"({s['transition_excluded_errors']} failed attempts excluded)")
            print(f"  repaired runs: {s['runs_revised']}; right->wrong {r['right_to_wrong']}, "
                  f"wrong->right {r['wrong_to_right']} (net {r['net']:+d})")
        u = s["unsupported"]
        invalid = "citations not checked" if not u["citations_checked"] else f"{u['invalid_citation']} invalid citation"
        print(f"  unsupported answers (lower bound) {_pct(u)}: {u['no_evidence']} without evidence, {invalid}; "
              f"{u['verified_empty']} independently verified empty, {u['unknown']} unknown")
        print(f"  empty-reference questions correct {_pct(s['empty_reference'])}; "
              f"false abstentions {_pct(s['false_abstention'])}")
    for system, s in report["recovery"].items():
        print(f"\nrecovery ({system}, omitted packet): fetched {_pct(s['fetched'])}, "
              f"answer workspace {_pct(s['retrieved'])}, cited {_pct(s['cited'])}, used {_pct(s['used'])}")
        print(f"  verifier only {_pct(s['verifier_only'])}; recovered and correct {_pct(s['recovered'])}, "
              f"correct {_pct(s['correct'])}; missing metadata {s['unknown']}")
        print(f"  correct when fetched back {_pct(s['correct_when_retrieved'])}; "
              f"correct among intact-correct {_pct(s['correct_given_intact_correct'])}")
    for name, state in report["skipped"].items():
        print(f"Skipped {name}: {state}")
