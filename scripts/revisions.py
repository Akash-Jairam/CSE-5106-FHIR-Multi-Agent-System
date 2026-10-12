"""Revision tracking (P3-22): what revisions changed, unsupported claims, empty answers, recovery.

Run through `python scripts/run_experiments.py revisions ...`, which locates
and checks each arm's result file; only complete, current arms are used.
Answers are scored with the typed scorer and never written or printed: the
per-question log holds only question IDs and flags.

- Revisions. Each tool-equipped run saves its first draft in `revision`.
  Scoring the draft and the released answer gives right->right, right->wrong,
  wrong->right, and wrong->wrong. A released answer is what is evaluated, so
  a correct draft that is withheld as unverified counts as right->wrong.
  Counts are given over all runs with a draft and over repaired runs only.
- Unsupported claims. A released (non-null) answer with no evidence, or with
  a citation that does not match its source record. Citations are checked
  only when an evidence source (FHIR store or local NDJSON) is given.
- Empty answers. Questions whose reference is empty ([] or [[None]]): the
  share answered correctly (abstained, or said none/zero). False abstentions
  are abstentions on questions that do have a reference answer.
- Recovery. On the omitted packet, whether the removed record was fetched
  back, cited, and whether the answer was right, also among questions the
  same system got right from the intact packet.
"""
import csv
import json
import os

from utils.answer_scorer import is_abstention, score_answer
from utils.evidence_scorer import check_citation, parse_citation

TRANSITIONS = ("right_to_right", "right_to_wrong", "wrong_to_right", "wrong_to_wrong")
LOG_COLUMNS = ["arm", "question_id", "family", "answered", "correct", "reference_empty", "has_draft", "draft_correct",
               "transition", "revised", "repair_role", "evidence_cited", "evidence_invalid", "unsupported",
               "omitted_resource", "omitted_retrieved", "omitted_cited"]


def _errored(row):
    error = row.get("error")
    return error is not None and str(error).strip() not in ("", "nan")


def _score(answer, reference):
    return score_answer(answer, reference["true_answer"], reference["question"], reference.get("template", ""))


def _cites(evidence, rtype, rid):
    for citation in evidence or []:
        parsed = parse_citation(citation)
        if parsed and parsed[1] == rid and parsed[0] in (rtype, None):
            return True
    return False


def question_record(arm, row, reference, perturbation=None, resources=None):
    """Flags for one question of one complete arm."""
    answer = None if _errored(row) else row.get("agent_answer")
    final = _score(answer, reference)
    correct = 0 if _errored(row) else final["typed_correct"]
    evidence = row.get("evidence") if isinstance(row.get("evidence"), list) else []
    revision = row.get("revision") if isinstance(row.get("revision"), dict) else {}
    draft = revision.get("draft") if isinstance(revision.get("draft"), dict) else None
    record = {"arm": arm, "question_id": reference["question_id"], "family": reference.get("family"),
              "answered": answer is not None and not is_abstention(answer), "correct": correct,
              "reference_empty": final["answer_type"] == "empty", "has_draft": draft is not None,
              "draft_correct": None, "transition": None, "revised": bool(revision.get("revised")),
              "repair_role": revision.get("repair_role"), "evidence_cited": len(evidence),
              "evidence_invalid": None, "unsupported": None,
              "omitted_resource": None, "omitted_retrieved": None, "omitted_cited": None}
    if draft is not None:
        record["draft_correct"] = _score(draft.get("answer"), reference)["typed_correct"]
        record["transition"] = (f"{'right' if record['draft_correct'] else 'wrong'}_to_"
                                f"{'right' if correct else 'wrong'}")
    if resources is not None:
        record["evidence_invalid"] = sum(check_citation(c, resources) != "valid" for c in evidence)
    if record["answered"]:
        record["unsupported"] = not evidence or bool(record["evidence_invalid"])
    if perturbation is not None:
        rtype, rid = perturbation["removed"]["resource_type"], perturbation["removed"]["id"]
        fetched = row.get("agent_fhir_resources") if isinstance(row.get("agent_fhir_resources"), dict) else {}
        record["omitted_resource"] = f"{rtype}/{rid}"
        record["omitted_retrieved"] = rid in (fetched.get(rtype) or [])
        record["omitted_cited"] = _cites(evidence, rtype, rid)
    return record


def _rate(numerator, denominator):
    return {"n": denominator, "count": numerator, "rate": numerator / denominator if denominator else None}


def summarize_arm(records, citations_checked):
    with_draft = [r for r in records if r["has_draft"]]
    revised = [r for r in with_draft if r["revised"]]
    answered = [r for r in records if r["answered"]]
    empty = [r for r in records if r["reference_empty"]]
    non_empty = [r for r in records if not r["reference_empty"]]
    summary = {
        "questions": len(records),
        "accuracy": _rate(sum(r["correct"] for r in records), len(records)),
        "transitions": {t: sum(r["transition"] == t for r in with_draft) for t in TRANSITIONS} if with_draft else None,
        "transitions_revised": {t: sum(r["transition"] == t for r in revised) for t in TRANSITIONS}
        if with_draft else None,
        "runs_with_draft": len(with_draft),
        "runs_revised": len(revised),
        "unsupported": {**_rate(sum(bool(r["unsupported"]) for r in answered), len(answered)),
                        "no_evidence": sum(r["evidence_cited"] == 0 for r in answered),
                        "invalid_citation": (sum(bool(r["evidence_invalid"]) for r in answered)
                                             if citations_checked else None),
                        "citations_checked": citations_checked},
        "empty_reference": _rate(sum(r["correct"] for r in empty), len(empty)),
        "false_abstention": _rate(sum(not r["answered"] for r in non_empty), len(non_empty)),
    }
    for kind in ("transitions", "transitions_revised"):
        if summary[kind]:
            summary[kind]["net"] = summary[kind]["wrong_to_right"] - summary[kind]["right_to_wrong"]
    return summary


def summarize_recovery(omitted, intact=None):
    """Recovery on the omitted packet; `intact` is the same system's intact-arm records, if complete."""
    summary = {"questions": len(omitted),
               "retrieved": _rate(sum(r["omitted_retrieved"] for r in omitted), len(omitted)),
               "cited": _rate(sum(r["omitted_cited"] for r in omitted), len(omitted)),
               "correct": _rate(sum(r["correct"] for r in omitted), len(omitted)),
               "correct_when_retrieved": _rate(sum(r["correct"] for r in omitted if r["omitted_retrieved"]),
                                               sum(r["omitted_retrieved"] for r in omitted)),
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
        print("No complete arms yet.")
    for name, s in report["arms"].items():
        print(f"\n{name}: accuracy {_pct(s['accuracy'])}")
        if s["transitions"]:
            t, r = s["transitions"], s["transitions_revised"]
            print(f"  draft -> released: right->wrong {t['right_to_wrong']}, wrong->right {t['wrong_to_right']} "
                  f"(net {t['net']:+d}) over {s['runs_with_draft']} drafts")
            print(f"  repaired runs: {s['runs_revised']}; right->wrong {r['right_to_wrong']}, "
                  f"wrong->right {r['wrong_to_right']} (net {r['net']:+d})")
        u = s["unsupported"]
        invalid = "citations not checked" if not u["citations_checked"] else f"{u['invalid_citation']} invalid citation"
        print(f"  unsupported answers {_pct(u)}: {u['no_evidence']} without evidence, {invalid}")
        print(f"  empty-reference questions correct {_pct(s['empty_reference'])}; "
              f"false abstentions {_pct(s['false_abstention'])}")
    for system, s in report["recovery"].items():
        print(f"\nrecovery ({system}, omitted packet): record fetched back {_pct(s['retrieved'])}, "
              f"cited {_pct(s['cited'])}, correct {_pct(s['correct'])}")
        print(f"  correct when fetched back {_pct(s['correct_when_retrieved'])}; "
              f"correct among intact-correct {_pct(s['correct_given_intact_correct'])}")
    for name, state in report["skipped"].items():
        print(f"Skipped {name}: {state}")
