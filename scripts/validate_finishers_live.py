"""Calibrate the V10 finishers on live FHIR records (design_v10.txt §5 todo 2). No model is called.

audit       For every docs/diagnosis_v8/root_causes.csv question of a finisher template
            (the patients of scripts/validate_convention_checks_live.py), run the
            finisher on two drafts: `gold` (the reference answer citing the reference
            FHIR ids) and `wrong` (the audit's typical wrong answer, built by
            validate_convention_checks_live.wrong_draft). A replacement on `gold` that
            scores wrong is harmful; a replacement on `wrong` that scores right is a fix.
candidates  Run the finisher on real candidates from finished arms (e.g. the B2 fresh
            single and the V9 team primary): released answer plus cited ids, against
            the live records, and count replaced wrong->right / right->wrong per
            template. This is the finisher's realised effect on those candidates
            (exploratory: the same development questions).

Reference answers are used only through the typed scorer's 0/1 and are never printed
or written; the report holds templates, outcomes, reasons and counts.

  python scripts/validate_finishers_live.py audit [--output output/diagnosis/finishers_live_audit.json]
  python scripts/validate_finishers_live.py candidates --arm single=<b2>/ordinary_single_agent.json \\
      --arm primary=<b2>/ordinary_multi_agent.json [--output output/diagnosis/finishers_live_candidates.json]
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.finishers import Fetcher, run_finisher
from scripts import validate_convention_checks_live as live
from scripts.replay_v10 import FINISHER_TEMPLATES, template_ids
from scripts.replay_v9_controller import DEFAULT_FAMILIES, DEFAULT_INPUT, _released, correct, load_references, load_rows
from utils import parse_question_context
from utils.answer_scorer import parse_reference


def _context(row):
    _, _, current_time = parse_question_context(f"Question: {row['question']}\nContext:\n{row['assumption']}")
    return {"question": row["question"], "patient_fhir_id": row["patient_fhir_id"], "current_time": current_time}


def _finish(context, draft):
    return run_finisher(context, draft, Fetcher(lambda: True, context["patient_fhir_id"], context["current_time"]))


def _outcome(record, reference, before):
    """Outcome label for one finisher run: replaced_<transition> or <finisher>[:reason]."""
    if record["finisher"] != "replaced":
        return record["finisher"] + (f":{record['reason']}" if record["reason"] else "")
    after = correct(record["result"], reference)
    return "replaced_" + {(1, 1): "right_to_right", (1, 0): "right_to_wrong",
                          (0, 1): "wrong_to_right", (0, 0): "wrong_to_wrong"}[(before, after)]


def audit(references):
    rows = [r for r in live.read_csv(live.ROOT_CAUSES) if r["template_id"] in FINISHER_TEMPLATES]
    dataset = {r["question_id"]: r for r in live.read_csv(live.DATASET) if r["question_id"] in {a["question_id"] for a in rows}}
    workspaces, table, records = {}, collections.defaultdict(collections.Counter), []
    for entry in rows:
        qid, template = entry["question_id"], entry["template_id"]
        row, reference = dataset[qid], references[qid]
        if row["patient_fhir_id"] not in workspaces:
            workspaces[row["patient_fhir_id"]] = live.fetch_workspace(row["patient_fhir_id"])
        workspace, _ = workspaces[row["patient_fhir_id"]]
        context = _context(row)
        cited = live.gold_ids(row)
        gold = {"answer": live.as_answer(parse_reference(row["true_answer"]), template), "contributing_ids": cited,
                "evidence": []}
        result = {"question_id": qid, "template_id": template}
        if gold["answer"] is not None:
            record = _finish(context, gold)
            result["gold"] = _outcome(record, reference, correct(gold["answer"], reference))
            table[template]["gold:" + result["gold"]] += 1
        wrong, _ = live.wrong_draft(template, gold, workspace, cited) if gold["answer"] is not None else (None, None)
        if wrong is not None:
            wrong = {"evidence": [], **wrong}
            record = _finish(context, wrong)
            result["wrong"] = _outcome(record, reference, correct(wrong.get("answer"), reference))
            table[template]["wrong:" + result["wrong"]] += 1
        records.append(result)
        print(f"{qid} {template} gold={result.get('gold', '-')} wrong={result.get('wrong', '-')}", flush=True)
    return {"rows": records, "by_template": {t: dict(sorted(c.items())) for t, c in sorted(table.items())}}


def candidates(arms, references, templates):
    dataset = {r["question_id"]: r for r in live.read_csv(live.DATASET)}
    report = {}
    for name, path in arms:
        table, total = collections.defaultdict(collections.Counter), collections.Counter()
        for qid, row in sorted(load_rows(path).items()):
            template = templates.get(qid)
            if template not in FINISHER_TEMPLATES:
                continue
            revision = row.get("revision") or {}
            if revision.get("primary") is not None and name != "single":
                candidate = {k: revision["primary"].get(k) for k in ("answer", "evidence", "contributing_ids")}
            else:
                candidate = {"answer": _released(row), "evidence": row.get("evidence") or [], "contributing_ids": []}
            reference = references[qid]
            before = correct(candidate["answer"], reference)
            outcome = _outcome(_finish(_context(dataset[qid]), candidate), reference, before)
            table[template][outcome] += 1
            total[outcome.split(":")[0]] += 1
            total["right_before"] += before
        report[name] = {"total": dict(sorted(total.items())),
                        "by_template": {t: dict(sorted(c.items())) for t, c in sorted(table.items())},
                        "net": total["replaced_wrong_to_right"] - total["replaced_right_to_wrong"]}
        print(name, json.dumps(report[name]["total"]), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["audit", "candidates"])
    parser.add_argument("--arm", action="append", default=[], help="candidates: name=path to a finished arm file")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    references = load_references(DEFAULT_INPUT, DEFAULT_FAMILIES)
    if args.command == "audit":
        report = audit(references)
        output = args.output or "output/diagnosis/finishers_live_audit.json"
    else:
        arms = [tuple(item.split("=", 1)) for item in args.arm]
        if not arms or any(len(arm) != 2 for arm in arms):
            parser.error("candidates needs --arm name=path")
        report = candidates(arms, references, template_ids())
        output = args.output or "output/diagnosis/finishers_live_candidates.json"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
