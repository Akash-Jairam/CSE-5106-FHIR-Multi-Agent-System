"""Offline, first-attempt comparisons across frozen architecture snapshots.

Unlike run_experiments status, this accepts different code fingerprints so
an archived baseline can be compared with a new design. It refuses partial
cohorts and mismatched model/budget/input identities. Never called by agents.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analysis import arm_records
from scripts.experiment_stats import clustered_mean, latency_summary, paired_difference
from scripts.variability import first_attempt, is_failed
from utils.answer_scorer import score_answer
from scripts.full_benchmark import iter_result_rows


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scoring_row(row):
    """Keep first-attempt scoring data without retaining multi-GB raw audits."""
    keys = ("question_id", "patient_fhir_id", "agent_answer", "tokens", "latency", "error", "budget")
    compact = {key: row.get(key) for key in keys}
    revision = row.get("revision")
    if isinstance(revision, dict):
        draft = revision.get("draft")
        compact["revision"] = {"draft": {"answer": draft.get("answer")} if isinstance(draft, dict) else None,
                               "revised": revision.get("revised")}
    history = row.get("attempt_history")
    if history is not None:
        if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
            raise ValueError("Attempt history must contain recorded result objects")
        compact["attempt_history"] = [scoring_row(attempt) for attempt in history]
    return compact


def compare(args):
    with open(args.input, newline="", encoding="utf-8") as stream:
        source = {r["question_id"]: r for r in csv.DictReader(stream)}
    with open(args.families, newline="", encoding="utf-8") as stream:
        families = {r["question_id"]: r for r in csv.DictReader(stream)}
    if args.full:
        expected = set(source)
    else:
        manifest = json.loads(Path(args.split_manifest).read_text())
        expected = set(manifest["splits"][args.split]["question_ids"])
    references = {qid: {"question": source[qid]["question"],
                        "true_answer": source[qid]["true_answer"],
                        "patient": source[qid]["patient_fhir_id"],
                        "template": families[qid]["template"],
                        "family": families[qid]["family"]} for qid in expected}
    records, arms, identities = {}, {}, {}
    binding = None
    for specification in args.arm:
        name, separator, filename = specification.partition("=")
        if not separator or not name or name in records:
            raise ValueError("Each --arm must have a unique name followed by = and its result path")
        path = Path(filename)
        indexed = {}
        for original in iter_result_rows(path):
            if not isinstance(original, dict) or original.get("question_id") in indexed:
                raise ValueError(f"{name}: result must contain unique question objects")
            indexed[original.get("question_id")] = scoring_row(original)
        rows = list(indexed.values())
        if len(rows) != len(expected):
            raise ValueError(f"{name}: result cohort has the wrong size")
        if len(indexed) != len(rows) or set(indexed) != expected:
            raise ValueError(f"{name}: result question IDs do not match the scheduled cohort")
        if any(first_attempt(row).get("latency") is None and not is_failed(first_attempt(row))
               for row in rows):
            raise ValueError(f"{name}: every scheduled question must have a first attempt")
        identity = json.loads(Path(str(path) + ".run.json").read_text())
        public_binding = {k: identity.get(k) for k in
                          ("model", "max_total_tokens", "max_tool_calls", "input_hash",
                           "split", "condition", "subset_hash", "repeat")}
        if binding is not None and public_binding != binding:
            raise ValueError(f"{name}: model, ceilings, or question/condition identity differs")
        binding = public_binding
        identities[name] = {"result_path": str(path.resolve()), "result_sha256": sha256(path),
                            "identity": identity}
        recs = records[name] = arm_records(indexed, references)
        family_reports = {}
        for family in ["all", *sorted({r["family"] for r in recs})]:
            selected = [r for r in recs if family == "all" or r["family"] == family]
            family_reports[family] = {"correct": sum(r["correct"] for r in selected),
                                      **clustered_mean([r["correct"] for r in selected],
                                                       [r["patient"] for r in selected],
                                                       n_boot=args.bootstrap, seed=5106)}
        first = [first_attempt(indexed[qid]) for qid in sorted(expected)]
        transitions = {"wrong_to_right": 0, "right_to_wrong": 0, "draft_correct": 0,
                       "drafts": 0, "semantic_repairs": 0}
        for qid in sorted(expected):
            row = first_attempt(indexed[qid])
            revision = row.get("revision") or {}
            draft = revision.get("draft")
            if not isinstance(draft, dict):
                continue
            ref = references[qid]
            draft_correct = score_answer(draft.get("answer"), ref["true_answer"],
                                         ref["question"], ref["template"])["typed_correct"]
            final_correct = next(r["correct"] for r in recs if r["question_id"] == qid)
            transitions["drafts"] += 1
            transitions["draft_correct"] += draft_correct
            transitions["wrong_to_right"] += not draft_correct and bool(final_correct)
            transitions["right_to_wrong"] += bool(draft_correct) and not final_correct
            transitions["semantic_repairs"] += bool(revision.get("revised"))
        tokens = [(r.get("tokens") or {}).get("total") for r in first]
        costs = [(r.get("tokens") or {}).get("cost") for r in first]
        arms[name] = {"accuracy": family_reports, "terminal_errors": sum(is_failed(r) for r in first),
                      "budget_stops": sum(bool((r.get("budget") or {}).get("stopped")) for r in first),
                      "mean_tokens": statistics.fmean(tokens) if None not in tokens else None,
                      "observed_cost": sum(costs) if None not in costs else None,
                      "latency": latency_summary([r.get("latency") for r in first]),
                      "transitions": transitions}
    comparisons = []
    names = list(records)
    for i, left in enumerate(names):
        for right in names[:i]:
            a, b = records[left], records[right]
            comparisons.append({"comparison": f"{left} - {right}",
                                "wins": sum(x["correct"] and not y["correct"] for x, y in zip(a, b)),
                                "losses": sum(y["correct"] and not x["correct"] for x, y in zip(a, b)),
                                **paired_difference([r["correct"] for r in a], [r["correct"] for r in b],
                                                    [r["patient"] for r in a],
                                                    n_boot=args.bootstrap, seed=5106)})
    return {"cohort": "full_released" if args.full else args.split,
            "selection_warning": "Fixed subset is reused for architecture development; gains on it are exploratory.",
            "questions": len(expected), "patients": len({r["patient"] for r in references.values()}),
            "question_ids_sha256": hashlib.sha256(json.dumps(sorted(expected)).encode()).hexdigest(),
            "dataset_sha256": sha256(args.input), "families_sha256": sha256(args.families),
            "scoring": "Unchanged typed scorer; first attempts, terminal errors incorrect; patient-block bootstrap",
            "identities": identities, "arms": arms, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--input", default=str(ROOT / "final_dataset/questions_answers_sql_fhir.csv"))
    parser.add_argument("--families", default=str(ROOT / "manifests/question_families.csv"))
    parser.add_argument("--split_manifest", default=str(ROOT / "manifests/split.json"))
    parser.add_argument("--split", default="held_out")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare(args)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, arm in report["arms"].items():
        overall = arm["accuracy"]["all"]
        print(f"{name}: {overall['correct']}/{overall['n']} ({overall['estimate']:.1%}), "
              f"errors={arm['terminal_errors']}, budget stops={arm['budget_stops']}")
    for comparison in report["comparisons"]:
        print(f"{comparison['comparison']}: {comparison['estimate']:+.1%} "
              f"95% CI [{comparison['ci_low']:+.1%}, {comparison['ci_high']:+.1%}], "
              f"wins/losses={comparison['wins']}/{comparison['losses']}")


if __name__ == "__main__":
    main()
