"""Offline, first-attempt comparisons across frozen architecture snapshots.

Unlike run_experiments status, this accepts different code fingerprints so
an archived baseline can be compared with a new design. It refuses partial
cohorts and mismatched model/budget/input identities. Never called by agents.
"""
import argparse
import csv
import hashlib
import json
import math
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
from scripts.evaluation_strata import load_evaluation_strata


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


def validate_full_identity(identity, name):
    required = ("model", "max_total_tokens", "max_tool_calls", "input_hash",
                "split", "condition", "subset_hash", "repeat")
    if not isinstance(identity, dict) or not all(k in identity for k in required):
        raise ValueError(f"{name}: explicit full-run identity bindings are required")
    if not isinstance(identity["model"], str) or not identity["model"].strip():
        raise ValueError(f"{name}: invalid full-run model identity")
    if any(type(identity[k]) is not int or identity[k] <= 0 for k in ("max_total_tokens", "max_tool_calls")):
        raise ValueError(f"{name}: invalid full-run ceiling identity")
    digest = identity["input_hash"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{name}: invalid full-run input identity")
    if any(identity[k] is not None for k in ("split", "condition", "subset_hash", "repeat")):
        raise ValueError(f"{name}: stratified/conditional identity is not a full ordinary run")


def validate_full_attempt(row, name):
    first = first_attempt(row)
    latency = first.get("latency")
    valid = lambda value: type(value) in (int, float) and math.isfinite(value) and value >= 0
    if (latency is None and not is_failed(first)) or (latency is not None and not valid(latency)):
        raise ValueError(f"{name}: invalid or missing first-attempt latency")
    tokens, budget = first.get("tokens"), first.get("budget")
    if (tokens is not None and not isinstance(tokens, dict)) or (budget is not None and not isinstance(budget, dict)):
        raise ValueError(f"{name}: malformed first-attempt usage/budget metadata")
    if any(value is not None and not valid(value) for value in ((tokens or {}).get("total"), (tokens or {}).get("cost"))):
        raise ValueError(f"{name}: invalid first-attempt token/cost metadata")


def compare(args):
    strata_path = getattr(args, "evaluation_strata", None)
    strata_digest = getattr(args, "evaluation_strata_sha256", None)
    if bool(strata_path) != bool(strata_digest) or (strata_path and not args.full):
        raise ValueError("--evaluation_strata and its frozen SHA256 must be supplied together with --full")
    if strata_path and getattr(args, "full_terminal_confirmed", False) is not True:
        raise ValueError("Full terminal/source/audit guard confirmation is required before strata scoring")
    with open(args.input, newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
        source = {r["question_id"]: r for r in source_rows}
    with open(args.families, newline="", encoding="utf-8") as stream:
        family_rows = list(csv.DictReader(stream))
        families = {r["question_id"]: r for r in family_rows}
    if strata_path and (len(source) != len(source_rows) or len(families) != len(family_rows)):
        raise ValueError("Full evaluation source/families must contain unique question IDs")
    if strata_path and args.bootstrap != 10000:
        raise ValueError("Frozen full-strata analysis requires 10000 patient-block bootstrap replicates")
    evaluation = (load_evaluation_strata(strata_path, strata_digest, ROOT, source, families,
                                         args.input, args.families) if strata_path else None)
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
    records, arms, identities, prepared = {}, {}, {}, {}
    binding = None
    for specification in args.arm:
        name, separator, filename = specification.partition("=")
        if not separator or not name or name in prepared:
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
        if evaluation is not None and any(row.get("patient_fhir_id") != references[row["question_id"]]["patient"]
                                          for row in rows):
            raise ValueError(f"{name}: result patient IDs differ from the authoritative full input")
        if any(first_attempt(row).get("latency") is None and not is_failed(first_attempt(row))
               for row in rows):
            raise ValueError(f"{name}: every scheduled question must have a first attempt")
        identity = json.loads(Path(str(path) + ".run.json").read_text())
        if evaluation is not None:
            validate_full_identity(identity, name)
            for row in rows:
                validate_full_attempt(row, name)
        public_binding = {k: identity.get(k) for k in
                          ("model", "max_total_tokens", "max_tool_calls", "input_hash",
                           "split", "condition", "subset_hash", "repeat")}
        if binding is not None and public_binding != binding:
            raise ValueError(f"{name}: model, ceilings, or question/condition identity differs")
        binding = public_binding
        identities[name] = {"result_path": str(path.resolve()), "result_sha256": sha256(path),
                            "identity": identity}
        prepared[name] = indexed
    # Validate every scheduled arm and identity before invoking any scorer.
    for name, indexed in prepared.items():
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
    report = {"cohort": "full_released" if args.full else args.split,
            "selection_warning": "Fixed subset is reused for architecture development; gains on it are exploratory.",
            "questions": len(expected), "patients": len({r["patient"] for r in references.values()}),
            "question_ids_sha256": hashlib.sha256(json.dumps(sorted(expected)).encode()).hexdigest(),
            "dataset_sha256": sha256(args.input), "families_sha256": sha256(args.families),
            "scoring": "Unchanged typed scorer; first attempts, terminal errors incorrect; patient-block bootstrap",
            "identities": identities, "arms": arms, "comparisons": comparisons}
    if evaluation is not None:
        reports = {}
        for name, ids in {**evaluation["splits"], **evaluation["views"]}.items():
            chosen = set(ids)
            subset = {arm: [r for r in rows if r["question_id"] in chosen] for arm, rows in records.items()}
            summaries = {arm: {"correct": sum(r["correct"] for r in rows),
                               "terminal_errors": sum(r["failed"] for r in rows),
                               "recorded_later_attempts": sum(r["retries"] for r in rows),
                               **clustered_mean([r["correct"] for r in rows], [r["patient"] for r in rows],
                                                n_boot=args.bootstrap, seed=5106)}
                         for arm, rows in subset.items()}
            pairs = []
            for i, left in enumerate(subset):
                for right in list(subset)[:i]:
                    a, b = subset[left], subset[right]
                    pairs.append({"comparison": f"{left} - {right}",
                                  "wins": sum(x["correct"] and not y["correct"] for x, y in zip(a, b)),
                                  "losses": sum(y["correct"] and not x["correct"] for x, y in zip(a, b)),
                                  **paired_difference([r["correct"] for r in a], [r["correct"] for r in b],
                                                      [r["patient"] for r in a], n_boot=args.bootstrap, seed=5106)})
            reports[name] = {"role": "primary_stratum" if name == "question_confirmation" else "descriptive",
                             "question_ids_sha256": hashlib.sha256(json.dumps(sorted(chosen)).encode()).hexdigest(),
                             "arms": summaries, "comparisons": pairs}
        report["evaluation_strata"] = {**evaluation, "bootstrap_replicates": args.bootstrap,
                                        "bootstrap_seed": 5106,
                                        "caller_asserted_operational_full_terminal_guard_passed": True,
                                        "primary_estimand": "Prespecified team minus contemporary single on question_confirmation; other contrasts are descriptive.",
                                        "reports": reports}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--input", default=str(ROOT / "final_dataset/questions_answers_sql_fhir.csv"))
    parser.add_argument("--families", default=str(ROOT / "manifests/question_families.csv"))
    parser.add_argument("--split_manifest", default=str(ROOT / "manifests/split.json"))
    parser.add_argument("--split", default="held_out")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--evaluation_strata", help="Frozen full-run exposure-strata JSON; analysis only")
    parser.add_argument("--evaluation_strata_sha256", help="Prespecified SHA256 of that exact manifest")
    parser.add_argument("--full_terminal_confirmed", action="store_true",
                        help="Assert the operational full-terminal/source/audit guard passed; required for strata scoring")
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
