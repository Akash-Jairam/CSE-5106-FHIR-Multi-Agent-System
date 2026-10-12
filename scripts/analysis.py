"""Experiment analysis (P3-21): accuracy with patient-clustered CIs, paired comparisons, latency, plot.

Run through `python scripts/run_experiments.py analyze ...`, which locates
and checks each arm's result file. Only complete, current arms are analyzed;
the rest are listed as skipped. Answers are scored with the typed scorer
(errors count as wrong) and never written or printed. Outputs, in
<output_dir>/analysis/:

  summary.json             everything below, plus settings and skipped arms
  accuracy.csv             accuracy per arm, overall and by family, with 95% CIs
  comparisons.csv          paired differences between systems and conditions, with 95% CIs
  latency.csv              median and p95 latency, and mean tokens, per arm
  accuracy_vs_tokens.svg   the ordinary arms' accuracy against mean tokens per question

The bootstrap seed is the split seed, so reruns give identical intervals.
"""
import csv
import json
import os

from scripts.experiment_plots import accuracy_vs_tokens_svg
from scripts.experiment_stats import clustered_mean, latency_summary, paired_difference
from scripts.variability import score_row

ALL = "all"
# (label, arm a, arm b): a - b on the questions both answered
COMPARISONS = [
    ("multi_agent - single_agent", "ordinary_multi_agent", "ordinary_single_agent"),
    ("multi_agent - direct_llm", "ordinary_multi_agent", "ordinary_direct_llm"),
    ("single_agent - direct_llm", "ordinary_single_agent", "ordinary_direct_llm"),
    ("single_agent: omitted - intact", "recovery_single_agent_omitted", "recovery_single_agent_intact"),
    ("multi_agent: omitted - intact", "recovery_multi_agent_omitted", "recovery_multi_agent_intact"),
    ("omitted: multi_agent - single_agent", "recovery_multi_agent_omitted", "recovery_single_agent_omitted"),
    ("intact: multi_agent - single_agent", "recovery_multi_agent_intact", "recovery_single_agent_intact"),
]


def arm_records(rows, references):
    """Per-question records for one complete arm, in question-ID order."""
    records = []
    for qid in sorted(rows):
        row, reference = rows[qid], references[qid]
        tokens = (row.get("tokens") or {}).get("total")
        records.append({"question_id": qid, "patient": reference["patient"], "family": reference["family"],
                        "correct": score_row(row, reference), "tokens": tokens, "latency": row.get("latency")})
    return records


def _families(records):
    return [ALL] + sorted({r["family"] for r in records if r["family"]})


def _select(records, family):
    return [r for r in records if family == ALL or r["family"] == family]


def analyze(arms, args, question_sets):
    from scripts import run_experiments as experiments

    with open(args.split_manifest, encoding="utf-8") as stream:
        seed = json.load(stream)["seed"]
    statuses = {arm.name: experiments.arm_results(arm, args, question_sets) for arm in arms}
    complete = {name: rows for name, (status, rows) in statuses.items() if status["state"] == "complete"}
    question_ids = {qid for rows in complete.values() for qid in rows}
    references = experiments.load_references(args, question_ids)
    records = {name: arm_records(rows, references) for name, rows in complete.items()}
    boot = {"n_boot": args.bootstrap, "seed": seed}

    accuracy = []
    for name, recs in records.items():
        for family in _families(recs):
            chosen = _select(recs, family)
            accuracy.append({"arm": name, "family": family,
                             **clustered_mean([r["correct"] for r in chosen], [r["patient"] for r in chosen], **boot)})

    comparisons = []
    for label, a, b in COMPARISONS:
        if a not in records or b not in records:
            continue
        left = {r["question_id"]: r for r in records[a]}
        right = {r["question_id"]: r for r in records[b]}
        shared = sorted(set(left) & set(right))
        for family in _families([left[q] for q in shared]):
            chosen = [q for q in shared if family == ALL or left[q]["family"] == family]
            comparisons.append({"comparison": label, "a": a, "b": b, "family": family, **paired_difference(
                [left[q]["correct"] for q in chosen], [right[q]["correct"] for q in chosen],
                [left[q]["patient"] for q in chosen], **boot)})

    latency, plot_points = [], {}
    for name, recs in records.items():
        tokens = [r["tokens"] for r in recs]
        token_ci = (clustered_mean(tokens, [r["patient"] for r in recs], **boot) if None not in tokens
                    else {"estimate": None, "ci_low": None, "ci_high": None})
        latency.append({"arm": name, **latency_summary([r["latency"] for r in recs]),
                        "mean_tokens": token_ci["estimate"], "tokens_ci_low": token_ci["ci_low"],
                        "tokens_ci_high": token_ci["ci_high"]})
        if name.startswith("ordinary_") and token_ci["estimate"] is not None:
            overall = next(row for row in accuracy if row["arm"] == name and row["family"] == ALL)
            plot_points[name.removeprefix("ordinary_")] = {"accuracy": overall, "tokens": token_ci}

    return {
        "settings": {"model": args.model, "split": args.split, "limit": args.limit,
                     "max_total_tokens": args.max_total_tokens, "max_tool_calls": args.max_tool_calls,
                     "bootstrap": args.bootstrap, "seed": seed, "level": 0.95, "clustered_by": "patient"},
        "analyzed": sorted(complete),
        "skipped": {name: status["state"] for name, (status, _) in statuses.items() if name not in complete},
        "accuracy": accuracy,
        "comparisons": comparisons,
        "latency": latency,
        "plot_points": plot_points,
    }


def _write_csv(rows, path, columns):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_analysis(report, directory):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "summary.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1, allow_nan=False)
        stream.write("\n")
    _write_csv(report["accuracy"], os.path.join(directory, "accuracy.csv"),
               ["arm", "family", "n", "patients", "estimate", "ci_low", "ci_high"])
    _write_csv(report["comparisons"], os.path.join(directory, "comparisons.csv"),
               ["comparison", "family", "a", "b", "n", "patients", "estimate", "ci_low", "ci_high", "p_bootstrap"])
    _write_csv(report["latency"], os.path.join(directory, "latency.csv"),
               ["arm", "n", "median", "p95", "mean_tokens", "tokens_ci_low", "tokens_ci_high"])
    settings = report["settings"]
    subtitle = (f"{settings['model']}, {settings['split']} questions; "
                f"whiskers are patient-clustered 95% CIs ({settings['bootstrap']:,} resamples)")
    with open(os.path.join(directory, "accuracy_vs_tokens.svg"), "w", encoding="utf-8") as stream:
        stream.write(accuracy_vs_tokens_svg(report["plot_points"], subtitle))


def _ci(row, fmt="{:.1%}"):
    if row["estimate"] is None:
        return "n/a"
    return f"{fmt.format(row['estimate'])} [{fmt.format(row['ci_low'])}, {fmt.format(row['ci_high'])}]"


def print_analysis(report):
    if not report["analyzed"]:
        print("No complete arms to analyze yet.")
    families = sorted({row["family"] for row in report["accuracy"]} - {ALL})
    if report["accuracy"]:
        print("Accuracy (95% CI, clustered by patient)")
        for arm in report["analyzed"]:
            rows = {row["family"]: row for row in report["accuracy"] if row["arm"] == arm}
            print(f"  {arm}: {_ci(rows[ALL])} over {rows[ALL]['n']} questions, {rows[ALL]['patients']} patients")
            for family in families:
                if family in rows:
                    print(f"    {family}: {_ci(rows[family])} (n={rows[family]['n']})")
    overall = [row for row in report["comparisons"] if row["family"] == ALL]
    if overall:
        print("\nPaired differences (a - b, 95% CI)")
        for row in overall:
            p = "<0.001" if row["p_bootstrap"] < 0.001 else f"={row['p_bootstrap']:.3f}"
            print(f"  {row['comparison']}: {_ci(row, "{:+.1%}")} (n={row['n']}, p{p})")
    if report["latency"]:
        print("\nLatency and tokens per question")
        for row in report["latency"]:
            median = "n/a" if row["median"] is None else f"{row['median']:.1f}s"
            p95 = "n/a" if row["p95"] is None else f"{row['p95']:.1f}s"
            tokens = "n/a" if row["mean_tokens"] is None else f"{row['mean_tokens']:,.0f}"
            print(f"  {row['arm']}: median {median}, p95 {p95}, mean tokens {tokens}")
    for name, state in report["skipped"].items():
        print(f"Skipped {name}: {state}")
