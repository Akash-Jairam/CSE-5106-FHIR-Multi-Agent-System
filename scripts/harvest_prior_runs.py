"""Harvest per-question outcomes from the completed paired runs (V8 diagnosis, step A1).

The raw result files of the V2-V7 experiments are gitignored, but each branch
committed per-question mechanism records. This script reads those records
straight from Git (`git show <ref>:<path>`) and normalizes them into one long
table with one row per (run, arm, question):

    run, arm, system, question_id, correct, terminal_error, answer_is_null,
    typed_reason, tokens, tool_calls

`system` is one of `direct`, `single`, `team`, or `a_only` (the natural
single-agent policy embedded as the V6/V7 primary). Not every run recorded
every arm per question; coverage is printed so later analysis can say exactly
which runs each set is based on. Answers themselves are not in these records
and are never printed.

Usage:
    python scripts/harvest_prior_runs.py [--ref origin/experiments/residual-trajectories-v7]
                                         [--output_dir output/diagnosis]
"""
import argparse
import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_REF = "origin/experiments/residual-trajectories-v7"
COLUMNS = ["run", "arm", "system", "question_id", "correct", "terminal_error",
           "answer_is_null", "typed_reason", "tokens", "tool_calls"]

# Each source: committed path, loader name, and how its arms map to systems.
SOURCES = [
    {"run": "archived", "path": "benchmarks/run-300q-20261012/investigations/midterm_20261012/paired_outcomes.csv",
     "loader": "paired_outcomes"},
    {"run": "v2", "path": "benchmarks/design-v2-20261013/mechanisms.csv", "loader": "arm_csv",
     "arms": {"refreshed_single": "single", "verified_team_v2": "team"}},
    {"run": "v3d", "path": "benchmarks/design-v3d-20261014/diagnostics/mechanisms_rows.json", "loader": "rows_json",
     "arms": {"v3d": "team"}},
    {"run": "v4", "path": "benchmarks/design-v4-20261014/diagnostics_corrected/mechanisms.csv", "loader": "arm_csv",
     "arms": {"ordinary_single_agent": "single", "ordinary_multi_agent": "team"}},
    {"run": "v5", "path": "benchmarks/design-v5-20261015/diagnostics/mechanisms_rows.json", "loader": "rows_json",
     "arms": {None: "team"}},
    {"run": "v6", "path": "benchmarks/design-v6-20261015/diagnostics/mechanisms_rows.json", "loader": "v6_rows"},
    {"run": "v7", "path": "benchmarks/design-v7-20261015/diagnostics/mechanisms_rows.json", "loader": "v7_rows"},
]


def git_show(ref, path, repo_root):
    result = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=repo_root, capture_output=True, text=True)
    if result.returncode != 0:
        raise FileNotFoundError(f"{ref}:{path}: {result.stderr.strip()}")
    return result.stdout


def to_bool(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def null_flag(record):
    for key in ("answer_is_null", "actual_answer_is_null"):
        if key in record:
            return to_bool(record[key])
    return None


def row(run, arm, system, record, correct_key="correct"):
    return {
        "run": run, "arm": arm, "system": system,
        "question_id": record["question_id"],
        "correct": to_bool(record.get(correct_key)),
        "terminal_error": to_bool(record.get("terminal_error")),
        "answer_is_null": null_flag(record),
        "typed_reason": record.get("typed_reason") or None,
        "tokens": to_int(record.get("tokens")),
        "tool_calls": to_int(record.get("tool_calls")),
    }


def load_paired_outcomes(run, text, _source):
    rows = []
    for record in csv.DictReader(io.StringIO(text)):
        base = {"question_id": record["question_id"]}
        rows.append(row(run, "ordinary_direct_llm", "direct", {**base, "correct": record["direct_correct"]}))
        rows.append(row(run, "ordinary_single_agent", "single", {**base, "correct": record["single_correct"]}))
        rows.append(row(run, "ordinary_multi_agent", "team",
                        {**base, "correct": record["team_correct"], "terminal_error": record.get("team_error")}))
    return rows


def load_arm_csv(run, text, source):
    rows = []
    for record in csv.DictReader(io.StringIO(text)):
        system = source["arms"].get(record.get("arm"))
        if system:
            rows.append(row(run, record["arm"], system, record))
    return rows


def load_rows_json(run, text, source):
    rows = []
    for record in json.loads(text):
        arm = record.get("arm")
        system = source["arms"].get(arm, source["arms"].get(None))
        if system:
            rows.append(row(run, arm or f"{run}_team", system, record))
    return rows


def load_v6_rows(run, text, _source):
    rows = []
    for record in json.loads(text):
        rows.append(row(run, "v6_team", "team", record))
        if record.get("a_only_available", True):
            rows.append(row(run, "v6_a_only", "a_only",
                            {"question_id": record["question_id"], "correct": record.get("a_only_correct"),
                             "terminal_error": record.get("a_only_error")}))
    return rows


def load_v7_rows(run, text, _source):
    rows = []
    for record in json.loads(text):
        rows.append(row(run, "v7_team", "team", record))
        a = record.get("A") or {}
        if a.get("available"):
            rows.append(row(run, "v7_a_only", "a_only",
                            {"question_id": record["question_id"], "correct": a.get("correct"),
                             "terminal_error": a.get("error"), "answer_is_null": a.get("exact_null"),
                             "tokens": a.get("tokens"), "tool_calls": a.get("tool_calls")}))
    return rows


LOADERS = {
    "paired_outcomes": load_paired_outcomes,
    "arm_csv": load_arm_csv,
    "rows_json": load_rows_json,
    "v6_rows": load_v6_rows,
    "v7_rows": load_v7_rows,
}


def harvest(ref, repo_root, sources=SOURCES):
    rows, missing = [], []
    for source in sources:
        try:
            text = git_show(ref, source["path"], repo_root)
        except FileNotFoundError as error:
            missing.append(str(error))
            continue
        rows.extend(LOADERS[source["loader"]](source["run"], text, source))
    return rows, missing


def coverage(rows):
    summary = {}
    for record in rows:
        key = (record["run"], record["system"])
        entry = summary.setdefault(key, {"questions": 0, "correct": 0, "terminal_errors": 0})
        entry["questions"] += 1
        entry["correct"] += 1 if record["correct"] else 0
        entry["terminal_errors"] += 1 if record["terminal_error"] else 0
    return {f"{run}/{system}": value for (run, system), value in sorted(summary.items())}


def write_outputs(rows, missing, ref, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    table = os.path.join(output_dir, "prior_runs.csv")
    with open(table, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"ref": ref, "rows": len(rows), "missing_sources": missing, "coverage": coverage(rows)}
    with open(os.path.join(output_dir, "prior_runs_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    return table, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--output_dir", default="output/diagnosis")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    rows, missing = harvest(args.ref, repo_root)
    table, summary = write_outputs(rows, missing, args.ref, args.output_dir)
    print(f"Wrote {len(rows)} rows to {table}")
    for key, value in summary["coverage"].items():
        print(f"  {key:18s} n={value['questions']:3d} correct={value['correct']:3d} errors={value['terminal_errors']}")
    for item in missing:
        print(f"  missing: {item}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
