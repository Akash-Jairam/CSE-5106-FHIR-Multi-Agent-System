"""Offline release rules for multi_agent's verification gate (a post-hoc analysis, not a Phase 3 arm).

multi_agent releases an answer only when its final check reports no issues.
That decision comes after every role has run, and each result keeps the draft,
the final answer and the issues, so another rule can be applied to saved runs
without rerunning the agents. The rules were fixed before any dev results:

  strict       release only when no issue remains (the system as built)
  field_check  withhold only when a deterministic field-check issue remains;
               issues resting on the verifier model's own judgment do not block
  always       release the final answer, or the draft when there is none

Choose a rule on the dev split (highest accuracy; ties go to the stricter
rule), then apply only that rule to held-out:

    python scripts/release_policy.py output/experiments/<model>/dev/ordinary_multi_agent.json
    python scripts/release_policy.py output/experiments/<model>/held_out/ordinary_multi_agent.json \
        --rules strict field_check --compare output/experiments/<model>/held_out/ordinary_single_agent.json

Scoring matches `run_experiments.py analyze`: first attempt, typed scorer,
failed runs count as wrong, patient-clustered 95% CIs seeded by the split.
Prints aggregates only, never answers.
"""
import argparse
import json
import os
import sys
from argparse import Namespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.experiment_stats import DEFAULT_BOOTSTRAP, clustered_mean, paired_difference  # noqa: E402
from scripts.run_experiments import DEFAULT_FAMILIES, DEFAULT_INPUT, DEFAULT_SPLIT, load_references  # noqa: E402
from scripts.variability import first_attempt, is_failed, score_row  # noqa: E402

RULES = ("strict", "field_check", "always")
# Raised by the deterministic field check (tools/field_check.py, utils/evidence_scorer.check_citation)
FIELD_CHECK_ISSUES = frozenset({
    "malformed", "missing_resource", "bad_path", "value_mismatch", "patient_scope", "time_field",
    "time_window", "unit", "encounter_link", "encounter_scope", "current_time", "no_evidence"})


def _revision(row):
    revision = row.get("revision")
    if isinstance(revision, str):
        try:
            revision = json.loads(revision)
        except json.JSONDecodeError:
            revision = None
    return revision if isinstance(revision, dict) else {}


def released_answer(row, rule):
    """The answer `rule` releases for one attempt, or None when it withholds one."""
    if rule == "strict":
        return row.get("agent_answer")
    revision = _revision(row)
    final = revision.get("final")
    candidate = final or revision.get("draft")
    if not candidate:
        return None
    issues = (revision.get("final_check_issues") if final else revision.get("check_issues")) or []
    if rule == "field_check" and any(issue.get("type") in FIELD_CHECK_ISSUES for issue in issues):
        return None
    return candidate.get("answer")


def score_rule(rows, references, rule):
    """Per-question 0/1 correctness (question-ID order) and release counts under `rule`."""
    scores, released, released_wrong = [], 0, 0
    for qid in sorted(rows):
        row = first_attempt(rows[qid])
        if is_failed(row):
            scores.append(0)
            continue
        answer = released_answer(row, rule)
        correct = score_row({**row, "agent_answer": answer, "attempt_history": None}, references[qid])
        scores.append(correct)
        if answer is not None:
            released += 1
            released_wrong += 1 - correct
    return scores, released, released_wrong


def _load(path):
    with open(path, encoding="utf-8") as stream:
        return {row["question_id"]: row for row in json.load(stream)}


def _ci(stats, sign=""):
    if stats["ci_low"] is None:
        return f"{stats['estimate']:{sign}.1%}"
    return f"{stats['estimate']:{sign}.1%} [{stats['ci_low']:{sign}.1%}, {stats['ci_high']:{sign}.1%}]"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Apply release rules to saved multi_agent results.")
    parser.add_argument("results", help="A multi_agent result file")
    parser.add_argument("--rules", nargs="+", choices=RULES, default=list(RULES))
    parser.add_argument("--compare", help="Another arm's result file on the same questions, e.g. single_agent")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    args = parser.parse_args(argv)

    rows = _load(args.results)
    references = load_references(Namespace(input=DEFAULT_INPUT, split_manifest=DEFAULT_SPLIT,
                                           families=DEFAULT_FAMILIES), list(rows))
    with open(DEFAULT_SPLIT, encoding="utf-8") as stream:
        seed = json.load(stream)["seed"]
    patients = [references[qid]["patient"] for qid in sorted(rows)]
    baseline = None
    if args.compare:
        other = _load(args.compare)
        if set(other) != set(rows):
            raise ValueError("--compare must cover exactly the same questions")
        baseline = [score_row(other[qid], references[qid]) for qid in sorted(other)]

    print(f"{len(rows)} questions, {len(set(patients))} patients")
    print(f"{'rule':<12} {'accuracy (95% CI)':<26} {'released':>8} {'wrong released':>15}")
    for rule in args.rules:
        scores, released, wrong = score_rule(rows, references, rule)
        accuracy = clustered_mean(scores, patients, args.bootstrap, seed)
        line = f"{rule:<12} {_ci(accuracy):<26} {released:>8} {wrong:>15}"
        if baseline is not None:
            line += f"   vs compare: {_ci(paired_difference(scores, baseline, patients, args.bootstrap, seed), '+')}"
        print(line)
    if baseline is not None:
        print(f"compare arm accuracy: {_ci(clustered_mean(baseline, patients, args.bootstrap, seed))}")


if __name__ == "__main__":
    main()
