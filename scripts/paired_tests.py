"""Exact McNemar tests between the ordinary arms (a supplementary check on `run_experiments.py analyze`).

For each pair of systems on the same held-out questions, counts the
questions only one of them answered correctly and gives the exact two-sided
McNemar p-value. The test treats questions as independent; the
patient-clustered intervals from `analyze` remain the primary result.
With several models, the pairs are also tested on all their questions pooled.

    python scripts/paired_tests.py output/experiments/gpt-oss:120b/held_out_v2 \
        output/experiments/qwen3:32b/held_out_v2

Scoring matches `run_experiments.py analyze`: first attempt, typed scorer,
failed runs count as wrong. Prints aggregates only, never answers.
"""
import argparse
import json
import os
import sys
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.experiment_stats import mcnemar_exact  # noqa: E402
from scripts.run_experiments import DEFAULT_FAMILIES, DEFAULT_INPUT, DEFAULT_SPLIT, load_references  # noqa: E402
from scripts.variability import first_attempt, is_failed, score_row  # noqa: E402

ARMS = ("multi_agent", "single_agent", "direct_llm")
PAIRS = [("multi_agent", "single_agent"), ("multi_agent", "direct_llm"), ("single_agent", "direct_llm")]


def arm_scores(path, references=None):
    """Per-question 0/1 correctness of one result file, keyed by question ID."""
    with open(path, encoding="utf-8") as stream:
        rows = {row["question_id"]: first_attempt(row) for row in json.load(stream)}
    if references is None:
        references = load_references(Namespace(input=DEFAULT_INPUT, split_manifest=DEFAULT_SPLIT,
                                               families=DEFAULT_FAMILIES), list(rows))
    return {qid: 0 if is_failed(row) else score_row(row, references[qid]) for qid, row in rows.items()}, references


def compare(scores, a, b):
    if set(scores[a]) != set(scores[b]):
        raise ValueError(f"{a} and {b} must cover exactly the same questions")
    keys = sorted(scores[a])
    test = mcnemar_exact([scores[a][k] for k in keys], [scores[b][k] for k in keys])
    return {**test, "correct_a": sum(scores[a].values()), "correct_b": sum(scores[b].values())}


def _line(label, a, b, test):
    return (f"{label:<16} {a} vs {b}: {test['correct_a']} vs {test['correct_b']} of {test['n']}, "
            f"only {a} {test['only_a']}, only {b} {test['only_b']}, exact p = {test['p_value']:.2g}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Exact McNemar tests between ordinary arms.")
    parser.add_argument("output_dirs", nargs="+", help="One experiment output directory per model")
    args = parser.parse_args(argv)

    pooled = {arm: {} for arm in ARMS}
    for directory in args.output_dirs:
        scores, references = {}, None
        for arm in ARMS:
            scores[arm], references = arm_scores(os.path.join(directory, f"ordinary_{arm}.json"), references)
            pooled[arm].update({(directory, qid): value for qid, value in scores[arm].items()})
        label = os.path.basename(os.path.dirname(os.path.normpath(directory)))
        for a, b in PAIRS:
            print(_line(label, a, b, compare(scores, a, b)))
    if len(args.output_dirs) > 1:
        for a, b in PAIRS:
            print(_line("pooled", a, b, compare(pooled, a, b)))


if __name__ == "__main__":
    main()
