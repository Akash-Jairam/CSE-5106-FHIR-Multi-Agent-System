"""Classify each question template into a question family and write a manifest.

Families are assigned by the template's primary operation. The trailing
[time_filter_global*] slot only scopes the search window (e.g. "in 2162",
"on the last hospital visit"), so it does not by itself make a question
temporal. Rules are checked in order and the first match wins:

  numerical_aggregation  reduce a set of numeric values: [agg_function]
                         (min/max/avg) or "total" (sum)
  temporal_selection     pick an event by time order or report a time:
                         [time_filter_exact*] (first/last/Nth), [sort],
                         "When was ...", "... have passed since ..."
  lookup_counting        everything else: existence checks, counts, and
                         attribute lookups (gender, marital status, ...)

All rows are kept, including questions whose true answer is empty ("[]") or
null ("[[None]]"); `answer_kind` records which.

Usage:
    python scripts/build_question_families.py
"""
import argparse
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(ROOT, "final_dataset", "questions_answers_sql_fhir.csv")
DEFAULT_OUTPUT = os.path.join(ROOT, "manifests", "question_families.csv")

FAMILY_RULES = [
    ("numerical_aggregation", re.compile(r"\[agg_function\]|\btotal\b")),
    ("temporal_selection", re.compile(r"\[time_filter_exact\d\]|\[sort\]|^When was|passed since")),
]
DEFAULT_FAMILY = "lookup_counting"


def classify_template(template):
    for family, pattern in FAMILY_RULES:
        if pattern.search(template):
            return family
    return DEFAULT_FAMILY


def answer_kind(true_answer):
    answer = str(true_answer).strip()
    if answer in ("[]", "nan", ""):
        return "empty"
    if answer == "[[None]]":
        return "null"
    return "value"


def build_manifest(df):
    templates = sorted(df["template"].unique())
    template_ids = {t: f"T{i:03d}" for i, t in enumerate(templates)}
    return pd.DataFrame({
        "question_id": df["question_id"],
        "split": df["split"],
        "family": df["template"].map(classify_template),
        "template_id": df["template"].map(template_ids),
        "template": df["template"],
        "main_table_name": df["main_table_name"],
        "answer_kind": df["true_answer"].map(answer_kind),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    manifest = build_manifest(df)
    if len(manifest) != len(df) or manifest["question_id"].duplicated().any():
        sys.exit("Manifest must have exactly one row per question_id")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    manifest.to_csv(args.output, index=False)

    print(f"Wrote {len(manifest)} rows to {os.path.relpath(args.output, ROOT)}\n")
    print(pd.crosstab(manifest["family"], manifest["split"], margins=True), "\n")
    print(pd.crosstab(manifest["family"], manifest["answer_kind"], margins=True), "\n")
    print(manifest.groupby("family")["template_id"].nunique().rename("templates"))


if __name__ == "__main__":
    main()
