"""Build manifests/variability_subset.json: the fixed questions for run-variability repeats (P3-20).

SUBSET_SIZE held-out questions are sampled with the split seed, stratified by
question family in proportion to the held-out mix. Only question IDs and
families are read; reference answers are never loaded. Each system answers
the subset REPEATS times (scripts/run_experiments.py), and
scripts/run_experiments.py variability reports the spread.

Usage:
    python scripts/build_variability_subset.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.build_split import DEFAULT_FAMILIES, DEFAULT_OUTPUT as DEFAULT_SPLIT, allocate, file_sha256  # noqa: E402

DEFAULT_OUTPUT = os.path.join(ROOT, "manifests", "variability_subset.json")
SUBSET_SIZE = 20
REPEATS = 5
SPLIT = "held_out"


def build_subset(split, families, size=SUBSET_SIZE):
    """Sorted question IDs and family counts; deterministic for a given split manifest."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("Subset size must be a positive integer")
    question_ids = split["splits"][SPLIT]["question_ids"]
    if (not question_ids or any(not isinstance(qid, str) or not qid.strip() for qid in question_ids)
            or len(set(question_ids)) != len(question_ids)):
        raise ValueError("Held-out question IDs must be unique, non-empty strings")
    pool = families[families["question_id"].isin(question_ids)].copy()
    if pool["question_id"].duplicated().any():
        raise ValueError("Each held-out question must have exactly one family row")
    if set(pool["question_id"]) != set(question_ids):
        raise ValueError("Some held-out questions are missing from the family manifest")
    if pool["family"].isna().any() or not pool["family"].isin(
            ("lookup_counting", "numerical_aggregation", "temporal_selection")).all():
        raise ValueError("Every held-out question must have an in-scope family")
    if size > len(pool):
        raise ValueError("Subset size exceeds the held-out question count")
    pool = pool.sort_values("question_id").reset_index(drop=True)
    rng = np.random.RandomState(split["seed"])
    quotas = allocate(pool["family"].value_counts().sort_index(), size)
    picked = pd.concat([pool[pool["family"] == family].sample(n=quota, random_state=rng)
                        for family, quota in quotas.items() if quota])
    return sorted(picked["question_id"]), picked["family"].value_counts().sort_index().to_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--families", default=DEFAULT_FAMILIES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with open(args.split, encoding="utf-8") as stream:
        split = json.load(stream)
    if split.get("family_manifest_sha256") != file_sha256(args.families):
        raise ValueError("The split was built from a different family manifest")
    families = pd.read_csv(args.families, usecols=["question_id", "family"])
    question_ids, family_counts = build_subset(split, families)
    manifest = {
        "version": 1,
        "split": SPLIT,
        "seed": split["seed"],
        "size": len(question_ids),
        "repeats": REPEATS,
        "split_sha256": file_sha256(args.split),
        "families_sha256": file_sha256(args.families),
        "family_counts": family_counts,
        "question_ids": question_ids,
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=1)
        stream.write("\n")
    print(f"Wrote {len(question_ids)} questions x {REPEATS} repeats to {os.path.relpath(args.output, ROOT)}: "
          f"{family_counts}")


if __name__ == "__main__":
    main()
