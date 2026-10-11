"""Build the patient-disjoint pilot / dev / held-out split.

Each patient belongs to exactly one split, so no patient's record is seen
during both tuning and evaluation:

  dev       proposal examples plus documented prior exposures; ~50 questions
  pilot     at least PILOT_MIN_PATIENTS randomly drawn patients; 20 questions
            for quick sanity runs
  held_out  every remaining patient; 300 questions for final evaluation

Only the proposal's three in-scope operations are sampled. Questions are
sampled from each split's patient pool stratified by question
family (manifests/question_families.csv, see build_question_families.py), so
each split keeps the dataset's family mix. Questions in a pool that are not
sampled are left unused; do not tune on held-out patients' questions.

The dataset's original train/valid/test column is ignored because it is not
patient-disjoint.

Usage:
    python scripts/build_split.py [--seed 5106]
"""
import argparse
import ast
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(ROOT, "final_dataset", "questions_answers_sql_fhir.csv")
DEFAULT_FAMILIES = os.path.join(ROOT, "manifests", "question_families.csv")
DEFAULT_OUTPUT = os.path.join(ROOT, "manifests", "split.json")
DEFAULT_EXPOSURES = os.path.join(ROOT, "manifests", "development_exposure.json")

DEFAULT_SEED = 5106
SPLIT_SIZES = {"pilot": 20, "dev": 50, "held_out": 300}
# MIMIC-IV subject_ids reserved for dev
DEV_PATIENTS = [10001217, 10004235, 10015860]
PILOT_MIN_PATIENTS = 3


def allocate(counts, total):
    """Split `total` across strata in proportion to `counts` (largest remainder)."""
    exact = counts / counts.sum() * total
    alloc = np.floor(exact).astype(int)
    remainder = (exact - alloc).sort_values(ascending=False)
    alloc[remainder.index[: total - alloc.sum()]] += 1
    return alloc


def sample_questions(pool, total, family_mix, rng):
    """Sample `total` questions from `pool`, matching `family_mix` proportions."""
    quotas = allocate(family_mix, total)
    picked = []
    for family, quota in quotas.items():
        candidates = pool[pool["family"] == family]
        if len(candidates) < quota:
            sys.exit(f"Not enough {family} questions in pool ({len(candidates)} < {quota})")
        picked.append(candidates.sample(n=quota, random_state=rng))
    return pd.concat(picked)


def choose_pilot_patients(df, candidates, quotas, rng):
    """Draw random patients until there are enough to fill every family quota."""
    chosen = []
    for patient in rng.permutation(candidates):
        chosen.append(patient)
        counts = df[df["patient_fhir_id"].isin(chosen)]["family"].value_counts()
        if len(chosen) >= PILOT_MIN_PATIENTS and all(counts.get(f, 0) >= q for f, q in quotas.items()):
            return chosen
    sys.exit("Could not build a pilot pool that covers every family")


def build_split(df, seed, exposed_subjects=()):
    # Preserve every manifest row, but evaluate only the proposed operations.
    df = df[df["family"].isin(("lookup_counting", "numerical_aggregation", "temporal_selection"))].copy()
    if df.empty:
        raise ValueError("No in-scope questions are available")
    df = df.sort_values("question_id").reset_index(drop=True)
    rng = np.random.RandomState(seed)
    family_mix = df["family"].value_counts().sort_index()
    subject_to_fhir = df.drop_duplicates("subject_id").set_index("subject_id")["patient_fhir_id"]

    reserved = sorted(set(DEV_PATIENTS) | set(exposed_subjects))
    missing = [p for p in reserved if p not in subject_to_fhir.index]
    if missing:
        sys.exit(f"Reserved dev patients not in dataset: {missing}")
    dev_patients = [subject_to_fhir[p] for p in reserved]

    others = sorted(set(df["patient_fhir_id"]) - set(dev_patients))
    pilot_quotas = allocate(family_mix, SPLIT_SIZES["pilot"])
    pilot_patients = choose_pilot_patients(df, others, pilot_quotas, rng)
    held_out_patients = sorted(set(others) - set(pilot_patients))

    pools = {"pilot": pilot_patients, "dev": dev_patients, "held_out": held_out_patients}
    splits = {}
    for name, patients in pools.items():
        pool = df[df["patient_fhir_id"].isin(patients)]
        sample = sample_questions(pool, SPLIT_SIZES[name], family_mix, rng)
        id_to_subject = pool.drop_duplicates("patient_fhir_id").set_index("patient_fhir_id")["subject_id"]
        splits[name] = {
            "patients": [
                {"patient_fhir_id": p, "subject_id": int(id_to_subject[p])}
                for p in sorted(patients, key=lambda p: id_to_subject[p])
            ],
            "question_ids": sorted(sample["question_id"]),
            "family_counts": sample["family"].value_counts().sort_index().to_dict(),
            "answer_kind_counts": sample["answer_kind"].value_counts().sort_index().to_dict(),
        }
    return splits


def subject_id(value):
    """Read the released structured template values without formatting assumptions."""
    return int(ast.literal_eval(value)["val_placeholder"]["patient_id"])


def file_sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_disjoint(splits):
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa = {p["patient_fhir_id"] for p in splits[a]["patients"]}
            pb = {p["patient_fhir_id"] for p in splits[b]["patients"]}
            if pa & pb or set(splits[a]["question_ids"]) & set(splits[b]["question_ids"]):
                sys.exit(f"Splits {a} and {b} overlap")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--families", default=DEFAULT_FAMILIES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--exposures", default=DEFAULT_EXPOSURES)
    args = parser.parse_args()

    df = pd.read_csv(args.input, usecols=["question_id", "patient_fhir_id", "val_dict"])
    df["subject_id"] = df["val_dict"].map(subject_id)
    families = pd.read_csv(args.families, usecols=["question_id", "family", "answer_kind"])
    df = df.merge(families, on="question_id", how="left", validate="one_to_one")
    if df["family"].isna().any():
        sys.exit("Some questions are missing from the family manifest; rerun build_question_families.py")

    with open(args.exposures) as handle:
        exposures = json.load(handle)
    exposed_subjects = {int(p) for item in exposures["exposures"] for p in item["subject_ids"]}
    splits = build_split(df, args.seed, exposed_subjects)
    check_disjoint(splits)

    manifest = {
        "seed": args.seed,
        "source": os.path.relpath(args.input, ROOT),
        "family_manifest": os.path.relpath(args.families, ROOT),
        "source_sha256": file_sha256(args.input),
        "family_manifest_sha256": file_sha256(args.families),
        "development_exposure": os.path.relpath(args.exposures, ROOT),
        "development_exposure_sha256": file_sha256(args.exposures),
        "in_scope_families": ["lookup_counting", "numerical_aggregation", "temporal_selection"],
        "splits": splits,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Wrote {os.path.relpath(args.output, ROOT)} (seed {args.seed})\n")
    for name, split in splits.items():
        print(f"{name:9s} {len(split['question_ids']):4d} questions from "
              f"{len(split['patients']):2d} patients  "
              f"families={split['family_counts']}  answers={split['answer_kind_counts']}")


if __name__ == "__main__":
    main()
