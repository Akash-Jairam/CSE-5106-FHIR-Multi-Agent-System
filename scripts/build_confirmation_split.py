"""Draw the once-only V9 confirmation sample of protected questions (docs/diagnosis_v8/launch_v9.md section 4).

Protected: in-scope questions in none of split.json's pilot/dev/held_out lists, on a
patient in neither manifests/development_exposure.json nor
docs/diagnosis_v8/audited_subject_ids.json. A family-stratified 300 (seed 5106,
the in-scope family mix, build_split.sample_questions) is written as a one-split
manifest named held_out so the unchanged runners accept it. Run once.

  python scripts/build_confirmation_split.py --output manifests/confirmation_v9.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_split import (DEFAULT_EXPOSURES, DEFAULT_FAMILIES, DEFAULT_INPUT, ROOT, file_sha256,
                         sample_questions, subject_id)

DEFAULT_SPLIT = os.path.join(ROOT, "manifests", "split.json")
DEFAULT_AUDITED = os.path.join(ROOT, "docs", "diagnosis_v8", "audited_subject_ids.json")
IN_SCOPE = ("lookup_counting", "numerical_aggregation", "temporal_selection")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=5106)
    parser.add_argument("--questions", type=int, default=300)
    args = parser.parse_args()
    if os.path.exists(args.output):
        sys.exit(f"{args.output} exists; the confirmation sample is drawn once")

    df = pd.read_csv(DEFAULT_INPUT, usecols=["question_id", "patient_fhir_id", "val_dict"])
    df["subject_id"] = df["val_dict"].map(subject_id)
    families = pd.read_csv(DEFAULT_FAMILIES, usecols=["question_id", "family", "answer_kind"])
    df = df.merge(families, on="question_id", how="left", validate="one_to_one")
    df = df[df["family"].isin(IN_SCOPE)].sort_values("question_id").reset_index(drop=True)
    family_mix = df["family"].value_counts().sort_index()

    with open(DEFAULT_SPLIT) as handle:
        used = {q for entry in json.load(handle)["splits"].values() for q in entry["question_ids"]}
    with open(DEFAULT_EXPOSURES) as handle:
        exposed = {int(p) for item in json.load(handle)["exposures"] for p in item["subject_ids"]}
    with open(DEFAULT_AUDITED) as handle:
        audited = {int(p) for p in json.load(handle)}
    pool = df[~df["question_id"].isin(used) & ~df["subject_id"].isin(exposed | audited)]
    sample = sample_questions(pool, args.questions, family_mix, np.random.RandomState(args.seed))
    patients = sample.drop_duplicates("patient_fhir_id").sort_values("subject_id")

    manifest = {
        "seed": args.seed,
        "purpose": "V9 once-only confirmation (design-v9-20261019); protected questions, scored once",
        "source": os.path.relpath(DEFAULT_INPUT, ROOT),
        "family_manifest": os.path.relpath(DEFAULT_FAMILIES, ROOT),
        "source_sha256": file_sha256(DEFAULT_INPUT),
        "family_manifest_sha256": file_sha256(DEFAULT_FAMILIES),
        "excluded_split_sha256": file_sha256(DEFAULT_SPLIT),
        "development_exposure_sha256": file_sha256(DEFAULT_EXPOSURES),
        "audited_subject_ids_sha256": file_sha256(DEFAULT_AUDITED),
        "protected_pool": {"questions": len(pool), "patients": int(pool["subject_id"].nunique())},
        "in_scope_families": list(IN_SCOPE),
        "splits": {"held_out": {
            "patients": [{"patient_fhir_id": p, "subject_id": int(s)}
                         for p, s in zip(patients["patient_fhir_id"], patients["subject_id"])],
            "question_ids": sorted(sample["question_id"]),
            "family_counts": sample["family"].value_counts().sort_index().to_dict(),
            "answer_kind_counts": sample["answer_kind"].value_counts().sort_index().to_dict(),
        }},
    }
    with open(args.output, "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    split = manifest["splits"]["held_out"]
    print(f"Wrote {args.output}: {len(split['question_ids'])} questions, {len(split['patients'])} patients, "
          f"pool {manifest['protected_pool']}, families {split['family_counts']}")


if __name__ == "__main__":
    main()
