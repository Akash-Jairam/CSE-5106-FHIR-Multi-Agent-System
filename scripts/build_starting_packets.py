"""Build starting packets for every question in manifests/split.json.

Reads patient records from the local MIMIC-IV FHIR demo NDJSON files (the
same data that is loaded into the FHIR store), so no GCP access is needed.
Writes manifests/starting_packets.jsonl with the resource IDs in each packet;
the packet text can be rebuilt with tools.starting_packet.

Gold resource IDs are never used to build packets. They are only used for the
diagnostics printed for the pilot and dev splits (held-out is not reported, so
it cannot leak into tuning). Use --sweep to compare token budgets on dev.

Usage:
    python scripts/build_starting_packets.py
    python scripts/build_starting_packets.py --sweep 2000 4000 8000 16000
"""
import argparse
import ast
import glob
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.starting_packet import PACKET_MAX_TOKENS, build_starting_packet, parse_current_time  # noqa: E402

DEFAULT_FHIR_DIR = os.path.join(ROOT, "physionet.org", "files", "mimic-iv-fhir-demo", "2.1.0", "fhir")
DEFAULT_INPUT = os.path.join(ROOT, "final_dataset", "questions_answers_sql_fhir.csv")
DEFAULT_SPLIT = os.path.join(ROOT, "manifests", "split.json")
DEFAULT_FAMILIES = os.path.join(ROOT, "manifests", "question_families.csv")
DEFAULT_OUTPUT = os.path.join(ROOT, "manifests", "starting_packets.jsonl")

# Not patient-scoped: loaded in full, as the store's patient search returns them
SHARED_FILES = ("MimicPatient", "MimicMedication", "MimicMedicationMix", "MimicLocation")
# Never used by the packet, and absent from the store's patient search
SKIPPED_FILES = ("MimicOrganization", "MimicMedicationAdministration", "MimicMedicationAdministrationICU",
                 "MimicMedicationDispense", "MimicMedicationDispenseED", "MimicMedicationStatementED")
SUBJECT_RE = re.compile(rb'"subject": \{"reference": "Patient/([0-9a-f-]+)"')


def index_patient_lines(fhir_dir, patients):
    """Return shared resources and {patient: [(path, offset), ...]} for patient-scoped files."""
    shared, offsets = [], defaultdict(list)
    for path in sorted(glob.glob(os.path.join(fhir_dir, "*.ndjson"))):
        name = os.path.basename(path)[: -len(".ndjson")]
        if name in SKIPPED_FILES:
            continue
        with open(path, "rb") as f:
            if name in SHARED_FILES:
                shared.extend(json.loads(line) for line in f)
                continue
            offset = 0
            for line in f:
                match = SUBJECT_RE.search(line)
                if match and match.group(1).decode() in patients:
                    offsets[match.group(1).decode()].append((path, offset))
                offset += len(line)
    return shared, offsets


def load_record(shared, locations):
    resources = list(shared)
    handles = {}
    try:
        for path, offset in locations:
            f = handles.setdefault(path, open(path, "rb"))
            f.seek(offset)
            resources.append(json.loads(f.readline()))
    finally:
        for f in handles.values():
            f.close()
    record = defaultdict(list)
    for r in resources:
        record[r["resourceType"]].append(r)
    return record


def gold_recall(packet_ids, gold):
    packet = {(t, i) for t, ids in packet_ids.items() for i in ids}
    gold_pairs = {(t, i) for t, ids in gold.items() for i in ids}
    if not gold_pairs:
        return None, None
    hit = len(packet & gold_pairs)
    return hit / len(gold_pairs), hit == len(gold_pairs)


def summarize(rows, label):
    df = pd.DataFrame(rows)
    with_gold = df.dropna(subset=["recall"])
    print(f"\n[{label}] {len(df)} questions ({len(with_gold)} with gold IDs)")
    print(f"  tokens: median {df.n_tokens.median():.0f}, max {df.n_tokens.max()}; "
          f"truncated {df.truncated.mean():.0%}; encounters omitted in {(df.omitted_encounters > 0).mean():.0%}")
    print(f"  gold recall: mean {with_gold.recall.mean():.1%}; all gold in packet {with_gold.complete.mean():.1%}")
    by_family = with_gold.groupby("family").agg(n=("recall", "size"), recall=("recall", "mean"),
                                                complete=("complete", "mean"))
    print(by_family.to_string(float_format=lambda x: f"{x:.1%}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fhir-dir", default=DEFAULT_FHIR_DIR)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--families", default=DEFAULT_FAMILIES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tokens", type=int, default=PACKET_MAX_TOKENS)
    parser.add_argument("--sweep", type=int, nargs="+",
                        help="Report dev diagnostics for these budgets instead of writing the manifest")
    args = parser.parse_args()

    if not os.path.isdir(args.fhir_dir):
        sys.exit(f"FHIR NDJSON directory not found: {args.fhir_dir} (run scripts/setup_fhir_store.sh)")

    split = json.load(open(args.split))
    split_of = {q: name for name, s in split["splits"].items() for q in s["question_ids"]}
    if args.sweep:
        split_of = {q: name for q, name in split_of.items() if name == "dev"}

    df = pd.read_csv(args.input, usecols=["question_id", "question", "assumption", "patient_fhir_id", "true_fhir_ids"])
    df = df[df.question_id.isin(split_of)]
    families = pd.read_csv(args.families, usecols=["question_id", "family"])
    df = df.merge(families, on="question_id", validate="one_to_one")

    print(f"Indexing {args.fhir_dir} for {df.patient_fhir_id.nunique()} patients...")
    shared, offsets = index_patient_lines(args.fhir_dir, set(df.patient_fhir_id))

    budgets = args.sweep or [args.max_tokens]
    rows = {b: [] for b in budgets}
    manifest = []
    for patient, questions in df.groupby("patient_fhir_id", sort=True):
        record = load_record(shared, offsets[patient])
        for q in questions.itertuples():
            gold = ast.literal_eval(q.true_fhir_ids)
            for budget in budgets:
                current_time = parse_current_time(q.assumption if isinstance(q.assumption, str) else "")
                packet = build_starting_packet(q.question, record, patient, budget, current_time)
                if packet["n_tokens"] > budget:
                    sys.exit(f"Packet for {q.question_id} exceeds budget ({packet['n_tokens']} > {budget})")
                recall, complete = gold_recall(packet["resource_ids"], gold)
                rows[budget].append({"split": split_of[q.question_id], "family": q.family,
                                     "n_tokens": packet["n_tokens"], "truncated": packet["truncated"],
                                     "omitted_encounters": packet["omitted_encounters"],
                                     "recall": recall, "complete": complete})
                if not args.sweep:
                    manifest.append({"question_id": q.question_id, "split": split_of[q.question_id],
                                     "patient_fhir_id": patient, "max_tokens": budget, "current_time": current_time,
                                     "n_tokens": packet["n_tokens"], "truncated": packet["truncated"],
                                     "omitted_encounters": packet["omitted_encounters"],
                                     "after_current_time": packet["after_current_time"],
                                     "resource_ids": packet["resource_ids"]})

    for budget in budgets:
        for name in ("pilot", "dev"):
            split_rows = [r for r in rows[budget] if r["split"] == name]
            if split_rows:
                summarize(split_rows, f"{name}, max_tokens={budget}")

    if not args.sweep:
        manifest.sort(key=lambda m: (m["split"], m["question_id"]))
        with open(args.output, "w") as f:
            for m in manifest:
                f.write(json.dumps(m) + "\n")
        print(f"\nWrote {len(manifest)} packets to {os.path.relpath(args.output, ROOT)}")


if __name__ == "__main__":
    main()
