"""Build starting packets for every question in manifests/split.json.

Reads patient records from the local MIMIC-IV FHIR demo NDJSON files (the
same data that is loaded into the FHIR store), so no GCP access is needed.
Writes manifests/starting_packets.jsonl with the resource IDs in each packet;
the packet text can be rebuilt with tools.starting_packet.

Gold labels are not read during packet generation or --sweep. The sweep
reports token and truncation statistics on dev only. Freeze the generated
manifest before --gold-diagnostics; that separate command measures pilot/dev
ID recall from the existing manifest without rebuilding or optimizing packets.

Usage:
    python scripts/build_starting_packets.py
    python scripts/build_starting_packets.py --sweep 2000 4000 8000 16000
"""
import argparse
import ast
import glob
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.starting_packet import PACKET_MAX_TOKENS, PACKET_VERSION, build_starting_packet, parse_current_time  # noqa: E402

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


def open_ndjson(path):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def file_sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def ndjson_paths(fhir_dir):
    paths = sorted(glob.glob(os.path.join(fhir_dir, "*.ndjson")))
    compressed = sorted(glob.glob(os.path.join(fhir_dir, "*.ndjson.gz")))
    paths.extend(path for path in compressed if path[:-3] not in paths)
    if not paths:
        raise ValueError(f"No FHIR NDJSON files found in {fhir_dir}")
    return [path for path in paths
            if os.path.basename(path).removesuffix(".gz").removesuffix(".ndjson") not in SKIPPED_FILES]


def index_patient_lines(fhir_dir, patients):
    """Return shared resources and {patient: [(path, offset), ...]} for patient-scoped files."""
    shared, offsets = [], defaultdict(list)
    for path in ndjson_paths(fhir_dir):
        name = os.path.basename(path).removesuffix(".gz").removesuffix(".ndjson")
        with open_ndjson(path) as f:
            if name in SHARED_FILES:
                shared.extend(json.loads(line) for line in f)
                continue
            offset = 0
            for line in f:
                resource = json.loads(line)
                patient = resource.get("subject", {}).get("reference", "").split("/")[-1]
                if patient in patients:
                    offsets[patient].append((path, offset))
                offset += len(line)
    return shared, offsets


def load_record(shared, locations):
    resources = list(shared)
    handles = {}
    try:
        for path, offset in locations:
            if path not in handles:
                handles[path] = open_ndjson(path)
            f = handles[path]
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
    print(f"\n[{label}] {len(df)} questions")
    print(f"  tokens: median {df.n_tokens.median():.0f}, max {df.n_tokens.max()}; "
          f"truncated {df.truncated.mean():.0%}; encounters omitted in {(df.omitted_encounters > 0).mean():.0%}")
    by_family = df.groupby("family").agg(n=("n_tokens", "size"), median_tokens=("n_tokens", "median"),
                                         truncated=("truncated", "mean"))
    print(by_family.to_string())


def frozen_gold_diagnostics(manifest_path, input_path, split):
    """Score already frozen pilot/dev IDs; never inspect held-out labels."""
    with open(manifest_path) as handle:
        manifest = [json.loads(line) for line in handle if line.strip()]
    eligible = {q: name for name in ("pilot", "dev") for q in split["splits"][name]["question_ids"]}
    packets = {m["question_id"]: m for m in manifest if m["question_id"] in eligible}
    if set(packets) != set(eligible):
        raise ValueError("Frozen packet manifest does not cover the current pilot/dev split")
    if any(m.get("packet_version") != PACKET_VERSION for m in packets.values()):
        raise ValueError("Freeze a manifest with the current packet version before diagnostics")
    source_hash = file_sha256(input_path)
    if any(m.get("source_sha256") != source_hash for m in packets.values()):
        raise ValueError("Frozen packet source does not match the diagnostic dataset")
    if any(m.get("split") != eligible[m["question_id"]] for m in packets.values()):
        raise ValueError("Frozen packets do not match the current split assignments")
    df = pd.read_csv(input_path, usecols=["question_id", "true_fhir_ids"])
    df = df[df.question_id.isin(eligible)]
    for name in ("pilot", "dev"):
        recalls = []
        for q in df.itertuples():
            if eligible[q.question_id] == name:
                recall, complete = gold_recall(packets[q.question_id]["resource_ids"], ast.literal_eval(q.true_fhir_ids))
                if recall is not None:
                    recalls.append((recall, complete))
        if recalls:
            print(f"[{name}] frozen gold-ID recall {sum(r for r, _ in recalls) / len(recalls):.1%}; "
                  f"all IDs included {sum(c for _, c in recalls) / len(recalls):.1%} ({len(recalls)} questions)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fhir-dir", default=DEFAULT_FHIR_DIR)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--families", default=DEFAULT_FAMILIES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tokens", type=int, default=PACKET_MAX_TOKENS)
    parser.add_argument("--sweep", type=int, nargs="+",
                        help="Report gold-free dev token/truncation statistics instead of writing the manifest")
    parser.add_argument("--gold-diagnostics", action="store_true",
                        help="Score pilot/dev IDs in the frozen --output manifest; never rebuild packets")
    parser.add_argument("--fhir-source-uri", help="Provenance URL for the local source files (saved with checksums)")
    args = parser.parse_args()

    with open(args.split) as handle:
        split = json.load(handle)
    if args.gold_diagnostics:
        if args.sweep:
            parser.error("--gold-diagnostics and --sweep cannot be combined")
        frozen_gold_diagnostics(args.output, args.input, split)
        return
    if not os.path.isdir(args.fhir_dir):
        sys.exit(f"FHIR NDJSON directory not found: {args.fhir_dir} (run scripts/setup_fhir_store.sh)")
    split_of = {q: name for name, s in split["splits"].items() for q in s["question_ids"]}
    if args.sweep:
        split_of = {q: name for q, name in split_of.items() if name == "dev"}

    df = pd.read_csv(args.input, usecols=["question_id", "question", "assumption", "patient_fhir_id"])
    df = df[df.question_id.isin(split_of)]
    families = pd.read_csv(args.families, usecols=["question_id", "family"])
    df = df.merge(families, on="question_id", validate="one_to_one")

    print(f"Indexing {args.fhir_dir} for {df.patient_fhir_id.nunique()} patients...")
    shared, offsets = index_patient_lines(args.fhir_dir, set(df.patient_fhir_id))

    budgets = args.sweep or [args.max_tokens]
    source_files = [{"file": os.path.basename(path), "sha256": file_sha256(path)} for path in ndjson_paths(args.fhir_dir)]
    fhir_source_hash = hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
    fingerprints = {"source_sha256": file_sha256(args.input), "split_sha256": file_sha256(args.split),
                    "retriever_sha256": file_sha256(os.path.join(ROOT, "tools", "starting_packet.py")),
                    "fhir_source_sha256": fhir_source_hash}
    rows = {b: [] for b in budgets}
    manifest = []
    for patient, questions in df.groupby("patient_fhir_id", sort=True):
        record = load_record(shared, offsets[patient])
        for q in questions.itertuples():
            for budget in budgets:
                current_time = parse_current_time(q.assumption if isinstance(q.assumption, str) else "")
                packet = build_starting_packet(q.question, record, patient, budget, current_time)
                if packet["n_tokens"] > budget:
                    sys.exit(f"Packet for {q.question_id} exceeds budget ({packet['n_tokens']} > {budget})")
                rows[budget].append({"split": split_of[q.question_id], "family": q.family,
                                     "n_tokens": packet["n_tokens"], "truncated": packet["truncated"],
                                     "omitted_encounters": packet["omitted_encounters"]})
                if not args.sweep:
                    manifest.append({"question_id": q.question_id, "split": split_of[q.question_id],
                                     "patient_fhir_id": patient, "max_tokens": budget, "current_time": current_time,
                                     "packet_version": PACKET_VERSION,
                                     **fingerprints,
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
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            for m in manifest:
                f.write(json.dumps(m) + "\n")
        source_output = os.path.splitext(args.output)[0] + ".source.json"
        with open(source_output, "w") as handle:
            json.dump({"source_uri": args.fhir_source_uri, "fhir_source_sha256": fhir_source_hash,
                       "files": source_files}, handle, indent=2)
            handle.write("\n")
        print(f"\nWrote {len(manifest)} packets to {os.path.relpath(args.output, ROOT)}")


if __name__ == "__main__":
    main()
