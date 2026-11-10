"""Build manifests/perturbations_multi.json: multi-record missing-evidence perturbations.

build_perturbations.py leaves exactly one answer-changing record out of a
starting packet, and skips multi-value answers. This builder makes three
richer kinds, with the same proof: over the exact visible packet fields
(scripts/packet_query.py), the original query reproduces the reference and
the omitted packet's result differs and scores wrong.

  link_target  leave out a Medication or Specimen that a visible record of
               the query's main table links to (the omission also redacts
               the reference, as tools.perturbation.omit_from_packet does),
               so the drug name or specimen type must be fetched
  tie_peer     multi-value answers (tie sets, e.g. T051/T052): keep one
               cited record of the tie and leave out the other k-1
  k_record     counts, existence checks and aggregates: leave out the
               smallest k (2..--max-k) cited records that changes the answer

Except for link targets, a removed record must have no incoming references
from other visible clinical records. Up to --size questions per kind are
sampled with the split seed, in proportion to family; each question gets at
most one kind (rarest kind first), since the runtime keys omissions by question.

tools/perturbation.load_perturbations runs this version-3 manifest; pass it
to scripts/run_experiments.py with --perturbations. Only aggregate counts
are printed.

Usage:
    python scripts/build_multi_perturbations.py
    python scripts/build_multi_perturbations.py --verify-manifest manifests/perturbations_multi.json
"""
import argparse
import ast
import json
import os
import sys
from collections import Counter

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.build_perturbations import (DEFAULT_PACKETS, SHARED_TYPES, _scalars, _sha256, carries,  # noqa: E402
                                         event_gold, references, same_result, sample)
from scripts.build_starting_packets import (DEFAULT_FHIR_DIR, DEFAULT_FAMILIES, DEFAULT_INPUT, DEFAULT_SPLIT,  # noqa: E402
                                            file_sha256, index_patient_lines, load_record)
from scripts.packet_query import PacketQuery, UnsupportedQuery, visible_resources  # noqa: E402
from tools.perturbation import omit_from_packet, text_sha256  # noqa: E402
from tools.starting_packet import PACKET_VERSION, _reference_key, _resource_time, build_starting_packet  # noqa: E402
from utils.answer_scorer import answer_type, parse_reference, score_answer  # noqa: E402

DEFAULT_OUTPUT = os.path.join(ROOT, "manifests", "perturbations_multi.json")
KINDS = ("link_target", "tie_peer", "k_record")
SIZE = 50
MAX_K = 5
LINK_TYPES = ("Medication", "Specimen")


def _answer(result):
    return result[0] if len(result) == 1 else result or None


def removal_sets(kind, question, oracle, original, by_id, values):
    """Candidate removal sets [(resourceType, id), ...] for one kind, most plausible first."""
    gold = ast.literal_eval(question["true_fhir_ids"]) if isinstance(question["true_fhir_ids"], str) else {}
    annotated = set(event_gold(gold))
    resources = visible_resources(original)
    clinical = [r for r in resources if r["resourceType"] not in SHARED_TYPES]
    candidates = [r for r in clinical if r["resourceType"] != "Specimen" and oracle.candidate(r)]

    def referenced(record):
        key = _reference_key(record)
        return any(key in references(other) for other in clinical if other is not record)

    if kind == "link_target":
        shown = {(r["resourceType"], r["id"]) for r in resources if r["resourceType"] in LINK_TYPES}
        targets = []
        for record in candidates:
            for ref in references({k: v for k, v in record.items() if k in ("medicationReference", "specimen")}):
                pair = tuple(ref.split("/", 1))
                if pair in shown and pair not in targets:
                    targets.append(pair)
        cited = [t for t in targets if t in annotated]
        return [[t] for t in cited + [t for t in targets if t not in cited]] + ([targets] if len(targets) > 1 else [])
    cited = sorted((r for r in candidates if (r["resourceType"], r["id"]) in annotated and not referenced(r)),
                   key=lambda r: (_resource_time(by_id[_reference_key(r)]), _reference_key(r)))
    pairs = [(r["resourceType"], r["id"]) for r in cited]
    if kind == "tie_peer":
        if len(values) < 2 or oracle.rule != "selection" or len(pairs) < 2:
            return []
        return [pairs[1:]]  # keep the earliest cited record of the tie
    if kind == "k_record":
        if oracle.rule == "selection":
            return []
        return [pairs[-k:] for k in range(2, min(MAX_K, len(pairs)) + 1)]  # the k most recent cited records
    raise ValueError(f"Unknown kind {kind}")


def prepare(question, packet_row, record):
    """(original packet, by_id, oracle, visible result) for one question, or (None, exclusion reason)."""
    if answer_type(question["true_answer"], question["template"]) == "empty":
        return None, "empty answer"
    by_id = {_reference_key(r): r for items in record.values() for r in items}
    original = build_starting_packet(question["question"], record, question["patient_fhir_id"],
                                     packet_row["max_tokens"], packet_row["current_time"])
    if _sha256(original["text"]) != packet_row["packet_text_sha256"]:
        raise ValueError(f"Rebuilt packet for {question['question_id']} does not match the frozen manifest")
    try:
        oracle = PacketQuery(question, by_id)
        before = oracle.run(original)
    except UnsupportedQuery as exc:
        return None, str(exc)
    if not same_result(parse_reference(question["true_answer"]), before):
        return None, "visible query does not reproduce reference"
    return (original, by_id, oracle, before), None


def evaluate(question, packet_row, prepared, family, template, kind):
    """(entry, None) for an eligible question of this kind, or (None, exclusion reason).

    `prepared` comes from prepare().
    """
    original, by_id, oracle, before = prepared
    values = parse_reference(question["true_answer"])
    patient, now = question["patient_fhir_id"], packet_row["current_time"]
    for removed in removal_sets(kind, question, oracle, original, by_id, values):
        trial = omit_from_packet(original, removed)
        try:
            after = oracle.run(trial)
        except UnsupportedQuery:
            continue
        if same_result(before, after):
            continue
        if score_answer(_answer(after), question["true_answer"], question["question"], template)["typed_correct"]:
            continue
        # A retained record must not literally state a removed selection value
        if kind != "tie_peer" and oracle.rule == "selection" and any(
                carries(v, leaf) for v in values for r in visible_resources(trial)
                for leaf in _scalars(r)):
            continue
        return {
            "question_id": question["question_id"],
            "kind": kind,
            "family": family,
            "template_id": question["template_id"],
            "patient_fhir_id": patient,
            "current_time": now,
            "rule": oracle.rule,
            "packet_version": PACKET_VERSION,
            "max_tokens": packet_row["max_tokens"],
            "question_sha256": text_sha256(question["question"]),
            "fhir_source_sha256": packet_row.get("fhir_source_sha256"),
            "removed": [{"resource_type": rtype, "id": rid} for rtype, rid in removed],
            "original_packet_sha256": packet_row["packet_text_sha256"],
            "perturbed_packet_sha256": _sha256(trial["text"]),
            "perturbed_resource_ids": trial["resource_ids"],
            "perturbed_n_tokens": trial["n_tokens"],
            "proof": {"query_sha256": text_sha256(oracle.sql),
                      "original_result_sha256": text_sha256(json.dumps(before)),
                      "perturbed_result_sha256": text_sha256(json.dumps(after)),
                      "perturbed_typed_correct": 0,
                      "oracle": "visible-packet-sql-v1"},
        }, None
    return None, f"no {kind} omission changes the visible query"


def load_inputs(args, question_ids):
    with open(args.packets, encoding="utf-8") as stream:
        packets = {row["question_id"]: row for row in map(json.loads, stream) if row["question_id"] in question_ids}
    if set(packets) != set(question_ids):
        raise ValueError("The starting-packet manifest does not cover the selected questions")
    df = pd.read_csv(args.input, usecols=["question_id", "question", "true_answer", "patient_fhir_id", "true_fhir_ids",
                                          "sql_query", "val_dict", "main_table_name", "assumption"])
    df = df[df.question_id.isin(question_ids)]
    families = pd.read_csv(args.families, usecols=["question_id", "family", "template_id", "template"])
    return packets, df.merge(families, on="question_id", validate="one_to_one")


def evaluate_all(df, packets, fhir_dir, kinds):
    shared, offsets = index_patient_lines(fhir_dir, set(df.patient_fhir_id))
    eligible, reasons = {kind: [] for kind in kinds}, {kind: Counter() for kind in kinds}
    for patient, questions in df.groupby("patient_fhir_id", sort=True):
        record = load_record(shared, offsets[patient])
        for q in questions.to_dict("records"):
            prepared, reason = prepare(q, packets[q["question_id"]], record)
            for kind in kinds:
                if prepared:
                    entry, reason = evaluate(q, packets[q["question_id"]], prepared, q["family"], q["template"], kind)
                else:
                    entry = None
                if entry:
                    eligible[kind].append(entry)
                else:
                    reasons[kind][reason] += 1
    return eligible, reasons


def verify_manifest(path, args):
    """Rebuild every frozen entry from local sources; returns counts only."""
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("version") != 3 or manifest.get("packet_version") != PACKET_VERSION:
        raise ValueError("Not a version-3 multi-record perturbation manifest for the current packets")
    entries = manifest["perturbations"]
    packets, df = load_inputs(args, {e["question_id"] for e in entries})
    shared, offsets = index_patient_lines(args.fhir_dir, set(df.patient_fhir_id))
    rows = df.set_index("question_id")
    records = {}
    for expected in entries:
        q = {**rows.loc[expected["question_id"]].to_dict(), "question_id": expected["question_id"]}
        patient = q["patient_fhir_id"]
        if patient not in records:
            records[patient] = load_record(shared, offsets[patient])
        prepared, reason = prepare(q, packets[q["question_id"]], records[patient])
        entry = None
        if prepared:
            entry, reason = evaluate(q, packets[q["question_id"]], prepared, q["family"], q["template"],
                                     expected["kind"])
        if entry != expected:
            raise ValueError(f"Frozen {expected['kind']} omission does not reproduce: {reason or 'entry differs'}")
    return {"verified": len(entries), "by_kind": dict(Counter(e["kind"] for e in entries))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fhir-dir", default=DEFAULT_FHIR_DIR)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--families", default=DEFAULT_FAMILIES)
    parser.add_argument("--packets", default=DEFAULT_PACKETS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=SIZE, help="Questions per kind (fewer if fewer are eligible)")
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--split-name", choices=("pilot", "dev", "held_out"), default="held_out")
    parser.add_argument("--verify-manifest", metavar="PATH")
    args = parser.parse_args()

    if args.verify_manifest:
        print(json.dumps(verify_manifest(args.verify_manifest, args), indent=1))
        return
    with open(args.split, encoding="utf-8") as stream:
        split = json.load(stream)
    question_ids = set(split["splits"][args.split_name]["question_ids"])
    packets, df = load_inputs(args, question_ids)
    source_hash = file_sha256(args.input)
    if any(row["source_sha256"] != source_hash for row in packets.values()):
        raise ValueError("The starting packets were built from a different dataset")

    print(f"Indexing {args.fhir_dir} for {df.patient_fhir_id.nunique()} patients...")
    eligible, reasons = evaluate_all(df, packets, args.fhir_dir, args.kinds)
    # One specification per question: kinds are sampled rarest first, each from questions not yet taken
    chosen, taken = [], set()
    for kind in sorted(args.kinds, key=lambda k: len(eligible[k])):
        available = [e for e in eligible[kind] if e["question_id"] not in taken]
        if available:
            picked = sample(available, min(args.size, len(available)), split["seed"])
            chosen.extend(picked)
            taken.update(e["question_id"] for e in picked)
    chosen.sort(key=lambda e: (KINDS.index(e["kind"]), e["question_id"]))
    manifest = {
        "version": 3,
        "packet_version": PACKET_VERSION,
        "split": args.split_name,
        "seed": split["seed"],
        "size": len(chosen),
        "source_sha256": source_hash,
        "split_sha256": file_sha256(args.split),
        "families_sha256": file_sha256(args.families),
        "packets_sha256": file_sha256(args.packets),
        "builder_sha256": file_sha256(os.path.abspath(__file__)),
        "oracle_sha256": file_sha256(os.path.join(ROOT, "scripts", "packet_query.py")),
        "omitter_sha256": file_sha256(os.path.join(ROOT, "tools", "perturbation.py")),
        "scorer_sha256": file_sha256(os.path.join(ROOT, "utils", "answer_scorer.py")),
        "fhir_source_sha256": next(iter(packets.values()))["fhir_source_sha256"],
        "eligibility": {kind: {"eligible": len(eligible[kind]),
                               "eligible_by_family": dict(sorted(Counter(e["family"] for e in eligible[kind]).items())),
                               "excluded_by_reason": dict(sorted(reasons[kind].items()))} for kind in args.kinds},
        "selected_by_kind": dict(sorted(Counter(e["kind"] for e in chosen).items())),
        "removed_records": dict(sorted(Counter(len(e["removed"]) for e in chosen).items())),
        "perturbations": chosen,
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=1)
        stream.write("\n")
    print(json.dumps({k: manifest[k] for k in ("selected_by_kind", "removed_records")}, indent=1))
    for kind in args.kinds:
        print(f"{kind}: {len(eligible[kind])} eligible of {len(df)}")
    print(f"Wrote {len(chosen)} perturbations to {args.output}")


if __name__ == "__main__":
    main()
