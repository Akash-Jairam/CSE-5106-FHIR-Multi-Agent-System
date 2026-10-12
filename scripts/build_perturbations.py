"""Build manifests/perturbations.json: missing-evidence perturbations of held-out questions.

For each eligible held-out question, exactly one answer-changing resource is
left out of the frozen starting packet, without reranking or filling its gap.
The FHIR store is unchanged. Eligibility uses a read-only offline SQL oracle
over the exact visible packet fields: the original query must reproduce the
reference, and the same query over the omitted packet must yield a different
result. Operation and scope come from SQL, never from matching gold values.
SQL/templates/labels remain confined to this diagnostic construction code.

Candidates must not have incoming clinical references whose removal would
hide other records' scope/time information. Retained metadata and records
must not state the removed selection value. Gold IDs prioritize candidates
but do not stand in for query completeness or answer-change proofs.

Multi-value answers are not eligible. 50 questions are sampled with the
split seed, allocated to families in proportion to their eligible counts.
Only aggregate counts are printed; held-out answers are never shown.

Usage:
    python scripts/build_perturbations.py
"""
import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.build_starting_packets import (DEFAULT_FHIR_DIR, DEFAULT_FAMILIES, DEFAULT_INPUT, DEFAULT_SPLIT,  # noqa: E402
                                            file_sha256, index_patient_lines, load_record, ndjson_paths)
from scripts.packet_query import PacketQuery, UnsupportedQuery, visible_resources  # noqa: E402
from tools.perturbation import (load_perturbations, omit_from_packet, text_sha256, validate_context,
                                verified_omission)  # noqa: E402
from tools.starting_packet import PACKET_VERSION, _label, _reference_key, _resource_time, build_starting_packet  # noqa: E402
from utils.answer_scorer import answer_type, parse_reference, score_answer  # noqa: E402

DEFAULT_PACKETS = os.path.join(ROOT, "manifests", "starting_packets.jsonl")
DEFAULT_OUTPUT = os.path.join(ROOT, "manifests", "perturbations.json")
SAMPLE_SIZE = 50
SHARED_TYPES = ("Patient", "Medication", "Location")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")


def event_gold(gold):
    """Gold (resourceType, id) pairs other than shared resources, in a stable order."""
    return sorted((rtype, rid) for rtype, ids in gold.items() if rtype not in SHARED_TYPES for rid in ids)


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def _scalars(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _scalars(item)
    elif isinstance(value, list):
        for item in value:
            yield from _scalars(item)
    elif value is not None:
        yield value


def leaves(resource, by_id):
    """Every scalar value of a resource, of the resources it references, and its label's parts."""
    values = list(_scalars(resource))
    for ref in _scalars({k: v for k, v in resource.items() if k not in ("subject", "patient")}):
        if isinstance(ref, str) and ref.count("/") == 1 and ref in by_id:
            values.extend(_scalars(by_id[ref]))
    label = _label(resource, by_id)
    values.extend(part.strip() for part in label.split(" / ") if part.strip())
    return values


def carries(value, leaf):
    """Whether a record leaf states the reference answer value."""
    if isinstance(value, str) and TIMESTAMP_RE.match(value.strip()):
        precision = 19 if len(value.strip()) >= 19 else 16
        return isinstance(leaf, str) and bool(TIMESTAMP_RE.match(leaf)) and \
            leaf[:precision].replace("T", " ") == value.strip()[:precision].replace("T", " ")
    number = _number(value)
    if number is not None:
        other = _number(leaf)
        return other is not None and _close(number, other)
    return isinstance(leaf, str) and leaf.strip().casefold() == str(value).strip().casefold()


def redundant(values, target, perturbed_ids, by_id):
    """Whether a remaining record of the target's type in the perturbed packet carries the answer."""
    rtype = target[0]
    for rid in perturbed_ids.get(rtype, []):
        record_leaves = leaves(by_id[f"{rtype}/{rid}"], by_id)
        if any(carries(value, leaf) for value in values for leaf in record_leaves):
            return True
    return False


def evaluate(question, packet_row, record, family, template):
    """(entry, None) for an eligible question, or (None, exclusion reason)."""
    gold = ast.literal_eval(question["true_fhir_ids"]) if isinstance(question["true_fhir_ids"], str) else {}
    values = parse_reference(question["true_answer"])
    kind = answer_type(question["true_answer"], template)
    if kind == "empty":
        return None, "empty answer"
    if kind == "set":
        return None, "multi-value answer"
    by_id = {_reference_key(r): r for items in record.values() for r in items}
    patient, now = question["patient_fhir_id"], packet_row["current_time"]
    original = build_starting_packet(question["question"], record, patient, packet_row["max_tokens"], now)
    if _sha256(original["text"]) != packet_row["packet_text_sha256"]:
        raise ValueError(f"Rebuilt packet for {question['question_id']} does not match the frozen manifest")
    try:
        oracle = PacketQuery(question, by_id)
        before = oracle.run(original)
    except UnsupportedQuery as exc:
        return None, str(exc)
    if not same_result(values, before):
        return None, "visible query does not reproduce reference"
    resources = visible_resources(original)
    annotated = set(event_gold(gold))
    candidates = [r for r in resources if r["resourceType"] not in SHARED_TYPES and oracle.candidate(r)]
    candidates.sort(key=lambda r: ((r["resourceType"], r["id"]) in annotated,
                                   _resource_time(by_id[_reference_key(r)]), _reference_key(r)), reverse=True)
    perturbed = target = after = None
    for candidate in candidates:
        pair = (candidate["resourceType"], candidate["id"])
        key = _reference_key(candidate)
        # A one-record omission must not erase other clinical records' linking,
        # ordering or answer fields to manufacture an answer change.
        if any(key in references(other) for other in resources
               if other is not candidate and other["resourceType"] not in SHARED_TYPES):
            continue
        trial = omit_from_packet(original, [pair])
        try:
            result = oracle.run(trial)
        except UnsupportedQuery:
            continue
        if same_result(before, result):
            continue
        answer = result[0] if len(result) == 1 else result or None
        if score_answer(answer, question["true_answer"], question["question"], template)["typed_correct"]:
            continue
        # Retained metadata or an unrelated record must not literally supply a
        # supposedly removed selection value (e.g. a drug name in Medication).
        if oracle.rule == "selection" and any(carries(v, leaf) for v in values
                                               for r in visible_resources(trial) for leaf in _scalars(r)):
            continue
        perturbed, target, after = trial, pair, result
        break
    if perturbed is None:
        return None, "no one-resource omission changes the visible query without redundant support"
    return {
        "question_id": question["question_id"],
        "family": family,
        "template_id": question["template_id"],
        "patient_fhir_id": patient,
        "current_time": now,
        "rule": oracle.rule,
        "packet_version": PACKET_VERSION,
        "max_tokens": packet_row["max_tokens"],
        "question_sha256": text_sha256(question["question"]),
        "fhir_source_sha256": packet_row.get("fhir_source_sha256"),
        "removed": {"resource_type": target[0], "id": target[1]},
        "original_packet_sha256": packet_row["packet_text_sha256"],
        "perturbed_packet_sha256": _sha256(perturbed["text"]),
        "perturbed_resource_ids": perturbed["resource_ids"],
        "perturbed_n_tokens": perturbed["n_tokens"],
        "proof": {"query_sha256": text_sha256(oracle.sql),
                  "original_result_sha256": text_sha256(json.dumps(before)),
                  "perturbed_result_sha256": text_sha256(json.dumps(after)),
                  "perturbed_typed_correct": 0,
                  "oracle": "visible-packet-sql-v1"},
    }, None


def references(value):
    if isinstance(value, dict):
        reference = value.get("reference")
        if isinstance(reference, str):
            yield "/".join(reference.rstrip("/").split("/")[-2:])
        for item in value.values():
            yield from references(item)
    elif isinstance(value, list):
        for item in value:
            yield from references(item)


def same_result(left, right):
    """Typed one-column SQL results, including normalized empty aggregate results."""
    left = [v for v in left if v is not None]
    right = [v for v in right if v is not None]
    if len(left) != len(right):
        return False
    left, right = sorted(left, key=str), sorted(right, key=str)
    return all(carries(a, b) for a, b in zip(left, right))


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sample(eligible, size, seed):
    """`size` entries, allocated to families in proportion (largest remainder), seeded within each."""
    by_family = defaultdict(list)
    for entry in eligible:
        by_family[entry["family"]].append(entry)
    if len(eligible) < size:
        raise ValueError(f"Only {len(eligible)} eligible questions for a sample of {size}")
    quotas = {f: size * len(items) / len(eligible) for f, items in by_family.items()}
    counts = {f: int(q) for f, q in quotas.items()}
    for f in sorted(quotas, key=lambda f: (-(quotas[f] - counts[f]), f))[:size - sum(counts.values())]:
        counts[f] += 1
    rng = random.Random(seed)
    chosen = []
    for family in sorted(by_family):
        items = sorted(by_family[family], key=lambda e: e["question_id"])
        chosen.extend(rng.sample(items, counts[family]))
    return sorted(chosen, key=lambda e: e["question_id"])


def verify_manifest(fhir_dir, path=DEFAULT_OUTPUT, input_path=DEFAULT_INPUT, families_path=DEFAULT_FAMILIES,
                    packets_path=DEFAULT_PACKETS):
    """Verify every frozen entry against source, visible SQL and the runtime path.

    Returns counts only; never prints gold answers or exposes them to agents.
    """
    entries = load_perturbations(path)
    source_files = [{"file": os.path.basename(p), "sha256": file_sha256(p)} for p in ndjson_paths(fhir_dir)]
    source_hash = text_sha256(json.dumps(source_files, sort_keys=True))
    if any(entry["fhir_source_sha256"] != source_hash for entry in entries.values()):
        raise ValueError("Local FHIR sources do not match the frozen omission experiment")
    with open(packets_path, encoding="utf-8") as stream:
        packets = {p["question_id"]: p for p in map(json.loads, stream)}
    questions = pd.read_csv(input_path)
    questions = questions[questions.question_id.isin(entries)]
    families = pd.read_csv(families_path, usecols=["question_id", "family", "template_id"])
    questions = questions.merge(families, on="question_id", validate="one_to_one")
    if set(questions.question_id) != set(entries):
        raise ValueError("The dataset does not cover every frozen perturbation")
    shared, offsets = index_patient_lines(fhir_dir, set(questions.patient_fhir_id))
    verified = 0
    for patient, group in questions.groupby("patient_fhir_id", sort=True):
        record = load_record(shared, offsets[patient])
        for q in group.to_dict("records"):
            expected = entries[q["question_id"]]
            entry, reason = evaluate(q, packets[q["question_id"]], record, q["family"], q["template"])
            if entry != expected:
                raise ValueError("Frozen omission does not reproduce its visible-query proof: " + (reason or "entry differs"))
            validate_context(q["question"], patient, expected, expected["max_tokens"], expected["current_time"])
            original = build_starting_packet(q["question"], record, patient, expected["max_tokens"], expected["current_time"])
            omitted = verified_omission(original, expected)
            removed = expected["removed"]
            if removed["id"] in omitted["text"]:
                raise ValueError("An omitted resource ID is still visible in the packet")
            for rtype, resources in original["resources"].items():
                if rtype in SHARED_TYPES:
                    continue
                kept = [r for r in resources if (rtype, r["id"]) != (removed["resource_type"], removed["id"])]
                if omitted["resources"].get(rtype, []) != kept:
                    raise ValueError("Omission changed another clinical record's fields")
            verified += 1
    return {"verified": verified, "patients": int(questions.patient_fhir_id.nunique()), "packet_version": PACKET_VERSION,
            "by_family": dict(sorted(Counter(e["family"] for e in entries.values()).items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fhir-dir", default=DEFAULT_FHIR_DIR)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--families", default=DEFAULT_FAMILIES)
    parser.add_argument("--packets", default=DEFAULT_PACKETS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--split-name", choices=("pilot", "dev", "held_out"), default="held_out",
                        help="Use pilot/dev only for offline rule validation; held_out is the final experiment")
    parser.add_argument("--verify-manifest", metavar="PATH",
                        help="Verify every frozen omission from local source and its visible query; do not resample")
    args = parser.parse_args()

    if args.verify_manifest:
        print(json.dumps(verify_manifest(args.fhir_dir, args.verify_manifest, args.input, args.families, args.packets), indent=1))
        return

    with open(args.split, encoding="utf-8") as stream:
        split = json.load(stream)
    held_out = set(split["splits"][args.split_name]["question_ids"])
    packets = {}
    with open(args.packets, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["question_id"] in held_out:
                packets[row["question_id"]] = row
    if set(packets) != held_out:
        raise ValueError("The starting-packet manifest does not cover the held-out split")
    source_hash = file_sha256(args.input)
    if any(row["source_sha256"] != source_hash for row in packets.values()):
        raise ValueError("The starting packets were built from a different dataset")

    df = pd.read_csv(args.input, usecols=["question_id", "question", "true_answer", "patient_fhir_id",
                                          "true_fhir_ids", "sql_query", "val_dict", "main_table_name", "assumption"])
    df = df[df.question_id.isin(held_out)]
    families = pd.read_csv(args.families, usecols=["question_id", "family", "template_id", "template"])
    df = df.merge(families, on="question_id", validate="one_to_one")

    print(f"Indexing {args.fhir_dir} for {df.patient_fhir_id.nunique()} held-out patients...")
    shared, offsets = index_patient_lines(args.fhir_dir, set(df.patient_fhir_id))
    eligible, reasons = [], Counter()
    for patient, questions in df.groupby("patient_fhir_id", sort=True):
        record = load_record(shared, offsets[patient])
        for q in questions.to_dict("records"):
            entry, reason = evaluate(q, packets[q["question_id"]], record, q["family"], q["template"])
            if entry:
                eligible.append(entry)
            else:
                reasons[reason] += 1

    chosen = sample(eligible, args.size, split["seed"])
    manifest = {
        "version": 2,
        "packet_version": PACKET_VERSION,
        "split": args.split_name,
        "seed": split["seed"],
        "size": args.size,
        "source_sha256": source_hash,
        "split_sha256": file_sha256(args.split),
        "families_sha256": file_sha256(args.families),
        "packets_sha256": file_sha256(args.packets),
        "builder_sha256": file_sha256(os.path.abspath(__file__)),
        "oracle_sha256": file_sha256(os.path.join(ROOT, "scripts", "packet_query.py")),
        "omitter_sha256": file_sha256(os.path.join(ROOT, "tools", "perturbation.py")),
        "scorer_sha256": file_sha256(os.path.join(ROOT, "utils", "answer_scorer.py")),
        "fhir_source_sha256": next(iter(packets.values()))["fhir_source_sha256"],
        "eligibility": {
            "held_out_questions": len(held_out),
            "eligible": len(eligible),
            "eligible_by_family": dict(sorted(Counter(e["family"] for e in eligible).items())),
            "eligible_by_rule": dict(sorted(Counter(e["rule"] for e in eligible).items())),
            "excluded_by_reason": dict(sorted(reasons.items())),
        },
        "selected_by_family": dict(sorted(Counter(e["family"] for e in chosen).items())),
        "selected_by_rule": dict(sorted(Counter(e["rule"] for e in chosen).items())),
        "perturbations": chosen,
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=1, sort_keys=False)
        stream.write("\n")
    print(json.dumps({k: manifest[k] for k in ("eligibility", "selected_by_family", "selected_by_rule")}, indent=1))
    print(f"Wrote {len(chosen)} perturbations to {args.output}")


if __name__ == "__main__":
    main()
