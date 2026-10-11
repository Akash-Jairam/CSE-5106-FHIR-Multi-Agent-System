"""Build manifests/perturbations.json: missing-evidence perturbations of held-out questions.

For each eligible held-out question, one answer-changing resource is left out
of the starting packet. The packet is rebuilt from the local MIMIC-IV FHIR
demo NDJSON as if that resource were not in the patient's record, so headers
and truncation stay consistent. The FHIR store is not changed.

A question is eligible when it has a non-empty reference answer and all of
its gold event resources (not the shared Patient, Medication, and Location
resources) are in its frozen starting packet. One rule then applies,
depending on the question type:

- existence (yes/no): exactly one gold record, so removing it turns "yes"
  into "no". With several records the evidence is redundant.
- count: the answer equals the number of gold records, so removing one (the
  latest) lowers the count.
- aggregate: the operation (min, max, sum, avg) is the one that reproduces
  the answer from the gold values. The removed record is the unique extreme
  value for min/max, or the latest record that moves the sum or average,
  and the result without it must differ.
- selection (a value, time, or name from one record): exactly one gold
  record, and no remaining record of the same type in the perturbed packet
  carries the answer value, including names resolved through references.
  Otherwise the evidence is redundant.

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
                                            file_sha256, index_patient_lines, load_record)
from tools.perturbation import without_resources  # noqa: E402
from tools.starting_packet import _label, _reference_key, _resource_time, build_starting_packet, parse_current_time  # noqa: E402
from utils.answer_scorer import answer_type, parse_reference  # noqa: E402

DEFAULT_PACKETS = os.path.join(ROOT, "manifests", "starting_packets.jsonl")
DEFAULT_OUTPUT = os.path.join(ROOT, "manifests", "perturbations.json")
SAMPLE_SIZE = 50
SHARED_TYPES = ("Patient", "Medication", "Location")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")
AGGREGATES = {
    "min": min,
    "max": max,
    "sum": sum,
    "avg": lambda values: sum(values) / len(values),
}


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


def amount(resource):
    """The numeric amount a record contributes to an aggregate (measured value or prescribed dose)."""
    value = (resource.get("valueQuantity") or {}).get("value")
    if value is None:
        for dosage in resource.get("dosageInstruction") or []:
            for rate in dosage.get("doseAndRate") or []:
                value = (rate.get("doseQuantity") or {}).get("value")
                if value is not None:
                    break
            if value is not None:
                break
    return _number(value)


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
        return isinstance(leaf, str) and bool(TIMESTAMP_RE.match(leaf)) and \
            leaf[:16].replace("T", " ") == value.strip()[:16].replace("T", " ")
    number = _number(value)
    if number is not None:
        other = _number(leaf)
        return other is not None and _close(number, other)
    return isinstance(leaf, str) and leaf.strip().casefold() == str(value).strip().casefold()


def latest(pairs, by_id):
    """The most recent record (ties by type and id)."""
    return max(pairs, key=lambda pair: (_resource_time(by_id[f"{pair[0]}/{pair[1]}"]), pair))


def choose_target(kind, values, template, family, events, by_id):
    """(rule, target, None) or (None, None, exclusion reason) before the packet is rebuilt."""
    if kind == "set":
        return None, None, "multi-value answer"
    if kind == "boolean":
        if _number(values[0]) == 0 or str(values[0]).strip().lower() == "false":
            return None, None, "negative answer"
        if len(events) != 1:
            return None, None, "redundant: several records support the answer"
        return "existence", events[0], None
    if template.startswith("Count the number of"):
        if _number(values[0]) != len(events):
            return None, None, "count is not one per gold record"
        return "count", latest(events, by_id), None
    if family == "numerical_aggregation":
        return _aggregate_target(values[0], events, by_id)
    if len(events) != 1:
        return None, None, "several gold records for a single value"
    return "selection", events[0], None


def _aggregate_target(answer, events, by_id):
    answer = _number(answer)
    amounts = {pair: amount(by_id[f"{pair[0]}/{pair[1]}"]) for pair in events}
    if answer is None or any(v is None for v in amounts.values()):
        return None, None, "aggregate values not readable"
    values = list(amounts.values())
    ops = [name for name, op in AGGREGATES.items() if _close(op(values), answer)]
    if len(ops) != 1:
        return None, None, "aggregate operation ambiguous" if ops else "aggregate not reproduced from gold values"
    op = ops[0]
    if op in ("min", "max"):
        extremes = [pair for pair, v in amounts.items() if _close(v, answer)]
        if len(extremes) != 1:
            return None, None, "redundant: tied extreme values"
        target = extremes[0]
    else:
        movers = [pair for pair, v in amounts.items()
                  if (op == "sum" and not _close(v, 0)) or (op == "avg" and not _close(v, answer))]
        if not movers:
            return None, None, "no record changes the aggregate"
        target = latest(movers, by_id)
    rest = [v for pair, v in amounts.items() if pair != target]
    if rest and _close(AGGREGATES[op](rest), answer):
        return None, None, "removal does not change the aggregate"
    return f"aggregate:{op}", target, None


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
    events = event_gold(gold)
    if not events:
        return None, "no gold event records"
    packet_ids = {(t, i) for t, ids in packet_row["resource_ids"].items() for i in ids}
    if not set(events) <= packet_ids:
        return None, "gold not all in the starting packet"
    by_id = {_reference_key(r): r for items in record.values() for r in items}
    rule, target, reason = choose_target(kind, values, template, family, events, by_id)
    if rule is None:
        return None, reason

    patient, now = question["patient_fhir_id"], packet_row["current_time"]
    original = build_starting_packet(question["question"], record, patient, packet_row["max_tokens"], now)
    if _sha256(original["text"]) != packet_row["packet_text_sha256"]:
        raise ValueError(f"Rebuilt packet for {question['question_id']} does not match the frozen manifest")
    perturbed = build_starting_packet(question["question"], without_resources(record, [target]), patient,
                                      packet_row["max_tokens"], now)
    if target[1] in perturbed["resource_ids"].get(target[0], []):
        raise ValueError(f"Removed resource is still in the perturbed packet for {question['question_id']}")
    if rule == "selection" and redundant(values, target, perturbed["resource_ids"], by_id):
        return None, "redundant: another packet record carries the answer"
    return {
        "question_id": question["question_id"],
        "family": family,
        "template_id": question["template_id"],
        "patient_fhir_id": patient,
        "current_time": now,
        "rule": rule,
        "removed": {"resource_type": target[0], "id": target[1]},
        "original_packet_sha256": packet_row["packet_text_sha256"],
        "perturbed_packet_sha256": _sha256(perturbed["text"]),
        "perturbed_resource_ids": perturbed["resource_ids"],
        "perturbed_n_tokens": perturbed["n_tokens"],
    }, None


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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fhir-dir", default=DEFAULT_FHIR_DIR)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--families", default=DEFAULT_FAMILIES)
    parser.add_argument("--packets", default=DEFAULT_PACKETS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    args = parser.parse_args()

    with open(args.split, encoding="utf-8") as stream:
        split = json.load(stream)
    held_out = set(split["splits"]["held_out"]["question_ids"])
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
                                          "true_fhir_ids"])
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
        "version": 1,
        "split": "held_out",
        "seed": split["seed"],
        "size": args.size,
        "source_sha256": source_hash,
        "split_sha256": file_sha256(args.split),
        "families_sha256": file_sha256(args.families),
        "packets_sha256": file_sha256(args.packets),
        "builder_sha256": file_sha256(os.path.abspath(__file__)),
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
