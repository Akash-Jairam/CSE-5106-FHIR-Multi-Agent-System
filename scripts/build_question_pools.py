"""Build extra held-out question pools from the validated questions split.json left unsampled.

manifests/split.json samples 300 of the held-out patients' questions; the
other held-out questions already have SQL labels and FHIR citations but are
never asked. Each pool here is a split manifest with the same patients and
the same pilot/dev questions as split.json, and a different held_out
question list, so it runs through the unchanged runner:

    python scripts/run_experiments.py run --arms ordinary \\
        --split_manifest manifests/pools/held_out_v2.json \\
        --output_dir output/experiments/<model>/held_out_v2 \\
        --max_total_tokens 200000 --max_tool_calls 30

Pools (none reuses a question from split.json's held_out):

  held_out_v2        300 fresh in-scope questions, family mix of the dataset
                     (a confirmation sample for the 300-question results)
  template_balanced  up to --per-template questions of every in-scope
                     template, including those absent from held_out
  out_of_scope       every duration / paired-difference / paired-comparison
                     question of the held-out patients
  zero_finisher      in-scope questions the finisher router does not match
                     (agent.template_router.classify_template is None)
  truncation         in-scope questions whose starting packet is truncated
                     and leaves out at least one cited clinical record
  disagreement       numerical aggregations with non-integer answers and
                     lookups/counts naming a multi-word concept, where
                     candidate answers tend to split the release vote

Selection may read gold answers and citations (truncation, disagreement):
this is offline construction code, and agents never see pool criteria.
Only aggregate counts are printed.

Usage:
    python scripts/build_question_pools.py [--pools held_out_v2 truncation]
"""
import argparse
import ast
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agent.template_router import classify_template  # noqa: E402
from scripts.build_split import DEFAULT_FAMILIES, DEFAULT_INPUT, DEFAULT_OUTPUT as DEFAULT_SPLIT  # noqa: E402
from scripts.build_split import allocate, file_sha256  # noqa: E402

DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "manifests", "pools")
IN_SCOPE = ("lookup_counting", "numerical_aggregation", "temporal_selection")
POOL_SIZE = 300
PER_TEMPLATE = 5
# Shared and context resources: a packet that omits them still shows the event
CONTEXT_TYPES = ("Patient", "Medication", "Location", "Encounter")
MULTI_WORD_SLOTS = ("lab_name", "vital_name", "drug_name", "procedure_name", "spec_name", "output_name")
POOLS = ("held_out_v2", "template_balanced", "out_of_scope", "zero_finisher", "truncation", "disagreement")


def load_questions(input_path, families_path):
    df = pd.read_csv(input_path)
    families = pd.read_csv(families_path, usecols=["question_id", "family", "template_id", "excluded_reason",
                                                   "answer_kind"])
    return df.merge(families, on="question_id", validate="one_to_one")


def unsampled_held_out(df, split):
    """Held-out patients' questions that split.json does not ask."""
    patients = {p["patient_fhir_id"] for p in split["splits"]["held_out"]["patients"]}
    asked = set(split["splits"]["held_out"]["question_ids"])
    return df[df.patient_fhir_id.isin(patients) & ~df.question_id.isin(asked)].sort_values("question_id")


def stratified(pool, size, mix, seed):
    """`size` questions of `pool`, allocated to families in proportion to `mix` (all of them if fewer)."""
    if len(pool) <= size:
        return pool
    mix = mix[mix.index.isin(pool.family.unique())]
    available = pool.family.value_counts()
    # A family with too few questions is taken whole; the others share its unused quota in proportion
    capped = {}
    while True:
        quotas = allocate(mix.drop(list(capped)), size - sum(capped.values()))
        short = [family for family, quota in quotas.items() if quota > available[family]]
        if not short:
            break
        capped.update({family: int(available[family]) for family in short})
    quotas = pd.concat([pd.Series(capped, dtype=int), quotas])
    rng = np.random.RandomState(seed)
    return pd.concat([pool[pool.family == family].sample(n=int(quota), random_state=rng)
                      for family, quota in quotas.sort_index().items() if quota])


def per_template(pool, k, seed):
    rng = np.random.RandomState(seed)
    return pd.concat([group.sample(n=min(k, len(group)), random_state=rng)
                      for _, group in pool.groupby("template_id", sort=True)])


def multi_word_concept(val_dict):
    values = ast.literal_eval(val_dict)["val_placeholder"]
    names = [str(values[k]) for k in values if k.rstrip("123") in MULTI_WORD_SLOTS]
    return any(len(name.split()) >= 3 or re.search(r"[(),/%]", name) for name in names)


def fractional(true_answer):
    from utils.answer_scorer import _as_number, parse_reference

    numbers = [_as_number(v) for v in parse_reference(true_answer)]
    return len(numbers) == 1 and numbers[0] is not None and not float(numbers[0]).is_integer()


def truncation_rows(pool, fhir_dir):
    """Per question: whether its starting packet is truncated, and the cited clinical records it leaves out."""
    from scripts.build_starting_packets import index_patient_lines, load_record
    from tools.starting_packet import PACKET_MAX_TOKENS, build_starting_packet, parse_current_time

    shared, offsets = index_patient_lines(fhir_dir, set(pool.patient_fhir_id))
    rows = []
    for patient, group in pool.groupby("patient_fhir_id", sort=True):
        record = load_record(shared, offsets[patient])
        for q in group.itertuples():
            now = parse_current_time(q.assumption if isinstance(q.assumption, str) else "")
            packet = build_starting_packet(q.question, record, patient, PACKET_MAX_TOKENS, now)
            shown = {(t, i) for t, ids in packet["resource_ids"].items() for i in ids}
            gold = ast.literal_eval(q.true_fhir_ids) if isinstance(q.true_fhir_ids, str) else {}
            cited = {(t, i) for t, ids in gold.items() if t not in CONTEXT_TYPES for i in ids}
            rows.append({"question_id": q.question_id, "truncated": packet["truncated"],
                         "cited": len(cited), "missing": len(cited - shown)})
    return pd.DataFrame(rows, columns=["question_id", "truncated", "cited", "missing"])


def select(name, unsampled, mix, args):
    """(questions, criteria) for one pool."""
    in_scope = unsampled[unsampled.family.isin(IN_SCOPE)]
    if name == "held_out_v2":
        return stratified(in_scope, args.size, mix, args.seed), "family-stratified in-scope sample"
    if name == "template_balanced":
        return per_template(in_scope, args.per_template, args.seed), f"up to {args.per_template} per in-scope template"
    if name == "out_of_scope":
        return unsampled[unsampled.family == "out_of_scope"], "every out-of-scope question"
    if name == "zero_finisher":
        routed = in_scope.question.map(classify_template).notna()
        return stratified(in_scope[~routed], args.size, mix, args.seed), "classify_template(question) is None"
    if name == "truncation":
        packets = truncation_rows(in_scope, args.fhir_dir)
        hit = set(packets.question_id[packets.truncated & (packets.missing > 0)])
        return (stratified(in_scope[in_scope.question_id.isin(hit)], args.size, mix, args.seed),
                "truncated starting packet that omits at least one cited non-context record")
    if name == "disagreement":
        numeric = (in_scope.family == "numerical_aggregation") & in_scope.true_answer.map(fractional)
        named = (in_scope.family == "lookup_counting") & in_scope.val_dict.map(multi_word_concept)
        return (stratified(in_scope[numeric | named], args.size, mix, args.seed),
                "non-integer aggregation answers, or lookups/counts naming a multi-word concept")
    raise ValueError(f"Unknown pool {name}")


def pool_manifest(split, split_path, name, chosen, criteria):
    """A copy of split.json whose held_out split asks `chosen` instead."""
    manifest = json.loads(json.dumps(split))
    held_out = manifest["splits"]["held_out"]
    held_out["question_ids"] = sorted(chosen.question_id)
    held_out["family_counts"] = chosen.family.value_counts().sort_index().to_dict()
    held_out["answer_kind_counts"] = chosen.answer_kind.value_counts().sort_index().to_dict()
    manifest["in_scope_families"] = sorted(set(chosen.family))
    manifest["pool"] = {
        "name": name,
        "criteria": criteria,
        "base_split": os.path.relpath(split_path, ROOT),
        "base_split_sha256": file_sha256(split_path),
        "builder_sha256": file_sha256(os.path.abspath(__file__)),
        "questions": len(chosen),
        "patients": int(chosen.patient_fhir_id.nunique()),
        "templates": int(chosen.template_id.nunique()),
        "template_counts": chosen.template_id.value_counts().sort_index().to_dict(),
    }
    return manifest


def main():
    from scripts.build_starting_packets import DEFAULT_FHIR_DIR

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--families", default=DEFAULT_FAMILIES)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--fhir-dir", default=DEFAULT_FHIR_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pools", nargs="+", choices=POOLS, default=list(POOLS))
    parser.add_argument("--size", type=int, default=POOL_SIZE)
    parser.add_argument("--per-template", type=int, default=PER_TEMPLATE)
    parser.add_argument("--seed", type=int, default=None, help="Default: the split's seed")
    args = parser.parse_args()

    with open(args.split, encoding="utf-8") as stream:
        split = json.load(stream)
    if split["source_sha256"] != file_sha256(args.input) or split["family_manifest_sha256"] != file_sha256(args.families):
        sys.exit("The split was built from a different dataset or family manifest")
    args.seed = split["seed"] if args.seed is None else args.seed
    df = load_questions(args.input, args.families)
    mix = df[df.family.isin(IN_SCOPE)].family.value_counts().sort_index()
    unsampled = unsampled_held_out(df, split)
    os.makedirs(args.output_dir, exist_ok=True)
    for name in args.pools:
        chosen, criteria = select(name, unsampled, mix, args)
        if chosen.empty:
            print(f"{name:18s} no questions match; skipped")
            continue
        manifest = pool_manifest(split, args.split, name, chosen, criteria)
        path = os.path.join(args.output_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        held_out = manifest["splits"]["held_out"]
        print(f"{name:18s} {len(chosen):4d} questions, {manifest['pool']['patients']:2d} patients, "
              f"{manifest['pool']['templates']:2d} templates  families={held_out['family_counts']}  "
              f"-> {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
