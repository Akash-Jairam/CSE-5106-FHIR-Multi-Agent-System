"""Build meaning-preserving paraphrases of the finisher-template questions.

agent/template_router.classify_template routes seven templates (T001, T051,
T052, T054, T055, T064, T073) to deterministic finishers by their wording.
Each released question of those templates gets two rewrites from its
structured slots (val_dict), with the same patient, current time, reference
answer and citations:

  routed    reworded with synonyms the router still recognizes ("earliest",
            "latest"), so the finisher still fires
  unrouted  reworded without the router's cue words ("oldest/newest ...
            start time" for first/last), so no finisher fires

Comparing a system on the original, routed and unrouted questions separates
the finishers' contribution from the analysis convention guidance. Each
rewrite records the template the router assigns it; unrouted rewrites must
get None and routed rewrites their own template, or the row is dropped.

Writes final_dataset/generated/paraphrases.csv and, in manifests/generated,
paraphrases.families.csv and paraphrases.split.json (the same patient pools
as split.json; each split asks the rewrites of its own sampled questions).

Usage:
    python scripts/build_paraphrases.py
"""
import argparse
import ast
import hashlib
import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agent.template_router import classify_template  # noqa: E402
from scripts.build_split import DEFAULT_FAMILIES, DEFAULT_INPUT, DEFAULT_OUTPUT as DEFAULT_SPLIT  # noqa: E402
from scripts.build_split import file_sha256  # noqa: E402
from scripts.generate_fhir_questions import GENERATED_DATA, GENERATED_MANIFESTS  # noqa: E402

NAME = "paraphrases"
FINISHER_TEMPLATES = ("T001", "T051", "T052", "T054", "T055", "T064", "T073")
ORDER = {  # time_filter_exact1 type -> (routed word, unrouted word)
    "exact-first": ("earliest", "oldest"),
    "exact-last": ("latest", "newest"),
}


def scope(values):
    """The template's canonical scope phrase ("on the first hospital visit", "since 17 months ago"), or ""."""
    phrase = values["time_placeholder"].get("time_filter_global1", {}).get("nlq", "") or ""
    return re.sub(r"\b(\d+) (day|week|month|year) ago\b", lambda m: f"{m[1]} {m[2]}{'' if m[1] == '1' else 's'} ago",
                  phrase).strip()


def rewrite(template_id, values, variant):
    """The paraphrase of one question, or None when its slots are not the template's usual ones."""
    slots = values["val_placeholder"]
    patient = f"patient {slots['patient_id']}"
    where = scope(values)
    if variant == "unrouted":
        # Visit scopes ("on the last hospital visit") use the same order words as the selection
        where = re.sub(r"\bfirst\b", "oldest", re.sub(r"\blast\b", "newest", where))
    where = f" {where}" if where else ""
    order = values["time_placeholder"].get("time_filter_exact1", {}).get("type")
    if template_id == "T001":
        if variant == "routed":
            return f"How many different medications were prescribed to {patient}{where}?"
        return f"How many separate medication names appear in the prescriptions of {patient}{where}?"
    if order not in ORDER:
        return None
    word = ORDER[order][variant == "unrouted"]
    route, specimen = slots.get("drug_route"), slots.get("spec_name")
    if template_id == "T051":
        if variant == "routed":
            return f"Which medication(s) were in the {word} prescription for {patient}{where}?"
        return f"Which medication(s) of {patient} have the {word} prescription start time{where}?"
    if template_id == "T052" and route:
        if variant == "routed":
            return f"Which medication(s) were in the {word} {route}-route prescription for {patient}{where}?"
        return f"Which medication(s) given by the {route} route to {patient} have the {word} prescription start time{where}?"
    if template_id == "T054":
        if variant == "routed":
            return f"Which specimen was collected for the {word} culture of {patient}{where}?"
        return f"Which specimen type was collected for the {word} microbiology test of {patient}{where}?"
    if template_id == "T055" and specimen:
        if variant == "routed":
            return f"Which organism(s) grew in the {word} {specimen} culture of {patient}{where}?"
        return f"Which organism(s) grew in the {word} {specimen} microbiology test of {patient}{where}?"
    if template_id == "T064":
        if variant == "routed":
            return f"When was the {word} microbiology test of {patient}{where}?"
        return f"At what date and time was the {word} microbiology test of {patient}{where} taken?"
    if template_id == "T073" and route:
        if variant == "routed":
            return f"When was the {word} prescription of a medication via the {route} route for {patient}{where}?"
        return f"At what date and time did the {word} prescription via the {route} route for {patient}{where} start?"
    return None


def paraphrase_rows(df):
    rows = []
    for q in df.to_dict("records"):
        values = ast.literal_eval(q["val_dict"])
        for variant in ("routed", "unrouted"):
            text = rewrite(q["template_id"], values, variant)
            if not text:
                continue
            routed = classify_template(text)
            if routed != (q["template_id"] if variant == "routed" else None):
                continue
            key = json.dumps([q["question_id"], variant, text])
            rows.append({**q, "question": text, "source_question_id": q["question_id"], "variant": variant,
                         "router_template": routed or "",
                         "question_id": hashlib.sha256(key.encode()).hexdigest()[:24]})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--families", default=DEFAULT_FAMILIES)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    args = parser.parse_args()

    with open(args.split, encoding="utf-8") as stream:
        split = json.load(stream)
    sampled = {q: name for name, entry in split["splits"].items() for q in entry["question_ids"]}
    df = pd.read_csv(args.input)
    columns = list(df.columns)
    families = pd.read_csv(args.families)
    df = df.merge(families[["question_id", "template_id"]], on="question_id", validate="one_to_one")
    finisher = df[df.template_id.isin(FINISHER_TEMPLATES)]
    rows = paraphrase_rows(finisher).sort_values("question_id")
    dropped = len(finisher) * 2 - len(rows)
    rows = rows[columns + ["source_question_id", "variant", "router_template"]]

    os.makedirs(GENERATED_DATA, exist_ok=True)
    os.makedirs(GENERATED_MANIFESTS, exist_ok=True)
    data_path = os.path.join(GENERATED_DATA, f"{NAME}.csv")
    families_path = os.path.join(GENERATED_MANIFESTS, f"{NAME}.families.csv")
    split_path = os.path.join(GENERATED_MANIFESTS, f"{NAME}.split.json")
    rows.to_csv(data_path, index=False)
    source = families.set_index("question_id")
    family_rows = source.loc[rows.source_question_id].reset_index(drop=True)
    family_rows.insert(0, "question_id", rows.question_id.values)
    family_rows.to_csv(families_path, index=False)

    splits = {}
    for name, entry in split["splits"].items():
        chosen = rows[rows.source_question_id.map(sampled) == name]
        kinds = family_rows.set_index("question_id").loc[chosen.question_id]
        splits[name] = {"patients": entry["patients"], "question_ids": sorted(chosen.question_id),
                        "family_counts": kinds.family.value_counts().sort_index().to_dict(),
                        "answer_kind_counts": kinds.answer_kind.value_counts().sort_index().to_dict(),
                        "variant_counts": chosen.variant.value_counts().sort_index().to_dict(),
                        "source_question_ids": sorted(set(chosen.source_question_id))}
    manifest = {
        "seed": split["seed"],
        "source": os.path.relpath(data_path, ROOT),
        "family_manifest": os.path.relpath(families_path, ROOT),
        "source_sha256": file_sha256(data_path),
        "family_manifest_sha256": file_sha256(families_path),
        "base_split": os.path.relpath(args.split, ROOT),
        "base_split_sha256": file_sha256(args.split),
        "builder_sha256": file_sha256(os.path.abspath(__file__)),
        "router_sha256": file_sha256(os.path.join(ROOT, "agent", "template_router.py")),
        "in_scope_families": sorted(set(family_rows.family)),
        "splits": splits,
    }
    with open(split_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")

    print(f"Wrote {len(rows)} paraphrases of {rows.source_question_id.nunique()} questions to "
          f"{os.path.relpath(data_path, ROOT)} ({dropped} rewrites dropped: unusual slots or router mismatch)")
    print(pd.crosstab(family_rows.template_id, rows.variant.values, margins=True).to_string())
    for name, entry in splits.items():
        print(f"{name:9s} {len(entry['question_ids']):4d} rewrites of {len(entry['source_question_ids'])} "
              f"sampled questions  variants={entry['variant_counts']}")


if __name__ == "__main__":
    main()
