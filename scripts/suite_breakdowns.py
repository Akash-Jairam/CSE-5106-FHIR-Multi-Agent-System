"""Breakdowns of the question-suite runs (docs/question_generation.md) beyond `run_experiments.py analyze`.

    python scripts/suite_breakdowns.py output/experiments/gpt-oss:120b

For one model's experiment directory, prints:
  - paraphrases: accuracy of each ordinary arm on the 67 held-out finisher questions as
    released (held_out_v2), routed and unrouted, with exact McNemar tests between variants;
  - generated: accuracy by polarity, feature and semantics sensitivity, for the generated
    and positive-only suites;
  - multi_perturbations: intact and omitted accuracy of each recovery system by kind;
  - errors: failed first attempts of every suite and arm, by kind. An "empty reply" is
    usually a tool call that was not valid JSON, which Ollama drops without an error.
Scoring matches `analyze`: first attempt, typed scorer, failed runs count as wrong.
Intervals are patient-clustered bootstraps. Prints aggregates only, never answers.
"""
import argparse
import collections
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.experiment_stats import clustered_mean, mcnemar_exact  # noqa: E402
from scripts.paired_tests import ARMS, arm_scores  # noqa: E402
from scripts.run_experiments import RECOVERY_SYSTEMS  # noqa: E402
from scripts.variability import first_attempt, is_failed  # noqa: E402

PARAPHRASES = ("final_dataset/generated/paraphrases.csv", "manifests/generated/paraphrases.split.json",
               "manifests/generated/paraphrases.families.csv")
GENERATED = "final_dataset/generated/fhir_native.csv"
GENERATED_SUITES = {"generated": "manifests/generated/fhir_native.split.json",
                    "generated_positive": "manifests/generated/fhir_native.positive.split.json"}
GENERATED_FAMILIES = "manifests/generated/fhir_native.families.csv"
MULTI_PERTURBATIONS = "manifests/perturbations_multi.json"


def _rate(values, patients):
    s = clustered_mean(list(values), list(patients))
    if not s["n"]:
        return "n/a"
    ci = "" if s["ci_low"] is None else f" [{s['ci_low']:.1%}, {s['ci_high']:.1%}]"
    return f"{s['estimate']:.1%}{ci} (n={s['n']})"


def _mcnemar(a, b):
    t = mcnemar_exact(a, b)
    return f"{sum(a) - sum(b):+d} of {t['n']}, discordant {t['only_a']}/{t['only_b']}, exact p = {t['p_value']:.2g}"


def paraphrases(model_dir):
    frame = pd.read_csv(PARAPHRASES[0], usecols=["question_id", "source_question_id", "variant"])
    variants = {}
    for arm in ARMS:
        scores, references = arm_scores(os.path.join(model_dir, "suite_paraphrases", f"ordinary_{arm}.json"),
                                        inputs=PARAPHRASES)
        asked = frame[frame["question_id"].isin(scores)]
        original, original_refs = arm_scores(os.path.join(model_dir, "held_out_v2", f"ordinary_{arm}.json"))
        by_variant = {"original": {q: original[q] for q in asked["source_question_id"]}}
        for variant, rows in asked.groupby("variant"):
            by_variant[variant] = dict(zip(rows["source_question_id"], (scores[q] for q in rows["question_id"])))
        variants[arm] = by_variant
        sources = sorted(by_variant["original"])
        patients = [original_refs[q]["patient"] for q in sources]
        print(f"\n  {arm}")
        for variant in ("original", "routed", "unrouted"):
            print(f"    {variant:<9} {_rate([by_variant[variant][q] for q in sources], patients)}")
        for a, b in (("original", "routed"), ("original", "unrouted"), ("routed", "unrouted")):
            print(f"    {a} - {b}: {_mcnemar([by_variant[a][q] for q in sources], [by_variant[b][q] for q in sources])}")
    sources = sorted(variants[ARMS[0]]["original"])
    for variant in ("original", "routed", "unrouted"):
        pick = lambda arm: [variants[arm][variant][q] for q in sources]  # noqa: E731
        print(f"  {variant:<9} multi - single: {_mcnemar(pick('multi_agent'), pick('single_agent'))}")


def generated(model_dir):
    frame = pd.read_csv(GENERATED, usecols=["question_id", "polarity", "features", "semantics_sensitive"])
    frame = frame.set_index("question_id")
    for suite, manifest in GENERATED_SUITES.items():
        print(f"\n  {suite}")
        scores = {}
        for arm in ARMS:
            scores[arm], references = arm_scores(os.path.join(model_dir, f"suite_{suite}", f"ordinary_{arm}.json"),
                                                 inputs=(GENERATED, manifest, GENERATED_FAMILIES))
        meta = frame.loc[sorted(scores[ARMS[0]])].copy()
        meta["patient"] = [references[q]["patient"] for q in meta.index]
        meta["features"] = meta["features"].fillna("none")
        for column in ("polarity", "features", "semantics_sensitive"):
            for value, rows in meta.groupby(column):
                print(f"    {column}={value}")
                for arm in ARMS:
                    print(f"      {arm:<13} {_rate((scores[arm][q] for q in rows.index), rows['patient'])}")
                if len(rows) >= 10:
                    pick = lambda arm: [scores[arm][q] for q in rows.index]  # noqa: E731
                    print(f"      multi - single: {_mcnemar(pick('multi_agent'), pick('single_agent'))}")


def multi_perturbations(model_dir):
    with open(MULTI_PERTURBATIONS, encoding="utf-8") as stream:
        kinds = {p["question_id"]: p["kind"] for p in json.load(stream)["perturbations"]}
    directory = os.path.join(model_dir, "suite_multi_perturbations")
    for strategy in RECOVERY_SYSTEMS:
        intact, references = arm_scores(os.path.join(directory, f"recovery_{strategy}_intact.json"))
        omitted, _ = arm_scores(os.path.join(directory, f"recovery_{strategy}_omitted.json"), references)
        print(f"\n  {strategy}")
        for kind in ["all", *sorted(set(kinds.values()))]:
            qids = sorted(q for q in omitted if kind == "all" or kinds[q] == kind)
            patients = [references[q]["patient"] for q in qids]
            kept = [q for q in qids if intact[q]]
            print(f"    {kind:<11} intact {_rate((intact[q] for q in qids), patients)}; "
                  f"omitted {_rate((omitted[q] for q in qids), patients)}; "
                  f"omitted given intact right {sum(omitted[q] for q in kept)}/{len(kept)}")


ERROR_KINDS = (("empty reply", "Empty or None output"), ("invalid output", "Invalid"), ("timeout", "Timeout"),
               ("input too long", "Input tokens exceeded"))


def _error_kind(error):
    return next((kind for kind, marker in ERROR_KINDS if marker in error), "other")


def errors(model_dir):
    for directory in sorted(glob.glob(os.path.join(model_dir, "suite_*"))):
        for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
            if path.endswith(".run.json"):
                continue
            with open(path, encoding="utf-8") as stream:
                rows = [first_attempt(row) for row in json.load(stream)]
            failed = collections.Counter(_error_kind(str(row["error"])) for row in rows if is_failed(row))
            arm = os.path.basename(path).removesuffix(".json")
            print(f"  {os.path.basename(directory):<28} {arm:<30} {sum(failed.values()):>3}/{len(rows):<3} "
                  + ", ".join(f"{kind} {n}" for kind, n in failed.most_common()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_dir", help="output/experiments/<model>")
    parser.add_argument("--only", choices=["paraphrases", "generated", "multi_perturbations", "errors"])
    args = parser.parse_args(argv)
    for name, report in (("paraphrases", paraphrases), ("generated", generated),
                         ("multi_perturbations", multi_perturbations), ("errors", errors)):
        if args.only in (None, name):
            print(f"\n== {name}: {args.model_dir}")
            report(args.model_dir)


if __name__ == "__main__":
    main()
