"""Error taxonomy over the completed paired runs (V8 diagnosis, step A2).

Reads the long table written by `scripts/harvest_prior_runs.py`, joins it with
the question-family manifest (template_id, answer_kind, main table) and the
question text, and classifies every development question into a stability set:

  both_wrong_core   wrong in every single-policy run and in most team runs
  single_only_wins  single policy mostly right, team mostly wrong
  team_only_wins    team mostly right, single policy mostly wrong
  flaky             right in some runs and wrong in others, in both systems
  both_right_core   right in nearly every run of both systems

"Single policy" pools the fresh single-agent arms (archived, V2, V4) with the
V6/V7 A-only primaries, which execute the same single-agent implementation.
Direct-LLM flags (archived run only) are carried along for reference.

Each question also gets a coarse automatic label from the modal typed-scorer
reason over the runs that recorded one (V3d, V4, V5 team; V4 single):

  false_abstention  "abstained (reference is non-empty)" - the agent said the
                    record has no answer although the reference has one
  wrong_value       "mismatch" / "mismatch (minute precision)" / "missing time"
  unparsed          output could not be typed (format, multiple values, units)
  value_vs_empty    "gave a value" / "gave a timestamp" against an empty reference
  terminal_error    the attempt failed outright

Outputs (default output/diagnosis/, gitignored) plus a committed markdown
summary. Reference answers are never written to the markdown or printed.

Usage:
    python scripts/error_taxonomy.py [--prior output/diagnosis/prior_runs.csv]
                                     [--output_dir output/diagnosis]
                                     [--markdown docs/diagnosis_v8/taxonomy.md]
"""
import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAMILIES = "manifests/question_families.csv"
DATASET = "final_dataset/questions_answers_sql_fhir.csv"
SINGLE_POLICY = ("single", "a_only")

REASON_LABELS = [
    ("abstained (reference is non-empty)", "false_abstention"),
    ("gave a", "value_vs_empty"),
    ("unit mismatch", "unparsed"),
    ("unparsed", "unparsed"),
    ("mismatch", "wrong_value"),
    ("missing time", "wrong_value"),
    ("abstained", "abstention_on_empty"),
    ("match", "correct"),
    ("zero", "correct"),
    ("said none", "correct"),
    ("negative statement", "correct"),
]


def coarse_label(reason, terminal_error, correct):
    if terminal_error:
        return "terminal_error"
    if correct:
        return "correct"
    if not reason:
        return "unknown"
    for prefix, label in REASON_LABELS:
        if reason.startswith(prefix):
            return label
    return "unknown"


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def to_bool(value):
    return {"True": True, "False": False, "1": True, "0": False}.get(str(value), None)


def fraction(values):
    values = [v for v in values if v is not None]
    return (sum(values) / len(values), len(values)) if values else (None, 0)


def classify(single_acc, team_acc):
    if single_acc is None or team_acc is None:
        return "uncovered"
    if single_acc == 0 and team_acc <= 0.3:
        return "both_wrong_core"
    if single_acc >= 0.9 and team_acc >= 0.9:
        return "both_right_core"
    if single_acc >= 0.8 and team_acc <= 0.4:
        return "single_only_wins"
    if team_acc >= 0.8 and single_acc <= 0.4:
        return "team_only_wins"
    return "flaky"


def build_rows(prior_rows, families, dataset):
    by_question = collections.defaultdict(list)
    for record in prior_rows:
        by_question[record["question_id"]].append(record)
    family_by_id = {r["question_id"]: r for r in families}
    question_by_id = {r["question_id"]: r for r in dataset}

    rows = []
    for question_id, records in sorted(by_question.items()):
        family = family_by_id.get(question_id, {})
        question = question_by_id.get(question_id, {})
        single = [to_bool(r["correct"]) for r in records if r["system"] in SINGLE_POLICY]
        team = [to_bool(r["correct"]) for r in records if r["system"] == "team"]
        direct = [to_bool(r["correct"]) for r in records if r["system"] == "direct"]
        single_acc, single_n = fraction(single)
        team_acc, team_n = fraction(team)
        direct_acc, _ = fraction(direct)

        labels = collections.Counter()
        reasons = collections.Counter()
        for r in records:
            if r["system"] == "direct":
                continue
            correct = to_bool(r["correct"])
            if correct:
                continue
            label = coarse_label(r["typed_reason"], to_bool(r["terminal_error"]), correct)
            labels[label] += 1
            if r["typed_reason"]:
                reasons[r["typed_reason"]] += 1
        known = {k: v for k, v in labels.items() if k != "unknown"}
        modal = max(known, key=known.get) if known else ("unknown" if labels else "correct")

        nulls = [to_bool(r["answer_is_null"]) for r in records if r["system"] != "direct"
                 and not to_bool(r["correct"]) and to_bool(r["answer_is_null"]) is not None]
        null_frac, _ = fraction(nulls)

        rows.append({
            "question_id": question_id,
            "family": family.get("family", ""),
            "template_id": family.get("template_id", ""),
            "answer_kind": family.get("answer_kind", ""),
            "main_table": family.get("main_table_name", ""),
            "single_runs": single_n, "single_acc": None if single_acc is None else round(single_acc, 3),
            "team_runs": team_n, "team_acc": None if team_acc is None else round(team_acc, 3),
            "direct_correct": direct_acc,
            "set": classify(single_acc, team_acc),
            "modal_error_label": modal,
            "error_labels": json.dumps(dict(labels), sort_keys=True),
            "error_reasons": json.dumps(dict(reasons), sort_keys=True),
            "wrong_answers_null_fraction": None if null_frac is None else round(null_frac, 2),
            "question": question.get("question", ""),
            "template": family.get("template", ""),
        })
    return rows


def template_table(rows, member_sets=("both_wrong_core",)):
    by_template = collections.defaultdict(lambda: {"n": 0, "core": 0, "single_wrong": 0.0, "team_wrong": 0.0})
    meta = {}
    for r in rows:
        entry = by_template[r["template_id"]]
        entry["n"] += 1
        entry["core"] += 1 if r["set"] in member_sets else 0
        entry["single_wrong"] += 1 - (r["single_acc"] or 0)
        entry["team_wrong"] += 1 - (r["team_acc"] or 0)
        meta[r["template_id"]] = (r["family"], r["answer_kind"], r["main_table"], r["template"])
    table = []
    for template_id, entry in by_template.items():
        family, answer_kind, main_table, template = meta[template_id]
        table.append({
            "template_id": template_id, "family": family, "answer_kind": answer_kind, "main_table": main_table,
            "questions": entry["n"], "both_wrong_core": entry["core"],
            "single_wrong_rate": round(entry["single_wrong"] / entry["n"], 3),
            "team_wrong_rate": round(entry["team_wrong"] / entry["n"], 3),
            "template": template,
        })
    table.sort(key=lambda t: (-t["both_wrong_core"], -t["team_wrong_rate"], t["template_id"]))
    return table


def summarize(rows, templates):
    sets = collections.Counter(r["set"] for r in rows)
    by_set_family = collections.defaultdict(collections.Counter)
    by_set_label = collections.defaultdict(collections.Counter)
    by_set_kind = collections.defaultdict(collections.Counter)
    for r in rows:
        by_set_family[r["set"]][r["family"]] += 1
        by_set_label[r["set"]][r["modal_error_label"]] += 1
        by_set_kind[r["set"]][r["answer_kind"]] += 1
    core_templates = [t for t in templates if t["both_wrong_core"]]
    return {
        "questions": len(rows),
        "sets": dict(sets),
        "by_set_family": {k: dict(v) for k, v in by_set_family.items()},
        "by_set_error_label": {k: dict(v) for k, v in by_set_label.items()},
        "by_set_answer_kind": {k: dict(v) for k, v in by_set_kind.items()},
        "core_template_count": len(core_templates),
        "core_covered_by_top5_templates": sum(t["both_wrong_core"] for t in core_templates[:5]),
    }


def markdown(summary, rows, templates):
    def table(headers, lines):
        out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        out += ["| " + " | ".join(str(c) for c in line) + " |" for line in lines]
        return "\n".join(out)

    parts = ["# V8 diagnosis: error taxonomy over the completed paired runs", "",
             "Generated by `scripts/error_taxonomy.py` from `scripts/harvest_prior_runs.py` output. "
             "Single policy pools the archived, V2 and V4 fresh single arms with the V6/V7 A-only primaries; "
             "team pools the seven team arms (archived, V2, V3d, V4, V5, V6, V7). Reference answers are not shown.",
             "", "## Stability sets", ""]
    order = ["both_wrong_core", "single_only_wins", "team_only_wins", "flaky", "both_right_core", "uncovered"]
    parts.append(table(["set", "questions", "lookup_counting", "temporal_selection", "numerical_aggregation"],
                       [[s, summary["sets"].get(s, 0)] + [summary["by_set_family"].get(s, {}).get(f, 0)
                        for f in ("lookup_counting", "temporal_selection", "numerical_aggregation")] for s in order]))
    parts += ["", "## Modal error label by set", ""]
    labels = ["false_abstention", "wrong_value", "unparsed", "value_vs_empty", "terminal_error", "abstention_on_empty", "unknown"]
    parts.append(table(["set"] + labels,
                       [[s] + [summary["by_set_error_label"].get(s, {}).get(l, 0) for l in labels]
                        for s in order if s != "both_right_core"]))
    parts += ["", "## Answer kind by set", ""]
    kinds = ["value", "empty", "null_value"]
    parts.append(table(["set"] + kinds, [[s] + [summary["by_set_answer_kind"].get(s, {}).get(k, 0) for k in kinds]
                                         for s in order]))
    parts += ["", "## Templates ranked by both-wrong-core membership", "",
              f"{summary['core_template_count']} templates contain core questions; the top five hold "
              f"{summary['core_covered_by_top5_templates']} of {summary['sets'].get('both_wrong_core', 0)}.", ""]
    parts.append(table(["template_id", "family", "kind", "n", "core", "single wrong", "team wrong", "template"],
                       [[t["template_id"], t["family"], t["answer_kind"], t["questions"], t["both_wrong_core"],
                         t["single_wrong_rate"], t["team_wrong_rate"], t["template"].replace("|", "/")]
                        for t in templates if t["both_wrong_core"] or t["team_wrong_rate"] >= 0.5]))
    parts += ["", "## Both-wrong core questions", "",
              "Question text only; the manual root-cause audit lives in `root_causes.csv`.", ""]
    parts.append(table(["question_id", "template", "kind", "single", "team", "direct", "modal label", "question"],
                       [[r["question_id"], r["template_id"], r["answer_kind"], r["single_acc"], r["team_acc"],
                         r["direct_correct"], r["modal_error_label"], r["question"].replace("|", "/")]
                        for r in rows if r["set"] == "both_wrong_core"]))
    parts += ["", "## Single-only wins (what the team loses)", ""]
    parts.append(table(["question_id", "template", "kind", "single", "team", "modal label", "question"],
                       [[r["question_id"], r["template_id"], r["answer_kind"], r["single_acc"], r["team_acc"],
                         r["modal_error_label"], r["question"].replace("|", "/")]
                        for r in rows if r["set"] == "single_only_wins"]))
    parts += ["", "## Team-only wins", ""]
    parts.append(table(["question_id", "template", "kind", "single", "team", "modal label", "question"],
                       [[r["question_id"], r["template_id"], r["answer_kind"], r["single_acc"], r["team_acc"],
                         r["modal_error_label"], r["question"].replace("|", "/")]
                        for r in rows if r["set"] == "team_only_wins"]))
    return "\n".join(parts) + "\n"


def write(rows, templates, summary, output_dir, markdown_path):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "taxonomy.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(output_dir, "templates.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(templates[0].keys()))
        writer.writeheader()
        writer.writerows(templates)
    with open(os.path.join(output_dir, "taxonomy_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    if markdown_path:
        os.makedirs(os.path.dirname(markdown_path), exist_ok=True)
        with open(markdown_path, "w") as handle:
            handle.write(markdown(summary, rows, templates))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prior", default="output/diagnosis/prior_runs.csv")
    parser.add_argument("--output_dir", default="output/diagnosis")
    parser.add_argument("--markdown", default="docs/diagnosis_v8/taxonomy.md")
    args = parser.parse_args(argv)
    rows = build_rows(read_csv(args.prior), read_csv(FAMILIES), read_csv(DATASET))
    templates = template_table(rows)
    summary = summarize(rows, templates)
    write(rows, templates, summary, args.output_dir, args.markdown)
    print(json.dumps({k: summary[k] for k in ("questions", "sets", "by_set_error_label")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
