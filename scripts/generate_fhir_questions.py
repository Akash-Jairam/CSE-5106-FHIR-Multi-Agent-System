"""Generate new questions by re-instantiating released EHRSQL questions for other patients and slot values.

Each released question is a skeleton: its template wording, SQL and slots
(val_dict). For a target patient, the generator swaps in that patient's ID,
their own current-time anchor and slot values (lab, vital, drug, route,
procedure, specimen, output, care unit, admission route) drawn from their
record, and evaluates the rewritten SQL over the patient's complete FHIR
record with scripts/fhir_oracle.RecordQuery. The answer and cited resources
come from the FHIR fields themselves, so there is no SQL-to-FHIR mapping
loss, and every question is answerable from the store.

Safeguards:
  - a skeleton is used only if the oracle reproduces its own released label
    on its own patient, and at least --min-agreement (0.9) of its template's
    supported labels (the oracle diverges from SQL on, e.g., durations);
  - skeletons with absolute dates ("in 11/2161", "since 2157") are skipped,
    because their dates belong to the source patient's timeline;
  - slot values come from the target patient's record, and for each
    (template, patient) the generator keeps one question with a non-empty
    answer and, where one exists, one whose answer is empty / 0 / False for
    a concept the patient does have elsewhere in their record (a scope test);
  - generated questions that repeat a released question are dropped.

Writes, under --name (default fhir_native):
  final_dataset/generated/<name>.csv          every generated question, in the
                                              released dataset's columns plus
                                              source_question_id, polarity, features
  manifests/generated/<name>.families.csv     family manifest rows
  manifests/generated/<name>.split.json       split manifest over the same
                                              patient pools as split.json,
                                              sampling in-scope families
                                              (--split-families to change)
  manifests/generated/<name>.positive.split.json
                                              the same pools, positive
                                              questions only, spread evenly
                                              over templates

--resplit rebuilds only the split manifest from an existing bank.

Usage:
    python scripts/generate_fhir_questions.py [--patients-per-split held_out] [--size 300]
"""
import argparse
import ast
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.build_question_pools import IN_SCOPE, stratified  # noqa: E402
from scripts.build_split import DEFAULT_FAMILIES, DEFAULT_INPUT, DEFAULT_OUTPUT as DEFAULT_SPLIT  # noqa: E402
from scripts.build_split import file_sha256  # noqa: E402
from scripts.fhir_oracle import RecordQuery, format_answer, full_packet, release  # noqa: E402
from scripts.packet_query import UnsupportedQuery  # noqa: E402
from tools.starting_packet import _display, _resource_kind, parse_current_time  # noqa: E402
from utils.answer_scorer import parse_reference, score_answer  # noqa: E402

DEFAULT_NAME = "fhir_native"
GENERATED_DATA = os.path.join(ROOT, "final_dataset", "generated")
GENERATED_MANIFESTS = os.path.join(ROOT, "manifests", "generated")
SPLIT_SIZES = {"pilot": 20, "dev": 50, "held_out": 300}
# Slots that can be redrawn from the target's record: (table, column[, filter column, value])
SLOT_SOURCES = {
    "lab_name": ("d_labitems", "label"),
    "vital_name": ("d_items", "label", "linksto", "chartevents"),
    "output_name": ("d_items", "label", "linksto", "outputevents"),
    "drug_name": ("prescriptions", "drug"),
    "drug_route": ("prescriptions", "route"),
    "procedure_name": ("d_icd_procedures", "long_title"),
    "spec_name": ("microbiologyevents", "spec_type_desc"),
    "careunit": ("icustays", "first_careunit"),
    "admission_route": ("admissions", "admission_location"),
    "diagnosis_name": ("d_icd_diagnoses", "long_title"),
    "ab_name": ("microbiologyevents", "ab_name"),
}
# Authored templates for multi-hop questions no released template asks (plan 4B):
# culture -> organism -> susceptibility, and ICU stay -> hospital admission -> diagnosis.
# They have no released labels, so they skip the agreement gate.
PLACEHOLDER_PATIENT = 99999999
_LAST_SUSCEPTIBILITY_CULTURE = (
    "microbiologyevents.charttime = ( SELECT microbiologyevents.charttime FROM microbiologyevents WHERE "
    f"microbiologyevents.subject_id = {PLACEHOLDER_PATIENT} AND microbiologyevents.spec_type_desc = 'SPEC0' AND "
    "microbiologyevents.ab_name IS NOT NULL ORDER BY microbiologyevents.charttime DESC LIMIT 1 )")
AUTHORED = [
    {"template_id": "T076", "family": "temporal_selection", "main_table_name": "microbiologyevents",
     "template": "What was the susceptibility of the organism in patient {patient_id}'s last {spec_name} culture "
                 "with susceptibility results to {ab_name}?",
     "question": f"What was the susceptibility of the organism in patient {PLACEHOLDER_PATIENT}'s last spec0 culture "
                 "with susceptibility results to ab0?",
     "sql_query": "SELECT DISTINCT microbiologyevents.interpretation FROM microbiologyevents WHERE "
                  f"microbiologyevents.subject_id = {PLACEHOLDER_PATIENT} AND microbiologyevents.spec_type_desc = "
                  f"'SPEC0' AND microbiologyevents.ab_name = 'AB0' AND {_LAST_SUSCEPTIBILITY_CULTURE}",
     "slots": {"spec_name": "spec0", "ab_name": "ab0"}},
    {"template_id": "T077", "family": "temporal_selection", "main_table_name": "microbiologyevents",
     "template": "Which antibiotics was the organism in patient {patient_id}'s last {spec_name} culture with "
                 "susceptibility results resistant to?",
     "question": f"Which antibiotics was the organism in patient {PLACEHOLDER_PATIENT}'s last spec0 culture with "
                 "susceptibility results resistant to?",
     "sql_query": "SELECT DISTINCT microbiologyevents.ab_name FROM microbiologyevents WHERE "
                  f"microbiologyevents.subject_id = {PLACEHOLDER_PATIENT} AND microbiologyevents.spec_type_desc = "
                  f"'SPEC0' AND microbiologyevents.interpretation = 'Resistant' AND {_LAST_SUSCEPTIBILITY_CULTURE}",
     "slots": {"spec_name": "spec0"}},
    {"template_id": "T078", "family": "lookup_counting", "main_table_name": "diagnoses_icd",
     "template": "Has_verb patient {patient_id} been diagnosed with {diagnosis_name} during the hospital stay of "
                 "their last icu visit?",
     "question": f"Has patient {PLACEHOLDER_PATIENT} been diagnosed with diag0 during the hospital stay of their "
                 "last ICU visit?",
     "sql_query": "SELECT COUNT(*)>0 FROM diagnoses_icd WHERE diagnoses_icd.icd_code IN ( SELECT "
                  "d_icd_diagnoses.icd_code FROM d_icd_diagnoses WHERE d_icd_diagnoses.long_title = 'DIAG0' ) AND "
                  "diagnoses_icd.hadm_id IN ( SELECT icustays.hadm_id FROM icustays WHERE icustays.subject_id = "
                  f"{PLACEHOLDER_PATIENT} AND icustays.outtime IS NOT NULL ORDER BY icustays.intime DESC LIMIT 1 )",
     "slots": {"diagnosis_name": "diag0"}},
]
FIXED_SLOTS = ("patient_id", "vital_value")
RELATIVE_TIME = re.compile(r"^(null|exact-|rel-|mix-)")
MAX_TRIALS = 12  # slot assignments evaluated per (skeleton, patient)
MAX_SKELETONS = 3  # skeletons tried per (template, patient)
# Templates whose released labels the oracle reproduces less often than this are not
# used: their FHIR projection may not mean what the EHRSQL template means
MIN_AGREEMENT = 0.9


def slot_kind(name):
    return name.rstrip("123")


def sql_literal(sql, natural):
    """The quoted SQL literal a natural-language slot value stands for (same text, any case)."""
    for literal in re.findall(r"'((?:[^']|'')*)'", sql):
        if literal.replace("''", "'").casefold() == str(natural).casefold():
            return literal.replace("''", "'")
    return None


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def relative_only(values):
    return all(RELATIVE_TIME.match(t.get("type", "")) for t in values.get("time_placeholder", {}).values())


def authored_skeletons():
    """The authored templates as skeleton rows in the released dataset's columns."""
    return pd.DataFrame([{
        "question_id": f"authored-{t['template_id']}", "question": t["question"], "sql_query": t["sql_query"],
        "true_answer": "[]", "assumption": None, "patient_fhir_id": None, "template": t["template"],
        "val_dict": repr({"val_placeholder": {"patient_id": PLACEHOLDER_PATIENT, **t["slots"]}, "op_placeholder": {},
                          "time_placeholder": {}}),
        "proc_query": None, "main_table_name": t["main_table_name"], "mappable_to_fhir": True, "true_fhir_ids": "{}",
        "family": t["family"], "template_id": t["template_id"], "excluded_reason": "", "split": "authored",
    } for t in AUTHORED])


def record_values(query, packet):
    """{slot kind: sorted distinct values} present in the patient's projected record, plus the
    projected `rows` so slots read from one table can be drawn together."""
    rows = query._rows(packet)
    values = {"rows": rows}
    for kind, (table, column, *where) in SLOT_SOURCES.items():
        found = {row.get(column) for row in rows.get(table, [])
                 if row.get(column) and (not where or row.get(where[0]) == where[1])}
        if kind == "careunit":
            found |= {row.get("last_careunit") for row in rows.get("icustays", []) if row.get("last_careunit")}
        if kind == "procedure_name":
            # The projection also files ED and ICU procedure events under procedures_icd;
            # procedure slots name ICD procedures only
            found = {_display(r.get("code")) for r in packet["resources"].get("Procedure", [])
                     if _resource_kind(r) == "procedure"} & found
        values[kind] = sorted(found)
    vitals = defaultdict(list)
    for row in rows.get("chartevents", []):
        if row.get("valuenum") is not None:
            vitals[row["itemid"]].append(row["valuenum"])
    labels = {row["itemid"]: row["label"] for row in rows.get("d_items", [])}
    values["vital_median"] = {labels.get(item): sorted(v)[len(v) // 2] for item, v in vitals.items()}
    return values


def instantiate(skeleton, patient, anchor, slots):
    """A new dataset row for `patient`, or None when a slot cannot be substituted cleanly.

    `slots` maps slot name -> new SQL-case value. The question gets the
    lowercased value, as the released questions do.
    """
    values = ast.literal_eval(skeleton["val_dict"])
    old = values["val_placeholder"]
    question, sql = skeleton["question"], skeleton["sql_query"]
    for name, new in slots.items():
        natural = old[name]
        if name == "vital_value":
            pattern = re.compile(rf"(?<![\d.]){re.escape(str(natural))}(?![\d.])")
            if not pattern.search(question) or not pattern.search(sql):
                return None
            question, sql = pattern.sub(str(new), question), pattern.sub(str(new), sql)
        else:
            literal = sql_literal(sql, natural)
            if literal is None or str(natural) not in question:
                return None
            question = question.replace(str(natural), str(new).lower())
            sql = sql.replace(quote(literal), quote(new))
        old = {**old, name: new.lower() if isinstance(new, str) else new}
    source_id = str(values["val_placeholder"]["patient_id"])
    id_pattern = re.compile(rf"\b{source_id}\b")
    if not id_pattern.search(question):
        return None
    question, sql = id_pattern.sub(str(patient["subject_id"]), question), id_pattern.sub(str(patient["subject_id"]), sql)
    old["patient_id"] = patient["subject_id"]
    assumption = skeleton["assumption"] if isinstance(skeleton["assumption"], str) else ""
    source_time = parse_current_time(assumption)
    time_values = json.dumps(values["time_placeholder"])
    if source_time:
        if not anchor:
            return None
        old_date, new_date = source_time[:10], anchor[:10]
        assumption = assumption.replace(source_time[:16], anchor[:16])
        sql = sql.replace(old_date, new_date)
        time_values = time_values.replace(old_date, new_date)
        if old_date != new_date and (old_date in sql or source_time[:16] in assumption):
            return None
    values = {**values, "val_placeholder": old, "time_placeholder": json.loads(time_values)}
    return {**skeleton, "question": question, "sql_query": sql, "assumption": assumption or skeleton["assumption"],
            "patient_fhir_id": patient["patient_fhir_id"], "val_dict": repr(values)}


def question_id(row):
    key = json.dumps([row["source_question_id"], row["patient_fhir_id"], row["question"], row["assumption"]])
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def is_negative(values):
    """Empty, zero or False answers: the concept is absent from the asked scope."""
    present = [v for v in values if v is not None]
    return not present or (len(present) == 1 and isinstance(present[0], (int, float)) and float(present[0]) == 0)


def features(query, cited, packet):
    """Reference hops the evidence needs, for multi-hop diagnostic suites."""
    tags = set()
    by_id = packet["by_id"]
    for rid in cited.get("Medication", []):
        if by_id.get(f"Medication/{rid}", {}).get("ingredient"):
            tags.add("medication_mix")
    if cited.get("Specimen"):
        tags.add("specimen_link")
    if query.main_table == "microbiologyevents" and any(by_id.get(f"Observation/{rid}", {}).get("derivedFrom")
                                                         for rid in cited.get("Observation", [])):
        tags.add("micro_organism_chain")
    if query.main_table in ("icustays", "chartevents", "outputevents") and "admissions" in query.tables:
        tags.add("icu_to_hospital")
    if query.susceptibility:
        tags.add("susceptibility_chain")
    if query.main_table == "diagnoses_icd" and "icustays" in query.tables:
        tags.add("icu_to_diagnosis")
    return sorted(tags)


def slot_assignments(skeleton, available, rng):
    """Candidate {slot: value} maps for a skeleton, drawn from the target's record values."""
    slots = [name for name in ast.literal_eval(skeleton["val_dict"])["val_placeholder"] if name not in FIXED_SLOTS]
    if any(slot_kind(name) not in SLOT_SOURCES for name in slots):
        return []
    if not slots:
        return [{}]
    choices = []
    tables = {SLOT_SOURCES[slot_kind(name)][0] for name in slots}
    if len(slots) > 1 and len(tables) == 1 and "rows" in available:
        # Slots of one table (specimen and antibiotic) are drawn from the same row, so they co-occur
        columns = {name: SLOT_SOURCES[slot_kind(name)][1] for name in slots}
        rows = [r for r in available["rows"].get(tables.pop(), []) if all(r.get(c) for c in columns.values())]
        combos = sorted({tuple(r[columns[name]] for name in slots) for r in rows})
        rng.shuffle(combos)
        choices = [dict(zip(slots, combo)) for combo in combos[:MAX_TRIALS // 2]]
    for _ in range(MAX_TRIALS * 3):
        picked, used = {}, set()
        for name in slots:
            options = [v for v in available[slot_kind(name)] if v not in used]
            if not options:
                return choices
            picked[name] = rng.choice(options)
            used.add(picked[name])
        if picked not in choices:
            choices.append(picked)
        if len(choices) >= MAX_TRIALS:
            break
    return choices


def generate_for_patient(skeletons, patient, anchor, packet, rng, existing=frozenset()):
    """Generated rows for one patient: per skeleton template, one positive and one negative question at most."""
    rows = []
    for template_id, group in skeletons.groupby("template_id", sort=True):
        candidates = group.sample(n=min(MAX_SKELETONS, len(group)), random_state=rng.randrange(2**32)).to_dict("records")
        kept = {}
        for skeleton in candidates:
            if len(kept) == 2:
                break
            try:
                probe = RecordQuery({**skeleton, "patient_fhir_id": patient["patient_fhir_id"]}, packet)
                available = record_values(probe, packet)
            except UnsupportedQuery:
                continue
            for slots in slot_assignments(skeleton, available, rng):
                if "vital_value" in ast.literal_eval(skeleton["val_dict"])["val_placeholder"]:
                    vital = next((v for k, v in slots.items() if slot_kind(k) == "vital_name"), None)
                    median = available["vital_median"].get(vital or sql_literal(skeleton["sql_query"], ast.literal_eval(
                        skeleton["val_dict"])["val_placeholder"].get("vital_name", "")))
                    if median is None:
                        continue
                    slots = {**slots, "vital_value": round(float(median), 1)}
                row = instantiate(skeleton, patient, anchor, slots)
                if row is None or (row["patient_fhir_id"], row["question"]) in existing:
                    continue
                try:
                    query = RecordQuery(row, packet)
                    values, cited = query.run_traced()
                except UnsupportedQuery:
                    continue
                if query.inferred_start and query.rule == "selection":
                    continue  # ordering by an approximated prescription start is not reliable
                polarity = "negative" if is_negative(values) else "positive"
                if polarity in kept:
                    continue
                kept[polarity] = {**row, "source_question_id": skeleton["question_id"], "polarity": polarity,
                                  "true_answer": format_answer(values), "true_fhir_ids": repr(cited),
                                  "features": ",".join(features(query, cited, packet)),
                                  "semantics_sensitive": semantics_sensitive(row, packet, values),
                                  "split": "generated", "mappable_to_fhir": True}
                if len(kept) == 2:
                    break
        rows.extend(kept.values())
    for row in rows:
        row["question_id"] = question_id(row)
    return rows


def semantics_sensitive(row, packet, values):
    """Whether the static-database reading of stays open at the current time gives another answer."""
    try:
        static, _ = RecordQuery(row, packet, static=True).run_traced()
    except UnsupportedQuery:
        return True
    return format_answer(static) != format_answer(values)


def label_agrees(values, q):
    reference = parse_reference(q["true_answer"])
    single = values[0] if len(values) == 1 else values or None
    return bool((not values and not reference)
                or score_answer(single, q["true_answer"], q["question"], q["template"])["typed_correct"])


def valid_skeletons(df, packets, min_agreement=MIN_AGREEMENT):
    """Rows whose own label the full-record oracle explains, without absolute dates, from templates
    where it explains at least `min_agreement` of the labels it supports.

    A label is explained if the oracle reproduces it, or if it is the static-database
    answer of a semantics-sensitive row (a stay open at the current time; the
    generated answers follow the agents' guidance instead). Selections that order
    by an approximated prescription start (no validityPeriod) are left out. Returns
    (skeletons, {template_id: share of supported labels explained}).
    """
    outcomes = []
    for q in df.to_dict("records"):
        packet = packets[q["patient_fhir_id"]]
        try:
            query = RecordQuery(q, packet)
            answer, _ = query.run_traced()
            static, _ = RecordQuery(q, packet, static=True).run_traced()
        except UnsupportedQuery:
            continue
        if query.inferred_start and query.rule == "selection":
            continue  # selection by an approximated prescription start: neither evidence for nor against
        explained = label_agrees(answer, q) or (format_answer(static) != format_answer(answer)
                                                and label_agrees(static, q))
        outcomes.append((q["question_id"], q["template_id"], explained, relative_only(ast.literal_eval(q["val_dict"]))))
    outcomes = pd.DataFrame(outcomes, columns=["question_id", "template_id", "agrees", "relative"])
    agreement = outcomes.groupby("template_id").agrees.mean()
    trusted = set(agreement[agreement >= min_agreement].index)
    keep = outcomes[outcomes.agrees & outcomes.relative & outcomes.template_id.isin(trusted)].question_id
    return df[df.question_id.isin(set(keep))], agreement.round(4).to_dict()


def anchors(df):
    """Each patient's most common released current time (None if their questions state none)."""
    times = df.assign(now=df.assumption.map(lambda a: parse_current_time(a) if isinstance(a, str) else None))
    return {p: (Counter(g.now.dropna()).most_common(1) or [(None,)])[0][0] for p, g in times.groupby("patient_fhir_id")}


def split_manifest(generated, families, split, split_path, data_path, families_path, seed, sampled=IN_SCOPE):
    """Same patient pools as split.json; each split samples that pool's generated questions of the
    `sampled` families (the proposal's three operations by default, as in split.json)."""
    merged = generated.merge(families[["question_id", "family", "answer_kind"]], on="question_id")
    merged = merged[merged.family.isin(sampled)]
    mix = merged.family.value_counts().sort_index()
    splits = {}
    for name, entry in split["splits"].items():
        patients = {p["patient_fhir_id"] for p in entry["patients"]}
        pool = merged[merged.patient_fhir_id.isin(patients)].sort_values("question_id")
        chosen = stratified(pool, SPLIT_SIZES[name], mix, seed)
        splits[name] = {"patients": entry["patients"], "question_ids": sorted(chosen.question_id),
                        "family_counts": chosen.family.value_counts().sort_index().to_dict(),
                        "answer_kind_counts": chosen.answer_kind.value_counts().sort_index().to_dict()}
    return {
        "seed": seed,
        "source": os.path.relpath(data_path, ROOT),
        "family_manifest": os.path.relpath(families_path, ROOT),
        "source_sha256": file_sha256(data_path),
        "family_manifest_sha256": file_sha256(families_path),
        "base_split": os.path.relpath(split_path, ROOT),
        "base_split_sha256": file_sha256(split_path),
        "generator_sha256": file_sha256(os.path.abspath(__file__)),
        "oracle_sha256": file_sha256(os.path.join(ROOT, "scripts", "fhir_oracle.py")),
        "in_scope_families": sorted(mix.index),
        "splits": splits,
    }


def main():
    from scripts.build_question_families import answer_kind
    from scripts.build_starting_packets import DEFAULT_FHIR_DIR, index_patient_lines, load_record
    from scripts.fhir_oracle import load_dataset

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--families", default=DEFAULT_FAMILIES)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--fhir-dir", default=DEFAULT_FHIR_DIR)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--splits", nargs="+", choices=("pilot", "dev", "held_out"), default=["pilot", "dev", "held_out"],
                        help="Generate for these splits' patients")
    parser.add_argument("--seed", type=int, default=None, help="Default: the split's seed")
    parser.add_argument("--min-agreement", type=float, default=MIN_AGREEMENT)
    parser.add_argument("--split-families", nargs="+", default=list(IN_SCOPE),
                        help="Families the split manifest samples; the bank keeps every family")
    parser.add_argument("--resplit", action="store_true",
                        help="Rebuild only the split manifest from the existing generated bank")
    args = parser.parse_args()
    data_path = os.path.join(GENERATED_DATA, f"{args.name}.csv")
    families_path = os.path.join(GENERATED_MANIFESTS, f"{args.name}.families.csv")
    split_path = os.path.join(GENERATED_MANIFESTS, f"{args.name}.split.json")

    with open(args.split, encoding="utf-8") as stream:
        split = json.load(stream)
    seed = split["seed"] if args.seed is None else args.seed
    if args.resplit:
        with open(split_path, encoding="utf-8") as stream:
            previous = json.load(stream)
        generated, family_rows = pd.read_csv(data_path), pd.read_csv(families_path, keep_default_na=False)
        write_manifest(generated, family_rows, split, args, data_path, families_path, split_path, seed,
                       previous["template_agreement"], previous["min_agreement"])
        return
    df = load_dataset(args.input, args.families)
    families = pd.read_csv(args.families)
    subjects = {p["patient_fhir_id"]: p["subject_id"] for name in args.splits for p in split["splits"][name]["patients"]}
    existing = set(zip(df.patient_fhir_id, df.question))
    times = anchors(df)

    print(f"Indexing {args.fhir_dir}...")
    shared, offsets = index_patient_lines(args.fhir_dir, set(df.patient_fhir_id))
    packets = {p: full_packet(load_record(shared, offsets[p]), p) for p in sorted(set(df.patient_fhir_id))}
    skeletons, agreement = valid_skeletons(df, packets, args.min_agreement)
    for packet in packets.values():
        release(packet)
    print(f"{len(skeletons)} of {len(df)} released questions are usable skeletons "
          f"({skeletons.template_id.nunique()} of {len(agreement)} supported templates)")
    skeletons = pd.concat([skeletons, authored_skeletons()], ignore_index=True)

    rng = random.Random(seed)
    rows = []
    for patient_fhir_id in sorted(subjects):
        patient = {"patient_fhir_id": patient_fhir_id, "subject_id": subjects[patient_fhir_id]}
        rows.extend(generate_for_patient(skeletons, patient, times.get(patient_fhir_id), packets[patient_fhir_id],
                                         rng, existing))
        release(packets[patient_fhir_id])
    generated = pd.DataFrame(rows).drop_duplicates("question_id").sort_values("question_id")
    columns = list(pd.read_csv(args.input, nrows=0).columns) + ["source_question_id", "polarity", "features",
                                                                "semantics_sensitive"]
    generated = generated[columns]

    os.makedirs(GENERATED_DATA, exist_ok=True)
    os.makedirs(GENERATED_MANIFESTS, exist_ok=True)
    generated.to_csv(data_path, index=False)
    source = pd.concat([families.set_index("question_id"),
                        authored_skeletons().set_index("question_id")[["family", "excluded_reason", "template_id"]]])
    family_rows = pd.DataFrame({
        "question_id": generated.question_id,
        "split": generated.split,
        "family": generated.source_question_id.map(source["family"]),
        "excluded_reason": generated.source_question_id.map(source["excluded_reason"]).fillna(""),
        "template_id": generated.source_question_id.map(source["template_id"]),
        "template": generated.template,
        "main_table_name": generated.main_table_name,
        "answer_kind": generated.true_answer.map(answer_kind),
    })
    family_rows.to_csv(families_path, index=False)
    print(f"Wrote {len(generated)} questions to {os.path.relpath(data_path, ROOT)}")
    print(pd.crosstab(family_rows.family, generated.polarity.values, margins=True).to_string())
    print(f"templates: {family_rows.template_id.nunique()}; patients: {generated.patient_fhir_id.nunique()}; "
          f"features: {dict(Counter(f for v in generated.features.fillna('') for f in v.split(',') if f))}")
    write_manifest(generated, family_rows, split, args, data_path, families_path, split_path, seed,
                   agreement, args.min_agreement)


def template_spread(pool, size, seed):
    """Up to `size` rows taking one per template in turn (seeded order within each template)."""
    rng = random.Random(seed)
    queues = {}
    for template_id, group in pool.sort_values("question_id").groupby("template_id", sort=True):
        rows = list(group.index)
        rng.shuffle(rows)
        queues[template_id] = rows
    chosen = []
    while len(chosen) < size and any(queues.values()):
        for template_id in sorted(queues):
            if queues[template_id] and len(chosen) < size:
                chosen.append(queues[template_id].pop())
    return pool.loc[chosen]


def positive_manifest(generated, families, manifest, sampled, seed):
    """A copy of `manifest` whose splits ask positive questions only, spread evenly over templates."""
    merged = generated.merge(families[["question_id", "family", "template_id", "answer_kind"]], on="question_id")
    merged = merged[merged.family.isin(sampled) & (merged.polarity == "positive")].reset_index(drop=True)
    positive = json.loads(json.dumps(manifest))
    for name, entry in positive["splits"].items():
        patients = {p["patient_fhir_id"] for p in entry["patients"]}
        chosen = template_spread(merged[merged.patient_fhir_id.isin(patients)], SPLIT_SIZES[name], seed)
        entry.update({"question_ids": sorted(chosen.question_id),
                      "family_counts": chosen.family.value_counts().sort_index().to_dict(),
                      "answer_kind_counts": chosen.answer_kind.value_counts().sort_index().to_dict(),
                      "template_counts": chosen.template_id.value_counts().sort_index().to_dict()})
    positive["polarity"] = "positive"
    return positive


def write_manifest(generated, family_rows, split, args, data_path, families_path, split_path, seed, agreement,
                   min_agreement):
    manifest = split_manifest(generated, family_rows, split, args.split, data_path, families_path, seed,
                              args.split_families)
    manifest["template_agreement"] = agreement
    manifest["min_agreement"] = min_agreement
    with open(split_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    positive = positive_manifest(generated, family_rows, manifest, args.split_families, seed)
    with open(split_path.replace(".split.json", ".positive.split.json"), "w", encoding="utf-8") as stream:
        json.dump(positive, stream, indent=2)
        stream.write("\n")
    for name, entry in positive["splits"].items():
        print(f"{name:9s} {len(entry['question_ids']):4d} positive questions over "
              f"{len(entry['template_counts'])} templates")
    for name, entry in manifest["splits"].items():
        print(f"{name:9s} {len(entry['question_ids']):4d} questions  families={entry['family_counts']}")


if __name__ == "__main__":
    main()
