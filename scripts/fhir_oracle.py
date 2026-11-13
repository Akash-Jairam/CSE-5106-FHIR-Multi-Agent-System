"""Offline SQL oracle over a patient's complete FHIR record, with traced citations.

scripts/packet_query.py evaluates a released EHRSQL query over the fields a
starting packet shows. `RecordQuery` runs the same projection over every one
of the patient's resources (no packet budget), and also records which FHIR
resource produced each row of the query's main table, so a query's answer
comes with the resources it was computed from.

The record projection adds relations the packet oracle does not represent:
diagnoses_icd / d_icd_diagnoses (from Condition), transfers (care-unit
periods of hospital Encounters, named by Location), and, when a query asks
for them, microbiology test names and antibiotic susceptibility columns
(test -> organism -> susceptibility Observations; the interpretation is the
FHIR display, e.g. "Resistant", which no released question asks about).

Stay semantics follow the agents' guidance (agent/ehr_semantics.py): a stay
still open at the assumed current time is the current stay and has no end,
so it is not the "last" (completed) visit. The released SQL labels were
computed on the static database, where that stay already has its discharge
time. `static=True` evaluates that reading instead; a question whose answer
differs between the two is semantics-sensitive.

Like PacketQuery, this is construction code: agents never import it.

Usage (agreement with the released labels; prints aggregate counts only):
    python scripts/fhir_oracle.py
"""
import argparse
import ast
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict

import pandas as pd
from sqlglot import exp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.packet_query import (PRIMARY_TYPES, SCHEMAS, PacketQuery, UnsupportedQuery, _clock, _coding,  # noqa: E402
                                  _identifier, _micro_type, _numeric, _ref)
from tools.starting_packet import (_display, _linked, _medication_name, _patient_resources, _reference_key,  # noqa: E402
                                   _wall_clock)
from utils.answer_scorer import parse_reference, score_answer  # noqa: E402

SOURCE_COLUMN = "_src"
MICRO_COLUMNS = ("test_name", "ab_name", "dilution_comparison", "dilution_value", "interpretation")
SUSCEPTIBILITY_COLUMNS = MICRO_COLUMNS[1:]
RECORD_SCHEMAS = {
    **SCHEMAS,
    "microbiologyevents": SCHEMAS["microbiologyevents"] + ", test_name TEXT, ab_name TEXT, "
                          "dilution_comparison TEXT, dilution_value REAL, interpretation TEXT",
    "diagnoses_icd": "subject_id INTEGER, hadm_id INTEGER, icd_code TEXT, charttime TEXT",
    "d_icd_diagnoses": "icd_code TEXT, long_title TEXT",
    "transfers": "subject_id INTEGER, hadm_id INTEGER, transfer_id INTEGER, eventtype TEXT, careunit TEXT, "
                 "intime TEXT, outtime TEXT",
}
RECORD_PRIMARY_TYPES = {**PRIMARY_TYPES, "diagnoses_icd": "Condition", "transfers": "Encounter"}
# How a Medication mixture names its prescription rows
MIX_NAMING = ("components", "components_or_parent", "parent_and_components")
DEFAULT_MIX_NAMING = "components"
# Resources outside the patient's own record that rows may reference
SHARED_TYPES = ("Medication", "Location")


def full_packet(record, patient_fhir_id):
    """A packet-shaped view of the patient's whole record: their Patient and
    own resources, plus the shared Medication and Location resources."""
    patient, own, by_id = _patient_resources(record, patient_fhir_id)
    resources = defaultdict(list)
    for r in patient + own:
        resources[r["resourceType"]].append(r)
    for rtype in SHARED_TYPES:
        resources[rtype].extend(r for r in record.get(rtype, []))
    return {"resources": dict(resources), "by_id": by_id}


def _row_key(row):
    return tuple(sorted(row.items(), key=lambda item: item[0]))


def format_answer(values):
    """An EHRSQL-style reference string ("[[56]]", "[['a'], ['b']]", "[]"), without float noise."""
    return repr([[float(f"{value:.12g}") if isinstance(value, float) else value] for value in values])


class RecordQuery(PacketQuery):
    """PacketQuery over a full record, whose main-table rows carry their source resource."""
    schemas = RECORD_SCHEMAS
    primary_types = RECORD_PRIMARY_TYPES

    def __init__(self, question, packet, static=False, mix_naming=DEFAULT_MIX_NAMING):
        if mix_naming not in MIX_NAMING:
            raise ValueError(f"Unknown mix naming {mix_naming}")
        self.static, self.mix_naming = static, mix_naming
        super().__init__(question, packet["by_id"])
        self.packet = packet
        columns = {column.name for column in self.tree.find_all(exp.Column)}
        self.susceptibility = bool(columns & set(SUSCEPTIBILITY_COLUMNS))
        self.micro_extended = bool(columns & set(MICRO_COLUMNS))

    def _candidate(self, resource):
        if self.main_table == "transfers":
            return resource["resourceType"] == "Encounter" and _identifier(resource, "encounter-hosp") is not None
        if self.susceptibility and resource["resourceType"] == "Observation":
            original = self.source.get(f"Observation/{resource['id']}", resource)
            return "microbiology" in _coding(original).get("system", "")
        return super().candidate(resource)

    def _visit_end(self, end):
        return end if self.static else super()._visit_end(end)

    def _rows(self, packet):
        requests = packet["resources"].get("MedicationRequest")
        if requests:
            packet = {**packet, "resources": {**packet["resources"],
                                              "MedicationRequest": [_as_prescription(r) for r in requests
                                                                    if not _is_order_entry(r)]}}
        return super()._rows(packet)

    def candidate(self, resource):
        if resource["resourceType"] == "MedicationRequest" and _is_order_entry(resource):
            return False
        return self._candidate(resource)

    def _drug_names(self, medication, by_id, seen=None):
        components = super()._drug_names(medication, by_id, seen)
        if not medication.get("ingredient") or self.mix_naming == "components":
            return components
        parent = _medication_name(medication)
        if self.mix_naming == "components_or_parent":
            return components or ([parent] if parent else [])
        return list(dict.fromkeys(([parent] if parent else []) + components))

    def _extra_rows(self, rows, resources, by_id, admission):
        for r in resources:
            if r["resourceType"] == "Condition":
                coding = _coding(r)
                rows["diagnoses_icd"].append({"subject_id": self.patient, "hadm_id": admission(r),
                                              "icd_code": coding.get("code"),
                                              "charttime": _wall_clock(r, by_id) or None})
                rows["d_icd_diagnoses"].append({"icd_code": coding.get("code"), "long_title": coding.get("display")})
            elif r["resourceType"] == "Encounter" and _identifier(r, "encounter-hosp") is not None:
                for index, stay in enumerate(r.get("location") or []):
                    period = stay.get("period") or {}
                    rows["transfers"].append({
                        "subject_id": self.patient, "hadm_id": _identifier(r, "encounter-hosp"), "transfer_id": index,
                        "eventtype": "transfer", "careunit": by_id.get(_ref(stay.get("location")), {}).get("name"),
                        "intime": _clock(period.get("start")), "outtime": self._visit_end(_clock(period.get("end")))})
        if self.micro_extended and "microbiologyevents" in self.tables:
            rows["microbiologyevents"] = self._micro_rows(resources, by_id, admission)

    def _micro_rows(self, resources, by_id, admission):
        """Microbiology rows with test names and, when asked for, one row per susceptibility result."""
        children = defaultdict(list)
        tests = []
        for r in resources:
            if r["resourceType"] != "Observation":
                continue
            original = self.source.get(f"Observation/{r['id']}", r)
            if "microbiology" not in _coding(original).get("system", ""):
                continue
            subtype = _micro_type(original)
            if subtype == "test":
                tests.append(r)
            for parent in r.get("derivedFrom") or []:
                children[_ref(parent)].append((subtype, r))
        rows = []
        for test in tests:
            if not test.get("encounter") and self.micro_hospital_used:
                raise UnsupportedQuery("microbiology hospital membership is absent from the visible FHIR fields")
            specimen = by_id.get(_ref(test.get("specimen")), {})
            base = {"subject_id": self.patient, "hadm_id": admission(test),
                    "charttime": _clock(test.get("effectiveDateTime")),
                    "spec_type_desc": _display(specimen.get("type")).upper() or None,
                    "test_name": _coding(test).get("display"), "org_name": None}
            organisms = [o for kind, o in children[f"Observation/{test['id']}"] if kind == "organism"]
            if not organisms:
                rows.append(base)
            for organism in organisms:
                row = {**base, "org_name": _coding(organism).get("display")}
                results = ([s for kind, s in children[f"Observation/{organism['id']}"] if kind == "antibiotic"]
                           if self.susceptibility else [])
                if not results:
                    rows.append(row)
                for result in results:
                    dilution = next((e.get("valueQuantity") or {} for e in result.get("extension") or []
                                     if e.get("url", "").endswith("dilution-details")), {})
                    rows.append({**row, "ab_name": _coding(result).get("display"),
                                 "dilution_comparison": dilution.get("comparator"),
                                 "dilution_value": _numeric(dilution.get("value")),
                                 "interpretation": _display(result.get("valueCodeableConcept")) or None})
        return rows

    def _leaf_parents(self):
        """Microbiology Observations whose own row is replaced by their children's rows."""
        key = ("leaf_parents", self.susceptibility)
        if key not in self.packet:
            parents = set()
            for r in self.packet["resources"].get("Observation", []):
                original = self.source.get(f"Observation/{r['id']}", r)
                if "microbiology" not in _coding(original).get("system", ""):
                    continue
                subtype = _micro_type(original)
                if subtype == "organism" or (subtype == "antibiotic" and self.susceptibility):
                    parents.update(_ref(parent) for parent in r.get("derivedFrom") or [])
            self.packet[key] = parents
        return self.packet[key]

    def _sources(self):
        """[(row, "Type/id")] for the main table, attributed by per-resource row differences."""
        table, rtype = self.main_table, self.primary_types[self.main_table]
        by_id = self.packet["by_id"]
        parents = self._leaf_parents() if rtype == "Observation" else set()
        attributed = []
        for resource in self.packet["resources"].get(rtype, []):
            if not self.candidate({**resource, "resourceType": rtype}):
                continue
            if _reference_key(resource) in parents:
                continue  # a microbiology test with organisms (or organism with results) adds no row of its own
            context = _linked(resource, by_id)
            without = Counter(map(_row_key, self._rows({"resources": _group(context)})[table]))
            with_rows = self._rows({"resources": _group(context + [resource])})[table]
            for row in with_rows:
                key = _row_key(row)
                if without[key]:
                    without[key] -= 1
                else:
                    attributed.append((row, _reference_key(resource)))
        return attributed

    def _projection_key(self):
        """What the projected rows depend on besides the record, so they can be reused across queries."""
        columns = {column.name for column in self.tree.find_all(exp.Column)}
        return (self.main_table, self.current_time, "microbiologyevents" in self.tables, self.micro_hospital_used,
                "spec_type_desc" in columns, "dose_val_rx" in columns, self.static, self.mix_naming,
                self.micro_extended, self.susceptibility)

    def traced_rows(self):
        """Projected rows of the full record, with the main table's rows carrying their source resource."""
        cache = self.packet.setdefault("cache", {})
        key = self._projection_key()
        if key not in cache:
            try:
                rows = self._rows(self.packet)
                traced = self._sources()
                if Counter(map(_row_key, rows[self.main_table])) != Counter(_row_key(row) for row, _ in traced):
                    raise UnsupportedQuery("main-table rows could not be attributed to single resources")
                rows[self.main_table] = [{**row, SOURCE_COLUMN: source} for row, source in traced]
                cache[key] = rows
            except UnsupportedQuery as exc:
                cache[key] = exc
        if isinstance(cache[key], UnsupportedQuery):
            raise cache[key]
        return cache[key]

    def _connection(self):
        """A read-only in-memory database of the traced rows, shared by queries with the same projection."""
        connections = self.packet.setdefault("connections", {})
        key = self._projection_key()
        if key in connections:
            return connections[key]
        rows = self.traced_rows()
        connection = sqlite3.connect(":memory:")
        try:
            self._load(connection, rows)
        except sqlite3.Error as exc:
            connection.close()
            raise UnsupportedQuery("record rows could not be loaded") from exc
        connections[key] = connection
        return connection

    def _load(self, connection, rows):
        for table, schema in self.schemas.items():
            extra = f", {SOURCE_COLUMN} TEXT" if table == self.main_table else ""
            connection.execute(f"CREATE TABLE {table} ({schema}{extra})")
        for table, table_rows in rows.items():
            seen = set()
            for row in table_rows:
                row_key = tuple(row.items())
                if table.startswith("d_") and row_key in seen:
                    continue
                seen.add(row_key)
                connection.execute(f"INSERT INTO {table} ({','.join(row)}) VALUES ({','.join('?' for _ in row)})",
                                   tuple(row.values()))
        connection.execute("PRAGMA query_only = ON")

    def run_traced(self):
        """(answer values, {resourceType: [ids]}) over the full record."""
        connection = self._connection()
        try:
            answer = [row[0] for row in connection.execute(self.sql).fetchall()]
            sources = [row[0] for sql in self.citation_sql() for row in connection.execute(sql).fetchall()]
        except sqlite3.Error as exc:
            raise UnsupportedQuery("query could not be evaluated from the record's FHIR fields") from exc
        return answer, self._citations(sources, answer)

    def citation_sql(self):
        """SQL returning the source resource of each main-table row the query reads.

        The outer query, or each subquery of its projection that reads the
        main table (as paired differences and comparisons do), is rewritten to
        select the source column. Selections keep their ORDER BY/LIMIT, so they
        cite the selected rows; counts, existence checks and aggregates cite
        every row they reduce.
        """
        tree = self.tree.copy()
        if self._reads_main(tree):
            return [self._cite(tree)]
        selects, queries = [], []
        for node in tree.expressions[0].find_all(exp.Select, bfs=True):
            if self._reads_main(node) and not any(node.find_ancestor(exp.Select) is chosen for chosen in selects):
                selects.append(node)
        for node in selects:
            queries.append(self._cite(node.copy()))
        return queries

    def _reads_main(self, select):
        source = select.args.get("from")
        return isinstance(source.this if source else None, exp.Table) and source.this.name == self.main_table

    def _cite(self, select):
        alias = select.args["from"].this.alias_or_name
        selected = select.expressions[0]
        selected = selected.this if isinstance(selected, exp.Alias) else selected
        reduces = any(isinstance(node, exp.AggFunc) for node in selected.walk())
        select.set("expressions", [exp.column(SOURCE_COLUMN, table=alias)])
        if reduces or select.args.get("group"):
            for key in ("group", "having", "order", "limit", "offset"):
                select.set(key, None)
        select.set("distinct", exp.Distinct())
        return select.sql(dialect="sqlite")

    def _citations(self, sources, answer=()):
        """Cited resources plus the Medication, Specimen and microbiology parents each one links to,
        and, for care-unit transfers, the Locations that name an answered unit."""
        by_id = self.packet["by_id"]
        cited = []
        for key in sources:
            if not key or key in cited:
                continue
            cited.append(key)
            resource = by_id.get(key, {})
            if self.main_table == "transfers":
                for stay in resource.get("location") or []:
                    ref = _ref(stay.get("location"))
                    if by_id.get(ref, {}).get("name") in answer and ref not in cited:
                        cited.append(ref)
            pending = [resource.get("medicationReference"), resource.get("specimen"),
                       *(resource.get("derivedFrom") or [])]
            while pending:
                ref = _ref(pending.pop())
                linked = by_id.get(ref)
                if not linked or ref in cited or linked["resourceType"] not in ("Medication", "Specimen", "Observation"):
                    continue
                cited.append(ref)
                pending.extend(i.get("itemReference") for i in linked.get("ingredient") or [])
                pending.append(linked.get("specimen"))
                pending.extend(linked.get("derivedFrom") or [])
        # Released start times of prescriptions without a validity period are only approximated
        self.inferred_start = any(key.startswith("MedicationRequest/") and _as_prescription(by_id.get(key, {}))
                                  is not by_id.get(key) for key in cited)
        ids = defaultdict(list)
        for key in cited:
            rtype, rid = key.split("/", 1)
            ids[rtype].append(rid)
        return dict(ids)


def _is_order_entry(request):
    """Provider order entries (POE) are MedicationRequests absent from the prescriptions table."""
    return any(i.get("system", "").endswith("medication-request-poe") for i in request.get("identifier") or [])


def _as_prescription(request):
    """A prescription without dispenseRequest.validityPeriod (651 pharmacy entries in the demo)
    starts at authoredOn to the minute, which reproduces its released prescription row."""
    period = (request.get("dispenseRequest") or {}).get("validityPeriod") or {}
    if period.get("start") or not request.get("authoredOn"):
        return request
    authored = request["authoredOn"]  # released starts have minute precision
    start = authored[:16] + ":00" + authored[19:] if len(authored) >= 19 else authored
    return {**request, "dispenseRequest": {**(request.get("dispenseRequest") or {}),
                                           "validityPeriod": {**period, "start": start}}}


def release(packet):
    """Drop a full packet's cached projections and databases."""
    for connection in packet.pop("connections", {}).values():
        connection.close()
    packet.pop("cache", None)


def _group(resources):
    grouped = defaultdict(list)
    for r in resources:
        grouped[r["resourceType"]].append(r)
    return dict(grouped)


def evaluate_row(question, packet):
    """(answer values, citations) for one dataset row over its patient's full record."""
    query = RecordQuery(question, packet)
    return query.run_traced()


def id_pairs(ids):
    return {(rtype, rid) for rtype, values in (ids or {}).items() for rid in values}


def _agrees(values, q):
    reference = parse_reference(q["true_answer"])
    answer = values[0] if len(values) == 1 else values or None
    return bool((not values and not reference)
                or score_answer(answer, q["true_answer"], q["question"], q["template"])["typed_correct"])


def agreement(df, fhir_dir):
    """Per-row agreement of the full-record oracle with the released labels.

    `agrees`: the oracle's answer (agents' stay semantics) reproduces the label.
    `explained`: it does, or the row is semantics-sensitive and the static
    reading reproduces it. `inferred`: a selection ordered by an approximated
    prescription start, left out of the agreement shares.
    """
    from scripts.build_starting_packets import index_patient_lines, load_record

    shared, offsets = index_patient_lines(fhir_dir, set(df.patient_fhir_id))
    rows = []
    for patient, group in df.groupby("patient_fhir_id", sort=True):
        packet = full_packet(load_record(shared, offsets[patient]), patient)
        for q in group.to_dict("records"):
            row = {"question_id": q["question_id"], "template_id": q["template_id"], "family": q["family"],
                   "supported": False, "agrees": False, "explained": False, "sensitive": False, "inferred": False,
                   "gold_recall": None}
            try:
                query = RecordQuery(q, packet)
                values, cited = query.run_traced()
                static, _ = RecordQuery(q, packet, static=True).run_traced()
            except UnsupportedQuery as exc:
                row["reason"] = str(exc)
                rows.append(row)
                continue
            row["supported"] = True
            row["agrees"] = _agrees(values, q)
            row["sensitive"] = format_answer(static) != format_answer(values)
            row["explained"] = row["agrees"] or (row["sensitive"] and _agrees(static, q))
            row["inferred"] = bool(query.inferred_start and query.rule == "selection")
            gold = id_pairs(ast.literal_eval(q["true_fhir_ids"]) if isinstance(q["true_fhir_ids"], str) else {})
            if gold:
                row["gold_recall"] = len(gold & id_pairs(cited)) / len(gold)
            rows.append(row)
        release(packet)
    return pd.DataFrame(rows)


def load_dataset(input_path, families_path):
    df = pd.read_csv(input_path)
    families = pd.read_csv(families_path, usecols=["question_id", "family", "template_id"])
    return df.merge(families, on="question_id", validate="one_to_one")


def main():
    from scripts.build_starting_packets import DEFAULT_FAMILIES, DEFAULT_FHIR_DIR, DEFAULT_INPUT

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fhir-dir", default=DEFAULT_FHIR_DIR)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--families", default=DEFAULT_FAMILIES)
    parser.add_argument("--output", help="Write per-template agreement as JSON")
    args = parser.parse_args()

    result = agreement(load_dataset(args.input, args.families), args.fhir_dir)
    scored = result[result.supported & ~result.inferred]
    by_template = result.groupby("template_id").agg(n=("question_id", "size"), supported=("supported", "mean")).join(
        scored.groupby("template_id").agg(agrees=("agrees", "mean"), explained=("explained", "mean"),
                                          sensitive=("sensitive", "mean"), gold_recall=("gold_recall", "mean")))
    print(by_template.round(3).to_string())
    print(f"\nsupported {int(result.supported.sum())}/{len(result)} ({int(result.inferred.sum())} selections by an "
          f"approximated prescription start left out); of the rest the oracle reproduces {scored.agrees.mean():.1%} "
          f"of labels and explains {scored.explained.mean():.1%} (semantics-sensitive: {scored.sensitive.mean():.1%}); "
          f"mean gold-ID recall {scored.gold_recall.mean():.1%}")
    print(result[~result.supported].reason.value_counts().to_string())
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump({t: {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in r.items()}
                       for t, r in by_template.iterrows()}, stream, indent=1)
            stream.write("\n")


if __name__ == "__main__":
    main()
