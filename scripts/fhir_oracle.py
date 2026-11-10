"""Offline SQL oracle over a patient's complete FHIR record, with traced citations.

scripts/packet_query.py evaluates a released EHRSQL query over the fields a
starting packet shows. `RecordQuery` runs the same projection over every one
of the patient's resources (no packet budget), and also records which FHIR
resource produced each row of the query's main table, so a query's answer
comes with the resources it was computed from.

packet_query.py is pinned by hash in the frozen perturbation manifest, so it
is extended here rather than edited. Like PacketQuery, this is construction
code: agents never import it.

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
from scripts.packet_query import PRIMARY_TYPES, SCHEMAS, PacketQuery, UnsupportedQuery, _ref  # noqa: E402
from tools.starting_packet import _linked, _patient_resources, _reference_key  # noqa: E402
from utils.answer_scorer import parse_reference, score_answer  # noqa: E402

SOURCE_COLUMN = "_src"
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
    # Observations that others derive from (microbiology tests with organisms)
    parents = {_ref(parent) for r in resources.get("Observation", []) for parent in r.get("derivedFrom") or []}
    return {"resources": dict(resources), "by_id": by_id, "parents": parents}


def _row_key(row):
    return tuple(sorted(row.items(), key=lambda item: item[0]))


def format_answer(values):
    """An EHRSQL-style reference string ("[[56]]", "[['a'], ['b']]", "[]"), without float noise."""
    return repr([[float(f"{value:.12g}") if isinstance(value, float) else value] for value in values])


class RecordQuery(PacketQuery):
    """PacketQuery over a full record, whose main-table rows carry their source resource."""

    def __init__(self, question, packet):
        super().__init__(question, packet["by_id"])
        self.packet = packet

    def _sources(self):
        """[(row, "Type/id")] for the main table, attributed by per-resource row differences."""
        table, rtype = self.main_table, PRIMARY_TYPES[self.main_table]
        by_id = self.packet["by_id"]
        parents = self.packet["parents"] if rtype == "Observation" else set()
        attributed = []
        for resource in self.packet["resources"].get(rtype, []):
            if not self.candidate({**resource, "resourceType": rtype}):
                continue
            if _reference_key(resource) in parents:
                continue  # a microbiology test with organisms adds no row of its own
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
                "spec_type_desc" in columns, "dose_val_rx" in columns)

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
        for table, schema in SCHEMAS.items():
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
        return answer, self._citations(sources)

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

    def _citations(self, sources):
        """Cited resources plus the Medication, Specimen and microbiology test each one links to."""
        by_id = self.packet["by_id"]
        cited = []
        for key in sources:
            if not key or key in cited:
                continue
            cited.append(key)
            resource = by_id.get(key, {})
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
        ids = defaultdict(list)
        for key in cited:
            rtype, rid = key.split("/", 1)
            ids[rtype].append(rid)
        return dict(ids)


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


def agreement(df, fhir_dir):
    """Per-template agreement of the full-record oracle with the released labels."""
    from scripts.build_starting_packets import index_patient_lines, load_record

    shared, offsets = index_patient_lines(fhir_dir, set(df.patient_fhir_id))
    rows = []
    for patient, group in df.groupby("patient_fhir_id", sort=True):
        packet = full_packet(load_record(shared, offsets[patient]), patient)
        for q in group.to_dict("records"):
            row = {"question_id": q["question_id"], "template_id": q["template_id"], "family": q["family"],
                   "supported": False, "agrees": False, "gold_recall": None}
            try:
                values, cited = evaluate_row(q, packet)
            except UnsupportedQuery as exc:
                row["reason"] = str(exc)
                rows.append(row)
                continue
            answer = values[0] if len(values) == 1 else values or None
            reference = parse_reference(q["true_answer"])
            row["supported"] = True
            row["agrees"] = bool(score_answer(answer, q["true_answer"], q["question"], q["template"])["typed_correct"]
                                 or (not values and not reference))
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
    supported_rows = result[result.supported]
    by_template = result.groupby("template_id").agg(n=("question_id", "size"), supported=("supported", "mean")).join(
        supported_rows.groupby("template_id").agg(agrees=("agrees", "mean"), gold_recall=("gold_recall", "mean")))
    print(by_template.round(3).to_string())
    supported = result[result.supported]
    print(f"\nsupported {len(supported)}/{len(result)}; agrees with label on "
          f"{supported.agrees.mean():.1%} of supported (agrees column: share of supported rows); mean gold-ID recall {supported.gold_recall.mean():.1%}")
    print(result[~result.supported].reason.value_counts().to_string())
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump({t: {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in r.items()}
                       for t, r in by_template.iterrows()}, stream, indent=1)
            stream.write("\n")


if __name__ == "__main__":
    main()
