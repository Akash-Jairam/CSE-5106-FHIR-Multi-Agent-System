"""Offline-only SQL oracle over the exact fields visible in a starting packet.

Labels/SQL are never imported by an agent. This projection implements the
released MIMIC question relations, not a general FHIR-to-SQL database. A
query requiring an unrepresented field is rejected rather than guessed.
"""
import ast
import math
import sqlite3

import sqlglot
from sqlglot import exp

from tools.starting_packet import _display, _medication_name, _resource_kind, parse_current_time

SCHEMAS = {
    "patients": "subject_id INTEGER, gender TEXT",
    "admissions": "subject_id INTEGER, hadm_id INTEGER, admittime TEXT, dischtime TEXT, admission_type TEXT, "
                  "admission_location TEXT, discharge_location TEXT",
    "icustays": "subject_id INTEGER, hadm_id INTEGER, stay_id INTEGER, intime TEXT, outtime TEXT, "
                "first_careunit TEXT, last_careunit TEXT",
    "labevents": "subject_id INTEGER, hadm_id INTEGER, itemid INTEGER, charttime TEXT, valuenum REAL, value TEXT",
    "chartevents": "subject_id INTEGER, hadm_id INTEGER, stay_id INTEGER, itemid INTEGER, charttime TEXT, "
                  "valuenum REAL, value TEXT",
    "outputevents": "subject_id INTEGER, hadm_id INTEGER, stay_id INTEGER, itemid INTEGER, charttime TEXT, value REAL",
    "prescriptions": "subject_id INTEGER, hadm_id INTEGER, starttime TEXT, stoptime TEXT, drug TEXT, route TEXT, "
                     "dose_val_rx REAL, dose_unit_rx TEXT",
    "procedures_icd": "subject_id INTEGER, hadm_id INTEGER, charttime TEXT, icd_code TEXT",
    "microbiologyevents": "subject_id INTEGER, hadm_id INTEGER, charttime TEXT, spec_type_desc TEXT, org_name TEXT",
    "d_labitems": "itemid INTEGER, label TEXT",
    "d_items": "itemid INTEGER, label TEXT, linksto TEXT",
    "d_icd_procedures": "icd_code TEXT, long_title TEXT",
}
PRIMARY_TYPES = {"patients": "Patient", "admissions": "Encounter", "icustays": "Encounter",
                 "labevents": "Observation", "chartevents": "Observation", "outputevents": "Observation",
                 "prescriptions": "MedicationRequest", "procedures_icd": "Procedure",
                 "microbiologyevents": "Observation"}


class UnsupportedQuery(ValueError):
    pass


def visible_resources(packet):
    return [{**r, "resourceType": rtype} for rtype, items in packet["resources"].items() for r in items]


def _clock(value):
    return value[:19].replace("T", " ") if isinstance(value, str) and value else None


def _ref(value):
    reference = (value or {}).get("reference", "")
    return "/".join(reference.rstrip("/").split("/")[-2:])


def _coding(resource):
    values = (resource.get("code") or {}).get("coding") or []
    return values[0] if values else {}


def _identifier(resource, suffix):
    return next((i.get("value") for i in resource.get("identifier", [])
                 if i.get("system", "").endswith(suffix)), None)


def _numeric(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


class PacketQuery:
    """An immutable query contract evaluated only against rendered packet values."""

    def __init__(self, question, source):
        try:
            statements = sqlglot.parse(question["sql_query"], read="sqlite")
        except (sqlglot.errors.ParseError, TypeError, KeyError) as exc:
            raise UnsupportedQuery("query is not a supported SQL SELECT") from exc
        if len(statements) != 1 or not isinstance(statements[0], exp.Select):
            raise UnsupportedQuery("only one SELECT is supported")
        self.tree = statements[0]
        if len(self.tree.expressions) != 1 or self.tree.args.get("with"):
            raise UnsupportedQuery("multi-column or recursive queries are not supported")
        self.tables = {t.name for t in self.tree.find_all(exp.Table)}
        if not self.tables <= set(SCHEMAS):
            raise UnsupportedQuery("query uses a relation without a visible FHIR projection")
        self.main_table = question.get("main_table_name")
        if self.main_table not in PRIMARY_TYPES:
            raise UnsupportedQuery("question has no supported primary relation")
        values = question.get("val_dict")
        values = ast.literal_eval(values) if isinstance(values, str) else values
        if not isinstance(values, dict) or "patient_id" not in values.get("val_placeholder", {}):
            raise UnsupportedQuery("question has no structured patient identity")
        self.patient = values["val_placeholder"]["patient_id"]
        assumption = question.get("assumption")
        self.current_time = parse_current_time(assumption) if isinstance(assumption, str) else None
        self.source = source
        self.sql = self.tree.sql(dialect="sqlite")
        selected = self.tree.expressions[0]
        counts = list(selected.find_all(exp.Count))
        aggregate = next((node for node in selected.walk() if isinstance(node, (exp.Min, exp.Max, exp.Sum, exp.Avg))), None)
        if counts:
            self.rule = "count" if isinstance(selected, exp.Count) else "existence"
        elif aggregate is not None:
            self.rule = "aggregate:" + aggregate.key
        else:
            self.rule = "selection"

    def candidate(self, resource):
        if resource["resourceType"] != PRIMARY_TYPES[self.main_table]:
            return False
        if self.main_table == "admissions":
            return _identifier(resource, "encounter-hosp") is not None
        if self.main_table == "icustays":
            return _identifier(resource, "encounter-icu") is not None
        if resource["resourceType"] == "Observation":
            original = self.source.get(f"Observation/{resource['id']}", resource)
            return _resource_kind(original) == {"labevents": "lab", "chartevents": "vital",
                                                "outputevents": "output", "microbiologyevents": "micro"}[self.main_table]
        return True

    def _rows(self, packet):
        resources = visible_resources(packet)
        by_id = {f"{r['resourceType']}/{r['id']}": r for r in resources}
        rows = {table: [] for table in SCHEMAS}

        def insert(table, **row):
            rows[table].append(row)

        def encounter(reference):
            return by_id.get(_ref(reference), {})

        def admission(resource):
            visit = encounter(resource.get("encounter"))
            if _identifier(visit, "encounter-icu") is not None:
                visit = encounter(visit.get("partOf"))
            return _identifier(visit, "encounter-hosp")

        def drug_names(medication, seen=None):
            seen = set() if seen is None else seen
            key = medication.get("id")
            if key in seen:
                return []
            seen.add(key)
            ingredients = medication.get("ingredient") or []
            if ingredients:
                return [name for ingredient in ingredients for name in drug_names(
                    by_id.get(_ref(ingredient.get("itemReference")), {}), seen)]
            name = _medication_name(medication)
            return [name] if name else []

        for r in resources:
            rtype, rid = r["resourceType"], r["id"]
            if rtype == "Patient":
                insert("patients", subject_id=self.patient, gender={"male": "M", "female": "F"}.get(r.get("gender"), r.get("gender")))
            elif rtype == "Encounter":
                period = r.get("period") or {}
                end = _clock(period.get("end"))
                if self.current_time and end and end > self.current_time:
                    end = None
                hospital, icu = _identifier(r, "encounter-hosp"), _identifier(r, "encounter-icu")
                if hospital is not None:
                    hospitalization = r.get("hospitalization") or {}
                    insert("admissions", subject_id=self.patient, hadm_id=hospital,
                           admittime=_clock(period.get("start")), dischtime=end,
                           admission_location=_display(hospitalization.get("admitSource")),
                           discharge_location=_display(hospitalization.get("dischargeDisposition")))
                if icu is not None:
                    locations = r.get("location") or []
                    names = [by_id.get(_ref(item.get("location")), {}).get("name") for item in locations]
                    insert("icustays", subject_id=self.patient, hadm_id=_identifier(encounter(r.get("partOf")), "encounter-hosp"),
                           stay_id=icu, intime=_clock(period.get("start")), outtime=end,
                           first_careunit=names[0] if names else None, last_careunit=names[-1] if names else None)
            elif rtype == "Observation":
                original = self.source.get(f"Observation/{rid}", r)
                kind = _resource_kind(original)
                table = {"lab": "labevents", "vital": "chartevents", "output": "outputevents"}.get(kind)
                if table:
                    coding = _coding(r)
                    value = _numeric((r.get("valueQuantity") or {}).get("value"))
                    row = {"subject_id": self.patient, "hadm_id": admission(r), "itemid": coding.get("code"),
                           "charttime": _clock(r.get("effectiveDateTime")), "value": value}
                    if table != "outputevents":
                        row["valuenum"] = value
                    if table != "labevents":
                        row["stay_id"] = _identifier(encounter(r.get("encounter")), "encounter-icu")
                    insert(table, **row)
                    if table == "labevents":
                        insert("d_labitems", itemid=coding.get("code"), label=coding.get("display"))
                    else:
                        insert("d_items", itemid=coding.get("code"), label=coding.get("display"), linksto=table)
                elif kind == "micro":
                    # Organism resources inherit the visible test/specimen; susceptibility
                    # resources are not extra independent test events.
                    system = (original.get("code", {}).get("coding") or [{}])[0].get("system", "")
                    if "susc" in system:
                        continue
                    parent = by_id.get(_ref((r.get("derivedFrom") or [{}])[0]), {})
                    test = parent or r
                    if not parent and any(_ref((other.get("derivedFrom") or [{}])[0]) == f"Observation/{rid}"
                                          for other in resources if other["resourceType"] == "Observation"):
                        continue
                    specimen = by_id.get(_ref(test.get("specimen")), {})
                    insert("microbiologyevents", subject_id=self.patient, hadm_id=admission(test),
                           charttime=_clock(test.get("effectiveDateTime")),
                           spec_type_desc=_display(specimen.get("type")).upper() or None,
                           org_name=_coding(r).get("display") if parent else None)
            elif rtype == "MedicationRequest":
                medication = by_id.get(_ref(r.get("medicationReference")), {})
                names = drug_names(medication) or [_display(r.get("medicationCodeableConcept"))]
                dosages = r.get("dosageInstruction") or [{}]
                dosage = dosages[0]
                dose = (dosage.get("doseAndRate") or [{}])[0].get("doseQuantity") or {}
                if len(names) > 1 and any(c.name == "dose_val_rx" for c in self.tree.find_all(exp.Column)):
                    raise UnsupportedQuery("mixed prescription doses are not attributable to individual drugs")
                period = (r.get("dispenseRequest") or {}).get("validityPeriod") or {}
                for name in names:
                    insert("prescriptions", subject_id=self.patient, hadm_id=admission(r),
                           starttime=_clock(period.get("start")), stoptime=_clock(period.get("end")), drug=name or None,
                           route=_display(dosage.get("route")), dose_val_rx=_numeric(dose.get("value")), dose_unit_rx=dose.get("unit"))
            elif rtype == "Procedure":
                coding = _coding(r)
                insert("procedures_icd", subject_id=self.patient, hadm_id=admission(r),
                       charttime=_clock(r.get("performedDateTime")), icd_code=coding.get("code"))
                insert("d_icd_procedures", icd_code=coding.get("code"), long_title=coding.get("display"))
        return rows

    def run(self, packet):
        connection = sqlite3.connect(":memory:")
        try:
            for table, schema in SCHEMAS.items():
                connection.execute(f"CREATE TABLE {table} ({schema})")
            for table, rows in self._rows(packet).items():
                # Dictionary metadata can be repeated through several linked records.
                seen = set()
                for row in rows:
                    key = tuple(row.items())
                    if table.startswith("d_") and key in seen:
                        continue
                    seen.add(key)
                    columns = ",".join(row)
                    connection.execute(f"INSERT INTO {table} ({columns}) VALUES ({','.join('?' for _ in row)})", tuple(row.values()))
            connection.execute("PRAGMA query_only = ON")
            remaining = [10000]

            def limit():
                remaining[0] -= 1
                return remaining[0] <= 0

            connection.set_progress_handler(limit, 1000)
            return [row[0] for row in connection.execute(self.sql).fetchall()]
        except sqlite3.Error as exc:
            raise UnsupportedQuery("query could not be evaluated from visible packet fields") from exc
        finally:
            connection.close()
