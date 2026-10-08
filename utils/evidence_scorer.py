"""Evidence scorer: check cited FHIR fields against the actual resources.

An agent's `evidence` is a list of citations:

    {"resource_id": "Observation/<id>", "field_path": "valueQuantity.value", "value": 98.7}

`resource_id` may also be a bare id, with the type in an optional
`resource_type` key. `field_path` uses dots and optional list indices
("code.coding[0].display"); a list without an index matches any element.

A citation is valid when the resource exists, the path resolves, and the
cited value matches the resource's value: numbers numerically, timestamps by
wall-clock time (an omitted UTC offset is fine), strings case-insensitively,
and objects when every cited key matches.

Per answer this reports the valid-evidence fraction and the precision/recall
of the cited resource ids against the gold `true_fhir_ids`.
"""
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

from .metrics_utils import retrieval_precision, retrieval_recall

SUPPORTED_TYPES = ["Patient", "Encounter", "Condition", "MedicationRequest", "Procedure", "Observation",
                   "MedicationAdministration", "Location", "Specimen", "Medication"]
PATH_SEGMENT_RE = re.compile(r"^([A-Za-z_][\w-]*)((?:\[\d+\])*)$")
ID_RE = re.compile(rb'"id": ?"([^"]+)"')


# ---------------------------------------------------------------------------
# Parsing citations
# ---------------------------------------------------------------------------

def parse_evidence(raw):
    """Agent `evidence` (list, JSON string, or missing) -> list of citations."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return [raw]  # scored as malformed
    return list(raw) if isinstance(raw, (list, tuple)) else [raw]


def parse_citation(citation):
    """(resource_type or None, resource_id, field_path, value), or None if malformed."""
    if not isinstance(citation, dict):
        return None
    ref, path = citation.get("resource_id"), citation.get("field_path")
    if not isinstance(ref, str) or not ref.strip() or not isinstance(path, str) or "value" not in citation:
        return None
    rtype = citation.get("resource_type")
    if "/" in ref:
        rtype, ref = ref.rsplit("/", 1)
    return rtype or None, ref.strip(), path.strip(), citation["value"]


# ---------------------------------------------------------------------------
# Resolving paths and comparing values
# ---------------------------------------------------------------------------

def resolve_path(resource, path):
    """Values at `path` in `resource`; None if the path is malformed or absent."""
    current = [resource]
    for segment in path.split("."):
        match = PATH_SEGMENT_RE.match(segment)
        if not match:
            return None
        key, indices = match.group(1), [int(i) for i in re.findall(r"\d+", match.group(2))]
        nxt = []
        for node in current:
            # A list without an index matches any element
            for item in (node if isinstance(node, list) else [node]):
                if isinstance(item, dict) and key in item:
                    value = item[key]
                    for index in indices:
                        value = value[index] if isinstance(value, list) and index < len(value) else None
                    if value is not None:
                        nxt.append(value)
        if not nxt:
            return None
        current = nxt
    flat = []
    for value in current:
        flat.extend(value if isinstance(value, list) else [value])
    return flat


def _as_datetime(value):
    if not isinstance(value, str) or not re.match(r"\d{4}-\d{2}-\d{2}", value.strip()):
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def _as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def values_match(cited, actual):
    if isinstance(cited, dict):
        return isinstance(actual, dict) and all(k in actual and values_match(v, actual[k]) for k, v in cited.items())
    if isinstance(cited, bool) or isinstance(actual, bool):
        return cited == actual or str(cited).strip().lower() == str(actual).strip().lower()
    cited_dt, actual_dt = _as_datetime(cited), _as_datetime(actual)
    if cited_dt and actual_dt:
        if (cited_dt.tzinfo is None) != (actual_dt.tzinfo is None):
            cited_dt, actual_dt = cited_dt.replace(tzinfo=None), actual_dt.replace(tzinfo=None)
        return cited_dt == actual_dt
    cited_num, actual_num = _as_number(cited), _as_number(actual)
    if cited_num is not None and actual_num is not None:
        return math.isclose(cited_num, actual_num, rel_tol=1e-9, abs_tol=1e-9)
    return str(cited).strip().casefold() == str(actual).strip().casefold()


def check_citation(citation, resources):
    """'valid' or the reason the citation is invalid."""
    parsed = parse_citation(citation)
    if parsed is None:
        return "malformed"
    rtype, rid, path, value = parsed
    resource = resources.get(rid)
    if resource is None or (rtype and resource.get("resourceType") != rtype):
        return "missing_resource"
    actual = resolve_path(resource, path)
    if actual is None:
        return "bad_path"
    if isinstance(value, list):
        ok = len(value) == len(actual) and all(values_match(v, a) for v, a in zip(value, actual))
    else:
        ok = any(values_match(value, a) for a in actual)
    return "valid" if ok else "value_mismatch"


# ---------------------------------------------------------------------------
# Fetching cited resources
# ---------------------------------------------------------------------------

def cited_refs(evidence_lists):
    """{(resource_type or None, id)} across all answers."""
    refs = set()
    for evidence in evidence_lists:
        for citation in evidence:
            parsed = parse_citation(citation)
            if parsed:
                refs.add(parsed[:2])
    return refs


def fetch_from_store(refs, batch_size=100):
    """{id: resource} from the FHIR store; ids without a type are looked up in every supported type."""
    from fhir_client import get_fhir_client
    client = get_fhir_client()
    by_type = defaultdict(set)
    for rtype, rid in refs:
        for t in ([rtype] if rtype else SUPPORTED_TYPES):
            by_type[t].add(rid)
    resources = {}
    for rtype, ids in by_type.items():
        ids = sorted(ids)
        for start in range(0, len(ids), batch_size):
            for resource in client.get_resources_by_resource_ids(rtype, ids[start:start + batch_size]):
                resources[resource["id"]] = resource
    return resources


def fetch_from_ndjson(refs, fhir_dir):
    """{id: resource} from the MIMIC-IV FHIR NDJSON files (same data as the store)."""
    wanted = {rid.encode() for _, rid in refs}
    resources = {}
    for path in sorted(glob.glob(os.path.join(fhir_dir, "*.ndjson"))):
        with open(path, "rb") as f:
            for line in f:
                match = ID_RE.search(line)
                if match and match.group(1) in wanted:
                    resource = json.loads(line)
                    resources[resource["id"]] = resource
    return resources


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_evidence(evidence, resources, gold_ids):
    """Evidence metrics for one answer. `gold_ids` is the flat list of gold resource ids."""
    statuses = Counter(check_citation(c, resources) for c in evidence)
    n_cited = len(evidence)
    cited_ids = sorted({p[1] for p in map(parse_citation, evidence) if p})
    return {
        "evidence_cited": n_cited,
        "evidence_valid": statuses["valid"],
        "valid_evidence_fraction": statuses["valid"] / n_cited if n_cited else float("nan"),
        "evidence_issues": {k: v for k, v in statuses.items() if k != "valid"},
        "evidence_precision": retrieval_precision(cited_ids, gold_ids, method="continuous"),
        "evidence_recall": retrieval_recall(cited_ids, gold_ids, method="continuous"),
    }
