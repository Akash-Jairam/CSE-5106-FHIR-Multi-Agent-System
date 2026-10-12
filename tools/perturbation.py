"""Missing-evidence perturbations (P2-18): starting packets with one resource held back.

manifests/perturbations.json lists, for each perturbed question, one
(resourceType, id) to leave out of its starting packet. The FHIR store is
not changed, so a system can recover the evidence by fetching it.
"""
import json
import hashlib
import re
from copy import deepcopy
from pathlib import Path

from .starting_packet import PACKET_MAX_TOKENS, PACKET_VERSION, build_starting_packet, count_tokens

DEFAULT_PATH = "manifests/perturbations.json"
ROOT = Path(__file__).resolve().parent.parent
COUNT_RE = re.compile(r"\((\d+) of (\d+)(; truncated)?\)")
_REMOVED = object()


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def without_resources(record, removed):
    """`record` (resource type -> resources) minus the (resourceType, id) pairs in `removed`."""
    removed = {tuple(pair) for pair in removed}
    return {rtype: [r for r in items if (r.get("resourceType", rtype), r.get("id")) not in removed]
            for rtype, items in record.items()}


def load_perturbations(path=DEFAULT_PATH):
    """Validated {question_id: frozen specification}; never discard packet hashes."""
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("version") != 2 or manifest.get("packet_version") != PACKET_VERSION:
        raise ValueError("Rebuild perturbations for the current omission protocol and packet version")
    if manifest.get("split") != "held_out" or manifest.get("size") != 50:
        raise ValueError("The experiment requires 50 frozen held-out perturbations")
    artifacts = {"source_sha256": "final_dataset/questions_answers_sql_fhir.csv", "split_sha256": "manifests/split.json",
                 "families_sha256": "manifests/question_families.csv", "packets_sha256": "manifests/starting_packets.jsonl",
                 "builder_sha256": "scripts/build_perturbations.py", "oracle_sha256": "scripts/packet_query.py",
                 "omitter_sha256": "tools/perturbation.py", "scorer_sha256": "utils/answer_scorer.py"}
    for key, relative in artifacts.items():
        if manifest.get(key) != _file_sha256(ROOT / relative):
            raise ValueError(f"Perturbation artifact mismatch: {relative}")
    split = json.loads((ROOT / "manifests/split.json").read_text())
    held_out = set(split["splits"]["held_out"]["question_ids"])
    packets = {p["question_id"]: p for p in map(json.loads, (ROOT / "manifests/starting_packets.jsonl").read_text().splitlines())}
    entries = manifest.get("perturbations")
    if not isinstance(entries, list) or len(entries) != manifest.get("size"):
        raise ValueError("Perturbation sample size does not match its entries")
    selected = {}
    for entry in entries:
        qid = entry.get("question_id")
        if qid in selected or qid not in held_out or qid not in packets:
            raise ValueError("Perturbation IDs must be distinct frozen held-out questions")
        original = packets[qid]
        removed = entry.get("removed") or {}
        proof = entry.get("proof") or {}
        hashes = [entry.get("question_sha256"), entry.get("perturbed_packet_sha256"),
                  proof.get("query_sha256"), proof.get("original_result_sha256"), proof.get("perturbed_result_sha256")]
        if (proof.get("oracle") != "visible-packet-sql-v1"
                or proof.get("perturbed_typed_correct") != 0
                or not all(isinstance(h, str) and re.fullmatch(r"[a-f0-9]{64}", h) for h in hashes)
                or proof["original_result_sha256"] == proof["perturbed_result_sha256"]):
            raise ValueError("Perturbation has no distinct before/after visible-query proof")
        if (entry.get("original_packet_sha256") != original["packet_text_sha256"]
                or entry.get("patient_fhir_id") != original["patient_fhir_id"]
                or entry.get("current_time") != original["current_time"]
                or entry.get("max_tokens") != original["max_tokens"]
                or entry.get("packet_version") != original["packet_version"]
                or entry.get("fhir_source_sha256") != original["fhir_source_sha256"]
                or manifest.get("fhir_source_sha256") != original["fhir_source_sha256"]
                or original.get("retriever_sha256") != _file_sha256(ROOT / "tools/starting_packet.py")
                or removed.get("id") not in original["resource_ids"].get(removed.get("resource_type"), [])):
            raise ValueError("Perturbation does not match its original frozen packet")
        expected = {rtype: [rid for rid in ids if (rtype, rid) != (removed["resource_type"], removed["id"])]
                    for rtype, ids in original["resource_ids"].items()}
        expected = {rtype: ids for rtype, ids in expected.items() if ids}
        if entry.get("perturbed_resource_ids") != expected:
            raise ValueError("Perturbation must remove exactly one resource without replacement")
        selected[qid] = deepcopy(entry)
    return selected


def _redact_references(value, omitted):
    if isinstance(value, dict):
        reference = value.get("reference")
        if isinstance(reference, str) and "/".join(reference.rstrip("/").split("/")[-2:]) in omitted:
            return _REMOVED
        return {k: cleaned for k, item in value.items()
                if (cleaned := _redact_references(item, omitted)) is not _REMOVED}
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := _redact_references(item, omitted)) is not _REMOVED]
    return value


def omit_from_packet(packet, removed):
    """Strictly omit selected packet resources, without reranking or filling gaps.

    This low-level helper is for construction/tests; normal experiment callers
    must use a validated frozen specification with get_perturbed_starting_packet.
    """
    omitted = {f"{rtype}/{rid}" for rtype, rid in removed}
    original = {(rtype, rid) for rtype, ids in packet["resource_ids"].items() for rid in ids}
    if not set(map(tuple, removed)) <= original:
        raise ValueError("Cannot omit a resource absent from the original packet")
    result = deepcopy(packet)
    result["resources"] = {rtype: [_redact_references(r, omitted) for r in items if f"{rtype}/{r['id']}" not in omitted]
                           for rtype, items in packet["resources"].items()}
    result["resources"] = {rtype: items for rtype, items in result["resources"].items() if items}
    result["resource_ids"] = {rtype: [r["id"] for r in items] for rtype, items in result["resources"].items()}
    lines, section, header_index = [], None, None
    for line in packet["text"].splitlines():
        if line.startswith("## "):
            section = line[3:].split(" ", 1)[0]
            header_index = len(lines)
            lines.append(line)
            continue
        if not line.startswith("{"):
            lines.append(line)
            continue
        resource = json.loads(line)
        rtype = resource.get("resourceType", section)
        if f"{rtype}/{resource.get('id')}" in omitted:
            if header_index is not None and ("resourceType" not in resource or (rtype == "Encounter" and section == "Hospital")):
                header = lines[header_index]
                match = COUNT_RE.search(header)
                if match:
                    kept, total = int(match[1]) - 1, int(match[2]) - 1
                    suffix = "; truncated" if kept < total else ""
                    lines[header_index] = COUNT_RE.sub(f"({kept} of {total}{suffix})", header)
            continue
        cleaned = _redact_references(resource, omitted)
        lines.append(line if cleaned == resource else json.dumps(cleaned, separators=(",", ":"), ensure_ascii=False, sort_keys=True))
    result["text"] = "\n".join(line for line in lines if not (line.startswith("## ") and "(0 of 0)" in line))
    result["n_tokens"] = count_tokens(result["text"])
    return result


def validate_context(question, patient_fhir_id, specification, max_tokens, current_time):
    """Check the immutable inputs before any record retrieval."""
    if not isinstance(specification, dict):
        raise ValueError("A frozen perturbation specification is required; use omit_from_packet for tests")
    if (specification.get("patient_fhir_id") != patient_fhir_id
            or specification.get("current_time") != current_time
            or specification.get("max_tokens") != max_tokens
            or specification.get("packet_version") != PACKET_VERSION
            or specification.get("question_sha256") != text_sha256(question)):
        raise ValueError("Question context does not match the frozen perturbation")


def verified_omission(original, specification):
    """The same source-snapshot checks used by runtime and offline verification."""
    if text_sha256(original["text"]) != specification.get("original_packet_sha256"):
        raise ValueError("Live starting packet does not match the frozen original")
    removed = specification["removed"]
    result = omit_from_packet(original, [(removed["resource_type"], removed["id"])])
    if (text_sha256(result["text"]) != specification.get("perturbed_packet_sha256")
            or result["resource_ids"] != specification.get("perturbed_resource_ids")
            or result["n_tokens"] != specification.get("perturbed_n_tokens")):
        raise ValueError("Live omitted packet does not match the frozen perturbation")
    return result


def get_perturbed_starting_packet(question, patient_fhir_id, specification, max_tokens=PACKET_MAX_TOKENS,
                                  current_time=None):
    """A verified frozen omission of this question's original starting packet."""
    from .resource_tools import get_all_resources_by_patient_fhir_id

    validate_context(question, patient_fhir_id, specification, max_tokens, current_time)
    record = get_all_resources_by_patient_fhir_id(patient_fhir_id=patient_fhir_id)
    if "error" in record:
        raise RuntimeError(f"Could not fetch record for {patient_fhir_id}: {record['error']}")
    original = build_starting_packet(question, record, patient_fhir_id, max_tokens, current_time)
    return verified_omission(original, specification)
