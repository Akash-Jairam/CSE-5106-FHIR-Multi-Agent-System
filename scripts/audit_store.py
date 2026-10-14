"""Lossless audit externalization selected explicitly by the runner.

The caller replaces large traces, retrieved resources, and revision messages
with one audit_artifacts descriptor before worker IPC or retained result rows.
Answer, evidence, usage, budgets, draft/final answers, and revision flags remain
inline. Restore before trace-based diagnostics. Raw audits are content-addressed
compressed JSON with an uncompressed SHA-256; no inference or configuration is
involved.
"""
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

ARTIFACT_PATTERN = re.compile(r"^audit/([a-f0-9]{64})\.json\.gz$")
MAX_RESTORED_BYTES = 512 * 1024 * 1024


def _json_chunks(value):
    # Preserve object field order and all message/code strings exactly.
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for chunk in encoder.iterencode(value):
        yield chunk.encode("utf-8")


def _save_artifact(payload, output_dir):
    output_dir = Path(output_dir).resolve()
    directory = output_dir / "audit"
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.resolve().is_relative_to(output_dir):
        raise ValueError("Audit storage must stay inside the selected output directory")
    digest = hashlib.sha256()
    for chunk in _json_chunks(payload):
        digest.update(chunk)
    sha256 = digest.hexdigest()
    destination = directory / f"{sha256}.json.gz"
    if destination.exists():
        _load_artifact({"path": f"audit/{sha256}.json.gz", "sha256": sha256}, output_dir)
    else:
        descriptor, name = tempfile.mkstemp(prefix=".audit-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
                    for chunk in _json_chunks(payload):
                        compressed.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, destination)
        finally:
            if os.path.exists(name):
                os.unlink(name)
    return {"version": 1, "codec": "json+gzip", "path": f"audit/{sha256}.json.gz", "sha256": sha256}


def _load_artifact(reference, output_dir):
    match = ARTIFACT_PATTERN.fullmatch(reference.get("path", ""))
    if not match or match[1] != reference.get("sha256"):
        raise ValueError("Invalid content-addressed audit reference")
    output_dir = Path(output_dir).resolve()
    path = (output_dir / reference["path"]).resolve()
    if not path.is_relative_to(output_dir):
        raise ValueError("Audit reference escapes the selected output directory")
    data = bytearray()
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            if len(data) + len(chunk) > MAX_RESTORED_BYTES:
                raise ValueError("Audit artifact exceeds the per-question restoration limit")
            data.extend(chunk)
            digest.update(chunk)
    if digest.hexdigest() != reference["sha256"]:
        raise ValueError("Audit artifact SHA-256 mismatch")
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("Audit artifact must contain a JSON object")
    return payload


def _marker(reference, field):
    return [{"type": "external_audit_field", "artifact_ref": {**reference, "field": field}}]


def _reference(value):
    if (isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict)
            and value[0].get("type") == "external_audit_field"):
        reference = value[0].get("artifact_ref")
        if not isinstance(reference, dict) or reference.get("version") != 1 or reference.get("codec") != "json+gzip":
            raise ValueError("Unsupported audit reference contract")
        return reference
    return None


def externalize_result_audit(output, output_dir, question_id=None):
    """Archive full audit data before a result is retained in the driver's DataFrame.

    The normal saved fields keep their public types: trace/messages become empty
    lists, resource-ID maps remain inline, and audit_artifacts identifies the exact
    full trace/resources/messages. No answer, citation, usage or budget is changed.
    """
    result = dict(output)
    if output.get("audit_artifacts") is not None:
        reference = output["audit_artifacts"]
        if not isinstance(reference, dict) or reference.get("version") != 2:
            raise ValueError("Unsupported inline audit-artifact contract")
        owner = question_id if question_id is not None else output.get("question_id")
        if owner is not None and reference.get("question_id") != owner:
            raise ValueError("Audit artifact belongs to another question")
        return result
    owner = question_id if question_id is not None else output.get("question_id")
    payload = {"_audit_meta": {"question_id": owner,
                              "audit_artifacts_was_present": "audit_artifacts" in output}}
    fields = []
    if "trace" in output:
        payload["trace"] = output["trace"]
        fields.append("trace")
    if "retrieved_fhir_resources" in output:
        payload["retrieved_fhir_resources"] = output["retrieved_fhir_resources"]
        fields.append("retrieved_fhir_resources")
    revision = output.get("revision")
    if isinstance(revision, dict) and "messages" in revision:
        payload["revision.messages"] = revision["messages"]
        fields.append("revision.messages")
    if not fields or (not any(payload[field] for field in fields)
                      and output.get("latency") is None and not output.get("error")):
        return result
    reference = {**_save_artifact(payload, output_dir), "version": 2,
                 "fields": fields, "question_id": owner}
    if "trace" in payload:
        result["trace"] = []
    if "retrieved_fhir_resources" in payload:
        result.pop("retrieved_fhir_resources")
    if "revision.messages" in payload:
        result["revision"] = {**revision, "messages": []}
    result["audit_artifacts"] = reference
    return result


def restore_result_audit(output, output_dir, question_id=None, restore_history=True):
    """Restore one attempt, and optionally its history, for trace-based analyses."""
    result = deepcopy(output)
    owner = question_id if question_id is not None else result.get("question_id")
    reference = result.get("audit_artifacts")
    if reference is not None:
        if not isinstance(reference, dict) or reference.get("version") != 2 or reference.get("codec") != "json+gzip":
            raise ValueError("Unsupported audit-artifact contract")
        fields = reference.get("fields")
        allowed = {"trace", "retrieved_fhir_resources", "revision.messages"}
        if (not isinstance(fields, list)
                or not all(isinstance(field, str) and field in allowed for field in fields)
                or len(set(fields)) != len(fields)):
            raise ValueError("Invalid audit-artifact fields")
        if owner is not None and reference.get("question_id") != owner:
            raise ValueError("Audit artifact belongs to another question")
        payload = _load_artifact(reference, output_dir)
        meta = payload.get("_audit_meta")
        if not isinstance(meta, dict) or meta.get("question_id") != reference.get("question_id"):
            raise ValueError("Audit artifact question identity differs from its reference")
        for field in fields:
            if field not in payload:
                raise ValueError("Audit artifact omitted a declared field")
            if field == "revision.messages":
                if not isinstance(result.get("revision"), dict):
                    raise ValueError("Audit messages require revision metadata")
                result["revision"]["messages"] = payload[field]
            else:
                result[field] = payload[field]
        if meta.get("audit_artifacts_was_present"):
            result["audit_artifacts"] = None
        else:
            result.pop("audit_artifacts")
    if restore_history and isinstance(result.get("attempt_history"), list):
        result["attempt_history"] = [restore_result_audit(attempt, output_dir, question_id=owner,
                                                         restore_history=False)
                                     for attempt in result["attempt_history"]]
    # Compatibility with the earlier, unused list-form reference prototype.
    trace_ref = _reference(result.get("trace"))
    messages_ref = _reference((result.get("revision") or {}).get("messages"))
    cached = {}
    for reference, target in ((trace_ref, "trace"), (messages_ref, "revision_messages")):
        if reference is None:
            continue
        if reference.get("field") != target:
            raise ValueError("Audit reference points to the wrong result field")
        key = reference["sha256"]
        if key not in cached:
            cached[key] = _load_artifact(reference, output_dir)
        value = cached[key][target]
        if target == "trace":
            result["trace"] = value
        else:
            result["revision"]["messages"] = value
    return result


def verify_result_audit(output, output_dir, question_id=None):
    """Verify current and historical audit references without retaining full audits."""
    owner = question_id if question_id is not None else output.get("question_id")
    if output.get("audit_artifacts") is not None:
        # The restored object is deliberately discarded before the next row.
        restore_result_audit(output, output_dir, question_id=owner, restore_history=False)
    for attempt in output.get("attempt_history") or []:
        if not isinstance(attempt, dict):
            raise ValueError("Invalid historical attempt audit")
        verify_result_audit(attempt, output_dir, question_id=owner)
