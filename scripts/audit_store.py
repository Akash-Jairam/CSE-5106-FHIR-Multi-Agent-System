"""Optional lossless audit externalization; not enabled in the active runner.

The caller can replace large trace and revision-message fields with list-form
artifact references before retaining result rows. Answer, evidence, usage,
budgets, draft/final answers, and revision flags remain inline. Restore before
performing trace-based diagnostics. Raw audits are content-addressed compressed
JSON with an uncompressed SHA-256; no inference or configuration is involved.
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
MAX_RESTORED_BYTES = 128 * 1024 * 1024


def _json_chunks(value):
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
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


def externalize_result_audit(output, output_dir):
    """Return a compact copy; preserve the exact full trace and bus messages in gzip."""
    result = dict(output)
    payload = {}
    if output.get("trace") and not _reference(output["trace"]):
        payload["trace"] = output["trace"]
    revision = output.get("revision")
    if isinstance(revision, dict) and revision.get("messages") and not _reference(revision["messages"]):
        payload["revision_messages"] = revision["messages"]
    if not payload:
        return result
    reference = _save_artifact(payload, output_dir)
    if "trace" in payload:
        result["trace"] = _marker(reference, "trace")
    if "revision_messages" in payload:
        result["revision"] = {**revision, "messages": _marker(reference, "revision_messages")}
    return result


def restore_result_audit(output, output_dir):
    """Resolve lossless references before existing trace-based analyses are used."""
    result = deepcopy(output)
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
