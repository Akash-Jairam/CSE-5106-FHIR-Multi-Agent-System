"""Standalone sandbox worker. Never imports repository configuration or tools."""
import contextlib
import datetime
import json
import math
import re
import resource
import statistics
import sys

SUPPORTED_TYPES = {"Patient", "Encounter", "Condition", "MedicationRequest", "Procedure", "Observation",
                   "MedicationAdministration", "Location", "Specimen", "Medication"}
RESOURCE_ID_RE = re.compile(r"^[A-Za-z0-9.-]{1,64}$")
PATH_SEGMENT_RE = re.compile(r"^([A-Za-z_][\w-]*)((?:\[\d+\])*)$")
FINAL_SCHEMA_EXAMPLE = (
    "Set answer, evidence, contributing_ids, operation, limitation in the same final code. "
    "Citation example: {'resource_id':'Observation/o1','field_path':'valueQuantity.value','value':7}; "
    "contributing_ids=['Observation/o1']; operation='filtered and computed from original records'; limitation=None. "
    "Use resource_id and field_path exactly; one concrete field path and its exact source value per citation.")
STDOUT_CHAR_LIMIT = 64 * 1024


class DiagnosticOutput:
    """Keep a finite prefix in memory while counting all printed diagnostic characters."""

    def __init__(self, limit=STDOUT_CHAR_LIMIT):
        self.limit = limit
        self.parts = []
        self.total_chars = 0
        self.kept_chars = 0

    def write(self, text):
        count = len(text)
        if count == 0:
            return 0
        available = self.limit - self.kept_chars
        if available > 0:
            prefix = text[:available]
            self.parts.append(prefix)
            self.kept_chars += len(prefix)
        self.total_chars += count
        return count

    def flush(self):
        pass

    def fields(self):
        if not self.total_chars:
            return {}
        return {"stdout": "".join(self.parts), "stdout_chars": self.total_chars,
                "stdout_truncated": self.total_chars > self.kept_chars}


def final_answer_value(value):
    """Diagnostics are valid inspection results but cannot be authoritative final answers."""
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, list) and all(final_answer_value(item) for item in value)


def valid_resource_id(value):
    if not isinstance(value, str) or value.count("/") != 1:
        return False
    resource_type, identifier = value.split("/")
    return resource_type in SUPPORTED_TYPES and bool(RESOURCE_ID_RE.fullmatch(identifier))


def final_metadata_error(result, held):
    """Validate executable final metadata before success reaches the role's next model turn."""
    missing = [key for key in ("evidence", "contributing_ids", "operation", "limitation") if key not in result]
    if missing:
        return "Final metadata missing variables: " + ", ".join(missing)
    evidence, contributing = result["evidence"], result["contributing_ids"]
    if not isinstance(evidence, list):
        return "Final evidence must be a list of exact source-field citation objects"
    for index, citation in enumerate(evidence):
        required = {"resource_id", "field_path", "value"}
        if (not isinstance(citation, dict) or not required <= set(citation)
                or not set(citation) <= required | {"scope_index"}):
            return f"Final evidence[{index}] requires resource_id, field_path, value and optional scope_index; aliases are not accepted"
        if "scope_index" in citation and (type(citation["scope_index"]) is not int or citation["scope_index"] not in (0, 1)):
            return f"Final evidence[{index}].scope_index must be the integer 0 or 1 for the question's textual operand"
        resource_id, path = citation["resource_id"], citation["field_path"]
        if not valid_resource_id(resource_id) or resource_id not in held:
            return f"Final evidence[{index}].resource_id must be a supported full ResourceType/id present in retrieved_resources"
        if not isinstance(path, str) or not path or not all(PATH_SEGMENT_RE.fullmatch(part) for part in path.split(".")):
            return f"Final evidence[{index}].field_path must be one concrete dot/index path, e.g. code.coding[0].display"
    if not isinstance(contributing, list) or not all(valid_resource_id(item) for item in contributing):
        return "Final contributing_ids must be a list of full supported ResourceType/id strings, not bare IDs"
    if len(set(contributing)) != len(contributing):
        return "Final contributing_ids must be unique; remove duplicates explicitly"
    unknown = [item for item in contributing if item not in held]
    if unknown:
        return "Final contributing_ids must already be retrieved; first missing IDs: " + ", ".join(unknown[:3])
    if any(citation["resource_id"] not in contributing for citation in evidence):
        return "Every final cited resource must also appear in contributing_ids"
    if not isinstance(result["operation"], str) or not result["operation"].strip():
        return "Final operation must explain the completed filter/search/calculation, including for empty results"
    if result["limitation"] is not None and not isinstance(result["limitation"], str):
        return "Final limitation must be a string or null"
    if bool(result["answer"]) and not evidence:
        return "A nonempty final answer requires source-field evidence; zero, false, null, [] can have none after a completed search"
    return None


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 * 1024, 4 * 1024 * 1024))
    payload = json.load(sys.stdin)
    records = payload["resources"]
    held = {f"{resource_type}/{record['id']}" for resource_type, items in records.items()
            for record in items if isinstance(record, dict) and isinstance(record.get("id"), str)}
    namespace = {"json": json, "datetime": datetime, "math": math, "re": re,
                 "statistics": statistics,
                 "retrieved_resources": records, "retrieved_fhir_resources": records}
    diagnostic_output = DiagnosticOutput()
    try:
        # stdout supports inspection; only an assigned, validated result can be final.
        with contextlib.redirect_stdout(diagnostic_output):
            exec(payload["code"], namespace)
        mode = payload.get("mode", "auto")
        if "answer" in namespace:
            result = {"answer": namespace["answer"], "mode": mode}
        elif diagnostic_output.total_chars and mode in ("inspect", "auto"):
            result = {"mode": mode, "diagnostic": True,
                      "diagnostic_note": "Printed inspection output only; no final answer was assigned. "
                                         "Prefer answer={'count':len(rows),'sample':rows[:3]} for structured inspection."}
        else:
            result = {"error": "Code executed successfully (no answer variable set). Assign actual data to answer, "
                               "e.g. answer={'count':len(rows),'sample':rows[:3]}; bare expressions are not returned. "
                               "Printed diagnostics never establish a final answer. Use mode=final with all result variables."}
        if "answer" not in namespace and payload.get("mode") == "final":
            result["error_type"] = "final_contract"
        if "answer" in result:
            for key in ("evidence", "operation", "contributing_ids", "limitation"):
                if key in namespace:
                    result[key] = namespace[key]
            if ("evidence" in result and not isinstance(result["evidence"], list)
                    or "operation" in result and not isinstance(result["operation"], str)
                    or "contributing_ids" in result and (not isinstance(result["contributing_ids"], list)
                        or any(not isinstance(item, str) for item in result["contributing_ids"]))):
                result = {"error": "Computation metadata requires evidence list, operation string, and contributing_ids string list"}
                if payload.get("mode") == "final":
                    result.update(error_type="final_contract", error=result["error"] + ". " + FINAL_SCHEMA_EXAMPLE)
            elif result["mode"] == "final" and not final_answer_value(result["answer"]):
                result = {"error": "Final answer must be a finite scalar, list of scalar/list/null values, or null. "
                                   "Use mode inspect for diagnostic objects, then recompute the final answer.",
                          "error_type": "final_contract"}
            elif result["mode"] == "final":
                problem = final_metadata_error(result, held)
                if problem:
                    result = {"error": problem + ". " + FINAL_SCHEMA_EXAMPLE, "error_type": "final_contract"}
    except SystemExit:
        result = {"error": "Code called exit()/SystemExit. Set the 'answer' variable instead."}
    except Exception as error:
        result = {"error": f"{type(error).__name__}: {error}"}
    result.update(diagnostic_output.fields())
    try:
        serialized = json.dumps(result, allow_nan=False)
    except (TypeError, ValueError):
        error = {"error": "The answer and metadata must contain only finite JSON values"}
        if payload.get("mode") == "final":
            error["error_type"] = "final_contract"
        serialized = json.dumps(error)
    if len(serialized) > 4 * 1024 * 1024:
        serialized = json.dumps({"error": "Python result exceeds the 4 MiB limit"})
    sys.stdout.write(serialized)


if __name__ == "__main__":
    main()
