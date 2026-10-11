"""Verification role (P2-15) for the fixed controller.

Checks a draft from a fresh context. The verifier never sees the retrieval
or analysis conversations or their workspace:

1. Re-fetch cited fields: the deterministic field check (tools/field_check.py)
   re-reads every citation from the store and checks the value, the time
   field, the unit, the patient, the encounter scope, and the current time.
   As for the single agent, this check counts as one tool call.
2. Recompute: the model starts from an empty workspace, re-searches the
   patient's records within the question's scope, and recomputes the answer
   in the sandbox. It reports issues the field check cannot see, such as
   records the draft missed or a wrong calculation.
3. Merge: the field-check issues, the model's issues, and a
   `recompute_mismatch` issue if the recomputed answer differs from the draft.

Every issue is {"type", "record", "field", "responsible_role", "action"}.
responsible_role is "retrieval" or "analysis", or None when no role can
repair the issue.
"""
import json
import math
import re

from .framework import PYTHON_TOOL, RETRIEVAL_TOOLS, Role, Toolbox
from .output_schema import _json_objects
from tools.field_check import check_fields
from tools.resource_tools import supported_types

REPAIR_ROLES = ("retrieval", "analysis")
TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T")

SYSTEM_PROMPT = f"""You are the verification agent in a team answering questions about one patient's FHIR record.
You check another agent's draft answer independently. You do not see how it was produced.

Your tools fetch the patient's records (get_resources_by_patient_fhir_id, get_resources_by_resource_id) into
`retrieved_resources`, which starts empty, and run Python over them (execute_python_code; set `answer`).
Available resource types: {', '.join(supported_types)}. Only the question's patient can be retrieved, and records
dated after the assumed current time are left out.

1. Fetch the records the question needs yourself; do not rely on the draft's record list being complete.
2. Recompute the answer in Python, applying the question's scope (visit, time window, current time).
3. Compare with the draft. The automatic field check of the draft's citations is given to you; do not repeat its
   issues.

Report each problem as an issue:
- responsible_role "retrieval" when records the answer needs were not used because they were never retrieved,
  or the wrong visit or time window was searched;
- responsible_role "analysis" when the records were there but were filtered, sorted, computed, or cited wrongly.
Reply with only a JSON object:
{{"recomputed_answer": <your answer, in the same answer format; null if you could not recompute it>,
 "issues": [{{"type": "<short_snake_case>", "record": "<ResourceType/id or null>", "field": "<field path or null>",
             "responsible_role": "retrieval" | "analysis", "action": "<what the responsible agent should do>"}}]}}
Reply with "issues": [] if the draft is correct."""


def _canonical(value):
    """Comparable form of an answer: numbers rounded, text case-folded, lists unordered."""
    if isinstance(value, list):
        if len(value) == 1:
            return _canonical(value[0])
        return tuple(sorted(repr(_canonical(v)) for v in value))
    if isinstance(value, bool):
        return ("bool", value)  # True == 1.0 in Python; a yes/no answer is not a count
    if value is None:
        return value
    if isinstance(value, str):
        text = value.strip()
        if TIMESTAMP_RE.match(text):
            text = text.replace("T", " ", 1)
        text = text.casefold()
        try:
            value = float(text)
        except ValueError:
            return text
    if isinstance(value, (int, float)) and math.isfinite(value):
        return round(float(value), 6)
    return value


def _issue(kind, record=None, field=None, responsible_role="analysis", action=""):
    return {"type": kind, "record": record, "field": field, "responsible_role": responsible_role, "action": action}


def field_check_issues(draft, check):
    """Structured issues from a check_fields result for `draft`."""
    issues = []
    for text in check["issues"]:
        if text.startswith("encounter_scope:"):
            issues.append(_issue("encounter_scope", action=text.split(":", 1)[1].strip()))
    for result in check["results"]:
        citation = result["citation"] if isinstance(result["citation"], dict) else {}
        record, field = citation.get("resource_id"), citation.get("field_path")
        if result["status"] != "valid":
            actual = f" The value there is {json.dumps(result['actual'], default=str)}." if "actual" in result else ""
            issues.append(_issue(result["status"], record, field,
                                 action=f"Cite a field that exists in the record with its exact value.{actual}"))
        for warning in result["warnings"]:
            kind, _, detail = warning.partition(":")
            issues.append(_issue(kind.strip(), record, field, action=detail.strip()))
    if draft.get("answer") is not None and not draft.get("evidence"):
        issues.append(_issue("no_evidence", action="Cite the fields that support the answer"))
    return issues


class VerificationAgent:
    """Handler for the controller's verification tasks. Owns its workspace; nothing is shared."""

    def __init__(self, host, context, max_turns=10):
        self.host = host
        self.context = context
        self.workspace = {}
        self.checks = 0
        toolbox = Toolbox(RETRIEVAL_TOOLS + (PYTHON_TOOL,), self.workspace,
                          patient_fhir_id=context.get("patient_fhir_id"), current_time=context.get("current_time"))
        self.role = Role("verification", SYSTEM_PROMPT, toolbox, host, max_turns=max_turns)

    def __call__(self, task):
        draft = task["draft"]
        self.workspace.clear()
        issues = self._field_check(draft)
        if issues is None:
            return {"issues": [_issue("source_check", responsible_role=None,
                                      action="The tool-call budget did not allow the evidence check")]}
        result = self.role.run(self._prompt(draft, issues))
        if result.error:
            return {"error": result.error}
        verdict = next((obj for obj in _json_objects(result.content or "") if "issues" in obj), None)
        if verdict is None:
            return {"error": "the reply had no JSON verdict with issues"}
        if not isinstance(verdict["issues"], list) or not all(
                isinstance(i, dict) and isinstance(i.get("type"), str) and i["type"].strip()
                for i in verdict["issues"]):
            return {"error": "verdict issues must be a list of objects with a type"}
        for item in verdict["issues"]:
            role = item.get("responsible_role")
            issues.append(_issue(item["type"].strip(), item.get("record"), item.get("field"),
                                 role if role in REPAIR_ROLES else None, str(item.get("action") or "")))
        recomputed = verdict.get("recomputed_answer")
        if (recomputed is not None and _canonical(recomputed) != _canonical(draft.get("answer"))
                and not any(i["type"] == "recompute_mismatch" for i in issues)):
            issues.append(_issue("recompute_mismatch", action=(
                f"The verifier recomputed {json.dumps(recomputed)}; the draft answered "
                f"{json.dumps(draft.get('answer'))}. Recheck the filters and calculation.")))
        unique = {(i["type"], json.dumps(i["record"]), json.dumps(i["field"])): i for i in issues}
        return {"issues": list(unique.values()), "recomputed_answer": recomputed}

    def _field_check(self, draft):
        """check_fields on the draft's citations, recorded in the trace like the single agent's check;
        None if the tool-call budget is used up."""
        if not self.host._take_tool_call():
            return None
        self.checks += 1
        tool_id = f"verification_field_check_{self.checks}"
        arguments = {"citations": draft.get("evidence") or []}
        self.host.messages.append({"role": "assistant", "content": None, "tool_calls": [{
            "id": tool_id, "type": "function",
            "function": {"name": "check_fields", "arguments": json.dumps(arguments, default=str)}}]})
        check = check_fields(arguments["citations"], question=self.context["question"],
                             current_time=self.context.get("current_time"),
                             patient_fhir_id=self.context.get("patient_fhir_id"))
        self.host.messages.append({"role": "tool", "tool_call_id": tool_id, "name": "check_fields",
                                   "content": json.dumps(check, default=str)})
        return field_check_issues(draft, check)

    def _prompt(self, draft, issues):
        shown = {k: draft.get(k) for k in ("answer", "evidence", "operation", "contributing_ids", "limitation")}
        return "\n".join([
            f"Question: {self.context['question']}", f"Context: {self.context.get('assumptions', '')}", "",
            "Draft to verify:", json.dumps(shown, default=str), "",
            "Automatic field check of the draft's citations (already reported):",
            json.dumps(issues, default=str) if issues else "no problems found"])
