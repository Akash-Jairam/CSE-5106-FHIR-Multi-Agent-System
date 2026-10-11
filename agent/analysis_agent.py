"""Analysis role (P2-14) for the fixed controller.

Filters, sorts, and computes over the records the retrieval role gathered,
using only the sandboxed Python tool. It cannot fetch: if evidence is
missing it says so, and repair routing sends that to retrieval. Its draft
extends the common answer object with how the answer was computed:

    {"answer": ..., "evidence": [{"resource_id", "field_path", "value"}],
     "operation": "max of valueQuantity.value over ...",
     "contributing_ids": ["Observation/o1", ...],
     "limitation": null | "..."}

`contributing_ids` lists every record the computation used, while
`evidence` cites the fields that support the answer (for a count, the counted
records may contribute without each being cited). A reply without a
well-formed draft is an error, as for the direct baseline.
"""
import json
import math

from .framework import PYTHON_TOOL, Role, Toolbox
from .output_schema import ANSWER_FORMAT_INSTRUCTIONS, _json_objects
from tools.starting_packet import render_resource

PREVIEW_CHARS = 1500

SYSTEM_PROMPT = f"""You are the analysis agent in a team answering questions about one patient's FHIR record.
Another agent has already retrieved the records. They are loaded in `retrieved_resources` (a dict from resource
type to a list of FHIR resources) for execute_python_code, your only tool. You cannot fetch more records.

Work in Python: filter the records to the question's scope, sort them, and compute the answer. Set the `answer`
variable to the JSON value you need back; printed output is discarded. Compare times as wall-clock strings or
timezone-naive datetimes.
- Observation measurement time is `effectiveDateTime` (`issued` is when the result was released).
- A prescription's start time is `dispenseRequest.validityPeriod.start` (`authoredOn` is when it was ordered).
- Hospital visits are Encounters whose identifier system ends in "encounter-hosp"; ICU stays end in
  "encounter-icu" and link to their hospital visit through partOf.
If the records needed are missing, answer null and say what is missing in the limitation.

{ANSWER_FORMAT_INSTRUCTIONS}
Also include in the same JSON object:
- "operation": one sentence on how the answer was computed (filters, sort order, aggregation, fields used).
- "contributing_ids": every record the computation used, as "<ResourceType>/<id>"."""


def _valid_answer(value):
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return not isinstance(value, float) or math.isfinite(value)
    return isinstance(value, list) and all(_valid_answer(item) for item in value)


def _is_citation(value):
    return (isinstance(value, dict) and "value" in value
            and all(isinstance(value.get(k), str) and value[k].strip() for k in ("resource_id", "field_path")))


class AnalysisAgent:
    """Handler for the controller's analysis tasks; `workspace` is shared with the retrieval role."""

    def __init__(self, host, context, workspace, max_turns=10):
        self.context = context
        self.workspace = workspace
        self.role = Role("analysis", SYSTEM_PROMPT, Toolbox([PYTHON_TOOL], workspace), host, max_turns=max_turns)

    def __call__(self, task):
        result = self.role.run(self._prompt(task))
        if result.error:
            return {"error": result.error}
        return self._draft(result.content)

    def _prompt(self, task):
        records = task.get("records") or {}
        parts = [f"Question: {self.context['question']}", f"Context: {self.context.get('assumptions', '')}", "",
                 "Retrieval report:", json.dumps({k: records.get(k) for k in ("resource_types", "scope", "limitation")}),
                 "", "retrieved_resources holds:"]
        for rtype, items in self.workspace.items():
            if items:
                sample = render_resource({"resourceType": rtype, **items[0]})
                if len(sample) > PREVIEW_CHARS:
                    sample = sample[:PREVIEW_CHARS] + "..."
                parts.append(f"- {rtype}: {len(items)} records, e.g. {sample}")
        if task.get("issues"):
            parts += ["", "A verifier found problems with your previous draft:",
                      json.dumps(task.get("previous_draft")), "Issues:", json.dumps(task["issues"], indent=1),
                      "Fix them, recomputing if needed, then reply with the corrected JSON object."]
        return "\n".join(parts)

    @staticmethod
    def _draft(content):
        draft = next((obj for obj in _json_objects(content or "") if "answer" in obj), None)
        if draft is None:
            return {"error": "the reply had no JSON draft with an answer"}
        evidence = draft.get("evidence") or []
        if not _valid_answer(draft["answer"]):
            return {"error": "the draft answer must be a scalar, a list of answer values, or null"}
        if not isinstance(evidence, list) or not all(_is_citation(c) for c in evidence):
            return {"error": "draft evidence must be a list of {resource_id, field_path, value} citations"}
        contributing = draft.get("contributing_ids")
        if contributing is None:
            contributing = list(dict.fromkeys(c["resource_id"] for c in evidence))
        if not isinstance(contributing, list) or not all(isinstance(i, str) and "/" in i for i in contributing):
            return {"error": "contributing_ids must be a list of \"ResourceType/id\" strings"}
        operation, limitation = draft.get("operation"), draft.get("limitation")
        return {
            "answer": draft["answer"],
            "evidence": evidence,
            "operation": operation if operation is None or isinstance(operation, str) else json.dumps(operation),
            "contributing_ids": contributing,
            "limitation": limitation if limitation is None or isinstance(limitation, str) else json.dumps(limitation),
        }
