"""Analysis role (P2-14) for the fixed controller.

Filters, sorts, and computes over the records the retrieval role gathered,
using the sandboxed Python tool. It can supplement retrieval when execution
reveals missing evidence. An explicit final execution captures the complete
draft, avoiding prose copying of large contributor lists. Its draft extends
the common answer object with how the answer was computed:

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

from .framework import PYTHON_TOOL, RETRIEVAL_TOOLS, Role, Toolbox, _bound_tool_reply
from .ehr_semantics import SEMANTIC_GUIDANCE
from .output_schema import ANSWER_FORMAT_INSTRUCTIONS, _json_objects
from tools.resource_tools import FHIR_ID_RE, supported_types
from tools.starting_packet import render_resource
from utils.evidence_scorer import PATH_SEGMENT_RE

PREVIEW_CHARS = 1500

SYSTEM_PROMPT = f"""You are the analysis agent in a team answering questions about one patient's FHIR record.
Another agent has retrieved initial records, loaded in `retrieved_resources` (a dict from resource
type to lists of FHIR resources). You can fetch additional types or linked records when inspection shows
missing evidence. Do not assume the retriever's scope or the starting packet is complete.

Work in Python: filter the records to the question's scope, sort them, and compute the answer. Set the `answer`
variable to the JSON value you need back. Printed output is only a bounded diagnostic channel; it cannot be
the final answer. Compare times as wall-clock strings or
timezone-naive datetimes.
- Observation measurement time is `effectiveDateTime` (`issued` is when the result was released).
- A prescription's start time is `dispenseRequest.validityPeriod.start` (`authoredOn` is when it was ordered).
- Hospital visits are Encounters whose identifier system ends in "encounter-hosp"; ICU stays end in
  "encounter-icu" and link to their hospital visit through partOf.
Use mode="inspect" for bounded diagnostic counts, distinct code labels, and small samples; never return
all patient records as a diagnostic. Every Python invocation starts a fresh worker.
For example, inspect with code:
  answer = {{"counts": {{kind: len(rows) for kind, rows in retrieved_resources.items()}},
            "sample": retrieved_resources.get("Observation", [])[:2]}}
Do not set answer=None to inspect records: assign the actual diagnostic object you need to read.
For your final calculation call execute_python_code with mode="final". In that code set:
  answer = <actual scalar, list of answer values, or null>
  evidence = [{{"resource_id": "<ResourceType>/<id>", "field_path": "<path>", "value": <exact source value>}}]
  contributing_ids = ["<ResourceType>/<id>", ...]  # all scoped inputs and cited dependencies
  operation = "<filters, ordering, calculation and fields used>"
  limitation = None  # or explanation of genuinely missing evidence
One citation contains exactly one path and the exact source value. For example, for a selected Observation r:
  evidence = [{{"resource_id": "Observation/" + r["id"], "field_path": "valueQuantity.value",
               "value": r["valueQuantity"]["value"]}}]
Use resource_id/field_path/value, never resource/field/fields aliases. Contributor IDs always include the
resource type ("Observation/" + r["id"]), never bare IDs. The tool returns a correctable error for invalid
final metadata; fix it in one new final execution rather than inventing inaccessible records.
These variables are captured from the executed code so complete contributors/evidence need not be copied
into your prose response. Then return the final JSON object. If you discover more records are needed after
the final calculation, fetch them and run a new final calculation.

{SEMANTIC_GUIDANCE}

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


def _is_resource_id(value):
    if not isinstance(value, str) or value.count("/") != 1:
        return False
    rtype, rid = value.split("/")
    return rtype in supported_types and bool(FHIR_ID_RE.fullmatch(rid))


def _is_citation(value):
    return (isinstance(value, dict) and "value" in value
            and _is_resource_id(value.get("resource_id"))
            and isinstance(value.get("field_path"), str)
            and all(PATH_SEGMENT_RE.fullmatch(segment) for segment in value["field_path"].split(".")))


class AnalysisAgent:
    """Handler for the controller's analysis tasks; `workspace` is shared with the retrieval role."""

    def __init__(self, host, context, workspace, max_turns=10):
        self.context = context
        self.workspace = workspace
        self.toolbox = Toolbox(RETRIEVAL_TOOLS + (PYTHON_TOOL,), workspace,
                               patient_fhir_id=context.get("patient_fhir_id"),
                               current_time=context.get("current_time"))
        self.role = Role("analysis", SYSTEM_PROMPT, self.toolbox, host,
                         max_turns=max_turns, max_tool_calls=9)

    def __call__(self, task):
        self.toolbox.invalidate_computation()
        result = self.role.run(self._prompt(task))
        if result.error:
            return {"error": result.error}
        computation = getattr(self.toolbox, "final_computation", None)
        if computation is not None:
            # The executable artifact is the candidate; the model's final prose
            # cannot change a computed value or drop a large contributor list.
            computed = {k: computation.get(k) for k in
                        ("answer", "evidence", "operation", "contributing_ids", "limitation")}
            if isinstance(computed.get("evidence"), list) and isinstance(computed.get("contributing_ids"), list):
                computed["contributing_ids"] = list(dict.fromkeys([
                    *computed["contributing_ids"],
                    *(c.get("resource_id") for c in computed["evidence"] if isinstance(c, dict))]))
            candidate = self._draft(json.dumps(computed), self.workspace)
            return candidate
        return self._draft(result.content, self.workspace)

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
                labels = sorted({str(c.get("display") or c.get("code") or "")
                                 for r in items for c in (r.get("code") or {}).get("coding", [])})
                if labels:
                    parts.append("  Available concept labels (bounded): " + json.dumps(labels[:50]))
        if task.get("issues"):
            shown = _bound_tool_reply({"content": json.dumps({"previous_draft": task.get("previous_draft"),
                                                              "issues": task["issues"]})})["content"]
            parts += ["", "A verifier found problems with your previous draft:",
                      shown,
                      "Fix them, recomputing if needed, then reply with the corrected JSON object."]
        return "\n".join(parts)

    @staticmethod
    def _draft(content, resources=None):
        draft = next((obj for obj in _json_objects(content or "") if "answer" in obj), None)
        if draft is None:
            return {"error": "the reply had no JSON draft with an answer"}
        evidence = draft.get("evidence")
        if not _valid_answer(draft["answer"]):
            return {"error": "the draft answer must be a scalar, a list of answer values, or null"}
        if not isinstance(evidence, list) or not all(_is_citation(c) for c in evidence):
            return {"error": "draft evidence must be a list of {resource_id, field_path, value} citations"}
        try:
            json.dumps(draft, allow_nan=False)
        except (TypeError, ValueError):
            return {"error": "draft evidence and metadata must contain only finite JSON values"}
        contributing = draft.get("contributing_ids")
        if not isinstance(contributing, list) or not all(_is_resource_id(i) for i in contributing):
            return {"error": "contributing_ids must be a list of \"ResourceType/id\" strings"}
        contributing = list(dict.fromkeys(contributing))
        if any(c["resource_id"] not in contributing for c in evidence):
            return {"error": "every cited resource must be included in contributing_ids"}
        if resources is not None:
            held = {f"{rtype}/{r['id']}" for rtype, items in resources.items() for r in items}
            unknown = [rid for rid in contributing if rid not in held]
            if unknown:
                return {"error": "contributing_ids are not among the retrieved records: " + ", ".join(unknown)}
        operation, limitation = draft.get("operation"), draft.get("limitation")
        if (operation is not None and not isinstance(operation, str)) or (
                draft["answer"] is not None and (not isinstance(operation, str) or not operation.strip())):
            return {"error": "a non-abstaining draft must report a non-empty operation string"}
        if limitation is not None and not isinstance(limitation, str):
            return {"error": "draft limitation must be a string or null"}
        return {
            "answer": draft["answer"],
            "evidence": evidence,
            "operation": operation,
            "contributing_ids": contributing,
            "limitation": limitation,
        }
