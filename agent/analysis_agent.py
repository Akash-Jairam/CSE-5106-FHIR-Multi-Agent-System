"""Analysis role (P2-14) for the fixed controller.

Filters, sorts, and computes over the records the retrieval role gathered,
using the sandboxed Python tool. It sees what the single agent sees (the
starting packet, worked examples and search guidance) plus the shared
EHR conventions, and it may fetch a record type it finds missing. Its draft
extends the common answer object with how the answer was computed:

    {"answer": ..., "evidence": [{"resource_id", "field_path", "value"}],
     "operation": "max of valueQuantity.value over ...",
     "contributing_ids": ["Observation/o1", ...],
     "limitation": null | "..."}

`contributing_ids` lists every record the computation used, while
`evidence` cites the fields that support the answer (for a count, the counted
records may contribute without each being cited). Only a reply without a
usable answer is an error; malformed citations and contributors are dropped
or completed rather than discarding the answer.
"""
import json
import math

from .ehr_semantics import SEMANTIC_GUIDANCE
from .framework import PYTHON_TOOL, RETRIEVAL_TOOLS, Role, Toolbox
from .multi_turn_code_resource_agent import WORKED_EXAMPLES, MultiTurnCodeResourceAgent
from .output_schema import ANSWER_FORMAT_INSTRUCTIONS, _json_objects
from tools.resource_tools import FHIR_ID_RE, supported_types
from tools.starting_packet import render_resource
from utils.evidence_scorer import PATH_SEGMENT_RE

PREVIEW_CHARS = 1500
DEFAULT_OPERATION = "computed with execute_python_code"

SYSTEM_PROMPT = f"""You are the analysis agent in a team answering questions about one patient's FHIR record.
Another agent has already retrieved the patient's records of every relevant type. They are loaded in
`retrieved_resources` (a dict from resource type to a list of FHIR resources) for execute_python_code.
The user message also shows the starting packet: an excerpt of the record chosen for this question.

Work in Python: filter the records to the question's scope, sort them, and compute the answer. Set the `answer`
variable to the JSON value you need back; printed output is discarded unless you return it in `answer`.
Compare times as wall-clock strings or timezone-naive datetimes.
If a record type you need is missing from `retrieved_resources`, fetch it with get_resources_by_patient_fhir_id
(available types: {', '.join(supported_types)}); it is then added to `retrieved_resources`.
Answer null only when the records establish nothing; an established absence is 0 for a count and false for a
yes/no question.
{MultiTurnCodeResourceAgent.search_guidance}
{SEMANTIC_GUIDANCE}
{WORKED_EXAMPLES}

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
    if not (isinstance(value, dict) and "value" in value
            and _is_resource_id(value.get("resource_id"))
            and isinstance(value.get("field_path"), str)
            and all(PATH_SEGMENT_RE.fullmatch(segment) for segment in value["field_path"].split("."))):
        return False
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def reply_problem(content):
    """Why a reply carries no usable draft, or None."""
    draft = next((obj for obj in _json_objects(content or "") if "answer" in obj), None)
    if draft is None:
        return "it had no JSON object with an \"answer\""
    if not _valid_answer(draft["answer"]):
        return "the answer must be a scalar, a list of answer values, or null"
    return None


class AnalysisAgent:
    """Handler for the controller's analysis tasks; `workspace` is shared with the retrieval role.

    `packet` returns the starting packet the retrieval role loaded (or None)."""

    def __init__(self, host, context, workspace, max_turns=12, packet=None):
        self.context = context
        self.workspace = workspace
        self.packet = packet or (lambda: None)
        toolbox = Toolbox(RETRIEVAL_TOOLS + (PYTHON_TOOL,), workspace, patient_fhir_id=context.get("patient_fhir_id"),
                          current_time=context.get("current_time"))
        self.role = Role("analysis", SYSTEM_PROMPT, toolbox, host, max_turns=max_turns)

    def __call__(self, task):
        result = self.role.run(self._prompt(task), check=reply_problem)
        if result.error:
            return {"error": result.error}
        return self._draft(result.content, self.workspace)

    def _prompt(self, task):
        records = task.get("records") or {}
        parts = [f"Question: {self.context['question']}", f"Context: {self.context.get('assumptions', '')}",
                 f"The patient's FHIR ID is {self.context.get('patient_fhir_id')}. "
                 f"Assume the current time is {self.context.get('current_time')}.", "",
                 "Retrieval report:", json.dumps({k: records.get(k) for k in ("resource_types", "scope", "limitation")}),
                 "", "retrieved_resources holds:"]
        for rtype, items in self.workspace.items():
            if items:
                sample = render_resource({"resourceType": rtype, **items[0]})
                if len(sample) > PREVIEW_CHARS:
                    sample = sample[:PREVIEW_CHARS] + "..."
                parts.append(f"- {rtype}: {len(items)} records, e.g. {sample}")
        packet = self.packet()
        if packet and packet.get("text"):
            parts += ["", "Starting packet (an excerpt; `retrieved_resources` holds the complete records):",
                      packet["text"]]
        if task.get("issues"):
            parts += ["", "A verifier found problems with your previous draft:",
                      json.dumps(task.get("previous_draft")), "Issues:", json.dumps(task["issues"], indent=1),
                      "Recheck them in Python. Keep your answer if it is right; otherwise correct it. "
                      "Reply with the JSON object."]
        return "\n".join(parts)

    @staticmethod
    def _draft(content, resources=None):
        """The reply's draft. Malformed citations are dropped and contributors are completed from the
        citations, so the answer is never discarded for its metadata."""
        problem = reply_problem(content)
        if problem:
            return {"error": f"the reply was not a usable draft: {problem}"}
        draft = next(obj for obj in _json_objects(content or "") if "answer" in obj)
        evidence = draft.get("evidence")
        evidence = [c for c in evidence if _is_citation(c)] if isinstance(evidence, list) else []
        contributing = draft.get("contributing_ids")
        contributing = [i for i in contributing if _is_resource_id(i)] if isinstance(contributing, list) else []
        contributing = list(dict.fromkeys(contributing + [c["resource_id"] for c in evidence]))
        if resources is not None:
            held = {f"{rtype}/{r['id']}" for rtype, items in resources.items() for r in items}
            contributing = [rid for rid in contributing if rid in held]
        operation, limitation = draft.get("operation"), draft.get("limitation")
        if not isinstance(operation, str) or not operation.strip():
            operation = None if draft["answer"] is None else DEFAULT_OPERATION
        if limitation is not None and not isinstance(limitation, str):
            limitation = json.dumps(limitation, default=str)
        return {
            "answer": draft["answer"],
            "evidence": evidence,
            "operation": operation,
            "contributing_ids": contributing,
            "limitation": limitation,
        }
