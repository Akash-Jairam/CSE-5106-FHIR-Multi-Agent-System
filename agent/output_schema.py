"""Common output contract shared by every system (direct LLM, single agent, multi-agent).

`BaseAgent.run()` returns an `AgentOutput`:

    {
      "answer": true | 42 | "2133-12-31 02:00:00" | "Heparin" | [...] | null,
      "evidence": [{"resource_id": "Observation/<id>", "field_path": "valueQuantity.value", "value": 98.7}],
      "limitation": null | "No weight was recorded on the first ICU visit",
      "tokens": {"prompt": 1200, "completion": 85, "total": 1285, "cost": 0.0004, "llm_calls": 2},
      "tool_calls": [{"name": "get_resources_by_patient_fhir_id", "arguments": {...}}],
      "latency": 3.42,
      ...diagnostics: "trace", "retrieved_fhir_resources", "error", "revision"
    }

`answer` is null when the system abstains; `limitation` says why, or what is
uncertain about a given answer. Agents written against this contract should
end with a JSON object as described in ANSWER_FORMAT_INSTRUCTIONS; older
agents that answer in free text are wrapped with the text as `answer`.
"""
import json
import re
from typing import Any, Optional, TypedDict


class Citation(TypedDict):
    resource_id: str  # "ResourceType/id"
    field_path: str   # e.g. "valueQuantity.value", "code.coding[0].display"
    value: Any        # the value at field_path


class Tokens(TypedDict):
    prompt: int
    completion: int
    total: int
    cost: float
    llm_calls: int


class ToolCall(TypedDict):
    name: str
    arguments: Any


class AgentOutput(TypedDict):
    answer: Any                # bool | number | str | list | None (abstained)
    evidence: list             # list[Citation]
    limitation: Optional[str]
    tokens: Tokens
    tool_calls: list           # list[ToolCall]
    latency: float             # wall-clock seconds for the whole run
    trace: list                # message history (diagnostic)
    retrieved_fhir_resources: dict
    error: Optional[str]
    revision: Optional[dict]   # agents with a revision step: draft answer, check issues, whether it changed


ANSWER_FORMAT_INSTRUCTIONS = """When you have the final answer, reply with only a JSON object:
{"answer": <value>, "evidence": [{"resource_id": "<ResourceType>/<id>", "field_path": "<path>", "value": <value at that path>}], "limitation": <null or a short explanation>}
- answer: true/false for yes/no questions; a number for counts and measurements (no units; use the unit the question asks for); "YYYY-MM-DD HH:MM:SS" for times; a string for names and categories; a list when there are several values; null if the patient's record does not contain the answer.
- evidence: the FHIR fields that support the answer, as dotted paths into the resource (e.g. "valueQuantity.value", "effectiveDateTime", "code.coding[0].display"), with the exact value found there.
- limitation: null if the answer is fully supported; otherwise what is missing or uncertain."""

def _json_objects(text):
    """JSON objects embedded in `text` (bare or in ``` fences), last one first."""
    decoder = json.JSONDecoder()
    objects = []
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            objects.append(obj)
    return objects[::-1]


def has_json_answer(text):
    """Whether `text` contains a JSON object with an "answer" key."""
    return isinstance(text, str) and any("answer" in obj for obj in _json_objects(text))


def parse_final_message(text):
    """(answer, evidence, limitation) from a model's final message.

    Uses the last JSON object with an "answer" key; otherwise the whole text
    is a free-text answer.
    """
    if not isinstance(text, str):
        return text, [], None
    for obj in _json_objects(text):
        if "answer" in obj:
            evidence = obj.get("evidence") or []
            limitation = obj.get("limitation")
            return (obj["answer"],
                    evidence if isinstance(evidence, list) else [evidence],
                    limitation if limitation is None else str(limitation))
    return text, [], None


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def extract_tool_calls(trace):
    """Tool calls made during a run, from the assistant messages in `trace`."""
    calls = []
    for message in trace or []:
        if _get(message, "role") != "assistant":
            continue
        for call in _get(message, "tool_calls") or []:
            function = _get(call, "function")
            arguments = _get(function, "arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            calls.append({"name": _get(function, "name"), "arguments": arguments})
    return calls


def to_tokens(usage):
    usage = usage or {}
    return {
        "prompt": usage.get("prompt_tokens", 0),
        "completion": usage.get("completion_tokens", 0),
        "total": usage.get("total_tokens", 0),
        "cost": usage.get("cost", 0.0),
        "llm_calls": usage.get("llm_calls", 0),
    }


def validate_output(output):
    """Problems with an AgentOutput (empty list if it follows the contract)."""
    problems = []
    for key in AgentOutput.__annotations__:
        if key not in output:
            problems.append(f"missing key: {key}")
    if not isinstance(output.get("evidence", []), list):
        problems.append("evidence must be a list")
    for i, citation in enumerate(output.get("evidence") or []):
        if not isinstance(citation, dict) or not {"resource_id", "field_path", "value"} <= set(citation):
            problems.append(f"evidence[{i}] must have resource_id, field_path, and value")
    if output.get("limitation") is not None and not isinstance(output["limitation"], str):
        problems.append("limitation must be a string or null")
    if not isinstance(output.get("tool_calls", []), list):
        problems.append("tool_calls must be a list")
    if not isinstance(output.get("latency", 0), (int, float)):
        problems.append("latency must be a number of seconds")
    if set(output.get("tokens") or {}) != set(Tokens.__annotations__):
        problems.append(f"tokens must have keys {sorted(Tokens.__annotations__)}")
    return problems
