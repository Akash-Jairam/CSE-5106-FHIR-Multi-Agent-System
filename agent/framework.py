"""Message passing for the Phase 2 multi-agent system (see docs/phase2_framework.md).

LangGraph executes fixed Python workflows; the adapters below retain
BaseAgent's budget/usage accounting and the existing tool sandbox:

- FixedWorkflow: local StateGraph execution with deterministic edges,
  no checkpoints, retries, cache, or external tracing.
- Bus: structured messages between the controller and roles. Payloads must
  be JSON values and are copied, so no role can share or mutate another's
  objects.
- Toolbox: one role's allow-listed view of the tool registry. Model
  arguments are limited to each tool's declared parameters; question
  context (patient, current time) is bound by the system, never the model.
- Role: one LLM role with a fresh context per task, its own tools, and the
  host agent's budget (BaseAgent._llm_call / _take_tool_call).

The host is a BaseAgent whose _run() is the fixed controller, so
BaseAgent.run() still returns the common output contract.
"""
import json
from copy import deepcopy
from dataclasses import asdict, dataclass

from tools import get_tool, get_tool_definitions
from tools.field_check import _resource_time
from tools.resource_tools import FHIR_ID_RE
from tools.starting_packet import render_resource

RETRIEVAL_TOOLS = ("get_resources_by_patient_fhir_id", "get_resources_by_resource_id")
PYTHON_TOOL = "execute_python_code"


class FixedWorkflow:
    """A LangGraph workflow over a JSON state object, with local-only execution.

    Nodes return the complete next state. Edges use None for the end node;
    branches are (source, router, destinations) with the same convention.
    Models and tools remain ordinary node code, never framework executors.
    """

    def __init__(self, nodes, entry_point, edges=(), branches=(), max_steps=16):
        from langgraph.graph import END, StateGraph

        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        builder = StateGraph(dict)
        for name, handler in nodes.items():
            builder.add_node(name, handler, retry_policy=None)
        builder.set_entry_point(entry_point)
        for source, target in edges:
            builder.add_edge(source, END if target is None else target)
        for source, router, destinations in branches:
            builder.add_conditional_edges(
                source, router, {key: END if target is None else target for key, target in destinations.items()})
        self.graph = builder.compile(checkpointer=None, cache=None)
        self.max_steps = max_steps

    def run(self, state):
        from langsmith import tracing_context

        # A developer's tracing environment must not export patient/question data.
        with tracing_context(enabled=False):
            return self.graph.invoke(deepcopy(state), config={
                "recursion_limit": self.max_steps, "callbacks": []})


@dataclass(frozen=True)
class Message:
    sender: str
    recipient: str
    kind: str       # e.g. "task", "records", "draft", "issues", "final"
    payload: dict

    def to_dict(self):
        return asdict(self)


class Bus:
    """Ordered log of structured messages for one question."""

    def __init__(self):
        self._log = []

    @property
    def log(self):
        return deepcopy(self._log)

    def send(self, sender, recipient, kind, payload):
        if not isinstance(payload, dict):
            raise TypeError("A message payload must be an object")
        try:
            # The round trip both validates and copies the payload
            payload = json.loads(json.dumps(payload, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"A message payload must be a finite JSON value: {exc}") from None
        message = Message(sender, recipient, kind, payload)
        self._log.append(message)
        return deepcopy(message)

    def to_list(self):
        return [m.to_dict() for m in self._log]


class Toolbox:
    """Allow-listed registry tools over one role's working records.

    `resources` is the role's working copy of retrieved records, by resource
    type; roles that should see the same records share the dict. `bound`
    maps a tool name to keyword arguments the system supplies (they replace
    any the model sends). Fetched records owned by another patient, or dated
    after `current_time` ("YYYY-MM-DD HH:MM:SS"), are left out.
    """

    def __init__(self, names, resources=None, patient_fhir_id=None, bound=None, current_time=None):
        definitions = {d["function"]["name"]: d for d in get_tool_definitions()}
        missing = [name for name in names if name not in definitions]
        if missing:
            raise KeyError(f"Tools not in the registry: {', '.join(missing)}")
        self.names = tuple(names)
        self.definitions = [definitions[name] for name in self.names]
        self.resources = {} if resources is None else resources
        self.patient_fhir_id = patient_fhir_id
        self.current_time = current_time
        self.bound = bound or {}
        self.scope_failures = {}

    def _parameters(self, name):
        definition = next(d for d in self.definitions if d["function"]["name"] == name)
        return set(definition["function"].get("parameters", {}).get("properties", {}))

    def call(self, tool_call):
        """Run one tool call; returns the tool message."""
        name = tool_call.function.name
        if name not in self.names:
            return self._error(tool_call, f"Tool '{name}' is not available to this role")
        try:
            args = json.loads(tool_call.function.arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("Arguments must be a JSON object")
        except (ValueError, TypeError) as exc:
            return self._error(tool_call, str(exc))
        unknown = set(args) - self._parameters(name)
        if unknown:
            return self._error(tool_call, f"Unknown arguments: {', '.join(sorted(unknown))}")
        args.update(self.bound.get(name, {}))
        try:
            if name in RETRIEVAL_TOOLS:
                content = self._retrieve(name, args)
            elif name == PYTHON_TOOL:
                content = json.dumps(get_tool(name)(**args, global_vars={"retrieved_resources": self.resources}),
                                     default=str)
            else:
                content = json.dumps(get_tool(name)(**args), default=str)
        except Exception as exc:
            return self._error(tool_call, str(exc))
        return {"tool_call_id": tool_call.id, "role": "tool", "name": name, "content": content}

    def _retrieve(self, name, args):
        if (name == "get_resources_by_patient_fhir_id" and self.patient_fhir_id
                and args.get("patient_fhir_id") != self.patient_fhir_id):
            raise ValueError("Retrieve only the patient identified in this question")
        output = get_tool(name)(**args)
        if not isinstance(output, dict) or "error" in output:
            raise ValueError(output.get("error") if isinstance(output, dict)
                             else "The retrieval tool returned an invalid result")
        resource_type = args["resource_type"]
        records = output.get(resource_type, [])
        if not isinstance(records, list) or not all(
                isinstance(r, dict) and r.get("resourceType") == resource_type
                and isinstance(r.get("id"), str) and FHIR_ID_RE.fullmatch(r["id"]) for r in records):
            raise ValueError("The retrieval tool returned an invalid resource list")
        if name == "get_resources_by_resource_id" and any(r["id"] != args["resource_id"] for r in records):
            raise ValueError("The retrieval tool returned a resource other than the requested ID")
        fetched = {r["id"] for r in records}
        records, excluded = self._in_scope(records)
        # Preserve earlier evidence across searches, but replace a fetched ID with
        # its original record (or remove it if the new source fails scope checks).
        merged = {r["id"]: r for r in self.resources.get(resource_type, []) if r["id"] not in fetched}
        merged.update({r["id"]: deepcopy(r) for r in records})
        self.resources[resource_type] = list(merged.values())
        reply = {"stored_as": f"retrieved_resources['{resource_type}']",
                 "ids": [f"{resource_type}/{r.get('id')}" for r in records],
                 "items": len(self.resources[resource_type])}
        if resource_type == "Encounter":
            # IDs alone cannot establish first/last visit scope if the packet
            # omitted the encounter's period or hospital/ICU identifier.
            reply["records"] = [json.loads(render_resource(r)) for r in records]
        return json.dumps({**reply, "excluded": excluded} if excluded else reply)

    @staticmethod
    def _references(value):
        if isinstance(value, dict):
            reference = value.get("reference")
            if isinstance(reference, str):
                parts = reference.rstrip("/").split("/")
                if len(parts) >= 2:
                    yield "/".join(parts[-2:])
            for item in value.values():
                yield from Toolbox._references(item)
        elif isinstance(value, list):
            for item in value:
                yield from Toolbox._references(item)

    @staticmethod
    def _source_time(record, by_id, seen=None):
        key = f"{record.get('resourceType')}/{record.get('id')}"
        seen = set() if seen is None else seen
        if key in seen:
            return ""
        seen.add(key)
        time = _resource_time(record)
        if time:
            return time
        # Specimens and period-valued events use additional standard FHIR paths.
        for value in ((record.get("collection") or {}).get("collectedDateTime"),
                      ((record.get("collection") or {}).get("collectedPeriod") or {}).get("start"),
                      (record.get("effectivePeriod") or {}).get("start")):
            if isinstance(value, str) and value:
                return value[:19].replace("T", " ")
        references = [record.get("encounter"), record.get("context")]
        references.extend(record.get("derivedFrom") or [])
        for reference in references:
            for target in Toolbox._references(reference):
                if target in by_id:
                    time = Toolbox._source_time(by_id[target], by_id, seen)
                    if time:
                        return time
        return ""

    def _in_scope(self, records):
        """Records within the question's patient and current time, and counts of those left out."""
        by_id = {f"{r.get('resourceType')}/{r.get('id')}": r
                 for items in self.resources.values() for r in items}
        by_id.update({f"{r.get('resourceType')}/{r.get('id')}": r for r in records})
        linked_metadata = {reference for items in self.resources.values() for r in items
                           for reference in self._references(r)}
        kept, excluded = [], {}
        for record in records:
            rtype = record["resourceType"]
            if rtype == "Patient":
                owner = record.get("id")
            else:
                reference = record.get("subject") or record.get("patient") or {}
                patient_ref = reference.get("reference", "") if isinstance(reference, dict) else ""
                parts = patient_ref.rstrip("/").split("/") if isinstance(patient_ref, str) else []
                owner = parts[-1] if len(parts) >= 2 and parts[-2] == "Patient" else None
            reason = None
            if self.patient_fhir_id:
                if rtype in ("Medication", "Location"):
                    if f"{rtype}/{record['id']}" not in linked_metadata:
                        reason = "unlinked_metadata"
                elif not owner:
                    reason = "unknown_patient"
                elif owner != self.patient_fhir_id:
                    reason = "other_patient"
            if not reason and self.current_time and rtype not in ("Patient", "Medication", "Location"):
                time = self._source_time(record, by_id)
                if not time:
                    reason = "unknown_time"
                elif time > self.current_time:
                    reason = "after_current_time"
            if reason:
                excluded[reason] = excluded.get(reason, 0) + 1
            else:
                kept.append(record)
            key = f"{rtype}/{record['id']}"
            if reason in ("unknown_patient", "unknown_time"):
                self.scope_failures[key] = reason
            else:
                self.scope_failures.pop(key, None)
        return kept, excluded

    @staticmethod
    def _error(tool_call, message):
        return {"tool_call_id": tool_call.id, "role": "tool", "name": tool_call.function.name,
                "content": json.dumps({"error": message}, default=str)}


@dataclass
class RoleResult:
    content: str = None   # the role's final reply, if it gave one
    error: str = None


class Role:
    """One LLM role. Each run() starts a fresh context; usage, budget, and
    trace go to the host agent so the whole system answers to one ceiling."""

    def __init__(self, name, system_prompt, toolbox, host, max_turns=10):
        self.name = name
        self.system_prompt = system_prompt
        self.toolbox = toolbox
        self.host = host
        self.max_turns = max_turns

    def run(self, content):
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content}]
        self.host.messages.extend(deepcopy(messages))
        tools = self.toolbox.definitions or None
        for _ in range(self.max_turns):
            response, error, _ = self.host._llm_call(messages, tools=tools)
            if error:
                return RoleResult(error=error)
            messages.append(response)
            self.host.messages.append(response)
            if not response.tool_calls:
                return RoleResult(content=response.content)
            for tool_call in response.tool_calls:
                if self.host._take_tool_call():
                    reply = self.toolbox.call(tool_call)
                else:
                    reply = self.host._tool_budget_message(tool_call)
                messages.append(reply)
                self.host.messages.append(reply)
        return RoleResult(error=f"{self.name}: no final reply within {self.max_turns} turns")
