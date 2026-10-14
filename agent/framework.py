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
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass

from tools import get_tool, get_tool_definitions
from tools.field_check import _resource_time
from tools.resource_tools import FHIR_ID_RE
from tools.starting_packet import render_resource

RETRIEVAL_TOOLS = ("get_resources_by_patient_fhir_id", "get_resources_by_resource_id")
PYTHON_TOOL = "execute_python_code"
MODEL_RETRIEVAL_ITEMS = 32
MODEL_RETRIEVAL_CHARS = 8_000


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
        self.last_execution = None
        self.final_computation = None

    def workspace_sha256(self):
        """Identity of the exact copied records used by the sandbox worker."""
        return hashlib.sha256(json.dumps(self.resources, sort_keys=True, default=str,
                                         allow_nan=False).encode()).hexdigest()

    def invalidate_computation(self):
        self.last_execution = None
        self.final_computation = None

    def _parameters(self, name):
        definition = next(d for d in self.definitions if d["function"]["name"] == name)
        return set(definition["function"].get("parameters", {}).get("properties", {}))

    def call(self, tool_call):
        """Run one tool call; returns the tool message."""
        name = tool_call.function.name
        if name in RETRIEVAL_TOOLS:
            # A failed fetch still leaves the intended calculation incomplete.
            self.invalidate_computation()
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
                mode = args.get("mode", "auto")
                workspace = self.workspace_sha256()
                if self.final_computation and self.final_computation["workspace_sha256"] != workspace:
                    self.invalidate_computation()
                if mode != "inspect":
                    self.invalidate_computation()
                output = get_tool(name)(**args, global_vars={"retrieved_resources": self.resources})
                if isinstance(output, dict) and "answer" in output and not output.get("error"):
                    self.last_execution = deepcopy({**output, "mode": mode, "code": args["code"],
                                                    "workspace_sha256": workspace})
                    if mode == "final":
                        self.final_computation = deepcopy(self.last_execution)
                content = json.dumps(output, default=str)
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
            if isinstance(output, dict) and isinstance(output.get("retrieval_metadata"), dict):
                return json.dumps({"error": str(output.get("error")),
                                   "retrieval_metadata": deepcopy(output["retrieval_metadata"])})
            raise ValueError(output.get("error") if isinstance(output, dict)
                             else "The retrieval tool returned an invalid result")
        if (isinstance(output.get("retrieval_metadata"), dict)
                and output["retrieval_metadata"].get("complete") is not True):
            return json.dumps({"error": "The retrieval tool did not complete all linked dependencies",
                               "retrieval_metadata": deepcopy(output["retrieval_metadata"])})
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
        if isinstance(output.get("retrieval_metadata"), dict):
            reply["retrieval_metadata"] = deepcopy(output["retrieval_metadata"])
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
        linked_metadata = set()
        for items in self.resources.values():
            for record in items:
                if record.get("resourceType") in ("Medication", "Location"):
                    continue  # Shared metadata is never an ownership root.
                owner_ref = record.get("subject") or record.get("patient") or {}
                parts = owner_ref.get("reference", "").rstrip("/").split("/") if isinstance(owner_ref, dict) else []
                owner = parts[-1] if len(parts) >= 2 and parts[-2] == "Patient" else ""
                if record.get("resourceType") == "Patient":
                    owner = record.get("id")
                if self.patient_fhir_id and owner != self.patient_fhir_id:
                    continue
                if self.current_time and record.get("resourceType") != "Patient":
                    moment = self._source_time(record, by_id)
                    if not moment or moment > self.current_time:
                        continue
                linked_metadata.update(self._references(record))
        # A mix and all ingredient components may arrive together in one batch. Traverse only
        # the standard Medication ingredient graph, rooted in held patient-owned source events.
        pending = {reference for reference in linked_metadata if reference.startswith("Medication/")}
        visited = set()
        while pending:
            key = pending.pop()
            if key in visited:
                continue
            visited.add(key)
            medication = by_id.get(key) or {}
            for ingredient in medication.get("ingredient", []):
                for reference in self._references(ingredient.get("itemReference")):
                    if reference.startswith("Medication/") and reference not in linked_metadata:
                        linked_metadata.add(reference)
                        pending.add(reference)
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


def _model_tool_reply(reply):
    """Bound every model-facing result; raw worker/check results stay in the audit."""
    model_reply = deepcopy(reply)
    if reply.get("name") not in RETRIEVAL_TOOLS:
        return _bound_tool_reply(model_reply)
    try:
        content = json.loads(reply["content"])
    except (KeyError, TypeError, ValueError):
        return _bound_tool_reply(model_reply)
    if not isinstance(content, dict) or "error" in content or not isinstance(content.get("ids"), list):
        return _bound_tool_reply(model_reply)
    ids, records = content["ids"], content.get("records")
    if (len(ids) <= MODEL_RETRIEVAL_ITEMS and len(reply["content"]) <= MODEL_RETRIEVAL_CHARS
            and (not isinstance(records, list) or len(records) <= MODEL_RETRIEVAL_ITEMS)):
        return model_reply
    preview = {**content, "ids": ids[:MODEL_RETRIEVAL_ITEMS], "ids_count": len(ids),
               "ids_truncated": len(ids) > MODEL_RETRIEVAL_ITEMS,
               "preview_note": "IDs and records are a bounded preview; all retrieved records remain in the workspace."}
    if isinstance(records, list):
        preview.update(records=[], records_count=len(records), records_truncated=bool(records))
        for record in records[:MODEL_RETRIEVAL_ITEMS]:
            candidate = {**preview, "records": [*preview["records"], record],
                         "records_truncated": len(preview["records"]) + 1 < len(records)}
            if len(json.dumps(candidate)) > MODEL_RETRIEVAL_CHARS:
                break
            preview = candidate
    model_reply["content"] = json.dumps(preview)
    return _bound_tool_reply(model_reply)


def _bound_tool_reply(reply):
    """Finite JSON previews of arbitrary tool payloads, including Python diagnostics."""
    raw = reply.get("content")
    if not isinstance(raw, str) or len(raw) <= MODEL_RETRIEVAL_CHARS:
        return reply
    try:
        value = json.loads(raw)
    except ValueError:
        value = raw
    for string_cap, item_cap, key_cap in ((2048, 32, 32), (512, 16, 16), (128, 8, 8), (32, 2, 4)):
        truncated = []

        def preview(item, path="$", depth=0):
            if depth > 6:
                truncated.append(path)
                return {"preview_omitted": True, "value_type": type(item).__name__}
            if isinstance(item, str) and len(item) > string_cap:
                truncated.append(path)
                return item[:string_cap] + " [truncated]"
            if isinstance(item, list):
                if len(item) > item_cap:
                    truncated.append(f"{path}: {len(item)} items")
                return [preview(child, f"{path}[{index}]", depth + 1)
                        for index, child in enumerate(item[:item_cap])]
            if isinstance(item, dict):
                priority = ("error", "mode", "answer", "operation", "stored_as", "items", "n_valid",
                            "ids_count", "records_count", "issues", "evidence", "contributing_ids")
                keys = [key for key in priority if key in item]
                keys += [key for key in item if key not in keys]
                if len(keys) > key_cap:
                    truncated.append(f"{path}: {len(keys)} keys")
                return {key[:128]: preview(item[key], f"{path}.{key[:64]}", depth + 1)
                        for key in keys[:key_cap]}
            return item

        bounded = preview(value)
        if not isinstance(bounded, dict):
            bounded = {"result_preview": bounded}
        bounded.update(preview_truncated=True, source_chars=len(raw),
                       preview_note="Bounded preview only. Full returned data is retained in the audit; "
                                    "all retrieved records remain available to Python.",
                       truncated_paths=truncated[:16])
        encoded = json.dumps(bounded, default=str)
        if len(encoded) <= MODEL_RETRIEVAL_CHARS:
            reply["content"] = encoded
            return reply
    reply["content"] = json.dumps({"preview_truncated": True, "source_chars": len(raw),
                                  "preview_note": "Result is too large to display. Use a focused Python inspection."})
    return reply


class Role:
    """One LLM role. Each run() starts a fresh context; usage, budget, and
    trace go to the host agent so the whole system answers to one ceiling."""

    def __init__(self, name, system_prompt, toolbox, host, max_turns=10, max_tool_calls=None):
        if max_tool_calls is not None and (isinstance(max_tool_calls, bool)
                or not isinstance(max_tool_calls, int) or max_tool_calls < 0):
            raise ValueError("max_tool_calls must be a nonnegative integer or None")
        self.name = name
        self.system_prompt = system_prompt
        self.toolbox = toolbox
        self.host = host
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls

    def run(self, content):
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content}]
        self.host.messages.extend(deepcopy(messages))
        tools = self.toolbox.definitions or None
        local_calls = 0
        for _ in range(self.max_turns):
            local_complete = self.max_tool_calls is not None and local_calls >= self.max_tool_calls
            if local_complete:
                final_task = self._final_task("This role's allocated tool calls are complete.")
                messages.append(final_task)
                self.host.messages.append(deepcopy(final_task))
            response, error, _ = self.host._llm_call(messages, tools=None if local_complete else tools)
            if error:
                return RoleResult(error=error)
            messages.append(response)
            self.host.messages.append(response)
            if not response.tool_calls:
                return RoleResult(content=response.content)
            if local_complete:
                return RoleResult(error=f"{self.name}: requested tools instead of the allocated final reply")
            for tool_call in response.tool_calls:
                if self.max_tool_calls is not None and local_calls >= self.max_tool_calls:
                    reply = {"tool_call_id": tool_call.id, "role": "tool", "name": tool_call.function.name,
                             "content": json.dumps({"error": "Not run: this role's tool allocation is complete. "
                                                        "Synthesize the final JSON from the available results."})}
                elif self.host._take_tool_call():
                    local_calls += 1
                    reply = self.toolbox.call(tool_call)
                else:
                    reply = self.host._tool_budget_message(tool_call)
                model_reply, audit_reply = _model_tool_reply(reply), deepcopy(reply)
                if model_reply.get("content") != audit_reply.get("content"):
                    audit_reply["model_content"] = model_reply["content"]
                messages.append(model_reply)
                self.host.messages.append(audit_reply)
        if self.max_turns < 1:
            return RoleResult(error=f"{self.name}: no final reply within {self.max_turns} turns")
        # The last tool result still needs a model reply. Give it one bounded
        # synthesis call under the host's existing token ceiling, with no tools.
        final_task = self._final_task("The tool turns are complete.")
        messages.append(final_task)
        self.host.messages.append(deepcopy(final_task))
        response, error, _ = self.host._llm_call(messages, tools=None)
        if error:
            return RoleResult(error=error)
        messages.append(response)
        self.host.messages.append(response)
        if response.tool_calls:
            return RoleResult(error=f"{self.name}: requested tools instead of a final reply after {self.max_turns} turns")
        return RoleResult(content=response.content)

    @staticmethod
    def _final_task(reason):
        return {"role": "user", "content": reason + (
            " Use the results already returned and reply now with the final JSON object required by your role. "
            "No more tools are available. If checks are incomplete, report that in the required response format; "
            "do not invent a successful result.")}
