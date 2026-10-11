"""Message passing for the Phase 2 multi-agent system (see docs/phase2_framework.md).

A thin layer over BaseAgent and tools/registry.py rather than an external
agent framework, so every role shares one question's budget and usage
accounting and generated code still runs only in tools/python_executor.py:

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

RETRIEVAL_TOOLS = ("get_resources_by_patient_fhir_id", "get_resources_by_resource_id")
PYTHON_TOOL = "execute_python_code"


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
        self.log = []

    def send(self, sender, recipient, kind, payload):
        if not isinstance(payload, dict):
            raise TypeError("A message payload must be an object")
        try:
            # The round trip both validates and copies the payload
            payload = json.loads(json.dumps(payload, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"A message payload must be a finite JSON value: {exc}") from None
        message = Message(sender, recipient, kind, payload)
        self.log.append(message)
        return message

    def to_list(self):
        return [m.to_dict() for m in self.log]


class Toolbox:
    """Allow-listed registry tools over one role's working records.

    `resources` is the role's working copy of retrieved records, by resource
    type; roles that should see the same records share the dict. `bound`
    maps a tool name to keyword arguments the system supplies (they replace
    any the model sends).
    """

    def __init__(self, names, resources=None, patient_fhir_id=None, bound=None):
        definitions = {d["function"]["name"]: d for d in get_tool_definitions()}
        missing = [name for name in names if name not in definitions]
        if missing:
            raise KeyError(f"Tools not in the registry: {', '.join(missing)}")
        self.names = tuple(names)
        self.definitions = [definitions[name] for name in self.names]
        self.resources = {} if resources is None else resources
        self.patient_fhir_id = patient_fhir_id
        self.bound = bound or {}

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
        if not isinstance(records, list):
            raise ValueError("The retrieval tool returned an invalid resource list")
        if name == "get_resources_by_resource_id":
            # A point fetch adds evidence; a patient-wide fetch replaces a partial packet.
            merged = {r["id"]: r for r in self.resources.get(resource_type, [])}
            merged.update({r["id"]: r for r in records})
            self.resources[resource_type] = list(merged.values())
        else:
            self.resources[resource_type] = records
        return json.dumps({"stored_as": f"retrieved_resources['{resource_type}']",
                           "ids": [f"{resource_type}/{r.get('id')}" for r in records],
                           "items": len(self.resources[resource_type])})

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
