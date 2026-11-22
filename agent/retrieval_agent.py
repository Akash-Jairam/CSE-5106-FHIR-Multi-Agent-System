"""Retrieval role (P2-13) for the fixed controller.

Starts from the same deterministic starting packet as the single agent. The
model then chooses which resource types to fetch and resolves the question's
patient and encounter scope. The role only reads: its tools are the two
patient/ID searches, and its Toolbox leaves out other patients' records and
records dated after the assumed current time.

Records go into the shared workspace, which the analysis role sees as
`retrieved_resources`. The result message carries their IDs and the scope:

    {"record_ids": {"Observation": ["Observation/o1", ...], ...},
     "resource_types": ["Observation", ...],
     "scope": {"patient_fhir_id": ..., "current_time": ..., "encounter_ids": [...]},
     "limitation": null | "..."}

With `prefetch`, the role first fetches the patient's complete records of
every type the packet or the question points to (Encounters first, so
undated Conditions resolve), then the Medications and Specimens those
records reference, before the model looks for anything else. With `direct`,
it also reads the packet once without tools, as the direct baseline does,
and reports that answer as `direct` for the controller's release vote.
"""
import json
import re
from copy import deepcopy

from .analysis_agent import _is_citation, _valid_answer
from .direct_llm_agent import SYSTEM_PROMPT as DIRECT_PROMPT
from .framework import RETRIEVAL_TOOLS, Role, Toolbox
from .output_schema import _json_objects, has_json_answer, parse_final_message
from tools.perturbation import get_perturbed_starting_packet
from tools.resource_tools import FHIR_ID_RE, supported_types
from tools.starting_packet import get_starting_packet

SYSTEM_PROMPT = f"""You are the retrieval agent in a team answering questions about one patient's FHIR record.
Your job is only to gather the records needed to answer the question; another agent will analyze them.

You start from a starting packet: an excerpt of the patient's record chosen for this question. It may be
incomplete. Use the tools to fetch what is missing:
- get_resources_by_patient_fhir_id fetches the patient's resources of one type and merges them into the workspace.
- get_resources_by_resource_id fetches one resource by ID and adds it.
Available resource types: {', '.join(supported_types)}.
Only the question's patient can be retrieved, and records dated after the assumed current time are left out.
Medication and Location metadata must be referenced by an already retrieved record. Fetch Encounters before
undated Conditions so their dates can be established. Excluded records are counted in the tool response;
unknown patient/time is not proof of an empty result. Encounter responses include fields for resolving scope.

Resolve the question's scope:
- Hospital visits are Encounters whose identifier system ends in "encounter-hosp"; ICU stays end in
  "encounter-icu" and link to their hospital visit through partOf. Order visits by period.start.
- If the question is about specific visits (e.g. "first hospital visit", "last ICU stay"), report their IDs.

Do not compute the answer. When the workspace holds what is needed, reply with only a JSON object:
{{"resource_types": [<types relevant to the question>], "encounter_ids": [<IDs of the encounters the question is
scoped to; empty if it is not about specific visits>], "limitation": <null, or what could not be found>}}"""


# Record types a question's wording points to, besides those in its starting packet
TYPE_KEYWORDS = (
    ("MedicationRequest", re.compile(r"drug|medic|prescri|dose|dosage|route|tablet|capsule|inject|insulin|antibiotic", re.I)),
    ("MedicationAdministration", re.compile(r"administ|infus|intake|input", re.I)),
    ("Procedure", re.compile(r"procedure|surg|operat|cathet|ventilat|bypass|excision|insertion|biopsy|transfusion", re.I)),
    ("Condition", re.compile(r"diagnos|condition|disease|disorder", re.I)),
    ("Observation", re.compile(r"\blab|test|specimen|organism|cultur|microbio|vital|measure|level|weight|height|"
                               r"heart rate|blood pressure|temperature|saturation|respiratory|glucose|output|bmi", re.I)),
)
EVENT_TYPES = ("Condition", "MedicationRequest", "MedicationAdministration", "Procedure", "Observation", "Specimen")
PREFETCH_LIMIT = 8
TOOL_RESERVE = 12   # tool calls left for analysis, verification and the finishers


class RetrievalAgent:
    """Handler for the controller's retrieval tasks; `workspace` is shared with the analysis role."""

    def __init__(self, host, context, workspace, max_turns=8, prefetch=False, direct=False):
        self.host = host
        self.context = context
        self.workspace = workspace
        self.prefetch = prefetch
        self.direct = direct
        self.direct_candidate = None
        self.prefetched = []
        self.packet = None
        self.perturbation = getattr(host, "perturbation", None)
        toolbox = Toolbox(RETRIEVAL_TOOLS, workspace, patient_fhir_id=context.get("patient_fhir_id"),
                          current_time=context.get("current_time"))
        self.role = Role("retrieval", SYSTEM_PROMPT, toolbox, host, max_turns=max_turns)

    def __call__(self, task):
        patient_fhir_id = self.context.get("patient_fhir_id")
        if not patient_fhir_id:
            return {"error": "The retrieval agent needs the patient FHIR ID in the question context"}
        if self.packet is None:
            if self.perturbation is not None:
                self.packet = get_perturbed_starting_packet(self.context["question"], patient_fhir_id,
                                                            self.perturbation,
                                                            current_time=self.context.get("current_time"))
            else:
                self.packet = get_starting_packet(self.context["question"], patient_fhir_id,
                                                  current_time=self.context.get("current_time"))
            for rtype, items in self.packet["resources"].items():
                self.workspace[rtype] = [{**deepcopy(r), "resourceType": rtype} for r in items]
            if self.direct:
                self.direct_candidate = self._read_packet()
            if self.prefetch:
                self._prefetch()
        result = self.role.run(self._prompt(task))
        if result.error:
            if not self.prefetched:
                return {"error": result.error}
            # The prefetched records are complete for their types; analysis can still use them
            return self._result("", f"The retrieval agent stopped ({result.error})")
        return self._result(result.content)

    def _read_packet(self):
        """The direct baseline's answer from the packet alone: one model call without tools."""
        question = f"{self.context['question']}\nContext: {self.context.get('assumptions', '')}"
        messages = [{"role": "system", "content": DIRECT_PROMPT},
                    {"role": "user", "content": f"{question}\n\n{self.packet['text']}"}]
        self.host.messages.extend(deepcopy(messages))
        response, error, _ = self.host._llm_call(messages)
        if error or response is None:
            return None
        self.host.messages.append(response)
        if not has_json_answer(response.content):
            return None
        answer, evidence, limitation = parse_final_message(response.content)
        if not _valid_answer(answer):
            return None
        return {"answer": answer, "evidence": [c for c in evidence or [] if _is_citation(c)],
                "limitation": limitation if isinstance(limitation, str) else None}

    def _wanted_types(self):
        question = self.context["question"]
        types = ["Encounter"]
        types += [t for t in self.packet["resources"] if t in EVENT_TYPES and t != "Specimen"]
        types += [t for t, pattern in TYPE_KEYWORDS if pattern.search(question)]
        return list(dict.fromkeys(types))

    def _prefetch(self):
        """Fetch every wanted type in full, then the metadata those records reference."""
        for rtype in self._wanted_types():
            self._fetch(rtype)
        held = lambda t: self.workspace.get(t) or []
        if any(r.get("medicationReference") for r in held("MedicationRequest") + held("MedicationAdministration")):
            self._fetch("Medication")
        if any(r.get("specimen") for r in held("Observation")):
            self._fetch("Specimen")

    def _fetch(self, rtype):
        host = self.host
        if len(self.prefetched) >= PREFETCH_LIMIT or (
                host.max_tool_calls is not None and host.max_tool_calls - host.tool_calls_made <= TOOL_RESERVE):
            return
        if not host._take_tool_call():
            return
        self.prefetched.append(rtype)
        tool_id = f"retrieval_prefetch_{len(self.prefetched)}"
        arguments = {"resource_type": rtype, "patient_fhir_id": self.context["patient_fhir_id"]}
        host.messages.append({"role": "assistant", "content": None, "tool_calls": [{
            "id": tool_id, "type": "function",
            "function": {"name": "get_resources_by_patient_fhir_id", "arguments": json.dumps(arguments)}}]})
        try:
            content = self.role.toolbox._retrieve("get_resources_by_patient_fhir_id", arguments)
        except Exception as exc:
            content = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
        host.messages.append({"role": "tool", "tool_call_id": tool_id,
                              "name": "get_resources_by_patient_fhir_id", "content": content})

    def _prompt(self, task):
        parts = [f"Question: {self.context['question']}", f"Context: {self.context.get('assumptions', '')}",
                 "", self.packet["text"]]
        if self.prefetched:
            held = {rtype: len(self.workspace.get(rtype) or []) for rtype in self.prefetched}
            parts += ["", "The patient's complete records of these types are already in the workspace "
                          f"(records per type): {json.dumps(held)}. Fetch only other types the question needs."]
        if task.get("issues"):
            held = {rtype: len(items) for rtype, items in self.workspace.items() if items}
            parts += ["", "A verifier found problems with the answer built from these records:",
                      json.dumps(task["issues"], indent=1),
                      f"The workspace now holds: {json.dumps(held)}.",
                      "Fetch what is missing or out of scope, then reply with the JSON object again."]
        return "\n".join(parts)

    def _result(self, content, note=None):
        held = {rtype: items for rtype, items in self.workspace.items() if items}
        summary = next((obj for obj in _json_objects(content or "")
                        if {"resource_types", "encounter_ids"} & set(obj)), None)
        notes = [note] if note else []
        if summary is None:
            summary = {}
            notes.append("The retrieval agent did not report the question's scope")
        own = summary.get("limitation")
        if own is not None and not isinstance(own, str):
            return {"error": "retrieval limitation must be a string or null"}
        types = summary.get("resource_types")
        if types is not None and (not isinstance(types, list)
                                  or not all(isinstance(t, str) and t in supported_types for t in types)):
            return {"error": "resource_types must be a list of supported resource types"}
        types = [t for t in types if t in held] if isinstance(types, list) else sorted(held)
        encounters = {r.get("id") for r in held.get("Encounter", [])}
        requested = summary.get("encounter_ids")
        if requested is not None and (not isinstance(requested, list) or not all(
                isinstance(e, str) and FHIR_ID_RE.fullmatch(e.removeprefix("Encounter/")) for e in requested)):
            return {"error": "encounter_ids must be a list of Encounter IDs"}
        requested = [str(e).rsplit("/", 1)[-1] for e in requested] if isinstance(requested, list) else []
        unknown = [e for e in requested if e not in encounters]
        if unknown:
            notes.append(f"Encounter scope not among the retrieved encounters: {', '.join(unknown)}")
        unresolved = self.role.toolbox.scope_failures
        if unresolved:
            notes.append("Patient/time scope could not be established for: " + ", ".join(sorted(unresolved)))
        limitation = ". ".join(([own] if own else []) + notes) or None
        return {
            "record_ids": {rtype: [f"{rtype}/{r.get('id')}" for r in items] for rtype, items in held.items()},
            "resource_types": types,
            "scope": {"patient_fhir_id": self.context.get("patient_fhir_id"),
                      "current_time": self.context.get("current_time"),
                      "encounter_ids": [e for e in requested if e in encounters]},
            "limitation": limitation,
            **({"direct": self.direct_candidate} if self.direct else {}),
        }
