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
"""
import json
from copy import deepcopy

from .framework import RETRIEVAL_TOOLS, Role, Toolbox
from .output_schema import _json_objects
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


class RetrievalAgent:
    """Handler for the controller's retrieval tasks; `workspace` is shared with the analysis role."""

    def __init__(self, host, context, workspace, max_turns=8):
        self.context = context
        self.workspace = workspace
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
        result = self.role.run(self._prompt(task))
        if result.error:
            return {"error": result.error}
        return self._result(result.content)

    def _prompt(self, task):
        parts = [f"Question: {self.context['question']}", f"Context: {self.context.get('assumptions', '')}",
                 "", self.packet["text"]]
        if task.get("issues"):
            held = {rtype: len(items) for rtype, items in self.workspace.items() if items}
            parts += ["", "A verifier found problems with the answer built from these records:",
                      json.dumps(task["issues"], indent=1),
                      f"The workspace now holds: {json.dumps(held)}.",
                      "Fetch what is missing or out of scope, then reply with the JSON object again."]
        return "\n".join(parts)

    def _result(self, content):
        held = {rtype: items for rtype, items in self.workspace.items() if items}
        summary = next((obj for obj in _json_objects(content or "")
                        if {"resource_types", "encounter_ids"} & set(obj)), None)
        notes = []
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
        }
