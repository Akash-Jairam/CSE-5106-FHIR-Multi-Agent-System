import json

from .multi_turn_code_resource_agent import MultiTurnCodeResourceAgent
from .output_schema import ANSWER_FORMAT_INSTRUCTIONS, has_json_answer, parse_final_message
from tools import get_tool_definitions
from tools.field_check import check_fields
from tools.resource_tools import get_all_resources_by_patient_fhir_id
from tools.starting_packet import get_starting_packet
from utils import parse_question_context

OUTPUT_GUIDANCE = f"""

STARTING PACKET AND OUTPUT FORMAT (these override the 'The final answer is:' format above):
- The user message includes a starting packet: an excerpt of the patient's record chosen for this question. Its
  resources are already loaded in `retrieved_resources`. The packet may be incomplete; fetching a resource type
  replaces that entry in `retrieved_resources` with all of the patient's resources of that type.
- Resources dated after the assumed current time are not in the packet; ignore them when you fetch more.
- Before answering, you can verify your evidence with check_fields. It flags citations that do not match the
  record, the wrong time field (Observation `issued` instead of `effectiveDateTime`; MedicationRequest `authoredOn`
  instead of `dispenseRequest.validityPeriod.start`), unit mismatches, resources outside a scoped encounter, and
  resources after the current time.
- Your first final answer is checked automatically. If problems are found you get one chance to revise.

{ANSWER_FORMAT_INSTRUCTIONS}"""


class SingleAgent(MultiTurnCodeResourceAgent):
    """Tool-equipped single-agent baseline: the multi-turn code agent, started from the
    deterministic starting packet, with the field-check tool and one revision after an
    automatic check of its first answer. Returns the common output contract."""

    def __init__(self, model: str, max_iterations: int = 30, verbose: bool = False, base_url=None):
        super().__init__(model, max_iterations=max_iterations, verbose=verbose, base_url=base_url)
        self.tools = self.tools + [t for t in get_tool_definitions() if t["function"]["name"] == "check_fields"]
        self.system_msg = [{"role": "system", "content": self.system_msg[0]["content"] + OUTPUT_GUIDANCE}]

    def _handle_tool_call(self, tool_call, retrieved_resources):
        if tool_call.function.name != "check_fields":
            return super()._handle_tool_call(tool_call, retrieved_resources)
        args = json.loads(tool_call.function.arguments)
        result = self._check(args.get("citations"), retrieved_resources, args.get("encounter_id"))
        return {"tool_call_id": tool_call.id, "role": "tool", "name": "check_fields",
                "content": json.dumps({"issues": result["issues"], "n_valid": result["n_valid"]}, default=str)}

    def _check(self, citations, retrieved_resources, encounter_id=None):
        """Field check, resolving citations against what is already loaded before the store."""
        known = dict(self._record_by_id)
        for items in retrieved_resources.values():
            for r in items if isinstance(items, list) else []:
                if isinstance(r, dict) and "id" in r:
                    known[r["id"]] = r
        return check_fields(citations, question=self._question, current_time=self._current_time,
                            encounter_id=encounter_id, resources=known)

    def _is_final(self, content):
        # The JSON answer object, or the base agent's "The final answer is:" if the model falls back to it
        return bool(content) and (has_json_answer(content) or "the final answer is:" in content.lower())

    def _run(self, question: str) -> dict:
        self._question, patient_fhir_id, self._current_time = parse_question_context(question)
        if not patient_fhir_id:
            raise ValueError("The single agent needs the patient FHIR ID in the question context")

        # Same packet as the direct LLM baseline; its full resources are preloaded for code
        packet = get_starting_packet(self._question, patient_fhir_id, current_time=self._current_time)
        record = get_all_resources_by_patient_fhir_id(patient_fhir_id=patient_fhir_id)
        self._record_by_id = {r["id"]: r for items in record.values() if isinstance(items, list) for r in items}
        retrieved_resources = {rtype: [self._record_by_id[i] for i in ids if i in self._record_by_id]
                               for rtype, ids in packet["resource_ids"].items()}

        self.messages = self.system_msg.copy()
        self.messages.append({"role": "user", "content": f"{question}\n\n{packet['text']}"})
        draft, final, check_issues = None, None, None
        for _ in range(self.max_iterations):
            response_message, error, usage_info = self._llm_call(self.messages, tools=self.tools)
            if error:
                if draft is not None and self.budget_stop:
                    break  # stopped by the token ceiling during the revision: keep the checked draft
                return {"answer": None, "error": error, "trace": self.messages,
                        "retrieved_fhir_resources": retrieved_resources,
                        "revision": self._revision(draft, check_issues, final)}
            self.messages.append(response_message)

            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    if not self._take_tool_call():
                        self.messages.append(self._tool_budget_message(tool_call))
                        continue
                    tool_message = self._handle_tool_call(tool_call, retrieved_resources)
                    if tool_message is None:
                        tool_message = {"tool_call_id": tool_call.id, "role": "tool", "name": tool_call.function.name,
                                        "content": f"Unknown tool '{tool_call.function.name}'"}
                    self.messages.append(tool_message)
                continue

            if not self._is_final(response_message.content):
                continue
            answer, evidence, limitation = parse_final_message(response_message.content)
            if draft is not None:
                final = (answer, evidence, limitation)
                break
            draft = (answer, evidence, limitation)
            check_issues = self._check(evidence, retrieved_resources)["issues"]
            if answer is not None and not evidence:
                check_issues.append("no evidence cited for a non-null answer")
            if not check_issues:
                final = draft
                break
            # One revision: report the problems and let the agent continue (with tools) once more
            self.messages.append({"role": "user", "content": (
                "The field check found problems with your answer's evidence:\n- " + "\n- ".join(check_issues) +
                "\nYou may revise once: fix the evidence or the answer if needed (you can use the tools), "
                "then reply with the final JSON object. This will be your final answer.")})

        if final is None:
            final = draft if draft is not None else (None, [], "No final answer within the iteration limit")
        answer, evidence, limitation = final
        if self.budget_stop and final is draft and check_issues:
            limitation = limitation or f"Not revised: stopped by the {self.budget_stop}"
        return {
            "answer": answer,
            "evidence": evidence,
            "limitation": limitation,
            "trace": self.messages,
            "retrieved_fhir_resources": retrieved_resources,
            "revision": self._revision(draft, check_issues, final),
        }

    @staticmethod
    def _revision(draft, check_issues, final):
        if draft is None:
            return None
        as_dict = lambda t: {"answer": t[0], "evidence": t[1], "limitation": t[2]}
        return {
            "draft": as_dict(draft),
            "check_issues": check_issues,
            "revised": bool(check_issues) and final is not None and final is not draft,
            "answer_changed": final is not None and final[0] != draft[0],
        }
