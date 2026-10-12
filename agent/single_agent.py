import json
from copy import deepcopy

from .multi_turn_code_resource_agent import MultiTurnCodeResourceAgent
from .output_schema import ANSWER_FORMAT_INSTRUCTIONS, has_json_answer, parse_final_message
from tools import get_tool_definitions
from tools.field_check import check_fields
from tools.starting_packet import get_starting_packet
from utils import parse_question_context

OUTPUT_GUIDANCE = f"""

STARTING PACKET AND OUTPUT FORMAT (these override the 'The final answer is:' format above):
- The user message includes a starting packet: an excerpt of the patient's record chosen for this question. Its
  visible fields are already loaded in `retrieved_resources`. The packet may be incomplete; fetching a resource type
  replaces that entry in `retrieved_resources` with all of the patient's resources of that type.
- Resources dated after the assumed current time are not in the packet; ignore them when you fetch more.
- Before answering, you can verify your evidence with check_fields. It flags citations that do not match the
  record, the wrong time field (Observation `issued` instead of `effectiveDateTime`; MedicationRequest `authoredOn`
  instead of `dispenseRequest.validityPeriod.start`), unit mismatches, resources outside a scoped encounter,
  resources outside the question's time window, and resources after the current time.
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
            if tool_call.function.name == "get_resources_by_patient_fhir_id":
                try:
                    args = json.loads(tool_call.function.arguments)
                except (ValueError, TypeError):
                    return self._tool_error(tool_call, "Arguments must be a JSON object")
                if not isinstance(args, dict) or args.get("patient_fhir_id") != self._patient_fhir_id:
                    return self._tool_error(tool_call, "Retrieve only the patient identified in this question")
            return super()._handle_tool_call(tool_call, retrieved_resources)
        try:
            args = json.loads(tool_call.function.arguments)
            if not isinstance(args, dict) or not isinstance(args.get("citations"), list):
                raise ValueError("citations must be a list")
        except (ValueError, TypeError) as exc:
            return self._tool_error(tool_call, str(exc))
        result = self._check(args["citations"], args.get("encounter_id"))
        return {"tool_call_id": tool_call.id, "role": "tool", "name": "check_fields",
                "content": json.dumps(result, default=str)}

    def _check(self, citations, encounter_id=None):
        """Re-fetch original fields; model-generated code can mutate its working copy."""
        try:
            return check_fields(citations, question=self._question, current_time=self._current_time,
                                encounter_id=encounter_id, patient_fhir_id=self._patient_fhir_id)
        except Exception as exc:
            return {"issues": [f"source_check: could not verify the original records ({exc})"], "n_valid": 0}

    def _automatic_check(self, output, index):
        if not self._take_tool_call():
            return ["source_check: the tool-call budget did not allow the final evidence check"]
        tool_id = f"automatic_field_check_{index}"
        args = {"citations": output[1]}
        self.messages.append({"role": "assistant", "content": None, "tool_calls": [{
            "id": tool_id, "type": "function", "function": {
                "name": "check_fields", "arguments": json.dumps(args)}}]})
        result = self._check(output[1])
        if output[0] is not None and not output[1]:
            result["issues"].append("no evidence cited for a non-null answer")
        self.messages.append({"role": "tool", "tool_call_id": tool_id, "name": "check_fields",
                              "content": json.dumps(result, default=str)})
        return result["issues"]

    def _is_final(self, content):
        # The JSON answer object, or the base agent's "The final answer is:" if the model falls back to it
        return bool(content) and (has_json_answer(content) or "the final answer is:" in content.lower())

    def _run(self, question: str) -> dict:
        self._question, patient_fhir_id, self._current_time = parse_question_context(question)
        if not patient_fhir_id:
            raise ValueError("The single agent needs the patient FHIR ID in the question context")
        self._patient_fhir_id = patient_fhir_id

        # Code starts with the same fields the direct baseline can read in the packet.
        packet = get_starting_packet(self._question, patient_fhir_id, current_time=self._current_time)
        retrieved_resources = {rtype: [{**deepcopy(r), "resourceType": rtype} for r in items]
                               for rtype, items in packet["resources"].items()}

        self.messages = self.system_msg.copy()
        self.messages.append({"role": "user", "content": f"{question}\n\n{packet['text']}"})
        draft, final, check_issues, final_check_issues = None, None, None, None
        revision_error = None
        for _ in range(self.max_iterations):
            response_message, error, usage_info = self._llm_call(self.messages, tools=self.tools)
            if error:
                if draft is not None:
                    revision_error = error
                    break
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
                final_check_issues = self._automatic_check(final, 2)
                break
            draft = (answer, evidence, limitation)
            check_issues = self._automatic_check(draft, 1)
            if not check_issues:
                final = draft
                final_check_issues = check_issues
                break
            if self.budget_stop:
                break
            # One revision: report the problems and let the agent continue (with tools) once more
            self.messages.append({"role": "user", "content": (
                "The field check found problems with your answer's evidence:\n- " + "\n- ".join(check_issues) +
                "\nYou may revise once: fix the evidence or the answer if needed (you can use the tools), "
                "then reply with the final JSON object. This will be your final answer.")})

        if final is None:
            final = draft if draft is not None else (None, [], "No final answer within the iteration limit")
            final_check_issues = check_issues
        answer, evidence, limitation = final
        if final_check_issues:
            reason = "Evidence could not be verified: " + "; ".join(final_check_issues)
            if revision_error:
                reason += f". Revision stopped: {revision_error}"
            elif final is draft:
                reason += ". No corrected final answer within the available budget or iteration limit"
            answer = None
            limitation = f"{limitation}. {reason}" if limitation else reason
        return {
            "answer": answer,
            "evidence": evidence,
            "limitation": limitation,
            "trace": self.messages,
            "retrieved_fhir_resources": retrieved_resources,
            "revision": self._revision(draft, check_issues, final, final_check_issues, revision_error),
            "error": revision_error if revision_error and not self.budget_stop else None,
        }

    @staticmethod
    def _revision(draft, check_issues, final, final_check_issues=None, revision_error=None):
        if draft is None:
            return None
        as_dict = lambda t: {"answer": t[0], "evidence": t[1], "limitation": t[2]}
        return {
            "draft": as_dict(draft),
            "check_issues": check_issues,
            "revised": bool(check_issues) and final is not None and final is not draft,
            "answer_changed": final is not None and final[0] != draft[0],
            "final": as_dict(final) if final is not None else None,
            "final_check_issues": final_check_issues,
            "revision_error": revision_error,
        }
