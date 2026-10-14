import ast
import hashlib
import json
from copy import deepcopy

from .multi_turn_code_resource_agent import MultiTurnCodeResourceAgent
from .analysis_agent import AnalysisAgent
from .framework import RETRIEVAL_TOOLS, _model_tool_reply
from .ehr_semantics import SEMANTIC_GUIDANCE
from .output_schema import ANSWER_FORMAT_INSTRUCTIONS, has_json_answer, parse_final_message
from tools import get_tool_definitions
from tools.field_check import check_fields
from tools.perturbation import get_perturbed_starting_packet
from tools.starting_packet import get_starting_packet
from tools.resource_tools import supported_types
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
- Python tool replies may be bounded previews. All retrieved resources remain in the Python workspace.
  Use mode="inspect" for small diagnostic results. For the final calculation call execute_python_code
  with mode="final" and set all of these variables in that same worker execution:
    answer = <actual scalar, list of answer values, or null>
    evidence = [{{"resource_id":"<ResourceType>/<id>", "field_path":"<path>", "value":<exact source value>}}]
    contributing_ids = ["<ResourceType>/<id>", ...]  # scoped inputs and cited dependencies
    operation = "<filters, ordering, calculation and fields used>"
    limitation = None  # or explanation of missing evidence
  The executed result becomes the candidate answer; your final prose cannot replace it or drop evidence.
  Fetching more records invalidates that result, so rerun the final calculation after a fetch.

{SEMANTIC_GUIDANCE}

{ANSWER_FORMAT_INSTRUCTIONS}"""

SINGLE_SYSTEM_PROMPT = f"""You answer questions about one patient's FHIR records using one conversation and tools.
Use get_resources_by_patient_fhir_id to fetch required resource types, get_resources_by_resource_id for linked
records, execute_python_code to inspect/filter/order/calculate, and check_fields to verify exact source citations.
Available resource types: {', '.join(supported_types)}. Retrieve only the patient in the question. Fetch the
patient-owned event records before linked Medication or Location metadata. Do not infer missing details from
resource IDs or assume the starting packet is the complete record.

The Python workspace is a dictionary already supplied as retrieved_resources; it is not an importable module.
Each Python invocation starts a fresh worker, so assign every needed local variable in that invocation.
For mode="inspect", prefer assigning the actual diagnostic data to answer. Printed stdout also supports
diagnostics, with an honest 64 KiB character cap and truncation flags; it never certifies a final answer.
Bare expressions are not returned. Avoid answer=None when you want a structured diagnostic result.
Keep inspection samples small; full records stay in the workspace.
Inspection example (mode="inspect"):
rows = retrieved_resources.get('Observation', [])
answer = {{'count': len(rows), 'sample': rows[:3],
          'labels': sorted({{str(c.get('display') or c.get('code') or '') for r in rows
                            for c in r.get('code', {{}}).get('coding', [])}})[:30]}}

For a final calculation, execute the complete scope/filter/calculation in one call with mode="final" and set
all five result variables: answer, evidence, contributing_ids, operation, limitation. Citations require exactly
resource_id, field_path, value. Use full ResourceType/id strings, one concrete dot/index field path, and the exact
source value. Do not use resource/field/fields aliases, bare IDs, prose paths, or omit citation values.
All cited resources must already be retrieved and included in the unique contributing_ids list. Fetch linked
metadata first if you cite it. A nonempty answer needs evidence. An established zero, false, null, or [] can have
empty evidence, with an operation explaining the completed search; unavailable data does not establish emptiness.
Final schema example for a simple patient gender lookup (mode="final"):
patient = retrieved_resources['Patient'][0]
answer = patient['gender']
evidence = [{{'resource_id': 'Patient/' + patient['id'], 'field_path': 'gender', 'value': patient['gender']}}]
contributing_ids = ['Patient/' + patient['id']]
operation = 'Read gender from the question patient record'
limitation = None
For numerical or temporal questions, replace the lookup with the question's complete filtering, ordering and
calculation; cite actual input fields and include every scoped input and cited dependency. A tool contract error
is recoverable: correct the code and run a new final calculation rather than declaring the records inaccessible.
Then return the final JSON object promptly. The executed artifact supplies the candidate; source checks can request
one semantic revision. Preserve exact source values in evidence even when the answer converts units or formats a date.

{OUTPUT_GUIDANCE}"""


class SingleAgent(MultiTurnCodeResourceAgent):
    """Tool-equipped single-agent baseline: the multi-turn code agent, started from the
    deterministic starting packet, with the field-check tool and one revision after an
    automatic check of its first answer. Returns the common output contract."""

    def __init__(self, model: str, max_iterations: int = 30, verbose: bool = False, base_url=None):
        super().__init__(model, max_iterations=max_iterations, verbose=verbose, base_url=base_url)
        self.tools = self.tools + [t for t in get_tool_definitions() if t["function"]["name"] == "check_fields"]
        self.system_msg = [{"role": "system", "content": SINGLE_SYSTEM_PROMPT}]
        self.final_computation = None
        self.last_execution = None
        self._requires_final_computation = False
        self._final_computation_error = None

    @staticmethod
    def _workspace_sha256(resources):
        return hashlib.sha256(json.dumps(resources, sort_keys=True, default=str,
                                         allow_nan=False).encode()).hexdigest()

    def _invalidate_computation(self, reason):
        self.final_computation = None
        self.last_execution = None
        self._final_computation_error = reason

    def _computed_candidate(self, resources):
        if self.final_computation is None:
            return None
        if self.final_computation["workspace_sha256"] != self._workspace_sha256(resources):
            self._invalidate_computation("The retrieved workspace changed after the final computation")
            return None
        computed = {key: self.final_computation.get(key) for key in
                    ("answer", "evidence", "operation", "contributing_ids", "limitation")}
        return AnalysisAgent._draft(json.dumps(computed), resources)

    def _handle_tool_call(self, tool_call, retrieved_resources):
        name = tool_call.function.name
        if name in RETRIEVAL_TOOLS:
            self._invalidate_computation("Retrieval requires a new final computation")
        mode, workspace, code = "auto", None, None
        if name == "execute_python_code":
            try:
                arguments = json.loads(tool_call.function.arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Arguments must be a JSON object")
                mode, code = arguments.get("mode", "auto"), arguments.get("code")
            except (ValueError, TypeError) as exc:
                self._invalidate_computation("The Python tool arguments were invalid")
                return self._tool_error(tool_call, str(exc))
            workspace = self._workspace_sha256(retrieved_resources)
            if self.final_computation and self.final_computation["workspace_sha256"] != workspace:
                self._invalidate_computation("The retrieved workspace changed after the final computation")
            if mode != "inspect":
                self._invalidate_computation("A new Python calculation did not produce a valid final artifact")
            if mode == "final":
                self._requires_final_computation = True
        if tool_call.function.name != "check_fields":
            if tool_call.function.name == "get_resources_by_patient_fhir_id":
                try:
                    args = json.loads(tool_call.function.arguments)
                except (ValueError, TypeError):
                    return self._tool_error(tool_call, "Arguments must be a JSON object")
                if not isinstance(args, dict) or args.get("patient_fhir_id") != self._patient_fhir_id:
                    return self._tool_error(tool_call, "Retrieve only the patient identified in this question")
            reply = super()._handle_tool_call(tool_call, retrieved_resources)
            if reply and tool_call.function.name == "execute_python_code":
                # The inherited dispatcher serializes Python dictionaries with repr().
                try:
                    reply["content"] = json.dumps(ast.literal_eval(reply["content"]), default=str)
                except (ValueError, SyntaxError, TypeError):
                    pass
                try:
                    output = json.loads(reply["content"])
                except (ValueError, TypeError):
                    output = None
                if isinstance(output, dict) and "answer" in output and not output.get("error"):
                    self.last_execution = deepcopy({**output, "mode": mode, "code": code,
                                                    "workspace_sha256": workspace})
                    if mode == "final":
                        computed = {key: output.get(key) for key in
                                    ("answer", "evidence", "operation", "contributing_ids", "limitation")}
                        if (isinstance(computed["evidence"], list) and isinstance(computed["contributing_ids"], list)
                                and all(isinstance(item, str) for item in computed["contributing_ids"])):
                            computed["contributing_ids"] = list(dict.fromkeys([
                                *computed["contributing_ids"], *(citation.get("resource_id")
                                for citation in computed["evidence"] if isinstance(citation, dict))]))
                        candidate = AnalysisAgent._draft(json.dumps(computed), retrieved_resources)
                        if candidate.get("error"):
                            self._invalidate_computation(candidate["error"])
                            return self._tool_error(tool_call, "Invalid final computation: " + candidate["error"])
                        self.final_computation = deepcopy({**self.last_execution, **candidate})
                        self._final_computation_error = None
                elif mode == "final" and isinstance(output, dict):
                    self._final_computation_error = output.get("error") or "The final worker returned no answer"
            return reply
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
        self._append_message({"role": "assistant", "content": None, "tool_calls": [{
            "id": tool_id, "type": "function", "function": {
                "name": "check_fields", "arguments": json.dumps(args)}}]})
        result = self._check(output[1])
        established_empty = (self.final_computation is not None
                             and self.final_computation.get("answer") == output[0]
                             and (output[0] is False or output[0] == 0 or output[0] == []))
        if output[0] is not None and not output[1] and not established_empty:
            result["issues"].append("no evidence cited for a non-null answer")
        self._append_message({"role": "tool", "tool_call_id": tool_id, "name": "check_fields",
                              "content": json.dumps(result, default=str)})
        return result["issues"]

    def _is_final(self, content):
        # The JSON answer object, or the base agent's "The final answer is:" if the model falls back to it
        return bool(content) and (has_json_answer(content) or "the final answer is:" in content.lower())

    def _append_message(self, message):
        """Separate bounded model observations from the complete saved audit trace."""
        model_message = _model_tool_reply(message) if isinstance(message, dict) and message.get("role") == "tool" else message
        audit_message = deepcopy(message)
        if isinstance(model_message, dict) and model_message.get("content") != message.get("content"):
            audit_message["model_content"] = model_message["content"]
        self.messages.append(audit_message)
        self.model_messages.append(model_message)

    def _run(self, question: str) -> dict:
        self._question, patient_fhir_id, self._current_time = parse_question_context(question)
        if not patient_fhir_id:
            raise ValueError("The single agent needs the patient FHIR ID in the question context")
        self._patient_fhir_id = patient_fhir_id
        self.final_computation, self.last_execution = None, None
        self._requires_final_computation, self._final_computation_error = False, None

        # Code starts with the same fields the direct baseline can read in the packet.
        if self.perturbation is not None:
            packet = get_perturbed_starting_packet(self._question, patient_fhir_id, self.perturbation,
                                                   current_time=self._current_time)
        else:
            packet = get_starting_packet(self._question, patient_fhir_id, current_time=self._current_time)
        retrieved_resources = {rtype: [{**deepcopy(r), "resourceType": rtype} for r in items]
                               for rtype, items in packet["resources"].items()}

        self.messages = self.system_msg.copy()
        self.messages.append({"role": "user", "content": f"{question}\n\n{packet['text']}"})
        self.model_messages = deepcopy(self.messages)
        draft, final, check_issues, final_check_issues = None, None, None, None
        revision_error = None
        for _ in range(self.max_iterations):
            response_message, error, usage_info = self._llm_call(self.model_messages, tools=self.tools)
            if error:
                if draft is not None:
                    revision_error = error
                    break
                return {"answer": None, "error": error, "trace": self.messages,
                        "retrieved_fhir_resources": retrieved_resources,
                        "revision": self._revision(draft, check_issues, final)}
            self._append_message(response_message)

            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    if not self._take_tool_call():
                        self._append_message(self._tool_budget_message(tool_call))
                        continue
                    tool_message = self._handle_tool_call(tool_call, retrieved_resources)
                    if tool_message is None:
                        tool_message = {"tool_call_id": tool_call.id, "role": "tool", "name": tool_call.function.name,
                                        "content": f"Unknown tool '{tool_call.function.name}'"}
                    self._append_message(tool_message)
                continue

            candidate = self._computed_candidate(retrieved_resources)
            if candidate and candidate.get("error"):
                self._invalidate_computation(candidate["error"])
                candidate = None
            if candidate is None and self._requires_final_computation:
                self._append_message({"role": "user", "content": (
                    "No valid final computation is available: " + (self._final_computation_error or "missing artifact") +
                    ". Correct the code and call execute_python_code with mode=final, setting answer, evidence, "
                    "contributing_ids, operation, and limitation. Then give the final JSON object.")})
                continue
            if candidate is None and not self._is_final(response_message.content):
                continue
            if candidate is not None:
                answer, evidence, limitation = candidate["answer"], candidate["evidence"], candidate["limitation"]
            else:
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
            self._invalidate_computation("The evidence check requested a revised final computation")
            # One revision: report the problems and let the agent continue (with tools) once more
            self._append_message({"role": "user", "content": (
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

    def _revision(self, draft, check_issues, final, final_check_issues=None, revision_error=None):
        if draft is None:
            return {"context_compactions": deepcopy(self.context_compactions)} if self.context_compactions else None
        as_dict = lambda t: {"answer": t[0], "evidence": t[1], "limitation": t[2]}
        return {
            "draft": as_dict(draft),
            "check_issues": check_issues,
            "revised": bool(check_issues) and final is not None and final is not draft,
            "answer_changed": final is not None and final[0] != draft[0],
            "final": as_dict(final) if final is not None else None,
            "final_check_issues": final_check_issues,
            "revision_error": revision_error,
            "context_compactions": deepcopy(self.context_compactions),
        }
