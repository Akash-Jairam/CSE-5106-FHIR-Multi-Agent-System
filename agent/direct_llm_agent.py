from .base_agent import BaseAgent
from .ehr_semantics import CONVENTION_GUIDANCE
from .output_schema import ANSWER_FORMAT_INSTRUCTIONS, has_json_answer, parse_final_message
from tools.perturbation import get_perturbed_starting_packet
from tools.starting_packet import get_starting_packet
from utils import parse_question_context


class DirectLLMAgent(BaseAgent):
    """Zero-shot baseline: one LLM call with the question and the deterministic
    starting packet (tools/starting_packet.py). No tools, no further retrieval."""

    def __init__(self, model: str, verbose: bool = False, base_url=None):
        super().__init__(model, verbose, base_url)
        self.system_msg = {"role": "system", "content": f"""You are a helpful assistant that answers questions about one patient's FHIR record.

You are given an excerpt of the record (the "starting packet"): the patient, their hospital and ICU encounters, and the resources that best match the question. You cannot retrieve anything else, so answer from the excerpt alone.
- Use the assumed current time in the context to resolve relative dates ("this year", "last hospital visit").
- Times in the record carry UTC offsets; report times as they appear (local wall-clock time), without the offset.
- If the excerpt does not contain what is needed, set "answer" to null and explain in "limitation". Do not guess.
{CONVENTION_GUIDANCE}
{ANSWER_FORMAT_INSTRUCTIONS}"""}

    def _run(self, question: str) -> dict:
        bare_question, patient_fhir_id, current_time = parse_question_context(question)
        if not patient_fhir_id:
            raise ValueError("The direct LLM agent needs the patient FHIR ID in the question context")
        # Rank the packet on the bare question (with the assumed current time as the cutoff)
        # so it matches manifests/starting_packets.jsonl
        if self.perturbation is not None:
            packet = get_perturbed_starting_packet(bare_question, patient_fhir_id, self.perturbation,
                                                   current_time=current_time)
        else:
            packet = get_starting_packet(bare_question, patient_fhir_id, current_time=current_time)
        retrieved = {}
        for rtype, ids in packet["resource_ids"].items():
            retrieved[rtype] = [{"id": i} for i in ids]

        self.messages = [self.system_msg,
                         {"role": "user", "content": f"{question}\n\n{packet['text']}"}]
        response_message, error, usage_info = self._llm_call(self.messages)
        if error:
            return {"answer": None, "limitation": None, "error": error,
                    "trace": self.messages, "retrieved_fhir_resources": retrieved, "tool_calls": []}

        self.messages.append(response_message)
        if not has_json_answer(response_message.content):
            return {"answer": None, "evidence": [],
                    "limitation": "The model did not return a complete JSON answer",
                    "error": "Invalid final output: expected a JSON answer object",
                    "trace": self.messages, "retrieved_fhir_resources": retrieved, "tool_calls": []}
        answer, evidence, limitation = parse_final_message(response_message.content)
        if self.verbose:
            print(f"Packet: {packet['n_tokens']} tokens (truncated={packet['truncated']}); answer: {answer!r}")
        return {
            "answer": answer,
            "evidence": evidence,
            "limitation": limitation,
            "trace": self.messages,
            "retrieved_fhir_resources": retrieved,
            "tool_calls": [],
        }
