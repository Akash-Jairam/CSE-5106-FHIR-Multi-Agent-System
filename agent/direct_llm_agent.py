import re

from .base_agent import BaseAgent
from .output_schema import ANSWER_FORMAT_INSTRUCTIONS, parse_final_message
from tools.starting_packet import get_starting_packet, parse_current_time
from utils import safe_llm_call

QUESTION_RE = re.compile(r"^Question:\s*(.*?)\s*\nContext:", re.S)
PATIENT_RE = re.compile(r"Patient FHIR ID is ([0-9a-fA-F-]+)\.")


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

{ANSWER_FORMAT_INSTRUCTIONS}"""}

    def _run(self, question: str) -> dict:
        question_match, patient_match = QUESTION_RE.search(question), PATIENT_RE.search(question)
        if not patient_match:
            raise ValueError("The direct LLM agent needs the patient FHIR ID in the question context")
        # Rank the packet on the bare question (with the assumed current time as the cutoff)
        # so it matches manifests/starting_packets.jsonl
        packet = get_starting_packet(question_match.group(1) if question_match else question,
                                     patient_match.group(1), current_time=parse_current_time(question))
        retrieved = {}
        for rtype, ids in packet["resource_ids"].items():
            retrieved[rtype] = [{"id": i} for i in ids]

        self.messages = [self.system_msg,
                         {"role": "user", "content": f"{question}\n\n{packet['text']}"}]
        response_message, error, usage_info = safe_llm_call(
            model=self.model,
            messages=self.messages,
            base_url=self.base_url
        )
        self._update_usage(usage_info)
        if error:
            return {"answer": None, "limitation": None, "error": error,
                    "trace": self.messages, "retrieved_fhir_resources": retrieved, "tool_calls": []}

        self.messages.append(response_message)
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
