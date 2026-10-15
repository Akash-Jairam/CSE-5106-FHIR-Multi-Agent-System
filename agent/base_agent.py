import sys
import os
import time
import math
import litellm

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import setup_api_keys, validate_model_api_key, get_model_provider, safe_llm_call
from utils.core_utils import compact_model_messages, count_tokens_in_messages
from .output_schema import AgentOutput, extract_tool_calls, parse_final_message, to_tokens, validate_output

BUDGET_EXCEEDED = "Budget exceeded"
PROVIDER_ACCOUNTING_ERROR = "Provider accounting failure"


class BaseAgent:
    """
    Base class for all FHIR agents.
    
    Provides essential functionality:
    - OpenAI API key setup from config.yml
    - Model configuration
    - Basic message management
    - run(): the common output contract (agent/output_schema.py)
    - per-question token and tool-call ceilings (set_budget, _llm_call, _take_tool_call)

    Subclasses implement _run(question) and return either the contract's
    keys ("answer", "evidence", "limitation", plus "trace" and
    "retrieved_fhir_resources") or the legacy {"final_answer", "trace",
    "usage", "retrieved_fhir_resources"}, whose free-text final answer is
    parsed for a JSON answer object.
    """
    
    def __init__(self, model: str, verbose: bool = False, base_url=None):
        """
        Initialize the base agent.
        
        Args:
            model (str): The LLM model to use
            verbose (bool): Whether to print debug information
        """
        self.model = model
        self.verbose = verbose
        self.messages = []
        self.base_url = base_url

        # Per-question ceilings (None = unlimited); see set_budget()
        self.max_total_tokens = None
        self.max_tool_calls = None
        self._reset_usage()

        # Frozen missing-evidence specification for this question (see set_perturbation)
        self.perturbation = None

        # Setup API keys from config.yml
        setup_api_keys()

    def set_budget(self, max_total_tokens=None, max_tool_calls=None):
        """Aggregate per-question ceilings, including each request's completion limit."""
        for name, value in (("max_total_tokens", max_total_tokens), ("max_tool_calls", max_tool_calls)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        self.max_total_tokens = max_total_tokens
        self.max_tool_calls = max_tool_calls

    def set_perturbation(self, specification=None):
        """Start from the frozen omitted packet in `specification` (P2-18) instead of the full one.

        The FHIR store is unchanged, so a tool-equipped system can still fetch the omitted record.
        """
        if specification is not None and not isinstance(specification, dict):
            raise ValueError("A perturbation must be a frozen specification from load_perturbations()")
        self.perturbation = specification

    def _reset_usage(self):
        self.total_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'cost': 0.0,
            'llm_calls': 0
        }
        self.call_usage = []        # per model call: prompt, completion, cached prompt tokens
        self.tool_calls_made = 0
        self.budget_stop = None     # why a ceiling stopped the run, if one did
        self.last_llm_error = None
        self.context_compactions = []
        
    def _update_usage(self, usage_info):
        """Update total usage statistics."""
        if usage_info:
            self.total_usage['prompt_tokens'] += usage_info.get('prompt_tokens', 0)
            self.total_usage['completion_tokens'] += usage_info.get('completion_tokens', 0)
            self.total_usage['total_tokens'] += usage_info.get('total_tokens', 0)
            # Not rounded per call: cheap calls (< $0.00005) would otherwise count as $0
            cost = usage_info.get('cost', 0.0)
            if cost is None or self.total_usage['cost'] is None:
                self.total_usage['cost'] = None
            else:
                self.total_usage['cost'] += cost
            self.total_usage['llm_calls'] += usage_info.get('llm_calls', 1)
            self.call_usage.extend(usage_info.get('per_call') or [{
                'prompt': usage_info.get('prompt_tokens', 0),
                'completion': usage_info.get('completion_tokens', 0),
                'cached': usage_info.get('cached_prompt_tokens', 0),
            }])

    def _llm_call(self, messages, tools=None):
        """safe_llm_call() under the token ceiling; records usage. Returns (message, error, usage)."""
        if self.max_tool_calls is not None and self.tool_calls_made >= self.max_tool_calls:
            tools = None
        prompt = count_tokens_in_messages(messages, tools)
        if prompt > 28_000:
            messages, compaction = compact_model_messages(messages, tools)
            if compaction:
                self.context_compactions.append(compaction)
                prompt = count_tokens_in_messages(messages, tools)
        limits = {}
        if self.max_total_tokens is not None:
            # Tokenizers and provider framing differ; reserve a margin before capping output.
            prompt_reserve = prompt + max(64, math.ceil(prompt * 0.05))
            remaining = self.max_total_tokens - self.total_usage['total_tokens'] - prompt_reserve
            if remaining < 1:
                self.budget_stop = (f"token budget: {self.total_usage['total_tokens']} tokens used, the next prompt "
                                    f"needs ~{prompt}, ceiling {self.max_total_tokens}")
                self.last_llm_error = f"{BUDGET_EXCEEDED}: {self.budget_stop}"
                return None, self.last_llm_error, None
            limits = {"max_completion_tokens": remaining, "max_retries": 1}
        response, error, usage_info = safe_llm_call(model=self.model, messages=messages, tools=tools,
                                                    base_url=self.base_url, **limits)
        self._update_usage(usage_info)
        self.last_llm_error = error
        if self.max_total_tokens is not None and not error and usage_info and any(
                c.get("usage_available") is False for c in usage_info.get("per_call", [])):
            self.budget_stop = "provider did not report token usage; aggregate ceiling cannot be verified"
            self.last_llm_error = f"{PROVIDER_ACCOUNTING_ERROR}: {self.budget_stop}"
            return None, self.last_llm_error, usage_info
        if self.max_total_tokens is not None and self.total_usage['total_tokens'] > self.max_total_tokens:
            self.budget_stop = "provider usage exceeded the reserved token ceiling"
            self.last_llm_error = f"{PROVIDER_ACCOUNTING_ERROR}: {self.budget_stop}"
            return None, self.last_llm_error, usage_info
        return response, error, usage_info

    def _take_tool_call(self):
        """Count one tool call; False once max_tool_calls is used up (the caller must not run it)."""
        if self.max_tool_calls is not None and self.tool_calls_made >= self.max_tool_calls:
            self.budget_stop = self.budget_stop or f"tool-call budget: all {self.max_tool_calls} calls used"
            return False
        self.tool_calls_made += 1
        return True

    def _tool_budget_message(self, tool_call):
        """Tool reply when the tool-call budget is used up."""
        return {"tool_call_id": tool_call.id, "role": "tool", "name": tool_call.function.name,
                "content": f"Not run: the tool-call budget ({self.max_tool_calls} calls) is used up. "
                           "Give your final answer now with the data you already have."}

    def run(self, question: str) -> AgentOutput:
        """Answer `question`; returns the common output contract."""
        self._reset_usage()
        self.messages = []
        start = time.perf_counter()
        try:
            result = self._run(question)
            if not isinstance(result, dict):
                raise ValueError("Agent result must be an object")
        except Exception as e:
            result = {"answer": None, "error": str(e), "trace": self.messages}
        latency = time.perf_counter() - start

        if "answer" in result:
            answer, evidence, limitation = result["answer"], result.get("evidence") or [], result.get("limitation")
        else:
            answer, evidence, limitation = parse_final_message(result.get("final_answer"))
        error = result.get("error") or self.last_llm_error
        legacy_answer = result.get("final_answer")
        if isinstance(legacy_answer, str) and legacy_answer in {
                "No final answer reached within iteration limit.",
                "No tool calls found in the response"}:
            error = result["final_answer"]
        # A refused call under the token ceiling is an abstention with a reason, not an error
        if any(isinstance(v, str) and BUDGET_EXCEEDED in v for v in (answer, error)):
            if isinstance(answer, str) and BUDGET_EXCEEDED in answer:
                answer, evidence = None, []
            error = None
            limitation = limitation or f"Stopped by the {self.budget_stop}"
        elif error:
            answer, evidence = None, []
        trace = result.get("trace", self.messages)
        output = {
            "answer": answer,
            "evidence": evidence,
            "limitation": limitation,
            "tokens": to_tokens(self.total_usage, self.call_usage),
            "tool_calls": result.get("tool_calls", extract_tool_calls(trace)),
            "latency": round(latency, 3),
            "trace": trace,
            "retrieved_fhir_resources": result.get("retrieved_fhir_resources", {}),
            "error": error,
            "revision": result.get("revision"),
            "budget": {
                "max_total_tokens": self.max_total_tokens,
                "max_tool_calls": self.max_tool_calls,
                "tool_calls_run": self.tool_calls_made,
                "stopped": self.budget_stop,
            },
        }
        problems = validate_output(output)
        if problems:
            output["answer"], output["evidence"] = None, []
            output["limitation"] = "The agent returned an invalid output contract"
            output["error"] = "Invalid output: " + "; ".join(problems)
            output["invalid_output"] = problems
            if not isinstance(output["tool_calls"], list):
                output["tool_calls"] = []
        return output

    def _run(self, question: str) -> dict:
        raise NotImplementedError
