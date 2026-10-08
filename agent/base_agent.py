import sys
import os
import time
import litellm

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import setup_api_keys, validate_model_api_key, get_model_provider
from .output_schema import AgentOutput, extract_tool_calls, parse_final_message, to_tokens


class BaseAgent:
    """
    Base class for all FHIR agents.
    
    Provides essential functionality:
    - OpenAI API key setup from config.yml
    - Model configuration
    - Basic message management
    - run(): the common output contract (agent/output_schema.py)

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
        
        # Initialize usage tracking
        self.total_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'cost': 0.0,
            'llm_calls': 0
        }
        
        # Setup API keys from config.yml
        setup_api_keys()
        
    def _update_usage(self, usage_info):
        """Update total usage statistics."""
        if usage_info:
            self.total_usage['prompt_tokens'] += usage_info.get('prompt_tokens', 0)
            self.total_usage['completion_tokens'] += usage_info.get('completion_tokens', 0)
            self.total_usage['total_tokens'] += usage_info.get('total_tokens', 0)
            self.total_usage['cost'] += round(usage_info.get('cost', 0.0), 4)
            self.total_usage['llm_calls'] += 1

    def run(self, question: str) -> AgentOutput:
        """Answer `question`; returns the common output contract."""
        start = time.perf_counter()
        result = self._run(question)
        latency = time.perf_counter() - start

        if "answer" in result:
            answer, evidence, limitation = result["answer"], result.get("evidence") or [], result.get("limitation")
        else:
            answer, evidence, limitation = parse_final_message(result.get("final_answer"))
        trace = result.get("trace", self.messages)
        return {
            "answer": answer,
            "evidence": evidence,
            "limitation": limitation,
            "tokens": to_tokens(result.get("usage", self.total_usage)),
            "tool_calls": result.get("tool_calls", extract_tool_calls(trace)),
            "latency": round(latency, 3),
            "trace": trace,
            "retrieved_fhir_resources": result.get("retrieved_fhir_resources", {}),
            "error": result.get("error"),
        }

    def _run(self, question: str) -> dict:
        raise NotImplementedError