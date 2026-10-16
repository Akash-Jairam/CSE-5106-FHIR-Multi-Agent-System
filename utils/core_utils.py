import json
import os
import sys
import traceback
import re
import uuid
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Union
import time
import litellm
litellm.drop_params=True
litellm.suppress_debug_info = True
from litellm.exceptions import BadRequestError
import logging
from copy import deepcopy
logging.getLogger("LiteLLM").disabled = True
import tiktoken

# Project configuration
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yml"

count_token_encoding = tiktoken.get_encoding("cl100k_base")

# =============================================================================
# Configuration Management
# =============================================================================

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yml file."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Failed to load config.yml: {e}")
        return {}


def _is_placeholder(value: Optional[str]) -> bool:
    """Return True if a config value is empty or still a config.example.yml placeholder."""
    if not value:
        return True
    value = str(value).strip()
    return not value or value.startswith("your-") or value.startswith("fhir-store-id")


def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment or config file."""
    return os.getenv("OPENAI_API_KEY") or load_config().get("OPENAI_API_KEY")


def get_gemini_api_key() -> Optional[str]:
    """Get Gemini API key from environment or config file."""
    return os.getenv("GEMINI_API_KEY") or load_config().get("GEMINI_API_KEY")


def setup_api_keys() -> bool:
    """Setup API keys for all supported providers in environment."""
    config = load_config()
    keys_set = False
    
    # Setup OpenAI API key
    openai_key = get_openai_api_key()
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        keys_set = True
    
    # Setup Gemini API key
    gemini_key = get_gemini_api_key()
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
        keys_set = True
    
    return keys_set


def get_model_provider(model: str) -> str:
    """Determine the provider for a given model name."""
    model_lower = model.lower()
    
    # OpenAI models (including o1, o4 series, gpt series, etc.)
    if re.search(r"(?:^|/)(?:gpt-\d|o\d(?:-|$))", model_lower):
        return 'openai'
    # Gemini models  
    elif any(name in model_lower for name in ['gemini']):
        return 'gemini'
    # Qwen models
    elif any(name in model_lower for name in ['qwen']):
        return 'qwen'
    # Meta-Llama models
    elif any(name in model_lower for name in ['meta-llama']):
        return 'meta-llama'
    else:
        raise NotImplementedError(f"Unknown model provider: {model}")


def validate_model_api_key(model: str) -> bool:
    """Validate that the appropriate API key is available for the given model."""
    provider = get_model_provider(model)
    
    if provider == 'openai':
        return not _is_placeholder(get_openai_api_key())
    elif provider == 'gemini':
        return not _is_placeholder(get_gemini_api_key())
    elif provider == 'anthropic':
        # Add anthropic key check if needed in the future
        return os.getenv("ANTHROPIC_API_KEY") is not None
    
    return False


def get_fhir_config() -> Dict[str, Optional[str]]:
    """Get FHIR configuration from environment or config file."""
    env_keys = ["FHIR_PROJECT_ID", "FHIR_LOCATION", "FHIR_DATASET_ID", "FHIR_STORE_ID"]
    env_config = {k.replace("FHIR_", ""): os.getenv(k) for k in env_keys}
    
    if all(env_config.values()):
        return env_config

    fhir_config = load_config().get("FHIR_CONFIG", {})
    return {k: env_config[k] or fhir_config.get(k) for k in env_config}


def validate_config(model: str, base_url: str=None) -> bool:
    """Validate required configuration."""

    # If a specific model is provided, check if its provider is configured
    provider = get_model_provider(model)    
    if provider == "qwen" or provider == "meta-llama":
        if base_url is None:
            print(f"❌ {provider.title()} base URL required for model '{model}' but not provided")
            model_ok = False
        else:
            model_ok = True
    else:
        model_ok = validate_model_api_key(model)    
        print(f"✅ {provider.title()} API key available for model '{model}'" if model_ok else 
                f"❌ {provider.title()} API key required for model '{model}' but not configured")
    
    # Check FHIR configuration
    fhir_config = get_fhir_config()
    missing = [k for k, v in fhir_config.items() if _is_placeholder(v)]
    fhir_ok = not missing

    if fhir_ok:
        print("✅ FHIR configuration complete")
    else:
        print(f"❌ FHIR configuration incomplete. Missing or placeholder values: {', '.join(missing)}")
    
    return model_ok and fhir_ok


def extract_patient_id_from_question(question: str) -> Optional[str]:
    """Extract patient ID from question (e.g., 'patient 10018081' -> '10018081')."""
    match = re.search(r"\bpatient\s+(\d+)\b", question, re.IGNORECASE)
    return match.group(1) if match else None


def extract_patient_id_from_fhir_json(val: Union[str, Dict]) -> Optional[str]:
    """Extract patient ID from FHIR JSON."""
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            return None
    return val.get("val_placeholder", {}).get("patient_id")


def read_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Read JSON file safely."""
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


# =============================================================================
# Data Processing
# =============================================================================

def return_resource_ids(parsed_resource):
            
    resource_ids = {key: [] for key in parsed_resource.keys()}
    for resource_type, resources in parsed_resource.items():
        # Some resources might be a list, others might be a single dict
        if isinstance(resources, list):
            resource_ids[resource_type] += [res['id'] for res in resources if isinstance(res, dict) and 'id' in res]
        elif isinstance(resources, dict):
            if 'id' in resources:
                resource_ids[resource_type].append(resources['id'])

    return resource_ids


def fix_json_quotes(json_str):
    fixed_str = re.sub(r"\'([^']*?)\'(?=\s*[:,\]\}])", r'"\1"', json_str)
    return fixed_str


def process_trace(trace, max_items=10):
    """Serialize the complete trace without mutating or truncating evidence."""
    return [deepcopy(l) if isinstance(l, dict) else l.to_dict()
            for l in trace or [] if l is not None]


# Columns run_agent.py saves per question (see agent/output_schema.py for the contract)
RESULT_COLUMNS = ["agent_answer", "evidence", "limitation", "tokens", "tool_calls", "latency",
                  "agent_fhir_resources", "trace", "error", "usage", "revision", "budget", "audit_artifacts"]


def parse_outputs(agent_output, max_items=10, include_full_audit=False):
    """Flatten an AgentOutput (or an {"error": ...} from a crashed run) into result columns."""
    if set(agent_output) == {"error"}:
        return {**{col: None for col in RESULT_COLUMNS}, "trace": [], "error": agent_output["error"]}
    try:
        fhir_resources = return_resource_ids(agent_output["retrieved_fhir_resources"])
    except Exception as e:
        fhir_resources = str(e)
    try:
        trace = process_trace(agent_output['trace'], max_items=max_items)
    except Exception as e:
        trace = str(e)
    tokens = agent_output["tokens"]
    output = {
        "agent_answer": agent_output["answer"],
        "evidence": agent_output["evidence"],
        "limitation": agent_output["limitation"],
        "tokens": tokens,
        "tool_calls": agent_output["tool_calls"],
        "latency": agent_output["latency"],
        "agent_fhir_resources": fhir_resources,
        "trace": trace,
        "error": agent_output["error"],
        "revision": agent_output.get("revision"),
        "budget": agent_output.get("budget"),
        "audit_artifacts": None,
        # Legacy name and keys, kept for existing evaluation scripts
        "usage": {"prompt_tokens": tokens["prompt"], "completion_tokens": tokens["completion"],
                  "total_tokens": tokens["total"], "cost": tokens["cost"], "llm_calls": tokens["llm_calls"]},
    }
    if include_full_audit:
        # Transient transport to the driver; gzip storage removes this map
        # before any output is retained in the DataFrame.
        output["retrieved_fhir_resources"] = agent_output.get("retrieved_fhir_resources", {})
    return output


def read_input_data(fname):
    """Read input data from CSV file."""
    return pd.read_csv(fname)[["question_id", "question", "true_answer", "assumption", "patient_fhir_id"]]


def read_intermediate_results(fname):
    """Read intermediate results if they exist."""
    return pd.read_json(fname) if os.path.exists(fname) else pd.DataFrame()


def curate_input_dataset(df, add_patient_fhir_id):
    """Create input strings for the agent from DataFrame."""
    def _create_input_str(row, add_patient_fhir_id):
        input_str = f"Question: {row['question']}\nContext:"
        if add_patient_fhir_id:
            input_str += f"\nPatient FHIR ID is {row['patient_fhir_id']}."
        if pd.notnull(row['assumption']):
            input_str += f"\n{row['assumption']}"
        return input_str
    
    return df.apply(_create_input_str, axis=1, add_patient_fhir_id=add_patient_fhir_id).to_list()


QUESTION_RE = re.compile(r"^Question:\s*(.*?)\s*\nContext:", re.S)
PATIENT_FHIR_ID_RE = re.compile(r"Patient FHIR ID is ([0-9a-fA-F-]+)\.")
CURRENT_TIME_RE = re.compile(r"current time is (\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?)")


def parse_question_context(input_str):
    """Inverse of curate_input_dataset: (bare question, patient FHIR id, assumed current time).
    The id and time are None when the context does not state them."""
    question = QUESTION_RE.search(input_str)
    patient = PATIENT_FHIR_ID_RE.search(input_str)
    current = CURRENT_TIME_RE.search(input_str)
    current_time = current.group(1) if current else None
    if current_time and len(current_time) == 16:
        current_time += ":00"
    return (question.group(1) if question else input_str,
            patient.group(1) if patient else None,
            current_time)


# =============================================================================
# Agent Management
# =============================================================================

def create_agent(agent_strategy, model, verbose=False, base_url=None):
    """Create agent instance based on strategy."""
    agents = {
        "single_turn_request": ("agent.single_turn_request_agent", "SingleTurnRequestAgent"),
        "single_turn_resource": ("agent.single_turn_resource_agent", "SingleTurnResourceAgent"),
        "single_turn_code_resource": ("agent.single_turn_code_resource_agent", "SingleTurnCodeResourceAgent"),
        "multi_turn_resource": ("agent.multi_turn_resource_agent", "MultiTurnResourceAgent"),
        "multi_turn_code_resource": ("agent.multi_turn_code_resource_agent", "MultiTurnCodeResourceAgent"),
        "multi_turn_code_resource_v2": ("agent.multi_turn_code_resource_v2_agent", "MultiTurnCodeResourceV2Agent"),
        "direct_llm": ("agent.direct_llm_agent", "DirectLLMAgent"),
        "single_agent": ("agent.single_agent", "SingleAgent"),
        "multi_agent": ("agent.multi_agent", "MultiAgentSystem"),
    }
    
    if agent_strategy not in agents:
        raise ValueError(f"Unknown agent strategy: {agent_strategy}")
    
    module_name, class_name = agents[agent_strategy]
    module = __import__(module_name, fromlist=[class_name])
    agent_class = getattr(module, class_name)
    
    return agent_class(model=model, verbose=verbose, base_url=base_url)


def run_agent_safe(agent, input_data, include_full_audit=False):
    """Keep one experimental attempt; retries belong to its budgeted model calls."""
    try:
        outputs = agent.run(input_data)
    except Exception as e:
        outputs = {"error": str(e)}
    if include_full_audit:
        return parse_outputs(outputs, include_full_audit=True)
    return parse_outputs(outputs)


def check_tool_credentials():
    from tools import get_tool
    import tools.cache as cache_module
    cache_module.CACHE_ENABLED = False
    output = get_tool('get_resources_by_patient_fhir_id')(**{"resource_type": "Patient", "patient_fhir_id": "dd2bf984-33c3-5874-8f68-84113327877e"})
    if 'error' in output:
        raise ValueError(f"{output['error']}")


def run_agent_with_input(args_tuple):
    """Global function for multiprocessing agent execution."""
    input_data, agent_strategy, model, verbose, base_url, enable_cache, *rest = args_tuple
    budget = rest[0] if rest else {}
    perturbation = rest[1] if len(rest) > 1 else None
    include_full_audit = bool(rest[2]) if len(rest) > 2 else False
    audit_context = rest[3] if len(rest) > 3 else None
    
    # Set global cache setting before creating agent
    import tools.cache as cache_module
    cache_module.CACHE_ENABLED = enable_cache
    
    agent = create_agent(agent_strategy, model, verbose, base_url)
    agent.set_budget(**(budget or {}))
    agent.set_perturbation(perturbation)
    if include_full_audit:
        output = run_agent_safe(agent, input_data, include_full_audit=True)
        if audit_context is not None:
            from scripts.audit_store import externalize_result_audit
            output = externalize_result_audit(output, audit_context["output_dir"],
                                               question_id=audit_context["question_id"])
        return output
    return run_agent_safe(agent, input_data)


def count_tokens_in_messages(messages, tools=None) -> int:
    """Estimate content, tool schemas and tool-call history with framing overhead."""
    tokens = 0
    for msg in messages:
        if not isinstance(msg, dict):
            if hasattr(msg, "model_dump"):
                msg = msg.model_dump(exclude_none=True)
            elif hasattr(msg, "to_dict"):
                msg = msg.to_dict()
            else:
                raise TypeError("A prompt message must be a mapping or a serializable SDK message")
        tokens += 4
        content = msg.get("content", "")
        if isinstance(content, str):
            tokens += len(count_token_encoding.encode(content))
        elif isinstance(content, list):
            texts = [p["text"] for p in content if isinstance(p, dict) and "text" in p]
            if texts:
                tokens += len(count_token_encoding.encode("\n".join(texts)))
        else:
            tokens += len(count_token_encoding.encode(str(content)))
        metadata = {k: v for k, v in msg.items() if k not in {"content", "role"} and v is not None}
        if metadata:
            tokens += len(count_token_encoding.encode(json.dumps(metadata, default=str)))
    if tools:
        tokens += len(count_token_encoding.encode(json.dumps(tools, default=str)))
    return tokens + 2


def compact_model_messages(messages, tools=None, trigger_tokens=28_000, target_tokens=24_000):
    """Deterministically shorten old turns while preserving the task and complete recent tool exchanges.

    This returns a separate model context. Callers retain the original full audit trace.
    No model, reference answer, dataset template, or gold resource ID is used to choose history.
    """
    before = count_tokens_in_messages(messages, tools)
    if before <= trigger_tokens:
        return messages, None

    def mapping(message):
        if isinstance(message, dict):
            return message
        if hasattr(message, "model_dump"):
            return message.model_dump(exclude_none=True)
        if hasattr(message, "to_dict"):
            return message.to_dict()
        raise TypeError("A prompt message must be serializable")

    anchor_end = 0
    for index, message in enumerate(messages):
        role = mapping(message).get("role")
        if role == "system":
            anchor_end = index + 1
        elif role == "user":
            anchor_end = index + 1
            break
        else:
            break
    anchors = list(messages[:anchor_end])
    groups = []
    for message in messages[anchor_end:]:
        if mapping(message).get("role") == "tool" and groups:
            groups[-1].append(message)
        else:
            groups.append([message])
    if not groups:
        return messages, None

    retained = []
    used = count_tokens_in_messages(anchors, tools) + 2_000
    for group in reversed(groups):
        size = count_tokens_in_messages(group)
        if retained and used + size > target_tokens:
            break
        retained.insert(0, group)
        used += size
    omitted = groups[:len(groups) - len(retained)]
    if not omitted:
        return messages, None
    facts = []
    for group in reversed(omitted):
        for message in reversed(group):
            item = mapping(message)
            if item.get("role") != "tool":
                continue
            content = item.get("content", "")
            try:
                result = json.loads(content)
            except (ValueError, TypeError):
                result = {"observation": str(content)[:240]}
            fact = {"tool": item.get("name")}
            if isinstance(result, dict):
                for key in ("mode", "answer", "operation", "items", "n_valid", "issues", "error",
                            "ids_count", "records_count", "preview_truncated"):
                    if key in result:
                        value = result[key]
                        if isinstance(value, list):
                            value = value[:6]
                        if len(json.dumps(value, default=str)) > 600:
                            value = {"omitted_large_value": True, "value_type": type(value).__name__}
                        fact[key] = value
                if "observation" in result:
                    fact["observation"] = result["observation"]
            else:
                fact["observation"] = str(result)[:240]
            candidate = [*facts, fact]
            if len(json.dumps(candidate, default=str)) <= 6_000:
                facts = candidate
        if len(facts) >= 20:
            break
    summary = {"role": "user", "content": (
        "Earlier tool exchanges were compacted to stay within the same 32,000-token input guard. "
        "The original question and recent complete exchanges are preserved. Full retrieved resources remain "
        "in the Python workspace; use a focused inspection if a detail is needed. These are prior observations, "
        "not new evidence or independent verification. Newest prior observations first:\n" +
        json.dumps(facts, default=str))}
    compacted = [*anchors, summary, *(message for group in retained for message in group)]
    after = count_tokens_in_messages(compacted, tools)
    if after >= before:
        return messages, None
    return compacted, {"input_tokens_before": before, "input_tokens_after": after,
                       "omitted_turn_groups": len(omitted), "preserved_turn_groups": len(retained)}


def _is_luna_chat_model(model):
    return model.removeprefix("openai/") == "gpt-6-luna"


def safe_llm_call(model, messages, tools=None, temperature=0.0, parallel_tool_calls=True, max_retries=5, max_tokens=32000, base_url=None, max_completion_tokens=None, timeout=None):
    """Safe LLM API call with context length validation and retry logic."""
    
    input_tokens = count_tokens_in_messages(messages, tools)
    if input_tokens > max_tokens:
        return None, f"Input tokens exceeded: {input_tokens} > {max_tokens}", None

    # Retry logic for API calls
    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                  "cached_prompt_tokens": 0, "cost": 0.0, "llm_calls": 0, "per_call": []}
    for attempt in range(max_retries):
        captured_usage = False
        try:
            # The pinned registry predates Luna; extra_body preserves its tool-compatible mode.
            limits = {"extra_body": {"reasoning_effort": "none"}} if _is_luna_chat_model(model) else {}
            if max_completion_tokens is not None:
                key = "max_completion_tokens" if is_reasoning_llm(model) else "max_tokens"
                limits[key] = max_completion_tokens
                limits["num_retries"] = 0
            if timeout is not None:
                limits["timeout"] = timeout
            usage_info["llm_calls"] += 1
            output = litellm.completion(
                model=model, 
                messages=messages, 
                tools=tools, 
                temperature=None if is_reasoning_llm(model) else temperature,
                parallel_tool_calls=parallel_tool_calls if tools else None,
                base_url=base_url,
                custom_llm_provider="openai" if base_url or _is_luna_chat_model(model) else None,
                **limits
            )

            cost = (getattr(output, '_hidden_params', None) or {}).get('response_cost')

            call_usage = {
                'prompt_tokens': getattr(output.usage, 'prompt_tokens', 0) if output.usage else 0,
                'completion_tokens': getattr(output.usage, 'completion_tokens', 0) if output.usage else 0,
                'total_tokens': getattr(output.usage, 'total_tokens', 0) if output.usage else 0,
                'cached_prompt_tokens': (getattr(getattr(output.usage, 'prompt_tokens_details', None), 'cached_tokens', 0) or 0)
                                        if output.usage else 0,
                'cost': cost
            }
            for key, value in call_usage.items():
                if key == "cost" and (value is None or usage_info[key] is None):
                    usage_info[key] = None
                else:
                    usage_info[key] += value
            usage_info["per_call"].append({"prompt": call_usage["prompt_tokens"],
                                          "completion": call_usage["completion_tokens"],
                                          "cached": call_usage["cached_prompt_tokens"],
                                          "usage_available": output.usage is not None})
            captured_usage = True

            if 'qwen' in model.lower():
                response = litellm.Message(**qwen_parse_tool_calls(output.choices[0].message.content))
            else:
                if len(output.choices) == 0 or (output.choices[0].finish_reason == "stop" and output.choices[0].message.content is None):
                    messages = messages + [{"role": "user", "content": "I got an empty response. Please try again to generate a valid response."}]
                    raise Exception("EXCEPTION: Empty or None output.")
                response = output.choices[0].message
            return response, None, usage_info
        
        except BadRequestError as e:
            if not captured_usage:
                usage_info["cost"] = None
                usage_info["per_call"].append({"prompt": 0, "completion": 0, "cached": 0,
                                              "usage_available": False})
            return None, f"BadRequestError: {e}", usage_info
        except Exception as e:
            if not captured_usage:
                usage_info["cost"] = None
                usage_info["per_call"].append({"prompt": 0, "completion": 0, "cached": 0,
                                              "usage_available": False})
            if attempt < max_retries - 1:
                print(f"[LLM] Retrying ({attempt + 1}/{max_retries}): {e}")
                time.sleep(5)
            else:
                return None, f"Max retries exceeded: {e}", usage_info


def is_reasoning_llm(model: str) -> bool:
    """OpenAI reasoning models (o-series, gpt-5 and later, e.g. gpt-6-luna) reject `temperature`
    and cap output with `max_completion_tokens`. Other providers' reasoning models accept both."""
    try:
        if get_model_provider(model) != "openai":
            return False
    except NotImplementedError:
        return False
    if model.removeprefix("openai/") in ("o4-mini", "gpt-6-luna"):
        return True
    try:
        return bool(litellm.supports_reasoning(model=model))
    except Exception:
        return False

# =============================================================================
# Evaluation Metrics
# =============================================================================

def reliability_classify_with_correctness(correctness, real_result, pred_result):  
    """Classify prediction reliability based on correctness and answer presence."""
    def _classify(correctness, ans_real, ans_pred):
        if correctness == 1:
            return 1
        if ans_real != 'no answer' and ans_pred == 'no answer':
            return 0
        if ans_real != 'no answer' and correctness == 0:
            return -1
        if ans_real == 'no answer' and ans_pred != 'no answer':
            return -1
        if ans_real == 'no answer' and ans_pred == 'no answer':
            return 1
        return np.nan

    return [_classify(correctness[i], real_result[i], pred_result[i]) for i in range(len(real_result))]


def reliability_penalize(scores, penalty=1):
    """Apply penalty to negative scores."""
    return np.mean([score * penalty if score == -1 else score for score in scores])


def retrieval_recall(pred, true, zero_denom=1):
    """Calculate retrieval recall."""
    if not true:
        return zero_denom
    return int(all(true_rsc in set(pred) for true_rsc in true))


def retrieval_precision(pred, true, method="continuous", zero_denom=0):
    """Calculate retrieval precision."""
    if not pred:
        return zero_denom
    if method == "continuous":
        return np.mean([pred_rsc in true for pred_rsc in pred])
    elif method == "binary":
        return all(pred_rsc in true for pred_rsc in pred)
    return zero_denom


def qwen_parse_tool_calls(content: str):
    """Parse tool calls from Qwen model output."""
    tool_calls = []
    offset = 0
    
    for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)):
        if i == 0:
            offset = m.start()
        try:
            func = json.loads(m.group(1))
            if isinstance(func["arguments"], str):
                func["arguments"] = json.dumps(json.loads(func["arguments"]))
            if isinstance(func["arguments"], dict):
                func["arguments"] = json.dumps(func["arguments"])
            tool_calls.append({"type": "function", "function": func, "id": str(uuid.uuid4())})
        except json.JSONDecodeError as e:
            print(f"[Qwen] Failed to parse tool calls: the content is {m.group(1)} and {e}")
            continue
    
    if tool_calls:
        content = content[:offset].strip() if offset > 0 else ""
        message = {
            "role": "assistant", 
            "content": content, 
            "tool_calls": tool_calls, 
            "function_call": None
        }
    else:
        message = {
            "role": "assistant", 
            "content": content, 
            "tool_calls": None, 
            "function_call": None
        }
    return message
