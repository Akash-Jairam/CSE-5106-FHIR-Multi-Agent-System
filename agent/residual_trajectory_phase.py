"""V7 phase isolation/accounting. Shared Single, Base, tools and runtime are unchanged."""
from copy import deepcopy
import hashlib
import json
import math
from types import FunctionType, MethodType, SimpleNamespace

from . import base_agent as base_module
from .base_agent import BaseAgent, BUDGET_EXCEEDED
from .continuous_primary import _adopt, _binding, _tagged
from .single_agent import SingleAgent

SDK_CAP = 30


def json_text(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(json_text(value).encode()).hexdigest()


def same(left, right):
    return json_text(left) == json_text(right)


def reserve(prompt):
    return prompt + max(64, math.ceil(prompt * .05))


def strict_object(content, limit):
    if not isinstance(content, str) or len(content) > limit:
        raise ValueError("selector response length/type differs")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(content, object_pairs_hook=pairs)
    json_text(value)
    if not isinstance(value, dict):
        raise ValueError("selector response is not one finite object")
    return value


def clean_ledger():
    return SimpleNamespace(total_usage={"prompt_tokens": 0, "completion_tokens": 0,
        "total_tokens": 0, "llm_calls": 0, "cost": 0.0}, call_usage=[], tool_calls_made=0,
        messages=[], budget_stop=None, last_llm_error=None, context_compactions=[])


def validated_ledger(local):
    """Use unchanged _adopt once on a NEW scratch host, never the dirty outer host."""
    scratch = clean_ledger()
    raw, problems, unknown = _adopt(scratch, local)
    return scratch, raw, problems, unknown


def merge_ledger(host, scratch, phase):
    if phase in host._v7_merged_phases:
        raise ValueError("phase ledger already merged")
    host._v7_merged_phases.add(phase)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "llm_calls"):
        host.total_usage[key] += scratch.total_usage[key]
    costs = (host.total_usage["cost"], scratch.total_usage["cost"])
    host.total_usage["cost"] = None if None in costs else sum(costs)
    host.call_usage.extend(deepcopy(scratch.call_usage))
    host.tool_calls_made += scratch.tool_calls_made
    host.context_compactions.extend(deepcopy(scratch.context_compactions))
    cost = host.total_usage["cost"]
    if cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
        # Retain the finite phase/raw amounts; an unrepresentable aggregate is
        # explicitly unknown, never a NaN/Infinity that destroys the fatal audit.
        host.total_usage["cost"] = None
        return ["aggregate cost overflow"]
    return []


def mark_unknown(local, before):
    """An admitted exception/missing ledger is not a synthetic free invocation."""
    old_calls, old_n = before
    if not isinstance(local.call_usage, list):
        local.call_usage = []
    if len(local.call_usage) <= old_n:
        local.call_usage.append({"prompt": 0, "completion": 0, "cached": 0,
                                 "usage_available": False, "usage_placeholder": True})
    value = local.total_usage.get("llm_calls")
    local.total_usage["llm_calls"] = max(value if type(value) is int else 0, old_calls + 1)
    local.total_usage["cost"] = None


def message_dict(message):
    if isinstance(message, dict):
        return deepcopy(message)
    if hasattr(message, "model_dump"):
        return message.model_dump()
    return message.to_dict()


def _guard_B_calls(single, sdk_cap):
    original = single._llm_call
    state = {"fatal": False, "errors": [], "call_limits": []}
    def guarded(messages, tools=None):
        if single.total_usage["llm_calls"] >= sdk_cap:
            single.budget_stop = "alternative SDK allowance exhausted"
            single.last_llm_error = BUDGET_EXCEEDED + ": " + single.budget_stop
            return None, single.last_llm_error, None
        before_tokens = single.total_usage["total_tokens"]
        before = (single.total_usage["llm_calls"], len(single.call_usage))
        # Observes exactly the same deterministic Base tokenizer/framing rule.
        actual_tools = None if single.max_tool_calls is not None and single.tool_calls_made >= single.max_tool_calls else tools
        prompt = base_module.count_tokens_in_messages(messages, actual_tools)
        compactions = len(single.context_compactions)
        try:
            response, error, usage = original(messages, tools=tools)
        except Exception:
            mark_unknown(single, before)
            state["fatal"] = True
            state["errors"].append("admitted alternative call escaped with unverifiable usage")
            raise
        if usage is None and response is None and isinstance(error, str) and (
                BUDGET_EXCEEDED in error or error.startswith("Input tokens exceeded:")):
            return response, error, usage
        # Any other returned invocation must expose exactly one canonical usage.
        if not isinstance(usage, dict) or usage.get("llm_calls") != 1 or type(usage.get("llm_calls")) is not int:
            mark_unknown(single, before)
            state["fatal"] = True
            state["errors"].append("alternative invocation usage is missing or not one attempt")
        scratch, raw, problems, unknown = validated_ledger(single)
        if problems or unknown:
            state["fatal"] = True
            state["errors"].extend(problems or ["alternative usage unavailable"])
            state["malformed_raw_ledger"] = raw
            # Prevent the unchanged common formatter losing the audit on bad types.
            single.total_usage = deepcopy(scratch.total_usage)
            single.call_usage = deepcopy(scratch.call_usage)
        if len(single.context_compactions) > compactions:
            prompt = single.context_compactions[-1]["input_tokens_after"]
        completion_limit = single.max_total_tokens - before_tokens - reserve(prompt)
        state["call_limits"].append({"prompt_tokens_estimate": prompt, "completion_limit": completion_limit})
        if isinstance(usage, dict) and type(usage.get("completion_tokens")) is int and usage["completion_tokens"] > completion_limit:
            state["fatal"] = True
            state["errors"].append("alternative reported completion overshoot")
        if single.total_usage["llm_calls"] > sdk_cap:
            state["fatal"] = True
            state["errors"].append("alternative SDK allowance overshoot")
        if state["fatal"]:
            single.last_llm_error = "Provider accounting failure: alternative invocation unverifiable"
            return None, single.last_llm_error, usage
        return response, error, usage
    single._llm_call = guarded
    return state


def _observe_initialization(single, host):
    """Same Single._run bytecode, private globals overlay; no process-wide patch."""
    state = {"charged": 0, "started": 0, "completed": 0, "kind": None}
    original = SingleAgent._run
    namespace = dict(original.__globals__)
    for name in ("get_starting_packet", "get_perturbed_starting_packet"):
        function = namespace[name]
        def observed(*args, _function=function, _name=name, **kwargs):
            state["started"] += 1
            state["kind"] = _name
            if state["started"] != 1:
                raise ValueError("alternative initialized more than once")
            if not host._take_tool_call():
                raise ValueError("alternative initialization allowance unavailable")
            state["charged"] = 1
            packet = _function(*args, **kwargs)
            state["packet_sha256"] = fingerprint(packet)
            state["completed"] += 1
            return packet
        namespace[name] = observed
    clone = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    clone.__kwdefaults__ = original.__kwdefaults__
    single._run = MethodType(clone, single)
    return state


def run_single_phase(host, question, phase, token_cap, tool_cap, sdk_cap, *, initialization_charge=False):
    single = SingleAgent(host.model, max_iterations=sdk_cap, verbose=host.verbose, base_url=host.base_url)
    single.set_budget(max_total_tokens=token_cap, max_tool_calls=tool_cap)
    single.set_perturbation(host.perturbation)
    call_state = _guard_B_calls(single, sdk_cap) if initialization_charge else {"fatal": False, "errors": []}
    init = _observe_initialization(single, host) if initialization_charge else {"charged": 0}
    exception = None
    try:
        released = single.run(question)
    except Exception as exc:
        exception = type(exc).__name__
        released = {"answer": None, "evidence": [], "limitation": None,
                    "error": "Phase output formatting failed: " + exception,
                    "revision": None, "trace": deepcopy(single.messages), "retrieved_fhir_resources": {}}
    scratch, raw, problems, unknown = validated_ledger(single)
    errors = list(problems) + list(call_state["errors"])
    if unknown:
        errors.append("phase provider usage unknown")
    if exception:
        errors.append("phase common formatter failed")
    if scratch.total_usage["total_tokens"] > token_cap or scratch.tool_calls_made > tool_cap or scratch.total_usage["llm_calls"] > sdk_cap:
        errors.append("phase actual ceiling overshoot")
    errors.extend(merge_ledger(host, scratch, phase))
    trace = deepcopy(released.get("trace") or single.messages)
    workspace = deepcopy(released.get("retrieved_fhir_resources") or {})
    artifact, reason, binding = _binding(released, deepcopy(single.final_computation), workspace, trace)
    source_check = binding.pop("source_check", None) if artifact else None
    if errors or call_state["fatal"]:
        artifact, reason, binding, source_check = None, "phase accounting fatal", {}, None
    if initialization_charge and (init["started"] != 1 or init["completed"] != 1):
        artifact, reason, binding, source_check = None, "original initialization did not complete", {}, None
    trace = [message_dict(item) for item in trace]
    tagged_trace = [{**deepcopy(item), "phase": phase} for item in trace]
    if initialization_charge:
        host.messages.append({"role": "tool", "name": "initialize_original_starting_packet",
            "tool_call_id": "v7-B-initialization", "phase": "alternative_initialization",
            "host_operation": True, "content": json_text({"question_sha256": fingerprint(question), **init})})
    host.messages.extend(tagged_trace)
    diagnostic = {"phase": phase, "token_cap": token_cap, "tool_cap": tool_cap,
        "sdk_cap": sdk_cap, "tokens": scratch.total_usage["total_tokens"],
        "tool_calls": scratch.tool_calls_made + init["charged"], "natural_tool_calls": scratch.tool_calls_made,
        "model_calls": scratch.total_usage["llm_calls"], "fatal": bool(errors or call_state["fatal"]),
        "usage_known": not unknown, "errors": errors, "eligible": artifact is not None,
        "skip_reason": reason, "budget_stop": single.budget_stop, "initialization": deepcopy(init),
        "call_limits": call_state.get("call_limits", [])}
    snapshot = {"version": 1, "kind": "candidate_snapshot", "candidate": "B" if initialization_charge else "A",
        "phase": phase, "released": {key: deepcopy(released.get(key)) for key in
            ("answer", "evidence", "limitation", "error", "revision", "budget")},
        "artifact": deepcopy(artifact), "workspace": workspace, "source_check": deepcopy(source_check),
        "usage": {"total_usage": deepcopy(scratch.total_usage), "per_call": deepcopy(scratch.call_usage),
                  "raw_accounting": raw, "observer": _tagged(call_state)},
        "binding": binding, "workspace_sha256": SingleAgent._workspace_sha256(workspace),
        "trace_sha256": fingerprint(trace), "tagged_trace_sha256": fingerprint(tagged_trace),
        "diagnostics": deepcopy(diagnostic)}
    return snapshot, diagnostic


def selector_call(host, messages, reservation, completion=2000):
    """One intact request, never Base's automatic message compaction."""
    prompt = base_module.count_tokens_in_messages(messages)
    needed = reserve(prompt) + completion
    diagnostic = {"phase": "selection", "tokens": 0, "tool_calls": 0, "model_calls": 0,
        "fatal": False, "errors": [], "prompt_tokens_estimate": prompt, "reservation": reservation}
    if (prompt > 32000 or needed > reservation or needed > host.max_total_tokens - host.total_usage["total_tokens"]
            or host.total_usage["llm_calls"] >= SDK_CAP):
        diagnostic["status"] = "known_preprovider_refusal"
        return None, diagnostic
    local = clean_ledger()
    try:
        response, error, usage = base_module.safe_llm_call(model=host.model, messages=messages,
            tools=None, base_url=host.base_url, max_completion_tokens=completion, max_retries=1)
        if usage is None and response is None and isinstance(error, str) and error.startswith("Input tokens exceeded:"):
            diagnostic["status"] = "known_preprovider_refusal"
            return None, diagnostic
        if usage:
            BaseAgent._update_usage(local, usage)
        if not isinstance(usage, dict) or type(usage.get("llm_calls")) is not int or usage.get("llm_calls") != 1:
            mark_unknown(local, (0, 0))
            diagnostic["errors"].append("selector invocation usage is missing or not one attempt")
        if isinstance(usage, dict) and type(usage.get("completion_tokens")) is int and usage["completion_tokens"] > completion:
            diagnostic["errors"].append("selector completion ceiling overshoot")
        diagnostic["provider_error"] = error
    except Exception as exc:
        mark_unknown(local, (0, 0))
        response, error = None, "selector invocation raised " + type(exc).__name__
        diagnostic["errors"].append(error)
    scratch, raw, problems, unknown = validated_ledger(local)
    diagnostic["errors"].extend(problems)
    if unknown:
        diagnostic["errors"].append("selector provider usage unknown")
    if scratch.total_usage["total_tokens"] > needed:
        diagnostic["errors"].append("selector reserved ceiling overshoot")
    diagnostic["errors"].extend(merge_ledger(host, scratch, "selection"))
    diagnostic.update(tokens=scratch.total_usage["total_tokens"], model_calls=scratch.total_usage["llm_calls"],
        fatal=bool(diagnostic["errors"]), raw_accounting=raw,
        status="provider_failed" if error else "responded")
    return response if not error and not diagnostic["fatal"] else None, diagnostic
