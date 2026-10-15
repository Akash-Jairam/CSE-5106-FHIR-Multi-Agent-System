"""V8 native Single capture and intact, explicitly bounded auxiliary requests.

No cache, provider setup, retrieval or model invocation happens at import time.
The primary calls the unchanged Single policy; only the auxiliary has this
different request/reservation policy. Source checks establish lineage, not truth.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib

from . import base_agent as base_module
from .base_agent import BaseAgent
from .continuous_primary import _binding, _tagged
from .residual_trajectory_phase import (clean_ledger, fingerprint, mark_unknown,
    message_dict, merge_ledger, reserve, validated_ledger)
from .single_agent import SingleAgent

GLOBAL_TOKENS = 200000
GLOBAL_TOOLS = 30
GLOBAL_SDK = 30
AUX_TOKENS = 60000
AUX_TOOLS = 6
AUX_SDK = 6
INPUT_TOKENS = 32000
CALL_COMPLETION = 8192
JUDGE_COMPLETION = 2000


def source_identity():
    """Only fixed public inference files; never inspect configuration or Git."""
    root = Path(__file__).resolve().parent
    names = ('single_agent.py', 'base_agent.py', 'continuous_primary.py',
        'residual_trajectory_phase.py', 'residual_trajectory_agent.py',
        'counterexample_investigator_agent.py', 'counterexample_investigator_phase.py')
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def caps(host):
    return {'max_total_tokens': min(GLOBAL_TOKENS, host.max_total_tokens)
                if host.max_total_tokens is not None else GLOBAL_TOKENS,
            'max_tool_calls': min(GLOBAL_TOOLS, host.max_tool_calls)
                if host.max_tool_calls is not None else GLOBAL_TOOLS}


def snapshot(single, released, phase, candidate, token_cap, tool_cap, sdk_cap,
             errors=(), observer=None):
    scratch, raw, problems, unknown = validated_ledger(single)
    errors = [*errors, *problems]
    if unknown:
        errors.append('phase provider usage unknown')
    if (scratch.total_usage['total_tokens'] > token_cap or scratch.tool_calls_made > tool_cap
            or scratch.total_usage['llm_calls'] > sdk_cap):
        errors.append('phase actual ceiling overshoot')
    trace = [message_dict(item) for item in (released.get('trace') or single.messages)]
    workspace = deepcopy(released.get('retrieved_fhir_resources') or {})
    artifact, reason, binding = _binding(released, deepcopy(single.final_computation), workspace, trace)
    source_check = binding.pop('source_check', None) if artifact else None
    if errors:
        artifact, reason, binding, source_check = None, 'phase accounting fatal', {}, None
    tagged_trace = [{**deepcopy(item), 'phase': phase} for item in trace]
    diagnostic = {'phase': phase, 'token_cap': token_cap, 'tool_cap': tool_cap,
        'sdk_cap': sdk_cap, 'tokens': scratch.total_usage['total_tokens'],
        'tool_calls': scratch.tool_calls_made, 'model_calls': scratch.total_usage['llm_calls'],
        'fatal': bool(errors), 'usage_known': not unknown, 'errors': errors,
        'eligible': artifact is not None, 'skip_reason': reason,
        'budget_stop': single.budget_stop,
        'released_latency_seconds': released.get('latency')}
    value = {'version': 1, 'kind': 'candidate_snapshot', 'candidate': candidate,
        'phase': phase, 'released': {key: deepcopy(released.get(key)) for key in
            ('answer', 'evidence', 'limitation', 'error', 'revision', 'budget')},
        'artifact': deepcopy(artifact), 'workspace': workspace,
        'source_check': deepcopy(source_check), 'binding': binding,
        'workspace_sha256': SingleAgent._workspace_sha256(workspace),
        'trace_sha256': fingerprint(trace), 'tagged_trace_sha256': fingerprint(tagged_trace),
        'usage': {'total_usage': deepcopy(scratch.total_usage),
            'per_call': deepcopy(scratch.call_usage), 'raw_accounting': raw,
            'observer': _tagged(observer or {})}, 'diagnostics': diagnostic}
    return value, tagged_trace, scratch


def capture_primary(host, question):
    limits = caps(host)
    single = SingleAgent(host.model, max_iterations=GLOBAL_SDK,
                        verbose=host.verbose, base_url=host.base_url)
    single.set_budget(**limits)
    single.set_perturbation(host.perturbation)
    errors = []
    try:
        released = single.run(question)
    except Exception as exc:
        errors.append('primary formatter failed: ' + type(exc).__name__)
        released = {'answer': None, 'evidence': [], 'limitation': None,
            'error': 'Primary output formatting failed', 'trace': single.messages,
            'retrieved_fhir_resources': {}}
    value, trace, _ = snapshot(single, released, 'natural_primary', 'A',
        limits['max_total_tokens'], limits['max_tool_calls'], GLOBAL_SDK, errors)
    payload = {'question_sha256': fingerprint(question), 'model': host.model,
        'base_url_sha256': fingerprint(host.base_url),
        'perturbation_sha256': fingerprint(host.perturbation), 'caps': limits,
        'source_identity': source_identity(), 'candidate': value, 'trace': trace,
        'model_messages': [message_dict(item) for item in getattr(single, 'model_messages', [])],
        'context_compactions': deepcopy(single.context_compactions)}
    return {'version': 1, 'kind': 'v8_primary_envelope', 'payload': payload,
            'sha256': fingerprint(payload)}


def restore_primary(host, question, envelope):
    value = deepcopy(envelope)
    if (value.get('version') != 1 or value.get('kind') != 'v8_primary_envelope'
            or value.get('sha256') != fingerprint(value.get('payload'))):
        raise ValueError('Primary envelope identity differs')
    payload = value['payload']
    if (payload['question_sha256'] != fingerprint(question) or payload['model'] != host.model
            or payload['base_url_sha256'] != fingerprint(host.base_url)
            or payload['caps'] != caps(host)
            or payload['perturbation_sha256'] != fingerprint(host.perturbation)
            or payload['source_identity'] != source_identity()):
        raise ValueError('Primary input/model/cap/source binding differs')
    a, trace = payload['candidate'], payload['trace']
    natural_trace = [{key: item for key, item in message.items() if key != 'phase'} for message in trace]
    if (a['candidate'] != 'A' or a['phase'] != 'natural_primary'
            or a['workspace_sha256'] != SingleAgent._workspace_sha256(a['workspace'])
            or a['trace_sha256'] != fingerprint(natural_trace)
            or a['tagged_trace_sha256'] != fingerprint(trace)):
        raise ValueError('Primary snapshot trace/workspace differs')
    if a['artifact'] is not None:
        artifact, _, binding = _binding(a['released'], a['artifact'], a['workspace'], natural_trace)
        source_check = binding.pop('source_check', None) if artifact else None
        if artifact is None or binding != a['binding'] or source_check != a['source_check']:
            raise ValueError('Primary released execution/source binding differs')
    local = SimpleNamespace(total_usage=deepcopy(a['usage']['total_usage']),
        call_usage=deepcopy(a['usage']['per_call']), tool_calls_made=a['diagnostics']['tool_calls'],
        context_compactions=payload['context_compactions'])
    scratch, _, problems, unknown = validated_ledger(local)
    # Fatal first attempts remain materializable; a clean eligibility assertion
    # cannot conceal malformed or unavailable accounting.
    if (problems or unknown) and not a['diagnostics']['fatal']:
        raise ValueError('Primary incorrectly claims known clean accounting')
    merge_errors = merge_ledger(host, scratch, 'natural_primary')
    if merge_errors:
        a['diagnostics']['fatal'] = True
        a['diagnostics']['errors'].extend(merge_errors)
    host.messages.extend(deepcopy(trace))
    return a, payload


def bounded_call(local, messages, tools, state, *, reserve_tokens=0,
                 reserve_sdk=0, completion=CALL_COMPLETION, phase='investigation'):
    """One real SDK attempt, no retries/compaction or implicit extra send.

    A known admission refusal is free; every admitted exception/missing usage is
    charged as unknown and fatal. Known billed provider errors can preserve A.
    """
    prompt = base_module.count_tokens_in_messages(messages, tools)
    available = local.max_total_tokens - local.total_usage['total_tokens'] - reserve_tokens
    output_cap = min(completion, available - reserve(prompt))
    diagnostic = {'phase': phase, 'prompt_tokens_estimate': prompt,
        'reserved_future_tokens': reserve_tokens, 'completion_cap': max(0, output_cap),
        'reserved_future_sdk': reserve_sdk, 'status': 'known_preprovider_refusal'}
    state['calls'].append(diagnostic)
    if (prompt > INPUT_TOKENS or output_cap < 1
            or local.total_usage['llm_calls'] >= state['sdk_cap'] - reserve_sdk):
        state['stop'] = 'intact request/future reservation unavailable'
        return None, state['stop']
    before = (local.total_usage['llm_calls'], len(local.call_usage))
    before_tokens = local.total_usage['total_tokens']
    try:
        response, error, usage = base_module.safe_llm_call(model=local.model,
            messages=messages, tools=tools, base_url=local.base_url,
            max_completion_tokens=output_cap, max_retries=1)
        if (response is None and usage is None and isinstance(error, str)
                and error.startswith('Input tokens exceeded:')):
            state['stop'] = 'provider adapter known input refusal'
            return None, state['stop']
        if usage:
            BaseAgent._update_usage(local, usage)
        if not isinstance(usage, dict) or type(usage.get('llm_calls')) is not int or usage['llm_calls'] != 1:
            mark_unknown(local, before)
            state['errors'].append('admitted SDK usage is missing or not one attempt')
        if (isinstance(usage, dict) and type(usage.get('completion_tokens')) is int
                and usage['completion_tokens'] > output_cap):
            state['errors'].append('reported completion exceeds requested ceiling')
        if local.total_usage['total_tokens'] - before_tokens > reserve(prompt) + output_cap:
            state['errors'].append('reported invocation exceeds reserved ceiling')
        diagnostic['status'] = 'provider_failed' if error else 'responded'
        diagnostic['provider_error_present'] = bool(error)
    except Exception as exc:
        mark_unknown(local, before)
        response, error = None, 'admitted SDK exception: ' + type(exc).__name__
        diagnostic['status'] = 'provider_exception'
        state['errors'].append(error)
    scratch, raw, problems, unknown = validated_ledger(local)
    state['errors'].extend(problems)
    if unknown:
        state['errors'].append('provider usage unknown')
    if (scratch.total_usage['total_tokens'] > local.max_total_tokens
            or scratch.total_usage['llm_calls'] > state['sdk_cap']
            or scratch.tool_calls_made > local.max_tool_calls):
        state['errors'].append('auxiliary actual ceiling overshoot')
    if problems:
        state['malformed_raw_accounting'] = raw
        local.total_usage, local.call_usage = scratch.total_usage, scratch.call_usage
    state['fatal'] = bool(state['errors'])
    if error:
        state['stop'] = 'provider request failed'
    return response if not error and not state['fatal'] else None, error
