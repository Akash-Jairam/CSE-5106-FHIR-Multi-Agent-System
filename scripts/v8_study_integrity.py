"""Postterminal V8 study integrity, without scorer, QA parsing or inference.

An externally pinned observed-process-exit statement is required. The four
policies share physical A once; provider observations are reconciled against A
once plus each suffix, never against four duplicated conceptual ledgers. The
check establishes retained execution/source/accounting provenance, not truth.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]
SHARED_PATH = ROOT / 'benchmarks/design-v7-20261015/integrity/v7_guard.py'
SHARED_SHA = 'bf47aa9e9c84735b0a096acbb6d209776473fcb7e81e8cfa2e366576d90e1115'
BASE_SHA = '98d272f8ed8e42914b5ed1776908cf6e25e209e5726d5386b4c168a270a499e8'
META = ('manifest.json', 'runtime_adapter_manifest.json', 'expected_public_binding.json',
        'prospective_protocol.json', 'launch_commands.json')
RUNTIME = {'.local-setup/azure_adapter.py', '.local-setup/sitecustomize.py',
           '.local-setup/azure.py', '.local-setup/phase3.py'}
REQUIRED = {'agent/counterexample_investigator_agent.py', 'agent/counterexample_investigator_phase.py',
            'agent/continuous_primary.py', 'agent/residual_trajectory_agent.py', 'agent/residual_trajectory_phase.py',
            'agent/single_agent.py', 'agent/base_agent.py', 'scripts/v8_mechanism_study.py',
            'scripts/v8_study_cache.py', 'scripts/v8_study_integrity.py',
            'scripts/launch_v8_study.py', 'scripts/audit_store.py', 'utils/answer_scorer.py'}
FIELDS = ('answer', 'evidence', 'operation', 'contributing_ids', 'limitation')
RENDERED = ('answer', 'evidence', 'limitation')
PROTOCOL = 'full-single-counterexample-investigator-v8'
MODES = ('A', 'W', 'R', 'C')
LEGACY_STUDY_MANIFEST = 'manifests/v8_mechanism_120_20261015.json'
HEX = re.compile(r'^[a-f0-9]{64}$')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def text(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(text(value).encode()).hexdigest()


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def equal(left, right):
    return text(left) == text(right)


def load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def integer(value):
    return type(value) is int and value >= 0


def shared():
    require(sha(SHARED_PATH) == SHARED_SHA, 'Reviewed common integrity helper differs')
    spec = importlib.util.spec_from_file_location('v8_reviewed_storage_source_journal', SHARED_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.BASE_SHA == BASE_SHA, 'Reviewed storage/journal dependency differs')
    return module


def owned(output, name):
    path = Path(output) / name
    require(not Path(name).is_absolute() and '..' not in Path(name).parts and not path.is_symlink()
            and path.is_file() and path.resolve().is_relative_to(Path(output).resolve()), 'Unsafe owned file')
    return path


def input_text(row):
    value = 'Question: ' + row['question'] + '\nContext:\nPatient FHIR ID is ' + row['patient_fhir_id'] + '.'
    assumption = row.get('assumption')
    if assumption is not None and not (type(assumption) is float and math.isnan(assumption)):
        value += '\n' + str(assumption)
    return value


def ledger(totals, calls, fatal, cap, tool_calls):
    require(isinstance(totals, dict) and isinstance(calls, list) and type(fatal) is bool, 'Ledger envelope malformed')
    require(all(integer(totals.get(k)) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'llm_calls'))
            and integer(tool_calls), 'Verified numeric counters malformed')
    require(all(isinstance(c, dict) and integer(c.get('prompt')) and integer(c.get('completion'))
                and type(c.get('usage_available')) is bool for c in calls), 'Projected per-call ledger malformed')
    unknown = any(c['usage_available'] is False for c in calls)
    invalid_cache = any('cached' in c and (not integer(c['cached']) or c['cached'] > c['prompt']) for c in calls)
    over = totals['total_tokens'] > cap[0] or tool_calls > cap[1] or totals['llm_calls'] > cap[2]
    if unknown or over or invalid_cache:
        require(fatal, 'Unknown usage or reported overshoot falsely nonfatal')
    if unknown:
        require(totals.get('cost') is None, 'Unknown usage has falsely known cost')
    if not fatal:
        require(totals['prompt_tokens'] + totals['completion_tokens'] == totals['total_tokens']
                and len(calls) == totals['llm_calls']
                and sum(c['prompt'] for c in calls) == totals['prompt_tokens']
                and sum(c['completion'] for c in calls) == totals['completion_tokens'], 'Nonfatal ledger unreconciled')
    cost = totals.get('cost')
    require(cost is None or type(cost) in (int, float) and math.isfinite(cost) and cost >= 0, 'Cost malformed')
    return Counter(unknown_calls=sum(c['usage_available'] is False for c in calls),
                   invalid_cache_views=int(invalid_cache),
                   reported_phase_overshoots=int(over), charged_minus_retained_call_entries=totals['llm_calls']-len(calls))


def observed_counts(trace):
    responses = tools = 0
    for message in trace:
        calls = message.get('tool_calls') or []
        automatic = (message.get('role') == 'assistant' and message.get('content') is None and len(calls) == 1
                     and calls[0].get('id') in {'automatic_field_check_1', 'automatic_field_check_2'}
                     and (calls[0].get('function') or {}).get('name') == 'check_fields')
        responses += message.get('role') == 'assistant' and not automatic
        tools += message.get('role') == 'tool' and not str(message.get('content', '')).startswith('Not run:')
    return responses, tools


def validate_candidate(snapshot, tagged_trace):
    """Bind the actual released candidate; no program/check is re-executed."""
    helper = shared()
    require(isinstance(snapshot, dict) and snapshot.get('version') == 1
            and snapshot.get('kind') == 'candidate_snapshot', 'Candidate snapshot differs')
    label, phase = snapshot.get('candidate'), snapshot.get('phase')
    require((label, phase) in {('A', 'natural_primary'), ('B', 'counterexample_investigation')}, 'Candidate phase differs')
    require(all(isinstance(m, dict) and m.get('phase') == phase for m in tagged_trace), 'Candidate trace phase differs')
    natural = [{k: v for k, v in m.items() if k != 'phase'} for m in tagged_trace]
    require(snapshot.get('trace_sha256') == fingerprint(natural)
            and snapshot.get('tagged_trace_sha256') == fingerprint(tagged_trace), 'Candidate trace SHA differs')
    workspace = snapshot.get('workspace')
    require(isinstance(workspace, dict) and snapshot.get('workspace_sha256') == fingerprint(workspace), 'Candidate own-workspace SHA differs')
    d, usage = snapshot.get('diagnostics'), snapshot.get('usage')
    require(isinstance(d, dict) and isinstance(usage, dict) and d.get('phase') == phase
            and all(type(d.get(k)) is bool for k in ('fatal', 'eligible', 'usage_known')), 'Candidate diagnostics malformed')
    for key in ('tokens', 'tool_calls', 'model_calls', 'token_cap', 'tool_cap', 'sdk_cap'):
        require(integer(d.get(key)), 'Candidate cap/counter malformed')
    limits = (200000, 30, 30) if label == 'A' else (60000, 6, 6)
    caps = tuple(d[k] for k in ('token_cap', 'tool_cap', 'sdk_cap'))
    require(all(value <= limit for value, limit in zip(caps, limits)), 'Candidate expands policy caps')
    if label == 'A':
        require(caps == limits, 'Primary must retain full original caps')
    totals, calls = usage.get('total_usage'), usage.get('per_call')
    counts = ledger(totals, calls, d['fatal'], caps, d['tool_calls'])
    require(totals['total_tokens'] == d['tokens'] and totals['llm_calls'] == d['model_calls']
            and d['usage_known'] is (not any(c['usage_available'] is False for c in calls)), 'Candidate diagnostics/ledger differ')
    if not d['fatal']:
        raw = usage.get('raw_accounting') or {}
        require(equal(raw.get('total_usage'), totals) and equal(raw.get('call_usage'), calls)
                and raw.get('tool_calls_made') == d['tool_calls'], 'Candidate raw accounting differs')
    responses, tools = observed_counts(natural)
    counts.update(retained_model_responses=responses, retained_charged_tool_replies=tools,
                  charged_minus_observed_model_attempts=d['model_calls']-responses,
                  charged_minus_observed_tool_calls=d['tool_calls']-tools)
    artifact, released = snapshot.get('artifact'), snapshot.get('released')
    require(isinstance(released, dict) and all(k in released for k in (*RENDERED, 'error')), 'Actual released candidate unavailable')
    require(d['eligible'] is (artifact is not None), 'Eligibility/artifact mismatch')
    if artifact is None:
        require(snapshot.get('source_check') is None and snapshot.get('binding') == {}, 'Unreleased candidate falsely bound')
        return counts
    require(not d['fatal'] and isinstance(artifact, dict) and all(k in artifact for k in FIELDS)
            and not released.get('error') and equal({k: artifact[k] for k in RENDERED},
                                                  {k: released[k] for k in RENDERED}), 'Computed-withheld candidate substituted')
    revision = released.get('revision') or {}
    require(revision.get('final_check_issues') == []
            and equal({k: (revision.get('final') or {}).get(k) for k in RENDERED},
                      {k: artifact[k] for k in RENDERED}), 'Actual source-clean release differs')
    known_failed = 0
    if label == 'B':
        observer = usage.get('observer') or {}
        invocations = observer.get('calls')
        require(isinstance(invocations, list) and all(isinstance(c, dict) and c.get('status') in
                {'known_preprovider_refusal', 'responded', 'provider_failed'} for c in invocations),
                'Eligible B invocation observer missing or malformed')
        known_failed = sum(c['status'] == 'provider_failed' and c.get('provider_error_present') is True
                           for c in invocations)
        require(sum(c['status'] != 'known_preprovider_refusal' for c in invocations) == d['model_calls'],
                'Eligible B admitted invocation ledger differs')
    require(responses + known_failed == d['model_calls'] and tools == d['tool_calls'],
            'Eligible candidate lacks complete charged-call evidence')
    five = {k: artifact[k] for k in FIELDS}
    code = helper.validate_primary(natural, five, workspace)
    require(artifact.get('code') == code and artifact.get('workspace_sha256') == snapshot['workspace_sha256'], 'Artifact code/workspace differs')
    binding = snapshot.get('binding') or {}
    ei, ci = binding.get('execution_trace_index'), binding.get('source_check_trace_index')
    require(integer(ei) and integer(ci) and ei < ci < len(natural), 'Source binding chronology differs')
    require(binding.get('code_sha256') == hashlib.sha256(code.encode()).hexdigest()
            and binding.get('workspace_sha256') == snapshot['workspace_sha256'], 'Source binding hashes differ')
    require(natural[ei].get('tool_call_id') == binding.get('execution_tool_call_id')
            and natural[ci].get('tool_call_id') == binding.get('source_check_tool_call_id'), 'Source binding own-call IDs differ')
    check = helper.json_object(natural[ci].get('content'))
    require(equal(check, snapshot.get('source_check')), 'Retained original source check differs')
    helper._source_summary(five, check, workspace)
    return counts


def validate_policy(row, restored, envelope):
    helper = shared()
    require(not row.get('attempt_history') and row.get('study_terminal') is True, 'Only retained terminal first outcomes allowed')
    revision = restored.get('revision')
    require(isinstance(revision, dict) and revision.get('protocol') == PROTOCOL
            and revision.get('mode') == row.get('study_arm'), 'V8 policy revision identity differs')
    snapshots = revision.get('messages')
    require(isinstance(snapshots, list) and 1 <= len(snapshots) <= 2, 'Candidate snapshot inventory differs')
    a = snapshots[0]
    require(equal(a, envelope['payload']['candidate']), 'Policy replaced immutable actual A')
    trace = restored.get('trace')
    require(isinstance(trace, list) and equal(trace[:len(envelope['payload']['trace'])], envelope['payload']['trace']), 'Policy A trace prefix differs')
    require(revision.get('candidate_snapshot_sha256') == {s['candidate']: fingerprint(s) for s in snapshots}, 'Snapshot hash index differs')
    phases = revision.get('phases')
    require(isinstance(phases, list) and 1 <= len(phases) <= 2 and equal(phases[0], a['diagnostics']), 'Phase primary diagnostics differ')
    counts = validate_candidate(a, envelope['payload']['trace'])
    calls = deepcopy(a['usage']['per_call'])
    totals = deepcopy(a['usage']['total_usage'])
    tools = a['diagnostics']['tool_calls']
    b = snapshots[1] if len(snapshots) == 2 else None
    b_trace = [m for m in trace if m.get('phase') == 'counterexample_investigation']
    if b is not None:
        counts.update(validate_candidate(b, b_trace))
        initial = next((m for m in b_trace if m.get('role') == 'user'), None)
        require(initial is not None and isinstance(initial.get('content'), str), 'Investigation input packet missing')
        packet = helper.json_object(initial['content'].splitlines()[-1])
        context = helper.context_from_row({**row, 'question_with_context': input_text(row)})
        inventory = [{'resource_type': name, 'held_count': len(values)} for name, values in sorted(a['workspace'].items())
                     if isinstance(values, list)]
        require(equal(packet.get('context'), context)
                and equal(packet.get('A'), helper.candidate_entry(a, envelope['payload']['trace']))
                and equal(packet.get('workspace_inventory'), inventory)
                and packet.get('retrieval_permitted') is (row['study_arm'] != 'W'), 'Investigation actual A/context/permission packet differs')
    else:
        require(not b_trace, 'Policy has unretained investigator snapshot')
    aux = phases[1] if len(phases) == 2 else None
    if aux is not None:
        require(aux.get('phase') == 'counterexample_auxiliary' and b is not None, 'Auxiliary phase/snapshot missing')
        admission = revision.get('admission') or {}
        caps = tuple(aux.get(k) for k in ('token_cap', 'tool_cap', 'sdk_cap'))
        require(caps == tuple(admission.get(k) for k in ('aux_token_cap', 'aux_tool_cap', 'aux_sdk_cap'))
                and caps == (min(60000, 200000-totals['total_tokens']), min(6, 30-tools), min(6, 30-totals['llm_calls'])), 'Actual residual auxiliary caps differ')
        raw = aux.get('raw_accounting') or {}
        extra, suffix = raw.get('total_usage'), raw.get('call_usage')
        counts.update(ledger(extra, suffix, aux.get('fatal'), caps, aux.get('tool_calls')))
        require(extra['total_tokens'] == aux.get('tokens') and extra['llm_calls'] == aux.get('model_calls')
                and raw.get('tool_calls_made') == aux.get('tool_calls'), 'Auxiliary aggregate/raw counters differ')
        require(equal(suffix[:len(b['usage']['per_call'])], b['usage']['per_call']), 'Investigator ledger is not exact auxiliary prefix')
        require(integer(aux.get('probe_interactions')) and aux['probe_interactions'] <= 2
                and integer(aux.get('automatic_checks')) and aux['automatic_checks'] <= 2, 'Auxiliary interaction cap differs')
        invocations = aux.get('calls')
        require(isinstance(invocations, list) and all(isinstance(c, dict) and c.get('status') in
                {'known_preprovider_refusal', 'responded', 'provider_failed', 'provider_exception'} for c in invocations),
                'Auxiliary admission diagnostics malformed')
        if not aux['fatal']:
            require(sum(c['status'] != 'known_preprovider_refusal' for c in invocations) == aux['model_calls'],
                    'Auxiliary admitted invocation count differs')
        costs = totals.get('cost'), extra.get('cost')
        total_cost = None if None in costs else sum(costs)
        totals['cost'] = total_cost if total_cost is None or math.isfinite(total_cost) else None
        for k in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'llm_calls'):
            totals[k] += extra[k]
        calls.extend(deepcopy(suffix));tools += aux['tool_calls']
    else:
        require(b is None, 'Investigator snapshot lacks retained aggregate phase')
    fatal = revision.get('fatal')
    require(type(fatal) is bool, 'Fatal flag unavailable')
    counts.update(ledger(totals, calls, fatal, (200000, 30, 30), tools))
    if any(p.get('fatal') for p in phases):
        require(fatal, 'Phase accounting fatal was hidden')
    if fatal:
        require(helper.B.failed(row), 'Whole-fatal policy lacks explicit terminal error')
    tokens = row.get('tokens') or {}
    require(all(equal(tokens.get(k), totals[v]) for k, v in
                [('prompt', 'prompt_tokens'), ('completion', 'completion_tokens'), ('total', 'total_tokens'), ('llm_calls', 'llm_calls')])
            and equal(tokens.get('per_call'), calls) and equal(tokens.get('cost'), totals.get('cost')),
            'Conceptual A-plus-suffix ledger differs')
    budget = row.get('budget') or {}
    require(budget.get('max_total_tokens') == 200000 and budget.get('max_tool_calls') == 30
            and budget.get('tool_calls_run') == tools, 'Aggregate budget/charged tools differ')
    selected = revision.get('selected')
    require(selected in {'A', 'B'} and revision.get('alternative_selected') is (selected == 'B'), 'Actual selected candidate differs')
    if not fatal:
        chosen = a if selected == 'A' else b
        require(chosen is not None and equal({k: row.get(k) for k in ('evidence', 'limitation', 'error')},
                                            {k: chosen['released'].get(k) for k in ('evidence', 'limitation', 'error')})
                and equal(row.get('agent_answer'), chosen['released'].get('answer')), 'Final is not actual released A/B')
    mechanism = revision.get('mechanism')
    expected_b = {k: b['artifact'][k] for k in FIELDS} if b is not None and b.get('artifact') is not None else None
    require(isinstance(mechanism, dict) and equal(row.get('mechanism'), mechanism)
            and equal(mechanism.get('released_source_bound_B'), expected_b)
            and mechanism.get('selected_B') is (selected == 'B' and not fatal)
            and mechanism.get('auxiliary_fatal') is (fatal and revision.get('status') != 'fatal_primary')
            and mechanism.get('status') == revision.get('status'), 'Explicit mechanism projection differs')
    observations, witness = revision.get('observations'), revision.get('witness')
    require(isinstance(observations, list) and len(observations) <= 2
            and mechanism.get('probe_interactions') == len(observations)
            and mechanism.get('witness_bound') is (witness is not None), 'Observation/mechanism counts differ')
    requests = {}
    replies = {}
    for message in b_trace:
        for call in message.get('tool_calls') or []:
            require(call.get('id') not in requests, 'Investigation reuses native tool-call ID')
            requests[call.get('id')] = call
        if message.get('role') == 'tool':
            require(message.get('tool_call_id') not in replies, 'Investigation reuses tool reply ID')
            replies[message.get('tool_call_id')] = message
    for i, observation in enumerate(observations, 1):
        identifier = observation.get('tool_call_id');call = requests.get(identifier) or {}
        function = call.get('function') or {}
        require(observation.get('observation_id') == 'observation_' + str(i)
                and observation.get('tool') == function.get('name')
                and equal(observation.get('arguments'), helper.json_object(function.get('arguments')))
                and equal(observation.get('actual_reply'), {k: v for k, v in (replies.get(identifier) or {}).items()
                                                         if k not in {'phase', 'model_content'}})
                and observation.get('A_program_sha256') == a['binding'].get('code_sha256')
                and observation.get('A_workspace_sha256') == a['workspace_sha256'], 'Observation is not an actual owned probe')
        if row['study_arm'] == 'W':
            require(observation['tool'] == 'execute_python_code', 'W performed a retrieval probe')
        reply = observation['actual_reply'];content = reply.get('content', '')
        try:
            value = json.loads(content)
        except (TypeError, ValueError):
            value = None
        success = isinstance(value, dict) and not value.get('error')
        if observation['tool'] in {'get_resources_by_patient_fhir_id', 'get_resources_by_resource_id'} and value is None:
            success = bool(re.fullmatch(r"retrieved_resources\['[^']+'\] \(\d+ items; \d+ fetched\)", content))
        truncated = isinstance(value, dict) and any(value.get(k) for k in
                    ('truncated', 'stdout_truncated', 'answer_truncated', 'output_truncated'))
        eligible = bool(success and not truncated)
        if observation['tool'] == 'execute_python_code' and observation.get('workspace_before_sha256') != observation.get('workspace_after_sha256'):
            eligible = False
        if len(text(observation).encode()) > 8192:
            eligible = False
        require(observation.get('witness_eligible') is eligible, 'Failed/truncated/mutating probe falsely certifies a witness')
    if witness is not None:
        require(isinstance(witness, dict) and isinstance(witness.get('obligation_span'), str)
                and witness['obligation_span'] and witness['obligation_span'] in input_text(row), 'Witness literal obligation differs')
        observation = next((o for o in observations if o['observation_id'] == witness.get('observation_id')), None)
        require(observation is not None and observation.get('witness_eligible') is True
                and witness.get('observation_sha256') == fingerprint(observation)
                and witness.get('A_snapshot_sha256') == fingerprint(a)
                and witness.get('A_program_sha256') == a['binding'].get('code_sha256')
                and witness.get('A_workspace_sha256') == a['workspace_sha256'], 'Witness actual-observation/A binding differs')
    judge = [m for m in trace if m.get('phase') == 'counterexample_judge']
    selection = revision.get('selection')
    if selection is not None:
        require(len(judge) == 3 and [m.get('role') for m in judge] == ['system', 'user', 'assistant']
                and not judge[-1].get('tool_calls') and equal(helper.json_object(judge[-1].get('content')), selection), 'Selection lacks actual tool-free response')
        helper.validate_selection({k: selection[k] for k in ('requirements', 'choice', 'reason')})
        require(witness is not None and selection.get('obligation_span') == witness['obligation_span']
                and selection.get('observation_id') == witness['observation_id'], 'Selection witness differs')
        packet = helper.json_object(judge[1].get('content'))
        require(set(packet) == {'context', 'A', 'B', 'witness', 'observation'}
                and equal(packet.get('context'), helper.context_from_row({**row, 'question_with_context': input_text(row)}))
                and equal(packet.get('A'), helper.candidate_entry(a, envelope['payload']['trace']))
                and equal(packet.get('B'), helper.candidate_entry(b, b_trace))
                and equal(packet.get('witness'), witness)
                and equal(packet.get('observation'), next(o for o in observations if o['observation_id'] == witness['observation_id']))
                and (revision.get('admission') or {}).get('judge_packet_sha256') == fingerprint(packet), 'Judge did not receive intact actual entries')
    if selected == 'B':
        require(not fatal and b is not None and b.get('artifact') is not None and selection is not None
                and selection['choice'] == 'B' and all(x['B']['status'] == 'supported' for x in selection['requirements'])
                and any(x['A']['status'] == 'contradicted' for x in selection['requirements']), 'B release lacks authorized actual selection')
    if row['study_arm'] == 'A':
        require(aux is None and selected == 'A' and row.get('physical_primary_inference_executed') is True, 'A materialization repeated or replaced inference')
    else:
        require(row.get('physical_primary_inference_executed') is False, 'Followup falsely bills A physically again')
    physical = calls if row['study_arm'] == 'A' else calls[len(a['usage']['per_call']):]
    p = row.get('physical_usage') or {}
    known = [c for c in physical if c['usage_available']]
    require(p.get('available') is True and p.get('observed_invocations') == len(physical)
            and p.get('known_usage_invocations') == len(known)
            and p.get('unknown_usage_invocations') == len(physical)-len(known), 'A-once physical ledger diagnostic differs')
    projected = {}
    for key in ('prompt', 'completion', 'cached'):
        values = [c.get(key) for c in known]
        projected[key] = sum(values) if all(integer(v) for v in values) else None
    require(equal(p.get('known_tokens'), projected), 'Physical known usage differs')
    counts.update(policies=1, fatal_policies=int(fatal), selected_B=int(selected == 'B'), eligible_B=int(expected_b is not None))
    return {'counts': dict(counts), 'physical_calls': physical, 'fatal': fatal}


def cache_inventory(directory):
    require(directory.is_dir() and not directory.is_symlink(), 'Private cache directory missing/unsafe')
    result = {}
    for path in sorted(directory.iterdir()):
        require(path.is_file() and not path.is_symlink() and path.name.endswith('.cache'), 'Unexpected cache file')
        result[path.name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    return result


def verify_case(rows, output_dir, store, expected_case=None, expected_model=None, expected_sources=None):
    helper = shared();output = Path(output_dir).resolve()
    require(len(rows) == 4 and {r.get('study_arm') for r in rows} == set(MODES), 'Case needs four unique first policy outcomes')
    index = {r['study_arm']: r for r in rows};base = index['A'];qid = base['question_id']
    if expected_case is not None:
        require(all(equal(base.get(k), v) for k, v in expected_case.items()), 'Case selection metadata differs')
    for row in rows:
        require(all(equal(row.get(k), base.get(k)) for k in
                    ('question_id', 'patient_fhir_id', 'question', 'assumption', 'true_answer', 'template', 'primary_envelope_sha256', 'primary_envelope_ref')),
                'Case identity/reference/A envelope differs across policies')
    path = owned(output, 'primaries/' + qid + '.json.gz')
    require(base.get('primary_envelope_ref') == str(path.relative_to(output)), 'Primary envelope owner/path differs')
    with gzip.open(path, 'rb') as stream:
        raw = stream.read(helper.B.MAX_AUDIT_BYTES+1)
    require(len(raw) <= helper.B.MAX_AUDIT_BYTES, 'Primary envelope too large')
    envelope = json.loads(raw)
    require(raw == compact(envelope) and hashlib.sha256(raw).hexdigest() == base.get('primary_envelope_sha256'), 'External primary canonical SHA differs')
    require(envelope.get('version') == 1 and envelope.get('kind') == 'v8_primary_envelope'
            and envelope.get('sha256') == fingerprint(envelope.get('payload')), 'Internal primary payload SHA differs')
    payload = envelope['payload']
    require(payload.get('question_sha256') == fingerprint(input_text(base))
            and payload.get('caps') == {'max_total_tokens': 200000, 'max_tool_calls': 30}
            and payload.get('base_url_sha256') == fingerprint(None)
            and payload.get('perturbation_sha256') == fingerprint(None), 'Primary input/caps binding differs')
    if expected_model is not None:
        require(payload.get('model') == expected_model, 'Primary model differs')
    if expected_sources is not None:
        expected = {Path(name).name: value for name, value in expected_sources.items() if name in
                    {'agent/'+name for name in ('single_agent.py', 'base_agent.py', 'continuous_primary.py',
                     'residual_trajectory_phase.py', 'residual_trajectory_agent.py',
                     'counterexample_investigator_agent.py', 'counterexample_investigator_phase.py')}}
        require(len(expected) == 7 and equal(payload.get('source_identity'), expected), 'Primary source-module identity differs')
    counts = Counter();physical = [];files = {str(path.relative_to(output)): sha(path)}
    for mode in MODES:
        row = index[mode]
        restored, _ = helper.verify_restored(row, output, store)
        proof = validate_policy(row, restored, envelope)
        counts.update(proof['counts']);physical.extend(proof['physical_calls'])
        ref = row['audit_artifacts'];files[ref['path']] = sha(owned(output, ref['path']))
    directory = output / 'case_cache' / qid
    inventory = cache_inventory(directory / 'post_A_base')
    cache_sha = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    inodes = set()
    for mode in ('post_A_base', 'W', 'R', 'C'):
        current = cache_inventory(directory / mode)
        require(set(inventory) <= set(current), 'Followup lost original cache entries')
        for name in current:
            p = directory / mode / name
            require(p.stat().st_ino not in inodes, 'Private cache files share inode')
            inodes.add(p.stat().st_ino)
            files[str(p.relative_to(output))] = current[name]['sha256']
        if mode != 'post_A_base':
            cp = index[mode].get('cache_isolation') or {}
            require(cp.get('base_unchanged') is True and cp.get('followup') == mode
                    and cp.get('overlay_files_after') == len(current)
                    and cp.get('new_overlay_files') == len(set(current)-set(inventory)), 'Private followup cache evidence differs')
    for row in rows:
        cp = row.get('cache_isolation') or {}
        require(cp.get('policy') == 'cold_private_A_then_identical_physical_copies_no_hardlinks'
                and cp.get('inventory_sha256') == cache_sha and cp.get('files') == len(inventory)
                and cp.get('bytes') == sum(v['bytes'] for v in inventory.values()), 'Post-A base proof differs')
    return {'counts': dict(counts), 'physical_calls': physical, 'files': files,
            'fatal_physical': any(r.get('revision', {}).get('fatal') for r in rows)}


def validate_journals(output_dir, physical_calls, fatal_physical=False):
    helper = shared();output = Path(output_dir).resolve();summaries = [];ends = []
    directory = output / 'provider-events'
    if directory.exists():
        require(directory.is_dir() and not directory.is_symlink(), 'Provider journal directory unsafe')
        for path in sorted(directory.iterdir()):
            summary = helper.B.parse_journal(path, output)
            require(summary['pairing_complete'], 'Terminal provider journal has unmatched/pending events')
            summaries.append(summary)
            for line in path.read_text().splitlines():
                event = json.loads(line)
                if event['event'] == 'end':
                    ends.append(event)
    known = Counter((c['prompt'], c['completion']) for c in physical_calls if c['usage_available'])
    observed = Counter((e['usage']['prompt_tokens'], e['usage']['completion_tokens']) for e in ends
                       if type(e['usage']['prompt_tokens']) is int and type(e['usage']['completion_tokens']) is int)
    unknown = sum(c['usage_available'] is False for c in physical_calls)
    missing_usage = len(ends)-sum(observed.values())
    quarantined = any(c.get('accounting_quarantined') is True for c in physical_calls)
    if not fatal_physical:
        require(known == observed and unknown == 0 and missing_usage == 0
                and len(ends) == len(physical_calls), 'Nonfatal physical ledger/journal mismatch')
    elif not quarantined:
        require(not (known-observed) and len(ends) <= len(physical_calls)
                and sum((observed-known).values()) + missing_usage <= unknown,
                'Fatal journal discrepancy exceeds explicitly unknown admitted calls')
    return {'journals': summaries, 'physical_ledger_calls': len(physical_calls), 'observed_sdk_ends': len(ends),
            'known_usage_calls': sum(known.values()), 'unknown_usage_calls': unknown,
            'journal_usage_unavailable': missing_usage, 'ledger_minus_observed_sdk_calls': len(physical_calls)-len(ends),
            'known_ledger_minus_known_journal_calls': sum((known-observed).values()),
            'known_journal_without_known_ledger_calls': sum((observed-known).values()),
            'quarantined_accounting_unverifiable': quarantined,
            'limits': 'Fatal paths retain pre-send/admitted accounting gaps; no unknown usage is zero billing.'}


def bound_study_manifest(root, record, sources, pins=None):
    """The pinned prospective protocol, never a CLI override, names selection.

    The original prospective protocol already declares study_manifest and its
    SHA. New pins additionally carry that path; old pins without a path remain
    valid only for the original filename. No old archive is rewritten.
    """
    root, record = Path(root).resolve(), Path(record).resolve()
    prospective = load(record/'prospective_protocol.json')
    name = prospective.get('study_manifest')
    require(isinstance(name, str), 'Prospective study manifest path malformed')
    if 'study_manifest_path' in prospective:
        require(prospective['study_manifest_path'] == name, 'Ambiguous prospective study manifest aliases')
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and len(relative.parts) >= 2
            and relative.parts[0] == 'manifests' and relative.suffix == '.json'
            and '..' not in relative.parts and '\\' not in name
            and relative.as_posix() == name, 'Study manifest must be confined under repo/manifests')
    path = root
    for part in relative.parts:
        path /= part
        require(not path.is_symlink(), 'Study manifest has a symlinked path')
    require(path.is_file() and path.resolve().is_relative_to(root/'manifests'),
            'Prospective study manifest missing/outside repo/manifests')
    require(isinstance(sources, dict) and name in sources and sha(path) == sources[name],
            'Prospective study manifest is not exactly source-pinned')
    require(prospective.get('study_manifest_sha256') == sha(path), 'Prospective study manifest SHA differs')
    if pins is not None:
        require(pins.get('study_manifest_path', LEGACY_STUDY_MANIFEST) == name
                and pins.get('study_manifest_sha256') == sha(path), 'Pinned prospective study manifest differs')
    return name, path, prospective


def prepare_pins(root, record, destination):
    root, record, destination = Path(root).resolve(), Path(record).resolve(), Path(destination)
    require(not destination.exists(), 'Pins destination must be fresh')
    manifest = load(record/'manifest.json')
    require(REQUIRED <= set(manifest['source_sha256']), 'Study source inventory incomplete')
    study_name, study_path, _ = bound_study_manifest(root, record, manifest['source_sha256'])
    require(manifest.get('study_manifest_sha256') == sha(study_path), 'Record study manifest SHA differs')
    value = {'version': 1, 'status': 'REQUIRES_ROOT_REVIEW_BEFORE_LAUNCH',
             'guard_sha256': sha(Path(__file__)), 'shared_guard_sha256': SHARED_SHA, 'reviewed_base_sha256': BASE_SHA,
             'metadata_sha256': {name: sha(record/name) for name in META},
             'source_inventory_sha256': fingerprint(manifest['source_sha256']),
             'source_file_count': len(manifest['source_sha256']),
             'study_manifest_path': study_name, 'study_manifest_sha256': sha(study_path)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')
    return value


def provenance(root, record, pins_path, pins_sha):
    helper = shared();require(HEX.fullmatch(pins_sha or '') and sha(pins_path) == pins_sha, 'External prelaunch pins SHA differs')
    pins = load(pins_path)
    require(pins.get('guard_sha256') == sha(Path(__file__)) and pins.get('shared_guard_sha256') == SHARED_SHA
            and pins.get('reviewed_base_sha256') == BASE_SHA and set(pins.get('metadata_sha256', {})) == set(META), 'Pinned integrity implementation differs')
    for name, value in pins['metadata_sha256'].items():
        require(sha(record/name) == value, 'Frozen record metadata differs')
    manifest = load(record/'manifest.json');sources = manifest.get('source_sha256')
    require(isinstance(sources, dict) and REQUIRED <= set(sources) and len(sources) == pins.get('source_file_count')
            and fingerprint(sources) == pins.get('source_inventory_sha256')
            and set(sources) == helper.actual_source_inventory(root), 'Frozen source inventory differs')
    for name, value in sources.items():
        require(sha(helper.public_path(root, name)) == value, 'Frozen public source differs: '+name)
    helper.B.source_zip(record/'source.zip', sources, manifest['source_archive_sha256'])
    require(sha(record/'design.txt') == manifest['design_sha256'], 'Frozen design differs')
    _, study_path, prospective = bound_study_manifest(root, record, sources, pins)
    require(sha(study_path) == pins.get('study_manifest_sha256') == manifest.get('study_manifest_sha256'), 'Frozen study selection differs')
    study = load(study_path)
    for name, value in study['source_sha256'].items():
        require(sha(helper.public_path(root, name)) == value, 'Study input metadata hash differs')
    require(study['policy']['aggregate'] == {'tokens': 200000, 'tools': 30, 'sdk_attempts': 30}
            and study['policy']['auxiliary'] == {'tokens': 60000, 'tools': 6, 'sdk_attempts': 6, 'probes': 2}, 'Prospective policy caps differ')
    require(equal(prospective.get('study_policy'), study['policy']), 'Prospective study policy differs')
    runtime = load(record/'runtime_adapter_manifest.json')
    require(set(runtime['source_sha256']) == RUNTIME, 'Public runtime inventory differs')
    for name, value in runtime['source_sha256'].items():
        require(sha(helper.public_path(root, name)) == value, 'Frozen public runtime differs')
    helper.B.source_zip(record/'runtime_adapter_sources.zip', runtime['source_sha256'], runtime['archive_sha256'])
    binding = load(record/'expected_public_binding.json')
    require(set(binding) == {'provider_profile_sha256', 'adapter_sha256', 'bootstrap_sha256'}
            and all(isinstance(v, str) and HEX.fullmatch(v) for v in binding.values())
            and binding['adapter_sha256'] == runtime['source_sha256']['.local-setup/azure_adapter.py']
            and binding['bootstrap_sha256'] == runtime['source_sha256']['.local-setup/sitecustomize.py'], 'Public provider binding differs')
    return manifest, study, runtime, binding


def run_check(root, record, pins_path, pins_sha, output, terminal_path, terminal_sha, report, engineering=False):
    root, record, output, report = map(lambda p: Path(p).resolve(), (root, record, output, report))
    require(not report.exists() and not report.is_relative_to(output), 'Integrity report must be fresh outside raw outputs')
    require(HEX.fullmatch(terminal_sha or '') and sha(terminal_path) == terminal_sha, 'External owned-process attestation SHA differs')
    attestation = load(terminal_path);expected = 4 if engineering else 480
    require(attestation.get('version') == 1 and attestation.get('owned_study_process_terminal') is True
            and type(attestation.get('owned_session_id')) is int and attestation['owned_session_id'] > 0
            and type(attestation.get('exit_code')) is int and attestation.get('accuracy_scored') is False
            and attestation.get('expected_policy_outcomes') == expected and attestation.get('saved_policy_outcomes') == expected
            and Path(attestation['output_directory']).resolve() == output
            and datetime.fromisoformat(attestation['observed_terminal_utc']).tzinfo is not None, 'Actual owned terminal process proof required')
    manifest, study, runtime, binding = provenance(root, record, Path(pins_path), pins_sha)
    if not engineering:
        require(Path(manifest['output_directory']).resolve() == output, 'Frozen output path differs')
    identity = load(owned(output, 'study_identity.json'));state = load(owned(output, 'study_state.json'))
    rows = load(owned(output, 'study_outcomes.json'))
    require(isinstance(rows, list) and len(rows) == expected and state.get('saved_policy_outcomes') == expected
            and state.get('expected_policy_outcomes') == expected and state.get('correctness_scored') is False,
            'All declared first outcomes must be terminal before integrity analysis')
    study_name, study_path, _ = bound_study_manifest(root, record, manifest['source_sha256'], load(pins_path))
    require(state.get('phase') in ({'development_interface_complete'} if engineering else {'generation_complete', 'generation_complete_with_errors'}), 'Generation not complete')
    require(identity.get('development_interface_only') is engineering and identity.get('expected_policy_outcomes') == expected
            and identity.get('manifest_sha256') == sha(study_path)
            and identity.get('model') == manifest['model'] and equal(identity.get('policy'), study['policy'])
            and equal(identity.get('freeze', {}).get('public_provider_binding'), binding), 'Actual study identity differs')
    frozen = identity.get('freeze') or {}
    actual_sources = frozen.get('source_sha256')
    require(isinstance(actual_sources, dict) and
            (all(manifest['source_sha256'].get(k) == v for k, v in actual_sources.items()) if engineering
             else equal(actual_sources, manifest['source_sha256']))
            and equal(frozen.get('runtime_sha256'), runtime['source_sha256'])
            and (frozen.get('freeze_manifest_sha256') in {None, sha(record/'manifest.json')} if engineering
                 else frozen.get('freeze_manifest_sha256') == sha(record/'manifest.json')), 'Actual runtime/source identity differs')
    require({name for name in REQUIRED if name.startswith('agent/') or name in
             {'scripts/v8_mechanism_study.py', 'scripts/v8_study_cache.py', 'scripts/launch_v8_study.py'}} <= set(actual_sources),
            'Actual inference identity omits required dependencies')
    cases = identity.get('runtime_case_metadata')
    require(isinstance(cases, list) and len(cases) == expected//4, 'Runtime case selection missing')
    if not engineering:
        require(equal(cases, study['cases']), 'Actual selected120/order differs')
    else:
        require(cases[0]['question_id'] not in {c['question_id'] for c in study['cases']}, 'Engineering case consumed frozen120')
    indexed = {}
    for row in rows:
        key = row.get('question_id'), row.get('study_arm')
        require(key not in indexed, 'Duplicate study first outcome')
        indexed[key] = row
    require(set(indexed) == {(c['question_id'], m) for c in cases for m in MODES}, 'Study outcome cohort differs')
    require([(r['question_id'], r['study_arm']) for r in rows] ==
            [(c['question_id'], m) for c in cases for m in ('A', *c['followup_order'])],
            'Actual followup/time-block order differs from frozen plan')
    helper = shared();store = helper.B.load_public_store(root)
    files = {name: sha(owned(output, name)) for name in ('study_identity.json', 'study_state.json', 'study_outcomes.json')}
    counts = Counter();physical = [];fatal = False
    for case in cases:
        proof = verify_case([indexed[case['question_id'], m] for m in MODES], output, store, case,
                            manifest['model'], manifest['source_sha256'])
        counts.update(proof['counts']);physical.extend(proof['physical_calls']);files.update(proof['files'])
        fatal |= proof['fatal_physical']
    journals = validate_journals(output, physical, fatal)
    for item in journals['journals']:
        files[item['path']] = item['sha256']
    # No accuracy import has occurred. Reject drift before publishing any PASS.
    provenance(root, record, Path(pins_path), pins_sha)
    require(sha(terminal_path) == terminal_sha, 'Owned-process proof changed during verification')
    for name, value in files.items():
        require(sha(owned(output, name)) == value, 'Raw study snapshot changed during verification')
    value = {'kind': 'v8_study_completion_integrity', 'status': 'PASS', 'engineering_only': engineering,
             'policy_outcomes': expected, 'questions': expected//4, 'accuracy_scored': False,
             'owned_terminal_attestation_sha256': terminal_sha, 'pins_sha256': pins_sha,
             'guard_sha256': sha(Path(__file__)), 'source_manifest_sha256': sha(record/'manifest.json'),
             'source_files': len(manifest['source_sha256']), 'runtime_files': len(runtime['source_sha256']),
             'study_manifest_path': study_name, 'study_manifest_sha256': sha(study_path),
             'counts': dict(counts), 'physical_provider': journals, 'raw_sha256': files,
             'archive_allowlist': sorted(n for n in files if not n.startswith('case_cache/')),
             'limits': ['Execution/source-field/retained-accounting provenance is not semantic correctness.',
                        'C retained-context construction follows frozen code; no wire-payload/tokenizer equivalence claim.',
                        'Initial identical cache-copy claim follows frozen runner plus retained base proof; current private files/inodes checked.',
                        'Known admission refusal is not an SDK call; fatal pre-send/observer gaps are reported, never zero billed usage.',
                        'Cache pickle files are hashed only, never unpickled or included in the archive allowlist.',
                        'No scorer imported, QA CSV parsed, model/FHIR request or program/source-check execution.']}
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('mode', choices=['pin-record', 'completion'])
    parser.add_argument('--root', default=str(ROOT));parser.add_argument('--record', required=True)
    parser.add_argument('--pins', required=True);parser.add_argument('--pins-sha256')
    parser.add_argument('--output');parser.add_argument('--terminal-proof');parser.add_argument('--terminal-proof-sha256')
    parser.add_argument('--report');parser.add_argument('--engineering', action='store_true')
    args = parser.parse_args(argv)
    if args.mode == 'pin-record':
        value = prepare_pins(args.root, args.record, args.pins)
        print(json.dumps({'pins_sha256': sha(args.pins), 'status': value['status']}))
    else:
        require(all((args.output, args.report, args.terminal_proof)), 'Completion paths required')
        value = run_check(args.root, args.record, args.pins, args.pins_sha256, args.output,
                          args.terminal_proof, args.terminal_proof_sha256, args.report, args.engineering)
        print(json.dumps({k: value[k] for k in ('status', 'questions', 'policy_outcomes', 'accuracy_scored')}))


if __name__ == '__main__':
    main()
