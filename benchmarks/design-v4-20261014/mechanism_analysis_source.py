"""Corrected V4 completed-pair mechanisms (Single execution proof). Root terminal confirmation and reviewed integrity proof are mandatory.

No scorer is loaded until the complete frozen source, both ordered 300-row first-
attempt files, identities, expected hashes and every audit have passed validation.
No inference, configuration bootstrap, network, FHIR or runtime feedback is used.
"""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import zipfile

COUNT = 300
FROZEN_FILE_COUNT = 142
EXPERIMENT = 'design-v4-20261014'
RUN_RELATIVE = 'output/experiments/gpt-6-luna/plan_conditioned_v4_300q'
ARMS = {'ordinary_single_agent':'single_agent', 'ordinary_multi_agent':'multi_agent_v4'}
PHASES = ('planner', 'solver')
PHASE_CAPS = {'planner':(10000,0), 'solver':(190000,30)}
PROTOCOL = 'plan-conditioned-v4'
PASS_STATUSES = {'PASS','PASS_WITH_JOURNAL_LIMITATIONS'}
CLEAN_POLICY = 'source_valid_primary_preserved'
HEX = re.compile(r'^[0-9a-f]{64}$')

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def iter_rows(path, chunk_chars=1024 * 1024):
    """Stream the same top-level result-array format as the benchmark runner."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding='utf-8') as stream:
        buffer, position, ended = '', 0, False
        def fill():
            nonlocal buffer, position, ended
            fragment = stream.read(chunk_chars)
            buffer, position = buffer[position:] + fragment, 0
            ended = not fragment
        def whitespace():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or ended:
                    return
                fill()
        whitespace()
        if position >= len(buffer) or buffer[position] != '[':
            raise ValueError('Result must be a JSON array')
        position += 1
        whitespace()
        if position < len(buffer) and buffer[position] == ']':
            position += 1
        else:
            while True:
                while True:
                    try:
                        record, end = decoder.raw_decode(buffer, position)
                        break
                    except json.JSONDecodeError:
                        if ended:
                            raise ValueError('Incomplete result JSON') from None
                        fill()
                position = end
                yield record
                whitespace()
                if position < len(buffer) and buffer[position] == ']':
                    position += 1
                    break
                if position >= len(buffer) or buffer[position] != ',':
                    raise ValueError('Invalid result array separator')
                position += 1
                whitespace()
        whitespace()
        if position < len(buffer):
            raise ValueError('Trailing result JSON')

def failed(row):
    error = row.get('error')
    return error is not None and str(error).strip() not in ('', 'nan')

def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0

def issue_types(items):
    if items is not None and not isinstance(items, list):
        return ['malformed_issue_list']
    result = []
    for item in items or []:
        if isinstance(item, dict):
            result.append(item.get('type', 'malformed_issue'))
        elif isinstance(item, str):
            result.append(item.partition(':')[0].strip() or 'unstructured_issue')
        else:
            result.append('malformed_issue')
    return sorted(set(result))

def same_json(left, right):
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError):
        return False

def artifact(value):
    return isinstance(value, dict) and not value.get('error') and all(
        key in value for key in ('answer', 'evidence', 'operation', 'contributing_ids', 'limitation'))

def single_release_support(trace, rendered):
    """Recover retained Single execution ancestry without treating rendered3 as full5.

    Successful explicit-final replies retain five fields in the complete audit.
    Retrieval and non-inspect Python replies invalidate prior execution; inspect
    preserves it. A matching clean automatic check must follow the retained final.
    A source-checked prose-only answer has unavailable execution provenance.
    """
    calls, current, checked = {}, None, False
    source_checked = False
    if not isinstance(trace, list) or not isinstance(rendered, dict):
        return {'artifact':None,'execution_status':'unavailable_trace_or_rendered_final',
                'automatic_source_check_matched':False}
    for message in trace:
        if not isinstance(message, dict): continue
        for call in message.get('tool_calls') or []:
            if not isinstance(call, dict): continue
            function = call.get('function') or {}
            try: args = json.loads(function.get('arguments') or '{}')
            except (TypeError,ValueError): args = None
            calls[call.get('id')] = (function.get('name'),args)
        if message.get('role') != 'tool': continue
        name,args = calls.get(message.get('tool_call_id'),(message.get('name'),None))
        name = message.get('name') or name
        try: result = json.loads(message.get('content') or '{}')
        except (TypeError,ValueError): result = None
        if name in ('get_resources_by_patient_fhir_id','get_resources_by_resource_id'):
            current,checked = None,False
        elif name == 'execute_python_code':
            mode = args.get('mode','auto') if isinstance(args,dict) else None
            if mode != 'inspect': current,checked = None,False
            if (mode == 'final' and isinstance(result,dict) and result.get('mode') == 'final'
                    and artifact(result)):
                current = result
        elif name == 'check_fields' and str(message.get('tool_call_id','')).startswith('automatic_field_check_'):
            matches = (isinstance(args,dict) and same_json(args.get('citations'),rendered.get('evidence'))
                       and isinstance(result,dict) and result.get('issues') == [])
            source_checked = bool(matches)
            checked = bool(matches and current is not None
                           and all(same_json(current.get(k),rendered.get(k)) for k in ('answer','evidence','limitation')))
            if not matches:
                # Single invalidates its final computation when an automatic
                # source check requests revision; a later check cannot revive it.
                current,checked = None,False
    if current is not None and checked:
        status = 'retained_current_final_and_matching_automatic_source_check'
    elif current is not None:
        status = 'retained_final_without_matching_automatic_source_check_or_rendered_binding'
    else:
        status = 'execution_ancestry_unavailable_or_invalidated'
    return {'artifact':current,'execution_status':status,'automatic_source_check_matched':source_checked,
            'execution_bound_and_checked':checked}


def terminal_kind(row):
    if not failed(row):
        return None
    text = str(row.get('error')).lower()
    for words, name in [ (('timeout', 'timed out'), 'timeout'),
                         (('budget', 'token ceiling', 'model-call allocation'), 'budget'),
                         (('input tokens', 'context'), 'input_guard'),
                         (('json serializable', 'serialization', 'serialize'), 'transport'),
                         (('badrequest', 'sdk', 'retries', 'provider'), 'provider'),
                         (('source', 'scope', 'field'), 'source'),
                         (('contract', 'final computation'), 'execution_contract') ]:
        if any(word in text for word in words):
            return name
    return 'other'

def trace_stats(trace):
    counts = Counter()
    arguments = {}
    if not isinstance(trace, list):
        return {'trace_malformed': 1}
    for message in trace:
        if not isinstance(message, dict):
            continue
        if message.get('model_call') is False and message.get('phase') == 'solver_boundary':
            counts['host_final_boundaries'] += 1
        for call in message.get('tool_calls') or []:
            if isinstance(call, dict):
                function = call.get('function') or {}
                try:
                    args = json.loads(function.get('arguments') or '{}')
                except (ValueError, TypeError):
                    args = {}
                arguments[call.get('id')] = (function.get('name'), args)
        if message.get('role') != 'tool':
            continue
        name = message.get('name')
        if name == 'check_fields':
            counts['source_check_replies'] += 1
        if name != 'execute_python_code':
            continue
        try:
            result = json.loads(message.get('content') or '{}')
        except (ValueError, TypeError):
            counts['python_reply_non_json'] += 1
            continue
        if not isinstance(result, dict):
            continue
        mode = result.get('mode') or (arguments.get(message.get('tool_call_id'), (None, {}))[1]).get('mode')
        if mode == 'final' and 'answer' in result and not result.get('error'):
            counts['worker_valid_final_replies'] += 1
        if result.get('error_type') == 'final_contract':
            counts['worker_final_contract_errors'] += 1
    return dict(counts)

def source_caps(path):
    values = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('PHASE_CAPS', 'PROTOCOL'):
                    values[target.id] = ast.literal_eval(node.value)
    return values

def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_file(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bound_file(root, relative):
    relative = PurePosixPath(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe public source path')
    require(relative.name != 'config.yml' and '.local-setup' not in relative.parts,
            'Private settings paths are forbidden')
    path = root / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(root), 'Source escapes root')
    return path


def verify_freeze(root, proof):
    record = root / 'benchmarks' / EXPERIMENT
    manifest_path = record / 'manifest.json'
    require(sha256(manifest_path) == proof.get('manifest_sha256'), 'Frozen manifest digest differs from reviewed proof')
    manifest = json_file(manifest_path)
    require(manifest.get('experiment_id') == EXPERIMENT, 'Different experiment manifest')
    require(manifest.get('expected_questions_per_arm') == COUNT, 'Manifest cohort size differs')
    require(set(manifest.get('arms', [])) == set(ARMS), 'Manifest paired arms differ')
    require(manifest.get('model') == 'gpt-6-luna' and manifest.get('max_total_tokens') == 200000
            and manifest.get('max_tool_calls') == 30, 'Manifest common profile/caps differ')
    require(Path(manifest.get('output_directory', '')).resolve() == root / RUN_RELATIVE,
            'Manifest output directory differs')
    frozen = manifest.get('source_sha256')
    require(isinstance(frozen, dict) and len(frozen) == FROZEN_FILE_COUNT, 'Exactly 142 frozen source files required')
    for required in ('utils/answer_scorer.py','scripts/audit_store.py','agent/plan_conditioned_agent.py',
                     'agent/query_contract_agent.py','agent/single_agent.py'):
        require(required in frozen, 'Required frozen public source missing: '+required)
    for relative, expected in frozen.items():
        require(isinstance(expected, str) and HEX.fullmatch(expected), 'Invalid frozen digest')
        require(sha256(bound_file(root, relative)) == expected, 'Frozen source differs: '+relative)
    archive = record / 'source.zip'
    digest = sha256(archive)
    require(digest == manifest.get('source_archive_sha256') == proof.get('source_archive_sha256'),
            'Frozen source archive differs')
    with zipfile.ZipFile(archive) as packed:
        members = packed.infolist()
        require(len(members) == len(frozen) and len({m.filename for m in members}) == len(members),
                'Frozen archive member count/uniqueness differs')
        require({m.filename for m in members} == set(frozen), 'Frozen archive inventory differs')
        for member in members:
            bound_file(root, member.filename)
            require(not member.is_dir() and ((member.external_attr >> 16) & 0o170000) != 0o120000,
                    'Frozen archive contains directory/symlink')
            require(hashlib.sha256(packed.read(member)).hexdigest() == frozen[member.filename],
                    'Frozen archive entry differs: '+member.filename)
    require(sha256(record / 'design.txt') == manifest.get('design_sha256'), 'Frozen design differs')
    dataset = root / 'final_dataset/questions_answers_sql_fhir.csv'
    split = root / 'manifests/split.json'
    require(sha256(dataset) == manifest.get('dataset_sha256') == proof.get('dataset_sha256'), 'Dataset digest differs')
    require(sha256(split) == manifest.get('split_sha256') == proof.get('split_sha256'), 'Split digest differs')
    caps = source_caps(root / 'agent/plan_conditioned_agent.py')
    require(caps == {'PHASE_CAPS':PHASE_CAPS,'PROTOCOL':PROTOCOL}, 'Frozen V4 protocol/caps differ')
    ids = json_file(split)['splits']['held_out']['question_ids']
    require(isinstance(ids,list) and len(ids) == COUNT and len(set(ids)) == COUNT
            and all(isinstance(q,str) and q for q in ids), 'Complete unique ordered fixed 300 required')
    with dataset.open(newline='', encoding='utf-8') as stream:
        source_rows = list(csv.DictReader(stream))
    require(len({r['question_id'] for r in source_rows}) == len(source_rows), 'Duplicate source questions')
    source = {r['question_id']:r for r in source_rows}
    with (root / 'manifests/question_families.csv').open(newline='',encoding='utf-8') as stream:
        family_rows = list(csv.DictReader(stream))
    require(len({r['question_id'] for r in family_rows}) == len(family_rows), 'Duplicate family questions')
    families = {r['question_id']:r for r in family_rows}
    require(set(ids).issubset(source) and set(ids).issubset(families), 'Missing fixed cohort metadata')
    require(all('family' in families[q] and 'template' in families[q] for q in ids), 'Missing family/template metadata')
    return manifest_path, manifest, frozen, ids, source, families


def phase_records(revision):
    entries = revision.get('phases')
    if entries is None:
        return {}, ['phase_records_not_recorded']
    if not isinstance(entries,list):
        return {}, ['malformed_phase_records']
    result, anomalies = {}, []
    for entry in entries:
        if not isinstance(entry,dict) or entry.get('phase') not in PHASES:
            anomalies.append('malformed_phase_record'); continue
        name = entry['phase']
        if name in result: anomalies.append('duplicate_phase:'+name)
        result[name] = entry
        if (entry.get('token_cap'),entry.get('tool_cap')) != PHASE_CAPS[name]:
            anomalies.append('phase_cap_mismatch:'+name)
        for measured, cap in (('tokens','token_cap'),('tool_calls','tool_cap')):
            if not finite(entry.get(measured)) or not finite(entry.get(cap)):
                anomalies.append('unknown_phase_measure:'+name+':'+measured)
            elif entry[measured] > entry[cap]:
                anomalies.append('phase_cap_exceeded:'+name+':'+measured)
    return result, anomalies


def plan_status(revision, trace):
    plan, visible = revision.get('plan'), revision.get('plan_visible')
    claimed_hash = revision.get('plan_sha256')
    status, anomalies, operators = 'not_recorded', [], Counter()
    if visible is True:
        if not isinstance(plan,dict):
            status = 'visible_missing_plan'; anomalies.append('visible_plan_missing')
        else:
            try:
                actual = hashlib.sha256(json.dumps(plan,sort_keys=True,default=str,allow_nan=False).encode()).hexdigest()
            except (ValueError,TypeError,RecursionError):
                actual = None
            status = 'visible_valid_hash' if actual and actual == claimed_hash else 'visible_hash_mismatch'
            if status != 'visible_valid_hash': anomalies.append('plan_hash_mismatch')
    elif visible is False:
        error = str(revision.get('plan_error') or '').lower()
        status = 'ignored_unknown' if 'declared unknown' in error else 'ignored_error' if error else 'ignored_without_reason'
        if plan is not None or claimed_hash is not None: anomalies.append('ignored_plan_retained')
    marker = any(isinstance(m,dict) and m.get('role') == 'system'
                 and 'Frozen advisory plan JSON:' in str(m.get('content') or '') for m in trace or [])
    hash_verified = status == 'visible_valid_hash'
    phases = revision.get('phases') if isinstance(revision.get('phases'),list) else []
    planner_fatal = any(isinstance(p,dict) and p.get('phase') == 'planner' and p.get('fatal') for p in phases)
    solver_started = (any(isinstance(p,dict) and p.get('phase') == 'solver' for p in phases)
                      or sum(isinstance(m,dict) and m.get('role') == 'system' for m in trace or []) > 1
                      or any(isinstance(m,dict) and m.get('role') == 'user'
                             and str(m.get('content') or '').startswith('Solver allocation remaining: about ')
                             for m in trace or []))
    if visible is True and not marker:
        if hash_verified and planner_fatal and not solver_started:
            status = 'declared_visible_but_fatal_before_solver'
        else:
            anomalies.append('visible_plan_context_not_observed')
    if visible is False and marker: anomalies.append('ignored_plan_context_observed')
    if isinstance(plan,dict):
        for step in plan.get('steps') or []:
            if isinstance(step,dict): operators[str(step.get('operator'))] += 1
    return {'plan_status':status,'plan_visible':visible is True,'plan_hash':claimed_hash,
            'plan_hash_verified':hash_verified,
            'plan_error_present':bool(revision.get('plan_error')),'plan_advice_context_observed':marker,
            'plan_steps':len(plan.get('steps') or []) if isinstance(plan,dict) else 0,
            'plan_unknowns':len(plan.get('unknowns') or []) if isinstance(plan,dict) else 0,
            'plan_operator_counts':dict(operators)}, anomalies


def model_trace_stats(trace):
    stats = trace_stats(trace)
    stats.update(planner_model_responses=0, solver_model_responses=0, other_model_responses=0,
                 host_generated_source_check_assistant_messages=0, automatic_source_check_calls=0,
                 automatic_source_check_replies=0, solver_invocation_attempt_signals=0)
    systems = 0
    for message in trace or []:
        if not isinstance(message,dict): continue
        if message.get('role') == 'system': systems += 1
        if (message.get('role') == 'user'
                and str(message.get('content') or '').startswith('Solver allocation remaining: about ')):
            # LedgerSolver records this before PhaseHost; it can precede a local
            # refusal, so these are invocation attempts, not verified SDK sends.
            stats['solver_invocation_attempt_signals'] += 1
        calls = [c for c in message.get('tool_calls') or [] if isinstance(c,dict)]
        automatic = [c for c in calls if re.fullmatch(r'automatic_field_check_[0-9]+',str(c.get('id') or ''))
                     and isinstance(c.get('function'),dict) and c['function'].get('name') == 'check_fields']
        host_automatic = message.get('role') == 'assistant' and bool(calls) and len(automatic) == len(calls)
        if host_automatic:
            stats['host_generated_source_check_assistant_messages'] += 1
        stats['automatic_source_check_calls'] += len(automatic)
        if (message.get('role') == 'tool' and message.get('name') == 'check_fields'
                and re.fullmatch(r'automatic_field_check_[0-9]+',str(message.get('tool_call_id') or ''))):
            stats['automatic_source_check_replies'] += 1
        if message.get('role') != 'assistant' or message.get('model_call') is False or host_automatic:
            continue
        for call in calls:
            if call in automatic: continue
            name = (call.get('function') or {}).get('name','unknown')
            stats['model_requested_tool_calls'] = stats.get('model_requested_tool_calls',0)+1
            key = 'model_requested_'+str(name)
            stats[key] = stats.get(key,0)+1
        field = 'planner_model_responses' if systems == 1 else 'solver_model_responses' if systems >= 2 else 'other_model_responses'
        stats[field] += 1
    return stats


def row_record(row, qid, family, score, is_team):
    revision = row.get('revision')
    revision = revision if isinstance(revision,dict) else {}
    draft,primary,final = (revision.get(k) for k in ('draft','primary','final'))
    candidate_correct = lambda c: bool(score(c['answer'],qid)['typed_correct']) if isinstance(c,dict) and 'answer' in c else None
    result_score = score(row.get('agent_answer'),qid)
    terminal = failed(row)
    correct = bool(not terminal and result_score['typed_correct'])
    issues = revision.get('final_check_issues')
    clean = not terminal and issues == []
    release_matches = (isinstance(final,dict) and 'answer' in final and 'evidence' in final
                       and same_json(row.get('agent_answer'),final['answer'])
                       and same_json(row.get('evidence'),final['evidence'])
                       and same_json(row.get('limitation'),final.get('limitation')))
    primary_revision = revision.get('primary_revision')
    primary_revision = primary_revision if isinstance(primary_revision,dict) else {}
    source_checked_rendered = bool(clean and release_matches)
    support = {'artifact':None,'execution_status':'not_applicable_team_retains_full_final',
               'automatic_source_check_matched':False,'execution_bound_and_checked':False}
    if is_team:
        source_valid = bool(clean and revision.get('protocol') == PROTOCOL and revision.get('release_policy') == CLEAN_POLICY
                            and artifact(final) and release_matches)
        phases,anomalies = phase_records(revision)
    else:
        # Single revision.final is rendered3; the full executed5 exists only in
        # successful tool replies. Do not report missing execution proof as false.
        support = single_release_support(row.get('trace'),final)
        source_valid = (True if source_checked_rendered and support.get('execution_bound_and_checked') else
                        None if source_checked_rendered else False)
        phases,anomalies = {},[]
    if clean and artifact(final) and not release_matches: anomalies.append('released_artifact_mismatch')
    if is_team and revision.get('protocol') is None: anomalies.append('protocol_not_recorded')
    tokens = row.get('tokens') if isinstance(row.get('tokens'),dict) else {}
    budget = row.get('budget') if isinstance(row.get('budget'),dict) else {}
    total,tools,calls = tokens.get('total'),budget.get('tool_calls_run'),tokens.get('llm_calls')
    unknown_usage = any(isinstance(c,dict) and c.get('usage_available') is False for c in tokens.get('per_call') or [])
    if finite(total) and total > 200000: anomalies.append('global_token_cap_exceeded')
    if finite(tools) and tools > 30: anomalies.append('global_tool_cap_exceeded')
    if is_team and finite(calls) and calls > 21: anomalies.append('global_model_call_cap_exceeded')
    if phases and all(finite(e.get('tokens')) for e in phases.values()) and finite(total):
        if sum(e['tokens'] for e in phases.values()) != total: anomalies.append('phase_token_sum_mismatch')
    if phases and all(finite(e.get('tool_calls')) for e in phases.values()) and finite(tools):
        if sum(e['tool_calls'] for e in phases.values()) != tools: anomalies.append('phase_tool_sum_mismatch')
    unresolved = revision.get('unresolved_scope')
    unresolved = unresolved if isinstance(unresolved,dict) else {}
    trace = row.get('trace') if isinstance(row.get('trace'),list) else []
    traces = model_trace_stats(trace) if is_team else trace_stats(trace)
    if is_team:
        if traces.get('solver_model_responses',0) > 20: anomalies.append('solver_model_response_cap_exceeded')
        if traces.get('solver_invocation_attempt_signals',0) > 20: anomalies.append('solver_invocation_attempt_cap_exceeded')
        if traces.get('host_final_boundaries',0) > 2: anomalies.append('host_source_boundary_cap_exceeded')
        plan,plan_anomalies = plan_status(revision,trace)
        anomalies.extend(plan_anomalies)
    else:
        plan = {'plan_status':'not_applicable','plan_visible':False,'plan_hash':None,'plan_hash_verified':False,
                'plan_error_present':False,'plan_advice_context_observed':False,'plan_steps':0,'plan_unknowns':0,'plan_operator_counts':{}}
    draft_correct,primary_correct = candidate_correct(draft),candidate_correct(primary)
    source_revised = bool(revision.get('solver_source_revised') if is_team else revision.get('revised'))
    empty = result_score.get('answer_type') == 'empty'
    fatal = any(bool(e.get('fatal')) for e in phases.values())
    stop = bool(budget.get('stopped'))
    accounting_overshoot = any('cap_exceeded' in a for a in anomalies)
    if fatal or unknown_usage or accounting_overshoot: source_valid = False
    return {'question_id':qid,'family':family['family'],'answer_kind':result_score.get('answer_type'),
        'protocol':revision.get('protocol'),'protocol_not_recorded':is_team and revision.get('protocol') is None,
        'terminal_error':terminal,'terminal_error_kind':terminal_kind(row),'correct':correct,
        'typed_reason':result_score.get('typed_reason'),'reference_empty':empty,
        'draft_correct':draft_correct,'primary_correct':primary_correct,'final_candidate_correct':candidate_correct(final),
        'primary_artifact_present':artifact(primary),'released_artifact_present':artifact(final) if is_team else artifact(support.get('artifact')),
        'source_checked_rendered_release':source_checked_rendered,
        'execution_proof_status':support['execution_status'],
        'execution_automatic_source_check_matched':support.get('automatic_source_check_matched',False),
        'source_valid_release_support_unavailable':source_valid is None,
        'primary_source_valid':(primary_revision.get('final_check_issues') == [] if artifact(primary) and 'final_check_issues' in primary_revision else None),
        'clean_final_source_check':clean,'released_artifact_matches_output':release_matches,
        'source_valid_release':source_valid,'release_policy':revision.get('release_policy'),
        'answer_is_null':row.get('agent_answer') is None,
        'scored_null_credit':correct and row.get('agent_answer') is None,
        'empty_reference_credit_outside_clean_release':correct and empty and source_valid is False,
        'null_credit_outside_clean_release':correct and row.get('agent_answer') is None and source_valid is False,
        'credited_without_source_valid_release':correct and source_valid is False,
        'credited_with_source_release_support_unavailable':correct and source_valid is None,
        'source_valid_correct_release':correct and source_valid is True,
        'source_valid_typed_wrong_release':not correct and not terminal and source_valid is True,
        'solver_source_revised':source_revised,
        'source_revision_helped':source_revised and draft_correct is False and primary_correct is True,
        'source_revision_harmed':source_revised and draft_correct is True and primary_correct is False,
        'initial_issue_types':issue_types(revision.get('check_issues')),'final_issue_types':issue_types(issues),
        'unresolved_scope_count':len(unresolved),'unresolved_scope_reason_counts':dict(Counter(str(v) for v in unresolved.values())),
        'budget_stop':stop,'phase_records':phases,'phase_fatal':fatal,
        'phase_local_refusal':any(bool(e.get('errors')) for e in phases.values()),
        'provider_usage_unknown':unknown_usage,'accounting_overshoot':accounting_overshoot,
        'accounting_anomalies':sorted(set(anomalies)),
        'tokens':total,'prompt_tokens':tokens.get('prompt'),'completion_tokens':tokens.get('completion'),
        'model_calls':calls,'tool_calls':tools,'cost':tokens.get('cost'),'latency':row.get('latency'),
        'context_compactions':len(revision.get('context_compactions') or []),
        'trace_stats':traces,**plan}


def mean_record(records,field,predicate=lambda row:True):
    values = [row[field] for row in records if predicate(row) and finite(row.get(field))]
    return {'mean':sum(values)/len(values) if values else None,'n':len(values)}


def summarize(records):
    flags = ('terminal_error','correct','reference_empty','source_valid_release','source_valid_correct_release',
             'source_valid_typed_wrong_release','scored_null_credit','empty_reference_credit_outside_clean_release',
             'null_credit_outside_clean_release','credited_without_source_valid_release','solver_source_revised',
             'source_checked_rendered_release','source_valid_release_support_unavailable',
             'credited_with_source_release_support_unavailable',
             'source_revision_helped','source_revision_harmed','phase_fatal','phase_local_refusal',
             'provider_usage_unknown','accounting_overshoot','budget_stop','protocol_not_recorded',
             'plan_visible','plan_hash_verified','plan_error_present','plan_advice_context_observed')
    summary = {'questions':len(records),**{k:sum(bool(r[k]) for r in records) for k in flags}}
    summary['accuracy_all_first_attempts'] = summary['correct']/len(records) if records else None
    summary['plan_visible_rate_all_first_attempts'] = summary['plan_visible']/len(records) if records else None
    ignored = sum(r['plan_status'].startswith('ignored_') for r in records)
    summary['plan_ignored'] = ignored
    summary['plan_ignored_rate_all_first_attempts'] = ignored/len(records) if records else None
    for field in ('terminal_error_kind','release_policy','plan_status','answer_kind','execution_proof_status'):
        summary[field+'_counts'] = dict(Counter(r[field] for r in records if r[field] is not None))
    for field in ('initial_issue_types','final_issue_types','accounting_anomalies'):
        summary[field] = dict(Counter(item for r in records for item in r[field]))
    for field in ('trace_stats','unresolved_scope_reason_counts','plan_operator_counts'):
        counts = Counter()
        for r in records: counts.update(r[field])
        summary[field] = dict(counts)
    summary['usage'] = {field:mean_record(records,field) for field in
        ('tokens','prompt_tokens','completion_tokens','model_calls','tool_calls','cost','latency','context_compactions')}
    summary['usage']['known_usage_tokens'] = mean_record(records,'tokens',lambda r:not r['provider_usage_unknown'])
    summary['usage']['accounted_tokens_are_not_verified_billing_for_unknown_usage'] = True
    summary['phases'] = {}
    for name in PHASES:
        entries = [r['phase_records'][name] for r in records if name in r['phase_records']]
        summary['phases'][name] = {'observed':len(entries),'missing':len(records)-len(entries),
            'fatal':sum(bool(e.get('fatal')) for e in entries),
            'local_refusals':sum(bool(e.get('errors')) for e in entries),
            'tokens_recorded':sum(e['tokens'] for e in entries if finite(e.get('tokens'))),
            'tools_recorded':sum(e['tool_calls'] for e in entries if finite(e.get('tool_calls'))),
            'tokens_mean_observed':sum(e['tokens'] for e in entries if finite(e.get('tokens')))/len(entries) if entries else None}
    return summary


def loss_kind(team):
    if team['terminal_error']: return 'terminal_'+str(team['terminal_error_kind'])
    if team['source_valid_release']: return 'source_valid_typed_wrong_semantic_or_scoring_mismatch'
    if team['phase_fatal'] or team['accounting_overshoot']: return 'fatal_accounting'
    if team['budget_stop']: return 'budget_stop'
    if team['unresolved_scope_count'] or 'source_scope' in team['final_issue_types']: return 'source_scope_refusal'
    if team['final_issue_types']: return 'source_or_execution_contract_refusal'
    return 'other_or_unrecorded_release'


def paired_report(single,team,full_acceptance=True):
    require([r['question_id'] for r in single] == [r['question_id'] for r in team], 'Paired diagnostic IDs differ')
    wins,losses = [],[]
    for s,t in zip(single,team):
        if t['correct'] and not s['correct']:
            wins.append({'question_id':t['question_id'],'family':t['family'],'answer_kind':t['answer_kind'],
                'source_valid_release':t['source_valid_release'],
                'credit_outside_clean_release':t['credited_without_source_valid_release'],
                'plan_status':t['plan_status']})
        elif s['correct'] and not t['correct']:
            losses.append({'question_id':t['question_id'],'family':t['family'],'answer_kind':t['answer_kind'],
                'mechanism':loss_kind(t),'plan_status':t['plan_status'],'plan_visible':t['plan_visible']})
    sc,tc = sum(r['correct'] for r in single),sum(r['correct'] for r in team)
    if full_acceptance: require(len(single) == COUNT,'Acceptance requires all 300 first attempts')
    return {'questions':len(single),'single_correct':sc,'team_correct':tc,'team_only_correct':len(wins),
        'single_only_correct':len(losses),'both_correct':sum(s['correct'] and t['correct'] for s,t in zip(single,team)),
        'both_wrong':sum(not s['correct'] and not t['correct'] for s,t in zip(single,team)),
        'difference_team_minus_single':(tc-sc)/len(single),
        'acceptance':{'best_observed_single_threshold':238,'requires_strictly_above_new_single':True,
                      'passed':tc > 238 and tc > sc,'all_300_first_attempts_errors_incorrect':True} if full_acceptance else None,
        'paired_loss_mechanism_counts':dict(Counter(r['mechanism'] for r in losses)),
        'paired_win_source_valid':sum(r['source_valid_release'] for r in wins),
        'paired_win_credit_outside_clean_release':sum(r['credit_outside_clean_release'] for r in wins),
        'wins':wins,'losses':losses}


def grouped(records,field):
    return {key:summarize([r for r in records if r[field] == key])
            for key in sorted({r[field] for r in records},key=str)}


def paired_groups(single,team,field):
    return {key:paired_report([s for s,t in zip(single,team) if t[field] == key],
                              [t for t in team if t[field] == key],full_acceptance=False)
            for key in sorted({r[field] for r in team},key=str)}


def run(root,completion_report,completion_report_sha256,output,confirm_both_owned_jobs_terminal=False,superseded_diagnostics=None):
    require(confirm_both_owned_jobs_terminal is True, 'Root must confirm both owned jobs are terminal before diagnostics')
    root,proof_path,output = Path(root).resolve(),Path(completion_report).resolve(),Path(output).resolve()
    require(isinstance(completion_report_sha256,str) and HEX.fullmatch(completion_report_sha256), 'Reviewed completion report SHA required')
    require(sha256(proof_path) == completion_report_sha256, 'Reviewed completion report digest differs')
    proof = json_file(proof_path)
    require(proof.get('status') in PASS_STATUSES, 'Completion integrity proof did not pass')
    require(isinstance(proof.get('arms'),dict) and set(proof['arms']) == set(ARMS), 'Completion proof must contain exactly both arms')
    require(all(isinstance(proof.get(k),str) and HEX.fullmatch(proof[k]) for k in
                ('code_hash','ordered_input_hash','manifest_sha256','dataset_sha256','split_sha256','source_archive_sha256')),
            'Completion proof lacks required global digest bindings')
    manifest_path,manifest,frozen,ids,source,families = verify_freeze(root,proof)
    audit_module = None
    prepared_paths,raw_inputs = {},{}
    checked = {proof_path:completion_report_sha256,manifest_path:proof['manifest_sha256']}
    for relative,digest in frozen.items(): checked[bound_file(root,relative)] = digest
    for p,k in ((root/'final_dataset/questions_answers_sql_fhir.csv','dataset_sha256'),
                (root/'manifests/split.json','split_sha256'),
                (manifest_path.parent/'source.zip','source_archive_sha256')):
        checked[p] = proof[k]
    checked[manifest_path.parent/'design.txt'] = manifest['design_sha256']
    identities = {}
    # No scorer imports or calls occur in this complete-pair integrity pass.
    for arm,strategy in ARMS.items():
        raw = root / RUN_RELATIVE / (arm+'.json')
        sidecar = Path(str(raw)+'.run.json')
        expected = proof['arms'][arm]
        require(isinstance(expected,dict), 'Malformed completion arm proof')
        require(sha256(raw) == expected.get('result_sha256'), arm+': raw digest differs')
        require(sha256(sidecar) == expected.get('identity_sha256'), arm+': identity digest differs')
        checked[raw],checked[sidecar] = expected['result_sha256'],expected['identity_sha256']
        identity = json_file(sidecar)
        required = {'version':2,'agent_strategy':strategy,'model':'gpt-6-luna','max_total_tokens':200000,
                    'max_tool_calls':30,'input_hash':proof['ordered_input_hash'],'code_hash':proof['code_hash'],
                    'split':'held_out','split_manifest_hash':proof['split_sha256'],'repeat':None,'condition':None,
                    'subset_hash':None,'perturbations_hash':None,'add_patient_fhir_id':True,'enable_cache':True,
                    'audit_storage':'gzip','audit_artifact_schema':2}
        require(all(k in identity and identity[k] == v for k,v in required.items()),arm+': run identity differs')
        identities[arm] = identity
        originals = list(iter_rows(raw))
        require(len(originals) == COUNT and all(isinstance(r,dict) for r in originals), arm+': complete 300 rows required')
        require([r.get('question_id') for r in originals] == ids, arm+': unique frozen question order differs')
        for row in originals:
            qid = row['question_id']
            require(row.get('patient_fhir_id') == source[qid]['patient_fhir_id'],arm+': patient identity differs')
            require(row.get('attempt_history') in (None,[]),arm+': attempt histories/replays forbidden')
            require(failed(row) or finite(row.get('latency')),arm+': unattempted first-attempt row')
            require(isinstance(row.get('audit_artifacts'),dict),arm+': missing gzip audit')
        if audit_module is None:
            audit_module = load_module(root/'scripts/audit_store.py','_v4_public_audit_store')
        artifacts = []
        for row in originals:
            qid = row['question_id']; reference = row['audit_artifacts']
            restored_row = audit_module.restore_result_audit(row,raw.parent,question_id=qid,restore_history=False)
            path = (raw.parent/reference['path']).resolve()
            require(not (raw.parent/reference['path']).is_symlink() and path.is_relative_to(raw.parent), 'Audit path escapes run')
            checked[path] = sha256(path)
            artifacts.append({'question_id':qid,'path':reference['path'],'uncompressed_sha256':reference['sha256'],
                              'compressed_sha256':checked[path]})
            protocol = (restored_row.get('revision') or {}).get('protocol') if isinstance(restored_row.get('revision'),dict) else None
            require(arm != 'ordinary_multi_agent' or protocol in (None,PROTOCOL),arm+': unexpected protocol')
        prepared_paths[arm] = raw
        raw_inputs[arm] = {'raw_sha256':expected['result_sha256'],'sidecar_sha256':expected['identity_sha256'],
                           'identity':identity,'audit_artifacts':artifacts}
    first,second = (identities[a] for a in ARMS)
    require({k:v for k,v in first.items() if k != 'agent_strategy'} ==
            {k:v for k,v in second.items() if k != 'agent_strategy'},'Paired run identity settings differ')
    for path,digest in checked.items(): require(sha256(path) == digest,'Input changed before scoring: '+str(path))
    # All 600 artifacts, both complete arms and 142 source files are now verified.
    scorer = load_module(root/'utils/answer_scorer.py','_v4_public_unchanged_typed_scorer')
    score = lambda answer,qid: scorer.score_answer(answer,source[qid]['true_answer'],source[qid]['question'],families[qid]['template'])
    records = {}
    for arm,raw in prepared_paths.items():
        records[arm] = []
        # Restore one full audit at a time; do not retain all 600 large traces.
        for original in iter_rows(raw):
            qid = original['question_id']
            row = audit_module.restore_result_audit(original,raw.parent,question_id=qid,restore_history=False)
            records[arm].append(row_record(row,qid,families[qid],score,arm == 'ordinary_multi_agent'))
    for path,digest in checked.items(): require(sha256(path) == digest,'Input changed during scoring; no report written: '+str(path))
    correction = {'kind':'posthoc_single_release_classification_only','superseded_helper_sha256':'cd26072933950ee2edcc01574af7c2850d1764528883889a402c6aa9c4c1d69c',
        'reason':'Single revision.final stores only rendered answer/evidence/limitation; full executed artifacts are in retained tool replies. Prior helper incorrectly required five fields in the rendered final.',
        'score_or_runtime_changes':False,'superseded_reports':{},
        'unavailable_execution_ancestry':'source_valid_release is null (N/A); source_checked_rendered_release is reported separately and unproven execution is not counted as confirmed outside-source credit.'}
    if superseded_diagnostics is not None:
        superseded = Path(superseded_diagnostics).resolve()
        require(superseded.is_dir() and superseded != output, 'Superseded diagnostics must be a separate directory')
        for name in ('mechanisms.json','mechanisms_rows.json','mechanisms.csv'):
            path = superseded/name
            require(path.is_file() and not path.is_symlink(), 'Missing safe superseded diagnostic output')
            correction['superseded_reports'][name] = {'path':str(path),'sha256':sha256(path)}
        require(json_file(superseded/'mechanisms.json').get('script_sha256') == correction['superseded_helper_sha256'], 'Unexpected superseded diagnostic helper')
    report = {'diagnostic_correction':correction,'schema_version':5,'experiment':EXPERIMENT,'scope':'Completed paired fixed 300, all scheduled first attempts; exploratory reused cohort.',
        'scoring':'Unchanged typed scorer loaded from its frozen public file after all completion checks; terminal errors incorrect.',
        'completion_report':{'path':str(proof_path),'sha256':completion_report_sha256,'status':proof['status']},
        'frozen_source_files_verified':len(frozen),'source_manifest_sha256':proof['manifest_sha256'],
        'source_archive_sha256':proof['source_archive_sha256'],'script_sha256':sha256(__file__),
        'protocol':{'name':PROTOCOL,'phase_caps':PHASE_CAPS,'solver_real_calls':20,'host_slots':22,
                    'model_requested_tools':28,'reserved_source_checks':2},
        'caveats':['Plan textual provenance and source checks do not prove clinical interpretation correctness.',
                   'Plan context exposure does not show causal use or establish the effect of planning alone.',
                   'Source-valid typed-wrong outcomes indicate a semantic/scoring mismatch, not an independent clinical adjudication.',
                   'Empty-reference null/absence credit outside a clean release is separate from successful source-validated answers.',
                   'Single rendered source-check support and retained execution-backed proof are reported separately; unavailable execution ancestry is null/N/A, not a failed source check.',
                   'Unknown provider usage placeholders do not establish zero usage or billing.',
                   'Terminal attempts with missing protocol/phases remain in the denominator and are reported as diagnostic gaps.',
                   'System-message segmentation counts observed model replies excluding host automatic source checks; failed provider attempts may have no reply.',
                   'Solver allocation signals count invocation attempts before PhaseHost, including possible local refusals; they are not verified SDK sends.',
                   'A validated plan declared visible before fatal planner accounting can remain unexposed because the solver never started.',
                   'The completion proof may pass with disclosed journal limitations; this helper does not infer missing requests.'],
        'raw_inputs':raw_inputs,'arms':{a:summarize(r) for a,r in records.items()},
        'paired':paired_report(records['ordinary_single_agent'],records['ordinary_multi_agent']),
        'per_family':{a:grouped(r,'family') for a,r in records.items()},
        'per_answer_kind':{a:grouped(r,'answer_kind') for a,r in records.items()},
        'per_plan_status_team':grouped(records['ordinary_multi_agent'],'plan_status'),
        'paired_per_family':paired_groups(records['ordinary_single_agent'],records['ordinary_multi_agent'],'family'),
        'paired_per_answer_kind':paired_groups(records['ordinary_single_agent'],records['ordinary_multi_agent'],'answer_kind')}
    require(not output.exists(),'Choose a fresh diagnostics output directory')
    output.mkdir(parents=True)
    (output/'mechanisms.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    all_records = [dict(row,arm=arm) for arm,rows in records.items() for row in rows]
    (output/'mechanisms_rows.json').write_text(json.dumps(all_records,indent=2,allow_nan=False)+'\n')
    with (output/'mechanisms.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(all_records[0])); writer.writeheader()
        for row in all_records:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--completion-report',required=True)
    parser.add_argument('--completion-report-sha256',required=True)
    parser.add_argument('--confirm-both-owned-jobs-terminal',action='store_true',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--superseded-diagnostics')
    args = parser.parse_args()
    report = run(args.root,args.completion_report,args.completion_report_sha256,args.output,args.confirm_both_owned_jobs_terminal,args.superseded_diagnostics)
    print(json.dumps({'arms':{a:{'questions':s['questions'],'correct':s['correct']} for a,s in report['arms'].items()},
                      'paired':{k:v for k,v in report['paired'].items() if k not in ('wins','losses')}},indent=2))


if __name__ == '__main__':
    main()
