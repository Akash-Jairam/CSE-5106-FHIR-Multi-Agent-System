"""V8 postterminal supplementary diagnostics; no live evaluation or routing.

Only the pinned official analysis defines the primary result and fixed R gate.
This program imports the scorer only after the external completion proof, current
source/runtime/archive bindings, all raw hashes, and all480 firsts pass. Audits
are then restored one at a time. No QA CSV is parsed or program re-executed.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]
MODES = ('A', 'W', 'R', 'C')
FIELDS = ('answer', 'evidence', 'operation', 'contributing_ids', 'limitation')
RENDERED = ('answer', 'evidence', 'limitation')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024*1024):
            result.update(block)
    return result.hexdigest()


def load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def equal(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def module_from_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def failed(row):
    return row.get('error') is not None and str(row['error']).strip() not in ('', 'nan')


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def measured(values):
    values = list(values);known = [v for v in values if finite_number(v)]
    return {'records': len(values), 'known_records': len(known), 'unknown_records': len(values)-len(known),
            'known_sum': sum(known), 'known_mean': statistics.mean(known) if known else None,
            'known_median': statistics.median(known) if known else None}


def expected_rows(records, manifest):
    cases = manifest.get('cases')
    require(isinstance(cases, list) and len(cases) == 120 and len({c['question_id'] for c in cases}) == 120,
            'Frozen120 manifest required')
    expected = {(c['question_id'], mode) for c in cases for mode in MODES};index = {}
    require(isinstance(records, list) and len(records) == 480, 'All480 outcomes required before correctness')
    for row in records:
        key = row.get('question_id'), row.get('study_arm')
        require(key in expected and key not in index and row.get('study_terminal') is True
                and not row.get('attempt_history'), 'Missing/duplicate/nonfirst/nonterminal policy')
        require({'agent_answer', 'error', 'question', 'true_answer', 'template', 'patient_fhir_id', 'mechanism'} <= set(row),
                'Actual saved fields unavailable; never impute null')
        index[key] = row
    require(set(index) == expected, 'Complete study cohort differs')
    for case in cases:
        a = index[case['question_id'], 'A']
        for mode in MODES:
            row = index[case['question_id'], mode]
            require(all(equal(row.get(k), a.get(k)) for k in
                        ('patient_fhir_id', 'question', 'true_answer', 'template', 'primary_envelope_sha256')),
                    'Shared case/reference/A identity differs')
            mechanism = row['mechanism']
            require(isinstance(mechanism, dict) and type(mechanism.get('selected_B')) is bool
                    and type(mechanism.get('auxiliary_fatal')) is bool, 'Explicit mechanism flags unavailable')
            selected, fatal = mechanism['selected_B'], mechanism['auxiliary_fatal']
            require(not (selected and fatal), 'Selected-B and auxiliary-fatal effects overlap')
            if fatal:
                require(failed(row), 'Auxiliary fatal lacks retained error')
            b = mechanism.get('released_source_bound_B')
            require(b is None or isinstance(b, dict) and set(b) == set(FIELDS), 'Released B artifact unavailable/malformed')
            if selected:
                require(b is not None and not failed(row) and equal(row['agent_answer'], b['answer'])
                        and all(equal(row.get(k), b[k]) for k in ('evidence', 'limitation')), 'Selected B is not released final')
            elif mode != 'A' and not fatal:
                require(all(equal(row.get(k), a.get(k)) for k in ('agent_answer', 'evidence', 'limitation', 'error')),
                        'Nonfatal nonselection changed actual A')
    return cases, index


def gate_and_load(args, guard_loader=None):
    """No scorer load and no correctness. Frozen integrity is reused by hash."""
    root, record, output = map(lambda p: Path(p).resolve(), (args.root, args.record, args.study_output))
    destination = Path(args.destination).resolve()
    require(not destination.exists() and not destination.is_relative_to(output), 'Fresh diagnostics directory outside raw study required')
    bindings = [(args.integrity_report, args.integrity_sha256), (args.terminal_proof, args.terminal_sha256),
                (args.pins, args.pins_sha256), (args.official_analysis, args.official_sha256)]
    for path, digest in bindings:
        require(isinstance(digest, str) and len(digest) == 64 and not Path(path).is_symlink()
                and sha(path) == digest, 'Externally pinned proof/official analysis differs')
    proof = load(args.integrity_report);terminal = load(args.terminal_proof)
    require(proof.get('kind') == 'v8_study_completion_integrity' and proof.get('status') == 'PASS'
            and proof.get('engineering_only') is False and proof.get('policy_outcomes') == 480
            and proof.get('questions') == 120 and proof.get('accuracy_scored') is False
            and proof.get('owned_terminal_attestation_sha256') == args.terminal_sha256
            and proof.get('pins_sha256') == args.pins_sha256, 'Deep actual480 completion PASS required')
    require(terminal.get('version') == 1 and terminal.get('owned_study_process_terminal') is True
            and terminal.get('accuracy_scored') is False and type(terminal.get('owned_session_id')) is int
            and terminal['owned_session_id'] > 0 and type(terminal.get('exit_code')) is int
            and terminal.get('saved_policy_outcomes') == 480 and terminal.get('expected_policy_outcomes') == 480
            and Path(terminal['output_directory']).resolve() == output
            and datetime.fromisoformat(terminal['observed_terminal_utc']).tzinfo is not None,
            'Observed owned process exit required')
    guard_path = root/'scripts/v8_study_integrity.py'
    require(sha(guard_path) == proof.get('guard_sha256'), 'Frozen guard source differs')
    guard = (guard_loader or (lambda path: module_from_file(path, 'v8_supplement_integrity')))(guard_path)
    source, manifest, runtime, binding = guard.provenance(root, record, Path(args.pins), args.pins_sha256)
    study_name, study_path, _ = guard.bound_study_manifest(
        root, record, source['source_sha256'], load(args.pins))
    require(sha(record/'manifest.json') == proof.get('source_manifest_sha256')
            and proof.get('study_manifest_path') == study_name
            and sha(study_path) == proof.get('study_manifest_sha256')
            and Path(source['output_directory']).resolve() == output, 'Integrity record/input/output binding differs')
    snapshots = proof.get('raw_sha256')
    require(isinstance(snapshots, dict) and {'study_outcomes.json', 'study_state.json', 'study_identity.json'} <= set(snapshots),
            'Deep guard raw inventory unavailable')
    for name, digest in snapshots.items():
        require(sha(guard.owned(output, name)) == digest, 'Raw snapshot changed after deep integrity PASS')
    rows = load(guard.owned(output, 'study_outcomes.json'))
    state = load(guard.owned(output, 'study_state.json'));identity = load(guard.owned(output, 'study_identity.json'))
    require(state.get('phase') in {'generation_complete', 'generation_complete_with_errors'}
            and state.get('saved_policy_outcomes') == 480 and state.get('expected_policy_outcomes') == 480
            and state.get('correctness_scored') is False and identity.get('development_interface_only') is False,
            'Actual generation not complete')
    expected_rows(rows, manifest)
    official = load(args.official_analysis)
    require(official.get('status') == 'COMPLETED_MECHANISM_STUDY' and official.get('primary_contrast') == 'R - A'
            and set(official.get('arms', {})) == set(MODES)
            and all(official['arms'][m].get('n') == 120 for m in MODES), 'Pinned complete official analysis required')
    return {'root': root, 'record': record, 'output': output, 'destination': destination,
            'guard': guard, 'source': source, 'manifest': manifest, 'runtime': runtime, 'binding': binding,
            'rows': rows, 'official': official, 'proof': proof, 'state': state, 'identity': identity,
            'bindings': bindings, 'script_sha256': sha(Path(__file__))}


def source_trace(trace):
    """Operational receipts, not an interpretation of question semantics."""
    counts = Counter();pending = {};latest = None;latest_check = None
    for message in trace or []:
        for call in message.get('tool_calls') or []:
            function = call.get('function') or {}
            try:
                arguments = json.loads(function.get('arguments') or '{}')
            except (ValueError, TypeError):
                arguments = {}
            pending[call.get('id')] = (function.get('name'), arguments)
        if message.get('role') != 'tool':
            continue
        name, arguments = pending.get(message.get('tool_call_id'), (message.get('name'), {}))
        counts['tool_replies'] += 1
        if str(message.get('content', '')).startswith('Not run:'):
            counts['refused_tool_replies'] += 1
        try:
            value = json.loads(message.get('content') or '{}')
        except (ValueError, TypeError):
            value = None
        counts['tool_error_replies'] += isinstance(value, dict) and bool(value.get('error'))
        if name == 'execute_python_code' and arguments.get('mode') == 'final':
            counts['native_final_requests_with_replies'] += 1
            if isinstance(value, dict) and value.get('mode') == 'final' and 'error' not in value and 'error_type' not in value and all(k in value for k in FIELDS):
                latest = {k: value[k] for k in FIELDS};latest_check = None
                counts['successful_final_executions'] += 1
        if name == 'check_fields':
            counts['source_check_replies'] += 1
            issues = value.get('issues') if isinstance(value, dict) else None
            if issues == []:
                counts['source_checks_recorded_clean'] += 1
            elif isinstance(issues, list):
                counts['source_checks_recorded_issues'] += 1
                for issue in issues:
                    lower = str(issue).lower()
                    category = next((key for key in ('patient_scope', 'paired_scope', 'encounter_scope',
                        'encounter_link', 'time_window', 'unit', 'no evidence') if key in lower),
                        'evidence_result' if lower.startswith('evidence[') else 'other')
                    counts['issue_category_'+category.replace(' ', '_')] += 1
            else:
                counts['source_check_unavailable_or_error'] += 1
            if latest is not None and equal(arguments.get('citations'), latest['evidence']):
                latest_check = issues
    return {'counts': dict(counts), 'last_successful_execution': latest, 'later_matching_check_issues': latest_check}


def reported_physical_cost(restored, actual_a):
    """Reported phase cost once; missing billing metadata never becomes zero."""
    revision = restored.get('revision') or {}
    if restored.get('study_arm') == 'A':
        a = next((s for s in revision.get('messages') or [] if s.get('candidate') == 'A'), None)
        value = (((a or {}).get('usage') or {}).get('total_usage') or {}).get('cost')
        return value if finite_number(value) else None, 'actual_A_reported_cost_once'
    phases = [p for p in revision.get('phases') or [] if p.get('phase') == 'counterexample_auxiliary']
    if len(phases) == 1:
        value = ((phases[0].get('raw_accounting') or {}).get('total_usage') or {}).get('cost')
        return value if finite_number(value) else None, 'actual_auxiliary_reported_cost'
    a_calls = (actual_a.get('tokens') or {}).get('llm_calls')
    calls = (restored.get('tokens') or {}).get('llm_calls')
    if not phases and type(a_calls) is int and type(calls) is int and calls == a_calls:
        return 0, 'absent_auxiliary_zero_admitted_suffix_calls'
    return None, 'phase_cost_unavailable_not_assumed_zero'


def summarize(records, restored_rows, manifest, scorer, official):
    """Pure analysis seam. Call only after gate_and_load, using saved120 refs."""
    cases, index = expected_rows(records, manifest)
    score = scorer.score_answer if hasattr(scorer, 'score_answer') else scorer
    parse_reference = getattr(scorer, 'parse_reference', None)
    require(callable(parse_reference), 'Unchanged reference parser required')
    credit = {key: bool(not failed(row) and score(row['agent_answer'], row['true_answer'], row['question'], row['template'])['typed_correct'])
              for key, row in index.items()}
    reference_empty = {case['question_id']: len(parse_reference(index[case['question_id'], 'A']['true_answer'])) == 0 for case in cases}
    arms = {};mechanisms = {};descriptive = {};flags = {};corrective_patients = set();corrective_ops = set()
    for mode in MODES:
        arms[mode] = {'n': 120, 'correct': sum(credit[c['question_id'], mode] for c in cases),
                      'terminal_errors': sum(failed(index[c['question_id'], mode]) for c in cases)}
        groups = defaultdict(lambda: Counter(n=0, correct=0, terminal_errors=0));counter = Counter()
        for case in cases:
            qid = case['question_id'];row = index[qid, mode];a = credit[qid, 'A'];final = credit[qid, mode]
            for view, label in [('family', case['study_family']), ('operation', case['operation_group']),
                                ('reference', 'empty' if reference_empty[qid] else 'nonempty')]:
                groups[view+'/'+label].update(n=1, correct=int(final), terminal_errors=int(failed(row)))
            mechanism = row['mechanism'];b = mechanism.get('released_source_bound_B')
            b_credit = None if b is None else bool(score(b['answer'], row['true_answer'], row['question'], row['template'])['typed_correct'])
            selected, fatal = mechanism['selected_B'], mechanism['auxiliary_fatal']
            opportunity = not a and b_credit is True
            if mode != 'A':
                counter['O'] += opportunity;counter['H'] += selected and not a and final
                counter['M'] += selected and a and not final;counter['F'] += fatal and a and not final
                counter['all_post_A_helpful'] += not a and final;counter['all_post_A_harmful'] += a and not final
                counter['selected_B'] += selected;counter['B_available_source_bound'] += b is not None
                counter['auxiliary_fatal'] += fatal
                if mode == 'R' and opportunity:
                    corrective_patients.add(case['patient_fhir_id']);corrective_ops.add(case['operation_group'])
            flags[qid, mode] = {'question_id': qid, 'study_arm': mode, 'family': case['study_family'],
                'operation': case['operation_group'], 'reference_empty': reference_empty[qid],
                'A_typed_correct': a, 'released_policy_typed_correct': final, 'source_bound_B_typed_correct': b_credit,
                'source_bound_B_available': b is not None, 'selected_B': selected, 'auxiliary_fatal': fatal,
                'O': opportunity if mode != 'A' else False, 'H': bool(selected and not a and final),
                'M': bool(selected and a and not final), 'F': bool(fatal and a and not final),
                'witness_bound': mechanism.get('witness_bound'), 'witness_semantic_validity': 'unknown',
                'warning': 'Typed B error or harmful choice does not by itself establish an incorrect witness.'}
        descriptive[mode] = {key: dict(value) for key, value in groups.items()}
        if mode != 'A':
            counter['net'] = counter['H']-counter['M']-counter['F']
            require(counter['net'] == arms[mode]['correct']-arms['A']['correct'], 'H-M-F differs from actual final-minus-A')
            mechanisms[mode] = dict(counter)
    require(equal(arms, official.get('arms')) and equal(mechanisms, official.get('mechanisms'))
            and len(corrective_patients) == official.get('R_corrective_patients')
            and len(corrective_ops) == official.get('R_corrective_operations')
            and arms['R']['correct']-arms['A']['correct'] == official.get('R_minus_A_correct'), 'Supplementary counts differ from frozen official analysis')
    require(equal({k: v for k, v in official['advance_gate'].items() if k != 'passed'}, manifest['policy']['advance_gate']),
            'Official frozen advancement rule differs')
    rule = manifest['policy']['advance_gate'];r = mechanisms['R']
    expected_gate = (r['O'] >= rule['O_min'] and r['net'] >= rule['net_min']
                     and r['M']+r['F'] <= rule['harm_plus_fatal_max']
                     and len(corrective_patients) >= rule['corrective_patients_min']
                     and len(corrective_ops) >= rule['corrective_operations_min'])
    require(official['advance_gate'].get('passed') is expected_gate, 'Official fixed gate does not reconcile its exact reported counts')
    operational = {mode: {'statuses': Counter(), 'phase_states': Counter(), 'admitted_calls': Counter(),
                         'source_and_tool_receipts': Counter(), 'candidate_states': Counter(), 'admission': defaultdict(list)} for mode in MODES}
    seen = set();physical_costs = [];cost_sources = Counter()
    for restored in restored_rows:
        key = restored.get('question_id'), restored.get('study_arm')
        require(key in index and key not in seen, 'Restored audit missing/duplicate/unexpected')
        seen.add(key);row = index[key];mode = key[1];revision = restored.get('revision') or {};op = operational[mode]
        require(equal(row.get('mechanism'), revision.get('mechanism')), 'Restored mechanism differs from saved inline fields')
        value, origin = reported_physical_cost(restored, index[key[0], 'A'])
        physical_costs.append(value);cost_sources[origin] += 1
        op['statuses'][revision.get('status') or 'unavailable'] += 1
        for phase in revision.get('phases') or []:
            name = phase.get('phase', 'unavailable')
            op['phase_states'][name+':fatal='+str(phase.get('fatal'))] += 1
            for invocation in phase.get('calls') or []:
                op['admitted_calls'][invocation.get('status') or 'unavailable'] += 1
        for name, value in (revision.get('admission') or {}).items():
            if finite_number(value):
                op['admission'][name].append(value)
        for snapshot in revision.get('messages') or []:
            label = snapshot.get('candidate', 'unavailable');artifact = snapshot.get('artifact')
            trace = [m for m in restored.get('trace', []) if m.get('phase') == snapshot.get('phase')]
            receipts = source_trace(trace)
            op['source_and_tool_receipts'].update(receipts['counts'])
            state = label+(':released_source_bound' if artifact is not None else ':no_released_source_bound_artifact')
            op['candidate_states'][state] += 1
            computed = receipts['last_successful_execution'];issues = receipts['later_matching_check_issues']
            if artifact is None and computed is not None:
                op['candidate_states'][label+':retained_successful_execution_not_a_released_artifact'] += 1
                flags[key][label+'_unreleased_execution_typed_correct'] = bool(score(computed['answer'], row['true_answer'], row['question'], row['template'])['typed_correct'])
                flags[key][label+'_unreleased_execution_check'] = ('recorded_issues' if issues else 'recorded_clean' if issues == [] else 'unavailable')
                flags[key][label+'_unreleased_execution_scope'] = 'Diagnostic execution only; may be stale, source-refused or accounting-ineligible; excluded from O/H/M/F.'
                if isinstance(issues, list) and issues:
                    op['candidate_states'][label+':retained_execution_with_later_source_issues'] += 1
    require(seen == set(index), 'Every owned policy audit must be restored before diagnostics publication')
    output_ops = {}
    for mode, values in operational.items():
        output_ops[mode] = {k: dict(v) for k, v in values.items() if k != 'admission'}
        output_ops[mode]['admission'] = {key: measured(values) for key, values in values['admission'].items()}
    accounting = {'conceptual_policies': {}, 'physical_A_once_plus_suffixes': {}}
    for mode in MODES:
        rows = [index[c['question_id'], mode] for c in cases]
        accounting['conceptual_policies'][mode] = {key: measured((r.get('tokens') or {}).get(key) for r in rows)
                                                 for key in ('prompt', 'completion', 'total', 'cached_prompt', 'cost', 'llm_calls')}
        accounting['conceptual_policies'][mode]['A_plus_suffix_latency_seconds'] = measured(r.get('latency') for r in rows)
        accounting['conceptual_policies'][mode]['charged_tools'] = measured((r.get('budget') or {}).get('tool_calls_run') for r in rows)
        for container, keys in [('physical_fhir', ('observed_physical_queries', 'physical_requests', 'pages', 'response_bytes', 'returned_records')),
                                ('cache_activity', ('load_calls', 'hits', 'misses'))]:
            values = {}
            for key in keys:
                conceptual = []
                for case in cases:
                    a = (index[case['question_id'], 'A'].get(container) or {}).get(key)
                    suffix = (index[case['question_id'], mode].get(container) or {}).get(key)
                    conceptual.append(a if mode == 'A' else a+suffix if finite_number(a) and finite_number(suffix) else None)
                values[key] = measured(conceptual)
            accounting['conceptual_policies'][mode]['A_plus_suffix_'+container] = values
    physical = accounting['physical_A_once_plus_suffixes']
    for key in ('prompt', 'completion', 'cached'):
        physical['known_'+key+'_tokens'] = measured((r.get('physical_usage') or {}).get('known_tokens', {}).get(key) for r in records)
    for key in ('observed_invocations', 'known_usage_invocations', 'unknown_usage_invocations', 'cost'):
        physical[key] = measured((r.get('physical_usage') or {}).get(key) for r in records)
    physical['physical_A_wall_seconds'] = measured(index[c['question_id'], 'A'].get('physical_A_wall_seconds') for c in cases)
    physical['followup_wall_seconds'] = measured(r.get('followup_wall_seconds') for r in records if r['study_arm'] != 'A')
    tools = []
    for row in records:
        a = (index[row['question_id'], 'A'].get('budget') or {}).get('tool_calls_run')
        total = (row.get('budget') or {}).get('tool_calls_run')
        tools.append(total if row['study_arm'] == 'A' else total-a if finite_number(total) and finite_number(a) else None)
    physical['charged_tools'] = measured(tools)
    physical['fhir_pagination'] = {key: measured((r.get('physical_fhir') or {}).get(key) for r in records)
                                    for key in ('observed_physical_queries', 'physical_requests', 'pages', 'response_bytes', 'returned_records')}
    physical['tool_cache'] = {key: measured((r.get('cache_activity') or {}).get(key) for r in records)
                              for key in ('load_calls', 'hits', 'misses')}
    physical['cache_observer_unavailable_rows'] = sum((r.get('cache_activity') or {}).get('cache_observer_available') is not True for r in records)
    physical['derived_reported_phase_cost'] = measured(physical_costs)
    physical['derived_reported_phase_cost_sources'] = dict(cost_sources)
    physical['derived_reported_phase_cost_scope'] = 'Actual A snapshot cost once plus actual auxiliary raw phase costs; no admitted suffix phase has0 cost only when its retained SDK-call difference is0. Reported estimate, not invoice/billing proof; None remains unknown.'
    return {'kind': 'v8_postterminal_supplementary', 'primary_authority': 'Externally SHA-pinned frozen official analysis; no new gate/selection.',
            'arms': arms, 'mechanisms_exact_official_reconciliation': mechanisms,
            'official_advance_gate': official['advance_gate'], 'descriptive_accuracy': descriptive,
            'balanced_followup_order_counts': dict(Counter(''.join(c['followup_order']) for c in cases)),
            'operational': output_ops, 'accounting': accounting,
            'case_flags': [flags[c['question_id'], m] for c in cases for m in MODES],
            'limits': ['Adaptive balanced120 development study, not protected confirmation or a causal pure-independence ablation.',
                       'Subgroups and W/R/C contrasts are descriptive; no new thresholds or favorable-view selection.',
                       'A physically runs once; conceptual policy totals duplicate A by design.',
                       'Reported cost is not an invoice; unknown usage/cost/cache fields remain unknown, not zero billing.',
                       'FHIR metrics cover existing pagination observers, not authentication or all network activity; tool cache is not provider prompt cache.',
                       'Only local FHIR/tool caches are snapshot-isolated. Azure prompt caching is opaque; actual reported cached tokens and balanced W/R/C order do not remove server-side caching effects.',
                       'Retained successful execution without released artifact is diagnostic only and may be stale; it never creates a corrective opportunity.',
                       'Source checks and literal witness lineage do not establish semantic validity; harmful typed switches do not localize the cause.']}


def recheck(context, args):
    require(sha(Path(__file__)) == context['script_sha256'], 'Supplementary source changed during analysis')
    for path, digest in context['bindings']:
        require(sha(path) == digest, 'Pinned report/proof/official analysis drift')
    context['guard'].provenance(context['root'], context['record'], Path(args.pins), args.pins_sha256)
    for name, digest in context['proof']['raw_sha256'].items():
        require(sha(context['guard'].owned(context['output'], name)) == digest, 'Raw snapshot drift before publication')


def run(args, guard_loader=None, scorer_loader=None):
    context = gate_and_load(args, guard_loader)
    # This is the first possible scorer import, after every prerequisite above.
    scorer = (scorer_loader or (lambda path: module_from_file(path, 'v8_supplement_pure_scorer')))(context['root']/'utils/answer_scorer.py')
    store = context['guard'].shared().B.load_public_store(context['root'])
    def restored_rows():
        for row in context['rows']:
            yield store.restore_result_audit(row, context['output'], question_id=row['question_id'], restore_history=False)
    result = summarize(context['rows'], restored_rows(), context['manifest'], scorer, context['official'])
    result['provenance'] = {'supplementary_sha256': context['script_sha256'], 'integrity_report_sha256': args.integrity_sha256,
                            'owned_terminal_proof_sha256': args.terminal_sha256, 'official_analysis_sha256': args.official_sha256,
                            'guard_pins_sha256': args.pins_sha256, 'source_manifest_sha256': context['proof']['source_manifest_sha256']}
    result['physical_generation_wall_seconds'] = context['state'].get('physical_generation_wall_seconds')
    provider = context['proof'].get('physical_provider') or {}
    journal_counts = Counter();causes = Counter();statuses = Counter()
    for journal in provider.get('journals') or []:
        journal_counts.update(journal.get('counts') or {});causes.update(journal.get('cause_types') or {});statuses.update(journal.get('http_status_counts') or {})
    native_usage = []
    for journal in provider.get('journals') or []:
        path = context['guard'].owned(context['output'], journal['path'])
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event['event'] == 'end':
                native_usage.append(event['usage'])
    result['observed_provider'] = {key: provider.get(key) for key in ('physical_ledger_calls', 'observed_sdk_ends', 'known_usage_calls',
        'unknown_usage_calls', 'journal_usage_unavailable', 'ledger_minus_observed_sdk_calls', 'quarantined_accounting_unverifiable')}
    result['observed_provider'].update(journal_counts=dict(journal_counts), cause_types=dict(causes), http_status_counts=dict(statuses))
    result['observed_provider']['native_sdk_reported_usage'] = {key: measured(v.get(key) for v in native_usage)
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cached_prompt_tokens')}
    result['observed_provider']['cache_scope'] = 'Native SDK cached_prompt_tokens distinguishes missing from0; conceptual cached_prompt is the retained normalized agent ledger. Azure prompt caching is opaque and not locally isolated.'
    recheck(context, args)
    context['destination'].mkdir(parents=True, exist_ok=False)
    with (context['destination']/'supplementary.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root', default=str(ROOT));parser.add_argument('--record', required=True)
    parser.add_argument('--study-output', required=True);parser.add_argument('--destination', required=True)
    for name in ('integrity', 'terminal', 'official'):
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--integrity-report', required=True);parser.add_argument('--terminal-proof', required=True)
    parser.add_argument('--official-analysis', required=True);parser.add_argument('--pins', required=True)
    parser.add_argument('--pins-sha256', required=True)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({'kind': result['kind'], 'policy_outcomes': 480, 'official_gate_reconciled': True}))


if __name__ == '__main__':
    main()
