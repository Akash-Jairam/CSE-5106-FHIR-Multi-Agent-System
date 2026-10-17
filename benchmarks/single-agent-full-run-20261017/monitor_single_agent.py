"""Private, answer-blind observer/verifier for the authorized single-agent scope.

No inference, configuration reads, scheduling, scoring, or process mutation.
Only --append-validation writes to the run, and only to an observer-owned journal.
"""
import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.audit_store import restore_result_audit
from scripts.v3_full_run import classify_attempt, classifier_for_manifest, provider_rejection_limit

ARM = 'single_agent'
UNSELECTED = ('multi_agent', 'direct_llm')
AMENDMENT = 'amendments/0003-single-agent-only-resume/pre_apply_metadata'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def semantic_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def preservation(run, snapshot):
    before, current = read(snapshot / 'acceptance.json'), read(run / 'acceptance.json')
    records = {}
    for arm in UNSELECTED:
        files = {}
        for name in (f'full_{arm}.json', f'full_{arm}.json.run.json'):
            expected, actual = sha(snapshot / name), sha(run / name)
            require(expected == actual, f'Unselected canonical/identity changed: {name}')
            files[name] = actual
        for part in ('arms', 'counts'):
            require(before[part][arm] == current[part][arm], f'Unselected ledger {part} changed: {arm}')
        records[arm] = {'files_byte_identical_sha256': files,
                        'ledger_choices_semantic_sha256': semantic_sha(current['arms'][arm]),
                        'ledger_counts_semantic_sha256': semantic_sha(current['counts'][arm]),
                        'unchanged': True}
    return records


def scope_state(state):
    require(state.get('selected_arms') == [ARM], 'Selected scope is not single_agent only')
    require(set(state.get('active', {})) <= {ARM}, 'Unauthorized arm active')
    require(state.get('benchmark_complete') is False, 'Must not claim original three-arm benchmark complete')


def native_status(state, terminal=False):
    if terminal:
        return {'required': False, 'reason': 'Verified selected execution scope completed; analysis remains separately pending', 'pid_comm': {}}
    pids = [state['supervisor_pid']] + [detail['pid'] for detail in state.get('active', {}).values()]
    if type(state.get('keep_awake_pid')) is int:
        pids.append(state['keep_awake_pid'])
    result = subprocess.run(['ps', '-p', ','.join(map(str, sorted(set(pids)))), '-o', 'pid=,stat=,comm='],
                            text=True, capture_output=True)
    found, states = {}, {}
    for line in result.stdout.splitlines():
        values = line.strip().split(None, 2)
        if len(values) == 3:
            states[values[0]], found[values[0]] = values[1], values[2]
    require(all(str(pid) in found for pid in pids), 'Owned process missing from native PID/state/comm snapshot')
    require(all('Z' not in states[str(pid)] for pid in pids), 'Owned process is a zombie')
    return {'required': True, 'pid_comm': found, 'pid_states': states, 'ps_exit_code': result.returncode}


def safe_projection(row):
    # Prediction presence is used only for terminal/missing-latency detection,
    # preserving the frozen classifier's exact None/NaN behavior. No prediction
    # values or reference columns are printed, scored, or used for selection.
    return {key: row.get(key) for key in ('agent_answer', 'question_id', 'patient_fhir_id', 'latency', 'error', 'tokens',
             'budget', 'trace', 'revision', 'audit_artifacts', 'attempt_history')}


def provider_events(run, round_number):
    ends, incomplete = [], 0
    for journal in sorted((run / 'diagnostics' / ARM / f'round-{round_number:04d}').glob('*.jsonl')):
        for line in journal.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                incomplete += 1
                continue
            if event.get('event') == 'end':
                ends.append(event)
    failures = []
    for event in ends:
        if event.get('outcome') != 'error':
            continue
        failures.append({key: event.get(key) for key in ('pid', 'counter', 'started_utc', 'finished_utc',
                           'duration_seconds', 'http_status', 'fd_count')} |
                        {'exception_classes': [(item.get('type') or {}).get('class') for item in event.get('exception_chain', [])],
                         'sdk_closed': (event.get('client') or {}).get('sdk_closed'),
                         'http_closed': (event.get('client') or {}).get('http_closed')})
    return {'completed_requests': len(ends), 'outcomes': dict(Counter(event.get('outcome') for event in ends)),
            'failures_safe_metadata': failures, 'partial_lines': incomplete}


def terminal_proof(state, ledger, selected_count, expected):
    if state.get('phase') != 'complete_selected_arms':
        return False
    require(state.get('selected_scope_complete') is True, 'Selected completion flag missing')
    require(state.get('benchmark_complete') is False, 'Original benchmark completion must stay false')
    require(state.get('pending_scoped_export') is True and state.get('analysis_status') == 'pending_scoped_export',
            'Selected completion must distinguish pending scoped analysis')
    require(bool(state.get('finished_at')), 'Selected completion timestamp missing')
    require(not state.get('active'), 'Selected terminal state still has active workers')
    require(not state.get('attention'), 'Selected terminal state requires attention')
    require(selected_count == expected == state.get('accepted', {}).get(ARM), 'Selected terminal cohort incomplete')
    require(ledger.get('complete') is False, 'Original three-arm acceptance must remain incomplete')
    return True


def cohort(manifest):
    path = Path(manifest['input']['path'])
    require(sha(path) == manifest['input']['sha256'], 'Authoritative full input changed')
    with path.open(newline='') as stream:
        patients = {}
        for row in csv.DictReader(stream):
            qid = row['question_id']
            require(qid and qid not in patients and row['patient_fhir_id'], 'Invalid authoritative cohort identities')
            patients[qid] = row['patient_fhir_id']
    require(len(patients) == manifest['questions_per_arm'], 'Authoritative cohort count changed')
    return patients


def identities_and_rows(run, manifest, batch_path, patients):
    batch = read(batch_path)
    require(batch.get('arm') == ARM, 'Wrong arm in selected batch')
    output = Path(batch['output'])
    require(output.resolve().is_relative_to((run / 'attempts' / ARM).resolve()), 'Attempt output escaped selected arm')
    seal_path = batch_path.with_suffix('.seal.json')
    seal, identity_path = read(seal_path), Path(str(output) + '.run.json')
    require(seal['batch_sha256'] == sha(batch_path), 'Batch seal mismatch')
    require(seal['result_sha256'] == (sha(output) if output.exists() else None), 'Result seal mismatch')
    require(seal['identity_sha256'] == (sha(identity_path) if identity_path.exists() else None), 'Identity seal mismatch')
    require(sha(batch['input']) == batch['input_sha256'], 'Shard input changed')
    with Path(batch['input']).open(newline='') as stream:
        shard = list(csv.DictReader(stream))
    require(len(shard) == len(batch['question_ids']) and {row['question_id'] for row in shard} == set(batch['question_ids']), 'Shard question identities changed')
    require(all(row['question_id'] in patients and row['patient_fhir_id'] == patients[row['question_id']] for row in shard), 'Shard patient binding differs from full input')
    if not output.exists():
        return batch, [], None
    identity = read(identity_path)
    require(identity == batch['run_identity'], 'Saved identity differs from immutable batch')
    expected = manifest['run_identities'][ARM]
    for key in ('agent_strategy', 'model', 'code_hash', 'max_total_tokens', 'max_tool_calls', 'base_url_hash',
                'enable_cache', 'add_patient_fhir_id', 'audit_storage', 'audit_artifact_schema', 'split',
                'condition', 'subset_hash', 'repeat'):
        require(identity.get(key) == expected.get(key), f'Selected inference identity changed: {key}')
    rows = read(output)
    require(len(rows) == len(batch['question_ids']), 'Selected batch omitted placeholders')
    require(len({row['question_id'] for row in rows}) == len(rows), 'Duplicate result question')
    require({row['question_id'] for row in rows} == set(batch['question_ids']), 'Selected cohort IDs changed')
    require(all(row['patient_fhir_id'] == patients[row['question_id']] for row in rows), 'Result patient binding changed')
    return batch, rows, identity


def replay(run, manifest):
    accepted, choices, retries, counts, summaries = {}, {}, {}, Counter(raw_attempts=0, runtime_failures=0, intrinsic_errors=0), []
    patients = cohort(manifest)
    classifier = classifier_for_manifest(manifest)
    provider_counts = Counter()
    policy = manifest.get('provider_rejection_policy')
    paths = sorted((run / 'attempts' / ARM).glob('round-*.batch.json'))
    for batch_path in paths:
        if not batch_path.with_suffix('.seal.json').exists():
            continue
        batch, rows, identity = identities_and_rows(run, manifest, batch_path, patients)
        require(not set(batch['question_ids']) & set(accepted), 'Accepted question scheduled again')
        if policy:
            require(all(provider_counts[qid] < provider_rejection_limit(manifest, qid, batch['round']) for qid in batch['question_ids']),
                    'Historical batch exceeded provider rejection unchanged-retry limit')
        decisions, audit_count, rejections = Counter(), 0, Counter()
        for row in rows:
            qid = row['question_id']
            decision = classifier(safe_projection(row), Path(batch['output']).parent)
            decisions[decision['status']] += 1
            if decision['status'] == 'pending':
                continue
            require(decision['status'] != 'attention', 'Unclassified selected attempt requires attention')
            require(row.get('audit_artifacts') is not None, 'Attempt audit reference missing')
            restore_result_audit(safe_projection(row), Path(batch['output']).parent, question_id=qid, restore_history=False)
            audit_count += 1
            counts['raw_attempts'] += 1
            if decision['status'] == 'runtime':
                retries[qid] = retries.get(qid, 0) + 1
                counts['runtime_failures'] += 1
                if decision.get('category') == 'provider_policy_rejection':
                    rejections[qid] += 1
                    provider_counts[qid] += 1
            else:
                accepted[qid] = {key: value for key, value in row.items() if key != 'attempt_history'}
                choices[qid] = {'attempt_file': str(Path(batch['output']).relative_to(run)), 'round': batch['round'],
                                'runtime_failures_before': retries.get(qid, 0), 'selection': decision['reason'], 'runtime_clean': True}
                counts['intrinsic_errors'] += int(bool(row.get('error')))
        summaries.append({'round': batch['round'], 'scheduled': len(batch['question_ids']),
                          'classifications': dict(decisions), 'audits_verified': audit_count,
                          'provider_rejections': dict(rejections),
                          'result_sha256': sha(batch['output']) if Path(batch['output']).exists() else None,
                          'seal_sha256': sha(batch_path.with_suffix('.seal.json'))})
    return accepted, choices, retries, dict(counts), summaries


def verify(run, snapshot, args):
    manifest, state, ledger = read(run / 'run_manifest.json'), read(run / 'supervisor_state.json'), read(run / 'acceptance.json')
    scope_state(state)
    preserved = preservation(run, snapshot)
    accepted, choices, retries, counts, summaries = replay(run, manifest)
    rejections = Counter()
    for summary in summaries:
        rejections.update(summary['provider_rejections'])
    policy = manifest.get('provider_rejection_policy')
    parked = {qid for qid, count in rejections.items() if policy and count >= provider_rejection_limit(manifest, qid) and qid not in accepted}
    if policy:
        derived = read(run / 'provider_rejections.json')
        require(derived.get('policy') == policy and derived.get('source') == 'reconstructed_from_sealed_immutable_attempts', 'Provider rejection provenance changed')
        require(derived.get('rejections') == {ARM: dict(rejections)} and derived.get('parked_question_ids') == {ARM: sorted(parked)}, 'Provider rejection counts/parking differ from sealed replay')
    require(set(ledger['arms'][ARM]) == set(accepted), 'Selected ledger does not include all sealed clean results')
    require(ledger['arms'][ARM] == choices, 'Selected ledger did not select the first runtime-clean outcome')
    require(ledger['counts'][ARM] == counts, 'Selected operational counts differ from sealed replay')
    canonical = read(run / f'full_{ARM}.json')
    require(len(canonical) == len(accepted) and len({row['question_id'] for row in canonical}) == len(canonical), 'Canonical count/IDs mismatch')
    require({row['question_id'] for row in canonical} == set(accepted), 'Canonical cohort differs from first-clean replay')
    for row in canonical:
        require(semantic_sha(row) == semantic_sha(accepted[row['question_id']]), 'Accepted row changed from immutable source attempt')
        restore_result_audit(safe_projection(row), run, question_id=row['question_id'], restore_history=False)
    require(read(run / f'full_{ARM}.json.run.json') == manifest['run_identities'][ARM], 'Canonical full-cohort identity changed')
    selected_count, expected = len(accepted), manifest['questions_per_arm']
    terminal = terminal_proof(state, ledger, selected_count, expected)
    require(not state.get('attention'), 'Supervisor requests attention')
    if args.round is not None:
        summary = next((row for row in summaries if row['round'] == args.round), None)
        require(summary is not None, 'Requested sealed round missing')
        if args.clean is not None:
            require(summary['classifications'].get('accepted', 0) == args.clean, 'Unexpected clean count')
        if args.runtime is not None:
            require(summary['classifications'].get('runtime', 0) == args.runtime, 'Unexpected runtime count')
    if args.accepted is not None:
        require(selected_count == args.accepted, 'Unexpected accepted total')
    if args.final_arm:
        require(terminal, 'Final verification requires selected-scope success proof')
    if not terminal:
        require(not state.get('selected_scope_complete'), 'Nonterminal selected completion flag inconsistent')
        active = state.get('active', {}).get(ARM)
        require(active is not None, 'No active selected successor')
        batch_path = run / 'attempts' / ARM / f'round-{active["round"]:04d}.batch.json'
        active_batch = read(batch_path)
        require(not set(active_batch['question_ids']) & set(accepted), 'Active batch repeats accepted question')
        if policy:
            require(all(rejections[qid] < provider_rejection_limit(manifest, qid, active_batch['round'])
                        for qid in active_batch['question_ids']),
                    'Persistently rejected request scheduled beyond its unchanged-retry limit')
        pending_retries = set(retries) - set(accepted) - parked
        require(pending_retries <= set(active_batch['question_ids']), 'Runtime retry missing from active selected queue')
    native = native_status(state, terminal)
    return {'status': 'SELECTED_EXECUTION_COMPLETE_ANALYSIS_PENDING' if terminal else 'SELECTED_SCOPE_IN_PROGRESS',
            'selected_arms': [ARM], 'selected_accepted': selected_count, 'questions_per_selected_arm': expected,
            'selected_scope_complete': terminal, 'original_three_arm_benchmark_complete': False,
            'analysis_status': state.get('analysis_status'), 'preserved_unselected': preserved,
            'replayed_counts': counts, 'sealed_batches': summaries, 'canonical_first_clean_verified': True,
            'provider_rejections_replayed': dict(rejections), 'parked_pending_question_ids': sorted(parked),
            'accepted_reruns': 0, 'current_native': native, 'state_phase': state.get('phase')}


def observe(run, snapshot, args):
    manifest, state, ledger = read(run / 'run_manifest.json'), read(run / 'supervisor_state.json'), read(run / 'acceptance.json')
    scope_state(state)
    preserved = preservation(run, snapshot)
    expected = manifest['questions_per_arm']
    selected = len(ledger['arms'][ARM])
    terminal = terminal_proof(state, ledger, selected, expected)
    active_report = None
    classifier = classifier_for_manifest(manifest)
    active = state.get('active', {}).get(ARM)
    if active:
        path = run / 'attempts' / ARM / f'round-{active["round"]:04d}.json'
        decisions = Counter()
        if path.exists():
            for row in read(path):
                decisions[classifier(safe_projection(row), path.parent)['status']] += 1
        active_report = {'pid': active['pid'], 'round': active['round'], 'checkpointed_reported': active.get('checkpointed_in_batch'),
                         'seconds_since_checkpoint': active.get('seconds_since_checkpoint'), 'saved_classifications': dict(decisions),
                         'provider_diagnostics': provider_events(run, active['round'])}
    free = shutil.disk_usage(run).free
    return {'status': 'SELECTED_EXECUTION_COMPLETE_ANALYSIS_PENDING' if terminal else 'SELECTED_SCOPE_ATTENTION' if state.get('attention') else 'SELECTED_SCOPE_IN_PROGRESS',
            'selected_arms': [ARM], 'state_phase': state.get('phase'), 'selected_scope_complete': terminal,
            'original_three_arm_benchmark_complete': False, 'analysis_status': state.get('analysis_status'),
            'supervisor_pid': state.get('supervisor_pid'), 'heartbeat_at': state.get('heartbeat_at'),
            'selected_accepted_persisted': selected, 'selected_accepted_state': state.get('accepted', {}).get(ARM),
            'selected_counts_persisted': ledger['counts'][ARM], 'active_single_agent': active_report,
            'attention': state.get('attention'), 'attention_reason': state.get('reason'),
            'provider_rejections': state.get('provider_rejections'),
            'disk_free_gib': free / 1024 ** 3, 'disk_minimum_gib': manifest['settings'].get('min_free_gb'),
            'disk_guard_clear': free >= manifest['settings'].get('min_free_gb', 5) * 1024 ** 3,
            'preserved_unselected': preserved,
            'current_native': native_status(state, terminal) if not args.skip_native else {'required': 'skipped_by_request'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('observe', 'verify'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--preserved-metadata', type=Path)
    parser.add_argument('--round', type=int)
    parser.add_argument('--accepted', type=int)
    parser.add_argument('--clean', type=int)
    parser.add_argument('--runtime', type=int)
    parser.add_argument('--final-arm', action='store_true')
    parser.add_argument('--skip-native', action='store_true', help='Observer only; verifier never skips liveness')
    parser.add_argument('--append-validation', action='store_true', help='Append verifier-owned selected-scope validation evidence only')
    args = parser.parse_args()
    require(not (args.command == 'verify' and args.skip_native), 'Verifier requires native liveness or terminal proof')
    run = args.run_dir.resolve()
    snapshot = args.preserved_metadata.resolve() if args.preserved_metadata else run / AMENDMENT
    record = (verify if args.command == 'verify' else observe)(run, snapshot, args)
    record.update(at_utc=datetime.now(timezone.utc).isoformat(), monitor_sha256=sha(__file__),
                  manifest_sha256=sha(run / 'run_manifest.json'), correctness_used_for_scheduling=False)
    if args.append_validation:
        require(args.command == 'verify', 'Only verifier may append validation evidence')
        with (run / 'single_agent_monitor_validations.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'REFUSED', 'error_type': type(error).__name__,
                          'reason': str(error) if isinstance(error, ValueError) else 'Monitor could not validate selected scope'}))
        raise SystemExit(2)
