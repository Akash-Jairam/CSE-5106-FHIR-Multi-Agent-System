"""Offline progress audit; never launches inference or modifies accepted results.

Run from the project root with the benchmark's .venv and dependency PYTHONPATH.
Native process inspection requires the same host permissions as normal monitoring.
This helper is outside the frozen inference and operational source lists.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.audit_store import restore_result_audit
from scripts.v3_full_run import classify_attempt, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--arm', choices=('multi_agent', 'single_agent', 'direct_llm'), default='multi_agent')
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--accepted', type=int, required=True)
    parser.add_argument('--clean', type=int, required=True)
    parser.add_argument('--runtime', type=int, required=True)
    parser.add_argument('--final-arm', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest = json.loads((run / 'run_manifest.json').read_text())
    output = run / 'attempts' / args.arm / f'round-{args.round:04d}.json'
    batch_path = output.with_suffix('.batch.json')
    batch = json.loads(batch_path.read_text())
    seal = json.loads(batch_path.with_suffix('.seal.json').read_text())
    identity = Path(str(output) + '.run.json')
    assert seal['batch_sha256'] == file_sha256(batch_path), 'batch seal mismatch'
    assert seal['result_sha256'] == file_sha256(output), 'result seal mismatch'
    assert seal['identity_sha256'] == file_sha256(identity), 'identity seal mismatch'
    assert json.loads(identity.read_text()) == batch['run_identity'], 'raw identity mismatch'
    assert batch['run_identity']['code_hash'] == manifest['run_identities'][args.arm]['code_hash'], 'inference identity changed'
    assert file_sha256(batch['input']) == batch['input_sha256'], 'input shard changed'
    rows = json.loads(output.read_text())
    decisions = {row['question_id']: classify_attempt(row, output.parent)['status'] for row in rows}
    expected = {key: value for key, value in {'accepted': args.clean, 'runtime': args.runtime}.items() if value}
    assert len(rows) == len(decisions) == len(batch['question_ids']), 'duplicate or missing results'
    assert set(decisions) == set(batch['question_ids']), 'question IDs changed'
    assert dict(Counter(decisions.values())) == expected, 'unexpected attempt classifications'
    for row in rows:
        restore_result_audit(row, output.parent, restore_history=False)
    failed = {qid for qid, status in decisions.items() if status == 'runtime'}
    ledger = json.loads((run / 'acceptance.json').read_text())
    canonical = json.loads((run / f'full_{args.arm}.json').read_text())
    accepted_ids = {row['question_id'] for row in canonical}
    assert len(canonical) == len(accepted_ids) == args.accepted, 'accepted count mismatch'
    assert set(ledger['arms'][args.arm]) == accepted_ids, 'canonical/ledger mismatch'
    counts = ledger['counts'][args.arm]
    assert counts['raw_attempts'] == args.accepted + counts['runtime_failures'], 'unresolved classified attempts'
    if args.final_arm:
        assert args.accepted == manifest['questions_per_arm'] and not failed, 'arm incomplete'
    else:
        next_path = run / 'attempts' / args.arm / f'round-{args.round + 1:04d}.batch.json'
        deadline = time.monotonic() + 45
        while not next_path.exists() and time.monotonic() < deadline:
            time.sleep(1)
        assert next_path.exists(), 'next batch not yet available; inspect supervisor'
        next_batch = json.loads(next_path.read_text())
        assert failed <= set(next_batch['question_ids']), 'runtime retry omitted'
        assert not (set(next_batch['question_ids']) & accepted_ids), 'accepted question repeated'
    state = json.loads((run / 'supervisor_state.json').read_text())
    assert not state.get('attention'), 'supervisor requests attention'
    successful_terminal = state.get('phase') == 'complete'
    if successful_terminal:
        assert args.final_arm, 'terminal verification requires --final-arm'
        assert type(state.get('analysis_return_code')) is int and state['analysis_return_code'] == 0, 'analysis did not succeed'
        assert state.get('finished_at'), 'successful terminal timestamp missing'
        assert ledger.get('complete') is True, 'complete acceptance ledger required'
        assert all(state.get('accepted', {}).get(arm) == manifest['questions_per_arm']
                   for arm in manifest['run_identities']), 'terminal arm counts incomplete'
        assert not state.get('active'), 'terminal state still has active workers'
    elif not args.final_arm:
        active_arm = state.get('active', {}).get(args.arm)
        assert active_arm and active_arm.get('round', 0) > args.round, 'selected arm has no active successor'
    live = {}
    if not successful_terminal:
        pids = [state['supervisor_pid']] + [detail['pid'] for detail in state.get('active', {}).values()]
        if state.get('phase') == 'analyzing' and type(state.get('analysis_pid')) is int:
            pids.append(state['analysis_pid'])
        check = subprocess.run(['ps', '-p', ','.join(map(str, pids)), '-o', 'pid=,stat=,comm='], text=True, capture_output=True)
        for line in check.stdout.splitlines():
            parts = line.strip().split(None, 2)
            if len(parts) == 3:
                live[parts[0]] = parts[1]
        assert all(str(pid) in live and 'Z' not in live[str(pid)] for pid in pids), 'inspect missing owned process'
    errors = []
    for journal in (run / 'diagnostics' / args.arm / f'round-{args.round:04d}').glob('*.jsonl'):
        for line in journal.read_text().splitlines():
            event = json.loads(line)
            if event.get('outcome') == 'error':
                errors.append({'http_status': event.get('http_status'), 'duration_seconds': event.get('duration_seconds'),
                               'exception_types': [item.get('type', {}).get('class') for item in event.get('exception_chain', [])]})
    record = {'version': 1, 'at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'COMPLETE' if successful_terminal else 'IN_PROGRESS_NOT_FINAL',
              'arm': args.arm, 'sealed_round': args.round, 'accepted': args.accepted, 'expected_questions_per_arm': manifest['questions_per_arm'],
              'counts': counts, 'batch_classifications': dict(Counter(decisions.values())),
              'batch_seal_result_identity_input_and_audits_verified': len(rows), 'canonical_ledger_ids_match': True,
              'all_runtime_questions_scheduled': not args.final_arm, 'accepted_rerun_overlap': 0,
              'supervisor_pid': state['supervisor_pid'], 'active': state.get('active'), 'native_live_states': live,
              'successful_terminal_state_verified': successful_terminal, 'native_liveness_required': not successful_terminal,
              'supervisor_phase': state.get('phase'), 'analysis_return_code': state.get('analysis_return_code'),
              'attention': state.get('attention'), 'provider_error_events': errors,
              'manifest_sha256': file_sha256(run / 'run_manifest.json'), 'correctness_used_for_scheduling': False,
              'verifier_sha256': file_sha256(Path(__file__))}
    with (run / 'monitor_validations.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
