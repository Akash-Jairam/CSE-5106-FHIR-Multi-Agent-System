"""Private full-cohort synthetic tests. No live benchmark data or model calls."""
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
REAL = Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
CANDIDATE = Path('/private/tmp/v3-paired-provider-amendment/export_complete_v3_single.py')
spec = importlib.util.spec_from_file_location('paired_candidate', CANDIDATE)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    with tempfile.TemporaryDirectory(prefix='paired-v3-single-fixture-', dir='/private/tmp') as temporary:
        base = Path(temporary)
        project, run, historical, amendment = base / 'project', base / 'run', base / 'old_v3_export', base / 'amendment'
        paths = ['benchmarks/v3-full-run-20261015/export_complete_v3.py', 'scripts/full_benchmark.py', 'scripts/audit_store.py',
                 'scripts/v3_full_analysis.py', 'scripts/experiment_stats.py', 'utils/answer_scorer.py',
                 'utils/evidence_scorer.py', 'utils/metrics_utils.py', 'run_agent.py']
        for relative in paths:
            destination = project / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REAL / relative, destination)
        for directory in ('scripts', 'utils'):
            (project / directory / '__init__.py').write_text('')
        old_runner = REAL / 'output/experiments/gpt-6-luna/v3_full_2931_20261015/amendments/0003-single-agent-only-resume/pre_apply_source/scripts/v3_full_run.py'
        runner = project / 'scripts/v3_full_run.py'
        runner.write_bytes(old_runner.read_bytes() + b'\n# Synthetic selected-arm scheduling-only amendment.\n')
        u = candidate.load_utilities(project)
        run.mkdir()
        inputs = [{'question_id': f'q{i:04}', 'question': 'Synthetic value?', 'true_answer': '[]', 'assumption': '',
                   'patient_fhir_id': f'p{i % 2}', 'true_fhir_ids': '{}'} for i in range(2931)]
        source, families = run / 'source.csv', run / 'families.csv'
        write_csv(source, inputs)
        write_csv(families, [{'question_id': row['question_id'], 'family': 'lookup_counting', 'template': 'Count', 'template_id': 'T001'} for row in inputs])
        split = {'splits': {'dev': {'patients': [{'patient_fhir_id': 'p0'}, {'patient_fhir_id': 'p1'}],
                                    'question_ids': [row['question_id'] for row in inputs]}}}
        write_json(run / 'split.json', split)
        pricing = {'label': 'Synthetic matched official-price fixture', 'currency': 'USD',
                   'per_million_tokens': {'input': .1, 'cached_input': .01, 'output': .5}}
        write_json(run / 'pricing.json', pricing)
        for name in sorted(candidate.SNAPSHOT_OMIT_PUBLIC):
            path = run / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('Synthetic public pin fixture only.\n')
        identities = {arm: {'version': 2, 'agent_strategy': 'multi_agent_v3' if arm == 'multi_agent' else arm,
                           'model': 'fixture', 'max_total_tokens': 200000, 'max_tool_calls': 30,
                           'input_hash': u.input_identity_hash(source), 'code_hash': candidate.FINGERPRINT,
                           'audit_storage': 'gzip'} for arm in ('direct_llm', 'single_agent', 'multi_agent')}
        source_pins = {relative: u.sha256(project / relative) for relative in paths if relative.endswith('.py') and not relative.startswith('benchmarks')}
        source_pins['scripts/v3_full_run.py'] = u.sha256(old_runner)
        old = {'version': 1, 'created_at': '2026-10-16T04:16:57+00:00', 'questions_per_arm': 2931,
               'protocol': 'v3-first-runtime-clean-terminal', 'classifier_version': 'v3-infrastructure-v3',
               'project_root': str(project), 'arms': {'direct_llm': 'direct_llm', 'single_agent': 'single_agent', 'multi_agent': 'multi_agent_v3'},
               'input': {'path': str(source), 'sha256': u.sha256(source)},
               'families': {'path': str(families), 'sha256': u.sha256(families)},
               'split_manifest': {'path': str(run / 'split.json'), 'sha256': u.sha256(run / 'split.json')},
               'pinned_files_sha256': {name: u.sha256(run / name) for name in candidate.PUBLIC_PINS},
               'source_sha256': source_pins, 'run_identities': identities,
               'settings': {'model': 'fixture', 'num_processes': 2, 'max_total_tokens': 200000, 'max_tool_calls': 30}}
        current = json.loads(json.dumps(old))
        current['source_sha256']['scripts/v3_full_run.py'] = u.sha256(runner)
        current['selected_arms'] = ['single_agent']
        write_json(run / 'run_manifest.json', current)
        write_json(amendment / 'pre_apply_metadata/run_manifest.json', old)
        write_json(historical / 'validated_v3_input_snapshot/run_manifest.json', old)
        archived_runner = amendment / 'pre_apply_source/scripts/v3_full_run.py'
        archived_runner.parent.mkdir(parents=True)
        shutil.copyfile(old_runner, archived_runner)
        ledger = {'version': 1, 'protocol': 'v3-first-runtime-clean-terminal', 'selection': 'first_runtime_clean_terminal',
                  'complete': False, 'arms': {arm: {} for arm in identities},
                  'counts': {arm: {'raw_attempts': 0, 'runtime_failures': 0, 'intrinsic_errors': 0} for arm in identities}}

        def raw_row(arm, index):
            return {'question_id': inputs[index]['question_id'], 'patient_fhir_id': inputs[index]['patient_fhir_id'],
                    'agent_answer': None if arm == 'multi_agent' or index % 2 == 0 else 'A nonempty wrong value',
                    'latency': 1 if arm == 'multi_agent' else 2, 'error': None, 'evidence': [], 'limitation': None,
                    'revision': None, 'tool_calls': [], 'budget': {'tool_calls_run': 0, 'stopped': None},
                    'trace': [{'role': 'system', 'content': 'Synthetic ' + arm}],
                    'tokens': {'prompt': 100 if arm == 'multi_agent' else 120, 'completion': 10 if arm == 'multi_agent' else 20,
                               'cached_prompt': 20 if arm == 'multi_agent' else 30, 'total': 110 if arm == 'multi_agent' else 140,
                               'cost': None, 'llm_calls': 1, 'repeated_prompt': 0,
                               'per_call': [{'prompt': 100 if arm == 'multi_agent' else 120,
                                             'completion': 10 if arm == 'multi_agent' else 20,
                                             'cached': 20 if arm == 'multi_agent' else 30, 'usage_available': True}]}}

        def batch(arm, number, selected, raw_rows):
            directory = run / 'attempts' / arm
            directory.mkdir(parents=True, exist_ok=True)
            shard, output = directory / f'round-{number:04}.csv', directory / f'round-{number:04}.json'
            write_csv(shard, selected)
            from scripts.audit_store import externalize_result_audit
            rows = [externalize_result_audit(row, directory, question_id=row['question_id']) for row in raw_rows]
            write_json(output, rows)
            identity = {**identities[arm], 'input_hash': u.input_identity_hash(shard)}
            write_json(Path(str(output) + '.run.json'), identity)
            descriptor = directory / f'round-{number:04}.batch.json'
            write_json(descriptor, {'arm': arm, 'round': number, 'input': str(shard), 'input_sha256': u.sha256(shard),
                                     'question_ids': [row['question_id'] for row in selected], 'output': str(output), 'run_identity': identity})
            write_json(descriptor.with_suffix('.seal.json'), {'batch_sha256': u.sha256(descriptor), 'result_sha256': u.sha256(output),
                       'identity_sha256': u.sha256(Path(str(output) + '.run.json')), 'sealed_at': '2026-10-17T04:00:00+00:00'})
            return rows

        for arm in ('multi_agent', 'single_agent'):
            number = 1
            if arm == 'single_agent':
                runtime = raw_row(arm, 0)
                runtime.update(error='Max retries exceeded: HTTP 429', latency=99,
                               tokens={'total': 999, 'per_call': [{'usage_available': False}]})
                batch(arm, 1, inputs[:1], [runtime])
                number = 2
            rows = batch(arm, number, inputs, [raw_row(arm, i) for i in range(2931)])
            write_json(run / f'full_{arm}.json', rows)
            write_json(run / f'full_{arm}.json.run.json', identities[arm])
            for row in rows:
                origin = run / 'attempts' / arm / row['audit_artifacts']['path']
                dest = run / row['audit_artifacts']['path']
                dest.parent.mkdir(exist_ok=True)
                os.link(origin, dest)
            ledger['arms'][arm] = {row['question_id']: {'attempt_file': f'attempts/{arm}/round-{number:04}.json', 'round': number,
                  'runtime_failures_before': int(arm == 'single_agent' and i == 0), 'selection': 'runtime_clean_terminal', 'runtime_clean': True}
                  for i, row in enumerate(rows)}
            ledger['counts'][arm] = {'raw_attempts': 2931 + int(arm == 'single_agent'), 'runtime_failures': int(arm == 'single_agent'), 'intrinsic_errors': 0}
        write_json(run / 'full_direct_llm.json', [{'question_id': 'q0000', 'agent_answer': 'DEFERRED_DIRECT_SENTINEL'}])
        ledger['arms']['direct_llm'] = {'q0000': {'operational_only_partial_sentinel': True}}
        write_json(run / 'acceptance.json', ledger)
        write_json(run / 'supervisor_state.json', {'started_at': '2026-10-16T04:16:57+00:00', 'phase': 'complete_selected_arms',
                                                  'selected_scope_complete': True, 'benchmark_complete': False, 'active': {},
                                                  'scope_started_at': '2026-10-17T04:00:00+00:00', 'finished_at': '2026-10-17T05:00:00+00:00'})
        (run / 'operations.jsonl').write_text(json.dumps({'event': 'supervisor_started', 'at': '2026-10-16T04:16:57+00:00'}) + '\n')
        historical_receipt = {'status': 'V3_COMPLETE_CONTROLS_PENDING', 'questions': 2931, 'v3_complete': True,
              'source_manifest_sha256': u.sha256(historical / 'validated_v3_input_snapshot/run_manifest.json'),
              'v3_canonical_sha256_after': u.sha256(run / 'full_multi_agent.json'),
              'v3_ledger_slice_sha256_after': u.semantic_hash({'selections': ledger['arms']['multi_agent'], 'counts': ledger['counts']['multi_agent']}),
              'pricing': pricing, 'helper_sha256': u.sha256(project / 'benchmarks/v3-full-run-20261015/export_complete_v3.py')}
        write_json(historical / 'V3_COMPLETE_CONTROLS_PENDING.json', historical_receipt)
        checks = []
        def refusal(expected):
            output = base / f'must_not_exist_{len(checks)}'
            try:
                candidate.deliver(project, run, historical, amendment, output)
            except ValueError as error:
                assert expected in str(error), str(error)
            else:
                raise AssertionError('Invalid paired export accepted')
            assert not output.exists()
            assert 'scripts.v3_full_analysis' not in sys.modules and 'utils.answer_scorer' not in sys.modules
            checks.append(expected)

        incomplete = json.loads(json.dumps(ledger))
        incomplete['arms']['single_agent'].pop('q0000')
        write_json(run / 'acceptance.json', incomplete)
        refusal('incomplete acceptance cohort')
        write_json(run / 'acceptance.json', ledger)
        bad = json.loads(json.dumps(current))
        bad['run_identities']['single_agent']['code_hash'] = 'wrong'
        write_json(run / 'run_manifest.json', bad)
        refusal('frozen inference identity changed')
        write_json(run / 'run_manifest.json', current)
        original_runner = runner.read_bytes()
        runner.write_bytes(original_runner.replace(b'CLASSIFIER_VERSION = "v3-infrastructure-v3"', b'CLASSIFIER_VERSION = "changed"'))
        changed = json.loads(json.dumps(current));changed['source_sha256']['scripts/v3_full_run.py'] = u.sha256(runner)
        write_json(run / 'run_manifest.json', changed)
        refusal('classifier semantics')
        runner.write_bytes(original_runner);write_json(run / 'run_manifest.json', current)
        identity_file = run / 'full_single_agent.json.run.json'
        identity = json.loads(identity_file.read_text());identity['max_tool_calls'] = 29
        write_json(identity_file, identity)
        refusal('frozen inference identity changed')
        write_json(identity_file, identities['single_agent'])
        public_pin = run / 'provider_profile.json'
        original_pin = public_pin.read_bytes()
        public_pin.write_bytes(original_pin + b'Tampered synthetic public fixture.\n')
        refusal('Frozen public pinned file changed: provider_profile.json')
        public_pin.write_bytes(original_pin)

        before = {name: u.sha256(run / name) for name in ['run_manifest.json', 'acceptance.json', 'full_multi_agent.json', 'full_single_agent.json', 'full_direct_llm.json']}
        output = base / 'complete_pair'
        receipt = candidate.deliver(project, run, historical, amendment, output)
        report = u.read_json(output / 'paired_complete/summary.json')
        global_report = u.read_json(output / 'global_partial/summary.json')
        assert receipt['status'] == 'V3_SINGLE_COMPLETE_DIRECT_PENDING' and not receipt['benchmark_complete']
        assert receipt['inference_fingerprint'] == candidate.FINGERPRINT and receipt['operational_source_changes'].keys() == {'scripts/v3_full_run.py'}
        assert set(report['arms']) == {'multi_agent', 'single_agent'} and 'all_arms_accepted_metrics' not in report
        assert global_report['status'] == 'PARTIAL_PROGRESS_DO_NOT_REPORT_AS_FINAL' and global_report['completeness']['direct_llm']['accepted'] == 0
        assert not (output / 'validated_paired_input_snapshot/full_direct_llm.json').exists()
        assert set(receipt['snapshot']['verified_public_pins_omitted']) == candidate.SNAPSHOT_OMIT_PUBLIC
        assert all(not (output / 'validated_paired_input_snapshot' / name).exists() for name in candidate.SNAPSHOT_OMIT_PUBLIC)
        assert len(report['comparisons']) == 1
        pair = report['comparisons'][0]
        assert pair['left'] == 'single_agent' and pair['right'] == 'multi_agent' and pair['matched_questions'] == 2931
        assert pair['left_only'] == pair['right_only'] == 0 and pair['left_wins'] == 0 and pair['right_wins'] == 1465
        assert abs(pair['accuracy_difference']['estimate'] + 1465 / 2931) < 1e-12
        assert pair['accuracy_difference']['ci_low'] == -1 and pair['accuracy_difference']['ci_high'] == 0
        assert pair['latency_seconds_difference_observed_pairs']['estimate'] == 1
        assert pair['latency_seconds_difference_observed_pairs']['ci_low'] == 1 and pair['latency_seconds_difference_observed_pairs']['ci_high'] == 1
        assert report['arms']['single_agent']['metrics']['latency_seconds']['total'] == 5862
        assert report['arms']['multi_agent']['metrics']['latency_seconds']['total'] == 2931
        assert report['arms']['single_agent']['metrics']['total_tokens']['total'] == 2931 * 140
        assert report['arms']['multi_agent']['metrics']['total_tokens']['total'] == 2931 * 110
        assert math.isclose(report['arms']['single_agent']['metrics']['estimated_cost_usd']['total'], 2931 * .0000193, rel_tol=1e-12)
        assert math.isclose(report['arms']['multi_agent']['metrics']['estimated_cost_usd']['total'], 2931 * .0000132, rel_tol=1e-12)
        assert receipt['selected_single_scope_timing']['wall_clock_seconds'] == 3600
        assert all(receipt['missing_measurements'][arm]['estimated_cost_usd'] == 0 for arm in candidate.ARMS)
        assert receipt['first_clean_replay_counts']['single_agent']['runtime_failures'] == 1
        csv.field_size_limit(sys.maxsize)
        for path in (output / 'paired_complete').glob('*.csv'):
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
                assert all(None not in row and row.get('arm', 'single_agent') in candidate.ARMS for row in rows)
                if path.name == 'per_question.csv':assert len(rows) == 5862
                if path.name == 'paired_comparisons.csv':assert len(rows) == 1
        assert len((output / 'paired_complete/per_question.jsonl').read_text().splitlines()) == 5862
        assert before == {name: u.sha256(run / name) for name in before}
        checks.extend(['full2931each_positive_export', 'no_direct_scores_or_comparisons', 'paired2931_accuracy_and_latency_CIs',
                       'runtime_cost_time_token_exclusion', 'all_csv_jsonl_parse', 'all_live_fixture_artifacts_unchanged'])
        evidence = {'status': 'PASS', 'checks': checks, 'helper_sha256': u.sha256(CANDIDATE), 'test_sha256': u.sha256(Path(__file__)),
                    'synthetic_questions_per_arm': 2931, 'actual_live_data_scored': False, 'private_fixture_only': True}
        write_json(Path('/private/tmp/v3-paired-provider-amendment/paired_export_test_results.json'), evidence)
        print(json.dumps(evidence))


if __name__ == '__main__':
    main()
