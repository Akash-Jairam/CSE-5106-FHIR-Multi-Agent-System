"""Independent read-only QA of a delivered COMPLETE 2,931-question V3 arm.

Never calls an answer scorer/analyzer/provider/launcher. Exported correctness
flags are used only after exact full-cohort/receipt validation, to independently
recalculate aggregate statistics. Writes only a fresh external QA directory.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
N = 2931
TOKEN_FIELDS = ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cached_prompt_tokens',
                'uncached_prompt_tokens', 'repeated_prompt_tokens')
REQUIRED_STATUS = 'V3_COMPLETE_CONTROLS_PENDING'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def semantic(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def close(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) <= set(actual), label + ': missing keys')
        for key, value in expected.items():
            close(actual[key], value, label + '.' + key)
    elif expected is None:
        require(actual is None, label + ': expected null')
    elif type(expected) in (int, float):
        require(type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-11), label + ': numeric mismatch')
    else:
        require(actual == expected, label + ': mismatch')


def independent_measure(values):
    import numpy as np
    finite = [value for value in values if number(value) is not None]
    observed = {key: None for key in ('total', 'mean', 'median', 'p95', 'stdev', 'min', 'max')}
    if finite:
        observed = {'total': math.fsum(finite), 'mean': statistics.fmean(finite), 'median': statistics.median(finite),
                    'p95': float(np.percentile(finite, 95)), 'stdev': statistics.stdev(finite) if len(finite) > 1 else None,
                    'min': min(finite), 'max': max(finite)}
    missing = len(values) - len(finite)
    return {'n': len(values), 'observed_n': len(finite), 'missing_n': missing,
            **{key: None if missing else value for key, value in observed.items()}, 'observed_only': observed}


def independent_clustered_mean(values, patients, intervals=True):
    import numpy as np
    by_patient = defaultdict(list)
    for patient, value in zip(patients, values):
        by_patient[patient].append(value)
    result = {'n': len(values), 'patients': len(by_patient),
              'estimate': statistics.fmean(values) if values else None, 'ci_low': None, 'ci_high': None}
    if intervals and len(by_patient) >= 2:
        groups = [by_patient[patient] for patient in sorted(by_patient)]
        sums = np.array([math.fsum(group) for group in groups])
        sizes = np.array([len(group) for group in groups])
        random = np.random.default_rng(5106)
        weights = random.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=10000)
        draws = (weights @ sums) / (weights @ sizes)
        result['ci_low'], result['ci_high'] = [float(value) for value in np.percentile(draws, [2.5, 97.5])]
    return result


def metric_values(rows, name):
    return [None if name in TOKEN_FIELDS and row['provider_usage_unknown'] else row[name] for row in rows]


def verify_aggregate(actual, rows, label, intervals=True):
    expected = {'questions': len(rows), 'correct': sum(row['correct'] for row in rows),
                'intrinsic_errors': sum(row['intrinsic_error'] for row in rows),
                'abstentions': sum(row['abstained'] for row in rows), 'limitations': sum(row['has_limitation'] for row in rows),
                'budget_stops': sum(row['budget_stopped'] for row in rows),
                'provider_usage_unknown': sum(row['provider_usage_unknown'] for row in rows),
                'evidence_validated_questions': sum(row['evidence_validation_available'] for row in rows)}
    close(actual, expected, label)
    close(actual['accuracy'], independent_clustered_mean([int(row['correct']) for row in rows],
          [row['patient_fhir_id'] for row in rows], intervals), label + '.accuracy')
    for name, metric in actual['metrics'].items():
        values = metric_values(rows, name)
        close(metric, independent_measure(values), label + '.' + name)
        if intervals and values and all(number(value) is not None for value in values) and name in (
                'latency_seconds', 'estimated_cost_usd', 'total_tokens'):
            close(metric['mean_patient_bootstrap'], independent_clustered_mean(values,
                  [row['patient_fhir_id'] for row in rows]), label + '.' + name + '.bootstrap')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--export-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    run, export, output = args.run_dir.resolve(), args.export_dir.resolve(), args.output_dir.resolve()
    require(not output.exists() and not output.is_relative_to(run) and not output.is_relative_to(export), 'QA output must be fresh and outside run/export')
    receipt = json.loads((export / 'V3_COMPLETE_CONTROLS_PENDING.json').read_text())
    require(receipt['status'] == REQUIRED_STATUS and receipt['v3_complete'] is True
            and receipt['benchmark_complete'] is False and receipt['questions'] == N, 'Complete V3 receipt required before QA')
    manifest = json.loads((run / 'run_manifest.json').read_text())
    require(digest(run / 'run_manifest.json') == receipt['source_manifest_sha256'], 'Run manifest changed')
    ledger = json.loads((run / 'acceptance.json').read_text())
    ledger_slice = {'selections': ledger['arms']['multi_agent'], 'counts': ledger['counts']['multi_agent']}
    require(len(ledger_slice['selections']) == N and semantic(ledger_slice) == receipt['v3_ledger_slice_sha256_after'], 'V3 accepted ledger changed or incomplete')
    canonical_path = run / 'full_multi_agent.json'
    canonical_hash = digest(canonical_path)
    require(canonical_hash == receipt['v3_canonical_sha256_before'] == receipt['v3_canonical_sha256_after'], 'Canonical V3 content changed')
    sys.path.insert(0, str(args.project_root.resolve()))
    from scripts.full_benchmark import iter_result_rows
    from scripts.v3_full_run import classify_attempt
    from scripts.audit_store import restore_result_audit
    require(digest(args.project_root / 'scripts/v3_full_run.py') == receipt['independently_approved_classifier']['sha256'], 'Approved classifier source changed')
    identity = json.loads(Path(str(canonical_path) + '.run.json').read_text())
    require(identity == manifest['run_identities']['multi_agent'] and digest(Path(str(canonical_path) + '.run.json')) == receipt['v3_identity_sha256'], 'V3 identity changed')
    patients = {}
    with Path(manifest['input']['path']).open(newline='') as stream:
        for row in csv.DictReader(stream):
            require(row['question_id'] not in patients, 'Duplicate source question')
            patients[row['question_id']] = row['patient_fhir_id']
    require(len(patients) == N and digest(manifest['input']['path']) == manifest['input']['sha256'], 'Full frozen source required')
    canonical = {}
    canonical_digests = {}
    for row in iter_result_rows(canonical_path):
        qid = row['question_id']
        require(qid in patients and qid not in canonical and row['patient_fhir_id'] == patients[qid]
                and not row.get('attempt_history'), 'Canonical question/patient/history mismatch')
        canonical_digests[qid] = semantic(row)
        canonical[qid] = {key: row.get(key) for key in ('question_id', 'patient_fhir_id', 'agent_answer', 'evidence', 'latency',
                                                       'limitation', 'tokens', 'budget', 'error')}
        canonical[qid]['tool_calls_count'] = len(row['tool_calls']) if isinstance(row.get('tool_calls'), list) else None
    require(set(canonical) == set(patients) == set(ledger_slice['selections']), 'All 2,931 canonical questions required before calculations')
    require(semantic(canonical_digests) == receipt['selected_raw_rows_sha256'], 'Selected canonical digest differs from receipt')
    records = {}
    with (export / 'v3_complete/per_question.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            qid = row['question_id']
            require(row['arm'] == 'multi_agent' and qid not in records, 'Unexpected or duplicate exported arm/question')
            records[qid] = row
    require(set(records) == set(canonical), 'Export does not contain exact 2,931-question cohort')
    pricing = json.loads((run / 'pricing.json').read_text())
    require(pricing == receipt['pricing'], 'Frozen price schedule mismatch')
    rates = pricing['per_million_tokens']
    require(rates == {'input': .1, 'cached_input': .01, 'output': .5}, 'Unexpected official frozen rates')
    for qid, row in records.items():
        raw, selected = canonical[qid], ledger_slice['selections'][qid]
        require(row['runtime_clean_selection_verified'] and row['acceptance_entry'] == selected and selected['runtime_clean'] is True, 'Export selection ledger differs')
        require(row['patient_fhir_id'] == raw['patient_fhir_id'] and semantic(row['final_answer']) == semantic(raw['agent_answer'])
                and row['evidence'] == raw['evidence'] and row['limitation'] == raw['limitation'], 'Exported answer/evidence changed')
        tokens, budget = raw.get('tokens') or {}, raw.get('budget') or {}
        prompt, cached, completion = (number(tokens.get(key)) for key in ('prompt', 'cached_prompt', 'completion'))
        unknown = any(call.get('usage_available') is False for call in tokens.get('per_call') or [])
        cost = None if unknown or any(value is None for value in (prompt, cached, completion)) or cached > prompt else (
            (prompt - cached) * rates['input'] + cached * rates['cached_input'] + completion * rates['output']) / 1e6
        expected = {'latency_seconds': number(raw.get('latency')), 'estimated_cost_usd': cost,
                    'reported_cost_usd': number(tokens.get('cost')), 'prompt_tokens': prompt,
                    'completion_tokens': completion, 'total_tokens': number(tokens.get('total')), 'cached_prompt_tokens': cached,
                    'uncached_prompt_tokens': prompt - cached if prompt is not None and cached is not None and cached <= prompt else None,
                    'repeated_prompt_tokens': number(tokens.get('repeated_prompt')), 'llm_calls': number(tokens.get('llm_calls')),
                    'tool_calls_executed': number(budget.get('tool_calls_run')),
                    'tool_calls_recorded': raw['tool_calls_count'],
                    'evidence_count': len(raw['evidence']) if isinstance(raw.get('evidence'), list) else None}
        close(row, expected, qid + '.measurements')
        require(row['provider_usage_unknown'] == unknown and row['per_call_usage'] == tokens.get('per_call'), 'Per-call usage changed')
        intrinsic = raw.get('error') is not None and str(raw['error']).strip() not in ('', 'nan')
        require(row['intrinsic_error'] == intrinsic and row['correct'] == (row['typed_answer_matches'] and not intrinsic), 'Final correctness/error flags inconsistent')
    # Replay sealed V3 attempts independently of the delivery gate. No scorer is called.
    choices, raw_counts, runtime_prefix = {}, Counter({'raw_attempts': 0, 'runtime_failures': 0, 'intrinsic_errors': 0}), Counter()
    for batch_path in sorted((run / 'attempts/multi_agent').glob('round-*.batch.json'), key=lambda path: int(path.name.split('-')[1].split('.')[0])):
        batch = json.loads(batch_path.read_text())
        seal_path = batch_path.with_suffix('.seal.json')
        seal = json.loads(seal_path.read_text())
        raw_file = Path(batch['output'])
        require(digest(batch_path) == seal['batch_sha256'] and digest(batch['input']) == batch['input_sha256'], 'Batch/shard seal changed')
        if not raw_file.exists():
            require(seal['result_sha256'] is None, 'Missing sealed raw result')
            continue
        require(digest(raw_file) == seal['result_sha256'] and digest(str(raw_file) + '.run.json') == seal['identity_sha256'], 'Result/identity seal changed')
        require(json.loads(Path(str(raw_file) + '.run.json').read_text()) == batch['run_identity'], 'Raw identity changed')
        seen = set()
        for row in iter_result_rows(raw_file):
            qid = row['question_id']
            require(qid in batch['question_ids'] and qid not in seen and row['patient_fhir_id'] == patients[qid], 'Raw ownership changed')
            seen.add(qid)
            full = restore_result_audit(row, raw_file.parent, restore_history=False)
            decision = classify_attempt(full, None)
            if decision['status'] == 'pending':
                continue
            require(qid not in choices, 'Accepted outcome was rerun')
            raw_counts['raw_attempts'] += 1
            if decision['status'] == 'runtime':
                raw_counts['runtime_failures'] += 1
                runtime_prefix[qid] += 1
            else:
                require(decision['status'] == 'accepted', 'Unresolved attention in sealed attempts')
                raw_counts['intrinsic_errors'] += int(bool(row.get('error')))
                choices[qid] = {'attempt_file': str(raw_file.relative_to(run)), 'round': batch['round'],
                                'runtime_failures_before': runtime_prefix[qid], 'selection': decision['reason'], 'runtime_clean': True}
                require(semantic({key: value for key, value in row.items() if key != 'attempt_history'}) == canonical_digests[qid], 'Canonical is not first clean selected raw row')
        require(seen == set(batch['question_ids']), 'Saved raw rows omitted scheduled placeholders')
    close(dict(raw_counts), ledger_slice['counts'], 'sealed_replay_counts')
    require(choices == ledger_slice['selections'], 'First-clean provenance differs from ledger')
    report = json.loads((export / 'v3_complete/summary.json').read_text())
    global_report = json.loads((export / 'global_partial/summary.json').read_text())
    require(report['status'] == REQUIRED_STATUS and report['benchmark_complete'] is False and set(report['arms']) == {'multi_agent'}
            and 'all_arms_accepted_metrics' not in report and not report['comparisons'], 'V3 delivery scope incorrectly labeled')
    require(global_report['status'] == 'PARTIAL_PROGRESS_DO_NOT_REPORT_AS_FINAL', 'Global partial label changed')
    rows = [records[qid] for qid in sorted(records)]
    verify_aggregate(report['arms']['multi_agent'], rows, 'overall')
    group_rows = 0
    for field, table in report['groups'].items():
        buckets = defaultdict(list)
        for row in rows:
            buckets[str(row[field] or 'unknown')].append(row)
        require({row['group'] for row in table} == set(buckets), 'Group partition differs from per-question data')
        for item in table:
            require(item['arm'] == 'multi_agent', 'Control aggregate leaked into V3 tables')
            verify_aggregate(item, buckets[item['group']], field + '.' + item['group'], intervals=field not in ('patient_fhir_id', 'template_id'))
            group_rows += 1
    for name, metric in report['arms']['multi_agent']['metrics'].items():
        require(receipt['missing_measurements'][name] == metric['missing_n'], 'Receipt missing-measurement count mismatch')
    csv.field_size_limit(sys.maxsize)
    csv_counts = {}
    for path in sorted((export / 'v3_complete').glob('*.csv')):
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            count = 0
            for row in reader:
                require(None not in row and row.get('arm', 'multi_agent') == 'multi_agent', 'Malformed CSV/control row')
                if path.name == 'per_question.csv':
                    expected = records[row['question_id']]
                    for key in ('latency_seconds', 'estimated_cost_usd', 'total_tokens'):
                        close(float(row[key]) if row[key] else None, expected[key], 'csv.' + key)
                count += 1
            csv_counts[path.name] = count
    require(csv_counts['per_question.csv'] == N and csv_counts['arm_summary.csv'] == 1, 'CSV row count mismatch')
    inventory = json.loads((export / 'delivery_artifact_hashes.json').read_text())
    for relative, metadata in inventory.items():
        path = (export / relative).resolve()
        require(path.is_relative_to(export) and path.stat().st_size == metadata['bytes'] and digest(path) == metadata['sha256'], 'Delivered artifact hash mismatch')
    require(digest(canonical_path) == canonical_hash, 'Canonical changed during QA')
    latest = json.loads((run / 'acceptance.json').read_text())
    require(semantic({'selections': latest['arms']['multi_agent'], 'counts': latest['counts']['multi_agent']}) == receipt['v3_ledger_slice_sha256_after'], 'V3 ledger changed during QA')
    output.mkdir(parents=True)
    primary = {key: {name: report['arms']['multi_agent']['metrics'][key][name] for name in ('total', 'mean', 'missing_n')}
               for key in ('latency_seconds', 'estimated_cost_usd', 'total_tokens')}
    evidence = {'status': 'PASS', 'generated_at': datetime.now(timezone.utc).isoformat(), 'questions': N,
                'run_directory': str(run), 'export_directory': str(export), 'canonical_sha256': canonical_hash,
                'v3_ledger_slice_sha256': receipt['v3_ledger_slice_sha256_after'], 'primary_measurements': primary,
                'first_clean_replay_counts': dict(raw_counts), 'aggregate_groups_verified': group_rows,
                'csv_rows_parsed': csv_counts, 'delivery_artifacts_verified': len(inventory),
                'bootstrap': {'independent_implementation': True, 'replicates': 10000, 'seed': 5106, 'unit': 'patient cluster', 'level': .95},
                'no_answer_scorer_or_analyzer_called': True, 'actual_incomplete_data_scored': False,
                'benchmark_complete': False, 'controls_omitted_from_v3_delivery': True,
                'checks': ['All 2931 per-question answers/evidence/latency/tokens/cost reconcile to canonical and accepted ledger',
                           'Every sealed V3 attempt replayed; first clean source-attempt choices unchanged',
                           'Totals/means/missingness/median/p95/stdev and patient-cluster bootstrap intervals independently recalculated',
                           'All V3 group aggregates and CSVs verified; global partial and controls-pending labels preserved',
                           'Delivered artifacts hash verified; V3 canonical and ledger slice stable during QA'],
                'verifier_sha256': digest(Path(__file__))}
    evidence_path = output / 'reconciliation.json'
    evidence_path.write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n')
    hashes = {name: digest(path) for name, path in [('reconciliation.json', evidence_path), ('verifier_source', Path(__file__))]}
    (output / 'reconciliation_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'questions': N, 'primary_measurements': primary, 'evidence_path': str(evidence_path)}))


if __name__ == '__main__':
    main()
