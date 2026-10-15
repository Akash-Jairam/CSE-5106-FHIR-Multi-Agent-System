"""Completed-run timing/error recording only. No scorer, model or credentials."""
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import statistics

ROOT = Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
OUTPUT = ROOT / 'output/experiments/gpt-6-luna/query_contract_v3c_300q'
spec = importlib.util.spec_from_file_location('read_only_results', ROOT / 'scripts/full_benchmark.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def kind(row):
    value = str(row.get('error') or '')
    if not value.strip():
        return 'nonterminal'
    status = re.search(r'Error code: (\d{3})', value)
    if status:
        return 'http_' + status.group(1)
    if re.search(r'APITimeout|Request timed out', value, re.I):
        return 'client_sdk_timeout'
    if re.search(r'APIConnectionError|Connection error', value, re.I):
        return 'api_connection_error'
    return 'other_terminal'

report = {'scope': 'Completed sequential first-attempt timing/error metadata; no answer/reference inspection or scoring',
          'model_calls': 0, 'arms': {}}
for arm, log_name in (('ordinary_single_agent', 'run_single.log'), ('ordinary_multi_agent', 'run_team.log')):
    path = OUTPUT / (arm + '.json')
    rows = list(module.iter_result_rows(path))
    if len(rows) != 300 or any(row.get('latency') is None and kind(row) == 'nonterminal' for row in rows):
        raise ValueError('Completed cohort required before recording diagnostics')
    types = Counter(kind(row) for row in rows)
    segments, start = [], None
    for index, row in enumerate(rows, 1):
        bad = kind(row) != 'nonterminal'
        if bad and start is None:
            start = index
        if not bad and start is not None:
            segments.append({'first_row': start, 'last_row': index - 1, 'length': index - start})
            start = None
    if start is not None:
        segments.append({'first_row': start, 'last_row': 300, 'length': 301 - start})
    timed = [row['latency'] for row in rows if isinstance(row.get('latency'), (int, float))]
    calls = [call for row in rows for call in (row.get('tokens') or {}).get('per_call') or []]
    failed_rows = [row for row in rows if kind(row) != 'nonterminal']
    phase_missing = sum(not (row.get('revision') or {}).get('phases') for row in failed_rows)
    log = OUTPUT / log_name
    stamp_count = 0
    earliest = latest = None
    with log.open(errors='replace') as stream:
        for line in stream:
            stamp = re.match(r'^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})', line)
            if stamp:
                stamp_count += 1
                earliest = earliest or stamp[1]
                latest = stamp[1]
    report['arms'][arm] = {'terminal_kinds': dict(types),
        'first_terminal_row': next((i for i,row in enumerate(rows,1) if kind(row) != 'nonterminal'), None),
        'longest_error_segments': sorted(segments, key=lambda item:item['length'], reverse=True)[:3],
        'recorded_latency_sum_seconds': sum(timed), 'recorded_latency_median_seconds': statistics.median(timed),
        'reported_model_calls': sum((row.get('tokens') or {}).get('llm_calls', 0) for row in rows),
        'usage_unknown_calls': sum(call.get('usage_available') is False for call in calls),
        'failed_rows_unknown_usage': sum(any(c.get('usage_available') is False for c in (row.get('tokens') or {}).get('per_call') or []) for row in failed_rows),
        'failed_rows_zero_accounted_tokens': sum((row.get('tokens') or {}).get('total') == 0 for row in failed_rows),
        'failed_rows_without_phase_records': phase_missing,
        'identity_mtime_utc': datetime.fromtimestamp(Path(str(path)+'.run.json').stat().st_mtime,timezone.utc).isoformat(),
        'final_output_mtime_utc': datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
        'timestamp_prefixed_log_lines': stamp_count, 'first_log_timestamp': earliest, 'last_log_timestamp': latest}
report['limitations'] = [
    'Result rows are sequential per arm; row indices identify execution order, not exact per-request UTC intervals.',
    'Per-question latency excludes driver gaps; identity/output file mtimes bracket recording and do not establish request-level times.',
    'Persisted generic APIConnectionError does not identify DNS/TLS/proxy/reset cause; private headers/config and raw error bodies were not printed.',
    'Early contract-fatal returns can omit revision.phases; observed fatal-phase counts are not exhaustive.',
    'Zero accounted tokens accompanying usage_available=false are missing-usage placeholders, not evidence of zero billed work.']
destination = Path(__file__).with_name('v3c_recording_diagnostics.json')
destination.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
