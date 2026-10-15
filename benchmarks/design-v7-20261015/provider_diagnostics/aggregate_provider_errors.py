"""Completed V7 safe provider-journal metadata only; no results/config imports."""
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
JOURNALS = REPO / 'output/experiments/gpt-6-luna/residual_trajectories_v7_300q/provider-events'
DEST = Path(__file__).parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def utc(value):
    parsed = datetime.fromisoformat(value)
    assert parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0
    return parsed


def delta(left, right):
    return round((utc(right) - utc(left)).total_seconds(), 6)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def summary(values):
    return {'n': len(values), 'min': min(values) if values else None,
            'max': max(values) if values else None,
            'mean': sum(values) / len(values) if values else None}


def class_name(node):
    value = node['type']['class']
    assert isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.]+', value)
    return value


paths = sorted(JOURNALS.glob('*.jsonl'))
assert len(paths) == 2
workers = []
errors = []
groups = defaultdict(list)
all_fds = []
all_starts = all_ends = all_success = 0
for worker_number, path in enumerate(paths, 1):
    worker = f'worker_{worker_number}'
    initial_sha = sha(path)
    records = [json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
               for line in path.read_text().splitlines()]
    assert records and all(isinstance(row, dict) for row in records)
    # All source records are the adapter's allowlisted schema. No request IDs,
    # process IDs, exception messages or client identities are emitted below.
    common = {'version', 'pid', 'ppid', 'counter', 'process_session', 'started_utc', 'event', 'client', 'fd_count'}
    end_only = {'finished_utc', 'duration_seconds', 'outcome', 'exception_chain', 'usage', 'http_status', 'request_ids'}
    sessions = set()
    starts, ends = {}, {}
    fds = []
    for record in records:
        assert record['version'] == 1
        assert record['event'] in {'start', 'end'}
        assert set(record) == common | (end_only if record['event'] == 'end' else set())
        sessions.add((record['pid'], record['ppid'], record['process_session']))
        assert type(record['counter']) is int and record['counter'] > 0
        utc(record['started_utc'])
        fd = record['fd_count']
        assert fd is None or type(fd) is int and fd >= 0
        if fd is not None:
            fds.append(fd)
        target = starts if record['event'] == 'start' else ends
        assert record['counter'] not in target
        target[record['counter']] = record
        if record['event'] == 'end':
            assert record['outcome'] in {'success', 'error'}
            assert finite(record['duration_seconds'])
            assert utc(record['finished_utc']) >= utc(record['started_utc'])
            assert record['http_status'] is None or type(record['http_status']) is int and 100 <= record['http_status'] <= 599
            assert isinstance(record['exception_chain'], list)
            for node in record['exception_chain']:
                assert set(node) == {'type', 'errno', 'winerror', 'verify_code', 'flags'}
                class_name(node)
                assert set(node['flags']) == {'dns', 'tls', 'connect', 'read', 'write', 'pool', 'timeout', 'resource_fd'}
                assert all(type(flag) is bool for flag in node['flags'].values())
                assert all(node[k] is None or type(node[k]) is int for k in ('errno', 'winerror', 'verify_code'))
            assert set(record['usage']) == {'prompt_tokens', 'completion_tokens', 'total_tokens', 'cached_prompt_tokens'}
            assert all(n is None or type(n) is int and n >= 0 for n in record['usage'].values())
    assert len(sessions) == 1
    assert set(starts) == set(ends) == set(range(1, len(starts) + 1))
    assert records[0]['event'] == 'start' and records[0]['counter'] == 1
    assert records[-1]['event'] == 'end' and records[-1]['counter'] == len(starts)
    for counter in starts:
        assert starts[counter]['started_utc'] == ends[counter]['started_utc']
        assert starts[counter]['pid'] == ends[counter]['pid']
        assert starts[counter]['process_session'] == ends[counter]['process_session']
        if counter > 1:
            assert utc(starts[counter]['started_utc']) >= utc(ends[counter - 1]['finished_utc'])
    ordered = [ends[counter] for counter in sorted(ends)]
    success = [row for row in ordered if row['outcome'] == 'success']
    worker_errors = [row for row in ordered if row['outcome'] == 'error']
    assert success
    for error_number, error in enumerate(worker_errors, 1):
        following = next((row for row in success if row['counter'] > error['counter']), None)
        chain = [class_name(node) for node in error['exception_chain']]
        flags = sorted({name for node in error['exception_chain'] for name, enabled in node['flags'].items() if enabled})
        key = (tuple(chain), error['http_status'])
        groups[key].append(error['duration_seconds'])
        entry = {'worker': worker, 'worker_error_index': error_number,
                 'exception_chain': chain, 'http_status': error['http_status'],
                 'duration_seconds': error['duration_seconds'], 'observed_exception_flags': flags,
                 'usage_reported': {name: value is not None for name, value in error['usage'].items()},
                 'same_worker_later_success': following is not None,
                 'first_later_success_start_seconds_after_error_end': delta(error['finished_utc'], following['started_utc']) if following else None,
                 'first_later_success_end_seconds_after_error_end': delta(error['finished_utc'], following['finished_utc']) if following else None,
                 'first_later_success_duration_seconds': following['duration_seconds'] if following else None,
                 'intervening_attempts': following['counter'] - error['counter'] - 1 if following else None,
                 'fd_at_error_end': error['fd_count']}
        errors.append(entry)
    workers.append({'worker': worker, 'journal_sha256': initial_sha, 'starts': len(starts), 'ends': len(ends),
                    'successes': len(success), 'errors': len(worker_errors), 'fd_range_all_events': [min(fds), max(fds)] if fds else None,
                    'first_observed_start_utc': starts[1]['started_utc'],
                    'last_observed_end_utc': ends[len(ends)]['finished_utc'],
                    'last_success_start_utc': success[-1]['started_utc'],
                    'last_success_end_utc': success[-1]['finished_utc'],
                    'final_observed_outcome': ordered[-1]['outcome'],
                    'unmatched_starts_at_end': 0, 'unmatched_ends': 0,
                    'counters_contiguous_from_one': True, 'synchronous_nonoverlapping_requests': True,
                    'all_errors_have_later_same_worker_success': all(item['same_worker_later_success'] for item in errors if item['worker'] == worker)})
    all_fds.extend(fds)
    all_starts += len(starts)
    all_ends += len(ends)
    all_success += len(success)
    assert sha(path) == initial_sha, 'Journal changed during analysis'

assert len(errors) == 10 and all_starts == all_ends and all_success + len(errors) == all_ends
report = {'status': 'SAFE_PROVIDER_METADATA_SUMMARY_PASS',
          'scope': 'Completed V7 two anonymous worker journals only; no raw results, request payloads or question data',
          'starts': all_starts, 'ends': all_ends, 'successes': all_success, 'errors': len(errors),
          'by_exception_chain_and_status': [
              {'exception_chain': list(key[0]), 'http_status': key[1], 'duration_seconds': summary(values)}
              for key, values in sorted(groups.items(), key=lambda item: repr(item[0]))],
          'fd_range_all_events': [min(all_fds), max(all_fds)] if all_fds else None,
          'all_errors_have_later_same_worker_success': all(item['same_worker_later_success'] for item in errors),
          'workers': workers, 'error_recovery_metadata': errors,
          'public_adapter_sha256': sha(REPO / '.local-setup/azure_adapter.py'),
          'limits': [
              'Worker labels are anonymous and are not attributed to benchmark arm from these journals alone.',
              'The journal records SDK-call starts/ends, not OS process startup/exit. First counter1 and final paired end establish observed journal boundary closure; terminal process evidence is separate.',
              'The first later success may be another phase or question, not a retry of the failed request. These are recovery observations, not retry claims.',
              'No HTTP status means no recorded HTTP response status; it does not establish which network/service component caused the error.',
              'Timeout/status and subsequent success do not establish Azure region, incident causality or incident resolution.',
              'Absent usage on failed calls remains unknown, never zero billing.',
              'File-descriptor counts are snapshots at event boundaries, not proof of no transient resource spike or leak.',
              'No model calls, network/FHIR, config values, result/payload/question/log reads, process changes or duplicate audit restorations.'
          ]}
with (DEST / 'provider_error_summary.json').open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({key: report[key] for key in ('status', 'starts', 'ends', 'successes', 'errors', 'by_exception_chain_and_status', 'fd_range_all_events', 'all_errors_have_later_same_worker_success', 'workers')}, indent=2))
