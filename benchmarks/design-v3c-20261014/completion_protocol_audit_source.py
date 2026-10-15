"""Post-completion recording/archival guard; never scores or invokes agents."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--owned_jobs_finished', action='store_true', required=True)
parser.add_argument('--report', type=Path, required=True)
parser.add_argument('--archive', type=Path)
args = parser.parse_args()
root = args.root.resolve()
record = root / 'benchmarks/design-v3c-20261014'
output = root / 'output/experiments/gpt-6-luna/query_contract_v3c_300q'

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def failed(row):
    return row.get('error') is not None and str(row['error']).strip() not in ('', 'nan')

def stamped(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns

manifest = json.loads((record / 'manifest.json').read_text())
for name, expected in manifest['source_sha256'].items():
    if name in ('config.yml', 'config.yaml') or name.startswith('.local-setup/config'):
        raise ValueError('Frozen source manifest contains a forbidden secret file')
    if sha(root / name) != expected:
        raise ValueError('Frozen source changed: ' + name)
for file_name, key in (('source.zip', 'source_archive_sha256'), ('design.txt', 'design_sha256')):
    if sha(record / file_name) != manifest[key]:
        raise ValueError('Frozen record differs')
if sha(root / 'final_dataset/questions_answers_sql_fhir.csv') != manifest['dataset_sha256']:
    raise ValueError('Frozen dataset differs')
if sha(root / 'manifests/split.json') != manifest['split_sha256']:
    raise ValueError('Frozen split differs')
adapter_manifest = json.loads((record / 'runtime_adapter_manifest.json').read_text())
for name, expected in adapter_manifest['source_sha256'].items():
    if name not in ('.local-setup/azure_adapter.py', '.local-setup/sitecustomize.py',
                    '.local-setup/azure.py', '.local-setup/phase3.py'):
        raise ValueError('Unexpected runtime-source archive allowlist')
    if sha(root / name) != expected:
        raise ValueError('Recorded runtime adapter changed')
if sha(record / 'runtime_adapter_sources.zip') != adapter_manifest['archive_sha256']:
    raise ValueError('Runtime adapter archive differs')
binding = json.loads((output / '.local-provider.json').read_text())
if set(binding) != {'provider_profile_sha256', 'adapter_sha256', 'bootstrap_sha256'}:
    raise ValueError('Public binding schema differs')
if (binding['adapter_sha256'] != adapter_manifest['source_sha256']['.local-setup/azure_adapter.py']
        or binding['bootstrap_sha256'] != adapter_manifest['source_sha256']['.local-setup/sitecustomize.py']):
    raise ValueError('Effective recorded adapter binding differs')
controls = {
    'archived_single': ('midterm_20261012_full_v1', '295f291b28cc3ca67f46799bd621b20a5d6a071be9d33e6590e9fac5b4377478'),
    'refreshed_single': ('verified_team_v2_300q', '6010afe99ca9c6b90e52ee3f359e64a3ab9ccf331a826bb8e1de306b2d0ee873')}
control_checks = {}
for name, (directory, expected) in controls.items():
    path = root / 'output/experiments/gpt-6-luna' / directory / 'ordinary_single_agent.json'
    if sha(path) != expected:
        raise ValueError('Historical control differs from committed checksum')
    if json.loads((path.parent / '.local-provider.json').read_text()) != binding:
        raise ValueError('Historical public provider binding differs')
    control_checks[name] = {'path': str(path), 'sha256': expected}
expected_ids = json.loads((root / 'manifests/split.json').read_text())['splits']['held_out']['question_ids']
if len(expected_ids) != 300 or len(set(expected_ids)) != 300:
    raise ValueError('The required fixed cohort differs')
streaming = load_module(root / 'scripts/full_benchmark.py', 'recording_stream_reader')
store = load_module(root / 'scripts/audit_store.py', 'recording_audit_store')
identities, arms = {}, {}
required = {'.local-provider.json', 'run_single.log', 'run_team.log'}
referenced = set()
for arm, strategy in (('ordinary_single_agent', 'single_agent'), ('ordinary_multi_agent', 'multi_agent_v3')):
    result = output / (arm + '.json')
    identity_file = output / (arm + '.json.run.json')
    required.update((result.name, identity_file.name))
    before = stamped(result)
    identity = json.loads(identity_file.read_text())
    if (identity['agent_strategy'] != strategy or identity['model'] != 'gpt-6-luna'
            or identity['max_total_tokens'] != 200000 or identity['max_tool_calls'] != 30
            or identity['split'] != 'held_out' or identity['split_manifest_hash'] != manifest['split_sha256']
            or identity['audit_storage'] != 'gzip' or identity['audit_artifact_schema'] != 2):
        raise ValueError('Completed arm identity differs from the scheduled protocol')
    counts = Counter()
    ids = []
    for row in streaming.iter_result_rows(result):
        ids.append(row['question_id'])
        if row.get('attempt_history'):
            raise ValueError('Fresh V3c unexpectedly contains retry history')
        if row.get('latency') is None and not failed(row):
            raise ValueError('Every scheduled first attempt must finish before any scoring or archive')
        counts['completed_first_attempts'] += 1
        counts['terminal_errors'] += int(failed(row))
        counts['budget_stops'] += int(bool((row.get('budget') or {}).get('stopped')))
        tokens, budget = row.get('tokens') or {}, row.get('budget') or {}
        counts['unknown_provider_usage_attempts'] += int(any(c.get('usage_available') is False for c in tokens.get('per_call') or []))
        total = tokens.get('total')
        counts['reported_aggregate_token_overshoots'] += int(isinstance(total, (int, float)) and total > 200000)
        counts['reported_tool_overshoots'] += int((budget.get('tool_calls_run') or 0) > 30)
        revision = row.get('revision') or {}
        phases = revision.get('phases') or []
        fatal = any(bool(phase.get('fatal')) for phase in phases)
        counts['fatal_phase_attempts'] += int(fatal)
        counts['fatal_phase_records'] += sum(bool(phase.get('fatal')) for phase in phases)
        counts['reported_phase_token_overshoots'] += sum((phase.get('tokens') or 0) > phase['token_cap'] for phase in phases)
        counts['reported_phase_tool_overshoots'] += sum((phase.get('tool_calls') or 0) > phase['tool_cap'] for phase in phases)
        clean = not failed(row) and revision.get('final_check_issues') == []
        counts['recorded_clean_release_attempts'] += int(clean)
        counts['clean_release_with_fatal_phase'] += int(clean and fatal)
        # These observed errors/fatal/overshoot outcomes remain completed attempts.
        # They are reported rather than falsely rejected as an incomplete cohort.
        store.verify_result_audit(row, output, question_id=row['question_id'])
        referenced.add(row['audit_artifacts']['path'])
        counts['verified_owned_gzip_audits'] += 1
    if ids != expected_ids or counts['completed_first_attempts'] != 300:
        raise ValueError('Completed cohort IDs/order differ')
    digest = sha(result)
    if before != stamped(result):
        raise ValueError('Result changed during completion verification')
    identities[arm] = identity
    arms[arm] = {**dict(counts), 'result_sha256': digest, 'identity_sha256': sha(identity_file)}
if ({k:v for k,v in identities['ordinary_single_agent'].items() if k != 'agent_strategy'}
        != {k:v for k,v in identities['ordinary_multi_agent'].items() if k != 'agent_strategy'}):
    raise ValueError('Concurrent arm identities differ beyond strategy')
audit_files = {str(path.relative_to(output)) for path in (output / 'audit').glob('*.json.gz')
               if re.fullmatch(r'[a-f0-9]{64}\.json\.gz', path.name)}
if not referenced <= audit_files:
    raise ValueError('Referenced audit missing from archive allowlist')
allowed = sorted(required | audit_files)
files = {}
for name in allowed:
    path = output / name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(output.resolve()):
        raise ValueError('Archive entry is missing, unsafe or nonregular')
    files[name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
report = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
          'scope': 'Completed first-attempt recording, frozen provenance and stored-audit integrity; no scoring or clinical validation',
          'expected_first_attempts': 600, 'arms': arms, 'historical_controls': control_checks,
          'provider_binding': binding, 'raw_files': files,
          'unreferenced_retained_audit_files': sorted(audit_files - referenced),
          'helper_sha256': sha(Path(__file__)), 'model_calls': 0}
if args.archive:
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
        for name in allowed:
            bundle.write(output / name, name)
    with zipfile.ZipFile(args.archive) as bundle:
        if bundle.testzip() is not None or sorted(bundle.namelist()) != allowed:
            raise ValueError('Raw archive member/CRC verification failed')
        for name, expected in files.items():
            digest = hashlib.sha256()
            with bundle.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected['sha256'] or sha(output / name) != expected['sha256']:
                raise ValueError('Archived bytes differ from completed source files')
    report.update(raw_archive_path=str(args.archive.resolve()), raw_archive_sha256=sha(args.archive),
                  raw_archive_bytes=args.archive.stat().st_size)
args.report.parent.mkdir(parents=True, exist_ok=True)
args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({'completed_first_attempts': 600, 'arms': arms,
                  'archive_written': bool(args.archive), 'report': str(args.report)}))
