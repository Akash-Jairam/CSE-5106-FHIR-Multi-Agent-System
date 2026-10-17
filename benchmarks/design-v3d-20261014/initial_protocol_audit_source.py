"""Read-only one-shot protocol audit. Never invokes agents or scorers."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import gzip
import re
import zipfile
from importlib import metadata
import importlib.util
import json
import math
from pathlib import Path
import sys

import pandas as pd
from packaging.requirements import Requirement


ROOT = Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
BASE = ROOT / 'benchmarks/design-v3d-20261014'
OUTPUT = ROOT / 'output/experiments/gpt-6-luna/query_contract_v3d_300q'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
manifest = json.loads((BASE / 'manifest.json').read_text())
snapshot_started_utc = datetime.now(timezone.utc).isoformat()
source_bad = [name for name, value in manifest['source_sha256'].items() if sha(ROOT/name) != value]
assert not source_bad, source_bad
assert sha(BASE/'source.zip') == manifest['source_archive_sha256']
with zipfile.ZipFile(BASE/'source.zip') as frozen_zip:
    assert set(frozen_zip.namelist()) == set(manifest['source_sha256']) and len(frozen_zip.namelist()) == len(manifest['source_sha256'])
    assert all(hashlib.sha256(frozen_zip.read(name)).hexdigest() == value for name,value in manifest['source_sha256'].items())
assert sha(BASE/'design.txt') == manifest['design_sha256']
assert sha(ROOT/'manifests/split.json') == manifest['split_sha256']
assert sha(ROOT/'final_dataset/questions_answers_sql_fhir.csv') == manifest['dataset_sha256']
adapter = json.loads((BASE/'runtime_adapter_manifest.json').read_text())
assert not [name for name, value in adapter['source_sha256'].items() if sha(ROOT/name) != value]
assert sha(BASE/'runtime_adapter_sources.zip') == adapter['archive_sha256']
assert set(adapter['source_sha256']) == {'.local-setup/azure_adapter.py', '.local-setup/azure.py', '.local-setup/phase3.py', '.local-setup/sitecustomize.py'}
with zipfile.ZipFile(BASE/'runtime_adapter_sources.zip') as runtime_zip:
    assert set(runtime_zip.namelist()) == set(adapter['source_sha256']) and len(runtime_zip.namelist()) == 4
    assert all(hashlib.sha256(runtime_zip.read(name)).hexdigest() == value for name,value in adapter['source_sha256'].items())
binding = json.loads((OUTPUT/'.local-provider.json').read_text())
assert binding['adapter_sha256'] == adapter['source_sha256']['.local-setup/azure_adapter.py']
assert binding['bootstrap_sha256'] == adapter['source_sha256']['.local-setup/sitecustomize.py']
preflight = json.loads((BASE/'health_preflight.json').read_text())
assert binding == adapter['expected_provider_binding']
assert binding['provider_profile_sha256'] == preflight['request_manifest']['provider_identity_profile_sha256']

# Execute only the two public source/dependency hash functions, with no runner,
# agent, setup, configuration or provider imports.
tree = ast.parse((ROOT/'run_agent.py').read_text())
pure = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name in {'code_fingerprint', 'runtime_dependency_versions'}], type_ignores=[])
namespace = dict(Path=Path, hashlib=hashlib, json=json, metadata=metadata,
                 Requirement=Requirement, __file__=str(ROOT/'run_agent.py'))
exec(compile(pure, '<public-source-hash-functions>', 'exec'), namespace)
# Reproduce the launcher's public dependency search path without importing its
# credential-loading bootstrap or reading configuration.
sys.path.insert(0, str(ROOT/'.local-setup/phase2-dependencies'))
code_hash = namespace['code_fingerprint']()
spec = importlib.util.spec_from_file_location('isolated_audit_store', ROOT/'scripts/audit_store.py')
store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(store)
expected = json.loads((ROOT/'manifests/split.json').read_text())['splits']['held_out']['question_ids']
caps = {'contract': (10000, 0), 'solver': (175000, 27), 'audit': (15000, 0), 'targeted': (0, 3)}
arms = {}
identities = {}
inputs = {}
anomalies = []

for arm, strategy in [('ordinary_single_agent', 'single_agent'), ('ordinary_multi_agent', 'multi_agent_v3')]:
    path = OUTPUT/(arm+'.json')
    raw = path.read_bytes()  # one stable atomic-save snapshot per arm
    rows = json.loads(raw)
    assert len(rows) == 300 and [r['question_id'] for r in rows] == expected
    identity = json.loads(Path(str(path)+'.run.json').read_text())
    assert identity['agent_strategy'] == strategy
    assert identity['model'] == 'gpt-6-luna'
    assert identity['max_total_tokens'] == 200000 and identity['max_tool_calls'] == 30
    assert identity['code_hash'] == code_hash
    assert identity['split_manifest_hash'] == manifest['split_sha256']
    assert identity['split'] == 'held_out' and identity['audit_storage'] == 'gzip'
    assert identity['audit_artifact_schema'] == 2
    # Identity-only input hashing; labels are never inspected, compared with
    # answers, sent anywhere, or scored.
    columns = ['question_id', 'question', 'true_answer', 'assumption', 'patient_fhir_id']
    input_json = pd.DataFrame(rows)[columns].to_json(orient='records', date_format='iso')
    assert hashlib.sha256(input_json.encode()).hexdigest() == identity['input_hash']
    inputs[arm] = input_json
    identities[arm] = identity
    saved = [r for r in rows if (type(r.get('latency')) in (int, float) and math.isfinite(r['latency'])) or r.get('error')]
    assert all(not r.get('attempt_history') for r in rows), 'unexpected retry history'
    restored_count = boundaries = 0
    phase_fatals = phase_overshoots = global_overshoots = missing_phase_records = 0
    independent_restored_bytes = 0
    phase_counts, release_policies = Counter(), Counter()
    for row in saved:
        reference = row.get('audit_artifacts')
        assert isinstance(reference, dict) and reference['version'] == 2 and reference['codec'] == 'json+gzip'
        assert reference['question_id'] == row['question_id']
        assert re.fullmatch(r'audit/[a-f0-9]{64}\.json\.gz', reference['path'])
        assert reference['path'] == 'audit/' + reference['sha256'] + '.json.gz'
        artifact_path = OUTPUT/reference['path']
        assert artifact_path.resolve().is_relative_to(OUTPUT.resolve())
        with gzip.open(artifact_path, 'rb') as compressed:
            payload_raw = compressed.read(512*1024*1024+1)
        assert len(payload_raw) <= 512*1024*1024
        assert hashlib.sha256(payload_raw).hexdigest() == reference['sha256']
        payload = json.loads(payload_raw)
        assert payload['_audit_meta']['question_id'] == row['question_id']
        assert set(payload) == {'_audit_meta', *reference['fields']}
        independent_restored_bytes += len(payload_raw)
        store.verify_result_audit(row, OUTPUT, question_id=row['question_id'])
        restored = store.restore_result_audit(row, OUTPUT, question_id=row['question_id'])
        assert isinstance(restored.get('trace'), list)
        for field in reference['fields']:
            actual = restored['revision']['messages'] if field == 'revision.messages' else restored[field]
            assert actual == payload[field]
        for field in ('agent_answer', 'evidence', 'limitation', 'tokens', 'usage', 'tool_calls', 'budget', 'error', 'latency'):
            assert restored.get(field) == row.get(field)
        restored_count += 1
        boundaries += sum(bool(m.get('phase') == 'solver_boundary' and m.get('model_call') is False)
                          for m in restored['trace'] if isinstance(m, dict))
        tokens, budget = row.get('tokens') or {}, row.get('budget') or {}
        assert budget['max_total_tokens'] == 200000 and budget['max_tool_calls'] == 30
        assert budget.get('tool_calls_run', 0) <= 30
        global_overshoots += bool(tokens.get('total', 0) > 200000)
        if strategy != 'multi_agent_v3':
            continue
        revision = restored.get('revision') or {}
        phases = revision.get('phases') or []
        if phases:
            assert sum(p['tokens'] for p in phases) == tokens['total']
            assert sum(p['tool_calls'] for p in phases) == budget['tool_calls_run']
        else:
            missing_phase_records += 1
            assert row.get('error'), 'nonterminal team row missing phase records'
        for phase in phases:
            assert (phase['token_cap'], phase['tool_cap']) == caps[phase['phase']]
            phase_overshoots += bool(phase['tokens'] > phase['token_cap'] or phase['tool_calls'] > phase['tool_cap'])
            phase_fatals += bool(phase['fatal'])
            phase_counts[phase['phase']] += 1
        policy = revision.get('release_policy')
        release_policies[policy] += 1
        if policy in {'source_valid_primary_preserved', 'source_checked_targeted_local_constraint'}:
            assert revision['final_check_issues'] == []
            final = revision['final']
            for display, artifact in [('agent_answer', 'answer'), ('evidence', 'evidence'), ('limitation', 'limitation')]:
                assert json.dumps(row.get(display), sort_keys=True, allow_nan=False) == json.dumps(final.get(artifact), sort_keys=True, allow_nan=False)
        if any(m.get('phase') == 'solver_boundary' and m.get('model_call') is not False
               for m in restored['trace'] if isinstance(m, dict)):
            anomalies.append({'arm': arm, 'question_id': row['question_id'], 'type': 'boundary_model_accounting'})
    arms[arm] = {'scheduled_rows':len(rows), 'saved_first_attempts_checked':len(saved),
                 'terminal_errors':sum(bool(r.get('error')) for r in saved),
                 'gzip_owner_hash_restorations':restored_count, 'host_boundaries':boundaries,
                 'phases_observed':dict(phase_counts), 'release_policies':dict(release_policies),
                 'raw_snapshot_sha256':hashlib.sha256(raw).hexdigest(),
                 'snapshot_bytes':len(raw), 'snapshot_observed_utc':datetime.now(timezone.utc).isoformat(),
                 'independent_uncompressed_artifact_bytes_verified':independent_restored_bytes,
                 'phase_fatal_records':phase_fatals,'phase_overshoot_records':phase_overshoots,
                 'reported_global_token_overshoots':global_overshoots,'team_errors_without_phase_records':missing_phase_records}

assert inputs['ordinary_single_agent'] == inputs['ordinary_multi_agent']
assert {k:v for k,v in identities['ordinary_single_agent'].items() if k != 'agent_strategy'} == {k:v for k,v in identities['ordinary_multi_agent'].items() if k != 'agent_strategy'}
report = {'checked_at_utc':datetime.now(timezone.utc).isoformat(),
          'scope':'One-shot live saved-file and per-journal byte snapshots; operational/provenance only; no scoring or ranking. Counts may differ between snapshots; not a run-completion claim.',
          'snapshot_started_utc':snapshot_started_utc,
          'frozen_source_files_verified':len(manifest['source_sha256']), 'source_mismatches':source_bad,
          'source_archive_verified':True, 'design_dataset_split_hashes_verified':True,
          'adapter_source_files_verified':len(adapter['source_sha256']), 'adapter_archive_verified':True,
          'provider_binding':binding, 'identical_ordered_300_inputs':True,
          'code_hash':code_hash, 'sidecar_settings_match_except_actual_strategy':True,
          'arms':arms, 'anomalies':anomalies}
assert not anomalies

# Read each append-only journal once. A current trailing partial line or last
# unmatched start is retained as incomplete evidence, never counted success.
base_keys = {'version','pid','ppid','counter','process_session','started_utc'}
start_keys = base_keys | {'event','client','fd_count'}
end_keys = start_keys | {'finished_utc','duration_seconds','outcome','exception_chain','usage','http_status','request_ids'}
class_pattern = re.compile(r'[A-Za-z0-9_.]{1,200}')
request_pattern = re.compile(r'[A-Za-z0-9_.:-]{1,160}')
request_headers = {'x-request-id','x-ms-request-id','apim-request-id'}
def integer_or_none(value, nonnegative=False):
    return value is None or type(value) is int and (not nonnegative or value >= 0)
def class_schema(value):
    if value is None:
        return
    assert isinstance(value,dict) and set(value)=={'class','mro'}
    assert isinstance(value['class'],str) and class_pattern.fullmatch(value['class'])
    assert isinstance(value['mro'],list) and 1 <= len(value['mro']) <= 16
    assert all(isinstance(x,str) and class_pattern.fullmatch(x) for x in value['mro'])
def client_schema(value):
    assert isinstance(value,dict) and set(value)=={'sdk','http','transport','pool','sdk_closed','http_closed','client_id','versions'}
    for key in ('sdk','http','transport','pool'):
        class_schema(value[key])
    assert all(value[k] is None or type(value[k]) is bool for k in ('sdk_closed','http_closed'))
    assert integer_or_none(value['client_id'],True)
    assert isinstance(value['versions'],dict) and set(value['versions'])=={'openai','httpx2'}
    assert all(x is None or isinstance(x,str) and re.fullmatch(r'[0-9A-Za-z.+_-]{1,64}',x) for x in value['versions'].values())
def event_schema(event):
    assert isinstance(event,dict) and event.get('event') in {'start','end'}
    assert set(event)==(start_keys if event['event']=='start' else end_keys)
    assert event['version']==1 and all(type(event[k]) is int and event[k]>0 for k in ('pid','ppid','counter'))
    assert isinstance(event['process_session'],str) and re.fullmatch(r'[a-f0-9]{32}',event['process_session'])
    assert datetime.fromisoformat(event['started_utc']).tzinfo is not None
    assert integer_or_none(event['fd_count'],True)
    client_schema(event['client'])
    if event['event']=='start':
        return
    assert datetime.fromisoformat(event['finished_utc'])>=datetime.fromisoformat(event['started_utc'])
    assert type(event['duration_seconds']) in (int,float) and math.isfinite(event['duration_seconds']) and event['duration_seconds']>=0
    assert event['outcome'] in {'success','error'}
    assert integer_or_none(event['http_status']) and (event['http_status'] is None or 100<=event['http_status']<=599)
    assert isinstance(event['request_ids'],dict) and set(event['request_ids'])<=request_headers
    assert all(isinstance(x,str) and request_pattern.fullmatch(x) for x in event['request_ids'].values())
    assert isinstance(event['usage'],dict) and set(event['usage'])=={'prompt_tokens','completion_tokens','total_tokens','cached_prompt_tokens'}
    assert all(integer_or_none(x,True) for x in event['usage'].values())
    assert isinstance(event['exception_chain'],list) and len(event['exception_chain'])<=8
    for cause in event['exception_chain']:
        assert isinstance(cause,dict) and set(cause)=={'type','errno','winerror','verify_code','flags'}
        class_schema(cause['type'])
        assert all(integer_or_none(cause[x]) for x in ('errno','winerror','verify_code'))
        assert isinstance(cause['flags'],dict) and set(cause['flags'])=={'dns','tls','connect','read','write','pool','timeout','resource_fd'}
        assert all(type(x) is bool for x in cause['flags'].values())

journal_summaries=[]
journal_totals=Counter()
all_default_ends=Counter()
for journal in sorted((OUTPUT/'provider-events').glob('*.jsonl')):
    assert re.fullmatch(r'azure-diagnostics-[0-9]+-[a-f0-9]{32}\.jsonl',journal.name)
    raw=journal.read_bytes()
    lines=raw.split(b'\n')
    tail=lines.pop()
    events=[json.loads(line) for line in lines]
    starts, ends={},{}
    fds=[]
    class_counts=Counter()
    clients=set()
    metadata_unavailable=0
    errors=Counter()
    status_counts=Counter()
    unknown_usage=0
    for event in events:
        event_schema(event)
        key=(event['pid'],event['process_session'],event['counter'])
        target=starts if event['event']=='start' else ends
        assert key not in target,'duplicate journal event key'
        target[key]=event
        if event['fd_count'] is not None:
            fds.append(event['fd_count'])
        client=event['client']
        if client['client_id'] is not None:
            clients.add(client['client_id'])
        transport=client['transport']
        class_counts[transport['class'] if transport else 'unavailable_before_client_creation']+=1
        if event['event']=='end':
            status_counts[str(event['http_status'])]+=1
            unknown_usage+=event['usage']['total_tokens'] is None
            if event['outcome']=='error':
                for cause in event['exception_chain']:
                    errors[cause['type']['class']]+=1
            if transport is None:
                metadata_unavailable+=1
            else:
                assert transport['class']=='httpx2.HTTPTransport'
                assert client['http']['class']=='openai._base_client.SyncHttpxClientWrapper'
                assert 'httpx2.Client' in client['http']['mro']
                assert client['versions']=={'openai':'3.26.1','httpx2':'2.13.1'}
                all_default_ends[transport['class']]+=1
    matched=set(starts)&set(ends)
    for key in matched:
        assert all(starts[key][k]==ends[key][k] for k in ('pid','ppid','process_session','counter','started_utc'))
    unmatched_start=sorted(set(starts)-set(ends))
    unmatched_end=sorted(set(ends)-set(starts))
    if unmatched_end:
        anomalies.append({'type':'journal_end_without_start','file':journal.name,'count':len(unmatched_end)})
    # Best-effort logging can lose an end; older unmatched starts must not be
    # silently described as current in-flight calls.
    newest_counter=max((k[2] for k in starts),default=0)
    unmatched=[{'pid':k[0],'counter':k[2],'started_utc':starts[k]['started_utc'],
                'most_recent_start_in_this_snapshot':k[2]==newest_counter}
               for k in unmatched_start]
    success=sum(x['outcome']=='success' for x in ends.values())
    failures=sum(x['outcome']=='error' for x in ends.values())
    journal_totals.update({'starts':len(starts),'ends':len(ends),'matched_pairs':len(matched),
                          'successes':success,'errors':failures,'unmatched_starts':len(unmatched),
                          'unmatched_ends':len(unmatched_end),'trailing_partial_lines':bool(tail),
                          'unknown_usage_ends':unknown_usage,'ends_without_client_metadata':metadata_unavailable})
    journal_summaries.append({'file':journal.name,'snapshot_sha256':hashlib.sha256(raw).hexdigest(),
       'snapshot_bytes':len(raw),'snapshot_observed_utc':datetime.now(timezone.utc).isoformat(),
       'starts':len(starts),'ends':len(ends),'matched_pairs':len(matched),'successes':success,'errors':failures,
       'unmatched_starts':unmatched,'unmatched_ends':len(unmatched_end),'trailing_partial_bytes':len(tail),
       'numeric_fd_min':min(fds) if fds else None,'numeric_fd_max':max(fds) if fds else None,
       'distinct_nonnull_sdk_client_ids':len(clients),'observed_transport_classes':dict(class_counts),
       'http_status_counts':dict(status_counts),'error_cause_class_counts':dict(errors),
       'unknown_usage_ends':unknown_usage})
assert journal_summaries,'missing provider journal snapshots'
report['provider_journal']={'schema_valid':True,'privacy_allowlist_valid':True,
   'totals':dict(journal_totals),'observed_default_transport_completed_events':dict(all_default_ends),
   'files':journal_summaries,
   'limits':['Per-file byte snapshots are not simultaneous with each other or saved result snapshots.',
             'Latest unmatched starts are consistent with current in-flight calls; unmatched/partial evidence is never credited as success or known usage.',
             'Journaling is best effort; missing events cannot prove that a request was absent.',
             'Native default HTTPX2 class/version observations do not establish the proxy route or sustained future availability.',
             'FD observations are descriptor counts at event boundaries, not a proof of leak absence.',
             'Successful parsed SDK responses have unknown HTTP status; null is retained rather than asserting HTTP 200.']}
report['status']='PASS' if not anomalies else 'ATTENTION'
report['helper_sha256']=sha(__file__)
report['no_config_model_network_scoring_calls']=True
report['fatal_or_overshoot_outcomes_are_recorded_not_rejected']=True
report_path=BASE/'initial_protocol_audit.json'
report_path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'arms':arms,'provider_totals':dict(journal_totals),
                  'report':str(report_path),'anomalies':anomalies},indent=2))
