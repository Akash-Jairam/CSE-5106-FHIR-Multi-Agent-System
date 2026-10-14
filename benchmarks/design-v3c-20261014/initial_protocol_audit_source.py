"""Read-only one-shot protocol audit. Never invokes agents or scorers."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import importlib.util
import json
import math
from pathlib import Path
import sys

import pandas as pd
from packaging.requirements import Requirement


ROOT = Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
BASE = ROOT / 'benchmarks/design-v3c-20261014'
OUTPUT = ROOT / 'output/experiments/gpt-6-luna/query_contract_v3c_300q'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
manifest = json.loads((BASE / 'manifest.json').read_text())
source_bad = [name for name, value in manifest['source_sha256'].items() if sha(ROOT/name) != value]
assert not source_bad, source_bad
assert sha(BASE/'source.zip') == manifest['source_archive_sha256']
assert sha(BASE/'design.txt') == manifest['design_sha256']
assert sha(ROOT/'manifests/split.json') == manifest['split_sha256']
assert sha(ROOT/'final_dataset/questions_answers_sql_fhir.csv') == manifest['dataset_sha256']
adapter = json.loads((BASE/'runtime_adapter_manifest.json').read_text())
assert not [name for name, value in adapter['source_sha256'].items() if sha(ROOT/name) != value]
assert sha(BASE/'runtime_adapter_sources.zip') == adapter['archive_sha256']
binding = json.loads((OUTPUT/'.local-provider.json').read_text())
assert binding['adapter_sha256'] == adapter['source_sha256']['.local-setup/azure_adapter.py']
assert binding['bootstrap_sha256'] == adapter['source_sha256']['.local-setup/sitecustomize.py']
preflight = json.loads((BASE/'health_preflight.json').read_text())

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
    phase_counts, release_policies = Counter(), Counter()
    for row in saved:
        store.verify_result_audit(row, OUTPUT, question_id=row['question_id'])
        restored = store.restore_result_audit(row, OUTPUT, question_id=row['question_id'])
        assert isinstance(restored.get('trace'), list)
        restored_count += 1
        boundaries += sum(bool(m.get('phase') == 'solver_boundary' and m.get('model_call') is False)
                          for m in restored['trace'] if isinstance(m, dict))
        tokens, budget = row.get('tokens') or {}, row.get('budget') or {}
        assert budget['max_total_tokens'] == 200000 and budget['max_tool_calls'] == 30
        assert tokens.get('total', 0) <= 200000 and budget.get('tool_calls_run', 0) <= 30
        if strategy != 'multi_agent_v3':
            continue
        revision = restored.get('revision') or {}
        phases = revision.get('phases') or []
        assert sum(p['tokens'] for p in phases) == tokens['total']
        assert sum(p['tool_calls'] for p in phases) == budget['tool_calls_run']
        for phase in phases:
            assert (phase['token_cap'], phase['tool_cap']) == caps[phase['phase']]
            assert phase['tokens'] <= phase['token_cap'] and phase['tool_calls'] <= phase['tool_cap']
            assert not phase['fatal']
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
                 'raw_snapshot_sha256':hashlib.sha256(raw).hexdigest()}

assert inputs['ordinary_single_agent'] == inputs['ordinary_multi_agent']
assert {k:v for k,v in identities['ordinary_single_agent'].items() if k != 'agent_strategy'} == {k:v for k,v in identities['ordinary_multi_agent'].items() if k != 'agent_strategy'}
report = {'checked_at_utc':datetime.now(timezone.utc).isoformat(),
          'scope':'Initial operational/provenance audit only; no scoring or ranking',
          'frozen_source_files_verified':len(manifest['source_sha256']), 'source_mismatches':source_bad,
          'source_archive_verified':True, 'design_dataset_split_hashes_verified':True,
          'adapter_source_files_verified':len(adapter['source_sha256']), 'adapter_archive_verified':True,
          'provider_binding':binding, 'identical_ordered_300_inputs':True,
          'code_hash':code_hash, 'sidecar_settings_match_except_actual_strategy':True,
          'arms':arms, 'anomalies':anomalies}
assert not anomalies
Path('/private/tmp/fhir-v3c-initial-protocol-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
