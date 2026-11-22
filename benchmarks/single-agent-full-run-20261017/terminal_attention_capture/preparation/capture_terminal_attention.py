"""Metadata bookkeeping around unchanged monitor commands; NEVER a completion gate.

Run only after the fixed authorized processes have stopped. No inference, scoring,
provider/configuration imports, run writes, process signals, or retry scheduling.
All output goes to a new sibling directory under /private/tmp. Failed captures
retain their partial evidence. File contents and clinical values are not printed.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
RUN = ROOT / 'output/experiments/gpt-6-luna/v3_full_2931_20261015'
MONITOR = ROOT / 'benchmarks/single-agent-full-run-20261017/monitor_single_agent.py'
WRAPPER = RUN / 'monitor_observations/observer_continuous.py'
PYTHON = ROOT / '.venv/bin/python'
DEPS = ROOT / '.local-setup/phase2-dependencies'
MANIFEST_SHA = '4630c7c9a41badad1b1bba9e800aa9b5e5296de34dc96aef10c441769e21543c'
RUNNER_SHA = 'd13f52fa060acbdd1aec73b985b8857dba6664c060ad3228dc2283b3295768da'
MONITOR_SHA = 'daa88848d68f0dbd9fbe586ecb4cea3fdd65bb8d574770f0bb42c943a88e9047'
WRAPPER_SHA = 'c95ee748e000d144c9b5bdd2f78e1384d2933c4054bf160d693eb108fa2e7b3f'
OWNED_PIDS = (12567, 12569, 50214)
ARM = 'single_agent'
PARKED = {'05a8454a6a71435a557db2cc', '0eb69f6e47ab63c055e2df90'}
SOURCE_NAMES = set('''agent/__init__.py agent/analysis_agent.py agent/base_agent.py
agent/controller.py agent/direct_llm_agent.py agent/ehr_semantics.py agent/framework.py
agent/multi_agent.py agent/multi_turn_code_resource_agent.py agent/multi_turn_code_resource_v2_agent.py
agent/multi_turn_resource_agent.py agent/output_schema.py agent/query_contract_agent.py
agent/retrieval_agent.py agent/single_agent.py agent/single_turn_code_resource_agent.py
agent/single_turn_request_agent.py agent/single_turn_resource_agent.py agent/verification_agent.py
fhir_client.py requirements.txt run_agent.py scripts/audit_store.py scripts/full_benchmark.py
scripts/launch_v3_full.py scripts/v3_full_analysis.py scripts/v3_full_run.py tools/__init__.py
tools/cache.py tools/field_check.py tools/perturbation.py tools/python_executor.py tools/python_worker.py
tools/registry.py tools/request_tools.py tools/resource_tools.py tools/starting_packet.py utils/__init__.py
utils/answer_scorer.py utils/core_utils.py utils/evidence_scorer.py utils/fhir_utils.py
utils/metrics_utils.py utils/model_config.py utils/sql_to_fhir_utils.py'''.split())
PUBLIC_NAMES = set('''baseline_decision.txt preflight.json pricing.json protocol.txt provider_profile.json
split.json runtime/azure.py runtime/azure_adapter.py runtime/sitecustomize.py'''.split())
DEPENDENCY_NAMES = set('''PyYAML Requests dask google-auth ijson langchain-core langgraph
langgraph-checkpoint langgraph-prebuilt langgraph-sdk langsmith litellm numpy pandas protobuf
python_dateutil sqlglot sqlparse streamlit tiktoken tqdm'''.split())
INPUTS = {
    'input': ('final_dataset/questions_answers_sql_fhir.csv', 'e2045692fef7f5f4f77496935160f5fc727e162d213e94feed61401948e512a0'),
    'families': ('manifests/question_families.csv', '158177320a895c74969984f49922ba0a66f52ee3720460fa9a386115c386f8ab'),
    'split_manifest': ('manifests/split.json', 'f28db462ebcfe4e0d87675c86de9a08adef4038e26dbddcf88c50ddf92e94015'),
}
POLICY = {'version': 'azure-prompt-rejection-v1', 'category': 'provider_policy_rejection',
          'max_rejections_per_question': 2, 'retry_request': 'unchanged',
          'persistent_action': 'park_pending_then_attention', 'scope': [ARM]}
ROUNDS = set(range(1, 32))
AUDIT_NAME = re.compile(r'[a-f0-9]{64}\.json\.gz')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def safe(path, base=ROOT, directory=False):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and path.is_relative_to(base), 'Path outside fixed allowlisted root')
    for part in (base, *path.relative_to(base).parents, path.relative_to(base)):
        candidate = part if part.is_absolute() else base / part
        require(not candidate.is_symlink(), 'Symlink in captured path')
    require(path.resolve() == path, 'Captured path is not canonical')
    require(path.is_dir() if directory else path.is_file(), 'Required metadata artifact missing')
    return path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def native_absence(output, label):
    result = subprocess.run(['/bin/ps', '-p', ','.join(map(str, OWNED_PIDS)), '-o', 'pid=,stat=,comm='],
                            text=True, capture_output=True)
    record = {'at_utc': now(), 'owned_pids': list(OWNED_PIDS), 'return_code': result.returncode,
              'stdout': result.stdout, 'stderr': result.stderr}
    (output / f'native_{label}.json').write_text(json.dumps(record, indent=2) + '\n')
    require(result.returncode in (0, 1) and not result.stdout.strip() and not result.stderr.strip(),
            'Native scoped PID absence not established; do not capture a live run')
    return record


def snapshot():
    files, audit_references = {}, set()

    def remember(path, expected=None):
        path = safe(path)
        detail = {'sha256': sha(path), 'bytes': path.stat().st_size}
        require(expected is None or detail['sha256'] == expected, 'Frozen byte hash mismatch')
        files[str(path.relative_to(ROOT))] = detail
        return detail['sha256']

    remember(RUN / 'run_manifest.json', MANIFEST_SHA)
    manifest = read(RUN / 'run_manifest.json')
    require(manifest.get('project_root') == str(ROOT) and manifest.get('selected_arms') == [ARM]
            and manifest.get('questions_per_arm') == 2931, 'Unexpected frozen selected scope')
    require(manifest.get('classifier_version') == 'v3-infrastructure-v4'
            and manifest.get('provider_rejection_policy') == POLICY, 'Unexpected approved operational policy')
    sources, pins, versions = (manifest.get(key) or {} for key in
                               ('source_sha256', 'pinned_files_sha256', 'runtime_dependency_versions'))
    require(set(sources) == SOURCE_NAMES and len(sources) == 45, 'Guarded source allowlist changed')
    discovered_sources = {name for name in SOURCE_NAMES if not name.startswith(('agent/', 'tools/', 'utils/'))}
    for directory in ('agent', 'tools', 'utils'):
        safe(ROOT / directory, directory=True)
        discovered_sources.update(str(path.relative_to(ROOT)) for path in (ROOT / directory).rglob('*.py'))
    require(discovered_sources == SOURCE_NAMES, 'Current guarded source-name inventory changed')
    require(set(pins) == PUBLIC_NAMES and len(pins) == 9, 'Public-file allowlist changed')
    safe(RUN / 'runtime', directory=True)
    require({str(path.relative_to(RUN)) for path in (RUN / 'runtime').glob('*.py')}
            == {name for name in PUBLIC_NAMES if name.startswith('runtime/')}, 'Current runtime source-name inventory changed')
    require(set(versions) == DEPENDENCY_NAMES and len(versions) == 21, 'Declared dependency allowlist changed')
    require(sources['scripts/v3_full_run.py'] == RUNNER_SHA, 'Approved operational runner pin changed')
    for name, expected in sorted(sources.items()):
        remember(ROOT / name, expected)
    for name, expected in sorted(pins.items()):
        remember(RUN / name, expected)
    for key, (name, expected) in INPUTS.items():
        require(manifest[key]['path'] == str(ROOT / name) and manifest[key]['sha256'] == expected,
                'Input descriptor escaped exact allowlist')
        remember(ROOT / name, expected)
    observed_versions = {}
    for name in sorted(DEPENDENCY_NAMES):
        try:
            observed_versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            observed_versions[name] = None
    require(observed_versions == versions, 'Declared installed dependency versions differ')
    remember(MONITOR, MONITOR_SHA)
    remember(WRAPPER, WRAPPER_SHA)
    for name in ('supervisor_state.json', 'acceptance.json', 'provider_rejections.json', 'operations.jsonl'):
        remember(RUN / name)
    for arm in ('single_agent', 'multi_agent', 'direct_llm'):
        remember(RUN / f'full_{arm}.json')
        remember(RUN / f'full_{arm}.json.run.json')
    archive = RUN / 'amendments/0003-single-agent-only-resume/pre_apply_metadata'
    for name in ('acceptance.json', 'full_multi_agent.json', 'full_multi_agent.json.run.json',
                 'full_direct_llm.json', 'full_direct_llm.json.run.json'):
        remember(archive / name)

    state, ledger, rejected = (read(RUN / name) for name in
                              ('supervisor_state.json', 'acceptance.json', 'provider_rejections.json'))
    require(state.get('phase') == 'attention' and state.get('attention') is True
            and state.get('reason') == 'provider_policy_review_required' and state.get('finished_at')
            and state.get('active') == {} and state.get('selected_arms') == [ARM]
            and state.get('questions_per_arm') == 2931, 'Not the authorized terminal attention state')
    require(state.get('supervisor_pid') in OWNED_PIDS and state.get('keep_awake_pid') in OWNED_PIDS
            and state.get('supervisor_pid') != state.get('keep_awake_pid'), 'Recorded supervisor/keep-awake ownership changed')
    require(state.get('benchmark_complete') is False and state.get('selected_scope_complete') is False
            and state.get('pending_scoped_export') is False and state.get('analysis_status') == 'not_started'
            and ledger.get('complete') is False, 'Incomplete scope flags changed')
    require(rejected.get('policy') == POLICY and rejected.get('classifier_version') == 'v3-infrastructure-v4'
            and rejected.get('source') == 'reconstructed_from_sealed_immutable_attempts'
            and rejected.get('rejections') == {ARM: {qid: 2 for qid in PARKED}}
            and rejected.get('parked_question_ids') == {ARM: sorted(PARKED)}
            and rejected.get('pending_question_ids') == {ARM: sorted(PARKED)}
            and state.get('provider_rejections') == rejected, 'Unexpected provider parking metadata')
    issues = state.get('issues')
    require(isinstance(issues, list) and len(issues) == 2
            and {x.get('question_id') for x in issues} == PARKED
            and all(x.get('arm') == ARM and x.get('provider_rejections') == 2 and x.get('status') == 'pending'
                    and x.get('action') == 'external_policy_review_required' for x in issues), 'Unexpected pending issues')
    with (ROOT / INPUTS['input'][0]).open(newline='') as stream:
        cohort_ids = [row['question_id'] for row in csv.DictReader(stream)]
    canonical = read(RUN / 'full_single_agent.json')
    accepted = [row['question_id'] for row in canonical]
    require(len(cohort_ids) == len(set(cohort_ids)) == 2931 and len(accepted) == len(set(accepted)) == 2929,
            'Unexpected full or accepted metadata count')
    require(set(accepted) == set(ledger['arms'][ARM]) and set(accepted) <= set(cohort_ids)
            and set(cohort_ids) - set(accepted) == PARKED and state.get('accepted', {}).get(ARM) == 2929,
            'Remaining cohort is not exactly the two parked questions')
    operations = [json.loads(line) for line in (RUN / 'operations.jsonl').read_text().splitlines() if line.strip()]
    require(operations and operations[-1].get('event') == 'supervisor_stopped'
            and operations[-1].get('phase') == 'attention' and operations[-1].get('attention') is True,
            'Final supervisor-stopped event missing')
    stopped = {key: operations[-1].get(key) for key in ('at', 'event', 'phase', 'attention')}
    require(datetime.fromisoformat(stopped['at']) >= datetime.fromisoformat(state['finished_at']), 'Stop event predates terminal state')

    def refs(rows, base):
        for row in rows:
            reference = row.get('audit_artifacts')
            if reference is not None:
                require(isinstance(reference, dict) and reference.get('version') == 2
                        and reference.get('codec') == 'json+gzip' and reference.get('question_id') == row.get('question_id'),
                        'Unexpected audit reference metadata')
                name = reference.get('path', '')
                require(re.fullmatch(r'audit/[a-f0-9]{64}\.json\.gz', name) is not None
                        and name == 'audit/' + str(reference.get('sha256')) + '.json.gz', 'Unsafe audit reference')
                path = safe(base / name)
                audit_references.add(str(path.relative_to(ROOT)))
                remember(path)
            history = row.get('attempt_history') or []
            require(isinstance(history, list), 'Unexpected historical attempt metadata')
            if history:
                refs(history, base)

    selected = safe(RUN / 'attempts/single_agent', directory=True)
    suffixes = ('.batch.json', '.batch.seal.json', '.json', '.json.run.json', '.csv')
    expected_names = {f'round-{number:04d}{suffix}' for number in ROUNDS for suffix in suffixes}
    actual_names = set()
    for path in selected.iterdir():
        if path.name == 'audit':
            safe(path, directory=True)
            continue
        require(path.name in expected_names or re.fullmatch(r'round-(00(?:0[1-9]|[12][0-9]|3[01]))\.log', path.name),
                'Unexpected selected round artifact or unsealed extra round')
        remember(path)
        if path.name in expected_names:
            actual_names.add(path.name)
    require(actual_names == expected_names, 'Expected exact 31 sealed round stems are incomplete')
    for number in sorted(ROUNDS):
        stem = selected / f'round-{number:04d}'
        batch, seal = read(Path(str(stem) + '.batch.json')), read(Path(str(stem) + '.batch.seal.json'))
        raw, identity, shard = (Path(str(stem) + suffix) for suffix in ('.json', '.json.run.json', '.csv'))
        require(batch.get('round') == number and batch.get('arm') == ARM and batch.get('output') == str(raw)
                and batch.get('input') == str(shard), 'Batch paths or selected ownership changed')
        require(seal.get('batch_sha256') == sha(Path(str(stem) + '.batch.json'))
                and seal.get('result_sha256') == sha(raw) and seal.get('identity_sha256') == sha(identity)
                and batch.get('input_sha256') == sha(shard), 'Sealed round metadata byte binding differs')
        require(read(identity) == batch.get('run_identity'), 'Sealed batch identity differs')
        refs(read(raw), selected)
    refs(canonical, RUN)
    audit_directories = {}
    for directory in (selected / 'audit', RUN / 'audit'):
        safe(directory, directory=True)
        names = []
        for path in sorted(directory.iterdir()):
            require(AUDIT_NAME.fullmatch(path.name) is not None, 'Unexpected audit inventory entry')
            remember(path)
            names.append(path.name)
        audit_directories[str(directory.relative_to(ROOT))] = names
    return {'files': dict(sorted(files.items())), 'dependency_versions': observed_versions,
            'round_numbers': sorted(ROUNDS), 'audit_directory_entries': audit_directories,
            'referenced_audit_paths': sorted(audit_references),
            'safe_terminal_projection': {'phase': state['phase'], 'reason': state['reason'], 'finished_at': state['finished_at'],
                'active': {}, 'selected_accepted': 2929, 'questions_per_selected_arm': 2931,
                'remaining_question_ids': sorted(PARKED), 'provider_rejections_each': 2,
                'selected_scope_complete': False, 'original_three_arm_benchmark_complete': False,
                'selected_counts': ledger['counts'][ARM], 'last_supervisor_stopped': stopped},
            'scope_counts': {'guarded_sources': 45, 'public_pins': 9, 'input_descriptors': 3, 'declared_dependency_versions': 21}}


def command(output, name, arguments):
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join((str(DEPS), str(ROOT)))
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    for key in ('FHIR_LOCAL_AZURE_CONFIG', 'FHIR_V3_PROVIDER_PROFILE_PATH', 'FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR'):
        environment.pop(key, None)
    argv = [str(PYTHON), '-B', str(MONITOR), *arguments, '--run-dir', str(RUN)]
    result = subprocess.run(argv, cwd=ROOT, env=environment, capture_output=True)
    (output / f'{name}.stdout').write_bytes(result.stdout)
    (output / f'{name}.stderr').write_bytes(result.stderr)
    record = {'argv': argv, 'return_code': result.returncode, 'stdout_file': f'{name}.stdout', 'stderr_file': f'{name}.stderr'}
    (output / f'{name}.command.json').write_text(json.dumps(record, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New /private/tmp sibling directory; must not exist')
    args = parser.parse_args()
    output = args.output
    require(output.is_absolute() and output.parent == Path('/private/tmp')
            and output.name.startswith('single-terminal-attention-capture-') and not output.exists()
            and output != Path(__file__).resolve().parent and not output.is_symlink(), 'Output must be a new private sibling directory')
    output.mkdir(mode=0o700)
    receipt = {'schema_version': 1, 'status': 'CAPTURE_FAILED_PARTIAL', 'started_at': now(),
               'capture_script_sha256': sha(Path(__file__).resolve()), 'run': str(RUN),
               'benchmark_complete': False, 'selected_scope_complete': False,
               'verifier_success': False, 'correctness_scoring_performed': False,
               'inference_or_provider_calls_performed': False, 'secret_configuration_read': False}
    code = 2
    try:
        require(str(DEPS) in sys.path, 'Use the authorized phase2 dependency metadata path')
        native_absence(output, 'before')
        before = snapshot()
        (output / 'bindings_before.json').write_text(json.dumps(before, indent=2) + '\n')
        observe = command(output, 'observe', ['observe', '--skip-native'])
        verify = command(output, 'verify', ['verify'])
        after = snapshot()
        (output / 'bindings_after.json').write_text(json.dumps(after, indent=2) + '\n')
        native_absence(output, 'after')
        receipt['stable_before_after_bindings'] = before == after
        require(before == after, 'Source/file/dependency bindings changed during capture')
        observed, refused = json.loads(observe.stdout), json.loads(verify.stdout)
        require(observe.returncode == 0 and not observe.stderr and observed.get('status') == 'SELECTED_SCOPE_ATTENTION'
                and observed.get('selected_accepted_persisted') == 2929 and observed.get('selected_accepted_state') == 2929
                and observed.get('attention_reason') == 'provider_policy_review_required'
                and observed.get('manifest_sha256') == MANIFEST_SHA and observed.get('monitor_sha256') == MONITOR_SHA,
                'Unexpected observer outcome; preserved without successful capture claim')
        require(verify.returncode == 2 and not verify.stderr and refused ==
                {'status': 'REFUSED', 'error_type': 'ValueError', 'reason': 'Supervisor requests attention'},
                'Unexpected verifier outcome; preserved without prefix-integrity claim')
        receipt.update(status='TERMINAL_INCOMPLETE_METADATA_CAPTURED', stable_before_after_bindings=True,
                       expected_attention_refusal_observed=True, source_bound_prefix_checks_reached=True,
                       safe_terminal_projection=before['safe_terminal_projection'], scope_counts=before['scope_counts'])
        code = 0
    except Exception as error:
        receipt.update(error_type=type(error).__name__, reason=str(error) if isinstance(error, ValueError)
                       else 'Capture could not complete; inspect preserved partial metadata and command outputs')
    finally:
        receipt.update(finished_at=now(), limitations=[
            'This records terminal INCOMPLETE 2929/2931 state. It is not monitor success, selected-arm completion, or full-benchmark completion.',
            'The exact unchanged-monitor attention refusal establishes only checks before its attention guard, with stable bound files. Later assertions and native/active-successor checks are not reached.',
            'Native absence is separately observed only for the three explicitly authorized PIDs; no process signals are sent.',
            'The separate capture requires all 31 selected round stems sealed and inventories shards/audits; it does not invent a successful monitor receipt or independently score answers.',
            'Audit payload ownership and uncompressed hashes are checked by the unchanged monitor before its attention guard; this capture binds compressed bytes and safe reference paths.',
            'Declared dependency versions are distribution metadata, not package-byte hashes; the three input descriptors do not cover the separate offline FHIR resource files.',
            'Provider-profile bytes are pinned public metadata. Credentials/current provider settings are not read or validated.',
            'No retry, prompt/filter/provider change, partial scoring, or export is authorized by this bookkeeping receipt.'])
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        checksums = {path.name: {'sha256': sha(path), 'bytes': path.stat().st_size}
                     for path in sorted(output.iterdir()) if path.is_file()}
        (output / 'ARTIFACT_SHA256SUMS.json').write_text(json.dumps(checksums, indent=2) + '\n')
        print(json.dumps({'status': receipt['status'], 'output': str(output), 'return_code': code,
                          'selected_scope_complete': False, 'benchmark_complete': False}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
