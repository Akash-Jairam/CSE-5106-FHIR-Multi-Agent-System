"""Generated V7 lifecycle records only: no inference, gold scoring, or credentials.

Run with the repository virtualenv Python -I. The public audit-store implementation
is copied as source without importing its package. All records and all other source,
runtime, split, dataset, identity and journal bytes are generated in TemporaryDirectory.
The fixed input/code/provider pins are substituted only within each fixture scope.
"""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd

PACKET = Path(__file__).parent
AUDIT_SOURCE = Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System/scripts/audit_store.py')
spec = importlib.util.spec_from_file_location('independent_v5_lifecycle_guard', PACKET / 'v7_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
GUARD_LOADED_SHA = guard.sha(PACKET / 'v7_guard.py')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_zip(path, root, names):
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            archive.write(root / name, name)


class LifecycleFixture:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.record = self.root / 'benchmarks/synthetic-v7'
        self.output = self.root / 'output/synthetic-v7'
        self.record.mkdir(parents=True)
        for name in guard.REQUIRED_SOURCE:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# Synthetic frozen public source fixture.\n')
        (self.root / 'agent/residual_trajectory_agent.py').write_text('PROTOCOL='+repr(guard.PROTOCOL)+'\nENTRY_BYTES=32768\nBOUNDARY_TOKENS=512\nSELECTOR_COMPLETION=2000\nSELECTOR_PROMPT="Synthetic selector"+SEMANTIC_GUIDANCE\n')
        (self.root / 'agent/residual_trajectory_phase.py').write_text('SDK_CAP=30\n')
        (self.root / 'agent/ehr_semantics.py').write_text('SEMANTIC_GUIDANCE="Synthetic guide"\n')
        (self.root / 'scripts/audit_store.py').write_bytes(AUDIT_SOURCE.read_bytes())
        self.inputs = [{
            'question_id': f'synthetic-question-{i:03d}',
            'question': f'Synthetic question {i}; no clinical records.',
            'true_answer': 'unused synthetic reference',
            'assumption': None,
            'patient_fhir_id': f'00000000-0000-0000-0000-{i:012d}',
        } for i in range(300)]
        encoded = pd.DataFrame(self.inputs).to_json(orient='records', date_format='iso')
        self.input_hash = hashlib.sha256(encoded.encode()).hexdigest()
        split = self.root / 'manifests/split.json'
        dump(split, {'splits': {'held_out': {'question_ids': [r['question_id'] for r in self.inputs]}}})
        dataset = self.root / 'final_dataset/questions_answers_sql_fhir.csv'
        dataset.parent.mkdir(parents=True)
        dataset.write_text('synthetic_fixture_only\nnot_a_benchmark\n')
        self.provider_sources = {}
        for i, name in enumerate(guard.PROVIDER_FILES):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f'# Synthetic public provider fixture {i}; no credentials.\n')
            self.provider_sources[name] = guard.sha(path)
        self.binding = {
            'provider_profile_sha256': hashlib.sha256(b'synthetic public profile').hexdigest(),
            'adapter_sha256': self.provider_sources['.local-setup/azure_adapter.py'],
            'bootstrap_sha256': self.provider_sources['.local-setup/sitecustomize.py'],
        }
        self.code_hash = hashlib.sha256(b'synthetic frozen code identity').hexdigest()
        names = guard.actual_source_inventory(self.root)
        mapping = {name: guard.sha(self.root / name) for name in sorted(names)}
        source_zip(self.record / 'source.zip', self.root, mapping)
        source_zip(self.record / 'runtime_adapter_sources.zip', self.root, self.provider_sources)
        (self.record / 'design.txt').write_text('Synthetic lifecycle fixture; no accuracy claim.\n')
        self.manifest = {
            'source_sha256': mapping,
            'source_archive_sha256': guard.sha(self.record / 'source.zip'),
            'design_sha256': guard.sha(self.record / 'design.txt'),
            'dataset_sha256': guard.sha(dataset), 'split_sha256': guard.sha(split),
            'model': 'gpt-6-luna', 'max_total_tokens': 200000, 'max_tool_calls': 30,
            'expected_questions_per_arm': 300, 'arms': list(guard.ARM_STRATEGIES),
            'output_directory': str(self.output),
        }
        dump(self.record / 'manifest.json', self.manifest)
        dump(self.record / 'runtime_adapter_manifest.json', {
            'source_sha256': self.provider_sources,
            'archive_sha256': guard.sha(self.record / 'runtime_adapter_sources.zip'),
        })
        dump(self.record / 'expected_public_binding.json', self.binding)
        dump(self.record / 'prospective_protocol.json', {
            'acceptance':guard.ACCEPTANCE,'recorded_A_only_gate':guard.A_ONLY_GATE,
            'model':'gpt-6-luna','fixed_questions_per_arm':300,'workers_per_arm':1,'first_attempt_only':True,
            'primary_iterations':30,'primary_tools_inclusive_automatic_checks':True,'reserved_primary_source_checks':0,
            'full_primary_allowance':[200000,30,30],'alternative_allocation':'actual_residual_less_initialization_and_selector_reservation',
            'max_team_model_calls':30,'max_control_model_calls':30,'global_caps':[200000,30],
            'estimated_input_guard':32000,'minimum_team_correct':240,'expected_public_binding':self.binding,
            'bounds':{'candidate_entry_bytes':32768,'selector_boundary_tokens':512,'selector_completion':2000,'selector_response_chars':16000},
            'alternative_initialization_tool_charge':1,'minimum_residual_tools':3,'minimum_residual_sdk_attempts':3})
        commands={}
        for label,arm in [('single','ordinary_single_agent'),('team','ordinary_multi_agent')]:
            commands[label]={'arm':arm,'argv':['caffeinate','-i','.venv/bin/python','.local-setup/phase3.py','run','--project-root','.',
                '--model','gpt-6-luna','--max_total_tokens','200000','--max_tool_calls','30','--num_processes','1','--save_interval','1',
                '--output_dir',str(self.output),'--split','held_out','--team_strategy','multi_agent_v7','--audit_storage','gzip','--arms',arm],
                'environment':{'FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR':str(self.output/'provider-events')},'log':str(self.output/('run_'+label+'.log'))}
        dump(self.record/'launch_commands.json',{'concurrent':True,'workers_per_arm':1,'commands':commands})
        self.pins = self.record / 'pins.json'
        self.pins_sha = None
        self.store = guard.B.load_public_store(self.root)
        self.rows = {}

    def patched_constants(self):
        stack = ExitStack()
        stack.enter_context(patch.multiple(guard,
            FIXED_INPUT_HASH=self.input_hash,
            FIXED_DATASET_SHA=self.manifest['dataset_sha256'],
            FIXED_SPLIT_SHA=self.manifest['split_sha256']))
        stack.enter_context(patch.object(guard.B, 'code_hash', return_value=self.code_hash))
        return stack

    def pin(self):
        guard.pin_record(self.record, self.pins)
        self.pins_sha = guard.sha(self.pins)

    def run(self, mode, name, terminal=False, archive=None):
        attestation=self.record/'owned.json'
        dump(attestation,{'both_owned_jobs_terminal':True,'accuracy_scored':False,'jobs':[
            {'arm':arm,'owned_session_id':i+1,'exit_code':1,'saved_first_attempts':300,'observed_terminal_utc':'2026-01-01T00:00:00+00:00'} for i,arm in enumerate(('single','team'))]})
        return guard.run_check(self.root, self.record, self.pins, self.pins_sha,
            mode, self.record / name, owned_finished=terminal, archive=archive,
            owned_attestation=attestation,owned_attestation_sha=guard.sha(attestation))

    def row(self, arm, index, completed):
        data = deepcopy(self.inputs[index])
        limitation = 'Synthetic refusal: no executed final computation.'
        revision = {'messages': [{'role': 'user', 'content': f'Synthetic inline revision {arm}/{index}'}]}
        if arm == 'ordinary_multi_agent':
            primary={'answer':None,'evidence':[],'limitation':limitation,'error':None,'revision':None,'budget':None}
            diagnostic={'phase':'natural_primary','token_cap':200000,'tool_cap':30,'sdk_cap':30,'tokens':0,'tool_calls':0,
                'natural_tool_calls':0,'model_calls':0,'eligible':False,'fatal':False,'usage_known':True,'errors':[],
                'skip_reason':'Synthetic no explicit final','budget_stop':None,'initialization':{'charged':0},'call_limits':[]}
            totals={'prompt_tokens':0,'completion_tokens':0,'total_tokens':0,'llm_calls':0,'cost':0.0}
            trace=[{'role':'user','content':f'Synthetic retained trace {arm}/{index}'}]
            tagged=[{**v,'phase':'natural_primary'} for v in trace]
            workspace={'Observation':[]}
            snapshot={'version':1,'kind':'candidate_snapshot','candidate':'A','phase':'natural_primary','released':primary,
                'artifact':None,'workspace':workspace,'source_check':None,'usage':{'total_usage':totals,'per_call':[],
                'raw_accounting':{'total_usage':deepcopy(totals),'call_usage':[],'tool_calls_made':0},'observer':{}},'binding':{},
                'workspace_sha256':guard.fingerprint(workspace),'trace_sha256':guard._aux_fingerprint(trace),
                'tagged_trace_sha256':guard._aux_fingerprint(tagged),'diagnostics':deepcopy(diagnostic)}
            digest=guard._aux_fingerprint(snapshot)
            revision.update(protocol=guard.PROTOCOL,status='A_packet_unavailable',selected='A',fatal=False,
                A_only={'available':True,'snapshot_sha256':digest,'error':None,'source_bound':False},
                candidate_snapshot_sha256={'A':digest},phases=[diagnostic],messages=[snapshot],
                selection=None,admission={'exception':'ValueError'},independent_candidate_selected=False,final_check_issues=None)
        data.update(question_with_context=f"Question: {data['question']}\nContext:\nPatient FHIR ID is {data['patient_fhir_id']}.",
            agent_answer=None, evidence=[], limitation=limitation,
            latency=0.01 if completed else None, error=None,
            tokens={'prompt': 0, 'completion': 0, 'total': 0, 'cost': 0.0,
                'llm_calls': 0, 'cached_prompt': 0, 'repeated_prompt': 0, 'per_call': []},
            budget={'max_total_tokens': 200000, 'max_tool_calls': 30,
                'tool_calls_run': 0, 'stopped': None}, revision=revision,
            trace=tagged if arm=='ordinary_multi_agent' else [{'role': 'user', 'content': f'Synthetic retained trace {arm}/{index}'}],
            retrieved_fhir_resources={'Observation': []})
        if completed and index == 0:
            data.update(error='Synthetic early terminal exception', latency=None)
            data['revision'] = {'messages': revision['messages']}
        if completed:
            return self.store.externalize_result_audit(data, self.output, question_id=data['question_id'])
        return data

    def write_arms(self, completed_count):
        self.output.mkdir(parents=True, exist_ok=True)
        dump(self.output / '.local-provider.json', self.binding)
        for arm, strategy in guard.ARM_STRATEGIES.items():
            rows = [self.row(arm, i, i < completed_count) for i in range(300)]
            self.rows[arm] = rows
            dump(self.output / (arm + '.json'), rows)
            identity = {
                'agent_strategy': strategy, 'model': 'gpt-6-luna',
                'max_total_tokens': 200000, 'max_tool_calls': 30,
                'split': 'held_out', 'split_manifest_hash': self.manifest['split_sha256'],
                'enable_cache': True, 'add_patient_fhir_id': True, 'audit_storage': 'gzip',
                'audit_artifact_schema': 2, 'code_hash': self.code_hash,
                'condition': None, 'perturbations_hash': None, 'subset_hash': None,
                'repeat': None, 'input_hash': self.input_hash,
            }
            dump(self.output / (arm + '.json.run.json'), identity)
        for name in ('run_single.log', 'run_team.log'):
            (self.output / name).write_text('Synthetic generated lifecycle only.\n')

    def write_journals(self, finished):
        for pid in (12345, 12346):
            session = f'{pid:032x}'
            client = {'sdk': None, 'http': {'class': 'openai._base_client.SyncHttpxClientWrapper',
                    'mro': ['openai._base_client.SyncHttpxClientWrapper', 'httpx2.Client']},
                'transport': {'class': 'httpx2.HTTPTransport', 'mro': ['httpx2.HTTPTransport']},
                'pool': None, 'sdk_closed': False, 'http_closed': False,
                'client_id': pid, 'versions': {'openai': '3.26.1', 'httpx2': '2.13.1'}}
            start = {'version': 1, 'pid': pid, 'ppid': 1, 'counter': 1,
                'process_session': session, 'started_utc': '2026-01-01T00:00:00+00:00',
                'event': 'start', 'client': client, 'fd_count': 4}
            events = [start]
            if finished:
                events.append({**start, 'event': 'end',
                    'finished_utc': '2026-01-01T00:00:01+00:00', 'duration_seconds': 1.0,
                    'outcome': 'success', 'exception_chain': [],
                    'usage': {'prompt_tokens': 0, 'completion_tokens': 0,
                        'total_tokens': 0, 'cached_prompt_tokens': 0},
                    'http_status': 200, 'request_ids': {}})
            path = self.output / f'provider-events/azure-diagnostics-{pid}-{session}.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(json.dumps(event) + '\n' for event in events))


class V7LifecycleTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        if guard.sha(PACKET / 'v7_guard.py') != GUARD_LOADED_SHA:
            raise AssertionError('Guard source changed during synthetic lifecycle suite; rerun on stable bytes')

    def test_generated_600_first_attempt_lifecycle_and_exact_archive(self):
        with tempfile.TemporaryDirectory(prefix='v7-synthetic-lifecycle-') as directory:
            fixture = LifecycleFixture(directory)
            with fixture.patched_constants():
                fixture.pin()
                prelaunch = fixture.run('prelaunch', 'prelaunch.json')
                self.assertEqual(prelaunch['status'], 'PRELAUNCH_PASS')
                self.assertFalse(fixture.output.exists())
                fixture.write_arms(1)
                fixture.write_journals(False)
                initial = fixture.run('initial', 'initial.json')
                self.assertEqual(initial['status'], 'INITIAL_SNAPSHOT_PASS')
                self.assertEqual(initial['restored_audit_records'], 2)
                self.assertEqual(initial['journal']['totals']['unmatched_starts'], 2)
                with self.assertRaisesRegex(ValueError, 'must finish'):
                    fixture.run('completion', 'premature.json', terminal=True)
                self.assertFalse((fixture.record / 'premature.json').exists())
                fixture.write_arms(300)
                fixture.write_journals(True)
                (fixture.output / 'unreferenced_synthetic_debug.txt').write_text('Exclude this unowned member.\n')
                archive = fixture.root / 'output/archives/synthetic.zip'
                result = fixture.run('completion', 'completion.json', terminal=True, archive=archive)
                self.assertEqual(result['status'], 'PASS')
                self.assertEqual(result['expected_first_attempts'], 600)
                self.assertEqual(result['restored_audit_records'], 600)
                self.assertEqual(result['unique_referenced_artifacts'], 600)
                self.assertEqual(result['actual_base_members'], 607)
                self.assertEqual(result['provider_journal_members'], 2)
                self.assertTrue(result['archive_verified'])
                self.assertTrue(result['caller_asserted_owned_jobs_terminal'])
                for summary in result['arms'].values():
                    self.assertEqual(summary['scheduled_rows'], 300)
                    self.assertEqual(summary['saved_first_attempts'], 300)
                    self.assertEqual(summary['terminal_errors'], 1)
                    self.assertEqual(summary['verified_owned_gzip_audits'], 300)
                self.assertEqual(result['arms']['ordinary_multi_agent']['terminal_errors_without_protocol_metadata'], 1)
                with zipfile.ZipFile(archive) as saved:
                    self.assertEqual(len(saved.namelist()), 609)
                    self.assertNotIn('unreferenced_synthetic_debug.txt', saved.namelist())
                    for name, metadata in result['raw_files'].items():
                        self.assertEqual(hashlib.sha256(saved.read(name)).hexdigest(), metadata['sha256'])
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    fixture.run('completion', 'completion.json', terminal=True, archive=archive)
                with self.assertRaisesRegex(ValueError, 'must be absent'):
                    fixture.run('prelaunch', 'second_prelaunch.json')

    def test_pinned_metadata_and_extra_source_fail_before_launch(self):
        with tempfile.TemporaryDirectory(prefix='v7-synthetic-pins-') as directory:
            fixture = LifecycleFixture(directory)
            with fixture.patched_constants():
                fixture.pin()
                extra = fixture.root / 'agent/late_edit.py'
                extra.write_text('# Added after source inventory was fixed.\n')
                with self.assertRaisesRegex(ValueError, 'inventory differs'):
                    fixture.run('prelaunch', 'extra_source.json')
                extra.unlink()
                prospective = fixture.record / 'prospective_protocol.json'
                value = json.loads(prospective.read_text())
                value['workers_per_arm'] = 2
                dump(prospective, value)
                with self.assertRaisesRegex(ValueError, 'metadata differs'):
                    fixture.run('prelaunch', 'mutated_metadata.json')
                self.assertFalse(fixture.output.exists())

    def test_completed_retry_or_corrupt_owned_gzip_blocks_report_and_archive(self):
        with tempfile.TemporaryDirectory(prefix='v7-synthetic-failure-') as directory:
            fixture = LifecycleFixture(directory)
            with fixture.patched_constants():
                fixture.pin()
                fixture.write_arms(300)
                fixture.write_journals(True)
                arm = 'ordinary_multi_agent'
                rows = fixture.rows[arm]
                rows[299]['attempt_history'] = [{'error': 'Synthetic forbidden retry'}]
                dump(fixture.output / (arm + '.json'), rows)
                archive = fixture.root / 'output/archives/blocked.zip'
                with self.assertRaisesRegex(ValueError, 'retry history'):
                    fixture.run('completion', 'retry_report.json', terminal=True, archive=archive)
                self.assertFalse((fixture.record / 'retry_report.json').exists())
                self.assertFalse(archive.exists())
                rows[299].pop('attempt_history')
                dump(fixture.output / (arm + '.json'), rows)
                audit = fixture.output / rows[299]['audit_artifacts']['path']
                raw = bytearray(audit.read_bytes())
                raw[-5] ^= 0xFF
                audit.write_bytes(raw)
                with self.assertRaises((ValueError, OSError, EOFError)):
                    fixture.run('completion', 'gzip_report.json', terminal=True, archive=archive)
                self.assertFalse((fixture.record / 'gzip_report.json').exists())
                self.assertFalse(archive.exists())


class V7LaunchBoundaryTests(unittest.TestCase):
    def test_completion_requires_explicit_owned_terminal_assertion_before_any_output(self):
        with tempfile.TemporaryDirectory(prefix='v7-terminal-oracle-') as directory:
            fixture=LifecycleFixture(directory)
            with fixture.patched_constants():
                fixture.pin()
                with self.assertRaisesRegex(ValueError,'confirm both owned jobs'):
                    fixture.run('completion','not_terminal.json',terminal=False)
                self.assertFalse((fixture.record/'not_terminal.json').exists())
                self.assertFalse(fixture.output.exists())

    def test_wrong_declared_launch_profile_rejected_even_when_newly_pinned(self):
        with tempfile.TemporaryDirectory(prefix='v7-launch-profile-') as directory:
            fixture=LifecycleFixture(directory)
            commands=json.loads((fixture.record/'launch_commands.json').read_text())
            argv=commands['commands']['team']['argv'];argv[argv.index('--num_processes')+1]='2'
            dump(fixture.record/'launch_commands.json',commands)
            with fixture.patched_constants():
                fixture.pin()
                with self.assertRaisesRegex(ValueError,'exact launch argv'):
                    fixture.run('prelaunch','bad_launch.json')
                self.assertFalse((fixture.record/'bad_launch.json').exists())

    def test_report_collision_cannot_overwrite_source(self):
        with tempfile.TemporaryDirectory(prefix='v7-report-collision-') as directory:
            fixture=LifecycleFixture(directory)
            protected=fixture.root/'agent/single_agent.py';before=protected.read_bytes()
            with fixture.patched_constants():
                fixture.pin()
                with self.assertRaisesRegex(ValueError,'already exists'):
                    guard.run_check(fixture.root,fixture.record,fixture.pins,fixture.pins_sha,'prelaunch',protected)
                self.assertEqual(protected.read_bytes(),before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
