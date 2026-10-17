"""Synthetic metadata-only capture/gate regressions; no real verifier/audit call."""
import ast
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import bound_completion as bound
import capture_final_proof as capture
import completed_gate as gate

BASE = Path(__file__).resolve().parent
REPO = Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')


class BoundGate(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/private/tmp',prefix='single-bound-proof-fixture-')
        self.base = Path(self.temp.name)
        self.root,self.run = self.base/'project',self.base/'run'
        self.run.mkdir()
        self.output,self.proof_dir = self.base/'audit_must_stay_absent',self.base/'captured_proof'
        monitor = self.root/bound.MONITOR_REL
        monitor.parent.mkdir(parents=True)
        monitor.write_bytes((REPO/bound.MONITOR_REL).read_bytes())
        self.reader = self.root/'scripts/full_benchmark.py'
        self.reader.parent.mkdir(parents=True)
        self.reader.write_text('# Synthetic, never imported or executed.\n')
        self.source = self.base/'input.csv'
        with self.source.open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=['question_id','patient_fhir_id']);writer.writeheader()
            writer.writerows({'question_id':f'q{i:04}','patient_fhir_id':f'p{i%94:02}'} for i in range(2931))
        self.identity = {'code_hash':gate.FINGERPRINT,'agent_strategy':'single_agent'}
        self.manifest = {'questions_per_arm':2931,'selected_arms':['single_agent'],'run_identities':{'single_agent':self.identity},
                         'source_sha256':{'scripts/full_benchmark.py':bound.sha(self.reader)},
                         'input':{'path':str(self.source),'sha256':bound.sha(self.source)}}
        self.ledger = {'protocol':'v3-first-runtime-clean-terminal','selection':'first_runtime_clean_terminal','complete':False,
                       'arms':{'single_agent':{f'q{i:04}':{'runtime_clean':True,'selection':'runtime_clean_terminal','round':1,
                               'attempt_file':'attempts/single_agent/round-0001.json','runtime_failures_before':0} for i in range(2931)}},
                       'counts':{'single_agent':{'raw_attempts':2931,'runtime_failures':0,'intrinsic_errors':0}}}
        self.state = {'phase':'complete_selected_arms','selected_scope_complete':True,'benchmark_complete':False,
                      'selected_arms':['single_agent'],'pending_scoped_export':True,'analysis_status':'pending_scoped_export',
                      'finished_at':'2026-10-17T12:00:00+00:00','active':{},'attention':False,'accepted':{'single_agent':2931}}
        write(self.run/'acceptance.json',self.ledger)
        write(self.run/'supervisor_state.json',self.state)
        write(self.run/'run_manifest.json',self.manifest)
        write(self.run/'full_single_agent.json',[])
        write(self.run/'full_single_agent.json.run.json',self.identity)
        self.add_round(1)
        self.verify_calls = 0
        with patch.object(capture.subprocess,'run',side_effect=self.fake_verify):
            capture.capture(self.root,self.run,self.proof_dir)
        self.proof_path,self.binding_path = self.proof_dir/'proof.json',self.proof_dir/'binding.json'
        self.proof_sha,self.binding_sha = bound.sha(self.proof_path),bound.sha(self.binding_path)

    def tearDown(self):
        self.temp.cleanup()

    def add_round(self,number,sealed=True):
        folder = self.run/'attempts/single_agent';folder.mkdir(parents=True,exist_ok=True)
        result = folder/f'round-{number:04}.json'
        identity = Path(str(result)+'.run.json')
        batch = folder/f'round-{number:04}.batch.json'
        shard = folder/f'round-{number:04}.csv'
        shard.write_text('question_id,patient_fhir_id\nq0000,p00\n')
        write(result,[]);write(identity,self.identity)
        write(batch,{'arm':'single_agent','round':number,'question_ids':['q0000'],'input':str(shard),
                     'input_sha256':bound.sha(shard),'output':str(result),'run_identity':self.identity})
        if sealed:
            write(batch.with_suffix('.seal.json'),{'batch_sha256':bound.sha(batch),'result_sha256':bound.sha(result),
                                                'identity_sha256':bound.sha(identity)})

    def fake_verify(self,command,**kwargs):
        self.verify_calls += 1
        self.assertNotIn('--append-validation',command)
        snap = bound.snapshot(self.root,self.run)
        proof = {'status':'SELECTED_EXECUTION_COMPLETE_ANALYSIS_PENDING','selected_arms':['single_agent'],
                 'selected_accepted':2931,'questions_per_selected_arm':2931,'selected_scope_complete':True,
                 'original_three_arm_benchmark_complete':False,'canonical_first_clean_verified':True,'accepted_reruns':0,
                 'parked_pending_question_ids':[],'monitor_sha256':bound.MONITOR_SHA,
                 'manifest_sha256':bound.sha(self.run/'run_manifest.json'),'replayed_counts':self.ledger['counts']['single_agent'],
                 'sealed_batches':[{'round':x['round'],'classifications':{'accepted':2931 if x['round']==1 else 0},
                                    'result_sha256':x['result_sha256'],'seal_sha256':x['seal_sha256']} for x in snap['sealed_batches']]}
        return SimpleNamespace(returncode=0,stdout=(json.dumps(proof)+'\n').encode(),stderr=b'')

    def call(self):
        return gate.completion_gate(self.root,self.run,self.proof_path,self.binding_path,self.output)

    def stale_refused(self,expected=''):
        with self.assertRaisesRegex(ValueError,expected):
            self.call()
        self.assertEqual(bound.sha(self.proof_path),self.proof_sha)
        self.assertEqual(bound.sha(self.binding_path),self.binding_sha)
        self.assertFalse(self.output.exists())

    def test_unchanged_control(self):
        result = self.call()
        self.assertEqual(len(result['selections']),2931)
        self.assertEqual(self.verify_calls,1)
        self.assertFalse(self.output.exists())

    def test_stale_canonical(self):
        (self.run/'full_single_agent.json').write_text('[{"changed":true}]\n')
        self.stale_refused('Stale final proof')

    def test_stale_ledger_choice(self):
        value = deepcopy(self.ledger)
        value['arms']['single_agent']['q0000'].update(selection='intrinsic_terminal_error',runtime_failures_before=999)
        write(self.run/'acceptance.json',value)
        self.stale_refused('Stale final proof')

    def test_stale_ledger_round99(self):
        value = deepcopy(self.ledger)
        value['arms']['single_agent']['q0000'].update(round=99,attempt_file='attempts/single_agent/round-0099.json')
        write(self.run/'acceptance.json',value)
        self.stale_refused('Stale final proof')

    def test_stale_whole_ledger_counts(self):
        value = deepcopy(self.ledger)
        value['counts']['single_agent'].update(raw_attempts=2932,runtime_failures=1)
        write(self.run/'acceptance.json',value)
        self.stale_refused('Stale final proof')

    def test_stale_state_timestamp(self):
        value = deepcopy(self.state);value['finished_at']='2099-01-01T00:00:00+00:00'
        write(self.run/'supervisor_state.json',value)
        self.stale_refused('Stale final proof')

    def test_later_exactly_sealed_round(self):
        self.add_round(2)
        self.stale_refused('Stale final proof')

    def test_extra_unsealed_round(self):
        self.add_round(2,sealed=False)
        self.stale_refused('inventory is incomplete')

    def test_stale_canonical_identity(self):
        write(self.run/'full_single_agent.json.run.json',{'agent_strategy':'changed'})
        self.stale_refused('Stale final proof')

    def test_stale_raw_result(self):
        (self.run/'attempts/single_agent/round-0001.json').write_text('[1]\n')
        self.stale_refused('seal no longer matches')

    def test_stale_shard(self):
        (self.run/'attempts/single_agent/round-0001.csv').write_text('question_id,patient_fhir_id\nchanged,p\n')
        self.stale_refused('shard changed')

    def test_changed_proof_bytes(self):
        self.proof_path.write_bytes(self.proof_path.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'proof digest/schema'):
            self.call()
        self.assertFalse(self.output.exists())

    def test_incomplete_refuses_before_canonical_or_verifier(self):
        value = deepcopy(self.ledger);value['arms']['single_agent'].pop('q0000')
        write(self.run/'acceptance.json',value)
        (self.run/'full_single_agent.json').unlink()
        with patch.object(capture.subprocess,'run',side_effect=AssertionError('Verifier must not run')):
            with self.assertRaisesRegex(ValueError,'2931 selected outcomes'):
                capture.capture(self.root,self.run,self.base/'incomplete_capture_must_stay_absent')
        with self.assertRaisesRegex(ValueError,'2931 selected outcomes'):
            self.call()
        self.assertFalse(self.output.exists())

    def test_change_during_capture_rejected_without_proof(self):
        target = self.base/'unstable_capture_must_stay_absent'
        def mutate(*args,**kwargs):
            result = self.fake_verify(*args,**kwargs)
            (self.run/'full_single_agent.json').write_text('[{"after":true}]\n')
            return result
        with patch.object(capture.subprocess,'run',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'changed during final monitor'):
                capture.capture(self.root,self.run,target)
        self.assertFalse(target.exists())

    def test_unknown_usage_canonical_validator_refuses(self):
        tree = ast.parse((BASE/'audit_complete_single_cache.py').read_text())
        main = next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
        outer = next(x for x in main.body if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='original')
        inner = next(x for x in outer.body if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='call')
        function = ast.FunctionDef(name='validate',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='calls')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[inner],decorator_list=[])
        module = ast.Module(body=[function],type_ignores=[]);ast.fix_missing_locations(module)
        namespace={'require':bound.require}
        exec(compile(module,'selected_call_validator','exec'),namespace)
        with self.assertRaisesRegex(ValueError,'unknown usage'):
            namespace['validate']([{'prompt':0,'completion':0,'cached':0,'usage_available':False}])


if __name__=='__main__':
    unittest.main()
