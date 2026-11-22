"""Synthetic metadata-only gate tests; no full audit, scorer, or provider."""
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

BASE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('gate_under_test',BASE/'completed_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


class SyntheticGate(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/private/tmp',prefix='single-cache-gate-fixture-')
        self.base = Path(self.temp.name)
        self.root,self.run = self.base/'project',self.base/'run'
        self.run.mkdir()
        self.output = self.base/'must_stay_absent'
        self.proof_path = self.base/'proof.json'
        self.monitor = self.root/'benchmarks/single-agent-full-run-20261017/monitor_single_agent.py'
        self.reader = self.root/'scripts/full_benchmark.py'
        for path in (self.monitor,self.reader):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('# Synthetic metadata fixture, never imported.\n')
        self.source = self.base/'input.csv'
        with self.source.open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=['question_id','patient_fhir_id'])
            writer.writeheader()
            writer.writerows({'question_id':f'q{i:04}','patient_fhir_id':f'p{i%94:02}'} for i in range(2931))
        identity = {'code_hash':gate.FINGERPRINT,'agent_strategy':'single_agent'}
        self.manifest = {'questions_per_arm':2931,'selected_arms':['single_agent'],
                         'run_identities':{'single_agent':identity},
                         'source_sha256':{'scripts/full_benchmark.py':gate.sha(self.reader)},
                         'input':{'path':str(self.source),'sha256':gate.sha(self.source)}}
        self.ledger = {'protocol':'v3-first-runtime-clean-terminal','selection':'first_runtime_clean_terminal','complete':False,
                       'arms':{'single_agent':{f'q{i:04}':{'runtime_clean':True,'selection':'runtime_clean_terminal','round':1,'attempt_file':'attempts/single_agent/round-0001.json'} for i in range(2931)}},
                       'counts':{'single_agent':{'raw_attempts':2931,'runtime_failures':0,'intrinsic_errors':0}}}
        self.state = {'phase':'complete_selected_arms','selected_scope_complete':True,'benchmark_complete':False,
                      'selected_arms':['single_agent'],'pending_scoped_export':True,'analysis_status':'pending_scoped_export',
                      'finished_at':'2026-10-17T12:00:00+00:00','active':{},'attention':False,'accepted':{'single_agent':2931}}
        self.proof = {'status':'SELECTED_EXECUTION_COMPLETE_ANALYSIS_PENDING','selected_arms':['single_agent'],
                      'selected_accepted':2931,'questions_per_selected_arm':2931,'selected_scope_complete':True,
                      'original_three_arm_benchmark_complete':False,'canonical_first_clean_verified':True,'accepted_reruns':0,
                      'parked_pending_question_ids':[],'monitor_sha256':gate.sha(self.monitor),
                      'replayed_counts':self.ledger['counts']['single_agent'],
                      'sealed_batches':[{'round':1,'classifications':{'accepted':2931}}]}
        write(self.run/'full_single_agent.json',[])
        write(self.run/'full_single_agent.json.run.json',identity)
        self.save()

    def save(self):
        write(self.run/'acceptance.json',self.ledger)
        write(self.run/'supervisor_state.json',self.state)
        write(self.run/'run_manifest.json',self.manifest)
        self.proof['manifest_sha256'] = gate.sha(self.run/'run_manifest.json')
        write(self.proof_path,self.proof)

    def tearDown(self):
        self.temp.cleanup()

    def call(self):
        return gate.completion_gate(self.root,self.run,self.proof_path,self.output)

    def test_full_metadata_gate_without_capture(self):
        result = self.call()
        self.assertEqual(len(result['selections']),2931)
        self.assertEqual(len(result['patients']),2931)
        self.assertFalse(self.output.exists())

    def test_incomplete_ledger_refused_before_proof_or_canonical(self):
        self.ledger['arms']['single_agent'].pop('q0000')
        self.save()
        self.proof_path.unlink()
        (self.run/'full_single_agent.json').unlink()
        with self.assertRaisesRegex(ValueError,'2931 selected outcomes'):
            self.call()
        self.assertFalse(self.output.exists())

    def test_first_clean_flag_required(self):
        self.ledger['arms']['single_agent']['q0000']['runtime_clean'] = False
        self.save()
        with self.assertRaisesRegex(ValueError,'first-clean proof'):
            self.call()

    def test_active_worker_refused(self):
        self.state['active'] = {'single_agent':{'round':2}}
        self.save()
        with self.assertRaisesRegex(ValueError,'supervisor proof'):
            self.call()

    def test_incomplete_monitor_receipt_refused(self):
        self.proof['selected_scope_complete'] = False
        self.save()
        with self.assertRaisesRegex(ValueError,'monitor receipt'):
            self.call()

    def test_parked_question_refused(self):
        self.proof['parked_pending_question_ids'] = ['q0000']
        self.save()
        with self.assertRaisesRegex(ValueError,'monitor receipt'):
            self.call()

    def test_raw_runtime_count_disagreement_refused(self):
        self.proof['replayed_counts'] = {'raw_attempts':2932,'runtime_failures':0,'intrinsic_errors':0}
        self.save()
        with self.assertRaisesRegex(ValueError,'attempt counts differ'):
            self.call()

    def test_identity_changed_refused(self):
        write(self.run/'full_single_agent.json.run.json',{'code_hash':'changed','agent_strategy':'single_agent'})
        with self.assertRaisesRegex(ValueError,'inference identity'):
            self.call()

    def test_wrong_input_patient_cohort_refused(self):
        with self.source.open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=['question_id','patient_fhir_id']);writer.writeheader()
            writer.writerows({'question_id':f'z{i:04}','patient_fhir_id':f'p{i%94:02}'} for i in range(2931))
        self.manifest['input']['sha256'] = gate.sha(self.source)
        self.save()
        with self.assertRaisesRegex(ValueError,'question cohort differs'):
            self.call()


if __name__=='__main__':
    unittest.main()
