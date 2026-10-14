"""Synthetic offline diagnostics checks; no clinical run or model calls."""
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

HELPER = Path('/private/tmp/fhir_v3_large_run_diagnostics.py')
REPO = Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
AUDIT = Path('/private/tmp/fhir-v3-combined-tSu0kb/scripts/audit_store.py')
V3 = Path('/private/tmp/fhir-v3-design/agent/query_contract_agent.py')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='fhir-v3-diagnostics-mock-', dir='/private/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('utils', 'scripts', 'agent', 'manifests', 'final_dataset', 'results'):
            (self.root/name).mkdir()
        shutil.copy2(REPO/'utils/answer_scorer.py', self.root/'utils/answer_scorer.py')
        shutil.copy2(AUDIT, self.root/'scripts/audit_store.py')
        shutil.copy2(V3, self.root/'agent/query_contract_agent.py')
        self.helper = module(HELPER, 'mock_v3_diagnostics')
        self.audit = module(self.root/'scripts/audit_store.py', 'mock_v3_audit')
        self.ids = [f'q{i}' for i in range(300)]
        self.split = self.root/'manifests/split.json'
        self.split.write_text(json.dumps({'splits':{'held_out':{'question_ids':self.ids}}}))
        self.dataset = self.root/'final_dataset/questions_answers_sql_fhir.csv'
        with self.dataset.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['question_id','question','true_answer','patient_fhir_id'])
            writer.writeheader()
            writer.writerows({'question_id':qid,'question':'How many weight measurements?','true_answer':'[]' if qid in ('q3','q13') else '[[1]]','patient_fhir_id':'synthetic-patient'} for qid in self.ids)
        with (self.root/'manifests/question_families.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['question_id','family','template'])
            writer.writeheader()
            writer.writerows({'question_id':qid,'family':'numeric','template':'Count'} for qid in self.ids)
        self.manifest=self.root/'source_manifest.json'
        self.manifest.write_text(json.dumps({'dataset_sha256':self.helper.sha256(self.dataset),'split_sha256':self.helper.sha256(self.split),
             'source_sha256':{name:self.helper.sha256(self.root/name) for name in ['utils/answer_scorer.py','scripts/audit_store.py','agent/query_contract_agent.py']}}))
        self.raw=self.root/'results/ordinary_multi_agent.json'
        self.sidecar=Path(str(self.raw)+'.run.json')
        self.sidecar.write_text(json.dumps({'agent_strategy':'multi_agent_v3','model':'synthetic','max_total_tokens':200000,'max_tool_calls':30,
             'audit_storage':'gzip','split_manifest_hash':self.helper.sha256(self.split),'code_hash':'synthetic-source-fingerprint','input_hash':'synthetic-input-fingerprint'}))
        self.rows=[self.base_row(qid) for qid in self.ids]
        self.modify_rows()
        self.save()

    def candidate(self,value):
        return {'answer':value,'evidence':[],'contributing_ids':[],'operation':'synthetic calculation','limitation':None}

    def base_row(self,qid):
        candidate=self.candidate(1)
        return {'question_id':qid,'patient_fhir_id':'synthetic-patient','agent_answer':1,'evidence':[], 'limitation':None,'error':None,'latency':1,
            'tokens':{'total':450,'per_call':[{'usage_available':True}]},'budget':{'tool_calls_run':3,'stopped':None},
            'trace':[{'role':'assistant','content':'immutable synthetic trace','phase':'solver_boundary','model_call':False}],
            'retrieved_fhir_resources':{'Observation':[{'id':'synthetic'}]},
            'revision':{'protocol':'query-contract-v3.2-static-audit','draft':deepcopy(candidate),'primary':deepcopy(candidate),'final':deepcopy(candidate),
               'final_check_issues':[],'check_issues':[],'solver_source_revised':False,'targeted_revised':False,'revised':False,
               'release_policy':'source_valid_primary_preserved','contract':{'requirements':[]},'audit':{'checks':[{'verdict':'PASS'}],'repair':None},'audit_error':None,
               'targeted_proof':None,'unresolved_scope':{},'phases':[{'phase':name,'token_cap':caps[0],'tool_cap':caps[1],
                   'tokens':150 if name!='targeted' else 0,'tool_calls':3 if name=='solver' else 0,'fatal':False,'errors':[]} for name,caps in [('contract',(10000,0)),('solver',(175000,27)),('audit',(15000,0))]]}}

    def modify_rows(self):
        r=self.rows[1]['revision'];r['draft']['answer']=0;r['solver_source_revised']=True;r['revised']=True
        r=self.rows[2]['revision'];r['draft']['answer']=0;r['primary']['answer']=0;r['targeted_revised']=True;r['revised']=True
        r['release_policy']='source_checked_targeted_local_constraint';r['audit']={'checks':[{'verdict':'FAIL'}],'repair':{'requirement_id':'c1'}}
        r['targeted_proof']={'status':'source_checked_local_constraint'}
        self.rows[3]['agent_answer']=None;self.rows[3]['revision']['final']=self.candidate(None)
        self.rows[3]['revision']['final_check_issues']=['source_scope: synthetic unknown time'];self.rows[3]['revision']['release_policy']='primary_failed_execution_or_source_contract'
        self.rows[4]['agent_answer']=None;self.rows[4]['error']='timeout';self.rows[4]['latency']=None
        self.rows[5]['agent_answer']=0;r=self.rows[5]['revision'];r['final']['answer']=0;r['targeted_revised']=True;r['revised']=True
        r['release_policy']='source_checked_targeted_local_constraint';r['targeted_proof']={'status':'source_checked_local_constraint'}
        self.rows[6]['revision']['audit']={'checks':[{'verdict':'UNKNOWN'}],'repair':None}
        self.rows[7]['revision']['audit']=None;self.rows[7]['revision']['audit_error']='auditor did not return a checks object'
        self.rows[8]['revision']['contract']=None;self.rows[8]['revision']['audit']=None
        self.rows[9]['agent_answer']=None;r=self.rows[9]['revision'];r['final']=self.candidate(None);r['final_check_issues']=['source_scope: unknown time']
        r['unresolved_scope']={'Observation/undated':'unknown_time'};r['release_policy']='primary_failed_execution_or_source_contract'
        self.rows[10]['revision']['audit']={'checks':[{'verdict':'FAIL'}],'repair':{'requirement_id':'c1'}}
        self.rows[10]['revision']['targeted_proof']={'status':'unsupported','reason':'direct count required'}
        self.rows[11]['agent_answer']=None;self.rows[11]['error']='Budget exceeded';self.rows[11]['budget']['stopped']='local token ceiling'
        self.rows[11]['revision']['phases'][0].update(tokens=10001,fatal=True,errors=['contract local ceiling'])
        self.rows[11]['tokens']['total']=10301
        first=deepcopy(self.rows[12]);first['agent_answer']=0;first['revision']['final']['answer']=0
        self.rows[12]['attempt_history']=[first]
        self.rows[13]['agent_answer']=None;self.rows[13]['error']='provider error';self.rows[13]['latency']=None

    def save(self):
        saved=[]
        for row in self.rows:
            item=deepcopy(row)
            history=item.pop('attempt_history',None)
            item=self.audit.externalize_result_audit(item,self.raw.parent,question_id=item['question_id'])
            if history:
                item['attempt_history']=[self.audit.externalize_result_audit(h,self.raw.parent,question_id=item['question_id']) for h in history]
            saved.append(item)
        self.raw.write_text(json.dumps(saved))

    def run_report(self):
        return self.helper.run(self.root,['v3='+str(self.raw)],self.root/'report',self.manifest)

    def test_complete_300_first_attempt_mechanisms_and_exact_artifacts(self):
        original_hash=self.helper.sha256(self.raw)
        report=self.run_report();s=report['arms']['v3']
        self.assertEqual(s['questions'],300)
        self.assertEqual(s['solver_source_revised'],1)
        self.assertEqual(s['source_revision_helped'],1)
        self.assertEqual(s['targeted_helped'],1)
        self.assertEqual(s['targeted_harmed'],1)
        self.assertEqual(s['terminal_error'],3)
        self.assertEqual(s['unverified_null_credit'],1)
        self.assertEqual(s['scored_null_credit'],1)
        self.assertEqual(s['phase_fatal'],1)
        self.assertEqual(s['unresolved_scope_reason_counts'],{'unknown_time':1})
        self.assertEqual(s['targeted_denial_reason_counts'],{'direct count required':1})
        self.assertEqual(s['audit_verdict_counts']['UNKNOWN'],1)
        self.assertIn('phase_cap_exceeded:contract:tokens',s['accounting_anomalies'])
        records=json.loads((self.root/'report/mechanisms_rows.json').read_text())
        prior=next(row for row in records if row['question_id']=='q12')
        self.assertFalse(prior['correct']);self.assertEqual(prior['recorded_retries'],1)
        self.assertEqual(self.helper.sha256(self.raw),original_hash)
        self.assertEqual(len(report['raw_inputs']['v3']['first_attempt_artifacts']),300)
        self.assertEqual(report['input_hashes']['scorer'],self.helper.sha256(REPO/'utils/answer_scorer.py'))

    def test_displayed_null_or_evidence_mismatch_cannot_receive_source_valid_credit(self):
        row=self.base_row('q3');row['agent_answer']=None;row['revision']['final']['answer']=0
        score=lambda answer,qid: {'typed_correct':int(answer is None),'answer_type':'empty','typed_reason':'synthetic'}
        record=self.helper.row_record(row,'q3',{'family':'numeric'},score,{}, {'max_total_tokens':200000,'max_tool_calls':30})
        self.assertTrue(record['scored_null_credit'])
        self.assertTrue(record['unverified_null_credit'])
        self.assertFalse(record['source_valid_release'])
        self.assertIn('released_artifact_mismatch',record['accounting_anomalies'])
        row=self.base_row('q0');row['evidence']=[{'resource_id':'Observation/different','field_path':'valueQuantity.value','value':1}]
        record=self.helper.row_record(row,'q0',{'family':'numeric'},score,{}, {'max_total_tokens':200000,'max_tool_calls':30})
        self.assertFalse(record['source_valid_release'])
        self.assertIn('released_artifact_mismatch',record['accounting_anomalies'])
        self.assertFalse(self.helper.same_json(False,0))

    def test_partial_cohort_cannot_score_or_write(self):
        rows=json.loads(self.raw.read_text());rows[-1]['latency']=None;rows[-1]['error']=None
        self.raw.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError,'incomplete first attempt'):
            self.run_report()
        self.assertFalse((self.root/'report').exists())

    def test_missing_cohort_row_cannot_write(self):
        rows=json.loads(self.raw.read_text());self.raw.write_text(json.dumps(rows[:-1]))
        with self.assertRaisesRegex(ValueError,'incomplete scheduled cohort'):
            self.run_report()
        self.assertFalse((self.root/'report').exists())

    def test_wrong_question_artifact_cannot_score(self):
        rows=json.loads(self.raw.read_text());rows[0]['audit_artifacts']=rows[1]['audit_artifacts']
        self.raw.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError,'another question'):
            self.run_report()
        self.assertFalse((self.root/'report').exists())

    def test_corrupt_artifact_cannot_write(self):
        row=json.loads(self.raw.read_text())[0]
        (self.raw.parent/row['audit_artifacts']['path']).write_bytes(b'not gzip')
        with self.assertRaises(OSError):
            self.run_report()
        self.assertFalse((self.root/'report').exists())

    def test_scorer_source_mismatch_rejected(self):
        with (self.root/'utils/answer_scorer.py').open('a') as stream:stream.write('\n# changed\n')
        with self.assertRaisesRegex(ValueError,'scorer differs'):
            self.run_report()
        self.assertFalse((self.root/'report').exists())

    def test_streaming_reader_refuses_trailing_or_truncated_json(self):
        for content in ('[{}] trailing','[{','[{},]'):
            with self.subTest(content=content):
                self.raw.write_text(content)
                with self.assertRaises(ValueError):list(self.helper.iter_rows(self.raw,chunk_chars=1))


if __name__=='__main__':
    unittest.main()
