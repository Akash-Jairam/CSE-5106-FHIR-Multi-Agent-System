"""Synthetic completed cohorts only; no actual run/result reads or inference."""
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HELPER = Path('/private/tmp/fhir_v4_large_run_diagnostics_corrected.py')
spec = importlib.util.spec_from_file_location('v4_diagnostics_under_test',HELPER)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
PUBLIC = Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False)+'\n')


def final(answer=1):
    return {'answer':answer,'evidence':[],'operation':'count','contributing_ids':[],'limitation':None}


class Fixture:
    def __init__(self,root):
        self.root=Path(root); self.record=self.root/'benchmarks'/m.EXPERIMENT
        self.out=self.root/m.RUN_RELATIVE; self.ids=['synthetic-%03d'%i for i in range(300)]
        self.record.mkdir(parents=True); self.out.mkdir(parents=True)
        self.marker=self.root/'SCORER_IMPORTED'
        prefix='from pathlib import Path\nPath('+repr(str(self.marker))+').write_text("imported")\n'
        required={'utils/answer_scorer.py':prefix+(PUBLIC/'utils/answer_scorer.py').read_text(),
                  'scripts/audit_store.py':(PUBLIC/'scripts/audit_store.py').read_text(),
                  'agent/plan_conditioned_agent.py':"PHASE_CAPS = {'planner':(10000,0),'solver':(190000,30)}\nPROTOCOL = 'plan-conditioned-v4'\n",
                  'agent/query_contract_agent.py':'# frozen inherited V3\n',
                  'agent/single_agent.py':'# frozen single\n',
                  'utils/__init__.py':'raise RuntimeError("utils bootstrap forbidden")\n'}
        for i in range(135): required['public/filler_%03d.py'%i]='# synthetic frozen public source\n'
        families=self.root/'manifests/question_families.csv';families.parent.mkdir(parents=True)
        with families.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['question_id','family','template']);writer.writeheader()
            for i,qid in enumerate(self.ids): writer.writerow({'question_id':qid,'family':'count_a' if i%2 else 'count_b','template':'How many observations?'})
        required['manifests/question_families.csv']=families.read_text()
        assert len(required)==142
        self.frozen={}
        for rel,text in required.items():
            path=self.root/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
            self.frozen[rel]=m.sha256(path)
        with zipfile.ZipFile(self.record/'source.zip','w') as packed:
            for rel in self.frozen: packed.write(self.root/rel,rel)
        dataset=self.root/'final_dataset/questions_answers_sql_fhir.csv';dataset.parent.mkdir()
        with dataset.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['question_id','patient_fhir_id','question','true_answer','assumption']);writer.writeheader()
            for i,qid in enumerate(self.ids): writer.writerow({'question_id':qid,'patient_fhir_id':'patient-synthetic',
                'question':'How many observations?','true_answer':'[]' if i in (0,1) else '[[1]]','assumption':''})
        write_json(self.root/'manifests/split.json',{'splits':{'held_out':{'question_ids':self.ids}}})
        (self.record/'design.txt').write_text('Synthetic selected V4 design.\n')
        self.manifest={'experiment_id':m.EXPERIMENT,'expected_questions_per_arm':300,'arms':list(m.ARMS),
            'model':'gpt-6-luna','max_total_tokens':200000,'max_tool_calls':30,'output_directory':str(self.out),
            'source_sha256':self.frozen,'source_archive_sha256':m.sha256(self.record/'source.zip'),
            'design_sha256':m.sha256(self.record/'design.txt'),'dataset_sha256':m.sha256(dataset),
            'split_sha256':m.sha256(self.root/'manifests/split.json')}
        write_json(self.record/'manifest.json',self.manifest)
        self.proof={'status':'PASS','code_hash':'c'*64,'ordered_input_hash':'d'*64,
            'manifest_sha256':m.sha256(self.record/'manifest.json'),'dataset_sha256':self.manifest['dataset_sha256'],
            'split_sha256':self.manifest['split_sha256'],'source_archive_sha256':self.manifest['source_archive_sha256'],'arms':{}}
        self.rows={};self.identities={}
        self.audit=m.load_module(self.root/'scripts/audit_store.py','synthetic_audit_fixture')
        for arm,strategy in m.ARMS.items():
            team=arm=='ordinary_multi_agent';rows=[]
            for i,qid in enumerate(self.ids):
                answer=None if i in (0,1) else 1
                if not team and i==4:answer=0
                if team and i==2:answer=2
                f=final(answer)
                rev={'draft':deepcopy(f),'final':deepcopy(f),'final_check_issues':[],'revised':False}
                if not team:
                    rev['draft']={k:f[k] for k in ('answer','evidence','limitation')}
                    rev['final']=deepcopy(rev['draft'])
                trace=[{'role':'system','content':'Unchanged solver'},
                       {'role':'assistant','content':None,'tool_calls':[{'id':'compute','function':{'name':'execute_python_code','arguments':json.dumps({'mode':'final','code':'answer = 1'})}}]},
                       {'role':'tool','name':'execute_python_code','tool_call_id':'compute','content':json.dumps(dict(f,mode='final'))},
                       {'role':'assistant','content':'computed'},
                       {'role':'assistant','content':None,'tool_calls':[{'id':'automatic_field_check_1','function':{'name':'check_fields','arguments':json.dumps({'citations':f['evidence']})}}]},
                       {'role':'tool','name':'check_fields','tool_call_id':'automatic_field_check_1','content':json.dumps({'issues':[]})}]
                if team:
                    plan={'status':'READY','steps':[{'id':'p1','operator':'aggregate','quote':'How many','span':[0,8],
                        'inputs':[],'requirement':'Count observations','supplied_span':[0,8],'span_normalization':'unchanged'}],'unknowns':[]}
                    ph=hashlib.sha256(json.dumps(plan,sort_keys=True,default=str,allow_nan=False).encode()).hexdigest()
                    rev.update(protocol=m.PROTOCOL,primary=deepcopy(f),primary_revision={'final_check_issues':[]},
                        release_policy=m.CLEAN_POLICY,solver_source_revised=False,targeted_revised=False,
                        plan=plan,plan_sha256=ph,plan_visible=True,plan_error=None,
                        phases=[{'phase':'planner','token_cap':10000,'tool_cap':0,'tokens':10,'tool_calls':0,'fatal':False,'errors':[]},
                                {'phase':'solver','token_cap':190000,'tool_cap':30,'tokens':20,'tool_calls':3,'fatal':False,'errors':[]}])
                    trace=[{'role':'system','content':'Planner'},{'role':'assistant','content':'plan'},
                           {'role':'system','content':'Solver. Frozen advisory plan JSON:'+json.dumps(plan)},
                           {'role':'assistant','content':'computed'},
                           {'role':'assistant','content':'Host phase boundary','model_call':False,'phase':'solver_boundary'},
                           {'role':'tool','name':'check_fields','content':'{}'}]
                    if i==1:
                        rev['release_policy']='primary_failed_execution_or_source_contract'
                        rev['final_check_issues']=[{'type':'source_scope'}];rev['unresolved_scope']={'Observation/synthetic':'missing_time'}
                        rev['final']={'answer':None,'evidence':[],'limitation':'Unresolved scope'}
                    if i==5:
                        rev.update(plan=None,plan_sha256=None,plan_visible=False,plan_error='planner declared UNKNOWN; advice ignored')
                        trace[2]['content']='Solver without advice'
                row={'question_id':qid,'patient_fhir_id':'patient-synthetic','agent_answer':answer,'evidence':[],
                    'limitation':'Unresolved scope' if team and i==1 else None,'error':None,'latency':0.2,
                    'tokens':{'prompt':20,'completion':10,'total':30,'llm_calls':2 if team else 1,'cost':None,
                              'per_call':[{'prompt':10,'completion':0,'usage_available':True},{'prompt':10,'completion':10,'usage_available':True}]},
                    'budget':{'tool_calls_run':3,'stopped':None},'revision':rev,'trace':trace}
                if team and i==3:
                    row.update(agent_answer=1,error='Max retries exceeded: synthetic provider timeout',revision=None)
                    row['tokens']['per_call']=[{'prompt':0,'completion':0,'usage_available':False}]
                rows.append(self.audit.externalize_result_audit(row,self.out,question_id=qid))
            self.rows[arm]=rows
            identity={'version':2,'agent_strategy':strategy,'model':'gpt-6-luna','max_total_tokens':200000,
                'max_tool_calls':30,'input_hash':self.proof['ordered_input_hash'],'code_hash':self.proof['code_hash'],
                'split':'held_out','split_manifest_hash':self.proof['split_sha256'],'repeat':None,'condition':None,
                'subset_hash':None,'perturbations_hash':None,'add_patient_fhir_id':True,'enable_cache':True,
                'audit_storage':'gzip','audit_artifact_schema':2,'base_url_hash':hashlib.sha256(b'').hexdigest()}
            self.identities[arm]=identity;self.save_arm(arm)
        self.proof_path=self.root/'completion_proof.json';self.save_proof()

    def save_arm(self,arm):
        raw=self.out/(arm+'.json');side=Path(str(raw)+'.run.json')
        write_json(raw,self.rows[arm]);write_json(side,self.identities[arm])
        self.proof['arms'][arm]={'result_sha256':m.sha256(raw),'identity_sha256':m.sha256(side)}

    def save_proof(self):write_json(self.proof_path,self.proof)

    def run(self,confirm=True,proof_digest=None):
        self.save_proof()
        return m.run(self.root,self.proof_path,proof_digest or m.sha256(self.proof_path),self.root/'diagnostics',confirm)


class V4DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='v4-diagnostics-synthetic-');self.addCleanup(self.temp.cleanup)
        self.f=Fixture(self.temp.name)

    def rejected_before_scoring(self,match):
        with self.assertRaisesRegex(ValueError,match):self.f.run()
        self.assertFalse(self.f.marker.exists(),'Scorer was loaded before completed-pair validation')
        self.assertFalse((self.f.root/'diagnostics').exists())

    def single_record(self,answer=1):
        f=final(answer); rendered={k:f[k] for k in ('answer','evidence','limitation')}
        trace=[{'role':'assistant','tool_calls':[{'id':'f','function':{'name':'execute_python_code','arguments':json.dumps({'mode':'final'})}}]},
               {'role':'tool','tool_call_id':'f','name':'execute_python_code','content':json.dumps(dict(f,mode='final'))},
               {'role':'assistant','tool_calls':[{'id':'automatic_field_check_1','function':{'name':'check_fields','arguments':json.dumps({'citations':[]})}}]},
               {'role':'tool','tool_call_id':'automatic_field_check_1','name':'check_fields','content':json.dumps({'issues':[]})}]
        row={'agent_answer':answer,'evidence':[],'limitation':None,'revision':{'final':rendered,'final_check_issues':[]},'trace':trace}
        return row

    def record(self,row):
        return m.row_record(row,'synthetic',{'family':'count'},lambda a,q:{'typed_correct':int(a==1),'answer_type':'number','typed_reason':'synthetic'},False)

    def test_actual_single_rendered3_binds_retained_executed5(self):
        row=self.single_record(); self.assertEqual(len(row['revision']['final']),3)
        r=self.record(row);self.assertIs(r['source_valid_release'],True)
        self.assertTrue(r['source_checked_rendered_release']);self.assertTrue(r['source_valid_correct_release'])
        self.assertFalse(r['credited_without_source_valid_release'])

    def test_clean_prose_single_execution_is_na_not_false_zero(self):
        row=self.single_record();row['trace']=row['trace'][2:]
        r=self.record(row);self.assertIsNone(r['source_valid_release'])
        self.assertTrue(r['source_checked_rendered_release']);self.assertTrue(r['source_valid_release_support_unavailable'])
        self.assertFalse(r['credited_without_source_valid_release']);self.assertTrue(r['credited_with_source_release_support_unavailable'])

    def test_single_retrieval_invalidates_earlier_execution(self):
        row=self.single_record();row['trace'].insert(2,{'role':'tool','name':'get_resources_by_resource_id','content':'failed retrieval'})
        r=self.record(row);self.assertIsNone(r['source_valid_release'])

    def test_single_auto_failure_invalidates_earlier_execution(self):
        row=self.single_record();row['trace'][2:2]=[
            {'role':'assistant','tool_calls':[{'id':'a','function':{'name':'execute_python_code','arguments':json.dumps({'mode':'auto'})}}]},
            {'role':'tool','tool_call_id':'a','name':'execute_python_code','content':json.dumps({'error':'failed'})}]
        self.assertIsNone(self.record(row)['source_valid_release'])

    def test_single_inspection_preserves_execution(self):
        row=self.single_record();row['trace'][2:2]=[
            {'role':'assistant','tool_calls':[{'id':'a','function':{'name':'execute_python_code','arguments':json.dumps({'mode':'inspect'})}}]},
            {'role':'tool','tool_call_id':'a','name':'execute_python_code','content':json.dumps({'answer':{'count':1},'mode':'inspect'})}]
        self.assertIs(self.record(row)['source_valid_release'],True)

    def test_single_rendered_bool_number_mismatch_not_bound(self):
        row=self.single_record();row['revision']['final']['answer']=True;row['agent_answer']=True
        self.assertIsNone(self.record(row)['source_valid_release'])

    def test_single_source_check_must_match_and_follow_execution(self):
        row=self.single_record();row['trace'][2:4],row['trace'][0:2]=row['trace'][0:2],row['trace'][2:4]
        self.assertIsNone(self.record(row)['source_valid_release'])

    def test_failed_first_automatic_check_requires_new_final(self):
        row=self.single_record()
        row['trace'][3]['content']=json.dumps({'issues':['source mismatch']})
        row['trace'].extend([
            {'role':'assistant','tool_calls':[{'id':'automatic_field_check_2','function':{'name':'check_fields','arguments':json.dumps({'citations':[]})}}]},
            {'role':'tool','tool_call_id':'automatic_field_check_2','name':'check_fields','content':json.dumps({'issues':[]})}])
        self.assertIsNone(self.record(row)['source_valid_release'])

    def test_single_source_check_failure_remains_false(self):
        row=self.single_record();row['revision']['final_check_issues']=['source mismatch']
        self.assertIs(self.record(row)['source_valid_release'],False)

    def test_full_pair_scores_all_first_attempts_and_keeps_null_credit_separate(self):
        report=self.f.run();s=report['arms']['ordinary_single_agent'];t=report['arms']['ordinary_multi_agent']
        self.assertEqual((s['questions'],t['questions']),(300,300))
        self.assertEqual((s['correct'],t['correct']),(299,298))
        self.assertEqual(t['terminal_error'],1);self.assertEqual(t['protocol_not_recorded'],1)
        self.assertEqual(t['provider_usage_unknown'],1)
        self.assertEqual(t['null_credit_outside_clean_release'],1)
        self.assertEqual(t['empty_reference_credit_outside_clean_release'],1)
        self.assertEqual(t['source_valid_correct_release'],297)
        self.assertEqual(t['source_valid_typed_wrong_release'],1)
        self.assertEqual(report['paired']['paired_win_source_valid'],1)
        self.assertEqual(report['paired']['paired_loss_mechanism_counts'],
            {'source_valid_typed_wrong_semantic_or_scoring_mismatch':1,'terminal_timeout':1})
        self.assertEqual(t['plan_visible'],298);self.assertEqual(t['plan_ignored'],1)
        self.assertEqual(t['plan_hash_verified'],298)
        self.assertEqual(t['phases']['planner']['missing'],1)
        self.assertEqual(t['usage']['tokens']['n'],300)
        self.assertEqual(t['usage']['known_usage_tokens']['n'],299)
        self.assertEqual(len(report['raw_inputs']['ordinary_multi_agent']['audit_artifacts']),300)
        self.assertTrue(self.f.marker.exists())
        self.assertTrue((self.f.root/'diagnostics/mechanisms_rows.json').is_file())

    def test_terminal_confirmation_is_required_before_any_input_read(self):
        with patch.object(m,'sha256',side_effect=AssertionError('read before confirmation')):
            with self.assertRaisesRegex(ValueError,'confirm both'):
                m.run(self.f.root,self.f.proof_path,'0'*64,self.f.root/'diagnostics',False)

    def test_bad_proof_hash_rejected(self):
        with self.assertRaisesRegex(ValueError,'proof|report digest'):self.f.run(proof_digest='0'*64)
        self.assertFalse(self.f.marker.exists())

    def test_failed_completion_proof_rejected(self):
        self.f.proof['status']='FAIL';self.rejected_before_scoring('did not pass')

    def test_journal_limitations_still_accept_full_integrity(self):
        self.f.proof['status']='PASS_WITH_JOURNAL_LIMITATIONS'
        self.assertEqual(self.f.run()['completion_report']['status'],'PASS_WITH_JOURNAL_LIMITATIONS')

    def test_missing_global_binding_rejected(self):
        del self.f.proof['ordered_input_hash'];self.rejected_before_scoring('digest bindings')

    def test_incomplete_second_arm_cannot_load_scorer_after_first_arm_valid(self):
        self.f.rows['ordinary_multi_agent'].pop();self.f.save_arm('ordinary_multi_agent')
        self.rejected_before_scoring('complete 300')

    def test_reordered_cohort_rejected(self):
        rows=self.f.rows['ordinary_multi_agent'];rows[4],rows[5]=rows[5],rows[4]
        self.f.save_arm('ordinary_multi_agent');self.rejected_before_scoring('order differs')

    def test_duplicate_question_rejected(self):
        rows=self.f.rows['ordinary_multi_agent'];rows[5]['question_id']=rows[4]['question_id']
        self.f.save_arm('ordinary_multi_agent');self.rejected_before_scoring('order differs')

    def test_histories_forbidden(self):
        self.f.rows['ordinary_multi_agent'][0]['attempt_history']=[{'error':'first failed'}]
        self.f.save_arm('ordinary_multi_agent');self.rejected_before_scoring('histories')

    def test_unattempted_nonterminal_row_rejected(self):
        self.f.rows['ordinary_multi_agent'][9]['latency']=None
        self.f.save_arm('ordinary_multi_agent');self.rejected_before_scoring('unattempted')

    def test_terminal_missing_protocol_phases_is_a_diagnostic_gap(self):
        report=self.f.run();row=json.loads((self.f.root/'diagnostics/mechanisms_rows.json').read_text())[303]
        self.assertTrue(row['terminal_error']);self.assertFalse(row['correct']);self.assertFalse(row['source_valid_release'])
        self.assertIn('phase_records_not_recorded',row['accounting_anomalies'])
        self.assertEqual(report['arms']['ordinary_multi_agent']['protocol_not_recorded'],1)

    def test_frozen_source_drift_rejected(self):
        (self.f.root/'agent/single_agent.py').write_text('# drift\n');self.rejected_before_scoring('Frozen source differs')

    def test_archive_drift_rejected(self):
        with (self.f.record/'source.zip').open('ab') as stream:stream.write(b'drift')
        self.rejected_before_scoring('archive differs')

    def test_design_drift_rejected(self):
        (self.f.record/'design.txt').write_text('changed');self.rejected_before_scoring('design differs')

    def test_raw_digest_drift_rejected(self):
        with (self.f.out/'ordinary_multi_agent.json').open('a') as stream:stream.write(' ')
        self.rejected_before_scoring('raw digest differs')

    def test_identity_binding_rejected(self):
        self.f.identities['ordinary_multi_agent']['agent_strategy']='multi_agent_v3'
        self.f.save_arm('ordinary_multi_agent');self.rejected_before_scoring('identity differs')

    def test_bad_owned_audit_rejected_before_scoring(self):
        self.f.rows['ordinary_multi_agent'][4]['audit_artifacts']['question_id']='other'
        self.f.save_arm('ordinary_multi_agent');self.rejected_before_scoring('another question')

    def test_corrupt_gzip_rejected_before_scoring(self):
        p=self.f.out/self.f.rows['ordinary_multi_agent'][4]['audit_artifacts']['path'];p.write_bytes(b'corrupt')
        with self.assertRaises((ValueError,OSError,EOFError)):self.f.run()
        self.assertFalse(self.f.marker.exists())

    def test_plan_hash_mismatch_is_reported_without_deleting_the_attempt(self):
        row=self.f.rows['ordinary_multi_agent'][4];row['revision']['plan_sha256']='e'*64
        self.f.save_arm('ordinary_multi_agent');t=self.f.run()['arms']['ordinary_multi_agent']
        self.assertEqual(t['questions'],300);self.assertEqual(t['correct'],298)
        self.assertEqual(t['accounting_anomalies']['plan_hash_mismatch'],1)

    def test_overshoot_and_fatal_are_findings_not_missing_questions_or_clean_releases(self):
        row=self.f.rows['ordinary_multi_agent'][7];row['revision']['phases'][0].update(tokens=10001,fatal=True)
        row['tokens']['total']=10021
        self.f.save_arm('ordinary_multi_agent');t=self.f.run()['arms']['ordinary_multi_agent']
        self.assertEqual(t['questions'],300);self.assertEqual(t['phase_fatal'],1)
        self.assertEqual(t['accounting_overshoot'],1);self.assertEqual(t['source_valid_correct_release'],296)
        self.assertEqual(t['accounting_anomalies']['phase_cap_exceeded:planner:tokens'],1)

    def test_runtime_protocol_mismatch_rejected(self):
        self.f.rows['ordinary_multi_agent'][7]['revision']['protocol']='query-contract-v3.2-static-audit'
        self.f.save_arm('ordinary_multi_agent');self.rejected_before_scoring('unexpected protocol')

    def test_host_automatic_checks_are_not_model_replies_or_requested_tools(self):
        trace=[{'role':'system','content':'Planner'}, {'role':'system','content':'Solver'},
               {'role':'user','content':'Solver allocation remaining: about 100 tokens and 28 model-requested tools.'},
               {'role':'assistant','content':None,'tool_calls':[{'id':'automatic_field_check_1','type':'function',
                    'function':{'name':'check_fields','arguments':'{}'}}]},
               {'role':'tool','name':'check_fields','tool_call_id':'automatic_field_check_1','content':'{}'},
               {'role':'assistant','content':None,'tool_calls':[{'id':'real-model-tool','type':'function',
                    'function':{'name':'execute_python_code','arguments':'{}'}}]},
               {'role':'assistant','content':'Host boundary','phase':'solver_boundary','model_call':False},
               {'role':'user','content':'Solver allocation remaining: about 50 tokens and 27 model-requested tools.'}]
        stats=m.model_trace_stats(trace)
        self.assertEqual(stats['solver_model_responses'],1)
        self.assertEqual(stats['model_requested_tool_calls'],1)
        self.assertEqual(stats['model_requested_execute_python_code'],1)
        self.assertNotIn('model_requested_check_fields',stats)
        self.assertEqual(stats['host_generated_source_check_assistant_messages'],1)
        self.assertEqual(stats['automatic_source_check_calls'],1)
        self.assertEqual(stats['automatic_source_check_replies'],1)
        self.assertEqual(stats['source_check_replies'],1)
        self.assertEqual(stats['host_final_boundaries'],1)
        self.assertEqual(stats['solver_invocation_attempt_signals'],2)
        self.assertFalse(any('sdk_send' in key for key in stats))
        row=self.f.audit.restore_result_audit(self.f.rows['ordinary_multi_agent'][4],self.f.out,
                                             question_id=self.f.ids[4],restore_history=False)
        row['trace']=trace;row['budget']['tool_calls_run']=2
        score=lambda answer,qid:{'typed_correct':True,'typed_reason':'synthetic','answer_type':'number'}
        record=m.row_record(row,self.f.ids[4],{'family':'synthetic'},score,True)
        self.assertEqual(record['tool_calls'],2)

    def test_valid_plan_declared_visible_but_planner_fatal_before_solver(self):
        revision=deepcopy(self.f.rows['ordinary_multi_agent'][4]['revision'])
        revision['phases']=revision['phases'][:1];revision['phases'][0]['fatal']=True
        trace=[{'role':'system','content':'Planner'}, {'role':'assistant','content':'validated plan'}]
        status,anomalies=m.plan_status(revision,trace)
        self.assertEqual(status['plan_status'],'declared_visible_but_fatal_before_solver')
        self.assertTrue(status['plan_visible']);self.assertTrue(status['plan_hash_verified'])
        self.assertFalse(status['plan_advice_context_observed']);self.assertEqual(anomalies,[])
        revision['phases'][0]['fatal']=False
        status,anomalies=m.plan_status(revision,trace)
        self.assertEqual(status['plan_status'],'visible_valid_hash')
        self.assertIn('visible_plan_context_not_observed',anomalies)

    def test_no_utils_bootstrap_from_public_file_scorer(self):
        self.f.run();self.assertTrue(self.f.marker.exists())


if __name__=='__main__':unittest.main(verbosity=2)
