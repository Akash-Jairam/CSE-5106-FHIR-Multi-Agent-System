"""Synthetic mechanisms and gate composition; no real run/results are read.

The guard double isolates this helper's trust-boundary wiring. The actual V5
integrity guard has its own independent protocol/source tests; this fixture does
not substitute a fake guard in production. Real public audit/scorer code is used
with 600 freshly generated, question-owned gzip audits.
"""
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
import zipfile

HELPER=Path('/private/tmp/fhir_v5_critic_mechanisms.py')
spec=importlib.util.spec_from_file_location('v5_mechanisms_under_test',HELPER)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
PUBLIC=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
REAL_MODULE=m.module


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False)+'\n')


def candidate(answer):
    return {'answer':answer,'evidence':[],'contributing_ids':[],'operation':'synthetic operation','limitation':None}


def row(primary=1,final=2,accepted=True):
    p,f=candidate(primary),candidate(final)
    c={'status':'released' if accepted else 'primary_preserved','model_calls':2,'tool_calls':3 if accepted else 0,
       'first':{'decision':'INSPECT'},'proposal':{'decision':'PATCH'} if accepted else {'decision':'KEEP'},
       'probe_output':{'answer':{}},'validated_probe':{},'alternate_execution':deepcopy(f),'alternate':deepcopy(f)}
    return {'question_id':'synthetic','agent_answer':final,'evidence':[],'limitation':None,'error':None,'latency':1,
      'tokens':{'total':30,'llm_calls':3,'per_call':[{'usage_available':True}]},
      'budget':{'tool_calls_run':5,'stopped':None},'trace':[],
      'revision':{'protocol':m.PROTOCOL,'draft':candidate(primary),'primary':p,'final':f,
        'primary_revision':{'final_check_issues':[]},'final_check_issues':[],
        'release_policy':'executed_source_checked_critic_proposal' if accepted else 'source_valid_primary_preserved',
        'critic_revised':accepted,'solver_source_revised':False,'critic':c,'unresolved_scope':{},
        'phases':[{'phase':'solver','tokens':20,'token_cap':175000,'tool_calls':2,'tool_cap':27,'fatal':False,'errors':[]},
                  {'phase':'critic','tokens':10,'token_cap':25000,'tool_calls':3 if accepted else 0,'tool_cap':3,'fatal':False,'errors':[]}]}}


def exact(reference):
    return lambda answer:{'typed_correct':m.same(answer,reference),'answer_type':'empty' if reference is None else 'number','typed_reason':'synthetic'}


class MechanismTests(unittest.TestCase):
    def test_helpful_harmful_and_neutral_accepted_edits(self):
        for p,f,expected in [(1,2,'helpful'),(2,1,'harmful'),(2,2,'neutral'),(0,1,'neutral')]:
            r=m.row_record(row(p,f),exact(2));self.assertEqual(r['critic_semantic_repair_effect'],expected)
            self.assertEqual(r['critic_stage_effect'],expected)

    def test_accounting_failure_loses_primary_credit_in_total_stage_only(self):
        value=row(2,None,False);value['error']='Critic aggregate accounting failed'
        rev=value['revision'];rev['release_policy']='aggregate_accounting_failed';rev['critic']['status']='accounting_failed'
        rev['phases'][1]['fatal']=True;value['tokens']['per_call']=[{'usage_available':False}]
        r=m.row_record(value,exact(2))
        self.assertEqual(r['critic_stage_effect'],'harmful')
        self.assertEqual(r['critic_semantic_repair_effect'],'not_applied')
        self.assertTrue(r['critic_stage_nonrepair_credit_lost']);self.assertFalse(r['correct'])

    def test_terminal_error_overrides_typed_null_credit(self):
        value=row(None,None,False);value['error']='unknown billed usage'
        value['revision']['release_policy']='aggregate_accounting_failed';value['revision']['critic']['status']='accounting_failed'
        value['revision']['phases'][1]['fatal']=True
        r=m.row_record(value,exact(None));self.assertFalse(r['correct']);self.assertFalse(r['typed_null_credit'])
        self.assertEqual(r['critic_stage_effect'],'harmful')

    def test_missing_primary_is_unavailable_and_actual_null_is_scored(self):
        value=row(None,None,False);r=m.row_record(value,exact(None))
        self.assertTrue(r['primary_is_actual_null']);self.assertIs(r['primary_typed_correct'],True)
        value['revision']['primary']=None;r=m.row_record(value,exact(None))
        self.assertIsNone(r['primary_typed_correct']);self.assertFalse(r['primary_is_actual_null'])
        self.assertEqual(r['critic_stage_effect'],'not_applied')
        value['revision'].pop('draft');r=m.row_record(value,exact(None));self.assertIsNone(r['draft_typed_correct'])

    def test_source_revision_and_critic_revision_are_separate(self):
        value=row(2,1);value['revision']['draft']=candidate(1);value['revision']['solver_source_revised']=True
        r=m.row_record(value,exact(2));self.assertEqual(r['solver_source_revision_effect'],'helpful')
        self.assertEqual(r['critic_semantic_repair_effect'],'harmful')

    def test_typed_false_and_integer_are_not_conflated(self):
        r=m.row_record(row(True,1),exact(1));self.assertEqual(r['critic_semantic_repair_effect'],'helpful')

    def test_phase_local_refusal_without_global_stop_is_counted(self):
        value=row(2,2,False);value['revision']['phases'][1]['errors']=['critic known-budget refusal before model call']
        r=m.row_record(value,exact(2));self.assertTrue(r['phase_local_refusal']);self.assertTrue(r['local_refusal_without_aggregate_stop'])
        self.assertFalse(r['aggregate_budget_stop']);self.assertFalse(r['recorded_unknown_provider_usage'])

    def test_provider_failure_and_completion_overshoot_are_not_local_refusals(self):
        value=row(2,None,False);value['revision']['phases'][1]['errors']=['provider usage unavailable','critic reported completion exceeded requested subcap']
        r=m.row_record(value,exact(2));self.assertTrue(r['phase_recorded_error']);self.assertFalse(r['phase_local_refusal'])
        self.assertTrue(r['critic_completion_subcap_overshoot'])

    def test_missing_usage_and_recorded_unknown_usage_remain_distinct(self):
        value=row(2,2,False);value['tokens']['per_call']=[]
        r=m.row_record(value,exact(2));self.assertTrue(r['usage_metadata_unavailable']);self.assertFalse(r['recorded_unknown_provider_usage'])
        value['tokens']['per_call']=[{}];r=m.row_record(value,exact(2));self.assertTrue(r['recorded_unknown_provider_usage'])

    def test_null_credit_outside_clean_release_is_not_source_success(self):
        value=row(None,None,False);value['revision']['release_policy']='primary_failed_execution_or_source_contract'
        value['revision']['final_check_issues']=['scope issue']
        r=m.row_record(value,exact(None));self.assertTrue(r['correct']);self.assertFalse(r['source_valid_release'])
        self.assertTrue(r['typed_null_credit_outside_source_valid_release'])

    def test_probe_and_patch_failures_are_distinct_from_accepted_repairs(self):
        for output,validated,expected in [({'error':'failed'},False,'probe_failed'),({'answer':{}},False,'probe_unusable')]:
            value=row(2,2,False);c=value['revision']['critic'];c.pop('proposal');c['probe_output']=output;c.pop('validated_probe')
            self.assertEqual(m.row_record(value,exact(2))['critic_stage'],expected)
        value=row(2,2,False);c=value['revision']['critic'];c['proposal']={'decision':'PATCH'};c['alternate_execution']={'error':'bad metadata'}
        self.assertEqual(m.row_record(value,exact(2))['critic_stage'],'candidate_execution_failed')

    def test_all300_denominator_includes_early_terminal_rows(self):
        records=[m.row_record(row(),exact(2)) for _ in range(299)]
        records.append(m.row_record({'question_id':'early','agent_answer':None,'error':'early crash'},exact(None)))
        summary=m.summarize(records);self.assertEqual(summary['questions'],300);self.assertEqual(summary['correct'],299)
        self.assertEqual(summary['critic_stage_denominator'],299);self.assertEqual(summary['terminal_error'],1)


class Fixture:
    def __init__(self,root):
        self.root=Path(root).resolve();self.record=self.root/'benchmarks'/m.EXPERIMENT;self.output=self.root/m.RUN_RELATIVE
        self.record.mkdir(parents=True);self.output.mkdir(parents=True);self.marker=self.root/'SCORER_IMPORTED';self.restored=0
        source={'utils/answer_scorer.py':"from pathlib import Path\nPath("+repr(str(self.marker))+").write_text('imported')\n"+(PUBLIC/'utils/answer_scorer.py').read_text(),
          'scripts/audit_store.py':(PUBLIC/'scripts/audit_store.py').read_text(),
          'scripts/compare_runs.py':'# synthetic frozen comparison source\n'}
        self.ids=['synthetic-%03d'%i for i in range(300)]
        dataset=self.root/'final_dataset/questions_answers_sql_fhir.csv';dataset.parent.mkdir()
        with dataset.open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=['question_id','question','true_answer','assumption','patient_fhir_id']);w.writeheader()
            for qid in self.ids:w.writerow({'question_id':qid,'question':'Return the maximum synthetic value.','true_answer':'[[2]]','assumption':'','patient_fhir_id':'synthetic-patient'})
        families=self.root/'manifests/question_families.csv';families.parent.mkdir()
        with families.open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=['question_id','family','template']);w.writeheader()
            for qid in self.ids:w.writerow({'question_id':qid,'family':'synthetic','template':'Return the maximum value.'})
        source['manifests/question_families.csv']=families.read_text()
        for rel,text in source.items():p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
        self.mapping={rel:m.sha(self.root/rel) for rel in source}
        write(self.root/'manifests/split.json',{'splits':{'held_out':{'question_ids':self.ids}}})
        with zipfile.ZipFile(self.record/'source.zip','w') as archive:
            for rel in source:archive.write(self.root/rel,rel)
        with zipfile.ZipFile(self.record/'runtime_adapter_sources.zip','w'):pass
        (self.record/'design.txt').write_text('synthetic selected design')
        self.adapter={'archive_sha256':m.sha(self.record/'runtime_adapter_sources.zip')}
        self.binding={'synthetic_profile_sha256':'b'*64}
        self.manifest={'source_sha256':self.mapping,'source_archive_sha256':m.sha(self.record/'source.zip'),
          'design_sha256':m.sha(self.record/'design.txt'),'dataset_sha256':m.sha(dataset),'split_sha256':m.sha(self.root/'manifests/split.json')}
        write(self.record/'manifest.json',self.manifest);write(self.record/'runtime_adapter_manifest.json',self.adapter)
        write(self.record/'expected_public_binding.json',self.binding);write(self.record/'prospective_protocol.json',{})
        self.pins={'record_metadata_sha256':{n:m.sha(self.record/n) for n in ['manifest.json','runtime_adapter_manifest.json','expected_public_binding.json','prospective_protocol.json']},
          'source_inventory_sha256':hashlib.sha256(json.dumps(self.mapping,sort_keys=True).encode()).hexdigest()}
        self.pins_path=self.root/'pins.json';write(self.pins_path,self.pins)
        self.guard_path=self.root/'synthetic_guard.py';self.guard_path.write_text('# synthetic external trust-boundary test double\n')
        self.base=self.root/'synthetic_base.py';self.base.write_text('# synthetic reviewed base\n')
        self.store=REAL_MODULE(self.root/'scripts/audit_store.py','synthetic_v5_audit_store')
        self.rows={};self.sidecars={}
        for arm,strategy in m.ARMS.items():
            rows=[]
            for qid in self.ids:
                r=row() if strategy=='multi_agent_v5' else {'agent_answer':2,'evidence':[],'limitation':None,'error':None,'latency':1,'revision':{}}
                r.update(question_id=qid,patient_fhir_id='synthetic-patient',question='Return the maximum synthetic value.',true_answer='[[2]]',assumption='',
                         trace=[{'role':'system','content':strategy}],retrieved_fhir_resources={'synthetic':[]})
                rows.append(self.store.externalize_result_audit(r,self.output,question_id=qid))
            self.rows[arm]=rows;self.sidecars[arm]={'agent_strategy':strategy,'code_hash':'c'*64};self.save_arm(arm)
        self.proof_path=self.root/'completion.json';self.official_path=self.record/'results.json'
        self.refresh()
        self.guard=SimpleNamespace(BASE_PATH=self.base,BASE_SHA=m.sha(self.base),PROVIDER_SOURCE={},
          equal=m.same,public_path=lambda root,name:root/name,actual_source_inventory=lambda root:set(self.mapping))
        def frozen(root,record,output,pins,digest):
            for rel,value in self.mapping.items():m.require(m.sha(root/rel)==value,'Frozen source changed')
            return self.pins,self.manifest,self.adapter,self.binding
        self.guard.frozen_provenance=frozen
        self.guard.B=SimpleNamespace(code_hash=lambda root:'c'*64,owned_file=self.owned,load_public_store=lambda root:self.store)
        self.guard.verify_arm_snapshot=self.verify_arm
        self.guard.verify_restored=lambda row,output,store:(store.restore_result_audit(row,output,question_id=row['question_id'],restore_history=False),0)

    def owned(self,output,relative):
        p=output/relative;m.require(not Path(relative).is_absolute() and p.resolve().is_relative_to(output) and not p.is_symlink(),'Unsafe owned path');return p

    def save_arm(self,arm):
        write(self.output/(arm+'.json'),self.rows[arm]);write(self.output/(arm+'.json.run.json'),self.sidecars[arm])

    def summary(self,arm):
        raw=self.output/(arm+'.json');rows=self.rows[arm]
        return {'saved_first_attempts':300,'verified_owned_gzip_audits':300,'result_sha256':m.sha(raw),
          'identity_sha256':m.sha(Path(str(raw)+'.run.json')),'bytes':raw.stat().st_size,
          'terminal_errors':sum(m.failed(r) for r in rows),'budget_stops':sum(bool((r.get('budget') or {}).get('stopped')) for r in rows)}

    def refresh(self):
        self.proof={'status':'PASS','mode':'completion','caller_asserted_owned_jobs_terminal':True,'expected_first_attempts':600,
          'restored_audit_records':600,'unique_referenced_artifacts':len({r['audit_artifacts']['path'] for rows in self.rows.values() for r in rows}),'scoring_performed':False,
          'guard_sha256':m.sha(self.guard_path),'pins_sha256':m.sha(self.pins_path),'provider_binding':self.binding,
          'manifest_sha256':m.sha(self.record/'manifest.json'),'source_inventory_sha256':self.pins['source_inventory_sha256'],
          'frozen_source_files_verified':len(self.mapping),'runtime_adapter_files_verified':4,'code_hash':'c'*64,
          'reviewed_base_sha256':m.sha(self.base),**{k:self.manifest[k] for k in ['source_archive_sha256','design_sha256','dataset_sha256','split_sha256']},
          'arms':{a:self.summary(a) for a in m.ARMS},'raw_files':{p.relative_to(self.output).as_posix():{'sha256':m.sha(p),'bytes':p.stat().st_size} for p in self.output.rglob('*') if p.is_file()}}
        self.official={'questions':300,'cohort':'held_out','dataset_sha256':self.manifest['dataset_sha256'],
          'families_sha256':self.mapping['manifests/question_families.csv'],'question_ids_sha256':hashlib.sha256(json.dumps(sorted(self.ids)).encode()).hexdigest(),
          'identities':{},'arms':{}}
        for arm,label in [('ordinary_single_agent','v5_single'),('ordinary_multi_agent','v5_team')]:
            self.official['identities'][label]={'result_path':str(self.output/(arm+'.json')),'result_sha256':self.summary(arm)['result_sha256'],'identity':self.sidecars[arm]}
            self.official['arms'][label]={'accuracy':{'all':{'correct':300}},'terminal_errors':0,'budget_stops':0}
        self.save_proofs()

    def save_proofs(self):write(self.proof_path,self.proof);write(self.official_path,self.official)

    def verify_arm(self,root,output,manifest,expected,code,store,arm,strategy,complete):
        raw=m.load(output/(arm+'.json'));m.require([r['question_id'] for r in raw]==expected and len(raw)==300,'Wrong cohort/order')
        m.require(m.load(output/(arm+'.json.run.json'))['agent_strategy']==strategy,'Wrong strategy')
        for r in raw:
            m.require(not r.get('attempt_history'),'Retries forbidden');m.require(m.failed(r) or r.get('latency') is not None,'Unattempted row')
            m.require(not self.marker.exists(),'Scorer imported before all audits restored')
            store.restore_result_audit(r,output,question_id=r['question_id'],restore_history=False);self.restored+=1
        return self.summary(arm),self.sidecars[arm],'same inputs',{r['audit_artifacts']['path'] for r in raw}

    def args(self):
        return SimpleNamespace(root=str(self.root),completion_report=str(self.proof_path),completion_report_sha256=m.sha(self.proof_path),
          official_report=str(self.official_path),official_report_sha256=m.sha(self.official_path),guard_module=str(self.guard_path),guard_sha256=m.sha(self.guard_path),
          guard_pins=str(self.pins_path),guard_pins_sha256=m.sha(self.pins_path),confirm_both_owned_jobs_terminal=True,
          output=str(self.record/'diagnostics'),single_label='v5_single',team_label='v5_team')

    def patch(self):return patch.object(m,'module',side_effect=lambda path,name:self.guard if Path(path)==self.guard_path else REAL_MODULE(path,name))


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='v5-mechanism-synthetic-');self.addCleanup(self.temp.cleanup)
        self.f=Fixture(self.temp.name)

    def rejected(self,mutate):
        mutate();args=self.f.args()
        with self.f.patch(),self.assertRaises((ValueError,EOFError,OSError)):
            m.run(args)
        self.assertFalse(self.f.marker.exists());self.assertFalse(Path(args.output).exists())

    def test_complete600_restored_before_scorer_import_and_totals_reconciled(self):
        with self.f.patch():report=m.run(self.f.args())
        self.assertEqual(self.f.restored,600);self.assertTrue(self.f.marker.exists())
        self.assertEqual(report['summary']['questions'],300);self.assertEqual(report['summary']['correct'],300)
        self.assertEqual(report['summary']['critic_semantic_repair_effect_counts']['helpful'],300)


    def test_valid_content_addressed_cross_arm_dedup_keeps600_restorations(self):
        old={self.f.rows[arm][0]['audit_artifacts']['path'] for arm in m.ARMS}
        for arm in m.ARMS:
            value={'question_id':self.f.ids[0],'patient_fhir_id':'synthetic-patient',
                'question':'Return the maximum synthetic value.','true_answer':'[[2]]','assumption':'',
                'agent_answer':None,'evidence':[],'limitation':None,'error':'synthetic terminal failure',
                'latency':1,'trace':[],'retrieved_fhir_resources':{},'revision':None}
            self.f.rows[arm][0]=self.f.store.externalize_result_audit(value,self.f.output,question_id=self.f.ids[0])
            self.f.save_arm(arm)
        referenced={r['audit_artifacts']['path'] for rows in self.f.rows.values() for r in rows}
        for relative in old-referenced:(self.f.output/relative).unlink()
        self.f.refresh()
        for label in ('v5_single','v5_team'):
            self.f.official['arms'][label]['accuracy']['all']['correct']=299
            self.f.official['arms'][label]['terminal_errors']=1
        self.f.save_proofs()
        self.assertEqual(self.f.proof['unique_referenced_artifacts'],599)
        with self.f.patch():report=m.run(self.f.args())
        self.assertEqual(self.f.restored,600);self.assertTrue(self.f.marker.exists())
        self.assertEqual(report['summary']['questions'],300)
        self.assertEqual(report['summary']['terminal_error'],1)

    def test_inconsistent_pinned_unique_count_blocks_scorer(self):
        self.f.proof['unique_referenced_artifacts']=599;self.f.save_proofs()
        self.rejected(lambda:None)

    def test_destination_blocks_raw_source_existing_escape_and_symlink_before_reads(self):
        link=self.f.root/'linked';link.symlink_to(self.f.record,target_is_directory=True)
        for path in (self.f.output/'diagnostics',self.f.root/'agent/new-diagnostics',
                     self.f.record,link/'diagnostics',self.f.record/'..'/'escape-diagnostics'):
            args=self.f.args();args.output=str(path)
            with self.subTest(path=path),patch.object(m,'verify_all',side_effect=AssertionError('must reject before reads')),self.assertRaises(ValueError):
                m.run(args)
        self.assertFalse((self.f.output/'diagnostics').exists())

    def test_formal_destination_is_allowed_and_fresh(self):
        args=self.f.args();self.assertEqual(m.diagnostics_destination(args),self.f.record/'diagnostics')
        Path(args.output).mkdir()
        with self.assertRaisesRegex(ValueError,'Fresh'):m.diagnostics_destination(args)

    def test_external_private_temporary_destination_allowed(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp',prefix='v5-diagnostics-output-') as temporary:
            args=self.f.args();args.output=str(Path(temporary)/'fresh')
            self.assertEqual(m.diagnostics_destination(args),Path(args.output))

    def test_terminal_assertion_rejected_before_any_read(self):
        args=SimpleNamespace(confirm_both_owned_jobs_terminal=False)
        with patch.object(m,'sha',side_effect=AssertionError('must not read')),self.assertRaisesRegex(ValueError,'terminal confirmation'):
            m.verify_all(args)

    def test_proof_partial_initial_or_not_terminal_blocks_scorer(self):
        for key,value in [('restored_audit_records',599),('mode','initial'),('caller_asserted_owned_jobs_terminal',False)]:
            original=self.f.proof[key];self.f.proof[key]=value;self.f.save_proofs()
            self.rejected(lambda:None);self.f.proof[key]=original;self.f.save_proofs()

    def test_external_digest_mismatch_blocks_guard_import(self):
        args=self.f.args();args.completion_report_sha256='0'*64
        with patch.object(m,'module',side_effect=AssertionError('must not import')),self.assertRaisesRegex(ValueError,'Pinned file'):
            m.verify_all(args)

    def test_frozen_scorer_or_compare_source_tamper_blocks_import(self):
        for relative in ['utils/answer_scorer.py','scripts/compare_runs.py']:
            p=self.f.root/relative;original=p.read_text();self.rejected(lambda:p.write_text(original+'\n# changed'))
            p.write_text(original)

    def test_raw_file_digest_tamper_blocks_import(self):
        self.rejected(lambda:(self.f.output/'ordinary_multi_agent.json').write_text('[]'))

    def test_identity_strategy_tamper_even_with_rehashed_proof_blocks_import(self):
        def mutate():
            self.f.sidecars['ordinary_multi_agent']['agent_strategy']='other';self.f.save_arm('ordinary_multi_agent');self.f.refresh()
        self.rejected(mutate)

    def test_corrupt_gzip_with_rehashed_compressed_pin_still_blocks_import(self):
        def mutate():
            path=self.f.output/self.f.rows['ordinary_multi_agent'][0]['audit_artifacts']['path'];path.write_bytes(b'not gzip');self.f.refresh()
        self.rejected(mutate)

    def test_missing_first_attempt_or_retry_history_blocks_import(self):
        arm='ordinary_multi_agent';r=self.f.rows[arm][0]
        r['latency']=None;self.f.save_arm(arm);self.f.refresh();self.rejected(lambda:None)
        r['latency']=1;r['attempt_history']=[{'error':'earlier'}];self.f.save_arm(arm);self.f.refresh();self.rejected(lambda:None)

    def test_official_raw_binding_mismatch_blocks_import(self):
        self.f.official['identities']['v5_team']['result_sha256']='0'*64;self.f.save_proofs();self.rejected(lambda:None)

    def test_official_total_disagreement_writes_no_diagnostics(self):
        self.f.official['arms']['v5_team']['accuracy']['all']['correct']=299;self.f.save_proofs()
        with self.f.patch(),self.assertRaisesRegex(ValueError,'Typed totals disagree'):m.run(self.f.args())
        self.assertTrue(self.f.marker.exists());self.assertFalse((self.f.record/'diagnostics').exists())

    def test_postgate_source_change_is_rechecked_before_scorer_import(self):
        with self.f.patch():prepared=m.verify_all(self.f.args())
        p=self.f.root/'utils/answer_scorer.py';p.write_text(p.read_text()+'\n# changed after gate')
        with self.assertRaisesRegex(ValueError,'before scorer import'):m.analyze(prepared)
        self.assertFalse(self.f.marker.exists())


if __name__=='__main__':unittest.main(verbosity=2)
