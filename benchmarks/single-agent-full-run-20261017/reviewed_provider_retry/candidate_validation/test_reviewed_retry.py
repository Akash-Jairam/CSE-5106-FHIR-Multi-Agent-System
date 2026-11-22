"""Targeted offline metadata tests for the two-question reviewed retry boundary."""
from collections import Counter
from copy import deepcopy
import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
candidate=load('reviewed_retry_candidate',HERE/'v3_full_run.py');candidate.ROOT=ROOT
with patch.dict(sys.modules,{'scripts.v3_full_run':candidate}):
 monitor=load('reviewed_retry_monitor',HERE/'monitor_single_agent.py')
A,B=candidate.PROVIDER_REJECTION_POLICY['reviewed_question_ids']
def decision(row,audit_dir=None):
 return {'status':'runtime','category':'provider_policy_rejection','reason':'provider_policy_rejection'} if row.get('error')=='fixture-runtime' else {'status':'accepted','reason':'runtime_clean_terminal'}
def manifest():return {'selected_arms':['single_agent'],'provider_rejection_policy':deepcopy(candidate.PROVIDER_REJECTION_POLICY)}

class ReviewedPolicyTests(unittest.TestCase):
 def test_exact_authorized_scope_and_boundary(self):
  m=manifest()
  for qid in (A,B):
   for number in (1,2,31):self.assertEqual(candidate.provider_rejection_limit(m,qid,number),2)
   for number in (32,33,None):self.assertEqual(candidate.provider_rejection_limit(m,qid,number),3)
  for number in (1,31,32,33,None):self.assertEqual(candidate.provider_rejection_limit(m,'unreviewed',number),2)
 def test_policy_or_scope_mutation_refused(self):
  for key,value in (('max_rejections_per_question',4),('first_reviewed_round',31),('reviewed_question_ids',[A]),('retry_request','rewrite')):
   m=manifest();m['provider_rejection_policy'][key]=value
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Selected provider-rejection policy'):
    candidate.provider_rejection_limit(m,A,32)
  m=manifest();m['selected_arms']=['direct_llm']
  with self.assertRaises(ValueError):candidate.provider_rejection_limit(m,A,32)
 def test_current_parking_preserves_counts_and_only_unparks_named_two(self):
  with tempfile.TemporaryDirectory(dir=HERE) as directory:
   accepted={arm:{} for arm in candidate.ARMS};counts={arm:{} for arm in candidate.ARMS}
   counts['single_agent']={A:2,B:2,'unreviewed':2}
   original=deepcopy(counts)
   parked,document=candidate.persist_provider_rejections(Path(directory),manifest(),accepted,counts,('single_agent',))
   self.assertEqual(parked['single_agent'],{'unreviewed'});self.assertEqual(counts,original)
   self.assertEqual(document['rejections']['single_agent'],original['single_agent'])
   counts['single_agent'][A]=3
   parked,_=candidate.persist_provider_rejections(Path(directory),manifest(),accepted,counts,('single_agent',))
   self.assertEqual(parked['single_agent'],{A,'unreviewed'})

class ReplayTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=HERE);self.addCleanup(self.temp.cleanup);self.run=Path(self.temp.name)
  self.patients={A:'fixture-patient-a',B:'fixture-patient-b','already-clean':'fixture-patient-c','unreviewed':'fixture-patient-d'}
  self.m=manifest()
 def batch(self,number,ids,clean=()):
  directory=self.run/'attempts/single_agent';directory.mkdir(parents=True,exist_ok=True)
  path=directory/f'round-{number:04d}.batch.json';output=directory/f'round-{number:04d}.json';source=directory/f'round-{number:04d}.csv'
  source.write_text('question_id\n'+'\n'.join(ids)+'\n')
  rows=[{'question_id':qid,'patient_fhir_id':self.patients[qid],'latency':1.0,'error':None if qid in clean else 'fixture-runtime','audit_artifacts':{'fixture':True}} for qid in ids]
  batch={'round':number,'arm':'single_agent','question_ids':list(ids),'input':str(source),'input_sha256':candidate.file_sha256(source),'output':str(output),'run_identity':{'fixture':'metadata-only'}}
  candidate.atomic_json(path,batch);candidate.atomic_json(output,rows);candidate.atomic_json(Path(str(output)+'.run.json'),batch['run_identity'])
  candidate.atomic_json(path.with_suffix('.seal.json'),{'batch_sha256':candidate.file_sha256(path),'result_sha256':candidate.file_sha256(output),'identity_sha256':candidate.file_sha256(Path(str(output)+'.run.json'))})
  return path
 def prefix(self,qid=A):
  self.batch(1,(qid,));self.batch(2,(qid,))
 def collect(self):
  import run_agent
  with patch.object(run_agent,'is_attempted',return_value=True),patch.object(candidate,'copy_audit'):
   return candidate.collect_accepted(self.run,self.m,self.patients,classifier=decision,arms=('single_agent',))
 def replay_monitor(self):
  def rows(run,m,path,patients):
   batch=json.loads(path.read_text());return batch,json.loads(Path(batch['output']).read_text()),batch['run_identity']
  with patch.object(monitor,'cohort',return_value=self.patients),patch.object(monitor,'identities_and_rows',side_effect=rows),patch.object(monitor,'classifier_for_manifest',return_value=decision),patch.object(monitor,'restore_result_audit'):
   return monitor.replay(self.run,self.m)
 def test_reviewed_third_clean_at32_accepted_and_earlier_clean_preserved(self):
  self.prefix();self.batch(3,('already-clean',),clean=('already-clean',));self.batch(32,(A,),clean=(A,))
  accepted,choices,retries,attention,counts=self.collect()
  self.assertEqual(set(accepted['single_agent']),{A,'already-clean'});self.assertEqual(choices['single_agent'][A]['runtime_failures_before'],2)
  self.assertEqual(choices['single_agent'][A]['round'],32);self.assertEqual(choices['single_agent']['already-clean']['round'],3)
  self.assertEqual(counts['single_agent']['raw_attempts'],4);self.assertEqual(counts['single_agent']['runtime_failures'],2);self.assertEqual(attention,[])
  ma,mc,_,_,_=self.replay_monitor();self.assertEqual(set(ma),{A,'already-clean'});self.assertEqual(mc,choices['single_agent'])
 def test_historical_third_clean_before32_still_refused(self):
  self.prefix();self.batch(31,(A,),clean=(A,))
  with self.assertRaisesRegex(ValueError,'parked question'):self.collect()
  with self.assertRaisesRegex(ValueError,'Historical batch exceeded'):self.replay_monitor()
 def test_unreviewed_third_clean_at32_refused(self):
  self.prefix('unreviewed');self.batch(32,('unreviewed',),clean=('unreviewed',))
  with self.assertRaises(ValueError):self.collect()
  with self.assertRaises(ValueError):self.replay_monitor()
 def test_third_rejection_retained_counts_and_fourth_clean_refused(self):
  self.prefix();self.batch(32,(A,))
  accepted,choices,retries,attention,counts=self.collect()
  self.assertEqual(accepted['single_agent'],{});self.assertEqual(retries['single_agent'][A],3)
  self.assertEqual(counts['single_agent']['runtime_failures'],3);self.assertEqual(attention,[])
  _,_,_,_,summaries=self.replay_monitor();self.assertEqual(sum(s['provider_rejections'].get(A,0) for s in summaries),3)
  self.batch(33,(A,),clean=(A,))
  with self.assertRaises(ValueError):self.collect()
  with self.assertRaises(ValueError):self.replay_monitor()
 def test_accepted_third_cannot_be_rerun(self):
  self.prefix();self.batch(32,(A,),clean=(A,));self.batch(33,(A,),clean=(A,))
  with self.assertRaisesRegex(ValueError,'accepted question was rerun'):self.collect()
  with self.assertRaisesRegex(ValueError,'Accepted question scheduled again'):self.replay_monitor()
 def test_monitor_active_round_guard_uses_historical_limit(self):
  m=manifest();m.update(questions_per_arm=3,run_identities={'single_agent':{}})
  for number,should_pass in ((31,False),(32,True)):
   state={'attention':False,'selected_scope_complete':False,'active':{'single_agent':{'round':number}},'analysis_status':'not_started'}
   ledger={'arms':{'single_agent':{}},'counts':{'single_agent':{}}}
   derived={'policy':m['provider_rejection_policy'],'source':'reconstructed_from_sealed_immutable_attempts','rejections':{'single_agent':{A:2}},'parked_question_ids':{'single_agent':[]}}
   def read(path):
    name=Path(path).name
    if name=='run_manifest.json':return m
    if name=='supervisor_state.json':return state
    if name=='acceptance.json':return ledger
    if name=='provider_rejections.json':return derived
    if name=='full_single_agent.json':return []
    if name=='full_single_agent.json.run.json':return {}
    if name.endswith('.batch.json'):return {'round':number,'question_ids':[A]}
    raise AssertionError(name)
   args=SimpleNamespace(round=None,clean=None,runtime=None,accepted=None,final_arm=False)
   with patch.object(monitor,'read',side_effect=read),patch.object(monitor,'scope_state'),patch.object(monitor,'preservation',return_value={}),patch.object(monitor,'replay',return_value=({}, {}, {}, {}, [{'provider_rejections':{A:2}}])),patch.object(monitor,'terminal_proof',return_value=False),patch.object(monitor,'native_status',return_value={}):
    if should_pass:self.assertEqual(monitor.verify(self.run,self.run,args)['status'],'SELECTED_SCOPE_IN_PROGRESS')
    else:
     with self.assertRaisesRegex(ValueError,'Persistently rejected request scheduled'):monitor.verify(self.run,self.run,args)

if __name__=='__main__':unittest.main(verbosity=2)
