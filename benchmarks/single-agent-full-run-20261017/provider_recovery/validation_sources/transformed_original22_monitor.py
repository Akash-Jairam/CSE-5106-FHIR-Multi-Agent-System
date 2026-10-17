"""Private synthetic tests; no inference or live benchmark reads/writes."""
import argparse
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from scripts.audit_store import externalize_result_audit
spec=importlib.util.spec_from_file_location('single_monitor','/private/tmp/v3-paired-provider-amendment/monitor_single_agent.py'); monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
root=Path(tempfile.mkdtemp(prefix='single-monitor-fixtures-'));results=[]
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2))
def csvfile(path,ids):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=('question_id','patient_fhir_id'));w.writeheader();w.writerows({'question_id':qid,'patient_fhir_id':'p'+qid[1:]} for qid in ids)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def fixture(name,terminal=False):
 run=root/name;run.mkdir();snapshot=run/monitor.AMENDMENT;snapshot.mkdir(parents=True)
 full=run/'full-input.csv';csvfile(full,['q0','q1']);identity={'agent_strategy':'single_agent','model':'fixture','code_hash':'fixed','max_total_tokens':200000,'max_tool_calls':30,'audit_storage':'gzip','audit_artifact_schema':2,'enable_cache':True}
 manifest={'questions_per_arm':2,'input':{'path':str(full),'sha256':sha(full)},'run_identities':{'single_agent':identity},'settings':{'min_free_gb':0}}
 write(run/'run_manifest.json',manifest)
 work=run/'attempts/single_agent';work.mkdir(parents=True)
 def batch(number,items):
  inputfile=work/f'round-{number:04d}.csv';csvfile(inputfile,[qid for qid,_ in items]);out=work/f'round-{number:04d}.json';rs=[]
  for qid,error in items:
   row={'question_id':qid,'patient_fhir_id':'p'+qid[1:],'agent_answer':None,'latency':1,'error':error,'tokens':{'total':10,'per_call':[{'usage_available':True}]},'trace':[{'role':'assistant','content':'synthetic audit'}]}
   rs.append(externalize_result_audit(row,work,question_id=qid))
  write(out,rs);write(Path(str(out)+'.run.json'),identity);bp=out.with_suffix('.batch.json');write(bp,{'arm':'single_agent','round':number,'input':str(inputfile),'input_sha256':sha(inputfile),'output':str(out),'question_ids':[qid for qid,_ in items],'run_identity':identity});write(bp.with_suffix('.seal.json'),{'batch_sha256':sha(bp),'result_sha256':sha(out),'identity_sha256':sha(Path(str(out)+'.run.json'))})
  for row in rs:
   ref=row['audit_artifacts'];target=run/ref['path'];target.parent.mkdir(exist_ok=True);target.write_bytes((work/ref['path']).read_bytes())
  return rs
 rs=batch(1,[('q0',None),('q1','ReadTimeout: request timed out')]);accepted=[rs[0]]
 choices={'q0':{'attempt_file':'attempts/single_agent/round-0001.json','round':1,'runtime_failures_before':0,'selection':'runtime_clean_terminal','runtime_clean':True}}
 if terminal:
  rs2=batch(2,[('q1',None)]);accepted+=rs2;choices['q1']={'attempt_file':'attempts/single_agent/round-0002.json','round':2,'runtime_failures_before':1,'selection':'runtime_clean_terminal','runtime_clean':True}
 else:write(work/'round-0002.batch.json',{'arm':'single_agent','round':2,'question_ids':['q1']})
 ledger={'complete':False,'arms':{'single_agent':choices,'multi_agent':{'v':{'runtime_clean':True}},'direct_llm':{}},'counts':{'single_agent':{'raw_attempts':3 if terminal else 2,'runtime_failures':1,'intrinsic_errors':0},'multi_agent':{'raw_attempts':1,'runtime_failures':0},'direct_llm':{'raw_attempts':0,'runtime_failures':0}}}
 write(run/'acceptance.json',ledger);write(snapshot/'acceptance.json',deepcopy(ledger))
 for arm in monitor.UNSELECTED:
  for suffix,value in [('.json',[{'question_id':'v'}] if arm=='multi_agent' else []),('.json.run.json',{'arm':arm})]:
   name=f'full_{arm}'+suffix;write(run/name,value);(snapshot/name).write_bytes((run/name).read_bytes())
 write(run/'full_single_agent.json',accepted);write(run/'full_single_agent.json.run.json',identity)
 state={'selected_arms':['single_agent'],'benchmark_complete':False,'selected_scope_complete':terminal,'phase':'complete_selected_arms' if terminal else 'running','attention':False,'supervisor_pid':123,'keep_awake_pid':125,'active':{} if terminal else {'single_agent':{'pid':124,'round':2,'checkpointed_in_batch':0}},'accepted':{'single_agent':len(accepted),'multi_agent':1,'direct_llm':0}}
 if terminal:state.update(pending_scoped_export=True,analysis_status='pending_scoped_export',finished_at='2026-10-17T06:00:00+00:00')
 write(run/'supervisor_state.json',state)
 return run,snapshot,batch
args=argparse.Namespace(round=None,clean=None,runtime=None,accepted=None,final_arm=False,skip_native=False)
for terminal in (False,True):
 run,snapshot,_=fixture('positive_'+str(terminal),terminal)
 with patch.object(monitor.subprocess,'run',side_effect=AssertionError('terminal must not inspect stopped PIDs')) if terminal else patch.object(monitor.subprocess,'run',return_value=SimpleNamespace(stdout='123 Ss supervisor\n124 S single_worker\n125 S caffeinate\n',returncode=0)):
  verified=monitor.verify(run,snapshot,args);observed=monitor.observe(run,snapshot,args)
 assert verified['selected_scope_complete']==terminal and verified['original_three_arm_benchmark_complete'] is False and observed['original_three_arm_benchmark_complete'] is False
 assert verified['selected_accepted']==(2 if terminal else 1)
 results.append({'case':'terminal_execution_only_no_analysis_claim' if terminal else 'active_selected_scope_with_retry_queue','result':'PASS'})
def refuse(name,mutate,terminal=False):
 run,snapshot,batch=fixture(name,terminal);mutate(run,snapshot,batch)
 try:
  with patch.object(monitor.subprocess,'run',return_value=SimpleNamespace(stdout='123 Ss supervisor\n124 S single_worker\n125 S caffeinate\n',returncode=0)):monitor.verify(run,snapshot,args)
 except ValueError as error:results.append({'case':name,'result':'REFUSED_AS_EXPECTED','reason':str(error)})
 else:raise AssertionError('Unsafe fixture accepted: '+name)
def update(path,key,value):d=json.loads(path.read_text());d[key]=value;write(path,d)
refuse('unselected_canonical_byte_mutation',lambda r,s,b:(r/'full_multi_agent.json').write_text('[ {"question_id": "v"} ]'))
refuse('direct_identity_mutation',lambda r,s,b:write(r/'full_direct_llm.json.run.json',{'changed':True}))
def change_ledger(r,s,b):d=json.loads((r/'acceptance.json').read_text());d['counts']['direct_llm']['raw_attempts']=1;write(r/'acceptance.json',d)
refuse('unselected_ledger_mutation',change_ledger)
refuse('unauthorized_scope',lambda r,s,b:update(r/'supervisor_state.json','selected_arms',['single_agent','direct_llm']))
refuse('unauthorized_active_arm',lambda r,s,b:update(r/'supervisor_state.json','active',{'direct_llm':{'pid':999,'round':1}}))
refuse('accepted_question_in_active_queue',lambda r,s,b:write(r/'attempts/single_agent/round-0002.batch.json',{'question_ids':['q0','q1']}))
refuse('runtime_retry_missing',lambda r,s,b:write(r/'attempts/single_agent/round-0002.batch.json',{'question_ids':[]}))
refuse('raw_seal_mismatch',lambda r,s,b:(r/'attempts/single_agent/round-0001.json').write_text('[]'))
refuse('accepted_question_rerun_sealed',lambda r,s,b:b(2,[('q0',None)]))
refuse('authoritative_input_changed',lambda r,s,b:(r/'full-input.csv').write_text('question_id,patient_fhir_id\nq0,p0\n'))
refuse('terminal_missing_scope_flag',lambda r,s,b:update(r/'supervisor_state.json','selected_scope_complete',False),True)
refuse('terminal_wrong_analysis_claim',lambda r,s,b:update(r/'supervisor_state.json','analysis_status','complete'),True)
refuse('terminal_false_original_complete',lambda r,s,b:update(r/'supervisor_state.json','benchmark_complete',True),True)
refuse('terminal_ledger_claims_original_complete',lambda r,s,b:update(r/'acceptance.json','complete',True),True)
refuse('terminal_wrong_count',lambda r,s,b:update(r/'supervisor_state.json','accepted',{'single_agent':1}),True)
refuse('terminal_still_active',lambda r,s,b:update(r/'supervisor_state.json','active',{'single_agent':{'pid':124,'round':3}}),True)
run,snapshot,_=fixture('missing_native_pid')
try:
 with patch.object(monitor.subprocess,'run',return_value=SimpleNamespace(stdout='123 Ss supervisor\n',returncode=0)):monitor.verify(run,snapshot,args)
except ValueError as e:results.append({'case':'missing_native_pid','result':'REFUSED_AS_EXPECTED','reason':str(e)})
else:raise AssertionError('Missing worker liveness passed')
for value in (False, 0, [], {}, '', float('nan')):
    projected = monitor.safe_projection({'agent_answer':value,'latency':None,'error':None})
    decision = monitor.classify_attempt(projected)
    assert decision['status']=='attention' and decision['reason']=='terminal_attempt_missing_latency'
results.append({'case':'present_falsy_or_nan_answer_with_missing_latency_never_pending','result':'REFUSED_AS_EXPECTED','variants':6})
assert monitor.classify_attempt(monitor.safe_projection({'agent_answer':None,'latency':None,'error':None}))['status']=='pending'
results.append({'case':'true_placeholder_answer_none_latency_none_remains_pending','result':'PASS'})
run,snapshot,_=fixture('zombie_worker')
try:
 with patch.object(monitor.subprocess,'run',return_value=SimpleNamespace(stdout='123 Ss supervisor\n124 Z single_worker\n125 S caffeinate\n',returncode=0)):monitor.verify(run,snapshot,args)
except ValueError as e:
 assert str(e)=='Owned process is a zombie'
 results.append({'case':'zombie_native_worker_present_but_rejected','result':'REFUSED_AS_EXPECTED','reason':str(e)})
else:raise AssertionError('Zombie worker passed')
report={'monitor_path':monitor.__file__,'monitor_sha256':sha(Path(monitor.__file__)),'cases':results,'case_count':len(results),'all_passed':True,'fixtures_directory':str(root),'inference_calls':0,'live_data_or_source_mutations':0}
Path('/private/tmp/v3-independent-provider-export-review/original22_monitor_results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
