"""Private synthetic regression and AST closure audit. No live run or scoring."""
import ast, contextlib, hashlib, importlib.util, io, json, sys, types
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
OUT=Path('/private/tmp/v3-independent-provider-export-review')
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
CANDIDATE=Path('/private/tmp/v3-provider-rejection-candidate/v3_full_run.py')
PROTOTYPE=Path('/private/tmp/v3-paired-provider-amendment/monitor_single_agent.py')
EXPORT=Path('/private/tmp/v3-paired-provider-amendment/export_complete_v3_single.py')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source_hashes={str(p):digest(p) for p in (CANDIDATE,PROTOTYPE,EXPORT)}
import scripts
candidate=types.ModuleType('scripts.v3_full_run');candidate.__file__=str(ROOT/'scripts/v3_full_run.py');candidate.__package__='scripts'
sys.modules[candidate.__name__]=candidate;scripts.v3_full_run=candidate
exec(compile(CANDIDATE.read_text(),str(CANDIDATE),'exec'),candidate.__dict__)
source=Path('/private/tmp/test_single_agent_monitor.py').read_text().replace("'/private/tmp/single_agent_monitor.py'",repr(str(PROTOTYPE))).replace("'/private/tmp/single_agent_monitor_tests.json'",repr(str(OUT/'original22_monitor_results.json')))
(OUT/'transformed_original22_monitor.py').write_text(source)
ns={'__name__':'independent_monitor_fixture_review'}
with contextlib.redirect_stdout(io.StringIO()):exec(compile(source,str(OUT/'transformed_original22_monitor.py'),'exec'),ns)
monitor=ns['monitor'];write=ns['write'];sha=ns['sha'];args=ns['args'];fixture=ns['fixture'];results=[]
payload={'error':{'code':'content_filter','param':'prompt','status':400,'innererror':{'code':'ResponsibleAIPolicyViolation','content_filter_result':{'jailbreak':{'detected':True,'filtered':True}}}}}
error=candidate.SDK_ERROR_PREFIX+repr(payload)
policy=candidate.PROVIDER_REJECTION_POLICY
native=SimpleNamespace(stdout='123 Ss supervisor\n124 S single_worker\n125 S caffeinate\n',returncode=0)
def load(p):return json.loads(p.read_text())
def update(p,**kw):v=load(p);v.update(kw);write(p,v)
def enable(run):update(run/'run_manifest.json',classifier_version=candidate.CLASSIFIER_VERSION,selected_arms=['single_agent'],provider_rejection_policy=policy)
def reject(run,number):
 out=run/f'attempts/single_agent/round-{number:04d}.json';rows=load(out)
 for row in rows:
  if row['question_id']=='q1':row.update(error=error,tokens={'total':0,'per_call':[{'usage_available':False}]})
 write(out,rows);seal=out.with_suffix('.batch.seal.json');update(seal,result_sha256=sha(out))
def sidecar(run,count,parked):write(run/'provider_rejections.json',{'policy':policy,'source':'reconstructed_from_sealed_immutable_attempts','rejections':{'single_agent':{'q1':count}},'parked_question_ids':{'single_agent':parked}})
def check(run,snapshot):
 with patch.object(monitor.subprocess,'run',return_value=native):return monitor.verify(run,snapshot,args)
def refuse(name,run,snapshot,reason):
 try:check(run,snapshot)
 except ValueError as e:
  assert reason in str(e),(name,str(e));results.append({'case':name,'result':'REFUSED_AS_EXPECTED','reason':str(e)})
 else:raise AssertionError(name+' accepted')
def counts(run,raw,runtime):
 p=run/'acceptance.json';v=load(p);v['counts']['single_agent']={'raw_attempts':raw,'runtime_failures':runtime,'intrinsic_errors':0};write(p,v)
# One unchanged retry remains eligible; the sidecar merely mirrors sealed replay.
run,snapshot,batch=fixture('v4_one_rejection_queued');enable(run);reject(run,1);sidecar(run,1,[])
v=check(run,snapshot);assert v['provider_rejections_replayed']=={'q1':1} and not v['parked_pending_question_ids'];results.append({'case':'one_rejection_retry_queued','result':'PASS'})
prior=monitor.replay(run,load(run/'run_manifest.json'))
sidecar(run,0,[]);after=monitor.replay(run,load(run/'run_manifest.json'));assert prior==after
refuse('mutable_sidecar_cannot_override_sealed_count',run,snapshot,'counts/parking differ')
sidecar(run,1,['q1']);refuse('false_parking_sidecar_refused',run,snapshot,'counts/parking differ')
sidecar(run,1,[]);p=run/'provider_rejections.json';update(p,source='mutable_counter');refuse('unsealed_sidecar_provenance_refused',run,snapshot,'provenance changed')
# Exact opt-in policy and scope only; no-policy cases are in the original22 suite.
manifest=load(run/'run_manifest.json')
for field,value in [('provider_rejection_policy',{**policy,'max_rejections_per_question':3}),('selected_arms',['direct_llm'])]:
 bad={**manifest,field:value}
 try: candidate.classifier_for_manifest(bad)
 except ValueError:pass
 else:raise AssertionError('invalid policy accepted')
results.append({'case':'invalid_policy_or_scope_refused','result':'PASS','variants':2})
# A valid second attempt may recover, with first-clean identity retained.
run,snapshot,batch=fixture('v4_second_attempt_clean_terminal',True);enable(run);reject(run,1);sidecar(run,1,[])
v=check(run,snapshot);assert v['selected_scope_complete'] and v['selected_accepted']==2;results.append({'case':'authorized_second_attempt_clean_terminal','result':'PASS'})
# A second provider rejection parks; a third queued attempt is refused.
run,snapshot,batch=fixture('v4_two_rejections_cannot_queue_third');enable(run);reject(run,1);batch(2,[('q1',error)]);reject(run,2);counts(run,3,2);sidecar(run,2,['q1'])
update(run/'supervisor_state.json',active={'single_agent':{'pid':124,'round':3}});write(run/'attempts/single_agent/round-0003.batch.json',{'arm':'single_agent','round':3,'question_ids':['q1']})
replay=monitor.replay(run,load(run/'run_manifest.json'));assert replay[2]=={'q1':2} and replay[3]['runtime_failures']==2
refuse('parked_qid_active_third_attempt_refused',run,snapshot,'Persistently rejected request scheduled')
sidecar(run,2,[]);refuse('parked_sidecar_missing_qid_refused',run,snapshot,'counts/parking differ')
# A completed third-clean attempt must never erase the earlier policy breach.
run,snapshot,batch=fixture('v4_historical_third_clean_refused',True);enable(run);reject(run,1);batch(2,[('q1',error)]);reject(run,2);third=batch(3,[('q1',None)])
p=run/'full_single_agent.json';v=load(p);v=[row for row in v if row['question_id']!='q1']+third;write(p,v)
p=run/'acceptance.json';v=load(p);v['arms']['single_agent']['q1'].update(attempt_file='attempts/single_agent/round-0003.json',round=3,runtime_failures_before=2);v['counts']['single_agent']={'raw_attempts':4,'runtime_failures':2,'intrinsic_errors':0};write(p,v);sidecar(run,2,[])
refuse('historical_third_clean_after_two_rejections_refused',run,snapshot,'Historical batch exceeded provider rejection unchanged-retry limit')
# A generic runtime does not consume the separate provider-rejection budget.
run,snapshot,batch=fixture('v4_generic_runtime_before_provider_retry');enable(run)
batch(2,[('q1',error)]);reject(run,2);counts(run,3,2);sidecar(run,1,[])
update(run/'supervisor_state.json',active={'single_agent':{'pid':124,'round':3}});write(run/'attempts/single_agent/round-0003.batch.json',{'arm':'single_agent','round':3,'question_ids':['q1']})
v=check(run,snapshot);assert v['provider_rejections_replayed']=={'q1':1} and v['replayed_counts']['runtime_failures']==2;results.append({'case':'generic_runtime_does_not_consume_provider_limit','result':'PASS'})
# Independently reconstruct the precise AST dependency closure, then invoke pure loader.
spec=importlib.util.spec_from_file_location('independent_private_export',EXPORT);exp=importlib.util.module_from_spec(spec);spec.loader.exec_module(exp)
tree=ast.parse(CANDIDATE.read_text());defs={}
for n in tree.body:
 if isinstance(n,ast.FunctionDef):defs[n.name]=n
 elif isinstance(n,ast.Assign):
  for t in n.targets:
   if isinstance(t,ast.Name):defs[t.id]=n
selected=set();pending=['classify_attempt','CLASSIFIER_VERSION']
while pending:
 name=pending.pop()
 if name in selected or name not in defs:continue
 selected.add(name);pending.extend(n.id for n in ast.walk(defs[name]) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id in defs and n.id not in selected)
assert selected==exp.V4_CLASSIFIER_NAMES
u=exp.load_utilities(ROOT);pure,version,ast_hash=exp.load_v4_classifier(CANDIDATE,u)
assert version=='v3-infrastructure-v4' and pure({'agent_answer':None,'latency':1,'error':error,'tokens':{'per_call':[{'usage_available':False}]}},allow_provider_rejection=True)['category']=='provider_policy_rejection'
assert all(digest(p)==h for p,h in source_hashes.items())
report={'all_passed':True,'original_monitor_cases':22,'additional_monitor_cases':len(results),'cases':results,'source_sha256':source_hashes,'AST_dependency_names':sorted(selected),'AST_dependency_count':len(selected),'AST_semantic_sha256':ast_hash,'no_inference_calls':True,'no_live_data_or_source_mutations':True,'no_actual_answer_scoring':True}
(OUT/'independent_monitor_v4_results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'all_passed':True,'original_cases':22,'additional_cases':len(results),'source_sha256':source_hashes,'evidence':str(OUT/'independent_monitor_v4_results.json')},indent=2))
