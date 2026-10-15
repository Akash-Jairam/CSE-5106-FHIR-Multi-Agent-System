"""Authorized completed V4 trajectory review. Private output; no model/store/config calls."""
import ast
from collections import Counter,defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
ROOT=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
OUT=Path('/private/tmp/fhir-v4-trajectory-review.json')
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
h=load('/private/tmp/fhir_v4_large_run_diagnostics_corrected.py','review_metadata_functions_only')
audit=load(ROOT/'scripts/audit_store.py','review_public_audit_store')
scorer=load(ROOT/'utils/answer_scorer.py','review_public_typed_scorer')
with (ROOT/'final_dataset/questions_answers_sql_fhir.csv').open(newline='') as stream:source={r['question_id']:r for r in csv.DictReader(stream)}
with (ROOT/'manifests/question_families.csv').open(newline='') as stream:families={r['question_id']:r for r in csv.DictReader(stream)}
ids=json.loads((ROOT/'manifests/split.json').read_text())['splits']['held_out']['question_ids']
v4rows=json.loads((ROOT/'benchmarks/design-v4-20261014/diagnostics/mechanisms_rows.json').read_text())
v3rows=json.loads((ROOT/'benchmarks/design-v3d-20261014/diagnostics/mechanisms_rows.json').read_text())
metadata={arm:{r['question_id']:r for r in v4rows if r['arm']==arm} for arm in ['ordinary_single_agent','ordinary_multi_agent']}
metadata['v3d']={r['question_id']:r for r in v3rows}
single,team,old=(metadata[k] for k in ['ordinary_single_agent','ordinary_multi_agent','v3d'])
v4proof=json.loads((ROOT/'benchmarks/design-v4-20261014/completion_protocol_audit.json').read_text())
v3proof=json.loads((ROOT/'benchmarks/design-v3d-20261014/diagnostics/mechanisms.json').read_text())
paths={'ordinary_single_agent':ROOT/'output/experiments/gpt-6-luna/plan_conditioned_v4_300q/ordinary_single_agent.json',
       'ordinary_multi_agent':ROOT/'output/experiments/gpt-6-luna/plan_conditioned_v4_300q/ordinary_multi_agent.json',
       'v3d':ROOT/'output/experiments/gpt-6-luna/query_contract_v3d_300q/ordinary_multi_agent.json'}
expected={k:v4proof['arms'][k]['result_sha256'] for k in ['ordinary_single_agent','ordinary_multi_agent']}
expected['v3d']=v3proof['raw_inputs']['v3d']['raw_sha256']
for arm,p in paths.items():assert h.sha256(p)==expected[arm]

def score(answer,q):return scorer.score_answer(answer,source[q]['true_answer'],source[q]['question'],families[q]['template'])
def itype(issue):
 s=issue.get('type','unstructured') if isinstance(issue,dict) else str(issue).partition(':')[0]
 return re.sub(r'evidence\[[0-9]+\]','evidence[*]',s)

def trajectory(row,q):
 calls={}; modes=Counter();errors=Counter();checks=Counter();last_final=None;last_code=None
 for msg in row.get('trace') or []:
  if not isinstance(msg,dict):continue
  for call in msg.get('tool_calls') or []:
   if not isinstance(call,dict):continue
   fn=call.get('function') or {}
   try:args=json.loads(fn.get('arguments') or '{}')
   except (ValueError,TypeError):args={}
   args=args if isinstance(args,dict) else {}
   calls[call.get('id')]=(fn.get('name'),args)
   if fn.get('name')=='execute_python_code':modes[str(args.get('mode','auto'))]+=1
  if msg.get('role')!='tool':continue
  name,args=calls.get(msg.get('tool_call_id'),(msg.get('name'),{}));name=msg.get('name') or name
  try:reply=json.loads(msg.get('content') or '{}')
  except (ValueError,TypeError):reply=None
  if name=='execute_python_code' and isinstance(reply,dict):
   if reply.get('error'):
    errors[str(reply.get('error_type') or 'unclassified_worker_error')]+=1
   elif args.get('mode')=='final' and 'answer' in reply:
    last_final=reply;last_code=args.get('code')
  if name=='check_fields' and isinstance(reply,dict):
   for issue in reply.get('issues') or []:checks[itype(issue)]+=1
 rev=row.get('revision') or {};primary=rev.get('primary_revision') or {}
 final_issues=[itype(i) for i in rev.get('final_check_issues') or []]
 code_fields=[];code_functions=[]
 if isinstance(last_code,str):
  try:
   tree=ast.parse(last_code)
   code_fields=sorted({n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)
       and n.value in {'effectiveDateTime','issued','onsetDateTime','authoredOn','period','start','end','valueQuantity','value','unit','status','encounter','subject','date','dateTime','dispenseRequest'}})
   code_functions=sorted({n.func.id if isinstance(n.func,ast.Name) else n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))})
  except (ValueError,SyntaxError):pass
 return {'question_id':q,'family':families[q]['family'],'answer_kind':team[q]['answer_kind'],
  'fresh_single_correct':single[q]['correct'],'v3d_correct':old[q]['correct'],'v4_correct':team[q]['correct'],
  'source_valid_release':team[q]['source_valid_release'],'plan_status':team[q]['plan_status'],
  'draft_correct':team[q]['draft_correct'],'primary_correct':team[q]['primary_correct'],
  'final_issue_kinds':final_issues,'unresolved_scope_reasons':dict(Counter(str(v) for v in (rev.get('unresolved_scope') or {}).values())),
  'solver_invocation_attempts':h.model_trace_stats(row.get('trace'))['solver_invocation_attempt_signals'],
  'host_source_boundaries':h.model_trace_stats(row.get('trace'))['host_final_boundaries'] if 'host_final_boundaries' in h.model_trace_stats(row.get('trace')) else 0,
  'source_check_issue_kinds':dict(checks),'python_mode_requests':dict(modes),'python_error_types':dict(errors),
  'last_successful_final_reply_present':last_final is not None,
  'last_successful_final_answer_typed_correct':bool(score(last_final['answer'],q)['typed_correct']) if last_final is not None else None,
  'last_final_code_field_names':code_fields,'last_final_code_functions':code_functions,
  'source_revision_attempted':bool(rev.get('solver_source_revised')),
  'primary_revision_error_present':bool(primary.get('revision_error')),
  'budget_stopped':bool((row.get('budget') or {}).get('stopped')),
  'model_calls':(row.get('tokens') or {}).get('llm_calls'),'tool_calls':(row.get('budget') or {}).get('tool_calls_run')}

wanted={q for q,t in team.items() if not t['source_valid_release'] or old[q]['correct'] and not t['correct']}
verified_scores={};details={};seen_sets={}
for arm,p in paths.items():
 seen=[];correct=0
 for row in h.iter_rows(p):
  q=row['question_id'];seen.append(q);assert row.get('attempt_history') in (None,[])
  answer_score=score(row.get('agent_answer'),q)
  value=bool(not h.failed(row) and answer_score['typed_correct'])
  assert value==metadata[arm][q]['correct'];correct+=value
  if arm=='ordinary_multi_agent' and q in wanted:
   restored=audit.restore_result_audit(row,p.parent,question_id=q,restore_history=False)
   details[q]=trajectory(restored,q)
 assert seen==ids
 verified_scores[arm]=correct;seen_sets[arm]=len(seen)
 assert h.sha256(p)==expected[arm]

def aggregate(rows):
 return {'n':len(rows),'correct':sum(r['correct'] for r in rows),'source_valid_releases':sum(r['source_valid_release'] for r in rows),
         'source_valid_typed_wrong':sum(r['source_valid_release'] and not r['correct'] for r in rows),
         'source_refusals':sum(not r['source_valid_release'] for r in rows),'null_credit_outside_clean_release':sum(r.get('null_credit_outside_clean_release',False) for r in rows),
         'family_counts':dict(Counter(r['family'] for r in rows)),'plan_status_counts':dict(Counter(r['plan_status'] for r in rows))}
losses=[t for q,t in team.items() if single[q]['correct'] and not t['correct']]
regressions=[t for q,t in team.items() if old[q]['correct'] and not t['correct']]
gains=[t for q,t in team.items() if t['correct'] and not old[q]['correct']]
gates=[t for t in team.values() if not t['source_valid_release']]
gatedetails=[details[r['question_id']] for r in gates]
report={'scope':'Completed 300 first attempts in each of three arms; all final scores independently recomputed with frozen public typed scorer. Selected exact audit trajectories restored. No patient resource content emitted.',
 'scores':verified_scores,'raw_hashes':expected,'cohort_counts':seen_sets,
 'all_v4_team':aggregate(list(team.values())),'fresh_single_paired_losses':aggregate(losses),
 'v3d_correct_v4_wrong':aggregate(regressions),'v4_correct_v3d_wrong':aggregate(gains),
 'plan_status_association':{key:aggregate([r for r in team.values() if r['plan_status']==key]) for key in sorted({r['plan_status'] for r in team.values()})},
 'source_refusal_trajectory':{'questions':len(gatedetails),'final_issue_kinds':dict(Counter(i for d in gatedetails for i in d['final_issue_kinds'])),
  'invocation_attempt_counts':dict(Counter(d['solver_invocation_attempts'] for d in gatedetails)),
  'boundary_counts':dict(Counter(d['host_source_boundaries'] for d in gatedetails)),
  'python_error_types':dict(sum((Counter(d['python_error_types']) for d in gatedetails),Counter())),
  'last_successful_final_typed_correct':sum(d['last_successful_final_answer_typed_correct'] is True for d in gatedetails),
  'budget_stops':sum(d['budget_stopped'] for d in gatedetails)},
 'trajectory_details':list(details.values()),
 'caveats':['Visible/ignored plan groups are selected by parse/availability outcomes, not randomized controls.',
 'Typed correctness of a rejected candidate does not validate its sources and does not justify bypassing the release gate.',
 'V3d/V4 comparison bundles advice visibility, removing late audit, and reallocating solver budget; stochastic effects are not isolated.',
 'Plan semantics are reviewed separately by the parent; these mechanism summaries do not prove a causal plan effect.']}
OUT.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in {'trajectory_details','caveats','scope'}},indent=2))
