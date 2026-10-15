"""Read-only completed fixed300 V5 auxiliary-counter reconciliation; no protected QA loads."""
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
ROOT=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
RECORD=ROOT/'benchmarks/design-v5-20261015'
OUT=ROOT/'output/experiments/gpt-6-luna/executable_critic_v5_300q'
HERE=Path(__file__).parent
PINS={
 'completion_protocol_audit.json':'7c80f4a6784fb8e141ad2a8b7b2e93357469789768b55a09631d67696a620289',
 'results.json':'1aa4ab40423063305e64378ce36ed62ef282b3ac766533fd590ff6eb3f642d4c'}
TEAM_SHA='b1f4d84fcc87d4460b84106e38f45069c1ad73e1624ae5faf4cbb05ea9a956b4'
def require(value,message):
 if not value:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def same(a,b):return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)
def artifact(value):return isinstance(value,dict) and not value.get('error') and all(k in value for k in ('answer','evidence','operation','contributing_ids','limitation'))
def transition(before,after):
 if before is None or after is None:return 'unavailable'
 return 'helpful' if not before and after else 'harmful' if before and not after else 'neutral'
def refs(value,path=''):
 if isinstance(value,dict):
  if isinstance(value.get('reference'),str):yield path+'.reference','/'.join(value['reference'].rstrip('/').split('/')[-2:])
  for key,item in value.items():yield from refs(item,(path+'.'+key).strip('.'))
 elif isinstance(value,list):
  for i,item in enumerate(value):yield from refs(item,f'{path}[{i}]')
for name,pin in PINS.items():require(sha(RECORD/name)==pin,'Report SHA mismatch: '+name)
completion=read(RECORD/'completion_protocol_audit.json');manifest=read(RECORD/'manifest.json')
require(completion['status']=='PASS' and completion['caller_asserted_owned_jobs_terminal'] is True
 and completion['restored_audit_records']==600 and completion['expected_first_attempts']==600,'Not completed provenance')
require(sha(RECORD/'manifest.json')==completion['manifest_sha256'],'Freeze mismatch')
sources=['scripts/compare_runs.py','utils/answer_scorer.py','scripts/audit_store.py','agent/executable_critic_agent.py','agent/query_contract_agent.py','agent/single_agent.py','agent/framework.py','tools/field_check.py']
for name in sources:require(sha(ROOT/name)==manifest['source_sha256'][name],'Frozen source drift: '+name)
require(sha(OUT/'ordinary_multi_agent.json')==TEAM_SHA==completion['arms']['ordinary_multi_agent']['result_sha256'],'Team raw SHA mismatch')
require(sha(OUT/'ordinary_multi_agent.json.run.json')==completion['arms']['ordinary_multi_agent']['identity_sha256'],'Team identity SHA mismatch')
expected=read(ROOT/'manifests/split.json')['splits']['held_out']['question_ids'];require(len(expected)==len(set(expected))==300,'Wrong development IDs')
expected_set=set(expected);families={}
with (ROOT/'manifests/question_families.csv').open(newline='') as stream:
 for row in csv.DictReader(stream):
  if row['question_id'] in expected_set:families[row['question_id']]=row['template']
require(set(families)==expected_set,'Development metadata missing')
scorer=load(ROOT/'utils/answer_scorer.py','v5_transition_pure_scorer')
audit=load(ROOT/'scripts/audit_store.py','v5_transition_pure_audit')
rows=read(OUT/'ordinary_multi_agent.json');require(len(rows)==300 and [r['question_id'] for r in rows]==expected,'Not exact complete300')
formal_rows={r['question_id']:r for r in read(RECORD/'diagnostics/mechanisms_rows.json')};require(set(formal_rows)==expected_set,'Formal diagnostics cohort mismatch')
official=read(RECORD/'results.json');counts=Counter();draft_shapes=Counter();audit_fields=Counter();by_revision={True:Counter(),False:Counter()};withheld=[];null_drafts=[];recomputed=[]
for row in rows:
 qid=row['question_id'];rev=row.get('revision') or {};draft=rev.get('draft');primary=rev.get('primary');final=rev.get('final');revised=bool(rev.get('solver_source_revised'))
 require(not row.get('attempt_history'),'Not first attempt')
 fields=tuple((row.get('audit_artifacts') or {}).get('fields',[]));audit_fields[','.join(fields)]+=1
 require('revision.messages' not in fields and isinstance(rev,dict),'Revision not inline')
 draft_shapes['|'.join(sorted(draft)) if isinstance(draft,dict) else type(draft).__name__]+=1
 def correct(answer):return bool(scorer.score_answer(answer,row['true_answer'],row['question'],families[qid])['typed_correct'])
 d=correct(draft['answer']) if isinstance(draft,dict) and 'answer' in draft else None
 p=correct(primary['answer']) if artifact(primary) else None
 f=not (row.get('error') is not None and str(row['error']).strip() not in ('','nan')) and correct(row.get('agent_answer'))
 require(isinstance(draft,dict)==(d is not None),'Placeholder draft')
 counts.update(correct=int(f),drafts=int(d is not None),draft_correct=int(d is True),primary_artifacts=int(p is not None),primary_correct=int(p is True),solver_source_revised=int(revised),critic_accepted=int(bool(rev.get('critic_revised'))))
 by_revision[revised].update({'draft_to_primary_'+transition(d,p):1,'draft_to_release_'+transition(d,f):1})
 if d is not None:
  counts.update(wrong_to_right=int(not d and f),right_to_wrong=int(d and not f),semantic_repairs=int(bool(rev.get('revised'))))
 else:null_drafts.append({'question_id':qid,'draft_type':type(draft).__name__,'release_typed_correct':f,'released_answer_is_none':row.get('agent_answer') is None})
 fm=formal_rows[qid]
 require(same([d,p,f,revised],[fm['draft_typed_correct'],fm['primary_typed_correct'],fm['correct'],fm['solver_source_revised']]),'Restored diagnostic row disagreement')
 rec={'question_id':qid,'draft_correct':d,'primary_correct':p,'released_correct':f,'solver_source_revised':revised}
 recomputed.append(rec)
 if not (revised and p is True and not f):continue
 full=audit.restore_result_audit(row,OUT,question_id=qid)
 require(same(full['revision'],rev),'Restoring changed inline revision')
 workspace=full['retrieved_fhir_resources'];resources={f'{item["resourceType"]}/{item["id"]}':item for group in workspace.values() for item in group}
 previous=rev.get('primary_revision') or {};issues=previous.get('final_check_issues') or []
 unresolved=rev.get('unresolved_scope') or {};cited={c['resource_id'] for c in primary['evidence']};contributing=set(primary['contributing_ids'])
 checks=[]
 for message in full['trace']:
  if isinstance(message,dict) and message.get('role')=='tool' and message.get('name')=='check_fields':
   try:value=json.loads(message['content'])
   except (ValueError,TypeError):continue
   if isinstance(value,dict):checks.append((message,value))
 require(checks and checks[-1][0].get('tool_call_id')=='automatic_field_check_2','Missing final source check')
 last_check=checks[-1][1];categories=Counter();links=[]
 for issue in issues:
  hit=re.match(r'evidence\[(\d+)\] ([a-z_]+):',str(issue));category=hit[2] if hit else str(issue).split(':',1)[0];categories[category]+=1
  if category!='encounter_link':continue
  i=int(hit[1]);cid=primary['evidence'][i]['resource_id'];scope=re.search(r'scoped (Encounter/[^ ]+)',str(issue))[1]
  parents=[]
  for key,resource in resources.items():
   for path,target in refs(resource):
    if target==cid and any(target2==scope and path2 in ('encounter.reference','context.reference') for path2,target2 in refs(resource)):
     parents.append({'resource_type':resource['resourceType'],'reference_path':path,'is_cited':key in cited,'is_contributing':key in contributing,'direct_encounter_equals_scope':True,'patient_equals_question_patient':(resource.get('subject') or resource.get('patient') or {}).get('reference','').rstrip('/').split('/')[-1]==row['patient_fhir_id']})
  links.append({'evidence_index':i,'child_type':cid.split('/')[0],'child_has_direct_encounter':bool(resources[cid].get('encounter') or resources[cid].get('context')),'parent_witnesses':parents})
 metadata_links=[]
 for cid in cited:
  if cid.split('/')[0] not in ('Medication','Location'):continue
  roots=[]
  for key,resource in resources.items():
   if resource['resourceType'] in ('Medication','Location'):continue
   for path,target in refs(resource):
    if target==cid:
     roots.append({'resource_type':resource['resourceType'],'reference_path':path,'is_cited':key in cited,'is_contributing':key in contributing,'patient_equals_question_patient':(resource.get('subject') or resource.get('patient') or {}).get('reference','').rstrip('/').split('/')[-1]==row['patient_fhir_id']})
  metadata_links.append({'metadata_type':cid.split('/')[0],'patient_owned_root_count':sum(x['patient_equals_question_patient'] for x in roots),'has_cited_patient_owned_root':any(x['is_cited'] and x['patient_equals_question_patient'] for x in roots),'has_contributing_patient_owned_root':any(x['is_contributing'] and x['patient_equals_question_patient'] for x in roots)})
 casesummary={**rec,'release_policy':rev.get('release_policy'),'released_answer_is_none':row.get('agent_answer') is None,'critic_recorded':bool(rev.get('critic')),'phase_fatal':any(p.get('fatal') for p in rev.get('phases',[])),
  'recorded_final_check_issue_categories':dict(categories),'citation_status_counts':dict(Counter(c.get('status') for c in last_check.get('results',[]))),
  'citation_warning_categories':dict(Counter(str(w).split(':',1)[0] for c in last_check.get('results',[]) for w in c.get('warnings',[]))),
  'unresolved_scope_count':len(unresolved),'unresolved_by_type_and_reason':{f'{k[0]}:{k[1]}':v for k,v in Counter((key.split('/')[0],reason) for key,reason in unresolved.items()).items()},
  'unresolved_cited_overlap':len(set(unresolved)&cited),'unresolved_contributing_overlap':len(set(unresolved)&contributing),
  'held_parent_link_witnesses':links,'cited_metadata_linkage':metadata_links,'audit_reference':row['audit_artifacts']}
 require(casesummary['release_policy']=='primary_failed_execution_or_source_contract' and not casesummary['critic_recorded'] and not casesummary['phase_fatal'],'Wrong gate ancestry')
 withheld.append(casesummary)
transition_keys=('wrong_to_right','right_to_wrong','draft_correct','drafts','semantic_repairs')
require({k:counts[k] for k in transition_keys}==official['arms']['v5_team']['transitions'],'Official transition mismatch')
require(counts['correct']==225 and len(withheld)==6 and counts['critic_accepted']==0,'Unexpected result')
require(by_revision[True]==Counter({'draft_to_primary_helpful':5,'draft_to_primary_neutral':18,'draft_to_release_helpful':4,'draft_to_release_neutral':14,'draft_to_release_harmful':5}),'Unexpected revised transition partition')
for name,pin in PINS.items():require(sha(RECORD/name)==pin,'Report changed during audit')
require(sha(OUT/'ordinary_multi_agent.json')==TEAM_SHA,'Raw changed during audit')
prior=Path('/private/tmp/fhir-v5-independent-score-wrmmao7a')
report={'status':'PASS_ENDPOINT_RECONCILIATION','scope':'Completed fixed300 V5 only; protected1397/full QA not opened',
 'helper_sha256':sha(__file__),'completion_report_sha256':PINS['completion_protocol_audit.json'],'official_results_sha256':PINS['results.json'],'raw_team_sha256':TEAM_SHA,
 'manifest_sha256':sha(RECORD/'manifest.json'),'source_sha256':{name:sha(ROOT/name) for name in sources},
 'formal_mechanism_helper_sha256':sha(RECORD/'fhir_v5_critic_mechanisms.py'),'formal_mechanism_summary_sha256':sha(RECORD/'diagnostics/mechanisms.json'),'formal_mechanism_rows_sha256':sha(RECORD/'diagnostics/mechanisms_rows.json'),
 'definitions':{'official':'draft answer -> final released answer over all present draft dicts; generic revised flag is labeled semantic_repairs','solver_source_effect':'draft answer -> computed primary answer only when solver_source_revised; no source-release assertion','critic_effect':'accepted executed/source-checked critic replacement; zero in this run'},
 'counts':dict(counts),'draft_shapes':dict(draft_shapes),'externalized_field_sets':dict(audit_fields),'all_300_revision_metadata_inline':True,
 'transition_partition':{'source_revised_23':dict(by_revision[True]),'not_source_revised_277':dict(by_revision[False])},
 'missing_draft_records':null_drafts,'six_computed_correct_primaries_withheld':withheld,
 'prior_independent_delivery':{name:{'path':str(prior/name),'sha256':sha(prior/name)} for name in ['independent_verification.json','verify_development.py']},
 'limits':['Typed-correct computed answer does not certify complete search or valid semantic derivation. Recorded citation status valid establishes only the checked values, not answer correctness.',
 'Three missing-direct-encounter citations have held cited/contributing patient-owned parents linking them to the scoped encounter. This is concrete evidence of a flat direct-link gate limitation, not an independent rerun of the full validator.',
 'Three global unresolved metadata vetoes concern no cited/contributing record; ordinary per-citation checks have no warnings. This is evidence of a broad unrelated-metadata refusal, not permission to use unlinked records or infer complete negative search.',
 'All six gzip records were independently owner/SHA/CRC restored; all600 had already passed the pinned completion guard. Only six were restored by this explanation script.',
 'Official transitions are numerically consistent as end-to-end available-draft -> release outcomes; they are unsuitable for attributing critic repairs or isolating solver computed-answer changes. No official report/scorer/source edits.'],
 'model_FHIR_network_private_configuration_calls':0}
(HERE/'reconciliation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(HERE/'recomputed_answer_credit_flags.json').write_text(json.dumps(recomputed,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'counts':dict(counts),'partition':report['transition_partition'],'missing_drafts':null_drafts,'cases':[{'qid':r['question_id'],'issues':r['recorded_final_check_issue_categories'],'unresolved':r['unresolved_scope_count']} for r in withheld],'prior_independent_delivery':report['prior_independent_delivery']},indent=2))
