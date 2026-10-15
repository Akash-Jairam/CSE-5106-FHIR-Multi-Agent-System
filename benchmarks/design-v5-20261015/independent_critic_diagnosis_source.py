"""Post-terminal fixed300 diagnosis only. Read frozen labels only from owned rows."""
import collections,csv,gzip,hashlib,importlib.util,json,math
from pathlib import Path

ROOT=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
REC=ROOT/'benchmarks/design-v5-20261015';RAW=ROOT/'output/experiments/gpt-6-luna/executable_critic_v5_300q'
OUT=Path('/private/tmp/fhir-v5-critic-diagnosis');OUT.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
proof=load(REC/'completion_protocol_audit.json');official=load(REC/'official_scoring_record.json')
assert proof['status']=='PASS' and proof['mode']=='completion' and proof['expected_first_attempts']==600 and proof['restored_audit_records']==600 and proof['caller_asserted_owned_jobs_terminal'] is True
assert sha(REC/'completion_protocol_audit.json')==official['completion_report_sha256']
assert sha(REC/'results.json')==official['official_results_sha256']
assert sha(ROOT/'agent/executable_critic_agent.py')=='e66f7b3af92a63fd80c92f00a4bae357dc01cdf5b1d11c2f5d3ef51f3ddb42c7'
for n in ['ordinary_multi_agent.json','ordinary_multi_agent.json.run.json']:
 assert sha(RAW/n)==proof['raw_files'][n]['sha256']
rows=load(RAW/'ordinary_multi_agent.json');assert len(rows)==300
manifest=load(REC/'manifest.json');assert sha(REC/'manifest.json')==proof['manifest_sha256']
family_path=ROOT/'manifests/question_families.csv'
assert sha(family_path)==manifest['source_sha256']['manifests/question_families.csv']
selected={row['question_id'] for row in rows}
with family_path.open(newline='') as stream:
 families={item['question_id']:item for item in csv.DictReader(stream) if item['question_id'] in selected}
assert set(families)==selected
# Scorer import happens only after completed guard, official-result and raw/source pins above.
spec=importlib.util.spec_from_file_location('frozen_postterminal_scorer',ROOT/'utils/answer_scorer.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
counts=collections.Counter();reasons=collections.Counter();case=[]
for row in rows:
 ref=row['audit_artifacts'];p=RAW/ref['path'];assert sha(p)==proof['raw_files'][ref['path']]['sha256']
 with gzip.open(p,'rb') as f:blob=f.read(128*1024*1024+1)
 assert len(blob)<=128*1024*1024 and hashlib.sha256(blob).hexdigest()==ref['sha256']
 payload=json.loads(blob);assert ref['question_id']==row['question_id']==payload['_audit_meta']['question_id']
 trace=payload.get('trace',row.get('trace',[]));rev=row['revision'];critic=rev.get('critic')
 workspace=payload.get('retrieved_fhir_resources',{})
 counts['rows']+=1
 if not critic:counts['no_critic']+=1;continue
 counts['critic_rows']+=1;counts['status:'+critic['status']]+=1;reasons[critic.get('reason','')]+=1
 started=False;responses=[];packet=None;tool=[]
 for index,m in enumerate(trace):
  if m.get('role')=='system' and str(m.get('content','')).startswith('Review the actual executed program against the ORIGINAL question'):
   started=True;continue
  if not started:continue
  if m.get('role')=='user' and packet is None:
   try:packet=json.loads(m.get('content') or '')
   except ValueError:pass
  if m.get('role')=='assistant' and m.get('phase')!='critic' and m.get('model_call') is not False:
   text=m.get('content');parsed=None
   try:parsed=json.loads(text or '')
   except ValueError:pass
   responses.append({'trace_index':index,'content':text,'parsed':parsed,'tool_calls':m.get('tool_calls')})
  if m.get('role')=='tool':tool.append({'trace_index':index,**m})
 primary=rev.get('primary');correct=lambda a:bool(S.score_answer(a,row['true_answer'],row.get('question',''),families[row['question_id']]['template'])['typed_correct'])
 primary_correct=correct(primary['answer']) if isinstance(primary,dict) and 'answer'in primary else None
 d={'question_id':row['question_id'],'question':row.get('question'),'assumption':row.get('assumption'),
    'reference':row['true_answer'],'template':families[row['question_id']]['template'],'primary':primary,'primary_correct':primary_correct,'final_correct':correct(row.get('agent_answer')),
    'critic':critic,'responses':responses,'critic_tools':tool,'packet':packet,'audit_reference':ref,
    'workspace_kinds':{k:len(v) for k,v in workspace.items()}}
 for i,r in enumerate(responses):
  v=r['parsed'];decision=v.get('decision') if isinstance(v,dict) else 'unparsed'
  counts[f'call{i+1}_decision:{decision}']+=1
 counts['responses']+=len(responses);counts['model_calls']+=critic.get('model_calls',0);counts['tools']+=critic.get('tool_calls',0)
 counts['primary_correct' if primary_correct else 'primary_wrong']+=1
 if critic.get('validated_probe'):counts['validated_probe']+=1
 if critic.get('probe_output'):counts['probe_attempt']+=1
 if critic.get('alternate_execution'):counts['alternate_execution']+=1
 if critic.get('source_check'):counts['candidate_source_check']+=1
 if primary_correct is False:
  for i,r in enumerate(responses):
   v=r['parsed'];counts[f'wrong_call{i+1}_decision:{v.get("decision") if isinstance(v,dict) else "unparsed"}']+=1
 case.append(d)
summary={'counts':dict(counts),'reasons':dict(reasons),'pins':{'completion':sha(REC/'completion_protocol_audit.json'),'official':sha(REC/'results.json'),
 'team':sha(RAW/'ordinary_multi_agent.json'),'frozen_agent':sha(ROOT/'agent/executable_critic_agent.py')},
 'scoring_caveat':'exact frozen typed scorer and fixed300 family templates; formal helper totals authoritative'}
(OUT/'trajectory_cases.json').write_text(json.dumps(case,indent=2,allow_nan=False)+'\n')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2))
