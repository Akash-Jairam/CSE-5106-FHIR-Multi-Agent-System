import collections,csv,hashlib,importlib.util,json,statistics
from pathlib import Path
R=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
O=Path('/private/tmp/fhir-v5-critic-diagnosis');D=R/'benchmarks/design-v5-20261015/diagnostics';RAW=R/'output/experiments/gpt-6-luna/executable_critic_v5_300q'
cases=json.loads((O/'trajectory_cases.json').read_text());formal=json.loads((D/'mechanisms_rows.json').read_text());formal={x['question_id']:x for x in formal};byid={x['question_id']:x for x in cases}
proof=json.loads((R/'benchmarks/design-v5-20261015/completion_protocol_audit.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
assert sha(RAW/'ordinary_single_agent.json')==proof['raw_files']['ordinary_single_agent.json']['sha256']
spec=importlib.util.spec_from_file_location('diagnostic_frozen_scorer',R/'utils/answer_scorer.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
with (R/'manifests/question_families.csv').open(newline='') as stream:families={x['question_id']:x for x in csv.DictReader(stream) if x['question_id'] in formal}
single={x['question_id']:x for x in json.loads((RAW/'ordinary_single_agent.json').read_text())}
scorrect={q:bool(not r.get('error') and S.score_answer(r.get('agent_answer'),r['true_answer'],r['question'],families[q]['template'])['typed_correct']) for q,r in single.items()}
assert sum(scorrect.values())==237 and sum(x['correct'] for x in formal.values())==225
for c in cases:
 f=formal[c['question_id']];assert c['primary_correct']==f['primary_typed_correct'] and c['final_correct']==f['correct']
summary={'stage_by_group':{},'mechanical_reason_by_group':{},'coverage':{},'probe_subtypes':{},'unknown_reason_markers':{}}
groups={'source_valid_wrong':[q for q,f in formal.items() if f['source_valid_typed_wrong_release']],
        'paired_team_wins':[q for q,f in formal.items() if f['correct'] and not scorrect[q]],
        'paired_team_losses':[q for q,f in formal.items() if scorrect[q] and not f['correct']]}
groups['all_critic']=list(byid)
for name,ids in groups.items():
 summary['stage_by_group'][name]=dict(collections.Counter(formal[q]['critic_stage'] for q in ids))
 summary['mechanical_reason_by_group'][name]=dict(collections.Counter(byid[q]['critic']['reason'] for q in ids if q in byid and ('Error:'in byid[q]['critic']['reason'])))
 summary['coverage'][name]={'n':len(ids),'empty_retrieval_ledger':sum(q in byid and not byid[q]['packet'].get('retrieval_ledger') for q in ids),
  'empty_complete_types':sum(q in byid and not byid[q]['packet'].get('complete_types') for q in ids),
  'missing_current_time':sum(q in byid and byid[q]['packet']['context']['current_time'] is None for q in ids)}
 probe=collections.Counter();unknown=collections.Counter()
 for q in ids:
  if q not in byid:continue
  c=byid[q];critic=c['critic'];reason=critic['reason'];v=(critic.get('probe_output') or {}).get('answer')
  if reason=='ValueError: inspection answer schema invalid':
   if isinstance(v,dict):
    for k in ['facts','observations']:
     z=v.get(k);probe[f'{k}:empty' if isinstance(z,list) and not z else f'{k}:over16' if isinstance(z,list) and len(z)>16 else f'{k}:wrong_type' if not isinstance(z,list) else f'{k}:within_range']+=1
   else:probe['answer:not_dict']+=1
  if reason=='ValueError: inspection observation schema invalid':
   names=[]
   for z in v['observations']:
    if len(json.dumps(z.get('value'),allow_nan=False))>1024:probe['observation:over1024']+=1
    if not isinstance(z.get('name'),str) or len(z['name'])>80:probe['observation:invalid_name']+=1
    if z.get('name') in names:probe['observation:duplicate_name']+=1
    names.append(z.get('name'))
  if reason=='ValueError: inspection failed':probe['runtime:'+str(critic['probe_output'].get('error','')).split(':')[0]]+=1
  if 'anchor quote missing'in reason:
   a=c['responses'][0]['parsed']['anchor'];context=c['packet']['context'];quote=a['quote'];channel=a['channel']
   other='question'if channel=='assumptions'else'assumptions'
   probe['anchor:wrong_original_channel'if quote in context[other] else'anchor:not_original_context']+=1
  if formal[q]['critic_stage']in ['first_unknown','second_unknown']:
   text=reason.lower()
   for mark,terms in {'missing_clock':['no assumed current time','no current_time','no current time','without the assumed reference time','no assumed-current-time'],
    'preview_limitation':['preview','small inventory','not shown','not available','available evidence'],
    'scope_or_missing_retrieval':['not complete','not completed','no complete','missing','does not establish','cannot establish'],
    'ambiguity':['ambiguous','ambiguity','tie','unspecified','unclear']}.items():
    if any(t in text for t in terms):unknown[mark]+=1
 summary['probe_subtypes'][name]=dict(probe);summary['unknown_reason_markers'][name]=dict(unknown)
summary['packet_tokens']={'min':min(c['critic']['first_prompt_tokens'] for c in cases),'median':statistics.median(c['critic']['first_prompt_tokens']for c in cases),'max':max(c['critic']['first_prompt_tokens']for c in cases)}
summary['binding']={'mechanisms_rows_sha256':sha(D/'mechanisms_rows.json'),'mechanisms_sha256':sha(D/'mechanisms.json')}
out=[]
for q,f in formal.items():
 if q not in byid:continue
 c=byid[q];c.update(formal=f,paired='team_win'if q in groups['paired_team_wins']else'team_loss'if q in groups['paired_team_losses']else'concordant',single_correct=scorrect[q])
 out.append(c)
(O/'joined_trajectories.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
(O/'deep_census.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2))
