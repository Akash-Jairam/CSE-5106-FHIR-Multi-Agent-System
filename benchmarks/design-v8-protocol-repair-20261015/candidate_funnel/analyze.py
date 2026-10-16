from pathlib import Path
import json,gzip,hashlib,importlib.util,collections,sys
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
REC=ROOT/'benchmarks/design-v8-protocol-repair-20261015'
RUN=ROOT/'output/experiments/gpt-6-luna/v8_mechanism_120_protocol_repair_20261015'
def load(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v): return json.dumps(v,sort_keys=True,ensure_ascii=True,allow_nan=False)
def dump(n,d): (OUT/n).write_text(json.dumps(d,indent=2,sort_keys=True,ensure_ascii=True,allow_nan=False)+'\n')
def score(v,row): return bool(scorer.score_answer(v,row['true_answer'],row['question'],row.get('template',''))['typed_correct'])
def failed(r): return bool(r.get('error'))
def parse(s):
 try:return json.loads(s)
 except (ValueError,TypeError):return None
manifest=load(REC/'manifest.json');runtime=load(REC/'runtime_adapter_manifest.json');integrity=load(REC/'completion_integrity.json')
assert sha(REC/'completion_integrity.json')=='677aaba828fd29b0546a5b6fc7ed7b1faa691a8d1b6f2e602c8c5113433613d7'
assert sha(REC/'official_analysis.json')=='8ecfb13c8f5491b6d57dbdb0eb0e3fa0be15addcd2dbbc1a66c206696bff71c4'
assert integrity['status']=='PASS' and integrity['questions']==120 and integrity['policy_outcomes']==480
hold_paths=set(manifest['source_sha256'])|set(runtime['source_sha256'])
meta=['manifest.json','runtime_adapter_manifest.json','expected_public_binding.json','prospective_protocol.json','launch_commands.json','design.txt','integrity_pins.json','source.zip','runtime_adapter_sources.zip']
hold_paths.update(str((REC/n).relative_to(ROOT)) for n in meta)
hold_before={n:sha(ROOT/n) for n in sorted(hold_paths)}
for n,h in {**manifest['source_sha256'],**runtime['source_sha256']}.items(): assert hold_before[n]==h,n
records=load(RUN/'study_outcomes.json');assert len(records)==480
assert sha(RUN/'study_outcomes.json')==integrity['raw_sha256']['study_outcomes.json']
spec=importlib.util.spec_from_file_location('frozen_typed_scorer',ROOT/'utils/answer_scorer.py');scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
index={(r['question_id'],r['study_arm']):r for r in records};assert len(index)==480
ids={q for q,m in index};assert len(ids)==120 and set(index)=={(q,m) for q in ids for m in 'AWRC'}
manifest_cases=load(ROOT/'manifests/v8_mechanism_120_protocol_repair_20261015.json')['cases'];assert ids=={c['question_id'] for c in manifest_cases}
envelopes={}; audit_used={}
for q in ids:
 r=index[q,'A'];p=RUN/r['primary_envelope_ref'];assert sha(p)==integrity['raw_sha256'][r['primary_envelope_ref']]
 uncompressed=gzip.decompress(p.read_bytes());e=json.loads(uncompressed);assert hashlib.sha256(canonical(e['payload']).encode()).hexdigest()==e['sha256'];assert hashlib.sha256(uncompressed).hexdigest()==r['primary_envelope_sha256'];envelopes[q]=e
for q in ids:
 for m in 'AWRC':
  r=index[q,m];a=index[q,'A'];assert all(canonical(r[k])==canonical(a[k]) for k in ['question','true_answer','template','patient_fhir_id','primary_envelope_sha256'])
scored={(q,m):not failed(r) and score(r['agent_answer'],r) for (q,m),r in index.items()}
def audit(r):
 ref=r['audit_artifacts'];rel=ref['path'];p=RUN/rel;raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==integrity['raw_sha256'][rel]
 data=gzip.decompress(raw);assert hashlib.sha256(data).hexdigest()==ref['sha256'];v=json.loads(data);assert v['_audit_meta']['question_id']==r['question_id'];audit_used[rel]=hashlib.sha256(raw).hexdigest();return v
rows=[];details=[]
for c in manifest_cases:
 q=c['question_id'];a=index[q,'A'];env=envelopes[q];acand=env['payload']['candidate'];a_correct=scored[q,'A']
 for m in 'WRC':
  r=index[q,m];rev=r['revision'];mech=r['mechanism'];b=mech['released_source_bound_B'];av=audit(r);snaps=av.get('revision.messages',rev['messages']);bs=next((v for v in snaps if v['candidate']=='B'),None)
  phases=rev['phases'];aux=phases[1] if len(phases)>1 else None;trace=av.get('trace',[]);bt=[(i,t) for i,t in enumerate(trace) if t.get('phase')=='counterexample_investigation'];calls={};events=[];messages=[]
  for i,t in bt:
   if t.get('role')=='assistant':
    for call in t.get('tool_calls',[]) or []:
     args=parse(call.get('function',{}).get('arguments'));args=args if isinstance(args,dict) else {};calls[call['id']]={'name':call.get('function',{}).get('name'),'args':args,'trace_i':i}
    content=t.get('content')
    if isinstance(content,str) and content.strip():messages.append({'trace_i':i,'content':content,'parsed':parse(content),'has_native_calls':bool(t.get('tool_calls'))})
   if t.get('role')=='tool':
    call=calls.get(t.get('tool_call_id'),{});args=call.get('args',{});reply=parse(t.get('content'));name=t.get('name',call.get('name'));mode=args.get('mode','auto') if name=='execute_python_code' else None
    is_final=name=='execute_python_code' and mode=='final';has_answer=isinstance(reply,dict) and 'answer' in reply and not reply.get('error')
    is_refusal=isinstance(t.get('content'),str) and t['content'].startswith('Not run: V8 admission:')
    evt={'trace_i':i,'call_trace_i':call.get('trace_i'),'name':name,'mode':mode,'refused':is_refusal,'reply_json':isinstance(reply,dict),'reply_keys':list(reply) if isinstance(reply,dict) else None,'error':reply.get('error') if isinstance(reply,dict) else t.get('content') if is_refusal else None,'has_answer':has_answer,'answer_typed_correct':score(reply['answer'],r) if has_answer and is_final else None,'artifact_keys_present':all(k in reply for k in ['answer','evidence','operation','contributing_ids','limitation']) if isinstance(reply,dict) else False,'check_issues':reply.get('issues') if isinstance(reply,dict) and name=='check_fields' else None,'code_sha256':hashlib.sha256(args.get('code','').encode()).hexdigest() if isinstance(args.get('code'),str) else None}
    events.append(evt)
  finals=[x for x in events if x['name']=='execute_python_code' and x['mode']=='final'];validfinals=[x for x in finals if x['has_answer']];correctfinals=[x for x in validfinals if x['answer_typed_correct']]
  observations=rev.get('observations') or [];witness=rev.get('witness');selection=rev.get('selection');bcorrect=score(b['answer'],r) if b is not None else None
  x={'question_id':q,'study_arm':m,'family':c.get('metadata_family'),'operation':c.get('operation_group'),'A_typed_correct':bool(a_correct),'A_terminal_error':failed(a),'A_source_bound':acand['artifact'] is not None,'A_ineligible_reason':acand['diagnostics']['skip_reason'],'status':rev['status'],'B_started':bs is not None,'B_source_bound':b is not None,'B_typed_correct':bcorrect,'B_agrees_A':canonical(b['answer'])==canonical(a['agent_answer']) if b is not None else None,'B_has_witness':witness is not None,'selected_B':mech['selected_B'],'auxiliary_fatal':mech['auxiliary_fatal'],'probe_interactions':len(observations),'eligible_observations':sum(o['witness_eligible'] is True for o in observations),'observation_tools':dict(collections.Counter(o['tool'] for o in observations)),'observation_coverage_reasons':dict(collections.Counter(o.get('coverage_reason','eligible_or_other') for o in observations)),'B_stop':bs['diagnostics']['investigation_stop'] if bs else None,'B_ineligible_reason':bs['diagnostics']['skip_reason'] if bs else None,'aux_stop':aux.get('stop') if aux else None,'automatic_checks':bs['diagnostics']['automatic_checks'] if bs else 0,'aux_sdk':aux.get('model_calls',0) if aux else 0,'aux_tool_calls':aux.get('tool_calls',0) if aux else 0,'aux_tokens':aux.get('tokens',0) if aux else 0,'attempted_final_requests':len(finals),'successful_final_answer_replies':len(validfinals),'scorer_compatible_final_replies':len(correctfinals),'scorer_compatible_unreleased_final':bool(correctfinals) and not bool(bcorrect),'requested_native_tools':dict(collections.Counter(v['name'] for v in calls.values())),'native_final_errors':dict(collections.Counter(str(v['error']) for v in finals if v['error'])),'investigator_assistant_contents':len(messages),'investigator_witness_shape_contents':sum(isinstance(v['parsed'],dict) and set(v['parsed'])=={'obligation_span','observation_id','A_behavior','contradiction'} for v in messages),'selector_present':selection is not None,'selector_choice':selection.get('choice') if isinstance(selection,dict) else None}
  rows.append(x)
  if not a_correct:
   details.append({'summary':x,'question':a['question'],'true_answer':a['true_answer'],'A_answer':a['agent_answer'],'A_code':acand['artifact'].get('code') if acand['artifact'] else None,'witness':witness,'observations':observations,'B_released_artifact':b,'B_snapshot_diagnostics':bs['diagnostics'] if bs else None,'B_released':bs['released'] if bs else None,'B_trace_messages':[{k:v for k,v in t.items() if k!='phase'} for _,t in bt],'event_projection':events,'selection':selection})

def cnt(rs,k):return dict(collections.Counter(str(r.get(k)) for r in rs))
def summarize(rs):
 d={'n':len(rs)}
 for k in ['A_source_bound','A_terminal_error','B_started','B_source_bound','B_typed_correct','B_agrees_A','B_has_witness','selected_B','auxiliary_fatal','status','B_stop','B_ineligible_reason','aux_stop','probe_interactions','eligible_observations','attempted_final_requests','successful_final_answer_replies','scorer_compatible_final_replies','scorer_compatible_unreleased_final','selector_present','selector_choice','automatic_checks','aux_sdk','aux_tool_calls']:
  d[k]=cnt(rs,k)
 d['total_aux_sdk']=sum(x['aux_sdk'] for x in rs);d['total_aux_tools']=sum(x['aux_tool_calls'] for x in rs)
 d['native_final_errors']=dict(sum((collections.Counter(x['native_final_errors']) for x in rs),collections.Counter()))
 d['requested_native_tools']=dict(sum((collections.Counter(x['requested_native_tools']) for x in rs),collections.Counter()))
 return d
summary={'kind':'postterminal_saved120_candidate_funnel','scope':'repaired V8 only; not previous aborted120 or newQA','arms':{},'A_correct':sum(scored[q,'A'] for q in ids),'A_wrong':sum(not scored[q,'A'] for q in ids),'A_errors':sum(failed(index[q,'A']) for q in ids),'used_audits':len(audit_used),'held_source_files':len(manifest['source_sha256']),'held_runtime_files':len(runtime['source_sha256'])}
for m in 'WRC':
 z=[r for r in rows if r['study_arm']==m];summary['arms'][m]={'all120':summarize(z),'A_wrong_including_errors':summarize([r for r in z if not r['A_typed_correct']]),'A_wrong_nonfatal':summarize([r for r in z if not r['A_typed_correct'] and not r['A_terminal_error']]),'A_correct':summarize([r for r in z if r['A_typed_correct']])}
official=load(REC/'official_analysis.json')
for m in 'WRC':
 z=[r for r in rows if r['study_arm']==m];assert sum(r['B_source_bound'] for r in z)==official['mechanisms'][m]['B_available_source_bound'];assert sum(not r['A_typed_correct'] and r['B_typed_correct'] is True for r in z)==official['mechanisms'][m]['O']
assert all(hold_before[n]==sha(ROOT/n) for n in hold_before)
dump('summary.json',summary);dump('rows.json',rows);dump('A_wrong_evidence_private.json',details);dump('hold_and_inputs.json',{'held_before_and_after_unchanged':hold_before,'study_outcomes_sha256':sha(RUN/'study_outcomes.json'),'integrity_sha256':sha(REC/'completion_integrity.json'),'official_analysis_sha256':sha(REC/'official_analysis.json'),'supplementary_sha256':sha(REC/'supplementary/supplementary.json'),'used_audits_sha256':audit_used})
print(json.dumps({'A_correct':summary['A_correct'],'A_wrong':summary['A_wrong'],'A_errors':summary['A_errors'],'used_audits':summary['used_audits'],'modes':{m:{'all':summary['arms'][m]['all120'],'A_wrong_nonfatal':summary['arms'][m]['A_wrong_nonfatal']} for m in 'WRC'}},indent=2))
