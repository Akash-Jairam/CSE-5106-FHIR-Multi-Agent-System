"""Independent metadata-only reconstruction from private captured artifacts only."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,math,re
from pathlib import Path
CAP=Path('/private/tmp/cache-provenance-checkpoint-891')
MEAS=Path('/private/tmp/single-measurement-checkpoint-891')
OUT=Path('/private/tmp/cache-provenance-891-independent-review')
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def hashes_valid(folder):
 entries=read(folder/'artifact_hashes.json')
 for name,v in entries.items():
  p=folder/name;need(p.resolve().is_relative_to(folder.resolve()) and p.is_file(),'Artifact path missing/escaped')
  need(sha(p)==v['sha256'] and p.stat().st_size==v['bytes'],'Captured artifact hash/size mismatch')
 return entries
artifact_manifest=hashes_valid(CAP);measurement_manifest=hashes_valid(MEAS)
e=read(CAP/'safe_evidence.json');summary=read(CAP/'coverage_summary.json');m_e=read(MEAS/'safe_evidence.json');proof=read(MEAS/'first_clean_proof.json')
need(e['script_sha256']==sha(CAP/'audit_cache_provenance.py'),'Receipt code hash mismatch')
need(e['association_file_sha256']==sha(CAP/'accepted_call_metadata_associations.jsonl'),'Receipt association hash mismatch')
for name,digest in e['fixed_capture_hashes'].items():need(sha(MEAS/name)==digest,'Fixed numeric capture binding mismatch')
need(sha(MEAS/'first_clean_proof.json')==e['first_clean_proof_sha256'],'Captured first-clean proof binding mismatch')
need(proof['selected_accepted']==891 and proof['canonical_first_clean_verified'] and proof['accepted_reruns']==0,'First-clean receipt summary differs')
need(e['diagnostic_source_sha256_before']==e['diagnostic_source_sha256_after'],'Recorded diagnostic capture stability differs')
need(m_e['source_hashes_before']==m_e['source_hashes_after'] and m_e['source_stable'],'Recorded measurement capture stability differs')
need(e['selected_questions']==m_e['accepted_measurements']==891 and e['selected_calls']==m_e['per_call_measurements']==5858,'Capture count binding differs')
sealed_summary={r['round']:r for r in proof['sealed_batches']}
for detail in m_e['selected_raw_measurement_bindings'].values():need(detail['sha256']==sealed_summary[detail['round']]['result_sha256'],'Captured raw digest differs from copied first-clean proof')
selected={}
allowed={'question_id','patient_fhir_id','latency_seconds','prompt_tokens','completion_tokens','total_tokens','cached_prompt_tokens','uncached_prompt_tokens','repeated_prompt_tokens','llm_calls','per_call_usage','provider_usage_available','tool_calls_executed','tool_calls_recorded','estimated_cost_usd','reported_cost_usd'}
for line in (MEAS/'per_question_measurements.jsonl').read_text().splitlines():
 r=json.loads(line);need(set(r)==allowed and r['question_id'] not in selected,'Unexpected/duplicate numeric projection')
 for call in r['per_call_usage']:need(set(call)=={'prompt','completion','cached','usage_available'} and all(type(call[k]) is int and call[k]>=0 for k in ('prompt','completion','cached')),'Unexpected per-call projection')
 selected[r['question_id']]=r
need(len(selected)==891 and sum(len(r['per_call_usage']) for r in selected.values())==5858,'Captured numeric sequence cohort differs')
reported={};reported_by_q=defaultdict(list)
association_keys={'question_id','call_index','round','association','file','pid','process_session','counter','started_utc','finished_utc','request_ids','usage','canonical_normalized_cached'}
for line in (CAP/'accepted_call_metadata_associations.jsonl').read_text().splitlines():
 r=json.loads(line);need(set(r)==association_keys,'Unexpected association field')
 k=(r['question_id'],r['call_index']);need(k not in reported,'Duplicate declared question/call association');reported[k]=r;reported_by_q[r['question_id']].append(r)
need(set(reported_by_q)==set(selected),'Association question cohort differs')
round_for_q={}
for qid,rs in reported_by_q.items():
 rs.sort(key=lambda r:r['call_index']);need([r['call_index'] for r in rs]==list(range(len(selected[qid]['per_call_usage']))),'Call indexes incomplete')
 need(len({r['round'] for r in rs})==1,'Question spans declared rounds');round_for_q[qid]=rs[0]['round']
# Reconstruct all end-event streams without discarding failures or unknown usage.
streams=defaultdict(dict);starts={};ends={};files=[];all_success=[];errors=[];timing_anomalies=[]
for p in sorted((CAP/'raw_diagnostics').glob('round-*/*.jsonl')):
 number=int(p.parent.name.split('-')[1]);raw=p.read_bytes();need(raw.endswith(b'\n'),'Truncated copied journal')
 original_relative=f'diagnostics/single_agent/round-{number:04d}/{p.name}'
 matches=[h for name,h in e['diagnostic_source_sha256_before'].items() if name.endswith('/'+original_relative)]
 need(matches==[sha(p)],'Copied journal does not match captured source digest')
 files.append({'relative_path':str(p.relative_to(CAP)),'sha256':sha(p),'bytes':len(raw)})
 for line in raw.splitlines():
  v=json.loads(line);event=v.get('event');need(event in ('start','end'),'Unexpected journal event')
  need(set(v)<= {'version','pid','ppid','counter','process_session','started_utc','event','client','fd_count','finished_utc','duration_seconds','outcome','exception_chain','usage','http_status','request_ids'},'Unexpected journal content field')
  if v.get('usage') is not None:need(set(v['usage'])<= {'prompt_tokens','completion_tokens','total_tokens','cached_prompt_tokens'},'Unexpected usage content field')
  need(type(v.get('pid')) is int and type(v.get('counter')) is int and v['counter']>0 and isinstance(v.get('process_session'),str),'Malformed request identity')
  k=(number,v['pid'],v['process_session'],v['counter']);v={key:v.get(key) for key in ('event','pid','process_session','counter','started_utc','finished_utc','outcome','duration_seconds','http_status','request_ids','usage')};v.update(round=number,file=original_relative)
  table=starts if event=='start' else ends;need(k not in table,'Duplicate recorded request event');table[k]=v
  if event=='end':
   streams[k[:3]][k[3]]=v
   if v['outcome']=='success':all_success.append(v)
   else:need(v['outcome']=='error','Unknown end outcome');errors.append(v)
need(len(files)==20,'Expected20 journals')
start_only=set(starts)-set(ends);end_only=set(ends)-set(starts);need(not end_only,'Recorded ends lack starts')
missing=[]
for key in {k[:3] for k in starts}|set(streams):
 counters={k[3] for k in starts if k[:3]==key}|set(streams[key]);missing.extend((key,c) for c in range(1,max(counters)+1) if c not in counters)
for k,event in ends.items():
 start=starts[k];need(start['started_utc']==event['started_utc'],'Start/end timestamp binding differs')
 need(datetime.fromisoformat(event['finished_utc'])>=datetime.fromisoformat(event['started_utc']),'Negative UTC request interval')
 need(type(event['duration_seconds']) in (int,float) and math.isfinite(event['duration_seconds']) and event['duration_seconds']>=0,'Invalid duration')
for key,counters in streams.items():
 vals=[counters[i] for i in sorted(counters)]
 for a,b in zip(vals,vals[1:]):
  if datetime.fromisoformat(b['started_utc'])<datetime.fromisoformat(a['finished_utc']):timing_anomalies.append({'round':key[0],'pid':key[1],'counter_before':a['counter'],'counter_after':b['counter']})
need(not timing_anomalies,'Unexpected concurrent/overlapping same-worker provider calls')
def sig_event(event):
 u=event.get('usage') or {}
 return (u.get('prompt_tokens'),u.get('completion_tokens')) if event.get('outcome')=='success' else None
def match_stream(targets,key,first):
 # Counter-addressed lookup deliberately leaves errors/missing ends as barriers.
 chunk=[]
 for offset,target in enumerate(targets):
  counter=first+offset;event=streams[key].get(counter);identity=key+(counter,)
  if event is None or identity not in starts or sig_event(event)!=target:return None
  chunk.append(event)
 return chunk
first_index=defaultdict(list);within_round_signature=defaultdict(Counter)
for key,values in streams.items():
 for counter,event in values.items():
  signature=sig_event(event)
  if signature is not None:first_index[signature].append((key,counter));within_round_signature[key[0]][signature]+=1
candidate_table=read(CAP/'per_question_candidate_counts.json');need(set(candidate_table)==set(selected),'Candidate table cohort differs')
all_matches={};claimed={};single_counts=Counter();global_candidate_counts=Counter();matched_events=[];duration_pass=0
for qid,r in selected.items():
 targets=[(c['prompt'],c['completion']) for c in r['per_call_usage']];need(bool(targets),'Empty question token sequence')
 matches=[]
 for key,counter in first_index[targets[0]]:
  chunk=match_stream(targets,key,counter)
  if chunk is not None:matches.append((key,chunk))
 global_candidate_counts[len(matches)]+=1
 scoped=[(key,chunk) for key,chunk in matches if key[0]==round_for_q[qid]]
 need(len(scoped)==1,'Expected unique per-round independent sequence match')
 need(candidate_table[qid]=={'round':round_for_q[qid],'calls':len(targets),'observed_sequence_candidates':1},'Per-question candidate count mismatch')
 key,chunk=scoped[0];need(sum(v['duration_seconds'] for v in chunk)<=r['latency_seconds']+.1,'Provider duration exceeds question latency');duration_pass+=1
 for target in targets:
  n=within_round_signature[key[0]][target];single_counts['single_call_unique' if n==1 else 'single_call_ambiguous' if n>1 else 'single_call_no_candidate']+=1
 for i,event in enumerate(chunk):
  eventkey=key+(event['counter'],);need(eventkey not in claimed,'Accepted questions overlap one provider event');claimed[eventkey]=(qid,i);matched_events.append(event)
  saved=reported[(qid,i)]
  need(all(saved[k]==event[k] for k in ('round','file','pid','process_session','counter','started_utc','finished_utc','request_ids','usage')),'Rebuilt match differs from captured association')
  call=r['per_call_usage'][i];need((event['usage']['prompt_tokens'],event['usage']['completion_tokens'])==(call['prompt'],call['completion']),'Sequence token values differ')
  need(event['usage']['total_tokens']==call['prompt']+call['completion'],'Per-call total token balance differs')
  need(type(event['usage']['cached_prompt_tokens']) is int and 0<=event['usage']['cached_prompt_tokens']<=call['prompt'],'Per-call cached subset differs')
  need(saved['canonical_normalized_cached']==call['cached'],'Association canonical cache differs')
 all_matches[qid]=(key,chunk)
need(len(claimed)==5858,'Matched call count differs')
def cache_counts(events):
 c=Counter(null=0,zero=0,positive=0,invalid=0)
 for v in events:
  x=(v.get('usage') or {}).get('cached_prompt_tokens');c['null' if x is None else 'invalid' if type(x) is not int or x<0 else 'zero' if x==0 else 'positive']+=1
 return dict(c)
cache_all=cache_counts(all_success);cache_match=cache_counts(matched_events)
need(cache_all==e['successful_event_cache_counts']==summary['successful_event_cache_counts'],'Observed cache presence counts mismatch')
need(cache_match=={'null':0,'zero':50,'positive':5808,'invalid':0},'Matched cache counts differ')
need(single_counts==Counter({k:v for k,v in e['join_candidate_counts'].items() if k.startswith('single_call_')}),'Single signature ambiguity count differs')
token_totals={k:sum(v['usage'][k] for v in matched_events) for k in ('prompt_tokens','completion_tokens','total_tokens','cached_prompt_tokens')}
canonical_totals={k:sum(r[k] for r in selected.values()) for k in ('prompt_tokens','completion_tokens','total_tokens','cached_prompt_tokens')}
all_success_totals={k:sum(v['usage'][k] for v in all_success) for k in token_totals}
unassociated_totals={k:all_success_totals[k]-token_totals[k] for k in token_totals}
need(token_totals==canonical_totals,'Matched/captured token aggregate mismatch')
for qid,(key,chunk) in all_matches.items():
 for call,event in zip(selected[qid]['per_call_usage'],chunk):need(event['usage']['cached_prompt_tokens']==call['cached'],'Cached detail disagrees after independent matching')
need(token_totals['cached_prompt_tokens']==summary['accepted_observed_sequence_association']['matched_cached_token_sum']==54118775,'Cached total mismatch')
need(len(starts)==summary['retained_diagnostic_start_events']==5944 and len(ends)==summary['retained_diagnostic_end_events']==5942 and len(all_success)==summary['retained_successful_end_events']==5921 and len(errors)==summary['retained_error_end_events']==21,'Request coverage counts mismatch')
need(len(start_only)==2 and not missing and all(k[0]==1 and k[3]==max(c for kk in starts if kk[:3]==k[:3] for c in [kk[3]]) for k in start_only),'Missing request barriers differ')
need(sum(b['selected_rows'] for b in m_e['selected_raw_measurement_bindings'].values())==891,'Captured raw source binding count differs')
for rnd,count in Counter(round_for_q.values()).items():
 need(count==sum(b['selected_rows'] for b in m_e['selected_raw_measurement_bindings'].values() if b['round']==rnd),'Question round-association count differs from source capture binding')
# Negative/limit examples: same counter-keyed matching rule, synthetic metadata only.
def example_match(records,targets):
 for first in sorted(records):
  if all((first+i) in records and records[first+i]==target for i,target in enumerate(targets)):return True
 return False
A=(100,10);B=(200,20)
barrier_tests={'visible_success_contiguous_matches':example_match({1:A,2:B},[A,B]),'explicit_failed_end_blocks':not example_match({1:A,2:None,3:B},[A,B]),'missing_end_counter_blocks':not example_match({1:A,3:B},[A,B]),'missing_whole_recorded_counter_blocks':not example_match({1:A,3:B},[A,B])}
need(all(barrier_tests.values()),'Independent barrier test failed')
limitation_counterexample={'canonical_target_sequence':[[100,10],[200,20]],'actual_events':'Target callA is logged atcounter1; target callB is unlogged before counter increment; next question callB is logged atcounter2.','retained_sequence':[[100,10],[200,20]],'matcher_observes_unique_contiguous_sequence':example_match({1:A,2:B},[A,B]),'why_not_identity_proof':'Without a stored question/request join, unique retained numeric sequences do not rule out a completely invisible logging failure and a coincidentally matching neighboring call.'}
request_sets=[tuple(sorted((v.get('request_ids') or {}).items())) for v in all_success]
need(all(request_sets) and len(set(request_sets))==len(request_sets),'Successful request identifier uniqueness differs')
need(hashes_valid(CAP)==artifact_manifest and hashes_valid(MEAS)==measurement_manifest,'Captured artifact set changed during review')
report={'status':'PASS_OBSERVED_METADATA_ASSOCIATIONS_WITH_EXPLICIT_LIMITS','reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'inputs':{'cache_package_manifest_sha256':sha(CAP/'artifact_hashes.json'),'numeric_package_manifest_sha256':sha(MEAS/'artifact_hashes.json'),'cache_package_script_sha256':sha(CAP/'audit_cache_provenance.py'),'captured_association_sha256':sha(CAP/'accepted_call_metadata_associations.jsonl'),'numeric_measurements_sha256':sha(MEAS/'per_question_measurements.jsonl'),'captured_first_clean_proof_sha256':sha(MEAS/'first_clean_proof.json')},'artifact_hash_size_checks':len(artifact_manifest)+len(measurement_manifest),'journal_count':len(files),'selected_questions':891,'selected_calls':5858,'independently_unique_scoped_sequences':len(all_matches),'global_sequence_candidate_count_distribution':dict(global_candidate_counts),'captured_raw_digest_bindings_match_copied_first_clean_proof':True,'journal_and_usage_keys_metadata_allowlist_verified':True,'matched_per_call_total_and_cached_subset_balances_verified':True,'cache_excluded_from_all_match_keys':True,'matching_key_fields':['round','pid','process_session','consecutive counter','ordered prompt_tokens/completion_tokens sequence'],'association_file_identity_values_all_reproduced':True,'matched_event_overlap':0,'matched_provider_duration_within_question_latency_count':duration_pass,'same_worker_temporal_overlap_anomalies':timing_anomalies,'single_call_candidate_counts':dict(single_counts),'retained_request_coverage':{'starts':len(starts),'ends':len(ends),'successes':len(all_success),'errors':len(errors),'starts_without_end':len(start_only),'ends_without_start':len(end_only),'missing_recorded_counters':len(missing),'start_only_entries_are_terminal_round1_events':True},'observed_success_cache_counts':cache_all,'matched_accepted_cache_counts':cache_match,'matched_token_totals':token_totals,'all_retained_success_token_totals_operational_only':all_success_totals,'unassociated_success_token_totals_excluded_from_accepted_metrics':unassociated_totals,'captured_numeric_token_totals':canonical_totals,'successful_unassociated_events':len(all_success)-len(matched_events),'matched_normalized_cache_disagreements':0,'successful_requests_without_request_id':sum(not x for x in request_sets),'duplicate_nonempty_success_request_id_sets':len(request_sets)-len(set(request_sets)),'visible_barrier_synthetic_tests':barrier_tests,'invisible_failure_counterexample':limitation_counterexample,'limits':['Observed failures remain explicit nonmatching barriers; absent ends/recorded counters break consecutive sequences. No accepted match crosses such a visible barrier.','Matching is unique relative to retained numeric events. It is inferred, not a directly persisted question/request-ID join; a fully invisible logging failure before counter increment can still defeat identity inference, as the synthetic counterexample shows.','This review rechecks captured numeric rows, copied first-clean receipt, and all20 copied journals. Original sealed raw attempts/inputs/ledger are not copied here, so external raw-to-question provenance remains attested by the captured proof and capture receipts rather than re-read from live files.','No unconditional complete-logging/cache-availability claim follows; the finding applies only to891 questions/5858 matched observed calls at this checkpoint. It does not establish coverage for V3 or later baseline calls.','No actual provider bill is verified, and no frozen metric, cache value, estimate, prompt, source, setting, or schedule is changed.'],'no_live_reads_or_changes':True,'no_question_answer_reference_clinical_or_correctness_values_read':True,'no_model_scorer_provider_calls':True}
(OUT/'safe_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'journals':len(files),'unique_question_sequences':len(all_matches),'matched_calls':len(claimed),'global_candidate_counts':dict(global_candidate_counts),'observed_cache':cache_all,'matched_cache':cache_match,'token_totals':token_totals,'starts_without_end':len(start_only),'visible_failure_or_missing_counter_barriers_crossed':0,'evidence':str(OUT/'safe_review.json'),'evidence_sha256':sha(OUT/'safe_review.json')},indent=2))
