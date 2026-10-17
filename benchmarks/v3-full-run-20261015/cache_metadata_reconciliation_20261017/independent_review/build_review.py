from pathlib import Path
prior=Path('/private/tmp/cache-provenance-891-independent-review/review.py').read_text()
body=prior[prior.index('reported={};'):prior.index("report={'status':")]
body=body.replace('5858','17715').replace("'Expected20 journals'","'Expected64 journals'").replace('len(files)==20','len(files)==64')
body=body.replace("original_relative=f'diagnostics/single_agent/", "original_relative=f'diagnostics/multi_agent/")
body=body.replace("round_for_q[qid]=rs[0]['round']", "round_for_q[qid]=rs[0]['round'];need(round_for_q[qid]==selections[qid]['round'],'Association round differs from captured first-clean ledger')")
body=body.replace("{'null':0,'zero':50,'positive':5808,'invalid':0}","{'null':0,'zero':6763,'positive':10952,'invalid':0}")
body=body.replace('==54118775','==27222449')
body=body.replace("need(len(starts)==summary['retained_diagnostic_start_events']==5944 and len(ends)==summary['retained_diagnostic_end_events']==5942 and len(all_success)==summary['retained_successful_end_events']==5921 and len(errors)==summary['retained_error_end_events']==21,'Request coverage counts mismatch')", "need(len(starts)==summary['retained_start_events']==18004 and len(ends)==18004 and len(all_success)==summary['retained_successful_end_events']==17923 and len(errors)==summary['retained_error_end_events']==81,'Request coverage counts mismatch')")
start=body.index("need(len(start_only)==2")
end=body.index('# Negative/limit examples:',start)
body=body[:start]+'''need(not start_only and not end_only and not missing,'Unexpected missing starts/ends/counters')
round_counts={}
for number in range(1,34):
 round_events=[v for k,v in ends.items() if k[0]==number]
 count={'calls':len(round_events),'usage_available':sum(v['outcome']=='success' for v in round_events),'usage_unavailable':sum(v['outcome']=='error' for v in round_events)}
 author=e['rounds'][str(number)]
 need(count=={k:author['all_sealed_raw_attempt_call_counts'].get(k,0) for k in ('calls','usage_available','usage_unavailable')},'Observed events differ from captured raw-call count attestation')
 need(sum(k[0]==number for k in starts)==author['start_events'] and count['usage_available']==author['end_success'] and count['usage_unavailable']==author['end_error'],'Per-round receipt event counts mismatch')
 need(author['recorded_end_count_minus_raw_calls']==author['recorded_success_count_minus_raw_known_usage_calls']==0,'Raw-call count reconciliation receipt differs')
 qcount=sum(selections[qid]['round']==number for qid in selected)
 ccount=sum(len(selected[qid]['per_call_usage']) for qid in selected if selections[qid]['round']==number)
 need(qcount==author['selected_questions'] and ccount==author['selected_calls'],'Per-round accepted cohort count differs')
 round_counts[str(number)]={'selected_questions':qcount,'selected_calls':ccount,'observed_end_counts_equal_attested_all_raw_calls':count}
''' +body[end:]
body=body.replace("need(hashes_valid(CAP)==artifact_manifest and hashes_valid(MEAS)==measurement_manifest,'Captured artifact set changed during review')", "need(hashes_valid(CAP)==artifact_manifest,'Captured artifact set changed during review')")
header='''"""Independent reconstruction using captured V3 numeric metadata ONLY.
Never imports or executes the author's script or inference/scoring/provider code.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import ast,hashlib,json,math,re
from pathlib import Path
CAP=Path('/private/tmp/cache-provenance-v3-complete')
MEAS=CAP
OUT=Path('/private/tmp/cache-provenance-v3-independent-review')
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def semantic(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def hashes_valid(folder):
 entries=read(folder/'artifact_hashes.json')
 for name,v in entries.items():
  p=folder/name;need(p.resolve().is_relative_to(folder.resolve()) and p.is_file(),'Artifact path missing/escaped')
  need(sha(p)==v['sha256'] and p.stat().st_size==v['bytes'],'Captured artifact hash/size mismatch')
 return entries
artifact_manifest=hashes_valid(CAP)
e=read(CAP/'safe_evidence.json');summary=read(CAP/'coverage_summary.json')
need(e['script_sha256']==sha(CAP/'audit_cache_provenance.py'),'Author code hash mismatch')
need(e['association_file_sha256']==sha(CAP/'accepted_call_metadata_associations.jsonl'),'Association hash mismatch')
need(e['measurement_projection_sha256']==sha(CAP/'per_question_measurements.jsonl'),'Numeric projection hash mismatch')
need(e['selected_questions']==2931 and e['selected_calls']==17715,'Complete cohort count mismatch')
need(e['diagnostic_source_sha256_before']==e['diagnostic_source_sha256_after'] and e['source_stable'],'Recorded journal source stability differs')
receipt=read(CAP/'input_provenance/V3_COMPLETE_CONTROLS_PENDING.json')
ledger=read(CAP/'input_provenance/accepted_selection_ledger_at_v3_completion.json')
manifest=read(CAP/'input_provenance/run_manifest_used_for_v3_export.json')
provenance=read(CAP/'input_provenance/verified_provenance.json')
reconciliation=read(CAP/'input_provenance/independent_reconciliation.json')
need(receipt['status']=='V3_COMPLETE_CONTROLS_PENDING' and receipt['questions']==2931 and receipt['v3_complete'] is True,'Complete historical receipt differs')
need(sha(CAP/'input_provenance/V3_COMPLETE_CONTROLS_PENDING.json')==e['first_clean_receipt_sha256'],'Receipt hash binding differs')
need(sha(CAP/'input_provenance/run_manifest_used_for_v3_export.json')==receipt['source_manifest_sha256'],'Historical manifest hash binding differs')
need(sha(CAP/'input_provenance/accepted_selection_ledger_at_v3_completion.json')==receipt['whole_ledger_sha256_at_gate']==receipt['whole_ledger_sha256_after'],'Whole ledger receipt binding differs')
selections=ledger['arms']['multi_agent'];slice_value={'selections':selections,'counts':ledger['counts']['multi_agent']}
need(semantic(slice_value)==receipt['v3_ledger_slice_sha256_before']==receipt['v3_ledger_slice_sha256_after'],'V3 ledger semantic receipt binding differs')
need(ledger['counts']['multi_agent']==receipt['raw_v3_attempt_counts']=={'raw_attempts':3012,'runtime_failures':81,'intrinsic_errors':0},'First-clean attempt counts differ')
need(receipt['sealed_v3_rounds']==list(range(1,34)),'Sealed source round scope differs')
canonical='1c8c974ccff99aae802f698ad043f4d959bb1808b3abee0f9e27c9c3ffa8097c'
need(canonical==e['canonical_sha256']==receipt['v3_canonical_sha256_before']==receipt['v3_canonical_sha256_after'],'Canonical digest binding differs')
need(reconciliation['status']=='PASS' and reconciliation['questions']==2931 and reconciliation['canonical_sha256']==canonical,'Independent reconciliation receipt binding differs')
# Verify all five captured provenance-file digests against the captured fixed-input ledger.
for p in (CAP/'input_provenance').glob('*.json'):
 matches=[v for name,v in e['fixed_completed_v3_input_hashes'].items() if name.endswith('/completed_v3/'+p.name)]
 need(matches==[sha(p)],'Captured provenance-file digest binding differs')
raw_hash_count=0
for name,digest in e['fixed_completed_v3_input_hashes'].items():
 if re.search(r'/attempts/multi_agent/round-[0-9]{4}\\.json$',name):
  need(name in provenance and provenance[name]['sha256']==digest,'Captured raw attempt digest does not match completed inventory');raw_hash_count+=1
 if name.endswith('/scripts/full_benchmark.py'):need(digest==manifest['source_sha256']['scripts/full_benchmark.py'],'Reader source digest differs from manifest')
need(raw_hash_count==33,'Missing raw attempt digest binding')
need(manifest['run_identities']['multi_agent']['code_hash']=='65d2bbfc7da05f97e646acb22d20f0367d7f7a4e1a2a7c8b9c244fd77ef8d409','Frozen inference fingerprint differs')
first=read(CAP/'first_pass/safe_evidence.json')
need(first['script_sha256']==sha(CAP/'first_pass/audit_cache_provenance.py'),'First-pass source hash binding differs')
need(first['association_file_sha256']==e['association_file_sha256'] and first['measurement_projection_sha256']==e['measurement_projection_sha256'],'First-pass association/projection differs')
# Static author-algorithm inspection only; no author code execution.
tree=ast.parse((CAP/'audit_cache_provenance.py').read_text())
functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
need(ast.unparse(functions['signature'].body[0].value)=="(call['prompt'], call['completion'])",'Author match key includes unexpected fields')
event_return=next(n.value for n in ast.walk(functions['event_signature']) if isinstance(n,ast.Return))
need('cached' not in ast.unparse(event_return),'Author event match signature uses cache')
selected={}
allowed={'question_id','patient_fhir_id','latency_seconds','per_call_usage','prompt_tokens','completion_tokens','cached_prompt_tokens','total_tokens','llm_calls'}
for line in (CAP/'per_question_measurements.jsonl').read_text().splitlines():
 r=json.loads(line);qid=r['question_id'];need(set(r)==allowed and qid not in selected and qid in selections,'Unexpected/duplicate projection identity/schema')
 need(re.fullmatch(r'[0-9a-f]{24}',qid) and isinstance(r['patient_fhir_id'],str) and re.fullmatch(r'[A-Za-z0-9._:-]{1,100}',r['patient_fhir_id']),'Unexpected identity representation')
 need(selections[qid]['runtime_clean'] is True and selections[qid]['selection']=='runtime_clean_terminal','Accepted selection is not runtime clean')
 need(selections[qid]['attempt_file']==f"attempts/multi_agent/round-{selections[qid]['round']:04d}.json",'Accepted raw pointer/round differs')
 calls=r['per_call_usage'];need(isinstance(calls,list) and calls and r['llm_calls']==len(calls),'Missing call sequence')
 for c in calls:need(set(c)=={'prompt','completion','cached','usage_available'} and c['usage_available'] is True and all(type(c[k]) is int and c[k]>=0 for k in ('prompt','completion','cached')) and c['cached']<=c['prompt'],'Invalid call measurement schema')
 need(sum(c['prompt'] for c in calls)==r['prompt_tokens'] and sum(c['completion'] for c in calls)==r['completion_tokens'] and sum(c['cached'] for c in calls)==r['cached_prompt_tokens'] and r['total_tokens']==r['prompt_tokens']+r['completion_tokens'],'Projected aggregate token balances differ')
 need(type(r['latency_seconds']) in (int,float) and math.isfinite(r['latency_seconds']) and r['latency_seconds']>=0,'Latency missing/nonfinite')
 selected[qid]=r
need(set(selected)==set(selections) and len(selected)==2931 and sum(len(r['per_call_usage']) for r in selected.values())==17715,'Complete captured numeric cohort differs')
'''
footer='''report={'status':'PASS_COMPLETE_V3_OBSERVED_METADATA_ASSOCIATIONS_WITH_EXPLICIT_LIMITS','reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'source_package':str(CAP),'source_package_artifact_manifest_sha256':sha(CAP/'artifact_hashes.json'),'source_package_artifact_hash_size_checks':len(artifact_manifest),'author_code_sha256':sha(CAP/'audit_cache_provenance.py'),'source_completion_receipt_sha256':sha(CAP/'input_provenance/V3_COMPLETE_CONTROLS_PENDING.json'),'source_manifest_sha256':sha(CAP/'input_provenance/run_manifest_used_for_v3_export.json'),'source_accepted_ledger_sha256':sha(CAP/'input_provenance/accepted_selection_ledger_at_v3_completion.json'),'historical_canonical_sha256_declared':canonical,'captured_provenance_receipt_ledger_manifest_raw_digest_bindings_verified':True,'raw_digest_bindings':raw_hash_count,'first_pass_equivalent_associations_and_projection_verified':True,'numeric_projection_sha256':sha(CAP/'per_question_measurements.jsonl'),'captured_association_sha256':sha(CAP/'accepted_call_metadata_associations.jsonl'),'selected_questions':2931,'selected_calls':17715,'sealed_rounds':33,'retained_journals':len(files),'unique_owned_round_question_sequences':len(all_matches),'global_sequence_candidate_count_distribution':dict(global_candidate_counts),'global_ambiguity_qids':[qid for qid,r in selected.items() if sum(match_stream([(c['prompt'],c['completion']) for c in r['per_call_usage']],key,counter) is not None for key,counter in first_index[(r['per_call_usage'][0]['prompt'],r['per_call_usage'][0]['completion'])])>1],'cache_not_used_in_match_keys':True,'matching_identity_scope':['round','worker PID','process session','consecutive counter','ordered prompt/completion counts'],'all_recorded_association_identities_and_metadata_reproduced':True,'matched_event_overlap':0,'single_call_candidate_counts':dict(single_counts),'matched_provider_duration_sums_fit_question_latency':duration_pass,'same_worker_temporal_overlap_anomalies':timing_anomalies,'retained_event_coverage':{'starts':len(starts),'ends':len(ends),'successes':len(all_success),'errors':len(errors),'starts_without_end':len(start_only),'ends_without_start':len(end_only),'missing_recorded_counters':len(missing)},'all33round_observed_event_counts_match_author_attested_raw_call_counts':True,'per_round_numeric_reconciliation':round_counts,'observed_successful_cache_counts':cache_all,'matched_accepted_cache_counts':cache_match,'normalized_cache_disagreements':0,'matched_token_totals':token_totals,'captured_projection_token_totals':canonical_totals,'all_retained_success_token_totals_operational_only':all_success_totals,'unassociated_success_token_totals_excluded_from_accepted_metrics':unassociated_totals,'unassociated_successful_events':len(all_success)-len(matched_events),'successful_requests_without_request_id':sum(not x for x in request_sets),'duplicate_nonempty_success_request_id_sets':len(request_sets)-len(set(request_sets)),'visible_barrier_tests':barrier_tests,'visible_error_or_missing_counter_barriers_crossed':0,'invisible_log_failure_counterexample':limitation_counterexample,'limitations':['These are unique observed numeric-sequence associations within the owned round/worker/session, not a directly persisted question/request-ID join. Global candidates are counted explicitly and any ambiguity remains recorded.','Visible failure events remain nonmatching sentinels; missing ends/counters break consecutive runs. No accepted match crosses a visible barrier. Best-effort logging failure before counter increment can be entirely invisible, so conditional observed coverage is not unconditional logging or identity proof.','Only captured files were read. The captured canonical/raw byte digests are linked through the original receipt, ledger and inventory; external canonical/raw bytes themselves were not re-read. All-attempt raw-call counts are independently compared to recorded journal counts as captured attestations, rather than recomputed from uncopied raw clinical files.','The conclusion covers the complete historical V3 cohort represented by this capture. It does not extend to the separately continuing baseline, future provider runs, missing logs or verified actual billing.','Cache null is never interpreted as an explicit provider zero, and no metric, estimate, prompt, source, setting or process was altered.'],'review_code_sha256':sha(Path(__file__)),'no_author_script_execution':True,'no_live_reads_or_changes':True,'no_clinical_question_answer_reference_or_correctness_values_read':True,'no_model_provider_scorer_calls':True}
(OUT/'safe_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\\n')
print(json.dumps({'status':report['status'],'questions':2931,'matched_calls':17715,'journals':len(files),'global_candidate_counts':dict(global_candidate_counts),'matched_cache':cache_match,'matched_tokens':token_totals,'event_coverage':report['retained_event_coverage'],'evidence':str(OUT/'safe_review.json'),'evidence_sha256':sha(OUT/'safe_review.json')},indent=2))
'''
Path('/private/tmp/cache-provenance-v3-independent-review/review.py').write_text(header+body+footer)
