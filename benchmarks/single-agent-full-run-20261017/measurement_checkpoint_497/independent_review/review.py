"""Independent captured-checkpoint arithmetic review. Reads ONLY the private capture."""
from collections import Counter
from datetime import datetime,timezone
from decimal import Decimal
import ast,hashlib,json,math,re,statistics
from pathlib import Path
CAP=Path('/private/tmp/single-measurement-checkpoint-497')
OUT=Path('/private/tmp/single-measurement-497-independent-review')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,message):
 if not ok:raise ValueError(message)
def num(v):return type(v) in (int,float) and math.isfinite(v) and v>=0
def near(a,b):
 return a is None and b is None or a is not None and b is not None and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10)
hashes=json.loads((CAP/'artifact_hashes.json').read_text())
verified=[]
for name,m in hashes.items():
 p=CAP/name;need(p.is_file() and p.parent==CAP,'Artifact escape or missing')
 need(sha(p)==m['sha256'] and p.stat().st_size==m['bytes'],'Captured artifact hash/size mismatch')
 verified.append(name)
e=json.loads((CAP/'safe_evidence.json').read_text())
need(e['status']=='PASS' and e['label']=='IN_PROGRESS_NOT_FINAL','Checkpoint label differs')
need(e['measurement_file_sha256']==sha(CAP/'per_question_measurements.jsonl'),'Evidence/measurement hash mismatch')
need(e['script_sha256']==sha(CAP/'audit_measurements.py'),'Evidence/script hash mismatch')
need(e['source_hashes_before']==e['source_hashes_after'] and e['source_stable'] is True,'Recorded source stability mismatch')
need(len(e['source_hashes_before'])==3 and all(re.fullmatch(r'[0-9a-f]{64}',v) for v in e['source_hashes_before'].values()),'Malformed recorded source hashes')
need(re.fullmatch(r'[0-9a-f]{64}',e['first_clean_proof_sha256']),'Malformed proof hash')
need(datetime.fromisoformat(e['started_at'])<=datetime.fromisoformat(e['finished_at']),'Invalid capture interval')
bindings=e['selected_raw_measurement_bindings']
need(set(b['round'] for b in bindings.values())==set(range(1,7)),'Unexpected selected source rounds')
need(sum(b['selected_rows'] for b in bindings.values())==e['accepted_measurements']==497,'Binding counts mismatch')
need(all(re.fullmatch(r'attempts/single_agent/round-[0-9]{4}\.json',p) and re.fullmatch(r'[0-9a-f]{64}',b['sha256']) and type(b['selected_rows']) is int and b['selected_rows']>0 for p,b in bindings.items()),'Malformed raw binding')
rows=[]
for line in (CAP/'per_question_measurements.jsonl').read_text().splitlines():
 need(bool(line.strip()),'Empty measurement line');rows.append(json.loads(line))
fields={'question_id','patient_fhir_id','latency_seconds','prompt_tokens','completion_tokens','total_tokens','cached_prompt_tokens','uncached_prompt_tokens','repeated_prompt_tokens','llm_calls','per_call_usage','provider_usage_available','tool_calls_executed','tool_calls_recorded','estimated_cost_usd','reported_cost_usd'}
callkeys={'prompt','completion','cached','usage_available'}
need(len(rows)==497 and len({r['question_id'] for r in rows})==497,'Captured cohort count/uniqueness mismatch')
metrics=fields-{'question_id','patient_fhir_id','per_call_usage','provider_usage_available'}
need(set(e['metrics'])==metrics,'Evidence metric schema mismatch')
price=e['pricing']['per_million_tokens'];need(price=={'input':.1,'cached_input':.01,'output':.5},'Captured rates differ')
unknown=0;bill_unknown=0;calls_total=0;cost_exact=Decimal(0);percall_cost_exact=Decimal(0)
for r in rows:
 need(set(r)==fields,'Unexpected projection key; possible content field')
 need(isinstance(r['question_id'],str) and bool(re.fullmatch(r'[0-9a-f]{24}',r['question_id'])),'Question ID schema differs')
 need(isinstance(r['patient_fhir_id'],str) and bool(re.fullmatch(r'[A-Za-z0-9._:-]{1,100}',r['patient_fhir_id'])),'Patient ID schema differs')
 need(type(r['provider_usage_available']) is bool and isinstance(r['per_call_usage'],list),'Per-call schema differs')
 calls=r['per_call_usage'];calls_total+=len(calls)
 for c in calls:
  need(isinstance(c,dict) and set(c)==callkeys and type(c['usage_available']) is bool,'Unexpected per-call field')
  need(all(type(c[k]) is int and c[k]>=0 for k in ('prompt','completion','cached')) and c['cached']<=c['prompt'],'Invalid per-call token counts')
 available=all(c['usage_available'] for c in calls);unknown+=not available
 need(r['provider_usage_available']==available,'Availability aggregation mismatch')
 need(r['llm_calls']==len(calls),'LLM call count mismatch')
 for key in ('llm_calls','tool_calls_executed','tool_calls_recorded'):need(type(r[key]) is int and r[key]>=0,'Non-integer call count')
 need(num(r['latency_seconds']),'Missing/nonfinite latency')
 bill_unknown+=r['reported_cost_usd'] is None
 need(r['reported_cost_usd'] is None or num(r['reported_cost_usd']),'Invalid billed-cost value')
 if available:
  p=sum(c['prompt'] for c in calls);o=sum(c['completion'] for c in calls);cached=sum(c['cached'] for c in calls)
  expect={'prompt_tokens':p,'completion_tokens':o,'cached_prompt_tokens':cached,'uncached_prompt_tokens':p-cached,'total_tokens':p+o,'repeated_prompt_tokens':sum(min(calls[i]['prompt'],calls[i-1]['prompt']+calls[i-1]['completion']) for i in range(1,len(calls)))}
  need(all(type(r[k]) is int and r[k]==v for k,v in expect.items()),'Question/per-call token balance mismatch')
  direct=(Decimal(p-cached)*Decimal('0.10')+Decimal(cached)*Decimal('0.01')+Decimal(o)*Decimal('0.50'))/Decimal(1000000)
  percall=sum(((Decimal(c['prompt']-c['cached'])*Decimal('0.10')+Decimal(c['cached'])*Decimal('0.01')+Decimal(c['completion'])*Decimal('0.50'))/Decimal(1000000) for c in calls),Decimal(0))
  need(direct==percall and near(float(direct),r['estimated_cost_usd']),'Cached-token price balance mismatch')
  cost_exact+=direct;percall_cost_exact+=percall
 else:
  need(all(r[k] is None for k in ('prompt_tokens','completion_tokens','cached_prompt_tokens','uncached_prompt_tokens','total_tokens','repeated_prompt_tokens','estimated_cost_usd')),'Unknown usage improperly priced')
need(unknown==e['provider_usage_unknown_rows']==0 and bill_unknown==e['reported_billing_unknown_rows']==497,'Missingness mismatch')
need(calls_total==e['per_call_measurements']==3302,'Total per-call count mismatch')
recomputed={}
for field in sorted(metrics):
 values=[r[field] for r in rows];observed=[v for v in values if num(v)]
 if observed:
  stat={'total':math.fsum(observed),'mean':math.fsum(observed)/len(observed),'median':statistics.median(observed),'p95':statistics.quantiles(observed,n=100,method='inclusive')[94],'sample_stdev':statistics.stdev(observed) if len(observed)>1 else None,'min':min(observed),'max':max(observed)}
 else:stat=dict.fromkeys(('total','mean','median','p95','sample_stdev','min','max'))
 expected=e['metrics'][field];missing=len(values)-len(observed)
 need(expected['n']==len(values) and expected['observed_n']==len(observed) and expected['missing_n']==missing,'Metric missingness count mismatch')
 need(all(near(stat[k],expected['observed_only'][k]) for k in stat),'Observed basic statistic mismatch')
 need(all(near(None if missing else stat[k],expected[k]) for k in stat),'Full-cohort basic statistic mismatch')
 recomputed[field]={'observed_n':len(observed),'missing_n':missing,**stat}
# Static source inspection only: never import or execute the fixed live reader.
tree=ast.parse((CAP/'audit_measurements.py').read_text())
imports=[]
for n in ast.walk(tree):
 if isinstance(n,ast.Import):imports.extend(a.name for a in n.names)
 elif isinstance(n,ast.ImportFrom):imports.append(n.module or '')
need(set(imports)<={'collections','datetime','hashlib','json','math','pathlib','statistics','sys'},'Unexpected imports in captured script')
need(all(e[k] is False for k in ('partial_accuracy_evaluated','answer_reference_prediction_values_inspected','answer_reference_prediction_values_printed','scorer_or_analyzer_calls','provider_calls','comparison_or_extrapolation_performed','repo_or_live_files_modified')),'Capture scope flags differ')
for name,m in hashes.items():need(sha(CAP/name)==m['sha256'],'Captured source changed during review')
report={'status':'PASS_WITH_PROVENANCE_LIMITATION','label':'IN_PROGRESS_NOT_FINAL','captured_checkpoint_directory':str(CAP),'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'captured_artifact_manifest_sha256':sha(CAP/'artifact_hashes.json'),'captured_artifacts_hash_and_size_verified':verified,'evidence_to_measurement_and_script_hash_links_verified':True,'recorded_source_hashes_before_after_equal':True,'recorded_raw_binding_selected_rows_sum':497,'recorded_raw_binding_rounds':[1,2,3,4,5,6],'captured_projection_schema_exact_allowlist':True,'clinical_question_prompt_answer_reference_correctness_content_fields_absent':True,'per_question_rows':497,'distinct_question_ids':497,'per_call_rows':calls_total,'provider_usage_unknown_rows':unknown,'actual_billing_unknown_rows':bill_unknown,'per_call_and_question_token_balances_passed':True,'cached_prompt_charged_once_at_cached_rate':True,'decimal_estimated_cost_total_usd':str(cost_exact),'decimal_per_call_estimated_cost_total_usd':str(percall_cost_exact),'all_basic_statistics_recomputed_and_matched':True,'statistics':recomputed,'limitations':['Only the captured directory was read. Original canonical/ledger/pricing bytes, raw attempts, seals and the separately hashed first-clean proof are not included in this capture; their live byte hashes and first-clean/runtime-exclusion claims cannot be independently revalidated from these captured files alone.','The recorded source hashes are internally consistent, and selected raw binding counts sum to497; this review does not treat those recorded digests as independently re-observed source files.','These497 accepted measurements are a progress checkpoint, not final cohort statistics. The15 excluded runtime attempts are an attested operational count, not independently reconstructible from the497 retained projections.','Dollar values are usage estimates under the captured frozen rates; actual billing is unavailable for all497rows.'],'scope_guarantees':{'fixed497_capture_script_executed':False,'live_canonical_or_run_files_read':False,'model_or_provider_calls':False,'scorer_or_analyzer_calls':False,'live_mutations':False,'external_messages':False}}
(OUT/'safe_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'rows':497,'per_calls':calls_total,'latency_total_seconds':recomputed['latency_seconds']['total'],'latency_mean_seconds':recomputed['latency_seconds']['mean'],'tokens_total':recomputed['total_tokens']['total'],'estimated_cost_total_usd':str(cost_exact),'estimated_cost_mean_usd':recomputed['estimated_cost_usd']['mean'],'schema_and_all_statistics_passed':True,'evidence':str(OUT/'safe_review.json'),'review_sha256':sha(OUT/'safe_review.json')}))
