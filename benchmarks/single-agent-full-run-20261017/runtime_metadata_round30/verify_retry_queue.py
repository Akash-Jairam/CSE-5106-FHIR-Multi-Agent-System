"""Read only the naturally created next shard to verify round30 runtime retries."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json
RUN=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System/output/experiments/gpt-6-luna/v3_full_2931_20261015')
OUT=Path('/private/tmp/single-round30-runtime-sealed-review')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,message):
 if not ok:raise ValueError(message)
reportpath=OUT/'safe_review.json'
report=json.loads(reportpath.read_text())
need(report['status']=='SEALED_ROUND30_METADATA_RECONCILED','Wait for natural round30 seal and ledger adoption')
runtime_ids={r['question_id'] for r in report['runtime_attempts']}
batchpath=RUN/'attempts/single_agent/round-0031.batch.json'
if not batchpath.exists():
 print(json.dumps({'status':'WAITING_FOR_NATURAL_NEXT_BATCH','runtime_retry_count':len(runtime_ids)}));raise SystemExit(0)
batchbytes=batchpath.read_bytes();batch=json.loads(batchbytes)
need(batch['arm']=='single_agent' and batch['round']==31,'Next shard ownership differs')
need(runtime_ids<=set(batch['question_ids']),'Round26 failed question is absent from natural next batch')
need(sha(Path(batch['input']))==batch['input_sha256'],'Next batch shard input changed')
manifest=json.loads((RUN/'run_manifest.json').read_text());expected=manifest['run_identities']['single_agent'];identity=batch['run_identity']
need(set(identity)==set(expected) and all(v==expected[k] for k,v in identity.items() if k!='input_hash'),'Next batch frozen inference identity differs')
ledgerpath=RUN/'acceptance.json';ledgerbytes=ledgerpath.read_bytes();ledger=json.loads(ledgerbytes)
for qid in runtime_ids:need(ledger['arms']['single_agent'].get(qid,{}).get('attempt_file')!='attempts/single_agent/round-0030.json','Runtime-tainted round30 attempt selected')
need(not set(batch['question_ids'])&set(ledger['arms']['single_agent']),'Next batch schedules an already accepted question')
need(not set(batch['question_ids'])&{'05a8454a6a71435a557db2cc','0eb69f6e47ab63c055e2df90'},'Next batch schedules a parked provider question')
result={'status':'PASS','captured_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'NATURAL_ROUND31_RUNTIME_RETRY_SCHEDULING_METADATA','round30_failed_attempts_excluded_from_ledger':True,'round30_runtime_question_ids':sorted(runtime_ids),'all_runtime_ids_queued_in_round31':True,'next_batch':{'relative_path':str(batchpath.relative_to(RUN)),'sha256':hashlib.sha256(batchbytes).hexdigest(),'shard_input_sha256':batch['input_sha256'],'frozen_identity_matches':True,'question_count':len(batch['question_ids']),'snapshot_not_round31_final_seal':True},'acceptance_snapshot_sha256':hashlib.sha256(ledgerbytes).hexdigest(),'no_accepted_or_parked_question_reruns_in_next_batch':True,'round30_review_sha256':sha(reportpath),'review_code_sha256':sha(Path(__file__)),'no_scoring_inference_or_live_mutation':True}
(OUT/'retry_queue_status.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'PASS','runtime_retry_count':len(runtime_ids),'queued_round':31,'evidence':str(OUT/'retry_queue_status.json'),'sha256':sha(OUT/'retry_queue_status.json')},indent=2))
