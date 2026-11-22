"""Read-only safe post-seal transport aggregate. No messages/configs/API calls."""
from collections import Counter
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
RUN=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System/output/experiments/gpt-6-luna/v3_full_2931_20261015')
OUT=Path('/private/tmp/single-round23-transport-health-evidence')
ORIGINAL=Path('/private/tmp/single-round23-runtime-progress-1133/safe_progress.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
original=ORIGINAL.read_bytes();snapshot=json.loads(original)
(OUT/'safe_progress_live_snapshot_1133.json').write_bytes(original)
need(sha(ORIGINAL)==sha(OUT/'safe_progress_live_snapshot_1133.json'),'Original snapshot changed while preserving')
sealpath=RUN/'attempts/single_agent/round-0023.batch.seal.json';seal=json.loads(sealpath.read_text())
need(seal['result_sha256']==sha(RUN/'attempts/single_agent/round-0023.json'),'Batch result changed after seal')
manifest=json.loads((RUN/'run_manifest.json').read_text());adapter=RUN/'runtime/azure_adapter.py'
need(sha(adapter)==manifest['pinned_files_sha256']['runtime/azure_adapter.py'],'Frozen diagnostic adapter changed')
workers=[];source_bindings={};total=Counter()
def safe_end(e):
 return {k:e.get(k) for k in ('pid','counter','started_utc','finished_utc','duration_seconds','outcome','http_status','fd_count')}|{'exception_classes':[(v.get('type') or {}).get('class') for v in e.get('exception_chain',[])]}
for p in sorted((RUN/'diagnostics/single_agent/round-0023').glob('*.jsonl')):
 raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest();events=[]
 for line in raw.splitlines():
  e=json.loads(line);need(set(e)<={'version','pid','ppid','counter','process_session','started_utc','event','client','fd_count','finished_utc','duration_seconds','outcome','exception_chain','usage','http_status','request_ids'},'Unexpected diagnostic content field');events.append(e)
 need(sha(p)==digest,'Post-seal journal bytes changed during read')
 ends=[e for e in events if e.get('event')=='end'];errors=[e for e in ends if e.get('outcome')=='error'];success=[e for e in ends if e.get('outcome')=='success'];total.update(e.get('outcome') for e in ends)
 timeouts=[e for e in errors if any('Timeout' in ((v.get('type') or {}).get('class') or '') for v in e.get('exception_chain',[]))]
 last=max(timeouts,key=lambda e:e['counter']) if timeouts else None
 later=[e for e in success if last and e['counter']>last['counter'] and e.get('process_session')==last.get('process_session') and (e.get('client') or {}).get('client_id')==(last.get('client') or {}).get('client_id')]
 fd=[e['fd_count'] for e in events if type(e.get('fd_count')) is int]
 workers.append({'pid':ends[0]['pid'],'process_session':ends[0]['process_session'],'starts':sum(e.get('event')=='start' for e in events),'ends':len(ends),'successes':len(success),'errors':len(errors),'timeout_errors_safe_metadata':[safe_end(e) for e in timeouts],'SDK_exception_class_counts':dict(Counter((e.get('exception_chain',[{}])[0].get('type') or {}).get('class') for e in errors)),'FD_min':min(fd),'FD_max':max(fd),'any_SDK_closed':any((e.get('client') or {}).get('sdk_closed') is True for e in events),'any_HTTP_closed':any((e.get('client') or {}).get('http_closed') is True for e in events),'same_process_session_client_successes_after_last_timeout':len(later),'first_same_client_success_after_last_timeout':safe_end(later[0]) if later else None,'last_recorded_success':safe_end(success[-1]) if success else None})
 source_bindings[p.name]={'live_snapshot_hash_at_1133':snapshot['diagnostic_source_hashes'].get(p.name),'post_batch_seal_stable_observation_sha256':digest,'post_batch_seal_bytes':len(raw),'post_batch_seal_file_relative_path':str(p.relative_to(RUN)),'snapshot_hash_scope':'Hash of bytes captured while this journal was still live; not an immutable/final journal hash.','post_seal_hash_scope':'Stable bytes independently observed after batch sealing. Diagnostic journal is not covered by batch.seal result/batch/identity hash fields.'}
need(len(workers)==2 and total['error']==6 and all(w['same_process_session_client_successes_after_last_timeout']>0 for w in workers),'Expected same-client recovery evidence absent')
report={'status':'READ_ONLY_TRANSPORT_GAP_AND_SAME_CLIENT_RECOVERY_DOCUMENTED','captured_at_utc':datetime.now(timezone.utc).isoformat(),'initial_snapshot_file':'safe_progress_live_snapshot_1133.json','initial_snapshot_sha256':sha(OUT/'safe_progress_live_snapshot_1133.json'),'initial_snapshot_recorded_at_utc':snapshot['captured_at_utc'],'initial_snapshot_journal_hashes_are_provisional':True,'batch_seal_sha256':sha(sealpath),'batch_result_sha256':seal['result_sha256'],'journal_hashes_are_not_covered_by_batch_seal':True,'frozen_adapter_sha256':sha(adapter),'source_bindings':source_bindings,'workers':workers,'post_seal_provider_end_outcomes':dict(total),'finding':'ConnectTimeout gap ended with subsequent successful requests in the same worker process/session and SDK client; recorded FD counts and closed flags show no client-close or FD-exhaustion signal. No restart or changes were performed by this audit.','limits':['Safe journal metadata does not locate the network/provider cause.','No direct diagnostic-to-question join is claimed.','Initial live journal snapshot hashes are not final sealed journal hashes. The separately recorded post-batch-seal hashes bind bytes observed stable here; the batch seal itself excludes journals.'],'source_reconstruction_note':'The initial snapshot was captured by inline Python. capture_snapshot_source_reconstructed.py reproduces that executed tool-cell text and was saved afterward; no pre-capture source-file checksum is claimed. It was not re-executed.','reconstructed_initial_source_sha256':sha(OUT/'capture_snapshot_source_reconstructed.py'),'final_review_source_sha256':sha(Path(__file__)),'no_raw_messages_clinical_answers_references_or_credentials_output':True,'no_provider_model_scorer_calls_or_run_changes':True}
(OUT/'safe_transport_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'outcomes':dict(total),'workers':[{'pid':w['pid'],'FD_min':w['FD_min'],'FD_max':w['FD_max'],'same_client_successes_after_last_timeout':w['same_process_session_client_successes_after_last_timeout']} for w in workers],'safe_transport_review_sha256':sha(OUT/'safe_transport_review.json'),'initial_snapshot_sha256':report['initial_snapshot_sha256']}))
