"""Answer-blind seal/ledger boundary audit; all writes stay in this private folder."""
import contextlib,hashlib,io,json
from datetime import datetime,timezone
from pathlib import Path
RUN=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System/output/experiments/gpt-6-luna/v3_full_2931_20261015')
OUT=Path('/private/tmp/single-round21-runtime-sealed-review')
ROUND=21
TARGETS=()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,reason):
 if not ok:raise ValueError(reason)
sealpath=RUN/'attempts/single_agent/round-0021.batch.seal.json'
if not sealpath.exists():
 print(json.dumps({'status':'WAITING_FOR_NATURAL_SEAL','round':ROUND}));raise SystemExit(0)
batchpath=RUN/'attempts/single_agent/round-0021.batch.json';result=RUN/'attempts/single_agent/round-0021.json';identity=Path(str(result)+'.run.json')
seal=json.loads(sealpath.read_text());batch=json.loads(batchpath.read_text());manifest=json.loads((RUN/'run_manifest.json').read_text())
need(batch['round']==ROUND and batch['arm']=='single_agent' and Path(batch['output']).resolve()==result.resolve(),'Sealed batch ownership differs')
need(sha(batchpath)==seal['batch_sha256'] and sha(result)==seal['result_sha256'] and sha(identity)==seal['identity_sha256'],'Immutable seal hash mismatch')
need(sha(Path(batch['input']))==batch['input_sha256'],'Sealed shard input hash mismatch')
saved=json.loads(identity.read_text());expected=manifest['run_identities']['single_agent']
need(saved==batch['run_identity'] and set(saved)==set(expected) and all(v==expected[k] for k,v in saved.items() if k!='input_hash'),'Frozen inference identity differs')
basepath=OUT/'sealed_classification_base.json'
if not basepath.exists():
 source=Path('/private/tmp/single-round21-runtime-check/review.py').read_text().replace('single-round21-runtime-check','single-round21-runtime-sealed-review')
 (OUT/'classification_review.py').write_text(source)
 with contextlib.redirect_stdout(io.StringIO()):exec(compile(source,str(OUT/'classification_review.py'),'exec'),{'__name__':'sealed_metadata_review','__file__':str(OUT/'classification_review.py')})
 basepath.write_bytes((OUT/'safe_review.json').read_bytes())
report=json.loads(basepath.read_text())
need(report['round21_result_snapshot']['sha256']==seal['result_sha256'],'Classified source differs from immutable seal')
need(not report['saved_classifications'].get('pending',0) and not report['saved_classifications'].get('attention',0),'Sealed classifications include pending/attention; inspect safe metadata')
rows=json.loads(result.read_text());qids={r['question_id'] for r in rows}
need(len(qids)==len(rows)==len(batch['question_ids']) and qids==set(batch['question_ids']),'Sealed question cohort differs')
runtime_ids={r['question_id'] for r in report['runtime_attempts']};accepted_ids=qids-runtime_ids
need(len(accepted_ids)==report['saved_classifications']['accepted'],'Accepted classification cohort count differs')
ledgerbytes=(RUN/'acceptance.json').read_bytes();ledger=json.loads(ledgerbytes);choices=ledger['arms']['single_agent'];attempt_file='attempts/single_agent/round-0021.json'
clean_adopted={qid for qid in accepted_ids if choices.get(qid,{}).get('attempt_file')==attempt_file and choices[qid].get('runtime_clean') is True}
for qid in runtime_ids:need(choices.get(qid,{}).get('attempt_file')!=attempt_file,'Runtime-tainted round21 attempt selected')
recoveries=[]
for prior in report['round20_runtime_recoveries']:
 qid=prior['question_id'];need(qid in TARGETS and prior['first_runtime_clean_in_round21'],'Expected prior failure is not first-clean in sealed round21')
 entry=choices.get(qid);before=sum(p['classification']=='runtime' for p in prior['prior_sealed_attempts'])
 adopted=bool(entry and entry.get('attempt_file')==attempt_file and entry.get('round')==ROUND and entry.get('runtime_clean') is True and entry.get('selection')=='runtime_clean_terminal' and entry.get('runtime_failures_before')==before)
 recoveries.append({'question_id':qid,'first_clean_in_sealed_round21':True,'ledger_adopted':adopted,'selection_metadata_only':entry})
need(len(recoveries)==len(TARGETS),'Expected recovery cohort incomplete')
complete=len(clean_adopted)==len(accepted_ids) and all(r['ledger_adopted'] for r in recoveries)
observation={'at_utc':datetime.now(timezone.utc).isoformat(),'accepted_snapshot_sha256':hashlib.sha256(ledgerbytes).hexdigest(),'sealed_clean_count':len(accepted_ids),'clean_adopted_count':len(clean_adopted),'prior_round20_recovery_adopted_count':sum(r['ledger_adopted'] for r in recoveries),'ledger_transition_complete':complete}
with (OUT/'ledger_observations.jsonl').open('a') as f:f.write(json.dumps(observation)+'\n')
report.update(status='SEALED_ROUND21_METADATA_RECONCILED' if complete else 'SEALED_ROUND21_LEDGER_TRANSITION_PENDING',scope='IMMUTABLE_ROUND21_RUNTIME_AND_NO_PRIOR_RUNTIME_RECOVERIES',reconciled_at_utc=observation['at_utc'],natural_seal={'batch_sha256':sha(batchpath),'seal_sha256':sha(sealpath),'result_sha256':sha(result),'identity_sha256':sha(identity),'shard_input_sha256':batch['input_sha256'],'frozen_identity_matches':True,'question_count':len(qids)},final_recoveries=recoveries,acceptance_ledger_observation=observation,all_sealed_runtime_attempts_excluded_from_ledger=True,review_code_sha256=sha(Path(__file__)),classification_review_code_sha256=sha(OUT/'classification_review.py'))
for r in report['round20_runtime_recoveries']:r['selection_persistence']='Immutable sealed raw classification verified; adoption reported separately in final_recoveries.'
inventory=report.pop('referenced_audit_hashes')
(OUT/'audit_hash_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
report['audit_integrity']={'restored_and_hash_verified_count':len(inventory),'inventory_file':'audit_hash_inventory.json','inventory_sha256':sha(OUT/'audit_hash_inventory.json')}
report['non_HTTP500_runtime_signatures']=[failure for failure in report['runtime_attempts'] if failure.get('new_provider_policy_rejection') or failure.get('HTTP_statuses')!=[500]]
(OUT/'safe_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'sealed_classifications':report['saved_classifications'],'runtime_attempts':report['runtime_attempts'],'new_provider_policy_rejections':report['provider_policy_rejection_count_in_captured_round21'],'recoveries':recoveries,'all_sealed_clean_results_adopted':complete,'evidence':str(OUT/'safe_review.json'),'evidence_sha256':sha(OUT/'safe_review.json')},indent=2))
