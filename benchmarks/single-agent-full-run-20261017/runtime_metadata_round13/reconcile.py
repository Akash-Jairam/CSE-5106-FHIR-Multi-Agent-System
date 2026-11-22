"""Read-only natural-seal reconciliation of runtime metadata and one recovery."""
import contextlib,hashlib,io,json
from datetime import datetime,timezone
from pathlib import Path
RUN=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System/output/experiments/gpt-6-luna/v3_full_2931_20261015')
OUT=Path('/private/tmp/single-round13-runtime-sealed-review')
sealpath=RUN/'attempts/single_agent/round-0013.batch.seal.json'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def need(ok,reason):
 if not ok:raise ValueError(reason)
if not sealpath.exists():
 print(json.dumps({'status':'WAITING_FOR_NATURAL_SEAL','round':13}));raise SystemExit(0)
batchpath=sealpath.with_suffix('').with_suffix('.json')
# Explicit file names avoid interpreting clinical data or trusting path arithmetic.
batchpath=RUN/'attempts/single_agent/round-0013.batch.json'
result=RUN/'attempts/single_agent/round-0013.json';identity=Path(str(result)+'.run.json')
seal=json.loads(sealpath.read_text());batch=json.loads(batchpath.read_text());manifest=json.loads((RUN/'run_manifest.json').read_text())
need(batch['round']==13 and batch['arm']=='single_agent' and Path(batch['output']).resolve()==result.resolve(),'Sealed batch ownership differs')
need(sha(batchpath)==seal['batch_sha256'] and sha(result)==seal['result_sha256'] and sha(identity)==seal['identity_sha256'],'Natural seal hash mismatch')
need(sha(Path(batch['input']))==batch['input_sha256'],'Sealed shard input hash mismatch')
saved_identity=json.loads(identity.read_text());need(saved_identity==batch['run_identity'],'Sealed run identity differs from batch')
expected=manifest['run_identities']['single_agent'];need(set(saved_identity)==set(expected) and all(v==expected[k] for k,v in saved_identity.items() if k!='input_hash'),'Sealed frozen inference identity differs')
# Reuse the independently authored, answer-blind classifier projection review.
source=Path('/private/tmp/single-round13-runtime-check/review.py').read_text().replace('single-round13-runtime-check','single-round13-runtime-sealed-review')
(OUT/'classification_review.py').write_text(source)
with contextlib.redirect_stdout(io.StringIO()):exec(compile(source,str(OUT/'classification_review.py'),'exec'),{'__name__':'sealed_metadata_review','__file__':str(OUT/'classification_review.py')})
report=json.loads((OUT/'safe_review.json').read_text())
need(report['round13_result_snapshot']['sha256']==seal['result_sha256']==sha(result),'Classified sealed result digest differs')
need(not report['saved_classifications'].get('pending',0),'Sealed batch retained unattempted placeholders')
rows=json.loads(result.read_text());need(len({r['question_id'] for r in rows})==len(rows)==len(batch['question_ids']) and {r['question_id'] for r in rows}==set(batch['question_ids']),'Sealed cohort differs')
ledgerpath=RUN/'acceptance.json';ledgerbytes=ledgerpath.read_bytes();ledger=json.loads(ledgerbytes);choices=ledger['arms']['single_agent']
recovery=report['round12_runtime_recoveries'][0];qid=recovery['question_id'];entry=choices.get(qid)
adopted=bool(entry and entry['round']==13 and entry['attempt_file']=='attempts/single_agent/round-0013.json' and entry['runtime_clean'] is True and entry['selection']=='runtime_clean_terminal' and entry['runtime_failures_before']==1)
need(recovery['first_runtime_clean_in_round13'],'Expected recovery not runtime clean in sealed round13')
for failure in report['runtime_attempts']:
 candidate=choices.get(failure['question_id']);need(not candidate or candidate.get('attempt_file')!='attempts/single_agent/round-0013.json','Runtime-tainted round13 attempt accepted in ledger')
report.update(status='SEALED_RUNTIME_METADATA_RECONCILED',scope='IMMUTABLE_ROUND13_RUNTIME_AND_ROUND12_RECOVERY',reconciled_at_utc=datetime.now(timezone.utc).isoformat(),natural_seal={'batch_sha256':sha(batchpath),'seal_sha256':sha(sealpath),'result_sha256':sha(result),'identity_sha256':sha(identity),'shard_input_sha256':batch['input_sha256'],'frozen_identity_matches':True,'question_count':len(rows),'all_scheduled_questions_accounted_for':True},recovery_canonical_ledger_selection_observed={'question_id':qid,'ledger_entry_matches_first_clean_round13':adopted,'selection_metadata_only':entry,'acceptance_snapshot_sha256':hashlib.sha256(ledgerbytes).hexdigest()},runtime_round13_attempts_not_selected_in_ledger=True,review_code_sha256=sha(Path(__file__)),classification_review_code_sha256=sha(OUT/'classification_review.py'))
for recovery in report['round12_runtime_recoveries']:recovery['selection_persistence']='Sealed raw result independently verified; see separately recorded acceptance-ledger adoption observation.'
(OUT/'safe_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'saved_classifications':report['saved_classifications'],'runtime_attempts':report['runtime_attempts'],'new_provider_policy_rejections':report['provider_policy_rejection_count_in_captured_round13'],'first_clean_recovery_verified_in_sealed_round13':True,'recovery_ledger_adoption_observed':adopted,'evidence':str(OUT/'safe_review.json'),'evidence_sha256':sha(OUT/'safe_review.json')},indent=2))
