"""Verify only reviewed provider-question scheduling/classification metadata through round25."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib, importlib.util, json, sys
ROOT=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
RUN=ROOT/'output/experiments/gpt-6-luna/v3_full_2931_20261015'
OUT=Path('/private/tmp/single-round25-runtime-sealed-review')
PARKED=('05a8454a6a71435a557db2cc','0eb69f6e47ab63c055e2df90')
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
from scripts import v3_full_run as runner
from scripts.audit_store import restore_result_audit
monitorpath=ROOT/'benchmarks/single-agent-full-run-20261017/monitor_single_agent.py'
spec=importlib.util.spec_from_file_location('private_round25_parked_monitor',monitorpath)
monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,message):
 if not ok:raise ValueError(message)
manifestpath=RUN/'run_manifest.json';manifest=json.loads(manifestpath.read_bytes())
source_hashes={'run_manifest.json':sha(manifestpath),'scripts/v3_full_run.py':sha(ROOT/'scripts/v3_full_run.py'),'monitor_single_agent.py':sha(monitorpath),'scripts/audit_store.py':sha(ROOT/'scripts/audit_store.py')}
need(source_hashes['scripts/v3_full_run.py']==manifest['source_sha256']['scripts/v3_full_run.py'],'Frozen runner mismatch')
classifier=runner.classifier_for_manifest(manifest)
records={qid:[] for qid in PARKED};bindings={};audits={}
for batchpath in sorted((RUN/'attempts/single_agent').glob('round-*.batch.json')):
 batch=json.loads(batchpath.read_bytes());number=batch['round']
 if number>25:continue
 included=set(batch['question_ids'])&set(PARKED)
 if number==25:need(not included,'Parked question scheduled in round25')
 if not included:continue
 sealpath=batchpath.with_suffix('.seal.json');need(sealpath.exists(),'Prior provider attempt is unsealed')
 seal=json.loads(sealpath.read_bytes());result=Path(batch['output']);identity=Path(str(result)+'.run.json')
 need(sha(batchpath)==seal['batch_sha256'] and sha(result)==seal['result_sha256'] and sha(identity)==seal['identity_sha256'],'Prior provider attempt immutable hashes differ')
 need(sha(Path(batch['input']))==batch['input_sha256'],'Prior provider shard input changed')
 for path in (batchpath,sealpath,result,identity):bindings[str(path.relative_to(RUN))]=sha(path)
 rows=[row for row in json.loads(result.read_bytes()) if row.get('question_id') in included]
 need(len(rows)==len(included),'Provider question omitted/duplicated in prior result')
 for row in rows:
  decision=classifier(monitor.safe_projection(row),result.parent)
  need(decision['status']=='pending' or (decision['status']=='runtime' and decision.get('category')=='provider_policy_rejection'),'Parked question has unexpected prior terminal decision')
  if decision['status']=='pending':
   records[row['question_id']].append({'round':number,'classification':'pending','category':'unattempted_placeholder_not_counted_as_rejection'});continue
  records[row['question_id']].append({'round':number,'classification':'runtime','category':'provider_policy_rejection'})
  projection=monitor.safe_projection(row);restore_result_audit(projection,result.parent,question_id=row['question_id'],restore_history=False)
  ref=row.get('audit_artifacts')
  if ref:
   p=result.parent/ref['path'];audits[str(p.relative_to(RUN))]={'compressed_sha256':sha(p),'uncompressed_sha256':ref['sha256']}
need(all(sum(r['classification']=='runtime' for r in v)==2 for v in records.values()),'Provider question rejection count changed')
ledgerpath=RUN/'acceptance.json';ledgerbytes=ledgerpath.read_bytes();ledger=json.loads(ledgerbytes)
need(not (set(ledger['arms']['single_agent'])&set(PARKED)),'Provider-pending question adopted as a result')
statepath=RUN/'supervisor_state.json';statebytes=statepath.read_bytes();state=json.loads(statebytes)
provider=state['provider_rejections'];derived=provider['rejections']['single_agent'];parked=provider['parked_question_ids']['single_agent']
need(all(derived.get(qid)==2 for qid in PARKED) and set(PARKED)<=set(parked),'Derived parked state disagrees with sealed counts')
need(sha(manifestpath)==source_hashes['run_manifest.json'] and sha(ROOT/'scripts/v3_full_run.py')==source_hashes['scripts/v3_full_run.py'] and sha(monitorpath)==source_hashes['monitor_single_agent.py'],'Frozen source binding changed')
report={'status':'PASS','captured_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'PARKED_PROVIDER_METADATA_THROUGH_SEALED_ROUND25','question_classifications':records,'exactly_two_rejections_each':True,'round25_excludes_parked_question_ids':True,'pending_not_in_acceptance_snapshot':True,'source_sha256':source_hashes,'immutable_prior_attempt_bindings':bindings,'audit_hashes':audits,'state_observation':{'sha256':hashlib.sha256(statebytes).hexdigest(),'snapshot_only_not_immutable':True,'provider_counts':{qid:derived[qid] for qid in PARKED}},'acceptance_observation':{'sha256':hashlib.sha256(ledgerbytes).hexdigest(),'snapshot_only_not_immutable':True},'review_code_sha256':sha(Path(__file__)),'no_answers_references_raw_errors_credentials_or_scoring':True}
(OUT/'parked_provider_status.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'exactly_two_rejections_each':True,'round25_excludes_parked_question_ids':True,'pending_not_in_acceptance_snapshot':True,'evidence':str(OUT/'parked_provider_status.json'),'sha256':sha(OUT/'parked_provider_status.json')},indent=2))
