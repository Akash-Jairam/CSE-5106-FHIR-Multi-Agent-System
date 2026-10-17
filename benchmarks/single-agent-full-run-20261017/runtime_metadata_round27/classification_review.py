"""Bounded, read-only round27 runtime/recovery classification; no answer scoring."""
from collections import Counter
from datetime import datetime,timezone
import hashlib,importlib.util,json,re,sys
from pathlib import Path
ROOT=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
RUN=ROOT/'output/experiments/gpt-6-luna/v3_full_2931_20261015'
OUT=Path('/private/tmp/single-round27-runtime-sealed-review')
TARGETS=('d24509cc1ab0dd341bb53f33',)
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
from scripts import v3_full_run as runner
from scripts.audit_store import restore_result_audit
monitorpath=ROOT/'benchmarks/single-agent-full-run-20261017/monitor_single_agent.py'
spec=importlib.util.spec_from_file_location('private_round27_monitor',monitorpath);monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,message):
 if not ok:raise ValueError(message)
manifestpath=RUN/'run_manifest.json';manifestbytes=manifestpath.read_bytes();manifest=json.loads(manifestbytes);classifier=runner.classifier_for_manifest(manifest)
sources={'run_manifest.json':hashlib.sha256(manifestbytes).hexdigest(),'scripts/v3_full_run.py':sha(ROOT/'scripts/v3_full_run.py'),'monitor_single_agent.py':sha(monitorpath),'scripts/audit_store.py':sha(ROOT/'scripts/audit_store.py')}
need(sources['scripts/v3_full_run.py']==manifest['source_sha256']['scripts/v3_full_run.py'] and runner.CLASSIFIER_VERSION==manifest['classifier_version'],'Frozen classifier mismatch')
rawpath=RUN/'attempts/single_agent/round-0027.json';data=rawpath.read_bytes();rows=json.loads(data)
need(len({r.get('question_id') for r in rows})==len(rows),'Duplicate checkpoint identities')
known_classes=('APIConnectionError','APITimeoutError','ConnectTimeout','ReadTimeout','PoolTimeout','ConnectError','ReadError','RemoteProtocolError','TransportError','ConnectionError','SSLError')
families=(('timeout',r'timed?\s*out|time[ -]?out'),('empty_provider_response',r'EXCEPTION: Empty or None output\.'),('provider_internal_server_error',r'internal server error'),('upstream_bad_gateway',r'bad gateway'),('upstream_unavailable_or_reset',r'upstream.*(?:unavailable|reset)'),('service_or_backend_unavailable',r'service unavailable|backend.*unavailable'),('rate_or_resource_limit',r'too many requests|rate[ -]?limit|resource exhausted'),('transport_closed_or_disconnected',r'cannot send a request.*client.*closed|client has been closed|connection closed|server disconnected|transport.*closed'),('connection_failure',r'connection (?:error|failure|reset|refused|aborted)'),('network_unreachable',r'network (?:is unreachable|error)'),('DNS_resolution_failure',r'nodename nor servname provided|temporary failure in name resolution|name or service not known'),('TLS_certificate_failure',r'certificate verify failed'),('sandbox_worker_failure',r'worker.*(?:unavailable|failed|died|crashed)|sandbox.*unavailable'),('python_execution_deadline_exceeded',r'^Python execution exceeded [0-9.]+ seconds$'),('python_host_OS_failure',r'^Python execution failed: (?:OSError|PermissionError|FileNotFoundError)$'))
def signal_records(full):
 signals=[('error',full['error'])] if isinstance(full.get('error'),str) and full['error'].strip() else []
 signals.extend(runner.diagnostic_errors(full.get('revision') or {}))
 for i,m in enumerate(full.get('trace') or []):
  if not isinstance(m,dict) or m.get('role')!='tool':continue
  content=m.get('content')
  if isinstance(content,str):
   try:content=json.loads(content)
   except (ValueError,TypeError):continue
  if isinstance(content,dict):signals.extend(runner.diagnostic_errors(content,f'trace[{i}]'))
 return signals
def signatures(full,decision):
 result=set();statuses=set()
 for path,message in signal_records(full):
  if path not in decision.get('diagnostic_paths',[]):continue
  for cls in known_classes:
   if re.search(r'(?<![A-Za-z])'+cls+r'(?![A-Za-z])',message):result.add(cls)
  for label,pattern in families:
   if re.search(pattern,message,re.I):result.add(label)
  for m in re.finditer(r'(?:HTTP(?:StatusError)?[ :]*|status(?:_code| code)?[ =:]*|error code[ :]*)(408|429|50[0234]|52[024])\b',message,re.I):statuses.add(int(m.group(1)))
  for m in re.finditer(r'(?:^|\()(408|429|50[0234]) (?:Client|Server) Error',message,re.I):statuses.add(int(m.group(1)))
 if runner.provider_jwt_paths(full):result.add('observed_provider_jwt_rejection')
 if decision.get('category')=='provider_policy_rejection':result.update(('azure_prompt_content_filter','ResponsibleAIPolicyViolation','jailbreak_detected_and_filtered'));statuses.add(400)
 if statuses:result.add('allowlisted_HTTP_status')
 if not result:result.add('allowlisted_infrastructure_failure_unspecified')
 return sorted(result),sorted(statuses)
audit_hashes={}
def audit(row,directory):
 projection=monitor.safe_projection(row);full=restore_result_audit(projection,directory,question_id=row['question_id'],restore_history=False)
 ref=row.get('audit_artifacts')
 if ref:
  p=directory/ref['path'];audit_hashes[str(p.relative_to(RUN))]={'compressed_sha256':sha(p),'uncompressed_sha256':ref['sha256']}
 return full
counts=Counter();runtime=[];current={}
for row in rows:
 decision=classifier(monitor.safe_projection(row),rawpath.parent);counts[decision['status']]+=1
 if row['question_id'] in TARGETS:
  current[row['question_id']]={'classification':decision['status'],'selection_reason':decision['reason']}
 if decision['status']!='pending':audit(row,rawpath.parent)
 if decision['status']=='runtime':
  sig,status=signatures(audit(row,rawpath.parent),decision)
  runtime.append({'question_id':row['question_id'],'classification':'runtime','category':decision.get('category','infrastructure_runtime'),'selection_reason':decision['reason'],'allowlisted_failure_signatures':sig,'HTTP_statuses':status,'new_provider_policy_rejection':decision.get('category')=='provider_policy_rejection'})
prior={qid:[] for qid in TARGETS}
for batchpath in sorted((RUN/'attempts/single_agent').glob('round-*.batch.json')):
 batch=json.loads(batchpath.read_text());number=batch.get('round');included=set(batch.get('question_ids',[]))&set(TARGETS)
 if not included or not isinstance(number,int) or number>=27:continue
 sealpath=batchpath.with_suffix('.seal.json')
 if not sealpath.exists():continue
 seal=json.loads(sealpath.read_text());p=Path(batch['output']);raw=p.read_bytes();identity=Path(str(p)+'.run.json')
 need(seal['batch_sha256']==sha(batchpath) and seal['result_sha256']==hashlib.sha256(raw).hexdigest() and seal['identity_sha256']==sha(identity),'Prior sealed attempt changed')
 sources[str(p.relative_to(RUN))]=hashlib.sha256(raw).hexdigest();sources[str(batchpath.relative_to(RUN))]=sha(batchpath);sources[str(sealpath.relative_to(RUN))]=sha(sealpath)
 for row in json.loads(raw):
  qid=row.get('question_id')
  if qid not in included:continue
  decision=classifier(monitor.safe_projection(row),p.parent)
  prior[qid].append({'round':number,'classification':decision['status'],'selection_reason':decision['reason']})
  audit(row,p.parent)
recovery=[]
for qid in TARGETS:
 need(any(p['round']==26 and p['classification']=='runtime' for p in prior[qid]),'Expected round26 runtime identity absent')
 recovery.append({'question_id':qid,'prior_sealed_attempts':prior[qid],'round27_captured_decision':current.get(qid),'first_runtime_clean_in_round27':current.get(qid,{}).get('classification')=='accepted' and not any(p['classification']=='accepted' for p in prior[qid]),'selection_persistence':'Captured checkpoint decision only; canonical/ledger adoption follows the normal sealed boundary.'})
need(sha(manifestpath)==sources['run_manifest.json'] and sha(ROOT/'scripts/v3_full_run.py')==sources['scripts/v3_full_run.py'] and sha(monitorpath)==sources['monitor_single_agent.py'],'Frozen classifier inputs changed')
need(not any(k in sys.modules for k in ('scripts.v3_full_analysis','utils.answer_scorer','utils.evidence_scorer')),'Unexpected scorer import')
report={'status':'PASS','captured_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'IN_PROGRESS_ROUND27_CLASSIFIER_SNAPSHOT','classifier_version':runner.CLASSIFIER_VERSION,'round':27,'saved_classifications':dict(counts),'runtime_attempts':runtime,'provider_policy_rejection_count_in_captured_round27':sum(r['new_provider_policy_rejection'] for r in runtime),'round26_runtime_recoveries':recovery,'source_sha256':sources,'round27_result_snapshot':{'relative_path':str(rawpath.relative_to(RUN)),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'sha256_at_review_end':sha(rawpath),'live_checkpoint_may_advance_normally':True},'referenced_audit_hashes':audit_hashes,'no_clinical_raw_error_answer_reference_or_correctness_output':True,'no_model_scorer_provider_calls_or_live_mutations':True}
(OUT/'safe_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'saved_classifications':report['saved_classifications'],'runtime_attempts':runtime,'new_provider_policy_rejections':report['provider_policy_rejection_count_in_captured_round27'],'recoveries':recovery,'evidence':str(OUT/'safe_review.json'),'evidence_sha256':sha(OUT/'safe_review.json')},indent=2))
