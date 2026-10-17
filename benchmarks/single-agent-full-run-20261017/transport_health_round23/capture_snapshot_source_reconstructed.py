import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
run=Path('output/experiments/gpt-6-luna/v3_full_2931_20261015');out=Path('/private/tmp/single-round23-runtime-progress-1133');s=json.loads((run/'supervisor_state.json').read_text());a=s.get('active',{}).get('single_agent',{});errors=[];latest=[];hashes={}
for p in sorted((run/'diagnostics/single_agent/round-0023').glob('*.jsonl')):
 data=p.read_bytes();hashes[p.name]=hashlib.sha256(data).hexdigest();events=[]
 for line in data.splitlines():
  try:e=json.loads(line)
  except ValueError:continue
  if e.get('event')!='end':continue
  record={k:e.get(k) for k in ('pid','counter','started_utc','finished_utc','duration_seconds','outcome','http_status','fd_count')};record['exception_classes']=[(v.get('type') or {}).get('class') for v in e.get('exception_chain',[])];record['sdk_closed']=(e.get('client') or {}).get('sdk_closed');record['http_closed']=(e.get('client') or {}).get('http_closed');events.append(record)
  if e.get('outcome')=='error':errors.append(record)
 latest.extend(events[-2:])
raw=run/'attempts/single_agent/round-0023.json';r={'captured_at_utc':datetime.now(timezone.utc).isoformat(),'phase':s.get('phase'),'attention':s.get('attention'),'heartbeat_at':s.get('heartbeat_at'),'round':a.get('round'),'checkpointed':a.get('checkpointed_in_batch'),'checkpoint_age_seconds':a.get('seconds_since_checkpoint'),'raw_mtime_utc':datetime.fromtimestamp(raw.stat().st_mtime,timezone.utc).isoformat(),'diagnostic_errors_safe_metadata':errors,'latest_provider_ends_safe_metadata':latest,'diagnostic_source_hashes':hashes,'question_error_join_not_claimed_before_checkpoint':True,'no_model_provider_scorer_calls_or_live_changes':True}
(out/'safe_progress.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
