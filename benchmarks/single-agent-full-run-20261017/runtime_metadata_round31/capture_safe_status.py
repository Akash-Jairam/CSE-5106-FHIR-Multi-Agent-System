"""Project only observed status metadata; this performs no completion/export/native checks."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib,json
RUN=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System/output/experiments/gpt-6-luna/v3_full_2931_20261015')
OUT=Path('/private/tmp/single-round31-runtime-sealed-review')
PARKED={'05a8454a6a71435a557db2cc','0eb69f6e47ab63c055e2df90'}
def sha(data):return hashlib.sha256(data).hexdigest()
statebytes=(RUN/'supervisor_state.json').read_bytes();state=json.loads(statebytes)
ledgerbytes=(RUN/'acceptance.json').read_bytes();ledger=json.loads(ledgerbytes)
manifestbytes=(RUN/'run_manifest.json').read_bytes();manifest=json.loads(manifestbytes)
operations=(RUN/'operations.jsonl').read_bytes()
provider=state.get('provider_rejections') or {};pending=provider.get('pending_question_ids',{}).get('single_agent',[])
accepted=ledger['arms']['single_agent'];total=manifest['questions_per_arm']
projection={key:state.get(key) for key in ('phase','reason','attention','started_at','resumed_at','scope_started_at','finished_at','heartbeat_at','selected_arms','selected_scope_complete','benchmark_complete','pending_scoped_export','analysis_status')}
projection['active_is_empty']=state.get('active')=={}
projection['single_agent_runtime_clean_count']=len(accepted)
projection['questions_per_arm']=total
projection['remaining_count_by_cohort_total']=total-len(accepted)
projection['pending_question_ids_from_provider_state']=sorted(pending)
projection['pending_ids_are_exactly_two_parked_cases']=set(pending)==PARKED
projection['parked_ids_absent_from_runtime_clean_results']=not(PARKED&set(accepted))
projection['provider_rejections']={qid:provider.get('rejections',{}).get('single_agent',{}).get(qid) for qid in sorted(PARKED)}
projection['issues']=[{key:item.get(key) for key in ('arm','question_id','provider_rejections','status','action')} for item in state.get('issues',[]) if item.get('arm')=='single_agent' and item.get('question_id') in PARKED]
stop_events=[]
for line in operations.splitlines():
 try:item=json.loads(line)
 except ValueError:continue
 if item.get('event')=='supervisor_stopped' and item.get('at','')>=(state.get('finished_at') or '9999'):
  stop_events.append({key:item.get(key) for key in ('at','event','phase','attention')})
terminal=state.get('phase')=='attention' and state.get('reason')=='provider_policy_review_required'
result={'scope':'OBSERVED_STATUS_METADATA_ONLY_NOT_COMPLETION_VERIFICATION','captured_at_utc':datetime.now(timezone.utc).isoformat(),'observed_state':projection,'observed_stop_events_after_finished_at':stop_events,'interpretation':'INCOMPLETE_PROVIDER_REVIEW_REQUIRED' if terminal else 'OBSERVED_TRANSITION_NOT_CLAIMED_COMPLETE','benchmark_completion_claimed':False,'wrong_answer_inference_claimed':False,'no_next_batch_expected_from_this_attention_reason':terminal,'no_next_batch_wait_or_terminal_helper_export_scoring_invoked':True,'source_snapshots':{'run_manifest.json':sha(manifestbytes),'supervisor_state.json':sha(statebytes),'acceptance.json':sha(ledgerbytes),'operations.jsonl':sha(operations)},'state_ledger_operation_hashes_are_snapshots_not_final_seals':True,'review_code_sha256':sha(Path(__file__).read_bytes()),'no_clinical_answer_reference_correctness_credential_output_or_calls':True}
(OUT/'safe_status_metadata.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'interpretation':result['interpretation'],'observed_state':projection,'stop_events':stop_events,'sha256':sha((OUT/'safe_status_metadata.json').read_bytes())},indent=2))
