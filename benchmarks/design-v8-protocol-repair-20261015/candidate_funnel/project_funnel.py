from pathlib import Path
import json,collections
r=Path(__file__).resolve().parent;rows=json.loads((r/'rows.json').read_text());ds=json.loads((r/'A_wrong_evidence_private.json').read_text());result={}
for m in 'WRC':
 result[m]={}
 for name,subset in [('all120',[x for x in rows if x['study_arm']==m]),('A_wrong_nonfatal',[x for x in rows if x['study_arm']==m and not x['A_typed_correct'] and not x['A_terminal_error']])]:
  keys=['n','source_bound_A','snapshot_created','actual_aux_send','zero_send_snapshot','probe_rows','eligible_observation_rows','source_bound_B','source_bound_B_no_witness','source_bound_B_with_witness','source_bound_disagreement','correct_alternative_to_wrong_A','any_final_request_rows','successful_final_reply_rows','correct_final_reply_rows','unreleased_correct_final_rows']
  vals=[len(subset),sum(x['A_source_bound'] for x in subset),sum(x['B_started'] for x in subset),sum(x['aux_sdk']>0 for x in subset),sum(x['B_started'] and x['aux_sdk']==0 for x in subset),sum(x['probe_interactions']>0 for x in subset),sum(x['eligible_observations']>0 for x in subset),sum(x['B_source_bound'] for x in subset),sum(x['B_source_bound'] and not x['B_has_witness'] for x in subset),sum(x['B_source_bound'] and x['B_has_witness'] for x in subset),sum(x['B_agrees_A'] is False for x in subset),sum(not x['A_typed_correct'] and x['B_typed_correct'] is True for x in subset),sum(x['attempted_final_requests']>0 for x in subset),sum(x['successful_final_answer_replies']>0 for x in subset),sum(x['scorer_compatible_final_replies']>0 for x in subset),sum(x['scorer_compatible_unreleased_final'] for x in subset)]
  z=dict(zip(keys,vals));z['requested_retrievals']=sum(sum(v for k,v in x['requested_native_tools'].items() if k.startswith('get_resources')) for x in subset)
  z['stops']=dict(collections.Counter(str(x['B_stop']) for x in subset if x['B_started']));z['observation_coverage_reasons']=dict(sum((collections.Counter(x['observation_coverage_reasons']) for x in subset),collections.Counter()))
  result[m][name]=z
wrong_events=[(x['summary']['study_arm'],e) for x in ds if not x['summary']['A_terminal_error'] for e in x['event_projection']]
result['wrong_nonfatal_event_counts']={m:{'native_errors':dict(collections.Counter(str(e['error']) for mode,e in wrong_events if mode==m and e['error'])),'sourcecheck_issues_nonempty':sum(bool(e['check_issues']) for mode,e in wrong_events if mode==m and e['name']=='check_fields')} for m in 'WRC'}
(r/'funnel_counts.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2))
