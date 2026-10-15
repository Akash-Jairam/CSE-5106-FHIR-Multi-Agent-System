from pathlib import Path
import collections,hashlib,json,math
ROOT=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System');REC=ROOT/'benchmarks/design-v6-20261015';OUT=ROOT/'output/experiments/gpt-6-luna/independent_recomputation_v6_300q';DEST=Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(REC/'completion_protocol_audit.json')=='ba7a383078218c39c45ec4b485414b1adb248d14cd6e73fe83afd28bcb9d92fd';proof=json.loads((REC/'completion_protocol_audit.json').read_text());assert proof['status']=='PASS' and proof['restored_audit_records']==600
cohorts={};safe_rows=[]
for arm in ('ordinary_multi_agent','ordinary_single_agent'):
 path=OUT/(arm+'.json');assert sha(path)==proof['raw_files'][path.name]['sha256'];raw=json.loads(path.read_text());assert len(raw)==300
 arms=[]
 for row in raw:
  if arm=='ordinary_multi_agent':
   rev=row.get('revision') or {};pd=rev.get('primary_diagnostics') or {};art=rev.get('primary_artifact') or {}
   tokens=pd.get('tokens');tools=pd.get('tool_calls');sdk=pd.get('model_call_attempts_recorded');known=pd.get('usage_known') is True and not pd.get('fatal')
   group='fatal_or_unknown' if pd.get('fatal') or not known else 'nonfatal_binding_ineligible' if not pd.get('eligible') else 'bound_with_limitation' if art.get('limitation') else 'bound_without_limitation'
   stop=pd.get('primary_budget_stop')
  else:
   t=row.get('tokens') or {};b=row.get('budget') or {};tokens=t.get('total');tools=b.get('tool_calls_run');sdk=t.get('llm_calls');known=bool(t.get('per_call')) and all(c.get('usage_available') is True for c in t.get('per_call') or [])
   group='terminal_error' if row.get('error') else 'released_no_error'
   stop=b.get('stopped')
  assert all(type(v) is int and v>=0 for v in (tokens,tools,sdk))
  item={'question_id':row['question_id'],'group':group,'tokens_reported':tokens,'tools':tools,'sdk_attempts':sdk,'usage_known':known,'budget_stop_category':'token' if stop and 'token' in str(stop) else 'tool' if stop and 'tool' in str(stop) else 'other' if stop else 'none'}
  if known:item.update(residual_tokens=200000-tokens,residual_tools=30-tools,residual_sdk=30-sdk)
  arms.append(item);safe_rows.append({'arm':arm,**item})
 cohorts[arm]=arms
def distribution(values):
 s=sorted(values);n=len(s)
 return {'n':n,**{name:s[max(0,math.ceil(q*n)-1)] if n else None for name,q in [('min',0),('p10',.10),('p25',.25),('median',.5),('p75',.75),('p90',.9),('p95',.95),('max',1)]},'mean':sum(s)/n if n else None}
def describe(rows):
 known=[r for r in rows if r['usage_known']]
 return {'n':len(rows),'known_usage':len(known),'unknown_usage_excluded_from_residual_admission':len(rows)-len(known),'budget_stops':dict(collections.Counter(r['budget_stop_category'] for r in rows)),
 'actual_reported':{k:distribution([r[k] for r in rows]) for k in ('tokens_reported','tools','sdk_attempts')},
 'known_residual':{k:distribution([r[k] for r in known]) for k in ('residual_tokens','residual_tools','residual_sdk')},
 'residual_threshold_counts_not_sufficiency':{f'tokens_ge_{n}':sum(r['residual_tokens']>=n for r in known) for n in (25000,50000,100000,150000)},
 'tool_residual_thresholds':{f'ge_{n}':sum(r['residual_tools']>=n for r in known) for n in (2,3,5,10,15,20)},
 'sdk_residual_thresholds':{f'ge_{n}':sum(r['residual_sdk']>=n for r in known) for n in (2,3,5,10,15,20)},
 'joint_residual_thresholds':{f'T{t}_U{u}_C{c}':sum(r['residual_tokens']>=t and r['residual_tools']>=u and r['residual_sdk']>=c for r in known) for t,u,c in [(25000,2,2),(25000,3,3),(50000,5,5),(100000,10,10)]},
 'same_observed_A_cost_second_solve_plus_one_selector_SDK_screen_only':sum(2*r['tokens_reported']<=200000 and 2*r['tools']<=30 and 2*r['sdk_attempts']+1<=30 for r in known)}
result={'method':'Only canonical token/tool/SDK counters and stored primary diagnostics inspected; no question/answer/reference fields accessed. Quantiles use nearest rank; known residuals exclude unknown usage. Team A had175k/27 caps, so its residuals are observed arithmetic, not a counterfactual full-cap natural A. Fresh Single control had200k/30/30 and directly measures full-policy consumption; it is a different stochastic trajectory and its source eligibility is not inferred here. Residual thresholds and repeated-own-cost screen are budget possibilities only, never predicted completed Bs.',
 'arms':{arm:{'all':describe(rows),'groups':{g:describe([r for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})}} for arm,rows in cohorts.items()},
 'completion_report_sha256':sha(REC/'completion_protocol_audit.json'),'raw_arm_sha256':{a:sha(OUT/(a+'.json')) for a in cohorts},'no_scoring_or_reference_access':True}
(DEST/'residual_metrics.json').write_text(json.dumps(result,indent=2)+'\n');(DEST/'private_residual_rows.json').write_text(json.dumps(safe_rows,indent=2)+'\n')
# Compact stdout only.
for a,data in result['arms'].items():
 print(a,'groups',{g:x['n'] for g,x in data['groups'].items()});print(json.dumps(data['all'],indent=2))
