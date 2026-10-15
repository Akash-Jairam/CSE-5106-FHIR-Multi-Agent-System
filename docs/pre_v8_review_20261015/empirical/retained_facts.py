"""Small manual-audit support; computes metadata facts, never executes agent code."""
import ast,gzip,json
from collections import Counter
from pathlib import Path
OUT=Path(__file__).parent
cases=json.loads(gzip.decompress((OUT/'selected_case_evidence.json.gz').read_bytes()))['cases']
matrix=json.loads((OUT/'exposed300_matrix.json').read_text())
summary=json.loads((OUT/'cross_version_summary.json').read_text())

def coding(x):return x.get('coding',[]) if isinstance(x,dict) else []
def names(m):return [i['value'] for i in m.get('identifier',[]) if i.get('system','').endswith('mimic-medication-name')]
def start(r):return r.get('dispenseRequest',{}).get('validityPeriod',{}).get('start')
def direct_name(r):return [c.get('code') for c in coding(r.get('medicationCodeableConcept')) if c.get('system','').endswith('mimic-medication-name')]
def routes(r):return [c.get('code','').strip().upper() for d in r.get('dosageInstruction',[]) for c in coding(d.get('route'))]
def values(x,path=''):
 if isinstance(x,dict):
  for k,v in x.items():yield from values(v,(path+'.' if path else '')+k)
 elif isinstance(x,list):
  for i,v in enumerate(x):yield from values(v,path+f'[{i}]')
 else:yield path,x

facts={}
for case in cases:
 qid=case['question_id'];snap=case['snapshots'][0];ws=snap['workspace'];art=snap.get('artifact') or {}
 f={'case_selection':case['selection_rationale'],'artifact_code_retained':bool(art.get('code')),
    'A_released':snap['released']['answer'],'A_eligible':snap['diagnostics']['eligible'],
    'source_check_issues':(snap.get('source_check') or {}).get('issues'),
    'workspace_counts':{k:len(v) for k,v in ws.items()},'snapshot_artifact_sha256':case['audit_sha256']}
 md={m['id']:m for m in ws.get('Medication',[])}
 ev=art.get('evidence') or [];ev_ids={e['resource_id'] for e in ev}
 if qid in {'1f2e37aaa091d50e833afad1','dbadc610a502fc0a6c69d44e'}:
  selected=[e for e in ws['Encounter'] if 'Encounter/'+e['id'] in ev_ids][0]
  f['selected_encounter_projection']={'class':selected.get('class'),'admitSource':selected.get('hospitalization',{}).get('admitSource')}
  f['exact_reference_label_occurs_in_held_workspace']=any(val in case['reference'] for _,val in values(ws) if isinstance(val,str) and val in {'EW EMER.','OBSERVATION ADMIT'})
 if qid in {'7866a9173d7aff1bce72d3a6','2d7404fcfff0b855a4fb2fe9'}:
  observations=[o for o in ws['Observation'] if 'Observation/'+o['id'] in ev_ids]
  f['selected_observations']=[{k:o.get(k) for k in ['id','code','effectiveDateTime','valueString','specimen']} for o in observations]
  refs={o.get('specimen',{}).get('reference','') for o in observations}
  f['linked_specimens']=[{k:s.get(k) for k in ['id','type','collection']} for s in ws.get('Specimen',[]) if 'Specimen/'+s['id'] in refs]
  f['reference_specimen_type_present']=[{k:s.get(k) for k in ['id','type','collection']} for s in ws.get('Specimen',[]) if any(c.get('display') and c['display'] in case['reference'] for c in coding(s.get('type')))]
 if qid in {'383586ac765d783e5374a6c0','41b9a346811b51fc23501b6f'}:
  rs=ws['MedicationRequest']
  if qid.startswith('383'):
   enc=next(i for i in ev_ids if i.startswith('Encounter/'));rs=[r for r in rs if r.get('encounter',{}).get('reference')==enc]
  else:rs=[r for r in rs if 'IV' in routes(r)]
  rs=[r for r in rs if start(r)];earliest=min(start(r) for r in rs);ties=[r for r in rs if start(r)==earliest]
  tied_names=sorted({name for r in ties for name in names(md.get(r.get('medicationReference',{}).get('reference','').split('/')[-1],{}))})
  gold=ast.literal_eval(case['reference'].replace('\n',','))
  gold_names=sorted({x for row in gold for x in row})
  f['tie_witness']={'min_validity_start':earliest,'requests_at_min':len(ties),'distinct_names':tied_names,
                    'exact_reference_set_match':gold_names==tied_names,'distinct_names_n':len(tied_names)}
 if qid=='c04f16fa062bc81f8c917962':
  rs=[o for o in ws['Observation'] if any(c.get('system','').endswith('mimic-microbiology-test') for c in coding(o.get('code'))) and o.get('effectiveDateTime')]
  rs.sort(key=lambda o:o['effectiveDateTime'][:19]);f['earliest_microbiology_observations']=[{k:o.get(k) for k in ['code','effectiveDateTime','valueString','valueCodeableConcept']} for o in rs[:3]]
  f['exact_reference_time_present_in_test_fields']=[{'field':k,'code':o.get('code'),'value':v} for o in rs for k,v in o.items() if isinstance(v,str) and v[:19]=='2155-12-03T19:58:00']
  f['saved_question_assumption']=matrix['references'][qid]['assumption']
 if qid=='bf8aee434927ad78efbb9c37':
  rs=[r for r in ws['MedicationRequest'] if 'IV' in routes(r) and start(r) and '2200-06-01'<=start(r)<='2200-12-31T23:59:59']
  f['latest_held_exact_IV_start_before_assumed_cutoff']=max(start(r) for r in rs)
  f['reference_time_at_any_held_request_field']=[{'field':k,'value':v} for r in ws['MedicationRequest'] for k,v in values(r) if isinstance(v,str) and v[:19]=='2200-09-28T20:00:00'][:5]
 if qid=='f8d725f820917d71a5301983':
  rs=[o for o in ws['Observation'] if any(c.get('display','').strip().casefold()=='ptt' for c in coding(o.get('code'))) and type(o.get('valueQuantity',{}).get('value')) in (int,float)]
  low=min(r['valueQuantity']['value'] for r in rs);f['PTT_minimum_witness']={'n_PTT':len(rs),'minimum_value':low,'minimum_rows':[{k:r.get(k) for k in ['effectiveDateTime','issued','valueQuantity']} for r in rs if r['valueQuantity']['value']==low]}
  f['reference_time_in_any_held_field']=[{'resource_type':typ,'field':k,'value':v} for typ,rs in ws.items() for r in rs for k,v in values(r) if isinstance(v,str) and v[:19]=='2120-01-31T03:33:00'][:5]
  f['nonempty_code_text_observations']=sum(bool(o.get('code',{}).get('text')) for o in ws['Observation'])
  f['PTT_system_cohort_evidence']=[{k:r.get(k) for k in ['code','effectiveDateTime','valueQuantity']} for r in ws['Observation'] if any('ptt' in c.get('display','').lower() for c in coding(r.get('code')))]
 if qid=='2ca14c9e3ae55c6d467fa8c1':
  matched=[r for r in ws['MedicationRequest'] if any(n.strip().casefold()=='furosemide' for n in direct_name(r)+names(md.get(r.get('medicationReference',{}).get('reference','').split('/')[-1],{})))]
  dose=lambda r:[d['doseQuantity']['value'] for i in r.get('dosageInstruction',[]) for d in i.get('doseAndRate',[]) if type(d.get('doseQuantity',{}).get('value')) in (int,float)]
  f['dose_availability']={'matched_requests':len(matched),'known_dose_sum':sum(sum(dose(r)) for r in matched),'requests_without_numeric_dose':sum(not dose(r) for r in matched),'unknown_dose_requests_have_no_dosageInstruction':all(not r.get('dosageInstruction') for r in matched if not dose(r))}
 if qid=='967175d0df47b1226680f9bc':
  obs={r['id']:r for r in ws['Observation']};parents=[]
  for child in obs.values():
   if not any(c.get('display') in {'ESCHERICHIA COLI','ENTEROCOCCUS SP.'} for c in coding(child.get('code'))):continue
   for d in child.get('derivedFrom',[]):
    par=obs.get(d.get('reference','').split('/')[-1])
    if par:parents.append({'child_code':child.get('code'),'child_direct_encounter':child.get('encounter'),'parent_encounter':par.get('encounter'),'parent_time':par.get('effectiveDateTime'),'parent_code':par.get('code')})
  f['organism_parent_edges']=parents[:12]
  f['recorded_final_issues']=snap['released'].get('revision',{}).get('final_check_issues')
 facts[qid]=f

template_counts=Counter(matrix['questions'][qid]['template'] for qid in summary['question_sets']['all14arms_wrong'])
result={'scope':'Retrospective15 purposive all14-wrong cases, generic retained-field/tie checks only; no agent program replay',
        'persistent37_template_counts':dict(template_counts),'facts':facts}
(OUT/'retained_facts.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'case_count':len(facts),'proven_exact_tie_reference_matches':sum((f.get('tie_witness') or {}).get('exact_reference_set_match',False) for f in facts.values())},indent=2))
