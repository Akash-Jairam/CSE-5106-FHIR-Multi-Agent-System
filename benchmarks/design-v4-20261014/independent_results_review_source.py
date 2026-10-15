"""Independent public typed-scoring/paired-statistics review; no runtime bootstrap."""
import ast,csv,hashlib,importlib.util,json,pathlib,statistics
R=pathlib.Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System');F=R/'benchmarks/design-v4-20261014'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def require(x,msg):
 if not x:raise AssertionError(msg)
def load(name,rel):
 p=R/rel;spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
proof=json.loads((F/'completion_protocol_audit.json').read_text());official=json.loads((F/'results.json').read_text());before=sha(F/'results.json')
require(proof['status']=='PASS' and proof['restored_audit_records']==600,'No full integrity gate')
manifest=json.loads((F/'manifest.json').read_text())
for rel in ['utils/answer_scorer.py','scripts/experiment_stats.py','scripts/full_benchmark.py']:
 require(sha(R/rel)==manifest['source_sha256'][rel],'Frozen scoring/helper source differs')
score=load('review_public_scorer','utils/answer_scorer.py');stats=load('review_public_stats','scripts/experiment_stats.py');stream=load('review_public_stream','scripts/full_benchmark.py')
with (R/'final_dataset/questions_answers_sql_fhir.csv').open(newline='') as f:source={r['question_id']:r for r in csv.DictReader(f)}
with (R/'manifests/question_families.csv').open(newline='') as f:families={r['question_id']:r for r in csv.DictReader(f)}
split=json.loads((R/'manifests/split.json').read_text());ids=sorted(split['splits']['held_out']['question_ids']);require(len(ids)==300,'Fixed subset differs')
require(sha(R/'final_dataset/questions_answers_sql_fhir.csv')==official['dataset_sha256'],'Dataset differs')
require(sha(R/'manifests/question_families.csv')==official['families_sha256'],'Families differs')
require(hashlib.sha256(json.dumps(ids).encode()).hexdigest()==official['question_ids_sha256'],'Cohort hash differs')
fields=None
for n in ast.parse((R/'utils/core_utils.py').read_text()).body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RESULT_COLUMNS' for t in n.targets):fields=ast.literal_eval(n.value)
require(fields is not None,'Missing result fields')
records={};details={};summaries={}
def failed(r):return r.get('error') is not None and str(r.get('error')).strip() not in ('','nan')
for name,binding in official['identities'].items():
 path=pathlib.Path(binding['result_path']);require(path.resolve().is_relative_to((R/'output').resolve()),'Unexpected result path')
 require(sha(path)==binding['result_sha256'],'Raw SHA differs: '+name)
 identity=json.loads(pathlib.Path(str(path)+'.run.json').read_text());require(identity==binding['identity'],'Recorded identity differs')
 indexed={}
 for saved in stream.iter_result_rows(path):
  q=saved['question_id'];require(q not in indexed,'Duplicate attempt')
  history=saved.get('attempt_history') or [];require(isinstance(history,list),'History shape')
  first={**saved,**({k:history[0].get(k) for k in fields} if history else {})}
  require(first.get('latency') is not None or failed(first),'Unattempted row')
  require(first['patient_fhir_id']==source[q]['patient_fhir_id'],'Patient differs')
  indexed[q]={k:first.get(k) for k in ['agent_answer','error','latency','tokens','budget','revision']}
 require(sorted(indexed)==ids,'Result cohort differs')
 scores={};rows=[]
 for q in ids:
  r=indexed[q];v=score.score_answer(r['agent_answer'],source[q]['true_answer'],source[q]['question'],families[q]['template'])
  correct=0 if failed(r) else v['typed_correct'];scores[q]=correct
  rows.append({'question_id':q,'correct':correct,'failed':failed(r),'family':families[q]['family'],'patient':source[q]['patient_fhir_id'],'answer_type':v['answer_type'],'typed_reason':v['typed_reason'],'agent_answer':r['agent_answer']})
 accuracy={}
 for fam in ['all',*sorted(set(families[q]['family'] for q in ids))]:
  chosen=[z for z in rows if fam=='all' or z['family']==fam]
  accuracy[fam]={'correct':sum(z['correct'] for z in chosen),**stats.clustered_mean([z['correct'] for z in chosen],[z['patient'] for z in chosen],n_boot=10000,seed=5106)}
 require(accuracy==official['arms'][name]['accuracy'],'Accuracy/CIs differ: '+name)
 transitions={'wrong_to_right':0,'right_to_wrong':0,'draft_correct':0,'drafts':0,'semantic_repairs':0}
 for q in ids:
  rev=indexed[q].get('revision') or {};draft=rev.get('draft')
  if not isinstance(draft,dict):continue
  dc=score.score_answer(draft.get('answer'),source[q]['true_answer'],source[q]['question'],families[q]['template'])['typed_correct'];fc=scores[q]
  transitions['drafts']+=1;transitions['draft_correct']+=dc;transitions['wrong_to_right']+=not dc and bool(fc);transitions['right_to_wrong']+=bool(dc) and not fc;transitions['semantic_repairs']+=bool(rev.get('revised'))
 obs={'terminal_errors':sum(failed(r) for r in indexed.values()),'budget_stops':sum(bool((r.get('budget') or {}).get('stopped')) for r in indexed.values()),'transitions':transitions,'latency':stats.latency_summary([indexed[q].get('latency') for q in ids])}
 tokens=[(indexed[q].get('tokens') or {}).get('total') for q in ids];obs['mean_tokens']=statistics.fmean(tokens) if None not in tokens else None
 for k,v in obs.items():require(v==official['arms'][name][k],f'{name} {k} differs')
 records[name]=scores;details[name]={r['question_id']:r for r in rows};summaries[name]={'correct':sum(scores.values()),'n':300,'terminal_errors':obs['terminal_errors'],'budget_stops':obs['budget_stops'],'result_sha256':binding['result_sha256']}
for cmp in official['comparisons']:
 a,b=cmp['comparison'].split(' - ');x=[records[a][q] for q in ids];y=[records[b][q] for q in ids]
 computed={'comparison':cmp['comparison'],'wins':sum(p and not v for p,v in zip(x,y)),'losses':sum(v and not p for p,v in zip(x,y)),**stats.paired_difference(x,y,[source[q]['patient_fhir_id'] for q in ids],n_boot=10000,seed=5106)}
 require(computed==cmp,'Paired comparison/CI differs: '+cmp['comparison'])
require(sha(F/'results.json')==before,'Official results changed')
current=summaries['v4_team']['correct'];fresh=summaries['v4_single']['correct'];gate=current>238 and current>fresh
report={'status':'PASS_OFFICIAL_RESULTS_REPRODUCED','results_sha256':before,'completion_report_sha256':sha(F/'completion_protocol_audit.json'),'question_ids':300,'patients':58,'all_six_arms_complete_first_attempts':True,'all_accuracy_family_intervals_and_15_pair_intervals_exact':True,'arms':summaries,'acceptance':{'team_correct':current,'historical_best_single':238,'fresh_single_correct':fresh,'minimum_predeclared':239,'team_exceeds_238':current>238,'team_exceeds_fresh_single':current>fresh,'accepted':gate},'limits':['Typed scoring does not adjudicate clinical/source correctness. Reused subset results are exploratory. Known timeout failure counts incorrect, with unknown failed usage retained.'],'scoring_performed':True,'model_calls':0,'network_calls':0}
pathlib.Path('/private/tmp/fhir_v4_official_acceptance_review.json').write_text(json.dumps(report,indent=2)+'\n')
pathlib.Path('/private/tmp/fhir_v4_scored_compact_rows.json').write_text(json.dumps(details,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2))
