"""Independent completed fixed300 only; no full-QA/config/model/restore access."""
import os
os.environ['FHIR_LOCAL_AZURE_CONFIG']=''
os.environ['FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR']=''
import csv,hashlib,importlib.util,json,math
from pathlib import Path
import numpy as np
REPO=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
RECORD=REPO/'benchmarks/design-v7-20261015'
OUT=Path(__file__).parent
FORBIDDEN=REPO/'final_dataset/questions_answers_sql_fhir.csv'
def sha(path):
    path=Path(path)
    assert path.resolve()!=FORBIDDEN.resolve(),'Full QA dataset access forbidden'
    assert path.name!='config.yml' and not ('.local-setup' in path.parts and path.name.startswith('config')),'Credential file access forbidden'
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(1024*1024):digest.update(chunk)
    return digest.hexdigest()
def load(path):return json.loads(Path(path).read_text(),parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite input')))
def same(left,right):return json.dumps(left,sort_keys=True,allow_nan=False)==json.dumps(right,sort_keys=True,allow_nan=False)
def failed(row):return row.get('error') is not None and str(row['error']).strip() not in ('','nan')
def pure(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
proof=load(RECORD/'completion_integrity.json');official=load(RECORD/'comparison.json');pins=load(RECORD/'integrity_pins.json');manifest=load(RECORD/'manifest.json')
assert sha(RECORD/'completion_integrity.json')=='a27e383569198753c326b41e4fc7537387c66ee2593d47f0d97dbccb48b38202'
assert sha(RECORD/'owned_jobs_completion.json')=='3c6682a2c760933cd56fa3101af7c7232a13ef2679dc2b269c97af2fd692b2a9'
assert sha(RECORD/'integrity_pins.json')=='eea97f39b4a7cfa58fb363c4f4bd92990c1c7484ba4ac0e0171fbed1c6e60827'
assert sha(RECORD/'comparison.json')=='2ef2ae8342ade4d77352f424fda9ff5087eefac718eb06dd07ceedf8bb27fe0e'
assert proof['status']=='PASS' and proof['mode']=='completion' and proof['scoring_performed'] is False
assert proof['restored_audit_records']==proof['expected_first_attempts']==600 and proof['frozen_source_files_verified']==182
assert proof['guard_sha256']=='bf47aa9e9c84735b0a096acbb6d209776473fcb7e81e8cfa2e366576d90e1115'
assert proof['caller_asserted_owned_jobs_terminal'] is True
owned=load(RECORD/'owned_jobs_completion.json');assert owned['both_owned_jobs_terminal'] is True and len(owned['jobs'])==2
assert all(j['saved_first_attempts']==300 and type(j['exit_code']) is int for j in owned['jobs'])
for name,digest in pins['record_metadata_sha256'].items():assert sha(RECORD/name)==digest
assert sha(RECORD/'source.zip')==manifest['source_archive_sha256']==proof['source_archive_sha256']
assert sha(RECORD/'design.txt')==manifest['design_sha256']==proof['design_sha256']
for name,digest in manifest['source_sha256'].items():assert sha(REPO/name)==digest,name
assert len(manifest['source_sha256'])==182
adapter=load(RECORD/'runtime_adapter_manifest.json')
assert sha(RECORD/'runtime_adapter_sources.zip')==adapter['archive_sha256']
assert len(adapter['source_sha256'])==4
for name,digest in adapter['source_sha256'].items():assert sha(REPO/name)==digest
assert sha(REPO/'utils/answer_scorer.py')=='ca02e5a8758f5bcef3eee08ab70f69fddf23917d267df6301e83275f337f3964'
assert sha(REPO/'scripts/compare_runs.py')=='607ba1b473bd1de87c95793b0a67819a02921e9c12d1b74780059cd95538192c'
scorer=pure(REPO/'utils/answer_scorer.py','independent_pure_typed_scorer')
stats=pure(REPO/'scripts/experiment_stats.py','independent_pure_statistics')
expected=load(REPO/'manifests/split.json')['splits']['held_out']['question_ids'];assert len(expected)==len(set(expected))==300
families={}
with (REPO/'manifests/question_families.csv').open(newline='') as stream:
    for row in csv.DictReader(stream):
        if row['question_id'] in set(expected):families[row['question_id']]={'template':row['template'],'family':row['family']}
assert set(families)==set(expected)
raw=REPO/'output/experiments/gpt-6-luna/residual_trajectories_v7_300q';arms={};identity={};evidence={}
for label,arm,strategy in [('v7_single','ordinary_single_agent','single_agent'),('v7_team','ordinary_multi_agent','multi_agent_v7')]:
    path=raw/(arm+'.json');side=raw/(arm+'.json.run.json');rows=load(path);identity[label]=load(side)
    assert len(rows)==300 and [r['question_id'] for r in rows]==expected
    assert all(not r.get('attempt_history') for r in rows)
    assert all(r.get('latency') is not None or failed(r) for r in rows)
    assert sha(path)==proof['arms'][arm]['result_sha256']==official['identities'][label]['result_sha256']
    assert sha(side)==proof['arms'][arm]['identity_sha256']
    assert same(identity[label],official['identities'][label]['identity'])
    assert identity[label]['agent_strategy']==strategy and identity[label]['model']=='gpt-6-luna'
    assert identity[label]['max_total_tokens']==200000 and identity[label]['max_tool_calls']==30
    assert identity[label]['audit_storage']=='gzip' and identity[label]['split']=='held_out'
    assert Path(official['identities'][label]['result_path']).resolve()==path
    values={}
    for row in rows:
        qid=row['question_id'];typed=scorer.score_answer(row.get('agent_answer'),row['true_answer'],row['question'],families[qid]['template'])
        correct=bool(not failed(row) and typed['typed_correct'])
        values[qid]={'correct':correct,'error':failed(row),'nonerror_null_credit':not failed(row) and row.get('agent_answer') is None and correct,
            'error_ignored_would_match':failed(row) and bool(typed['typed_correct']),'patient':row['patient_fhir_id']}
    arms[label]={'rows':{r['question_id']:r for r in rows},'values':values}
    evidence[label]={'result_sha256':sha(path),'identity_sha256':sha(side),'n':300,'correct':sum(v['correct'] for v in values.values()),
        'terminal_errors':sum(v['error'] for v in values.values()),'all_terminal_errors_incorrect':all(not v['correct'] for v in values.values() if v['error']),
        'nonerror_exact_null_credits':sum(v['nonerror_null_credit'] for v in values.values()),'terminal_rows_that_would_match_if_error_ignored':sum(v['error_ignored_would_match'] for v in values.values())}
    assert evidence[label]['correct']==official['arms'][label]['accuracy']['all']['correct']
    assert evidence[label]['terminal_errors']==official['arms'][label]['terminal_errors']==proof['arms'][arm]['terminal_errors']
for qid in sorted(expected):
    a,b=(arms[label]['rows'][qid] for label in ('v7_single','v7_team'))
    assert all(same(a.get(k),b.get(k)) for k in ('question_id','question','true_answer','assumption','patient_fhir_id','question_with_context'))
assert same({k:v for k,v in identity['v7_single'].items() if k!='agent_strategy'}, {k:v for k,v in identity['v7_team'].items() if k!='agent_strategy'})
ids=sorted(expected);a=np.array([arms['v7_team']['values'][qid]['correct'] for qid in ids],dtype=float);b=np.array([arms['v7_single']['values'][qid]['correct'] for qid in ids],dtype=float)
patients=np.array([arms['v7_team']['values'][qid]['patient'] for qid in ids],dtype=object)
# Independent implementation of unchanged compare's seed path: every paired call
# seeds its own RNG with5106; clusters sorted by np.unique; multinomial weights;
# question-weighted ratio, not mean of patient means; 10000 percentile replicates.
_,index=np.unique(patients.astype(str),return_inverse=True)
sums=np.bincount(index,weights=a-b);counts=np.bincount(index).astype(float)
rng=np.random.default_rng(5106);weights=rng.multinomial(len(counts),np.full(len(counts),1/len(counts)),size=10000)
boot=weights@sums/(weights@counts);tail=(1-.95)/2*100;lo,hi=np.percentile(boot,[tail,100-tail])
pair={'comparison':'v7_team - v7_single','n':300,'patients':len(counts),'wins':int(np.sum((a==1)&(b==0))),'losses':int(np.sum((a==0)&(b==1))),
    'estimate':float(np.mean(a-b)),'ci_low':float(lo),'ci_high':float(hi),'bootstrap_replicates':10000,'seed':5106}
shared=stats.paired_difference(a,b,patients,n_boot=10000,seed=5106)
assert all(pair[k]==shared[k] for k in ('n','patients','estimate','ci_low','ci_high'))
comparison=next(c for c in official['comparisons'] if c['comparison']=='v7_team - v7_single')
assert all(pair[k]==comparison[k] for k in ('n','patients','wins','losses','estimate','ci_low','ci_high'))
assert hashlib.sha256(json.dumps(ids).encode()).hexdigest()==official['question_ids_sha256']
prospective=load(RECORD/'prospective_protocol.json')
assert prospective['minimum_team_correct']==240
assert prospective['acceptance']=='Team >239 AND Team >fresh contemporary Single AND Team >actual recorded full-cap A-only; all300 first attempts/errors, no replacements/retries. Missing A reconstruction makes the last gate unavailable and prevents selection.'
assert prospective['recorded_A_only_gate']=='Strict positive net auxiliary contribution; actual normalized released A/error and primary fatal state, never computed withheld or fabricated null. All300 reconstructible outcomes required.'
gate={'team_beats_prior239':evidence['v7_team']['correct']>239,'team_beats_fresh_single':evidence['v7_team']['correct']>evidence['v7_single']['correct'],
    'A_only_gate':'Not recomputed here; root separately runs reviewed posthoc. All three conjuncts are required.','selection_possible':False}
assert not gate['team_beats_prior239'] and not gate['team_beats_fresh_single']
result={'status':'INDEPENDENT_FIXED300_VERIFICATION_PASS','all300_first_attempts_and_errors_retained':True,'arms':evidence,'paired':pair,'gates':gate,
    'completion_report_sha256':sha(RECORD/'completion_integrity.json'),'owned_attestation_sha256':sha(RECORD/'owned_jobs_completion.json'),
    'official_report_sha256':sha(RECORD/'comparison.json'),'pins_sha256':sha(RECORD/'integrity_pins.json'),'source_files_rehashed':182,'public_runtime_files_rehashed':4,
    'source_sha256':{p:sha(REPO/p) for p in ['utils/answer_scorer.py','scripts/compare_runs.py','scripts/experiment_stats.py','scripts/analysis.py','scripts/variability.py']},
    'cohort_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'seed_path':'Separate default_rng(5106) for each pair;10000 multinomial patient-cluster weight draws; question-weighted ratio; percentile95%CI, exactly unchanged compare path.',
    'limits':['Fixed300 is adaptive development, not protected confirmation or generalization evidence.','Normal nonerror null/empty-reference credit follows unchanged typed scorer; terminal errors are always wrong, even if their raw answer would match.','Source/gzip proof600 is reused; no programs or deep CRCs were rerun here.','No broader QA CSV was opened or hashed. References/question text used solely from the two authorized completed saved300 cohorts; family templates only for those IDs.','Actual A-only/selection mechanisms are not inferred from compact raw revision data and are left to reviewed posthoc.'],
    'model_calls':0,'FHIR_calls':0,'network_calls':0,'config_values_read':False,'protected_scores_computed':False,'repository_edits':False}
# Recheck every scored raw/official/proof/policy/source binding before publication.
assert sha(RECORD/'completion_integrity.json')==result['completion_report_sha256']
assert sha(RECORD/'comparison.json')==result['official_report_sha256']
for label,arm in [('v7_single','ordinary_single_agent'),('v7_team','ordinary_multi_agent')]:assert sha(raw/(arm+'.json'))==evidence[label]['result_sha256']
for name,digest in manifest['source_sha256'].items():assert sha(REPO/name)==digest
(OUT/'independent_verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':result['status'],'arms':evidence,'paired':pair,'gate':gate},indent=2))
