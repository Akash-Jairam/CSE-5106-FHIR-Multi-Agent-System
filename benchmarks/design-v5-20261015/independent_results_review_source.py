"""Independent completed V5 DEVELOPMENT scoring only; no full/confirmation QA loads."""
import csv
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True

ROOT=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
RECORD=ROOT/'benchmarks/design-v5-20261015'
HERE=Path(__file__).parent
GUARD_SHA='10e7b2b537f529a146695606ca8d89690cfdb3fac8d9b3465d82b5eec6f01f29'
PINS_SHA='38b75204e50cbb2c0e7f929daf931f4ffa37ad6df4002cb566812eb01e1e7b50'
COMPLETION_SHA='7c80f4a6784fb8e141ad2a8b7b2e93357469789768b55a09631d67696a620289'


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def failed(row):
    return row.get('error') is not None and str(row['error']).strip() not in ('','nan')


require(sha(RECORD/'v5_guard.py')==GUARD_SHA,'Completion guard source differs')
require(sha(RECORD/'guard_pins.json')==PINS_SHA,'Prelaunch pins differ')
require(sha(RECORD/'completion_protocol_audit.json')==COMPLETION_SHA,'Pinned completed integrity report differs')
guard=module(RECORD/'v5_guard.py','independent_completed_v5_guard')
completion=read(RECORD/'completion_protocol_audit.json');manifest=read(RECORD/'manifest.json')
output=Path(manifest['output_directory'])
require(completion['status']=='PASS' and completion['mode']=='completion'
        and completion['caller_asserted_owned_jobs_terminal'] is True and completion['expected_first_attempts']==600
        and completion['restored_audit_records']==600,'Not complete first-attempt provenance')
pins,manifest,adapter,binding=guard.frozen_provenance(ROOT,RECORD,output,RECORD/'guard_pins.json',PINS_SHA)
require(guard.actual_source_inventory(ROOT)==set(manifest['source_sha256']),'Frozen source inventory differs')
require(completion['manifest_sha256']==sha(RECORD/'manifest.json') and completion['pins_sha256']==PINS_SHA
        and completion['guard_sha256']==GUARD_SHA,'Completion provenance differs')
for name,meta in completion['raw_files'].items():
    path=guard.B.owned_file(output,name)
    require(sha(path)==meta['sha256'] and path.stat().st_size==meta['bytes'],'Raw completion bytes differ')
expected=read(ROOT/'manifests/split.json')['splits']['held_out']['question_ids']
require(len(expected)==len(set(expected))==300,'Wrong development cohort')
expected_set=set(expected)
# This metadata file has no reference answers/SQL. Only fixed300 metadata is retained.
families={}
with (ROOT/'manifests/question_families.csv').open(newline='') as stream:
    for row in csv.DictReader(stream):
        if row['question_id'] in expected_set:families[row['question_id']]={k:row[k] for k in ('family','template')}
require(set(families)==expected_set,'Fixed development family metadata missing')
# Load exact pure scorer directly; do not import utils' provider/config bootstrap.
for name in ['utils/answer_scorer.py','scripts/experiment_stats.py','scripts/full_benchmark.py']:
    require(sha(ROOT/name)==manifest['source_sha256'][name],'Frozen public scoring source differs')
scorer=module(ROOT/'utils/answer_scorer.py','independent_v5_pure_answer_scorer')
reader=module(ROOT/'scripts/full_benchmark.py','independent_v5_streaming_reader')
import numpy as np


def cluster_interval(values,patients):
    # Independently aggregate patient clusters rather than calling paired_difference.
    groups=sorted(set(patients));totals=np.array([sum(v for v,p in zip(values,patients) if p==g) for g in groups],dtype=float)
    sizes=np.array([sum(p==g for p in patients) for g in groups],dtype=float)
    draws=np.random.default_rng(5106).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=10000)
    sampled=(draws@totals)/(draws@sizes);tail=(1-.95)/2*100
    low,high=np.percentile(sampled,[tail,100-tail])
    return {'n':len(values),'patients':len(groups),'estimate':sum(values)/len(values),'ci_low':float(low),'ci_high':float(high)}


records={};summaries={};identities={};input_columns=['question_id','question','true_answer','assumption','patient_fhir_id']
for arm,strategy in [('single','single_agent'),('team','multi_agent_v5')]:
    logical='ordinary_single_agent' if arm=='single' else 'ordinary_multi_agent'
    path=output/(logical+'.json');identity_path=Path(str(path)+'.run.json');identity=read(identity_path)
    evidence=completion['arms'][logical]
    require(sha(path)==evidence['result_sha256'] and sha(identity_path)==evidence['identity_sha256'],'Current arm SHA differs')
    require(identity['agent_strategy']==strategy and identity['model']=='gpt-6-luna'
            and identity['max_total_tokens']==200000 and identity['max_tool_calls']==30
            and identity['audit_storage']=='gzip' and identity['input_hash']==completion['ordered_input_hash'],'Actual development identity differs')
    kept={};ordered=[];counts=Counter();by_family={}
    for row in reader.iter_result_rows(path):
        qid=row['question_id'];ordered.append(qid)
        require(qid in expected_set and qid not in kept and not row.get('attempt_history') and guard.B.complete(row),'Invalid first-attempt cohort')
        result=scorer.score_answer(row.get('agent_answer'),row['true_answer'],row['question'],families[qid]['template'])
        error=failed(row);correct=int(not error and bool(result['typed_correct']))
        value={'question_id':qid,'patient':row['patient_fhir_id'],'correct':correct,'failed':error,
               'is_none':row.get('agent_answer') is None,'would_be_typed_correct_ignoring_error':int(result['typed_correct']),
               'answer_type':result['answer_type'],'inputs':{k:row[k] for k in input_columns}}
        kept[qid]=value
        counts.update(correct=correct,terminal_errors=int(error),exact_none_answers=int(value['is_none']),
                      correct_none_answers=int(value['is_none'] and correct),
                      error_answers_that_typed_scorer_alone_would_credit=int(error and result['typed_correct']))
        by_family.setdefault(families[qid]['family'],Counter()).update(n=1,correct=correct,errors=int(error))
    require(ordered==expected and len(kept)==300,'Ordered fixed300 differs')
    counts['n']=300;counts['accuracy']=counts['correct']/300
    counts['answer_types']=dict(Counter(r['answer_type'] for r in kept.values()))
    counts['families']={k:dict(v) for k,v in by_family.items()}
    records[arm]=kept;summaries[arm]=dict(counts);identities[arm]={'identity':identity,'result_sha256':sha(path),'identity_sha256':sha(identity_path)}
for qid in expected:
    require(guard.equal(records['single'][qid]['inputs'],records['team'][qid]['inputs']),'Pair source inputs differ')
ids=sorted(expected);patients=[records['single'][q]['patient'] for q in ids]
a=[records['team'][q]['correct'] for q in ids];b=[records['single'][q]['correct'] for q in ids]
d=[x-y for x,y in zip(a,b)]
paired={**cluster_interval(d,patients),'wins':sum(x and not y for x,y in zip(a,b)),
        'losses':sum(y and not x for x,y in zip(a,b)),'both_correct':sum(x and y for x,y in zip(a,b)),
        'both_wrong':sum(not x and not y for x,y in zip(a,b)),'comparison':'V5 team - contemporary single'}
# Reuse public stats only as an after-the-fact method-equivalence crosscheck.
public_stats=module(ROOT/'scripts/experiment_stats.py','independent_v5_public_statistics_crosscheck')
require(guard.equal(public_stats.paired_difference(a,b,patients,n_boot=10000,seed=5106),
                    {k:paired[k] for k in ('n','patients','estimate','ci_low','ci_high')}),'Independent bootstrap differs from frozen method')
strict_team=[int(records['team'][q]['correct'] and not records['team'][q]['is_none']) for q in ids]
strict_single=[int(records['single'][q]['correct'] and not records['single'][q]['is_none']) for q in ids]
null_sensitivity={'label':'Posthoc exact-None-credit sensitivity only; unchanged typed scorer remains primary, no questions excluded',
    'single_correct_if_no_exact_None_credit':sum(strict_single),'team_correct_if_no_exact_None_credit':sum(strict_team),
    'paired':cluster_interval([x-y for x,y in zip(strict_team,strict_single)],patients)}
for name,meta in completion['raw_files'].items():require(sha(output/name)==meta['sha256'],'Raw source changed during scoring')
guard.frozen_provenance(ROOT,RECORD,output,RECORD/'guard_pins.json',PINS_SHA)
require(sha(RECORD/'completion_protocol_audit.json')==COMPLETION_SHA,'Completion report changed')
report={'status':'INDEPENDENT_COMPLETED_DEVELOPMENT_RECOMPUTATION','scope':'Original fixed300 only; no full or protected1397 references loaded',
    'script_sha256':sha(Path(__file__)),'integrity_report_sha256':COMPLETION_SHA,'guard_sha256':GUARD_SHA,'pins_sha256':PINS_SHA,
    'source_files_verified':len(manifest['source_sha256']),'runtime_files_verified':4,'raw_files_hashed':len(completion['raw_files']),
    'ordered_input_hash':completion['ordered_input_hash'],'identities':identities,'arms':summaries,'paired':paired,
    'bootstrap':{'replicates':10000,'seed':5106,'unit':'patient clusters','estimand':'question-weighted mean paired correctness difference'},
    'gate':{'team_above_historical_single_239':summaries['team']['correct']>239,
            'team_above_contemporary_single':summaries['team']['correct']>summaries['single']['correct'],
            'pass':summaries['team']['correct']>239 and summaries['team']['correct']>summaries['single']['correct']},
    'null_credit_sensitivity':null_sensitivity,
    'limits':['All300 firsts retained; the one terminal Single failure is wrong regardless of output. Unreported failed usage remains unknown.',
              'Reference question/answer inputs came only from SHA-bound saved fixed300 rows; no full dataset QA/SQL parsing or protected-case access.',
              'Exact-None credit follows the unchanged benchmark empty-reference convention; it is not evidence of search completeness/source-supported abstention.',
              'This is adaptive development selection, not confirmation or patient/model/dataset generalization.',
              'No source/critic mechanism claim is made from compact revision fields; separate restored analysis is required.'],
    'actual_model_FHIR_network_private_configuration_calls':0,'protected_full_confirmation_scoring':False}
(HERE/'independent_verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(HERE/'paired_outcome_metadata.json').write_text(json.dumps([{'question_id':q,'patient':records['single'][q]['patient'],
    **{arm:{k:records[arm][q][k] for k in ('correct','failed','is_none','answer_type')} for arm in ('single','team')}} for q in ids],indent=2)+'\n')
print(json.dumps({'arms':summaries,'paired':paired,'gate':report['gate'],'null_credit_sensitivity':null_sensitivity},indent=2))
