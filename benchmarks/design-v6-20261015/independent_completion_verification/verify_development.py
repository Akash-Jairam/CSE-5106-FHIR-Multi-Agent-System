"""Independent completed FIXED300 V6 verification; no full CSV parsing/inference.

Uses only SHA-bound completed two-arm fixed300 rows for reference values. Reuses
completion evidence for all600 gzip CRC/restoration; does not repeat that audit.
"""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

OFFICIAL_SHA='678ed113da6e7b785d7e5ef2ce12dfb4db0c6ae7c11584aeff4a91316542f37c'
COMPLETION_SHA='ba7a383078218c39c45ec4b485414b1adb248d14cd6e73fe83afd28bcb9d92fd'
COMPARE_SHA='607ba1b473bd1de87c95793b0a67819a02921e9c12d1b74780059cd95538192c'
SCORER_SHA='ca02e5a8758f5bcef3eee08ab70f69fddf23917d267df6301e83275f337f3964'
PINS_SHA='e877d4e227a19f519e785c675ced1283eac3878022a0332a7e4f6f91a058017a'
GUARD_SHA='49ec8be98372cbeaf9c9ab917d02fdf0bbbdce06471a8a0745a02aae9c485c18'
ARCHIVE_SHA='0fc40fac89f079b1f7bb29bd8adf28ed976a387d22a5561476de2cc7f79b2bdb'
INPUT_COLUMNS=['question_id','question','true_answer','assumption','patient_fhir_id']
ARMS={'v6_single':('ordinary_single_agent','single_agent'),'v6_team':('ordinary_multi_agent','multi_agent_v6')}
MAX_RESULT_RECORD_CHARS=64*1024*1024


def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def pairs(items):
    result={}
    for key,value in items:
        require(key not in result,'Duplicate JSON key');result[key]=value
    return result

def bad_constant(_):raise ValueError('Nonfinite JSON')

def load(path):return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=bad_constant)

def same(a,b):return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)

def failed(row):return row.get('error') is not None and str(row['error']).strip() not in ('','nan')

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def iter_rows(path, chunk_chars=1024 * 1024):
    """Reviewed supervisor array parser, with finite/duplicate-key and size bounds."""
    decoder = json.JSONDecoder(object_pairs_hook=pairs, parse_constant=bad_constant)
    with Path(path).open(encoding='utf-8') as stream:
        buffer, position, ended = '', 0, False

        def fill():
            nonlocal buffer, position, ended
            fragment = stream.read(chunk_chars)
            buffer, position = buffer[position:] + fragment, 0
            require(len(buffer) <= MAX_RESULT_RECORD_CHARS, 'Oversized result record')
            ended = not fragment

        def whitespace():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or ended:
                    return
                fill()

        whitespace()
        require(position < len(buffer) and buffer[position] == '[', 'Result must be a JSON array')
        position += 1
        whitespace()
        if position < len(buffer) and buffer[position] == ']':
            position += 1
        else:
            while True:
                while True:
                    try:
                        row, end = decoder.raw_decode(buffer, position)
                        break
                    except json.JSONDecodeError:
                        require(not ended, 'Incomplete result record')
                        fill()
                position = end
                yield row
                whitespace()
                if position < len(buffer) and buffer[position] == ']':
                    position += 1
                    break
                require(position < len(buffer) and buffer[position] == ',', 'Invalid result separator')
                position += 1
                whitespace()
        whitespace()
        require(position == len(buffer), 'Data after result array')


def independently_clustered(values,patients,n_boot=10000,seed=5106):
    # Match the exact frozen compare -> paired_difference -> clustered_mean ->
    # _resampled_means RNG path, independently grouped without calling stats.
    import numpy as np
    y=np.asarray(values,dtype=float);labels=np.asarray(patients,dtype=object).astype(str)
    unique=sorted(set(labels));totals=np.asarray([y[labels==patient].sum() for patient in unique],dtype=float)
    sizes=np.asarray([sum(labels==patient) for patient in unique],dtype=float)
    rng=np.random.default_rng(seed)
    weights=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=n_boot)
    bootstrap=(weights@totals)/(weights@sizes)
    tail=(1-0.95)/2*100
    low,high=np.percentile(bootstrap,[tail,100-tail])
    return {'n':len(y),'patients':len(unique),'estimate':float(y.mean()),'ci_low':float(low),'ci_high':float(high)}


def verify(root,report):
    root=Path(root).resolve();report=Path(report).absolute()
    require(report.suffix=='.json' and report.is_relative_to(Path('/private/tmp')) and not report.is_relative_to(root)
      and not report.exists() and not any(x.is_symlink() for x in (report,*report.parents)),'Fresh external private report required')
    record=root/'benchmarks/design-v6-20261015';output=root/'output/experiments/gpt-6-luna/independent_recomputation_v6_300q'
    checked={}
    def pin(path,digest):
        require(not path.is_symlink() and path.is_file() and sha(path)==digest,'SHA differs: '+str(path))
        checked[path]=digest
    pin(record/'results.json',OFFICIAL_SHA);pin(record/'completion_protocol_audit.json',COMPLETION_SHA)
    pin(record/'guard_pins.json',PINS_SHA);pin(record/'v6_guard.py',GUARD_SHA)
    official=load(record/'results.json');completion=load(record/'completion_protocol_audit.json');pins=load(record/'guard_pins.json')
    require(completion['status']=='PASS' and completion['mode']=='completion' and completion['caller_asserted_owned_jobs_terminal'] is True
      and completion['scoring_performed'] is False and completion['expected_first_attempts']==600
      and completion['restored_audit_records']==600,'Complete unscored deep600 PASS required')
    require(completion['guard_sha256']==GUARD_SHA and completion['pins_sha256']==PINS_SHA,'Guard/pin evidence mismatch')
    for name,digest in pins['record_metadata_sha256'].items():pin(record/name,digest)
    manifest=load(record/'manifest.json');runtime=load(record/'runtime_adapter_manifest.json')
    require(len(manifest['source_sha256'])==pins['source_file_count']==completion['frozen_source_files_verified']==176,'Source freeze count differs')
    require(manifest['code_commit']=='33b2d9d87db48fd38abba28803b4116ee8aca59c','Unexpected frozen source commit')
    require(manifest['source_sha256']['scripts/compare_runs.py']==COMPARE_SHA and manifest['source_sha256']['utils/answer_scorer.py']==SCORER_SHA,'Scoring source differs')
    for name,digest in manifest['source_sha256'].items():
        require(name!='config.yml' and not name.startswith('.local-setup/config'),'Secret configuration forbidden')
        pin(root/name,digest)
    for filename,mapping,digest in [('source.zip',manifest['source_sha256'],manifest['source_archive_sha256']),
      ('runtime_adapter_sources.zip',runtime['source_sha256'],runtime['archive_sha256'])]:
        path=record/filename;pin(path,digest)
        with zipfile.ZipFile(path) as archive:
            require(len(archive.namelist())==len(mapping) and set(archive.namelist())==set(mapping),'Source ZIP inventory differs')
            for name,expected in mapping.items():require(hashlib.sha256(archive.read(name)).hexdigest()==expected,'Source ZIP member differs')
    for name,digest in runtime['source_sha256'].items():pin(root/name,digest)
    require(len(runtime['source_sha256'])==completion['runtime_adapter_files_verified']==4,'Runtime count differs')
    require(load(record/'expected_public_binding.json')==completion['provider_binding'],'Provider binding differs')
    for name,digest in [('design.txt',manifest['design_sha256'])]:pin(record/name,digest)
    # The full QA/SQL CSV is only opaque-hashed, NEVER parsed or decoded.
    pin(root/'final_dataset/questions_answers_sql_fhir.csv',manifest['dataset_sha256'])
    pin(root/'manifests/split.json',manifest['split_sha256'])
    ids=load(root/'manifests/split.json')['splits']['held_out']['question_ids'];require(len(ids)==len(set(ids))==300,'Fixed300 differs')
    require(official['questions']==300 and official['patients']==58 and official['cohort']=='held_out'
      and official['dataset_sha256']==manifest['dataset_sha256']
      and official['question_ids_sha256']==hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest(),'Official cohort differs')
    evidence=load(record/'completion_evidence_pins.json');owned=load(record/'owned_jobs_completion.json')
    require(evidence['completion_report_sha256']==COMPLETION_SHA and evidence['both_owned_jobs_terminal'] is True,'Evidence pin differs')
    pin(record/'owned_jobs_completion.json',evidence['owned_jobs_completion_sha256'])
    require(owned['both_owned_jobs_terminal'] is True and {(r['arm'],r['exit_code'],r['saved_first_attempts']) for r in owned['jobs']}=={('single',1,300),('team',1,300)},'Root terminal record differs')
    archive=root/evidence['archive']['path'];pin(archive,ARCHIVE_SHA)
    require(archive.stat().st_size==evidence['archive']['bytes']==217850594,'Archive size differs')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();require(len(names)==len(set(names))==609==evidence['archive']['members'],'Archive member count differs')
        require(set(names)==set(completion['raw_files']),'Archive allowlist differs from deep report')
        require(all(z.getinfo(name).file_size==completion['raw_files'][name]['bytes'] for name in names),'Archive member size differs')
    family_path=root/'manifests/question_families.csv';require(official['families_sha256']==manifest['source_sha256']['manifests/question_families.csv'],'Family binding differs')
    families={}
    with family_path.open(newline='') as stream:
        for value in csv.DictReader(stream):
            if value['question_id'] in ids:require(value['question_id'] not in families,'Duplicate metadata');families[value['question_id']]=value
    require(set(families)==set(ids),'Missing fixed300 template metadata')
    rows={};identities={}
    keep={*INPUT_COLUMNS,'agent_answer','error','latency','tokens','budget','attempt_history','audit_artifacts'}
    for label,(arm,strategy) in ARMS.items():
        path=output/(arm+'.json');sidecar=output/(arm+'.json.run.json');facts=completion['arms'][arm]
        for item in (path,sidecar):
            entry=completion['raw_files'][item.name];require(item.stat().st_size==entry['bytes'],'Raw size differs');pin(item,entry['sha256'])
        require(facts['saved_first_attempts']==facts['verified_owned_gzip_audits']==300 and facts['result_sha256']==sha(path),'Completed arm differs')
        identity=load(sidecar);identities[label]=identity
        require(identity['agent_strategy']==strategy and identity['model']=='gpt-6-luna'
          and identity['max_total_tokens']==200000 and identity['max_tool_calls']==30
          and identity['input_hash']==completion['ordered_input_hash'] and identity['code_hash']==completion['code_hash']
          and identity['split']=='held_out' and identity['audit_storage']=='gzip' and identity['audit_artifact_schema']==2,'Actual identity/caps differ')
        binding=official['identities'][label]
        require(Path(binding['result_path']).resolve()==path.resolve() and binding['result_sha256']==sha(path) and same(binding['identity'],identity),'Official raw/identity differs')
        compact=[{key:row.get(key) for key in keep} for row in iter_rows(path)]
        require([r['question_id'] for r in compact]==ids,'Ordered first300 cohort differs')
        require(all(not r['attempt_history'] and (failed(r) or type(r['latency']) in (int,float)) for r in compact),'Retry or missing first outcome')
        require(all(isinstance(r['audit_artifacts'],dict) and r['audit_artifacts']['question_id']==r['question_id'] for r in compact),'Missing owned audit reference')
        rows[label]={r['question_id']:r for r in compact}
    for qid in ids:require(all(same(rows['v6_single'][qid][key],rows['v6_team'][qid][key]) for key in INPUT_COLUMNS),'Paired input metadata differs')
    import pandas as pd
    for label in ARMS:
        frame=pd.DataFrame([rows[label][qid] for qid in ids])[INPUT_COLUMNS]
        digest=hashlib.sha256(frame.to_json(orient='records',date_format='iso').encode()).hexdigest()
        require(digest==identities[label]['input_hash'],'Recorded fixed300 authoritative input hash differs')
    for path,digest in checked.items():require(sha(path)==digest,'Input changed before pure scorer import')
    scorer=module(root/'utils/answer_scorer.py','independent_V6_typed_scorer')
    stats=module(root/'scripts/experiment_stats.py','independent_V6_frozen_statistics')
    import numpy as np
    scored={};arms={};patients=[rows['v6_single'][qid]['patient_fhir_id'] for qid in sorted(ids)]
    for label in ARMS:
        records=[]
        for qid in sorted(ids):
            row=rows[label][qid];typed=scorer.score_answer(row['agent_answer'],row['true_answer'],row['question'],families[qid]['template'])
            records.append({'qid':qid,'correct':int(not failed(row) and typed['typed_correct']),
              'failed':failed(row),'typed_ignoring_error':bool(typed['typed_correct']),
              'is_None':row['agent_answer'] is None,'reference_empty':typed['answer_type']=='empty',
              'known_budget_stop':bool((row.get('budget') or {}).get('stopped')),
              'unknown_usage':any(c.get('usage_available') is not True for c in (row.get('tokens') or {}).get('per_call') or [])})
        scored[label]=records;correct=[r['correct'] for r in records]
        summary={'correct':sum(correct),'n':300,'terminal_errors':sum(r['failed'] for r in records),
          'all_error_rows_scored_wrong':all(not r['correct'] for r in records if r['failed']),
          'error_rows_that_would_get_typed_credit_if_error_ignored':sum(r['failed'] and r['typed_ignoring_error'] for r in records),
          'actual_nonerror_None_rows':sum(r['is_None'] and not r['failed'] for r in records),
          'actual_nonerror_None_typed_credits':sum(r['is_None'] and r['correct'] for r in records),
          'reference_empty_rows':sum(r['reference_empty'] for r in records),
          'known_budget_stops':sum(r['known_budget_stop'] for r in records),'unknown_usage_attempts':sum(r['unknown_usage'] for r in records),
          'accuracy':independently_clustered(correct,patients)}
        target=official['arms'][label]
        require(summary['correct']==target['accuracy']['all']['correct'] and same(summary['accuracy'],{k:target['accuracy']['all'][k] for k in summary['accuracy']})
          and summary['terminal_errors']==target['terminal_errors'] and summary['known_budget_stops']==target['budget_stops'],'Independent arm score differs from official')
        require(same(summary['accuracy'],stats.clustered_mean(correct,patients,n_boot=10000,seed=5106)),'Independent grouping differs from frozen statistics')
        arms[label]=summary
    a=np.asarray([r['correct'] for r in scored['v6_team']],dtype=float);b=np.asarray([r['correct'] for r in scored['v6_single']],dtype=float)
    pair=independently_clustered(a-b,patients)
    pair.update(wins=int(sum((a==1)&(b==0))),losses=int(sum((a==0)&(b==1))),both_correct=int(sum((a==1)&(b==1))),both_wrong=int(sum((a==0)&(b==0))))
    official_pair=next(x for x in official['comparisons'] if x['comparison']=='v6_team - v6_single')
    require(all(same(pair[key],official_pair[key]) for key in ('n','patients','estimate','ci_low','ci_high','wins','losses')),'Independent pair differs from official')
    require(same({k:pair[k] for k in ('n','patients','estimate','ci_low','ci_high')},stats.paired_difference(a,b,patients,n_boot=10000,seed=5106)),'Frozen paired API differs')
    gate={'team_exceeds_239':arms['v6_team']['correct']>239,'team_exceeds_fresh_single':arms['v6_team']['correct']>arms['v6_single']['correct']}
    gate['passed']=all(gate.values())
    require(gate['passed'] is False,'Unexpected conditional selection conclusion')
    for path,digest in checked.items():require(sha(path)==digest,'Evidence changed during verification')
    result={'status':'PASS_INDEPENDENT_RECOMPUTATION_OFFICIAL_GATE_FAILED','scope':'Completed fixed300 two arms only; no protected/full CSV parsing, no new scoring rule or mechanism analysis.',
      'source_commit':manifest['code_commit'],'frozen_source_files_verified':176,'public_runtime_files_verified':4,
      'external_completion_report_sha256':COMPLETION_SHA,'external_official_report_sha256':OFFICIAL_SHA,
      'actual_root_terminal_attestation_sha256':evidence['owned_jobs_completion_sha256'],
      'source_archive_sha256':manifest['source_archive_sha256'],'runtime_archive_sha256':runtime['archive_sha256'],
      'raw_sha256':{label:official['identities'][label]['result_sha256'] for label in ARMS},
      'input_hash':completion['ordered_input_hash'],'paired_patients':len(set(patients)),
      'arms':arms,'team_minus_single':pair,'predeclared_gate':gate,
      'bootstrap_reproduction':{'replicates':10000,'seed':5106,'question_order':'sorted question IDs',
        'cluster_order':'sorted patient ID strings','draw':'fresh numpy.default_rng(5106).multinomial(58, uniform58, size=10000) for each mean/contrast',
        'estimand':'question weighted cluster resampling; all questions retained within each sampled patient',
        'percentiles':'(1-.95)/2*100 and 100-tail, numpy.percentile default linear',
        'numpy_version':np.__version__,'independent_implementation_matches_frozen_stats':True,
        'seed_is_not_shifted_by_arm_pair_index':True},
      'source_sha256':{name:manifest['source_sha256'][name] for name in ('utils/answer_scorer.py','scripts/compare_runs.py','scripts/experiment_stats.py','scripts/analysis.py','scripts/variability.py')},
      'completed_archive':{'sha256':ARCHIVE_SHA,'bytes':217850594,'members':609,'central_directory_allowlist_sizes_verified':True,'audit_CRC_restoration_not_repeated':True},
      'journal_totals_reused_from_pinned_deep_report':completion['journal']['totals'],
      'reproduction_script_sha256':sha(Path(__file__)),
      'limits':['The root-owned actual exit record is an external attestation; this script cannot independently reconstruct past process state.',
        'All600 gzip ownership/SHA/CRC restorations and journal pairing are reused from the externally SHA-pinned completion PASS, not falsely claimed as a new deep audit.',
        'Actual non-error None credit follows unchanged typed scorer and is descriptive; it does not establish source support, complete search or justified abstention.',
        'All six saved provider/accounting failures are incorrect. Their unreported billed usage is unknown, not zero. No failure was retried or excluded.',
        'This adaptive fixed300 result is exploratory. Its interval crosses zero; it does not establish superiority, reliable harm, or protected1397/patient/model/task generalization.',
        'Raw fixed300 inputs supply the authoritative references, bound by the prelaunch expected input hash and deep guard. Full CSV contents are only opaque-hashed.',
        'Auxiliary draft/revision summaries are not assessed here; separate posthoc restoration owns mechanisms. No promotion or full launch follows this failed gate.']}
    with report.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'single':arms['v6_single']['correct'],'team':arms['v6_team']['correct'],'paired':pair,'gate':gate},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args();verify(args.root,args.report)
