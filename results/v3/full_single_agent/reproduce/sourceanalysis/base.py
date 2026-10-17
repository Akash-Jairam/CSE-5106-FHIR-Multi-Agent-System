"""Package the stopped 2,929-row Single Agent cohort: measurements, never scoring.

All imports are answer-free. The frozen analyzer's measurement helpers and the
runner's operational classifier are loaded as reviewed pure AST subsets only.
Clinical result objects are archived verbatim/losslessly, never displayed.
"""
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
sys.path.insert(0, str(ROOT))
from scripts.audit_store import restore_result_audit, ARTIFACT_PATTERN
from scripts.full_benchmark import iter_result_rows
from scripts.experiment_stats import clustered_mean

RUN = ROOT / 'output/experiments/gpt-6-luna/v3_full_2931_20261015'
OUT = ROOT / "results/v3/full_single_agent"
EXPECTED, FULL, SEED, BOOTSTRAP = 2929, 2931, 5106, 10000
STATUS = 'PARTIAL_COHORT_MEASUREMENTS_ONLY_NO_ACCURACY'
PENDING = ['05a8454a6a71435a557db2cc', '0eb69f6e47ab63c055e2df90']
TOKEN_FIELDS = ('prompt_tokens','completion_tokens','total_tokens','cached_prompt_tokens','uncached_prompt_tokens','repeated_prompt_tokens')
MEASURES = ('latency_seconds','estimated_cost_usd','reported_cost_usd',*TOKEN_FIELDS,'llm_calls','tool_calls_executed',
            'tool_calls_recorded','evidence_count','evidence_valid','valid_evidence_fraction','evidence_precision','evidence_recall')
PUBLIC = {'baseline_decision.txt','preflight.json','protocol.txt','provider_profile.json','pricing.json','split.json',
          'runtime/azure.py','runtime/azure_adapter.py','runtime/sitecustomize.py'}
INVENTORY = {}


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def semantic(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',',':'), allow_nan=False).encode()).hexdigest()


def bind(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    require(expected is None or digest == expected, 'Frozen source or input hash mismatch: ' + path.name)
    INVENTORY[str(path)] = {'sha256':digest,'bytes':path.stat().st_size}
    return digest


def read(path): return json.loads(Path(path).read_bytes())


def save(path, value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def copy_bound(source, relative, expected=None):
    bind(source,expected);target=OUT/relative;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    require(sha(target)==INVENTORY[str(Path(source).resolve())]['sha256'],'Archive copy mismatch')


def pure_subset(path, allowed, seeds):
    """Use the unchanged dependency closure, with no module imports/setup code."""
    tree=ast.parse(Path(path).read_text());definitions={}
    for node in tree.body:
        if isinstance(node,ast.FunctionDef):definitions[node.name]=node
        elif isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name):definitions[target.id]=node
    selected,pending=set(),list(seeds)
    while pending:
        name=pending.pop()
        if name in selected or name not in definitions:continue
        selected.add(name)
        pending.extend(n.id for n in ast.walk(definitions[name]) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id in definitions and n.id not in selected)
    require(selected==set(allowed),'Unexpected pure measurement/classifier dependencies')
    nodes=[n for n in tree.body if any(definitions[k] is n for k in selected)]
    require(not any(isinstance(n,ast.FunctionDef) and n.decorator_list for n in nodes),'Decorators forbidden')
    scope={'ast':ast,'re':re,'math':math,'hashlib':hashlib,'json':json,'statistics':statistics,
           'defaultdict':defaultdict,'restore_result_audit':restore_result_audit}
    module=ast.Module(body=nodes,type_ignores=[])
    exec(compile(module,str(path),'exec'),scope)
    return scope,semantic(ast.dump(module,include_attributes=False))


class RichChunks:
    """Concatenated gzip members, each containing one complete JSONL record."""
    def __init__(self, directory, prefix, limit=32*1024*1024):
        self.directory,self.prefix,self.limit=directory,prefix,limit
        directory.mkdir(parents=True,exist_ok=True)
        self.stream=None;self.parts=[];self.rows=0
    def add(self,row):
        raw=(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
        compressed=gzip.compress(raw,compresslevel=6,mtime=0)
        require(len(compressed)<=self.limit,'One rich record exceeds chunk size bound')
        if self.stream is None or self.stream.tell()+len(compressed)>self.limit:
            if self.stream:self.stream.close()
            path=self.directory/f'{self.prefix}-{len(self.parts)+1:04}.jsonl.gz'
            self.stream=path.open('wb');self.parts.append({'path':str(path.relative_to(OUT)),'records':0})
        self.stream.write(compressed);self.parts[-1]['records']+=1;self.rows+=1
    def close(self):
        if self.stream:self.stream.close()
        for entry in self.parts:
            path=OUT/entry['path'];entry.update(sha256=sha(path),bytes=path.stat().st_size)
        return self.parts


def canonical_archive(source):
    """Byte-exact original JSON split BEFORE gzip; decompress+concatenate restores it."""
    directory=OUT/'archive/canonical';directory.mkdir(parents=True,exist_ok=True)
    parts=[];digest=hashlib.sha256()
    with source.open('rb') as stream:
        while raw:=stream.read(32*1024*1024):
            digest.update(raw);path=directory/f'full_single_agent.json.part-{len(parts)+1:04}.gz'
            path.write_bytes(gzip.compress(raw,compresslevel=6,mtime=0))
            require(path.stat().st_size<100*1024*1024,'Git file limit exceeded')
            parts.append({'path':str(path.relative_to(OUT)),'uncompressed_bytes':len(raw),'bytes':path.stat().st_size,'sha256':sha(path)})
    require(digest.hexdigest()==INVENTORY[str(source.resolve())]['sha256'],'Canonical archive source changed')
    return {'format':'decompress each numbered part then concatenate bytes; original JSON array, not JSONL',
            'original_sha256':digest.hexdigest(),'parts':parts}


def flatten(obj,prefix=''):
    result={}
    for k,v in obj.items():
        name=f'{prefix}.{k}' if prefix else k
        if isinstance(v,dict):result.update(flatten(v,name))
        elif isinstance(v,(list,tuple)):result[name]=json.dumps(v,separators=(',',':'),allow_nan=False)
        else:result[name]=v
    return result


def csv_rows(path,rows):
    rows=list(rows);keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def jsonl(path,rows):
    with Path(path).open('w') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def measurement(row, families, split_index, pricing, numeric, cost_estimate):
    # Same field selection and null handling as frozen question_record + metric_values.
    tokens=row.get('tokens') if isinstance(row.get('tokens'),dict) else {}
    budget=row.get('budget') if isinstance(row.get('budget'),dict) else {}
    calls=tokens.get('per_call');unknown=any(c.get('usage_available') is False for c in calls or [] if isinstance(c,dict))
    prompt,cached=numeric(tokens.get('prompt')),numeric(tokens.get('cached_prompt'))
    qid,patient=row['question_id'],row['patient_fhir_id'];family=families.get(qid,{})
    splits,patient_splits=split_index
    exposure=('development_exposed_patient' if patient_splits.get(patient,set()) & {'dev','pilot'} else
              'held_out_patient_descriptive' if 'held_out' in patient_splits.get(patient,set()) else 'unassigned_patient')
    tools=row.get('tool_calls');evidence=row.get('evidence')
    out={'question_id':qid,'patient_fhir_id':patient,'family':family.get('family','unknown'),'template_id':family.get('template_id'),
         'dataset_split':family.get('split'),'analysis_split':splits.get(qid,'unassigned'),'exposure':exposure,
         'latency_seconds':numeric(row.get('latency')),'estimated_cost_usd':cost_estimate(tokens,pricing),'reported_cost_usd':numeric(tokens.get('cost')),
         'prompt_tokens':prompt,'completion_tokens':numeric(tokens.get('completion')),'total_tokens':numeric(tokens.get('total')),
         'cached_prompt_tokens':cached,'uncached_prompt_tokens':prompt-cached if prompt is not None and cached is not None and cached<=prompt else None,
         'repeated_prompt_tokens':numeric(tokens.get('repeated_prompt')),'provider_usage_unknown':unknown,
         'llm_calls':numeric(tokens.get('llm_calls')),'tool_calls_executed':numeric(budget.get('tool_calls_run')),
         'tool_calls_recorded':len(tools) if isinstance(tools,list) else None,'evidence_count':len(evidence) if isinstance(evidence,list) else None,
         'evidence_valid':None,'valid_evidence_fraction':None,'evidence_precision':None,'evidence_recall':None,
         'per_call_usage':calls,'tool_counts_by_name':dict(Counter(str(c.get('name','unknown')) for c in tools if isinstance(c,dict))) if isinstance(tools,list) else None,
         'phase_measurements':(row.get('revision') or {}).get('phases'),'context_compactions':(row.get('revision') or {}).get('context_compactions')}
    if unknown:
        for key in TOKEN_FIELDS:out[key]=None
    return out


def summarize(rows,measure,finite,intervals=True):
    result={'questions':len(rows),'patients':len({r['patient_fhir_id'] for r in rows}),
            'metrics':{key:measure([r[key] for r in rows]) for key in MEASURES}}
    if intervals:
        for key in ('latency_seconds','estimated_cost_usd','total_tokens'):
            values=[r[key] for r in rows]
            if values and all(finite(v) for v in values):
                result['metrics'][key]['mean_patient_bootstrap']=clustered_mean(values,[r['patient_fhir_id'] for r in rows],n_boot=BOOTSTRAP,seed=SEED)
    return result


def main():
    require(not (OUT/'summary.json').exists(),'Refuse to overwrite completed measurements package')
    started=datetime.now(timezone.utc).isoformat()
    manifest=read(RUN/'run_manifest.json');ledger=read(RUN/'acceptance.json');state=read(RUN/'supervisor_state.json')
    pricing=read(RUN/'pricing.json');provider=read(RUN/'provider_rejections.json')
    require(state['phase']=='attention' and state.get('active')=={} and ledger.get('complete') is False,'Stopped partial state required')
    selections=ledger['arms']['single_agent'];counts=ledger['counts']['single_agent']
    require(len(selections)==EXPECTED and counts=={'raw_attempts':2984,'runtime_failures':55,'intrinsic_errors':0},'Terminal accepted/runtime counts changed')
    require(provider['rejections']=={'single_agent':{q:3 for q in PENDING}},'Terminal provider counts changed')
    require(pricing['per_million_tokens']=={'input':.1,'cached_input':.01,'output':.5},'Frozen official price schedule changed')
    for name in ('run_manifest.json','acceptance.json','supervisor_state.json','pricing.json','provider_rejections.json','operations.jsonl','full_single_agent.json.run.json'):
        copy_bound(RUN/name,'provenance/'+name)
    for name,expected in manifest['source_sha256'].items():
        require(Path(name).suffix=='.py' or name=='requirements.txt','Unexpected source path')
        copy_bound(ROOT/name,'provenance/frozen_source/'+name,expected)
    require(set(manifest['pinned_files_sha256'])==PUBLIC,'Public pin allowlist changed')
    for name,expected in manifest['pinned_files_sha256'].items():copy_bound(RUN/name,'provenance/public/'+name,expected)
    for key in ('input','families','split_manifest'):bind(manifest[key]['path'],manifest[key]['sha256'])
    copy_bound(manifest['families']['path'],'provenance/families.csv',manifest['families']['sha256'])
    copy_bound(manifest['split_manifest']['path'],'provenance/split.json',manifest['split_manifest']['sha256'])
    patients={}
    with Path(manifest['input']['path']).open(newline='') as stream:
        for row in csv.DictReader(stream):
            require(row['question_id'] not in patients,'Duplicated source question ID');patients[row['question_id']]=row['patient_fhir_id']
    require(len(patients)==FULL and set(patients)-set(selections)==set(PENDING),'Pending cohort changed')
    with Path(manifest['families']['path']).open(newline='') as stream:families={r['question_id']:r for r in csv.DictReader(stream)}
    analysis,measurement_ast=pure_subset(ROOT/'scripts/v3_full_analysis.py',
        {'finite','numeric','percentiles','measure','cost_estimate','exposure_index'},['measure','cost_estimate','exposure_index'])
    classify_scope,classifier_ast=pure_subset(ROOT/'scripts/v3_full_run.py',
        {'CLASSIFIER_VERSION','RUNTIME','INTRINSIC','PROVIDER_JWT_REJECTION','diagnostic_errors','intrinsic_tool_validation','provider_jwt_paths',
         'is_observed_provider_prompt_rejection','SDK_ERROR_PREFIX','MAX_ERROR_CHARS','classify_attempt'},['classify_attempt','CLASSIFIER_VERSION'])
    require(classify_scope['CLASSIFIER_VERSION']=='v3-infrastructure-v4','Classifier changed')
    classifier=lambda row:classify_scope['classify_attempt'](row,None,allow_provider_rejection=True)
    split_index=analysis['exposure_index'](read(RUN/'split.json'))
    canonical=RUN/'full_single_agent.json';canonical_sha=bind(canonical)
    require(canonical_sha=='b4c1cb4b824a7395733854e7013ecacd28e85b0d450d4f863b0b464bdee43969','Frozen 2929 canonical changed')
    require(read(RUN/'full_single_agent.json.run.json')==manifest['run_identities']['single_agent'],'Canonical identity changed')
    rich=RichChunks(OUT/'archive/rich_accepted_results','accepted')
    records,index,audits=[],{},[]
    for i,row in enumerate(iter_result_rows(canonical),1):
        qid=row.get('question_id');require(qid in selections and qid not in index and row['patient_fhir_id']==patients[qid],'Canonical owner/cohort changed')
        require(not row.get('attempt_history') and selections[qid]['runtime_clean'] is True,'Non-first-clean canonical row')
        index[qid]=semantic(row)
        ref=row.get('audit_artifacts');require(isinstance(ref,dict) and ref.get('question_id')==qid and ARTIFACT_PATTERN.fullmatch(ref['path']),'Audit descriptor changed')
        audit=RUN/ref['path'];compressed_sha=bind(audit)
        restored=restore_result_audit(row,RUN,question_id=qid,restore_history=False)
        decision=classifier(restored);require(decision['status']=='accepted' and decision['reason']==selections[qid]['selection'],'Accepted row is not runtime-clean')
        rich.add(restored)
        audits.append({'question_id':qid,'original_audit_path':ref['path'],'compressed_sha256':compressed_sha,
                       'uncompressed_sha256':ref['sha256'],'compressed_bytes':audit.stat().st_size,
                       'canonical_row_semantic_sha256':index[qid],'restored_row_semantic_sha256':semantic(restored)})
        record=measurement(row,families,split_index,pricing,analysis['numeric'],analysis['cost_estimate'])
        record.update(selection=selections[qid]['selection'],attempt_file=selections[qid]['attempt_file'],round=selections[qid]['round'],runtime_failures_before=selections[qid]['runtime_failures_before'])
        tokens=row['tokens'];calls=tokens.get('per_call') or []
        require(tokens.get('llm_calls')==len(calls),'LLM call aggregate mismatch')
        for field,key in (('prompt','prompt'),('completion','completion'),('cached','cached_prompt')):
            require(sum(c.get(field,0) for c in calls)==tokens[key],'Per-call usage sum mismatch')
        require(tokens['prompt']+tokens['completion']==tokens['total'],'Token accounting mismatch')
        repeated=sum(min(cur['prompt'],prev['prompt']+prev['completion']) for prev,cur in zip(calls,calls[1:]))
        require(repeated==tokens['repeated_prompt'],'Repeated-input estimate mismatch')
        records.append(record)
        if i%100==0:print(json.dumps({'packaged_accepted_rows':i,'expected':EXPECTED}),flush=True)
    require(len(index)==EXPECTED and set(index)==set(selections),'Canonical cohort incomplete')
    rich_parts=rich.close()
    # Reconstruct first-clean choice order from sealed raw rows; accepted content is
    # compared by full semantic hash. Only failed audits need restoration here.
    retries=Counter();selected={};choices={};raw_counts={'raw_attempts':0,'runtime_failures':0,'intrinsic_errors':0}
    failed_records=[];failed_rich=RichChunks(OUT/'diagnostics/rich_runtime_attempts','runtime')
    batches=[]
    for batch_path in sorted((RUN/'attempts/single_agent').glob('round-*.batch.json')):
        batch=read(batch_path);number=batch['round'];sealpath=batch_path.with_suffix('.seal.json');seal=read(sealpath)
        require(batch['arm']=='single_agent' and bind(batch_path)==seal['batch_sha256'],'Raw batch seal mismatch')
        copy_bound(batch_path,f'provenance/sealed_batches/{batch_path.name}')
        copy_bound(sealpath,f'provenance/sealed_batches/{sealpath.name}')
        output,shard=Path(batch['output']),Path(batch['input'])
        require(output.resolve().is_relative_to(RUN) and shard.resolve().is_relative_to(RUN),'Raw path escapes run')
        bind(output,seal['result_sha256']);bind(Path(str(output)+'.run.json'),seal['identity_sha256']);bind(shard,batch['input_sha256'])
        copy_bound(Path(str(output)+'.run.json'),f'provenance/sealed_batches/{output.name}.run.json')
        # CSV shards contain the original input/reference fields and are archived
        # for later handoff without evaluating any question or reference value.
        copy_bound(shard,f'provenance/sealed_batches/{shard.name}')
        identity=read(Path(str(output)+'.run.json'));expected=manifest['run_identities']['single_agent']
        require(identity==batch['run_identity'] and all(v==expected[k] for k,v in identity.items() if k!='input_hash'),'Raw frozen identity differs')
        seen=set()
        for row in iter_result_rows(output):
            qid=row['question_id'];require(qid in batch['question_ids'] and qid not in seen and row['patient_fhir_id']==patients[qid],'Raw ownership mismatch');seen.add(qid)
            # All observed rows in this closed run are attempted or placeholders.
            attempted=(row.get('error') not in (None,'') or row.get('latency') is not None)
            if not attempted:continue
            require(qid not in selected,'Accepted question was rerun')
            raw_counts['raw_attempts']+=1
            digest=semantic(row)
            if qid in selections and selections[qid]['attempt_file']==str(output.relative_to(RUN)):
                require(index[qid]==digest,'Canonical selected raw row differs')
                selected[qid]=digest
                choices[qid]={'attempt_file':str(output.relative_to(RUN)),'round':number,'runtime_failures_before':retries[qid],
                              'selection':'runtime_clean_terminal','runtime_clean':True}
            else:
                ref=row.get('audit_artifacts');require(ref is not None and ref.get('question_id')==qid,'Runtime audit missing')
                bind(output.parent/ref['path'])
                restored=restore_result_audit(row,output.parent,question_id=qid,restore_history=False)
                decision=classifier(restored);require(decision['status']=='runtime','Unselected attempted row is not runtime')
                retries[qid]+=1;raw_counts['runtime_failures']+=1
                failed_rich.add(restored)
                item=measurement(row,families,split_index,pricing,analysis['numeric'],analysis['cost_estimate'])
                item.update(round=number,attempt_file=str(output.relative_to(RUN)),classification=decision,
                            excluded_from_accepted_primary_metrics=True)
                failed_records.append(item)
        require(seen==set(batch['question_ids']),'Raw scheduled rows missing')
        batches.append({'round':number,'question_count':len(seen),'batch_sha256':sha(batch_path),'seal_sha256':sha(sealpath),
                        'result_sha256':sha(output),'identity_sha256':sha(Path(str(output)+'.run.json')),'input_sha256':sha(shard)})
    failed_parts=failed_rich.close()
    require([b['round'] for b in batches]==list(range(1,33)),'32 sealed rounds required')
    require(selected==index and choices==selections and raw_counts==counts,'First-clean replay ledger/count mismatch')
    categories=Counter(r['classification'].get('category','other_infrastructure') for r in failed_records)
    require(categories=={'provider_policy_rejection':6,'other_infrastructure':49},'Failed-attempt category counts changed')
    print(json.dumps({'accepted':len(records),'excluded_runtime':len(failed_records),'first_clean_replay_verified':True}),flush=True)
    archive=canonical_archive(canonical)
    jsonl(OUT/'per_question_measurements.jsonl',records)
    scalar_keys=[k for k in records[0] if k not in {'per_call_usage','tool_counts_by_name','phase_measurements','context_compactions'}]
    csv_rows(OUT/'per_question_measurements.csv',({k:r[k] for k in scalar_keys} for r in records))
    jsonl(OUT/'diagnostics/runtime_attempt_measurements.jsonl',failed_records)
    csv_rows(OUT/'diagnostics/runtime_attempt_measurements.csv',(flatten(r) for r in failed_records))
    jsonl(OUT/'provenance/accepted_audit_inventory.jsonl',audits)
    save(OUT/'provenance/single_agent_ledger_slice.json',{'selections':selections,'counts':counts})
    save(OUT/'provenance/first_clean_validation.json',{'status':'PASS','accepted':EXPECTED,'scheduled':FULL,'pending':PENDING,
        'counts':raw_counts,'choices_equal_frozen_ledger':True,'canonical_selected_raw_semantic_equal':True,
        'accepted_restored_classifier_clean':True,'sealed_batches':batches,'classifier_AST_sha256':classifier_ast,
        'canonical_row_semantic_sha256':index})
    overall=summarize(records,analysis['measure'],analysis['finite'])
    groups={}
    for field in ('family','template_id','patient_fhir_id','analysis_split','dataset_split','exposure'):
        bucket=defaultdict(list)
        for row in records:bucket[str(row.get(field) or 'unknown')].append(row)
        table=[{'group':name,**summarize(rows,analysis['measure'],analysis['finite'],intervals=field not in ('template_id','patient_fhir_id'))} for name,rows in sorted(bucket.items())]
        groups[field]=table;save(OUT/f'groups_by_{field}.json',table);csv_rows(OUT/f'groups_by_{field}.csv',(flatten(r) for r in table))
    phases=defaultdict(list);question_phase_count=0
    for row in records:
        if row.get('phase_measurements'):question_phase_count+=1
        for phase in row.get('phase_measurements') or []:
            if isinstance(phase,dict):phases[str(phase.get('phase','unknown'))].append(phase)
    phase_rows=[{'phase':name,'questions_with_phase':len(rows),'tokens':analysis['measure']([r.get('tokens') for r in rows]),
                 'tool_calls':analysis['measure']([r.get('tool_calls') for r in rows]),'fatal':sum(bool(r.get('fatal')) for r in rows),
                 'has_diagnostic_errors':sum(bool(r.get('errors')) for r in rows)} for name,rows in sorted(phases.items())]
    save(OUT/'phase_usage.json',{'questions_with_phase_metadata':question_phase_count,'rows':phase_rows,
          'definition':'Unchanged frozen analyzer phase aggregation; absent phase metadata is unavailable, not zero work.'})
    csv_rows(OUT/'phase_usage.csv',(flatten(r) for r in phase_rows))
    tool_counts=Counter()
    calls_by_index=defaultdict(list)
    for row in records:
        tool_counts.update(row.get('tool_counts_by_name') or {})
        for i,call in enumerate(row.get('per_call_usage') or [],1):calls_by_index[i].append(call)
    save(OUT/'tool_name_counts.json',dict(tool_counts))
    call_rows=[{'call_index':i,'questions_with_call':len(calls),'prompt_tokens':analysis['measure']([c.get('prompt') for c in calls]),
                'completion_tokens':analysis['measure']([c.get('completion') for c in calls]),'cached_prompt_tokens':analysis['measure']([c.get('cached') for c in calls]),
                'usage_available_false':sum(c.get('usage_available') is False for c in calls)} for i,calls in sorted(calls_by_index.items())]
    save(OUT/'call_ordinal_usage.json',call_rows);csv_rows(OUT/'call_ordinal_usage.csv',(flatten(r) for r in call_rows))
    # This is a diagnostics-only summary of recorded values; no failed cost/time
    # or token is mixed into the accepted cohort or imputed as correctness.
    failed_summary=summarize(failed_records,analysis['measure'],analysis['finite'],intervals=False)
    failed_summary.update(status='EXCLUDED_RUNTIME_DIAGNOSTICS_ONLY',categories=dict(categories),
        note='Recorded observed values only. Provider-usage flags may be unknown; absent costs are not zero, and actual billing is unknown.')
    save(OUT/'diagnostics/runtime_summary.json',failed_summary)
    summary={'schema_version':1,'status':STATUS,'benchmark_complete':False,'selected_scope_complete':False,'accuracy_evaluated':False,
        'generated_at':datetime.now(timezone.utc).isoformat(),'accepted':EXPECTED,'scheduled':FULL,'pending':2,'completion_coverage':EXPECTED/FULL,
        'pending_question_ids':PENDING,'patients':len({r['patient_fhir_id'] for r in records}),'runtime_attempts_excluded':55,
        'runtime_categories':dict(categories),'accepted_metrics':overall,'groups':{k:len(v) for k,v in groups.items()},
        'pricing':pricing,'actual_billing_usd':None,'phase_metadata_questions':question_phase_count,
        'cache_provenance':{'full_baseline_cached_field_audit_completed':False,'completed_checkpoint_accepted_rows':891,
            'availability_flag_note':'Per-call usage_available indicates aggregate usage availability; cached-field presence/provenance is not guaranteed by this flag. Cached zeros may reflect fallback defaults. Full baseline cache provenance audit was not performed.'},
        'statistics':{'bootstrap':'unchanged scripts.experiment_stats.clustered_mean; question-weighted mean, resample patient clusters',
            'n_boot':BOOTSTRAP,'seed':SEED,'ci_level':.95,'CI_metrics':['latency_seconds','estimated_cost_usd','total_tokens'],
            'interval_scope':'Observed accepted2929 cohort only; no inference for pending2 or claim about the complete2931 outcome',
            'descriptives':'Unchanged frozen measure: linear-interpolated p95, sample SD; complete accepted-cohort totals/means null if any measurement missing; observed-only values separate'},
        'operational_timing':{k:state.get(k) for k in ('started_at','scope_started_at','finished_at')},
        'scope_notes':['Partial cohort; no accuracy, reference correctness, evidence validation, paired comparison or full completion claim.',
            'All55 runtime attempts excluded from accepted primary metrics:6 provider rejections and49 other infrastructure outcomes.',
            'Full detailed accepted content is in lossless restored rich JSONL gzip chunks; clinical fields were not evaluated or displayed.',
            'Actual billing remains unknown. Costs use frozen official OpenAI rates as the user-authorized Azure pricing proxy.',
            'Development exposure splits are descriptive; the full released cohort informed development and is not a pristine held-out evaluation.'],
        'canonical_archive':archive,'rich_accepted_archive':{'format':'numbered JSONL gzip chunks; concatenated gzip members supported by standard gzip readers',
            'records':EXPECTED,'parts':rich_parts,'self_contained_result_content':True},
        'runtime_diagnostic_archive':{'records':len(failed_records),'parts':failed_parts},
        'provenance':{'run_dir_historical_absolute':str(RUN),'canonical_sha256':canonical_sha,
            'manifest_sha256':sha(RUN/'run_manifest.json'),'ledger_sha256':sha(RUN/'acceptance.json'),
            'inference_fingerprint':manifest['run_identities']['single_agent']['code_hash'],'measurement_AST_sha256':measurement_ast,
            'classifier_AST_sha256':classifier_ast,'external_audit_files_duplicated':False,
            'note':'Original externalized canonical descriptors name historical audit paths. Full restored content is portable in rich JSONL; audit descriptor hashes and original compressed/uncompressed hashes are retained. Original raw batch JSON files are hash-bound rather than duplicated; runtime raw content is separately archived.'}}
    save(OUT/'summary.json',summary)
    csv_rows(OUT/'metric_summary.csv',({'metric':k,**flatten(v)} for k,v in overall['metrics'].items()))
    # Independent arithmetic balances; no answer or reference field is read here.
    checks={}
    for key in MEASURES:
        values=[r[key] for r in records];valid=[v for v in values if analysis['finite'](v)];detail=overall['metrics'][key]
        checks[key]={'rows':len(values),'observed_n':len(valid),'missing_n':len(values)-len(valid),
                     'sum_reconciles':detail['observed_only']['total']==(sum(valid) if valid else None),
                     'mean_reconciles':detail['observed_only']['mean']==(statistics.fmean(valid) if valid else None)}
        require(checks[key]['sum_reconciles'] and checks[key]['mean_reconciles'],'Measurement arithmetic mismatch')
    for path,expected in INVENTORY.items():require(sha(path)==expected['sha256'],'Frozen source/run changed during packaging')
    require('utils.answer_scorer' not in sys.modules and 'scripts.v3_full_analysis' not in sys.modules,'Scorer/analyzer unexpectedly imported')
    validation={'status':'PASS','accepted_rows':EXPECTED,'rich_rows':sum(p['records'] for p in rich_parts),'numeric_rows':len(records),
        'raw_attempts_replayed':raw_counts['raw_attempts'],'runtime_excluded':raw_counts['runtime_failures'],'first_clean_ledger_equal':True,
        'all_source_hashes_stable':True,'measurement_arithmetic':checks,'scorer_or_analyzer_imported':False,
        'accuracy_or_reference_correctness_evaluated':False,'provider_calls':False,'clinical_text_printed':False,
        'script_sha256':sha(Path(__file__)),'started_at':started,'finished_at':datetime.now(timezone.utc).isoformat()}
    save(OUT/'validation.json',validation)
    save(OUT/'provenance/source_bindings.json',INVENTORY)
    def fmt(v):return 'unknown' if v is None else f'{v:.12g}'
    lines=[STATUS,'','Single Agent full benchmark attempt: 2929/2931 accepted (99.9317639% coverage),94 patients;2 pending.',
           'This package reports accepted-cohort measurements only. No accuracy or answer/reference correctness evaluation was performed.',
           'The two pending questions each reached3 provider-policy rejections. No fourth request was made.',
           'All55 failed runtime attempts (6 provider-policy,49 other infrastructure) are preserved separately and excluded from primary measurements.','',
           'Accepted-task measurements (sum of per-task latency is not elapsed wall time):']
    for key,detail in overall['metrics'].items():
        observed=detail['observed_only'];lines.append(f"{key}: n={detail['n']} observed={detail['observed_n']} missing={detail['missing_n']}; total={fmt(detail['total'])}; mean={fmt(detail['mean'])}; median={fmt(detail['median'])}; p95={fmt(detail['p95'])}; sampleSD={fmt(detail['stdev'])}; min={fmt(detail['min'])}; max={fmt(detail['max'])}")
        if 'mean_patient_bootstrap' in detail:
            ci=detail['mean_patient_bootstrap'];lines.append(f"  Question-weighted mean95% patient-cluster bootstrap CI: [{fmt(ci['ci_low'])}, {fmt(ci['ci_high'])}] (10000 resamples, seed5106)")
    lines+=['','Missingness and interpretation:',
        'Evidence validity/precision/recall are unavailable because no reference/evidence scoring was run. Evidence counts are descriptive only.',
        'Reported dollar cost is null when the runtime did not supply it; actual billed cost is unknown.',
        'Estimated USD=((prompt-cached)*0.10+cached*0.01+completion*0.50)/1000000. Cached tokens are a subset of prompt tokens.',
        'If any per-call usage_available is false, frozen semantics withhold total/cached/etc token measures and cost estimates for that question.',
        'usage_available=true does not prove cached-token field presence or exact request association. The only completed baseline cached-field provenance checkpoint covers891 accepted questions. No full2929 baseline cached-field audit was performed.',
        'Repeated_prompt_tokens is the existing overlap estimate sum(min(current_prompt, previous_prompt+previous_completion)); it is not observed cached billing.',
        'Subgroup summaries by family,template,patient,split and development exposure describe the available accepted cohort only.',
        'Bootstrap intervals quantify variation within the accepted cohort. They do not impute either pending question or establish independent held-out generalization.',
        f'Explicit phase metadata appears on{question_phase_count} accepted questions. Missing phase metadata means unavailable rather than zero phase work.',
        'Per-call-ordinal tables are usage-by-call-index summaries, not inferred role or phase assignments.',
        'Operational first-start/scope-start/finish timestamps are in summary.json. They can include user deferral, transport recovery and explicit pauses, so they are distinct from the accepted-task latency sum.','',
        'Handoff and reproduction:',
        'summary.json and metric_summary.csv contain the overall measurement distribution; per_question_measurements.jsonl/.csv preserve numeric records and availability flags.',
        'groups_by_*.json/.csv contain per-family/template/patient/split/exposure tables. phase_usage and call_ordinal_usage retain available phase/call metadata; tool_name_counts counts recorded tool names.',
        'archive/rich_accepted_results/*.jsonl.gz contain2929 losslessly restored result objects, including answers, evidence, traces, retrieved resources, usage and revision metadata. These clinical fields were archived without evaluating or printing them.',
        'archive/canonical/*.gz restore the exact original externalized JSON by decompressing each numbered part and concatenating bytes. The expected SHA is in summary.json.',
        'Rich restored results are selfcontained for content analysis. Original canonical audit descriptors and raw batch absolute paths remain historical; source_bindings and accepted_audit_inventory record their hashes. Individual original audit.gz files and entire raw batch result JSON are not duplicated.',
        'diagnostics/ preserves all55 restored failed attempts and their measurement tables; these diagnostics never enter accepted totals.',
        'provenance/ contains frozen source45 files, all9 public pins, identity, complete ledger, the selected ledger slice, source bindings, input-family/split tables, all32 batch descriptors/seals/shards/identities and exact first-clean choice validation.',
        'The original full3-arm and paired2931-each completion gates remain unchanged. This2929-row package must never be represented as a completed paired or full benchmark.',
        'To refresh, first preserve this snapshot; rerun package_measurements.py only into a new destination after intentionally revising its exact stopped-cohort bindings. It refuses overwriting a completed package.']
    (OUT/'full_single_analysis.txt').write_text('\n'.join(lines)+'\n')
    files={str(p.relative_to(OUT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in {'ARTIFACT_SHA256.json','ARTIFACT_SHA256SUMS'}}
    save(OUT/'ARTIFACT_SHA256.json',files)
    (OUT/'ARTIFACT_SHA256SUMS').write_text(''.join(f"{d['sha256']}  {name}\n" for name,d in files.items()))
    require(max(d['bytes'] for d in files.values())<100*1024*1024,'A packaged file exceeds Git size limit')
    print(json.dumps({'status':STATUS,'accepted':EXPECTED,'pending':2,'runtime_excluded':55,'package_files':len(files),
          'package_bytes':sum(d['bytes'] for d in files.values()),'primary_latency_seconds':overall['metrics']['latency_seconds']['total'],
          'estimated_cost_usd':overall['metrics']['estimated_cost_usd']['total'],'total_tokens':overall['metrics']['total_tokens']['total'],
          'accuracy_evaluated':False,'output':str(OUT)}),flush=True)

if __name__=='__main__':main()
