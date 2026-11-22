"""Consolidate fixed, completed development observations; no inference or retries."""
import csv, hashlib, importlib.util, json, math, pathlib, statistics, subprocess, sys, zipfile
from datetime import datetime, timezone
from collections import Counter
import numpy as np

ROOT = pathlib.Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
B = ROOT / 'benchmarks/design-v3d-20261014'
RAW = ROOT / 'output/experiments/gpt-6-luna/query_contract_v3d_300q'
OUT = ROOT / 'results/v3/development'
PIN = '906327dee1823178e3ac6319061f7d9ee5d2e01f'
FULL = '4c7fbd13c1fd6e0b38abd002b072af8654837f28'
DEV = 'f00804f186768a3672ae6a6075f366a27dc0ca9a'
sys.dont_write_bytecode = True
sources = {}

def sha_bytes(data): return hashlib.sha256(data).hexdigest()
def sha(path): return sha_bytes(path.read_bytes())
def read(path):
    sources[str(path.relative_to(ROOT))] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return json.loads(path.read_text())
def write(name,value): (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def jsonl(name,rows): (OUT/name).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
def finite(x): return type(x) in (int,float) and math.isfinite(x)
def number(x): return float(x) if finite(x) else None
def gitbytes(ref,name): return subprocess.run(['git','show',ref+':'+name],capture_output=True,check=True).stdout
def load_source(path,name):
    specification=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(specification);specification.loader.exec_module(module);return module

OUT.mkdir(parents=True,exist_ok=True)
manifest=read(B/'manifest.json'); frozen=read(B/'results.json'); audit=read(B/'completion_protocol_audit.json')
mechanisms=read(B/'diagnostics/mechanisms.json'); mechanism_rows=read(B/'diagnostics/mechanisms_rows.json')
runtime_equality=read(B/'runtime_equality.json')
full_manifest=read(ROOT/'output/experiments/gpt-6-luna/v3_full_2931_20261015/run_manifest.json')
pricing=read(ROOT/'output/experiments/gpt-6-luna/v3_full_2931_20261015/pricing.json')
pricing={**pricing,'development_use_note':'Same frozen official OpenAI proxy rates for comparison. Original development observed_cost remains unavailable; no Azure bill is inferred.'}
rates=pricing['per_million_tokens']

# Preserve every copied original Git blob, plus its SHA256 and byte count.
tree=subprocess.run(['git','ls-tree','-r',PIN,'--','benchmarks/design-v3d-20261014'],capture_output=True,text=True,check=True).stdout
copied={}
for line in tree.splitlines():
    fields,name=line.split('\t');mode,kind,blob=fields.split();path=ROOT/name
    actual=subprocess.run(['git','hash-object',str(path)],capture_output=True,text=True,check=True).stdout.strip()
    assert actual==blob and path.read_bytes()==gitbytes(PIN,name)
    copied[name]={'git_blob':blob,'mode':mode,'sha256':sha(path),'bytes':path.stat().st_size}
    sources[name]={'sha256':sha(path),'bytes':path.stat().st_size}
assert len(copied)==33
write('original_record_inventory.json',{'source_commit':PIN,'files':copied,'all_exact_original_git_blobs':True})
with zipfile.ZipFile(B/'source.zip') as archive:
    entries={name for name in archive.namelist() if not name.endswith('/')}
    assert entries==set(manifest['source_sha256'])
    assert all(pathlib.PurePosixPath(name).name!='config.yml' and '..' not in pathlib.PurePosixPath(name).parts for name in entries)
    assert all(sha_bytes(archive.read(name))==expected for name,expected in manifest['source_sha256'].items())
assert sha(B/'source.zip')==manifest['source_archive_sha256']

# Compare exact allowlisted runtime sources without importing inference modules.
comparison=[]
for name,expected in sorted(runtime_equality['source_sha256'].items()):
    assert name.startswith(('agent/','tools/','utils/','scripts/')) or name in ('run_agent.py','fhir_client.py','requirements.txt')
    a,b=gitbytes(FULL,name),gitbytes(DEV,name)
    assert sha_bytes(b)==expected==manifest['source_sha256'][name]
    comparison.append({'path':name,'full_sha256':sha_bytes(a),'development_sha256':sha_bytes(b),'equal':a==b})
assert len(comparison)==57 and [r['path'] for r in comparison if not r['equal']]==['agent/query_contract_agent.py']
fingerprint_names=sorted(name for name in full_manifest['source_sha256'] if name.startswith(('agent/','tools/','utils/')) or name in ('run_agent.py','fhir_client.py','requirements.txt','scripts/audit_store.py','scripts/full_benchmark.py'))
fingerprints={}
for ref in (FULL,DEV):
    digest=hashlib.sha256()
    for name in fingerprint_names:
        digest.update(name.encode());digest.update(b'\0');digest.update(gitbytes(ref,name));digest.update(b'\0')
    digest.update(json.dumps(full_manifest['runtime_dependency_versions'],sort_keys=True).encode())
    fingerprints[ref]=digest.hexdigest()
assert fingerprints[FULL]==full_manifest['run_identities']['multi_agent']['code_hash']
assert fingerprints[DEV]==frozen['identities']['v3d_team']['identity']['code_hash']
source_comparison={'full_source_commit':FULL,'development_source_commit':DEV,'comparison_files':comparison,
    'files_compared':57,'equal':56,'different':1,'original_57_equality_claim_against':runtime_equality['against'],
    'original_57_claim_scope_note':'The archived equality record compares development with its preceding development source, not the full-run freeze.',
    'verified_development_archive_sources':len(manifest['source_sha256']),
    'semantic_difference':'LedgerSolver max_iterations changed from 21 to 22, reserving two host-only final-source-check boundary slots instead of one. The actual solver model-call cap remains 20. No implementation files were merged into the results branch.',
    'fingerprint_source_files':fingerprint_names,'fingerprint_source_count':len(fingerprint_names),
    'same_dependency_version_map':full_manifest['runtime_dependency_versions'],'reproduced_fingerprints':fingerprints,
    'fingerprint_conclusion':'The one source-byte change exactly explains both recorded fingerprints when the same frozen 21 dependency versions are used. The two run implementations are therefore closely related but not byte-identical.'}
write('source_comparison.json',source_comparison)

# Only the unchanged frozen typed scorer is loaded, explicitly authorized offline.
scorer_path=ROOT/'utils/answer_scorer.py'
assert sha(scorer_path)==manifest['source_sha256']['utils/answer_scorer.py']
sources[str(scorer_path.relative_to(ROOT))]={'sha256':sha(scorer_path),'bytes':scorer_path.stat().st_size}
scorer=load_source(scorer_path,'unchanged_development_typed_scorer')
source_path=ROOT/'final_dataset/questions_answers_sql_fhir.csv'; family_path=ROOT/'manifests/question_families.csv'
assert sha(source_path)==manifest['dataset_sha256'] and sha(family_path)==frozen['families_sha256']
for p in (source_path,family_path,ROOT/'manifests/split.json'):
    sources[str(p.relative_to(ROOT))]={'sha256':sha(p),'bytes':p.stat().st_size}
with source_path.open(newline='') as stream: references={r['question_id']:r for r in csv.DictReader(stream)}
with family_path.open(newline='') as stream: families={r['question_id']:r for r in csv.DictReader(stream)}
team_mech={r['question_id']:r for r in mechanism_rows}
raw_arms={}; rows=[]; phase_rows=[]
for label,original in [('single_agent','ordinary_single_agent'),('multi_agent_v3','ordinary_multi_agent')]:
    raw_path=RAW/(original+'.json'); data=read(raw_path); identity=read(pathlib.Path(str(raw_path)+'.run.json'))
    key='v3d_single' if label=='single_agent' else 'v3d_team'
    assert sha(raw_path)==frozen['identities'][key]['result_sha256'] and identity==frozen['identities'][key]['identity']
    assert len(data)==300 and len({r['question_id'] for r in data})==300
    assert all(not r.get('attempt_history') for r in data)
    raw_arms[label]=data
assert {r['question_id'] for r in raw_arms['single_agent']}==set(team_mech)=={r['question_id'] for r in raw_arms['multi_agent_v3']}
patients=sorted({r['patient_fhir_id'] for r in raw_arms['single_agent']});cluster={p:i+1 for i,p in enumerate(patients)}
assert len(patients)==58
for arm,data in raw_arms.items():
    for raw in data:
        qid=raw['question_id'];assert raw['patient_fhir_id']==references[qid]['patient_fhir_id']
        tokens=raw.get('tokens') or {};calls=tokens.get('per_call') or [];revision=raw.get('revision') or {};budget=raw.get('budget') or {}
        available=[c for c in calls if c.get('usage_available') is True]
        assert calls and all(isinstance(c,dict) for c in calls)
        complete=all(c.get('usage_available') is True for c in calls)
        prompt=sum(c['prompt'] for c in available);completion=sum(c['completion'] for c in available);cached=sum(c['cached'] for c in available)
        assert all(finite(v) and v>=0 for v in (prompt,completion,cached)) and cached<=prompt
        assert prompt==tokens['prompt'] and completion==tokens['completion'] and prompt+completion==tokens['total'] and cached==tokens['cached_prompt']
        known=((prompt-cached)*rates['input']+cached*rates['cached_input']+completion*rates['output'])/1e6
        terminal=bool(raw.get('error'))
        if arm=='single_agent':correct=bool(not terminal and scorer.score_answer(raw.get('agent_answer'),references[qid]['true_answer'],references[qid]['question'],families[qid]['template'])['typed_correct'])
        else:correct=bool(team_mech[qid]['correct'])
        record={'arm':arm,'question_id':qid,'patient_cluster':cluster[raw['patient_fhir_id']],'family':families[qid]['family'],
            'correct':correct,'latency_seconds':number(raw.get('latency')),'prompt_tokens_measured':prompt,'completion_tokens_measured':completion,
            'total_tokens_measured':prompt+completion,'cached_prompt_tokens_measured':cached,'llm_calls':len(calls),
            'tool_calls_run':number(budget.get('tool_calls_run')),'provider_usage_complete':complete,
            'proxy_cost_usd':known if complete else None,'known_usage_proxy_lower_bound_usd':known,
            'observed_azure_billing_usd':None,'budget_stopped':bool(budget.get('stopped')),
            'revised':bool(revision.get('revised')),'answer_changed':bool(revision.get('answer_changed')),
            'first_attempt_only':True,'terminal_error':terminal}
        if arm=='multi_agent_v3':
            for k,v in team_mech[qid].items():
                if type(v) is bool or (v is None and k.endswith('_correct')):record['mechanism_'+k]=v
            for phase,values in team_mech[qid]['phase_records'].items():
                phase_rows.append({'question_id':qid,'patient_cluster':record['patient_cluster'],'family':record['family'],'phase':phase,
                    **{k:v for k,v in values.items() if type(v) in (int,float,bool) or v is None},
                    'recorded_error_count':len(values.get('errors') or [])})
        rows.append(record)
assert sum(r['correct'] for r in rows if r['arm']=='single_agent')==238
assert sum(r['correct'] for r in rows if r['arm']=='multi_agent_v3')==232
for arm,key in [('single_agent','v3d_single'),('multi_agent_v3','v3d_team')]:
    for family,stats in frozen['arms'][key]['accuracy'].items():
        selected=[r for r in rows if r['arm']==arm and (family=='all' or r['family']==family)]
        assert len(selected)==stats['n'] and sum(r['correct'] for r in selected)==stats['correct']
jsonl('per_question.jsonl',rows);jsonl('phase_measurements.jsonl',phase_rows)

def describe(values):
    known=[float(x) for x in values if finite(x)]; a=np.asarray(known,dtype=float);n=len(a)
    result={'n_expected':len(values),'n_observed':n,'n_missing':len(values)-n,'sum_observed':float(a.sum()) if n else None,
            'total_complete':float(a.sum()) if n==len(values) and n else None}
    result.update({k:None for k in ('mean','sample_sd','median','min','p25','p75','p90','p95','p99','max')})
    if n:result.update(mean=float(a.mean()),sample_sd=float(a.std(ddof=1)) if n>1 else None,min=float(a.min()),max=float(a.max()),**dict(zip(('p25','median','p75','p90','p95','p99'),map(float,np.percentile(a,[25,50,75,90,95,99])))))
    return result
def clustered(values,clusters,seed=5106):
    keep=[i for i,v in enumerate(values) if finite(v)];v=np.array([values[i] for i in keep],dtype=float);p=np.array([clusters[i] for i in keep]);unique,index=np.unique(p,return_inverse=True)
    if not len(v):return {'n':0,'patients':0,'mean':None,'ci_low':None,'ci_high':None}
    sums=np.bincount(index,weights=v);counts=np.bincount(index);weights=np.random.default_rng(seed).multinomial(len(unique),np.full(len(unique),1/len(unique)),size=10000)
    boot=weights@sums/(weights@counts)
    return {'n':len(v),'patients':len(unique),'mean':float(v.mean()),'ci_low':float(np.percentile(boot,2.5)),'ci_high':float(np.percentile(boot,97.5))}
metrics={'latency_seconds':'seconds','prompt_tokens_measured':'tokens','completion_tokens_measured':'tokens','total_tokens_measured':'tokens',
    'cached_prompt_tokens_measured':'tokens','llm_calls':'calls','tool_calls_run':'calls','proxy_cost_usd':'USD','known_usage_proxy_lower_bound_usd':'USD'}
summary=[];accuracy=[]
for arm,key in [('single_agent','v3d_single'),('multi_agent_v3','v3d_team')]:
    for family in ['all','lookup_counting','numerical_aggregation','temporal_selection']:
        selected=[r for r in rows if r['arm']==arm and (family=='all' or r['family']==family)]
        for metric,unit in metrics.items():summary.append({'arm':arm,'family':family,'metric':metric,'unit':unit,**describe([r[metric] for r in selected])})
        accuracy.append({'arm':arm,'family':family,**frozen['arms'][key]['accuracy'][family],'interval_source':'Copied frozen patient-block bootstrap result; no accuracy interval refit'})
by_arm={arm:{r['question_id']:r for r in rows if r['arm']==arm} for arm in raw_arms}
paired=[]
for qid in sorted(team_mech):
    a,b=by_arm['multi_agent_v3'][qid],by_arm['single_agent'][qid]
    pair={'question_id':qid,'patient_cluster':a['patient_cluster'],'family':a['family'],'team_correct':a['correct'],'single_correct':b['correct'],'correct_difference':int(a['correct'])-int(b['correct'])}
    for metric in ('latency_seconds','total_tokens_measured','llm_calls','tool_calls_run','proxy_cost_usd'):
        pair['team_'+metric]=a[metric];pair['single_'+metric]=b[metric]
        pair['difference_'+metric]=a[metric]-b[metric] if finite(a[metric]) and finite(b[metric]) else None
    paired.append(pair)
assert sum(r['correct_difference']==1 for r in paired)==8 and sum(r['correct_difference']==-1 for r in paired)==14
jsonl('paired_measurements.jsonl',paired)
pair_stats=[]
for metric in ('latency_seconds','total_tokens_measured','llm_calls','tool_calls_run','proxy_cost_usd'):
    values=[r['difference_'+metric] for r in paired]
    pair_stats.append({'metric':metric,'unit':metrics[metric],'direction':'team_minus_single',**describe(values),
                       'patient_block_mean_ci95':clustered(values,[r['patient_cluster'] for r in paired])})
paired_accuracy=next(x for x in frozen['comparisons'] if x['comparison']=='v3d_team - v3d_single')
paired_accuracy={**paired_accuracy,'both_correct':sum(r['team_correct'] and r['single_correct'] for r in paired),
    'both_incorrect':sum(not r['team_correct'] and not r['single_correct'] for r in paired),'direction':'team_minus_single',
    'interval_source':'Copied frozen 10000-resample patient-block percentile interval; question-weighted'}
mechanism_counts=[]
for key,value in mechanisms['arms']['v3d'].items():
    if type(value) in (int,float):mechanism_counts.append({'metric':key,'count_or_value':value,'cohort_questions':300,'proportion_of_questions':value/300 if type(value) is int and 0<=value<=300 else None})
phase_summary=[{'phase':name,**{k:v for k,v in values.items() if k!='errors'}} for name,values in mechanisms['arms']['v3d']['phases'].items()]
stats={'policy':'Historical development first attempts, all 300 scheduled rows per arm. Original terminal outcomes and typed scores are preserved.',
    'cohort':{'questions_per_arm':300,'patients':58,'families':dict(Counter(r['family'] for r in rows if r['arm']=='single_agent')),
              'question_ids_sha256':frozen['question_ids_sha256'],'development_reused':True},
    'accuracy':accuracy,'measurement_statistics':summary,'paired_accuracy':paired_accuracy,'paired_measurement_statistics':pair_stats,
    'team_mechanisms':mechanisms['arms']['v3d'],'team_phase_summary':phase_summary,
    'single_transitions_frozen':frozen['arms']['v3d_single']['transitions'],'team_transitions_frozen':frozen['arms']['v3d_team']['transitions'],
    'pricing_proxy':pricing,'observed_azure_billing_usd':None,
    'measurement_bootstrap':{'resamples':10000,'seed':5106,'unit':'patient','estimand':'question-weighted mean of paired differences','interval':'2.5th and97.5th percentiles; no calibrated hypothesis-test p-value'},
    'missingness':{arm:{'proxy_cost_usd':sum(r['proxy_cost_usd'] is None for r in rows if r['arm']==arm),'observed_azure_billing_usd':300} for arm in raw_arms},
    'limits':['Fixed300 has been reused for architecture development; intervals are descriptive and do not make it a fresh confirmatory cohort.',
              'Historical first-attempt development metrics and full-run first-runtime-clean metrics follow different policies; do not pool them.',
              'Proxy costs apply the pinned official OpenAI rates to measured provider usage; they are not Azure bills. Missing whole-question costs stay null; sums over observed costs are labeled incomplete.',
              'Measured token totals and known-usage proxy lower bounds do not establish total usage/cost where provider usage is unavailable.',
              'Patient-cluster IDs are local numeric grouping labels, not patient FHIR identifiers. No questions, clinical resources, predictions, or reference answers appear in new numeric exports.',
              'Mechanism transitions are post-hoc associations; no causal claim about audit or source feedback follows from them.',
              'The development and full implementations differ by one host-boundary slot reservation, despite the shared V3 design.']}
write('statistics.json',stats)
tables={'per_question':rows,'paired_measurements':paired,'metric_statistics':summary,'accuracy_by_family':accuracy,
        'phase_measurements':phase_rows,'mechanism_counts':mechanism_counts,'phase_summary':phase_summary}
pathlib.Path('/private/tmp/v3-development-consolidation/tables.json').write_text(json.dumps(tables,allow_nan=False))
write('manifest.json',{'schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_branch':'experiments/query-contract-v3',
    'source_record_commit':PIN,'source_code_commit':DEV,'original_record_directory':'benchmarks/design-v3d-20261014',
    'new_results_directory':'results/v3/development','complete':True,'questions_per_arm':300,'arms':['single_agent','multi_agent_v3'],
    'scoring':'Original team per-question scores reused; fresh Single Agent scored offline with the unchanged frozen typed scorer, reconciled exactly to all frozen family totals and238/300 overall.',
    'source_file_hashes':sources,'original_source_archive_files_verified':len(manifest['source_sha256']),
    'new_analysis_source_sha256':sha(pathlib.Path(__file__)),'source_bindings_stable':all(sha(ROOT/k)==v['sha256'] for k,v in sources.items()),
    'numeric_export_policy':'No question text, clinical data, predictions, or gold values in new exports. Historical original records remain byte-identical.',
    'original_record_inventory':'original_record_inventory.json','source_comparison':'source_comparison.json','statistics':'statistics.json',
    'proxy_cost_basis':'Pinned official OpenAI prices, adopted as an Azure proxy, not verified Azure bills. Original observed_cost was null.'})
assert all(sha(ROOT/k)==v['sha256'] for k,v in sources.items())
main=[f'Development V3 paired comparison: {300} questions per arm across58 patient clusters.',
    'Fresh Single Agent:238/300 correct (79.333%). V3 team:232/300 (77.333%).',
    'Team minus Single Agent:-2.000 percentage points. Frozen patient-block95% interval:'+f"[{paired_accuracy['ci_low']*100:.3f},{paired_accuracy['ci_high']*100:.3f}] percentage points.",
    f"Paired outcomes:8 team-only correct,14 single-only correct,{paired_accuracy['both_correct']} both correct,{paired_accuracy['both_incorrect']} both incorrect.",
    'The interval crosses zero. This exploratory development sample does not establish a team accuracy advantage over its fresh contemporaneous control.',
    '', 'The original fixed300 development scoring policy uses each scheduled first attempt. The historical typed outcomes were preserved, rather than retroactively applying the later full-run collection policy.',
    'The sample includes92 lookup/counting,49 numerical aggregation and159 temporal-selection questions; it was repeatedly used during development.',
    '', 'Per-question latency, measured token usage, tool/model calls and cost-proxy distributions are in statistics.json and metric_statistics.csv. Latency is seconds, token measures are token counts, proxy costs are USD.',
    'Proxy rates per million tokens:input$0.10,cached input$0.01,output$0.50. These are the frozen official OpenAI prices adopted for Azure comparison, not verified Azure charges. Missing whole-question costs remain null, not zero.',
    '', 'Source/revision mechanisms (team):283 source-valid primary releases;19 source-feedback revisions;4 wrong-to-right draft/primary transitions and0 losses at that transition. The static audit inspected283 valid primaries, proposed no material local repair, and triggered0 targeted revisions. Five initially correct drafts were not credited at final release; this is not evidence that the audit caused those losses. Six null credits lack a source-valid release.',
    'Contract, solver and audit phases recorded432057,13817221 and1258421 tokens respectively; solver executed1545 tools. Full phase records and mechanism counts retain their original definitions.',
    '', 'Implementation provenance:development codef00804f used LedgerSolver host-loop cap22; the full-run freeze4c7fbd13 used21. The model-call cap remained20. Across57 declared shared/runtime sources,56 are byte-identical and query_contract_agent.py is the sole difference. The same21 dependency versions and42 fingerprint files reproduce208a88f... versus65d2bbfc... exactly. Keep the runs separate; do not describe them as byte-identical implementations.',
    '', 'Accuracy confidence intervals and mechanism scores are copied from frozen records. New paired measurement intervals use10000 patient-block resamples with seed5106 and retain question weighting; they are descriptive intervals, not p-values. All new numeric exports omit clinical/question/prediction/reference content.',
    'See manifest.json for source hashes and original_record_inventory.json for the33 exact Git blobs restored from906327d. No inference, provider calls, implementation merge, or original output rewrite occurred.']
(OUT/'development_analysis.txt').write_text('\n'.join(main)+'\n')
print(json.dumps({'output':str(OUT),'per_question_rows':len(rows),'paired_rows':len(paired),'phase_rows':len(phase_rows),
    'correct':{arm:sum(r['correct'] for r in rows if r['arm']==arm) for arm in raw_arms},'paired_accuracy':paired_accuracy,
    'measurement_stats':[{k:v for k,v in x.items() if k in ('arm','metric','n_observed','n_missing','mean','total_complete','sum_observed','median','p95')} for x in summary if x['family']=='all' and x['metric'] in ('latency_seconds','proxy_cost_usd','total_tokens_measured')]}))
