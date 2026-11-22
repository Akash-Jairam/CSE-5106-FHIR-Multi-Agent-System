"""Human-authorized posthoc matched completed cohort, unchanged frozen scoring/stats."""
import importlib.util
from pathlib import Path
import sys
spec=importlib.util.spec_from_file_location('base',Path(__file__).with_name('base.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
from collections import Counter,defaultdict
import csv,gzip,hashlib,json,math,shutil,statistics
from datetime import datetime,timezone
ROOT,RUN,OUT=b.ROOT,b.RUN,b.OUT
INTERNAL=OUT/'internal';INTERNAL.mkdir(parents=True,exist_ok=True)
N=2929

def main():
    started=datetime.now(timezone.utc).isoformat()
    require=b.require;save=b.save;bind=b.bind;read=b.read
    manifest,ledger,state,pricing=map(read,[RUN/'run_manifest.json',RUN/'acceptance.json',RUN/'supervisor_state.json',RUN/'pricing.json'])
    choices=ledger['arms']['single_agent'];counts=ledger['counts']['single_agent']
    require(len(choices)==N and counts=={'raw_attempts':2984,'runtime_failures':55,'intrinsic_errors':0},'Frozen stopped cohort changed')
    require(state['phase']=='attention' and state['active']=={},'Run is not stopped')
    require(pricing['per_million_tokens']=={'input':.1,'cached_input':.01,'output':.5},'Frozen official price schedule changed')
    source_sha=bind(RUN/'full_single_agent.json','b4c1cb4b824a7395733854e7013ecacd28e85b0d450d4f863b0b464bdee43969')
    for name in ('run_manifest.json','acceptance.json','supervisor_state.json','pricing.json','provider_rejections.json','operations.jsonl','full_single_agent.json.run.json'):
        b.copy_bound(RUN/name,'internal/provenance/'+name)
    for name,expected in manifest['source_sha256'].items():
        require(Path(name).suffix=='.py' or name=='requirements.txt','Unexpected source pin');bind(ROOT/name,expected)
    for name,expected in manifest['pinned_files_sha256'].items():
        require(name in b.PUBLIC,'Unexpected public pin');bind(RUN/name,expected)
    for key in ('input','families','split_manifest'):bind(manifest[key]['path'],manifest[key]['sha256'])
    b.copy_bound(manifest['families']['path'],'internal/provenance/families.csv',manifest['families']['sha256'])
    b.copy_bound(manifest['split_manifest']['path'],'internal/provenance/split.json',manifest['split_manifest']['sha256'])
    require(read(RUN/'full_single_agent.json.run.json')==manifest['run_identities']['single_agent'],'Identity mismatch')
    with Path(manifest['input']['path']).open(newline='') as stream:reference={r['question_id']:r for r in csv.DictReader(stream)}
    with Path(manifest['families']['path']).open(newline='') as stream:families={r['question_id']:r for r in csv.DictReader(stream)}
    require(len(reference)==2931 and set(reference)-set(choices)==set(b.PENDING),'Shared completed cohort changed')
    sys.path.insert(0,str(ROOT))
    import scripts.v3_full_analysis as a
    split_index=a.exposure_index(read(RUN/'split.json'))
    base_records=[];canonical_hashes={};compact_index=[]
    # Numeric/score analysis uses unchanged frozen question_record and scorer.
    for i,row in enumerate(b.iter_result_rows(RUN/'full_single_agent.json'),1):
        qid=row['question_id'];require(qid in choices and qid not in canonical_hashes,'Duplicate/foreign canonical row')
        require(row['patient_fhir_id']==reference[qid]['patient_fhir_id'] and not row.get('attempt_history') and choices[qid]['runtime_clean'] is True,'Canonical ownership/selection mismatch')
        canonical_hashes[qid]=b.semantic(row)
        rec=a.question_record('single_agent',row,reference[qid],families[qid],pricing,split_index,RUN)
        rec.update(runtime_clean_selection_verified=True,**choices[qid])
        base_records.append(rec)
        if i%500==0:print(json.dumps({'scored_accepted_rows':i,'N':N}),flush=True)
    require(len(base_records)==N,'Canonical count changed')
    # Match against the immutable V3 export, preserving all frozen flags/metrics.
    historical=ROOT/'output/experiments/gpt-6-luna/v3_complete_2931_20261016'
    receipt=read(historical/'V3_COMPLETE_CONTROLS_PENDING.json');bind(historical/'V3_COMPLETE_CONTROLS_PENDING.json')
    old_manifest=read(historical/'validated_v3_input_snapshot/run_manifest.json')
    require(bind(historical/'validated_v3_input_snapshot/run_manifest.json')==receipt['source_manifest_sha256'],'Historical V3 manifest changed')
    require(old_manifest['input']==manifest['input'] and old_manifest['run_identities']['multi_agent']==manifest['run_identities']['multi_agent'],'V3 matched source/identity differs')
    for name in ('utils/answer_scorer.py','scripts/v3_full_analysis.py'):
        require(old_manifest['source_sha256'][name]==manifest['source_sha256'][name],'Frozen scoring/statistics source mismatch')
    import subprocess
    stats_frozen=subprocess.check_output(['git','show','780f9bc:scripts/experiment_stats.py'],cwd=ROOT)
    require(stats_frozen==(ROOT/'scripts/experiment_stats.py').read_bytes(),'Patient-bootstrap implementation differs from original v3 freeze')
    bind(ROOT/'scripts/experiment_stats.py')
    v3path=historical/'v3_complete/per_question.jsonl';bind(v3path)
    saved_manifest=read(historical/'delivery_artifact_hashes.json')
    # file_manifest is relative-name->metadata; bind its output first, then require the historical file hash.
    bind(historical/'delivery_artifact_hashes.json')
    entry=saved_manifest['v3_complete/per_question.jsonl']
    require(b.sha(v3path)==(entry['sha256'] if isinstance(entry,dict) else entry),'Frozen V3 question export changed')
    v3all=[json.loads(line) for line in v3path.open() if line.strip()]
    require(len(v3all)==2931 and len({r['question_id'] for r in v3all})==2931,'V3 full cohort changed')
    v3_by_id={r['question_id']:r for r in v3all}
    v3=[v3_by_id[r['question_id']] for r in base_records]
    require(all(x['patient_fhir_id']==y['patient_fhir_id']==reference[x['question_id']]['patient_fhir_id'] for x,y in zip(base_records,v3)),'Matched patient identity differs')
    require(all(r.get('runtime_clean_selection_verified') for r in v3),'V3 selection unverified')
    overall_single=a.summarize(base_records,n_boot=10000)
    overall_v3=a.summarize(v3,n_boot=10000)
    pair=[p for p in a.paired({'direct_llm':[],'single_agent':base_records,'multi_agent':v3},10000) if {p['left'],p['right']}=={'single_agent','multi_agent'}]
    pair[0]['difference_convention']='left minus right = single_agent minus multi_agent_v3; accuracy is a fraction (multiply by100 for percentage points)'
    require(len(pair)==1 and pair[0]['matched_questions']==N and pair[0]['left_only']==pair[0]['right_only']==0,'Matched pair incomplete')
    # Freeze original source answers inside canonical archive; public numeric tables
    # contain flags and measurements, while detailed full clinical records are archived.
    keep=['arm','question_id','patient_fhir_id','family','template_id','dataset_split','analysis_split','exposure','answer_type','correct','typed_answer_matches','typed_reason',
          'intrinsic_error','abstained','draft_present','draft_correct','primary_present','primary_correct','revised','answer_changed','has_limitation','budget_stopped',
          *a.MEASURES,'provider_usage_unknown','per_call_usage','tool_counts_by_name','revision_protocol','phases','context_compactions','runtime_clean_selection_verified',
          'attempt_file','round','runtime_failures_before','selection','runtime_clean']
    numeric=[{k:r.get(k) for k in keep} for r in base_records]
    b.jsonl(OUT/'per_question_measurements_and_scores.jsonl',numeric);a.write_csv(OUT/'per_question_measurements_and_scores.csv',numeric)
    b.jsonl(OUT/'matched_v3_measurements_and_scores.jsonl',({k:r.get(k) for k in keep} for r in v3))
    a.write_csv(OUT/'matched_v3_measurements_and_scores.csv',({k:r.get(k) for k in keep} for r in v3))
    save(OUT/'single_agent_summary.json',overall_single);save(OUT/'matched_v3_summary.json',overall_v3);save(OUT/'paired_comparison.json',pair[0])
    a.write_csv(OUT/'metric_summary.csv',({'metric':k,**a.flatten(v)} for k,v in overall_single['metrics'].items()))
    a.write_csv(OUT/'matched_arm_summaries.csv',({'arm':arm,**a.flatten(d)} for arm,d in [('single_agent',overall_single),('multi_agent_v3',overall_v3)]))
    a.write_csv(OUT/'paired_comparison.csv',(a.flatten(p) for p in pair))
    groups={};pairgroups={}
    for field in ('family','template_id','patient_fhir_id','answer_type','analysis_split','dataset_split','exposure'):
        buckets=defaultdict(list)
        for r in base_records:buckets[str(r.get(field) or 'unknown')].append(r)
        table=[];pt=[]
        for name,rows in sorted(buckets.items()):
            other=[v3_by_id[r['question_id']] for r in rows]
            for arm,data in [('single_agent',rows),('multi_agent_v3',other)]:
                table.append({'arm':arm,'group':name,**a.summarize(data,n_boot=10000,intervals=field not in ('template_id','patient_fhir_id'))})
            detail=[p for p in a.paired({'direct_llm':[],'single_agent':rows,'multi_agent':other},10000) if {p['left'],p['right']}=={'single_agent','multi_agent'}][0]
            detail['difference_convention']='single_agent minus multi_agent_v3 (left minus right)'
            pt.append({'group':name,**detail})
        groups[field]=table;pairgroups[field]=pt
        save(OUT/f'by_{field}.json',table);a.write_csv(OUT/f'by_{field}.csv',(a.flatten(r) for r in table))
        save(OUT/f'paired_by_{field}.json',pt);a.write_csv(OUT/f'paired_by_{field}.csv',(a.flatten(r) for r in pt))
    phases=defaultdict(list);tool=Counter();callindex=defaultdict(list)
    for r in base_records:
        tool.update(r.get('tool_counts_by_name') or {})
        for i,c in enumerate(r.get('per_call_usage') or [],1):callindex[i].append(c)
        for p in r.get('phases') or []:
            if isinstance(p,dict):phases[str(p.get('phase','unknown'))].append(p)
    phase=[{'phase':n,'questions_with_phase':len(rs),'tokens':a.measure([r.get('tokens') for r in rs]),'tool_calls':a.measure([r.get('tool_calls') for r in rs]),
            'fatal':sum(bool(r.get('fatal')) for r in rs),'has_diagnostic_errors':sum(bool(r.get('errors')) for r in rs)} for n,rs in sorted(phases.items())]
    save(OUT/'phase_usage.json',{'metadata_available_questions':sum(bool(r.get('phases')) for r in base_records),'rows':phase,
        'note':'Unchanged frozen phase semantics; absent metadata means unavailable rather than zero work.'});a.write_csv(OUT/'phase_usage.csv',(a.flatten(r) for r in phase))
    save(OUT/'tool_name_counts.json',dict(tool))
    callrows=[{'call_index':i,'questions_with_call':len(cs),'prompt_tokens':a.measure([c.get('prompt') for c in cs]),'completion_tokens':a.measure([c.get('completion') for c in cs]),
               'cached_prompt_tokens':a.measure([c.get('cached') for c in cs]),'usage_available_false':sum(c.get('usage_available') is False for c in cs)} for i,cs in sorted(callindex.items())]
    save(OUT/'call_ordinal_usage.json',callrows);a.write_csv(OUT/'call_ordinal_usage.csv',(a.flatten(r) for r in callrows))
    mechanism={'single_agent':a.mechanism_summary(base_records),'multi_agent_v3_matched':a.mechanism_summary(v3)}
    save(OUT/'mechanisms.json',mechanism)
    draft_scope={}
    for arm,rs in [('single_agent',base_records),('multi_agent_v3_matched',v3)]:
        draft_scope[arm]={'all_final_correct':sum(r['correct'] for r in rs),'draft_present_questions':sum(r['draft_present'] for r in rs),
            'draft_present_final_correct':sum(r['correct'] and r['draft_present'] for r in rs),'draft_absent_final_correct':sum(r['correct'] and not r['draft_present'] for r in rs),
            'draft_present_draft_correct':sum(bool(r['draft_correct']) and r['draft_present'] for r in rs),
            'draft_present_wrong_to_right':sum(r['draft_present'] and not r['draft_correct'] and r['correct'] for r in rs),
            'draft_present_right_to_wrong':sum(r['draft_present'] and r['draft_correct'] and not r['correct'] for r in rs),
            'draft_absent_questions':sum(not r['draft_present'] for r in rs)}
    save(OUT/'draft_final_scope_reconciliation.json',{'note':'Draft transition counts concern only records with a draft. Final correctness includes records without a draft. Mechanism counters can also use typed stage correctness whereas final correctness includes intrinsic error handling; preserve frozen definitions.', 'arms':draft_scope})
    # Reconcile selected rows and all excluded raw attempts without re-running anything.
    pure,astsha=b.pure_subset(ROOT/'scripts/v3_full_run.py',{'CLASSIFIER_VERSION','RUNTIME','INTRINSIC','PROVIDER_JWT_REJECTION','diagnostic_errors','intrinsic_tool_validation','provider_jwt_paths',
        'is_observed_provider_prompt_rejection','SDK_ERROR_PREFIX','MAX_ERROR_CHARS','classify_attempt'},['classify_attempt','CLASSIFIER_VERSION'])
    retries=Counter();selected={};replayed={};raw_counts={'raw_attempts':0,'runtime_failures':0,'intrinsic_errors':0};failed=[];batches=[]
    failedrich=b.RichChunks(INTERNAL/'runtime_details','runtime')
    for batchpath in sorted((RUN/'attempts/single_agent').glob('round-*.batch.json')):
        batch,seal=read(batchpath),read(batchpath.with_suffix('.seal.json'));output=Path(batch['output']);number=batch['round']
        require(bind(batchpath)==seal['batch_sha256'],'Batch seal mismatch');bind(batchpath.with_suffix('.seal.json'))
        bind(output,seal['result_sha256']);bind(Path(str(output)+'.run.json'),seal['identity_sha256']);bind(batch['input'],batch['input_sha256'])
        for path in (batchpath,batchpath.with_suffix('.seal.json'),Path(batch['input']),Path(str(output)+'.run.json')):
            b.copy_bound(path,'internal/provenance/sealed_batches/'+path.name)
        identity=read(Path(str(output)+'.run.json'));expected=manifest['run_identities']['single_agent']
        require(identity==batch['run_identity'] and set(identity)==set(expected) and all(v==expected[k] for k,v in identity.items() if k!='input_hash'),'Raw identity mismatch')
        seen=set()
        for row in b.iter_result_rows(output):
            qid=row['question_id'];require(qid in batch['question_ids'] and qid not in seen and row['patient_fhir_id']==reference[qid]['patient_fhir_id'],'Raw owner mismatch');seen.add(qid)
            if row.get('error') in (None,'') and row.get('latency') is None:continue
            require(qid not in selected,'Accepted question rerun');raw_counts['raw_attempts']+=1
            if qid in choices and choices[qid]['attempt_file']==str(output.relative_to(RUN)):
                require(canonical_hashes[qid]==b.semantic({k:v for k,v in row.items() if k!='attempt_history'}),'Canonical selected raw row changed');selected[qid]=canonical_hashes[qid]
                replayed[qid]={'attempt_file':str(output.relative_to(RUN)),'round':number,'runtime_failures_before':retries[qid],'selection':'runtime_clean_terminal','runtime_clean':True}
            else:
                full=b.restore_result_audit(row,output.parent,question_id=qid,restore_history=False)
                decision=pure['classify_attempt'](full,None,allow_provider_rejection=True);require(decision['status']=='runtime','Unselected nonruntime result')
                ref=row['audit_artifacts'];bind(output.parent/ref['path']);failedrich.add(full);retries[qid]+=1;raw_counts['runtime_failures']+=1
                m=b.measurement(row,families,split_index,pricing,a.numeric,a.cost_estimate);m.update(round=number,attempt_file=str(output.relative_to(RUN)),classification=decision,excluded_from_primary=True);failed.append(m)
        require(seen==set(batch['question_ids']),'Scheduled rows missing')
        batches.append({'round':number,'batch_sha256':b.sha(batchpath),'seal_sha256':b.sha(batchpath.with_suffix('.seal.json')),'result_sha256':b.sha(output),
                        'identity_sha256':b.sha(Path(str(output)+'.run.json')),'input_sha256':b.sha(batch['input'])})
    failureparts=failedrich.close()
    require(selected==canonical_hashes and replayed==choices and raw_counts==counts,'First-clean ledger replay differs')
    require([p['round'] for p in batches]==list(range(1,33)),'Sealed32round cohort differs')
    categories=Counter(r['classification'].get('category','other_infrastructure') for r in failed)
    require(categories=={'provider_policy_rejection':6,'other_infrastructure':49},'Runtime exclusion counts differ')
    save(INTERNAL/'first_clean_validation.json',{'status':'PASS','accepted':N,'counts':counts,'selection_equal':True,'canonical_rows_equal_selected_raw':True,
        'sealed_rounds':batches,'row_semantic_sha256':canonical_hashes,'classifier_AST_sha256':astsha})
    b.jsonl(INTERNAL/'runtime_measurements.jsonl',failed);a.write_csv(INTERNAL/'runtime_measurements.csv',(b.flatten(r) for r in failed))
    save(INTERNAL/'runtime_summary.json',{'status':'EXCLUDED_DIAGNOSTICS_ONLY','counts':dict(categories),'measurements':b.summarize(failed,a.measure,a.finite,False),'rich_archive':failureparts,
        'note':'No failed measurements enter primary summaries. Unknown usage and billed costs remain unknown.'})
    terminal=ROOT/'benchmarks/single-agent-full-run-20261017/reviewed_provider_retry/terminal_metadata.json'
    proof=read(terminal);b.copy_bound(terminal,'internal/provenance/terminal_metadata.json')
    require(proof['all_prior_single_agent_selections_identical'] is True and proof['selected_accepted']==N,'Historical first-clean proof changed')
    for name,expected in proof['terminal_core_sha256'].items():require(b.sha(RUN/name)==expected,'Terminal proof/core hash mismatch')
    shared_ids=sorted(choices);save(OUT/'matched_cohort.json',{'status':'MATCHED_COMPLETED_COHORT','N':N,'patients':94,'question_ids':shared_ids,
        'cohort_selection':'Posthoc availability intersection of completed first-runtime-clean Single Agent results and complete V3 results.',
        'pending_questions_not_imputed_wrong':True,'full2931_completion_claimed':False,'clinical_reference_source_sha256':manifest['input']['sha256']})
    report={'status':'MATCHED_COMPLETED_COHORT','N':N,'patients':len({r['patient_fhir_id'] for r in base_records}),'full_benchmark_complete':False,
        'arms':{'single_agent':overall_single,'multi_agent_v3':overall_v3},'paired':pair[0],
        'pricing':pricing,'actual_billing_usd':None,'statistics':{'method':'Unchanged frozen typed scorer and analyzer summary/paired functions; patient-cluster bootstrap10000 seed5106;95%percentileCI; difference=single_agent minus V3.',
            'descriptives':'Frozen17metrics, totals/mean/median/linear-p95/sampleSD/min/max/missingness; incomplete measurements withhold complete-cohort aggregate and report observed_only separately.'},
        'interpretation':['This analysis uses the matched completed cohort, N=2929, selected posthoc by shared availability. Both arms use exactly the same question and patient IDs.',
            'The complete V3 N=2931 result is a separate overall evaluation; these matched V3 numbers use the shared2929 only.',
            'The two unavailable Single Agent questions are excluded from both sides and are never marked incorrect.',
            'The full released dataset informed development; split/exposure tables are descriptive, not an untouched held-out evaluation.',
            'Estimated dollar costs use frozen official OpenAI rates as the user-approved Azure proxy. Actual billing is unknown.',
            'The completed cached-field provenance checkpoint covers891 Single Agent questions. Aggregate usage availability flags do not prove cached-field presence. Full baseline cache provenance audit was not performed.',
            'Evidence-count is descriptive; field validation/precision/recall remain unavailable unless already present in frozen V3 results. No new evidence validation was performed.'],
        'scorer':{'sha256':b.sha(ROOT/'utils/answer_scorer.py'),'analysis_sha256':b.sha(ROOT/'scripts/v3_full_analysis.py'),'statistics_sha256':b.sha(ROOT/'scripts/experiment_stats.py')},
        'source_canonical_sha256':source_sha,'source_manifest_sha256':b.sha(RUN/'run_manifest.json'),'frozen_inference_fingerprint':manifest['run_identities']['single_agent']['code_hash'],
        'archive_note':'Data archive manifest is added separately. Operational diagnostics and incomplete-arm execution metadata are retained under internal/.'}
    save(OUT/'summary.json',report)
    cross=Counter((r['correct'],r['abstained']) for r in base_records)
    save(OUT/'correctness_empty_flag_crosstab.json',[{'correct':k[0],'syntactic_empty_no_value_flag':k[1],'questions':v} for k,v in sorted(cross.items())])
    lines=['MATCHED COMPLETED COHORT N=2929 (94 patients)','',
        'The comparison uses the same2929 completed questions in both arms. Availability-based selection is posthoc and descriptive. The overall V3 N2931 evaluation is reported separately.','']
    for arm,d in report['arms'].items():
        ci=d['accuracy'];lines.append(f"{arm}: {d['correct']}/{N} typed-correct; accuracy{ci['estimate']:.10%};95%patient-clusterCI[{ci['ci_low']:.10%},{ci['ci_high']:.10%}]")
        for key,m in d['metrics'].items():
            lines.append(f"  {key}: observed{m['observed_n']}/{m['n']},missing{m['missing_n']}; total={m['total']};mean={m['mean']};median={m['median']};p95={m['p95']};sampleSD={m['stdev']};min={m['min']};max={m['max']}")
            if 'mean_patient_bootstrap' in m:lines.append('    Mean patient-clusterCI: '+json.dumps(m['mean_patient_bootstrap']))
    lines+=['','Paired differences (Single Agent minus V3):',json.dumps(pair[0],indent=2),'',*report['interpretation'],'',
        'Tables: per_question_measurements_and_scores.*, matched_v3_measurements_and_scores.*, matched_arm_summaries.csv, metric_summary.csv, paired_comparison.*, by_*.{json,csv},paired_by_*.{json,csv}.',
        'phase_usage.json records available phase metadata; absent phase labels mean unavailable rather than zero work. call_ordinal_usage describes call index, not inferred phases. tool_name_counts and mechanisms retain descriptive behavior.',
        'The syntactic empty/no-value flag can include correct empty answers. See correctness_empty_flag_crosstab.json; do not equate every flagged answer with failure.',
        'Complete available clinical output is archived losslessly without displaying it. Archive manifest documents exact canonical row and audit payload preservation. Original gzip headers/bytes are not reconstructed.',
        'Internal provenance retains frozen source identities and hashes, first-clean replay, terminal metadata and excluded diagnostics. Original full2931 completion gates remain unchanged.']
    (OUT/'full_single_analysis.txt').write_text('\n'.join(lines)+'\n')
    checks={}
    for key in a.MEASURES:
        vals=a.metric_values(base_records,key);valid=[v for v in vals if a.finite(v)];m=overall_single['metrics'][key]
        require(m['observed_n']==len(valid) and m['missing_n']==N-len(valid),'Metric missingness mismatch')
        require(m['observed_only']['total']==(sum(valid) if valid else None),'Metric total mismatch')
        require(m['observed_only']['mean']==(statistics.fmean(valid) if valid else None),'Metric mean mismatch')
        checks[key]={'observed_n':len(valid),'missing_n':N-len(valid),'sum_and_mean_reconcile':True}
    for path,expected in b.INVENTORY.items():require(b.sha(path)==expected['sha256'],'Frozen artifact changed during analysis')
    save(INTERNAL/'source_bindings.json',b.INVENTORY)
    save(OUT/'validation.json',{'status':'PASS','N':N,'pair_N':pair[0]['matched_questions'],'shared_qid_patient_exact':True,'first_clean_choices_equal':True,
        'source_hashes_stable':True,'arithmetic_checks':checks,'correctness_crosstab_total':sum(cross.values()),'paired_table_sum':sum(pair[0][k] for k in ('left_wins','right_wins','both_correct','both_wrong')),
        'frozen_typed_scorer_used':True,'scorer_source_unchanged':True,'provider_calls':False,'clinical_text_printed':False,'new_full_completion_claim':False,
        'started_at':started,'finished_at':datetime.now(timezone.utc).isoformat(),'package_script_sha256':b.sha(Path(__file__))})
    print(json.dumps({'status':'PASS','N':N,'single_correct':overall_single['correct'],'V3_matched_correct':overall_v3['correct'],
        'single_latency_total':overall_single['metrics']['latency_seconds']['total'],'single_estimated_cost_total':overall_single['metrics']['estimated_cost_usd']['total'],
        'single_tokens_total':overall_single['metrics']['total_tokens']['total'],'pair_wins':pair[0]['left_wins'],'pair_losses':pair[0]['right_wins'],'output':str(OUT)}),flush=True)
if __name__=='__main__':main()
