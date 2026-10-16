from pathlib import Path
import json,hashlib,importlib.util,collections
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sources=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p,role):
 p=Path(p);p=p if p.is_absolute() else ROOT/p;data=p.read_bytes();sources.append({'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'role':role});return json.loads(data)
def dump(n,v):(OUT/n).write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n')
script=ROOT/'scripts/v8_mechanism_study.py';sources.append({'path':str(script),'sha256':sha(script),'bytes':script.stat().st_size,'role':'Exact reviewed metadata-only loader; no execute/analyze/scorer invoked'})
spec=importlib.util.spec_from_file_location('v8_selection_metadata_only',script);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
paths=[ROOT/'final_dataset/questions_answers_sql_fhir.csv',ROOT/'manifests/question_families.csv',ROOT/'manifests/split.json',ROOT/'manifests/final_evaluation_strata_20261015.json']
for p,role in zip(paths,['Parsed ONLY question_id/patient_fhir_id through explicit usecols','Parsed ONLY question_id/family/excluded_reason/template_id through exact loader','Question/patient split metadata','Frozen confirmation/development question IDs only; no QA']):sources.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'role':role})
metadata,families,split,exposure=m.read_selection_inputs(*paths)
byqid={x['question_id']:x['patient_fhir_id'] for x in metadata};family={x['question_id']:x for x in families};assert len(byqid)==len(metadata)==2931
op={q:(f['excluded_reason'] if f['family']=='out_of_scope' else f['family']) for q,f in family.items()}
reserved=set(exposure['splits']['development_patients']['question_ids'])|set(exposure['splits']['pilot_patients']['question_ids']);fixed=set(exposure['splits']['adaptive_questions']['question_ids']);protected=set(exposure['splits']['question_confirmation']['question_ids'])
assert len(reserved)==1234 and len(fixed)==300 and len(protected)==1397 and len(reserved|fixed|protected)==2931 and not(reserved&fixed|reserved&protected|fixed&protected)
history=read('docs/evaluation_exposure_history_20261015.json','Prior committed historical metadata scan attestation; no listed answer files reopened')
histpath=Path(history['attempted_id_union_artifact']['path']);hist=read(histpath,'Prior SHA-bound explicit attempted question/patient IDs, not patient-expanded exposure');assert sha(histpath)==history['attempted_id_union_artifact']['sha256']=='41e0246ff531e34a3fedf304a0cd5bbd51393faeb09e75f23b73e008de44eab9'
known=set(hist['attempted_question_ids']);covering=hist['covering_files'];assert len(known)==321 and set.union(*(set(x['attempted_question_ids']) for x in covering.values()))==known
patient_exposure=read('manifests/development_exposure.json','Subject/patient exposure ONLY; not an attempted-question list')
old=read('manifests/v8_mechanism_120_20261015.json','Original V8 full scheduled120, conservatively excluded including uncompleted43')
new=read('manifests/v8_mechanism_120_protocol_repair_20261015.json','Completed repaired V8 scheduled120; parent attests all120 terminal, no answer files read')
oldids={x['question_id'] for x in old['cases']};newids={x['question_id'] for x in new['cases']};assert len(oldids)==len(newids)==120 and not oldids&newids
interface_ids=set();interface_sources={}
for suffix,review in [('a','live_interface_review.json'),('b','live_interface_review_b.json')]:
 p='output/development/v8_interface_20261015'+suffix+'/study_identity.json';d=read(p,'Explicitly named V8 development interface case metadata only')
 reviewd=read('output/development/v8_engineering_20261015/'+review,'Named live interface completion metadata; no outcome values read')
 assert sha(ROOT/p)==reviewd['identity_sha256']
 ids={x['question_id'] for x in d['runtime_case_metadata']};assert len(ids)==1
 interface_ids|=ids;interface_sources[p]=sorted(ids)
assert len(interface_ids)==2
legacy={}
for n in ['benchmarks/design-v5-20261015/live_development_interface.json','benchmarks/design-v6-20261015/live_development_interface_initial.json','benchmarks/design-v6-20261015/live_development_interface_clarified.json','benchmarks/design-v7-20261015/live_development_interface.json']:
 d=read(n,'Legacy live interface metadata; QIDs absent, identity not assumed');legacy[n]={'reported_questions_or_attempts':d['questions'],'question_IDs_present':False,'reuse_assertion':d.get('previous_reserved_development_question_reused',d.get('previously_queried_reserved_development_input')),'input_sha256':d.get('input_sha256'),'input_split_sha256':d.get('input_split_sha256')}
v6identity=read('output/development/v6_native_interface_20261015/ordinary_multi_agent.json.run.json','Explicitly named V6 run identity only; no answers/audits read')
legacy_v6_hash=v6identity['input_hash']
selection_review=read('benchmarks/design-v8-protocol-repair-20261015/engineering/root_selection_review.json','Committed cohort separation assertions and mention of later diagnostic; no QA')
known_union=known|oldids|newids|interface_ids
assert not known_union&protected and not(oldids|newids|interface_ids)&fixed
remaining=reserved-known_union
qid_groups={
 'historical_explicit_attempted321':known,
 'fixed_adaptive300':fixed,
 'original_V8_all_scheduled120':oldids,
 'repaired_V8_all_scheduled120':newids,
 'known_V8_engineering_interfaces2':interface_ids,
 'protected1397_exclusion_not_exposure':protected}
source_union={name:{'role':'exclusion_not_exposure' if name.startswith('protected') else 'known_attempted' if name.startswith('historical') else 'adaptive_exclusion' if name.startswith('fixed') else 'known_scheduled' if 'scheduled' in name else 'known_interface_attempts','question_ids':sorted(ids),'question_count':len(ids),'patient_count':len({byqid[q] for q in ids})} for name,ids in qid_groups.items()}
overlap={}
for name,ids in [('original_V8',oldids),('repaired_V8',newids)]:
 intersection=ids&known
 provenance={}
 for src,x in covering.items():
  hits=intersection&set(x['attempted_question_ids'])
  if hits:
   prior=next(v for v in history['files'] if v['path']==src)
   provenance[src]={'question_ids':sorted(hits),'identity_sha256_recorded_by_prior_audit':prior['identity_sha256'],'scheduled_metadata_rows':prior['scheduled_metadata_rows'],'attempted_metadata_rows':prior['attempted_metadata_rows'],'evidence':'Copied only from prior SHA-bound metadata extraction; source answer file not reopened'}
 overlap[name]={'historical_attempt_overlap':sorted(intersection),'historical_attempt_overlap_n':len(intersection),'known_V8_interface_overlap':sorted(ids&interface_ids),'fixed300_overlap_n':len(ids&fixed),'protected1397_overlap_n':len(ids&protected),'prior_extraction_provenance':provenance}
# Patient-level ledger maps through already exposed split metadata; it never expands into attempted QIDs.
subject_to_patient={str(p['subject_id']):p['patient_fhir_id'] for name in ['dev','pilot','held_out'] for p in split['splits'][name]['patients'] if 'subject_id' in p}
subjects={str(v) for x in patient_exposure['exposures'] for v in x['subject_ids']}
documented_patients={subject_to_patient[s] for s in subjects if s in subject_to_patient}
patient_conservative_remaining={q for q in remaining if byqid[q] not in documented_patients}
known_q_patients={byqid[q] for q in known_union};known_templates={family[q]['template_id'] for q in known_union};remaining_templates={family[q]['template_id'] for q in remaining}
patient_rows=[]
for patient in sorted({byqid[q] for q in reserved}):
 ids={q for q in reserved if byqid[q]==patient};left=ids-known_union
 patient_rows.append({'patient_fhir_id':patient,'reserved_questions':len(ids),'known_question_exclusions':len(ids&known_union),'remaining_known_ID_clean':len(left),'remaining_by_operation':dict(collections.Counter(op[q] for q in left)),'documented_patient_exposure':patient in documented_patients,'known_question_exposure_or_schedule':patient in known_q_patients})
ops={k:{'available_questions':sum(op[q]==k for q in remaining),'available_patients':len({byqid[q] for q in remaining if op[q]==k}),'quota':v,'surplus':sum(op[q]==k for q in remaining)-v} for k,v in m.QUOTAS.items()}
assert all(z['surplus']>=0 for z in ops.values());assert ops['lookup_counting']['available_patients']>=20
known_attempted_lower=known|newids|interface_ids
summary={'status':'METADATA_ONLY_EXPOSURE_AUDIT_NO_NEW_MANIFEST_OR_QA','dataset_questions':2931,'dataset_patients':len(set(byqid.values())),'reserved_dev_pilot_questions':len(reserved),'reserved_dev_pilot_patients':len({byqid[q] for q in reserved}),'historical_explicit_attempted_questions':len(known),'historical_explicit_attempted_patients':len({byqid[q] for q in known}),'known_attempted_ID_union_lower_bound':len(known_attempted_lower),'known_attempted_ID_union_note':'Prior explicit321 plus completed repaired120 plus2 verified interfaces. Original aborted120 exact attempted-prefix identities not independently re-audited; all120 excluded as scheduled.','known_scheduled_attempted_interface_union':len(known_union),'known_scheduled_attempted_interface_patients':len(known_q_patients),'reserved_known_question_exclusions':len(known_union&reserved),'remaining_known_ID_clean_upper_bound':len(remaining),'remaining_patients':len({byqid[q] for q in remaining}),'remaining_by_operation':ops,'balanced120_feasible_for_identified_exclusions':True,'feasibility_proof':'Every disjoint operation pool meets its quota. Lookup/counting alone has34 available patients and30slots, so20patient coverage can be achieved without selecting question IDs here.','proposed_next_seed':5110,'reviewed_V8_selection_seeds':[old['seed'],new['seed']],'seed_claim':'5110 is unused in the two reviewed formal study manifests; no assertion about every unrecorded local trial. No new IDs selected.','overlap':overlap,'legacy_interface_identity_gaps':legacy,'legacy_V6_named_run_input_hash':legacy_v6_hash,'V4_historical_interface_input_hash':next(x['input_hash'] for x in history['files'] if x['path']=='output/local_azure/v4_dev_interface_20261014/ordinary_multi_agent.json'),'legacy_interface_total_reported_slots_upper_bound':sum(x['reported_questions_or_attempts'] for x in legacy.values()),'later_informal_diagnostic_mentioned_without_QIDs':selection_review.get('cohort_and_repair_fixed_before_later_informal_diagnostic'),'remaining_templates':len(remaining_templates),'remaining_template_IDs_absent_from_known_question_exclusion_union':len(remaining_templates-known_templates),'documented_subject_exposure_groups':len(patient_exposure['exposures']),'documented_subjects':len(subjects),'documented_subjects_mapped_to_patients':len(documented_patients),'remaining_questions_on_documented_development_patients':sum(byqid[q] in documented_patients for q in remaining),'patient_expanded_conservative_pool':{'interpretation':'Optional stronger exclusion of all32 patients in the coarse documented QA/gold exposure ledger, NOT an assertion that every question on these patients was attempted','questions':len(patient_conservative_remaining),'patients':len({byqid[q] for q in patient_conservative_remaining}),'operation_counts':dict(collections.Counter(op[q] for q in patient_conservative_remaining)),'balanced120_feasible':False},'reserved_patients_without_any_known_question_exposure_or_schedule':len({byqid[q] for q in reserved}-known_q_patients),'protected1397_unchanged':True,'protected_overlap_with_known_exposure_union':len(protected&known_union),'limits':['Remaining977 are clean against identified question-ID sets, not proven untouched against all engineering history.','V5/V6/V7 interface metadata omits QIDs; named V6/V7 custom input hashes differ from the old V4 interface, and V5 input identity is unavailable. Do not silently assume their IDs belong to321.','The32-subject coarse ledger records substantive QA/answer/few-shot/gold-packet exposure without exact QIDs. This further limits fully uninspected-question claims, beyond the six unresolved interface slots.','All35 reserved patients have prior documented or scheduled exposure; this is fresh-relative-to-known-ID adaptive development, not new-patient/template/model confirmation.','Prior metadata scan attestation reused; no answer file was reopened and no historical raw correctness was inspected.','Metadata template IDs and patient IDs do not establish identical question semantics or independent task transfer.','No scorer/model/FHIR invocation, cache/program replay, new QA field or protected-case content inspected.']}
for name,obj in source_union.items(): assert set(obj['question_ids'])<=set(byqid)
dump('exposure_ID_sets.json',source_union);dump('known_question_exclusion_union.json',{'question_ids':sorted(known_union),'count':len(known_union),'scope':'Explicit prior attempted IDs plus all scheduled V8 IDs plus2 verified V8 interface IDs; NOT patient-expanded','reserved_question_ids':sorted(known_union&reserved),'reserved_count':len(known_union&reserved)})
dump('historical_overlap_provenance.json',{'historical_ledger_path':str(histpath),'historical_ledger_sha256':sha(histpath),'committed_attestation_path':'docs/evaluation_exposure_history_20261015.json','committed_attestation_sha256':sha(ROOT/'docs/evaluation_exposure_history_20261015.json'),'union_proof':hist['union_proof'],'overlaps':overlap})
dump('remaining_counts_by_patient.json',patient_rows);dump('summary.json',summary);dump('source_catalog.json',sources)
print(json.dumps(summary,indent=2))
