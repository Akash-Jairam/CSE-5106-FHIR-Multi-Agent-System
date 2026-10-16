"""Independent completed120 audit; unchanged pure scorer, no runtime/data service."""
from collections import Counter,defaultdict
from datetime import datetime
import hashlib,importlib.util,json,math,re,statistics
from pathlib import Path,PurePosixPath
import zipfile
ROOT=Path(__file__).resolve().parents[3]
RECORD=ROOT/'benchmarks/design-v8-protocol-repair-20261015'
DEST=Path(__file__).resolve().parent
PINS={'terminal_proof.json':'5749508694750f6d2aab0318842ff14fd8fb87d490400f91e21bf722eb82972c',
 'completion_integrity.json':'677aaba828fd29b0546a5b6fc7ed7b1faa691a8d1b6f2e602c8c5113433613d7',
 'official_analysis.json':'8ecfb13c8f5491b6d57dbdb0eb0e3fa0be15addcd2dbbc1a66c206696bff71c4',
 'integrity_pins.json':'3cac5e151acd882e40e348d733380a9f354513605349f5f28dc3280c7278bd7b',
 'supplementary/supplementary.json':'6f6eb89c62b8ce27d1a17a55b07656005b37d8a86c149c8a5f81c1651e8aea49'}
MODES=('A','W','R','C');FIELDS={'answer','evidence','operation','contributing_ids','limitation'}
def require(x,msg):
 if not x:raise ValueError(msg)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
def load(path):return json.loads(Path(path).read_text(),parse_constant=lambda _:(_ for _ in()).throw(ValueError('NonfiniteJSON')))
def canon(v):return json.dumps(v,sort_keys=True,allow_nan=False)
def failed(r):return r.get('error')is not None and str(r['error']).strip()not in('','nan')
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def zip_check(path,inventory,expected):
 require(sha(path)==expected,'ArchiveSHA')
 with zipfile.ZipFile(path)as z:
  require(len(z.namelist())==len(inventory)and set(z.namelist())==set(inventory),'Archiveinventory')
  for name,digest in inventory.items():require(hashlib.sha256(z.read(name)).hexdigest()==digest,'Archivedsourcebytes')
def measured(xs):
 xs=list(xs);known=[x for x in xs if type(x)in(int,float)and math.isfinite(x)]
 return {'n':len(xs),'known':len(known),'unknown':len(xs)-len(known),'known_sum':sum(known),'mean':statistics.mean(known)if known else None,'median':statistics.median(known)if known else None}
def main():
 for name,digest in PINS.items():require(sha(RECORD/name)==digest,'Externalpin:'+name)
 terminal=load(RECORD/'terminal_proof.json');proof=load(RECORD/'completion_integrity.json');pins=load(RECORD/'integrity_pins.json');official=load(RECORD/'official_analysis.json');supp=load(RECORD/'supplementary/supplementary.json')
 require(terminal['owned_study_process_terminal']is True and type(terminal['exit_code'])is int and terminal['saved_policy_outcomes']==480,'Observedterminal')
 require(proof['status']=='PASS'and proof['engineering_only']is False and proof['policy_outcomes']==480 and proof['questions']==120,'Actual480deepPASS')
 require(proof['owned_terminal_attestation_sha256']==PINS['terminal_proof.json']and proof['pins_sha256']==PINS['integrity_pins.json'],'Proofbindings')
 for name,digest in pins['metadata_sha256'].items():require(sha(RECORD/name)==digest,'Fivemetadata')
 freeze=load(RECORD/'manifest.json');sources=freeze['source_sha256'];runtime=load(RECORD/'runtime_adapter_manifest.json')
 require(len(sources)==pins['source_file_count']==proof['source_files']==226,'Sourcecount')
 require(hashlib.sha256(json.dumps(sources,sort_keys=True,ensure_ascii=True,allow_nan=False).encode()).hexdigest()==pins['source_inventory_sha256'],'SourceinventorySHA')
 for name,digest in sources.items():require(sha(ROOT/name)==digest,'CurrentsourceSHA')
 require(sources['scripts/v8_study_integrity.py']==pins['guard_sha256']==proof['guard_sha256'],'GuardSHA')
 zip_check(RECORD/'source.zip',sources,freeze['source_archive_sha256'])
 require(len(runtime['source_sha256'])==proof['runtime_files']==4,'Runtimecount')
 for name,digest in runtime['source_sha256'].items():require(sha(ROOT/name)==digest,'CurrentpublicruntimeSHA')
 zip_check(RECORD/'runtime_adapter_sources.zip',runtime['source_sha256'],runtime['archive_sha256'])
 require(sha(RECORD/'design.txt')==freeze['design_sha256'],'DesignSHA')
 study=load(ROOT/pins['study_manifest_path']);require(sha(ROOT/pins['study_manifest_path'])==pins['study_manifest_sha256']==proof['study_manifest_sha256'],'StudySHA')
 require(proof['study_manifest_path']==pins['study_manifest_path'],'Studypath')
 prospective=load(RECORD/'prospective_protocol.json');require(canon(prospective['study_policy'])==canon(study['policy']),'Prospectivepolicy')
 out=Path(terminal['output_directory']);require(out.resolve()==Path(freeze['output_directory']).resolve(),'Outputowner')
 # Reuse the external deep480 proof; recheck raw top-level/journal snapshots, not all CRCs/caches again.
 rawfiles=['study_outcomes.json','study_state.json','study_identity.json']+[j['path']for j in proof['physical_provider']['journals']]
 for name in rawfiles:require(sha(out/name)==proof['raw_sha256'][name],'Rawsnapshot')
 rows=load(out/'study_outcomes.json');state=load(out/'study_state.json');require(len(rows)==480 and state['saved_policy_outcomes']==480 and state['phase']=='generation_complete_with_errors','Completefirsts')
 cases=study['cases'];require(len(cases)==120 and len({c['question_id']for c in cases})==120,'120cases')
 index={}
 for r in rows:
  key=(r['question_id'],r['study_arm']);require(key not in index and r['study_terminal']is True and not r.get('attempt_history'),'Uniquecompletedfirst');index[key]=r
 require(set(index)=={(c['question_id'],m)for c in cases for m in MODES},'Exact480cohort')
 for c in cases:
  a=index[c['question_id'],'A']
  for m in MODES:
   r=index[c['question_id'],m];require(all(canon(r.get(k))==canon(a.get(k))for k in('question','true_answer','template','patient_fhir_id','primary_envelope_sha256')),'Pairedreference/Aidentity')
 # First possible scorer imports follow terminal/provenance/raw/full-cohort gates.
 scorer=module(ROOT/'utils/answer_scorer.py','independent_v8_scorer');stats=module(ROOT/'scripts/experiment_stats.py','independent_v8_stats')
 correct={k:not failed(r)and scorer.score_answer(r['agent_answer'],r['true_answer'],r['question'],r['template'])['typed_correct']for k,r in index.items()}
 arms={m:{'n':120,'correct':sum(bool(correct[c['question_id'],m])for c in cases),'terminal_errors':sum(failed(index[c['question_id'],m])for c in cases)}for m in MODES}
 mechanisms={};patients=set();operations=set()
 for m in('W','R','C'):
  counts=Counter()
  for c in cases:
   key=(c['question_id'],m);r=index[key];a=bool(correct[c['question_id'],'A']);f=bool(correct[key]);d=r['mechanism'];b=d.get('released_source_bound_B')
   require(b is None or isinstance(b,dict)and set(b)==FIELDS,'Bshape')
   selected=d['selected_B'];fatal=d['auxiliary_fatal'];require(type(selected)is bool and type(fatal)is bool and not(selected and fatal),'Effectflags')
   bc=b is not None and scorer.score_answer(b['answer'],r['true_answer'],r['question'],r['template'])['typed_correct']
   opportunity=not a and b is not None and bool(bc)
   counts.update({'O':int(opportunity),'H':int(selected and not a and f),'M':int(selected and a and not f),'F':int(fatal and a and not f),
    'all_post_A_helpful':int(not a and f),'all_post_A_harmful':int(a and not f),'selected_B':int(selected),'B_available_source_bound':int(b is not None),'auxiliary_fatal':int(fatal)})
   if m=='R'and opportunity:patients.add(c['patient_fhir_id']);operations.add(c['operation_group'])
  counts['net']=counts['H']-counts['M']-counts['F'];require(counts['net']==arms[m]['correct']-arms['A']['correct'],'HMFexactnet');mechanisms[m]=dict(counts)
 rule=study['policy']['advance_gate'];require(rule=={'O_min':6,'net_min':4,'harm_plus_fatal_max':2,'corrective_patients_min':3,'corrective_operations_min':2},'Predeclaredgate')
 rr=mechanisms['R'];passed=rr['O']>=6 and rr['net']>=4 and rr['M']+rr['F']<=2 and len(patients)>=3 and len(operations)>=2
 require(canon(arms)==canon(official['arms'])and canon(mechanisms)==canon(official['mechanisms']),'Officialcounts')
 require(official['advance_gate']=={**rule,'passed':passed}and official['R_corrective_patients']==len(patients)and official['R_corrective_operations']==len(operations),'Officialgate')
 paired={left+'-'+right:stats.paired_difference([int(correct[c['question_id'],left])for c in cases],[int(correct[c['question_id'],right])for c in cases],[c['patient_fhir_id']for c in cases],n_boot=10000,seed=5106)for left,right in(('R','A'),('R','W'),('R','C'))}
 require(canon(paired)==canon(official['paired_question_weighted_patient_bootstrap']),'Bootstrapreproduction')
 # Distinguish physically executed suffixes from four conceptual policies.
 physical=[];phase_unknown=Counter();concept={};cost=[];phys_tools=0
 for c in cases:
  a=index[c['question_id'],'A'];prefix=a['tokens']['per_call'];physical.extend(('A',x)for x in prefix);phase_unknown['A']+=sum(x.get('usage_available')is False for x in prefix)
  phys_tools+=a['budget']['tool_calls_run']
  for m in c['followup_order']:
   r=index[c['question_id'],m];calls=r['tokens']['per_call'];require(canon(calls[:len(prefix)])==canon(prefix),'ExactAusageprefix');suffix=calls[len(prefix):];physical.extend((m,x)for x in suffix);phase_unknown[m]+=sum(x.get('usage_available')is False for x in suffix);phys_tools+=r['budget']['tool_calls_run']-a['budget']['tool_calls_run']
 for m in MODES:
  selected=[index[c['question_id'],m]for c in cases]
  concept[m]={'known_prompt':sum(r['tokens']['prompt']for r in selected),'known_completion':sum(r['tokens']['completion']for r in selected),'known_cached_prompt':sum(r['tokens']['cached_prompt']for r in selected),'accounted_invocations':sum(r['tokens']['llm_calls']for r in selected),'unknown_usage_invocations':sum(x.get('usage_available')is False for r in selected for x in r['tokens']['per_call']),
   'reported_cost':measured(r['tokens']['cost']for r in selected),'policy_latency':measured(r['latency']for r in selected),'charged_tools':sum(r['budget']['tool_calls_run']for r in selected)}
 known=[x for m,x in physical if x.get('usage_available')is True];physical_usage={'actual_invocations':len(physical),'known_usage_invocations':len(known),'unknown_usage_invocations':len(physical)-len(known),'unknown_by_actual_phase':dict(phase_unknown),'known_prompt':sum(x['prompt']for x in known),'known_completion':sum(x['completion']for x in known),'known_cached_prompt':sum(x['cached']for x in known),'physical_charged_tools':phys_tools}
 require(physical_usage['actual_invocations']==proof['physical_provider']['observed_sdk_ends']==1209 and physical_usage['unknown_usage_invocations']==11,'PhysicalSDKcount')
 journal=[]
 for j in proof['physical_provider']['journals']:
  journal.extend(json.loads(line)for line in(out/j['path']).read_text().splitlines())
 ends=sorted((x for x in journal if x['event']=='end'),key=lambda x:x['counter']);require(len(ends)==len(physical),'Journalphysicalprefix')
 for (_,call),event in zip(physical,ends):
  if call.get('usage_available')is True:require((call['prompt'],call['completion'])==(event['usage']['prompt_tokens'],event['usage']['completion_tokens']),'Journalnumericusage')
  else:require(event['outcome']=='error','Physicalunknownerror')
 duration=measured(e['duration_seconds']for e in ends);error_durations=measured(e['duration_seconds']for e in ends if e['outcome']=='error')
 # Safe metadata reconciliation with the separately frozen supplementary report.
 require(canon(supp['arms'])==canon(arms)and canon(supp['mechanisms_exact_official_reconciliation'])==canon(mechanisms),'Supplementcounts')
 for m in MODES:
  c=concept[m];s=supp['accounting']['conceptual_policies'][m]
  for own,key in(('known_prompt','prompt'),('known_completion','completion'),('known_cached_prompt','cached_prompt'),('accounted_invocations','llm_calls'),('charged_tools','charged_tools')):require(c[own]==s[key]['known_sum'],'Supplementconceptual')
  require(math.isclose(c['policy_latency']['mean'],s['A_plus_suffix_latency_seconds']['known_mean'],abs_tol=1e-9),'Supplementlatency')
 ps=supp['accounting']['physical_A_once_plus_suffixes']
 for own,key in(('known_prompt','known_prompt_tokens'),('known_completion','known_completion_tokens'),('known_cached_prompt','known_cached_tokens'),('physical_charged_tools','charged_tools')):require(physical_usage[own]==ps[key]['known_sum'],'Supplementphysical')
 inventory=[];groups=Counter();sizes=Counter()
 for name in proof['archive_allowlist']:
  rel=PurePosixPath(name);require(not rel.is_absolute()and'..'not in rel.parts and not name.startswith('case_cache/'),'Safeallowlist')
  require(name in('study_identity.json','study_outcomes.json','study_state.json')or re.fullmatch(r'audit/[a-f0-9]{64}\.json\.gz',name)or re.fullmatch(r'primaries/[a-f0-9]{24}\.json\.gz',name)or re.fullmatch(r'provider-events/azure-diagnostics-[0-9]+-[a-f0-9]{32}\.jsonl',name),'Expectedrawmembertype')
  p=out/name;require(not p.is_symlink()and p.is_file(),'Ownedrawmember');kind=rel.parts[0]if len(rel.parts)>1 else'top_level';n=p.stat().st_size;groups[kind]+=1;sizes[kind]+=n;inventory.append({'path':name,'bytes':n,'guard_raw_sha256':proof['raw_sha256'][name]})
 require(len(inventory)==len({x['path']for x in inventory})==506,'Exactlyallowlist')
 archived=load(RECORD/'raw_archive_manifest.json');archive=ROOT/archived['archive_path']
 require(archived['completion_integrity_sha256']==PINS['completion_integrity.json']and archived['archive_sha256']=='2e51e77019b8a958c3a5e79112507d2b977b81505ead560fe7c4be5f2462ef60','Archiveexternalbinding')
 require(sha(archive)==archived['archive_sha256']and archive.stat().st_size==archived['archive_bytes']==348733378,'CompletedZIPbytes')
 members={x['path']:x for x in archived['members']}
 require(len(members)==len(archived['members'])==506 and set(members)==set(proof['archive_allowlist']),'CompletedZIPallowlist')
 for x in inventory:require(members[x['path']]['sha256']==x['guard_raw_sha256']and members[x['path']]['bytes']==x['bytes'],'Completedmembermanifest')
 with zipfile.ZipFile(archive)as z:
  require(len(z.infolist())==506 and {x.filename for x in z.infolist()}==set(members),'ZIPcentralinventory')
  for x in z.infolist():require(x.file_size==members[x.filename]['bytes'],'ZIPcentralsizes')
 require(all(failed(index[c['question_id'],m])for c in cases if failed(index[c['question_id'],'A'])for m in MODES),'SharedAerrorsallfour')
 report={'status':'INDEPENDENT_COMPLETED120_REVIEW_PASS','provenance':{'external_pins':PINS,'source_manifest_sha256':sha(RECORD/'manifest.json'),'source_files_verified':226,'runtime_files_verified':4,'source_archive_sha256':sha(RECORD/'source.zip'),'runtime_archive_sha256':sha(RECORD/'runtime_adapter_sources.zip'),'scorer_sha256':sha(ROOT/'utils/answer_scorer.py'),'statistics_sha256':sha(ROOT/'scripts/experiment_stats.py'),'raw_top_level_and_journal_SHA_rechecked':True,'all480_CRC_repeated':False},'arms':arms,'mechanisms':mechanisms,'fixed_gate':{**rule,'passed':passed},'patient_cluster_bootstrap':paired,'physical_usage':physical_usage,'conceptual_usage':concept,'runtime':{'physical_generation_wall_seconds':state['physical_generation_wall_seconds'],'completed_SDK_seconds':duration,'error_SDK_seconds':error_durations,'sum_A_once_and_followups_seconds':ps['physical_A_wall_seconds']['known_sum']+ps['followup_wall_seconds']['known_sum'],'reported_physical_raw_cost':ps['cost'],'reported_physical_phase_cost':ps['derived_reported_phase_cost']},'error_distinction':{'conceptual_failed_policies':sum(a['terminal_errors']for a in arms.values()),'physical_errors':sum(e['outcome']=='error'for e in ends),'physical_primary_errors':phase_unknown['A'],'physical_auxiliary_errors':sum(phase_unknown[m]for m in('W','R','C')),'all_observed_error_chains':dict(Counter(' -> '.join(x['type']['class']for x in e['exception_chain'])for e in ends if e['outcome']=='error')),'primary_error_policy_replication':4},'archive_plan':{'allowed_members':len(inventory),'member_counts':dict(groups),'uncompressed_member_bytes':dict(sizes),'sum_member_bytes':sum(sizes.values()),'max_member_bytes':max(x['bytes']for x in inventory),'zip_size_not_yet_measured':True,'cache_pickle_members':0,'raw_sha256_proof_entries_not_an_archive_allowlist':len(proof['raw_sha256'])},'limits':['Official typed scores are benchmark labels, not source/clinical semantic truth.','Shared-A dependent policies, adaptive development cohort and partial coverage do not establish protected1397/generalization efficacy.','Known token subtotals exclude11 physically unreported error usages; reported monetary costs unavailable, not zero.','Raw gzip/ownership/CRC complete guard is externally pinned and reused; this audit does not re-execute or restore every480again.','No programs/checkers/inference/FHIR/network/config or protected question/gold accessed.']}
 # Fail closed on any late source/report/raw-top drift before publication.
 for name,digest in PINS.items():require(sha(RECORD/name)==digest,'Lateexternaldrift')
 for name,digest in sources.items():require(sha(ROOT/name)==digest,'Latesourcedrift')
 for name in rawfiles:require(sha(out/name)==proof['raw_sha256'][name],'Laterawdrift')
 report['archive_plan']['completed_archive']={'path':archived['archive_path'],'bytes':archived['archive_bytes'],'sha256':archived['archive_sha256'],'member_inventory_and_sizes_match':True,'root_manifest_all_member_SHA_roundtrip_verified':archived['roundtrip_all_member_sha256_verified'],'independent_CRC_repeat':False,'archive_manifest_sha256':sha(RECORD/'raw_archive_manifest.json')}
 report['archive_plan']['zip_size_not_yet_measured']=False
 report['error_distinction']['each_of_10_primary_error_cases_fails_all_four_policies']=True
 report['conceptual_dedup_reconciliation']={'total_known_tokens':sum(v['known_prompt']+v['known_completion']for v in concept.values()),'total_accounted_invocations':sum(v['accounted_invocations']for v in concept.values()),'known_tokens_equal_physical_plus_3A':sum(v['known_prompt']+v['known_completion']for v in concept.values())==physical_usage['known_prompt']+physical_usage['known_completion']+3*(concept['A']['known_prompt']+concept['A']['known_completion']),'attempts_equal_physical_plus_3A':sum(v['accounted_invocations']for v in concept.values())==physical_usage['actual_invocations']+3*concept['A']['accounted_invocations']}
 require(report['conceptual_dedup_reconciliation']['known_tokens_equal_physical_plus_3A']and report['conceptual_dedup_reconciliation']['attempts_equal_physical_plus_3A'],'Exactphysicalconceptualdedup')
 (DEST/'independent_review.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 (DEST/'archive_allowlist_plan.json').write_text(json.dumps({'basis_completion_report_sha256':PINS['completion_integrity.json'],'output_directory':str(out),'members':inventory},indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':report['status'],'arms':arms,'fixed_gate':report['fixed_gate'],'physical_usage':physical_usage,'generation_wall_seconds':state['physical_generation_wall_seconds']}))
if __name__=='__main__':main()
