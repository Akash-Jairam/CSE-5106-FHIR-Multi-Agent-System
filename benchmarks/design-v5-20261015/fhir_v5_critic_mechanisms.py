"""Post-terminal V5 mechanisms only; no inference or partial-result scoring.

All root-supplied SHA pins, complete raw files, identities, frozen code and 600
losslessly restored audits must pass before the unchanged scorer is imported.
Official paired totals remain authoritative; this helper reconciles them and
separates accepted semantic edits from all critic-stage gains/losses.
"""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

COUNT = 300
EXPERIMENT = 'design-v5-20261015'
RUN_RELATIVE = 'output/experiments/gpt-6-luna/executable_critic_v5_300q'
ARMS = {'ordinary_single_agent':'single_agent','ordinary_multi_agent':'multi_agent_v5'}
PROTOCOL = 'frozen-workspace-executable-critic-v5'
CLEAN = {'source_valid_primary_preserved','executed_source_checked_critic_proposal'}
HEX = re.compile(r'^[a-f0-9]{64}$')


def require(test, reason):
    if not test: raise ValueError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(1024*1024): digest.update(data)
    return digest.hexdigest()


def load(path):
    def pairs(items):
        result = {}
        for key,value in items:
            require(key not in result,'Duplicate JSON key'); result[key]=value
        return result
    def bad(value): raise ValueError('Nonfinite JSON')
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=bad)


def same(a,b):
    return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def failed(row):
    error=row.get('error')
    return error is not None and str(error).strip() not in ('','nan')


def artifact(value):
    return isinstance(value,dict) and not value.get('error') and all(k in value for k in
        ('answer','evidence','operation','contributing_ids','limitation'))


def transition(before,after,applicable=True):
    if not applicable: return 'not_applied'
    if before is None or after is None: return 'unavailable'
    return 'helpful' if not before and after else 'harmful' if before and not after else 'neutral'


def phase_refusal(error):
    text=str(error).lower()
    return any(s in text for s in ('phase token ceiling','phase tool ceiling',
        'model-requested tool ceiling','global tool allowance reserved','known-budget refusal',
        'budget exceeded: token budget:','tool-call budget:'))


def critic_responses(trace):
    started=False;result=[]
    for message in trace if isinstance(trace,list) else []:
        if not isinstance(message,dict):continue
        if (message.get('role')=='system' and isinstance(message.get('content'),str)
                and message['content'].startswith('Review the actual executed program against the ORIGINAL question')):
            started=True;continue
        if not started or message.get('role')!='assistant' or message.get('phase')=='critic' or message.get('model_call') is False:
            continue
        if message.get('tool_calls'):
            result.append('native_tool_call_rejected');continue
        try:
            value=json.loads(message.get('content') or '')
            decision=value.get('decision') if isinstance(value,dict) else None
        except (ValueError,TypeError):decision=None
        result.append(decision if decision in ('KEEP','UNKNOWN','INSPECT','PATCH') else 'malformed')
    return result


def critic_stage(critic,trace):
    if not isinstance(critic,dict): return 'not_reached_or_diagnostics_unavailable'
    if critic.get('status')=='coverage_skip':return 'coverage_skip'
    if critic.get('status')=='accounting_failed':return 'accounting_failed'
    if critic.get('status')=='released':return 'released'
    first=critic.get('first') or {};proposal=critic.get('proposal') or {}
    if first.get('decision') in ('KEEP','UNKNOWN'):return 'first_'+first['decision'].lower()
    observed=critic_responses(trace)
    if not first:
        if observed[:1]==['native_tool_call_rejected']:return 'first_native_tool_call_rejected'
        if observed:return 'first_json_or_schema_rejected'
        return 'first_response_unavailable' if critic.get('model_calls') else 'preflight_unusable' 
    if 'probe_output' not in critic:return 'inspect_not_executed'
    if not isinstance(critic['probe_output'],dict) or critic['probe_output'].get('error'):return 'probe_failed'
    if 'validated_probe' not in critic:return 'probe_unusable'
    if proposal.get('decision') in ('KEEP','UNKNOWN'):return 'second_'+proposal['decision'].lower()
    if not proposal:
        if len(observed)>1 and observed[1]=='PATCH':return 'patch_rejected_before_execution'
        if len(observed)>1 and observed[1]=='native_tool_call_rejected':return 'second_native_tool_call_rejected'
        if len(observed)>1:return 'second_json_or_schema_rejected'
        return 'second_response_unavailable' if critic.get('model_calls',0)>=2 else 'second_budget_refusal_or_unavailable' 
    output=critic.get('alternate_execution')
    if output is None:return 'candidate_not_executed'
    if not isinstance(output,dict) or output.get('error'):return 'candidate_execution_failed'
    if 'source_check' in critic:
        return 'candidate_limitation' if (critic.get('alternate') or {}).get('limitation') else 'source_check_rejected'
    if 'no substantive executed effect' in str(critic.get('reason')):return 'no_substantive_effect'
    source_attempt=any(isinstance(m,dict) and m.get('tool_call_id')=='v5-critic-source-check' for m in trace or [])
    return 'source_check_unusable' if source_attempt else 'candidate_validation_or_tool_allowance_failed'


def row_record(row,score):
    """score(answer) is injected; reference labels never enter the runtime."""
    revision=row.get('revision') if isinstance(row.get('revision'),dict) else {}
    critic=revision.get('critic') if isinstance(revision.get('critic'),dict) else None
    phases=revision.get('phases') if isinstance(revision.get('phases'),list) else []
    phases={p['phase']:p for p in phases if isinstance(p,dict) and isinstance(p.get('phase'),str)}
    primary=revision.get('primary');draft=revision.get('draft');final=revision.get('final')
    def candidate(value,full=False):
        if not isinstance(value,dict) or 'answer' not in value or (full and not artifact(value)):return None
        return bool(score(value['answer'])['typed_correct'])
    final_scored=score(row.get('agent_answer'));terminal=failed(row)
    correct=bool(not terminal and final_scored['typed_correct'])
    primary_correct=candidate(primary,True);draft_correct=candidate(draft)
    policy=revision.get('release_policy');primary_revision=revision.get('primary_revision') or {}
    fatal=any(bool(p.get('fatal')) for p in phases.values())
    tokens=row.get('tokens') if isinstance(row.get('tokens'),dict) else {}
    calls=tokens.get('per_call')
    usage='metadata_unavailable' if not isinstance(calls,list) or not calls else (
        'recorded_known' if all(isinstance(c,dict) and c.get('usage_available') is True for c in calls) else 'recorded_unknown')
    matches=isinstance(final,dict) and all(k in final for k in ('answer','evidence','limitation')) and same(
        {k:final[k] for k in ('answer','evidence','limitation')},
        {'answer':row.get('agent_answer'),'evidence':row.get('evidence'),'limitation':row.get('limitation')})
    source_valid=bool(not terminal and not fatal and usage=='recorded_known' and policy in CLEAN
        and revision.get('final_check_issues')==[] and not revision.get('unresolved_scope') and artifact(final) and matches)
    precritic_valid=bool(artifact(primary) and primary_revision.get('final_check_issues')==[]
        and not revision.get('unresolved_scope') and not phases.get('solver',{}).get('fatal')
        and policy in CLEAN|{'aggregate_accounting_failed'} and 'critic' in phases)
    accepted=bool(source_valid and precritic_valid and policy=='executed_source_checked_critic_proposal'
        and critic and critic.get('status')=='released' and revision.get('critic_revised') is True)
    source_revised=bool(revision.get('solver_source_revised'))
    errors={name:list(p.get('errors') or []) for name,p in phases.items()}
    local={name:[str(e) for e in items if phase_refusal(e)] for name,items in errors.items()}
    local={name:items for name,items in local.items() if items}
    budget=row.get('budget') if isinstance(row.get('budget'),dict) else {}
    stop=bool(budget.get('stopped'))
    trace=row.get('trace') or []
    decisions=critic_responses(trace)
    return {'question_id':row.get('question_id'),'correct':correct,'terminal_error':terminal,
        'reference_answer_kind':final_scored.get('answer_type'),'typed_reason':final_scored.get('typed_reason'),
        'draft_present':isinstance(draft,dict) and 'answer' in draft,'draft_typed_correct':draft_correct,
        'primary_artifact_present':artifact(primary),'primary_typed_correct':primary_correct,
        'primary_is_actual_null':artifact(primary) and primary['answer'] is None,
        'final_candidate_typed_correct':candidate(final),'source_valid_release':source_valid,
        'precritic_source_valid_primary':precritic_valid,'release_policy':policy,
        'critic_accepted_repair':accepted,'critic_semantic_repair_effect':transition(primary_correct,correct,accepted),
        'critic_stage_effect':transition(primary_correct,correct,precritic_valid),
        'critic_stage_primary_credit_lost':precritic_valid and primary_correct is True and not correct,
        'critic_stage_nonrepair_credit_lost':precritic_valid and primary_correct is True and not correct and not accepted,
        'solver_source_revised':source_revised,
        'solver_source_revision_effect':transition(draft_correct,primary_correct,source_revised),
        'critic_status':critic.get('status') if critic else 'not_recorded',
        'critic_reason':str(critic.get('reason') or '') if critic else None,
        'critic_first_prompt_tokens':critic.get('first_prompt_tokens') if critic else None,
        'critic_two_call_token_reservation':critic.get('two_call_token_reservation') if critic else None,
        'critic_stage':critic_stage(critic,trace),'critic_first_decision':(critic.get('first') or {}).get('decision') if critic else None,
        'critic_second_decision':(critic.get('proposal') or {}).get('decision') if critic else None,
        'observed_critic_decisions':decisions,'critic_patch_attempted':bool('PATCH' in decisions or
            (critic and (critic.get('proposal') or {}).get('decision')=='PATCH')),
        'critic_admitted_invocations':critic.get('model_calls') if critic else None,
        'critic_charged_tools':critic.get('tool_calls') if critic else None,
        'phase_fatal':fatal,'usage_status':usage,'recorded_unknown_provider_usage':usage=='recorded_unknown',
        'usage_metadata_unavailable':usage=='metadata_unavailable','phase_errors':errors,'phase_local_refusals':local,
        'phase_recorded_error':any(errors.values()),'phase_local_refusal':bool(local),
        'aggregate_budget_stop':stop,'local_refusal_without_aggregate_stop':bool(local) and not stop,
        'reported_global_overshoot':bool((type(tokens.get('total')) in (int,float) and tokens['total']>200000)
            or (type(budget.get('tool_calls_run')) in (int,float) and budget['tool_calls_run']>30)),
        'reported_phase_overshoot':any(type(p.get(k)) in (int,float) and type(p.get(cap)) in (int,float)
            and p[k]>p[cap] for p in phases.values() for k,cap in [('tokens','token_cap'),('tool_calls','tool_cap')]),
        'critic_completion_subcap_overshoot':any('completion exceeded requested subcap' in str(e).lower()
            for items in errors.values() for e in items),
        'actual_answer_is_null':row.get('agent_answer') is None,
        'typed_null_credit':correct and row.get('agent_answer') is None,
        'typed_null_credit_outside_source_valid_release':correct and row.get('agent_answer') is None and not source_valid,
        'empty_reference_credit_outside_source_valid_release':correct and final_scored.get('answer_type')=='empty' and not source_valid,
        'source_valid_typed_wrong_release':source_valid and not correct,
        'phase_records':phases,'tokens':tokens.get('total'),'model_calls':tokens.get('llm_calls'),
        'tool_calls':budget.get('tool_calls_run'),'latency':row.get('latency')}


def summarize(records):
    flags=('correct','terminal_error','primary_artifact_present','primary_is_actual_null',
      'source_valid_release','source_valid_typed_wrong_release','precritic_source_valid_primary',
      'critic_accepted_repair','critic_patch_attempted','critic_stage_primary_credit_lost','critic_stage_nonrepair_credit_lost',
      'solver_source_revised','phase_fatal','recorded_unknown_provider_usage','usage_metadata_unavailable',
      'phase_recorded_error','phase_local_refusal','aggregate_budget_stop','local_refusal_without_aggregate_stop',
      'reported_global_overshoot','reported_phase_overshoot','critic_completion_subcap_overshoot',
      'actual_answer_is_null','typed_null_credit','typed_null_credit_outside_source_valid_release',
      'empty_reference_credit_outside_source_valid_release')
    result={'questions':len(records),**{k:sum(bool(r[k]) for r in records) for k in flags}}
    for key in ('critic_semantic_repair_effect','critic_stage_effect','solver_source_revision_effect',
                'critic_status','critic_stage','critic_first_decision','critic_second_decision','usage_status','release_policy'):
        result[key+'_counts']=dict(Counter(r[key] for r in records))
    valid=[r for r in records if r['precritic_source_valid_primary']]
    accepted=[r for r in records if r['critic_accepted_repair']]
    result['critic_stage_delta_correct']=sum(r['correct']-r['primary_typed_correct'] for r in valid)
    result['accepted_repair_delta_correct']=sum(r['correct']-r['primary_typed_correct'] for r in accepted)
    result['critic_stage_denominator']=len(valid)
    result['accepted_semantic_repair_denominator']=len(accepted)
    result['all_first_attempt_accuracy']=result['correct']/len(records) if records else None
    return result


def verify_all(args):
    require(args.confirm_both_owned_jobs_terminal is True,'Explicit root terminal confirmation required')
    root=Path(args.root).resolve();record=root/'benchmarks'/EXPERIMENT;output=root/RUN_RELATIVE
    checked={}
    def pinned(path,digest):
        path=Path(path)
        require(isinstance(digest,str) and HEX.fullmatch(digest),'External SHA256 pin required')
        require(path.is_file() and not path.is_symlink() and sha(path)==digest,'Pinned file differs: '+str(path))
        checked[path]=digest;return path
    proof_path=pinned(args.completion_report,args.completion_report_sha256)
    official_path=pinned(args.official_report,args.official_report_sha256)
    guard_path=pinned(args.guard_module,args.guard_sha256)
    pins_path=pinned(args.guard_pins,args.guard_pins_sha256)
    proof,official=load(proof_path),load(official_path)
    require(proof.get('status') in {'PASS','PASS_WITH_JOURNAL_LIMITATIONS'} and proof.get('mode')=='completion'
        and proof.get('caller_asserted_owned_jobs_terminal') is True,'Completed terminal integrity PASS required')
    require(proof.get('expected_first_attempts')==600 and proof.get('restored_audit_records')==600
        and type(proof.get('unique_referenced_artifacts')) is int
        and 1<=proof['unique_referenced_artifacts']<=600
        and proof.get('scoring_performed') is False,'All600 unscored restored firsts required')
    require(proof.get('guard_sha256')==args.guard_sha256 and proof.get('pins_sha256')==args.guard_pins_sha256,
        'Completion proof does not bind supplied guard/pins')
    guard=module(guard_path,'v5_posthoc_reviewed_guard')
    pins,manifest,adapter,binding=guard.frozen_provenance(root,record,output,pins_path,args.guard_pins_sha256)
    require(guard.actual_source_inventory(root)==set(manifest['source_sha256']),'Frozen source inventory changed')
    require(guard.equal(proof.get('provider_binding'),binding),'Completion provider binding differs')
    for key,value in {'manifest_sha256':pins['record_metadata_sha256']['manifest.json'],
        'source_inventory_sha256':pins['source_inventory_sha256'],'frozen_source_files_verified':len(manifest['source_sha256']),
        'runtime_adapter_files_verified':4,'source_archive_sha256':manifest['source_archive_sha256'],
        'design_sha256':manifest['design_sha256'],'dataset_sha256':manifest['dataset_sha256'],
        'split_sha256':manifest['split_sha256'],'code_hash':guard.B.code_hash(root),
        'reviewed_base_sha256':guard.BASE_SHA}.items():
        require(same(proof.get(key),value),'Completion source binding differs: '+key)
    for name,digest in manifest['source_sha256'].items():pinned(guard.public_path(root,name),digest)
    require({'utils/answer_scorer.py','scripts/compare_runs.py','scripts/audit_store.py','manifests/question_families.csv'}<=set(manifest['source_sha256']),
        'Frozen scoring/comparison/store/family files missing')
    pinned(guard.BASE_PATH,guard.BASE_SHA)
    for name,digest in pins['record_metadata_sha256'].items():pinned(record/name,digest)
    for path,digest in [(record/'source.zip',manifest['source_archive_sha256']),
        (record/'design.txt',manifest['design_sha256']),
        (record/'runtime_adapter_sources.zip',adapter['archive_sha256']),
        (root/'final_dataset/questions_answers_sql_fhir.csv',manifest['dataset_sha256']),
        (root/'manifests/split.json',manifest['split_sha256'])]:pinned(path,digest)
    for name,digest in guard.PROVIDER_SOURCE.items():pinned(guard.public_path(root,name),digest)
    require(set(proof.get('arms') or {})==set(ARMS),'Completion proof must contain both arms')
    raw_files=proof.get('raw_files');require(isinstance(raw_files,dict) and raw_files,'Completion raw allowlist missing')
    for relative,entry in raw_files.items():
        path=guard.B.owned_file(output,relative)
        require(isinstance(entry,dict) and path.stat().st_size==entry.get('bytes'),'Raw size differs')
        pinned(path,entry.get('sha256'))
    expected=load(root/'manifests/split.json')['splits']['held_out']['question_ids']
    require(isinstance(expected,list) and len(expected)==len(set(expected))==300,'Fixed300 order missing')
    require(official.get('questions')==300 and official.get('cohort')=='held_out'
        and official.get('dataset_sha256')==manifest['dataset_sha256']
        and official.get('families_sha256')==manifest['source_sha256']['manifests/question_families.csv']
        and official.get('question_ids_sha256')==hashlib.sha256(json.dumps(sorted(expected)).encode()).hexdigest(),
        'Frozen official report cohort/source differs')
    store=guard.B.load_public_store(root);rows={};refs=set()
    labels={'ordinary_single_agent':args.single_label,'ordinary_multi_agent':args.team_label}
    require(args.single_label!=args.team_label,'Current paired official labels must differ')
    for arm,strategy in ARMS.items():
        summary,identity,encoded,references=guard.verify_arm_snapshot(root,output,manifest,expected,proof['code_hash'],store,arm,strategy,True)
        require(same(summary,proof['arms'][arm]),'Completion arm facts changed: '+arm)
        require(summary.get('saved_first_attempts')==300 and summary.get('verified_owned_gzip_audits')==300,'All300 per arm required')
        for relative in [arm+'.json',arm+'.json.run.json',*references]:require(relative in raw_files,'Raw proof omits arm/audit file')
        refs.update(references)
        entry=(official.get('identities') or {}).get(labels[arm]) or {}
        require(Path(entry.get('result_path','')).resolve()==(output/(arm+'.json')).resolve()
            and entry.get('result_sha256')==summary['result_sha256'] and same(entry.get('identity'),identity),
            'Official arm raw/identity binding differs: '+arm)
        stats=(official.get('arms') or {}).get(labels[arm]) or {}
        require(type((stats.get('accuracy') or {}).get('all',{}).get('correct')) is int
            and stats.get('terminal_errors')==summary.get('terminal_errors',0)
            and stats.get('budget_stops')==summary.get('budget_stops',0),'Official arm accounting differs')
        rows[arm]=load(output/(arm+'.json'))
    require(len(refs)==proof['unique_referenced_artifacts'],
        'Restored audit references disagree with pinned unique-artifact count')
    for path,digest in checked.items():require(sha(path)==digest,'Input changed before scorer import')
    return {'root':root,'record':record,'output':output,'guard':guard,'store':store,'proof':proof,
            'official':official,'manifest':manifest,'rows':rows,'checked':checked,'labels':labels,'ids':expected}


def analyze(prepared,loader=module):
    root=prepared['root'];manifest=prepared['manifest'];guard=prepared['guard'];official=prepared['official']
    source={};families={}
    with (root/'final_dataset/questions_answers_sql_fhir.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['question_id'] in prepared['ids']:
                require(row['question_id'] not in source,'Duplicate source question');source[row['question_id']]=row
    with (root/'manifests/question_families.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['question_id'] in prepared['ids']:
                require(row['question_id'] not in families,'Duplicate family question');families[row['question_id']]=row
    require(set(source)==set(families)==set(prepared['ids']),'Missing fixed300 reference metadata')
    for path,digest in prepared['checked'].items():require(sha(path)==digest,'Input changed before scorer import')
    scorer=loader(root/'utils/answer_scorer.py','v5_posthoc_unchanged_scorer')
    results={};team_records=[]
    for arm,rows in prepared['rows'].items():
        correct=0
        for row in rows:
            qid=row['question_id'];reference=source[qid];family=families[qid]
            def score(answer):return scorer.score_answer(answer,reference['true_answer'],reference['question'],family['template'])
            if arm=='ordinary_multi_agent':
                restored,_=guard.verify_restored(row,prepared['output'],prepared['store'])
                record=row_record(restored,score);record['family']=family['family'];team_records.append(record)
                correct+=record['correct']
            else:correct+=not failed(row) and bool(score(row.get('agent_answer'))['typed_correct'])
        label=prepared['labels'][arm]
        require(correct==official['arms'][label]['accuracy']['all']['correct'],'Typed totals disagree with frozen official report')
        results[arm]={'questions':len(rows),'correct':correct,'official_label':label}
    require(len(team_records)==300,'Mechanism denominator must include every team first attempt')
    for path,digest in prepared['checked'].items():require(sha(path)==digest,'Input changed during scoring')
    report={'experiment':EXPERIMENT,'protocol':PROTOCOL,'schema_version':1,
      'scope':'Completed fixed300 first attempts; adaptive development evidence only.',
      'summary':summarize(team_records),'official_totals_reconciled':results,
      'completion_status':prepared['proof']['status'],'source_manifest_sha256':prepared['proof']['manifest_sha256'],
      'scorer_sha256':manifest['source_sha256']['utils/answer_scorer.py'],
      'compare_source_sha256':manifest['source_sha256']['scripts/compare_runs.py'],
      'helper_sha256':sha(__file__),
      'limits':['Accepted repair transitions are distinct from total critic-stage effects, which include terminal/accounting losses.',
        'A missing primary/draft is unavailable; an explicitly retained null artifact is scored as null.',
        'Source/execution provenance and probe observations do not establish semantic correctness.',
        'Unknown usage placeholders do not imply zero billing; phase-local refusals differ from provider/accounting failures.',
        'No Single execution-ancestry classification is attempted; official current-arm totals are reconciled only.',
        'All600 row-owned audits restore before scorer import; identical content-addressed payloads may share files.',
        'Current source/identity/raw/report SHA bindings and the pinned unique-artifact count are rechecked.',
        'No new inference, source re-fetches, or code re-execution occur in this helper.']}
    return report,team_records


def diagnostics_destination(args):
    """Only the formal metadata directory or explicit temporary space is writable."""
    raw=Path(args.output)
    require(raw.is_absolute() and '..' not in raw.parts,'Absolute diagnostics path without escapes required')
    # Check unresolved ancestors so resolve() cannot conceal a symlink route.
    for ancestor in (raw,*raw.parents):
        require(not ancestor.is_symlink(),'Diagnostics path has a symlink ancestor')
    root=Path(args.root).resolve();destination=raw.resolve()
    formal=root/'benchmarks'/EXPERIMENT/'diagnostics'
    temporary=Path('/private/tmp').resolve()
    require(destination==formal or (destination.is_relative_to(temporary) and not destination.is_relative_to(root)),
        'Diagnostics must use the formal metadata directory or external temporary space')
    require(not destination.exists(),'Fresh diagnostics destination required')
    return destination


def run(args):
    destination=diagnostics_destination(args)
    prepared=verify_all(args)
    report,records=analyze(prepared)
    report['external_bindings']={k:getattr(args,k) for k in ('completion_report_sha256','official_report_sha256',
        'guard_sha256','guard_pins_sha256')}
    require(diagnostics_destination(args)==destination,'Diagnostics destination changed during analysis')
    destination.mkdir(parents=True)
    (destination/'mechanisms.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (destination/'mechanisms_rows.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ('root','completion-report','completion-report-sha256','official-report','official-report-sha256',
                   'guard-module','guard-sha256','guard-pins','guard-pins-sha256','output'):
        parser.add_argument('--'+option,required=True)
    parser.add_argument('--confirm-both-owned-jobs-terminal',action='store_true',required=True)
    parser.add_argument('--single-label',default='v5_single');parser.add_argument('--team-label',default='v5_team')
    report=run(parser.parse_args(argv));print(json.dumps(report['summary'],indent=2,allow_nan=False))


if __name__=='__main__':main()
