"""Post-terminal V6 recorded-primary/independent-candidate mechanisms only.

No live/partial results may be analyzed. All externally pinned terminal deep600
integrity/source/raw/official checks and restorations finish before scorer import.
The A-only endpoint is the RECORDED175k/27 actual Single primary, not a hypothetical
200k/30 control. Missing primary metadata is unavailable, never fabricated null.
"""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re

COUNT=300
EXPERIMENT='design-v6-20261015'
RUN_RELATIVE='output/experiments/gpt-6-luna/independent_recomputation_v6_300q'
ARMS={'ordinary_single_agent':'single_agent','ordinary_multi_agent':'multi_agent_v6'}
PROTOCOL='continuous-primary-independent-recompute-v6'
HEX=re.compile(r'^[a-f0-9]{64}$')
FIELDS=('answer','evidence','operation','contributing_ids','limitation')
RENDERED=('answer','evidence','limitation')


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


def rendered(value):
    return {key:value.get(key) for key in RENDERED}


def proof_flags(row,guard):
    """Mechanical counterfactual-candidate bindings, before any scorer import."""
    revision=row.get('revision') if isinstance(row.get('revision'),dict) else {}
    diag=revision.get('primary_diagnostics') or {};a=revision.get('primary');execution=revision.get('primary_artifact')
    workspace=row.get('retrieved_fhir_resources') or {};trace=row.get('trace') or []
    result={'a_source_bound':False,'b_source_bound':False,'a_proof_issue':None,'b_proof_issue':None}
    if diag.get('eligible') is True:
        try:
            require(artifact(execution) and execution.get('mode')=='final' and all(key in (a or {}) for key in RENDERED),'Eligible A lacks actual metadata')
            five={key:execution[key] for key in FIELDS}
            require(same(rendered(five),rendered(a)) and execution.get('workspace_sha256')==guard.fingerprint(workspace),'A execution/workspace differs')
            previous=revision.get('primary_revision') or {}
            require(previous.get('final_check_issues')==[] and isinstance(previous.get('final'),dict)
              and same(rendered(previous['final']),rendered(a)),'A original source gate did not release this A')
            start=next((i for i,message in enumerate(trace) if isinstance(message,dict)
              and message.get('phase') in {'independent_generation','independent_selection','independent_recompute'}),len(trace))
            require(guard.validate_primary(trace[:start],five,workspace)==execution.get('code'),'A actual executed code differs')
            checks=[json.loads(message['content']) for message in trace[:start] if isinstance(message,dict)
              and message.get('role')=='tool' and message.get('tool_call_id') in {'automatic_field_check_1','automatic_field_check_2'}]
            guard._source_summary(five,checks[-1],workspace)
            result['a_source_bound']=True
        except (ValueError,TypeError,KeyError,IndexError,AttributeError):result['a_proof_issue']='eligible_A_execution_or_source_binding_unavailable'
    aux=revision.get('recompute') or {}
    if aux.get('B_eligible') is True:
        try:
            b=aux.get('B');output=aux.get('B_execution');require(artifact(b) and artifact(output) and output.get('mode')=='final'
              and 'error' not in output and 'error_type' not in output and same(guard.canonical_artifact(output),b),'B actual final envelope differs')
            calls={};replies={}
            for message in trace:
                if not isinstance(message,dict):continue
                if message.get('role')=='assistant' and message.get('phase') in {'independent_generation','independent_recompute'}:
                    for call in message.get('tool_calls') or []:
                        function=call['function'];calls[call['id']]=(function['name'],json.loads(function['arguments']))
                if message.get('role')=='tool' and message.get('phase')=='independent_recompute':replies[message['tool_call_id']]=json.loads(message['content'])
            ids=[key for key,(name,args) in calls.items() if name=='execute_python_code']
            require(len(ids)==1 and calls[ids[0]][1].get('mode')=='final' and same(replies.get(ids[0]),output),'B execution trace differs')
            check=aux.get('B_source_check');require(same(replies.get('v6-independent-source-check'),check)
              and same(calls.get('v6-independent-source-check'),('check_fields',{'citations':b['evidence']})),'B original-source trace differs')
            guard._source_summary(b,check,workspace)
            result['b_source_bound']=True
        except (ValueError,TypeError,KeyError,IndexError,AttributeError):result['b_proof_issue']='eligible_B_execution_or_source_binding_unavailable'
    return result


def row_record(row,score,proof):
    """score(answer) is injected; references are never part of runtime design."""
    revision=row.get('revision') if isinstance(row.get('revision'),dict) else {}
    diag=revision.get('primary_diagnostics') if isinstance(revision.get('primary_diagnostics'),dict) else None
    a=revision.get('primary');aux=revision.get('recompute') if isinstance(revision.get('recompute'),dict) else {}
    previous=revision.get('primary_revision') if isinstance(revision.get('primary_revision'),dict) else {}
    policy=revision.get('release_policy');terminal=failed(row)
    typed=score(row.get('agent_answer'));correct=bool(not terminal and typed['typed_correct'])
    a_available=False;a_correct=None;a_error=None;a_typed=None;reason='missing_primary_revision_or_rendered_A'
    primary_fatal=bool(diag and diag.get('fatal'))
    if revision.get('protocol')==PROTOCOL and diag is not None and isinstance(a,dict) and all(key in a for key in RENDERED):
        if primary_fatal:
            a_available=True;a_correct=False;a_error=True;reason='recorded_primary_fatal_failure'
        elif diag.get('released_available') is not True:
            reason='actual_primary_common_output_unavailable'
        elif diag.get('eligible') is False:
            # This branch preserves the actual Single error on the same row.
            a_available=True;a_error=terminal;a_typed=bool(score(a['answer'])['typed_correct'])
            a_correct=bool(not a_error and a_typed);reason='actual_ineligible_or_prose_A_inherits_row_error'
        elif diag.get('eligible') is True and proof.get('a_source_bound') is True:
            # A was valid before B. A later auxiliary fatal error cannot be
            # retroactively assigned to this counterfactual A-only endpoint.
            a_available=True;a_error=False;a_typed=bool(score(a['answer'])['typed_correct'])
            a_correct=a_typed;reason='actual_eligible_A_before_auxiliary'
        else:reason='declared_eligible_A_without_mechanical_binding'
    b=aux.get('B');b_bound=aux.get('B_eligible') is True and proof.get('b_source_bound') is True and artifact(b)
    b_correct=bool(score(b['answer'])['typed_correct']) if b_bound else None
    exact_agreement=same(a['answer'],b['answer']) if proof.get('a_source_bound') is True and isinstance(a,dict) and b_bound else None
    switched=policy=='selected_executed_independent_candidate' and revision.get('independent_candidate_selected') is True
    source_revised=revision.get('primary_source_revised') is True
    draft=previous.get('draft');draft_correct=bool(score(draft['answer'])['typed_correct']) if isinstance(draft,dict) and all(key in draft for key in RENDERED) else None
    before_gate=previous.get('final');before_gate_correct=bool(score(before_gate['answer'])['typed_correct']) if isinstance(before_gate,dict) and all(key in before_gate for key in RENDERED) else None
    source_issues=previous.get('final_check_issues')
    aux_entered=bool(aux) or any(isinstance(p,dict) and p.get('phase')=='independent_recompute' for p in revision.get('phases') or [])
    return {'question_id':row.get('question_id'),'correct':correct,'terminal_error':terminal,
      'reference_answer_kind':typed.get('answer_type'),'a_only_available':a_available,'a_only_correct':a_correct,
      'a_only_error':a_error,'a_only_typed_ignoring_error':a_typed,'a_only_disposition':reason,
      'a_only_is_recorded175k27_primary':True,'primary_fatal_recorded':primary_fatal,
      'primary_eligible_recorded':diag.get('eligible') if diag else None,'primary_source_bound':proof.get('a_source_bound') is True,
      'primary_known_budget_stop':bool(diag and diag.get('primary_budget_stop')),
      'primary_source_issue_recorded':bool(source_issues),'primary_initial_source_issues_recorded':bool(previous.get('check_issues')),'primary_source_revised':source_revised,
      'primary_initial_draft_correct':draft_correct,'primary_prewithholding_candidate_correct':before_gate_correct,
      'primary_source_revision_effect_to_released_A':transition(draft_correct,a_correct,source_revised),
      'primary_source_withholding_lost_typed_credit':bool(a_available and a_error is False and before_gate_correct is True and a_correct is False and source_issues),
      'b_eligible_recorded':aux.get('B_eligible') is True,'b_source_bound':b_bound,'b_typed_correct':b_correct,
      'b_answer_agreement_exact':exact_agreement,'b_answer_agreement_recorded':aux.get('answer_agreement'),
      'independent_candidate_selected':switched,'released_B_effect':transition(a_correct,correct,switched),
      'auxiliary_stage_effect':transition(a_correct,correct,aux_entered),'auxiliary_entered':aux_entered,
      'auxiliary_non_switch_credit_lost':bool(aux_entered and not switched and a_correct is True and not correct),
      'auxiliary_fatal_recorded':aux.get('fatal') is True,'auxiliary_status':aux.get('status','not_entered'),
      'auxiliary_reason':str(aux.get('reason') or '') if aux else None,'release_policy':policy,
      'generic_revised_flag_informational_only':bool(revision.get('revised')),
      'unknown_usage_recorded':any(call.get('usage_available') is not True for call in (row.get('tokens') or {}).get('per_call') or []),
      'team_exact_None_typed_credit':row.get('agent_answer') is None and correct,
      'a_exact_None_typed_credit':bool(a_available and isinstance(a,dict) and a['answer'] is None and a_correct),
      'a_proof_issue':proof.get('a_proof_issue'),'b_proof_issue':proof.get('b_proof_issue')}


def summarize(records):
    result={'questions':len(records),'team_correct':sum(r['correct'] for r in records)}
    flags=('terminal_error','a_only_available','primary_fatal_recorded','primary_source_bound','primary_known_budget_stop',
      'primary_source_issue_recorded','primary_initial_source_issues_recorded','primary_source_revised','primary_source_withholding_lost_typed_credit','b_eligible_recorded',
      'b_source_bound','independent_candidate_selected','auxiliary_entered','auxiliary_non_switch_credit_lost','auxiliary_fatal_recorded',
      'generic_revised_flag_informational_only','unknown_usage_recorded','team_exact_None_typed_credit','a_exact_None_typed_credit')
    for name in flags:result[name]=sum(bool(r[name]) for r in records)
    available=[r for r in records if r['a_only_available']];n=len(records);known=sum(r['a_only_correct'] for r in available)
    result.update(a_only_correct_observed=known,a_only_unavailable=n-len(available),
      a_only_accuracy_all_firsts_if_complete=known/n if n and len(available)==n else None,
      a_only_fullcohort_accuracy_bounds=[known/n,(known+n-len(available))/n] if n else None,
      a_to_team_available_denominator=len(available),a_to_team_delta_correct_available=sum(r['correct']-r['a_only_correct'] for r in available))
    for key in ('a_only_disposition','primary_source_revision_effect_to_released_A','released_B_effect','auxiliary_stage_effect',
                'auxiliary_status','release_policy'):
        result[key]=dict(Counter(str(r[key]) for r in records))
    result['b_exact_agreement_counts']=dict(Counter('unavailable' if r['b_answer_agreement_exact'] is None else 'agree' if r['b_answer_agreement_exact'] else 'disagree' for r in records))
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
    require(proof.get('status')=='PASS' and proof.get('mode')=='completion'
        and proof.get('caller_asserted_owned_jobs_terminal') is True,'Completed terminal integrity PASS required')
    require(proof.get('expected_first_attempts')==600 and proof.get('restored_audit_records')==600
        and type(proof.get('unique_referenced_artifacts')) is int
        and 1<=proof['unique_referenced_artifacts']<=600
        and proof.get('scoring_performed') is False,'All600 unscored restored firsts required')
    require(proof.get('guard_sha256')==args.guard_sha256 and proof.get('pins_sha256')==args.guard_pins_sha256,
        'Completion proof does not bind supplied guard/pins')
    guard=module(guard_path,'v6_posthoc_reviewed_guard')
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
    require({'utils/answer_scorer.py','scripts/compare_runs.py','scripts/audit_store.py','manifests/question_families.csv','scripts/experiment_stats.py'}<=set(manifest['source_sha256']),
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
    proofs={}
    for row in rows['ordinary_multi_agent']:
        restored,_=guard.verify_restored(row,output,store)
        proofs[row['question_id']]=proof_flags(restored,guard)
    for path,digest in checked.items():require(sha(path)==digest,'Input changed before scorer import')
    return {'root':root,'record':record,'output':output,'guard':guard,'store':store,'proof':proof,
            'official':official,'manifest':manifest,'rows':rows,'proofs':proofs,'checked':checked,'labels':labels,'ids':expected}


def analyze(prepared,loader=module):
    root=prepared['root'];guard=prepared['guard'];official=prepared['official'];manifest=prepared['manifest']
    families={}
    with (root/'manifests/question_families.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['question_id'] in prepared['ids']:
                require(row['question_id'] not in families,'Duplicate family metadata');families[row['question_id']]=row
    require(set(families)==set(prepared['ids']),'Missing fixed300 family/template metadata')
    for path,digest in prepared['checked'].items():require(sha(path)==digest,'Input changed before scorer import')
    scorer=loader(root/'utils/answer_scorer.py','v6_posthoc_unchanged_typed_scorer')
    stats=loader(root/'scripts/experiment_stats.py','v6_posthoc_existing_compare_statistics')
    totals={};records=[];patients={}
    for arm,rows in prepared['rows'].items():
        correct=0
        for row in rows:
            qid=row['question_id'];family=families[qid]
            def score(answer):return scorer.score_answer(answer,row['true_answer'],row['question'],family['template'])
            if arm=='ordinary_multi_agent':
                restored,_=guard.verify_restored(row,prepared['output'],prepared['store'])
                item=row_record(restored,score,prepared['proofs'][qid]);item['family']=family['family'];records.append(item)
                patients[qid]=row['patient_fhir_id'];correct+=item['correct']
            else:correct+=not failed(row) and bool(score(row.get('agent_answer'))['typed_correct'])
        label=prepared['labels'][arm]
        require(correct==official['arms'][label]['accuracy']['all']['correct'],'Unchanged typed totals differ from official completed result')
        totals[arm]={'questions':len(rows),'correct':correct,'official_label':label}
    require(len(records)==COUNT,'All300 team firsts must remain in mechanism denominator')
    available=sorted((r for r in records if r['a_only_available']),key=lambda r:r['question_id'])
    paired=stats.paired_difference([int(r['correct']) for r in available],[int(r['a_only_correct']) for r in available],
      [patients[r['question_id']] for r in available],n_boot=10000,seed=5106) if available else None
    for path,digest in prepared['checked'].items():require(sha(path)==digest,'Input changed during analysis')
    report={'experiment':EXPERIMENT,'protocol':PROTOCOL,'schema_version':1,'summary':summarize(records),
      'scope':'Posthoc mechanisms on all completed fixed300 adaptive development first attempts; no protected/full QA parsed.',
      'official_totals_reconciled':totals,'team_minus_recorded_A_only_paired_available':paired,
      'paired_scope':'Secondary posthoc question-weighted patient-cluster comparison on recorded available A outcomes; missing A never filled with null/zero. Full300 only if all A outcomes available.',
      'completion_status':prepared['proof']['status'],'source_manifest_sha256':prepared['proof']['manifest_sha256'],
      'scorer_sha256':manifest['source_sha256']['utils/answer_scorer.py'],'compare_source_sha256':manifest['source_sha256']['scripts/compare_runs.py'],
      'statistics_source_sha256':manifest['source_sha256']['scripts/experiment_stats.py'],'helper_sha256':sha(__file__),
      'limits':['A-only is the actual recorded175k/27 natural Single primary, not a hypothetical200k/30 control or a new model run.',
        'A primary fatal outcome is wrong; an ineligible A inherits its same row error; a source-bound eligible A remains counterfactually scoreable if later auxiliary accounting failed.',
        'Missing primary/revision/common-output metadata is unavailable. No missing draft/artifact is converted into typed null credit.',
        'Primary source-revision effects use initial draft -> actual released A, including source-gate withholding; generic revised is informational and not an independent repair count.',
        'Released-B effects count actual selected existing B only. Eligible but unselected B and fatal/non-switch losses are separate.',
        'Source validity and successful execution do not prove semantic correctness, complete search, or independent evidence; A and B share the frozen workspace.',
        'Typed null credit follows unchanged scorer conventions; it does not certify a justified search/abstention. Unknown usage is not zero billing.',
        'All600 audits restore and source/raw/official SHA checks pass before scorer import. Saved fixed300 inputs alone supply references; full CSV is only byte-hashed by the guard.',
        'No inference, new retrieval, program re-execution, acceptance threshold or favorable-view selection is introduced.']}
    return report,records


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
    parser.add_argument('--single-label',default='v6_single');parser.add_argument('--team-label',default='v6_team')
    report=run(parser.parse_args(argv));print(json.dumps(report['summary'],indent=2,allow_nan=False))

if __name__=='__main__':main()
