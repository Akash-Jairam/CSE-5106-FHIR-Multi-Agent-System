"""Post-terminal fixed300 V7 mechanisms and prospective all300 actual-A gate only.

No inference imports, program replay or full-QA parsing. References come solely
from the completed saved300 inputs, after externally pinned gates/owned restores.
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
EXPERIMENT='design-v7-20261015'
PROTOCOL='full-single-residual-trajectories-v7'
ARMS={'ordinary_single_agent':'single_agent','ordinary_multi_agent':'multi_agent_v7'}
PRIOR_BEST_SINGLE=239
ACCEPTANCE='Team >239 AND Team >fresh contemporary Single AND Team >actual recorded full-cap A-only; all300 first attempts/errors, no replacements/retries. Missing A reconstruction makes the last gate unavailable and prevents selection.'
A_ONLY_GATE='Strict positive net auxiliary contribution; actual normalized released A/error and primary fatal state, never computed withheld or fabricated null. All300 reconstructible outcomes required.'
HEX=re.compile(r'^[a-f0-9]{64}$')
RENDERED=('answer','evidence','limitation')


def require(value,reason):
    if not value:raise ValueError(reason)

def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data:=stream.read(1024*1024):result.update(data)
    return result.hexdigest()

def load(path):
    def pairs(items):
        value={}
        for key,item in items:
            require(key not in value,'Duplicate JSON key');value[key]=item
        return value
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))

def same(a,b):return json.dumps(a,sort_keys=True,ensure_ascii=True,allow_nan=False)==json.dumps(b,sort_keys=True,ensure_ascii=True,allow_nan=False)

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

def failed(value):return value.get('error') is not None and str(value['error']).strip() not in ('','nan')

def transition(before,after,applied=True):
    if not applied:return 'not_applied'
    if before is None or after is None:return 'unavailable'
    return 'helpful' if not before and after else 'harmful' if before and not after else 'neutral'

def present(value):return isinstance(value,dict) and all(k in value for k in (*RENDERED,'error'))

def candidate(value):return isinstance(value,dict) and all(k in value for k in RENDERED)


def compact_verified(row):
    """ONLY call after frozen guard validation; drop workspaces/code before scoring."""
    revision=row.get('revision') if isinstance(row.get('revision'),dict) else {}
    snapshots=revision.get('messages') or []
    result={'status':revision.get('status'),'selected':revision.get('selected'),
        'fatal':revision.get('fatal') is True,'protocol':revision.get('protocol'),
        'selection':revision.get('selection'),'phases':revision.get('phases') or [],'snapshots':{}}
    trace=row.get('trace') or []
    for snapshot in snapshots:
        if not isinstance(snapshot,dict) or snapshot.get('kind')!='candidate_snapshot':continue
        label=snapshot.get('candidate');released=snapshot.get('released');diag=snapshot.get('diagnostics') or {}
        previous=released.get('revision') if isinstance(released,dict) else None
        previous=previous if isinstance(previous,dict) else {}
        local=[m for m in trace if m.get('phase')==snapshot.get('phase')]
        checks=[m for m in local if m.get('role')=='tool' and m.get('name')=='check_fields'
                and m.get('tool_call_id') in ('automatic_field_check_1','automatic_field_check_2')]
        # Candidate values remain in memory only for unchanged typed scoring.
        value={'released_present':present(released),'released':{k:released[k] for k in (*RENDERED,'error')} if present(released) else None,
            'fatal':diag.get('fatal') is True,'source_bound':snapshot.get('artifact') is not None,
            'source_revision':{'draft':previous.get('draft'),'computed_final':previous.get('final'),
                'initial_issues':bool(previous.get('check_issues')),'final_issues':bool(previous.get('final_check_issues')),
                'revised':previous.get('revised') is True,'automatic_checks_observed':len(checks),
                'second_automatic_check_observed':any(m.get('tool_call_id')=='automatic_field_check_2' for m in checks)},
            'budget_stop':diag.get('budget_stop'),'tokens':diag.get('tokens'),'tool_calls':diag.get('tool_calls'),
            'natural_tool_calls':diag.get('natural_tool_calls'),'model_calls':diag.get('model_calls'),
            'token_cap':diag.get('token_cap'),'tool_cap':diag.get('tool_cap'),'sdk_cap':diag.get('sdk_cap'),
            'initialization_charge':(diag.get('initialization') or {}).get('charged',0),
            'usage_known':diag.get('usage_known'),'unknown_invocations':sum(c.get('usage_available') is not True
                for c in (snapshot.get('usage') or {}).get('per_call') or []),
            'evidence_count':len((snapshot.get('artifact') or {}).get('evidence') or [])}
        result['snapshots'][label]=value
    return result


def phase_record(value,score,fatal_override=False):
    available=isinstance(value,dict) and value.get('released_present') is True
    result={'available':available,'correct':None,'typed_ignoring_error':None,'error':None,'source_bound':False,
        'source_revised':False,'source_revision_effect':'unavailable','source_withholding_lost_credit':False}
    if not available:return result
    released=value['released'];fatal=value['fatal'] or fatal_override;terminal=fatal or failed(released)
    typed=None if fatal else bool(score(released['answer'])['typed_correct'])
    result.update(correct=bool(not terminal and typed),typed_ignoring_error=typed,error=terminal,
        source_bound=value['source_bound'],fatal=fatal,exact_null=released['answer'] is None,
        known_budget_stop=bool(value['budget_stop']),unknown_invocations=value['unknown_invocations'],
        tokens=value['tokens'],tool_calls=value['tool_calls'],model_calls=value['model_calls'],
        token_cap=value['token_cap'],tool_cap=value['tool_cap'],sdk_cap=value['sdk_cap'],
        initialization_charge=value['initialization_charge'])
    source=value['source_revision'];draft=source['draft'];computed=source['computed_final']
    draft_correct=bool(score(draft['answer'])['typed_correct']) if candidate(draft) else None
    computed_correct=bool(score(computed['answer'])['typed_correct']) if candidate(computed) else None
    result.update(source_revised=source['revised'],source_feedback_requested=source['initial_issues'],
        source_final_issues=source['final_issues'],source_automatic_checks=source['automatic_checks_observed'],
        source_second_check=source['second_automatic_check_observed'],draft_correct=draft_correct,
        computed_final_typed_correct=computed_correct,
        source_revision_effect=transition(draft_correct,result['correct'],source['revised']),
        source_withholding_lost_credit=bool(not terminal and computed_correct is True and not result['correct'] and source['final_issues']))
    return result


def row_record(row,score,facts):
    typed=score(row.get('agent_answer'));correct=bool(not failed(row) and typed['typed_correct'])
    a_valid=facts['protocol']==PROTOCOL and facts['status']!='fatal_mutation'
    a=phase_record(facts['snapshots'].get('A') if a_valid else None,score,facts['fatal'] and facts['status']=='fatal_primary')
    b=phase_record(facts['snapshots'].get('B'),score,facts['fatal'] and facts['status']=='fatal_alternative')
    exact=None
    if a['available'] and b['available'] and b['source_bound']:
        exact=same(facts['snapshots']['A']['released']['answer'],facts['snapshots']['B']['released']['answer'])
    selected_B=facts['selected']=='B' and facts['status']=='selected_B' and not facts['fatal']
    aux_entered='B' in facts['snapshots'] or any(p.get('phase') in ('natural_alternative','selection') for p in facts['phases'])
    aux_fatal=(facts['fatal'] and facts['status'] in ('fatal_alternative','fatal_selection')) or any(
        p.get('fatal') is True for p in facts['phases'] if p.get('phase')!='natural_primary')
    return {'question_id':row.get('question_id'),'correct':correct,'terminal_error':failed(row),
        'reference_answer_kind':typed.get('answer_type'),'A':a,'B':b,'status':facts['status'],
        'selected_B':selected_B,'selector_choice':(facts.get('selection') or {}).get('choice'),
        'exact_JSON_answer_agreement':exact,'released_B_effect':transition(a['correct'],correct,selected_B),
        'all_post_A_effect':transition(a['correct'],correct,a['available']),
        'auxiliary_entered':aux_entered,'auxiliary_fatal':aux_fatal,
        'auxiliary_fatal_correct_A_loss':bool(aux_fatal and a['available'] and a['correct'] and not correct),
        'team_exact_null_credit':row.get('agent_answer') is None and correct,
        'A_exact_null_credit':bool(a['available'] and a.get('exact_null') and a['correct']),
        'A_gate_unavailable_reason':None if a['available'] else 'missing_actual_A_or_frozen_mutation',
        'phase_budget_stops':{p.get('phase'):p.get('budget_stop') for p in facts['phases'] if p.get('budget_stop')},
        'unknown_usage_invocations':sum(c.get('usage_available') is not True for c in (row.get('tokens') or {}).get('per_call') or [])}


def summarize(records,single_correct,team_correct):
    require(len(records)==COUNT,'All300 mechanism rows required')
    require(sum(r['correct'] for r in records)==team_correct,'Team mechanism totals differ from official')
    available=[r for r in records if r['A']['available']];known=sum(r['A']['correct'] for r in available)
    complete=len(available)==COUNT
    gate={'prior_best_single':PRIOR_BEST_SINGLE,'fresh_single_correct':single_correct,'team_correct':team_correct,
        'A_only_available':len(available),'A_only_unavailable':COUNT-len(available),
        'A_only_correct_all300':known if complete else None,'A_only_correct_observed':known,
        'A_only_full300_correct_bounds':[known,known+COUNT-len(available)],
        'team_beats_prior_best':team_correct>PRIOR_BEST_SINGLE,'team_beats_fresh_single':team_correct>single_correct,
        'team_beats_actual_A_only':team_correct>known if complete else None,
        'passed':complete and team_correct>max(PRIOR_BEST_SINGLE,single_correct,known),
        'status':'evaluated_all300' if complete else 'unavailable_missing_A_prevents_selection',
        'rule':'Strict integer Team>239 AND Team>freshSingle AND Team>actual full-cap recorded A-only, all300 first attempts.'}
    result={'questions':COUNT,'development_acceptance':gate,
        'all_A_available_denominator':len(available),'all_A_to_team_net_correct':sum(r['correct']-r['A']['correct'] for r in available),
        'A_B_source_bound_joint_correctness':dict(Counter(('A_right' if r['A']['correct'] else 'A_wrong')+'_'+('B_right' if r['B']['correct'] else 'B_wrong')
            for r in records if r['A']['available'] and r['B']['available'] and r['B']['source_bound'])),
        'A_B_source_bound_joint_denominator':sum(r['A']['available'] and r['B']['available'] and r['B']['source_bound'] for r in records)}
    for key in ('terminal_error','selected_B','auxiliary_entered','auxiliary_fatal','auxiliary_fatal_correct_A_loss','team_exact_null_credit','A_exact_null_credit'):
        result[key]=sum(bool(r[key]) for r in records)
    for key in ('status','selector_choice','released_B_effect','all_post_A_effect','exact_JSON_answer_agreement'):
        result[key]=dict(Counter(str(r[key]) for r in records))
    for label in ('A','B'):
        result[label]={key:sum(bool(r[label].get(key)) for r in records) for key in ('available','correct','error','fatal','source_bound','source_revised','source_feedback_requested','source_second_check','source_withholding_lost_credit','known_budget_stop')}
        result[label]['source_revision_effect']=dict(Counter(r[label]['source_revision_effect'] for r in records))
        result[label]['unknown_invocations']=sum(r[label].get('unknown_invocations',0) for r in records)
    result['unknown_usage_invocations']=sum(r['unknown_usage_invocations'] for r in records)
    return result


def raw_inventory(root):
    return {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() or p.is_symlink()}


def verify_all(args):
    require(args.confirm_both_owned_jobs_terminal is True,'Explicit actual-owned-jobs terminal assertion required')
    root=Path(args.root).resolve();record=root/'benchmarks'/EXPERIMENT;checked={}
    def pinned(path,digest):
        path=Path(path).absolute()
        require(isinstance(digest,str) and HEX.fullmatch(digest),'External SHA256 required')
        require(path.is_file() and not path.is_symlink() and sha(path)==digest,'Pinned bytes differ: '+str(path))
        checked[path]=digest;return path
    pinned(Path(__file__),args.helper_sha256)
    proof_path=pinned(args.completion_report,args.completion_report_sha256)
    official_path=pinned(args.official_report,args.official_report_sha256)
    guard_path=pinned(args.guard_module,args.guard_sha256);pins_path=pinned(args.guard_pins,args.guard_pins_sha256)
    attestation=pinned(args.owned_jobs_attestation,args.owned_jobs_attestation_sha256)
    proof=load(proof_path)
    require(proof.get('status') in {'PASS','PASS_WITH_JOURNAL_LIMITATIONS'} and proof.get('mode')=='completion'
        and proof.get('protocol')==PROTOCOL and proof.get('caller_asserted_owned_jobs_terminal') is True,'Terminal V7 completion PASS required')
    require(same(proof.get('expected_first_attempts'),600) and same(proof.get('restored_audit_records'),600)
        and proof.get('scoring_performed') is False and proof.get('guard_sha256')==args.guard_sha256
        and proof.get('pins_sha256')==args.guard_pins_sha256 and proof.get('owned_jobs_attestation_sha256')==args.owned_jobs_attestation_sha256,
        'All600 restored/owned/source bindings required')
    official=load(official_path);guard=module(guard_path,'v7_posthoc_reviewed_guard')
    guard.verify_owned_attestation(attestation,args.owned_jobs_attestation_sha256)
    pins,manifest=guard.verify_pins(record,pins_path,args.guard_pins_sha256)
    prospective=load(record/'prospective_protocol.json')
    require(prospective.get('acceptance')==ACCEPTANCE and prospective.get('recorded_A_only_gate')==A_ONLY_GATE
        and same(prospective.get('minimum_team_correct'),240),'Prospective three-condition acceptance differs')
    raw=Path(manifest['output_directory']).resolve()
    require(raw.is_relative_to(root/'output'),'Raw output outside owned project output')
    pins,manifest,adapter,binding=guard.frozen_provenance(root,record,raw,pins_path,args.guard_pins_sha256)
    require(guard.actual_source_inventory(root)==set(manifest['source_sha256']),'Frozen inventory differs')
    require(same(proof.get('provider_binding'),binding),'Provider binding differs')
    expected_proof={'manifest_sha256':pins['record_metadata_sha256']['manifest.json'],
        'source_inventory_sha256':pins['source_inventory_sha256'],'frozen_source_files_verified':len(manifest['source_sha256']),
        'runtime_adapter_files_verified':4,'source_archive_sha256':manifest['source_archive_sha256'],
        'design_sha256':manifest['design_sha256'],'dataset_sha256':manifest['dataset_sha256'],
        'split_sha256':manifest['split_sha256'],'code_hash':guard.B.code_hash(root),'reviewed_base_sha256':guard.BASE_SHA}
    require(all(same(proof.get(k),v) for k,v in expected_proof.items()),'Completion source/policy binding differs')
    pinned(guard.BASE_PATH,guard.BASE_SHA)
    for name,digest in manifest['source_sha256'].items():pinned(guard.public_path(root,name),digest)
    for name,digest in pins['record_metadata_sha256'].items():pinned(record/name,digest)
    for path,digest in ((record/'source.zip',manifest['source_archive_sha256']),
        (record/'design.txt',manifest['design_sha256']),(record/'runtime_adapter_sources.zip',adapter['archive_sha256'])):pinned(path,digest)
    for name,digest in adapter['source_sha256'].items():pinned(guard.public_path(root,name),digest)
    raw_files=proof.get('raw_files');require(isinstance(raw_files,dict) and raw_files,'Completion raw byte inventory absent')
    for relative,value in raw_files.items():
        path=guard.B.owned_file(raw,relative)
        require(isinstance(value,dict) and set(value)=={'sha256','bytes'} and type(value['bytes']) is int
            and value['bytes']>=0 and path.stat().st_size==value['bytes'],'Raw size envelope differs')
        pinned(path,value['sha256'])
    require(raw_inventory(raw)==set(raw_files),'Raw inventory changed')
    expected=load(root/'manifests/split.json')['splits']['held_out']['question_ids']
    require(len(expected)==len(set(expected))==COUNT,'Fixed300 identity absent')
    require(official.get('cohort')=='held_out' and same(official.get('questions'),COUNT)
        and official.get('dataset_sha256')==manifest['dataset_sha256']
        and official.get('families_sha256')==manifest['source_sha256']['manifests/question_families.csv']
        and official.get('question_ids_sha256')==hashlib.sha256(json.dumps(sorted(expected)).encode()).hexdigest(),'Official cohort/source differs')
    labels={'ordinary_single_agent':args.single_label,'ordinary_multi_agent':args.team_label}
    require(all(isinstance(v,str) and v for v in labels.values()) and len(set(labels.values()))==2,'Official current labels differ')
    rows={};facts={};store=guard.B.load_public_store(root);selector=guard.read_selector_prompt(root)
    for arm,strategy in ARMS.items():
        path=raw/(arm+'.json');saved=load(path);identity=load(raw/(arm+'.json.run.json'));summary=proof['arms'][arm]
        require(isinstance(saved,list) and len(saved)==COUNT and [r.get('question_id') for r in saved]==expected
            and all(guard.B.complete(r) and not r.get('attempt_history') for r in saved),'All300 ordered first attempts required')
        require(same(summary.get('saved_first_attempts'),COUNT) and same(summary.get('verified_owned_gzip_audits'),COUNT)
            and summary.get('result_sha256')==sha(path) and summary.get('identity_sha256')==sha(raw/(arm+'.json.run.json')),
            'Completion arm raw/identity/count differs')
        entry=(official.get('identities') or {}).get(labels[arm]) or {}
        require(identity.get('agent_strategy')==strategy and same(entry.get('identity'),identity)
            and entry.get('result_sha256')==sha(path) and Path(entry.get('result_path','')).resolve()==path.resolve(),
            'Official result raw/identity binding differs')
        require(sum(failed(r) for r in saved)==(official['arms'][labels[arm]]).get('terminal_errors')==summary.get('terminal_errors',0),
            'Official terminal-error denominator differs')
        for row in saved:
            ref=row.get('audit_artifacts') or {}
            require(ref.get('question_id')==row['question_id'] and ref.get('path') in raw_files,'Owned audit identity differs')
            if arm=='ordinary_multi_agent':
                restored,_=guard.verify_restored(row,raw,store)
                guard.v7_protocol_counts(row,restored,selector)
                facts[row['question_id']]=compact_verified(restored)
        rows[arm]=saved
    # Only pinned metadata templates for fixed300; never parse the full QA CSV.
    families={}
    with (root/'manifests/question_families.csv').open(newline='') as stream:
        for value in csv.DictReader(stream):
            if value['question_id'] in set(expected):
                require(value['question_id'] not in families,'Duplicate fixed300 family metadata');families[value['question_id']]=value
    require(set(families)==set(expected),'Fixed300 template metadata missing')
    require({'utils/answer_scorer.py','scripts/compare_runs.py'}<=set(manifest['source_sha256']),'Frozen scorer/comparison source missing')
    prepared={'root':root,'record':record,'raw':raw,'guard':guard,'proof':proof,'manifest':manifest,'official':official,
        'labels':labels,'rows':rows,'facts':facts,'families':families,'checked':checked,'raw_files':raw_files}
    recheck(prepared)
    return prepared


def recheck(prepared):
    for path,digest in prepared['checked'].items():require(sha(path)==digest,'Pinned input changed')
    require(raw_inventory(prepared['raw'])==set(prepared['raw_files']),'Raw inventory changed')
    require(prepared['guard'].actual_source_inventory(prepared['root'])==set(prepared['manifest']['source_sha256']),'Source inventory changed')


def analyze(prepared,loader=module):
    recheck(prepared)  # Every byte/source/restoration gate precedes the scorer import.
    scorer=loader(prepared['root']/'utils/answer_scorer.py','v7_posthoc_unchanged_typed_scorer')
    totals={};records=[]
    for arm,rows in prepared['rows'].items():
        correct=0
        for row in rows:
            family=prepared['families'][row['question_id']]
            def score(answer):return scorer.score_answer(answer,row['true_answer'],row['question'],family['template'])
            if arm=='ordinary_multi_agent':
                item=row_record(row,score,prepared['facts'][row['question_id']]);records.append(item);correct+=item['correct']
            else:correct+=not failed(row) and bool(score(row.get('agent_answer'))['typed_correct'])
        label=prepared['labels'][arm];official=prepared['official']['arms'][label]['accuracy']['all']
        require(same(official.get('n'),COUNT) and same(correct,official.get('correct')),'Unchanged typed totals disagree with official current arm')
        totals[arm]={'correct':correct,'n':COUNT,'official_label':label}
    summary=summarize(records,totals['ordinary_single_agent']['correct'],totals['ordinary_multi_agent']['correct'])
    recheck(prepared)
    return {'experiment':EXPERIMENT,'protocol':PROTOCOL,'schema_version':1,'summary':summary,
        'official_totals_reconciled':totals,'completion_status':prepared['proof']['status'],
        'scorer_sha256':prepared['manifest']['source_sha256']['utils/answer_scorer.py'],
        'compare_source_sha256':prepared['manifest']['source_sha256']['scripts/compare_runs.py'],
        'scope':'All completed fixed300 first attempts; saved references only; no full-QA parsing or confirmation scoring.',
        'limits':['A-only is actual full-cap natural Single with its own normalized release/error/phasefatal state, before auxiliary failures.',
            'Missing A remains unavailable and blocks acceptance; no null substitution or excluded denominator.',
            'Source feedback effects are draft-to-actual-phase-release descriptions, including source withholding; not generic outer revised attribution.',
            'Actual selected-B effects, eligible complementarity, strictJSON agreement, typed correctness and fatal auxiliary losses are distinct.',
            'Source/check/selector claims are fallible and do not prove semantic correctness or complete retrieval.',
            'The adaptive three-condition development gate adds no CI/significance/generalization claim; full confirmation protocol is unchanged.',
            'Externally pinned completion proof supplies deep600 restoration; this helper restores only necessary Team300 audits, with all raw bytes pinned.',
            'Terminality/pin timing remain root attestations; byte rechecks are not atomic OS attestation.']},records


def destination(args):
    path=Path(args.output)
    require(path.is_absolute() and '..' not in path.parts,'Absolute fresh diagnostics path required')
    require(not any(x.is_symlink() for x in (path,*path.parents)),'Symlink diagnostics ancestor forbidden')
    root=Path(args.root).resolve();formal=root/'benchmarks'/EXPERIMENT/'diagnostics';resolved=path.resolve()
    require(resolved==formal or (resolved.is_relative_to(Path('/private/tmp').resolve()) and not resolved.is_relative_to(root)),
        'Diagnostics must be formal metadata or external private temporary path')
    require(not resolved.exists(),'Fresh diagnostics destination required')
    return resolved


def run(args):
    output=destination(args);prepared=verify_all(args);report,records=analyze(prepared)
    report['external_bindings']={k:getattr(args,k) for k in ('helper_sha256','completion_report_sha256','official_report_sha256',
        'guard_sha256','guard_pins_sha256','owned_jobs_attestation_sha256')}
    report_payload=json.dumps(report,indent=2,allow_nan=False)+'\n';row_payload=json.dumps(records,indent=2,allow_nan=False)+'\n'
    recheck(prepared);require(destination(args)==output,'Diagnostics destination changed')
    output.mkdir(parents=True)
    (output/'mechanisms.json').write_text(report_payload);(output/'mechanisms_rows.json').write_text(row_payload)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for option in ('root','helper-sha256','completion-report','completion-report-sha256','official-report','official-report-sha256',
        'guard-module','guard-sha256','guard-pins','guard-pins-sha256','owned-jobs-attestation','owned-jobs-attestation-sha256','output'):
        p.add_argument('--'+option,required=True)
    p.add_argument('--confirm-both-owned-jobs-terminal',action='store_true',required=True)
    p.add_argument('--single-label',default='v7_single');p.add_argument('--team-label',default='v7_team')
    report=run(p.parse_args(argv));print(json.dumps(report['summary'],indent=2,allow_nan=False))

if __name__=='__main__':main()
