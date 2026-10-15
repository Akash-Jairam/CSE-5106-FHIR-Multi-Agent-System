"""Completed fixed300 V6 Team mechanisms; no inference or program execution."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True

PRIOR=Path('/private/tmp/fhir-v6-independent-verification-pf981jd9')
PRIOR_CODE='3b6689112b7d446d71d3bb0d33a60ff9ec63078884a79a1f4753f31ed9ca969e'
PRIOR_REPORT='64a19e856acc07677c1bdb9b4706b441dfa0a4f2decc9681299d85a3b49276fd'
ROOT_POSTHOC='870de0f130c4248a06a0b468beb55a6fe39ecf1d40f51b862df6556201ae098a'
ROOT_ROWS='387cd00bb01a80c940b86fb53dab54397b27c733af96ac861b23a6d5ec602168'
FIELDS=('answer','evidence','operation','contributing_ids','limitation')
RENDERED=('answer','evidence','limitation')
PROTOCOL='continuous-primary-independent-recompute-v6'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def require(value,message):
    if not value:raise ValueError(message)

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def same(a,b):return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)

def rendered(value):return {key:value[key] for key in RENDERED}

def effect(before,after):return 'helpful' if not before and after else 'harmful' if before and not after else 'neutral'


def verify(root,report):
    root=Path(root).resolve();report=Path(report).absolute()
    require(report.is_relative_to(Path('/private/tmp')) and not report.is_relative_to(root) and not report.exists()
      and not any(p.is_symlink() for p in (report,*report.parents)),'Fresh private report required')
    require(sha(PRIOR/'verify_development.py')==PRIOR_CODE and sha(PRIOR/'independent_verification.json')==PRIOR_REPORT,'Prior official verification changed')
    U=module(PRIOR/'verify_development.py','immutable_prior_V6_verification_utilities')
    prior=U.load(PRIOR/'independent_verification.json');record=root/'benchmarks/design-v6-20261015'
    output=root/'output/experiments/gpt-6-luna/independent_recomputation_v6_300q'
    checks={}
    def pin(path,digest):require(sha(path)==digest,'SHA differs: '+str(path));checks[path]=digest
    pin(record/'completion_protocol_audit.json',prior['external_completion_report_sha256'])
    pin(record/'results.json',prior['external_official_report_sha256'])
    pin(record/'v6_guard.py','49ec8be98372cbeaf9c9ab917d02fdf0bbbdce06471a8a0745a02aae9c485c18')
    pin(record/'reviewed_v4_guard.py','98d272f8ed8e42914b5ed1776908cf6e25e209e5726d5386b4c168a270a499e8')
    pin(record/'diagnostics/mechanisms.json',ROOT_POSTHOC);pin(record/'diagnostics/mechanisms_rows.json',ROOT_ROWS)
    completion=U.load(record/'completion_protocol_audit.json');official=U.load(record/'results.json')
    require(completion['status']=='PASS' and completion['mode']=='completion' and completion['restored_audit_records']==600
      and completion['caller_asserted_owned_jobs_terminal'] is True,'Actual terminal deep600 PASS required')
    manifest=U.load(record/'manifest.json')
    pin(root/'utils/answer_scorer.py',prior['source_sha256']['utils/answer_scorer.py'])
    pin(root/'scripts/audit_store.py',manifest['source_sha256']['scripts/audit_store.py'])
    for name in ('agent/continuous_primary.py','agent/independent_recompute_agent.py','agent/independent_recompute_aux.py','agent/base_agent.py','agent/single_agent.py'):
        pin(root/name,manifest['source_sha256'][name])
    raw=output/'ordinary_multi_agent.json';pin(raw,prior['raw_sha256']['v6_team'])
    pin(root/'manifests/split.json',manifest['split_sha256'])
    ids=U.load(root/'manifests/split.json')['splits']['held_out']['question_ids'];require(len(ids)==len(set(ids))==300,'Fixed300 differs')
    rootposthoc=U.load(record/'diagnostics/mechanisms.json');rootrows=U.load(record/'diagnostics/mechanisms_rows.json')
    require(len(rootrows)==300 and {r['question_id'] for r in rootrows}==set(ids),'Root mechanisms cohort differs')
    rootrows={r['question_id']:r for r in rootrows}
    guard=module(record/'v6_guard.py','V6_actual_original_mechanical_bindings')
    store=module(root/'scripts/audit_store.py','V6_actual_owned_audit_restore')
    scorer=module(root/'utils/answer_scorer.py','V6_unchanged_typed_scorer')
    templates={}
    import csv
    pin(root/'manifests/question_families.csv',manifest['source_sha256']['manifests/question_families.csv'])
    with (root/'manifests/question_families.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['question_id'] in ids:templates[row['question_id']]=row['template']
    require(set(templates)==set(ids),'Missing fixed300 template metadata')
    seen=[];facts=[];joint=Counter();disagreements=Counter();Bswitch=Counter();source_revision=Counter();source_kinds=Counter()
    source_withheld=[];switched_cases=[];aux_losses=[];restored_count=0;Aavailable=0
    compared_fields=['correct','a_only_available','a_only_correct','a_only_error','a_only_disposition',
      'primary_fatal_recorded','primary_source_bound','primary_source_revised','primary_initial_draft_correct',
      'primary_prewithholding_candidate_correct','primary_source_revision_effect_to_released_A',
      'primary_source_withholding_lost_typed_credit','b_source_bound','b_typed_correct','b_answer_agreement_exact',
      'independent_candidate_selected','released_B_effect','auxiliary_stage_effect','auxiliary_non_switch_credit_lost']
    for compact in U.iter_rows(raw):
        qid=compact['question_id'];seen.append(qid);require(not compact.get('attempt_history'),'Retried outcome forbidden')
        full,_=guard.verify_restored(compact,output,store);restored_count+=1
        rev=full.get('revision');require(isinstance(rev,dict) and rev.get('protocol')==PROTOCOL,'Missing actual primary revision')
        diag=rev.get('primary_diagnostics');A=rev.get('primary')
        require(isinstance(diag,dict) and isinstance(A,dict) and set(A)==set(RENDERED),'Missing primary outcome; never fabricate null')
        previous=rev.get('primary_revision') or {};aux=rev.get('recompute') or {};workspace=full.get('retrieved_fhir_resources') or {};trace=full.get('trace') or []
        def typed(value):return bool(scorer.score_answer(value,full['true_answer'],full['question'],templates[qid])['typed_correct'])
        final_correct=not U.failed(full) and typed(full.get('agent_answer'))
        fatal=diag.get('fatal') is True;eligible=diag.get('eligible') is True;boundA=False
        if fatal:
            require(U.failed(full),'Primary accounting failure lacks recorded terminal error')
            Aerror=True;Acorrect=False;disposition='recorded_primary_fatal_failure'
        else:
            require(diag.get('released_available') is True,'Primary common outcome unavailable; cannot score A-only')
            if not eligible:
                require(diag.get('eligible') is False and not aux and rev.get('release_policy')=='continuous_primary_preserved_auxiliary_ineligible','Ineligible primary must preserve actual row.error')
                require(same(A,rendered({'answer':full['agent_answer'],'evidence':full['evidence'],'limitation':full['limitation']})),'Ineligible primary did not remain released A')
                Aerror=U.failed(full);Acorrect=not Aerror and typed(A['answer']);disposition='actual_ineligible_or_prose_A_inherits_row_error'
            else:
                execution=rev.get('primary_artifact');require(isinstance(execution,dict) and execution.get('mode')=='final','Eligible primary missing actual execution')
                a5={key:execution[key] for key in FIELDS}
                require(same(rendered(a5),A) and execution['workspace_sha256']==guard.fingerprint(workspace),'Actual released A execution/workspace mismatch')
                require(previous.get('final_check_issues')==[] and same(rendered(previous['final']),A),'Eligible primary final original check did not release A')
                boundary=next((i for i,m in enumerate(trace) if isinstance(m,dict) and m.get('phase') in {'independent_generation','independent_recompute','independent_selection'}),len(trace))
                require(guard.validate_primary(trace[:boundary],a5,workspace)==execution['code'],'Actual original Single final/check code mismatch')
                checksA=[json.loads(m['content']) for m in trace[:boundary] if isinstance(m,dict) and m.get('role')=='tool' and m.get('tool_call_id') in {'automatic_field_check_1','automatic_field_check_2'}]
                guard._source_summary(a5,checksA[-1],workspace);boundA=True
                Aerror=False;Acorrect=typed(A['answer']);disposition='actual_eligible_A_before_auxiliary'
        Aavailable+=1;source_kinds[disposition]+=1
        previous_revised=rev.get('primary_source_revised') is True
        require(previous_revised==bool(previous.get('revised')),'Primary source-revision ancestry differs')
        draft=previous.get('draft');draft_correct=typed(draft['answer']) if isinstance(draft,dict) and set(RENDERED)<=set(draft) else None
        computed=previous.get('final');computed_correct=typed(computed['answer']) if isinstance(computed,dict) and set(RENDERED)<=set(computed) else None
        source_effect='not_applied'
        if previous_revised:
            require(draft_correct is not None,'Completed primary source revision lacks actual initial draft')
            source_effect=effect(draft_correct,Acorrect);source_revision[source_effect]+=1
        withheld=bool(not Aerror and computed_correct is True and not Acorrect and previous.get('final_check_issues'))
        if withheld:source_withheld.append({'question_id':qid,'computed_candidate_correct':True,'released_A_correct':False,'actual_source_issues_recorded':True,'is_primary_source_revision':previous_revised})
        B=aux.get('B');boundB=False;Bcorrect=None;agreement=None
        if aux.get('B_eligible') is True:
            require(boundA and isinstance(B,dict) and set(B)==set(FIELDS),'B eligible without actual valid A')
            executed=aux.get('B_execution');require(isinstance(executed,dict) and executed.get('mode')=='final' and 'error' not in executed and 'error_type' not in executed
              and same(guard.canonical_artifact(executed),B),'B not bound to actual successful final')
            calls={};replies={}
            for message in trace:
                if not isinstance(message,dict):continue
                if message.get('role')=='assistant' and message.get('phase') in {'independent_generation','independent_recompute'}:
                    for call in message.get('tool_calls') or []:calls[call['id']]=(call['function']['name'],json.loads(call['function']['arguments']))
                if message.get('role')=='tool' and message.get('phase')=='independent_recompute':replies[message['tool_call_id']]=json.loads(message['content'])
            native=[key for key,(name,args) in calls.items() if name=='execute_python_code']
            require(len(native)==1 and calls[native[0]][1].get('mode')=='final' and same(replies[native[0]],executed),'B actual native reply mismatch')
            require(same(calls.get('v6-independent-source-check'),('check_fields',{'citations':B['evidence']}))
              and same(replies.get('v6-independent-source-check'),aux.get('B_source_check')),'B original check trace mismatch')
            guard._source_summary(B,aux['B_source_check'],workspace);boundB=True;Bcorrect=typed(B['answer']);agreement=same(A['answer'],B['answer'])
            require(agreement is aux.get('answer_agreement'),'Exact answer-agreement recording mismatch')
            joint[f'A{int(Acorrect)}_B{int(Bcorrect)}']+=1
            if not agreement:disagreements[f'A{int(Acorrect)}_B{int(Bcorrect)}']+=1
        switched=rev.get('release_policy')=='selected_executed_independent_candidate'
        require(switched is (rev.get('independent_candidate_selected') is True),'B switch policy/flag mismatch')
        B_effect='not_applied'
        if switched:
            require(boundB and not agreement and aux.get('selected')=='B' and aux.get('selection',{}).get('choice')=='B'
              and same(rev.get('final'),B) and same(rendered(B),{'answer':full['agent_answer'],'evidence':full['evidence'],'limitation':full['limitation']}) and not U.failed(full),'Actual released selected B differs')
            selections=[m for m in trace if isinstance(m,dict) and m.get('phase')=='independent_selection' and m.get('role')=='assistant']
            require(len(selections)==1 and not selections[0].get('tool_calls') and same(json.loads(selections[0]['content']),aux['selection']),'Actual selector did not choose existing B')
            B_effect=effect(Acorrect,final_correct);Bswitch[B_effect]+=1
            switched_cases.append({'question_id':qid,'A_correct':Acorrect,'B_correct':Bcorrect,'final_correct':final_correct,'effect':B_effect})
        aux_entered=bool(aux);aux_effect=effect(Acorrect,final_correct) if aux_entered else 'not_applied'
        non_switch_loss=bool(aux_entered and not switched and Acorrect and not final_correct)
        if non_switch_loss:
            require(boundA and not Aerror and aux.get('fatal') is True and U.failed(full),'Non-switch A loss lacks actual later auxiliary fatal')
            aux_losses.append({'question_id':qid,'actual_A_correct_before_auxiliary':True,'A_error':False,'Team_error':True,'auxiliary_fatal':True})
        item={'question_id':qid,'correct':bool(final_correct),'a_only_available':True,'a_only_correct':bool(Acorrect),'a_only_error':bool(Aerror),
          'a_only_disposition':disposition,'primary_fatal_recorded':fatal,'primary_source_bound':boundA,'primary_source_revised':previous_revised,
          'primary_initial_draft_correct':draft_correct,'primary_prewithholding_candidate_correct':computed_correct,
          'primary_source_revision_effect_to_released_A':source_effect,'primary_source_withholding_lost_typed_credit':withheld,
          'b_source_bound':boundB,'b_typed_correct':Bcorrect,'b_answer_agreement_exact':agreement,
          'independent_candidate_selected':switched,'released_B_effect':B_effect,'auxiliary_stage_effect':aux_effect,'auxiliary_non_switch_credit_lost':non_switch_loss}
        require(all(same(item[key],rootrows[qid][key]) for key in compared_fields),'Independent row endpoint differs from root posthoc: '+qid)
        facts.append(item)
    require(seen==ids and restored_count==Aavailable==300,'Missing/duplicated owned Team first attempt')
    summary={'questions':300,'owned_Team_audits_restored_now':restored_count,'a_outcomes_reconstructible':Aavailable,
      'actual_released_A_correct':sum(r['a_only_correct'] for r in facts),'released_Team_correct':sum(r['correct'] for r in facts),
      'A_outcome_classes':dict(source_kinds),'primary_source_bound':sum(r['primary_source_bound'] for r in facts),
      'primary_fatal_wrong':sum(r['primary_fatal_recorded'] and not r['a_only_correct'] for r in facts),
      'source_bound_B_candidates':sum(joint.values()),'B_joint_typed_correctness':dict(joint),
      'A_correct_on_B_pool':joint['A1_B1']+joint['A1_B0'],'B_correct_on_B_pool':joint['A1_B1']+joint['A0_B1'],
      'A_wrong_B_correct':joint['A0_B1'],'strict_JSON_disagreements':sum(disagreements.values()),'disagreement_correctness':dict(disagreements),
      'actual_B_switches':sum(Bswitch.values()),'B_switch_effect':dict(Bswitch),'auxiliary_fatal_correct_A_losses':len(aux_losses),
      'primary_source_revisions':sum(source_revision.values()),'primary_source_revision_draft_to_released_A':dict(source_revision),
      'computed_correct_then_source_withheld_A':len(source_withheld),'all300_root_posthoc_row_endpoint_fields_match':True}
    require(summary['actual_released_A_correct']==rootposthoc['summary']['a_only_correct_observed']
      and summary['released_Team_correct']==official['arms']['v6_team']['accuracy']['all']['correct']==231,'Official/posthoc totals differ')
    require(summary['released_Team_correct']==summary['actual_released_A_correct']+Bswitch['helpful']-Bswitch['harmful']-len(aux_losses),'A to Team decomposition mismatch')
    for path,digest in checks.items():require(sha(path)==digest,'Evidence changed during mechanism verification')
    require(sha(PRIOR/'verify_development.py')==PRIOR_CODE and sha(PRIOR/'independent_verification.json')==PRIOR_REPORT,'Prior verification mutated')
    result={'status':'PASS_INDEPENDENT_ACTUAL_RELEASED_A_AND_B_MECHANISMS','summary':summary,
      'edge_cases':{'actual_B_switches':switched_cases,'auxiliary_fatal_valid_A_losses':aux_losses,'computed_correct_but_withheld_not_released_A':source_withheld},
      'source_proof':{'completion_report_sha256':prior['external_completion_report_sha256'],'official_report_sha256':prior['external_official_report_sha256'],
        'raw_Team_sha256':prior['raw_sha256']['v6_team'],'root_mechanisms_sha256':ROOT_POSTHOC,'root_mechanism_rows_sha256':ROOT_ROWS,
        'prior_independent_report_sha256':PRIOR_REPORT,'prior_independent_code_sha256':PRIOR_CODE,
        'unchanged_scorer_sha256':prior['source_sha256']['utils/answer_scorer.py'],'reproduction_sha256':sha(Path(__file__)),
        'all300_compared_field_inventory':compared_fields},
      'limits':['A-only is the actual recorded175k/27 natural Single primary, not a hypothetical200k/30 control or causal primary-cap ablation.',
        'Primary fatal is wrong; ineligible A keeps actual row.error; eligible A is scored before a later auxiliary fatal outcome. Missing A would reject reconstruction, never fabricate a null credit.',
        'Primary source revision uses actual draft to Base-normalized released A. primary_revision.final is a pre-withholding candidate and never substitutes for released A.',
        'Typed-correct withheld candidates still have recorded source issues. This does not establish validator false positives or source support.',
        'The37 source-bound B pool is conditional on eligibility, coverage and execution. Zero observed helpful alternatives there is not a statement about all263 unavailable B cases.',
        'Strict JSON answer difference and unchanged typed correctness are distinct; semantic repair counts do not come from generic revision.revised.',
        'Actual final executions/source replies were bound to traces and immutable workspace, not re-executed/refetched. Source checks do not establish semantic correctness or search completeness.',
        'All300 first attempts/errors retained; no full comparison/deep600 guard rerun or QA outside completed fixed300. Protected1397 and fullCSV contents were not read.']}
    with report.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args();verify(args.root,args.report)
