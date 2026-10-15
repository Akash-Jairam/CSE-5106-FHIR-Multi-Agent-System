"""Independent completed fixed300 A/B mechanism verification; no inference or full QA.

Original scorer is loaded standalone only after literal deep-completion PASS,
raw300 byte pins and exact scorer/store source hashes pass. No code is replayed.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import csv,hashlib,importlib.util,json

ROOT=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
RECORD=ROOT/'benchmarks/design-v7-20261015'
OUT=ROOT/'output/experiments/gpt-6-luna/residual_trajectories_v7_300q'
HERE=Path(__file__).parent
COMPLETION_SHA='a27e383569198753c326b41e4fc7537387c66ee2593d47f0d97dbccb48b38202'
SCORER_SHA='ca02e5a8758f5bcef3eee08ab70f69fddf23917d267df6301e83275f337f3964'
OFFICIAL_SHA='2ef2ae8342ade4d77352f424fda9ff5087eefac718eb06dd07ceedf8bb27fe0e'
PROTOCOL='full-single-residual-trajectories-v7'
FIELDS=('answer','evidence','limitation','error')
BINDING_REASONS={'primary returned an error','no surviving explicit final artifact','final workspace differs from execution',
 'executed final program is missing','executed final schema is invalid','released answer does not bind a clean executed final',
 'actual final execution or automatic source-check trace is missing','automatic source-check result does not bind the released artifact',
 'a later requested mutation was refused before dispatch','final execution/source binding is malformed or nonfinite',
 'phase accounting fatal','original initialization did not complete'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def enc(v):return json.dumps(v,sort_keys=True,ensure_ascii=True,allow_nan=False)
def fp(v):return hashlib.sha256(enc(v).encode()).hexdigest()
def same(a,b):return enc(a)==enc(b)
def load(p):return json.loads(Path(p).read_text())
def require(v,r):
    if not v:raise ValueError(r)
def module(p,n):
    spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def failed(v):return v.get('error') is not None and str(v['error']).strip() not in ('','nan')
def correct(score,value,error=False):return bool(not error and score(value)['typed_correct'])
def effect(a,b):return 'helpful' if not a and b else 'harmful' if a and not b else 'neutral'
def joint(a,b):return ('A_right' if a else 'A_wrong')+'_'+('B_right' if b else 'B_wrong')
def initial(trace):
    out=[]
    for message in trace:
        if message.get('role')=='assistant':break
        out.append({k:v for k,v in message.items() if k!='phase'})
    return out

def main():
    proof_path=RECORD/'completion_integrity.json';require(sha(proof_path)==COMPLETION_SHA,'Deep proof SHA differs');proof=load(proof_path)
    require(proof['status']=='PASS' and proof['mode']=='completion' and proof['protocol']==PROTOCOL
      and proof['expected_first_attempts']==600 and proof['restored_audit_records']==600
      and proof['caller_asserted_owned_jobs_terminal'] is True and proof['scoring_performed'] is False,'Actual terminal deep600 PASS required')
    official_path=RECORD/'comparison.json';require(sha(official_path)==OFFICIAL_SHA,'Official result SHA differs');official=load(official_path)
    manifest=load(RECORD/'manifest.json');store_path=ROOT/'scripts/audit_store.py'
    require(sha(ROOT/'utils/answer_scorer.py')==SCORER_SHA==manifest['source_sha256']['utils/answer_scorer.py'],'Original scorer SHA differs')
    require(sha(store_path)==manifest['source_sha256']['scripts/audit_store.py'],'Owned storage source differs')
    rows={};identities={};rawpins={}
    for arm in ('ordinary_single_agent','ordinary_multi_agent'):
        p=OUT/(arm+'.json');identity=OUT/(arm+'.json.run.json');expected=proof['arms'][arm]
        require(sha(p)==expected['result_sha256'] and sha(identity)==expected['identity_sha256'],'Completed arm bytes differ')
        values=load(p);require(isinstance(values,list) and len(values)==300 and all(not r.get('attempt_history') for r in values),'Exact first300 required')
        require(all((r.get('latency') is not None or failed(r)) for r in values),'Incomplete row')
        rows[arm]=values;identities[arm]=load(identity);rawpins[arm]={'result_sha256':sha(p),'identity_sha256':sha(identity)}
    srows=rows['ordinary_single_agent'];trows=rows['ordinary_multi_agent']
    require([r['question_id'] for r in srows]==[r['question_id'] for r in trows] and len({r['question_id'] for r in trows})==300,'Matched cohort differs')
    inputs=('question_id','question','true_answer','assumption','patient_fhir_id')
    require(all(same({k:a.get(k) for k in inputs},{k:b.get(k) for k in inputs}) for a,b in zip(srows,trows)),'Saved fixed300 references differ')
    expected_ids={r['question_id'] for r in trows};families={}
    # Template metadata only, retain only completed fixed300 IDs; no QA/SQL CSV.
    with (ROOT/'manifests/question_families.csv').open(newline='') as stream:
        for r in csv.DictReader(stream):
            if r['question_id'] in expected_ids:families[r['question_id']]=r['template']
    require(set(families)==expected_ids and sha(ROOT/'manifests/question_families.csv')==manifest['source_sha256']['manifests/question_families.csv'],'Fixed300 template metadata differs')
    scorer=module(ROOT/'utils/answer_scorer.py','independent_original_typed_scorer')
    store=module(store_path,'independent_owned_audit_store')
    totals=Counter();status=Counter();joint_all=Counter();joint_eligible=Counter();effects=Counter();selected=Counter();lost_flags=Counter();ineligible_reasons=Counter();computed=Counter();parity=Counter();parity_fields=Counter();restore_count=0
    for sr,tr in zip(srows,trows):
        def score(v):return scorer.score_answer(v,tr['true_answer'],tr['question'],families[tr['question_id']])
        sc=correct(score,sr.get('agent_answer'),failed(sr));tc=correct(score,tr.get('agent_answer'),failed(tr));totals['single_correct']+=sc;totals['team_correct']+=tc
        require(tr['audit_artifacts']['path'] in proof['raw_files'] and sr['audit_artifacts']['path'] in proof['raw_files'],'Unowned audit reference')
        restored=store.restore_result_audit(tr,OUT,question_id=tr['question_id'],restore_history=False);restore_count+=1
        rev=restored['revision'];require(rev.get('protocol')==PROTOCOL and rev.get('status')!='fatal_mutation','Actual A not reconstructible')
        snapshots=rev.get('messages');require(isinstance(snapshots,list),'Missing snapshots')
        by={v['candidate']:v for v in snapshots};require('A' in by and len(by)==len(snapshots),'Actual A unavailable')
        require({k:fp(v) for k,v in by.items()}==rev['candidate_snapshot_sha256'],'Snapshot identity differs')
        a=by['A'];released=a['released'];ad=a['diagnostics'];require(all(k in released for k in FIELDS),'A actual released/error missing')
        ac=correct(score,released['answer'],failed(released) or ad['fatal'] or (rev.get('fatal') and rev.get('status')=='fatal_primary'))
        totals['A_available']+=1;totals['A_correct']+=ac;totals['A_fatal']+=bool(ad['fatal']);totals['A_source_bound']+=a['artifact'] is not None
        totals['A_known_budget_stop']+=bool(ad.get('budget_stop'));effects[effect(ac,tc)]+=1;status[rev['status']]+=1
        if rev['status']=='selected_B' and rev['selected']=='B' and not rev['fatal']:
            require('B' in by and by['B']['artifact'] is not None and same({k:tr.get('agent_answer') if k=='answer' else tr.get(k) for k in FIELDS},{k:by['B']['released'].get(k) for k in FIELDS}),'Selected B not actual released B')
            selected[effect(ac,tc)]+=1
        elif not rev['fatal']:
            require(same({k:tr.get('agent_answer') if k=='answer' else tr.get(k) for k in FIELDS},{k:released.get(k) for k in FIELDS}),'Fallback not actual A')
        aux_fatal=bool(rev['fatal'] and rev['status'] in {'fatal_alternative','fatal_selection'})
        totals['auxiliary_fatal']+=aux_fatal;totals['auxiliary_fatal_correct_A_loss']+=aux_fatal and ac and not tc
        if 'B' in by:
            totals['B_entered']+=1;b=by['B'];br=b['released'];bd=b['diagnostics'];require(all(k in br for k in FIELDS),'B actual released/error missing')
            bc=correct(score,br['answer'],failed(br) or bd['fatal'] or (rev.get('fatal') and rev.get('status')=='fatal_alternative'))
            eligible=b['artifact'] is not None;totals['B_available']+=1;totals['B_correct']+=bc;totals['B_source_bound']+=eligible
            totals['B_known_budget_stop']+=bool(bd.get('budget_stop'));totals['B_fatal']+=bool(bd['fatal']);joint_all[joint(ac,bc)]+=1
            if eligible:joint_eligible[joint(ac,bc)]+=1
            if not ac and bc and not eligible:
                lost_flags['actual_released_B_correct_but_ineligible']+=1;lost_flags['released_B_answer_is_None']+=br['answer'] is None
                lost_flags['released_B_explicit_limitation']+=bool(br.get('limitation'));lost_flags['known_B_budget_stop']+=bool(bd.get('budget_stop'))
                prev=br.get('revision') or {};lost_flags['released_B_final_source_issues']+=bool(prev.get('final_check_issues'))
                reason=bd.get('skip_reason');ineligible_reasons[reason if reason in BINDING_REASONS else 'other_or_missing']+=1
            previous=br.get('revision') if isinstance(br.get('revision'),dict) else {};final=previous.get('final');draft=previous.get('draft')
            if isinstance(final,dict) and 'answer' in final:
                fc=bool(score(final['answer'])['typed_correct']);computed['B_computed_final_available']+=1;computed['B_computed_final_typed_correct']+=fc
                withheld=fc and not bc
                computed['B_computed_correct_not_actual_released_credit']+=withheld
                computed['A_wrong_B_computed_correct_not_actual_released_credit']+=not ac and withheld
                computed['A_wrong_B_computed_correct_unbound']+=not ac and fc and not eligible
                computed['A_wrong_B_computed_correct_unbound_known_budget_stop']+=not ac and fc and not eligible and bool(bd.get('budget_stop'))
                computed['A_wrong_B_computed_correct_unbound_explicit_limitation']+=not ac and fc and not eligible and bool(br.get('limitation'))
                computed['B_computed_correct_not_released_with_final_source_issues']+=withheld and bool(previous.get('final_check_issues'))
        # A single additional control restoration is needed for initial-message parity only.
        srestored=store.restore_result_audit(sr,OUT,question_id=sr['question_id'],restore_history=False);restore_count+=1
        st=srestored.get('trace') or [];at=[m for m in restored.get('trace') or [] if m.get('phase')=='natural_primary']
        require(fp(at)==a['tagged_trace_sha256'],'Actual A tagged trace SHA differs')
        clean_a=[{k:v for k,v in m.items() if k!='phase'} for m in at];require(fp(clean_a)==a['trace_sha256'],'Actual A natural trace SHA differs')
        si=initial(st);ai=initial(clean_a);parity['matched_pairs']+=1;parity['initial_message_prefix_exact_equal']+=same(si,ai)
        parity['initial_system_message_subset_exact_equal']+=same([m for m in si if m.get('role')=='system'],[m for m in ai if m.get('role')=='system'])
        su=[m for m in si if m.get('role')=='user'];au=[m for m in ai if m.get('role')=='user'];parity['initial_user_message_subset_exact_equal']+=same(su,au)
        parity['original_question_context_exact_equal']+=same(sr.get('question_with_context'),tr.get('question_with_context'))
        if not same(si,ai):
            for left,right in zip(si,ai):
                for key in set(left)|set(right):
                    if not same(left.get(key),right.get(key)):parity_fields[key]+=1
            parity['initial_prefix_length_mismatches']+=len(si)!=len(ai)
        parity['Single_control_any_context_compaction']+=bool((srestored.get('revision') or {}).get('context_compactions'))
        parity['A_any_context_compaction']+=bool((released.get('revision') or {}).get('context_compactions'))
    require(totals['A_available']==300 and totals['single_correct']==238 and totals['team_correct']==233,'Official totals/denominator differ')
    mechanism=load(RECORD/'diagnostics/mechanisms.json')['summary']
    require(totals['A_correct']==mechanism['development_acceptance']['A_only_correct_all300'],'A-only disagrees')
    require(dict(joint_eligible)==mechanism['A_B_source_bound_joint_correctness'],'Source-bound complementarity disagrees')
    require(totals['auxiliary_fatal_correct_A_loss']==mechanism['auxiliary_fatal_correct_A_loss'],'Auxiliary fatal loss differs')
    require(sha(proof_path)==COMPLETION_SHA and sha(official_path)==OFFICIAL_SHA,'Evidence changed')
    for arm,v in rawpins.items():require(sha(OUT/(arm+'.json'))==v['result_sha256'] and sha(OUT/(arm+'.json.run.json'))==v['identity_sha256'],'Raw changed')
    outcome={'status':'PASS_INDEPENDENT_ACTUAL_A_AND_B_MECHANISMS','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
      'completion_report_sha256':COMPLETION_SHA,'official_report_sha256':OFFICIAL_SHA,'scorer_sha256':SCORER_SHA,
      'script_sha256':sha(Path(__file__)),'raw_pins':rawpins,'actual_audit_restorations_for_mechanisms_and_parity':restore_count,
      'totals':dict(totals),'all_post_A_effect':dict(effects),'selected_B_effect':dict(selected),'status_counts':dict(status),
      'all_available_B_joint_actual_released_correctness':dict(joint_all),'source_bound_B_joint_actual_released_correctness':dict(joint_eligible),
      'A_wrong_actual_released_B_correct_unbound_overlapping_flags':dict(lost_flags),
      'A_wrong_actual_released_B_correct_unbound_skip_reasons':dict(ineligible_reasons),'computed_unreleased_B_diagnostics_separate_from_actual_credit':dict(computed),
      'initial_trace_message_parity':dict(parity),'initial_trace_message_differing_field_counts':dict(parity_fields),
      'sidecar_model_settings_exact_equal':same({k:v for k,v in identities['ordinary_single_agent'].items() if k!='agent_strategy'},
        {k:v for k,v in identities['ordinary_multi_agent'].items() if k!='agent_strategy'}),
      'limits':['A-only uses all300 ACTUAL normalized released/error outcomes with primary-fatal wrong; no computed candidate substitution.',
        'B all-available counterfactuals are actual released/error scores; source-bound pool is separately conditioned. Null typed credit is not proof of justified abstention or semantic correctness.',
        'Computed-final diagnostics refer to retained draft metadata only and never add released credit or prove a feasible eligible correction.',
        'Initial audit-message equality is a policy-prefix sanity check, not SDK wire attestation. Model/native tool schema actual first-request objects are not separately retained; same source hashes/settings cannot establish their on-wire byte identity.',
        'Context compaction can change actual model input relative to lossless audit messages; no request replay or retokenization was performed.',
        'A233 vs independently sampled Single238 does not identify a causal primary deficit or its cause. Legacy compare revised/transition fields are not used as V7 mechanism measures.',
        'All data are completed adaptive fixed300 only; no full QA/SQL CSV or protected1397 content read/scored; no source/runtime changes, models/FHIR/network or process interaction.'],
      'full_QA_CSV_read':False,'protected_QA_read':False,'model_calls':0,'FHIR_calls':0,'source_runtime_changes':False}
    destination=HERE/'review.json';require(not destination.exists(),'Fresh private review required');destination.write_text(json.dumps(outcome,indent=2,allow_nan=False)+'\n')
    print(json.dumps(outcome,indent=2))

if __name__=='__main__':main()
