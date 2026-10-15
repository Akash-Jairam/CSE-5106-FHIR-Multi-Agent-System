"""Bounded handwritten synthetic tests; no frozen runtime or actual study inputs."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('synthetic_supplementary',HERE/'supplementary.py')
supp=importlib.util.module_from_spec(spec);spec.loader.exec_module(supp)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,sort_keys=True)+'\n')
def artifact(answer):
    return {'answer':answer,'evidence':[],'operation':'Synthetic read','contributing_ids':[],'limitation':None}


def fixtures():
    cases=[];records=[];restored=[]
    orders=[['W','R','C'],['W','C','R'],['R','W','C'],['R','C','W'],['C','W','R'],['C','R','W']]
    rule={'O_min':6,'net_min':4,'harm_plus_fatal_max':2,
          'corrective_patients_min':2,'corrective_operations_min':2}
    for i in range(120):
        qid='synthetic-'+str(i)
        cases.append({'question_id':qid,'patient_fhir_id':'synthetic-patient-'+str(i),
            'study_family':'family-'+str(i%3),'operation_group':'operation-'+str(i%2),
            'followup_order':deepcopy(orders[i%6])})
        for mode,m in zip(supp.MODES,range(4)):
            answer=0 if i<7 else 7;error=None;b=None;selected=False;fatal=False
            if mode=='R':
                if i<6:b=artifact(7)
                if i<5:answer=7;selected=True
                if i==7:b=artifact(0);answer=0;selected=True
                if i==8:answer=None;error='Synthetic known fatal';fatal=True
            mechanism={'released_source_bound_B':b,'selected_B':selected,'auxiliary_fatal':fatal,
                'witness_bound':b is not None,'probe_interactions':int(mode!='A'),'status':'synthetic'}
            prompt=100+10*m;completion=20+m;cached=10+m
            row={'question_id':qid,'patient_fhir_id':cases[-1]['patient_fhir_id'],
                'study_arm':mode,'study_terminal':True,'question':'Synthetic recorded measurement?',
                'true_answer':'[7]','template':'synthetic','agent_answer':answer,'error':error,
                'evidence':[],'limitation':None,'primary_envelope_sha256':str(i).zfill(64),
                'mechanism':mechanism,'tokens':{'prompt':prompt,'completion':completion,'total':prompt+completion,
                    'cached_prompt':cached,'llm_calls':1 if mode=='A' else 2,'cost':None},
                'budget':{'tool_calls_run':2 if mode=='A' else 3},'latency':10+m,
                'physical_A_wall_seconds':10 if mode=='A' else None,'shared_A_wall_seconds':10,
                'followup_wall_seconds':None if mode=='A' else m,
                'physical_usage':{'available':True,'observed_invocations':1,'known_usage_invocations':1,
                    'unknown_usage_invocations':0,'cost':None,
                    'known_tokens':{'prompt':100 if mode=='A' else 10*m,
                                    'completion':20 if mode=='A' else m,'cached':10 if mode=='A' else m}},
                'physical_fhir':{'observed_physical_queries':int(mode!='W'),
                    'physical_requests':2 if mode=='A' else int(mode!='W'),
                    'pages':2 if mode=='A' else int(mode!='W'),'response_bytes':50 if mode=='A' else 10*int(mode!='W'),
                    'returned_records':5 if mode=='A' else 2*int(mode!='W')},
                'cache_activity':{'cache_observer_available':True,'load_calls':2,'hits':1,'misses':1}}
            records.append(row)
            audit={'question_id':qid,'study_arm':mode,'revision':{'status':'synthetic',
                'mechanism':deepcopy(mechanism),'messages':[],'phases':[],'admission':{}},'trace':[]}
            if i==6 and mode=='R':
                # Execution succeeded, but a later source check prevented release.
                audit['revision']['messages']=[{'candidate':'B','phase':'counterexample_investigation','artifact':None}]
                audit['trace']=[{'role':'assistant','phase':'counterexample_investigation','tool_calls':[
                    {'id':'unreleased-final','function':{'name':'execute_python_code',
                     'arguments':json.dumps({'code':'answer=7','mode':'final'})}}]},
                    {'role':'tool','phase':'counterexample_investigation','tool_call_id':'unreleased-final',
                     'content':json.dumps({**artifact(7),'mode':'final'})},
                    {'role':'assistant','phase':'counterexample_investigation','tool_calls':[
                    {'id':'refused-source','function':{'name':'check_fields','arguments':json.dumps({'citations':[]})}}]},
                    {'role':'tool','phase':'counterexample_investigation','tool_call_id':'refused-source',
                     'content':json.dumps({'issues':['unit mismatch']})}]
            restored.append(audit)
    zero={'O':0,'H':0,'M':0,'F':0,'all_post_A_helpful':0,'all_post_A_harmful':0,
          'selected_B':0,'B_available_source_bound':0,'auxiliary_fatal':0,'net':0}
    r={**zero,'O':6,'H':5,'M':1,'F':1,'all_post_A_helpful':5,'all_post_A_harmful':2,
       'selected_B':6,'B_available_source_bound':7,'auxiliary_fatal':1,'net':3}
    official={'status':'COMPLETED_MECHANISM_STUDY','primary_contrast':'R - A',
        'arms':{m:{'n':120,'correct':116 if m=='R' else 113,'terminal_errors':int(m=='R')} for m in supp.MODES},
        'mechanisms':{'W':dict(zero),'R':r,'C':dict(zero)},'R_corrective_patients':6,
        'R_corrective_operations':2,'R_minus_A_correct':3,'advance_gate':{**rule,'passed':False}}
    scorer=SimpleNamespace(score_answer=lambda answer,ref,question,template:{'typed_correct':answer in json.loads(ref)},
                           parse_reference=json.loads)
    return records,restored,{'cases':cases,'policy':{'advance_gate':rule}},scorer,official


def gate_fixture(tmp):
    records,restored,manifest,scorer,official=fixtures()
    root=Path(tmp)/'synthetic-root';root.mkdir();output=root/'raw';output.mkdir();record=root/'record';record.mkdir()
    (root/'scripts').mkdir();(root/'scripts/v8_study_integrity.py').write_text('# Synthetic guard identity only\n')
    (root/'manifests').mkdir();save(root/'manifests/v8_mechanism_120_protocol_repair_20261015.json',manifest)
    save(record/'manifest.json',{'synthetic':True})
    save(output/'study_outcomes.json',records)
    save(output/'study_state.json',{'phase':'generation_complete','saved_policy_outcomes':480,
        'expected_policy_outcomes':480,'correctness_scored':False,'physical_generation_wall_seconds':1920})
    save(output/'study_identity.json',{'development_interface_only':False})
    pins=root/'pins.json';save(pins,{'synthetic':True})
    terminal=root/'terminal.json';save(terminal,{'version':1,'owned_study_process_terminal':True,
        'accuracy_scored':False,'owned_session_id':100,'exit_code':0,'saved_policy_outcomes':480,
        'expected_policy_outcomes':480,'output_directory':str(output),
        'observed_terminal_utc':'2026-10-15T21:00:00+00:00'})
    report=root/'integrity.json';official_path=root/'official.json';save(official_path,official)
    args=SimpleNamespace(root=root,record=record,study_output=output,destination=root/'diagnostics',
        integrity_report=report,integrity_sha256='',terminal_proof=terminal,terminal_sha256=digest(terminal),
        pins=pins,pins_sha256=digest(pins),official_analysis=official_path,official_sha256=digest(official_path))
    proof={'kind':'v8_study_completion_integrity','status':'PASS','engineering_only':False,'policy_outcomes':480,
        'questions':120,'accuracy_scored':False,'owned_terminal_attestation_sha256':args.terminal_sha256,
        'pins_sha256':args.pins_sha256,'guard_sha256':digest(root/'scripts/v8_study_integrity.py'),
        'source_manifest_sha256':digest(record/'manifest.json'),
        'study_manifest_path':'manifests/v8_mechanism_120_protocol_repair_20261015.json',
        'study_manifest_sha256':digest(root/'manifests/v8_mechanism_120_protocol_repair_20261015.json')}
    def repin():
        args.terminal_sha256=digest(terminal);proof['owned_terminal_attestation_sha256']=args.terminal_sha256
        proof['raw_sha256']={name:digest(output/name) for name in ('study_outcomes.json','study_state.json','study_identity.json')}
        save(report,proof);args.integrity_sha256=digest(report)
    repin()
    audits={(r['question_id'],r['study_arm']):r for r in restored}
    store=SimpleNamespace(restore_result_audit=lambda row,*a,**kw:deepcopy(audits[row['question_id'],row['study_arm']]))
    study_name='manifests/v8_mechanism_120_protocol_repair_20261015.json'
    sources={study_name:digest(root/study_name)}
    guard=SimpleNamespace(provenance=Mock(return_value=({'output_directory':str(output),
            'source_sha256':sources},manifest,{},{})),
        bound_study_manifest=Mock(return_value=(study_name,root/study_name,{'study_manifest':study_name})),
        owned=lambda directory,name:Path(directory)/name,
        shared=lambda:SimpleNamespace(B=SimpleNamespace(load_public_store=lambda root:store)))
    return args,guard,proof,repin,scorer


class SupplementaryIndependentTests(unittest.TestCase):
    def test_reported_physical_phase_cost_uses_A_once_and_never_infers_unknown_suffix_zero(self):
        actual_a={'tokens':{'llm_calls':2}}
        a={'study_arm':'A','revision':{'messages':[{'candidate':'A','usage':{'total_usage':{'cost':0.25}}}]}}
        self.assertEqual(supp.reported_physical_cost(a,actual_a),(0.25,'actual_A_reported_cost_once'))
        aux={'study_arm':'R','tokens':{'llm_calls':4},'revision':{'phases':[
            {'phase':'counterexample_auxiliary','raw_accounting':{'total_usage':{'cost':0.1}}}]}}
        self.assertEqual(supp.reported_physical_cost(aux,actual_a),(0.1,'actual_auxiliary_reported_cost'))
        aux['revision']['phases'][0]['raw_accounting']['total_usage']['cost']=None
        self.assertEqual(supp.reported_physical_cost(aux,actual_a),(None,'actual_auxiliary_reported_cost'))
        a['revision']['messages'][0]['usage']['total_usage']['cost']=None
        self.assertEqual(supp.reported_physical_cost(a,actual_a),(None,'actual_A_reported_cost_once'))
        noaux={'study_arm':'W','tokens':{'llm_calls':2},'revision':{'phases':[]}}
        self.assertEqual(supp.reported_physical_cost(noaux,actual_a),(0,'absent_auxiliary_zero_admitted_suffix_calls'))
        noaux['tokens']['llm_calls']=3
        self.assertEqual(supp.reported_physical_cost(noaux,actual_a),(None,'phase_cost_unavailable_not_assumed_zero'))

    def test_handwritten_transitions_reconcile_and_effects_do_not_overlap(self):
        records,restored,manifest,scorer,official=fixtures()
        result=supp.summarize(records,restored,manifest,scorer,official)
        r=result['mechanisms_exact_official_reconciliation']['R']
        self.assertEqual([r[k] for k in ('O','H','M','F','net')],[6,5,1,1,3])
        self.assertEqual(result['arms']['R']['correct']-result['arms']['A']['correct'],3)
        self.assertFalse(result['official_advance_gate']['passed'])
        for flag in result['case_flags']:
            self.assertLessEqual(sum(flag[k] for k in ('H','M','F')),1)
            self.assertEqual(flag['witness_semantic_validity'],'unknown')

    def test_computed_source_refused_B_is_diagnostic_and_not_an_opportunity(self):
        result=supp.summarize(*fixtures())
        flag=next(f for f in result['case_flags'] if f['question_id']=='synthetic-6' and f['study_arm']=='R')
        self.assertFalse(flag['O']);self.assertFalse(flag['source_bound_B_available'])
        self.assertTrue(flag['B_unreleased_execution_typed_correct'])
        self.assertEqual(flag['B_unreleased_execution_check'],'recorded_issues')
        self.assertEqual(result['operational']['R']['candidate_states']['B:retained_execution_with_later_source_issues'],1)

    def test_hand_totals_count_physical_A_once_and_separate_conceptual_latency(self):
        result=supp.summarize(*fixtures());account=result['accounting']
        physical=account['physical_A_once_plus_suffixes'];concept=account['conceptual_policies']
        self.assertEqual(physical['known_prompt_tokens']['known_sum'],19200)
        self.assertEqual(sum(concept[m]['prompt']['known_sum'] for m in supp.MODES),55200)
        self.assertEqual(physical['known_completion_tokens']['known_sum'],3120)
        self.assertEqual(physical['known_cached_tokens']['known_sum'],1920)
        self.assertEqual(physical['observed_invocations']['known_sum'],480)
        self.assertEqual(physical['physical_A_wall_seconds']['known_sum'],1200)
        self.assertEqual(physical['followup_wall_seconds']['known_sum'],720)
        self.assertEqual(concept['R']['A_plus_suffix_latency_seconds']['known_sum'],1440)
        self.assertEqual(physical['fhir_pagination']['physical_requests']['known_sum'],480)
        self.assertEqual(physical['fhir_pagination']['returned_records']['known_sum'],1080)
        self.assertEqual(physical['tool_cache']['load_calls']['known_sum'],960)
        self.assertEqual(physical['tool_cache']['hits']['known_sum'],480)

    def test_absent_cost_cached_resource_and_cache_metadata_stays_unknown(self):
        records,restored,manifest,scorer,official=fixtures();row=records[2]
        row['physical_usage']['known_tokens'].pop('cached');row['physical_fhir']['physical_requests']=None
        row['cache_activity']={'cache_observer_available':False}
        result=supp.summarize(records,restored,manifest,scorer,official)
        physical=result['accounting']['physical_A_once_plus_suffixes']
        self.assertEqual(physical['cost']['unknown_records'],480)
        self.assertIsNone(physical['cost']['known_mean'])
        self.assertEqual(physical['known_cached_tokens']['unknown_records'],1)
        self.assertEqual(physical['known_cached_tokens']['known_sum'],1918)
        self.assertEqual(physical['fhir_pagination']['physical_requests']['unknown_records'],1)
        self.assertEqual(physical['tool_cache']['hits']['unknown_records'],1)
        self.assertEqual(physical['cache_observer_unavailable_rows'],1)

    def test_cohort_overlap_and_official_drift_rejected(self):
        for mutation in ('479','overlap','official','flipped_gate'):
            records,restored,manifest,scorer,official=fixtures()
            if mutation=='479':records.pop()
            elif mutation=='overlap':records[2]['mechanism']['selected_B']=True;records[2]['mechanism']['auxiliary_fatal']=True
            elif mutation=='official':official['mechanisms']['R']['H']+=1
            else:official['advance_gate']['passed']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):supp.summarize(records,restored,manifest,scorer,official)

    def test_complete_synthetic_gate_and_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,guard,proof,repin,scorer=gate_fixture(tmp);loader=Mock(return_value=scorer)
            result=supp.run(args,guard_loader=lambda _:guard,scorer_loader=loader)
            loader.assert_called_once()
            self.assertEqual(result['physical_generation_wall_seconds'],1920)
            self.assertTrue((args.destination/'supplementary.json').exists())
            self.assertEqual(guard.provenance.call_count,2)

    def test_pinned_completion_and_current_raw_failures_precede_scorer_import(self):
        for mutation in ('report_fail','engineering','terminal_bool_exit','raw_drift','479','duplicate','development','external_pin','old_manifest_path','missing_manifest_path'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                args,guard,proof,repin,scorer=gate_fixture(tmp);loader=Mock(return_value=scorer)
                if mutation=='report_fail':proof['status']='FAIL';repin()
                elif mutation=='engineering':proof['engineering_only']=True;repin()
                elif mutation=='terminal_bool_exit':
                    value=supp.load(args.terminal_proof);value['exit_code']=True;save(args.terminal_proof,value);repin()
                elif mutation=='raw_drift':
                    path=args.study_output/'study_outcomes.json';path.write_text(path.read_text()+'\n')
                elif mutation in ('479','duplicate'):
                    value=supp.load(args.study_output/'study_outcomes.json')
                    if mutation=='479':value.pop()
                    else:value[-1]=deepcopy(value[0])
                    save(args.study_output/'study_outcomes.json',value);repin()
                elif mutation=='development':save(args.study_output/'study_identity.json',{'development_interface_only':True});repin()
                elif mutation=='old_manifest_path':
                    proof['study_manifest_path']='manifests/v8_mechanism_120_20261015.json';repin()
                elif mutation=='missing_manifest_path':
                    proof.pop('study_manifest_path');repin()
                else:args.official_analysis.write_text(args.official_analysis.read_text()+'\n')
                with self.assertRaises(ValueError):supp.run(args,guard_loader=lambda _:guard,scorer_loader=loader)
                loader.assert_not_called();self.assertFalse(args.destination.exists())

    def test_provenance_failure_precedes_scorer_and_postscore_drift_blocks_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,guard,proof,repin,scorer=gate_fixture(tmp);loader=Mock(return_value=scorer)
            guard.provenance.side_effect=ValueError('Synthetic source/runtime/archive drift')
            with self.assertRaises(ValueError):supp.run(args,guard_loader=lambda _:guard,scorer_loader=loader)
            loader.assert_not_called();self.assertFalse(args.destination.exists())
        with tempfile.TemporaryDirectory() as tmp:
            args,guard,proof,repin,scorer=gate_fixture(tmp)
            def changed_loader(path):
                args.terminal_proof.write_text(args.terminal_proof.read_text()+'\n')
                return scorer
            with self.assertRaisesRegex(ValueError,'drift'):
                supp.run(args,guard_loader=lambda _:guard,scorer_loader=changed_loader)
            self.assertFalse(args.destination.exists())


if __name__=='__main__':unittest.main(verbosity=2)
