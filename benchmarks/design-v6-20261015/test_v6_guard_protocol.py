"""Synthetic V6 phase/artifact metadata only; no runtime import or scoring."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('v6_protocol_guard',HERE/'v6_guard.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
START_SHA=m.sha(HERE/'v6_guard.py')


def source(candidate):
    return {'issues':[],'n_valid':len(candidate['evidence']),
      'results':[{'citation':deepcopy(c),'status':'valid','warnings':[]} for c in candidate['evidence']]}


def candidate(answer,identifier):
    return {'answer':answer,'evidence':[{'resource_id':'Observation/'+identifier,'field_path':'valueQuantity.value','value':answer}],
      'operation':'Read one synthetic field','contributing_ids':['Observation/'+identifier],'limitation':None}


def pair(identifier,name,args,result,phase=None):
    suffix={'phase':phase} if phase else {}
    return [{'role':'assistant','content':None,**suffix,'tool_calls':[{'id':identifier,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]},
      {'role':'tool','name':name,'tool_call_id':identifier,'content':json.dumps(result),**suffix}]


def fixture(agreement=False,coverage=False):
    workspace={'Observation':[{'resourceType':'Observation','id':'a','valueQuantity':{'value':1}},
      {'resourceType':'Observation','id':'b','valueQuantity':{'value':2}}]}
    a=candidate(1,'a');b=candidate(1,'a') if agreement else candidate(2,'b')
    code_a='answer = 1\n';code_b='answer = '+repr(b['answer'])+'\n'
    sha=m.fingerprint(workspace)
    artifact={**deepcopy(a),'mode':'final','code':code_a,'workspace_sha256':sha}
    row={'question_id':'synthetic','question':'Return a synthetic field.','patient_fhir_id':'synthetic-patient',
      'question_with_context':'Question: Return a synthetic field.\nContext:\nPatient FHIR ID is synthetic-patient.',
      'error':None,'latency':1,'tokens':{'total':100 if coverage else 150,'llm_calls':2 if coverage else 3 if agreement else 4,
        'per_call':[{'usage_available':True} for _ in range(2 if coverage else 3 if agreement else 4)]},
      'budget':{'max_total_tokens':200000,'max_tool_calls':30,'tool_calls_run':2 if coverage else 4,'stopped':None}}
    primary_diag={'phase':'primary','token_cap':175000,'tool_cap':27,'max_iterations':30,'model_call_attempts_recorded':2,
      'tokens':100,'tool_calls':2,'fatal':False,'errors':[],'eligible':True}
    phase={'phase':'independent_recompute','token_cap':25000,'tool_cap':3,'model_call_cap':2,
      'model_calls':0 if coverage else 1 if agreement else 2,'tokens':0 if coverage else 50,
      'tool_calls':0 if coverage else 2,'fatal':False,'errors':[]}
    aux={'protocol':'independent-frozen-recomputation-v6','status':'coverage_skip' if coverage else 'agreement_preserved' if agreement else 'switched',
      'assurance':m.ASSURANCE,'fatal':False,'selected':'A' if agreement or coverage else 'B','switched':not agreement and not coverage,
      'model_calls':phase['model_calls'],'tool_calls':phase['tool_calls'],'phase':phase,
      'packet_admitted':not coverage,'selector_admitted':not agreement and not coverage}
    trace=[{'role':'system','content':'Synthetic actual Single context'}]
    trace+=pair('a-final','execute_python_code',{'code':code_a,'mode':'final'},{**a,'mode':'final'})
    trace+=[{'role':'assistant','content':'Natural primary final response'}]
    trace+=pair('automatic_field_check_1','check_fields',{'citations':a['evidence']},source(a))
    context=m.context_from_row(row)
    inventory={kind:{'count':len(items),'source_example':items[:1],'concept_labels':[]} for kind,items in workspace.items()}
    packet={'context':context,'workspace_sha256':sha,'held_inventory':inventory,'retrieval_allowed':False}
    if not coverage:
        aux.update(generation_packet_sha256=m._aux_fingerprint(packet),B=deepcopy(b),B_execution={**b,'mode':'final'},
          B_source_check=source(b),B_eligible=True,answer_agreement=agreement)
        trace += [{'role':'system','phase':'independent_generation','content':'Synthetic blind generation'},
          {'role':'user','phase':'independent_generation','content':json.dumps(packet)}]
        request,response=pair('native-b','execute_python_code',{'code':code_b,'mode':'final'},{**b,'mode':'final'},'independent_generation')
        response['phase']='independent_recompute';trace.extend([request,response])
        trace+=pair('v6-independent-source-check','check_fields',{'citations':b['evidence']},source(b),'independent_recompute')
        if not agreement:
            selection={'context':context,'workspace_sha256':sha,
              'A':{'program':code_a,'artifact':a,'source_check':m._source_summary(a,source(a),workspace),'program_sha256':m.sha_text(code_a),'workspace_sha256':sha},
              'B':{'program':code_b,'artifact':b,'source_check':m._source_summary(b,source(b),workspace),'program_sha256':m.sha_text(code_b),'workspace_sha256':sha}}
            aux.update(selector_packet_sha256=m._aux_fingerprint(selection),selection={'choice':'B','reason':'Synthetic comparison only'})
            trace += [{'role':'system','phase':'independent_selection','content':'Synthetic A/B choice'},
              {'role':'user','phase':'independent_selection','content':json.dumps(selection)},
              {'role':'assistant','phase':'independent_selection','content':json.dumps(aux['selection'])}]
    chosen=a if agreement or coverage else b
    revision={'protocol':m.PROTOCOL,'primary':m.rendered(a),'primary_artifact':artifact,'primary_revision':{'final_check_issues':[]},
      'primary_diagnostics':primary_diag,'phases':[deepcopy(primary_diag),deepcopy(phase)],'final':deepcopy(chosen),
      'final_check_issues':[],'recompute':aux,'primary_source_revised':False,
      'independent_candidate_selected':not agreement and not coverage,
      'release_policy':'continuous_primary_preserved' if agreement or coverage else 'selected_executed_independent_candidate'}
    row.update(agent_answer=chosen['answer'],evidence=deepcopy(chosen['evidence']),limitation=None,revision=revision,
      trace=trace,retrieved_fhir_resources=workspace)
    return row


class V6ProtocolTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        if m.sha(HERE/'v6_guard.py')!=START_SHA:raise AssertionError('Guard bytes changed during synthetic suite')

    def test_selected_B_binds_actual_program_source_selector_and_frozen_A(self):
        row=fixture();counts=m.attempt_counts(row,row,True)
        self.assertEqual(counts['independent_candidates_selected'],1)
        self.assertEqual(counts['bound_candidate_blind_packets'],1)
        self.assertEqual(counts['executed_source_bound_B_candidates'],1)

    def test_agreement_retains_A_without_selector(self):
        row=fixture(agreement=True);counts=m.attempt_counts(row,row,True)
        self.assertEqual(counts['auxiliary_status_agreement_preserved'],1)
        self.assertEqual(counts['auxiliary_responses_observed'],1)
        self.assertEqual(counts['independent_candidates_selected'],0)

    def test_coverage_skip_preserves_actual_A(self):
        row=fixture(coverage=True);counts=m.attempt_counts(row,row,True)
        self.assertEqual(counts['auxiliary_status_coverage_skip'],1)
        self.assertEqual(counts['bound_candidate_blind_packets'],0)

    def test_changed_preserved_A_is_rejected(self):
        row=fixture(coverage=True);row['revision']['final']['answer']=99;row['agent_answer']=99
        with self.assertRaisesRegex(ValueError,'Fallback changed A'):m.attempt_counts(row,row,True)

    def test_ineligible_prose_A_preserved_without_execution_claim(self):
        row=fixture(coverage=True);rev=row['revision'];rev['primary_artifact']=None
        rev['primary_diagnostics']['eligible']=False;rev['phases']=[deepcopy(rev['primary_diagnostics'])]
        rev.pop('recompute');rev['release_policy']='continuous_primary_preserved_auxiliary_ineligible'
        rev['final']=deepcopy(rev['primary']);row['tokens']['total']=100;row['tokens']['llm_calls']=2
        counts=m.attempt_counts(row,row,True)
        self.assertEqual(counts['ineligible_primary_preserved'],1)
        self.assertEqual(counts['source_valid_executed_releases'],0)

    def test_actual_generation_packet_cannot_include_A_even_with_rehashed_metadata(self):
        row=fixture();aux=row['revision']['recompute']
        message=next(x for x in row['trace'] if x.get('phase')=='independent_generation' and x['role']=='user')
        packet=json.loads(message['content']);packet['A']=row['revision']['primary'];message['content']=json.dumps(packet)
        aux['generation_packet_sha256']=m._aux_fingerprint(packet)
        with self.assertRaisesRegex(ValueError,'Candidate-blind generation'):m.attempt_counts(row,row,True)

    def test_selector_A_source_summary_must_match_actual_primary_check(self):
        row=fixture();aux=row['revision']['recompute']
        message=next(x for x in row['trace'] if x.get('phase')=='independent_selection' and x['role']=='user')
        packet=json.loads(message['content']);packet['A']['source_check']['raw_check_sha256']='0'*64
        message['content']=json.dumps(packet);aux['selector_packet_sha256']=m._aux_fingerprint(packet)
        with self.assertRaisesRegex(ValueError,'Selector A source summary'):m.attempt_counts(row,row,True)

    def test_B_source_warnings_block_claimed_switch(self):
        row=fixture();aux=row['revision']['recompute'];aux['B_source_check']['results'][0]['warnings']=['synthetic warning']
        for message in row['trace']:
            if message.get('role')=='tool' and message.get('tool_call_id')=='v6-independent-source-check':message['content']=json.dumps(aux['B_source_check'])
        with self.assertRaisesRegex(ValueError,'warning-free'):m.attempt_counts(row,row,True)

    def test_selector_cannot_release_third_candidate(self):
        row=fixture();aux=row['revision']['recompute'];aux['selection']['choice']='C'
        row['trace'][-1]['content']=json.dumps(aux['selection'])
        with self.assertRaisesRegex(ValueError,'exact existing B'):m.attempt_counts(row,row,True)

    def test_unknown_usage_retained_as_terminal_fatal_not_deleted(self):
        row=fixture(coverage=True);row['tokens']['per_call'][0]['usage_available']=False;row['error']='Synthetic accounting failure'
        row['agent_answer']=None;row['evidence']=[];row['limitation']='Accounting unavailable'
        row['revision'].update(final={'answer':None,'evidence':[],'limitation':row['limitation']},release_policy='fatal_execution_or_accounting_failure')
        row['revision']['primary_diagnostics']['fatal']=True;row['revision']['phases'][0]['fatal']=True
        counts=m.attempt_counts(row,row,True)
        self.assertEqual(counts['saved_first_attempts'],1);self.assertEqual(counts['recorded_unknown_provider_usage_attempts'],1)
        self.assertEqual(counts['fatal_release_outcomes'],1)

    def test_unknown_usage_cannot_silently_receive_nonterminal_null_credit(self):
        row=fixture(coverage=True);row['tokens']['per_call'][0]['usage_available']=False
        with self.assertRaisesRegex(ValueError,'nonterminal null'):m.attempt_counts(row,row,True)

    def test_terminal_phase_gap_is_reported_without_refusing_complete_first(self):
        row=fixture(coverage=True);row['error']='Synthetic unexpected auxiliary failure';row['tokens']['total']=101
        row['agent_answer']=None;row['evidence']=[];row['limitation']='Uncertain usage'
        row['revision'].update(final={'answer':None,'evidence':[],'limitation':row['limitation']},release_policy='fatal_execution_or_accounting_failure')
        counts=m.attempt_counts(row,row,True)
        self.assertEqual(counts['terminal_phase_accounting_sum_gaps'],1)

    def test_primary_inspect_reply_cannot_certify_executed_A(self):
        row=fixture()
        reply=next(message for message in row['trace'] if message.get('role')=='tool' and message.get('tool_call_id')=='a-final')
        result=json.loads(reply['content']);result['mode']='inspect';reply['content']=json.dumps(result)
        with self.assertRaisesRegex(ValueError,'actual explicit final execution'):m.attempt_counts(row,row,True)

    def test_independent_inspect_reply_cannot_certify_executed_B(self):
        row=fixture();aux=row['revision']['recompute'];aux['B_execution']['mode']='inspect'
        reply=next(message for message in row['trace'] if message.get('role')=='tool' and message.get('tool_call_id')=='native-b')
        reply['content']=json.dumps(aux['B_execution'])
        with self.assertRaisesRegex(ValueError,'actual native final execution'):m.attempt_counts(row,row,True)

    def test_primary_contradictory_execution_error_type_cannot_certify_A(self):
        row=fixture()
        reply=next(message for message in row['trace'] if message.get('role')=='tool' and message.get('tool_call_id')=='a-final')
        result=json.loads(reply['content']);result['error_type']='synthetic';reply['content']=json.dumps(result)
        with self.assertRaisesRegex(ValueError,'actual explicit final execution'):m.attempt_counts(row,row,True)

    def test_independent_contradictory_execution_error_type_cannot_certify_B(self):
        row=fixture();aux=row['revision']['recompute'];aux['B_execution']['error_type']='synthetic'
        reply=next(message for message in row['trace'] if message.get('role')=='tool' and message.get('tool_call_id')=='native-b')
        reply['content']=json.dumps(aux['B_execution'])
        with self.assertRaisesRegex(ValueError,'actual native final execution'):m.attempt_counts(row,row,True)

    def test_contradictory_source_error_cannot_certify_A_or_B(self):
        for owner in ('A','B'):
            with self.subTest(owner=owner):
                row=fixture();identifier='automatic_field_check_1' if owner=='A' else 'v6-independent-source-check'
                reply=next(message for message in row['trace'] if message.get('role')=='tool' and message.get('tool_call_id')==identifier)
                check=json.loads(reply['content']);check['error']='Synthetic failure';reply['content']=json.dumps(check)
                if owner=='B':row['revision']['recompute']['B_source_check']=deepcopy(check)
                with self.assertRaisesRegex(ValueError,'source-check envelope'):m.attempt_counts(row,row,True)

    def test_phase_counter_and_auxiliary_counter_must_reconcile(self):
        row=fixture();row['revision']['recompute']['phase']['model_calls']=0;row['revision']['phases'][1]['model_calls']=0
        with self.assertRaisesRegex(ValueError,'invocation accounting'):m.attempt_counts(row,row,True)

    def test_nonfatal_aggregate_invocations_must_reconcile(self):
        row=fixture();row['tokens']['llm_calls']=3
        with self.assertRaisesRegex(ValueError,'invocation accounting'):m.attempt_counts(row,row,True)

    def test_ineligible_fatal_primary_requires_actual_terminal_error(self):
        row=fixture(coverage=True);rev=row['revision'];rev['primary_artifact']=None
        rev['primary_diagnostics']['eligible']=False;rev['primary_diagnostics']['fatal']=True
        rev['phases']=[deepcopy(rev['primary_diagnostics'])];rev.pop('recompute')
        rev['release_policy']='continuous_primary_preserved_auxiliary_ineligible';rev['final']=deepcopy(rev['primary'])
        row['tokens']['total']=100
        with self.assertRaisesRegex(ValueError,'fatal accounting lacks'):m.attempt_counts(row,row,True)

    def test_selector_requesting_tool_cannot_release_B(self):
        row=fixture();row['trace'][-1]['tool_calls']=[{'id':'forbidden-selector-tool','function':{'name':'execute_python_code','arguments':'{}'}}]
        with self.assertRaisesRegex(ValueError,'actual response'):m.attempt_counts(row,row,True)

    def test_certified_B_tool_counter_must_match_actual_replies(self):
        row=fixture(agreement=True);rev=row['revision'];rev['recompute']['tool_calls']=3
        rev['recompute']['phase']['tool_calls']=3;rev['phases'][1]['tool_calls']=3;row['budget']['tool_calls_run']=5
        with self.assertRaisesRegex(ValueError,'tool counter differs'):m.attempt_counts(row,row,True)

    def test_discarded_B_source_exception_tool_gap_is_retained_as_fallback(self):
        row=fixture();rev=row['revision'];aux=rev['recompute'];a={k:rev['primary_artifact'][k] for k in ('answer','evidence','operation','contributing_ids','limitation')}
        row.update(agent_answer=a['answer'],evidence=deepcopy(a['evidence']),limitation=a['limitation'])
        row['trace']=[message for message in row['trace'] if message.get('phase')!='independent_selection'
          and not (message.get('phase')=='independent_recompute' and (message.get('tool_call_id')=='v6-independent-source-check'
          or any(c.get('id')=='v6-independent-source-check' for c in message.get('tool_calls') or [])))]
        aux.update(status='primary_preserved',selected='A',switched=False,model_calls=1,selector_admitted=False,
          reason='Synthetic known deterministic source-check exception after tool charge')
        for key in ('B_eligible','B_source_check','answer_agreement','selection','selector_packet_sha256'):aux.pop(key,None)
        aux['phase']['model_calls']=1;rev['phases'][1]['model_calls']=1;row['tokens']['llm_calls']=3
        rev.update(final=deepcopy(a),release_policy='continuous_primary_preserved',independent_candidate_selected=False)
        counts=m.attempt_counts(row,row,True)
        self.assertEqual(counts['auxiliary_charged_tool_reply_gaps'],1)
        self.assertEqual(counts['auxiliary_charged_minus_observed_tool_replies'],1)
        self.assertEqual(counts['executed_source_bound_B_candidates'],0)

    def test_unexecuted_malformed_generation_arguments_preserve_A(self):
        row=fixture(coverage=True);rev=row['revision'];aux=rev['recompute'];workspace=row['retrieved_fhir_resources']
        packet={'context':m.context_from_row(row),'workspace_sha256':m.fingerprint(workspace),
          'held_inventory':{'Observation':{'count':2,'source_example':workspace['Observation'][:1],'concept_labels':[]}},'retrieval_allowed':False}
        row['trace'] += [{'role':'user','phase':'independent_generation','content':json.dumps(packet)},
          {'role':'assistant','phase':'independent_generation','tool_calls':[{'id':'bad','type':'function','function':{'name':'execute_python_code','arguments':'{malformed'}}]}]
        aux.update(status='primary_preserved',packet_admitted=True,generation_packet_sha256=m._aux_fingerprint(packet),model_calls=1)
        aux['phase']['model_calls']=1;rev['phases'][1]['model_calls']=1;row['tokens']['llm_calls']=3
        counts=m.attempt_counts(row,row,True)
        self.assertEqual(counts['unexecuted_malformed_auxiliary_requests'],1)
        self.assertEqual(counts['executed_source_bound_B_candidates'],0)


if __name__=='__main__':unittest.main(verbosity=2)
