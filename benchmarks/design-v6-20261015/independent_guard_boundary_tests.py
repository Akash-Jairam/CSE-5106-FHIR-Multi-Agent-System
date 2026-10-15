"""Independent synthetic audit-boundary cases; imports only private guard fixtures."""
import copy
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, '/private/tmp/fhir-v6-integrity-bfjeklza')
import test_v6_guard_protocol as t

class IndependentBoundaryTests(unittest.TestCase):
    def test_raw_primary_and_B_execution_must_be_final(self):
        for identifier in ['a-final', 'native-b']:
            for mode in ['inspect', None]:
                with self.subTest(identifier=identifier, mode=mode):
                    row=t.fixture()
                    for message in row['trace']:
                        if message.get('role')=='tool' and message.get('tool_call_id')==identifier:
                            value=json.loads(message['content'])
                            if mode is None: value.pop('mode')
                            else: value['mode']=mode
                            message['content']=json.dumps(value)
                            if identifier=='native-b': row['revision']['recompute']['B_execution']=value
                    with self.assertRaises(ValueError): t.m.attempt_counts(row,row,True)

    def test_raw_execution_failure_marker_cannot_certify_A_or_B(self):
        for identifier in ['a-final','native-b']:
            with self.subTest(identifier=identifier):
                row=t.fixture()
                for message in row['trace']:
                    if message.get('role')=='tool' and message.get('tool_call_id')==identifier:
                        value=json.loads(message['content']);value['error_type']='SyntheticWorkerFailure'
                        message['content']=json.dumps(value)
                        if identifier=='native-b': row['revision']['recompute']['B_execution']=value
                with self.assertRaises(ValueError): t.m.attempt_counts(row,row,True)

    def test_contradictory_raw_source_error_is_rejected(self):
        for key in ['error','failure']:
            with self.subTest(key=key):
                row=t.fixture();aux=row['revision']['recompute']
                aux['B_source_check'][key]='Synthetic raw checker failure'
                for message in row['trace']:
                    if message.get('role')=='tool' and message.get('tool_call_id')=='v6-independent-source-check':
                        message['content']=json.dumps(aux['B_source_check'])
                # Binding the contradictory bytes to a packet does not make them a clean check.
                with self.assertRaises(ValueError):
                    t.m._source_summary(aux['B'],aux['B_source_check'],row['retrieved_fhir_resources'])

    def test_auxiliary_phase_invocations_must_reconcile(self):
        row=t.fixture();row['revision']['recompute']['phase']['model_calls']=0
        row['revision']['phases'][1]['model_calls']=0
        with self.assertRaises(ValueError): t.m.attempt_counts(row,row,True)

    def test_primary_plus_auxiliary_calls_must_reconcile_aggregate(self):
        row=t.fixture();row['tokens']['llm_calls']=3
        with self.assertRaises(ValueError): t.m.attempt_counts(row,row,True)

    def test_ineligible_fatal_A_cannot_be_nonterminal(self):
        row=t.fixture(coverage=True);rev=row['revision'];rev['primary_artifact']=None
        rev['primary_diagnostics']['eligible']=False;rev['primary_diagnostics']['fatal']=True
        rev['phases']=[copy.deepcopy(rev['primary_diagnostics'])];rev.pop('recompute')
        rev['release_policy']='continuous_primary_preserved_auxiliary_ineligible'
        rev['final']=copy.deepcopy(rev['primary'])
        with self.assertRaises(ValueError): t.m.attempt_counts(row,row,True)

    def test_selector_native_tool_request_cannot_release_B(self):
        row=t.fixture();row['trace'][-1]['tool_calls']=[{'id':'forbidden','type':'function',
          'function':{'name':'execute_python_code','arguments':'{}'}}]
        with self.assertRaises(ValueError): t.m.attempt_counts(row,row,True)

    def test_nonfatal_tool_counter_must_match_actual_executed_replies(self):
        row=t.fixture();rev=row['revision'];aux=rev['recompute']
        a={k:rev['primary_artifact'][k] for k in ('answer','evidence','operation','contributing_ids','limitation')}
        rev['final']=copy.deepcopy(a);row.update(agent_answer=a['answer'],evidence=copy.deepcopy(a['evidence']))
        rev['release_policy']='continuous_primary_preserved';rev['independent_candidate_selected']=False
        aux.update(status='primary_preserved',selected='A',switched=False,tool_calls=3)
        aux['phase']['tool_calls']=3;rev['phases'][1]['tool_calls']=3
        row['budget']['tool_calls_run']=5
        with self.assertRaises(ValueError): t.m.attempt_counts(row,row,True)

    def test_verified_error_without_model_response_preserves_A_and_gap(self):
        row=t.fixture();rev=row['revision'];aux=rev['recompute'];a={k:rev['primary_artifact'][k] for k in
          ('answer','evidence','operation','contributing_ids','limitation')}
        rev['final']=copy.deepcopy(a);row.update(agent_answer=a['answer'],evidence=copy.deepcopy(a['evidence']))
        rev['release_policy']='continuous_primary_preserved';rev['independent_candidate_selected']=False
        for key in ['B','B_execution','B_source_check','B_eligible','answer_agreement','selection','selector_packet_sha256']:
            aux.pop(key,None)
        aux.update(status='primary_preserved',selected='A',switched=False,model_calls=1,tool_calls=0,selector_admitted=False)
        aux['phase'].update(model_calls=1,tool_calls=0,tokens=20);rev['phases'][1]=copy.deepcopy(aux['phase'])
        row['tokens'].update(total=120,llm_calls=3,per_call=[{'usage_available':True} for _ in range(3)])
        row['budget']['tool_calls_run']=2
        row['trace']=[m for m in row['trace'] if m.get('phase') not in
          ('independent_recompute','independent_selection') and not (m.get('phase')=='independent_generation' and m.get('role')=='assistant')]
        result=t.m.attempt_counts(row,row,True)
        self.assertEqual(result['source_valid_executed_releases'],1)
        self.assertEqual(result['auxiliary_invocation_response_gaps'],1)

if __name__=='__main__':
    before=t.m.sha(Path(t.m.__file__))
    suite=unittest.defaultTestLoader.loadTestsFromName('test_v6_guard_protocol')
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName('test_v6_guard_lifecycle'))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentBoundaryTests))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert t.m.sha(Path(t.m.__file__))==before,'Guard source changed during review'
    sys.exit(not result.wasSuccessful())
