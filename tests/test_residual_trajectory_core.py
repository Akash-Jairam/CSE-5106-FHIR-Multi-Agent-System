"""V7 natural Single phases, isolation, selection and accounting; fully synthetic."""
from contextlib import ExitStack
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from litellm import Message
from agent.residual_trajectory_agent import ResidualTrajectoryAgent, selection_object
from agent.residual_trajectory_phase import _observe_initialization, _guard_B_calls, clean_ledger, fingerprint
from agent.single_agent import SingleAgent
from tests.test_continuous_primary import (PACKET, QUESTION, RESOURCE, computation,
                                          source_check, usage)


def native(value=7, identifier="same-local-id"):
    return Message(role="assistant", tool_calls=[{"id": identifier, "type": "function", "function": {
        "name": "execute_python_code", "arguments": json.dumps({"code": f"answer={value}", "mode": "final"})}}])


def prose(value=7):
    return Message(role="assistant", content=json.dumps({"answer": value, "evidence": [], "limitation": None}))


def selector(choice="B", A="contradicted", B="supported", **changes):
    value = {"requirements": [{"requirement": "Select the requested original value", "A": {
        "status": A, "basis": "A program selects the other synthetic field"}, "B": {
        "status": B, "basis": "B program selects the requested synthetic input field"}}],
        "choice": choice, "reason": "Compare the existing programs and actual cited fields"}
    value.update(changes)
    return Message(role="assistant", content=json.dumps(value))


class V7CoreTests(unittest.TestCase):
    def exercise(self, responses, outputs=(), usages=None, *, cap=200000, tools=30, check=source_check,
                 packet=None, model="arbitrary-model", base_url="https://fixture.invalid"):
        stack = ExitStack(); self.addCleanup(stack.close)
        stack.enter_context(patch("agent.base_agent.setup_api_keys"))
        stack.enter_context(patch("agent.base_agent.count_tokens_in_messages", return_value=10))
        packet_mock = stack.enter_context(patch("agent.single_agent.get_starting_packet", side_effect=packet or
            (lambda *a, **k: deepcopy(PACKET))))
        stack.enter_context(patch("agent.single_agent.check_fields", side_effect=check))
        pending_outputs = iter(deepcopy(list(outputs)))
        def execute(**kwargs):
            return next(pending_outputs)
        stack.enter_context(patch("agent.multi_turn_code_resource_agent.get_tool", return_value=execute))
        pending = iter(responses); bills = iter(usages) if usages is not None else None; requests = []
        def provider(**kwargs):
            requests.append(deepcopy(kwargs))
            response = next(pending)
            if isinstance(response, Exception): raise response
            if isinstance(response, tuple): return response
            return response, None, deepcopy(next(bills) if bills is not None else usage())
        stack.enter_context(patch("agent.base_agent.safe_llm_call", side_effect=provider))
        agent = ResidualTrajectoryAgent(model, base_url=base_url)
        agent.set_budget(max_total_tokens=cap, max_tool_calls=tools)
        result = agent.run(QUESTION)
        return result, requests, packet_mock, agent

    def test_natural_A_full_caps_B_fresh_same_model_and_own_workspace_selected_existing_B(self):
        b = computation(8); b['evidence'] = computation()['evidence']
        result, requests, packet, agent = self.exercise([native(), prose(999), native(8), prose(999), selector()],
            [computation(), b])
        self.assertIsNone(result['error'], result)
        self.assertEqual(result['answer'], 8)
        self.assertEqual(len(requests), 5)
        self.assertEqual(packet.call_count, 2)
        self.assertEqual(result['budget']['tool_calls_run'], 5)  # A2 + Binit1 + B2
        self.assertEqual(result['tokens']['llm_calls'], 5)
        self.assertTrue(all(x['model']=='arbitrary-model' and x['base_url']=='https://fixture.invalid' for x in requests))
        self.assertEqual(len(requests[2]['messages']), 2)
        self.assertNotIn('same-local-id', json.dumps(requests[2]['messages']))
        self.assertIn('same-local-id', json.dumps(requests[1]['messages'], default=lambda x:x.model_dump()))
        self.assertEqual(requests[-1]['max_completion_tokens'], 2000)
        self.assertIsNone(requests[-1]['tools'])
        snapshots=result['revision']['messages']
        self.assertEqual([s['candidate'] for s in snapshots], ['A','B'])
        self.assertEqual(snapshots[0]['diagnostics']['token_cap'], 200000)
        self.assertEqual(snapshots[0]['diagnostics']['tool_cap'], 30)
        self.assertEqual(snapshots[0]['diagnostics']['sdk_cap'], 30)
        self.assertTrue(all(s['artifact'] and s['source_check'] for s in snapshots))
        self.assertEqual(result['revision']['candidate_snapshot_sha256']['A'], fingerprint(snapshots[0]))
        self.assertIsNot(snapshots[0]['workspace'],snapshots[1]['workspace'])
        self.assertEqual(agent.total_usage['total_tokens'], 60)

    def test_exact_answer_agreement_preserves_actual_A_and_skips_selector(self):
        b=computation(operation="Different real B operation", limitation="Honest B limitation")
        result, requests, _, _=self.exercise([native(),prose(),native(),prose()], [computation(),b])
        self.assertEqual(result['revision']['status'],'agreement_preserved')
        self.assertEqual(len(requests),4)
        self.assertIsNone(result['limitation'])

    def test_B_worker_failure_can_receive_own_feedback_without_A_history(self):
        b=computation(8);b['evidence']=computation()['evidence']
        result, requests, _, _=self.exercise([native(),prose(),native(8),native(8,identifier='B-repair'),prose(),selector()],
            [computation(),{'error':'Synthetic worker contract failure'},b])
        self.assertEqual(result['answer'],8)
        self.assertIn('Synthetic worker contract failure',json.dumps(requests[3]['messages'], default=lambda x:x.model_dump()))
        self.assertEqual(result['tokens']['llm_calls'],6)
        self.assertEqual(result['budget']['tool_calls_run'],6)

    def test_nonfatal_prose_A_is_retained_as_unverified_and_can_be_replaced(self):
        b=computation(8);b['evidence']=computation()['evidence']
        result, requests, _, _=self.exercise([Message(role='assistant',content=json.dumps({'answer':9,'evidence':computation()['evidence'],'limitation':None})),native(8),prose(),selector()], [b])
        self.assertEqual(result['answer'],8)
        packet=json.loads(requests[-1]['messages'][1]['content'])
        self.assertEqual(packet['A']['verification'],'released_unverified')
        self.assertIsNone(packet['A']['program'])
        self.assertEqual(packet['A']['released']['answer'],9)

    def test_selector_B_without_contradicted_A_preserves_A(self):
        b=computation(8);b['evidence']=computation()['evidence']
        result,_,_,_=self.exercise([native(),prose(),native(8),prose(),selector(A='not_found')], [computation(),b])
        self.assertEqual(result['answer'],7)
        self.assertEqual(result['revision']['status'],'selector_preserved_A')

    def test_low_token_or_tool_remaining_skips_B_without_initialization(self):
        result,requests,packet,_=self.exercise([native(),prose()], [computation()], cap=36000)
        self.assertEqual(len(requests),2);self.assertEqual(packet.call_count,1)
        self.assertEqual(result['revision']['status'],'residual_coverage_skip')
        self.assertEqual(result['answer'],7)
        result,requests,packet,_=self.exercise([native(),prose()], [computation()],tools=4)
        self.assertEqual(len(requests),2);self.assertEqual(packet.call_count,1)
        self.assertEqual(result['budget']['tool_calls_run'],2)

    def test_unknown_B_usage_and_escaping_provider_are_fatal_with_attempt_retained(self):
        for response, bill in [(native(),usage(known=False,cost=None)),(RuntimeError('mock transport'),None)]:
            with self.subTest(response=type(response).__name__):
                bills=[usage(),usage(),bill] if bill else None
                result,requests,_,_=self.exercise([native(),prose(),response], [computation()],usages=bills)
                self.assertIsNotNone(result['error']);self.assertIsNone(result['answer'])
                self.assertEqual(result['tokens']['llm_calls'],3)
                self.assertTrue(result['revision']['fatal'])
                self.assertFalse(result['tokens']['per_call'][-1]['usage_available'])

    def test_actual_B_initialization_failure_is_charged_once_and_preserves_A(self):
        calls=[]
        def packets(*args,**kwargs):
            calls.append(1)
            if len(calls)==2:raise RuntimeError('synthetic init unavailable')
            return deepcopy(PACKET)
        result,requests,_,_=self.exercise([native(),prose()], [computation()],packet=packets)
        self.assertIsNone(result['error'])
        self.assertEqual(result['answer'],7);self.assertEqual(len(requests),2)
        self.assertEqual(result['budget']['tool_calls_run'],3)
        init=result['revision']['messages'][1]['diagnostics']['initialization']
        self.assertEqual((init['started'],init['charged'],init['completed']),(1,1,0))

    def test_initializer_overlay_keeps_actual_code_closure_globals_and_no_early_charge(self):
        with patch('agent.base_agent.setup_api_keys'):
            single=SingleAgent('arbitrary-model');host=ResidualTrajectoryAgent('arbitrary-model');host.set_budget(max_tool_calls=30)
        original=SingleAgent._run;globals_before=original.__globals__.copy()
        state=_observe_initialization(single,host)
        self.assertIs(single._run.__func__.__code__,original.__code__)
        self.assertEqual(single._run.__func__.__closure__,original.__closure__)
        self.assertEqual(original.__globals__,globals_before)
        self.assertEqual((state['started'],state['charged'],host.tool_calls_made),(0,0,0))

    def test_actual_A_request_prefix_matches_standalone_Single_full_policy(self):
        with patch('agent.residual_trajectory_agent.ENTRY_BYTES', 0):
            result, requests, _, _ = self.exercise([native(),prose()], [computation()])
        expected=[];pending=iter([native(),prose()])
        def provider(**kwargs):
            expected.append(deepcopy(kwargs));return next(pending),None,usage()
        with patch('agent.base_agent.safe_llm_call',side_effect=provider), \
                patch('agent.multi_turn_code_resource_agent.get_tool',return_value=lambda **kwargs:computation()):
            reference=SingleAgent('arbitrary-model',max_iterations=30,base_url='https://fixture.invalid')
            reference.set_budget(max_total_tokens=200000,max_tool_calls=30)
            actual=reference.run(QUESTION)
        encode=lambda v:json.dumps(v,default=lambda x:x.model_dump(),sort_keys=True)
        self.assertEqual(encode(requests),encode(expected))
        for key in ('answer','evidence','limitation','error'):
            self.assertEqual(result[key],actual[key])
        self.assertEqual(result['tokens'],actual['tokens'])

    def test_A_uses_all30_natural_SDK_calls_no_B_or_extra_initialization(self):
        responses=[Message(role='assistant',content='Continue reasoning') for _ in range(28)]+[native(),prose()]
        result,requests,packet,_=self.exercise(responses,[computation()])
        self.assertEqual(len(requests),30);self.assertEqual(packet.call_count,1)
        self.assertEqual(result['tokens']['llm_calls'],30)
        self.assertEqual(result['answer'],7)
        self.assertEqual(result['revision']['status'],'residual_coverage_skip')

    def test_B_malformed_cached_usage_stops_before_execution_and_is_fatal(self):
        malformed=usage();malformed['per_call'][0]['cached']='invalid-cache-count'
        result,requests,_,_=self.exercise([native(),prose(),native(8)],[computation()],
                                        usages=[usage(),usage(),malformed])
        self.assertIsNotNone(result['error']);self.assertTrue(result['revision']['fatal'])
        self.assertEqual(result['tokens']['llm_calls'],3)
        self.assertEqual(result['budget']['tool_calls_run'],3)  # A2+Binitialization; no B execution
        self.assertEqual(result['tokens']['per_call'][-1]['cached'],0)
        self.assertFalse(result['tokens']['per_call'][-1]['usage_available'])
        self.assertEqual(len(requests),3)

    def test_known_B_input_refusal_is_not_an_unknown_or_free_attempt(self):
        result,requests,_,_=self.exercise([native(),prose(),(None,'Input tokens exceeded: synthetic',None)],
                                       [computation()])
        self.assertIsNone(result['error']);self.assertEqual(result['answer'],7)
        self.assertEqual(result['tokens']['llm_calls'],2)
        self.assertEqual(result['budget']['tool_calls_run'],3)
        self.assertEqual(result['revision']['status'],'B_ineligible')

    def test_selector_completion_overshoot_and_unknown_are_fatal(self):
        b=computation(8);b['evidence']=computation()['evidence']
        for last in (usage(completion=2001),usage(known=False,cost=None)):
            with self.subTest(last=last['completion_tokens']):
                result,requests,_,_=self.exercise([native(),prose(),native(8),prose(),selector()],
                    [computation(),b],usages=[usage(),usage(),usage(),usage(),last])
                self.assertIsNone(result['answer']);self.assertIsNotNone(result['error'])
                self.assertEqual(result['tokens']['llm_calls'],5)
                self.assertTrue(result['revision']['fatal'])

    def test_bound_B_honest_limitation_is_not_deleted_or_a_special_veto(self):
        b=computation(8,limitation='Synthetic honest limitation');b['evidence']=computation()['evidence']
        result,requests,_,_=self.exercise([native(),prose(),native(8),prose(),selector()], [computation(),b])
        self.assertEqual(result['answer'],8)
        self.assertEqual(result['limitation'],'Synthetic honest limitation')
        self.assertEqual(json.loads(requests[-1]['messages'][1]['content'])['B']['released']['limitation'],result['limitation'])

    def test_B_computed_but_source_withheld_never_replaces_A(self):
        bad=computation(8)
        result,requests,_,_=self.exercise([native(),prose(),native(8),prose(),native(8,identifier='B-source-repair'),prose()],
                                        [computation(),bad,bad])
        self.assertEqual(result['answer'],7)
        self.assertEqual(result['revision']['status'],'B_ineligible')
        b=result['revision']['messages'][1]
        self.assertIsNone(b['artifact']);self.assertIsNone(b['released']['answer'])
        self.assertEqual(b['released']['revision']['final']['answer'],8)
        self.assertEqual(len(requests),6);self.assertEqual(result['budget']['tool_calls_run'],7)

    def test_oversized_A_entry_preserves_untruncated_A_without_B(self):
        a=Message(role='assistant',content=json.dumps({'answer':'x'*40000,'evidence':computation()['evidence'],'limitation':None}))
        result,requests,packet,_=self.exercise([a])
        self.assertEqual(result['answer'],'x'*40000)
        self.assertEqual(result['revision']['status'],'A_entry_coverage_skip')
        self.assertEqual(len(requests),1);self.assertEqual(packet.call_count,1)

    def test_oversized_B_entry_is_retained_in_snapshot_and_preserves_A(self):
        b=computation('x'*40000);b['evidence']=computation()['evidence']
        result,requests,_,_=self.exercise([native(),prose(),native(8),prose()], [computation(),b])
        self.assertEqual(result['answer'],7)
        self.assertEqual(result['revision']['status'],'B_entry_coverage_skip')
        self.assertEqual(result['revision']['messages'][1]['released']['answer'],'x'*40000)
        self.assertEqual(len(requests),4)

    def test_B_SDK_guard_refuses_before_an_extra_provider_invocation(self):
        with patch('agent.base_agent.setup_api_keys'),patch('agent.base_agent.count_tokens_in_messages',return_value=10), \
                patch('agent.base_agent.safe_llm_call',return_value=(prose(),None,usage())) as provider:
            single=SingleAgent('arbitrary-model');single.set_budget(max_total_tokens=5000,max_tool_calls=30)
            state=_guard_B_calls(single,1)
            single._llm_call([{'role':'user','content':'synthetic'}])
            response,error,reported=single._llm_call([{'role':'user','content':'synthetic'}])
        self.assertEqual(provider.call_count,1);self.assertEqual(single.total_usage['llm_calls'],1)
        self.assertIsNone(response);self.assertIsNone(reported)
        self.assertIn('Budget exceeded',error);self.assertFalse(state['fatal'])

    def test_finite_phase_cost_sum_overflow_is_fatal_and_audit_stays_JSON(self):
        b=computation(8);b['evidence']=computation()['evidence']
        result,requests,_,_=self.exercise([native(),prose(),native(8),prose()], [computation(),b],
                                       usages=[usage(cost=5e307) for _ in range(4)])
        self.assertIsNotNone(result['error']);self.assertTrue(result['revision']['fatal'])
        self.assertIsNone(result['tokens']['cost']);self.assertEqual(len(requests),4)
        self.assertEqual(result['tokens']['llm_calls'],4)
        self.assertEqual(result['revision']['messages'][0]['usage']['total_usage']['cost'],1e308)
        self.assertEqual(result['revision']['messages'][1]['usage']['total_usage']['cost'],1e308)
        json.dumps(result,allow_nan=False)

    def test_strict_selector_duplicate_nonfinite_tools_and_no_evidence_rejected(self):
        for text in ['{"choice":"A","choice":"B"}', '{"choice":NaN}', '{}']:
            with self.assertRaises(ValueError):selection_object(Message(role='assistant',content=text))
        m=selector();m.tool_calls=native().tool_calls
        with self.assertRaises(ValueError):selection_object(m)
        with self.assertRaises(ValueError):selection_object(selector(requirements=[]))


if __name__=='__main__':unittest.main()
