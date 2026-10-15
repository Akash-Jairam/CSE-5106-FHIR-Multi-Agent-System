"""Private V7 boundary tests: synthetic Single/provider/tools only."""
from copy import deepcopy
import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
REPO=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
sys.path.insert(0,str(REPO))
sys.path.insert(1,str(REPO/'.local-setup/phase2-dependencies'))
from litellm import Message
from agent.residual_trajectory_agent import ResidualTrajectoryAgent, candidate_entry
from agent.residual_trajectory_phase import clean_ledger, merge_ledger, run_single_phase, selector_call, validated_ledger, json_text, reserve
import tests.test_residual_trajectory_core as core
from tests.test_residual_trajectory_core import native, prose, selector
from tests.test_continuous_primary import computation, source_check, usage

class IndependentBoundaries(unittest.TestCase):
    exercise=core.V7CoreTests.exercise
    @staticmethod
    def different(value=8, **kw):
        result=computation(value,**kw)
        result['evidence']=computation()['evidence']
        return result
    def test_exact_three_residual_tools_counts_initialization_and_two_native_tools(self):
        result,requests,packet,_=self.exercise([native(),prose(),native(8),prose(),selector()],
            [computation(),self.different()],tools=5)
        self.assertEqual(result['answer'],8,result)
        self.assertEqual(result['budget']['tool_calls_run'],5)
        self.assertEqual(result['revision']['admission']['B_natural_tool_cap'],2)
        self.assertEqual(packet.call_count,2)
        self.assertEqual(len(requests),5)
    def test_exact_sdk_residual_27_A_plus_two_B_plus_selector(self):
        responses=[Message(role='assistant',content='Natural continuation') for _ in range(25)]
        responses += [native(),prose(),native(8),prose(),selector()]
        result,requests,packet,_=self.exercise(responses,[computation(),self.different()])
        self.assertEqual(result['answer'],8,result)
        self.assertEqual(result['tokens']['llm_calls'],30)
        self.assertEqual(len(requests),30)
        self.assertEqual(result['revision']['admission']['B_sdk_cap'],2)
        self.assertEqual(packet.call_count,2)
    def test_28_A_requests_skip_B_and_do_not_charge_initialization(self):
        responses=[Message(role='assistant',content='Natural continuation') for _ in range(26)] + [native(),prose()]
        result,requests,packet,_=self.exercise(responses,[computation()])
        self.assertEqual(result['answer'],7)
        self.assertEqual(result['revision']['status'],'residual_coverage_skip')
        self.assertEqual((len(requests),packet.call_count,result['budget']['tool_calls_run']),(28,1,2))
    def test_positive_B_allowance_does_not_guarantee_admissible_first_prompt(self):
        reservation=reserve(10+32768+512)+2000
        a_tokens=2*usage()['total_tokens']
        result,requests,packet,_=self.exercise([native(),prose()],[computation()],cap=reservation+a_tokens+1)
        self.assertEqual(result['answer'],7,result)
        self.assertEqual(result['revision']['status'],'B_ineligible')
        self.assertEqual(result['revision']['admission']['B_token_cap'],1)
        self.assertEqual((len(requests),packet.call_count,result['budget']['tool_calls_run']),(2,2,3))
    def test_exact_reserved_tokens_skips_before_B_init(self):
        reservation=reserve(10+32768+512)+2000
        result,requests,packet,_=self.exercise([native(),prose()],[computation()],cap=reservation+2*usage()['total_tokens'])
        self.assertEqual(result['revision']['status'],'residual_coverage_skip')
        self.assertEqual((len(requests),packet.call_count,result['budget']['tool_calls_run']),(2,1,2))
    def test_invalid_B_context_never_reaches_packet_and_has_no_init_charge(self):
        with patch('agent.base_agent.setup_api_keys'), patch('agent.single_agent.get_starting_packet') as packet:
            host=ResidualTrajectoryAgent('arbitrary-model');host.set_budget(max_total_tokens=200000,max_tool_calls=30)
            host._v7_merged_phases=set()
            snap,diag=run_single_phase(host,'not a valid contextual question','natural_alternative',1000,2,2,initialization_charge=True)
        self.assertFalse(diag['fatal'])
        self.assertEqual(diag['initialization']['started'],0)
        self.assertEqual(diag['initialization']['charged'],0)
        self.assertEqual(host.tool_calls_made,0)
        self.assertEqual(host.total_usage['llm_calls'],0)
        self.assertIsNone(snap['artifact'])
        self.assertIsNotNone(snap['released']['error'])
        packet.assert_not_called()
    def test_B_computed_but_twice_withheld_is_never_selected(self):
        checks=[]
        def check(citations,**kwargs):
            checks.append(1)
            value=source_check(citations,**kwargs)
            if len(checks)>1:value['issues']=['synthetic original source mismatch']
            return value
        result,requests,_,_=self.exercise([native(),prose(),native(8,'B-first'),prose(),native(9,'B-revised'),prose()],
            [computation(),self.different(8),self.different(9)],check=check)
        self.assertEqual(result['answer'],7,result)
        self.assertEqual(result['revision']['status'],'B_ineligible')
        b=result['revision']['messages'][1]
        self.assertIsNone(b['released']['answer'])
        self.assertEqual(b['released']['revision']['final']['answer'],9)
        self.assertIsNone(b['artifact'])
        self.assertEqual(len(requests),6)
        self.assertEqual(len(checks),3)
    def test_selected_B_honest_limitation_is_unchanged(self):
        result,requests,_,_=self.exercise([native(),prose(),native(8),prose(),selector()],
            [computation(),self.different(limitation='Synthetic available records may be incomplete')])
        self.assertEqual(result['answer'],8,result)
        self.assertEqual(result['limitation'],'Synthetic available records may be incomplete')
        packet=json.loads(requests[-1]['messages'][1]['content'])
        self.assertEqual(packet['B']['released']['limitation'],result['limitation'])
        self.assertEqual(packet['B']['artifact']['limitation'],result['limitation'])
    def _base_entry_length(self,label):
        with patch('agent.residual_trajectory_agent.ENTRY_BYTES',0):
            result,_,_,_=self.exercise([native(),prose()],[computation(operation='x')])
        snap=result['revision']['messages'][0]
        trace=[m for m in result['trace'] if m.get('phase')=='natural_primary']
        return len(json_text(candidate_entry(snap,trace)).encode())
    def test_A_full_ascii_byte_boundary_not_character_count(self):
        base=self._base_entry_length('A')
        for target in (32768,32769):
            with self.subTest(target=target):
                a=computation(operation='x'*(1+target-base))
                result,requests,packet,_=self.exercise([native(),prose(),native(8),prose(),selector()], [a,self.different()])
                if target==32768:
                    self.assertEqual(result['revision']['admission']['A_entry_bytes'],32768)
                    self.assertEqual(result['answer'],8,result)
                    self.assertEqual(packet.call_count,2)
                else:
                    self.assertEqual(result['revision']['status'],'A_entry_coverage_skip')
                    self.assertEqual((len(requests),packet.call_count),(2,1))
        a=computation(limitation='é'*6000)
        result,requests,packet,_=self.exercise([native(),prose()],[a])
        self.assertEqual(result['revision']['status'],'A_entry_coverage_skip')
        self.assertEqual(result['limitation'],'é'*6000)
        self.assertEqual((len(requests),packet.call_count),(2,1))
    def test_B_oversized_packet_falls_back_without_erasing_B_cost(self):
        b=self.different(operation='x'*33000)
        result,requests,packet,_=self.exercise([native(),prose(),native(8),prose()],[computation(),b])
        self.assertEqual(result['answer'],7,result)
        self.assertEqual(result['revision']['status'],'B_entry_coverage_skip')
        self.assertEqual((len(requests),packet.call_count,result['tokens']['llm_calls']),(4,2,4))
        self.assertEqual(result['budget']['tool_calls_run'],5)
    def test_non_supported_or_tied_selector_cannot_switch(self):
        cases=[('B','contradicted','not_found'),('B','contradicted','contradicted'),('B','supported','supported'),('UNKNOWN','contradicted','supported'),('A','contradicted','supported')]
        for choice,astatus,bstatus in cases:
            with self.subTest(case=(choice,astatus,bstatus)):
                result,requests,_,_=self.exercise([native(),prose(),native(8),prose(),selector(choice,astatus,bstatus)], [computation(),self.different()])
                self.assertEqual(result['answer'],7,result)
                self.assertEqual(result['tokens']['llm_calls'],5)
                self.assertFalse(result['revision']['independent_candidate_selected'])
    def test_selector_native_tools_rejected_after_known_usage(self):
        response=selector();response.tool_calls=native(99,'selector-native').tool_calls
        result,requests,_,_=self.exercise([native(),prose(),native(8),prose(),response],[computation(),self.different()])
        self.assertEqual(result['answer'],7,result)
        self.assertEqual(result['revision']['status'],'selector_malformed_or_unavailable')
        self.assertEqual(result['tokens']['llm_calls'],5)
        self.assertFalse(result['revision']['fatal'])
    def test_unknown_marker_merge_once_preserves_cost_unknown(self):
        local=clean_ledger();local.total_usage.update(llm_calls=1,cost=None)
        local.call_usage=[{'prompt':0,'completion':0,'cached':0,'usage_available':False}]
        scratch,raw,problems,unknown=validated_ledger(local)
        self.assertEqual(problems,[]);self.assertTrue(unknown)
        host=clean_ledger();host._v7_merged_phases=set()
        self.assertEqual(merge_ledger(host,scratch,'B'),[])
        before=deepcopy(host.total_usage)
        with self.assertRaises(ValueError):merge_ledger(host,scratch,'B')
        self.assertEqual(host.total_usage,before)
        self.assertEqual(host.total_usage['llm_calls'],1)
        self.assertIsNone(host.total_usage['cost'])
        self.assertFalse(host.call_usage[0]['usage_available'])
    def test_malformed_selector_post_provider_usage_is_fatal_not_parse_fallback(self):
        for field,value in [('cached','bad'),('prompt',True),('completion',-1)]:
            with self.subTest(field=field):
                bill=usage();bill['per_call'][0][field]=value
                result,requests,_,_=self.exercise([native(),prose(),native(8),prose(),selector()],
                    [computation(),self.different()],usages=[usage(),usage(),usage(),usage(),bill])
                self.assertIsNotNone(result['error'],result)
                self.assertTrue(result['revision']['fatal'])
                self.assertEqual(result['tokens']['llm_calls'],5)
                self.assertIsNone(result['tokens']['cost'])
                self.assertFalse(result['tokens']['per_call'][-1]['usage_available'])

if __name__=='__main__':unittest.main(verbosity=2)
