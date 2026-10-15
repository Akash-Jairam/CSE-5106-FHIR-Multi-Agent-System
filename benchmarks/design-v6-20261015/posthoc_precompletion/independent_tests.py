"""Actual frozen V6/Single/Base normalization; all records/provider replies synthetic."""
import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path
sys.path.insert(0,'/private/tmp/fhir-v6-posthoc-sj4zhs4u')
import test_v6_posthoc as t
from tests.test_continuous_primary import ContinuousPrimaryTests, final, prose, reply, computation, usage, QUESTION
from agent.independent_recompute_agent import IndependentRecomputeAgent
from utils.answer_scorer import score_answer
from utils import run_agent_safe
from litellm import Message

class FrozenNormalization(unittest.TestCase):
    def actual_row(self,responses,executions=(),usages=None,cap=200000):
        harness=ContinuousPrimaryTests();self.addCleanup(harness.doCleanups)
        responses=[Message(role='assistant',content=r.content,tool_calls=[{'id':c.id,'type':'function',
          'function':{'name':c.function.name,'arguments':c.function.arguments}} for c in r.tool_calls] if r.tool_calls else None) for r in responses]
        host,requests,_,_=harness.environment(responses,executions,usages)
        agent=IndependentRecomputeAgent('synthetic');agent.set_budget(max_total_tokens=cap,max_tool_calls=30)
        output=run_agent_safe(agent,QUESTION,include_full_audit=True)
        output['question_id']='synthetic-actual-normalization'
        return output,requests

    def score_record(self,value,reference):
        proof=t.m.proof_flags(value,t.guard_fixture.m)
        def score(answer):
            return score_answer(answer,[] if reference is None else [[reference]],'What is the synthetic value?','')
        return t.m.row_record(value,score,proof)

    def test_actual_known_budget_null_stays_nonerror_credit(self):
        value,requests=self.actual_row([],cap=0)
        self.assertEqual(requests,[]);self.assertIsNone(value['error'])
        item=self.score_record(value,None)
        self.assertTrue(item['a_only_available']);self.assertTrue(item['a_only_correct'])
        self.assertFalse(item['a_only_error']);self.assertTrue(item['primary_known_budget_stop'])
        self.assertFalse(item['primary_source_bound']);self.assertTrue(item['a_exact_None_typed_credit'])

    def test_actual_source_withholding_uses_released_A_not_correct_computed_final(self):
        value,_=self.actual_row([final(value=99),prose(99),final('again',99),prose(99)],
          [computation(99),computation(99)])
        self.assertIsNone(value['agent_answer']);self.assertIsNone(value['error'])
        self.assertEqual(value['revision']['primary_revision']['final']['answer'],99)
        item=self.score_record(value,99)
        self.assertTrue(item['primary_prewithholding_candidate_correct'])
        self.assertFalse(item['a_only_correct']);self.assertFalse(item['correct'])
        self.assertTrue(item['primary_source_revised'])
        self.assertEqual(item['primary_source_revision_effect_to_released_A'],'harmful')
        self.assertTrue(item['primary_source_withholding_lost_typed_credit'])
        self.assertFalse(item['independent_candidate_selected'])

    def test_actual_unknown_usage_auxiliary_loss_does_not_retroactively_error_A(self):
        value,requests=self.actual_row([final(),prose(),reply('unused auxiliary response')],[computation()],
          [usage(),usage(),usage(0,0,known=False,cost=None)])
        self.assertEqual(len(requests),3)
        self.assertIsNone(value['agent_answer']);self.assertTrue(value['error'])
        self.assertEqual(value['revision']['primary']['answer'],7)
        item=self.score_record(value,7)
        self.assertTrue(item['primary_source_bound']);self.assertTrue(item['a_only_correct'])
        self.assertFalse(item['a_only_error']);self.assertFalse(item['correct'])
        self.assertTrue(item['auxiliary_fatal_recorded']);self.assertTrue(item['unknown_usage_recorded'])
        self.assertEqual(item['auxiliary_stage_effect'],'harmful')
        self.assertEqual(item['released_B_effect'],'not_applied')
        self.assertTrue(item['auxiliary_non_switch_credit_lost'])

if __name__=='__main__':
    path=Path('/private/tmp/fhir-v6-posthoc-sj4zhs4u/v6_posthoc.py')
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    assert before=='ca0c3820298d7595e5fe062b41c24c6303b7938f82530ef2dc67004be0c2db8a'
    suite=unittest.defaultTestLoader.loadTestsFromName('test_v6_posthoc')
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(FrozenNormalization))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    sys.exit(not result.wasSuccessful())
