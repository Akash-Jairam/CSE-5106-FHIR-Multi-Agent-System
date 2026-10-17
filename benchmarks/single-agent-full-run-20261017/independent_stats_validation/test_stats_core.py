"""Synthetic-only independent-statistics edge cases; never opens a benchmark."""
import importlib.util
import json
import math
from pathlib import Path
import unittest

PATH = Path('/private/tmp/v3-single-independent-stats/reconcile_final_pair.py')
spec = importlib.util.spec_from_file_location('qa', PATH)
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


class SyntheticStats(unittest.TestCase):
    def test_complete_sample(self):
        result = qa.measure([1,2,3,4])
        self.assertEqual(result['total'],10)
        self.assertEqual(result['mean'],2.5)
        self.assertEqual(result['median'],2.5)
        self.assertAlmostEqual(result['p95'],3.85)
        self.assertAlmostEqual(result['stdev'],math.sqrt(5/3))

    def test_missingness_never_becomes_zero(self):
        result = qa.measure([1,None,3])
        self.assertEqual(result['missing_n'],1)
        self.assertIsNone(result['total'])
        self.assertIsNone(result['mean'])
        self.assertEqual(result['observed_only']['total'],4)
        self.assertEqual(result['observed_only']['mean'],2)

    def test_empty_measure(self):
        result = qa.measure([])
        self.assertEqual(result['n'],0)
        self.assertIsNone(result['total'])
        self.assertIsNone(result['observed_only']['mean'])

    def test_invalid_values_missing(self):
        result = qa.measure([False,-1,float('inf'),float('nan'),2])
        self.assertEqual(result['observed_n'],1)
        self.assertEqual(result['missing_n'],4)
        self.assertIsNone(result['observed_only']['stdev'])

    def test_one_sample_sd_unavailable(self):
        self.assertIsNone(qa.measure([0])['stdev'])

    def test_cache_subset_no_double_charge(self):
        rates = {'input':.1,'cached_input':.01,'output':.5}
        self.assertAlmostEqual(qa.cost({'prompt':120,'cached_prompt':30,'completion':20},rates),.0000193)
        self.assertNotEqual(qa.cost({'prompt':120,'cached_prompt':30,'completion':20},rates),.0000223)

    def test_unknown_usage_has_no_dollar_estimate(self):
        tokens = {'prompt':0,'cached_prompt':0,'completion':0,'per_call':[{'usage_available':False}]}
        self.assertIsNone(qa.cost(tokens,{'input':.1,'cached_input':.01,'output':.5}))

    def test_invalid_cached_subset_unknown(self):
        self.assertIsNone(qa.cost({'prompt':20,'cached_prompt':30,'completion':1},{'input':.1,'cached_input':.01,'output':.5}))

    def test_missing_price_unknown(self):
        self.assertIsNone(qa.cost({'prompt':20,'cached_prompt':10,'completion':1},{'input':.1,'cached_input':.01}))

    def test_one_patient_interval_unavailable(self):
        self.assertEqual(qa.cluster_interval([1,0],['p','p']),{'n':2,'patients':1,'estimate':.5,'ci_low':None,'ci_high':None})

    def test_cluster_question_weight_not_patient_average(self):
        result = qa.cluster_interval([1,1,0],['a','a','b'])
        self.assertAlmostEqual(result['estimate'],2/3)
        self.assertEqual(result['ci_low'],0)
        self.assertEqual(result['ci_high'],1)

    def test_paired_sign_preserved(self):
        result = qa.cluster_interval([-1,-1,0],['a','a','b'])
        self.assertAlmostEqual(result['estimate'],-2/3)
        self.assertEqual(result['ci_low'],-1)
        self.assertEqual(result['ci_high'],0)

    def test_unknown_usage_masks_tokens(self):
        self.assertEqual(qa.metric_values([{'provider_usage_unknown':True,'total_tokens':20}],'total_tokens'),[None])
        self.assertEqual(qa.metric_values([{'provider_usage_unknown':True,'latency_seconds':2}],'latency_seconds'),[2])

    def test_reproducible_patient_bootstrap(self):
        self.assertEqual(qa.cluster_interval([1,3,5,7],['a','b','c','a']),qa.cluster_interval([1,3,5,7],['a','b','c','a']))


if __name__ == '__main__':
    unittest.main()
