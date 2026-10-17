"""Private offline regression suite; no live writes or provider calls."""
import csv
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/private/tmp/v3-provider-rejection-review')
spec = importlib.util.spec_from_file_location('private_v4_runner', Path(__file__).with_name('v3_full_run.py'))
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
candidate.ROOT = ROOT
from tests import test_v3_full_run as original
from tests import test_v3_full_analysis as analysis_tests
from tests import test_v3_provider_jwt as jwt_tests
from test_independent_predicate import payload, exclusion_cases, row as predicate_row
from independent_predicate import is_observed_provider_prompt_rejection as independent_predicate

original.runner = candidate
jwt_tests.classify_candidate = candidate.classify_attempt


def rejected(qid='q0', patient='p0', answer=None):
    value = original.row(answer, candidate.SDK_ERROR_PREFIX + repr(payload()),
                         trace=[{'role': 'system', 'content': 'Synthetic unchanged prompt'}])
    value.update(question_id=qid, patient_fhir_id=patient)
    value['tokens']['per_call'] = [{'prompt': 0, 'completion': 0, 'cached': 0, 'usage_available': False}]
    value['tokens']['total'] = 0
    return value


class PredicateTests(unittest.TestCase):
    def test_independent_exclusion_suite_matches_embedded_rule(self):
        for name, value in exclusion_cases():
            with self.subTest(case=name):
                self.assertFalse(candidate.is_observed_provider_prompt_rejection(value))
                self.assertEqual(candidate.is_observed_provider_prompt_rejection(value), independent_predicate(value))

    def test_provider_decision_is_answer_blind_and_explicitly_enabled(self):
        for answer in (None, 'wrong', False, 0, [1, 2], {'error': 'unrelated'}):
            with self.subTest(answer=answer):
                value = rejected(answer=answer)
                self.assertEqual(candidate.classify_attempt(value)['status'], 'attention')
                decision = candidate.classify_attempt(value, allow_provider_rejection=True)
                self.assertEqual(decision['status'], 'runtime')
                self.assertEqual(decision['reason'], 'provider_policy_rejection')
                self.assertEqual(decision['category'], 'provider_policy_rejection')

    def test_model_owned_echo_and_generic_http400_never_enable_provider_recovery(self):
        value = original.row(candidate.SDK_ERROR_PREFIX + repr(payload()))
        value['trace'] = [{'role': 'assistant', 'content': value['agent_answer']},
                          {'role': 'tool', 'name': 'execute_python_code', 'content': {'error': value['agent_answer']}}]
        value['revision'] = {'contract': {'error': value['agent_answer']}, 'audit': {'error': value['agent_answer']}}
        self.assertEqual(candidate.classify_attempt(value, allow_provider_rejection=True)['status'], 'accepted')
        value = rejected()
        value['error'] = "Max retries exceeded: Error code: 400 - {'error': {'code': 'other'}}"
        self.assertEqual(candidate.classify_attempt(value, allow_provider_rejection=True)['status'], 'attention')


class RecoveryTests(original.SupervisorTests):
    # Inherited methods are exercised once by the original module's suite.
    def setUp(self):
        super().setUp()
        with self.input.open(newline='') as stream:
            records = list(csv.DictReader(stream))
        records.append({**records[0], 'question_id': 'q2', 'patient_fhir_id': 'p2'})
        with self.input.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
        with self.families.open('a') as stream:
            stream.write('q2,fixture,fixture\n')
        self.patients['q2'] = 'p2'
        args = candidate.build_parser().parse_args(['run', '--run-dir', str(self.directory), '--input', str(self.input),
                '--families', str(self.families), '--min-free-gb', '0', '--retry-delay-seconds', '0', '--batch-size', '2'])
        with patch('scripts.full_benchmark.QUESTIONS', 3):
            self.manifest = candidate.manifest_for(args)
        self.manifest['analysis_command'] = ['fixture-analysis']
        candidate.atomic_json(self.directory / 'run_manifest.json', self.manifest)

    def prepare_selected_resume(self):
        super().prepare_selected_resume()
        self.manifest['provider_rejection_policy'] = deepcopy(candidate.PROVIDER_REJECTION_POLICY)
        candidate.atomic_json(self.directory / 'run_manifest.json', self.manifest)

    def write_rejection(self, batch, rejected_ids):
        output = Path(batch['output'])
        self.write_batch_results(batch)
        records = json.loads(output.read_text())
        for index, value in enumerate(records):
            qid = value['question_id']
            if qid in rejected_ids:
                records[index] = original.externalize_result_audit(rejected(qid, self.patients[qid]), output.parent, question_id=qid)
        candidate.atomic_json(output, records)

    def assert_parked_unchanged(self, before):
        self.assertEqual(self.parked_evidence(), before)

    def test_new_policy_requires_explicit_exact_manifest_optin(self):
        self.prepare_selected_resume()
        correct = deepcopy(self.manifest)
        absent = deepcopy(correct)
        absent.pop('provider_rejection_policy')
        self.assertEqual(candidate.execution_arms(absent, ['single_agent'], resume=True), ('single_agent',))
        self.assertEqual(candidate.classifier_for_manifest(absent)(rejected())['status'], 'attention')
        for policy in (None, {}, {**candidate.PROVIDER_REJECTION_POLICY, 'max_rejections_per_question': 20},
                       {**candidate.PROVIDER_REJECTION_POLICY, 'retry_request': 'rewrite'}):
            value = deepcopy(correct)
            value['provider_rejection_policy'] = policy
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, 'Selected provider-rejection'):
                candidate.execution_arms(value, ['single_agent'], resume=True)
        with self.assertRaisesRegex(ValueError, 'unscoped'):
            candidate.execution_arms({'provider_rejection_policy': deepcopy(candidate.PROVIDER_REJECTION_POLICY)})

    def test_one_unchanged_retry_then_park_and_finish_fresh_questions_across_resume(self):
        self.prepare_selected_resume()
        # Setting the generic limit to zero proves provider rejections have their
        # own two-attempt budget rather than consuming generic retry capacity.
        self.manifest['settings']['max_runtime_retries'] = 0
        candidate.atomic_json(self.directory / 'run_manifest.json', self.manifest)
        path, batch = candidate.create_batch(self.directory, 'single_agent', {'q0'}, self.manifest, 1)
        self.write_rejection(batch, {'q0'}); candidate.seal_batch(path, 0, 'fixture')
        parked = self.parked_evidence(); launches = []
        class Child:
            returncode = 0
            pid = 999999
            def poll(self): return 0
            def wait(self, timeout=None): return 0
        def launch(command, **kwargs):
            self.assertEqual(command[command.index('--agent_strategy') + 1], 'single_agent')
            output = Path(command[command.index('--output') + 1])
            batch = json.loads(output.with_suffix('.batch.json').read_text())
            self.assertEqual(batch['run_identity']['code_hash'], self.manifest['run_identities']['single_agent']['code_hash'])
            launches.append(batch['question_ids'])
            self.write_rejection(batch, {'q0'})
            return Child()
        self.assertEqual(self.selected_run(launch), 2)
        self.assertEqual(launches, [['q0', 'q1'], ['q2']])
        self.assert_parked_unchanged(parked)
        ledger = json.loads((self.directory / 'acceptance.json').read_text())
        self.assertEqual(set(ledger['arms']['single_agent']), {'q1', 'q2'})
        self.assertEqual(ledger['counts']['single_agent'], {'raw_attempts': 4, 'runtime_failures': 2, 'intrinsic_errors': 0})
        self.assertFalse(ledger['complete'])
        self.assertEqual({row['agent_answer'] for row in json.loads((self.directory / 'full_single_agent.json').read_text())}, {'wrong'})
        state = json.loads((self.directory / 'supervisor_state.json').read_text())
        self.assertEqual(state['reason'], 'provider_policy_review_required')
        self.assertFalse(state['selected_scope_complete'])
        derived = json.loads((self.directory / 'provider_rejections.json').read_text())
        self.assertEqual(derived['rejections'], {'single_agent': {'q0': 2}})
        self.assertEqual(derived['parked_question_ids'], {'single_agent': ['q0']})
        # The derived counter is never the resume authority: tampering/resetting
        # it cannot purchase another provider retry.
        candidate.atomic_json(self.directory / 'provider_rejections.json', {'rejections': {'single_agent': {}}})
        with patch.object(candidate.subprocess, 'Popen') as child:
            self.assertEqual(self.selected_run(child), 2)
        child.assert_not_called()
        self.assertEqual(json.loads((self.directory / 'provider_rejections.json').read_text())['rejections'], {'single_agent': {'q0': 2}})
        self.assert_parked_unchanged(parked)

    def test_successful_unchanged_retry_retains_wrong_terminal_and_original_clean_choice(self):
        self.prepare_selected_resume()
        path, batch = candidate.create_batch(self.directory, 'single_agent', {'q0', 'q1'}, self.manifest, 1)
        self.write_rejection(batch, {'q0'}); candidate.seal_batch(path, 0, 'fixture')
        before = self.parked_evidence(); launches = []
        class Child:
            returncode = 0
            pid = 999999
            def poll(self): return 0
            def wait(self, timeout=None): return 0
        def launch(command, **kwargs):
            output = Path(command[command.index('--output') + 1])
            batch = json.loads(output.with_suffix('.batch.json').read_text())
            launches.append(batch['question_ids']); self.write_batch_results(batch)
            return Child()
        self.assertEqual(self.selected_run(launch), 0)
        self.assertEqual(launches, [['q0', 'q2']])
        ledger = json.loads((self.directory / 'acceptance.json').read_text())
        self.assertEqual(ledger['arms']['single_agent']['q1']['round'], 1)
        self.assertEqual(ledger['arms']['single_agent']['q0']['runtime_failures_before'], 1)
        self.assertEqual(set(ledger['arms']['single_agent']), set(self.patients))
        self.assertFalse(ledger['complete'])
        derived = json.loads((self.directory / 'provider_rejections.json').read_text())
        self.assertEqual(derived['rejections']['single_agent'], {'q0': 1})
        self.assertEqual(derived['parked_question_ids']['single_agent'], [])
        self.assertEqual(derived['pending_question_ids']['single_agent'], [])
        self.assert_parked_unchanged(before)

    def test_historical_third_clean_attempt_after_two_rejections_is_rejected(self):
        self.prepare_selected_resume()
        for number in (1, 2, 3):
            path, batch = candidate.create_batch(self.directory, 'single_agent', {'q0'}, self.manifest, number)
            self.write_rejection(batch, {'q0'} if number < 3 else set())
            candidate.seal_batch(path, 0, 'fixture')
        parked = self.parked_evidence()
        # The public replay API must enforce the budget even when the caller
        # does not request a derived provider counter dictionary.
        with self.assertRaisesRegex(ValueError, 'forbids scheduling a parked question'):
            candidate.collect_accepted(self.directory, self.manifest, self.patients, arms=('single_agent',))
        with patch.object(candidate.subprocess, 'Popen') as child:
            self.assertEqual(self.selected_run(child), 2)
        child.assert_not_called()
        self.assert_parked_unchanged(parked)
        self.assertFalse((self.directory / 'provider_rejections.json').exists())


if __name__ == '__main__':
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite(loader.loadTestsFromModule(module) for module in (original, analysis_tests, jwt_tests))
    suite.addTests(loader.loadTestsFromTestCase(PredicateTests))
    for name in ('test_new_policy_requires_explicit_exact_manifest_optin',
                 'test_one_unchanged_retry_then_park_and_finish_fresh_questions_across_resume',
                 'test_successful_unchanged_retry_retains_wrong_terminal_and_original_clean_choice',
                 'test_historical_third_clean_attempt_after_two_rejections_is_rejected'):
        suite.addTest(RecoveryTests(name))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    raise SystemExit(not result.wasSuccessful())
