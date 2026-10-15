"""Metadata-only fresh-cohort selection; no provider or QA/reference imports."""
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('v8_selection_under_test', ROOT / 'scripts/v8_mechanism_study.py')
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def metadata_fixture():
    questions, families = [], []
    for group, quota in study.QUOTAS.items():
        for index in range(quota * 3):
            qid = group + '_' + str(index)
            questions.append({'question_id': qid, 'patient_fhir_id': 'p' + str(index % 20)})
            other = group in ('duration', 'paired_comparison', 'paired_difference')
            families.append({'question_id': qid, 'family': 'out_of_scope' if other else group,
                             'excluded_reason': group if other else '', 'template_id': group})
    dev_ids = [row['question_id'] for row in questions]
    for qid in ('adaptive', 'confirmation'):
        questions.append({'question_id': qid, 'patient_fhir_id': qid + '_patient'})
        families.append({'question_id': qid, 'family': 'lookup_counting', 'excluded_reason': '',
                         'template_id': 'reserved_elsewhere'})
    split = {'splits': {'dev': {'patients': [{'patient_fhir_id': 'p' + str(i)} for i in range(20)]},
                        'pilot': {'patients': []}, 'held_out': {'question_ids': ['adaptive']}}}
    exposure = {'splits': {'development_patients': {'question_ids': dev_ids},
                           'pilot_patients': {'question_ids': []},
                           'adaptive_questions': {'question_ids': ['adaptive']},
                           'question_confirmation': {'question_ids': ['confirmation']}}}
    return questions, families, split, exposure


class FreshStudySelectionTests(unittest.TestCase):
    def test_prior_scheduled_ids_excluded_before_balanced_selection(self):
        inputs = metadata_fixture()
        old = study.select_manifest(*inputs, seed=5108)
        old_ids = {row['question_id'] for row in old['cases']}
        fresh = study.select_manifest(*inputs, seed=5109, excluded_study_ids=old_ids)
        fresh_ids = {row['question_id'] for row in fresh['cases']}
        self.assertEqual(len(fresh_ids), 120)
        self.assertFalse(fresh_ids & (old_ids | {'adaptive', 'confirmation'}))
        self.assertEqual(fresh['operation_counts'], study.QUOTAS)
        self.assertEqual(fresh['excluded_previous_study_questions'], 120)
        self.assertEqual(len(study.validate_manifest(fresh)), 120)
        self.assertEqual(fresh, study.select_manifest(*inputs, seed=5109, excluded_study_ids=old_ids))

    def test_exclusion_cannot_modify_other_partitions_or_hide_unknown_ids(self):
        for invalid in ('adaptive', 'confirmation', 'missing'):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'reserved development pool'):
                study.select_manifest(*metadata_fixture(), excluded_study_ids={invalid})

    def test_insufficient_fresh_stratum_fails_without_relaxing_quotas(self):
        inputs = metadata_fixture()
        exclusions = {row['question_id'] for row in inputs[1] if row['excluded_reason'] == 'duration'}
        with self.assertRaisesRegex(ValueError, 'Insufficient metadata candidates for duration'):
            study.select_manifest(*inputs, excluded_study_ids=exclusions)

    def test_default_and_empty_exclusions_preserve_original_selection(self):
        inputs = metadata_fixture()
        default = study.select_manifest(*inputs)
        self.assertEqual(default, study.select_manifest(*inputs, excluded_study_ids=[]))
        self.assertNotIn('excluded_previous_study_questions', default)


if __name__ == '__main__':
    unittest.main()
