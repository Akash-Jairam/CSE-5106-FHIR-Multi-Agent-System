"""Independent V7 posthoc synthetic edge/gate probes only; no actual inputs."""
from copy import deepcopy
import hashlib,importlib.util,sys,unittest
from pathlib import Path
from unittest.mock import Mock
P=Path('/private/tmp/fhir-v7-posthoc-candidate')
assert hashlib.sha256((P/'v7_posthoc.py').read_bytes()).hexdigest()=='4e3c460fba42855715ebef94d064b9f8c9cca827d94122042ce113dc29d3a6d4'
spec=importlib.util.spec_from_file_location('v7_synthetic_posthoc_fixtures',P/'test_v7_posthoc.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);m=t.M
scorer=m.module(P/'fixtures/answer_scorer.py','private_synthetic_exact_typed_scorer')
def actual_record(row,reference='[]'):
    return m.row_record(row,lambda value:scorer.score_answer(value,reference,'Synthetic question','Synthetic question'),m.compact_verified(row))

class IndependentPosthoc(unittest.TestCase):
    fixture=t.CompleteProofWire.fixture
    def test_actual_scorer_known_budget_null_is_preserved_without_sourceproof(self):
        row=t.row(a=None,final=None);a=row['revision']['messages'][0]
        a['artifact']=None;a['released']['limitation']='Stopped by the actual token budget'
        a['diagnostics']['budget_stop']='token budget exhausted'
        rec=actual_record(row)
        self.assertTrue(rec['A']['correct']);self.assertTrue(rec['A_exact_null_credit'])
        self.assertTrue(rec['A']['known_budget_stop']);self.assertFalse(rec['A']['source_bound'])
    def test_actual_scorer_auxfatal_null_does_not_retroactively_poison_A(self):
        row=t.row(a=None,b=2,final=None,status='fatal_selection',fatal=True)
        # All phase local diagnostics nonfatal; controller-level late fatal alone.
        rec=actual_record(row)
        self.assertTrue(rec['A']['correct']);self.assertFalse(rec['correct'])
        self.assertTrue(rec['auxiliary_fatal']);self.assertTrue(rec['auxiliary_fatal_correct_A_loss'])
        self.assertTrue(rec['A_exact_null_credit']);self.assertFalse(rec['team_exact_null_credit'])
        self.assertEqual(rec['all_post_A_effect'],'harmful')
    def test_actual_scorer_primary_topfatal_null_has_no_credit(self):
        row=t.row(a=None,final=None,status='fatal_primary',fatal=True)
        rec=actual_record(row)
        self.assertFalse(rec['A']['correct']);self.assertTrue(rec['A']['fatal'])
        self.assertIsNone(rec['A']['typed_ignoring_error']);self.assertFalse(rec['A_exact_null_credit'])
        self.assertFalse(rec['auxiliary_fatal_correct_A_loss'])
    def test_source_issue_error_overlaps_are_not_standalone_withholding(self):
        for error,fatal in [('Known released error',False),(None,True)]:
            with self.subTest(error=error,fatal=fatal):
                row=t.row(a=None,final=None,status='fatal_primary' if fatal else 'A_packet_unavailable',fatal=fatal)
                a=row['revision']['messages'][0];a['artifact']=None;a['released']['error']=error
                a['released']['revision'].update(final=t.three(7),final_check_issues=['Original source issue'])
                rec=actual_record(row,'[[7]]')
                self.assertFalse(rec['A']['correct']);self.assertTrue(rec['A']['computed_final_typed_correct'])
                self.assertTrue(rec['A']['source_final_issues']);self.assertFalse(rec['A']['source_withholding_lost_credit'])
    def test_nonerror_withheld_computed_correct_is_not_released_or_completed_revision(self):
        row=t.row(a=None,final=None);a=row['revision']['messages'][0];a['artifact']=None
        a['released']['revision'].update(draft=t.three(7),final=t.three(7),check_issues=['Original source issue'],final_check_issues=['Original source issue'],revised=False)
        rec=actual_record(row,'[[7]]')
        self.assertFalse(rec['A']['correct']);self.assertTrue(rec['A']['source_withholding_lost_credit'])
        self.assertTrue(rec['A']['source_feedback_requested']);self.assertFalse(rec['A']['source_revised'])
        self.assertEqual(rec['A']['source_revision_effect'],'not_applied')
    def test_missing_released_error_key_is_unavailable_not_null(self):
        row=t.row(a=None,final=None);del row['revision']['messages'][0]['released']['error']
        rec=actual_record(row)
        self.assertFalse(rec['A']['available']);self.assertIsNone(rec['A']['correct'])
        self.assertFalse(rec['A_exact_null_credit'])
    def test_pinned_proof_pins_owned_drift_stops_before_scorer_import(self):
        for field in ('completion_report','guard_pins','owned_jobs_attestation'):
            with self.subTest(field=field):
                f,args,_=self.fixture();prepared=m.verify_all(args)
                target=Path(getattr(args,field));target.write_bytes(target.read_bytes()+b'\n')
                loader=Mock()
                with self.assertRaisesRegex(ValueError,'Pinned input changed'):m.analyze(prepared,loader=loader)
                loader.assert_not_called();self.assertFalse(Path(args.output).exists())
    def test_template_or_raw_drift_after_admission_blocks_publication(self):
        for target_kind in ('template','raw'):
            with self.subTest(target_kind=target_kind):
                f,args,_=self.fixture();prepared=m.verify_all(args)
                def loader(path,name):
                    result=m.module(path,name)
                    target=f.root/'manifests/question_families.csv' if target_kind=='template' else f.output/'run_team.log'
                    target.write_bytes(target.read_bytes()+b'\n')
                    return result
                with self.assertRaisesRegex(ValueError,'Pinned input changed'):m.analyze(prepared,loader=loader)
                self.assertFalse(Path(args.output).exists())
    @classmethod
    def tearDownClass(cls):
        assert hashlib.sha256((P/'v7_posthoc.py').read_bytes()).hexdigest()=='4e3c460fba42855715ebef94d064b9f8c9cca827d94122042ce113dc29d3a6d4'

if __name__=='__main__':unittest.main(verbosity=2)
