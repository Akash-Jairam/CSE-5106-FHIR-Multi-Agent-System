"""Pure synthetic counterfactual/gate fixtures; no actual rows/model/settings/FHIR."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock,patch

HERE=Path(__file__).parent
sp=importlib.util.spec_from_file_location('v7_posthoc_synthetic',HERE/'v7_posthoc.py');M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M)


def three(value):return {'answer':value,'evidence':[],'limitation':None}
def snapshot(label,value,*,fatal=False,error=None,bound=True):
    revision={'draft':three(value),'final':three(value),'check_issues':[],'final_check_issues':[],'revised':False}
    return {'kind':'candidate_snapshot','candidate':label,'phase':'natural_primary' if label=='A' else 'natural_alternative',
        'released':{**three(value),'error':error,'revision':revision},'artifact':{'evidence':[]} if bound else None,
        'diagnostics':{'fatal':fatal,'budget_stop':None,'tokens':10,'tool_calls':2 if label=='A' else 3,
            'natural_tool_calls':2,'model_calls':2,'token_cap':200000 if label=='A' else 150000,'tool_cap':30 if label=='A' else 27,
            'sdk_cap':30 if label=='A' else 27,'initialization':{'charged':int(label=='B')},'usage_known':not fatal},
        'usage':{'per_call':[{'usage_available':not fatal}]}}

def row(a=1,b=None,final=1,status='agreement_preserved',selected='A',fatal=False):
    snapshots=[snapshot('A',a)]
    if b is not None:snapshots.append(snapshot('B',b))
    return {'question_id':'synthetic','agent_answer':final,'error':'Synthetic auxiliary failure' if fatal else None,
        'tokens':{'per_call':[{'usage_available':not fatal}]},'trace':[],
        'revision':{'protocol':M.PROTOCOL,'status':status,'selected':selected,'fatal':fatal,'messages':snapshots,
            'phases':[s['diagnostics']|{'phase':s['phase']} for s in snapshots]}}

def record(value,reference=1):
    return M.row_record(value,lambda answer:{'typed_correct':M.same(answer,reference),'answer_type':'synthetic'},M.compact_verified(value))


class Reconstruction(unittest.TestCase):
    def test_actual_nonfatal_A_error_is_not_inherited_from_later_row_error(self):
        r=row(b=2,final=None,status='fatal_alternative',fatal=True)
        r['revision']['messages'][1]['diagnostics']['fatal']=True;r['revision']['phases'][1]['fatal']=True
        q=record(r);self.assertTrue(q['A']['correct']);self.assertFalse(q['A']['error'])
        self.assertFalse(q['correct']);self.assertTrue(q['auxiliary_fatal_correct_A_loss'])
        self.assertEqual(q['all_post_A_effect'],'harmful')

    def test_aggregate_only_auxiliary_fatal_loss_keeps_valid_A(self):
        for status in ('fatal_alternative','fatal_selection'):
            with self.subTest(status=status):
                q=record(row(a=1,b=2,final=None,status=status,fatal=True))
                self.assertTrue(q['A']['correct']);self.assertTrue(q['auxiliary_fatal'])
                self.assertTrue(q['auxiliary_fatal_correct_A_loss'])
                self.assertEqual(q['all_post_A_effect'],'harmful')
        self.assertFalse(record(row(status='fatal_primary',fatal=True))['auxiliary_fatal'])

    def test_ineligible_prose_A_preserves_own_error_not_row(self):
        r=row(a=None,b=1,final=1,status='selected_B',selected='B')
        r['revision']['messages'][0]['artifact']=None;r['revision']['messages'][0]['released']['error']='Original A failed'
        q=record(r,reference=None);self.assertFalse(q['A']['correct']);self.assertTrue(q['A']['error'])
        self.assertTrue(q['A']['typed_ignoring_error']);self.assertFalse(q['A_exact_null_credit'])

    def test_fatal_A_never_gets_null_credit(self):
        r=row(a=None,final=None,status='fatal_primary',fatal=True)
        r['revision']['messages'][0]['diagnostics']['fatal']=True
        q=record(r,reference=None);self.assertTrue(q['A']['available']);self.assertFalse(q['A']['correct'])
        self.assertIsNone(q['A']['typed_ignoring_error']);self.assertFalse(q['A_exact_null_credit'])

    def test_primary_aggregate_accounting_fatal_is_not_a_valid_A_credit(self):
        q=record(row(a=1,final=None,status='fatal_primary',fatal=True))
        self.assertTrue(q['A']['available']);self.assertTrue(q['A']['fatal']);self.assertFalse(q['A']['correct'])
        self.assertIsNone(q['A']['typed_ignoring_error'])

    def test_missing_A_is_unavailable_and_never_scored_as_null(self):
        r=row(a=None,final=None);r['revision']['messages']=[];calls=[]
        def score(value):calls.append(value);return {'typed_correct':True}
        q=M.row_record(r,score,M.compact_verified(r));self.assertFalse(q['A']['available'])
        self.assertIsNone(q['A']['correct']);self.assertEqual(calls,[None])

    def test_frozen_A_mutation_does_not_certify_original_A(self):
        q=record(row(status='fatal_mutation',fatal=True));self.assertFalse(q['A']['available'])

    def test_A_withheld_computed_final_is_not_the_released_endpoint(self):
        r=row(a=None,final=None);s=r['revision']['messages'][0];s['artifact']=None
        s['released']['revision'].update(final=three(1),final_check_issues=['Synthetic issue'])
        q=record(r);self.assertFalse(q['A']['correct']);self.assertTrue(q['A']['computed_final_typed_correct'])
        self.assertTrue(q['A']['source_withholding_lost_credit'])

    def test_source_withholding_does_not_standalone_attribute_accounting_or_terminal_error(self):
        for fatal,error in ((True,None),(False,'Synthetic terminal phase error')):
            with self.subTest(fatal=fatal):
                r=row(a=None,final=None,status='fatal_primary' if fatal else 'B_ineligible',fatal=fatal)
                s=r['revision']['messages'][0];s['artifact']=None;s['released']['error']=error;s['diagnostics']['fatal']=fatal
                s['released']['revision'].update(final=three(1),final_check_issues=['Synthetic issue'])
                q=record(r);self.assertTrue(q['A']['computed_final_typed_correct']);self.assertTrue(q['A']['source_final_issues'])
                self.assertFalse(q['A']['source_withholding_lost_credit'])

    def test_A_and_B_source_revisions_do_not_become_selected_B_repair(self):
        r=row(a=1,b=2,final=2,status='selected_B',selected='B')
        for s in r['revision']['messages']:
            s['released']['revision'].update(draft=three(0),check_issues=['Synthetic issue'],revised=True)
        q=record(r);self.assertEqual(q['A']['source_revision_effect'],'helpful')
        self.assertEqual(q['B']['source_revision_effect'],'neutral');self.assertEqual(q['released_B_effect'],'harmful')

    def test_feedback_without_actual_revised_flag_is_not_completed_revision(self):
        r=row();r['revision']['messages'][0]['released']['revision']['check_issues']=['Synthetic issue']
        q=record(r);self.assertTrue(q['A']['source_feedback_requested']);self.assertFalse(q['A']['source_revised'])
        self.assertEqual(q['A']['source_revision_effect'],'not_applied')

    def test_selected_B_help_harm_neutral_and_unselected_complement_are_separate(self):
        for a,b,reference,effect in [(0,1,1,'helpful'),(1,2,1,'harmful'),(1,1,1,'neutral')]:
            with self.subTest(effect=effect):self.assertEqual(record(row(a,b,b,'selected_B','B'),reference)['released_B_effect'],effect)
        q=record(row(a=0,b=1,final=0,status='selector_preserved_A'))
        self.assertTrue(q['B']['correct']);self.assertEqual(q['released_B_effect'],'not_applied')

    def test_strict_JSON_agreement_is_distinct_from_typed_correctness(self):
        r=row(a=False,b=0,final=False);q=M.row_record(r,lambda x:{'typed_correct':x in (False,0)},M.compact_verified(r))
        self.assertTrue(q['A']['correct']);self.assertTrue(q['B']['correct']);self.assertFalse(q['exact_JSON_answer_agreement'])

    def test_actual_nonerror_null_credit_is_labelled_outside_sourceproof(self):
        r=row(a=None,final=None);r['revision']['messages'][0]['artifact']=None
        q=record(r,reference=None);self.assertTrue(q['A_exact_null_credit']);self.assertTrue(q['team_exact_null_credit'])
        self.assertFalse(q['A']['source_bound'])


class DevelopmentGate(unittest.TestCase):
    def base(self,n=240,a_correct=239):
        return [record(row(a=1 if i<a_correct else 0,final=1 if i<n else 0)) for i in range(300)]
    def test_strict_three_way_gate_and_all300_A(self):
        g=M.summarize(self.base(),239,240)['development_acceptance'];self.assertTrue(g['passed'])
        for fresh,team,a in [(240,240,239),(239,240,240),(238,239,238)]:
            self.assertFalse(M.summarize(self.base(team,a),fresh,team)['development_acceptance']['passed'])
    def test_one_missing_A_blocks_selection_without_being_wrong_or_excluded(self):
        values=self.base();values[0]['A'].update(available=False,correct=None)
        gate=M.summarize(values,239,240)['development_acceptance']
        self.assertFalse(gate['passed']);self.assertIsNone(gate['team_beats_actual_A_only'])
        self.assertIsNone(gate['A_only_correct_all300']);self.assertEqual(gate['A_only_available'],299)
    def test_partial_denominator_or_official_mismatch_rejected(self):
        with self.assertRaises(ValueError):M.summarize(self.base()[:-1],239,240)
        with self.assertRaises(ValueError):M.summarize(self.base(),239,241)


class RejectedGates(unittest.TestCase):
    def fixture(self):
        temp=tempfile.TemporaryDirectory(dir='/private/tmp');self.addCleanup(temp.cleanup)
        d=Path(temp.name);root=d/'project';(root/'benchmarks'/M.EXPERIMENT).mkdir(parents=True)
        values={'completion':{'status':'PASS','mode':'completion','protocol':M.PROTOCOL,'caller_asserted_owned_jobs_terminal':True,
            'expected_first_attempts':600,'restored_audit_records':600,'scoring_performed':False},'official':{},'pins':{},'owned':{}}
        files={}
        for k,v in values.items():files[k]=d/(k+'.json');files[k].write_text(json.dumps(v))
        guard=d/'guard.py';guard.write_text('raise AssertionError("No rejected gate may import this")\n')
        args=SimpleNamespace(root=str(root),output=str(d/'fresh'),helper_sha256=M.sha(HERE/'v7_posthoc.py'),
            completion_report=str(files['completion']),completion_report_sha256=M.sha(files['completion']),
            official_report=str(files['official']),official_report_sha256=M.sha(files['official']),guard_module=str(guard),guard_sha256=M.sha(guard),
            guard_pins=str(files['pins']),guard_pins_sha256=M.sha(files['pins']),owned_jobs_attestation=str(files['owned']),owned_jobs_attestation_sha256=M.sha(files['owned']),
            confirm_both_owned_jobs_terminal=True,single_label='v7_single',team_label='v7_team')
        proof=values['completion'];proof.update(guard_sha256=args.guard_sha256,pins_sha256=args.guard_pins_sha256,owned_jobs_attestation_sha256=args.owned_jobs_attestation_sha256)
        files['completion'].write_text(json.dumps(proof));args.completion_report_sha256=M.sha(files['completion'])
        return args,proof,files
    def test_partial_wrong_mode_missing600_and_nonterminal_reject_before_import(self):
        for key,value in [('status','PARTIAL'),('mode','initial'),('restored_audit_records',599),('expected_first_attempts',True),('caller_asserted_owned_jobs_terminal',False),('protocol','wrong')]:
            with self.subTest(key=key):
                args,proof,files=self.fixture();proof[key]=value;files['completion'].write_text(json.dumps(proof));args.completion_report_sha256=M.sha(files['completion'])
                with patch.object(M,'module') as loader:
                    with self.assertRaises(ValueError):M.run(args)
                loader.assert_not_called();self.assertFalse(Path(args.output).exists())
    def test_wrong_external_SHA_self_guard_report_pins_owned_reject_without_import(self):
        for field in ('helper_sha256','completion_report_sha256','official_report_sha256','guard_sha256','guard_pins_sha256','owned_jobs_attestation_sha256'):
            with self.subTest(field=field):
                args,_,_=self.fixture();setattr(args,field,'0'*64)
                with patch.object(M,'module') as loader:
                    with self.assertRaises(ValueError):M.run(args)
                loader.assert_not_called();self.assertFalse(Path(args.output).exists())
    def test_destination_existing_escape_raw_source_or_symlink_rejected(self):
        args,_,_=self.fixture();root=Path(args.root)
        for bad in (root/'output'/'new',root/'agent'/'new',Path(args.output).parent/'..'/'escape'):
            args.output=str(bad)
            with self.assertRaises(ValueError):M.destination(args)
        link=root.parent/'link';link.symlink_to(root,target_is_directory=True);args.output=str(link/'new')
        with self.assertRaises(ValueError):M.destination(args)


class CompleteProofWire(unittest.TestCase):
    """Real pure guard/store/scorer on generated600; no inference imports."""
    def fixture(self):
        import zipfile
        packet=Path('/private/tmp/fhir-v7-prospective-integrity-v2-20261015')
        self.assertEqual(M.sha(packet/'v7_guard.py'),'bf47aa9e9c84735b0a096acbb6d209776473fcb7e81e8cfa2e366576d90e1115')
        lifecycle=M.module(packet/'test_v7_guard_lifecycle.py','posthoc_synthetic_lifecycle')
        lifecycle.AUDIT_SOURCE=HERE/'fixtures/audit_store.py'
        temp=tempfile.TemporaryDirectory(dir='/private/tmp');self.addCleanup(temp.cleanup)
        f=lifecycle.LifecycleFixture(Path(temp.name)/'project')
        for item in f.inputs:item['true_answer']='[[7]]'
        f.input_hash=M.hashlib.sha256(lifecycle.pd.DataFrame(f.inputs).to_json(orient='records',date_format='iso').encode()).hexdigest()
        # Only frozen pure public modules, independently hashed fixture copies.
        source_hashes=M.load(HERE/'fixtures/public_source_sha256.json')
        for name,digest in source_hashes.items():
            source=HERE/'fixtures'/Path(name).name;self.assertEqual(M.sha(source),digest)
            target=f.root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        families=f.root/'manifests/question_families.csv'
        families.write_text('question_id,template\n'+''.join(r['question_id']+',Synthetic question\n' for r in f.inputs))
        f.manifest['source_sha256']={name:M.sha(f.root/name) for name in sorted(lifecycle.guard.actual_source_inventory(f.root))}
        (f.record/'source.zip').unlink();lifecycle.source_zip(f.record/'source.zip',f.root,f.manifest['source_sha256'])
        f.manifest['source_archive_sha256']=M.sha(f.record/'source.zip');lifecycle.dump(f.record/'manifest.json',f.manifest)
        scope=f.patched_constants();scope.__enter__();self.addCleanup(scope.__exit__,None,None,None)
        f.pin();f.write_arms(300);f.write_journals(True);proof=f.run('completion','completion.json',terminal=True)
        self.assertEqual(proof['status'],'PASS');self.assertEqual(proof['restored_audit_records'],600)
        labels={'ordinary_single_agent':'v7_single','ordinary_multi_agent':'v7_team'}
        official={'cohort':'held_out','questions':300,'dataset_sha256':f.manifest['dataset_sha256'],
            'families_sha256':f.manifest['source_sha256']['manifests/question_families.csv'],
            'question_ids_sha256':M.hashlib.sha256(json.dumps(sorted(r['question_id'] for r in f.inputs)).encode()).hexdigest(),
            'identities':{},'arms':{}}
        for arm,label in labels.items():
            result=f.output/(arm+'.json');identity=f.output/(arm+'.json.run.json')
            official['identities'][label]={'identity':M.load(identity),'result_path':str(result),'result_sha256':M.sha(result)}
            official['arms'][label]={'terminal_errors':1,'accuracy':{'all':{'correct':0,'n':300}}}
        official_path=f.record/'official.synthetic.json';lifecycle.dump(official_path,official)
        args=SimpleNamespace(root=str(f.root),output=str(Path(temp.name)/'diagnostics'),helper_sha256=M.sha(HERE/'v7_posthoc.py'),
            completion_report=str(f.record/'completion.json'),completion_report_sha256=M.sha(f.record/'completion.json'),
            official_report=str(official_path),official_report_sha256=M.sha(official_path),guard_module=str(packet/'v7_guard.py'),guard_sha256=M.sha(packet/'v7_guard.py'),
            guard_pins=str(f.pins),guard_pins_sha256=f.pins_sha,owned_jobs_attestation=str(f.record/'owned.json'),owned_jobs_attestation_sha256=M.sha(f.record/'owned.json'),
            confirm_both_owned_jobs_terminal=True,single_label='v7_single',team_label='v7_team')
        original=M.module
        def loader(path,name):
            return lifecycle.guard if Path(path)==packet/'v7_guard.py' else original(path,name)
        self.addCleanup(patch.stopall)
        patch.object(M,'EXPERIMENT','synthetic-v7').start();patch.object(M,'module',side_effect=loader).start()
        return f,args,lifecycle

    def test_complete_pinned_proof_owned_Team300_and_unchanged_typed_scorer(self):
        f,args,lifecycle=self.fixture()
        # Deep proof already supplied600 CRC; helper only restores Team300.
        with patch.object(lifecycle.guard,'verify_restored',wraps=lifecycle.guard.verify_restored) as restored:
            report=M.run(args)
        self.assertEqual(restored.call_count,300)
        self.assertEqual(report['official_totals_reconciled']['ordinary_multi_agent']['correct'],0)
        self.assertEqual(report['summary']['development_acceptance']['A_only_available'],299)
        self.assertFalse(report['summary']['development_acceptance']['passed'])
        self.assertEqual(len(M.load(Path(args.output)/'mechanisms_rows.json')),300)
        self.assertEqual(M.sha(f.root/'utils/answer_scorer.py'),M.load(HERE/'fixtures/public_source_sha256.json')['utils/answer_scorer.py'])

    def test_raw_drift_rejects_before_any_scorer_import_and_output(self):
        f,args,_=self.fixture();(f.output/'run_team.log').write_text('Synthetic drift')
        calls=[];original=M.module
        def loader(path,name):calls.append(str(path));return original(path,name)
        with patch.object(M,'module',side_effect=loader):
            with self.assertRaisesRegex(ValueError,'Pinned bytes differ|Raw size envelope differs'):M.run(args)
        self.assertFalse(any('answer_scorer' in name for name in calls));self.assertFalse(Path(args.output).exists())

    def test_source_and_report_drift_after_score_import_prevents_publish(self):
        for what in ('source','official'):
            with self.subTest(what=what):
                f,args,_=self.fixture();prepared=M.verify_all(args);original=M.module
                def loader(path,name):
                    scorer=original(path,name)
                    target=f.root/'scripts/compare_runs.py' if what=='source' else Path(args.official_report)
                    target.write_text(target.read_text()+'\n')
                    return scorer
                with self.assertRaisesRegex(ValueError,'Pinned input changed'):M.analyze(prepared,loader=loader)
                self.assertFalse(Path(args.output).exists())


if __name__=='__main__':unittest.main(verbosity=2)

