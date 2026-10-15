"""Synthetic values only; no result paths, private settings, model or network calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE=Path(__file__).parent

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

m=module(HERE/'v6_posthoc.py','synthetic_posthoc')
guard_fixture=module(Path('/private/tmp/fhir-v6-integrity-bfjeklza/test_v6_guard_protocol.py'),'synthetic_guard_fixture')

def three(answer):return {'answer':answer,'evidence':[],'limitation':None}
def five(answer):return {**three(answer),'operation':'Synthetic operation','contributing_ids':[]}
def score_for(reference):
    return lambda answer:{'typed_correct':m.same(answer,reference),'answer_type':'empty' if reference is None else 'number'}

def row(a=1,final=1,eligible=True,fatal=False):
    return {'question_id':'synthetic','agent_answer':final,'error':None,'tokens':{'per_call':[{'usage_available':True}]},
      'revision':{'protocol':m.PROTOCOL,'primary':three(a),'primary_diagnostics':{'released_available':True,'eligible':eligible,'fatal':fatal},
        'primary_revision':{'draft':three(a),'final':three(a),'final_check_issues':[],'check_issues':[],'revised':False},
        'primary_source_revised':False,'independent_candidate_selected':False,'revised':False,
        'release_policy':'continuous_primary_preserved'}}

def record(value,reference=1,**proof):
    return m.row_record(value,score_for(reference),{'a_source_bound':True,'b_source_bound':True,**proof})

def aux(value,b=2,switched=True):
    r=value['revision'];r['recompute']={'B':five(b),'B_eligible':True,'answer_agreement':m.same(r['primary']['answer'],b),
      'fatal':False,'status':'switched' if switched else 'primary_preserved','selected':'B' if switched else 'A'}
    r['independent_candidate_selected']=switched;r['revised']=switched or r['primary_source_revised']
    r['release_policy']='selected_executed_independent_candidate' if switched else 'continuous_primary_preserved'
    if switched:value['agent_answer']=b
    return value


class Reconstruction(unittest.TestCase):
    def test_selected_B_harm_is_not_primary_source_revision(self):
        item=record(aux(row(),b=2))
        self.assertEqual(item['released_B_effect'],'harmful');self.assertEqual(item['auxiliary_stage_effect'],'harmful')
        self.assertFalse(item['primary_source_revised']);self.assertTrue(item['generic_revised_flag_informational_only'])
        self.assertEqual(item['primary_source_revision_effect_to_released_A'],'not_applied')

    def test_selected_B_help(self):
        item=record(aux(row(a=2,final=2),b=1));self.assertEqual(item['released_B_effect'],'helpful')
        self.assertTrue(item['correct']);self.assertFalse(item['a_only_correct'])

    def test_selected_B_same_correctness_is_neutral(self):
        for reference in (1,3):
            with self.subTest(reference=reference):
                value=aux(row(a=1),b=1 if reference==1 else 2)
                self.assertEqual(record(value,reference)['released_B_effect'],'neutral')

    def test_unselected_correct_B_is_not_a_repair(self):
        item=record(aux(row(a=2,final=2),b=1,switched=False))
        self.assertTrue(item['b_typed_correct']);self.assertEqual(item['released_B_effect'],'not_applied')
        self.assertFalse(item['independent_candidate_selected']);self.assertEqual(item['auxiliary_stage_effect'],'neutral')

    def test_primary_source_revision_has_released_A_endpoint(self):
        value=aux(row(a=1),b=2);r=value['revision'];r['primary_source_revised']=True;r['primary_revision']['draft']=three(2)
        item=record(value)
        self.assertEqual(item['primary_source_revision_effect_to_released_A'],'helpful')
        self.assertEqual(item['released_B_effect'],'harmful')

    def test_source_feedback_without_distinct_second_candidate_is_not_revision(self):
        value=row(a=2,final=2);value['revision']['primary_revision']['check_issues']=['Synthetic source issue']
        item=record(value);self.assertTrue(item['primary_initial_source_issues_recorded'])
        self.assertFalse(item['primary_source_revised']);self.assertEqual(item['primary_source_revision_effect_to_released_A'],'not_applied')

    def test_source_withheld_A_uses_normalized_released_not_computed_final(self):
        value=row(a=None,final=None,eligible=False);r=value['revision'];r['primary_revision']['final']=three(1)
        r['primary_revision']['final_check_issues']=['Synthetic source issue']
        item=record(value)
        self.assertFalse(item['a_only_correct']);self.assertTrue(item['primary_prewithholding_candidate_correct'])
        self.assertTrue(item['primary_source_withholding_lost_typed_credit'])

    def test_actual_nonerror_null_credit_is_retained_and_labeled(self):
        item=record(row(a=None,final=None,eligible=False),reference=None,a_source_bound=False)
        self.assertTrue(item['a_only_correct']);self.assertTrue(item['team_exact_None_typed_credit'])
        self.assertTrue(item['a_exact_None_typed_credit']);self.assertFalse(item['primary_source_bound'])

    def test_primary_fatal_cannot_get_null_reference_credit(self):
        value=row(a=None,final=None,fatal=True);value['error']='Synthetic provider accounting failure'
        value['revision']['primary_diagnostics']['released_available']=False
        item=record(value,reference=None)
        self.assertTrue(item['a_only_available']);self.assertFalse(item['a_only_correct']);self.assertTrue(item['a_only_error'])
        self.assertFalse(item['correct']);self.assertIsNone(item['a_only_typed_ignoring_error'])

    def test_ineligible_A_inherits_its_actual_error(self):
        value=row(a=None,final=None,eligible=False);value['error']='Synthetic timeout'
        item=record(value,reference=None)
        self.assertTrue(item['a_only_available']);self.assertTrue(item['a_only_typed_ignoring_error'])
        self.assertFalse(item['a_only_correct']);self.assertTrue(item['a_only_error'])

    def test_eligible_A_remains_scoreable_after_auxiliary_fatal(self):
        value=aux(row(),switched=False);value['revision']['recompute'].update(fatal=True,status='accounting_failed')
        value['error']='Synthetic auxiliary accounting failure';value['agent_answer']=None
        item=record(value)
        self.assertTrue(item['a_only_correct']);self.assertFalse(item['a_only_error']);self.assertFalse(item['correct'])
        self.assertTrue(item['auxiliary_non_switch_credit_lost']);self.assertEqual(item['auxiliary_stage_effect'],'harmful')

    def test_missing_primary_metadata_is_unavailable_not_null(self):
        for where in ('revision','primary','primary_diagnostics'):
            with self.subTest(where=where):
                value=row(a=None,final=None)
                if where=='revision':value.pop('revision')
                else:
                    value['revision'].pop(where)
                    value['revision'].pop('primary_revision')
                calls=[]
                def scoring(answer):calls.append(answer);return score_for(None)(answer)
                item=m.row_record(value,scoring,{'a_source_bound':True,'b_source_bound':True})
                self.assertFalse(item['a_only_available']);self.assertIsNone(item['a_only_correct'])
                self.assertEqual(len(calls),1) # Only actual Team outcome; no synthesized A.

    def test_nonfatal_unavailable_actual_common_output_is_unavailable(self):
        value=row();value['revision']['primary_diagnostics']['released_available']=False
        self.assertFalse(record(value)['a_only_available'])

    def test_unbound_declared_eligible_A_is_unavailable(self):
        self.assertFalse(record(row(),a_source_bound=False)['a_only_available'])

    def test_placeholder_draft_and_computed_final_do_not_get_null_credit(self):
        value=row(a=None,final=None,eligible=False);value['revision']['primary_revision'].update(draft={'answer':None},final={'answer':None})
        item=record(value,reference=None);self.assertIsNone(item['primary_initial_draft_correct'])
        self.assertIsNone(item['primary_prewithholding_candidate_correct'])

    def test_source_loss_does_not_mislabel_accounting_error(self):
        value=row(a=None,final=None,eligible=False);value['error']='Synthetic failure'
        value['revision']['primary_revision'].update(final=three(1),final_check_issues=['Synthetic issue'])
        self.assertFalse(record(value)['primary_source_withholding_lost_typed_credit'])

    def test_agreement_is_exact_json_not_boolean_numeric_equality(self):
        value=aux(row(a=False,final=False),b=0,switched=False)
        self.assertFalse(record(value,reference=False)['b_answer_agreement_exact'])
        self.assertTrue(record(value,b_source_bound=False)['b_answer_agreement_exact'] is None)

    def test_b_eligibility_without_actual_source_binding_is_not_scored(self):
        item=record(aux(row(),b=1),b_source_bound=False)
        self.assertTrue(item['b_eligible_recorded']);self.assertFalse(item['b_source_bound']);self.assertIsNone(item['b_typed_correct'])

    def test_known_primary_budget_stop_separate_from_terminal_failure(self):
        value=row(a=None,final=None,eligible=False);value['revision']['primary_diagnostics']['primary_budget_stop']='Synthetic known refusal'
        item=record(value,reference=None);self.assertTrue(item['primary_known_budget_stop']);self.assertFalse(item['terminal_error'])

    def test_unknown_usage_is_not_zero_or_dropped(self):
        value=row();value['tokens']['per_call'].append({'usage_available':False})
        self.assertTrue(record(value)['unknown_usage_recorded'])

    def test_all_firsts_and_unavailable_bounds_preserved(self):
        items=[record(row()),record(row(a=2,final=2)),record({'agent_answer':None,'question_id':'missing'},reference=None)]
        summary=m.summarize(items)
        self.assertEqual(summary['questions'],3);self.assertEqual(summary['a_only_unavailable'],1)
        self.assertIsNone(summary['a_only_accuracy_all_firsts_if_complete'])
        self.assertEqual(summary['a_only_fullcohort_accuracy_bounds'],[1/3,2/3])
        self.assertEqual(summary['a_to_team_available_denominator'],2)

    def test_all_available_primary_accuracy_uses_full_denominator(self):
        summary=m.summarize([record(row()),record(row(a=2,final=2))])
        self.assertEqual(summary['a_only_accuracy_all_firsts_if_complete'],0.5)


class ActualEnvelopeBindings(unittest.TestCase):
    def fixture(self):
        value=guard_fixture.fixture();r=value['revision'];r['primary_diagnostics']['released_available']=True
        r['primary_revision']['final']=deepcopy(r['primary'])
        return value

    def test_synthetic_actual_execution_and_original_source_checks_bind_A_B(self):
        proof=m.proof_flags(self.fixture(),guard_fixture.m)
        self.assertTrue(proof['a_source_bound']);self.assertTrue(proof['b_source_bound'])

    def test_primary_inspect_cannot_bind_actual_A(self):
        value=self.fixture();reply=next(x for x in value['trace'] if x.get('tool_call_id')=='a-final')
        body=json.loads(reply['content']);body['mode']='inspect';reply['content']=json.dumps(body)
        proof=m.proof_flags(value,guard_fixture.m);self.assertFalse(proof['a_source_bound']);self.assertTrue(proof['b_source_bound'])

    def test_B_source_error_and_nonfinal_execution_cannot_bind_candidate(self):
        for error in ('mode','source'):
            with self.subTest(error=error):
                value=self.fixture();aux=value['revision']['recompute']
                if error=='mode':
                    aux['B_execution']['mode']='inspect'
                    next(x for x in value['trace'] if x.get('tool_call_id')=='native-b')['content']=json.dumps(aux['B_execution'])
                else:
                    aux['B_source_check']['error']='Synthetic contradiction'
                    next(x for x in value['trace'] if x.get('tool_call_id')=='v6-independent-source-check')['content']=json.dumps(aux['B_source_check'])
                self.assertFalse(m.proof_flags(value,guard_fixture.m)['b_source_bound'])

    def test_prewithholding_final_does_not_replace_changed_actual_released_A(self):
        value=self.fixture();value['revision']['primary']['answer']=None
        self.assertFalse(m.proof_flags(value,guard_fixture.m)['a_source_bound'])


class GateOrderAndDestinations(unittest.TestCase):
    def test_terminal_assertion_required_before_any_file_or_scorer_read(self):
        with patch.object(m,'load',side_effect=AssertionError('Must not load')),patch.object(m,'module',side_effect=AssertionError('Must not import')):
            with self.assertRaisesRegex(ValueError,'terminal confirmation'):m.verify_all(SimpleNamespace(confirm_both_owned_jobs_terminal=False))

    def test_599_completion_rejected_before_guard_or_scorer_import(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as td:
            root=Path(td);args=SimpleNamespace(root=root,confirm_both_owned_jobs_terminal=True)
            data={'completion_report':{'status':'PASS','mode':'completion','caller_asserted_owned_jobs_terminal':True,'expected_first_attempts':600,'restored_audit_records':599},
                  'official_report':{},'guard_module':{},'guard_pins':{}}
            for name,value in data.items():
                p=root/name;p.write_text(json.dumps(value));setattr(args,name,p);setattr(args,('guard_sha256' if name=='guard_module' else name+'_sha256'),m.sha(p))
            with patch.object(m,'module',side_effect=AssertionError('No imports')):
                with self.assertRaisesRegex(ValueError,'All600'):m.verify_all(args)

    def test_changed_input_fails_before_scorer_import(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as td:
            root=Path(td);(root/'manifests').mkdir();(root/'manifests/question_families.csv').write_text('question_id,family,template\nsynthetic,synthetic,synthetic\n')
            path=root/'raw';path.write_text('before');digest=m.sha(path);path.write_text('after')
            prepared={'root':root,'guard':None,'official':{},'manifest':{},'ids':['synthetic'],'checked':{path:digest}}
            loader=Mock(side_effect=AssertionError('No imports'))
            with self.assertRaisesRegex(ValueError,'before scorer import'):m.analyze(prepared,loader=loader)
            loader.assert_not_called()

    def test_destination_cannot_overwrite_or_escape_formal_record(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as td:
            root=Path(td);existing=root/'existing';existing.mkdir()
            with self.assertRaises(ValueError):m.diagnostics_destination(SimpleNamespace(root=root,output=existing))
            with self.assertRaises(ValueError):m.diagnostics_destination(SimpleNamespace(root=root,output=root/'runtime'/'mechanisms'))

    def test_no_symlink_destination_ancestor(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as td:
            root=Path(td);target=root/'target';target.mkdir();link=root/'link';link.symlink_to(target,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'symlink'):m.diagnostics_destination(SimpleNamespace(root=root,output=link/'fresh'))

    def test_posthoc_module_has_no_import_side_effect_scorer_or_provider(self):
        import ast
        tree=ast.parse((HERE/'v6_posthoc.py').read_text())
        imports=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('scorer' in ast.unparse(node) or 'agent' in ast.unparse(node) or 'azure' in ast.unparse(node) for node in imports))


if __name__=='__main__':unittest.main()
