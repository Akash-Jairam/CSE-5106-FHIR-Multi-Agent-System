"""Synthetic V5 operational records only; no real results or runtime imports."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('v5_guard_under_test',HERE/'v5_guard.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def candidate(answer):
    return {'answer':answer,'evidence':[{'resource_id':'Observation/a','field_path':'valueQuantity.value','value':1},
                                       {'resource_id':'Observation/b','field_path':'valueQuantity.value','value':2}],
            'operation':'Select the maximum value','contributing_ids':['Observation/a','Observation/b'],'limitation':None}


def tool(identifier,name,args,result):
    return [{'role':'assistant','content':None,'phase':'critic','tool_calls':[{'id':identifier,'type':'function',
             'function':{'name':name,'arguments':json.dumps(args)}}]},
            {'role':'tool','tool_call_id':identifier,'phase':'critic','name':name,'content':json.dumps(result)}]


def fixture(status='released'):
    workspace={'Observation':[{'id':'a','resourceType':'Observation','valueQuantity':{'value':1}},
                              {'id':'b','resourceType':'Observation','valueQuantity':{'value':2}}]}
    primary=candidate(1);alternate=candidate(2);code='answer = 1\nevidence = []\n'
    row={'question_id':'synthetic','question':'Return the maximum synthetic value.',
      'patient_fhir_id':'synthetic-patient','question_with_context':'Question: Return the maximum synthetic value.\nContext:\nPatient FHIR ID is synthetic-patient.',
      'agent_answer':2,'evidence':alternate['evidence'],'limitation':None,'latency':1,'error':None,
      'tokens':{'total':150,'llm_calls':3,'per_call':[{'usage_available':True} for _ in range(3)]},
      'budget':{'max_total_tokens':200000,'max_tool_calls':30,'tool_calls_run':5}}
    packet={'context':m.context_from_row(row),'final_code':code,'code_sha256':m.sha_text(code),
      'workspace_sha256':m.fingerprint(workspace),'retrieval_ledger':[],'complete_types':['Observation'],
      'unresolved_scope':{},'inventory_preview':{}}
    first={'decision':'INSPECT','anchor':{'channel':'question','quote':'maximum'},'claim':'The candidate is below a held value',
           'inspect_code':'answer = {"facts": [], "observations": []}'}
    fact={'resource_id':'Observation/b','field_path':'valueQuantity.value','value':2}
    probe={'facts':[fact],'observations':[{'name':'candidate_value','value':1}]}
    proposal={'decision':'PATCH','anchor':first['anchor'],'observed_discrepancy':'A larger held value exists',
      'expected_obligation':'Return maximum','preserved_obligations':['Patient scope'],
      'witnesses':[{'fact':fact,'observation_name':'candidate_value','observed_value':1,'relevance':'Larger source value'}],
      'patch':{'old_text':'answer = 1','new_text':'answer = 2'}}
    execution={**alternate,'mode':'final'}
    source={'issues':[],'results':[{'citation':deepcopy(c),'status':'valid','warnings':[]} for c in alternate['evidence']]}
    critic={'status':'released','assurance':m.ASSURANCE,'model_calls':2,'tool_calls':3,
      'primary_code_sha256':packet['code_sha256'],'workspace_sha256':packet['workspace_sha256'],
      'packet_sha256':m.fingerprint(packet),'first':first,'probe_output':{'answer':probe,'mode':'inspect'},
      'validated_probe':probe,'proposal':proposal,'alternate_execution':execution,'alternate':alternate,
      'source_check':source,'candidate_issues':[],'candidate_code_sha256':m.sha_text(code.replace('answer = 1','answer = 2'))}
    trace=[{'role':'system','content':'Synthetic solver'},
      {'role':'assistant','tool_calls':[{'id':'primary','function':{'name':'execute_python_code','arguments':json.dumps({'mode':'final','code':code})}}]},
      {'role':'tool','tool_call_id':'primary','name':'execute_python_code','content':json.dumps({**primary,'mode':'final'})},
      {'role':'assistant','content':'Host boundary','phase':'solver_boundary','model_call':False},
      {'role':'assistant','tool_calls':[{'id':'automatic_field_check_1','function':{'name':'check_fields','arguments':json.dumps({'citations':primary['evidence']})}}]},
      {'role':'tool','name':'check_fields','tool_call_id':'automatic_field_check_1','content':json.dumps({'issues':[],'results':[{'citation':deepcopy(c),'status':'valid','warnings':[]} for c in primary['evidence']]})},
      {'role':'system','content':'Review the actual executed program against the ORIGINAL question. Synthetic fixture.'},
      {'role':'user','content':json.dumps(packet)},
      {'role':'assistant','content':json.dumps(first)}]
    trace+=tool('v5-critic-inspect','execute_python_code',{'code':first['inspect_code'],'mode':'inspect'},critic['probe_output'])
    trace+=[{'role':'user','content':'Synthetic observed probe'},{'role':'assistant','content':json.dumps(proposal)}]
    trace+=tool('v5-critic-final','execute_python_code',{'code':code.replace('answer = 1','answer = 2'),'mode':'final'},execution)
    trace+=tool('v5-critic-source-check','check_fields',{'citations':alternate['evidence']},source)
    revision={'protocol':m.PROTOCOL,'primary':primary,'primary_revision':{'final_check_issues':[]},'final':alternate,
      'phases':[{'phase':'solver','token_cap':175000,'tool_cap':27,'tokens':100,'tool_calls':2,'fatal':False,'errors':[]},
                {'phase':'critic','token_cap':25000,'tool_cap':3,'tokens':50,'tool_calls':3,'fatal':False,'errors':[]}],
      'retrieval_ledger':[],'complete_types':['Observation'],'unresolved_scope':{},'solver_source_revised':False,
      'critic_revised':True,'targeted_revised':True,'final_check_issues':[],
      'release_policy':'executed_source_checked_critic_proposal','critic':critic}
    if status in {'primary_preserved','coverage_skip'}:
        revision.update(final=deepcopy(primary),critic_revised=False,targeted_revised=False,release_policy='source_valid_primary_preserved')
        row.update(agent_answer=1,evidence=primary['evidence']);row['budget']['tool_calls_run']=2
        revision['phases'][1].update(tool_calls=0,tokens=0)
        row['tokens']['total']=100;row['tokens']['llm_calls']=1 if status=='coverage_skip' else 2
        row['tokens']['per_call']=[{'usage_available':True} for _ in range(row['tokens']['llm_calls'])]
        critic.update(status=status,model_calls=0 if status=='coverage_skip' else 1,tool_calls=0)
        for k in ['first','probe_output','validated_probe','proposal','alternate_execution','alternate','source_check','candidate_issues','candidate_code_sha256']:critic.pop(k,None)
        trace=trace[:6] if status=='coverage_skip' else trace[:8]+[{'role':'assistant','content':'{"decision":"KEEP","reason":"No discrepancy"}'}]
    elif status=='accounting_failed':
        revision.update(final={'answer':None,'evidence':[],'limitation':'Accounting unavailable'},release_policy='aggregate_accounting_failed',
                        final_check_issues=[{'type':'accounting'}],critic_revised=False,targeted_revised=False)
        revision['phases'][1].update(fatal=True,tool_calls=0)
        critic.update(status=status,model_calls=1,tool_calls=0)
        row.update(agent_answer=None,evidence=[],limitation='Accounting unavailable',error='Critic aggregate accounting failed')
        row['tokens']['per_call']=[{'usage_available':None}];row['budget']['tool_calls_run']=2
        trace=trace[:8]
    row['revision']=revision;row['trace']=trace;row['retrieved_fhir_resources']=workspace
    return row


class V5GuardTests(unittest.TestCase):
    def test_released_repair_has_two_responses_three_tools_not_five_model_calls(self):
        row=fixture();counts=m.v5_protocol_counts(row,row)
        self.assertEqual(counts['observed_critic_model_responses'],2)
        self.assertEqual(counts['observed_critic_host_tool_replies'],3)
        self.assertEqual(counts['executed_source_checked_repairs'],1)

    def test_primary_preserved_and_coverage_skip_do_not_require_repair_fields(self):
        for status in ['primary_preserved','coverage_skip']:
            row=fixture(status);c=m.v5_protocol_counts(row,row)
            self.assertEqual(c['source_valid_releases'],1);self.assertEqual(c['critic_status_'+status],1)

    def test_fallback_cannot_change_valid_primary(self):
        row=fixture('primary_preserved');row['revision']['final']['answer']=2;row['agent_answer']=2
        with self.assertRaisesRegex(ValueError,'fallback changed'):m.v5_protocol_counts(row,row)

    def test_missing_usage_is_fatal_without_claiming_send_or_zero_usage(self):
        row=fixture('accounting_failed');c=m.v5_protocol_counts(row,row)
        self.assertEqual(c['source_valid_releases'],0);self.assertEqual(c['critic_admitted_invocations'],1)
        self.assertEqual(c['observed_critic_model_responses'],0)
        row['revision']['phases'][1]['fatal']=False
        with self.assertRaisesRegex(ValueError,'fail closed'):m.v5_protocol_counts(row,row)

    def test_unknown_usage_cannot_keep_primary_clean(self):
        row=fixture('primary_preserved');row['tokens']['per_call']=[{}]
        with self.assertRaisesRegex(ValueError,'Unsafe claimed'):m.v5_protocol_counts(row,row)

    def test_workspace_or_packet_tampering_rejected(self):
        row=fixture();row['retrieved_fhir_resources']['Observation'][0]['valueQuantity']['value']=9
        with self.assertRaisesRegex(ValueError,'workspace hash|frozen workspace'):m.v5_protocol_counts(row,row)
        row=fixture();packet=json.loads(row['trace'][7]['content']);packet['context']['question']='Different';row['trace'][7]['content']=json.dumps(packet)
        with self.assertRaisesRegex(ValueError,'context differs'):m.v5_protocol_counts(row,row)

    def test_false_number_witness_and_unobserved_fact_rejected(self):
        row=fixture();row['revision']['critic']['proposal']['witnesses'][0]['observed_value']=True
        with self.assertRaisesRegex(ValueError,'not observed|actual response'):m.v5_protocol_counts(row,row)
        row=fixture();row['revision']['critic']['validated_probe']['facts'][0]['value']=False
        with self.assertRaisesRegex(ValueError,'differs|probe differs'):m.v5_protocol_counts(row,row)

    def test_patch_execution_or_source_binding_tampering_rejected(self):
        row=fixture();row['revision']['critic']['candidate_code_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'patch code differs'):m.v5_protocol_counts(row,row)
        row=fixture();row['revision']['critic']['source_check']['results'][0]['warnings']=['not verified']
        with self.assertRaisesRegex(ValueError,'source check differs|warning-free|actual response'):m.v5_protocol_counts(row,row)

    def test_duplicate_or_overbudget_critic_tools_rejected(self):
        row=fixture();row['trace'].extend(deepcopy(row['trace'][-2:]))
        with self.assertRaisesRegex(ValueError,'duplicate critic host'):m.v5_protocol_counts(row,row)
        row=fixture();row['revision']['critic']['model_calls']=3
        with self.assertRaisesRegex(ValueError,'exceeds critic caps'):m.v5_protocol_counts(row,row)

    def test_released_actual_judgments_and_host_order_must_match(self):
        row=fixture()
        for message in row['trace']:
            if message.get('role')=='assistant' and message.get('content') and message.get('phase') is None:
                message['content']='{"decision":"KEEP","reason":"No discrepancy"}'
        with self.assertRaisesRegex(ValueError,'actual response'):m.v5_protocol_counts(row,row)
        row=fixture();start=next(i for i,x in enumerate(row['trace']) if x.get('tool_calls',[{}])[0].get('id')=='v5-critic-inspect')
        pair=row['trace'][start:start+2];del row['trace'][start:start+2];row['trace'].extend(pair)
        with self.assertRaisesRegex(ValueError,'out of order'):m.v5_protocol_counts(row,row)

    def test_invalid_source_status_rejected_even_with_empty_issue_list(self):
        row=fixture();row['revision']['critic']['source_check']['results'][0]['status']='invalid'
        for message in row['trace']:
            if message.get('tool_call_id')=='v5-critic-source-check':message['content']=json.dumps(row['revision']['critic']['source_check'])
        with self.assertRaisesRegex(ValueError,'valid warning-free'):m.v5_protocol_counts(row,row)

    def test_clean_missing_packet_per_call_or_solver_cap_is_rejected(self):
        row=fixture('primary_preserved');row['trace']=[x for x in row['trace'] if not (x.get('role')=='user' and 'final_code' in (x.get('content') or ''))]
        with self.assertRaisesRegex(ValueError,'actual packet'):m.v5_protocol_counts(row,row)
        row=fixture();row['tokens']['per_call']=[]
        with self.assertRaisesRegex(ValueError,'per-call'):m.v5_protocol_counts(row,row)
        row=fixture();row['trace'][:0]=[{'role':'user','content':'Solver allocation remaining: synthetic'} for _ in range(21)]
        with self.assertRaisesRegex(ValueError,'exceeds solver'):m.v5_protocol_counts(row,row)

    def test_released_patch_exact_schema_and_bounds(self):
        for mutation in ['missing','blank','region','program']:
            row=fixture();proposal=row['revision']['critic']['proposal']
            if mutation=='missing':proposal.pop('preserved_obligations')
            elif mutation=='blank':proposal['expected_obligation']=' '
            elif mutation=='region':proposal['patch']['new_text']='answer = 2'+(' '*8000)
            elif mutation=='program':
                proposal['patch']['new_text']='answer = 2'+(' '*7900)
                packet=json.loads(row['trace'][7]['content']);packet['final_code']+='\n'+(' '*25000)
                packet['code_sha256']=m.sha_text(packet['final_code']);row['trace'][7]['content']=json.dumps(packet)
                row['revision']['critic'].update(primary_code_sha256=packet['code_sha256'],packet_sha256=m.fingerprint(packet))
                row['trace'][1]['tool_calls'][0]['function']['arguments']=json.dumps({'mode':'final','code':packet['final_code']})
            for message in row['trace']:
                if message.get('role')=='assistant' and message.get('content') and '"PATCH"' in message['content']:message['content']=json.dumps(proposal)
            with self.assertRaises(ValueError):m.v5_protocol_counts(row,row)

    def test_runtime_contributor_canonicalization_keeps_raw_audit_intact(self):
        row=fixture();raw=json.loads(row['trace'][2]['content']);raw['contributing_ids']=[]
        row['trace'][2]['content']=json.dumps(raw)
        self.assertEqual(m.v5_protocol_counts(row,row)['source_valid_releases'],1)
        row=fixture();raw=deepcopy(row['revision']['critic']['alternate_execution']);raw['contributing_ids'].append('Observation/a')
        row['revision']['critic']['alternate_execution']=raw
        for message in row['trace']:
            if message.get('tool_call_id')=='v5-critic-final':message['content']=json.dumps(raw)
        self.assertEqual(m.v5_protocol_counts(row,row)['source_valid_releases'],1)
        self.assertEqual(len(row['revision']['critic']['alternate_execution']['contributing_ids']),3)
        raw['answer']=99
        with self.assertRaisesRegex(ValueError,'final execution'):m.v5_protocol_counts(row,row)

    def test_primary_execution_source_check_chronology_and_invalidation(self):
        row=fixture();auto=row['trace'][4:6];del row['trace'][4:6];row['trace'][1:1]=auto
        with self.assertRaisesRegex(ValueError,'later actual'):m.v5_protocol_counts(row,row)
        row=fixture();row['trace'][6:6]=[{'role':'assistant','tool_calls':[{'id':'retrieve','function':{'name':'get_resources_by_resource_id','arguments':'{}'}}]}]
        with self.assertRaisesRegex(ValueError,'current matching'):m.v5_protocol_counts(row,row)
        row=fixture();bad=deepcopy(row['trace'][4:6]);bad[1]['content']='{"issues":["bad"],"results":[]}'
        row['trace'][4:4]=bad
        with self.assertRaisesRegex(ValueError,'current matching'):m.v5_protocol_counts(row,row)
        row=fixture();row['trace'][6:6]=tool('synthetic-inspect','execute_python_code',{'code':'answer=7','mode':'inspect'},{'answer':7,'mode':'inspect'})
        # An ordinary inspection has no critic phase; it preserves the checked final.
        for message in row['trace'][6:8]:message.pop('phase',None)
        self.assertEqual(m.v5_protocol_counts(row,row)['source_valid_releases'],1)

    def test_known_usage_missing_auxiliary_response_preserves_primary_with_gap(self):
        row=fixture();rev=row['revision'];critic=rev['critic']
        rev.update(final=deepcopy(rev['primary']),critic_revised=False,targeted_revised=False,release_policy='source_valid_primary_preserved')
        row.update(agent_answer=1,evidence=rev['primary']['evidence'])
        critic.update(status='primary_preserved',model_calls=2,tool_calls=1,reason='Known-usage second response unavailable')
        for key in ['proposal','alternate_execution','alternate','source_check','candidate_issues','candidate_code_sha256']:critic.pop(key,None)
        # Keep the actual first response, inspection pair and follow-up input.
        second=next(i for i,x in enumerate(row['trace']) if x.get('role')=='assistant' and '"PATCH"' in (x.get('content') or ''))
        row['trace']=row['trace'][:second]
        rev['phases'][1]['tool_calls']=1;row['budget']['tool_calls_run']=3
        counts=m.v5_protocol_counts(row,row)
        self.assertEqual(counts['source_valid_releases'],1)
        self.assertEqual(counts['critic_invocations_without_retained_response'],1)
        self.assertEqual(counts['executed_source_checked_repairs'],0)

    def test_terminal_missing_protocol_metadata_retained(self):
        row={'error':'synthetic early crash','revision':None}
        c=m.v5_protocol_counts(row,row);self.assertEqual(c['terminal_errors_without_protocol_metadata'],1)

    def test_repair_can_change_evidence_without_changing_answer(self):
        row=fixture();p=row['revision']['primary'];p['answer']=2
        p['evidence']=p['evidence'][:1];row['trace'][2]['content']=json.dumps(dict(p,mode='final'))
        row['trace'][4]['tool_calls'][0]['function']['arguments']=json.dumps({'citations':p['evidence']})
        row['trace'][5]['content']=json.dumps({'issues':[],'results':[{'citation':c,'status':'valid','warnings':[]} for c in p['evidence']]})
        # The primary program still differs locally; this checks that answer equality
        # is not used as a substitute for full answer/evidence/contributor identity.
        c=m.v5_protocol_counts(row,row);self.assertEqual(c['executed_source_checked_repairs'],1)

    def test_fatal_overshoot_is_retained_not_clean(self):
        row=fixture('accounting_failed');row['tokens']['total']=200001
        row['revision']['phases'][0]['tokens']=175000;row['revision']['phases'][1]['tokens']=25001
        c=m.attempt_counts(row,row,True)
        self.assertEqual(c['reported_aggregate_token_overshoots'],1);self.assertEqual(c['reported_phase_token_overshoots'],1)

    def test_completion_requires_terminal_assertion_before_any_read(self):
        with self.assertRaisesRegex(ValueError,'both owned jobs'):
            m.run_check(Path('/nonexistent'),Path('/nonexistent'),Path('/nonexistent'),'0'*64,'completion',Path('/nonexistent'))

    def test_external_pins_reject_rewritten_manifest_and_source_map(self):
        with tempfile.TemporaryDirectory(prefix='v5-pins-synthetic-') as d:
            record=Path(d);mapping={k:'1'*64 for k in m.REQUIRED_SOURCE}
            manifest={'source_sha256':mapping}
            for name in m.RECORD_META:(record/name).write_text(json.dumps(manifest if name=='manifest.json' else {}))
            pins_path=record/'pins.json';pins=m.pin_record(record,pins_path);digest=m.sha(pins_path)
            m.verify_pins(record,pins_path,digest)
            mapping['agent/new.py']='2'*64;(record/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'metadata differs'):m.verify_pins(record,pins_path,digest)
            pins['record_metadata_sha256']['manifest.json']=m.sha(record/'manifest.json')
            pins_path.write_text(json.dumps(pins))
            with self.assertRaisesRegex(ValueError,'pins differ'):m.verify_pins(record,pins_path,digest)
            with self.assertRaisesRegex(ValueError,'source inventory differs'):m.verify_pins(record,pins_path,m.sha(pins_path))

    def test_source_path_cannot_read_secret_or_escape_symlink(self):
        with tempfile.TemporaryDirectory(prefix='v5-path-synthetic-') as d:
            root=Path(d)
            for name in ['config.yml','.git/config','.local-setup/config.yml','../elsewhere']:
                with self.assertRaises(ValueError):m.public_path(root,name)
            (root/'agent').mkdir();(root/'target.txt').write_text('synthetic')
            (root/'agent/linked.py').symlink_to(root/'target.txt')
            with self.assertRaisesRegex(ValueError,'Unsafe public'):m.public_path(root,'agent/linked.py')


if __name__=='__main__':unittest.main(verbosity=2)
