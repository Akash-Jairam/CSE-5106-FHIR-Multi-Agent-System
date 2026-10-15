"""Synthetic V7 retained snapshots/traces only; no inference, QA, scoring or credentials."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('v7_protocol_guard',HERE/'v7_guard.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
START_SHA=m.sha(HERE/'v7_guard.py')


def pair(identifier,name,args,result):
    return [{'role':'assistant','content':None,'tool_calls':[{'id':identifier,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]},
      {'role':'tool','name':name,'tool_call_id':identifier,'content':json.dumps(result)}]


def snapshot(label,answer,tokens,token_cap,tool_cap,sdk_cap,limitation=None,eligible=True):
    kind='natural_primary' if label=='A' else 'natural_alternative'
    workspace={'Observation':[{'resourceType':'Observation','id':label,'valueQuantity':{'value':answer}}]}
    artifact={'answer':answer,'evidence':[{'resource_id':'Observation/'+label,'field_path':'valueQuantity.value','value':answer}],
      'operation':'Synthetic single field','contributing_ids':['Observation/'+label],'limitation':limitation}
    check={'issues':[],'results':[{'citation':deepcopy(c),'status':'valid','warnings':[]} for c in artifact['evidence']],'n_valid':1}
    code='answer = '+repr(answer)+'\n'
    trace=[{'role':'system','content':'Synthetic original Single task'}]
    trace+=pair('same-native-id','execute_python_code',{'code':code,'mode':'final'},{**artifact,'mode':'final'})
    trace+=[{'role':'assistant','content':'Natural final'}]
    trace+=pair('automatic_field_check_1','check_fields',{'citations':artifact['evidence']},check)
    tagged=[{**deepcopy(item),'phase':kind} for item in trace]
    init={'charged':0} if label=='A' else {'charged':1,'started':1,'completed':1,'kind':'get_starting_packet','packet_sha256':'b'*64}
    diag={'phase':kind,'token_cap':token_cap,'tool_cap':tool_cap,'sdk_cap':sdk_cap,'tokens':tokens,'tool_calls':2+init['charged'],
      'natural_tool_calls':2,'model_calls':2,'fatal':False,'usage_known':True,'errors':[],'eligible':eligible,
      'skip_reason':None if eligible else 'Synthetic no surviving artifact','budget_stop':None,'initialization':init,'call_limits':[]}
    totals={'prompt_tokens':tokens-20,'completion_tokens':20,'total_tokens':tokens,'llm_calls':2,'cost':0.0}
    calls=[{'prompt':tokens-20,'completion':10,'cached':0,'usage_available':True},{'prompt':0,'completion':10,'cached':0,'usage_available':True}]
    released={**m.rendered(artifact),'error':None,'revision':{'final':m.rendered(artifact),'final_check_issues':[]},'budget':None}
    result={'version':1,'kind':'candidate_snapshot','candidate':label,'phase':kind,'released':released,
      'artifact':{**artifact,'mode':'final','code':code,'workspace_sha256':m.fingerprint(workspace)} if eligible else None,
      'workspace':workspace,'source_check':check if eligible else None,'usage':{'total_usage':totals,'per_call':calls,
      'raw_accounting':{'total_usage':deepcopy(totals),'call_usage':deepcopy(calls),'tool_calls_made':2},'observer':{}},
      'binding':{'execution_tool_call_id':'same-native-id','execution_trace_index':2,'source_check_tool_call_id':'automatic_field_check_1',
        'source_check_trace_index':5,'code_sha256':m.sha_text(code),'workspace_sha256':m.fingerprint(workspace)} if eligible else {},
      'workspace_sha256':m.fingerprint(workspace),'trace_sha256':m._aux_fingerprint(trace),'tagged_trace_sha256':m._aux_fingerprint(tagged),'diagnostics':deepcopy(diag)}
    return result,diag,tagged


def rehash(row):
    rev=row['revision'];snaps=rev['messages'];trace=row['trace']
    for snap in snaps:
        local=[t for t in trace if t['phase']==snap['phase']]
        snap['tagged_trace_sha256']=m._aux_fingerprint(local)
        snap['trace_sha256']=m._aux_fingerprint([{k:v for k,v in t.items() if k!='phase'} for t in local])
        snap['diagnostics']=deepcopy(next(p for p in rev['phases'] if p['phase']==snap['phase']))
    rev['candidate_snapshot_sha256']={s['candidate']:m._aux_fingerprint(s) for s in snaps}
    rev['A_only']['snapshot_sha256']=rev['candidate_snapshot_sha256']['A']


def fixture(status='selected_B',limited_B=False,unverified_A=False):
    a,ap,ta=snapshot('A',1,100,200000,30,30,eligible=not unverified_A)
    fixed=1000;bound=fixed+32768+512;reservation=m.reserve(bound)+2000
    admission={'A_entry_bytes':len(m.finite_json(m.candidate_entry(a,ta)).encode()),'future_B_entry_byte_bound':32768,
      'selector_fixed_prompt_tokens':fixed,'selector_prompt_bound':bound,'selector_token_reservation':reservation,
      'residual_tokens':199900,'residual_tools':28,'residual_sdk_attempts':28,'B_token_cap':199900-reservation,
      'B_natural_tool_cap':27,'B_sdk_cap':27}
    b,bp,tb=snapshot('B',1 if status=='agreement_preserved' else 2,1000,admission['B_token_cap'],27,27,
      limitation='Honest bounded observation' if limited_B else None,eligible=status!='B_ineligible')
    context={'question':'Return a synthetic field.','patient_fhir_id':'synthetic-patient','current_time':None,'assumptions':'Patient FHIR ID is synthetic-patient.'}
    q='Question: Return a synthetic field.\nContext:\nPatient FHIR ID is synthetic-patient.'
    init={'role':'tool','name':'initialize_original_starting_packet','tool_call_id':'v7-B-initialization','phase':'alternative_initialization',
      'host_operation':True,'content':m.finite_json({'question_sha256':m._aux_fingerprint(q),**bp['initialization']})}
    trace=ta+[init]+tb
    phases=[ap,bp];selection=None;chosen='B' if status=='selected_B' else 'A'
    if status in {'selected_B','selector_preserved_A'}:
        packet={'context':context,'A':m.candidate_entry(a,ta),'B':m.candidate_entry(b,tb)}
        admission.update(B_entry_bytes=len(m.finite_json(packet['B']).encode()),selector_prompt_tokens=2000,selector_packet_sha256=m._aux_fingerprint(packet))
        selection={'choice':'B' if chosen=='B' else 'A','reason':'Synthetic comparison',
          'requirements':[{'requirement':'Read the requested field','A':{'status':'contradicted','basis':'Synthetic A program'},'B':{'status':'supported','basis':'Synthetic B field'}}]}
        trace += [{'role':'system','phase':'selection','content':'Synthetic selector'},
          {'role':'user','phase':'selection','content':m.finite_json(packet)},
          {'role':'assistant','phase':'selection','content':m.finite_json(selection)}]
        phases.append({'phase':'selection','tokens':200,'tool_calls':0,'model_calls':1,'fatal':False,'errors':[],
          'prompt_tokens_estimate':2000,'reservation':reservation,'status':'responded','raw_accounting':{
            'total_usage':{'prompt_tokens':180,'completion_tokens':20,'total_tokens':200,'llm_calls':1,'cost':0.0},
            'call_usage':[{'prompt':180,'completion':20,'cached':0,'usage_available':True}],'tool_calls_made':0}})
    selected=a if chosen=='A' else b
    rev={'protocol':m.PROTOCOL,'status':status,'selected':chosen,'fatal':False,
      'A_only':{'available':True,'error':None,'source_bound':a['artifact'] is not None},'candidate_snapshot_sha256':{},
      'phases':phases,'selection':selection,'admission':admission,'messages':[a,b],
      'final_check_issues':[] if selected['artifact'] is not None else None,'independent_candidate_selected':chosen=='B'}
    row={'question_id':'synthetic','question':context['question'],'patient_fhir_id':'synthetic-patient','question_with_context':q,
      'agent_answer':selected['released']['answer'],'evidence':deepcopy(selected['released']['evidence']),
      'limitation':selected['released']['limitation'],'error':None,'latency':1,'revision':rev,'trace':trace,
      'retrieved_fhir_resources':deepcopy(selected['workspace']),
      'tokens':{'total':sum(p['tokens'] for p in phases),'llm_calls':sum(p['model_calls'] for p in phases),
        'per_call':[c for s in (a,b) for c in s['usage']['per_call']]+([{'prompt':180,'completion':20,'usage_available':True}] if len(phases)==3 else [])},
      'budget':{'max_total_tokens':200000,'max_tool_calls':30,'tool_calls_run':5,'stopped':None}}
    rehash(row);return row


class V7ProtocolTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        if m.sha(HERE/'v7_guard.py')!=START_SHA:raise AssertionError('Guard changed during synthetic suite')
    def validate(self,row):return m.attempt_counts(row,row,True,'Synthetic selector')
    def test_clean_switch_two_own_workspaces_duplicate_auto_ids_legal(self):
        c=self.validate(fixture());self.assertEqual(c['actual_B_source_bound'],1);self.assertEqual(c['source_valid_executed_releases'],1)
    def test_actual_A_preserved_on_selector_A(self):self.validate(fixture('selector_preserved_A'))
    def test_exact_agreement_preserves_A_without_selector(self):self.validate(fixture('agreement_preserved'))
    def test_ineligible_B_preserves_A(self):self.validate(fixture('B_ineligible'))
    def test_honest_B_limitation_is_visible_not_eligibility_veto(self):self.validate(fixture(limited_B=True))
    def test_unverified_prose_A_can_be_compared(self):self.validate(fixture(unverified_A=True))
    def test_workspace_swap_rejected(self):
        r=fixture();r['retrieved_fhir_resources']=r['revision']['messages'][0]['workspace']
        with self.assertRaisesRegex(ValueError,'Displayed workspace'):self.validate(r)
    def test_snapshot_digest_tamper_rejected(self):
        r=fixture();r['revision']['messages'][0]['released']['answer']=7
        with self.assertRaisesRegex(ValueError,'Snapshot SHA'):self.validate(r)
    def test_computed_unreleased_candidate_rejected(self):
        r=fixture();r['revision']['messages'][1]['released']['answer']=9;rehash(r)
        with self.assertRaisesRegex(ValueError,'Computed but unreleased'):self.validate(r)
    def test_source_check_from_other_namespace_rejected(self):
        r=fixture();r['revision']['messages'][1]['source_check']=r['revision']['messages'][0]['source_check'];rehash(r)
        with self.assertRaisesRegex(ValueError,'own source check'):self.validate(r)
    def test_missing_later_source_reply_rejected(self):
        r=fixture();r['trace']=[t for t in r['trace'] if not(t.get('phase')=='natural_alternative' and t.get('role')=='tool' and t.get('tool_call_id')=='automatic_field_check_1')];rehash(r)
        with self.assertRaises(ValueError):self.validate(r)
    def test_wrong_binding_index_rejected(self):
        r=fixture();r['revision']['messages'][1]['binding']['execution_trace_index']=1;rehash(r)
        with self.assertRaisesRegex(ValueError,'Binding call IDs'):self.validate(r)
    def test_false_requirement_not_found_cannot_switch(self):
        r=fixture();s=r['revision']['selection'];s['requirements'][0]['A']['status']='not_found';r['trace'][-1]['content']=m.finite_json(s)
        with self.assertRaisesRegex(ValueError,'strict existing'):self.validate(r)
    def test_unsupported_B_cannot_switch(self):
        r=fixture();s=r['revision']['selection'];s['requirements'][0]['B']['status']='not_found';r['trace'][-1]['content']=m.finite_json(s)
        with self.assertRaisesRegex(ValueError,'strict existing'):self.validate(r)
    def test_third_answer_schema_cannot_switch(self):
        r=fixture();s=r['revision']['selection'];s['answer']='invented';r['trace'][-1]['content']=m.finite_json(s)
        with self.assertRaisesRegex(ValueError,'choice schema'):self.validate(r)
    def test_selector_native_tools_rejected_when_claimed_parsed(self):
        r=fixture();r['trace'][-1]['tool_calls']=[{'id':'forbidden'}]
        with self.assertRaisesRegex(ValueError,'native tools'):self.validate(r)
    def test_selector_packet_truncation_rejected(self):
        r=fixture();p=json.loads(r['trace'][-2]['content']);p['B']['program']='';r['trace'][-2]['content']=m.finite_json(p)
        with self.assertRaisesRegex(ValueError,'intact candidate'):self.validate(r)
    def test_selector_system_prompt_rejected(self):
        r=fixture();r['trace'][-3]['content']='Changed prompt'
        with self.assertRaisesRegex(ValueError,'system prompt'):self.validate(r)
    def test_residual_wrong_token_cap_rejected(self):
        r=fixture();r['revision']['admission']['B_token_cap']+=1
        with self.assertRaisesRegex(ValueError,'Residual B caps'):self.validate(r)
    def test_omitted_B_initialization_charge_rejected(self):
        r=fixture();r['revision']['phases'][1]['tool_calls']=2;rehash(r)
        with self.assertRaisesRegex(ValueError,'initialization once'):self.validate(r)
    def test_initialization_mutation_rejected(self):
        r=fixture();p=json.loads(r['trace'][6]['content']);p['charged']=0;r['trace'][6]['content']=m.finite_json(p)
        with self.assertRaisesRegex(ValueError,'initialization metadata'):self.validate(r)
    def test_nonfatal_unknown_usage_rejected(self):
        r=fixture();r['tokens']['per_call'][0]['usage_available']=False
        with self.assertRaisesRegex(ValueError,'Unknown/overshot'):self.validate(r)
    def test_nonfatal_model_overshoot_rejected(self):
        r=fixture();r['tokens']['llm_calls']=31
        with self.assertRaisesRegex(ValueError,'Unknown/overshot'):self.validate(r)
    def test_duplicate_selector_keys_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):m.json_object('{"choice":"A","choice":"B"}')
    def test_nonfinite_selector_rejected(self):
        with self.assertRaises(ValueError):m.json_object('{"reason":NaN}')
    def test_unknown_auxiliary_usage_is_retained_failed_first_attempt(self):
        r=fixture('agreement_preserved');rev=r['revision'];b=rev['messages'][1];phase=rev['phases'][1]
        phase.update(fatal=True,usage_known=False,eligible=False,errors=['phase provider usage unknown'])
        b['artifact']=None;b['source_check']=None;b['binding']={};b['usage']['per_call'][0]['usage_available']=False
        b['usage']['total_usage']['cost']=None
        r['tokens']['per_call'][2]['usage_available']=False
        rev.update(status='fatal_alternative',fatal=True,final_check_issues=None)
        r.update(agent_answer=None,evidence=[],limitation='alternative usage unverifiable',error='V7 execution/accounting failure: unknown')
        rehash(r);c=self.validate(r)
        self.assertEqual(c['fatal_release_outcomes'],1);self.assertEqual(c['recorded_unknown_provider_usage_attempts'],1)
    def test_known_primary_overshoot_is_retained_failed_first_attempt(self):
        r=fixture('agreement_preserved');rev=r['revision'];a=rev['messages'][0];phase=rev['phases'][0]
        rev['messages']=[a];rev['phases']=[phase];phase.update(tokens=200001,fatal=True,eligible=False,errors=['actual ceiling overshoot'])
        a.update(artifact=None,source_check=None,binding={});a['usage']['total_usage']['total_tokens']=200001
        rev.update(status='fatal_primary',fatal=True,final_check_issues=None,admission=None)
        rev['A_only']['source_bound']=False
        r['trace']=[t for t in r['trace'] if t['phase']=='natural_primary'];r['tokens'].update(total=200001,llm_calls=2)
        r['tokens']['per_call']=r['tokens']['per_call'][:2];r['budget']['tool_calls_run']=2
        r.update(agent_answer=None,evidence=[],limitation='actual ceiling exceeded',error='V7 execution/accounting failure: ceiling')
        rehash(r);c=self.validate(r);self.assertEqual(c['reported_aggregate_token_overshoots'],1)
    def test_completion_overshot_nonfatal_selector_rejected(self):
        r=fixture();phase=r['revision']['phases'][2];raw=phase['raw_accounting'];raw['call_usage'][0]['completion']=2001
        with self.assertRaisesRegex(ValueError,'raw usage/completion'):self.validate(r)
    def test_fallback_error_preserved_without_computed_credit(self):
        r=fixture('B_ineligible',unverified_A=True);rev=r['revision'];a=rev['messages'][0]
        a['released'].update(answer=None,error='Original primary error')
        r.update(agent_answer=None,error='Original primary error');rev['A_only']['error']='Original primary error';rehash(r)
        rev['admission']['A_entry_bytes']=len(m.finite_json(m.candidate_entry(a,[t for t in r['trace'] if t['phase']=='natural_primary'])).encode())
        self.validate(r)
    def test_fallback_error_cannot_be_silently_dropped(self):
        r=fixture('B_ineligible',unverified_A=True);a=r['revision']['messages'][0]
        a['released']['error']='Original error';r['revision']['A_only']['error']='Original error';rehash(r)
        with self.assertRaisesRegex(ValueError,'actual released outcome or error'):self.validate(r)
    def test_root_exit_attestation_requires_distinct_owned_sessions(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'owned.json';p.write_text(m.finite_json({'both_owned_jobs_terminal':True,'accuracy_scored':False,'jobs':[
              {'arm':a,'owned_session_id':1,'exit_code':1,'saved_first_attempts':300,'observed_terminal_utc':'2026-01-01T00:00:00+00:00'} for a in ('single','team')]}))
            with self.assertRaisesRegex(ValueError,'one session'):m.verify_owned_attestation(p,m.sha(p))


if __name__=='__main__':unittest.main(verbosity=2)
