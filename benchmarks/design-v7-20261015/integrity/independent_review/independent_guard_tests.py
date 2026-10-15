"""Narrow synthetic independent guard regression; no actual run paths or main()."""
from copy import deepcopy
import argparse,hashlib,importlib.util,json,sys,unittest
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--guard-sha',required=True)
args,rest=parser.parse_known_args();sys.argv=[sys.argv[0],*rest]
assert hashlib.sha256((args.packet/'v7_guard.py').read_bytes()).hexdigest()==args.guard_sha
spec=importlib.util.spec_from_file_location('v7_synthetic_review_fixtures',args.packet/'test_v7_guard_protocol.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);m=t.m

def refresh(row):
    rev=row['revision'];t.rehash(row)
    if len(rev['messages'])>1:
        a,b=rev['messages'];ta=[x for x in row['trace'] if x['phase']=='natural_primary'];tb=[x for x in row['trace'] if x['phase']=='natural_alternative']
        rev['admission']['A_entry_bytes']=len(m.finite_json(m.candidate_entry(a,ta)).encode())
        selection=[x for x in row['trace'] if x['phase']=='selection']
        if selection:
            packet=json.loads(selection[1]['content']);packet['A']=m.candidate_entry(a,ta);packet['B']=m.candidate_entry(b,tb)
            selection[1]['content']=m.finite_json(packet)
            rev['admission']['selector_packet_sha256']=m._aux_fingerprint(packet)
            rev['admission']['B_entry_bytes']=len(m.finite_json(packet['B']).encode())
    row['tokens']['total']=sum(p['tokens'] for p in rev['phases'])
    row['tokens']['llm_calls']=sum(p['model_calls'] for p in rev['phases'])
    row['budget']['tool_calls_run']=sum(p['tool_calls'] for p in rev['phases'])

def zero_init(row):
    p=row['revision']['phases'][1];p['initialization']={'started':0,'completed':0,'charged':0,'kind':None}
    p['tool_calls']=p['natural_tool_calls']
    init=next(x for x in row['trace'] if x['phase']=='alternative_initialization')
    init['content']=m.finite_json({'question_sha256':m._aux_fingerprint(row['question_with_context']),**p['initialization']})

def zero_selector(row):row['revision']['phases'][2]['model_calls']=0

def undercharge_B_tools(row):
    p=row['revision']['phases'][1];p['natural_tool_calls']=1;p['tool_calls']=2
    row['revision']['messages'][1]['usage']['raw_accounting']['tool_calls_made']=1

def phantom_B_call(row):
    p=row['revision']['phases'][1];p['model_calls']=3
    u=row['revision']['messages'][1]['usage'];u['total_usage']['llm_calls']=3
    u['per_call'].append({'prompt':0,'completion':0,'cached':0,'usage_available':True})
    u['raw_accounting']['total_usage']=deepcopy(u['total_usage']);u['raw_accounting']['call_usage']=deepcopy(u['per_call'])

def known_error_base():
    row=t.fixture('B_ineligible');b=row['revision']['messages'][1]
    b['released'].update(answer=None,evidence=[],error='Known-usage admitted synthetic failure')
    row['revision']['phases'][1]['skip_reason']='primary returned an error'
    return row

class IndependentGuardBoundaries(unittest.TestCase):
    def check(self,row):return m.attempt_counts(row,row,True,'Synthetic selector')
    def test_clean_two_owned_natural_contexts_still_pass(self):self.check(t.fixture())
    def test_eligible_B_requires_successful_original_initialization(self):
        row=t.fixture();zero_init(row);refresh(row)
        with self.assertRaises(ValueError):self.check(row)
    def test_parsed_selector_cannot_be_an_uncharged_zero_call(self):
        row=t.fixture();zero_selector(row);refresh(row)
        with self.assertRaises(ValueError):self.check(row)
    def test_eligible_B_charges_both_actual_execution_and_source_tools(self):
        row=t.fixture();undercharge_B_tools(row);refresh(row)
        with self.assertRaises(ValueError):self.check(row)
    def test_eligible_B_cannot_invent_a_model_call_without_retained_response(self):
        row=t.fixture();phantom_B_call(row);refresh(row)
        with self.assertRaises(ValueError):self.check(row)
    def test_discarded_B_known_usage_provider_failure_may_lack_response(self):
        row=known_error_base();phase=row['revision']['phases'][1];b=row['revision']['messages'][1]
        # First ordinary source check requests revision; the next billed request
        # fails after known usage was captured, hence no third natural response.
        last=next(x for x in reversed(row['trace']) if x.get('phase')=='natural_alternative' and x.get('role')=='tool')
        check=json.loads(last['content']);check['issues']=['synthetic source issue'];last['content']=m.finite_json(check)
        row['trace'].append({'role':'user','phase':'natural_alternative','content':'Synthetic original source feedback'})
        b['released']['revision'].update(final_check_issues=['synthetic source issue'])
        phase['model_calls']=3;phase['tokens']+=110
        usage=b['usage'];usage['per_call'].append({'prompt':100,'completion':10,'cached':0,'usage_available':True})
        usage['total_usage'].update(llm_calls=3,total_tokens=phase['tokens'],prompt_tokens=1080,completion_tokens=30)
        usage['raw_accounting']['total_usage']=deepcopy(usage['total_usage']);usage['raw_accounting']['call_usage']=deepcopy(usage['per_call'])
        refresh(row);counts=self.check(row)
        self.assertEqual(counts['status_B_ineligible'],1)
        self.assertEqual(row['agent_answer'],1)
        self.assertEqual(counts['phase_B_charged_minus_observed_model_attempts'],1)
    def test_discarded_B_charged_tool_exception_may_lack_tool_reply(self):
        row=known_error_base();phase=row['revision']['phases'][1];b=row['revision']['messages'][1]
        local=[x for x in row['trace'] if x.get('phase')=='natural_alternative']
        # Single admitted one native execution request, charged it, then its
        # tool handler raised before appending the reply. Base returns error.
        row['trace']=[x for x in row['trace'] if x.get('phase')!='natural_alternative']+local[:2]
        phase.update(model_calls=1,natural_tool_calls=1,tool_calls=2,tokens=990)
        usage=b['usage'];usage['per_call']=usage['per_call'][:1]
        usage['total_usage'].update(llm_calls=1,total_tokens=990,prompt_tokens=980,completion_tokens=10)
        usage['raw_accounting'].update(total_usage=deepcopy(usage['total_usage']),call_usage=deepcopy(usage['per_call']),tool_calls_made=1)
        refresh(row);counts=self.check(row)
        self.assertEqual(counts['status_B_ineligible'],1)
        self.assertEqual(row['agent_answer'],1)
        self.assertEqual(counts['phase_B_charged_minus_observed_tool_calls'],1)
    def test_discarded_B_failure_before_original_packet_has_zero_charge(self):
        row=known_error_base();zero_init(row)
        phase=row['revision']['phases'][1];b=row['revision']['messages'][1]
        row['trace']=[x for x in row['trace'] if x.get('phase')!='natural_alternative']
        phase.update(model_calls=0,natural_tool_calls=0,tool_calls=0,tokens=0)
        b['workspace']={};b['workspace_sha256']=m.fingerprint({})
        b['released'].update(error='Synthetic failure before original initialization',revision=None)
        totals={'prompt_tokens':0,'completion_tokens':0,'total_tokens':0,'llm_calls':0,'cost':0.0}
        b['usage'].update(total_usage=totals,per_call=[],raw_accounting={'total_usage':deepcopy(totals),'call_usage':[],'tool_calls_made':0})
        refresh(row);counts=self.check(row)
        self.assertEqual(counts['status_B_ineligible'],1)
        self.assertEqual(row['budget']['tool_calls_run'],2)
    @classmethod
    def tearDownClass(cls):
        assert hashlib.sha256((args.packet/'v7_guard.py').read_bytes()).hexdigest()==args.guard_sha

if __name__=='__main__':unittest.main(verbosity=2)
