from copy import deepcopy
import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path('/private/tmp/fhir-v7-prospective-integrity-20261015')
assert hashlib.sha256((ROOT/'v7_guard.py').read_bytes()).hexdigest()=='e0ba8a58e5f36c7db9e728b418601fccdce01aa52a8a78eb7c2b08c3ecf5db93'
spec=importlib.util.spec_from_file_location('synthetic_guard_fixtures',ROOT/'test_v7_guard_protocol.py')
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
    p=row['revision']['phases'][1]
    p['initialization']={'started':0,'completed':0,'charged':0,'kind':None}
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

results=[]
for name,mutation in [('eligible_B_without_initialized_packet',zero_init),('parsed_selector_response_zero_calls',zero_selector),('eligible_B_two_actual_tools_charged_one',undercharge_B_tools),('eligible_B_phantom_model_call_without_reply',phantom_B_call)]:
    row=t.fixture();mutation(row);refresh(row)
    try:
        counts=m.attempt_counts(row,row,True,'Synthetic selector')
        results.append({'case':name,'accepted':True,'selected':row['revision']['selected'],'counts':dict(counts)})
    except Exception as e:results.append({'case':name,'accepted':False,'exception':type(e).__name__,'message':str(e)})
print(json.dumps(results,indent=2))
