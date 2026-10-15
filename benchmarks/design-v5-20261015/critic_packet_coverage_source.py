"""Count-only packet sizing on already exposed completed V3d primary artifacts.
Never reads reference answers, scorer output or correctness for selection.
"""
import gzip,hashlib,json
from pathlib import Path
from collections import Counter
from agent.executable_critic_agent import critic_packet, LIMITS, _prompt_reserve
from agent.query_contract_agent import fingerprint
from agent.controller import FixedController
from utils.core_utils import count_tokens_in_messages

root=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
run=root/'output/experiments/gpt-6-luna/query_contract_v3d_300q'
rows=json.loads((run/'ordinary_multi_agent.json').read_text())
sizes=[]; reasons=Counter(); components={k:[] for k in ['system','context','code','ledger','preview']}
for row in rows:
    revision=row.get('revision') or {}
    if not revision.get('primary') or revision.get('final_check_issues')!=[] or revision.get('unresolved_scope'):
        reasons['primary_not_source_valid']+=1;continue
    with gzip.open(run/row['audit_artifacts']['path'],'rt') as handle:
        audit=json.load(handle)
    trace=audit['trace']; workspace=audit['retrieved_fhir_resources']
    calls={}; final=None
    for message in trace:
        if not isinstance(message,dict):continue
        if message.get('role')=='assistant':
            for call in message.get('tool_calls') or []:
                function=call.get('function') or {}
                if function.get('name')=='execute_python_code':
                    args=json.loads(function['arguments'])
                    if args.get('mode')=='final': calls[call['id']]=args['code']
        if message.get('role')=='tool' and message.get('tool_call_id') in calls:
            output=json.loads(message['content'])
            if not output.get('error') and 'answer' in output:
                final={'code':calls[message['tool_call_id']], 'workspace_sha256':fingerprint(workspace)}
    if final is None:reasons['no_executed_code_recovered']+=1;continue
    context=FixedController.question_context(row['question_with_context'])
    packet,messages=critic_packet(context,final,workspace,revision['retrieval_ledger'],revision['complete_types'],revision['unresolved_scope'])
    p=count_tokens_in_messages(messages)
    reserve=(_prompt_reserve(p)+LIMITS['first_completion']+
        _prompt_reserve(p+LIMITS['first_response_tokens']+LIMITS['probe_tokens']+300)+LIMITS['second_completion'])
    sizes.append(p)
    reasons['fits' if p<=LIMITS['first_prompt_tokens'] and reserve<=25000 else 'intact_packet_exceeds_cap']+=1
    for name,value in [('system',messages[0]['content']),('context',packet['context']),('code',packet['final_code']),('ledger',packet['retrieval_ledger']),('preview',packet['inventory_preview'])]:
        components[name].append(count_tokens_in_messages([{'role':'user','content':value if isinstance(value,str) else json.dumps(value)}]))
def distribution(values):
    values=sorted(values)
    if not values:return {}
    return {str(p):values[round((len(values)-1)*p/100)] for p in [0,25,50,75,90,95,100]}
source=Path(__import__('agent.executable_critic_agent',fromlist=['x']).__file__)
report={'purpose':'score-blind engineering packet sizing, completed/exposed V3d only',
 'input_rows':len(rows),'eligible_packets':len(sizes),'counts':dict(reasons),'first_prompt_token_percentiles':distribution(sizes),
 'component_token_percentiles':{k:distribution(v) for k,v in components.items()},
 'limits':LIMITS,'agent_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'no_model_calls':True,'no_scoring_or_correctness_selection':True,'no_unseen_question_inputs':True}
Path('/private/tmp/fhir-v5-critic-coverage.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
