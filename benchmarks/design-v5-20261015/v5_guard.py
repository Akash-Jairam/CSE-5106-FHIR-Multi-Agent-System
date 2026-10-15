"""Model-free V5 subset prelaunch, initial and completion integrity checks.

Root must preregister the pins file and its SHA before launch. Initial checks are
saved snapshots, not completion/accuracy claims. Completion requires an explicit
external owned-jobs-terminal assertion. No scorer, model, FHIR or credential use.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

BASE_SHA = '98d272f8ed8e42914b5ed1776908cf6e25e209e5726d5386b4c168a270a499e8'
BASE_PATH = Path(__file__).with_name('reviewed_v4_guard.py')
assert hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() == BASE_SHA, 'Reviewed base guard differs'
spec = importlib.util.spec_from_file_location('v5_reviewed_integrity_base', BASE_PATH)
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
require, sha, equal = B.require, B.sha, B.json_equal
ARM_STRATEGIES = {'ordinary_single_agent':'single_agent', 'ordinary_multi_agent':'multi_agent_v5'}
PROTOCOL = 'frozen-workspace-executable-critic-v5'
PHASE_CAPS = {'solver':(175000,27), 'critic':(25000,3)}
ASSURANCE = 'fallible_model_semantics_not_global_proof'
FIXED_INPUT_HASH = '25e57321644744d5c0718c949c1f90e8be4f53b8aadce4d0118843a99c7b96ce'
FIXED_DATASET_SHA = 'e2045692fef7f5f4f77496935160f5fc727e162d213e94feed61401948e512a0'
FIXED_SPLIT_SHA = 'f28db462ebcfe4e0d87675c86de9a08adef4038e26dbddcf88c50ddf92e94015'
HEX = re.compile(r'^[a-f0-9]{64}$')
RECORD_META = ('manifest.json','runtime_adapter_manifest.json','expected_public_binding.json','prospective_protocol.json')
REQUIRED_SOURCE = {'agent/executable_critic_agent.py','agent/query_contract_agent.py','agent/single_agent.py',
                   'scripts/audit_store.py','run_agent.py','requirements.txt'}
PROVIDER_SOURCE = {
 '.local-setup/azure_adapter.py':'41f300c5adad350a41fc8e9d296820c526cd4689ab3621436bfe1ad39646d4bf',
 '.local-setup/sitecustomize.py':'fb3eea8ce917715e6b553f501cc4abf2fa5408570f198e7aaa6392e72f22e9fe',
 '.local-setup/azure.py':'1144197dae1282b7472ddd90dc703ac02d17f87efa9b6327d42adce1921b13c4',
 '.local-setup/phase3.py':'85182719e3c5020d527c97a454d80f4616caa7abc247a9b9b352e25077ce61d3'}
PUBLIC_BINDING = {'provider_profile_sha256':'493987643d4523481d1c2ed077747d43390f806f1665ab6a9d605f8a2251df1c',
 'adapter_sha256':PROVIDER_SOURCE['.local-setup/azure_adapter.py'],
 'bootstrap_sha256':PROVIDER_SOURCE['.local-setup/sitecustomize.py']}
TOOL_IDS = {'v5-critic-inspect':('execute_python_code','inspect'),
            'v5-critic-final':('execute_python_code','final'),
            'v5-critic-source-check':('check_fields',None)}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str,allow_nan=False).encode()).hexdigest()


def load(path):
    def pairs(items):
        result = {}
        for key,value in items:
            require(key not in result,'Duplicate metadata JSON key')
            result[key]=value
        return result
    def bad(value): raise ValueError('Non-finite metadata JSON')
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=bad)


def public_path(root,name):
    p = B.public_source_path(root,name)
    require(not any(part.startswith('.') for part in Path(name).parts) or name in PROVIDER_SOURCE,
            'Unallowlisted hidden source path')
    require(not p.is_symlink() and p.is_file() and p.resolve().is_relative_to(root.resolve()),'Unsafe public source file')
    return p


def pin_record(record,destination):
    """Prepare, but never silently trust, pins for root review BEFORE launch."""
    require(not destination.exists() and not destination.is_symlink(),'Pins destination already exists')
    manifest=load(record/'manifest.json')
    require(REQUIRED_SOURCE <= set(manifest['source_sha256']),'Required V5 source inventory is missing')
    pins={'version':1,'status':'REQUIRES_ROOT_REVIEW_AND_PRESPECIFICATION_BEFORE_LAUNCH',
          'source_inventory_sha256':fingerprint(manifest['source_sha256']),
          'source_file_count':len(manifest['source_sha256']),
          'record_metadata_sha256':{name:sha(record/name) for name in RECORD_META},
          'guard_sha256':sha(Path(__file__)),'reviewed_base_sha256':BASE_SHA}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as stream:stream.write(json.dumps(pins,indent=2,allow_nan=False)+'\n')
    return pins


def verify_pins(record,path,digest):
    require(isinstance(digest,str) and HEX.fullmatch(digest),'Externally prespecified pins SHA required')
    require(sha(path)==digest,'Prespecified guard pins differ')
    pins=load(path)
    require(pins.get('version')==1 and pins.get('guard_sha256')==sha(Path(__file__))
            and pins.get('reviewed_base_sha256')==BASE_SHA,'Guard implementation/pin version differs')
    meta=pins.get('record_metadata_sha256')
    require(isinstance(meta,dict) and set(meta)==set(RECORD_META),'Pin metadata inventory differs')
    for name,value in meta.items():
        require(isinstance(value,str) and HEX.fullmatch(value) and sha(record/name)==value,'Prespecified record metadata differs: '+name)
    manifest=load(record/'manifest.json');mapping=manifest.get('source_sha256')
    require(isinstance(mapping,dict) and REQUIRED_SOURCE <= set(mapping),'V5 source inventory incomplete')
    require(type(pins.get('source_file_count')) is int and len(mapping)==pins['source_file_count']
            and fingerprint(mapping)==pins.get('source_inventory_sha256'),'Prespecified source inventory differs')
    return pins,manifest


def verify_prospective(prospective,binding):
    expected={'model':'gpt-6-luna','fixed_questions_per_arm':300,'workers_per_arm':1,'first_attempt_only':True,
      'phase_caps':{k:list(v) for k,v in PHASE_CAPS.items()},'solver_requested_tools':25,
      'reserved_counted_source_checks':2,'max_real_solver_calls':20,'max_host_slots':22,
      'max_critic_invocations':2,'global_caps':[200000,30],'estimated_input_guard':32000,
      'minimum_team_correct':240,'expected_public_binding':binding}
    require(all(equal(prospective.get(k),v) for k,v in expected.items()),'Prospective V5 profile/caps differ')


def frozen_provenance(root,record,output,pins_path,pins_sha,prelaunch=False):
    pins,m=verify_pins(record,pins_path,pins_sha)
    for name,value in m['source_sha256'].items():
        require(isinstance(value,str) and HEX.fullmatch(value) and sha(public_path(root,name))==value,'Frozen source differs: '+name)
    B.source_zip(record/'source.zip',m['source_sha256'],m['source_archive_sha256'])
    require(sha(record/'design.txt')==m['design_sha256'],'Frozen design differs')
    require(m['dataset_sha256']==FIXED_DATASET_SHA and sha(root/'final_dataset/questions_answers_sql_fhir.csv')==FIXED_DATASET_SHA,'Fixed input dataset differs')
    require(m['split_sha256']==FIXED_SPLIT_SHA and sha(root/'manifests/split.json')==FIXED_SPLIT_SHA,'Fixed subset manifest differs')
    require(m.get('model')=='gpt-6-luna' and equal(m.get('max_total_tokens'),200000)
            and equal(m.get('max_tool_calls'),30) and equal(m.get('expected_questions_per_arm'),300)
            and set(m.get('arms',[]))==set(ARM_STRATEGIES),'Frozen paired run profile differs')
    require(Path(m['output_directory']).resolve()==output,'Manifest output path differs')
    values={}
    for node in ast.parse((root/'agent/executable_critic_agent.py').read_text()).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in {'PROTOCOL','PHASE_CAPS','ASSURANCE'}:
                    values[target.id]=ast.literal_eval(node.value)
    require(values=={'PROTOCOL':PROTOCOL,'PHASE_CAPS':PHASE_CAPS,'ASSURANCE':ASSURANCE},'Frozen V5 literal protocol differs')
    adapter=load(record/'runtime_adapter_manifest.json')
    require(equal(adapter.get('source_sha256'),PROVIDER_SOURCE),'Expected unchanged four public provider sources')
    for name,value in PROVIDER_SOURCE.items():require(sha(public_path(root,name))==value,'Public provider source differs')
    B.source_zip(record/'runtime_adapter_sources.zip',PROVIDER_SOURCE,adapter['archive_sha256'])
    binding=load(record/'expected_public_binding.json')
    require(equal(binding,PUBLIC_BINDING),'Prespecified common public provider binding differs')
    verify_prospective(load(record/'prospective_protocol.json'),binding)
    if prelaunch:require(not output.exists(),'Prelaunch output must be absent; no silent resume')
    else:require(equal(load(B.owned_file(output,'.local-provider.json')),binding),'Actual public binding differs')
    return pins,m,adapter,binding


def full_artifact(value):
    return isinstance(value,dict) and not value.get('error') and all(k in value for k in ('answer','evidence','operation','contributing_ids','limitation'))


def rendered(value):
    return {k:value.get(k) for k in ('answer','evidence','limitation')}


def context_from_row(row):
    text=row['question_with_context'];match=re.search(r'\nContext:(.*)\Z',text,re.S)
    current=re.search(r'current time is (\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?)',text)
    now=current.group(1) if current else None
    if now and len(now)==16:now+=':00'
    return {'question':row['question'],'patient_fhir_id':row['patient_fhir_id'],'current_time':now,
            'assumptions':match.group(1).strip() if match else ''}


def field_value(workspace,fact):
    require(isinstance(fact,dict) and set(fact)=={'resource_id','field_path','value'},'Malformed source fact')
    identifier=fact['resource_id'];require(isinstance(identifier,str) and identifier.count('/')==1,'Malformed fact ID')
    kind,key=identifier.split('/');records=[r for r in workspace.get(kind,[]) if r.get('id')==key]
    require(len(records)==1,'Source fact owner is not held uniquely')
    value=records[0]
    require(isinstance(fact['field_path'],str) and fact['field_path'],'Malformed fact field path')
    for part in fact['field_path'].split('.'):
        match=re.fullmatch(r'([A-Za-z_][\w-]*)((?:\[\d+\])*)',part)
        require(match and isinstance(value,dict) and match[1] in value,'Source fact path is missing/nonconcrete')
        value=value[match[1]]
        for number in re.findall(r'\[(\d+)\]',part):
            require(isinstance(value,list) and int(number)<len(value),'Source fact index is missing')
            value=value[int(number)]
    require(equal(value,fact['value']),'Source fact differs from frozen workspace')


def v5_protocol_counts(row,restored):
    counts=Counter();revision=restored.get('revision') or {}
    if revision.get('protocol')!=PROTOCOL:
        require(B.failed(row),'Nonterminal V5 row lacks protocol metadata')
        counts['terminal_errors_without_protocol_metadata']+=1;return counts
    phases=revision.get('phases') or [];names=[p['phase'] for p in phases]
    require(names in (['solver'],['solver','critic']),'V5 phase order differs')
    for phase in phases:require(equal([phase['token_cap'],phase['tool_cap']],list(PHASE_CAPS[phase['phase']])),'V5 declared phase caps differ')
    fatal=any(bool(p.get('fatal')) for p in phases);tokens=row.get('tokens') or {};total=tokens.get('total')
    unknown=any(p.get('usage_available') is not True for p in tokens.get('per_call') or [])
    policy=revision.get('release_policy');primary=revision.get('primary');final=revision.get('final')
    require(policy in {'primary_failed_execution_or_source_contract','source_valid_primary_preserved',
                       'executed_source_checked_critic_proposal','aggregate_accounting_failed'},'Unexpected V5 release policy')
    clean=policy in {'source_valid_primary_preserved','executed_source_checked_critic_proposal'}
    require(isinstance(final,dict),'Missing retained V5 final')
    require(equal(rendered(final),{'answer':row.get('agent_answer'),'evidence':row.get('evidence'),'limitation':row.get('limitation')}),'Rendered final differs from inline output')
    if clean:
        require(not B.failed(row) and not fatal and not unknown and revision.get('final_check_issues')==[]
                and not revision.get('unresolved_scope') and full_artifact(primary) and full_artifact(final),'Unsafe claimed source-valid release')
    trace=restored.get('trace') or [];calls={};replies={};reply_positions={};packets=[];packet_positions=[];critic_start=None
    for index,message in enumerate(trace):
        if not isinstance(message,dict):continue
        if message.get('role')=='system' and isinstance(message.get('content'),str) and message['content'].startswith('Review the actual executed program against the ORIGINAL question'):
            require(critic_start is None,'Duplicate critic system segment');critic_start=index
        if message.get('role')=='user' and isinstance(message.get('content'),str):
            try:packet=json.loads(message['content'])
            except (TypeError,ValueError):packet=None
            if isinstance(packet,dict) and {'final_code','code_sha256','workspace_sha256','context'}<=set(packet):
                packets.append(packet);packet_positions.append(index)
        if message.get('phase')=='critic' and message.get('role')=='assistant':
            for call in message.get('tool_calls') or []:
                require(call.get('id') in TOOL_IDS and call['id'] not in calls,'Unexpected/duplicate critic host tool')
                function=call.get('function') or {};args=json.loads(function.get('arguments') or '{}');name,mode=TOOL_IDS[call['id']]
                require(function.get('name')==name and (mode is None or args.get('mode')==mode),'Critic host tool/mode differs')
                require(set(args)==({'citations'} if mode is None else {'code','mode'}),'Unexpected critic host arguments')
                calls[call['id']]=(args,index)
        if message.get('phase')=='critic' and message.get('role')=='tool':
            key=message.get('tool_call_id');require(key in calls and key not in replies and message.get('name')==TOOL_IDS[key][0],'Critic tool reply has no unique owned call')
            replies[key]=json.loads(message.get('content') or '{}');reply_positions[key]=index
            counts['observed_critic_host_tool_replies']+=1
    require(len(packets)<=1,'Duplicate critic packet')
    primary_code=None
    if clean:
        require((revision.get('primary_revision') or {}).get('final_check_issues')==[], 'Clean primary lacks its clean final source check')
        primary_code=validate_primary(trace[:critic_start] if critic_start is not None else trace,primary,restored.get('retrieved_fhir_resources',{}))
    critic=revision.get('critic')
    if names==['solver']:
        require(policy=='primary_failed_execution_or_source_contract' and critic is None and not calls and not packets,'Critic present after failed primary')
    else:
        require(isinstance(critic,dict) and critic.get('assurance')==ASSURANCE,'Missing critic diagnostics/assurance')
        status=critic.get('status');require(status in {'coverage_skip','primary_preserved','released','accounting_failed'},'Unexpected critic status')
        counts['critic_status_'+status]+=1
        invocations=critic.get('model_calls');tools=critic.get('tool_calls')
        require(type(invocations) is int and invocations>=0 and type(tools) is int and tools>=0,'Malformed critic invocation/tool counters')
        counts['critic_admitted_invocations']+=invocations
        counts['critic_charged_tools']+=tools
        violation=invocations>2 or tools>3
        counts['reported_critic_cap_violations']+=violation
        if violation:require(not clean and (fatal or B.failed(row)),'Nonfatal release exceeds critic caps')
        if len(calls)!=tools or set(calls)!=set(replies):counts['critic_host_trace_counter_gaps']+=1
        if 'workspace_sha256' in critic:require(critic['workspace_sha256']==fingerprint(restored.get('retrieved_fhir_resources',{})),'Critic frozen workspace hash differs')
        if packets:
            packet=packets[0]
            require(critic_start is not None and equal(packet['context'],context_from_row(row)),'Actual critic context differs')
            require(packet['workspace_sha256']==fingerprint(restored['retrieved_fhir_resources'])
                    and packet['workspace_sha256']==critic.get('workspace_sha256'),'Critic packet workspace differs')
            require(sha_text(packet['final_code'])==packet['code_sha256']==critic.get('primary_code_sha256'),'Critic packet program differs')
            require(fingerprint(packet)==critic.get('packet_sha256'),'Actual critic packet hash differs')
            require(equal(packet.get('retrieval_ledger'),revision.get('retrieval_ledger'))
                    and equal(packet.get('complete_types'),revision.get('complete_types'))
                    and equal(packet.get('unresolved_scope'),revision.get('unresolved_scope')),'Critic packet ledger/scope differs')
            if clean:require(primary_code==packet['final_code'],'Critic program does not bind the current successful primary execution')
            counts['critic_observed_packet_bound_to_primary']+=1
        elif invocations or status=='released':
            require(not clean,'Invoked critic lacks its actual packet')
            counts['critic_invocation_packet_gaps']+=1
        response_positions=[i for i,m in enumerate(trace) if i>(critic_start if critic_start is not None else len(trace)) and isinstance(m,dict)
                   and m.get('role')=='assistant' and m.get('phase')!='critic' and m.get('model_call') is not False]
        responses=[trace[i] for i in response_positions]
        counts['observed_critic_model_responses']+=len(responses)
        if len(responses)>invocations:require(not clean,'Critic response count exceeds admitted invocations')
        if clean:
            require(len(calls)==tools and set(calls)==set(replies), 'Clean critic host-tool evidence has gaps')
            counts['critic_invocations_without_retained_response']+=max(0,invocations-len(responses))
            if invocations:
                require(critic_start is not None and len(packets)==1 and critic_start<packet_positions[0]
                        and (not response_positions or packet_positions[0]<response_positions[0]), 'Clean critic packet order differs')
            for key,position in (('first',0),('proposal',1)):
                if key in critic:
                    require(len(responses)>position and equal(strict_response(responses[position].get('content'),12000 if key=='first' else 24000),critic[key]), 'Critic retained judgment differs from actual response')
        if status=='released':
            require(clean and policy=='executed_source_checked_critic_proposal' and invocations==2 and len(responses)==2
                    and tools==3 and all(not r.get('tool_calls') for r in responses) and set(calls)==set(TOOL_IDS) and set(replies)==set(TOOL_IDS),'Repair path lacks exact bounded execution trace')
            require(revision.get('critic_revised') is True and revision.get('targeted_revised') is True,'Released critic repair is not recorded as revised')
            require(response_positions[0]<calls['v5-critic-inspect'][1]<reply_positions['v5-critic-inspect']<response_positions[1]
                    <calls['v5-critic-final'][1]<reply_positions['v5-critic-final']<calls['v5-critic-source-check'][1]<reply_positions['v5-critic-source-check'], 'Repair host stages are out of order')
            validate_repair(critic,packets[0],restored['retrieved_fhir_resources'],calls,replies,primary,final)
            counts['executed_source_checked_repairs']+=1
        elif clean:
            require(policy=='source_valid_primary_preserved' and equal(final,primary)
                    and revision.get('critic_revised') is False and revision.get('targeted_revised') is False,'Auxiliary fallback changed the valid primary')
        if status=='accounting_failed':require(not clean and policy=='aggregate_accounting_failed' and fatal and row.get('agent_answer') is None,'Unknown/fatal critic accounting did not fail closed')
    signals=sum(isinstance(m,dict) and m.get('role')=='user' and isinstance(m.get('content'),str)
                and m['content'].startswith('Solver allocation remaining:') for m in trace)
    boundaries=[m for m in trace if isinstance(m,dict) and m.get('phase')=='solver_boundary']
    automatic=[m for m in trace if isinstance(m,dict) and m.get('role')=='tool'
               and m.get('name')=='check_fields' and str(m.get('tool_call_id','')).startswith('automatic_field_check_')]
    require(all(m.get('model_call') is False for m in boundaries),'Host boundary labeled as a model call')
    counts['solver_invocation_attempt_signals']+=signals;counts['solver_host_boundaries']+=len(boundaries)
    counts['solver_automatic_source_checks']+=len(automatic)
    counts['reported_solver_limit_violations']+=signals>20 or signals+len(boundaries)>22 or len(automatic)>2
    solver_tools=phases[0]['tool_calls'];counts['reported_ordinary_solver_tool_violations']+=solver_tools-len(automatic)>25
    counts['reported_model_call_violations']+=(tokens.get('llm_calls') or 0)>22
    if clean:
        require(not counts['reported_solver_limit_violations'] and not counts['reported_ordinary_solver_tool_violations']
                and not counts['reported_model_call_violations'], 'Clean release exceeds solver/model limits')
        require(type(tokens.get('llm_calls')) is int and tokens['llm_calls']==len(tokens.get('per_call') or [])
                and tokens['llm_calls']>0, 'Clean release lacks canonical per-call usage evidence')
    counts['source_valid_releases']+=clean;counts['source_valid_null_answers']+=clean and row.get('agent_answer') is None
    counts['null_answers_outside_source_valid_release']+=not clean and row.get('agent_answer') is None
    counts['solver_source_revised']+=bool(revision.get('solver_source_revised'))
    return counts


def sha_text(value):
    require(isinstance(value,str),'Missing retained code text')
    return hashlib.sha256(value.encode()).hexdigest()


def canonical_artifact(execution,primary=False):
    """Mirror only existing public _draft/Single contributor normalization.

    Raw tool envelopes remain checked byte-for-JSON-value against their retained
    diagnostic copies. This permits ordered contributor dedup and Single's cited
    ID union; it never repairs answer, evidence, operation or limitation values.
    """
    value={k:execution.get(k) for k in ('answer','evidence','operation','contributing_ids','limitation')}
    ids=value['contributing_ids']
    require(isinstance(ids,list) and all(isinstance(i,str) for i in ids),'Malformed executed contributor IDs')
    if primary:
        ids=[*ids,*(c.get('resource_id') for c in value['evidence'] if isinstance(c,dict))]
    value['contributing_ids']=list(dict.fromkeys(ids))
    return value


def text(value,limit,empty=False):
    return isinstance(value,str) and len(value)<=limit and (empty or bool(value.strip()))


def strict_response(content,limit):
    require(text(content,limit),'Critic response exceeds text bounds')
    def pairs(items):
        value={}
        for key,item in items:
            require(key not in value,'Duplicate critic JSON key');value[key]=item
        return value
    value=json.loads(content,object_pairs_hook=pairs)
    json.dumps(value,allow_nan=False)
    require(isinstance(value,dict),'Critic response is not an object')
    return value


def validate_primary(trace,primary,workspace):
    """Bind a current explicit final to its later clean automatic source check."""
    pending={};current=None;checked=None
    for message in trace:
        if not isinstance(message,dict):continue
        for call in message.get('tool_calls') or []:
            f=call.get('function') or {};name=f.get('name');identifier=call.get('id')
            try:args=json.loads(f.get('arguments') or '{}')
            except (TypeError,ValueError):args={}
            pending[identifier]=(name,args)
            if name in {'get_resources_by_patient_fhir_id','get_resources_by_resource_id'}:
                current=None;checked=None
            if name=='execute_python_code' and args.get('mode')!='inspect':current=None;checked=None
        if message.get('role')!='tool' or message.get('tool_call_id') not in pending:continue
        identifier=message['tool_call_id'];name,args=pending[identifier]
        try:result=json.loads(message.get('content') or '{}')
        except (TypeError,ValueError):result={}
        if name=='execute_python_code' and args.get('mode')=='final' and full_artifact(result):
            current=(canonical_artifact(result,primary=True),args.get('code'));checked=None
        if name=='check_fields' and str(identifier).startswith('automatic_field_check_'):
            results=result.get('results') if isinstance(result,dict) else None
            valid=(current is not None and result.get('issues')==[] and equal(args.get('citations'),current[0]['evidence'])
                   and isinstance(results,list) and len(results)==len(current[0]['evidence'])
                   and all(isinstance(r,dict) and equal(r.get('citation'),c) and r.get('status')=='valid'
                           and r.get('warnings')==[] for c,r in zip(current[0]['evidence'],results)))
            if valid:checked=result
            else:current=None;checked=None
    require(current is not None and equal(current[0],primary),'Clean primary lacks current matching actual explicit final execution')
    require(checked is not None,'Clean primary lacks a later actual automatic source check')
    validate_citations(primary.get('evidence'),checked,workspace)
    return current[1]


def validate_citations(evidence,check,workspace):
    require(isinstance(evidence,list),'Missing source evidence list')
    results=check.get('results');require(isinstance(results,list) and len(results)==len(evidence),'Source-check result coverage differs')
    for citation,result in zip(evidence,results):
        # Ordinary citation matching follows the frozen source checker (including
        # its existing numeric/date/string equivalences), not V5 probe equality.
        require(isinstance(result,dict) and equal(result.get('citation'),citation) and result.get('status')=='valid'
                and result.get('warnings')==[], 'Source check does not bind valid warning-free citation')


def validate_repair_schema(first,proposal,probe,context):
    require(set(first)=={'decision','anchor','claim','inspect_code'} and first.get('decision')=='INSPECT'
            and text(first.get('claim'),1500) and text(first.get('inspect_code'),6000),'Inspection schema differs')
    require(set(proposal)=={'decision','anchor','observed_discrepancy','expected_obligation','preserved_obligations','witnesses','patch'}
            and proposal.get('decision')=='PATCH' and text(proposal.get('observed_discrepancy'),1500)
            and text(proposal.get('expected_obligation'),1500),'Patch explanation/schema differs')
    preserved=proposal['preserved_obligations']
    require(isinstance(preserved,list) and 1<=len(preserved)<=8 and all(text(i,500) for i in preserved), 'Preserved obligations invalid')
    anchor=first['anchor']
    require(isinstance(anchor,dict) and set(anchor)=={'channel','quote'} and anchor['channel'] in {'question','assumptions'}
            and text(anchor['quote'],2000),'Anchor schema differs')
    require(set(probe)=={'facts','observations'},'Probe answer schema differs')
    for fact in probe['facts']:
        require(text(fact.get('field_path'),256) and len(json.dumps(fact['value'],allow_nan=False))<=1024,'Probe fact bounds differ')
    for observation in probe['observations']:
        require(isinstance(observation,dict) and set(observation)=={'name','value'} and text(observation['name'],80)
                and len(json.dumps(observation['value'],allow_nan=False))<=1024,'Probe observation schema differs')
    for witness in proposal['witnesses']:
        require(isinstance(witness,dict) and set(witness)=={'fact','observation_name','observed_value','relevance'}
                and text(witness['observation_name'],80) and text(witness['relevance'],500),'Patch witness schema differs')


def validate_repair(critic,packet,workspace,calls,replies,primary,final):
    proposal=critic.get('proposal') or {};first=critic.get('first') or {};probe=critic.get('validated_probe') or {}
    require(first.get('decision')=='INSPECT' and proposal.get('decision')=='PATCH','Repair lacks inspect/patch decisions')
    validate_repair_schema(first,proposal,probe,packet['context'])
    anchor=proposal.get('anchor');require(equal(anchor,first.get('anchor')) and isinstance(anchor,dict),'Repair switched anchors')
    source=packet['context'].get(anchor.get('channel'));quote=anchor.get('quote')
    require(isinstance(source,str) and isinstance(quote,str) and quote
            and sum(source.startswith(quote,i) for i in range(len(source)))==1,'Repair anchor is not unique original text')
    require(equal(calls['v5-critic-inspect'][0]['code'],first.get('inspect_code'))
            and equal(replies['v5-critic-inspect'],critic.get('probe_output'))
            and equal(replies['v5-critic-inspect'].get('answer'),probe),'Repair probe differs from actual tool execution')
    require(isinstance(probe.get('facts'),list) and 1<=len(probe['facts'])<=16
            and isinstance(probe.get('observations'),list) and 1<=len(probe['observations'])<=16,'Repair probe schema differs')
    for fact in probe['facts']:field_value(workspace,fact)
    observations={o['name']:o['value'] for o in probe['observations']}
    require(len(observations)==len(probe['observations']),'Duplicate observed fact names')
    witnesses=proposal.get('witnesses');require(isinstance(witnesses,list) and 1<=len(witnesses)<=8,'Missing repair witnesses')
    for witness in witnesses:
        field_value(workspace,witness['fact'])
        require(any(equal(witness['fact'],fact) for fact in probe['facts']) and witness['observation_name'] in observations
                and equal(witness['observed_value'],observations[witness['observation_name']]),'Repair witness was not observed')
    patch=proposal.get('patch') or {};old=patch.get('old_text');new=patch.get('new_text');code=packet['final_code']
    require(set(patch)=={'old_text','new_text'} and text(old,8000) and text(new,8000,empty=True) and old!=new,'Malformed local code patch')
    starts=[i for i in range(len(code)) if code.startswith(old,i)];require(len(starts)==1,'Patch region is not unique')
    result=code[:starts[0]]+new+code[starts[0]+len(old):]
    require(len(result)<=32000,'Replacement program exceeds code bounds');ast.parse(result)
    require(equal(calls['v5-critic-final'][0]['code'],result) and sha_text(result)==critic.get('candidate_code_sha256'),'Executed patch code differs')
    execution=replies['v5-critic-final'];require(equal(execution,critic.get('alternate_execution')) and full_artifact(execution)
        and not execution.get('error') and equal(canonical_artifact(execution),final)
        and equal(final,critic.get('alternate')),'Released candidate differs from actual final execution')
    require(not equal({k:primary.get(k) for k in ('answer','evidence','contributing_ids')},
                      {k:final.get(k) for k in ('answer','evidence','contributing_ids')}),'Patch had no substantive effect')
    check=replies['v5-critic-source-check'];require(equal(check,critic.get('source_check')) and check.get('issues')==[]
        and critic.get('candidate_issues')==[] and not final.get('limitation')
        and equal(calls['v5-critic-source-check'][0]['citations'],final.get('evidence')),'Repair source check differs')
    validate_citations(final['evidence'],check,workspace)


def actual_source_inventory(root):
    roots=('run_agent.py','fhir_client.py','requirements.txt','config.py','config.example.yml','README.md','AGENTS.md')
    paths={root/name for name in roots if (root/name).is_file()}
    for directory in ('agent','tools','utils','scripts','tests','docs','manifests'):
        paths.update(p for p in (root/directory).rglob('*') if p.is_file()
                     and p.suffix in {'.py','.md','.txt','.json','.csv','.yml','.yaml'} and '__pycache__' not in p.parts)
    return {p.relative_to(root).as_posix() for p in paths}


def verify_restored(row,output,store):
    payload,nbytes=B.independent_restore(row,output)
    restored=store.restore_result_audit(row,output,question_id=row['question_id'],restore_history=False)
    for field in row['audit_artifacts']['fields']:
        actual=restored['revision']['messages'] if field=='revision.messages' else restored[field]
        require(equal(actual,payload[field]),'Restored field differs from exact gzip payload')
    for field,value in row.items():
        if field in {'trace','retrieved_fhir_resources','audit_artifacts'}:continue
        if field=='revision' and 'revision.messages' in row['audit_artifacts']['fields']:
            require(equal({k:v for k,v in restored[field].items() if k!='messages'},
                          {k:v for k,v in value.items() if k!='messages'}),'Inline revision changed on restoration')
        else:require(equal(restored.get(field),value),'Inline result changed on restoration')
    return restored,nbytes


def attempt_counts(row,restored,is_team):
    counts=Counter(saved_first_attempts=1,terminal_errors=int(B.failed(row)),verified_owned_gzip_audits=1)
    budget=row.get('budget');tokens=row.get('tokens');revision=restored.get('revision') or {}
    if budget is None and B.failed(row):counts['early_terminal_budget_metadata_gaps']+=1
    else:
        require(isinstance(budget,dict) and equal(budget.get('max_total_tokens'),200000)
                and equal(budget.get('max_tool_calls'),30),'Declared global per-question caps differ')
    budget=budget or {};tokens=tokens or {}
    counts['budget_stops']+=bool(budget.get('stopped'))
    unknown=any(c.get('usage_available') is not True for c in tokens.get('per_call') or [])
    counts['recorded_unknown_provider_usage_attempts']+=unknown
    if not tokens and B.failed(row):counts['early_terminal_usage_metadata_gaps']+=1
    counts['reported_aggregate_token_overshoots']+=(tokens.get('total') or 0)>200000
    counts['reported_tool_overshoots']+=(budget.get('tool_calls_run') or 0)>30
    phases=revision.get('phases') or [];fatal=any(bool(p.get('fatal')) for p in phases)
    counts['fatal_phase_records']+=sum(bool(p.get('fatal')) for p in phases)
    if is_team:
        counts.update(v5_protocol_counts(row,restored))
        if phases:
            require(equal(sum(p['tokens'] for p in phases),tokens.get('total'))
                    and equal(sum(p['tool_calls'] for p in phases),budget.get('tool_calls_run')),'Phase accounting sums differ')
            counts['reported_phase_token_overshoots']+=sum(p['tokens']>p['token_cap'] for p in phases)
            counts['reported_phase_tool_overshoots']+=sum(p['tool_calls']>p['tool_cap'] for p in phases)
        else:
            require(B.failed(row),'Nonterminal team result lacks phases')
            counts['terminal_errors_without_phase_records']+=1
        if counts['source_valid_releases']:
            require(not counts['reported_aggregate_token_overshoots'] and not counts['reported_tool_overshoots']
                    and not counts['reported_phase_token_overshoots'] and not counts['reported_phase_tool_overshoots'],
                    'Over-budget result claims a clean release')
    counts['recorded_clean_rendered_source_checks']+=not B.failed(row) and revision.get('final_check_issues')==[]
    counts['clean_check_with_fatal_phase']+=revision.get('final_check_issues')==[] and fatal
    return counts


def verify_arm_snapshot(root,output,manifest,expected,code,store,arm,strategy,completion):
    import pandas as pd
    path=B.owned_file(output,arm+'.json');raw=path.read_bytes();rows=json.loads(raw)
    require(isinstance(rows,list) and len(rows)==300 and [r['question_id'] for r in rows]==expected,'Scheduled fixed300/order differs')
    require(all(not r.get('attempt_history') for r in rows),'Fresh run contains retry history')
    if completion:require(all(B.complete(r) for r in rows),'All 600 first attempts must finish before completion audit/archive')
    side=B.owned_file(output,arm+'.json.run.json');identity=load(side)
    settings={'agent_strategy':strategy,'model':'gpt-6-luna','max_total_tokens':200000,'max_tool_calls':30,
      'split':'held_out','split_manifest_hash':manifest['split_sha256'],'enable_cache':True,'add_patient_fhir_id':True,
      'audit_storage':'gzip','audit_artifact_schema':2,'code_hash':code,'condition':None,
      'perturbations_hash':None,'subset_hash':None,'repeat':None}
    require(all(k in identity and equal(identity[k],v) for k,v in settings.items()),'Actual run identity differs')
    columns=['question_id','question','true_answer','assumption','patient_fhir_id']
    encoded=pd.DataFrame(rows)[columns].to_json(orient='records',date_format='iso')
    require(hashlib.sha256(encoded.encode()).hexdigest()==identity.get('input_hash')==FIXED_INPUT_HASH,'Actual ordered fixed-subset inputs differ')
    counts=Counter();policies=Counter();references=set();artifact_bytes=0
    for row in rows:
        if not B.complete(row):continue
        restored,nbytes=verify_restored(row,output,store);artifact_bytes+=nbytes
        counts.update(attempt_counts(row,restored,strategy=='multi_agent_v5'))
        references.add(row['audit_artifacts']['path'])
        policies[str((restored.get('revision') or {}).get('release_policy'))]+=1
    if not completion:require(counts['saved_first_attempts']>0,'Initial audit requires saved attempts in both arms')
    if completion:require(path.read_bytes()==raw,'Completed result changed during verification')
    return {'scheduled_rows':300,**dict(counts),'release_policies':dict(policies),
            'restored_uncompressed_bytes':artifact_bytes,'result_sha256':hashlib.sha256(raw).hexdigest(),
            'identity_sha256':sha(side),'bytes':len(raw)},identity,encoded,references


def run_check(root,record,pins_path,pins_sha,mode,report,owned_finished=False,archive=None):
    require(mode in {'prelaunch','initial','completion'},'Unknown audit mode')
    if mode=='completion':require(owned_finished is True,'Root must confirm both owned jobs are terminal')
    else:require(archive is None,'Only completion can create a raw archive')
    root,record=Path(root).resolve(),Path(record).resolve()
    require(record.is_relative_to(root/'benchmarks'),'Record must be in this repository benchmarks directory')
    manifest=load(record/'manifest.json');output=Path(manifest['output_directory']).resolve()
    require(output.is_relative_to(root/'output'),'Run output is outside owned repository output')
    B.guard_destinations(root,record,output,report,archive)
    pins,manifest,adapter,binding=frozen_provenance(root,record,output,pins_path,pins_sha,mode=='prelaunch')
    require(actual_source_inventory(root)==set(manifest['source_sha256']),'Actual public source inventory differs from prespecified file list')
    expected=load(root/'manifests/split.json')['splits']['held_out']['question_ids']
    require(len(expected)==len(set(expected))==300,'Fixed question cohort differs')
    code=B.code_hash(root)
    result={'status':'PRELAUNCH_PASS' if mode=='prelaunch' else 'INITIAL_SNAPSHOT_PASS' if mode=='initial' else 'PASS',
      'mode':mode,'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'protocol':PROTOCOL,
      'manifest_sha256':pins['record_metadata_sha256']['manifest.json'],'source_inventory_sha256':pins['source_inventory_sha256'],
      'frozen_source_files_verified':len(manifest['source_sha256']),'runtime_adapter_files_verified':4,
      'pins_sha256':pins_sha,'guard_sha256':sha(Path(__file__)),'reviewed_base_sha256':BASE_SHA,
      'source_archive_sha256':manifest['source_archive_sha256'],'design_sha256':manifest['design_sha256'],
      'dataset_sha256':manifest['dataset_sha256'],'split_sha256':manifest['split_sha256'],
      'code_hash':code,'provider_binding':binding,'model_calls':0,'network_calls':0,'scoring_performed':False,
      'caller_asserted_owned_jobs_terminal':owned_finished if mode=='completion' else None,
      'limits':['Critic invocations, retained model replies and SDK journal events are distinct evidence; no equality is inferred on failed/fatal paths.',
                'Text anchors, exact probe facts, executed code and recorded clean citations do not establish semantic/clinical correctness.',
                'Ordinary source citations bind exact retained check results; the guard does not refetch records or substitute stricter equality for the frozen checker semantics.',
                'Unavailable usage remains unknown; failures and phase overshoots are retained outcomes, not deleted questions.',
                'Initial files/journals are individually captured snapshots, not a simultaneous state; unmatched latest starts can be in flight.',
                'Completion terminality is an external root process-supervision assertion; this helper validates complete saved cohorts and provenance.',
                'No critic code is rerun and no source records are refetched. Mechanical artifact/witness checks use retained immutable data only.',
                'A coverage skip or malformed auxiliary judgment may preserve the exact valid primary; it does not establish that a semantic review happened.',
                'The fixed300 remains adaptive development validation, not confirmation of generalization.']}
    references=set();journals=[]
    if mode!='prelaunch':
        store=B.load_public_store(root);identities={};inputs={};arms={}
        for arm,strategy in ARM_STRATEGIES.items():
            summary,identity,encoded,refs=verify_arm_snapshot(root,output,manifest,expected,code,store,arm,strategy,mode=='completion')
            arms[arm]=summary;identities[arm]=identity;inputs[arm]=encoded;references|=refs
        require(inputs['ordinary_single_agent']==inputs['ordinary_multi_agent'],'Arm ordered inputs differ')
        require(equal({k:v for k,v in identities['ordinary_single_agent'].items() if k!='agent_strategy'},
                      {k:v for k,v in identities['ordinary_multi_agent'].items() if k!='agent_strategy'}),'Arm identities differ beyond strategy')
        journals=[B.parse_journal(p,output) for p in sorted((output/'provider-events').glob('*.jsonl'))]
        require(journals,'Missing safe provider journals');totals=Counter()
        for journal in journals:totals.update(journal['counts'])
        result.update(arms=arms,ordered_input_hash=FIXED_INPUT_HASH,
          restored_audit_records=sum(a['verified_owned_gzip_audits'] for a in arms.values()),
          unique_referenced_artifacts=len(references),journal={'totals':dict(totals),'files':journals})
        if mode=='completion':
            allowed=B.allowed_paths(output,references,[j['path'] for j in journals])
            files={name:{'sha256':sha(output/name),'bytes':(output/name).stat().st_size} for name in allowed}
            result.update(expected_first_attempts=600,raw_files=files,actual_base_members=len(B.BASE_FILES|references),
                          provider_journal_members=len(journals),archive_requested=bool(archive),archive_verified=False)
            if not all(j['pairing_complete'] for j in journals):result['status']='PASS_WITH_JOURNAL_LIMITATIONS'
    frozen_provenance(root,record,output,pins_path,pins_sha,mode=='prelaunch')
    require(actual_source_inventory(root)==set(manifest['source_sha256']),'Source inventory changed during audit')
    report.parent.mkdir(parents=True,exist_ok=True)
    with report.open('x') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n');stream.flush()
        if archive:
            require(all(j['safe_to_archive'] for j in journals),'Partial unvalidated journal bytes block archive; report preserved')
            require(all(sha(output/n)==v['sha256'] for n,v in files.items()),'Completed raw members changed')
            result.update(B.write_archive(archive,output,allowed,files))
            frozen_provenance(root,record,output,pins_path,pins_sha)
            require(actual_source_inventory(root)==set(manifest['source_sha256']),'Source inventory changed during archive')
            result['archive_verified']=True
            stream.seek(0);stream.truncate();stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['pins','prelaunch','initial','completion'],required=True)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--pins',type=Path,required=True);parser.add_argument('--pins_sha256')
    parser.add_argument('--report',type=Path);parser.add_argument('--owned_jobs_finished',action='store_true')
    parser.add_argument('--archive',type=Path)
    args=parser.parse_args(argv)
    if args.mode=='pins':
        root=args.root.resolve();record=args.record.resolve();manifest=load(record/'manifest.json');output=Path(manifest['output_directory']).resolve()
        require(record.is_relative_to(root/'benchmarks') and not output.exists(),'Prepare pins only for an unlaunched formal record')
        B.guard_destinations(root,record,output,args.pins)
        pin_record(record,args.pins)
        print(json.dumps({'status':'PINS_PREPARED_NOT_APPROVAL','pins':str(args.pins),'sha256':sha(args.pins)}));return
    require(args.report is not None,'Fresh report destination required')
    result=run_check(args.root,args.record,args.pins,args.pins_sha256,args.mode,args.report,args.owned_jobs_finished,args.archive)
    print(json.dumps({'status':result['status'],'mode':args.mode,'report':str(args.report),
                      'arms':result.get('arms'),'journal_totals':result.get('journal',{}).get('totals')},allow_nan=False))


if __name__=='__main__':main()
