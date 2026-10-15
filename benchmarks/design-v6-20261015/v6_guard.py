"""Model-free V6 subset prelaunch, initial and completion integrity checks.

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
ARM_STRATEGIES = {'ordinary_single_agent':'single_agent', 'ordinary_multi_agent':'multi_agent_v6'}
PROTOCOL = 'continuous-primary-independent-recompute-v6'
PHASE_CAPS = {'primary':(175000,27), 'independent_recompute':(25000,3)}
ASSURANCE = 'fallible_model_selection_of_executed_source_checked_candidates'
FIXED_INPUT_HASH = '25e57321644744d5c0718c949c1f90e8be4f53b8aadce4d0118843a99c7b96ce'
FIXED_DATASET_SHA = 'e2045692fef7f5f4f77496935160f5fc727e162d213e94feed61401948e512a0'
FIXED_SPLIT_SHA = 'f28db462ebcfe4e0d87675c86de9a08adef4038e26dbddcf88c50ddf92e94015'
HEX = re.compile(r'^[a-f0-9]{64}$')
RECORD_META = ('manifest.json','runtime_adapter_manifest.json','expected_public_binding.json','prospective_protocol.json','launch_commands.json')
REQUIRED_SOURCE = {'agent/continuous_primary.py','agent/independent_recompute_agent.py','agent/independent_recompute_aux.py','agent/single_agent.py','agent/base_agent.py',
                   'scripts/audit_store.py','run_agent.py','requirements.txt'}
PROVIDER_SOURCE = {
 '.local-setup/azure_adapter.py':'5423a46145d070090a58514aad285782c33db3751bcab085771d365946146cc8',
 '.local-setup/sitecustomize.py':'fb3eea8ce917715e6b553f501cc4abf2fa5408570f198e7aaa6392e72f22e9fe',
 '.local-setup/azure.py':'1144197dae1282b7472ddd90dc703ac02d17f87efa9b6327d42adce1921b13c4',
 '.local-setup/phase3.py':'85182719e3c5020d527c97a454d80f4616caa7abc247a9b9b352e25077ce61d3'}
PUBLIC_BINDING = {'provider_profile_sha256':'493987643d4523481d1c2ed077747d43390f806f1665ab6a9d605f8a2251df1c',
 'adapter_sha256':PROVIDER_SOURCE['.local-setup/azure_adapter.py'],
 'bootstrap_sha256':PROVIDER_SOURCE['.local-setup/sitecustomize.py']}


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
    require(REQUIRED_SOURCE <= set(manifest['source_sha256']),'Required V6 source inventory is missing')
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
    require(isinstance(mapping,dict) and REQUIRED_SOURCE <= set(mapping),'V6 source inventory incomplete')
    require(type(pins.get('source_file_count')) is int and len(mapping)==pins['source_file_count']
            and fingerprint(mapping)==pins.get('source_inventory_sha256'),'Prespecified source inventory differs')
    return pins,manifest

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
        if (name=='execute_python_code' and args.get('mode')=='final' and full_artifact(result)
                and result.get('mode')=='final' and 'error' not in result and 'error_type' not in result):
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
        # its existing numeric/date/string equivalences), not V6 probe equality.
        require(isinstance(result,dict) and equal(result.get('citation'),citation) and result.get('status')=='valid'
                and result.get('warnings')==[], 'Source check does not bind valid warning-free citation')

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


def verify_prospective(prospective,binding):
    expected={'model':'gpt-6-luna','fixed_questions_per_arm':300,'workers_per_arm':1,'first_attempt_only':True,
      'phase_caps':{k:list(v) for k,v in PHASE_CAPS.items()},'primary_iterations':30,
      'primary_tools_inclusive_automatic_checks':True,'reserved_primary_source_checks':0,
      'max_auxiliary_invocations':2,'max_team_model_calls':32,'max_control_model_calls':30,
      'global_caps':[200000,30],'estimated_input_guard':32000,'minimum_team_correct':240,
      'expected_public_binding':binding}
    require(all(equal(prospective.get(k),v) for k,v in expected.items()),'Prospective V6 profile/caps differ')


def validate_launch_commands(record,output):
    commands=load(record/'launch_commands.json')
    require(commands.get('concurrent') is True and equal(commands.get('workers_per_arm'),1),'Declared launch concurrency differs')
    require(set(commands.get('commands',{}))=={'single','team'},'Declared launch arm inventory differs')
    for label,arm in [('single','ordinary_single_agent'),('team','ordinary_multi_agent')]:
        entry=commands['commands'][label]
        expected=['caffeinate','-i','.venv/bin/python','.local-setup/phase3.py','run','--project-root','.',
          '--model','gpt-6-luna','--max_total_tokens','200000','--max_tool_calls','30',
          '--num_processes','1','--save_interval','1','--output_dir',str(output),
          '--split','held_out','--team_strategy','multi_agent_v6','--audit_storage','gzip','--arms',arm]
        require(entry.get('arm')==arm and equal(entry.get('argv'),expected),'Frozen exact launch argv differs: '+label)
        require(equal(entry.get('environment'),{'FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR':str(output/'provider-events')}),'Launch environment is not the safe owned journal directory')
        require(Path(entry.get('log','')).resolve()==output/('run_'+label+'.log'),'Declared arm log differs')


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
    literals={}
    for name in ('agent/continuous_primary.py','agent/independent_recompute_agent.py','agent/independent_recompute_aux.py'):
        values={}
        for node in ast.parse((root/name).read_text()).body:
            if isinstance(node,ast.Assign):
                try:value=ast.literal_eval(node.value)
                except (ValueError,TypeError):continue
                for target in node.targets:
                    if isinstance(target,ast.Name):values[target.id]=value
                    elif isinstance(target,(ast.Tuple,ast.List)) and isinstance(value,(list,tuple)):
                        values.update({key.id:item for key,item in zip(target.elts,value) if isinstance(key,ast.Name)})
        literals[name]=values
    require(all(equal(literals['agent/continuous_primary.py'].get(k),v) for k,v in
        {'PRIMARY_TOKENS':175000,'PRIMARY_TOOLS':27,'PRIMARY_ITERATIONS':30}.items()),'Primary literal allocation differs')
    require(literals['agent/independent_recompute_agent.py'].get('PROTOCOL')==PROTOCOL,'Team protocol literal differs')
    require(all(equal(literals['agent/independent_recompute_aux.py'].get(k),v) for k,v in
        {'PROTOCOL':'independent-frozen-recomputation-v6','TOKEN_CAP':25000,'TOOL_CAP':3,'CALL_CAP':2,'ASSURANCE':ASSURANCE}.items()),'Auxiliary literal allocation differs')
    adapter=load(record/'runtime_adapter_manifest.json')
    require(equal(adapter.get('source_sha256'),PROVIDER_SOURCE),'Reviewed four public provider sources differ')
    for name,value in PROVIDER_SOURCE.items():require(sha(public_path(root,name))==value,'Public provider source differs')
    B.source_zip(record/'runtime_adapter_sources.zip',PROVIDER_SOURCE,adapter['archive_sha256'])
    binding=load(record/'expected_public_binding.json')
    require(equal(binding,PUBLIC_BINDING),'Prespecified common public provider binding differs')
    verify_prospective(load(record/'prospective_protocol.json'),binding)
    validate_launch_commands(record,output)
    if prelaunch:require(not output.exists(),'Prelaunch output must be absent; no silent resume')
    else:require(equal(load(B.owned_file(output,'.local-provider.json')),binding),'Actual public binding differs')
    return pins,m,adapter,binding


def _aux_fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,allow_nan=False).encode()).hexdigest()


def _source_summary(candidate,check,workspace):
    json.dumps(check,allow_nan=False)
    require(isinstance(check,dict) and set(check)<={'issues','results','n_valid','question_scopes','scope_membership_only'}
      and check.get('issues')==[] and type(check.get('n_valid')) is int
      and check['n_valid']==len(candidate['evidence']),'Candidate source-check envelope differs')
    validate_citations(candidate['evidence'],check,workspace)
    return {'issues':[],'n_valid':len(candidate['evidence']),'citation_count':len(candidate['evidence']),
      'all_results_valid':True,'all_warnings_empty':True,'raw_check_sha256':_aux_fingerprint(check),
      'assurance':'source_fields_only_not_semantics_or_search_completeness'}


def v6_protocol_counts(row,restored):
    counts=Counter();revision=restored.get('revision') or {}
    if revision.get('protocol')!=PROTOCOL:
        require(B.failed(row),'Nonterminal V6 row lacks protocol metadata')
        counts['terminal_errors_without_protocol_metadata']+=1;return counts
    phases=revision.get('phases') or [];names=[p['phase'] for p in phases]
    require(names in (['primary'],['primary','independent_recompute']),'V6 phase order differs')
    for phase in phases:
        require(equal([phase['token_cap'],phase['tool_cap']],list(PHASE_CAPS[phase['phase']])),'V6 declared phase caps differ')
        require(type(phase.get('fatal')) is bool,'Malformed phase fatal flag')
        if phase['phase']=='independent_recompute':
            require(equal(phase.get('model_call_cap'),2),'Declared auxiliary model-call cap differs')
    primary_diag=revision.get('primary_diagnostics') or {};primary=revision.get('primary');artifact=revision.get('primary_artifact')
    require(equal(primary_diag,phases[0]),'Retained primary phase diagnostics differ')
    require(equal(primary_diag.get('max_iterations'),30),'Primary natural iteration declaration differs')
    require(type(primary_diag.get('model_call_attempts_recorded')) is int and primary_diag['model_call_attempts_recorded']>=0,'Malformed primary recorded invocation counter')
    counts['reported_primary_call_overshoots']+=primary_diag['model_call_attempts_recorded']>30
    if counts['reported_primary_call_overshoots']:require(B.failed(row),'Nonterminal primary exceeds30 reported model calls')
    policy=revision.get('release_policy');final=revision.get('final');aux=revision.get('recompute')
    require(policy in {'continuous_primary_preserved_auxiliary_ineligible','continuous_primary_preserved',
      'selected_executed_independent_candidate','fatal_execution_or_accounting_failure'},'Unexpected V6 release policy')
    require(isinstance(final,dict) and equal(rendered(final),{'answer':row.get('agent_answer'),'evidence':row.get('evidence'),'limitation':row.get('limitation')}),'Rendered final differs from output')
    fatal=any(p['fatal'] for p in phases) or bool(isinstance(aux,dict) and aux.get('fatal'))
    if fatal:require(B.failed(row),'Declared fatal accounting lacks an explicit terminal error')
    if policy=='fatal_execution_or_accounting_failure':
        require(B.failed(row) and row.get('agent_answer') is None,'Fatal V6 attempt lacks retained terminal error')
        counts['fatal_release_outcomes']+=1
    elif policy=='continuous_primary_preserved_auxiliary_ineligible':
        require(primary_diag.get('eligible') is False and artifact is None and aux is None
          and names==['primary'] and equal(rendered(final),rendered(primary)),'Ineligible primary was changed or sent to auxiliary')
        counts['ineligible_primary_preserved']+=1
    else:
        require(not B.failed(row) and not fatal and full_artifact(artifact) and full_artifact(final)
          and primary_diag.get('eligible') is True and names==['primary','independent_recompute']
          and revision.get('final_check_issues')==[],'Clean V6 release lacks bound eligible primary')
        five={k:artifact[k] for k in ('answer','evidence','operation','contributing_ids','limitation')}
        require(equal(rendered(five),rendered(primary)),'Executed A differs from actual released primary')
        trace=restored.get('trace') or [];start=next((i for i,m in enumerate(trace) if isinstance(m,dict)
            and m.get('phase') in {'independent_generation','independent_selection','independent_recompute'}),len(trace))
        code=validate_primary(trace[:start],five,restored['retrieved_fhir_resources'])
        primary_checks=[json.loads(message['content']) for message in trace[:start] if isinstance(message,dict)
          and message.get('role')=='tool' and message.get('tool_call_id') in {'automatic_field_check_1','automatic_field_check_2'}]
        a_summary=_source_summary(five,primary_checks[-1],restored['retrieved_fhir_resources'])
        require(code==artifact.get('code') and artifact.get('workspace_sha256')==fingerprint(restored['retrieved_fhir_resources']),'Actual A program/workspace identity differs')
        require(isinstance(aux,dict) and aux.get('protocol')=='independent-frozen-recomputation-v6'
          and aux.get('assurance')==ASSURANCE and equal(aux.get('phase'),phases[1]),'Auxiliary protocol/phase differs')
        invocations=aux.get('model_calls');tools=aux.get('tool_calls')
        require(type(invocations) is int and 0<=invocations<=2 and type(tools) is int and 0<=tools<=3,'Nonfatal auxiliary caps differ')
        require(equal(invocations,phases[1].get('model_calls')) and equal(tools,phases[1].get('tool_calls'))
          and equal((row.get('tokens') or {}).get('llm_calls'),primary_diag['model_call_attempts_recorded']+invocations),
          'Nonfatal primary/auxiliary invocation accounting differs')
        counts['auxiliary_invocations_reported']+=invocations;counts['auxiliary_tools_reported']+=tools
        status=aux.get('status');require(status in {'coverage_skip','primary_preserved','agreement_preserved','switched'},'Unexpected nonfatal auxiliary status')
        counts['auxiliary_status_'+status]+=1
        responses=[m for m in trace if isinstance(m,dict) and m.get('role')=='assistant'
          and m.get('phase') in {'independent_generation','independent_selection'}]
        counts['auxiliary_responses_observed']+=len(responses)
        counts['auxiliary_invocation_response_gaps']+=invocations!=len(responses)
        generation=[];selection=[];calls={};replies={}
        for message in trace[start:]:
            if not isinstance(message,dict):continue
            phase=message.get('phase')
            if message.get('role')=='user' and phase in {'independent_generation','independent_selection'}:
                packet=json.loads(message['content']);(generation if phase=='independent_generation' else selection).append(packet)
            if message.get('role')=='assistant' and phase in {'independent_generation','independent_recompute'}:
                for call in message.get('tool_calls') or []:
                    identifier=call.get('id');function=call.get('function') or {}
                    if not isinstance(identifier,str) or identifier in calls:
                        require(not aux.get('B_eligible'),'Certified B has malformed/repeated native tool identity')
                        counts['unexecuted_malformed_auxiliary_requests']+=1;continue
                    try:arguments=json.loads(function.get('arguments') or '')
                    except (TypeError,ValueError):
                        require(not aux.get('B_eligible'),'Certified B has malformed native arguments')
                        arguments=None;counts['unexecuted_malformed_auxiliary_requests']+=1
                    calls[identifier]=(function.get('name'),arguments)
            if message.get('role')=='tool' and phase=='independent_recompute':
                key=message['tool_call_id'];require(key in calls and key not in replies and message.get('name')==calls[key][0],'Unpaired/misnamed auxiliary tool result')
                replies[key]=json.loads(message['content'])
        if len(replies)!=tools:
            counts['auxiliary_charged_tool_reply_gaps']+=1
            counts['auxiliary_charged_minus_observed_tool_replies']+=tools-len(replies)
        if aux.get('B_eligible'):require(len(replies)==tools,'Certified B tool counter differs from actual returned tool evidence')
        if aux.get('packet_admitted'):
            require(len(generation)==1,'Admitted generation lacks one intact packet')
            packet=generation[0]
            require(set(packet)=={'context','workspace_sha256','held_inventory','retrieval_allowed'}
              and equal(packet['context'],context_from_row(row)) and packet['retrieval_allowed'] is False
              and packet['workspace_sha256']==fingerprint(restored['retrieved_fhir_resources'])
              and _aux_fingerprint(packet)==aux.get('generation_packet_sha256'),'Candidate-blind generation packet differs')
            inventory={kind:{'count':len(items),'source_example':items[:1],
              'concept_labels':sorted({str(c.get('display') or c.get('code') or '') for r in items for c in (r.get('code') or {}).get('coding',[])})}
              for kind,items in restored['retrieved_fhir_resources'].items()}
            require(equal(packet['held_inventory'],inventory),'Generation held inventory differs')
            counts['bound_candidate_blind_packets']+=1
        else:require(not generation and not responses,'Unadmitted generation has model evidence')
        if policy=='continuous_primary_preserved':
            require(equal(final,five) and revision.get('independent_candidate_selected') is False
              and aux.get('switched') is False,'Fallback changed A')
        else:
            require(status=='switched' and invocations==2 and tools==2 and len(responses)==2
              and aux.get('switched') is True and aux.get('selected')=='B'
              and revision.get('independent_candidate_selected') is True,'Selected B lacks the declared complete path')
        if aux.get('B_eligible'):
            b=aux.get('B');require(full_artifact(b) and not b.get('limitation'),'Eligible B artifact is missing')
            execution_ids=[key for key,(name,args) in calls.items() if name=='execute_python_code']
            require(len(execution_ids)==1,'B lacks one actual native execution')
            identifier=execution_ids[0];name,args=calls[identifier];execution=replies.get(identifier)
            require(args.get('mode')=='final' and isinstance(args.get('code'),str)
              and equal(execution,aux.get('B_execution')) and full_artifact(execution)
              and execution.get('mode')=='final' and 'error' not in execution and 'error_type' not in execution
              and equal(canonical_artifact(execution),b),'Retained B differs from actual native final execution')
            check=replies.get('v6-independent-source-check')
            require(equal(check,aux.get('B_source_check')) and equal(calls.get('v6-independent-source-check'),('check_fields',{'citations':b['evidence']})),'B original source-check call/result differs')
            _source_summary(b,check,restored['retrieved_fhir_resources'])
            counts['executed_source_bound_B_candidates']+=1
            require(aux.get('answer_agreement') is equal(five['answer'],b['answer']),'Typed agreement metadata differs')
            if status=='agreement_preserved':require(equal(five['answer'],b['answer']) and invocations==1 and not selection,'Agreement must retain A without selector')
            if aux.get('selector_admitted'):
                require(len(selection)==1,'Admitted selector lacks intact packet')
                packet=selection[0];require(set(packet)=={'context','workspace_sha256','A','B'}
                  and equal(packet['context'],context_from_row(row)) and packet['workspace_sha256']==artifact['workspace_sha256']
                  and _aux_fingerprint(packet)==aux.get('selector_packet_sha256'),'Selector packet identity differs')
                for key,candidate,program in [('A',five,code),('B',b,args['code'])]:
                    entry=packet[key]
                    require(equal(entry.get('artifact'),candidate) and entry.get('program')==program
                      and entry.get('program_sha256')==sha_text(program) and entry.get('workspace_sha256')==artifact['workspace_sha256'],'Selector candidate entry differs')
                require(equal(packet['A']['source_check'],a_summary),'Selector A source summary differs')
                require(equal(packet['B']['source_check'],_source_summary(b,check,restored['retrieved_fhir_resources'])),'Selector B source summary differs')
                choices=[m for m in responses if m.get('phase')=='independent_selection']
                if aux.get('selection') is not None:
                    require(len(choices)==1 and not choices[0].get('tool_calls') and equal(json.loads(choices[0]['content']),aux['selection']),'Retained selection differs from actual response')
            if status=='switched':require(equal(final,b) and aux.get('selection',{}).get('choice')=='B' and len(selection)==1,'Selected output is not exact existing B')
        if policy=='selected_executed_independent_candidate':counts['independent_candidates_selected']+=1
        counts['source_valid_executed_releases']+=1
        counts['source_valid_executed_null_answers']+=row.get('agent_answer') is None
    counts['null_answers_outside_executed_release']+=row.get('agent_answer') is None and policy not in {'continuous_primary_preserved','selected_executed_independent_candidate'}
    counts['primary_source_revised']+=bool(revision.get('primary_source_revised'))
    return counts


def attempt_counts(row,restored,is_team):
    counts=Counter(saved_first_attempts=1,terminal_errors=int(B.failed(row)),verified_owned_gzip_audits=1)
    budget=row.get('budget');tokens=row.get('tokens');revision=restored.get('revision') or {}
    if budget is None and B.failed(row):counts['early_terminal_budget_metadata_gaps']+=1
    else:require(isinstance(budget,dict) and equal(budget.get('max_total_tokens'),200000) and equal(budget.get('max_tool_calls'),30),'Declared global caps differ')
    budget=budget or {};tokens=tokens or {};unknown=any(c.get('usage_available') is not True for c in tokens.get('per_call') or [])
    counts['budget_stops']+=bool(budget.get('stopped'));counts['recorded_unknown_provider_usage_attempts']+=unknown
    counts['reported_aggregate_token_overshoots']+=(tokens.get('total') or 0)>200000
    counts['reported_tool_overshoots']+=(budget.get('tool_calls_run') or 0)>30
    counts['reported_model_call_overshoots']+=(tokens.get('llm_calls') or 0)>(32 if is_team else 30)
    if not tokens and B.failed(row):counts['early_terminal_usage_metadata_gaps']+=1
    if unknown or counts['reported_aggregate_token_overshoots'] or counts['reported_tool_overshoots'] or counts['reported_model_call_overshoots']:
        require(B.failed(row),'Unknown/overshot accounting was released as a nonterminal null/outcome')
    phases=revision.get('phases') or []
    counts['fatal_phase_records']+=sum(bool(p.get('fatal')) for p in phases)
    if is_team:
        counts.update(v6_protocol_counts(row,restored))
        if phases:
            sums=equal(sum(p['tokens'] for p in phases),tokens.get('total')) and equal(sum(p['tool_calls'] for p in phases),budget.get('tool_calls_run'))
            if not sums:
                require(B.failed(row),'Nonterminal phase accounting sums differ')
                counts['terminal_phase_accounting_sum_gaps']+=1
            counts['reported_phase_token_overshoots']+=sum(p['tokens']>p['token_cap'] for p in phases)
            counts['reported_phase_tool_overshoots']+=sum(p['tool_calls']>p['tool_cap'] for p in phases)
            counts['reported_auxiliary_call_overshoots']+=sum(p.get('model_calls',0)>2 for p in phases if p['phase']=='independent_recompute')
        else:
            require(B.failed(row),'Nonterminal team lacks phase metadata');counts['terminal_errors_without_phase_records']+=1
        if counts['source_valid_executed_releases']:
            require(not unknown and not counts['reported_aggregate_token_overshoots'] and not counts['reported_tool_overshoots']
              and not counts['reported_phase_token_overshoots'] and not counts['reported_phase_tool_overshoots'],'Overshot/unknown result claims executed source-valid release')
    counts['recorded_clean_rendered_source_checks']+=not B.failed(row) and revision.get('final_check_issues')==[]
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
        counts.update(attempt_counts(row,restored,strategy=='multi_agent_v6'))
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
      'limits':['Auxiliary invocations, retained model replies and SDK journal events are distinct evidence; no equality is inferred on failed/fatal paths.',
                'Executed programs and recorded clean source citations do not establish semantic/clinical correctness or search completeness.',
                'Ordinary source citations bind exact retained check results; the guard does not refetch records or substitute stricter equality for the frozen checker semantics.',
                'Unavailable usage remains unknown; failures and phase overshoots are retained outcomes, not deleted questions.',
                'Initial files/journals are individually captured snapshots, not a simultaneous state; unmatched latest starts can be in flight.',
                'Completion terminality is an external root process-supervision assertion; this helper validates complete saved cohorts and provenance.',
                'No program is rerun and no source records are refetched. Mechanical artifact checks use retained immutable data only.',
                'Ineligible/prose-only primaries remain their actual Single outcomes; source-bound execution and source-valid nulls are separate from benchmark answer credit.',
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
