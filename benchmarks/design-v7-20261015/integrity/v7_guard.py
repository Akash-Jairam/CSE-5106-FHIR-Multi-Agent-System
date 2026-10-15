"""Model-free V7 subset prelaunch, initial and completion integrity checks.

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
ARM_STRATEGIES = {'ordinary_single_agent':'single_agent', 'ordinary_multi_agent':'multi_agent_v7'}
PROTOCOL = 'full-single-residual-trajectories-v7'
PHASE_CAPS = {'natural_primary':(200000,30)}
ENTRY_BYTES=32768
BOUNDARY_TOKENS=512
SELECTOR_COMPLETION=2000
ACCEPTANCE='Team >239 AND Team >fresh contemporary Single AND Team >actual recorded full-cap A-only; all300 first attempts/errors, no replacements/retries. Missing A reconstruction makes the last gate unavailable and prevents selection.'
A_ONLY_GATE='Strict positive net auxiliary contribution; actual normalized released A/error and primary fatal state, never computed withheld or fabricated null. All300 reconstructible outcomes required.'
ASSURANCE = 'fallible_model_selection_of_executed_source_checked_candidates'
FIXED_INPUT_HASH = '25e57321644744d5c0718c949c1f90e8be4f53b8aadce4d0118843a99c7b96ce'
FIXED_DATASET_SHA = 'e2045692fef7f5f4f77496935160f5fc727e162d213e94feed61401948e512a0'
FIXED_SPLIT_SHA = 'f28db462ebcfe4e0d87675c86de9a08adef4038e26dbddcf88c50ddf92e94015'
HEX = re.compile(r'^[a-f0-9]{64}$')
RECORD_META = ('manifest.json','runtime_adapter_manifest.json','expected_public_binding.json','prospective_protocol.json','launch_commands.json')
REQUIRED_SOURCE = {'agent/continuous_primary.py','agent/residual_trajectory_agent.py','agent/residual_trajectory_phase.py','agent/single_agent.py','agent/base_agent.py','agent/ehr_semantics.py',
                   'scripts/audit_store.py','run_agent.py','requirements.txt'}
PROVIDER_FILES = {'.local-setup/azure_adapter.py','.local-setup/sitecustomize.py','.local-setup/azure.py','.local-setup/phase3.py'}



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
    require(not any(part.startswith('.') for part in Path(name).parts) or name in PROVIDER_FILES,
            'Unallowlisted hidden source path')
    require(not p.is_symlink() and p.is_file() and p.resolve().is_relative_to(root.resolve()),'Unsafe public source file')
    return p

def pin_record(record,destination):
    """Prepare, but never silently trust, pins for root review BEFORE launch."""
    require(not destination.exists() and not destination.is_symlink(),'Pins destination already exists')
    manifest=load(record/'manifest.json')
    require(REQUIRED_SOURCE <= set(manifest['source_sha256']),'Required V7 source inventory is missing')
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
    require(isinstance(mapping,dict) and REQUIRED_SOURCE <= set(mapping),'V7 source inventory incomplete')
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
        # its existing numeric/date/string equivalences), not V7 probe equality.
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
    expected={'acceptance':ACCEPTANCE,'recorded_A_only_gate':A_ONLY_GATE,'model':'gpt-6-luna','fixed_questions_per_arm':300,'workers_per_arm':1,'first_attempt_only':True,
      'primary_iterations':30,'primary_tools_inclusive_automatic_checks':True,'reserved_primary_source_checks':0,
      'full_primary_allowance':[200000,30,30],'alternative_allocation':'actual_residual_less_initialization_and_selector_reservation',
      'max_team_model_calls':30,'max_control_model_calls':30,'global_caps':[200000,30],
      'estimated_input_guard':32000,'minimum_team_correct':240,'expected_public_binding':binding,
      'bounds':{'candidate_entry_bytes':ENTRY_BYTES,'selector_boundary_tokens':BOUNDARY_TOKENS,
                'selector_completion':SELECTOR_COMPLETION,'selector_response_chars':16000},
      'alternative_initialization_tool_charge':1,'minimum_residual_tools':3,'minimum_residual_sdk_attempts':3}
    require(all(equal(prospective.get(k),v) for k,v in expected.items()),'Prospective V7 profile/caps differ')


def validate_launch_commands(record,output):
    commands=load(record/'launch_commands.json')
    require(commands.get('concurrent') is True and equal(commands.get('workers_per_arm'),1),'Declared launch concurrency differs')
    require(set(commands.get('commands',{}))=={'single','team'},'Declared launch arm inventory differs')
    for label,arm in [('single','ordinary_single_agent'),('team','ordinary_multi_agent')]:
        entry=commands['commands'][label]
        expected=['caffeinate','-i','.venv/bin/python','.local-setup/phase3.py','run','--project-root','.',
          '--model','gpt-6-luna','--max_total_tokens','200000','--max_tool_calls','30',
          '--num_processes','1','--save_interval','1','--output_dir',str(output),
          '--split','held_out','--team_strategy','multi_agent_v7','--audit_storage','gzip','--arms',arm]
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
    for name in ('agent/residual_trajectory_agent.py','agent/residual_trajectory_phase.py'):
        values={}
        for node in ast.parse((root/name).read_text()).body:
            if isinstance(node,ast.Assign):
                try:value=ast.literal_eval(node.value)
                except (ValueError,TypeError):continue
                for target in node.targets:
                    if isinstance(target,ast.Name):values[target.id]=value
        literals[name]=values
    require(all(equal(literals['agent/residual_trajectory_agent.py'].get(k),v) for k,v in
       {'PROTOCOL':PROTOCOL,'ENTRY_BYTES':ENTRY_BYTES,'BOUNDARY_TOKENS':BOUNDARY_TOKENS,
        'SELECTOR_COMPLETION':SELECTOR_COMPLETION}.items()),'V7 literal protocol/bounds differ')
    require(literals['agent/residual_trajectory_phase.py'].get('SDK_CAP')==30,'V7 SDK literal differs')
    adapter=load(record/'runtime_adapter_manifest.json');provider=adapter.get('source_sha256')
    require(isinstance(provider,dict) and set(provider)==PROVIDER_FILES,'Four public provider-source inventory differs')
    for name,value in provider.items():
        require(isinstance(value,str) and HEX.fullmatch(value) and sha(public_path(root,name))==value,'Public provider source differs')
    B.source_zip(record/'runtime_adapter_sources.zip',provider,adapter['archive_sha256'])
    binding=load(record/'expected_public_binding.json')
    require(isinstance(binding,dict) and set(binding)=={'provider_profile_sha256','adapter_sha256','bootstrap_sha256'}
      and all(isinstance(v,str) and HEX.fullmatch(v) for v in binding.values())
      and binding['adapter_sha256']==provider['.local-setup/azure_adapter.py']
      and binding['bootstrap_sha256']==provider['.local-setup/sitecustomize.py'],'Pinned public provider binding differs')
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


def finite_json(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=True,allow_nan=False)


def integer(value):return type(value) is int and value>=0


def json_object(text):
    def pairs(items):
        out={}
        for key,value in items:
            require(key not in out,'Duplicate retained selector JSON key');out[key]=value
        return out
    value=json.loads(text,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    finite_json(value);require(isinstance(value,dict),'Retained selector is not an object');return value


def reserve(prompt):return prompt+max(64,(prompt*5+99)//100)


def read_selector_prompt(root):
    def assignment(path,name):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):return node.value
        raise ValueError('Missing frozen selector/guide literal')
    node=assignment(root/'agent/residual_trajectory_agent.py','SELECTOR_PROMPT')
    require(isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add) and isinstance(node.right,ast.Name)
      and node.right.id=='SEMANTIC_GUIDANCE','Selector prompt composition differs')
    return ast.literal_eval(node.left)+ast.literal_eval(assignment(root/'agent/ehr_semantics.py','SEMANTIC_GUIDANCE'))


def successful_retrievals(trace):
    requests={};out=[];names={'get_resources_by_patient_fhir_id','get_resources_by_resource_id'}
    for message in trace:
        if message.get('role')=='assistant':
            for call in message.get('tool_calls') or []:
                f=call.get('function') or {}
                if f.get('name') in names:requests[call.get('id')]=f
        if message.get('role')!='tool' or message.get('name') not in names:continue
        f=requests.get(message.get('tool_call_id'))
        if not f:continue
        try:output=json.loads(message['content']);arguments=json.loads(f['arguments'])
        except (TypeError,ValueError,KeyError):
            try:arguments=json.loads(f['arguments'])
            except (TypeError,ValueError,KeyError):continue
            match=re.fullmatch(r"retrieved_resources\['([^']+)'\] \((\d+) items; (\d+) fetched\)",message.get('content') or '')
            if not match or arguments.get('resource_type')!=match[1]:continue
            output={'recorded_success_reply':message['content'],'held_count':int(match[2]),'fetched_count':int(match[3])}
        if not isinstance(output,dict) or output.get('error'):continue
        out.append({'tool_call_id':message.get('tool_call_id'),'tool':f['name'],'arguments':arguments,
          'recorded_reply':output,'physical_metadata_available':'retrieval_metadata' in output,
          'assurance':'request_and_retained_reply_not_search_completeness'})
    return out


def candidate_entry(snapshot,trace):
    artifact=snapshot['artifact']
    return {'released':{k:snapshot['released'].get(k) for k in ('answer','evidence','limitation','error')},
      'verification':'released_execution_and_original_source_bound' if artifact else 'released_unverified',
      'program':artifact.get('code') if artifact else None,
      'artifact':{k:artifact[k] for k in ('answer','evidence','operation','contributing_ids','limitation')} if artifact else None,
      'source_check':snapshot['source_check'],'successful_retrievals':successful_retrievals(trace),
      'workspace_sha256':snapshot['workspace_sha256'],'snapshot_sha256':_aux_fingerprint(snapshot)}


def observed_phase_counts(trace):
    """Retained natural replies; automatic check requests are host synthesis."""
    responses=0;tools=0
    for message in trace:
        if message.get('role')=='assistant':
            calls=message.get('tool_calls') or []
            automatic=(message.get('content') is None and len(calls)==1
              and calls[0].get('id') in {'automatic_field_check_1','automatic_field_check_2'}
              and (calls[0].get('function') or {}).get('name')=='check_fields')
            responses+=not automatic
        if message.get('role')=='tool':
            tools+=not str(message.get('content','')).startswith('Not run: the tool-call budget')
    return responses,tools


def validate_snapshot(snapshot,phase,trace,label):
    require(isinstance(snapshot,dict) and set(snapshot)=={'version','kind','candidate','phase','released','artifact','workspace',
      'source_check','usage','binding','workspace_sha256','trace_sha256','tagged_trace_sha256','diagnostics'},'Snapshot envelope differs')
    require(snapshot['version']==1 and snapshot['kind']=='candidate_snapshot' and snapshot['candidate']==label
      and snapshot['phase']==phase['phase'] and equal(snapshot['diagnostics'],phase),'Snapshot phase/version differs')
    require(isinstance(snapshot['released'],dict) and set(snapshot['released'])=={'answer','evidence','limitation','error','revision','budget'},'Actual released envelope differs')
    require(isinstance(snapshot['workspace'],dict) and snapshot['workspace_sha256']==fingerprint(snapshot['workspace']),'Snapshot own-workspace SHA differs')
    require(_aux_fingerprint(trace)==snapshot['tagged_trace_sha256'],'Snapshot phase-tagged trace SHA differs')
    untagged=[{k:v for k,v in message.items() if k!='phase'} for message in trace]
    require(_aux_fingerprint(untagged)==snapshot['trace_sha256'],'Snapshot own natural trace SHA differs')
    require(type(phase.get('fatal')) is bool and type(phase.get('eligible')) is bool and type(phase.get('usage_known')) is bool,'Phase state flags malformed')
    for k in ('tokens','tool_calls','natural_tool_calls','model_calls','token_cap','tool_cap','sdk_cap'):require(integer(phase.get(k)),'Phase integer/cap malformed')
    require(phase['sdk_cap']<=30,'Phase SDK cap exceeds global30')
    init=phase.get('initialization');require(isinstance(init,dict) and integer(init.get('charged')) and init['charged']<=1,'Initialization charge malformed')
    require(phase['tool_calls']==phase['natural_tool_calls']+init['charged'],'Inclusive phase tools do not include initialization once')
    if label=='A':require(init=={'charged':0},'A/common initialization must remain excluded equally')
    else:
        require(all(integer(init.get(k)) and init[k]<=1 for k in ('started','completed'))
          and init['completed']<=init['charged']<=init['started'],'B initialization lifecycle differs')
    usage=snapshot['usage'];totals=usage.get('total_usage') if isinstance(usage,dict) else None;calls=usage.get('per_call') if isinstance(usage,dict) else None
    require(isinstance(totals,dict) and isinstance(calls,list),'Snapshot usage envelope malformed')
    require(equal(totals.get('total_tokens'),phase['tokens']) and equal(totals.get('llm_calls'),phase['model_calls']),'Phase usage totals differ')
    unknown=any(c.get('usage_available') is not True for c in calls if isinstance(c,dict))
    require(phase['usage_known'] is (not unknown),'Phase known-usage flag differs')
    over=phase['tokens']>phase['token_cap'] or phase['natural_tool_calls']>phase['tool_cap'] or phase['model_calls']>phase['sdk_cap']
    if unknown or over:require(phase['fatal'],'Unknown or overshot phase falsely nonfatal')
    if not phase['fatal']:
        require(all(isinstance(c,dict) and integer(c.get('prompt')) and integer(c.get('completion')) and c.get('usage_available') is True for c in calls),'Nonfatal per-call usage malformed')
        require(all(integer(totals.get(k)) for k in ('prompt_tokens','completion_tokens','total_tokens','llm_calls'))
          and totals['prompt_tokens']+totals['completion_tokens']==totals['total_tokens']
          and len(calls)==totals['llm_calls'] and sum(c['prompt'] for c in calls)==totals['prompt_tokens']
          and sum(c['completion'] for c in calls)==totals['completion_tokens'],'Nonfatal phase usage unreconciled')
        raw=usage.get('raw_accounting') or {}
        require(equal(raw.get('total_usage'),totals) and equal(raw.get('call_usage'),calls)
          and raw.get('tool_calls_made')==phase['natural_tool_calls'],'Nonfatal phase raw ledger differs')
    observed_sdk,observed_tools=observed_phase_counts(untagged)
    observed=Counter({f'phase_{label}_retained_natural_model_responses':observed_sdk,
      f'phase_{label}_retained_charged_tool_replies':observed_tools,
      f'phase_{label}_charged_minus_observed_model_attempts':phase['model_calls']-observed_sdk,
      f'phase_{label}_charged_minus_observed_tool_calls':phase['natural_tool_calls']-observed_tools})
    artifact=snapshot['artifact']
    require(phase['eligible'] is (artifact is not None),'Snapshot eligibility differs')
    if artifact is None:
        require(snapshot['source_check'] is None and snapshot['binding']=={},'Unverified snapshot falsely has binding')
        return observed
    require(not phase['fatal'] and full_artifact(artifact),'Eligible snapshot fatal or malformed')
    require(phase['model_calls']==observed_sdk,'Eligible natural phase charged SDK attempts differ from actual retained model responses')
    require(phase['natural_tool_calls']==observed_tools,'Eligible natural phase charged tools differ from actual retained tool replies')
    if label=='B':
        require(all(init.get(k)==1 for k in ('started','charged','completed'))
          and init.get('kind') in {'get_starting_packet','get_perturbed_starting_packet'}
          and isinstance(init.get('packet_sha256'),str) and HEX.fullmatch(init['packet_sha256']),
          'Eligible B lacks actual completed original initialization')
    require(equal(rendered(artifact),rendered(snapshot['released'])) and not snapshot['released'].get('error'),'Computed but unreleased artifact substituted')
    released_revision=snapshot['released'].get('revision') or {}
    require(released_revision.get('final_check_issues')==[] and equal(rendered(released_revision.get('final') or {}),rendered(artifact)), 'Snapshot actual Single release/source result differs')
    code=validate_primary(untagged,{k:artifact[k] for k in ('answer','evidence','operation','contributing_ids','limitation')},snapshot['workspace'])
    require(code==artifact.get('code') and artifact.get('workspace_sha256')==snapshot['workspace_sha256'],'Snapshot code/workspace binding differs')
    binding=snapshot['binding'];require(isinstance(binding,dict),'Missing own source binding')
    for name in ('execution_trace_index','source_check_trace_index'):require(integer(binding.get(name)) and binding[name]<len(untagged),'Binding trace index differs')
    ei,ci=binding['execution_trace_index'],binding['source_check_trace_index'];require(ei<ci,'Source check precedes own execution')
    require(binding.get('code_sha256')==sha_text(code) and binding.get('workspace_sha256')==snapshot['workspace_sha256'],'Binding code/workspace SHA differs')
    require(untagged[ei].get('tool_call_id')==binding.get('execution_tool_call_id')
      and untagged[ci].get('tool_call_id')==binding.get('source_check_tool_call_id'),'Binding call IDs differ')
    check=json_object(untagged[ci]['content']);require(equal(check,snapshot['source_check']),'Snapshot own source check differs')
    _source_summary(artifact,check,snapshot['workspace'])
    requests={}
    for message in untagged:
        if message.get('role')=='assistant':
            for call in message.get('tool_calls') or []:
                require(call.get('id') not in requests,'Eligible phase has repeated local tool-call ID')
                requests[call.get('id')]=call
    execution_function=requests[binding['execution_tool_call_id']]['function'];source_function=requests[binding['source_check_tool_call_id']]['function']
    require(execution_function['name']=='execute_python_code' and source_function['name']=='check_fields','Binding names differ')
    execution_args=json_object(execution_function['arguments']);execution_result=json_object(untagged[ei]['content'])
    require(execution_args.get('mode')=='final' and execution_args.get('code')==code
      and execution_result.get('mode')=='final' and 'error' not in execution_result and 'error_type' not in execution_result
      and equal(canonical_artifact(execution_result,primary=True),{k:artifact[k] for k in ('answer','evidence','operation','contributing_ids','limitation')}),
      'Binding execution does not match actual released artifact/code')
    require(binding['source_check_tool_call_id'] in {'automatic_field_check_1','automatic_field_check_2'}
      and equal(json_object(source_function['arguments']).get('citations'),artifact['evidence']),'Binding own automatic source citations differ')
    return observed


def validate_selection(value):
    require(isinstance(value,dict) and set(value)=={'requirements','choice','reason'} and value['choice'] in {'A','B','UNKNOWN'},'Selector choice schema differs')
    require(isinstance(value['reason'],str) and 0<len(value['reason'].strip()) and len(value['reason'])<=2000,'Selector reason differs')
    rows=value['requirements'];require(isinstance(rows,list) and 1<=len(rows)<=32,'Selector requirements differ')
    for item in rows:
        require(isinstance(item,dict) and set(item)=={'requirement','A','B'} and isinstance(item['requirement'],str)
          and item['requirement'].strip() and len(item['requirement'])<=1000,'Selector requirement schema differs')
        for label in ('A','B'):
            fact=item[label];require(isinstance(fact,dict) and set(fact)=={'status','basis'}
              and fact['status'] in {'supported','not_found','contradicted'} and isinstance(fact['basis'],str)
              and fact['basis'].strip() and len(fact['basis'])<=2000,'Selector factual basis/status differs')
    return value['choice']=='B' and all(r['B']['status']=='supported' for r in rows) and any(r['A']['status']=='contradicted' for r in rows)


def v7_protocol_counts(row,restored,selector_prompt=None):
    counts=Counter();revision=restored.get('revision') or {}
    if revision.get('protocol')!=PROTOCOL:
        require(B.failed(row),'Nonterminal V7 row lacks protocol metadata');counts['terminal_errors_without_protocol_metadata']+=1;return counts
    phases=revision.get('phases');require(isinstance(phases,list) and phases,'Missing V7 phases')
    names=[p.get('phase') for p in phases];require(names in (['natural_primary'],['natural_primary','natural_alternative'],['natural_primary','natural_alternative','selection']),'V7 phase order differs')
    snapshots=revision.get('messages');require(isinstance(snapshots,list) and len(snapshots)==(1 if len(phases)==1 else 2),'Snapshot inventory differs')
    by={s.get('candidate'):s for s in snapshots};require(set(by)==({'A'} if len(snapshots)==1 else {'A','B'}),'Snapshot candidate namespaces differ')
    index={k:_aux_fingerprint(v) for k,v in by.items()};require(equal(index,revision.get('candidate_snapshot_sha256')),'Snapshot SHA inventory differs')
    trace=restored.get('trace') or [];require(isinstance(trace,list) and all(isinstance(t,dict) for t in trace),'Outer trace malformed')
    require(all(t.get('phase') in {'natural_primary','alternative_initialization','natural_alternative','selection'} for t in trace),'Unexpected outer trace namespace')
    labels=[t['phase'] for t in trace];order={'natural_primary':0,'alternative_initialization':1,'natural_alternative':2,'selection':3}
    require(labels==sorted(labels,key=order.get),'Outer phase chronology differs')
    local={label:[t for t in trace if t.get('phase')==phase] for label,phase in [('A','natural_primary'),('B','natural_alternative')]}
    for label,phase in zip(('A','B'),phases):counts.update(validate_snapshot(by[label],phase,local[label],label))
    primary=phases[0];require(equal([primary['token_cap'],primary['tool_cap'],primary['sdk_cap']],[200000,30,30]),'Full A allowance differs')
    require(equal(revision.get('A_only'),{'available':True,'snapshot_sha256':index['A'],'error':by['A']['released'].get('error'),'source_bound':by['A']['artifact'] is not None}),'Actual A-only identity differs')
    status=revision.get('status');chosen=revision.get('selected');fatal=revision.get('fatal')
    require(type(fatal) is bool and chosen in by and revision.get('independent_candidate_selected') is (chosen=='B'),'Selection metadata differs')
    statuses={'fatal_primary','A_entry_coverage_skip','residual_coverage_skip','A_packet_unavailable','fatal_mutation','fatal_alternative','B_ineligible','agreement_preserved','B_entry_coverage_skip','selector_coverage_skip','fatal_selection','selector_unavailable','selected_B','selector_preserved_A','selector_malformed_or_unavailable'}
    require(status in statuses and fatal is status.startswith('fatal_'),'V7 status/fatal differs')
    if any(p.get('fatal') for p in phases):require(fatal,'Fatal phase released nonfatal')
    if fatal:require(B.failed(row) and row.get('agent_answer') is None and row.get('evidence')==[],'Fatal attempt lacks retained error/null');counts['fatal_release_outcomes']+=1
    else:
        require(equal({'answer':row.get('agent_answer'),'evidence':row.get('evidence'),'limitation':row.get('limitation')},rendered(by[chosen]['released']))
          and equal(row.get('error'),by[chosen]['released'].get('error')),'Fallback/selection changed actual released outcome or error')
    require(equal(restored.get('retrieved_fhir_resources'),by[chosen]['workspace']),'Displayed workspace does not belong to selected candidate')
    require(equal(revision.get('final_check_issues'),[] if by[chosen]['artifact'] is not None and not fatal else None),'Final source claim differs')
    counts['status_'+status]+=1;counts['actual_A_source_bound']+=by['A']['artifact'] is not None
    admission=revision.get('admission')
    if len(snapshots)==1:
        require(chosen=='A' and not local['B'] and not any(t['phase'] in {'alternative_initialization','selection'} for t in trace),'No-B path has B/selection operations')
    if len(snapshots)==2:
        require(isinstance(admission,dict),'B lacks prior residual admission')
        rt=200000-primary['tokens'];ru=30-primary['tool_calls'];rs=30-primary['model_calls']
        require(equal([admission.get('residual_tokens'),admission.get('residual_tools'),admission.get('residual_sdk_attempts')],[rt,ru,rs]),'Residual accounting differs')
        require(ru>=3 and rs>=3 and admission.get('future_B_entry_byte_bound')==ENTRY_BYTES,'B admitted without intact tool/SDK reserves')
        bound=admission.get('selector_prompt_bound');fixed=admission.get('selector_fixed_prompt_tokens');reservation=admission.get('selector_token_reservation')
        require(integer(fixed) and bound==fixed+ENTRY_BYTES+BOUNDARY_TOKENS and reservation==reserve(bound)+SELECTOR_COMPLETION and rt>reservation,'Selector prospective reservation differs')
        require(equal([admission.get('B_token_cap'),admission.get('B_natural_tool_cap'),admission.get('B_sdk_cap')],[rt-reservation,ru-1,min(30,rs-1)]),'Residual B caps differ')
        alternative=phases[1];require(equal([alternative['token_cap'],alternative['tool_cap'],alternative['sdk_cap']],[rt-reservation,ru-1,min(30,rs-1)]),'Actual B phase caps differ from admission')
        init=[t for t in trace if t['phase']=='alternative_initialization'];require(len(init)==1,'B initialization trace inventory differs')
        require(init[0].get('role')=='tool' and init[0].get('name')=='initialize_original_starting_packet'
          and init[0].get('tool_call_id')=='v7-B-initialization' and init[0].get('host_operation') is True,'B initialization actual host charge trace differs')
        require(equal(json_object(init[0]['content']),{'question_sha256':_aux_fingerprint(row['question_with_context']),**alternative['initialization']}),'B internal initialization metadata differs')
        entry_A=candidate_entry(by['A'],local['A']);require(admission.get('A_entry_bytes')==len(finite_json(entry_A).encode())<=ENTRY_BYTES,'Intact A entry coverage differs')
        if by['B']['artifact'] is not None:counts['actual_B_source_bound']+=1
    if chosen=='B':require(status=='selected_B' and len(phases)==3 and by['B']['artifact'] is not None and not fatal,'B selection lacks clean existing B')
    elif status=='selected_B':raise ValueError('Selected B status released A')
    if status=='B_ineligible':require(by['B']['artifact'] is None and chosen=='A','Ineligible B replaced A')
    if status=='agreement_preserved':require(by['B']['artifact'] is not None and equal(by['A']['released'].get('answer'),by['B']['released'].get('answer')) and len(phases)==2 and chosen=='A','Exact JSON agreement did not preserve A')
    selected_messages=[t for t in trace if t['phase']=='selection'];selection=revision.get('selection')
    if selected_messages:
        require(len(phases)==3 and len(selected_messages) in {2,3} and selected_messages[0].get('role')=='system'
          and selected_messages[1].get('role')=='user','Selector trace packet differs')
        require(selector_prompt is None or selected_messages[0].get('content')==selector_prompt,'Selector system prompt differs')
        packet=json_object(selected_messages[1]['content'])
        require(set(packet)=={'context','A','B'} and equal(packet['context'],context_from_row(row))
          and equal(packet['A'],candidate_entry(by['A'],local['A'])) and equal(packet['B'],candidate_entry(by['B'],local['B']))
          and _aux_fingerprint(packet)==admission.get('selector_packet_sha256'),'Selector intact candidate/context packet differs')
        require(len(finite_json(packet['B']).encode())==admission.get('B_entry_bytes')<=ENTRY_BYTES,'Intact B entry coverage differs')
        require(by['B']['artifact'] is not None and not equal(by['A']['released'].get('answer'),by['B']['released'].get('answer')),'Selector lacks source-bound disagreeing B')
        diag=phases[2];require(diag.get('tool_calls')==0 and integer(diag.get('model_calls')) and diag['model_calls']<=1,'Selector used tools/multiple calls')
        require(diag.get('reservation')==admission['selector_token_reservation'] and diag.get('prompt_tokens_estimate')==admission.get('selector_prompt_tokens')<=admission['selector_prompt_bound'],'Actual selector bound differs')
        require(type(diag.get('fatal')) is bool and integer(diag.get('tokens')),'Selector phase accounting malformed')
        if diag['tokens']>reserve(diag['prompt_tokens_estimate'])+SELECTOR_COMPLETION:
            require(diag['fatal'],'Selector actual reserved ceiling overshoot falsely nonfatal')
        if not diag['fatal'] and diag['model_calls']:
            raw=diag.get('raw_accounting') or {};usage=raw.get('total_usage') or {};calls=raw.get('call_usage')
            require(isinstance(calls,list) and len(calls)==1 and isinstance(calls[0],dict)
              and calls[0].get('usage_available') is True and integer(calls[0].get('prompt'))
              and integer(calls[0].get('completion')) and calls[0]['completion']<=SELECTOR_COMPLETION
              and usage.get('llm_calls')==1 and usage.get('total_tokens')==diag['tokens']
              and usage.get('prompt_tokens')==calls[0]['prompt'] and usage.get('completion_tokens')==calls[0]['completion']
              and calls[0]['prompt']+calls[0]['completion']==diag['tokens'] and raw.get('tool_calls_made')==0,
              'Nonfatal selector raw usage/completion unreconciled')
        if selection is not None:
            require(diag['fatal'] is False and diag['model_calls']==1,'Parsed nonfatal selector response lacks exactly one charged SDK attempt')
            require(len(selected_messages)==3 and selected_messages[2].get('role')=='assistant' and not selected_messages[2].get('tool_calls'),'Retained selector native tools or missing response')
            response=selected_messages[2].get('content');require(isinstance(response,str) and len(response)<=16000 and equal(json_object(response),selection),'Selector retained response differs')
            allowed=validate_selection(selection);require((chosen=='B') is allowed,'Host strict existing-candidate selection differs')
        elif status in {'selected_B','selector_preserved_A'}:raise ValueError('Completed selection lacks retained parsed object')
    else:require(selection is None and len(phases)<3,'Missing selector trace or unexpected selection')
    if by[chosen]['artifact'] is not None and not fatal:
        counts['source_valid_executed_releases']+=1;counts['source_valid_executed_null_answers']+=row.get('agent_answer') is None
    counts['null_answers_outside_executed_release']+=row.get('agent_answer') is None and not counts['source_valid_executed_releases']
    return counts


def attempt_counts(row,restored,is_team,selector_prompt=None):
    counts=Counter(saved_first_attempts=1,terminal_errors=int(B.failed(row)),verified_owned_gzip_audits=1)
    budget=row.get('budget');tokens=row.get('tokens');revision=restored.get('revision') or {}
    if budget is None and B.failed(row):counts['early_terminal_budget_metadata_gaps']+=1
    else:require(isinstance(budget,dict) and equal(budget.get('max_total_tokens'),200000) and equal(budget.get('max_tool_calls'),30),'Declared global caps differ')
    budget=budget or {};tokens=tokens or {};unknown=any(c.get('usage_available') is not True for c in tokens.get('per_call') or [])
    counts['budget_stops']+=bool(budget.get('stopped'));counts['recorded_unknown_provider_usage_attempts']+=unknown
    counts['reported_aggregate_token_overshoots']+=(tokens.get('total') or 0)>200000
    counts['reported_tool_overshoots']+=(budget.get('tool_calls_run') or 0)>30
    counts['reported_model_call_overshoots']+=(tokens.get('llm_calls') or 0)>30
    if not tokens and B.failed(row):counts['early_terminal_usage_metadata_gaps']+=1
    if unknown or any(counts[k] for k in ('reported_aggregate_token_overshoots','reported_tool_overshoots','reported_model_call_overshoots')):
        require(B.failed(row),'Unknown/overshot accounting released as nonterminal outcome')
    phases=revision.get('phases') or []
    if is_team:
        counts.update(v7_protocol_counts(row,restored,selector_prompt))
        if phases:
            sums=equal(sum(p['tokens'] for p in phases),tokens.get('total')) and equal(sum(p['tool_calls'] for p in phases),budget.get('tool_calls_run')) and equal(sum(p['model_calls'] for p in phases),tokens.get('llm_calls'))
            if not sums:require(B.failed(row),'Nonterminal phase accounting sums differ');counts['terminal_phase_accounting_sum_gaps']+=1
            counts['fatal_phase_records']+=sum(bool(p.get('fatal')) for p in phases)
        else:require(B.failed(row),'Nonterminal team lacks phase metadata');counts['terminal_errors_without_phase_records']+=1
    counts['recorded_clean_rendered_source_checks']+=not B.failed(row) and revision.get('final_check_issues')==[]
    return counts


def verify_owned_attestation(path,digest):
    require(path is not None and isinstance(digest,str) and HEX.fullmatch(digest),'Externally root-owned terminal attestation SHA required')
    path=Path(path);require(not path.is_symlink() and path.is_file() and sha(path)==digest,'Owned terminal attestation pin differs')
    value=load(path);require(value.get('both_owned_jobs_terminal') is True and value.get('accuracy_scored') is False,'Owned terminal assertion differs')
    jobs=value.get('jobs');require(isinstance(jobs,list) and len(jobs)==2 and {j.get('arm') for j in jobs}=={'single','team'},'Owned terminal job inventory differs')
    for job in jobs:
        require(type(job.get('owned_session_id')) is int and job['owned_session_id']>0 and type(job.get('exit_code')) is int
          and job.get('saved_first_attempts')==300 and isinstance(job.get('observed_terminal_utc'),str)
          and datetime.fromisoformat(job['observed_terminal_utc']).tzinfo is not None,'Owned job is not terminal with exact300 saved attempts')
    require(jobs[0]['owned_session_id']!=jobs[1]['owned_session_id'],'Owned jobs share one session identity')
    return value


def verify_arm_snapshot(root,output,manifest,expected,code,store,arm,strategy,completion,selector_prompt=None):
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
        counts.update(attempt_counts(row,restored,strategy=='multi_agent_v7',selector_prompt))
        references.add(row['audit_artifacts']['path'])
        policies[str((restored.get('revision') or {}).get('status'))]+=1
    if not completion:require(counts['saved_first_attempts']>0,'Initial audit requires saved attempts in both arms')
    if completion:require(path.read_bytes()==raw,'Completed result changed during verification')
    return {'scheduled_rows':300,**dict(counts),'release_policies':dict(policies),
            'restored_uncompressed_bytes':artifact_bytes,'result_sha256':hashlib.sha256(raw).hexdigest(),
            'identity_sha256':sha(side),'bytes':len(raw)},identity,encoded,references

def run_check(root,record,pins_path,pins_sha,mode,report,owned_finished=False,archive=None,owned_attestation=None,owned_attestation_sha=None):
    require(mode in {'prelaunch','initial','completion'},'Unknown audit mode')
    if mode=='completion':
        require(owned_finished is True,'Root must confirm both owned jobs are terminal')
        attestation=verify_owned_attestation(owned_attestation,owned_attestation_sha)
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
      'owned_jobs_attestation_sha256':owned_attestation_sha if mode=='completion' else None,
      'limits':['Auxiliary invocations, retained model replies and SDK journal events are distinct evidence; no equality is inferred on failed/fatal paths.',
                'Executed programs and recorded clean source citations do not establish semantic/clinical correctness or search completeness.',
                'Ordinary source citations bind exact retained check results; the guard does not refetch records or substitute stricter equality for the frozen checker semantics.',
                'Unavailable usage remains unknown; failures and phase overshoots are retained outcomes, not deleted questions.',
                'Initial files/journals are individually captured snapshots, not a simultaneous state; unmatched latest starts can be in flight.',
                'Completion terminality is an external root process-supervision assertion; this helper validates complete saved cohorts and provenance.',
                'No program is rerun and no source records are refetched. Mechanical artifact checks use retained immutable data only.',
                'Ineligible/prose-only primaries remain their actual Single outcomes; source-bound execution and source-valid nulls are separate from benchmark answer credit.',
                'The fixed300 remains adaptive development validation, not confirmation of generalization.']}
    selector_prompt=read_selector_prompt(root)
    references=set();journals=[]
    if mode!='prelaunch':
        store=B.load_public_store(root);identities={};inputs={};arms={}
        for arm,strategy in ARM_STRATEGIES.items():
            summary,identity,encoded,refs=verify_arm_snapshot(root,output,manifest,expected,code,store,arm,strategy,mode=='completion',selector_prompt)
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
            for job in attestation['jobs']:
                require(job['saved_first_attempts']==arms['ordinary_single_agent' if job['arm']=='single' else 'ordinary_multi_agent']['saved_first_attempts'],'Owned exit attestation saved count differs')
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
    parser.add_argument('--owned_jobs_attestation',type=Path);parser.add_argument('--owned_jobs_attestation_sha256')
    args=parser.parse_args(argv)
    if args.mode=='pins':
        root=args.root.resolve();record=args.record.resolve();manifest=load(record/'manifest.json');output=Path(manifest['output_directory']).resolve()
        require(record.is_relative_to(root/'benchmarks') and not output.exists(),'Prepare pins only for an unlaunched formal record')
        B.guard_destinations(root,record,output,args.pins)
        pin_record(record,args.pins)
        print(json.dumps({'status':'PINS_PREPARED_NOT_APPROVAL','pins':str(args.pins),'sha256':sha(args.pins)}));return
    require(args.report is not None,'Fresh report destination required')
    result=run_check(args.root,args.record,args.pins,args.pins_sha256,args.mode,args.report,args.owned_jobs_finished,args.archive,args.owned_jobs_attestation,args.owned_jobs_attestation_sha256)
    print(json.dumps({'status':result['status'],'mode':args.mode,'report':str(args.report),
                      'arms':result.get('arms'),'journal_totals':result.get('journal',{}).get('totals')},allow_nan=False))

if __name__=='__main__':main()
