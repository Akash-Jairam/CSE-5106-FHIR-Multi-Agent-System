"""Completion provenance and exact allowlisted archive; no model calls or scoring.

Run only after both owned jobs exit. Credentials and runtime bootstrap are never
loaded. A completed terminal failure remains a first attempt; unknown failed-call
usage remains unknown. This does not judge clinical correctness or acceptance.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
from importlib import metadata
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

ARM_STRATEGIES = {'ordinary_single_agent':'single_agent', 'ordinary_multi_agent':'multi_agent_v3'}
PHASE_CAPS = {'contract':(10000,0),'solver':(175000,27),'audit':(15000,0),'targeted':(0,3)}
ADAPTER_FILES = {'.local-setup/azure_adapter.py','.local-setup/sitecustomize.py','.local-setup/azure.py','.local-setup/phase3.py'}
BASE_FILES = {'.local-provider.json','run_single.log','run_team.log',
              'ordinary_single_agent.json','ordinary_single_agent.json.run.json',
              'ordinary_multi_agent.json','ordinary_multi_agent.json.run.json'}
AUDIT_RE = re.compile(r'audit/([a-f0-9]{64})\.json\.gz')
JOURNAL_RE = re.compile(r'provider-events/azure-diagnostics-([0-9]+)-([a-f0-9]{32})\.jsonl')
MAX_AUDIT_BYTES = 512*1024*1024
MAX_JOURNAL_LINE_BYTES = 64*1024
CLASS_RE = re.compile(r'[A-Za-z0-9_.]{1,200}')
REQUEST_RE = re.compile(r'[A-Za-z0-9_.:-]{1,160}')

def require(value, message):
    if not value:
        raise ValueError(message)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def json_equal(left,right):
    # Python equality conflates False/0, True/1 and int/float. Finite JSON
    # equality preserves the recorded JSON type and all strings/array order.
    encode=lambda x:json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    return encode(left)==encode(right)

def guard_destinations(root,record,output,report,archive=None):
    resolved=[]
    for kind,path,suffix in [('report',report,'.json'),('archive',archive,'.zip')]:
        if path is None:
            continue
        require(not path.is_symlink() and not path.exists(),'Destination already exists; choose a fresh '+kind+' path')
        target=path.resolve()
        require(target.suffix==suffix,'Unexpected '+kind+' extension')
        require(not target.is_relative_to(output.resolve()),'Destination would alter owned raw run files')
        if target.is_relative_to(root.resolve()):
            allowed=record.resolve() if kind=='report' else (root/'output/archives').resolve()
            require(target.is_relative_to(allowed),'Destination would alter protected repository inputs/source')
        resolved.append(target)
    if len(resolved)==2:
        require(resolved[0]!=resolved[1] and not resolved[0].is_relative_to(resolved[1]) and not resolved[1].is_relative_to(resolved[0]),'Report/archive destinations collide')

def stamp(path):
    s=Path(path).stat()
    return (s.st_size,s.st_mtime_ns)

def public_source_path(root,name):
    p=PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts,'Unsafe source path')
    require(p.name not in {'config.yml','config.yaml','.env'} and not str(p).startswith('.local-setup/config'),'Forbidden secret source file')
    return root.joinpath(*p.parts)

def owned_file(output,name):
    p=output/name
    require(not p.is_symlink() and p.is_file() and p.resolve().is_relative_to(output.resolve()),'Missing or unsafe owned file: '+name)
    return p

def source_zip(path,mapping,digest):
    require(sha(path)==digest,'Frozen archive SHA differs')
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        require(len(names)==len(set(names)) and set(names)==set(mapping),'Frozen archive inventory differs')
        for name,value in mapping.items():
            require(hashlib.sha256(z.read(name)).hexdigest()==value,'Frozen archive entry differs: '+name)

def failed(row):
    return row.get('error') is not None and str(row['error']).strip() not in ('','nan')

def complete(row):
    latency=row.get('latency')
    return failed(row) or type(latency) in (int,float) and math.isfinite(latency) and latency>=0

def independent_restore(row,output):
    ref=row.get('audit_artifacts')
    require(isinstance(ref,dict) and ref.get('version')==2 and ref.get('codec')=='json+gzip','Missing schema2 gzip audit')
    require(ref.get('question_id')==row['question_id'],'Gzip reference question owner differs')
    match=AUDIT_RE.fullmatch(ref.get('path',''))
    require(match is not None and match[1]==ref.get('sha256'),'Invalid audit content address')
    path=owned_file(output,ref['path'])
    with gzip.open(path,'rb') as stream:
        raw=stream.read(MAX_AUDIT_BYTES+1)
    require(len(raw)<=MAX_AUDIT_BYTES,'Oversized restored audit')
    require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'Audit payload SHA differs')
    payload=json.loads(raw)
    require(isinstance(payload,dict) and isinstance(payload.get('_audit_meta'),dict) and payload['_audit_meta'].get('question_id')==row['question_id'],'Audit payload owner differs')
    fields=ref.get('fields')
    require(isinstance(fields,list) and len(fields)==len(set(fields)) and set(fields)<={'trace','retrieved_fhir_resources','revision.messages'},'Audit fields invalid')
    require(set(payload)=={'_audit_meta',*fields},'Audit payload fields differ')
    return payload,len(raw)

def nullable_int(value,nonnegative=False):
    return value is None or type(value) is int and (not nonnegative or value>=0)

def class_schema(value):
    if value is None:
        return
    require(isinstance(value,dict) and set(value)=={'class','mro'},'Unexpected class metadata keys')
    require(isinstance(value['class'],str) and CLASS_RE.fullmatch(value['class']),'Invalid class label')
    require(isinstance(value['mro'],list) and 1<=len(value['mro'])<=16 and all(isinstance(x,str) and CLASS_RE.fullmatch(x) for x in value['mro']),'Invalid class MRO')

def journal_event(event):
    base={'version','pid','ppid','counter','process_session','started_utc','event','client','fd_count'}
    end={'finished_utc','duration_seconds','outcome','exception_chain','usage','http_status','request_ids'}
    require(isinstance(event,dict) and event.get('event') in {'start','end'},'Unknown journal event')
    require(set(event)==(base if event['event']=='start' else base|end),'Journal event contains unallowlisted keys')
    require(event['version']==1 and all(type(event[k]) is int and event[k]>0 for k in ('pid','ppid','counter')),'Invalid journal identity')
    require(isinstance(event['process_session'],str) and re.fullmatch(r'[a-f0-9]{32}',event['process_session']),'Invalid journal session')
    require(datetime.fromisoformat(event['started_utc']).tzinfo is not None,'Naive journal timestamp')
    require(nullable_int(event['fd_count'],True),'Invalid FD count')
    c=event['client']
    require(isinstance(c,dict) and set(c)=={'sdk','http','transport','pool','sdk_closed','http_closed','client_id','versions'},'Unallowlisted client metadata')
    for k in ('sdk','http','transport','pool'):
        class_schema(c[k])
    require(all(c[k] is None or type(c[k]) is bool for k in ('sdk_closed','http_closed')),'Invalid closed flags')
    require(nullable_int(c['client_id'],True),'Invalid numeric client identity')
    require(isinstance(c['versions'],dict) and set(c['versions'])=={'openai','httpx2'} and all(x is None or isinstance(x,str) and re.fullmatch(r'[0-9A-Za-z.+_-]{1,64}',x) for x in c['versions'].values()),'Unallowlisted package metadata')
    if event['event']=='start':
        return
    require(datetime.fromisoformat(event['finished_utc'])>=datetime.fromisoformat(event['started_utc']),'Journal time reversed')
    duration=event['duration_seconds']
    require(type(duration) in (int,float) and math.isfinite(duration) and duration>=0,'Invalid duration')
    require(event['outcome'] in {'success','error'},'Invalid journal outcome')
    require(nullable_int(event['http_status']) and (event['http_status'] is None or 100<=event['http_status']<=599),'Invalid HTTP status')
    ids=event['request_ids']
    require(isinstance(ids,dict) and set(ids)<={'x-request-id','x-ms-request-id','apim-request-id'} and all(isinstance(x,str) and REQUEST_RE.fullmatch(x) for x in ids.values()),'Unsafe request identifier metadata')
    usage=event['usage']
    require(isinstance(usage,dict) and set(usage)=={'prompt_tokens','completion_tokens','total_tokens','cached_prompt_tokens'} and all(nullable_int(x,True) for x in usage.values()),'Invalid usage metadata')
    chain=event['exception_chain']
    require(isinstance(chain,list) and len(chain)<=8,'Unbounded exception chain')
    for cause in chain:
        require(isinstance(cause,dict) and set(cause)=={'type','errno','winerror','verify_code','flags'},'Unsafe exception metadata')
        class_schema(cause['type'])
        require(all(nullable_int(cause[k]) for k in ('errno','winerror','verify_code')),'Invalid exception code')
        flags=cause['flags']
        require(isinstance(flags,dict) and set(flags)=={'dns','tls','connect','read','write','pool','timeout','resource_fd'} and all(type(x) is bool for x in flags.values()),'Unsafe exception flags')

def parse_journal(path,output):
    name=path.relative_to(output).as_posix()
    match=JOURNAL_RE.fullmatch(name)
    require(match is not None,'Unexpected journal filename')
    owned_file(output,name)
    raw=path.read_bytes()
    parts=raw.split(b'\n')
    tail=parts.pop()
    lines=parts
    unparsed_tail_bytes=0
    if tail:
        try:
            json.loads(tail)
        except (ValueError,UnicodeDecodeError):
            unparsed_tail_bytes=len(tail)
        else:
            lines.append(tail)
    starts,ends={},{}
    counts=Counter()
    fds=[]
    clients=set()
    cause_types,cause_codes,flags,http_statuses,transports=Counter(),Counter(),Counter(),Counter(),Counter()
    for line in lines:
        require(0<len(line)<=MAX_JOURNAL_LINE_BYTES,'Invalid or oversized journal line')
        event=json.loads(line)
        journal_event(event)
        require(event['pid']==int(match[1]) and event['process_session']==match[2],'Journal event is not owned by filename identity')
        target=starts if event['event']=='start' else ends
        key=event['counter']
        require(key not in target,'Duplicate journal key')
        target[key]=event
        counts[event['event']+'s']+=1
        if event['fd_count'] is not None:
            fds.append(event['fd_count'])
        c=event['client']
        if c['client_id'] is not None:
            clients.add(c['client_id'])
        if event['event']=='end':
            counts[event['outcome']+'es' if event['outcome']=='success' else 'errors']+=1
            counts['unknown_usage_ends']+=event['usage']['total_tokens'] is None
            http_statuses[str(event['http_status'])]+=1
            t=c['transport']
            transports[t['class'] if t else 'unavailable']+=1
            if t is not None:
                require(t['class']=='httpx2.HTTPTransport' and c['http']['class']=='openai._base_client.SyncHttpxClientWrapper' and 'httpx2.Client' in c['http']['mro'] and c['versions']=={'openai':'3.26.1','httpx2':'2.13.1'},'Unexpected effective SDK default transport')
            for cause in event['exception_chain']:
                cause_types[cause['type']['class']]+=1
                for code in ('errno','winerror','verify_code'):
                    if cause[code] is not None:
                        cause_codes[code+':'+str(cause[code])]+=1
                flags.update(k for k,v in cause['flags'].items() if v)
    for key in set(starts)&set(ends):
        require(all(starts[key][k]==ends[key][k] for k in ('pid','ppid','process_session','counter','started_utc')),'Journal pair identity differs')
    counts['matched_pairs']=len(set(starts)&set(ends))
    counts['unmatched_starts']=len(set(starts)-set(ends))
    counts['unmatched_ends']=len(set(ends)-set(starts))
    counts['unvalidated_partial_bytes']=unparsed_tail_bytes
    return {'path':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),
       'counts':dict(counts),'safe_to_archive':unparsed_tail_bytes==0,
       'pairing_complete':not (set(starts)^set(ends)) and unparsed_tail_bytes==0,
       'unmatched_start_counters':sorted(set(starts)-set(ends)),
       'unmatched_end_counters':sorted(set(ends)-set(starts)),
       'cause_types':dict(cause_types),'cause_codes':dict(cause_codes),'cause_flags':dict(flags),
       'http_status_counts':dict(http_statuses),'completed_default_transports':dict(transports),
       'numeric_fd_min':min(fds) if fds else None,'numeric_fd_max':max(fds) if fds else None,
       'distinct_nonnull_sdk_client_ids':len(clients)}

def frozen_provenance(root,record,output):
    m=load(record/'manifest.json')
    require(len(m['source_sha256'])==138,'Expected138 frozen source files')
    for name,value in m['source_sha256'].items():
        require(sha(public_source_path(root,name))==value,'Frozen source differs: '+name)
    source_zip(record/'source.zip',m['source_sha256'],m['source_archive_sha256'])
    require(sha(record/'design.txt')==m['design_sha256'],'Frozen design differs')
    require(sha(root/'manifests/split.json')==m['split_sha256'],'Frozen split differs')
    require(sha(root/'final_dataset/questions_answers_sql_fhir.csv')==m['dataset_sha256'],'Frozen dataset differs')
    a=load(record/'runtime_adapter_manifest.json')
    require(set(a['source_sha256'])==ADAPTER_FILES,'Expected four allowlisted adapter sources')
    for name,value in a['source_sha256'].items():
        require(sha(public_source_path(root,name))==value,'Frozen adapter differs')
    source_zip(record/'runtime_adapter_sources.zip',a['source_sha256'],a['archive_sha256'])
    b=load(owned_file(output,'.local-provider.json'))
    require(set(b)=={'provider_profile_sha256','adapter_sha256','bootstrap_sha256'} and b==a['expected_provider_binding'],'Public binding differs')
    require(b['adapter_sha256']==a['source_sha256']['.local-setup/azure_adapter.py'] and b['bootstrap_sha256']==a['source_sha256']['.local-setup/sitecustomize.py'],'Adapter/source binding differs')
    require(b['provider_profile_sha256']==load(record/'health_preflight.json')['request_manifest']['provider_identity_profile_sha256'],'Public profile differs from normal preflight')
    return m,a,b

def code_hash(root):
    # Execute only public hash/dependency functions, no runner/agent/bootstrap.
    from packaging.requirements import Requirement
    tree=ast.parse((root/'run_agent.py').read_text())
    pure=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'code_fingerprint','runtime_dependency_versions'}],type_ignores=[])
    require(len(pure.body)==2,'Public hash functions missing')
    namespace={'Path':Path,'hashlib':hashlib,'json':json,'metadata':metadata,
               'Requirement':Requirement,'__file__':str(root/'run_agent.py')}
    sys.path.insert(0,str(root/'.local-setup/phase2-dependencies'))
    exec(compile(pure,'<public-source-hash-functions>','exec'),namespace)
    return namespace['code_fingerprint']()

def load_public_store(root):
    path=root/'scripts/audit_store.py'
    spec=importlib.util.spec_from_file_location('completion_audit_store',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def allowed_paths(output,references,journal_names):
    require(all(AUDIT_RE.fullmatch(x) for x in references),'Unsafe archive audit path')
    require(all(JOURNAL_RE.fullmatch(x) for x in journal_names),'Unsafe archive journal path')
    allowed=sorted(BASE_FILES|set(references)|set(journal_names))
    for name in allowed:
        owned_file(output,name)
    return allowed

def write_archive(archive,output,allowed,files):
    archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name in allowed:
            z.write(owned_file(output,name),name)
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None,'Archive CRC verification failed')
        require(len(z.namelist())==len(allowed) and sorted(z.namelist())==allowed,'Archive inventory differs')
        for name,expected in files.items():
            digest=hashlib.sha256()
            with z.open(name) as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    digest.update(block)
            require(digest.hexdigest()==expected['sha256'] and sha(output/name)==expected['sha256'],'Archived bytes differ from verified file')
    return {'raw_archive_path':str(archive.resolve()),'raw_archive_sha256':sha(archive),'raw_archive_bytes':archive.stat().st_size}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--owned_jobs_finished',action='store_true',required=True,
                        help='Root must confirm both owned processes exited before invoking')
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--archive',type=Path)
    args=parser.parse_args(argv)
    root=args.root.resolve()
    record=root/'benchmarks/design-v3d-20261014'
    output=root/'output/experiments/gpt-6-luna/query_contract_v3d_300q'
    guard_destinations(root,record,output,args.report,args.archive)
    m,a,b=frozen_provenance(root,record,output)
    expected=load(root/'manifests/split.json')['splits']['held_out']['question_ids']
    require(len(expected)==len(set(expected))==300,'Fixed cohort differs')
    expected_code=code_hash(root)
    store=load_public_store(root)
    from packaging.requirements import Requirement  # public dependency only
    import pandas as pd
    controls={}
    for name,directory,digest in (
        ('archived_single','midterm_20261012_full_v1','295f291b28cc3ca67f46799bd621b20a5d6a071be9d33e6590e9fac5b4377478'),
        ('refreshed_single','verified_team_v2_300q','6010afe99ca9c6b90e52ee3f359e64a3ab9ccf331a826bb8e1de306b2d0ee873')):
        p=root/'output/experiments/gpt-6-luna'/directory/'ordinary_single_agent.json'
        require(sha(p)==digest,'Historical control checksum differs')
        cb=load(p.parent/'.local-provider.json')
        require(cb['provider_profile_sha256']==b['provider_profile_sha256'] and cb['bootstrap_sha256']==b['bootstrap_sha256'],'Historical common profile/bootstrap differs')
        controls[name]={'path':str(p),'sha256':digest,'public_profile_equal':True,'bootstrap_equal':True,
                        'adapter_equal':cb['adapter_sha256']==b['adapter_sha256'],
                        'adapter_difference':'V3d adds only the frozen optional observer; not byte-identical to older controls'}
    identities,inputs,arms,references={},{},{},set()
    initial_stamps={}
    artifact_bytes=0
    for arm,strategy in ARM_STRATEGIES.items():
        path=owned_file(output,arm+'.json')
        initial_stamps[path.name]=stamp(path)
        raw=path.read_bytes()
        rows=json.loads(raw)
        require(isinstance(rows,list) and len(rows)==300 and [r['question_id'] for r in rows]==expected,'Scheduled300 ordered cohort differs')
        require(all(complete(r) for r in rows),'Both arms must finish all300 first attempts before completion audit/scoring/archive')
        require(all(not r.get('attempt_history') for r in rows),'Fresh run contains retry history')
        ip=owned_file(output,arm+'.json.run.json')
        identity=load(ip)
        checks={'agent_strategy':strategy,'model':'gpt-6-luna','max_total_tokens':200000,'max_tool_calls':30,
                'split':'held_out','split_manifest_hash':m['split_sha256'],'enable_cache':True,
                'add_patient_fhir_id':True,'audit_storage':'gzip','audit_artifact_schema':2,'code_hash':expected_code,
                'condition':None,'perturbations_hash':None,'subset_hash':None,'repeat':None}
        require(all(json_equal(identity.get(k),v) for k,v in checks.items()),'Run identity differs from protocol')
        columns=['question_id','question','true_answer','assumption','patient_fhir_id']
        input_json=pd.DataFrame(rows)[columns].to_json(orient='records',date_format='iso')
        require(hashlib.sha256(input_json.encode()).hexdigest()==identity['input_hash'],'Actual ordered input digest differs')
        require(identity['input_hash']=='25e57321644744d5c0718c949c1f90e8be4f53b8aadce4d0118843a99c7b96ce','Historical fixed cohort/input digest differs')
        inputs[arm]=input_json
        identities[arm]=identity
        counts=Counter()
        policies=Counter()
        for row in rows:
            counts['completed_first_attempts']+=1
            counts['terminal_errors']+=failed(row)
            payload,nbytes=independent_restore(row,output)
            artifact_bytes+=nbytes
            restored=store.restore_result_audit(row,output,question_id=row['question_id'])
            for field in row['audit_artifacts']['fields']:
                actual=restored['revision']['messages'] if field=='revision.messages' else restored[field]
                require(json_equal(actual,payload[field]),'Restored full field differs from exact gzip payload')
            for field,value in row.items():
                if field in {'trace','retrieved_fhir_resources','audit_artifacts'}:
                    continue
                if field=='revision' and 'revision.messages' in row['audit_artifacts']['fields']:
                    require(json_equal({k:v for k,v in restored[field].items() if k!='messages'},{k:v for k,v in value.items() if k!='messages'}),'Inline revision changed on restore')
                else:
                    require(json_equal(restored.get(field),value),'Inline result field changed on restore')
            require(isinstance(restored.get('trace'),list),'Full trace not restored')
            counts['verified_owned_gzip_audits']+=1
            references.add(row['audit_artifacts']['path'])
            tokens,budget=row.get('tokens') or {},row.get('budget') or {}
            require(json_equal(budget.get('max_total_tokens'),200000) and json_equal(budget.get('max_tool_calls'),30),'Declared per-question budget differs')
            counts['budget_stops']+=bool(budget.get('stopped'))
            counts['unknown_provider_usage_attempts']+=any(c.get('usage_available') is False for c in tokens.get('per_call') or [])
            counts['reported_aggregate_token_overshoots']+=(tokens.get('total') or 0)>200000
            counts['reported_tool_overshoots']+=(budget.get('tool_calls_run') or 0)>30
            revision=row.get('revision') or {}
            phases=revision.get('phases') or []
            fatal=any(bool(p.get('fatal')) for p in phases)
            counts['fatal_phase_attempts']+=fatal
            counts['fatal_phase_records']+=sum(bool(p.get('fatal')) for p in phases)
            if strategy=='multi_agent_v3':
                if phases:
                    require(sum(p['tokens'] for p in phases)==tokens['total'] and sum(p['tool_calls'] for p in phases)==budget['tool_calls_run'],'Reported phase sums differ from aggregate')
                else:
                    require(failed(row),'Nonterminal team result lacks phase records')
                    counts['terminal_errors_without_phase_records']+=1
                for phase in phases:
                    require(phase['phase'] in PHASE_CAPS and json_equal([phase['token_cap'],phase['tool_cap']],list(PHASE_CAPS[phase['phase']])),'Declared phase caps differ')
                    counts['reported_phase_token_overshoots']+=phase['tokens']>phase['token_cap']
                    counts['reported_phase_tool_overshoots']+=phase['tool_calls']>phase['tool_cap']
                policy=revision.get('release_policy')
                policies[str(policy)]+=1
                if policy in {'source_valid_primary_preserved','source_checked_targeted_local_constraint'}:
                    require(revision.get('final_check_issues')==[],'Source-valid release contains final source issues')
                    final=revision['final']
                    require(all(json_equal(row.get(display),final.get(key)) for display,key in [('agent_answer','answer'),('evidence','evidence'),('limitation','limitation')]),'Released artifact differs from inline answer/evidence')
            clean=not failed(row) and revision.get('final_check_issues')==[]
            counts['recorded_clean_release_attempts']+=clean
            counts['clean_release_with_fatal_phase']+=clean and fatal
        require(stamp(path)==initial_stamps[path.name],'Result changed during verification')
        arms[arm]={**dict(counts),'release_policies':dict(policies),'result_sha256':hashlib.sha256(raw).hexdigest(),
                   'identity_sha256':sha(ip),'bytes':len(raw)}
        del rows,raw
    require(inputs['ordinary_single_agent']==inputs['ordinary_multi_agent'],'Arm ordered inputs differ')
    require(json_equal({k:v for k,v in identities['ordinary_single_agent'].items() if k!='agent_strategy'},{k:v for k,v in identities['ordinary_multi_agent'].items() if k!='agent_strategy'}),'Arm identities differ beyond actual strategy')
    journals=[]
    for p in sorted((output/'provider-events').glob('*.jsonl')):
        journals.append(parse_journal(p,output))
    require(journals,'No provider journal files')
    journal_names=[j['path'] for j in journals]
    allowed=allowed_paths(output,references,journal_names)
    files={n:{'sha256':sha(output/n),'bytes':(output/n).stat().st_size} for n in allowed}
    # References are the allowlist, never a recursive archive of the output tree.
    on_disk_audits={p.relative_to(output).as_posix() for p in (output/'audit').glob('*.json.gz') if re.fullmatch(r'[a-f0-9]{64}\.json\.gz',p.name)}
    totals=Counter()
    for j in journals:
        totals.update(j['counts'])
        require(files[j['path']]['sha256']==j['sha256'],'Journal changed during completion verification')
    report={'status':'PASS','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
       'scope':'Completed first-attempt provenance and lossless stored-audit verification; no accuracy scoring or clinical validation',
       'expected_first_attempts':600,'frozen_source_files_verified':138,'runtime_adapter_files_verified':4,
       'source_archive_sha256':m['source_archive_sha256'],'design_sha256':m['design_sha256'],
       'dataset_sha256':m['dataset_sha256'],'split_sha256':m['split_sha256'],'code_hash':expected_code,
       'provider_binding':b,'arms':arms,'historical_controls':controls,'raw_files':files,
       'restored_audit_records':600,'independent_restored_bytes':artifact_bytes,
       'unique_referenced_artifacts':len(references),'expected_base_members_without_dedup':607,
       'actual_base_members':len(BASE_FILES|references),'provider_journal_members':len(journals),
       'unreferenced_local_audit_files_excluded':sorted(on_disk_audits-references),
       'journal':{'schema_privacy_allowlist_valid':True,'totals':dict(totals),'files':journals},
       'helper_sha256':sha(Path(__file__)),'model_calls':0,'network_calls':0,'scoring_performed':False,
       'archive_requested':bool(args.archive),'archive_verified':False,
       'limits':['Root asserts both owned jobs exited; timestamps/hashes are rechecked but are not a substitute for process supervision.',
                 'Historical controls have the same public profile/bootstrap; V3d optional observer changes adapter bytes.',
                 'All600 attempts are retained including failures, unknown usage, reported overshoots and missing early-fatal phase summaries.',
                 'Journal missing pairs are incomplete evidence, never successful calls or known usage; causes are counted from recorded classes/codes only.',
                 'Provider events deliberately lack question identifiers, so journal errors cannot be assigned to a particular row or phase by this helper.',
                 'Default HTTPX2 observations do not prove the proxy route; parsed successes with null status are not asserted HTTP200.',
                 'Exact payload restoration verifies stored provenance, not clinical correctness or equality to uncaptured live FHIR store values.',
                 'Content-address deduplication may yield fewer than607 distinct base members while preserving all600 owned restored records. Only referenced artifacts enter the ZIP.']}
    if not all(j['pairing_complete'] for j in journals):
        report['status']='PASS_WITH_JOURNAL_LIMITATIONS'
    # Revalidate the frozen source and all candidate member bytes immediately
    # before archiving. Failed API attempts do not affect this integrity gate.
    frozen_provenance(root,record,output)
    require(all(sha(output/n)==v['sha256'] for n,v in files.items()),'Owned completed files changed before archive')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation prevents the report from overwriting any prior file.
    # Keep this newly owned handle open for the optional archive metadata update.
    with args.report.open('x',encoding='utf-8') as report_stream:
        report_stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
        report_stream.flush()
        if args.archive:
            require(all(j['safe_to_archive'] for j in journals),'Journal contains unvalidated partial bytes; completion report preserved, archive requires review')
            report.update(write_archive(args.archive,output,allowed,files))
            frozen_provenance(root,record,output)
            report['archive_verified']=True
            report_stream.seek(0)
            report_stream.truncate()
            report_stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
            report_stream.flush()
    print(json.dumps({'status':report['status'],'completed_first_attempts':600,'arms':arms,
                      'archive_members':len(allowed),'archive_written':bool(args.archive),
                      'journal_totals':dict(totals),'report':str(args.report)},allow_nan=False))
    return report

if __name__=='__main__':
    main()
