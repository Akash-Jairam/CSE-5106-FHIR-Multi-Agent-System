# Complements census.py; fixed300 engineering only, no answer/reference fields inspected.
from pathlib import Path
import ast,collections,hashlib,importlib.util,json,re
ROOT=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System');R=ROOT/'benchmarks/design-v6-20261015';O=ROOT/'output/experiments/gpt-6-luna/independent_recomputation_v6_300q';D=Path(__file__).parent;F=Path('/private/tmp/fhir-v6-conditional-integration-v3-321ju7iu/base')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(R/'completion_protocol_audit.json')=='ba7a383078218c39c45ec4b485414b1adb248d14cd6e73fe83afd28bcb9d92fd'
proof=json.loads((R/'completion_protocol_audit.json').read_text());assert proof['status']=='PASS' and proof['restored_audit_records']==600
assert sha(O/'ordinary_multi_agent.json')==proof['raw_files']['ordinary_multi_agent.json']['sha256']
spec=importlib.util.spec_from_file_location('v6_deep_store',F/'scripts/audit_store.py');store=importlib.util.module_from_spec(spec);spec.loader.exec_module(store)
counts=collections.defaultdict(collections.Counter);details=[];metrics=collections.defaultdict(list)
for raw in json.loads((O/'ordinary_multi_agent.json').read_text()):
 ref=raw['audit_artifacts'];path=O/ref['path'];assert sha(path)==proof['raw_files'][ref['path']]['sha256']
 row=store.restore_result_audit(raw,O,question_id=raw['question_id'],restore_history=False);rev=row.get('revision') or {};c=rev.get('recompute') or {};trace=row.get('trace') or [];reason=c.get('reason') or '';e=c.get('B_execution') or {};error=str(e.get('error') or '')
 gen=[m for m in trace if m.get('role')=='assistant' and m.get('phase')=='independent_generation'];code=None;args=None
 if len(gen)==1:
  tc=gen[0].get('tool_calls') or []
  if len(tc)==1:
   counts['native_function_names'][tc[0].get('function',{}).get('name','absent')]+=1
   try:args=json.loads(tc[0]['function']['arguments']);code=args.get('code') if isinstance(args,dict) else None
   except (ValueError,TypeError):counts['generation_arguments']['invalid_JSON']+=1
   if isinstance(args,dict):
    if args.get('mode')!='final':counts['generation_arguments']['mode_not_final']+=1
    if isinstance(code,str) and len(code)>6000:counts['generation_arguments']['program_over6000chars']+=1
    if set(args)-{'code','mode','timeout'}:counts['generation_arguments']['unexpected_keys']+=1
    if not isinstance(code,str) or not code.strip():counts['generation_arguments']['missing_or_empty_code']+=1
    if 'timeout' in args and (type(args['timeout']) is not int or not 0<args['timeout']<=60):counts['generation_arguments']['invalid_timeout']+=1
    if not isinstance(tc[0].get('id'),str) or not tc[0].get('id') or len(tc[0]['id'])>200:counts['generation_arguments']['invalid_id']+=1
  else:counts['generation_arguments']['not_exactly_one_native']+=1
 category='no_execution' if not e else 'executed_without_error'
 if error:
  if e.get('error_type')=='final_contract':
   if error.startswith('Final metadata missing variables:'):
    category='contract_missing_metadata';missing=error.split(': ',1)[1].split('. Set answer',1)[0].split(', ')
    counts['missing_metadata_fields'].update(x for x in missing if x in ('evidence','contributing_ids','operation','limitation'))
   elif error.startswith('Final evidence['):
    category=('citation_keys_schema' if 'requires resource_id, field_path, value and optional scope_index' in error else 'citation_resource_id_not_held_or_invalid' if '.resource_id must' in error else 'citation_field_path_invalid' if '.field_path must' in error else 'citation_scope_index_invalid' if '.scope_index must' in error else 'citation_other')
   elif error.startswith('Final contributing_ids must be unique'):category='duplicate_contributing_ids'
   elif error.startswith('Final contributing_ids must be a list'):category='contributor_id_schema'
   elif error.startswith('Final contributing_ids must already'):category='contributor_id_not_held'
   elif error.startswith('Every final cited resource'):category='citation_not_in_contributors'
   elif error.startswith('Final operation must'):category='operation_empty'
   elif error.startswith('Code executed successfully'):category='answer_variable_absent'
   elif error.startswith('Final answer must'):category='answer_not_finite_scalar_list_null'
   elif error.startswith('Computation metadata'):category='metadata_container_types'
   elif error.startswith('A nonempty final answer'):category='nonempty_answer_without_evidence'
   elif error.startswith('The answer and metadata'):category='nonfinite_or_nonJSON_output'
   else:category='other_final_contract'
  else:
   m=re.match(r'([A-Za-z_][A-Za-z_0-9]*(?:Error|Exception)):',error);category='runtime_'+m[1] if m else 'other_execution_error'
 counts['worker_outcome'][category]+=1
 assigned=set();modules=set();tree=None
 if isinstance(code,str):
  metrics['all_program_chars'].append(len(code))
  try:
   tree=ast.parse(code)
   for n in ast.walk(tree):
    if isinstance(n,(ast.Assign,ast.AnnAssign,ast.NamedExpr)):
     targets=n.targets if isinstance(n,ast.Assign) else [n.target]
     for t in targets:
      if isinstance(t,ast.Name):assigned.add(t.id)
    if isinstance(n,ast.Import):modules.update(x.name.split('.')[0] for x in n.names)
    if isinstance(n,ast.ImportFrom) and n.module:modules.add(n.module.split('.')[0])
  except SyntaxError:counts['AST']['syntax_invalid']+=1
  counts['programs_importing_module'].update(x for x in modules if re.fullmatch(r'[a-zA-Z_][a-zA-Z_0-9]*',x))
  if category=='runtime_ModuleNotFoundError':counts['module_failure_import_names'].update(modules)
  if category=='contract_missing_metadata':counts['missing_metadata_assign_patterns'][','.join(k for k in ('answer','evidence','contributing_ids','operation','limitation') if k in assigned)]+=1
  if category=='citation_keys_schema' and tree:
   for n in ast.walk(tree):
    if isinstance(n,ast.Dict):
     keys={k.value for k in n.keys if isinstance(k,ast.Constant) and isinstance(k.value,str)}
     # Whitelist schema words only. Never expose arbitrary code literals.
     if {'resource_id','field_path','value','path','field','resource_type','id','source','resource','reference'}&keys:
      safe={k for k in keys if k in {'resource_id','field_path','value','path','field','resource_type','id','source','resource','reference','scope_index'}}
      if safe:counts['invalid_evidence_code_schema_keysets'][','.join(sorted(safe))]+=1
 if reason.startswith('auxiliary unavailable:'):
  mt=re.match(r'auxiliary unavailable: ([A-Za-z_][A-Za-z_0-9]*):',reason)
  counts['aux_exception_type'][mt[1] if mt else 'unknown']+=1
 if not c:
  counts['no_aux_primary_disposition']['fatal_primary' if (rev.get('primary_diagnostics') or {}).get('fatal') else 'nonfatal_ineligible_primary']+=1
 if 'primary schema/limitation' in reason:counts['primary_aux_schema_skip']['limitation' if bool((rev.get('primary_artifact') or {}).get('limitation')) else 'other_schema']+=1
 if 'executed artifact is ineligible' in reason:counts['B_ineligible_after_worker_success']['explicit_limitation' if bool(e.get('limitation')) else 'schema_without_limitation']+=1
 if c.get('status')=='coverage_skip':
  metrics['coverage_skip_reservations'].append(c.get('two_call_token_reservation'))
  counts['coverage_skip_cause']['reservation_over25000' if c.get('two_call_token_reservation',0)>25000 else 'remaining_global_or_tools']+=1
 if c.get('packet_admitted'):metrics['admitted_reservations'].append(c.get('two_call_token_reservation'))
 if c.get('B_entry_bytes') is not None:
  metrics['candidate_entry_bytes'].append(c['B_entry_bytes'])
  if c['B_entry_bytes']>8000:metrics['rejected_candidate_entry_bytes'].append(c['B_entry_bytes'])
 check=c.get('B_source_check') or {}
 if check and not c.get('B_eligible'):
  counts['source_rejected_counts']['checks']+=1
  for result in check.get('results') or []:
   counts['source_rejected_result_status'][str(result.get('status'))]+=1
   for warning in result.get('warnings') or []:
    counts['source_rejected_warning_categories']['encounter' if 'encounter' in str(warning).lower() else 'patient' if 'patient' in str(warning).lower() else 'time' if 'time' in str(warning).lower() else 'other']+=1
  counts['source_rejected_counts']['issues']+=len(check.get('issues') or [])
 # Evidence/workspace presence only, never source record values.
 held=row.get('retrieved_fhir_resources') or {}
 countheld=sum(len(v) for v in held.values() if isinstance(v,list))
 if c.get('status')=='coverage_skip':metrics['coverage_skips_held_records'].append(countheld)
 elif c.get('packet_admitted'):metrics['admitted_held_records'].append(countheld)
 details.append({'question_id':raw['question_id'],'worker_category':category,'program_chars':len(code) if isinstance(code,str) else None,'static_final_assignments':[x for x in ('answer','evidence','contributing_ids','operation','limitation') if x in assigned],'explicit_B_limitation':bool(e.get('limitation')),'primary_aux_limitation':bool((rev.get('primary_artifact') or {}).get('limitation')),'native_args_valid_JSON':isinstance(args,dict),'held_records':countheld})
def stat(values):
 values=sorted(x for x in values if type(x) in (int,float));n=len(values)
 return {'n':n,'min':min(values) if n else None,'median':values[n//2] if n else None,'p90':values[min(n-1,int(n*.9))] if n else None,'max':max(values) if n else None}
summary={'counts':{k:dict(v) for k,v in counts.items()},'metrics':{k:stat(v) for k,v in metrics.items()},'no_reference_scoring':True}
(D/'deep_census.json').write_text(json.dumps(summary,indent=2)+'\n');(D/'deep_rows.json').write_text(json.dumps(details,indent=2)+'\n');print(json.dumps(summary,indent=2))
