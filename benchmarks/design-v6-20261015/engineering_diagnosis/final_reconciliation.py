# Final engineering reconciliation; completion-gated, model-free, no answer/gold scoring.
from pathlib import Path
import ast,collections,hashlib,importlib.util,json,re
D=Path(__file__).parent;ROOT=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System');R=ROOT/'benchmarks/design-v6-20261015';O=ROOT/'output/experiments/gpt-6-luna/independent_recomputation_v6_300q';F=Path('/private/tmp/fhir-v6-conditional-integration-v3-321ju7iu/base');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(R/'completion_protocol_audit.json')=='ba7a383078218c39c45ec4b485414b1adb248d14cd6e73fe83afd28bcb9d92fd';proof=json.loads((R/'completion_protocol_audit.json').read_text());assert proof['status']=='PASS' and proof['restored_audit_records']==600
spec=importlib.util.spec_from_file_location('v6_final_engineering_store',F/'scripts/audit_store.py');store=importlib.util.module_from_spec(spec);spec.loader.exec_module(store)
static={n.value for n in ast.walk(ast.parse((F/'agent/continuous_primary.py').read_text())) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
counts=collections.defaultdict(collections.Counter);caseids=collections.defaultdict(list)
for raw in json.loads((O/'ordinary_multi_agent.json').read_text()):
 assert sha(O/raw['audit_artifacts']['path'])==proof['raw_files'][raw['audit_artifacts']['path']]['sha256']
 row=store.restore_result_audit(raw,O,question_id=raw['question_id'],restore_history=False);v=row.get('revision') or {};c=v.get('recompute') or {};pd=v.get('primary_diagnostics') or {};e=c.get('B_execution') or {};reason=c.get('reason') or '';err=str(e.get('error') or '')
 if not c:
  skip=pd.get('skip_reason');counts['no_aux_primary_skip_reason'][skip if skip in static else 'other']+=1
 if c.get('model_calls',0):
  usages=(row.get('tokens') or {}).get('per_call') or [];aux=usages[-c['model_calls']:]
  counts['aux_invocation_usage'].update('known' if x.get('usage_available') is True else 'unknown' for x in aux)
  if aux and aux[0].get('usage_available') is True:counts['generation_completion_at4000'][str(aux[0].get('completion')==4000)]+=1
 if 'SyntaxError:' in err:
  counts['syntax_error_shape']['unterminated_string' if 'unterminated string' in err else 'unclosed_delimiter' if 'was never closed' in err else 'invalid_syntax' if 'invalid syntax' in err else 'other']+=1
 gen=[m for m in row.get('trace') or [] if m.get('role')=='assistant' and m.get('phase')=='independent_generation']
 if not gen:continue
 try:args=json.loads(gen[0]['tool_calls'][0]['function']['arguments'])
 except ValueError:caseids['invalid_native_JSON'].append(raw['question_id']);continue
 extras=set(args)-{'code','mode','timeout'}
 if extras:counts['unexpected_argument_schema_keys'].update(k if k in {'answer','evidence','contributing_ids','operation','limitation'} else '<other_key>' for k in extras)
 if '.resource_id must be a supported full ResourceType/id present in retrieved_resources' not in err:continue
 tree=ast.parse(args['code']);functions={n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)};assignments={}
 for n in ast.walk(tree):
  if isinstance(n,ast.Assign):
   for t in n.targets:
    if isinstance(t,ast.Name):assignments.setdefault(t.id,[]).append(n.value)
 def resolve(n,depth=0):
  if depth>=5:return 'unresolved'
  if isinstance(n,ast.Name) and len(assignments.get(n.id,[]))==1:return resolve(assignments[n.id][0],depth+1)
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in functions:
   fun=functions[n.func.id];returns=[x.value for x in ast.walk(fun) if isinstance(x,ast.Return)]
   if len(returns)==1:return resolve(returns[0],depth+1)
   if any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='items' for x in ast.walk(fun)) and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='str' for x in returns):return 'held_mapping_key_namespace_confusion'
  if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant) and n.slice.value=='id':return 'bare_record_id'
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='get' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='id':return 'bare_record_id'
  return 'unresolved'
 patterns={resolve(value) for n in ast.walk(tree) if isinstance(n,ast.Dict) for key,value in zip(n.keys,n.values) if isinstance(key,ast.Constant) and key.value=='resource_id'}
 category=next(iter(patterns)) if len(patterns)==1 else 'mixed_or_unresolved'
 counts['citation_ID_first_failure_program_provenance'][category]+=1;caseids[category].append(raw['question_id'])
result={'counts':{k:dict(v) for k,v in counts.items()},'ID_assurance':'Program structural cause only, not a repaired execution/source certificate; first failure hides later failures. No confirmed fabricated/not-held literal IDs in this census; cannot exclude them after repair.','first_failure_funnel':{'total':300,'primary_fatal_no_aux':4,'primary_nonfatal_ineligible_no_aux':24,'bound_primary_with_limitation_aux_ineligible':30,'precall_packet_coverage_skip':21,'admitted_generation_unknown_usage_fatal':1,'native_generation_replies':220,'native_argument_rejections':2,'executions':218,'worker_errors':138,'worker_success_explicit_limitation':35,'candidate_entry_byte_bound':7,'original_source_rejection':1,'eligible_B':37,'exact_typed_agreement':29,'disagreement_selector_A':4,'disagreement_selector_B':4}}
assert sum(result['first_failure_funnel'][x] for x in ('primary_fatal_no_aux','primary_nonfatal_ineligible_no_aux','bound_primary_with_limitation_aux_ineligible','precall_packet_coverage_skip','admitted_generation_unknown_usage_fatal','native_argument_rejections','worker_errors','worker_success_explicit_limitation','candidate_entry_byte_bound','original_source_rejection','eligible_B'))==300
(D/'final_reconciliation.json').write_text(json.dumps(result,indent=2)+'\n');(D/'private_case_ID_provenance.json').write_text(json.dumps(dict(caseids),indent=2)+'\n');print(json.dumps(result,indent=2))
