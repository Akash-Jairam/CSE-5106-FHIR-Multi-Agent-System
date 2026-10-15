# Static program/interface patterns only; no program replay and no answer/reference use.
from pathlib import Path
import ast,collections,hashlib,importlib.util,json,re
ROOT=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System');R=ROOT/'benchmarks/design-v6-20261015';O=ROOT/'output/experiments/gpt-6-luna/independent_recomputation_v6_300q';D=Path(__file__).parent;F=Path('/private/tmp/fhir-v6-conditional-integration-v3-321ju7iu/base')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();p=json.loads((R/'completion_protocol_audit.json').read_text());assert sha(R/'completion_protocol_audit.json')=='ba7a383078218c39c45ec4b485414b1adb248d14cd6e73fe83afd28bcb9d92fd'
spec=importlib.util.spec_from_file_location('v6_pattern_store',F/'scripts/audit_store.py');store=importlib.util.module_from_spec(spec);spec.loader.exec_module(store)
counts=collections.defaultdict(collections.Counter);cases=[]
for raw in json.loads((O/'ordinary_multi_agent.json').read_text()):
 assert sha(O/raw['audit_artifacts']['path'])==p['raw_files'][raw['audit_artifacts']['path']]['sha256']
 row=store.restore_result_audit(raw,O,question_id=raw['question_id'],restore_history=False);rev=row.get('revision') or {};c=rev.get('recompute') or {};e=c.get('B_execution') or {};reason=c.get('reason') or '';err=str(e.get('error') or '')
 if 'executed artifact is ineligible' in reason and e.get('limitation'):
  text=str(e['limitation']).lower();tags=[]
  for label,pattern in [('absence_or_no_match',r'\b(no |not found|not recorded|none |absent|missing|not available|unavailable|not present|could not find|cannot find)'),('uncertainty_or_ambiguity',r'\b(uncertain|ambigu|cannot determine|could not determine|unable to|insufficient|incomplete|not enough)'),('preview_or_sample',r'\b(preview|sample|example|not shown|not provided)'),('temporal_scope',r'\b(encounter|admission|discharge|time|date|first|last|after|before|recent)'),('concept_or_unit_mapping',r'\b(concept|code|match|mapping|label|unit|convert|rate)')]:
   if re.search(pattern,text):tags.append(label);counts['B_limitation_tags_overlapping'][label]+=1
  if not tags:counts['B_limitation_tags_overlapping']['other']+=1
  counts['B_limitation_evidence_presence']['nonempty' if e.get('evidence') else 'empty']+=1
  cases.append({'question_id':raw['question_id'],'category':'B_explicit_limitation','tags':tags or ['other'],'evidence_count':len(e.get('evidence') or [])})
 if '.resource_id must be a supported full ResourceType/id present in retrieved_resources' not in err:continue
 gen=[m for m in row['trace'] if m.get('role')=='assistant' and m.get('phase')=='independent_generation'];args=json.loads(gen[0]['tool_calls'][0]['function']['arguments']);tree=ast.parse(args['code'])
 # Describe only fixed interface structure. Never emit expressions, names, literals, IDs or values.
 patterns=[];assignments={}
 for n in ast.walk(tree):
  if isinstance(n,ast.Assign):
   for t in n.targets:
    if isinstance(t,ast.Name):assignments.setdefault(t.id,[]).append(n.value)
 def classify(n,depth=0):
  if isinstance(n,ast.Name) and depth<3 and len(assignments.get(n.id,[]))==1:return 'via_variable:'+classify(assignments[n.id][0],depth+1)
  const=[x.value for x in ast.walk(n) if isinstance(x,ast.Constant) and isinstance(x.value,str)]
  idaccess=any(isinstance(x,ast.Subscript) and isinstance(x.slice,ast.Constant) and x.slice.value=='id' or isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='get' and x.args and isinstance(x.args[0],ast.Constant) and x.args[0].value=='id' for x in ast.walk(n))
  typed=any(re.match(r'^(Patient|Encounter|Condition|MedicationRequest|Procedure|Observation|MedicationAdministration|Location|Specimen|Medication)/',x) for x in const)
  dynamic='resourceType' in const
  if isinstance(n,ast.Constant) and isinstance(n.value,str):return 'literal_full_type_id' if typed else 'literal_without_supported_type_prefix'
  if idaccess and not typed and not dynamic and isinstance(n,(ast.Subscript,ast.Call)):return 'bare_record_id_access'
  if typed:return 'explicit_type_prefix_expression'
  if dynamic:return 'dynamic_resourceType_expression'
  if idaccess:return 'id_access_other_expression'
  return 'other_expression'
 for n in ast.walk(tree):
  if isinstance(n,ast.Dict):
   for k,v in zip(n.keys,n.values):
    if isinstance(k,ast.Constant) and k.value=='resource_id':patterns.append(classify(v))
 counts['bad_resource_id_programs']['n']+=1;counts['bad_resource_id_static_expressions'].update(patterns or ['no_literal_resource_id_dictionary_key'])
 normalized={x.split(':')[-1] for x in patterns};counts['bad_resource_id_program_patterns'][','.join(sorted(normalized)) or 'no_literal_resource_id_dictionary_key']+=1
 # Held ID validity verifies this is not wholesale invalid input IDs, without exposing any ID.
 held=row.get('retrieved_fhir_resources') or {};valid=all(isinstance(r,dict) and isinstance(r.get('id'),str) and re.fullmatch(r'[A-Za-z0-9.-]{1,64}',r['id']) for records in held.values() if isinstance(records,list) for r in records)
 counts['bad_resource_id_cases_held_record_id_format']['all_valid' if valid else 'contains_nonconforming_input']+=1
 cases.append({'question_id':raw['question_id'],'category':'worker_citation_id_rejection','static_expression_patterns':patterns,'held_record_ids_all_valid':valid})
result={'counts':{k:dict(v) for k,v in counts.items()},'assurance':'Static structural patterns and overlapping keyword tags only; no proof of execution path, missing data or correctness.'}
(D/'provenance_patterns.json').write_text(json.dumps(result,indent=2)+'\n');(D/'private_case_categories.json').write_text(json.dumps(cases,indent=2)+'\n');print(json.dumps(result,indent=2))
