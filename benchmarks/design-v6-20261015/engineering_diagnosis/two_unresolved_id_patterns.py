# Prints only the two unresolved citation-ID program structures with every string/numeric value redacted.
from pathlib import Path
import ast,importlib.util,json
ROOT=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System');O=ROOT/'output/experiments/gpt-6-luna/independent_recomputation_v6_300q';F=Path('/private/tmp/fhir-v6-conditional-integration-v3-321ju7iu/base');D=Path(__file__).parent
rows=json.loads((D/'private_case_categories.json').read_text());wanted={r['question_id'] for r in rows if r['category']=='worker_citation_id_rejection' and r['static_expression_patterns']==['other_expression'] or r['category']=='worker_citation_id_rejection' and set(r['static_expression_patterns'])=={'other_expression'}}
spec=importlib.util.spec_from_file_location('two_pattern_store',F/'scripts/audit_store.py');store=importlib.util.module_from_spec(spec);spec.loader.exec_module(store)
allowed={'resource_id','field_path','value','id','resourceType','retrieved_resources','retrieved_fhir_resources','contributing_ids','evidence','operation','limitation','answer'}
class Redact(ast.NodeTransformer):
 def visit_Constant(self,n):
  if isinstance(n.value,str):return ast.copy_location(ast.Constant(n.value if n.value in allowed else '<REDACTED_STRING>'),n)
  if type(n.value) in (int,float):return ast.copy_location(ast.Constant('<REDACTED_NUMBER>'),n)
  return n
for raw in json.loads((O/'ordinary_multi_agent.json').read_text()):
 if raw['question_id'] not in wanted:continue
 row=store.restore_result_audit(raw,O,question_id=raw['question_id'],restore_history=False)
 gen=next(m for m in row['trace'] if m.get('role')=='assistant' and m.get('phase')=='independent_generation');code=json.loads(gen['tool_calls'][0]['function']['arguments'])['code'];tree=ast.parse(code)
 print('PRIVATE_CASE_ID',raw['question_id']);print(ast.unparse(Redact().visit(tree)))
