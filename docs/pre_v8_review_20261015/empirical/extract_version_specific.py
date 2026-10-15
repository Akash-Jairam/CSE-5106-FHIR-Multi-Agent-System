"""Five purposive exposed300 V4 regressions; preserves retained evidence only."""
import ast,gzip,hashlib,json,re
from pathlib import Path
REPO=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
OUT=Path(__file__).parent
ns={'json':json};module=ast.parse((OUT/'cross_version.py').read_text())
exec(compile(ast.Module(body=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='iter_rows'],type_ignores=[]),'private_iter_rows','exec'),ns)
qids={'00b98d0bf4d50497625b257e','65f468deef71db0042d17c04','461ed0b2f30a1fe2a7aca6e3','d6121ad75765bc51dceaf4c9','f517e3cce89ac464b8a1e6e1'}
root=REPO/'output/experiments/gpt-6-luna/plan_conditioned_v4_300q';cases=[]
for row in ns['iter_rows'](root/'ordinary_multi_agent.json'):
 if row['question_id'] not in qids:continue
 ref=row['audit_artifacts'];assert ref['question_id']==row['question_id'] and re.fullmatch(r'audit/[0-9a-f]{64}\.json\.gz',ref['path'])
 data=gzip.decompress((root/ref['path']).read_bytes());assert hashlib.sha256(data).hexdigest()==ref['sha256'];payload=json.loads(data)
 assert payload['_audit_meta']['question_id']==row['question_id']
 cases.append({'row':row,'audit':payload})
assert len(cases)==5
with gzip.open(OUT/'selected_version_specific_evidence.json.gz','wt') as stream:json.dump(cases,stream,allow_nan=False)
print(json.dumps({'cases':len(cases),'selection':'V3d andV7 correct,V4 wrong;purposive operation/clock/output regressions,not prevalence sample.'}))
