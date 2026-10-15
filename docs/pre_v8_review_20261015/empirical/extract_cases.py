"""Retained exposed300 evidence only; never reruns model-generated programs."""
import ast, gzip, hashlib, json, re
from pathlib import Path
REPO=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
OUT=Path(__file__).parent
ns={'json':json}
module=ast.parse((OUT/'cross_version.py').read_text())
exec(compile(ast.Module(body=[node for node in module.body if isinstance(node,ast.FunctionDef) and node.name=='iter_rows'],type_ignores=[]),'private_iter_rows','exec'),ns)
SELECTED={
 '1f2e37aaa091d50e833afad1':'admission type projection',
 'dbadc610a502fc0a6c69d44e':'admission type projection',
 '2d7404fcfff0b855a4fb2fe9':'specimen value versus test label',
 '7866a9173d7aff1bce72d3a6':'specimen value versus test label',
 'c04f16fa062bc81f8c917962':'microbiology time/source selection',
 '0a8c46b684e72300d29c18aa':'tied latest prescription set',
 '383586ac765d783e5374a6c0':'tied earliest prescription set',
 '41b9a346811b51fc23501b6f':'tied earliest route prescription set',
 'bf8aee434927ad78efbb9c37':'prescription time convention',
 'f8d725f820917d71a5301983':'minimum measurement timestamp',
 '204b55b1a8a2dbeafa89b76c':'year aggregation versus empty reference',
 'f6274c47226150acc9cef0e2':'distinct medication identity',
 '0c6cdc444ee911941bfd23f0':'relative time and medication identity',
 '2ca14c9e3ae55c6d467fa8c1':'missing medication dose source',
 '967175d0df47b1226680f9bc':'source encounter-link refusal'}
matrix=json.loads((OUT/'exposed300_matrix.json').read_text())
root=REPO/'output/experiments/gpt-6-luna/residual_trajectories_v7_300q'
cases=[]
for row in ns['iter_rows'](root/'ordinary_multi_agent.json'):
 qid=row['question_id']
 if qid not in SELECTED:continue
 ref=row['audit_artifacts'];assert ref['version']==2 and ref['question_id']==qid
 assert re.fullmatch(r'audit/[0-9a-f]{64}\.json\.gz',ref['path'])
 data=gzip.decompress((root/ref['path']).read_bytes());assert hashlib.sha256(data).hexdigest()==ref['sha256']
 payload=json.loads(data);assert payload['_audit_meta']['question_id']==qid
 snapshots=payload['revision.messages'];assert isinstance(snapshots,list)
 cases.append({'question_id':qid,'selection_rationale':SELECTED[qid],
               'question':row['question'],'reference':row['true_answer'],
               'family':matrix['questions'][qid]['family'],'template':matrix['questions'][qid]['template'],
               'all14_outcomes':matrix['questions'][qid]['outcomes'],
               'audit_sha256':ref['sha256'],'revision':row['revision'],
               'snapshots':snapshots,'trace':payload['trace']})
assert len(cases)==len(SELECTED)
with gzip.open(OUT/'selected_case_evidence.json.gz','wt') as stream:
 json.dump({'selection':'Purposive15 of37 all14-wrong exposed300 cases, not a prevalence sample. No model-generated programs executed.','cases':cases},stream,allow_nan=False)
print(json.dumps([{'qid':c['question_id'],'rationale':c['selection_rationale'],'snapshots':[{'candidate':s['candidate'],'answer':s['released']['answer'],'eligible':s['diagnostics']['eligible'],'resources':{k:len(v) for k,v in s['workspace'].items()}} for s in c['snapshots']]} for c in cases],indent=2))
