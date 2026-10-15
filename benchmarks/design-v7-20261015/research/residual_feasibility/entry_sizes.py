from pathlib import Path
import hashlib,importlib.util,json,math
ROOT=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System');R=ROOT/'benchmarks/design-v6-20261015';O=ROOT/'output/experiments/gpt-6-luna/independent_recomputation_v6_300q';F=Path('/private/tmp/fhir-v6-conditional-integration-v3-321ju7iu/base');D=Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();proof=json.loads((R/'completion_protocol_audit.json').read_text());assert sha(R/'completion_protocol_audit.json')=='ba7a383078218c39c45ec4b485414b1adb248d14cd6e73fe83afd28bcb9d92fd'
storefile=F/'scripts/audit_store.py';assert sha(storefile)=='9ca06bd7310ae8f245d80e1274b09e78f4b78153bae599dfaa88a9acfa030caf';sp=importlib.util.spec_from_file_location('v6_residual_size_audit_store',storefile);store=importlib.util.module_from_spec(sp);sp.loader.exec_module(store)
rows=[];FIELDS=('answer','evidence','operation','contributing_ids','limitation')
def js(v):return json.dumps(v,sort_keys=True,ensure_ascii=True,allow_nan=False)
for raw in json.loads((O/'ordinary_multi_agent.json').read_text()):
 path=O/raw['audit_artifacts']['path'];assert sha(path)==proof['raw_files'][raw['audit_artifacts']['path']]['sha256'];row=store.restore_result_audit(raw,O,question_id=raw['question_id'],restore_history=False)
 rev=row.get('revision') or {};pd=rev.get('primary_diagnostics') or {};a=rev.get('primary_artifact');bound=pd.get('eligible') is True and isinstance(a,dict);group='fatal' if pd.get('fatal') else 'source_bound' if bound else 'nonfatal_ineligible'
 # Values are serialized only for byte length; never printed, compared to answers or retained.
 item={'question_id':raw['question_id'],'group':group,'released_rendered_bytes':len(js(rev.get('primary')).encode())}
 if bound:
  code=a['code'];artifact={k:a[k] for k in FIELDS};n=len(artifact['evidence']);summary={'issues':[],'n_valid':n,'citation_count':n,'all_results_valid':True,'all_warnings_empty':True,'raw_check_sha256':'0'*64,'assurance':'source_fields_only_not_semantics_or_search_completeness'}
  entry={'program':code,'artifact':artifact,'source_check':summary,'program_sha256':hashlib.sha256(code.encode()).hexdigest(),'workspace_sha256':a['workspace_sha256']}
  item.update(program_chars=len(code),program_utf8_bytes=len(code.encode()),five_field_artifact_ASCII_JSON_bytes=len(js(artifact).encode()),V6_shaped_A_entry_ASCII_JSON_bytes=len(js(entry).encode()),evidence_citations=n,contributing_ID_count=len(artifact['contributing_ids']),workspace_record_count=sum(len(v) for v in (row.get('retrieved_fhir_resources') or {}).values() if isinstance(v,list)))
 rows.append(item)
def stats(v):
 v=sorted(v);n=len(v)
 return {'n':n,**{name:v[max(0,math.ceil(q*n)-1)] if n else None for name,q in [('min',0),('p25',.25),('median',.5),('p75',.75),('p90',.9),('p95',.95),('max',1)]}}
keys=('released_rendered_bytes','program_chars','program_utf8_bytes','five_field_artifact_ASCII_JSON_bytes','V6_shaped_A_entry_ASCII_JSON_bytes','evidence_citations','contributing_ID_count','workspace_record_count')
result={'method':'Exact serialized byte sizes only; no generated/reference answer values inspected or retained. Team recorded175k/27 A only; cannot claim full-cap A future distribution. A-entry size reproduces existing V6 structural schema with fixed-width placeholder source SHA; hashes do not affect length. No question/context or selector-system tokens included.', 'groups':{g:{'n':sum(x['group']==g for x in rows),'metrics':{k:stats([r[k] for r in rows if r['group']==g and k in r]) for k in keys}} for g in ('source_bound','nonfatal_ineligible','fatal')},'bound_A_entry_threshold_counts':{f'over_{b}':sum(r.get('V6_shaped_A_entry_ASCII_JSON_bytes',0)>b for r in rows) for b in (8000,16000,32000,64000)}}
(D/'entry_sizes.json').write_text(json.dumps(result,indent=2)+'\n');(D/'private_entry_sizes_rows.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(result,indent=2))
