from pathlib import Path
import ast,collections,hashlib,importlib.util,json,re
ROOT=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
RECORD=ROOT/'benchmarks/design-v6-20261015';OUT=ROOT/'output/experiments/gpt-6-luna/independent_recomputation_v6_300q'
DEST=Path(__file__).parent;FROZEN=Path('/private/tmp/fhir-v6-conditional-integration-v3-321ju7iu/base')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while chunk:=f.read(1024*1024):h.update(chunk)
 return h.hexdigest()
def load(path):return json.loads(path.read_text())
assert sha(RECORD/'completion_protocol_audit.json')=='ba7a383078218c39c45ec4b485414b1adb248d14cd6e73fe83afd28bcb9d92fd'
assert sha(RECORD/'results.json')=='678ed113da6e7b785d7e5ef2ce12dfb4db0c6ae7c11584aeff4a91316542f37c'
proof=load(RECORD/'completion_protocol_audit.json')
assert proof['status']=='PASS' and proof['mode']=='completion' and proof['caller_asserted_owned_jobs_terminal'] is True
assert proof['restored_audit_records']==proof['expected_first_attempts']==600
assert sha(FROZEN/'agent/independent_recompute_aux.py')=='f6659d5de56841068666f50c83d9bf3673e180b21c97fe08cedbfca4f1a8fdd6'
for rel,entry in proof['raw_files'].items():
 path=OUT/rel
 assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(OUT.resolve())
 assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
store_path=FROZEN/'scripts/audit_store.py';assert sha(store_path)=='9ca06bd7310ae8f245d80e1274b09e78f4b78153bae599dfaa88a9acfa030caf'
spec=importlib.util.spec_from_file_location('fixed_v6_model_free_audit_store',store_path);store=importlib.util.module_from_spec(spec);spec.loader.exec_module(store)
rawpath=OUT/'ordinary_multi_agent.json';rawrows=load(rawpath);assert len(rawrows)==300 and len({r['question_id'] for r in rawrows})==300
# Exact static source failure text only; never emit arbitrary model/tool strings.
source_text=(FROZEN/'agent/independent_recompute_aux.py').read_text()
static={n.value for n in ast.walk(ast.parse(source_text)) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
counts=collections.defaultdict(collections.Counter);rows=[]
for raw in rawrows:
 row=store.restore_result_audit(raw,OUT,question_id=raw['question_id'],restore_history=False)
 # Input reference answers are never accessed, scored, printed, or retained.
 rev=row.get('revision') or {};rec=rev.get('recompute') or {};trace=row.get('trace') or []
 reason=rec.get('reason') or ''
 staticreason=reason.removeprefix('auxiliary unavailable: ValueError: ')
 reasonkey=staticreason if staticreason in static else 'unclassified:'+hashlib.sha256(reason.encode()).hexdigest()[:12]
 if not rec:reasonkey='no_recompute_diagnostic'
 counts['status'][str(rec.get('status','absent'))]+=1;counts['reason'][reasonkey]+=1
 counts['primary_eligibility'][str((rev.get('primary_diagnostics') or {}).get('eligible','absent'))]+=1
 counts['fatal'][str(rec.get('fatal','absent'))]+=1
 generation=[m for m in trace if m.get('role')=='assistant' and m.get('phase')=='independent_generation']
 tools=[m for m in trace if m.get('role')=='tool' and m.get('phase')=='independent_recompute']
 selection=[m for m in trace if m.get('role')=='assistant' and m.get('phase')=='independent_selection']
 counts['generation_response_count'][len(generation)]+=1;counts['aux_tool_reply_count'][len(tools)]+=1
 counts['selection_response_count'][len(selection)]+=1
 calls=(generation[0].get('tool_calls') or []) if len(generation)==1 else []
 native='no_response' if not generation else 'zero_tool_prose' if not calls and generation[0].get('content') else 'zero_tool_empty' if not calls else 'one_native' if len(calls)==1 else 'multiple_native'
 counts['native_generation'][native]+=1
 execution=rec.get('B_execution') or {};err=execution.get('error');errtype='none'
 if err:
  if execution.get('error_type')=='final_contract':
   prefixes=['Missing final metadata','Final evidence must be','Final evidence[','Final contributing_ids','Every final cited resource','Final operation must','Final limitation must','A nonempty final answer','Computation metadata','Final answer must','The answer and metadata','Code executed successfully']
   errtype='contract:'+next((p for p in prefixes if str(err).startswith(p)),'other')
  else:
   match=re.match(r'([A-Za-z_][A-Za-z_0-9]*(?:Error|Exception)):',str(err));errtype=match[1] if match else 'other_execution_error'
 counts['execution_error'][errtype]+=1
 item={'question_id':raw['question_id'],'status':rec.get('status'),'reason':reasonkey,'primary_eligible':(rev.get('primary_diagnostics') or {}).get('eligible'),'fatal':rec.get('fatal'),'packet_admitted':rec.get('packet_admitted'),'B_eligible':rec.get('B_eligible',False),'answer_agreement':rec.get('answer_agreement'),'selected':rec.get('selected'),'model_calls':rec.get('model_calls',0),'tool_calls':rec.get('tool_calls',0),'native_generation':native,'execution_error_category':errtype,'generation_response_present':len(generation)==1,'selection_response_present':len(selection)==1,'execution_present':'B_execution' in rec,'source_check_present':'B_source_check' in rec,'source_check_issue_count':len((rec.get('B_source_check') or {}).get('issues') or []),'source_check_valid_citations':(rec.get('B_source_check') or {}).get('n_valid'),'source_check_results':len((rec.get('B_source_check') or {}).get('results') or []),'B_entry_bytes':rec.get('B_entry_bytes'),'generation_prompt_tokens':rec.get('generation_prompt_tokens'),'selector_fixed_prompt_tokens':rec.get('selector_fixed_prompt_tokens'),'two_call_token_reservation':rec.get('two_call_token_reservation'),'primary_tokens':(rev.get('primary_diagnostics') or {}).get('tokens')}
 rows.append(item)
summary={'rows':len(rows),'counts':{k:dict(v) for k,v in counts.items()},'totals':{k:sum(bool(x.get(k)) for x in rows) for k in ('packet_admitted','B_eligible','answer_agreement','execution_present','source_check_present')},'model_calls':sum(x['model_calls'] for x in rows),'tool_calls':sum(x['tool_calls'] for x in rows),'bindings':{'completion_report_sha256':sha(RECORD/'completion_protocol_audit.json'),'official_report_sha256_only_no_scores_loaded':sha(RECORD/'results.json'),'raw_team_sha256':sha(rawpath),'frozen_aux_sha256':sha(FROZEN/'agent/independent_recompute_aux.py'),'audit_store_sha256':sha(store_path),'scorer_imports':0,'actual_reference_answer_accesses':0}}
(DEST/'engineering_census.json').write_text(json.dumps(summary,indent=2)+'\n');(DEST/'engineering_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(summary,indent=2))
