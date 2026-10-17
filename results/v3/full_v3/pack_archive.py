"""Portable, lossless parsed-row + original-audit-payload packaging. No inference imports."""
import argparse, gzip, hashlib, json, lzma, shutil, tempfile, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
 return h.hexdigest()
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=True).encode('utf-8')
def semantic(value):return hashlib.sha256(canonical(value)).hexdigest()
def need(ok,why):
 if not ok:raise ValueError(why)
def iter_array(path):
 decoder=json.JSONDecoder();buffer='';position=0;ended=False
 with Path(path).open('r',encoding='utf-8',newline='') as source:
  def read_more():
   nonlocal buffer,position,ended
   block=source.read(1024*1024)
   if not block:ended=True;return False
   buffer=buffer[position:]+block;position=0;return True
  def space():
   nonlocal position
   while True:
    while position<len(buffer) and buffer[position].isspace():position+=1
    if position<len(buffer) or ended:return
    read_more()
  read_more();space();need(position<len(buffer) and buffer[position]=='[','Canonical source is not a JSON array');position+=1
  while True:
   space()
   if position<len(buffer) and buffer[position]==']':position+=1;break
   while True:
    try:value,end=decoder.raw_decode(buffer,position);position=end;break
    except json.JSONDecodeError:
     if ended:raise
     read_more()
   yield value
   space();need(position<len(buffer),'Unterminated canonical array')
   if buffer[position]==',':position+=1;continue
   need(buffer[position]==']','Invalid canonical array delimiter');position+=1;break
  space()
  while not ended:read_more();space()
  need(position==len(buffer),'Trailing content after canonical array')

def package(args):
 source=args.canonical.resolve();run=source.parent;output=args.output.resolve()
 need(not output.exists(),'Output must be new');output.mkdir(parents=True)
 chunks=output/'patient_chunks';chunks.mkdir()
 source_hash=sha(source);ledger_bytes=args.ledger.read_bytes();ledger=json.loads(ledger_bytes);choices=ledger['arms'][args.arm]
 filters=[{'id':lzma.FILTER_LZMA2,'preset':args.preset,'dict_size':args.dict_mib*1024*1024}]
 groups=defaultdict(list);seen=set();original_order=[];index=[];archives=[];started=time.monotonic()
 with tempfile.TemporaryDirectory(prefix='parsed-canonical-spool-',dir=args.temp_dir) as temp:
  temporary=Path(temp)
  for n,row in enumerate(iter_array(source)):
   qid=row.get('question_id');patient=row.get('patient_fhir_id');need(isinstance(qid,str) and qid and isinstance(patient,str) and patient,'Missing question/patient identity');need(qid not in seen,'Duplicate canonical question');seen.add(qid)
   need(not row.get('attempt_history'),'Canonical history requires additional payload handling')
   ref=row.get('audit_artifacts');need(isinstance(ref,dict) and ref.get('version')==2 and ref.get('question_id')==qid,'Owned version2 audit reference required')
   p=(run/ref['path']).resolve();need(p.is_relative_to((run/'audit').resolve()) and p.suffixes[-2:]==['.json','.gz'],'Unexpected canonical audit path')
   spool=temporary/f'{n:05d}.json';spool.write_text(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=True),encoding='utf-8',newline='')
   groups[patient].append({'question_id':qid,'patient_fhir_id':patient,'source_row_index':n,'row_semantic_sha256':semantic(row),'spool':spool,'audit_path':p,'audit_ref':ref});original_order.append(qid)
  need(len(seen)==args.expected and seen==set(choices),'Canonical/accepted cohort mismatch')
  print(json.dumps({'event':'CANONICAL_SPOOLED','rows':len(seen),'patients':len(groups),'source_bytes':source.stat().st_size,'elapsed_seconds':round(time.monotonic()-started,2)}),flush=True)
  for patient_number,patient in enumerate(sorted(groups),1):
   part=0;writer=None;current=None;part_raw=0;part_rows=0
   def close_part():
    nonlocal writer,current,part_raw,part_rows
    if writer is None:return
    writer.close();need(current.stat().st_size<args.max_archive_mb*1000*1000,'Compressed chunk exceeds archive size limit')
    archives.append({'path':str(current.relative_to(output)),'sha256':sha(current),'bytes':current.stat().st_size,'envelope_utf8_bytes':part_raw,'rows':part_rows,'patient_ordinal':patient_number,'part':part})
    writer=None
   for record in sorted(groups[patient],key=lambda x:x['question_id']):
    row=json.loads(record['spool'].read_bytes());compressed=record['audit_path'].read_bytes();payload=gzip.decompress(compressed);ref=record['audit_ref']
    need(hashlib.sha256(payload).hexdigest()==ref['sha256'],'Audit uncompressed hash mismatch');audit_text=payload.decode('utf-8');audit_meta=json.loads(audit_text).get('_audit_meta',{})
    need(audit_meta.get('question_id')==record['question_id'],'Audit payload owner mismatch')
    envelope=json.dumps({'canonical':row,'audit_json':audit_text},ensure_ascii=False,separators=(',',':'),allow_nan=True).encode('utf-8')+b'\n'
    if writer is not None and part_raw+len(envelope)>args.raw_chunk_mib*1024*1024:close_part()
    if writer is None:
     part+=1;part_raw=part_rows=0;current=chunks/f'patient-{patient_number:03d}-part-{part:03d}.jsonl.xz';writer=lzma.LZMAFile(current,'wb',format=lzma.FORMAT_XZ,check=lzma.CHECK_CRC64,filters=filters)
    writer.write(envelope);part_raw+=len(envelope);part_rows+=1
    index.append({key:record[key] for key in ('question_id','patient_fhir_id','source_row_index','row_semantic_sha256')}|{'archive':str(current.relative_to(output)),'archive_line':part_rows,'source_audit_relative_path':ref['path'],'source_audit_compressed_sha256':hashlib.sha256(compressed).hexdigest(),'source_audit_compressed_bytes':len(compressed),'audit_uncompressed_sha256':ref['sha256'],'audit_uncompressed_bytes':len(payload),'accepted_selection':choices[record['question_id']]})
    record['spool'].unlink()
   close_part()
   print(json.dumps({'event':'PATIENT_PACKED','patients_done':patient_number,'patients_total':len(groups),'rows_done':len(index),'compressed_bytes':sum(x['bytes'] for x in archives),'elapsed_seconds':round(time.monotonic()-started,2)}),flush=True)
  need(len(index)==args.expected,'Packed row count mismatch')
 by_qid={x['question_id']:x for x in index};verified=set();payload_bytes=0
 for archive in archives:
  p=output/archive['path'];need(sha(p)==archive['sha256'],'Archive changed before verification');line_count=0
  with lzma.open(p,'rt',encoding='utf-8',newline='') as stream:
   for line_count,line in enumerate(stream,1):
    envelope=json.loads(line);need(set(envelope)=={'canonical','audit_json'},'Envelope fields differ');row=envelope['canonical'];qid=row['question_id'];need(qid not in verified,'Duplicate restored question');verified.add(qid);entry=by_qid[qid]
    need(entry['archive']==archive['path'] and entry['archive_line']==line_count,'Archive location index differs');need(semantic(row)==entry['row_semantic_sha256'],'Restored canonical row differs')
    payload=envelope['audit_json'].encode('utf-8');need(hashlib.sha256(payload).hexdigest()==entry['audit_uncompressed_sha256'] and len(payload)==entry['audit_uncompressed_bytes'],'Restored audit payload bytes differ');payload_bytes+=len(payload)
  need(line_count==archive['rows'] and sha(p)==archive['sha256'],'Archive count/hash changed')
 need(verified==seen,'Restored full cohort differs');need(sha(source)==source_hash,'Original canonical source changed')
 for entry in index:need(sha(run/entry['source_audit_relative_path'])==entry['source_audit_compressed_sha256'],'Original source audit bytes changed')
 index_path=output/'question_index.jsonl'
 with index_path.open('w',encoding='utf-8',newline='\n') as stream:
  for entry in sorted(index,key=lambda x:x['source_row_index']):stream.write(json.dumps(entry,sort_keys=True,separators=(',',':'))+'\n')
 identity_path=Path(str(source)+'.run.json');identity_copy=output/'source_inference_identity.json';shutil.copyfile(identity_path,identity_copy)
 aggregate=hashlib.sha256()
 for qid in sorted(by_qid):aggregate.update((qid+'\0'+by_qid[qid]['row_semantic_sha256']+'\n').encode())
 manifest={'schema':'patient_jsonl_xz_canonical_and_owned_audit_v1','status':'LOSSLESS_PARSED_CANONICAL_AND_AUDIT_PAYLOAD_VERIFIED','created_at_utc':datetime.now(timezone.utc).isoformat(),'source_canonical':{'path':str(source),'sha256':source_hash,'bytes':source.stat().st_size,'rows':len(index)},'source_selection_ledger':{'path':str(args.ledger.resolve()),'sha256':hashlib.sha256(ledger_bytes).hexdigest(),'arm':args.arm,'arm_choices_semantic_sha256':semantic(choices)},'source_inference_identity':{'path':str(identity_path),'sha256':sha(identity_path),'copy':identity_copy.name},'patient_count':len(groups),'question_count':len(index),'archive_count':len(archives),'archives':archives,'archive_total_bytes':sum(x['bytes'] for x in archives),'original_owned_audit_compressed_bytes_referenced':sum(x['source_audit_compressed_bytes'] for x in index),'original_owned_audit_uncompressed_bytes':payload_bytes,'question_index':{'path':index_path.name,'sha256':sha(index_path),'bytes':index_path.stat().st_size},'rows_semantic_digest_sorted_by_question_id':aggregate.hexdigest(),'compression':{'format':'XZ','check':'CRC64','preset':args.preset,'dictionary_bytes':args.dict_mib*1024*1024,'target_max_envelope_bytes_per_chunk':args.raw_chunk_mib*1024*1024,'compressed_file_limit_bytes_exclusive':args.max_archive_mb*1000*1000,'sort':'patient_fhir_id then question_id'},'losslessness':{'all_parsed_original_canonical_fields':True,'original_canonical_row_order_recoverable_via_source_row_index':True,'owned_audit_utf8_payload_bytes':True,'original_canonical_whitespace_reproduced':False,'original_gzip_container_bytes_headers_reproduced':False,'source_compressed_and_uncompressed_hashes_retained':True,'no_discarded_attempts_in_primary_accepted_archive':True},'verification':{'restored_question_count':len(verified),'every_parsed_row_semantic_hash_matches':True,'every_audit_utf8_payload_bytes_hash_matches':True,'source_canonical_and_every_source_gzip_hash_reobserved_unchanged':True,'every_archive_below_file_limit':True},'packager_source_sha256':sha(Path(__file__)),'elapsed_seconds':round(time.monotonic()-started,2)}
 (output/'ARCHIVE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'event':'ARCHIVE_VERIFIED','rows':len(index),'patients':len(groups),'archives':len(archives),'archive_bytes':manifest['archive_total_bytes'],'audit_payload_bytes':payload_bytes,'manifest_sha256':sha(output/'ARCHIVE_MANIFEST.json'),'elapsed_seconds':manifest['elapsed_seconds']}),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--canonical',type=Path,required=True);parser.add_argument('--ledger',type=Path,required=True);parser.add_argument('--arm',required=True);parser.add_argument('--expected',type=int,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--temp-dir',type=Path,default=Path('/private/tmp'));parser.add_argument('--preset',type=int,default=4);parser.add_argument('--dict-mib',type=int,default=64);parser.add_argument('--raw-chunk-mib',type=int,default=64);parser.add_argument('--max-archive-mb',type=int,default=90);package(parser.parse_args())
