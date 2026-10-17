"""Losslessly archive nonselected sealed attempts, without scoring or classification."""
import argparse,gzip,hashlib,importlib.util,json,lzma,tempfile,time
from pathlib import Path
spec=importlib.util.spec_from_file_location('private_archive_primitives','/private/tmp/package_patient_archive.py');u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
def package(args):
 run=args.run.resolve();out=args.output.resolve();u.need(not out.exists(),'Output must be new');out.mkdir(parents=True);(out/'chunks').mkdir()
 ledgerpath=run/'acceptance.json';ledger=json.loads(ledgerpath.read_bytes());choices=ledger['arms'][args.arm];sources={};selected=0;total=0;records=[];started=time.monotonic()
 def bind(p):
  sources[str(p.relative_to(run))]={'sha256':u.sha(p),'bytes':p.stat().st_size}
 with tempfile.TemporaryDirectory(prefix='nonselected-attempt-spool-') as temp:
  temp=Path(temp)
  for batchpath in sorted((run/'attempts'/args.arm).glob('round-*.batch.json')):
   batch=json.loads(batchpath.read_bytes());sealpath=batchpath.with_suffix('.seal.json');u.need(sealpath.exists(),'All source attempts must be sealed');seal=json.loads(sealpath.read_bytes());result=Path(batch['output']);identity=Path(str(result)+'.run.json');inputpath=Path(batch['input'])
   u.need(u.sha(batchpath)==seal['batch_sha256'] and u.sha(result)==seal['result_sha256'] and u.sha(identity)==seal['identity_sha256'] and u.sha(inputpath)==batch['input_sha256'],'Source attempt binding mismatch')
   for p in (batchpath,sealpath,result,identity,inputpath):bind(p)
   relative=str(result.relative_to(run))
   for source_index,row in enumerate(u.iter_array(result)):
    total+=1;qid=row['question_id']
    if choices.get(qid,{}).get('attempt_file')==relative:selected+=1;continue
    spool=temp/f'{len(records):04d}.json';spool.write_text(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=True),encoding='utf-8',newline='')
    records.append({'attempt_id':f'{batch["round"]:04d}:{qid}','question_id':qid,'patient_fhir_id':row['patient_fhir_id'],'round':batch['round'],'source_result_relative_path':relative,'source_row_index':source_index,'parsed_attempt_semantic_sha256':u.semantic(row),'spool':spool})
  u.need(len(records)==args.expected and selected==len(choices) and total==selected+args.expected,'Selected/nonselected cohort count mismatch')
  records.sort(key=lambda x:(x['patient_fhir_id'],x['question_id'],x['round']));index=[];archives=[];writer=None;current=None;raw_bytes=0;lines=0;part=0
  def close():
   nonlocal writer
   if writer is None:return
   writer.close();u.need(current.stat().st_size<90000000,'Archive exceeds90MB limit');archives.append({'path':str(current.relative_to(out)),'sha256':u.sha(current),'bytes':current.stat().st_size,'rows':lines,'envelope_utf8_bytes':raw_bytes});writer=None
  for record in records:
   row=json.loads(record['spool'].read_bytes());ref=row.get('audit_artifacts');audit_text=None;auditmeta={}
   if ref:
    p=(run/record['source_result_relative_path']).parent/ref['path'];raw=p.read_bytes();payload=gzip.decompress(raw);u.need(hashlib.sha256(payload).hexdigest()==ref['sha256'],'Audit hash mismatch');audit_text=payload.decode('utf-8');u.need(json.loads(audit_text).get('_audit_meta',{}).get('question_id')==record['question_id'],'Audit owner mismatch');bind(p)
    auditmeta={'source_audit_relative_path':str(p.relative_to(run)),'source_audit_compressed_sha256':hashlib.sha256(raw).hexdigest(),'source_audit_compressed_bytes':len(raw),'audit_uncompressed_sha256':ref['sha256'],'audit_uncompressed_bytes':len(payload)}
   origin={key:record[key] for key in ('attempt_id','question_id','patient_fhir_id','round','source_result_relative_path','source_row_index')}
   line=json.dumps({'origin':origin,'attempt':row,'audit_json':audit_text},ensure_ascii=False,separators=(',',':'),allow_nan=True).encode('utf-8')+b'\n'
   if writer is not None and raw_bytes+len(line)>64*1024*1024:close()
   if writer is None:
    part+=1;current=out/'chunks'/f'excluded-{part:03d}.jsonl.xz';writer=lzma.LZMAFile(current,'wb',format=lzma.FORMAT_XZ,check=lzma.CHECK_CRC64,filters=[{'id':lzma.FILTER_LZMA2,'preset':4,'dict_size':64*1024*1024}]);raw_bytes=lines=0
   writer.write(line);raw_bytes+=len(line);lines+=1;index.append(origin|{'parsed_attempt_semantic_sha256':record['parsed_attempt_semantic_sha256'],'archive':str(current.relative_to(out)),'archive_line':lines}|auditmeta);record['spool'].unlink()
  close();by_id={r['attempt_id']:r for r in index};verified=set()
  for archive in archives:
   u.need(u.sha(out/archive['path'])==archive['sha256'],'Archive hash changed')
   with lzma.open(out/archive['path'],'rt',encoding='utf-8') as stream:
    for n,line in enumerate(stream,1):
     envelope=json.loads(line);aid=envelope['origin']['attempt_id'];r=by_id[aid];u.need(aid not in verified and r['archive']==archive['path'] and r['archive_line']==n,'Attempt archive location/uniqueness mismatch');verified.add(aid);u.need(u.semantic(envelope['attempt'])==r['parsed_attempt_semantic_sha256'],'Parsed attempt differs')
     payload=envelope['audit_json'];u.need((payload is None and 'audit_uncompressed_sha256'not in r) or (isinstance(payload,str) and hashlib.sha256(payload.encode('utf-8')).hexdigest()==r.get('audit_uncompressed_sha256')),'Audit payload differs')
  u.need(len(verified)==args.expected,'Readback count mismatch')
  for relative,entry in sources.items():u.need(u.sha(run/relative)==entry['sha256'],'Original source changed')
  ip=out/'attempt_index.jsonl';ip.write_text(''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in index))
  manifest={'schema':'nonselected_sealed_attempts_jsonl_xz_v1','status':'LOSSLESS_PARSED_ATTEMPTS_AND_AUDIT_PAYLOADS_VERIFIED','arm':args.arm,'nonselected_attempts':args.expected,'selected_source_rows_omitted_from_this_separate_archive':selected,'total_source_rows':total,'selection_ledger_sha256':u.sha(ledgerpath),'arm_choice_semantic_sha256':u.semantic(choices),'source_run':str(run),'source_file_inventory':sources,'archives':archives,'archive_total_bytes':sum(x['bytes'] for x in archives),'attempt_index':{'path':ip.name,'sha256':u.sha(ip),'bytes':ip.stat().st_size},'verification':{'every_original_parsed_attempt_and_owned_payload_restored':True,'every_source_byte_hash_reobserved_unchanged':True,'no_scoring_classification_or_provider_calls':True},'canonical_whitespace_and_original_gzip_headers_not_recreated':True,'packager_source_sha256':u.sha(Path(__file__)),'elapsed_seconds':round(time.monotonic()-started,2)}
  (out/'ARCHIVE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
  (out/'README.txt').write_text('Internal nonselected sealed-attempt raw archive. These81source rows are separate from primary accepted V3 data. No scoring/classification was rerun. Envelopes retain original parsed attempt, origin metadata and original owned decompressed audit text; index binds source compressed/uncompressed hashes. Canonical whitespace/gzip containers are not recreated. Complete XZ/payload/row readback and all original source hashes were verified. Original selection ledger and source bindings are recorded in ARCHIVE_MANIFEST.json.\n')
  print(json.dumps({'status':manifest['status'],'attempts':args.expected,'archives':len(archives),'archive_bytes':manifest['archive_total_bytes'],'manifest_sha256':u.sha(out/'ARCHIVE_MANIFEST.json'),'elapsed_seconds':manifest['elapsed_seconds']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--expected',type=int,required=True);p.add_argument('--output',type=Path,required=True);package(p.parse_args())
