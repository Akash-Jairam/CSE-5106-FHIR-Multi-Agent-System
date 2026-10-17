"""Verify or reconstruct portable patient XZ envelopes using only Python's standard library."""
import argparse, gzip, hashlib, json, lzma, shutil, tempfile
from pathlib import Path

def sha(path):
 digest=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
 return digest.hexdigest()
def packed(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=True).encode('utf-8')
def semantic(value):return hashlib.sha256(packed(value)).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def load_index(directory,manifest):
 p=directory/manifest['question_index']['path'];require(sha(p)==manifest['question_index']['sha256'],'Question index hash differs')
 rows=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line]
 require(len(rows)==manifest['question_count'],'Index row count differs');index={r['question_id']:r for r in rows}
 require(len(index)==len(rows) and sorted(r['source_row_index'] for r in rows)==list(range(len(rows))),'Question identities/source order differ')
 return index

def records(directory,manifest):
 for archive in manifest['archives']:
  path=(directory/archive['path']).resolve();require(path.is_relative_to(directory.resolve()),'Archive path escapes package')
  require(path.stat().st_size==archive['bytes'] and sha(path)==archive['sha256'],'Compressed archive size/hash differs')
  count=0
  with lzma.open(path,'rt',encoding='utf-8',newline='') as stream:
   for count,line in enumerate(stream,1):yield archive,count,json.loads(line)
  require(count==archive['rows'],'Archive row count differs')

def verify(directory):
 directory=Path(directory).resolve();manifest=json.loads((directory/'ARCHIVE_MANIFEST.json').read_text(encoding='utf-8'))
 require(manifest['schema']=='patient_jsonl_xz_canonical_and_owned_audit_v1','Unsupported archive schema')
 index=load_index(directory,manifest);seen=set();audit_bytes=0;patients=set()
 for archive,line,envelope in records(directory,manifest):
  require(set(envelope)=={'canonical','audit_json'},'Envelope fields differ');row=envelope['canonical'];qid=row['question_id'];require(qid in index and qid not in seen,'Unknown or duplicate restored question');seen.add(qid);entry=index[qid]
  require(entry['archive']==archive['path'] and entry['archive_line']==line,'Archive location differs')
  require(row['patient_fhir_id']==entry['patient_fhir_id'] and semantic(row)==entry['row_semantic_sha256'],'Canonical parsed row differs')
  payload=envelope['audit_json'].encode('utf-8');audit_bytes+=len(payload)
  require(len(payload)==entry['audit_uncompressed_bytes'] and hashlib.sha256(payload).hexdigest()==entry['audit_uncompressed_sha256'],'Original audit payload bytes differ')
  audit=json.loads(envelope['audit_json']);require(audit.get('_audit_meta',{}).get('question_id')==qid,'Audit ownership differs')
  require(row['audit_artifacts']['sha256']==entry['audit_uncompressed_sha256'] and row['audit_artifacts']['path']==entry['source_audit_relative_path'],'Original audit reference differs');patients.add(row['patient_fhir_id'])
 require(seen==set(index) and len(patients)==manifest['patient_count'] and audit_bytes==manifest['original_owned_audit_uncompressed_bytes'],'Restored cohort totals differ')
 digest=hashlib.sha256()
 for qid in sorted(index):digest.update((qid+'\0'+index[qid]['row_semantic_sha256']+'\n').encode())
 require(digest.hexdigest()==manifest['rows_semantic_digest_sorted_by_question_id'],'Parsed row aggregate differs')
 identity=directory/manifest['source_inference_identity']['copy'];require(sha(identity)==manifest['source_inference_identity']['sha256'],'Inference identity differs')
 return manifest,index,{'status':'PORTABLE_ARCHIVE_VERIFIED','questions':len(seen),'patients':len(patients),'archives':len(manifest['archives']),'audit_payload_bytes':audit_bytes,'all_parsed_canonical_fields_and_owned_audit_bytes_match':True,'original_canonical_whitespace_or_gzip_container_bytes_claimed':False}

def reconstruct(directory,output):
 directory,output=Path(directory).resolve(),Path(output).resolve();require(not output.exists(),'Reconstruction output must be new');require(not output.is_relative_to(directory) and not directory.is_relative_to(output),'Reconstruction must be outside the archive')
 manifest,index,proof=verify(directory);output.mkdir(parents=True)
 with tempfile.TemporaryDirectory(prefix='canonical-rebuild-') as temporary:
  spool=Path(temporary)
  for archive,line,envelope in records(directory,manifest):
   row=envelope['canonical'];entry=index[row['question_id']]
   (spool/f'{entry["source_row_index"]:05d}.json').write_text(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=True),encoding='utf-8',newline='')
   target=(output/entry['source_audit_relative_path']).resolve();require(target.is_relative_to(output),'Audit path escapes reconstructed output');target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('wb') as stream:
    with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0,compresslevel=6) as compressed:compressed.write(envelope['audit_json'].encode('utf-8'))
   require(hashlib.sha256(gzip.decompress(target.read_bytes())).hexdigest()==entry['audit_uncompressed_sha256'],'Recreated audit payload differs')
  canonical_path=output/Path(manifest['source_canonical']['path']).name
  with canonical_path.open('wb') as stream:
   stream.write(b'[\n')
   for number in range(len(index)):
    if number:stream.write(b',\n')
    with (spool/f'{number:05d}.json').open('rb') as row_stream:shutil.copyfileobj(row_stream,stream)
   stream.write(b'\n]\n')
 shutil.copyfile(directory/manifest['source_inference_identity']['copy'],Path(str(canonical_path)+'.run.json'))
 receipt=proof|{'status':'PARSED_CANONICAL_AND_AUDIT_PAYLOADS_RECONSTRUCTED','canonical_path':str(canonical_path),'reconstructed_canonical_sha256':sha(canonical_path),'original_canonical_sha256':manifest['source_canonical']['sha256'],'original_row_order_preserved':True,'canonical_whitespace_and_gzip_headers_are_new':True,'source_archive_manifest_sha256':sha(directory/'ARCHIVE_MANIFEST.json')}
 (output/'RECONSTRUCTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
 return receipt

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('verify','reconstruct'));parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--output',type=Path);args=parser.parse_args()
 if args.command=='reconstruct':require(args.output is not None,'Reconstruction requires --output');result=reconstruct(args.archive,args.output)
 else:result=verify(args.archive)[2]
 print(json.dumps(result,indent=2))
