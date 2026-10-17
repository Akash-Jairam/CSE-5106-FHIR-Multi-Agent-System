"""Require a fresh trusted bound final-monitor proof before canonical parsing."""
import csv
import hashlib
import json
from pathlib import Path

from bound_completion import N, ARM, load_bound_proof, require, sha, snapshot

FINGERPRINT = '65d2bbfc7da05f97e646acb22d20f0367d7f7a4e1a2a7c8b9c244fd77ef8d409'


def completion_gate(root,run,proof_path,binding_path,output):
    root,run,proof_path,binding_path,output = map(lambda p:Path(p).resolve(),(root,run,proof_path,binding_path,output))
    require(not output.exists() and not output.is_relative_to(run) and not run.is_relative_to(output),
            'Output must be NEW and outside the live run')
    proof,current,rounds,fixed = load_bound_proof(root,run,proof_path,binding_path)
    def read(path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest()==fixed.get(str(path)),'Bound completed metadata changed before parsing')
        return json.loads(raw)
    ledger,manifest = read(run/'acceptance.json'),read(run/'run_manifest.json')
    selections = ledger['arms'][ARM]
    canonical = run/'full_single_agent.json'
    identity = read(Path(str(canonical)+'.run.json'))
    require(identity==manifest['run_identities'][ARM] and identity.get('code_hash')==FINGERPRINT
            and identity.get('agent_strategy')==ARM,'Canonical full inference identity differs')
    reader_source = root/'scripts/full_benchmark.py'
    require(fixed.get(str(reader_source))==manifest['source_sha256']['scripts/full_benchmark.py'],
            'Frozen streaming reader differs')
    source = Path(manifest['input']['path']).resolve()
    require(sha(source)==manifest['input']['sha256']==fixed.get(str(source)),'Bound frozen authoritative input differs')
    patients = {}
    with source.open(newline='',encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            qid,patient = row.get('question_id'),row.get('patient_fhir_id')
            require(qid and patient and qid not in patients,'Input identity missing or duplicated')
            patients[qid] = patient
    require(len(patients)==N and set(selections)==set(patients),'Full selected/input question cohort differs')
    for qid,entry in selections.items():
        require(entry.get('round') in rounds,'Selected ledger points to an unproved round')
        actual = next(x for x in current['sealed_batches'] if x['round']==entry['round'])
        require((run/entry.get('attempt_file','')).resolve()==Path(actual['result']),
                'Selected ledger points outside its bound round result')
    require(snapshot(root,run)==current,'Bound completed bytes/inventory changed during metadata gate')
    return {'root':root,'run':run,'output':output,'proof_path':proof_path,'binding_path':binding_path,
            'proof':proof,'ledger':ledger,'manifest':manifest,'selections':selections,'rounds':rounds,
            'canonical':canonical,'reader_source':reader_source,'patients':patients,'fixed':fixed,
            'bound_snapshot':current}
