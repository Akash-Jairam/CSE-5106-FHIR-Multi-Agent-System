"""Read-only metadata gate for a completed 2,931-question selected baseline."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

N = 2931
ARM = 'single_agent'
FINGERPRINT = '65d2bbfc7da05f97e646acb22d20f0367d7f7a4e1a2a7c8b9c244fd77ef8d409'


def require(test,message):
    if not test:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


def completion_gate(root,run,proof_path,output):
    root,run,proof_path,output = map(lambda p:Path(p).resolve(),(root,run,proof_path,output))
    require(not output.exists() and not output.is_relative_to(run) and not run.is_relative_to(output),
            'Output must be NEW and outside the live run')
    # Count/scope gate is intentionally first: an incomplete run is refused
    # before canonical parsing, proof imports, diagnostic reads or output writes.
    fixed = {}
    def read(path):
        raw = path.read_bytes()
        fixed[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    ledger_path,state_path,manifest_path = [run/name for name in ('acceptance.json','supervisor_state.json','run_manifest.json')]
    ledger = read(ledger_path)
    selections = (ledger.get('arms') or {}).get(ARM) or {}
    require(len(selections)==N and all(isinstance(qid,str) and qid for qid in selections),
            'Full baseline requires exactly 2931 selected outcomes; current cohort is incomplete')
    require(ledger.get('protocol')=='v3-first-runtime-clean-terminal' and ledger.get('selection')=='first_runtime_clean_terminal'
            and ledger.get('complete') is False,'Selected first-clean protocol or original benchmark scope differs')
    for entry in selections.values():
        require(isinstance(entry,dict) and entry.get('runtime_clean') is True
                and entry.get('selection') in ('runtime_clean_terminal','intrinsic_terminal_error'),
                'A selected outcome lacks first-clean proof')
    state,manifest = read(state_path),read(manifest_path)
    require(state.get('phase')=='complete_selected_arms' and state.get('selected_scope_complete') is True
            and state.get('benchmark_complete') is False and state.get('selected_arms')==[ARM]
            and state.get('pending_scoped_export') is True and state.get('analysis_status')=='pending_scoped_export'
            and state.get('finished_at') and not state.get('active') and not state.get('attention')
            and state.get('accepted',{}).get(ARM)==N,'Completed selected-arm supervisor proof required')
    require(manifest.get('questions_per_arm')==N and manifest.get('selected_arms')==[ARM], 'Full selected baseline manifest required')
    proof = read(proof_path)
    require(proof.get('status')=='SELECTED_EXECUTION_COMPLETE_ANALYSIS_PENDING'
            and proof.get('selected_arms')==[ARM] and proof.get('selected_accepted')==N
            and proof.get('questions_per_selected_arm')==N and proof.get('selected_scope_complete') is True
            and proof.get('original_three_arm_benchmark_complete') is False
            and proof.get('canonical_first_clean_verified') is True and proof.get('accepted_reruns')==0
            and not proof.get('parked_pending_question_ids'),'Saved completed first-clean monitor receipt required')
    monitor = root/'benchmarks/single-agent-full-run-20261017/monitor_single_agent.py'
    require(sha(monitor)==proof.get('monitor_sha256') and sha(manifest_path)==proof.get('manifest_sha256'),
            'Completion receipt monitor/manifest binding differs')
    fixed[str(monitor)] = sha(monitor)
    counts = (ledger.get('counts') or {}).get(ARM) or {}
    require(proof.get('replayed_counts')==counts and counts.get('raw_attempts',0)-counts.get('runtime_failures',0)==N,
            'Completed first-clean attempt counts differ')
    rounds = {}
    for item in proof.get('sealed_batches') or []:
        number = item.get('round')
        require(type(number) is int and number>0 and number not in rounds,'Completion receipt sealed round missing/duplicated')
        rounds[number] = item
    require(rounds and set(rounds)==set(range(1,max(rounds)+1)),'Completion receipt sealed rounds have gaps')
    require(sum(item.get('classifications',{}).get('accepted',0) for item in rounds.values())==N,
            'Completed sealed accepted count differs')
    canonical = run/'full_single_agent.json'
    fixed[str(canonical)] = sha(canonical)
    identity_path = Path(str(canonical)+'.run.json')
    identity = read(identity_path)
    require(identity==manifest['run_identities'][ARM] and identity.get('code_hash')==FINGERPRINT
            and identity.get('agent_strategy')==ARM,'Canonical full inference identity differs')
    reader_source = root/'scripts/full_benchmark.py'
    require(sha(reader_source)==manifest['source_sha256']['scripts/full_benchmark.py'],'Frozen streaming reader differs')
    fixed[str(reader_source)] = sha(reader_source)
    source = Path(manifest['input']['path'])
    require(sha(source)==manifest['input']['sha256'],'Frozen authoritative input differs')
    fixed[str(source)] = manifest['input']['sha256']
    patients = {}
    with source.open(newline='',encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            qid,patient = row.get('question_id'),row.get('patient_fhir_id')
            require(qid and patient and qid not in patients,'Input identity missing or duplicated')
            patients[qid] = patient
    require(len(patients)==N and set(selections)==set(patients),'Full selected/input question cohort differs')
    for path,digest in fixed.items():
        require(sha(path)==digest,'Completed metadata changed during gate')
    return {'root':root,'run':run,'output':output,'proof_path':proof_path,'proof':proof,'ledger':ledger,
            'manifest':manifest,'selections':selections,'rounds':rounds,'canonical':canonical,
            'reader_source':reader_source,'patients':patients,'fixed':fixed}
