"""Stable metadata/byte bindings around the unchanged reviewed final monitor."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

N, ARM = 2931, 'single_agent'
MONITOR_REL = 'benchmarks/single-agent-full-run-20261017/monitor_single_agent.py'
MONITOR_SHA = 'daa88848d68f0dbd9fbe586ecb4cea3fdd65bb8d574770f0bb42c943a88e9047'
CAPTURE_METHOD = 'reviewed_final_monitor_stable_bytes_and_exact_inventory_v1'


def require(test,message):
    if not test:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


def snapshot(root,run):
    """No canonical parsing. Incomplete count/scope is rejected before hashing it."""
    root,run = Path(root).resolve(),Path(run).resolve()
    files = {}
    def remember(path,optional=False):
        path = Path(path).resolve()
        if not path.exists():
            require(optional,'Required completed metadata file absent')
            files[str(path)] = None
            return None
        require(path.is_file(),'Completed metadata is not a regular file')
        detail = {'sha256':sha(path),'bytes':path.stat().st_size}
        files[str(path)] = detail
        return detail['sha256']
    def read(path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        files[str(path)] = {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
        return json.loads(raw)
    ledger = read(run/'acceptance.json')
    selections = (ledger.get('arms') or {}).get(ARM) or {}
    require(len(selections)==N and all(isinstance(qid,str) and qid for qid in selections),
            'Full baseline requires exactly 2931 selected outcomes; current cohort is incomplete')
    require(ledger.get('protocol')=='v3-first-runtime-clean-terminal' and ledger.get('selection')=='first_runtime_clean_terminal'
            and ledger.get('complete') is False,'Selected first-clean protocol or original benchmark scope differs')
    require(all(isinstance(x,dict) and x.get('runtime_clean') is True
                and x.get('selection') in ('runtime_clean_terminal','intrinsic_terminal_error') for x in selections.values()),
            'A selected outcome lacks first-clean proof')
    state,manifest = read(run/'supervisor_state.json'),read(run/'run_manifest.json')
    require(state.get('phase')=='complete_selected_arms' and state.get('selected_scope_complete') is True
            and state.get('benchmark_complete') is False and state.get('selected_arms')==[ARM]
            and state.get('pending_scoped_export') is True and state.get('analysis_status')=='pending_scoped_export'
            and state.get('finished_at') and not state.get('active') and not state.get('attention')
            and state.get('accepted',{}).get(ARM)==N,'Completed selected-arm supervisor proof required')
    require(manifest.get('questions_per_arm')==N and manifest.get('selected_arms')==[ARM], 'Full selected baseline manifest required')
    counts = (ledger.get('counts') or {}).get(ARM) or {}
    require(counts.get('raw_attempts',0)-counts.get('runtime_failures',0)==N,'Completed first-clean attempt counts differ')
    monitor = root/MONITOR_REL
    require(remember(monitor)==MONITOR_SHA,'Reviewed final monitor source changed')
    remember(run/'full_single_agent.json')
    remember(run/'full_single_agent.json.run.json')
    # Public source pins are byte-bound around verification without executing them.
    for relative,expected in (manifest.get('source_sha256') or {}).items():
        path = (root/relative).resolve()
        require(path.is_relative_to(root) and (path.suffix=='.py' or path.name.startswith('requirements')),
                'Unexpected public source pin path')
        require(remember(path)==expected,'Pinned verification/runtime source changed')
    source = Path(manifest['input']['path']).resolve()
    require(remember(source)==manifest['input']['sha256'],'Frozen authoritative input differs')
    folder = run/'attempts'/ARM
    all_round_files = sorted(path for path in folder.glob('round-*') if path.is_file())
    for path in all_round_files:
        require(path.resolve().is_relative_to(folder.resolve()),'Round file escapes selected arm')
        remember(path)
    batches = sorted(folder.glob('round-*.batch.json'))
    seals = sorted(folder.glob('round-*.batch.seal.json'))
    require(batches and {p.with_suffix('.seal.json') for p in batches}==set(seals),
            'Actual selected batch/seal inventory is incomplete or contains extra seals')
    entries = []
    for batch_path in batches:
        match = re.fullmatch(r'round-(\d{4})\.batch\.json',batch_path.name)
        require(match is not None,'Selected batch filename changed')
        number = int(match[1])
        batch,seal = read(batch_path),read(batch_path.with_suffix('.seal.json'))
        require(batch.get('arm')==ARM and batch.get('round')==number,'Selected batch identity differs')
        result = Path(batch['output']).resolve()
        shard = Path(batch['input']).resolve()
        require(result.is_relative_to(folder.resolve()) and shard.is_relative_to(folder.resolve()),'Selected result/shard escapes arm')
        result_hash = remember(result,optional=True)
        identity_hash = remember(Path(str(result)+'.run.json'),optional=True)
        require(seal.get('batch_sha256')==remember(batch_path) and seal.get('result_sha256')==result_hash
                and seal.get('identity_sha256')==identity_hash,'Selected batch seal no longer matches bytes')
        require(remember(shard)==batch['input_sha256'],'Selected shard changed')
        entries.append({'round':number,'batch':str(batch_path.resolve()),'seal':str(batch_path.with_suffix('.seal.json').resolve()),
                        'result':str(result),'identity':str(Path(str(result)+'.run.json')),'shard':str(shard),
                        'result_sha256':result_hash,'seal_sha256':files[str(batch_path.with_suffix('.seal.json').resolve())]['sha256']})
    numbers = [x['round'] for x in entries]
    require(numbers==list(range(1,max(numbers)+1)),'Actual sealed selected rounds have gaps or duplicates')
    return {'root':str(root),'run':str(run),'selected_count':N,'counts':counts,'files':dict(sorted(files.items())),
            'all_round_files':[str(p.resolve()) for p in all_round_files],'sealed_batches':entries}


def validate_proof(proof,snap):
    require(proof.get('status')=='SELECTED_EXECUTION_COMPLETE_ANALYSIS_PENDING'
            and proof.get('selected_arms')==[ARM] and proof.get('selected_accepted')==N
            and proof.get('questions_per_selected_arm')==N and proof.get('selected_scope_complete') is True
            and proof.get('original_three_arm_benchmark_complete') is False
            and proof.get('canonical_first_clean_verified') is True and proof.get('accepted_reruns')==0
            and not proof.get('parked_pending_question_ids'),'Saved completed first-clean monitor receipt required')
    require(proof.get('monitor_sha256')==MONITOR_SHA
            and proof.get('manifest_sha256')==snap['files'][str(Path(snap['run'])/'run_manifest.json')]['sha256']
            and proof.get('replayed_counts')==snap['counts'],'Final monitor proof source/count bindings differ')
    rounds = {}
    for item in proof.get('sealed_batches') or []:
        require(type(item.get('round')) is int and item['round'] not in rounds,'Duplicate/malformed proof round')
        rounds[item['round']] = item
    actual = {x['round']:x for x in snap['sealed_batches']}
    require(set(rounds)==set(actual),'Saved final proof differs from exact current sealed round inventory')
    for number,item in rounds.items():
        require(item.get('result_sha256')==actual[number]['result_sha256'] and item.get('seal_sha256')==actual[number]['seal_sha256'],
                'Final proof round byte bindings differ')
    require(sum(x.get('classifications',{}).get('accepted',0) for x in rounds.values())==N,'Final proof accepted count differs')
    return rounds


def load_bound_proof(root,run,proof_path,binding_path):
    # Snapshot prechecks the live exact count before opening a canonical or a proof.
    current = snapshot(root,run)
    proof_path,binding_path = Path(proof_path).resolve(),Path(binding_path).resolve()
    raw,binding_raw = proof_path.read_bytes(),binding_path.read_bytes()
    proof,binding = json.loads(raw),json.loads(binding_raw)
    here = Path(__file__).resolve().parent
    require(binding.get('schema_version')==1 and binding.get('capture_method')==CAPTURE_METHOD
            and binding.get('proof_file')==proof_path.name and proof_path.parent==binding_path.parent
            and binding.get('proof_sha256')==hashlib.sha256(raw).hexdigest(),'Bound final proof digest/schema differs')
    require(binding.get('capture_wrapper_sha256')==sha(here/'capture_final_proof.py')
            and binding.get('binding_module_sha256')==sha(Path(__file__)),'Reviewed capture/binding helper source changed')
    require(binding.get('before')==binding.get('after')==current,'Stale final proof: current byte/inventory bindings differ')
    rounds = validate_proof(proof,current)
    fixed = {path:detail['sha256'] for path,detail in current['files'].items() if detail is not None}
    fixed[str(proof_path)] = hashlib.sha256(raw).hexdigest()
    fixed[str(binding_path)] = hashlib.sha256(binding_raw).hexdigest()
    fixed[str(here/'capture_final_proof.py')] = sha(here/'capture_final_proof.py')
    fixed[str(Path(__file__).resolve())] = sha(Path(__file__))
    return proof,current,rounds,fixed
