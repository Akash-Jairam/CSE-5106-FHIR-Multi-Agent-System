#!/usr/bin/env python3
"""Synthetic gate-only adversarial review; never invokes the full audit."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCE = Path('/private/tmp/cache-provenance-single-final-preparation')
REPO = Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
spec = importlib.util.spec_from_file_location('synthetic_completion_fixture', BASE/'test_completion_gate.py')
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+'\n')


def run_case(name, mutate):
    fixture = fixture_module.SyntheticGate()
    fixture.setUp()
    try:
        original = {name: sha(path) for name,path in {
            'proof': fixture.proof_path,
            'canonical': fixture.run/'full_single_agent.json',
            'acceptance': fixture.run/'acceptance.json',
            'state': fixture.run/'supervisor_state.json',
            'manifest': fixture.run/'run_manifest.json',
        }.items()}
        details = mutate(fixture) or {}
        changed = {name: sha(path) != original[name] for name,path in {
            'proof': fixture.proof_path,
            'canonical': fixture.run/'full_single_agent.json',
            'acceptance': fixture.run/'acceptance.json',
            'state': fixture.run/'supervisor_state.json',
            'manifest': fixture.run/'run_manifest.json',
        }.items()}
        try:
            result = fixture.call()
            outcome = {'outcome': 'ACCEPTED', 'reported_rounds': sorted(result['rounds'])}
        except Exception as error:
            outcome = {'outcome': 'REFUSED', 'exception': type(error).__name__, 'reason': str(error)}
        assert changed['proof'] is False and changed['manifest'] is False
        assert fixture.output.exists() is False
        return {'case': name, 'changes_after_saved_proof': changed,
                'proof_reused_byte_identical': True, 'fixture_save_called_after_mutation': False,
                'output_created': False, **details, **outcome}
    finally:
        fixture.tearDown()


def mutate_canonical(fixture):
    (fixture.run/'full_single_agent.json').write_bytes(b'{"different_canonical_bytes": true}\n')
    return {'mutation': 'Replace canonical bytes after saving monitor proof; no canonical parse performed by gate'}


def mutate_ledger_choice(fixture):
    ledger = deepcopy(fixture.ledger)
    selection = ledger['arms']['single_agent']['q0000']
    selection['runtime_failures_before'] = 999
    selection['selection'] = 'intrinsic_terminal_error'
    write(fixture.run/'acceptance.json', ledger)
    return {'mutation': 'Change first-clean selection reason and prior-runtime count, retaining cohort/protocol/counts'}


def mutate_ledger_round(fixture):
    ledger = deepcopy(fixture.ledger)
    selection = ledger['arms']['single_agent']['q0000']
    selection['round'] = 99
    selection['attempt_file'] = 'attempts/single_agent/round-0099.json'
    write(fixture.run/'acceptance.json', ledger)
    return {'mutation': 'Point accepted q0000 at round 99 absent from saved round receipt'}


def mutate_state(fixture):
    state = deepcopy(fixture.state)
    state['finished_at'] = '2099-01-01T00:00:00+00:00'
    write(fixture.run/'supervisor_state.json', state)
    return {'mutation': 'Change completion timestamp after proof without changing required completion flags'}


def add_later_sealed_round(fixture):
    folder = fixture.run/'attempts/single_agent'
    output = folder/'round-0002.json'
    identity = Path(str(output)+'.run.json')
    batch = folder/'round-0002.batch.json'
    seal = folder/'round-0002.batch.seal.json'
    shard = folder/'round-0002.csv'
    folder.mkdir(parents=True)
    shard.write_text('question_id,patient_fhir_id\nq0000,p00\n')
    write(output, [{'question_id':'q0000','patient_fhir_id':'p00','latency':1.0,'error':None,
                    'tokens':{'total':2,'prompt':1,'completion':1,'cached_prompt':0,'llm_calls':1,
                              'per_call':[{'prompt':1,'completion':1,'cached':0,'usage_available':True}]}}])
    write(identity, fixture.manifest['run_identities']['single_agent'])
    write(batch, {'arm':'single_agent','round':2,'question_ids':['q0000'],
                  'input':str(shard),'input_sha256':sha(shard),'output':str(output),
                  'run_identity':fixture.manifest['run_identities']['single_agent']})
    write(seal, {'batch_sha256':sha(batch),'result_sha256':sha(output),'identity_sha256':sha(identity)})
    return {'mutation':'Add later sealed round 2 scheduling already accepted q0000 while retaining saved round-1-only proof',
            'later_round_files': [path.name for path in (batch,seal,output,identity,shard)],
            'later_round_seal_exact_hashes_verified': json.loads(seal.read_bytes()) ==
                {'batch_sha256':sha(batch),'result_sha256':sha(output),'identity_sha256':sha(identity)}}


def false_clean_control(fixture):
    ledger = deepcopy(fixture.ledger)
    ledger['arms']['single_agent']['q0000']['runtime_clean'] = False
    write(fixture.run/'acceptance.json', ledger)
    return {'mutation':'Negative control: explicit runtime_clean=False'}


def active_control(fixture):
    state = deepcopy(fixture.state)
    state['active'] = {'single_agent': {'round':2}}
    write(fixture.run/'supervisor_state.json', state)
    return {'mutation':'Negative control: active selected worker'}


def extract_call_validator(source_path):
    tree = ast.parse(source_path.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    canonical_loop = next(node for node in main.body if isinstance(node, ast.For) and
                          isinstance(node.target,ast.Name) and node.target.id == 'original')
    call_loop = next(node for node in canonical_loop.body if isinstance(node,ast.For) and
                     isinstance(node.target,ast.Name) and node.target.id == 'call')
    code = ast.Module(body=[ast.FunctionDef(name='validate',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='calls')],
                       kwonlyargs=[],kw_defaults=[],defaults=[]),body=[deepcopy(call_loop)],decorator_list=[])],type_ignores=[])
    ast.fix_missing_locations(code)
    def require(value, message):
        if not value:
            raise ValueError(message)
    namespace = {'require':require}
    exec(compile(code,str(source_path),'exec'),namespace)
    return namespace['validate']


def usage_validation():
    new = extract_call_validator(BASE/'audit_complete_single_cache.py')
    old = extract_call_validator(BASE/'original_complete_v3_audit.py')
    calls = [{'prompt':0,'completion':0,'cached':0,'usage_available':False}]
    result = {}
    for name,validate in [('adapted_bool_gate',new),('original_true_only_gate',old)]:
        try:
            validate(calls)
            result[name] = {'outcome':'ACCEPTED'}
        except ValueError as error:
            result[name] = {'outcome':'REFUSED','reason':str(error)}
    tree = ast.parse((REPO/'scripts/v3_full_run.py').read_text())
    classify = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='classify_attempt')
    usage_if = next(node for node in classify.body if isinstance(node,ast.If) and
                    any(isinstance(child,ast.Constant) and child.value=='unverified_provider_usage' for child in ast.walk(node)))
    function = ast.FunctionDef(name='usage_branch',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='tokens')],
                        kwonlyargs=[],kw_defaults=[],defaults=[]),body=[deepcopy(usage_if)],decorator_list=[])
    module = ast.Module(body=[function],type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module,'extracted_classify_attempt_usage_branch','exec'),namespace)
    result['runner_exact_unknown_usage_branch'] = namespace['usage_branch']({'per_call':calls})
    result['synthetic_call'] = calls[0]
    result['full_audit_invoked'] = False
    result['method'] = 'Execute only AST-extracted canonical per-call validation loops and runner unknown-usage branch; no module main/runtime imports'
    return result


cases = [run_case('unchanged_saved_proof_control',lambda fixture:None),
         run_case('canonical_bytes_changed_after_proof',mutate_canonical),
         run_case('ledger_first_clean_selection_changed_after_proof',mutate_ledger_choice),
         run_case('ledger_selection_points_to_unproved_round',mutate_ledger_round),
         run_case('completion_state_changed_after_proof',mutate_state),
         run_case('later_exactly_sealed_round_added_after_proof',add_later_sealed_round),
         run_case('negative_runtime_clean_false',false_clean_control),
         run_case('negative_active_worker',active_control)]
evidence = {
    'review':'Independent synthetic gate-only stale-proof and unknown-usage review',
    'reviewed_snapshots_sha256':{name:sha(BASE/name) for name in ('completed_gate.py','test_completion_gate.py','audit_complete_single_cache.py','original_complete_v3_audit.py')},
    'original_sources_unchanged_since_snapshot':{name:sha(BASE/name)==sha(SOURCE/name) for name in ('completed_gate.py','test_completion_gate.py','audit_complete_single_cache.py','original_complete_v3_audit.py')},
    'cases':cases,
    'usage_available_review':usage_validation(),
    'blocking_findings':[
        'Saved completion receipt is not bound to canonical/acceptance/state hashes or exact current batch inventory; within-audit hash stability cannot establish freshness relative to first-clean verification.',
        'A later sealed round is ignored because gate trusts only receipt round list and never inventories current rounds.',
        'Adapted canonical call validator accepts usage_available=False, although frozen runner treats it as attention; require True for selected calls or prove this through fresh exact bound replay before capture.'
    ],
    'scope':{'private_synthetic_fixtures_only':True,'fixture_save_called_after_mutation':False,
             'full_audit_run':False,'live_run_read':False,'live_edits':False,'answers_or_correctness_read':False,
             'config_or_credentials_read':False,'scoring_or_provider_calls':False}
}
write(BASE/'adversarial_evidence.json', evidence)
print(json.dumps({'evidence':str(BASE/'adversarial_evidence.json'),'cases':[{k:item[k] for k in ('case','outcome')} for item in cases],
                  'usage_available_review':evidence['usage_available_review'],'sources_unchanged':evidence['original_sources_unchanged_since_snapshot']},indent=2))
