"""Prospective shared-primary W/R/C mechanism study, with offline selection.

select/plan use metadata only. execute is an explicitly launched inference job.
analyze requires complete terminal saved outcomes and uses their references only.
No correctness is computed during generation. No project/provider imports occur
until execute; the scorer is imported only inside postterminal analysis.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / 'manifests/v8_mechanism_120_20261015.json'
SEED = 5108
MODES = ('A', 'W', 'R', 'C')
QUOTAS = {'lookup_counting': 30, 'temporal_selection': 30,
          'numerical_aggregation': 30, 'duration': 10,
          'paired_comparison': 10, 'paired_difference': 10}
POLICY = {'protocol': 'counterexample-investigator-mechanism-v8',
          'primary': {'tokens': 200000, 'tools': 30, 'sdk_attempts': 30},
          'aggregate': {'tokens': 200000, 'tools': 30, 'sdk_attempts': 30},
          'auxiliary': {'tokens': 60000, 'tools': 6, 'sdk_attempts': 6, 'probes': 2},
          'primary_contrast': 'R - A',
          'advance_gate': {'O_min': 6, 'net_min': 4, 'harm_plus_fatal_max': 2,
                           'corrective_patients_min': 3, 'corrective_operations_min': 2},
          'operation_structure_definition': 'Six metadata operation_group values used for prospective quotas; never posthoc case labels or runtime routing',
          'cache': 'cold_private_A_then_identical_physical_copies_no_hardlinks',
          'workers': 1, 'retries': 0,
          'scope': 'Broader development; no protected or adaptive300 IDs; not independent patient/template confirmation'}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def _rank(seed, *values):
    return hashlib.sha256(canonical([seed, *values])).hexdigest()


def _patient_round_robin(candidates, quota, seed, stratum):
    grouped = defaultdict(list)
    for row in candidates:
        grouped[row['patient_fhir_id']].append(row)
    patients = sorted(grouped, key=lambda patient: _rank(seed, stratum, 'patient', patient))
    for patient in patients:
        grouped[patient].sort(key=lambda row: _rank(seed, stratum, 'question', row['question_id']))
    if sum(map(len, grouped.values())) < quota:
        raise ValueError('Insufficient metadata candidates for ' + stratum)
    chosen = []
    while len(chosen) < quota:
        for patient in patients:
            if grouped[patient]:
                chosen.append(grouped[patient].pop(0))
                if len(chosen) == quota:
                    break
    return chosen


def select_manifest(question_metadata, family_metadata, split, exposure, seed=SEED):
    """Pure ID/patient/family selection; inputs must not contain QA/reference data."""
    metadata = {row['question_id']: row['patient_fhir_id'] for row in question_metadata}
    if len(metadata) != len(question_metadata):
        raise ValueError('Duplicate question metadata')
    family = {row['question_id']: row for row in family_metadata}
    if len(family) != len(family_metadata) or set(family) != set(metadata):
        raise ValueError('Family/question metadata cohort differs')
    strata = exposure['splits']
    partitions = [set(strata[name]['question_ids']) for name in
                  ('development_patients', 'pilot_patients', 'adaptive_questions', 'question_confirmation')]
    if set.union(*partitions) != set(metadata) or sum(map(len, partitions)) != len(metadata):
        raise ValueError('Exposure strata must partition the complete metadata IDs exactly')
    if partitions[2] != set(split['splits']['held_out']['question_ids']):
        raise ValueError('Adaptive fixed300 identity differs')
    permitted_patients = {row['patient_fhir_id'] for name in ('dev', 'pilot')
                          for row in split['splits'][name]['patients']}
    permitted = partitions[0] | partitions[1]
    if {qid for qid, patient in metadata.items() if patient in permitted_patients} != permitted:
        raise ValueError('Reserved-patient pool does not match the frozen exposure strata')
    pools = defaultdict(list)
    for qid in sorted(permitted):
        row = family[qid]
        group = row['excluded_reason'] if row['family'] == 'out_of_scope' else row['family']
        if group in QUOTAS:
            pools[group].append({'question_id': qid, 'patient_fhir_id': metadata[qid],
                                 'metadata_family': row['family'], 'operation_group': group,
                                 'template_id': row['template_id']})
    selected = {group: _patient_round_robin(pools[group], quota, seed, group)
                for group, quota in QUOTAS.items()}
    broad = {group: selected[group] for group in ('lookup_counting', 'temporal_selection', 'numerical_aggregation')}
    broad['other'] = [selected[group][index] for index in range(10)
                      for group in ('duration', 'paired_comparison', 'paired_difference')]
    orders = list(itertools.permutations(('W', 'R', 'C')))
    cases = []
    block = 0
    for round_index in range(5):
        for group in ('lookup_counting', 'temporal_selection', 'numerical_aggregation', 'other'):
            block += 1
            block_orders = sorted(orders, key=lambda order: _rank(seed, 'arm-order', block, list(order)))
            for index, row in enumerate(broad[group][round_index * 6:(round_index + 1) * 6]):
                cases.append({**row, 'case_index': len(cases) + 1, 'time_block': block,
                              'study_family': group, 'followup_order': list(block_orders[index])})
    ids = [row['question_id'] for row in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('Study must contain unique120 IDs')
    if len(ids) != 120 or set(ids) & (partitions[2] | partitions[3]):
        raise ValueError('Study count or protected/fixed300 separation failed')
    patients = len({row['patient_fhir_id'] for row in cases})
    if patients < 20:
        raise ValueError('Metadata selection did not cover the required20 patients')
    return {'schema_version': 1, 'status': 'PROSPECTIVE_METADATA_ONLY_NOT_RUN',
            'seed': seed, 'selection': 'SHA256-ranked patient round-robin within fixed operation quotas; no QA/reference/SQL selection',
            'order': '20 six-case time blocks, cycling four broad strata; all six W/R/C permutations once per block',
            'policy': deepcopy(POLICY), 'questions': 120, 'patients': patients,
            'family_counts': dict(Counter(row['study_family'] for row in cases)),
            'operation_counts': dict(Counter(row['operation_group'] for row in cases)),
            'followup_order_counts': dict(Counter(''.join(row['followup_order']) for row in cases)),
            'excluded_fixed300': len(partitions[2]), 'excluded_confirmation': len(partitions[3]),
            'cases': cases}


def read_selection_inputs(dataset, families, split, exposure):
    # Explicit usecols prevents question/reference/SQL materialization. No
    # val_dict or answer-kind field is used to choose cases.
    import csv
    import pandas as pd
    metadata = pd.read_csv(dataset, usecols=['question_id', 'patient_fhir_id'], dtype=str).to_dict('records')
    with Path(families).open(newline='') as stream:
        family = [{key: row[key] for key in ('question_id', 'family', 'excluded_reason', 'template_id')}
                  for row in csv.DictReader(stream)]
    return metadata, family, load(split), load(exposure)


def validate_manifest(manifest):
    if manifest.get('policy') != POLICY or manifest.get('questions') != 120:
        raise ValueError('Study policy differs from the prospective fixed protocol')
    cases = manifest.get('cases', [])
    if len(cases) != 120 or len({row['question_id'] for row in cases}) != 120:
        raise ValueError('Study must contain all120 unique cases')
    if Counter(row['operation_group'] for row in cases) != Counter(QUOTAS):
        raise ValueError('Operation quotas differ')
    if len({row['patient_fhir_id'] for row in cases}) < 20:
        raise ValueError('Study patient coverage differs')
    for block in range(1, 21):
        subset = [row for row in cases if row['time_block'] == block]
        if len(subset) != 6 or {tuple(row['followup_order']) for row in subset} != set(itertools.permutations(('W', 'R', 'C'))):
            raise ValueError('Followup orders are not balanced within each time block')
    return cases


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.write('\n')
    os.replace(temporary, path)


def plan(manifest, output, model):
    cases = validate_manifest(manifest)
    return {'status': 'PLAN_ONLY_NO_INFERENCE', 'model': model, 'output_directory': str(Path(output).resolve()),
            'questions': len(cases), 'primary_physical_runs': 120, 'followup_physical_runs': 360,
            'saved_policy_outcomes': 480, 'workers': 1, 'policy': POLICY,
            'cache': 'A cold percase; same frozen postA bytes physically copied into W/R/C; sibling writes isolated',
            'physical_A_billing_count': 1, 'conceptual_A_usage_per_policy': 1,
            'automatic_accuracy_analysis': False, 'retry_or_resume': False}


def choose_development_case(manifest, requested='auto'):
    """Metadata-only interface case outside the fixed120,300 and1397."""
    paths = manifest['source_sha256']
    inputs = read_selection_inputs(ROOT / 'final_dataset/questions_answers_sql_fhir.csv',
                                   ROOT / 'manifests/question_families.csv', ROOT / 'manifests/split.json',
                                   ROOT / 'manifests/final_evaluation_strata_20261015.json')
    metadata, families, split, exposure = inputs
    reserved = set(exposure['splits']['development_patients']['question_ids']) | set(exposure['splits']['pilot_patients']['question_ids'])
    candidates = reserved - {row['question_id'] for row in manifest['cases']}
    if requested == 'auto':
        requested = min(candidates, key=lambda qid: _rank(manifest['seed'], 'interface-outside120', qid))
    if requested not in candidates:
        raise ValueError('Interface case must be reserved and outside the frozen120/300/1397')
    for relative, expected_hash in paths.items():
        if digest(ROOT / relative) != expected_hash:
            raise ValueError('Metadata changed before development-case selection')
    row = next(row for row in families if row['question_id'] == requested)
    patient = next(row['patient_fhir_id'] for row in metadata if row['question_id'] == requested)
    operation = row['excluded_reason'] if row['family'] == 'out_of_scope' else row['family']
    return {'question_id': requested, 'patient_fhir_id': patient, 'metadata_family': row['family'],
            'operation_group': operation, 'template_id': row['template_id'],
            'case_index': 1, 'time_block': 0, 'study_family': operation if operation in QUOTAS and not row['family'] == 'out_of_scope' else 'other',
            'followup_order': ['W', 'R', 'C'], 'development_interface_only': True}


def verify_execution_freeze(args, manifest):
    """Existing public record/source/runtime bindings, before any runtime import."""
    binding = load(args.provider_binding)
    if set(binding) != {'provider_profile_sha256', 'adapter_sha256', 'bootstrap_sha256'}:
        raise ValueError('Public provider binding must contain exactly three hashes')
    if binding['adapter_sha256'] != digest(ROOT / '.local-setup/azure_adapter.py') or binding['bootstrap_sha256'] != digest(ROOT / '.local-setup/sitecustomize.py'):
        raise ValueError('Actual public adapter/bootstrap differs from runtime binding')
    required = {str(path.relative_to(ROOT)) for directory in ('agent', 'tools', 'utils')
                for path in (ROOT / directory).rglob('*.py')}
    required |= {'fhir_client.py', 'scripts/v8_mechanism_study.py', 'scripts/v8_study_cache.py',
                 'scripts/launch_v8_study.py'}
    if args.freeze_record:
        directory = Path(args.freeze_record).resolve()
        freeze = load(directory / 'manifest.json')
        if freeze.get('model') != args.model:
            raise ValueError('Requested model differs from the formal provider freeze')
        if freeze.get('study_manifest_sha256') != digest(args.manifest):
            raise ValueError('Formal freeze does not pin this exact120 manifest')
        sources = freeze.get('source_sha256')
        if not isinstance(sources, dict) or not required <= set(sources):
            raise ValueError('Formal source inventory omits current inference dependencies')
        if load(directory / 'expected_public_binding.json') != binding:
            raise ValueError('Actual public provider binding differs from formal freeze')
        runtime = load(directory / 'runtime_adapter_manifest.json')['source_sha256']
        record_sha = digest(directory / 'manifest.json')
    elif args.development_case:
        if args.model != load(ROOT / 'benchmarks/design-v7-20261015/manifest.json')['model']:
            raise ValueError('Development model differs from the unchanged provider profile')
        sources = {name: digest(ROOT / name) for name in required}
        runtime = load(ROOT / 'benchmarks/design-v7-20261015/runtime_adapter_manifest.json')['source_sha256']
        if load(ROOT / 'benchmarks/design-v7-20261015/expected_public_binding.json') != binding:
            raise ValueError('Development interface must use unchanged public V7 provider profile')
        record_sha = None
    else:
        raise ValueError('Formal120 execution requires a pre-inference --freeze-record')
    for name, expected_hash in {**sources, **runtime}.items():
        target = (ROOT / name).resolve()
        if not target.is_relative_to(ROOT) or target.name == 'config.yml' or digest(target) != expected_hash:
            raise ValueError('Frozen public source/runtime changed: ' + name)
    return {'source_sha256': sources, 'runtime_sha256': runtime, 'public_provider_binding': binding,
            'freeze_manifest_sha256': record_sha, 'study_manifest_sha256': digest(args.manifest),
            'model': args.model}


def _failed(row):
    return row.get('error') is not None and str(row['error']).strip() not in ('', 'nan')


def physical_usage(policy_output, primary_output=None):
    """Count A once physically; retain unknown markers in each admitted suffix."""
    calls = (policy_output.get('tokens') or {}).get('per_call')
    if not isinstance(calls, list):
        return {'available': False, 'reason': 'per_call ledger unavailable', 'cost': None}
    prefix = ((primary_output or {}).get('tokens') or {}).get('per_call', [])
    if not isinstance(prefix, list) or canonical(calls[:len(prefix)]) != canonical(prefix):
        raise ValueError('Conceptual policy does not contain the exact A ledger prefix')
    observed = calls[len(prefix):]
    known = [call for call in observed if call.get('usage_available') is True]
    totals = {}
    for key in ('prompt', 'completion', 'cached'):
        values = [call.get(key) for call in known]
        totals[key] = sum(values) if all(type(value) is int and value >= 0 for value in values) else None
    return {'available': True, 'observed_invocations': len(observed),
            'known_usage_invocations': len(known), 'unknown_usage_invocations': len(observed) - len(known),
            'known_tokens': totals, 'cost': None,
            'scope': 'Physical A once; each followup only its suffix. Unknown is not zero billing.'}


def run_case(input_data, case, output_dir, *, agent_factory, parse_output,
             externalize, reference_metadata, cache_module, client_class,
             on_record=None):
    """One physical A, three independent declared followups, no correctness call.

    Injected seams support offline testing without importing provider/config.
    reference_metadata is recorder-only and is NEVER passed to an agent.
    """
    from scripts.v8_study_cache import CaseCache, activated_cache, PhysicalSourceObserver
    output_dir = Path(output_dir).resolve()
    cache = CaseCache(output_dir / 'case_cache' / case['question_id'])
    records = []
    parent = agent_factory('R')
    primary_started = time.monotonic()
    with activated_cache(cache.primary, cache_module) as primary_cache, PhysicalSourceObserver(client_class) as observed:
        envelope = parent.run_primary(input_data)
    primary_wall = time.monotonic() - primary_started
    # Exact core envelope, not a reconstructed answer or computed-withheld draft.
    envelope_sha = hashlib.sha256(canonical(envelope)).hexdigest()
    cache_proof = cache.freeze()
    primary_dir = output_dir / 'primaries'
    primary_dir.mkdir(exist_ok=True)
    import gzip
    envelope_path = primary_dir / (case['question_id'] + '.json.gz')
    with envelope_path.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            stream.write(canonical(envelope))
    primary_physical = observed.report()
    primary_output = None
    for mode in ('A', *case['followup_order']):
        agent = agent_factory(mode)
        started = time.monotonic()
        if mode == 'A':
            raw_output = agent.run_from_primary(input_data, deepcopy(envelope), mode='A')
            physical = primary_physical
            cache_after = cache_proof
            cache_activity = primary_cache
        else:
            target = cache.fork(mode)
            with activated_cache(target, cache_module) as cache_activity, PhysicalSourceObserver(client_class) as observed:
                raw_output = agent.run_from_primary(input_data, deepcopy(envelope), mode=mode)
            physical = observed.report()
            cache_after = cache.after_followup(mode)
        if hashlib.sha256(canonical(envelope)).hexdigest() != envelope_sha:
            raise ValueError('A envelope changed between policies')
        if not isinstance(raw_output, dict):
            raise ValueError('Core returned a nonnormalized policy output')
        if mode == 'A':
            primary_output = deepcopy(raw_output)
        followup_wall = time.monotonic() - started
        core_latency = raw_output.get('latency')
        normalized = parse_output(raw_output, include_full_audit=True)
        normalized['latency'] = primary_wall + (followup_wall if mode != 'A' else 0)
        row = {**reference_metadata, **case, 'study_arm': mode, 'study_terminal': True,
               'primary_envelope_sha256': envelope_sha,
               'primary_envelope_ref': str(envelope_path.relative_to(output_dir)),
               'physical_primary_inference_executed': mode == 'A',
               'physical_fhir': physical, 'cache_isolation': cache_after,
               'cache_activity': deepcopy(cache_activity),
               'physical_usage': physical_usage(raw_output, None if mode == 'A' else primary_output),
               'physical_A_wall_seconds': primary_wall if mode == 'A' else None,
               'shared_A_wall_seconds': primary_wall,
               'followup_wall_seconds': followup_wall if mode != 'A' else None,
               'A_materialization_wall_seconds': followup_wall if mode == 'A' else None,
               'raw_core_latency_seconds': core_latency,
               'latency_scope': 'Observed A wall plus followup controller/cache-copy wall; excludes output artifact writes',
               **normalized}
        # The core owns the exact diagnostic schema; preserve it in full before
        # a postterminal extractor binds any mechanism classification.
        row['mechanism'] = mechanism_record(raw_output)
        row = externalize(row, output_dir, question_id=case['question_id'])
        records.append(row)
        if on_record is not None:
            on_record(row)
    cache.verify_base()
    return records


def mechanism_record(raw_output):
    """Projection of explicit core diagnostics, not generic revised attribution."""
    revision = raw_output.get('revision')
    if not isinstance(revision, dict):
        raise ValueError('Core mechanism revision unavailable')
    # Filled against the settled core schema before inference freeze. Refuse to
    # turn absent metadata into an apparently successful no-op mechanism.
    value = revision.get('mechanism')
    if not isinstance(value, dict):
        raise ValueError('Core must expose explicit revision.mechanism diagnostics')
    return deepcopy(value)


def _runtime_inputs(path, cases, families=None):
    """Runtime-only row filtering; no SQL column is selected or sent to agents."""
    import csv
    wanted = {row['question_id']: row['patient_fhir_id'] for row in cases}
    found = {}
    with Path(path).open(newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        positions = {name: header.index(name) for name in ('question_id', 'patient_fhir_id', 'question', 'true_answer', 'assumption')}
        for values in reader:
            qid = values[positions['question_id']]
            if qid not in wanted:
                continue
            if qid in found or values[positions['patient_fhir_id']] != wanted[qid]:
                raise ValueError('Runtime selected question/patient identity differs')
            found[qid] = {key: values[index] for key, index in positions.items()}
    if set(found) != set(wanted):
        raise ValueError('Runtime input is missing selected study cases')
    # The unchanged typed scorer uses this generic template metadata. It stays
    # recorder-side; curate_input_dataset passes only question/context to core.
    family_path = Path(families) if families is not None else ROOT / 'manifests/question_families.csv'
    templates = {}
    with family_path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            qid = row['question_id']
            if qid in wanted:
                if qid in templates:
                    raise ValueError('Duplicate selected template metadata')
                templates[qid] = row['template']
    if set(templates) != set(wanted):
        raise ValueError('Selected scoring template metadata unavailable')
    for qid in found:
        found[qid]['template'] = templates[qid]
    return found


def execute_study(args):
    manifest = load(args.manifest)
    cases = validate_manifest(manifest)
    for relative, expected_hash in manifest['source_sha256'].items():
        if digest(ROOT / relative) != expected_hash:
            raise ValueError('Frozen input metadata changed: ' + relative)
    if digest(args.input) != manifest['source_sha256']['final_dataset/questions_answers_sql_fhir.csv']:
        raise ValueError('Runtime dataset differs from selected metadata input')
    frozen = verify_execution_freeze(args, manifest)
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    if args.development_case:
        cases = [choose_development_case(manifest, args.development_case)]
    identity = {**plan(manifest, directory, args.model), 'status': 'GENERATION_STARTED_NOT_SCORED',
                'manifest_sha256': digest(args.manifest), 'model': args.model,
                'expected_policy_outcomes': len(cases) * 4,
                'development_interface_only': bool(args.development_case),
                'freeze': frozen, 'runtime_case_metadata': cases,
                'source_sha256': frozen['source_sha256']}
    if args.development_case:
        identity.update({'parent_manifest_questions': 120, 'questions': 1,
                         'primary_physical_runs': 1, 'followup_physical_runs': 3,
                         'saved_policy_outcomes': 4})
    atomic_json(directory / 'study_identity.json', identity)
    atomic_json(directory / 'study_state.json', {'phase': 'generation_started',
                'saved_policy_outcomes': 0, 'expected_policy_outcomes': len(cases) * 4,
                'correctness_scored': False, 'no_automatic_retry_or_resume': True})
    # Only execute enters normal runtime imports. The external Azure bootstrap
    # handles credentials privately, using the same adapter as normal run_agent.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import pandas as pd
    import tools.cache as cache_module
    from fhir_client import FHIRClient
    from utils.core_utils import curate_input_dataset, parse_outputs
    from scripts.audit_store import externalize_result_audit
    from agent.counterexample_investigator_agent import CounterexampleInvestigatorAgent
    inputs = _runtime_inputs(args.input, cases)
    records = []
    generation_started = time.monotonic()
    output_file = directory / 'study_outcomes.json'
    def save(row):
        records.append(row)
        atomic_json(output_file, records)
        atomic_json(directory / 'study_state.json', {'phase': 'generation_running',
                    'saved_policy_outcomes': len(records), 'expected_policy_outcomes': len(cases) * 4,
                    'correctness_scored': False, 'current_case_index': row['case_index']})
    def factory(mode):
        agent = CounterexampleInvestigatorAgent(args.model, verbose=False, mode=mode)
        agent.set_budget(max_total_tokens=200000, max_tool_calls=30)
        return agent
    try:
        for case in cases:
            row = inputs[case['question_id']]
            # Only question/context is passed to core. References and metadata
            # operations remain recorder-side; no gold/template routing.
            model_input = curate_input_dataset(pd.DataFrame([row]), True)[0]
            run_case(model_input, case, directory, agent_factory=factory,
                     parse_output=parse_outputs, externalize=externalize_result_audit,
                     reference_metadata=row, cache_module=cache_module,
                     client_class=FHIRClient, on_record=save)
        if len(records) != len(cases) * 4:
            raise ValueError('Generation did not retain every declared policy outcome')
        for relative, expected_hash in identity['source_sha256'].items():
            if digest(ROOT / relative) != expected_hash:
                raise ValueError('Study runner source changed during generation')
        if verify_execution_freeze(args, manifest) != frozen:
            raise ValueError('Frozen identity drifted during generation')
        state = {'phase': 'development_interface_complete' if args.development_case else
                         'generation_complete_with_errors' if any(_failed(row) for row in records) else 'generation_complete',
                 'saved_policy_outcomes': len(records), 'expected_policy_outcomes': len(cases) * 4,
                 'terminal_errors': sum(_failed(row) for row in records), 'correctness_scored': False,
                 'physical_generation_wall_seconds': time.monotonic() - generation_started,
                 'owned_process_terminal_proof': 'Caller must record actual process exit separately before analysis'}
        atomic_json(directory / 'study_state.json', state)
        return int(any(_failed(row) for row in records))
    except BaseException as exception:
        # Provider/tool failures normally arrive as normalized policy outcomes.
        # An unexpected recorder/runtime failure is retained without fabricating
        # an answer, a billing ledger or a completed policy that never returned.
        atomic_json(directory / 'study_exception.json', {
                    'kind': 'unexpected_driver_or_runtime_exception',
                    'exception_type': type(exception).__name__,
                    'question_id': case['question_id'],
                    'saved_policy_outcomes': len(records),
                    'attempt_replacement': False, 'usage': 'Available only in retained returned outcomes/envelope/journal; unreturned usage unknown',
                    'message_omitted': 'May contain patient/provider details; original exception remains in owned run log'})
        atomic_json(directory / 'study_state.json', {'phase': 'interrupted_or_attention',
                    'saved_policy_outcomes': len(records), 'expected_policy_outcomes': len(cases) * 4,
                    'correctness_scored': False, 'no_automatic_retry_or_resume': True})
        raise


def analyze_records(records, manifest, score_answer, *, terminal_confirmed):
    """Pure postterminal analysis of saved120 references; never reads a QA CSV.

    Callers supply the frozen scorer after operational integrity/terminal checks.
    Each record includes a normalized actual policy result and study diagnostics.
    No unavailable A is replaced by null, and computed-withheld B is not O.
    """
    if terminal_confirmed is not True:
        raise ValueError('All owned study jobs must be explicitly terminal before scoring')
    cases = validate_manifest(manifest)
    expected = {(case['question_id'], mode) for case in cases for mode in MODES}
    indexed = {}
    for row in records:
        if not {'question_id', 'patient_fhir_id', 'study_arm', 'agent_answer', 'error',
                'question', 'true_answer', 'template', 'primary_envelope_sha256'} <= set(row):
            raise ValueError('Saved actual policy fields unavailable; never impute a null answer')
        key = row.get('question_id'), row.get('study_arm')
        if key not in expected or key in indexed or row.get('study_terminal') is not True:
            raise ValueError('Incomplete, duplicate or unexpected study outcome')
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError('All480 policy outcomes are required before any scoring')
    for case in cases:
        qid = case['question_id'];base = indexed[qid, 'A']
        for mode in MODES:
            row = indexed[qid, mode]
            if any(canonical(row.get(key)) != canonical(base.get(key)) for key in ('question', 'true_answer', 'template', 'patient_fhir_id')):
                raise ValueError('Saved study references/patient differ between policies')
            if row.get('primary_envelope_sha256') != base.get('primary_envelope_sha256'):
                raise ValueError('Policies did not share the same immutable A envelope')
            if mode != 'A':
                mechanism = row.get('mechanism')
                if not isinstance(mechanism, dict):
                    raise ValueError('Mechanism metadata unavailable before scoring')
                if type(mechanism.get('selected_B')) is not bool or type(mechanism.get('auxiliary_fatal')) is not bool:
                    raise ValueError('Mechanism selection/fatal flags must be explicit booleans')
                selected, fatal = mechanism['selected_B'], mechanism['auxiliary_fatal']
                b = mechanism.get('released_source_bound_B')
                if b is not None and (not isinstance(b, dict) or set(b) !=
                        {'answer', 'evidence', 'operation', 'contributing_ids', 'limitation'}):
                    raise ValueError('Source-bound B must retain the complete actual five-field artifact')
                if selected and fatal:
                    raise ValueError('A fatal policy cannot also claim an actually released B')
                if fatal and not _failed(row):
                    raise ValueError('Auxiliary fatal must retain a terminal policy error')
                if selected:
                    if not isinstance(b, dict) or b.get('error') or _failed(row):
                        raise ValueError('Selected B lacks a successful source-bound candidate')
                    if any(canonical(row.get(output)) != canonical(b.get(candidate)) for output, candidate in
                           [('agent_answer', 'answer'), ('evidence', 'evidence'), ('limitation', 'limitation')]):
                        raise ValueError('Selected B does not match actual released policy fields')
                elif not fatal:
                    if any(canonical(row.get(key)) != canonical(base.get(key)) for key in
                           ('agent_answer', 'evidence', 'limitation', 'error')):
                        raise ValueError('Nonfatal nonselection did not preserve actual A')
    # All completeness/reference gates above precede the first scorer call.
    scored = {}
    for key, row in indexed.items():
        scored[key] = bool(not _failed(row) and score_answer(row.get('agent_answer'), row['true_answer'],
                                                           row['question'], row.get('template', ''))['typed_correct'])
    arms = {}
    transitions = {}
    opportunity_patients, opportunity_operations = set(), set()
    for mode in MODES:
        arms[mode] = {'n': 120, 'correct': sum(scored[case['question_id'], mode] for case in cases),
                      'terminal_errors': sum(_failed(indexed[case['question_id'], mode]) for case in cases)}
        if mode == 'A':
            continue
        counters = Counter()
        for case in cases:
            qid = case['question_id'];row = indexed[qid, mode];a = scored[qid, 'A'];final = scored[qid, mode]
            mechanism = row.get('mechanism')
            if not isinstance(mechanism, dict):
                raise ValueError('Mechanism metadata unavailable; do not fabricate counters')
            b = mechanism.get('released_source_bound_B')
            b_correct = False
            if b is not None:
                if not isinstance(b, dict) or 'answer' not in b or b.get('error'):
                    raise ValueError('Invalid released B diagnostic')
                b_correct = bool(score_answer(b['answer'], row['true_answer'], row['question'], row.get('template', ''))['typed_correct'])
            opportunity = not a and b is not None and b_correct
            selected = mechanism.get('selected_B') is True
            fatal = mechanism.get('auxiliary_fatal') is True
            counters['O'] += opportunity
            counters['H'] += selected and not a and final
            counters['M'] += selected and a and not final
            counters['F'] += fatal and a and not final
            counters['all_post_A_helpful'] += not a and final
            counters['all_post_A_harmful'] += a and not final
            counters['selected_B'] += selected
            counters['B_available_source_bound'] += b is not None
            counters['auxiliary_fatal'] += fatal
            if mode == 'R' and opportunity:
                opportunity_patients.add(case['patient_fhir_id'])
                opportunity_operations.add(case['operation_group'])
        counters['net'] = counters['H'] - counters['M'] - counters['F']
        if counters['net'] != arms[mode]['correct'] - arms['A']['correct']:
            raise ValueError('Declared H-M-F does not reconcile actual final-minus-A credit')
        transitions[mode] = dict(counters)
    r = transitions['R'];gate = POLICY['advance_gate']
    passed = (r['O'] >= gate['O_min'] and r['net'] >= gate['net_min'] and
              r['M'] + r['F'] <= gate['harm_plus_fatal_max'] and
              len(opportunity_patients) >= gate['corrective_patients_min'] and
              len(opportunity_operations) >= gate['corrective_operations_min'])
    # File entrypoints launched with Python -I do not put ROOT on sys.path.
    # Load only the pure statistics module after the terminal/cohort gates.
    import importlib.util
    stats_spec = importlib.util.spec_from_file_location('v8_pure_statistics', ROOT / 'scripts/experiment_stats.py')
    stats = importlib.util.module_from_spec(stats_spec)
    stats_spec.loader.exec_module(stats)
    paired_difference = stats.paired_difference
    ids = [case['question_id'] for case in cases]
    patients = [case['patient_fhir_id'] for case in cases]
    intervals = {left + '-' + right: paired_difference(
                    [int(scored[qid, left]) for qid in ids], [int(scored[qid, right]) for qid in ids],
                    patients, n_boot=10000, seed=5106)
                 for left, right in [('R', 'A'), ('R', 'W'), ('R', 'C')]}
    return {'status': 'COMPLETED_MECHANISM_STUDY', 'primary_contrast': 'R - A',
            'arms': arms, 'mechanisms': transitions,
            'R_corrective_patients': len(opportunity_patients), 'R_corrective_operations': len(opportunity_operations),
            'R_minus_A_correct': arms['R']['correct'] - arms['A']['correct'],
            'paired_question_weighted_patient_bootstrap': intervals,
            'bootstrap': {'replicates': 10000, 'seed': 5106, 'primary': 'R-A', 'other_contrasts': 'diagnostic'},
            'advance_gate': {**gate, 'passed': passed},
            'diagnostic_contrasts_correct': {'R-W': arms['R']['correct'] - arms['W']['correct'],
                                             'R-C': arms['R']['correct'] - arms['C']['correct']},
            'limits': ['Adaptive balanced development mix, not a full-benchmark effect or patient-generalization test.',
                       'A is shared once physically; policies are dependent. W/R/C contrasts are diagnostic.',
                       'A correct/null credit uses unchanged typed scorer, not an independently established absence search.',
                       'Source-bound alternative does not prove semantic correctness; O is posthoc scorer-based.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    selection = commands.add_parser('select', help='Metadata-only fixed120 selection')
    selection.add_argument('--dataset', default=str(ROOT / 'final_dataset/questions_answers_sql_fhir.csv'))
    selection.add_argument('--families', default=str(ROOT / 'manifests/question_families.csv'))
    selection.add_argument('--split', default=str(ROOT / 'manifests/split.json'))
    selection.add_argument('--exposure', default=str(ROOT / 'manifests/final_evaluation_strata_20261015.json'))
    selection.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    selection.add_argument('--seed', type=int, default=SEED)
    for name in ('plan', 'execute'):
        p = commands.add_parser(name)
        p.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
        p.add_argument('--output', required=True)
        p.add_argument('--model', default='gpt-6-luna')
        p.add_argument('--input', default=str(ROOT / 'final_dataset/questions_answers_sql_fhir.csv'))
        p.add_argument('--development-case', help='auto or reserved qid outside120/300/1397; one unscored interface case only')
        if name == 'execute':
            p.add_argument('--freeze-record', help='Existing formal source/runtime study freeze; required for full120')
            p.add_argument('--provider-binding', required=True, help='Credential-free actual3hash binding written by normal launcher')
    analysis = commands.add_parser('analyze', help='Postterminal saved-study analysis only')
    analysis.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    analysis.add_argument('--input', required=True, help='Completed saved study JSON, never a QA CSV')
    analysis.add_argument('--output', required=True)
    analysis.add_argument('--terminal-confirmed', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'select':
        manifest = select_manifest(*read_selection_inputs(args.dataset, args.families, args.split, args.exposure), args.seed)
        manifest['source_sha256'] = {str(Path(path).resolve().relative_to(ROOT)): digest(path)
                                     for path in (args.dataset, args.families, args.split, args.exposure)}
        path = Path(args.manifest)
        if path.exists():
            raise ValueError('Refusing to replace a frozen study manifest')
        atomic_json(path, manifest)
        print(json.dumps({key: manifest[key] for key in ('questions', 'patients', 'family_counts', 'operation_counts', 'followup_order_counts')}, indent=2))
    elif args.command == 'plan':
        manifest = load(args.manifest)
        result = plan(manifest, args.output, args.model)
        if args.development_case:
            result.update({'development_interface_only': True, 'runtime_case_metadata': choose_development_case(manifest, args.development_case),
                           'parent_manifest_questions': 120, 'questions': 1,
                           'primary_physical_runs': 1, 'followup_physical_runs': 3, 'saved_policy_outcomes': 4})
        print(json.dumps(result, indent=2))
    elif args.command == 'execute':
        return execute_study(args)
    else:
        manifest = load(args.manifest)
        records = load(args.input)
        if not args.terminal_confirmed or len(records) != 480 or any(row.get('study_terminal') is not True for row in records):
            raise ValueError('Require explicit terminal assertion and all480 completed outcomes before scorer import')
        import importlib.util
        spec = importlib.util.spec_from_file_location('v8_pure_scorer', ROOT / 'utils/answer_scorer.py')
        scorer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scorer)
        result = analyze_records(records, manifest, scorer.score_answer, terminal_confirmed=True)
        if Path(args.output).exists():
            raise ValueError('Analysis output must be fresh')
        atomic_json(args.output, result)


if __name__ == '__main__':
    raise SystemExit(main())
