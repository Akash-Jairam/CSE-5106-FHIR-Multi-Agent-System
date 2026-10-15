"""Ignored V8 interface probe: scripted synthetic A, real bounded auxiliary.

There are no benchmark questions, expected-answer comparisons, scorer imports,
or FHIR network calls. The investigator/judge prompts are the production prompts.
--prepare-only is fully offline. A paid auxiliary requires the normal existing
Azure bootstrap AND an explicit --allow-paid-auxiliary argument. Root owns launch.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PATIENT = '11111111-1111-1111-1111-111111111111'
QUESTION = ('Question: Which names belong to all records with the greatest numeric value?\n'
    'Context:\nPatient FHIR ID is ' + PATIENT + '.\n'
    'Assume the current time is 2131-01-01 00:00:00.')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def fixture():
    resources = {'Observation': [
        {'resourceType': 'Observation', 'id': identifier,
         'subject': {'reference': 'Patient/' + PATIENT},
         'effectiveDateTime': '2130-01-01T12:00:00Z',
         'code': {'text': name}, 'valueQuantity': {'value': value, 'unit': 'mg'}}
        for identifier, name, value in [('a', 'Alpha', 2), ('b', 'Beta', 9),
                                        ('c', 'Gamma', 5), ('d', 'Delta', 9)]]}
    program = """rows = retrieved_resources['Observation']
answer = max(rows, key=lambda row: row['valueQuantity']['value'])['code']['text']
evidence = []
for row in rows:
    for path in ['valueQuantity.value', 'code.text']:
        value = row
        for part in path.split('.'):
            value = value[part]
        evidence.append({'resource_id': 'Observation/' + row['id'], 'field_path': path, 'value': value})
contributing_ids = sorted({'Observation/' + row['id'] for row in rows})
operation = 'Return the code text of the record selected by maximum numeric value'
limitation = None
"""
    return resources, program


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='gpt-6-luna')
    parser.add_argument('--mode', choices=['W', 'R', 'C'], default='R')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--allow-paid-auxiliary', action='store_true')
    args = parser.parse_args(argv)
    if not args.prepare_only and not args.allow_paid_auxiliary:
        raise ValueError('A real auxiliary requires explicit --allow-paid-auxiliary')
    directory = Path(args.output).resolve()
    if directory.exists():
        raise ValueError('Output must be fresh; no retry or resume')
    directory.mkdir(parents=True, exist_ok=False)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from litellm import Message
    from agent import base_agent
    from agent.counterexample_investigator_agent import CounterexampleInvestigatorAgent
    from agent.counterexample_investigator_phase import source_identity
    from tools import get_tool

    provider_binding = None
    if not args.prepare_only:
        import azure_adapter
        settings = getattr(azure_adapter, '_settings', None)
        if settings is None or not settings.matches(args.model):
            raise ValueError('Use the existing normal Azure child bootstrap for this model')
        profile = settings.public()
        profile.pop('api_key_configured', None)
        provider_binding = {
            'provider_profile_sha256': hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
            'adapter_sha256': hashlib.sha256((ROOT / '.local-setup/azure_adapter.py').read_bytes()).hexdigest(),
            'bootstrap_sha256': hashlib.sha256((ROOT / '.local-setup/sitecustomize.py').read_bytes()).hexdigest()}
        os.environ['FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR'] = str(directory / 'provider-events')

    resources, primary_program = fixture()
    scripted = [Message(role='assistant', content=None, tool_calls=[{
        'id': 'synthetic-A-final', 'type': 'function', 'function': {
            'name': 'execute_python_code', 'arguments': canonical({'mode': 'final', 'code': primary_program})}}]),
        Message(role='assistant', content='{"answer":null,"evidence":[],"limitation":null}')]
    counts = {'scripted_primary_responses': 0, 'real_auxiliary_sdk_calls': 0,
        'synthetic_source_reads': 0, 'synthetic_retrieval_calls': 0,
        'real_python_worker_calls': 0}
    stage = {'primary': True}
    real_provider = base_agent.safe_llm_call
    def provider(**kwargs):
        if stage['primary']:
            index = counts['scripted_primary_responses']
            if index >= len(scripted):
                raise AssertionError('Unexpected extra primary request; never fall through to paid inference')
            counts['scripted_primary_responses'] += 1
            return scripted[index], None, {'prompt_tokens': 0, 'completion_tokens': 0,
                'total_tokens': 0, 'llm_calls': 1, 'cost': 0.0,
                'per_call': [{'prompt': 0, 'completion': 0, 'cached': 0, 'usage_available': True}]}
        if args.prepare_only:
            raise AssertionError('Offline preparation cannot invoke the auxiliary')
        counts['real_auxiliary_sdk_calls'] += 1
        return real_provider(**kwargs)
    def source_fetch(refs):
        counts['synthetic_source_reads'] += 1
        return {row['id']: deepcopy(row) for kind, rows in resources.items()
                for row in rows if (kind, row['id']) in refs}
    def dispatch(name):
        if name == 'execute_python_code':
            actual = get_tool(name)
            def execute(**kwargs):
                counts['real_python_worker_calls'] += 1
                return actual(**kwargs)
            return execute
        if name in {'get_resources_by_patient_fhir_id', 'get_resources_by_resource_id'}:
            def retrieve(**kwargs):
                counts['synthetic_retrieval_calls'] += 1
                if kwargs.get('patient_fhir_id', PATIENT) != PATIENT:
                    return {'error': 'Synthetic fixture contains one patient only'}
                kind = kwargs.get('resource_type')
                rows = deepcopy(resources.get(kind, []))
                ids = kwargs.get('resource_ids')
                if isinstance(ids, list):
                    rows = [row for row in rows if row['id'] in ids]
                return {kind: rows}
            return retrieve
        raise AssertionError('No other synthetic tool is allowed to reach a network boundary')

    sources = source_identity()
    with ExitStack() as stack:
        if args.prepare_only:
            stack.enter_context(patch('agent.base_agent.setup_api_keys'))
        stack.enter_context(patch('agent.base_agent.safe_llm_call', side_effect=provider))
        stack.enter_context(patch('agent.single_agent.get_starting_packet', side_effect=lambda *_a, **_k:
            {'text': 'Handwritten synthetic records: ' + canonical(resources), 'resources': deepcopy(resources)}))
        stack.enter_context(patch('tools.field_check.fetch_from_store', side_effect=source_fetch))
        stack.enter_context(patch('agent.multi_turn_code_resource_agent.get_tool', side_effect=dispatch))
        primary = CounterexampleInvestigatorAgent(args.model)
        primary.set_budget(max_total_tokens=200000, max_tool_calls=30)
        envelope = primary.run_primary(QUESTION)
        if (counts['scripted_primary_responses'] != 2
                or envelope['payload']['candidate']['artifact'] is None
                or envelope['payload']['candidate']['diagnostics']['fatal']):
            raise ValueError('Scripted synthetic A did not bind actual execution and original source check')
        with gzip.open(directory / 'scripted_primary.json.gz', 'wb') as stream:
            stream.write(canonical(envelope).encode())
        stage['primary'] = False
        output = None
        if not args.prepare_only:
            agent = CounterexampleInvestigatorAgent(args.model, mode=args.mode)
            agent.set_budget(max_total_tokens=200000, max_tool_calls=30)
            output = agent.run_from_primary(QUESTION, envelope, mode=args.mode)
    if source_identity() != sources:
        raise ValueError('Core source changed during the interface probe')
    if output is not None:
        from utils.core_utils import parse_outputs
        from scripts.audit_store import externalize_result_audit
        saved = externalize_result_audit(parse_outputs(output, include_full_audit=True), directory,
                                          question_id='synthetic-tied-extrema')
        (directory / 'outcome.json').write_text(canonical(saved) + '\n')
    summary = {'synthetic_interface_only': True, 'prepare_only': args.prepare_only,
        'benchmark_data_used': False, 'accuracy_scored': False, 'FHIR_network_calls': 0,
        'primary_is_scripted_not_a_model_discovery_result': True,
        'scripted_primary_ledger_is_not_real_billing': True,
        'model': args.model, 'mode': args.mode, **counts,
        'status': (output.get('revision') or {}).get('status') if output is not None else 'primary_prepared_offline',
        'source_identity': sources, 'public_provider_binding': provider_binding,
        'error_present': bool(output.get('error')) if output is not None else False}
    (directory / 'safe_summary.json').write_text(canonical(summary) + '\n')
    print(canonical(summary))
    return int(summary['error_present'])


if __name__ == '__main__':
    raise SystemExit(main())
