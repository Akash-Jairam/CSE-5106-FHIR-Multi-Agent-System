"""One unscored, entirely synthetic request through the existing Azure bootstrap."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT / 'output'):
        raise ValueError('Ignored output directory required')
    if not args.child:
        if output.exists():
            raise ValueError('Fresh test output required; no retry')
        sys.path.insert(0, str(ROOT / '.local-setup'))
        import azure
        from azure_adapter import AzureSettings
        settings = AzureSettings.load(ROOT / 'config.yml')
        if not settings.key_present or not settings.matches('gpt-6-luna'):
            raise ValueError('Existing configured deployment required')
        output.mkdir(parents=True)
        profile = settings.public()
        profile.pop('api_key_configured', None)
        binding = {'provider_profile_sha256': hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
                   'adapter_sha256': sha(ROOT / '.local-setup/azure_adapter.py'),
                   'bootstrap_sha256': sha(ROOT / '.local-setup/sitecustomize.py')}
        (output / 'provider_binding.json').write_text(json.dumps(binding, indent=2) + '\n')
        env = azure.child_environment()
        dependencies = ROOT / '.local-setup/phase2-dependencies'
        if dependencies.exists():
            env['PYTHONPATH'] += os.pathsep + str(dependencies)
        env.update({'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'PYTHONDONTWRITEBYTECODE': '1',
                    'FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR': str(output / 'provider-events')})
        return subprocess.call([str(ROOT / '.venv/bin/python'), '-B', str(Path(__file__).resolve()),
                                '--child', '--output', str(output)], cwd=ROOT, env=env)
    from utils.core_utils import safe_llm_call
    tools = [{'type': 'function', 'function': {'name': 'inspect_synthetic',
              'description': 'Inspect a synthetic test value.',
              'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}]
    messages = [
        {'role': 'system', 'content': 'This is an entirely synthetic message serialization test. Reply only OK after tool results and feedback. Do not call tools.'},
        {'role': 'user', 'content': 'Inspect synthetic value seven.'},
        {'role': 'assistant', 'content': '{"obligation_span":"synthetic value","observation_id":"synthetic-observation"}',
         'tool_calls': [{'id': 'call_synthetic_wire_probe', 'type': 'function',
                         'function': {'name': 'inspect_synthetic', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'call_synthetic_wire_probe', 'content': '{"value":7}'},
        {'role': 'user', 'content': 'Witness lineage recorded; tool results are complete. Reply exactly OK without another tool call.'}]
    response, error, usage = safe_llm_call('gpt-6-luna', messages, tools=tools,
                                          max_retries=1, max_completion_tokens=64)
    report = {'status': 'SYNTHETIC_WIRE_ACCEPTED' if response is not None and error is None else 'SYNTHETIC_WIRE_FAILED',
              'model': 'gpt-6-luna', 'sdk_request_limit': 1, 'completion_cap': 64,
              'roles': [m['role'] for m in messages], 'assistant_had_content_and_native_call': True,
              'matching_tool_reply_precedes_feedback': True, 'usage': usage,
              'provider_error_present': error is not None, 'response_present': response is not None,
              'benchmark_questions': 0, 'FHIR_calls': 0, 'accuracy_scored': False,
              'scope': 'Azure acceptance of the repaired synthetic wire shape; not production investigator discovery or accuracy evidence.'}
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return 0 if report['status'] == 'SYNTHETIC_WIRE_ACCEPTED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
