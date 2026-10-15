"""Use the existing private Azure bootstrap for an explicitly requested V8 study.

Configuration is consumed only by the normal AzureSettings runtime. This helper
records three public hashes, never credentials, and starts one owned child with
no retries or replacement outcomes. A small development case is unscored.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='gpt-6-luna')
    parser.add_argument('--manifest', default=str(ROOT / 'manifests/v8_mechanism_120_20261015.json'))
    parser.add_argument('--freeze-record')
    parser.add_argument('--development-case')
    args = parser.parse_args(argv)
    if not args.freeze_record and not args.development_case:
        raise ValueError('A substantial study requires a frozen record')
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Choose a fresh output directory; no retries or resume')
    if not output.is_relative_to(ROOT / 'output'):
        raise ValueError('Study outputs must stay in the ignored repository output directory')

    sys.path.insert(0, str(ROOT / '.local-setup'))
    import azure
    from azure_adapter import AzureSettings

    settings = AzureSettings.load(ROOT / 'config.yml')
    if not settings.key_present or not settings.matches(args.model):
        raise ValueError('The configured deployment cannot run the requested model')
    profile = settings.public()
    profile.pop('api_key_configured', None)
    binding = {
        'provider_profile_sha256': hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
        'adapter_sha256': hashlib.sha256((ROOT / '.local-setup/azure_adapter.py').read_bytes()).hexdigest(),
        'bootstrap_sha256': hashlib.sha256((ROOT / '.local-setup/sitecustomize.py').read_bytes()).hexdigest(),
    }
    # This file is outside the fresh run directory, so the runner can still
    # reject a reused run directory before making a request.
    output.parent.mkdir(parents=True, exist_ok=True)
    binding_path = output.with_name(output.name + '.provider.json')
    with binding_path.open('x') as stream:
        json.dump(binding, stream, indent=2)
        stream.write('\n')
    azure.PROJECT = ROOT
    environment = azure.child_environment()
    dependencies = ROOT / '.local-setup/phase2-dependencies'
    if dependencies.exists():
        environment['PYTHONPATH'] += ':' + str(dependencies)
    environment['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR'] = str(output / 'provider-events')
    command = [str(ROOT / '.venv/bin/python'), '-B', str(ROOT / 'scripts/v8_mechanism_study.py'),
               'execute', '--output', str(output), '--model', args.model,
               '--manifest', str(Path(args.manifest).resolve()), '--provider-binding', str(binding_path)]
    if args.freeze_record:
        command += ['--freeze-record', str(Path(args.freeze_record).resolve())]
    if args.development_case:
        command += ['--development-case', args.development_case]
    return subprocess.call(command, cwd=ROOT, env=environment)


if __name__ == '__main__':
    try:
        code = main()
    except Exception:
        print('V8 launcher could not validate the fresh output, runtime or deployment.', file=sys.stderr)
        code = 2
    raise SystemExit(code)
