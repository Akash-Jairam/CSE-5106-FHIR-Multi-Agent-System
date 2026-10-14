"""Personal Azure experiment launcher; provider configuration stays local."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import azure
from azure_adapter import AzureSettings

ROOT = Path(__file__).resolve().parent.parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "status", "run", "analyze", "variability", "revisions"])
    parser.add_argument("--project-root", type=Path, default=ROOT, help="Checkout to execute")
    args, forwarded = parser.parse_known_args()
    project = args.project_root.resolve()
    script = project / "scripts/run_experiments.py"
    if not script.exists():
        raise ValueError("Choose a checkout containing the Phase 3 experiment script")
    settings = AzureSettings.load(ROOT / "config.yml")
    model = azure.value(forwarded, "--model")
    if model and not settings.matches(model):
        raise ValueError("The requested model does not match the local deployment")
    if azure.value(forwarded, "--base_url"):
        raise ValueError("Omit --base_url; the Azure endpoint is configured locally")
    azure.default(forwarded, "--model", settings.model)
    azure.default(forwarded, "--num_processes", 1)
    azure.default(forwarded, "--save_interval", 1)
    output = azure.value(forwarded, "--output_dir")
    if not output:
        raise ValueError("Specify --output_dir explicitly to keep smoke and full runs separate")
    output = Path(output)
    if not output.is_absolute():
        output = project / output
    profile = settings.public()
    profile.pop("api_key_configured", None)
    identity = {"provider_profile_sha256": digest(profile), "adapter_sha256": hashlib.sha256(
        (ROOT / ".local-setup/azure_adapter.py").read_bytes()).hexdigest(),
        "bootstrap_sha256": hashlib.sha256((ROOT / ".local-setup/sitecustomize.py").read_bytes()).hexdigest()}
    binding = output / ".local-provider.json"
    if binding.exists() and json.loads(binding.read_text()) != identity:
        raise ValueError("Local provider settings changed; choose a new --output_dir")
    if args.command == "run":
        if not settings.key_present:
            raise ValueError("The local Azure API key is not configured")
        if not binding.exists() and any(output.glob("*.run.json")):
            raise ValueError("Existing results lack a local provider binding; choose a fresh --output_dir")
        output.mkdir(parents=True, exist_ok=True)
        if not binding.exists():
            binding.write_text(json.dumps(identity, indent=2) + "\n")
    azure.PROJECT = project
    environment = azure.child_environment()
    dependencies = ROOT / ".local-setup/phase2-dependencies"
    if dependencies.exists():
        environment["PYTHONPATH"] += ":" + str(dependencies)
    return subprocess.call([str(ROOT / ".venv/bin/python"), str(script), args.command, *forwarded],
                           cwd=project, env=environment)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("Local Phase 3 setup could not validate the checkout, output path, or provider profile.", file=sys.stderr)
        raise SystemExit(2)
