"""Local commands for Jeffrey's Azure deployment; no shared source changes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / ".local-setup"
PROJECT = Path(os.getenv("FHIR_LOCAL_PROJECT_ROOT", str(ROOT))).resolve()
sys.path.insert(0, str(LOCAL))
from azure_adapter import AzureSettings


def value(args, flag):
    for i, arg in enumerate(args):
        if arg == flag:
            if i + 1 >= len(args) or args[i + 1].startswith("--"):
                raise ValueError(flag + " needs a value")
            return args[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def default(args, flag, setting):
    if value(args, flag) is None:
        args.extend([flag, str(setting)])


def child_environment():
    env = os.environ.copy()
    env["FHIR_LOCAL_AZURE_CONFIG"] = str(ROOT / "config.yml")
    paths = [str(LOCAL), str(PROJECT)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["TIKTOKEN_CACHE_DIR"] = str(ROOT / ".cache/tiktoken")
    env["GOOGLE_CLOUD_PROJECT"] = "gen-lang-client-0053799010"
    return env


def main():
    parser = argparse.ArgumentParser(description="Personal Azure setup and execution commands")
    parser.add_argument("command", choices=["info", "offline-check", "check", "run", "evaluate"])
    # Parse only the local command; --help after run/evaluate belongs to the
    # underlying project entry point.
    args = parser.parse_args(sys.argv[1:2])
    forwarded = sys.argv[2:]
    settings = AzureSettings.load(ROOT / "config.yml")
    if args.command in {"info", "offline-check"}:
        print(json.dumps(settings.public(), indent=2))
        if args.command == "offline-check":
            print("Azure configuration is complete; API key is the only pending setting."
                  if not settings.key_present else "Azure configuration is complete.")
            print("No Azure request was made.")
        return 0

    requested_model = value(forwarded, "--model")
    if requested_model is not None and not settings.matches(requested_model):
        raise ValueError("This launcher is configured for " + settings.model)
    if value(forwarded, "--base_url") is not None:
        raise ValueError("Azure ENDPOINT is already configured; omit --base_url")
    default(forwarded, "--model", settings.model)
    helping = "--help" in forwarded or "-h" in forwarded
    if args.command == "check":
        script = PROJECT / "scripts/check_setup.py"
    elif args.command == "run":
        script = PROJECT / "run_agent.py"
        if not helping:
            default(forwarded, "--limit", 5)
            if int(value(forwarded, "--limit")) < 1:
                raise ValueError("--limit must be positive")
            default(forwarded, "--agent_strategy", "multi_turn_code_resource")
            default(forwarded, "--num_processes", 1)
            default(forwarded, "--save_interval", 1)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            default(forwarded, "--output", "output/local_azure/" + stamp + ".json")
    else:
        script = PROJECT / "evaluation_metrics.py"
        if not helping:
            default(forwarded, "--num_processes", 1)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            default(forwarded, "--output_dir", "output_eval/local_azure/" + stamp)
    if not helping and args.command != "check" and not settings.key_present:
        raise ValueError("Fill only AZURE_OPENAI.API_KEY in your local config.yml first")
    return subprocess.call([str(ROOT / ".venv/bin/python"), str(script), *forwarded],
                           cwd=PROJECT, env=child_environment())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print("Local Azure setup: " + str(error), file=sys.stderr)
        raise SystemExit(2)
