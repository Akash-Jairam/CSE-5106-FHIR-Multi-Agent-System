"""Launch the V3 benchmark with a pinned local Azure adapter and no secret output.

Only the normal Azure runtime loads the credential configuration. This wrapper
records the adapter's public settings and copies three allowlisted Python source
files; it never copies configuration, credentials, or arbitrary local files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
RUNTIME_SOURCES = ("azure.py", "azure_adapter.py", "sitecustomize.py")
PUBLIC_FIELDS = ("provider", "endpoint", "api_version", "deployment", "model",
                 "reasoning_effort", "max_completion_tokens")
PROFILE_ENV = "FHIR_V3_PROVIDER_PROFILE_PATH"


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def profile_digest(profile):
    # Preserve the canonicalization used by the earlier local Azure launchers.
    return sha256_bytes(json.dumps(profile, sort_keys=True).encode())


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "resume", "status"))
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--run-dir", "--output_dir", "--output-dir", dest="run_dir",
                        type=Path, required=True)
    parser.add_argument("--model")
    return parser


def public_profile(settings):
    # An explicit allowlist also prevents future settings.public() additions
    # from accidentally recording a credential or its presence.
    public = settings.public()
    return {name: public[name] for name in PUBLIC_FIELDS}


def snapshot_runtime(run_dir, local_dir, profile, *, resume):
    """Create one public binding, or verify and reuse its frozen source copies."""
    record_path = run_dir / "provider_profile.json"
    runtime = run_dir / "runtime"
    if record_path.is_symlink() or runtime.is_symlink():
        raise ValueError("Provider artifacts must be regular local paths")
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if (record.get("version") != 1 or record.get("public_profile") != profile
                or record.get("provider_profile_sha256") != profile_digest(profile)
                or record.get("runtime_directory") != str(runtime)):
            raise ValueError("Provider profile does not match the frozen run")
        hashes = record.get("runtime_source_sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(RUNTIME_SOURCES):
            raise ValueError("Incomplete runtime source binding")
        for name in RUNTIME_SOURCES:
            path = runtime / name
            if path.is_symlink() or sha256_bytes(path.read_bytes()) != hashes[name]:
                raise ValueError("A frozen runtime source changed")
        # This file is used to prepare paths; the adapter/bootstrap themselves
        # execute from the frozen copies even if local development moves on.
        if sha256_bytes((local_dir / "azure.py").read_bytes()) != hashes["azure.py"]:
            raise ValueError("Local Azure launcher changed since the source freeze")
        return record_path, runtime
    if resume:
        raise ValueError("Resume requires the original provider source binding")
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(exist_ok=True)
    hashes = {}
    for name in RUNTIME_SOURCES:
        source = (local_dir / name).read_bytes()
        destination = runtime / name
        if destination.is_symlink():
            raise ValueError("Runtime source copies must be regular files")
        if destination.exists():
            if destination.read_bytes() != source:
                raise ValueError("Existing runtime files do not match the provider source")
        else:
            with destination.open("xb") as stream:
                stream.write(source)
        hashes[name] = sha256_bytes(source)
    record = {"version": 1, "public_profile": profile,
              "provider_profile_sha256": profile_digest(profile),
              "runtime_source_sha256": hashes, "runtime_directory": str(runtime)}
    with record_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return record_path, runtime


def main(argv=None):
    args, forwarded = build_parser().parse_known_args(argv)
    project = args.project_root.resolve()
    script = project / "scripts/v3_full_run.py"
    if not script.is_file():
        raise ValueError("Choose a project containing the V3 benchmark runner")
    run_dir = args.run_dir if args.run_dir.is_absolute() else project / args.run_dir
    run_dir = run_dir.resolve()
    local_dir = ROOT / ".local-setup"
    executable = str(ROOT / ".venv/bin/python")
    dependencies = local_dir / "phase2-dependencies"
    if any(item in ("--base_url", "--base-url")
           or item.startswith(("--base_url=", "--base-url=")) for item in forwarded):
        raise ValueError("Use the configured local provider endpoint")
    if args.command == "status":
        # Reading aggregate progress must work without valid credentials and
        # must not bootstrap a provider client merely to inspect saved state.
        environment = os.environ.copy()
        environment.pop("FHIR_LOCAL_AZURE_CONFIG", None)
        environment.pop(PROFILE_ENV, None)
        paths = [str(project)]
        if dependencies.is_dir():
            paths.append(str(dependencies))
        if environment.get("PYTHONPATH"):
            paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(paths)
        model = args.model
    else:
        sys.path.insert(0, str(local_dir))
        import azure
        from azure_adapter import AzureSettings

        # Loading credentials is confined to the existing authorized runtime.
        settings = AzureSettings.load(ROOT / "config.yml")
        model = args.model or settings.model
        if not settings.matches(model):
            raise ValueError("Requested model differs from the local deployment")
        if args.command in ("run", "resume") and not settings.key_present:
            raise ValueError("Local provider credentials are unavailable")
        azure.PROJECT = project
        environment = azure.child_environment()
        profile = public_profile(settings)
        runtime = local_dir
        if args.command in ("run", "resume"):
            record_path, runtime = snapshot_runtime(run_dir, local_dir, profile,
                                                   resume=args.command == "resume")
            environment[PROFILE_ENV] = str(record_path)
            environment["FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR"] = str(run_dir / "provider-events")
        else:
            # Planning is read-only, including when the output directory does
            # not yet exist. The runner can plan without a provider record.
            environment.pop(PROFILE_ENV, None)
            environment.pop("FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR", None)
        paths = [str(runtime), str(project)]
        if dependencies.is_dir():
            paths.append(str(dependencies))
        if environment.get("PYTHONPATH"):
            paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(paths)
    command = [executable, str(script), args.command, *forwarded, "--run-dir", str(run_dir)]
    if model is not None:
        command.extend(("--model", model))
    os.chdir(project)
    # Preserve the supervisor's PID and signal delivery for detached runs.
    os.execve(executable, command, environment)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Configuration errors can include sensitive YAML text; never print
        # exceptions, tracebacks, settings objects, or provider response bodies.
        print("V3 benchmark launcher could not validate its local setup or frozen provider binding.",
              file=sys.stderr)
        raise SystemExit(2)
