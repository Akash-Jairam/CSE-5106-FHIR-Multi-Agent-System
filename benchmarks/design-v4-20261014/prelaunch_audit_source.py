"""Independent prelaunch integrity audit; no agents, configuration or providers.

Run with repo Python and public phase2-dependencies on PYTHONPATH only. Reads
labels solely inside the existing identity hash calculation; never scores or
prints question/answer/patient content. Writes benchmark metadata only.
"""
import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
from types import SimpleNamespace
import zipfile

import pandas as pd
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
RECORD = Path(__file__).resolve().parent
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
canonical = lambda value: json.dumps(value, sort_keys=True, allow_nan=False)


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def load(path):
    return json.loads(Path(path).read_text())


def extract(path, names, namespace):
    """Execute reviewed public pure functions only; never top-level imports."""
    tree = ast.parse(Path(path).read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    require({node.name for node in nodes} == set(names), "Missing requested public helper")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def check_archive(path, expected, archive_sha):
    require(sha(path) == archive_sha, "Archive byte hash changed: " + path.name)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(expected) and set(names) == set(expected), "Archive members differ")
        require(archive.testzip() is None, "Archive CRC failure")
        for name, digest in expected.items():
            relative = PurePosixPath(name)
            require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe archive member")
            require(relative.name != "config.yml" and not name.startswith(".local-setup/config"),
                    "Credential configuration must not enter audit")
            require(hashlib.sha256(archive.read(name)).hexdigest() == digest, "Archive member differs: " + name)
            require(sha(ROOT / name) == digest, "Current frozen source differs: " + name)


def main():
    started = datetime.now(timezone.utc).isoformat()
    manifest, protocol = load(RECORD / "manifest.json"), load(RECORD / "prospective_protocol.json")
    runtime = load(RECORD / "runtime_adapter_manifest.json")
    binding = load(RECORD / "expected_public_binding.json")
    commands = load(RECORD / "launch_commands.json")
    require(len(manifest["source_sha256"]) == 142, "Expected exactly142 frozen source files")
    check_archive(RECORD / "source.zip", manifest["source_sha256"], manifest["source_archive_sha256"])
    require(sha(RECORD / "design.txt") == manifest["design_sha256"], "Design differs")
    require(sha(ROOT / "docs/design-v4-plan-conditioned.txt") == manifest["design_sha256"], "Design/source mismatch")
    allowed_runtime = {".local-setup/azure_adapter.py", ".local-setup/sitecustomize.py",
                       ".local-setup/azure.py", ".local-setup/phase3.py"}
    require(set(runtime["source_sha256"]) == allowed_runtime, "Runtime allowlist differs")
    check_archive(RECORD / "runtime_adapter_sources.zip", runtime["source_sha256"], runtime["archive_sha256"])
    require(set(binding) == {"provider_profile_sha256", "adapter_sha256", "bootstrap_sha256"}, "Binding schema differs")
    require(all(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) for value in binding.values()), "Invalid binding hash")
    require(binding["adapter_sha256"] == runtime["source_sha256"][".local-setup/azure_adapter.py"], "Adapter binding differs")
    require(binding["bootstrap_sha256"] == runtime["source_sha256"][".local-setup/sitecustomize.py"], "Bootstrap binding differs")
    for path in [ROOT / "output/local_azure/v4_dev_interface_20261014/.local-provider.json",
                 ROOT / "output/experiments/gpt-6-luna/query_contract_v3d_300q/.local-provider.json"]:
        require(canonical(load(path)) == canonical(binding), "Previously emitted public binding differs")
    require(canonical(protocol["expected_public_binding"]) == canonical(binding), "Prospective binding differs")
    require(protocol["phase_caps"] == {"planner": [10000, 0], "solver": [190000, 30]}, "Phase caps differ")
    require(protocol["global_caps"] == [200000, 30], "Global caps differ")
    for key, value in {"solver_requested_tools": 28, "reserved_counted_source_checks": 2,
                       "max_real_solver_calls": 20, "max_host_slots": 22,
                       "fixed_questions_per_arm": 300, "workers_per_arm": 1,
                       "estimated_input_guard": 32000, "minimum_team_correct": 239}.items():
        require(type(protocol[key]) is int and protocol[key] == value, "Protocol differs: " + key)
    require(protocol["first_attempt_only"] is True and "team > 238 AND team > fresh" in protocol["acceptance"], "Acceptance differs")
    source_tree = ast.parse((ROOT / "agent/plan_conditioned_agent.py").read_text())
    caps = next(ast.literal_eval(n.value) for n in source_tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "PHASE_CAPS" for t in n.targets))
    require(caps == {"planner": (10000, 0), "solver": (190000, 30)}, "Runtime phase caps differ")
    require(manifest["code_commit"] == "a64adaf497a2c7bbbc39c9aaec0830fcbfda6d43", "Unexpected source commit")
    commit_paths = set(subprocess.check_output(["git", "ls-tree", "-r", "--name-only", manifest["code_commit"]], cwd=ROOT, text=True).splitlines())
    untracked_frozen = sorted(set(manifest["source_sha256"]) - commit_paths)
    prior_manifest = load(ROOT / "benchmarks/design-v3d-20261014/manifest.json")
    require(untracked_frozen == ["AGENTS.md", "docs/audit/audit_questions.py", "docs/experiment_protocol.md",
            "docs/implementation_plan.md", "docs/model_choices.md", "docs/question_taxonomy.md", "docs/repository_audit.md"],
            "Unexpected non-Git frozen source")
    require(all(prior_manifest["source_sha256"].get(name) == manifest["source_sha256"][name]
                for name in untracked_frozen), "Legacy non-Git source differs from V3d freeze")
    for name in set(manifest["source_sha256"]) & commit_paths:
        frozen = subprocess.check_output(["git", "show", manifest["code_commit"] + ":" + name], cwd=ROOT)
        require(hashlib.sha256(frozen).hexdigest() == manifest["source_sha256"][name], "Frozen commit/source differs: " + name)
    require(sha(ROOT / "final_dataset/questions_answers_sql_fhir.csv") == manifest["dataset_sha256"], "Dataset differs")
    require(sha(ROOT / "manifests/split.json") == manifest["split_sha256"], "Split differs")
    split = load(ROOT / "manifests/split.json")
    expected = split["splits"]["held_out"]["question_ids"]
    require(len(expected) == 300 and len(set(expected)) == 300, "Fixed cohort differs")
    require(split["source_sha256"] == manifest["dataset_sha256"], "Split/dataset binding differs")

    # Same reviewed pure loader, runner-argv and identity functions as production.
    # get_default_model is stubbed because explicit --model makes it irrelevant;
    # this prevents a private configuration import during argparse construction.
    namespace = dict(Path=Path, hashlib=hashlib, json=json, metadata=metadata,
                     Requirement=Requirement, pd=pd, argparse=argparse,
                     get_default_model=lambda: "gpt-6-luna", __file__=str(ROOT / "run_agent.py"),
                     INPUT_COLUMNS=["question_id", "question", "true_answer", "assumption", "patient_fhir_id"],
                     CONDITIONS=("intact", "omitted"))
    extract(ROOT / "utils/core_utils.py", {"read_input_data", "curate_input_dataset"}, namespace)
    extract(ROOT / "run_agent.py", {"_boolean", "build_parser", "load_inputs", "select_split",
            "code_fingerprint", "runtime_dependency_versions", "run_identity"}, namespace)
    runner = SimpleNamespace(**{key: namespace[key] for key in ("build_parser", "load_inputs", "run_identity", "CONDITIONS")})
    exp = dict(os=os, dataclass=dataclass, argparse=argparse, run_agent=runner,
               get_default_model=lambda: "gpt-6-luna", ROOT=str(ROOT), HELD_OUT="held_out",
               DEFAULT_INPUT=str(ROOT / "final_dataset/questions_answers_sql_fhir.csv"),
               DEFAULT_SPLIT=str(ROOT / "manifests/split.json"),
               DEFAULT_PERTURBATIONS=str(ROOT / "manifests/perturbations.json"),
               DEFAULT_SUBSET=str(ROOT / "manifests/variability_subset.json"),
               DEFAULT_FAMILIES=str(ROOT / "manifests/question_families.csv"))
    extract(ROOT / "scripts/run_experiments.py", {"Arm", "build_parser", "runner_argv", "output_dir"}, exp)
    output = Path(manifest["output_directory"])
    require(output == ROOT / "output/experiments/gpt-6-luna/plan_conditioned_v4_300q", "Output destination differs")
    require(not output.exists(), "Formal output already exists; not a fresh prelaunch")
    require(commands["concurrent"] is True and commands["workers_per_arm"] == 1, "Paired concurrency declaration differs")
    require(set(commands["commands"]) == {"single", "team"}, "Unexpected command arm")
    identities, inputs, model_inputs, emitted = {}, {}, {}, {}
    for label, strategy in [("single", "single_agent"), ("team", "multi_agent_v4")]:
        record = commands["commands"][label]
        argv = record["argv"]
        require(argv[:5] == ["caffeinate", "-i", ".venv/bin/python", ".local-setup/phase3.py", "run"], "Wrong launcher")
        require(argv.count("--project-root") == 1, "Project flag repeated or missing")
        index = argv.index("--project-root")
        require(argv[index + 1] == ".", "Unexpected project root")
        forwarded = argv[5:index] + argv[index + 2:]
        require(len([item for item in forwarded if item.startswith("--")]) == len(set(item for item in forwarded if item.startswith("--"))), "Repeated launch flag")
        args = exp["build_parser"]().parse_args(["run", *forwarded])
        arm = exp["Arm"]("ordinary", "single_agent" if label == "single" else "multi_agent")
        require(args.arms == [arm.name] and record["arm"] == arm.name, "Wrong arm selection")
        require(args.team_strategy == "multi_agent_v4" and args.model == "gpt-6-luna", "Wrong strategy/model")
        require((args.num_processes, args.save_interval, args.audit_storage, args.split, args.limit) == (1, 1, "gzip", "held_out", None), "Operational/input settings differ")
        require((args.max_total_tokens, args.max_tool_calls) == (200000, 30), "CLI ceilings differ")
        require(Path(args.input) == ROOT / "final_dataset/questions_answers_sql_fhir.csv" and Path(args.split_manifest) == ROOT / "manifests/split.json", "Runner input default differs")
        job_args = exp["runner_argv"](arm, args)
        settings = runner.build_parser().parse_args(job_args)
        require(settings.agent_strategy == strategy and settings.enable_cache is True and settings.add_patient_fhir_id is True, "Worker strategy/cache/context differs")
        frame = runner.load_inputs(settings.input, settings.limit, settings.split_manifest, settings.split)
        require(list(frame["question_id"]) == expected, "Exact ordered300 differ")
        identity = runner.run_identity(settings, frame)
        old = load(ROOT / "output/experiments/gpt-6-luna/query_contract_v3d_300q" / (arm.name + ".json.run.json"))
        require(identity["input_hash"] == old["input_hash"], "Inputs differ from completed V3d fixed300")
        for name in ["condition", "perturbations_hash", "subset_hash", "repeat"]:
            require(identity[name] is None, "Unexpected routing or repeated/subset arm")
        require(identity["audit_storage"] == "gzip" and identity["audit_artifact_schema"] == 2, "Audit storage identity differs")
        identities[label] = identity
        inputs[label] = frame[namespace["INPUT_COLUMNS"]].to_json(orient="records", date_format="iso")
        model_inputs[label] = namespace["curate_input_dataset"](frame, True)
        emitted[label] = job_args
        require(record["environment"] == {"FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR": str(output / "provider-events")}, "Unexpected launch environment")
        require(Path(record["log"]) == output / ("run_" + label + ".log"), "Unexpected log path")
        shell = shlex.split(record["shell_command"])
        require(shell[0] == "FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR=" + str(output / "provider-events"), "Shell journal differs")
        require(shell[1:-3] == argv and shell[-3:] == [">", record["log"], "2>&1"], "Shell/argv disagreement")
        require(record["pty"] is True, "Unexpected launch terminal mode")
    require(inputs["single"] == inputs["team"] and model_inputs["single"] == model_inputs["team"], "Paired inputs differ")
    stripped = lambda obj: {key: value for key, value in obj.items() if key != "agent_strategy"}
    require(canonical(stripped(identities["single"])) == canonical(stripped(identities["team"])), "Paired identities differ beyond strategy")
    health = load(ROOT / "benchmarks/design-v3d-20261014/health_preflight.json")["request_manifest"]
    require(health["provider_identity_profile_sha256"] == binding["provider_profile_sha256"], "Public ceiling/profile provenance differs")
    require((health["caps"], health["sdk_timeout_seconds"], health["sdk_max_retries"], health["application_max_retries"], health["reasoning_effort"]) == ([16384], 60, 0, 1, "none"), "Historical provider cap provenance differs")
    # Recheck hashes and freshness after all identity reads, narrowing observation drift.
    require(all(sha(ROOT / name) == digest for name, digest in manifest["source_sha256"].items()), "Source changed during audit")
    require(all(sha(ROOT / name) == digest for name, digest in runtime["source_sha256"].items()), "Runtime changed during audit")
    require(not output.exists(), "Output created during prelaunch audit")
    report = {"status": "PASS", "scope": "Read-only prelaunch provenance/interface audit; no scoring, configuration or model/FHIR/network calls",
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "helper_sha256": sha(__file__), "frozen_source_files_verified": 142,
        "source_archive_sha256": manifest["source_archive_sha256"], "source_commit": manifest["code_commit"],
        "source_zip_member_hashes_crc_verified": True, "git_source_members_verified": 135,
        "non_git_sources_verified_by_manifest_archive_and_current_bytes": untracked_frozen,
        "design_dataset_split_hashes_verified": True,
        "runtime_sources_and_exact4_archive_members_verified": True, "runtime_archive_sha256": runtime["archive_sha256"],
        "public_binding": binding, "binding_matches_development_and_v3d": True,
        "exact_ordered_questions_per_arm": 300, "ordered_question_ids_sha256": hashlib.sha256(canonical(expected).encode()).hexdigest(),
        "input_hash": identities["single"]["input_hash"], "inputs_match_completed_v3d": True,
        "model_input_strings_equal_and_no_reference_answer_projection": True,
        "current_code_hash": identities["single"]["code_hash"], "prospective_run_identities": identities,
        "runner_generated_argv": emitted, "explicit_concurrent_one_worker_commands_verified": True,
        "global_caps": [200000, 30], "phase_caps": protocol["phase_caps"],
        "source_check_reservation": [28, 2], "solver_calls_host_slots": [20, 22],
        "request_ceiling_timeout_retries_effort_public_provenance": [16384, 60, 0, 1, "none"],
        "formal_output_absent_before_and_after_audit": True,
        "limitations": ["Expected binding is matched to public bindings; private settings were not loaded. Root must verify newly emitted formal binding after single bootstrap and before team launch.",
                       "Concurrency is a two-command launch declaration, not observed running-process evidence; the manifest joined command string is descriptive.",
                       "This is a point-in-time source/metadata observation, not an atomic runtime attestation or provider-availability guarantee."]}
    destination = RECORD / "prelaunch_audit.json"
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("PASS:142 frozen source files, exact4 runtime sources, identical ordered300 inputs and explicit paired one-worker commands")
    print("report_sha256=" + sha(destination))


if __name__ == "__main__":
    main()
