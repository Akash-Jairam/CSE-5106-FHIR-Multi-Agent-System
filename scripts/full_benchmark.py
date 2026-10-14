"""Supervise one frozen, first-attempt run of all 2,931 released questions.

Run through the local provider launcher when using Azure. A run requires a
fresh output directory; this supervisor never resumes or retries failed rows.
All child processes inherit the launcher's provider bootstrap and credentials
internally. Only public settings, hashes, commands, and aggregate health are
written by this supervisor. No question, answer, trace, or credential is printed.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUESTIONS = 2931
SYSTEMS = ("direct_llm", "single_agent", "multi_agent")
INPUT_GUARD = 32000
LARGE_RESULT_BYTES = 512 * 1024 * 1024
LARGE_RESULT_HEARTBEAT = 60


def utc_time():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def dataset_cohort(path):
    """Validate the full cohort without disclosing any reference values."""
    patients = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"question_id", "patient_fhir_id"} <= set(reader.fieldnames or []):
            raise ValueError("Full dataset needs question_id and patient_fhir_id columns")
        for row in reader:
            question_id, patient = row["question_id"], row["patient_fhir_id"]
            if not question_id or not patient or question_id in patients:
                raise ValueError("Full dataset must have unique nonempty question IDs and nonempty patient IDs")
            patients[question_id] = patient
    if len(patients) != QUESTIONS:
        raise ValueError(f"Full released cohort must contain exactly {QUESTIONS} unique questions")
    return patients


def source_hashes(root=ROOT):
    paths = [root / name for name in ("run_agent.py", "fhir_client.py", "requirements.txt")]
    for directory in ("agent", "tools", "utils"):
        paths.extend((root / directory).rglob("*.py"))
    paths.extend(root / "scripts" / name for name in (
        "full_benchmark.py", "audit_store.py", "compare_runs.py", "analysis.py", "experiment_stats.py", "variability.py"))
    return {str(path.relative_to(root)): file_sha256(path) for path in sorted(paths)}


def arm_command(strategy, args):
    return [sys.executable, str(ROOT / "run_agent.py"), "--agent_strategy", strategy,
            "--model", args.model, "--input", str(args.input),
            "--output", str(args.output_dir / f"full_{strategy}.json"),
            "--limit", str(QUESTIONS), "--max_total_tokens", str(args.max_total_tokens),
            "--max_tool_calls", str(args.max_tool_calls), "--num_processes", str(args.num_processes),
            "--save_interval", str(args.save_interval), "--enable_cache", "true", "--quiet"]


def build_manifest(args, patients):
    import run_agent

    inputs = run_agent.load_inputs(str(args.input), limit=QUESTIONS)
    commands = {strategy: arm_command(strategy, args) for strategy in SYSTEMS}
    identities = {strategy: run_agent.run_identity(run_agent.build_parser().parse_args(command[2:]), inputs)
                  for strategy, command in commands.items()}
    binding_path = args.output_dir / ".local-provider.json"
    binding = json.loads(binding_path.read_text()) if binding_path.exists() else None
    return {"version": 1, "created_at": utc_time(), "cohort": "full_released_2931",
            "first_attempt_only": True, "automatic_retries": False,
            "expected_questions_per_arm": QUESTIONS, "expected_first_attempts": QUESTIONS * len(SYSTEMS),
            "settings": {"model": args.model, "max_total_tokens": args.max_total_tokens,
                         "max_tool_calls": args.max_tool_calls, "input_guard_tokens": INPUT_GUARD,
                         "num_processes_per_arm": args.num_processes, "save_interval": args.save_interval,
                         "enable_cache": True, "heartbeat_seconds": args.heartbeat_seconds,
                         "large_result_heartbeat_seconds": max(args.heartbeat_seconds, LARGE_RESULT_HEARTBEAT),
                         "large_result_threshold_bytes": LARGE_RESULT_BYTES,
                         "stall_seconds": args.stall_seconds},
            "input": {"path": str(args.input), "sha256": file_sha256(args.input),
                      "questions": len(patients), "patients": len(set(patients.values())),
                      "question_ids_sha256": hashlib.sha256(json.dumps(sorted(patients)).encode()).hexdigest()},
            "families": {"path": str(args.families), "sha256": file_sha256(args.families)},
            "source_sha256": source_hashes(), "local_provider_binding": binding,
            "runtime_dependency_versions": run_agent.runtime_dependency_versions(ROOT / "requirements.txt"),
            "commands": commands, "run_identities": identities,
            "comparison_command": [sys.executable, str(ROOT / "scripts/compare_runs.py"), "--full",
                "--input", str(args.input), "--families", str(args.families), "--bootstrap", str(args.bootstrap),
                "--output", str(args.output_dir / "full_comparison.json"),
                *(item for strategy in SYSTEMS for item in ("--arm", f"{strategy}={args.output_dir / ('full_' + strategy + '.json')}"))],
            "interpretation": "Same model, input, budgets, cache, and worker count across all arms. "
                              "Full cohort contains development-exposed questions; architecture improvements "
                              "combine shared tool/contract changes and team changes, not topology alone."}


def require_fresh_output(directory):
    forbidden = [directory / "supervisor_state.json", directory / "full_manifest.json"]
    forbidden.extend(directory / f"full_{strategy}.json{suffix}" for strategy in SYSTEMS
                     for suffix in ("", ".run.json"))
    if any(path.exists() for path in forbidden):
        raise ValueError("This full-run directory already has a manifest, state, or results; use a fresh directory. "
                         "No automatic resume or first-attempt replacement is allowed")


def finite_nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def iter_json_array(path, chunk_chars=1024 * 1024):
    """Stream top-level array records so multi-GB audits do not occupy memory at once."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as stream:
        buffer, position, ended = "", 0, False

        def fill():
            nonlocal buffer, position, ended
            fragment = stream.read(chunk_chars)
            buffer, position = buffer[position:] + fragment, 0
            ended = not fragment

        def whitespace():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or ended:
                    return
                fill()

        whitespace()
        if position >= len(buffer) or buffer[position] != "[":
            raise ValueError("Scheduled result must be a JSON array")
        position += 1
        whitespace()
        if position < len(buffer) and buffer[position] == "]":
            position += 1
        else:
            while True:
                while True:
                    try:
                        record, end = decoder.raw_decode(buffer, position)
                        break
                    except json.JSONDecodeError:
                        if ended:
                            raise ValueError("Scheduled result contains an incomplete JSON record") from None
                        fill()
                position = end
                yield record
                whitespace()
                if position < len(buffer) and buffer[position] == "]":
                    position += 1
                    break
                if position >= len(buffer) or buffer[position] != ",":
                    raise ValueError("Scheduled result has an invalid array separator")
                position += 1
                whitespace()
        whitespace()
        if position < len(buffer):
            raise ValueError("Scheduled result has data after the array")


def iter_result_rows(path, chunk_chars=1024 * 1024):
    """Public streaming reader for offline comparison; raw files remain unchanged."""
    yield from iter_json_array(path, chunk_chars=chunk_chars)


def arm_health(strategy, directory, patients, identity):
    """Load one result at a time; return aggregates and discard the large traces."""
    result_path = directory / f"full_{strategy}.json"
    identity_path = directory / f"full_{strategy}.json.run.json"
    health = {"questions": QUESTIONS, "attempted": 0, "completed": 0, "errors": 0,
              "budget_stops": 0, "limitations": 0, "identity_verified": False,
              "tokens_recorded": 0, "token_unknown_attempts": 0, "provider_usage_unknown_attempts": 0,
              "cost_known_subtotal": 0, "cost_unknown_attempts": 0, "problems": []}
    if identity_path.exists():
        try:
            health["identity_verified"] = json.loads(identity_path.read_text()) == identity
            if not health["identity_verified"]:
                health["problems"].append("run_identity_mismatch")
        except (OSError, ValueError):
            health["problems"].append("run_identity_unreadable")
    if not result_path.exists():
        return health
    if not health["identity_verified"]:
        health["problems"].append("result_without_verified_identity")
    try:
        seen = set()
        for row in iter_result_rows(result_path):
            if not isinstance(row, dict):
                raise ValueError("Invalid scheduled result shape")
            question_id = row.get("question_id")
            if not isinstance(question_id, str) or question_id in seen or question_id not in patients:
                raise ValueError("Invalid scheduled question identities")
            seen.add(question_id)
            if row.get("patient_fhir_id") != patients[row["question_id"]]:
                raise ValueError("Invalid scheduled patient identity")
            if row.get("attempt_history"):
                health["problems"].append("unexpected_retry_history")
            failed = row.get("error") is not None and row.get("error") != ""
            attempted = failed or row.get("latency") is not None or row.get("agent_answer") is not None
            if not attempted:
                continue
            health["attempted"] += 1
            health["errors"] += int(failed)
            health["completed"] += int(not failed)
            health["limitations"] += int(bool(row.get("limitation")))
            budget = row.get("budget") or {}
            health["budget_stops"] += int(bool(budget.get("stopped")))
            tokens = row.get("tokens") or {}
            if finite_nonnegative(tokens.get("total")):
                health["tokens_recorded"] += tokens["total"]
            else:
                health["token_unknown_attempts"] += 1
            health["provider_usage_unknown_attempts"] += int(any(
                call.get("usage_available") is False for call in tokens.get("per_call") or []))
            if finite_nonnegative(tokens.get("cost")):
                health["cost_known_subtotal"] += tokens["cost"]
            else:
                health["cost_unknown_attempts"] += 1
        if len(seen) != QUESTIONS:
            raise ValueError("Scheduled result cohort has the wrong size")
        health["problems"] = sorted(set(health["problems"]))
        health["cost_total"] = None if health["cost_unknown_attempts"] else health["cost_known_subtotal"]
        health["tokens_total"] = None if health["token_unknown_attempts"] or health["provider_usage_unknown_attempts"] else health["tokens_recorded"]
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        health["problems"].append("result_unreadable_or_invalid")
    return health


def source_unchanged(manifest):
    return (source_hashes() == manifest["source_sha256"]
            and file_sha256(manifest["input"]["path"]) == manifest["input"]["sha256"]
            and file_sha256(manifest["families"]["path"]) == manifest["families"]["sha256"])


def stop_owned(children):
    """Terminate only groups created by this supervisor, including their pool workers."""
    alive = [child for child in children.values() if child.poll() is None]
    for child in alive:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    for child in alive:
        try:
            child.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)


class SupervisorStopped(Exception):
    pass


def interrupt(signum, frame):
    raise SupervisorStopped()


def compact_summary(state):
    return {"phase": state["phase"], "all_arms_started": state.get("all_arms_started", False),
            "started_verified": state.get("started_verified", False),
            "all_first_attempts_complete": state.get("all_first_attempts_complete", False),
            "arms": {strategy: {key: detail.get(key) for key in
                ("pid", "process_alive", "return_code", "attempted", "completed", "errors", "budget_stops",
                 "identity_verified", "stall_status", "cost_unknown_attempts", "problems")}
                     for strategy, detail in state.get("arms", {}).items()}}


def monitor(state, children, manifest, patients, progress, cached, args):
    for strategy in SYSTEMS:
        path = args.output_dir / f"full_{strategy}.json"
        identity_path = args.output_dir / f"full_{strategy}.json.run.json"
        stamps = tuple((item.stat().st_mtime_ns, item.stat().st_size) if item.exists() else None
                       for item in (path, identity_path))
        child = children.get(strategy)
        alive = child is not None and child.poll() is None
        if strategy not in cached or cached[strategy][0] != stamps:
            cached[strategy] = (stamps, arm_health(strategy, args.output_dir, patients,
                                                 manifest["run_identities"][strategy]))
        detail = dict(cached[strategy][1])
        prior = progress[strategy]
        if detail["attempted"] > prior["attempted"]:
            prior.update(attempted=detail["attempted"], monotonic=time.monotonic(), at=utc_time())
        elapsed = time.monotonic() - prior["monotonic"]
        detail.update(pid=child.pid if child else None, process_alive=alive,
                      return_code=child.poll() if child else None,
                      started_at=state["arms"].get(strategy, {}).get("started_at"),
                      last_progress_at=prior["at"], seconds_since_progress=round(elapsed),
                      stall_status="stalled_45min" if alive and elapsed >= args.stall_seconds else "normal")
        if child is not None and not alive and not state["arms"].get(strategy, {}).get("finished_at"):
            detail["finished_at"] = utc_time()
        elif state["arms"].get(strategy, {}).get("finished_at"):
            detail["finished_at"] = state["arms"][strategy]["finished_at"]
        state["arms"][strategy] = detail
    state["heartbeat_at"] = utc_time()
    large_result = any((args.output_dir / f"full_{strategy}.json").exists()
                       and (args.output_dir / f"full_{strategy}.json").stat().st_size > LARGE_RESULT_BYTES
                       for strategy in SYSTEMS)
    state["effective_heartbeat_seconds"] = max(args.heartbeat_seconds, LARGE_RESULT_HEARTBEAT) if large_result else args.heartbeat_seconds
    state["all_arms_started"] = all(strategy in children for strategy in SYSTEMS)
    state["started_verified"] = state["all_arms_started"] and all(
        detail["identity_verified"] and detail["attempted"] > 0 and not detail["problems"]
        for detail in state["arms"].values())
    state["all_first_attempts_complete"] = all(
        detail["attempted"] == QUESTIONS and not detail["problems"]
        for detail in state["arms"].values())
    state["attention"] = any(detail["problems"] or detail["stall_status"] != "normal"
                             or (not detail["process_alive"] and detail["pid"] is not None
                                 and detail["attempted"] < QUESTIONS)
                             for detail in state["arms"].values())
    atomic_json(args.output_dir / "supervisor_state.json", state)


def run_supervisor(args, manifest, patients):
    require_fresh_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation makes duplicate starts fail before a child is launched.
    with (args.output_dir / "full_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    children, logs, awake = {}, [], None
    state = {"version": 1, "supervisor_pid": os.getpid(), "started_at": utc_time(), "phase": "starting",
             "automatic_retries": False, "manifest_sha256": file_sha256(args.output_dir / "full_manifest.json"),
             "arms": {strategy: {} for strategy in SYSTEMS}}
    progress = {strategy: {"attempted": 0, "monotonic": time.monotonic(), "at": state["started_at"]}
                for strategy in SYSTEMS}
    cached = {}
    atomic_json(args.output_dir / "supervisor_state.json", state)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        executable = shutil.which("caffeinate")
        if executable:
            awake = subprocess.Popen([executable, "-i", "-w", str(os.getpid())],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     start_new_session=True)
            state["keep_awake_pid"] = awake.pid
            state["keep_awake_policy"] = "Prevent idle sleep only while this supervisor lives"
        for strategy in SYSTEMS:
            log = (args.output_dir / f"full_{strategy}.log").open("a", encoding="utf-8")
            logs.append(log)
            children[strategy] = subprocess.Popen(manifest["commands"][strategy], cwd=ROOT,
                env=environment, stdout=log, stderr=log, start_new_session=True)
            state["arms"][strategy] = {"started_at": utc_time(), "pid": children[strategy].pid}
        state["phase"] = "running"
        while True:
            monitor(state, children, manifest, patients, progress, cached, args)
            if not source_unchanged(manifest) or file_sha256(args.output_dir / "full_manifest.json") != state["manifest_sha256"]:
                state.update(phase="attention_source_changed", attention=True)
                stop_owned(children)
                monitor(state, children, manifest, patients, progress, cached, args)
                state["attention"] = True
                atomic_json(args.output_dir / "supervisor_state.json", state)
                return 2
            if not any(child.poll() is None for child in children.values()):
                break
            time.sleep(state["effective_heartbeat_seconds"])
        if not state["all_first_attempts_complete"]:
            state.update(phase="attention_partial_run", attention=True, finished_at=utc_time())
            atomic_json(args.output_dir / "supervisor_state.json", state)
            return 2
        state.update(phase="analyzing", finished_at=utc_time())
        atomic_json(args.output_dir / "supervisor_state.json", state)
        with (args.output_dir / "full_analysis.log").open("a", encoding="utf-8") as log:
            analysis = subprocess.Popen(manifest["comparison_command"], cwd=ROOT,
                env=environment, stdout=log, stderr=log, start_new_session=True)
            children["comparison"] = analysis
            state["analysis_pid"] = analysis.pid
            atomic_json(args.output_dir / "supervisor_state.json", state)
            while True:
                try:
                    state["analysis_return_code"] = analysis.wait(timeout=args.heartbeat_seconds)
                    break
                except subprocess.TimeoutExpired:
                    state["heartbeat_at"] = utc_time()
                    atomic_json(args.output_dir / "supervisor_state.json", state)
        errors = sum(detail["errors"] + int(detail["return_code"] != 0) for detail in state["arms"].values())
        state.update(phase=("attention_analysis_failed" if state["analysis_return_code"] else
                            "complete_with_errors" if errors else "complete"),
                     attention=bool(state["analysis_return_code"]), analysis_finished_at=utc_time())
        summary = compact_summary(state)
        if not state["analysis_return_code"]:
            report = json.loads((args.output_dir / "full_comparison.json").read_text())
            summary["accuracy"] = {name: detail["accuracy"]["all"] for name, detail in report["arms"].items()}
            summary["comparisons"] = report["comparisons"]
        atomic_json(args.output_dir / "full_summary.json", summary)
        atomic_json(args.output_dir / "supervisor_state.json", state)
        print(json.dumps(summary, indent=2))
        return 0 if not state["attention"] else 2
    except SupervisorStopped:
        stop_owned(children)
        monitor(state, children, manifest, patients, progress, cached, args)
        state.update(phase="interrupted", attention=True, finished_at=utc_time())
        atomic_json(args.output_dir / "supervisor_state.json", state)
        return 130
    except Exception as error:
        stop_owned(children)
        state.update(phase="attention_supervisor_failure", attention=True, error_type=type(error).__name__,
                     finished_at=utc_time())
        atomic_json(args.output_dir / "supervisor_state.json", state)
        print("Full benchmark supervisor needs attention; inspect aggregate state and local logs.", file=sys.stderr)
        return 2
    finally:
        if awake is not None and awake.poll() is None:
            awake.terminate()
            try:
                awake.wait(timeout=5)
            except subprocess.TimeoutExpired:
                awake.kill()
                awake.wait(timeout=5)
        for log in logs:
            log.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "status"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "final_dataset/questions_answers_sql_fhir.csv")
    parser.add_argument("--families", type=Path, default=ROOT / "manifests/question_families.csv")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--max_total_tokens", type=int, default=200000)
    parser.add_argument("--max_tool_calls", type=int, default=30)
    parser.add_argument("--num_processes", type=int, default=2)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--heartbeat_seconds", type=int, default=30)
    parser.add_argument("--stall_seconds", type=int, default=45 * 60)
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    for name in ("max_total_tokens", "max_tool_calls", "num_processes", "save_interval",
                 "heartbeat_seconds", "stall_seconds", "bootstrap"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    args.output_dir, args.input, args.families = (path.resolve() for path in (args.output_dir, args.input, args.families))
    if args.command == "status":
        path = args.output_dir / "supervisor_state.json"
        if not path.exists():
            print(json.dumps({"phase": "not_started"}))
            return 0
        print(json.dumps(compact_summary(json.loads(path.read_text())), indent=2))
        return 0
    patients = dataset_cohort(args.input)
    manifest = build_manifest(args, patients)
    if args.command == "plan":
        print(json.dumps({key: manifest[key] for key in ("cohort", "expected_first_attempts", "settings",
                         "input", "families", "commands", "comparison_command", "interpretation")}, indent=2))
        return 0
    return run_supervisor(args, manifest, patients)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("Full benchmark setup could not validate the cohort, frozen inputs, or fresh output directory "
              f"({type(error).__name__}).", file=sys.stderr)
        raise SystemExit(2)
