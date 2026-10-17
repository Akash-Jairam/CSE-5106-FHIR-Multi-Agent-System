"""Frozen V3 full benchmark with answer-blind infrastructure recovery.

The first runtime-clean terminal attempt is retained, including incorrect answers,
abstentions and intrinsic design errors. Infrastructure attempts remain immutable
under attempts/ and never enter accuracy, accepted latency, or accepted cost.
No scorer or reference answer participates in scheduling or recovery.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.full_benchmark import atomic_json, dataset_cohort, file_sha256, iter_result_rows, stop_owned
from scripts.audit_store import restore_result_audit, ARTIFACT_PATTERN

ARMS = {"direct_llm": "direct_llm", "single_agent": "single_agent", "multi_agent": "multi_agent_v3"}
CLASSIFIER_VERSION = "v3-infrastructure-v4"
PROVIDER_REJECTION_POLICY = {
    "version": "azure-prompt-rejection-v1", "category": "provider_policy_rejection",
    "max_rejections_per_question": 2, "retry_request": "unchanged",
    "persistent_action": "park_pending_then_attention", "scope": ["single_agent"]}
SDK_ERROR_PREFIX = "Max retries exceeded: Error code: 400 - "
MAX_ERROR_CHARS = 8192
PROVIDER_JWT_REJECTION = "Max retries exceeded: Jwt verification fails"
RUNTIME = re.compile(
    r"timed?\s*out|time[ -]?out|(?:HTTP(?:StatusError)?[ :]*|status(?:_code| code)?[ =:]*|error code[ :]*)(?:408|429|50[0234]|52[024])\b|"
    r"(?:^|\()(?:408|429|50[0234]) (?:Client|Server) Error|bad gateway|EXCEPTION: Empty or None output\.|"
    r"APIConnectionError|APITimeoutError|ConnectError|ReadError|ReadTimeout|ConnectTimeout|"
    r"PoolTimeout|RemoteProtocolError|TransportError|SSLError|certificate verify failed|"
    r"nodename nor servname provided|ConnectionError|connection (?:error|failure|reset|refused|closed|aborted)|"
    r"cannot send a request.*client.*closed|client has been closed|transport.*closed|"
    r"server disconnected|temporary failure in name resolution|name or service not known|"
    r"network (?:is unreachable|error)|service unavailable|too many requests|"
    r"rate[ -]?limit|resource exhausted|upstream.*(?:unavailable|reset)|"
    r"backend.*unavailable|internal server error|^Python execution exceeded [0-9.]+ seconds$|"
    r"worker.*(?:unavailable|failed|died|crashed)|sandbox.*unavailable|"
    r"^Python execution failed: (?:OSError|PermissionError|FileNotFoundError)$", re.I)
INTRINSIC = re.compile(
    r"budget|allocation|iteration limit|max(?:imum)? iterations|did not finalize|"
    r"no final answer|no tool calls found|input tokens exceeded|context (?:length|window)|"
    r"invalid output|invalid.*contract|output.*contract|source[_ -]check|"
    r"source[- _]valid|execution[- _]contract|final.*computation|"
    r"NameError|TypeError|KeyError|IndexError|ValueError|SyntaxError|ZeroDivisionError|"
    r"assertion|sandbox|python.*(?:execution|error)|unsupported.*(?:tool|operation)|"
    r"malformed|invalid JSON|JSONDecodeError|empty or None output", re.I)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def diagnostic_errors(value, path="revision"):
    """Collect explicit error fields, excluding prose, answers and executable code."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"answer", "agent_answer", "draft", "primary", "final", "code", "content", "messages", "trace", "contract", "audit"}:
                continue
            location = f"{path}.{key}"
            if key == "reason" and isinstance(child, str) and child.startswith("targeted check failed:"):
                yield location, child
                continue
            if key == "action" and isinstance(child, str) and child.startswith("source_check:"):
                yield location, child
                continue
            if key in {"issues", "check_issues", "final_check_issues"} and isinstance(child, list):
                for item in child:
                    if isinstance(item, str) and item.startswith("source_check: could not verify the original records ("):
                        yield location, item
            if key in {"error", "errors", "contract_error", "audit_error"}:
                for item in child if isinstance(child, list) else [child]:
                    if isinstance(item, str) and item.strip():
                        yield location, item
                    elif isinstance(item, (dict, list)):
                        yield from diagnostic_errors(item, location)
            elif isinstance(child, (dict, list)):
                yield from diagnostic_errors(child, location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from diagnostic_errors(child, f"{path}[{index}]")


def intrinsic_tool_validation(error):
    """Recognize fixed host validation templates before examining echoed values.

    Tool names and argument keys are model-controlled, so their spelling cannot
    establish that an actual timeout or network request occurred.
    """
    fixed = {"Arguments must be a JSON object", "Unsupported FHIR resource type",
             "A non-empty FHIR resource ID is required", "Retrieve only the patient identified in this question",
             "citations must be a list", "mode must be auto, inspect, or final", "timeout must be positive"}
    if error in fixed:
        return True
    tools = (r"(?:get_resources_by_patient_fhir_id|get_resources_by_resource_id|"
             r"get_all_resources_by_patient_fhir_id|execute_python_code|check_fields|fhir_request_get)")
    return bool(re.fullmatch(
        r"Unknown arguments: [\s\S]+|Tool '[\s\S]*' is not available to this role|"
        rf"(?:TypeError: )?{tools}\(\) got an unexpected keyword argument '[\s\S]+'|"
        rf"(?:TypeError: )?{tools}\(\) missing [1-9][0-9]* required (?:positional|keyword-only) arguments?: [\s\S]+|"
        r"(?:JSONDecodeError: )?(?:Expecting (?:value|property name enclosed in double quotes|',' delimiter|':' delimiter)|"
        r"Unterminated string starting at|Invalid control character at|Extra data|Invalid \\escape|Invalid \\uXXXX escape)"
        r": line [1-9][0-9]* column [1-9][0-9]* \(char [0-9]+\)", error))


def provider_jwt_paths(row):
    """Require the exact host model-call error and an unaccounted failed call."""
    tokens = row.get("tokens")
    if not isinstance(tokens, dict) or not any(
            isinstance(call, dict) and call.get("usage_available") is False
            for call in tokens.get("per_call") or []):
        return []
    signals = [("error", row.get("error"))]
    revision = row.get("revision")
    if isinstance(revision, dict):
        signals.extend((f"revision.{name}", revision.get(name))
                       for name in ("contract_error", "audit_error"))
        for index, phase in enumerate(revision.get("phases") or []):
            if not isinstance(phase, dict):
                continue
            errors = phase.get("errors")
            if isinstance(errors, list):
                signals.extend((f"revision.phases[{index}].errors", error)
                               for error in errors)
    return [path for path, error in signals if error == PROVIDER_JWT_REJECTION]


def is_observed_provider_prompt_rejection(row):
    """Recognize only the observed SDK-owned rejected prompt, without answers."""
    if not isinstance(row, dict):
        return False
    error = row.get("error")
    tokens = row.get("tokens")
    calls = tokens.get("per_call") if isinstance(tokens, dict) else None
    if not (isinstance(error, str) and error.startswith(SDK_ERROR_PREFIX)
            and len(error) <= MAX_ERROR_CHARS and isinstance(calls, list) and calls
            and isinstance(calls[-1], dict) and calls[-1].get("usage_available") is False):
        return False
    try:
        payload = ast.literal_eval(error[len(SDK_ERROR_PREFIX):])
    except (ValueError, SyntaxError, TypeError, RecursionError):
        return False
    if not (isinstance(payload, dict) and set(payload) == {"error"} and isinstance(payload["error"], dict)):
        return False
    provider_error = payload["error"]
    if not (provider_error.get("code") == "content_filter" and provider_error.get("param") == "prompt"
            and type(provider_error.get("status")) is int and provider_error["status"] == 400):
        return False
    inner = provider_error.get("innererror")
    if not (isinstance(inner, dict) and inner.get("code") == "ResponsibleAIPolicyViolation"):
        return False
    filters = inner.get("content_filter_result")
    jailbreak = filters.get("jailbreak") if isinstance(filters, dict) else None
    return (isinstance(jailbreak, dict) and jailbreak.get("detected") is True
            and jailbreak.get("filtered") is True)


def classify_attempt(row, audit_dir=None, allow_provider_rejection=False):
    """Return pending/runtime/accepted/attention without access to gold or scoring.

    Unknown top-level errors halt. Nested model/debug errors are intrinsic unless
    an explicit infrastructure signature is present. Execution timeouts are
    retried too; persistent timeouts stop with attention rather than becoming wrong.
    """
    if row.get("attempt_history"):
        return {"status": "attention", "reason": "unexpected_attempt_history"}
    top = row.get("error")
    top = top if isinstance(top, str) and top.strip() else None
    latency = row.get("latency")
    if top is None and latency is None and row.get("agent_answer") is None:
        return {"status": "pending", "reason": "unattempted"}
    if latency is not None and (type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0):
        return {"status": "attention", "reason": "invalid_latency"}
    full = restore_result_audit(row, audit_dir, restore_history=False) if audit_dir is not None else row
    if allow_provider_rejection and is_observed_provider_prompt_rejection(full):
        return {"status": "runtime", "reason": "provider_policy_rejection",
                "category": "provider_policy_rejection", "diagnostic_paths": ["error"]}
    signals = [("error", top)] if top else []
    external_unknown = []
    signals.extend(diagnostic_errors(full.get("revision") or {}))
    # Only actual tool replies are inspected; user/assistant text can discuss
    # timeouts without an infrastructure event having happened.
    for index, message in enumerate(full.get("trace") or []):
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        content = message.get("content")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (ValueError, TypeError):
                continue
        if isinstance(content, dict):
            for location, error in diagnostic_errors(content, f"trace[{index}]"):
                if "python" in str(message.get("name", "")).lower() and not re.match(
                        r"^Python (?:execution exceeded [0-9.]+ seconds$|sandbox (?:unavailable:|worker failed$)|execution failed:)", error):
                    continue
                if intrinsic_tool_validation(error):
                    continue
                signals.append((location, error))
                if "python" not in str(message.get("name", "")).lower() and not (RUNTIME.search(error) or INTRINSIC.search(error)):
                    external_unknown.append(location)
    ambiguous_arguments = [path for path, error in signals if not intrinsic_tool_validation(error) and re.fullmatch(
        r"(?:Max retries exceeded: )?(?:TypeError: )?[A-Za-z_][A-Za-z0-9_.<>]*\(\) got an unexpected keyword argument '[\s\S]+'", error)]
    if ambiguous_arguments:
        return {"status": "attention", "reason": "unclassified_argument_failure", "diagnostic_paths": ambiguous_arguments}
    runtime_paths = [path for path, error in signals if not intrinsic_tool_validation(error) and RUNTIME.search(error) and not re.match(
        r"^(?:KeyError|IndexError|ValueError|TypeError|NameError|SyntaxError|ZeroDivisionError):", error)]
    runtime_paths.extend(provider_jwt_paths(full))
    if runtime_paths:
        return {"status": "runtime", "reason": "allowlisted_infrastructure_failure", "diagnostic_paths": runtime_paths}
    ambiguous_host = [path for path, error in signals if error.startswith("targeted check failed:") and "HTTPError" in error]
    if external_unknown or ambiguous_host:
        return {"status": "attention", "reason": "unclassified_external_failure", "diagnostic_paths": external_unknown + ambiguous_host}
    if top and not (INTRINSIC.search(top) or intrinsic_tool_validation(top)):
        return {"status": "attention", "reason": "unclassified_terminal_error", "error_sha256": hashlib.sha256(top.encode()).hexdigest()}
    if latency is None:
        # A crashed outer invocation has no attempt latency; do not invent it.
        return {"status": "attention", "reason": "terminal_attempt_missing_latency"}
    tokens = row.get("tokens")
    if not isinstance(tokens, dict):
        return {"status": "attention", "reason": "terminal_attempt_missing_usage"}
    total = tokens.get("total")
    if type(total) not in (int, float) or not math.isfinite(total) or total < 0:
        return {"status": "attention", "reason": "invalid_terminal_token_usage"}
    if any(call.get("usage_available") is False for call in tokens.get("per_call") or []):
        return {"status": "attention", "reason": "unverified_provider_usage"}
    return {"status": "accepted", "reason": "intrinsic_terminal_error" if top else "runtime_clean_terminal"}


def source_hashes(root=ROOT):
    names = [root / name for name in ("run_agent.py", "fhir_client.py", "requirements.txt")]
    for directory in ("agent", "tools", "utils"):
        names.extend((root / directory).rglob("*.py"))
    names.extend(root / "scripts" / name for name in
                 ("audit_store.py", "full_benchmark.py", "v3_full_run.py", "v3_full_analysis.py", "launch_v3_full.py"))
    return {str(path.relative_to(root)): file_sha256(path) for path in sorted(set(names)) if path.is_file()}


def pinned_files(directory):
    names = [directory / name for name in ("provider_profile.json", "pricing.json", "split.json", "preflight.json", "protocol.txt", "baseline_decision.txt")]
    names.extend((directory / "runtime").glob("*.py"))
    return {str(path.relative_to(directory)): file_sha256(path) for path in sorted(names) if path.is_file()}


def runner_command(arm, settings, input_path, output_path, count):
    return [sys.executable, str(ROOT / "run_agent.py"), "--agent_strategy", ARMS[arm],
            "--model", settings["model"], "--input", str(input_path), "--output", str(output_path),
            "--limit", str(count), "--max_total_tokens", str(settings["max_total_tokens"]),
            "--max_tool_calls", str(settings["max_tool_calls"]),
            "--num_processes", str(settings["num_processes"]), "--save_interval", "1",
            "--audit_storage", "gzip", "--enable_cache", "true", "--quiet"]


def manifest_for(args):
    import run_agent
    patients = dataset_cohort(args.input)
    inputs = run_agent.load_inputs(str(args.input))
    settings = {name: getattr(args, name) for name in ("model", "max_total_tokens", "max_tool_calls", "num_processes",
                "heartbeat_seconds", "stall_seconds", "min_free_gb", "max_runtime_retries", "retry_delay_seconds", "batch_size")}
    settings.update(save_interval=1, audit_storage="gzip", enable_cache=True)
    identities = {}
    for arm in ARMS:
        command = runner_command(arm, settings, args.input, args.run_dir / f"full_{arm}.json", len(patients))
        identities[arm] = run_agent.run_identity(run_agent.build_parser().parse_args(command[2:]), inputs)
    return {"version": 1, "created_at": utc_now(), "protocol": "v3-first-runtime-clean-terminal",
            "classifier_version": CLASSIFIER_VERSION, "project_root": str(ROOT), "arms": ARMS,
            "questions_per_arm": len(patients), "expected_accepted_questions": len(patients) * len(ARMS),
            "input": {"path": str(args.input), "sha256": file_sha256(args.input), "questions": len(patients),
                      "patients": len(set(patients.values())),
                      "question_ids_sha256": hashlib.sha256(json.dumps(sorted(patients)).encode()).hexdigest()},
            "families": {"path": str(args.families), "sha256": file_sha256(args.families)},
            "settings": settings, "source_sha256": source_hashes(), "pinned_files_sha256": pinned_files(args.run_dir),
            "split_manifest": {"path": str(ROOT / "manifests/split.json"), "sha256": file_sha256(ROOT / "manifests/split.json")},
            "offline_fhir_dir": str(ROOT / ".local-setup/phase2-fhir") if (ROOT / ".local-setup/phase2-fhir").is_dir() else None,
            "runtime_dependency_versions": run_agent.runtime_dependency_versions(ROOT / "requirements.txt"),
            "run_identities": identities,
            "policy": {"selection": "First infrastructure-clean terminal attempt per scheduled question; no answer-based reruns",
                       "infrastructure_attempts": "Preserved separately, excluded from accuracy and accepted time/cost",
                       "intrinsic_failures": "Retained as final, including budgets, invalid model output, source gates and solver limits",
                       "unknown_failures": "Halt with attention; never impute an incorrect outcome",
                       "partial_processes": "Preserve saved attempts; unsaved work remains pending",
                       "no_scorer_in_scheduling": True, "schedule": "team_then_controls",
                       "circuit_breaker": {"minimum_recent_failures": 8, "recent_window": 10, "minimum_failure_fraction": 0.8}},
            "analysis_command": [sys.executable, str(ROOT / "scripts/v3_full_analysis.py"), "--run-dir", str(args.run_dir)]}


def validate_manifest(manifest, directory):
    import run_agent
    policy = manifest.get("provider_rejection_policy")
    if "provider_rejection_policy" in manifest and (policy != PROVIDER_REJECTION_POLICY or manifest.get("selected_arms") != ["single_agent"]):
        raise ValueError("Selected provider-rejection policy is invalid or outside its approved scope")
    if manifest.get("classifier_version") != CLASSIFIER_VERSION or manifest.get("arms") != ARMS:
        raise ValueError("Frozen run protocol changed")
    if manifest.get("project_root") != str(ROOT) or manifest.get("source_sha256") != source_hashes():
        raise ValueError("Frozen source changed")
    for key in ("input", "families", "split_manifest"):
        if file_sha256(manifest[key]["path"]) != manifest[key]["sha256"]:
            raise ValueError("Frozen input changed")
    if manifest.get("pinned_files_sha256") != pinned_files(directory):
        raise ValueError("Frozen provider/runtime/pricing files changed")
    if manifest["runtime_dependency_versions"] != run_agent.runtime_dependency_versions(ROOT / "requirements.txt"):
        raise ValueError("Frozen runtime dependencies changed")
    validate_provider(directory)


def validate_provider(directory):
    """Use the normal runtime loader internally; compare public values only."""
    metadata_path = directory / "provider_profile.json"
    selected = os.environ.get("FHIR_V3_PROVIDER_PROFILE_PATH")
    if not selected or Path(selected).resolve() != metadata_path.resolve() or not metadata_path.is_file():
        raise ValueError("Frozen provider profile environment is absent or mismatched")
    import azure_adapter
    if Path(azure_adapter.__file__).resolve() != (directory / "runtime/azure_adapter.py").resolve():
        raise ValueError("Frozen provider adapter was not bootstrapped")
    config_path = os.environ.get("FHIR_LOCAL_AZURE_CONFIG")
    if not config_path:
        raise ValueError("Frozen provider configuration environment is absent")
    expected = json.loads(metadata_path.read_text())["public_profile"]
    current = azure_adapter.AzureSettings.load(config_path).public()
    current.pop("api_key_configured", None)
    if current != expected:
        raise ValueError("Frozen public provider settings changed")


def event(directory, kind, **details):
    with (directory / "operations.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": utc_now(), "event": kind, **details}, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def process_alive(pid):
    if type(pid) is not int or pid < 1:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def copy_audit(row, origin, directory):
    ref = row.get("audit_artifacts")
    if not isinstance(ref, dict) or not ARTIFACT_PATTERN.fullmatch(ref.get("path", "")):
        raise ValueError("Terminal attempt missing owned gzip audit")
    source, destination = origin / ref["path"], directory / ref["path"]
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        if file_sha256(destination) != file_sha256(source):
            raise ValueError("Conflicting accepted audit artifact")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def attempt_rounds(directory, arm):
    root = directory / "attempts" / arm
    return sorted(root.glob("round-*.batch.json"), key=lambda path: int(path.name.split("-")[1].split(".")[0]))


def classifier_for_manifest(manifest, classifier=classify_attempt):
    """The narrow policy extension requires an explicit frozen scope opt-in."""
    policy = manifest.get("provider_rejection_policy")
    if "provider_rejection_policy" not in manifest:
        return classifier
    if policy != PROVIDER_REJECTION_POLICY or manifest.get("selected_arms") != ["single_agent"]:
        raise ValueError("Selected provider-rejection policy is absent, invalid, or outside its approved scope")
    if classifier is classify_attempt:
        return lambda row, audit_dir=None: classify_attempt(row, audit_dir, allow_provider_rejection=True)
    return classifier


def collect_accepted(directory, manifest, patients, classifier=classify_attempt, arms=None, provider_rejections=None):
    """Rebuild accepted rows only from sealed immutable batches in execution order."""
    import run_agent
    classifier = classifier_for_manifest(manifest, classifier)
    if provider_rejections is None and "provider_rejection_policy" in manifest:
        provider_rejections = {}
    if provider_rejections is not None:
        provider_rejections.clear()
        provider_rejections.update({arm: {} for arm in ARMS})
    accepted, choices, retries, attention = {arm: {} for arm in ARMS}, {arm: {} for arm in ARMS}, {arm: {} for arm in ARMS}, []
    counts = {arm: {"raw_attempts": 0, "runtime_failures": 0, "intrinsic_errors": 0} for arm in ARMS}
    for arm in ARMS if arms is None else arms:
        for batch_path in attempt_rounds(directory, arm):
            batch = json.loads(batch_path.read_text())
            seal = batch_path.with_suffix(".seal.json")
            if not seal.exists():
                continue
            sealed = json.loads(seal.read_text())
            output = Path(batch["output"])
            identity_file = Path(str(output) + ".run.json")
            if (sealed.get("batch_sha256") != file_sha256(batch_path)
                    or sealed.get("identity_sha256") != (file_sha256(identity_file) if identity_file.exists() else None)
                    or sealed.get("result_sha256") != (file_sha256(output) if output.exists() else None)):
                raise ValueError("Sealed attempt batch changed")
            if "provider_rejection_policy" in manifest and any(
                    provider_rejections[arm].get(qid, 0) >= manifest["provider_rejection_policy"]["max_rejections_per_question"]
                    for qid in batch["question_ids"]):
                raise ValueError("Selected provider-rejection policy forbids scheduling a parked question")
            if not output.exists():
                continue
            if not identity_file.exists() or json.loads(identity_file.read_text()) != batch["run_identity"]:
                raise ValueError("Attempt run identity mismatch")
            if file_sha256(batch["input"]) != batch["input_sha256"]:
                raise ValueError("Attempt shard input changed")
            selected, seen = set(batch["question_ids"]), set()
            for row in iter_result_rows(output):
                qid = row.get("question_id")
                if qid in seen or qid not in selected or row.get("patient_fhir_id") != patients.get(qid):
                    raise ValueError("Attempt question/patient identities changed")
                seen.add(qid)
                if not run_agent.is_attempted(row):
                    continue
                if qid in accepted[arm]:
                    raise ValueError("Completed accepted question was rerun")
                decision = classifier(row, output.parent)
                counts[arm]["raw_attempts"] += 1
                if decision["status"] == "runtime":
                    counts[arm]["runtime_failures"] += 1
                    retries[arm][qid] = retries[arm].get(qid, 0) + 1
                    if provider_rejections is not None and decision.get("category") == "provider_policy_rejection":
                        provider_rejections[arm][qid] = provider_rejections[arm].get(qid, 0) + 1
                elif decision["status"] == "accepted":
                    copy_audit(row, output.parent, directory)
                    row = {key: value for key, value in row.items() if key != "attempt_history"}
                    accepted[arm][qid] = row
                    choices[arm][qid] = {"attempt_file": str(output.relative_to(directory)),
                                        "round": batch["round"], "runtime_failures_before": retries[arm].get(qid, 0),
                                        "selection": decision["reason"], "runtime_clean": True}
                    counts[arm]["intrinsic_errors"] += int(bool(row.get("error")))
                elif decision["status"] == "attention":
                    attention.append({"arm": arm, "question_id": qid, "attempt_file": str(output.relative_to(directory)), **decision})
            if seen != selected:
                raise ValueError("Saved attempt file omitted scheduled placeholders")
    return accepted, choices, retries, attention, counts


def save_accepted(directory, manifest, accepted, choices, counts, arms=None, cohort=None):
    if arms is None:
        ledger = {"version": 1, "protocol": manifest["protocol"],
                  "selection": "first_runtime_clean_terminal", "counts": counts, "arms": choices}
    else:
        # A scoped resume owns only these slices. In particular, replaying a
        # deferred arm must never discard its attention or overwrite its results.
        if cohort is None:
            raise ValueError("Selected persistence requires the original question cohort")
        ledger = read_acceptance(directory, manifest)
        for arm in arms:
            ledger["counts"][arm], ledger["arms"][arm] = counts[arm], choices[arm]
    for arm in accepted if arms is None else arms:
        indexed = accepted[arm]
        atomic_json(directory / f"full_{arm}.json", [indexed[qid] for qid in sorted(indexed)])
        # The identity binds the authoritative complete cohort. Raw shard identities
        # remain untouched and the manifest explicitly records this consolidation.
        atomic_json(directory / f"full_{arm}.json.run.json", manifest["run_identities"][arm])
    if arms is None:
        ledger["complete"] = all(len(rows) == manifest["questions_per_arm"] for rows in accepted.values())
    else:
        ledger["complete"] = all(set(ledger["arms"][arm]) == set(cohort) for arm in ARMS)
    atomic_json(directory / "acceptance.json", ledger)


def execution_arms(manifest, only_arms=None, resume=False):
    """Require both an explicit request and a matching frozen scheduling scope."""
    pinned = manifest.get("selected_arms")
    if pinned is None and only_arms is None:
        if "provider_rejection_policy" in manifest:
            raise ValueError("Selected provider-rejection policy cannot enable an unscoped run")
        return tuple(ARMS)
    if not resume or pinned != ["single_agent"] or only_arms != ["single_agent"]:
        raise ValueError("Selected arms require resume with --only-arm single_agent and an identical manifest pin")
    if "provider_rejection_policy" in manifest and manifest["provider_rejection_policy"] != PROVIDER_REJECTION_POLICY:
        raise ValueError("Selected provider-rejection policy requires the exact explicit manifest opt-in")
    return ("single_agent",)


def persist_provider_rejections(directory, manifest, accepted, rejections, arms):
    """Derived operational state; immutable sealed attempts remain authoritative."""
    limit = manifest["provider_rejection_policy"]["max_rejections_per_question"]
    parked = {arm: {qid for qid, count in rejections[arm].items()
                    if count >= limit and qid not in accepted[arm]} for arm in arms}
    document = {"version": 1, "classifier_version": CLASSIFIER_VERSION,
                "policy": manifest["provider_rejection_policy"], "updated_at": utc_now(),
                "source": "reconstructed_from_sealed_immutable_attempts",
                "rejections": {arm: rejections[arm] for arm in arms},
                "parked_question_ids": {arm: sorted(parked[arm]) for arm in arms},
                "pending_question_ids": {arm: sorted(qid for qid in rejections[arm] if qid not in accepted[arm]) for arm in arms}}
    atomic_json(directory / "provider_rejections.json", document)
    return parked, document


def read_acceptance(directory, manifest):
    ledger = json.loads((directory / "acceptance.json").read_text())
    if (ledger.get("version") != 1 or ledger.get("protocol") != manifest["protocol"]
            or ledger.get("selection") != "first_runtime_clean_terminal"
            or not isinstance(ledger.get("arms"), dict) or set(ledger["arms"]) != set(ARMS)
            or not isinstance(ledger.get("counts"), dict) or set(ledger["counts"]) != set(ARMS)
            or any(not isinstance(ledger[section][arm], dict) for section in ("arms", "counts") for arm in ARMS)):
        raise ValueError("Selected resume requires the original three-arm acceptance ledger")
    return ledger


def validate_completed_v3(directory, manifest, patients):
    """Read-only prerequisite for scheduling the separately requested baseline."""
    ledger = read_acceptance(directory, manifest)
    cohort = set(patients)
    if (len(cohort) != manifest["questions_per_arm"]
            or set(ledger["arms"]["multi_agent"]) != cohort
            or any(not set(ledger["arms"][arm]).issubset(cohort) for arm in ARMS)):
        raise ValueError("Selected resume requires the exact completed V3 cohort")
    identity_path = directory / "full_multi_agent.json.run.json"
    if json.loads(identity_path.read_text()) != manifest["run_identities"]["multi_agent"]:
        raise ValueError("Selected resume requires the original completed V3 identity")
    seen = set()
    for row in iter_result_rows(directory / "full_multi_agent.json"):
        qid = row.get("question_id")
        if qid not in cohort or qid in seen or row.get("patient_fhir_id") != patients[qid] or row.get("attempt_history"):
            raise ValueError("Selected resume requires unchanged completed V3 question/patient identities")
        choice = ledger["arms"]["multi_agent"][qid]
        if (not isinstance(choice, dict) or choice.get("runtime_clean") is not True
                or choice.get("selection") not in {"runtime_clean_terminal", "intrinsic_terminal_error"}):
            raise ValueError("Selected resume requires runtime-clean completed V3 selections")
        seen.add(qid)
    if seen != cohort:
        raise ValueError("Selected resume requires the exact completed V3 canonical cohort")
    return ledger


def create_batch(directory, arm, pending, manifest, number):
    import run_agent
    root = directory / "attempts" / arm
    root.mkdir(parents=True, exist_ok=True)
    input_path, output_path = root / f"round-{number:04d}.csv", root / f"round-{number:04d}.json"
    batch_path = root / f"round-{number:04d}.batch.json"
    if any(path.exists() for path in (input_path, output_path, batch_path)):
        raise ValueError("Attempt batch path already exists")
    with open(manifest["input"]["path"], newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        with input_path.open("x", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row["question_id"] in pending:
                    writer.writerow(row)
    command = runner_command(arm, manifest["settings"], input_path, output_path, len(pending))
    inputs = run_agent.load_inputs(str(input_path))
    batch = {"version": 1, "arm": arm, "round": number, "created_at": utc_now(),
             "question_ids": inputs["question_id"].tolist(), "input": str(input_path),
             "input_sha256": file_sha256(input_path), "output": str(output_path), "command": command,
             "run_identity": run_agent.run_identity(run_agent.build_parser().parse_args(command[2:]), inputs)}
    atomic_json(batch_path, batch)
    return batch_path, batch


def seal_batch(batch_path, return_code, reason):
    batch = json.loads(batch_path.read_text())
    output = Path(batch["output"])
    seal = batch_path.with_suffix(".seal.json")
    if seal.exists():
        return
    atomic_json(seal, {"sealed_at": utc_now(), "batch_sha256": file_sha256(batch_path),
                      "result_sha256": file_sha256(output) if output.exists() else None,
                      "identity_sha256": file_sha256(str(output) + ".run.json") if Path(str(output) + ".run.json").exists() else None,
                      "return_code": return_code, "reason": reason})


def recover_unsealed(directory, arms=None):
    """A prior interrupted supervisor must have no surviving owned children."""
    state_file = directory / "supervisor_state.json"
    prior = json.loads(state_file.read_text()) if state_file.exists() else {}
    if process_alive(prior.get("supervisor_pid")) and prior.get("supervisor_pid") != os.getpid():
        raise ValueError("Prior supervisor is still running")
    for detail in prior.get("active", {}).values():
        if process_alive(detail.get("pid")):
            raise ValueError("Prior owned child is still running; inspect before resuming")
    for arm in ARMS if arms is None else arms:
        for batch_path in attempt_rounds(directory, arm):
            seal_batch(batch_path, None, "recovered_after_interruption")


class StopRequested(Exception):
    pass


def stop_signal(signum, frame):
    raise StopRequested()


def compact_state(state):
    return {key: state.get(key) for key in ("phase", "heartbeat_at", "supervisor_pid", "started_at", "resumed_at", "finished_at",
            "questions_per_arm", "accepted", "counts", "active", "attention", "reason", "safe_message", "analysis_return_code",
            "selected_arms", "scope_started_at", "selected_scope_complete", "benchmark_complete", "pending_scoped_export", "analysis_status",
            "provider_rejections")}


def eligible_arms(accepted, count):
    """Complete the requested V3 run before starting fresh matched controls."""
    return ("multi_agent",) if len(accepted["multi_agent"]) < count else ("single_agent", "direct_llm")


def run_supervisor(directory, manifest, patients, resume=False, classifier=classify_attempt, only_arms=None):
    arms = execution_arms(manifest, only_arms, resume)
    selected = only_arms is not None
    provider_policy = "provider_rejection_policy" in manifest
    classifier = classifier_for_manifest(manifest, classifier)
    provider_rejections = {} if provider_policy else None
    parked = {arm: set() for arm in arms}
    settings = manifest["settings"]
    lock = (directory / ".supervisor.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError("This run already has an active supervisor")
    children, logs, batches, progress = {}, {}, {}, {}
    awake = None
    prior_state_path = directory / "supervisor_state.json"
    prior_state = json.loads(prior_state_path.read_text()) if resume and prior_state_path.exists() else {}
    state = {"version": 1, "supervisor_pid": os.getpid(), "started_at": prior_state.get("started_at") or utc_now(),
             "phase": "starting", "questions_per_arm": len(patients), "active": {}, "attention": False}
    if resume:
        state["resumed_at"] = utc_now()
    if selected:
        state.update(selected_arms=list(arms), selected_scope_complete=False, benchmark_complete=False,
                     pending_scoped_export=False, analysis_status="not_started",
                     scope_started_at=(prior_state.get("scope_started_at") if prior_state.get("selected_arms") == list(arms) else None) or utc_now())
    manifest_hash = file_sha256(directory / "run_manifest.json")
    handlers = {sig: signal.signal(sig, stop_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        validate_manifest(manifest, directory)
        retained = validate_completed_v3(directory, manifest, patients) if selected else None
        if resume:
            recover_unsealed(directory, arms if selected else None)
        accepted, choices, retries, attention, counts = collect_accepted(directory, manifest, patients, classifier, arms if selected else None, provider_rejections)
        if provider_policy:
            parked, state["provider_rejections"] = persist_provider_rejections(directory, manifest, accepted, provider_rejections, arms)
        if selected:
            # Recover saved clean results before scheduling any retry. Deferred
            # canonical files, identities, ledger entries and counts stay intact.
            save_accepted(directory, manifest, accepted, choices, counts, arms, patients)
        executable = shutil.which("caffeinate")
        if executable:
            awake = subprocess.Popen([executable, "-i", "-w", str(os.getpid())], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True)
            state["keep_awake_pid"] = awake.pid
        event(directory, "supervisor_started", pid=os.getpid(), resumed=resume, **({"selected_arms": list(arms)} if selected else {}))
        next_launch = {arm: 0 for arm in arms}
        no_progress_rounds = {arm: 0 for arm in arms}
        prior_counts = {arm: len(rows) for arm, rows in accepted.items()}
        while True:
            validate_manifest(manifest, directory)
            if file_sha256(directory / "run_manifest.json") != manifest_hash:
                raise ValueError("Run manifest changed")
            if shutil.disk_usage(directory).free < settings["min_free_gb"] * 1024 ** 3:
                raise ValueError("Disk reserve guard reached")
            finished = []
            for arm, child in list(children.items()):
                batch_path, batch = batches[arm]
                path = Path(batch["output"])
                stamp = (path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else None
                if stamp != progress[arm]["stamp"]:
                    checkpointed = [row for row in iter_result_rows(path) if row.get("latency") is not None or row.get("error")] if path.exists() else []
                    recent = checkpointed[-10:]
                    # Detect infrastructure trouble only from explicit runtime
                    # diagnostics; never inspect answer correctness.
                    runtime_recent = sum(classifier(row, path.parent)["status"] == "runtime" for row in recent) if len(recent) >= 8 else 0
                    progress[arm] = {"stamp": stamp, "monotonic": time.monotonic(), "checkpointed": len(checkpointed)}
                    if runtime_recent >= 8 and runtime_recent / len(recent) >= 0.8:
                        stop_owned({arm: child})
                        event(directory, "runtime_circuit_breaker", arm=arm, round=batch["round"], recent_runtime=runtime_recent, recent_questions=len(recent))
                        progress[arm]["circuit_breaker"] = True
                if child.poll() is None and time.monotonic() - progress[arm]["monotonic"] >= settings["stall_seconds"]:
                    stop_owned({arm: child})
                    event(directory, "stalled_process_recovered", arm=arm, round=batch["round"], pid=child.pid)
                    reason = "stalled_process"
                elif progress[arm].get("circuit_breaker"):
                    reason = "runtime_circuit_breaker"
                else:
                    reason = "terminal_process"
                if child.poll() is not None:
                    seal_batch(batch_path, child.returncode, reason)
                    event(directory, "batch_sealed", arm=arm, round=batch["round"], return_code=child.returncode, reason=reason)
                    logs.pop(arm).close()
                    del children[arm]
                    finished.append(arm)
                    next_launch[arm] = time.monotonic() + settings["retry_delay_seconds"]
            if finished:
                accepted, choices, retries, attention, counts = collect_accepted(directory, manifest, patients, classifier, arms if selected else None, provider_rejections)
                if provider_policy:
                    parked, state["provider_rejections"] = persist_provider_rejections(directory, manifest, accepted, provider_rejections, arms)
                for arm in finished:
                    no_progress_rounds[arm] = no_progress_rounds[arm] + 1 if len(accepted[arm]) == prior_counts[arm] else 0
                    prior_counts[arm] = len(accepted[arm])
                if selected:
                    save_accepted(directory, manifest, accepted, choices, counts, arms, patients)
            if selected:
                # Even an attention boundary reports and retains the clean
                # rows sealed alongside the infrastructure/unknown failure.
                state.update(accepted={arm: len(accepted[arm]) if arm in arms else len(retained["arms"][arm]) for arm in ARMS},
                             counts={arm: counts[arm] if arm in arms else retained["counts"][arm] for arm in ARMS})
            if attention:
                state.update(phase="attention", attention=True, reason="unclassified_or_invalid_attempt", issues=attention)
                break
            exhausted = [{"arm": arm, "question_id": qid, "runtime_attempts": count}
                         for arm in arms for qid, total in retries[arm].items()
                         for count in [total - (provider_rejections[arm].get(qid, 0) if provider_policy else 0)]
                         if count > settings["max_runtime_retries"] and qid not in accepted[arm]]
            if exhausted:
                state.update(phase="attention", attention=True, reason="runtime_retry_limit", issues=exhausted)
                break
            if provider_policy and not children and any(parked.values()) and all(
                    set(patients) - set(accepted[arm]) <= parked[arm] for arm in arms):
                state.update(phase="attention", attention=True, reason="provider_policy_review_required", finished_at=utc_now(),
                             issues=[{"arm": arm, "question_id": qid, "provider_rejections": provider_rejections[arm][qid],
                                      "status": "pending", "action": "external_policy_review_required"}
                                     for arm in arms for qid in sorted(parked[arm])])
                break
            for arm in arms if selected else eligible_arms(accepted, len(patients)):
                if arm in children or len(accepted[arm]) == len(patients) or time.monotonic() < next_launch[arm]:
                    continue
                # Repeated pre-result process failures are bounded separately from
                # per-question infrastructure retries, without inventing outcomes.
                prior = attempt_rounds(directory, arm)
                if no_progress_rounds[arm] >= settings["max_runtime_retries"] + 2:
                    state.update(phase="attention", attention=True, reason="process_round_limit")
                    break
                pending = set([qid for qid in patients if qid not in accepted[arm] and qid not in parked[arm]][:settings["batch_size"]])
                if not pending:
                    continue
                number = max([int(path.name.split("-")[1].split(".")[0]) for path in prior] or [0]) + 1
                batch_path, batch = create_batch(directory, arm, pending, manifest, number)
                log = Path(batch["output"]).with_suffix(".log").open("x", encoding="utf-8")
                environment = os.environ.copy()
                environment["PYTHONUNBUFFERED"] = "1"
                environment["FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR"] = str(directory / "diagnostics" / arm / f"round-{number:04d}")
                children[arm] = subprocess.Popen(batch["command"], cwd=ROOT, env=environment,
                    stdout=log, stderr=log, start_new_session=True)
                logs[arm], batches[arm] = log, (batch_path, batch)
                progress[arm] = {"stamp": None, "monotonic": time.monotonic(), "checkpointed": 0}
                event(directory, "batch_started", arm=arm, round=number, questions=len(pending), pid=children[arm].pid)
            state.update(heartbeat_at=utc_now(),
                         accepted={arm: len(accepted[arm]) if not selected or arm in arms else len(retained["arms"][arm]) for arm in ARMS},
                         counts={arm: counts[arm] if not selected or arm in arms else retained["counts"][arm] for arm in ARMS},
                         active={arm: {"pid": child.pid, "round": batches[arm][1]["round"],
                                       "checkpointed_in_batch": progress[arm].get("checkpointed", 0),
                                       "seconds_since_checkpoint": round(time.monotonic() - progress[arm]["monotonic"])}
                                 for arm, child in children.items()})
            if (finished and not selected) or not (directory / "acceptance.json").exists():
                save_accepted(directory, manifest, accepted, choices, counts, arms if selected else None, patients if selected else None)
            if state["attention"]:
                break
            if selected and all(set(accepted[arm]) == set(patients) for arm in arms):
                state.update(phase="complete_selected_arms", selected_scope_complete=True, benchmark_complete=False,
                             pending_scoped_export=True, analysis_status="pending_scoped_export", finished_at=utc_now())
                break
            if not selected and all(len(rows) == len(patients) for rows in accepted.values()):
                state["phase"] = "analyzing"
                atomic_json(directory / "supervisor_state.json", state)
                with (directory / "analysis.log").open("a", encoding="utf-8") as analysis_log:
                    analysis = subprocess.Popen(manifest["analysis_command"], cwd=ROOT, env=os.environ.copy(),
                                                stdout=analysis_log, stderr=analysis_log, start_new_session=True)
                    children["analysis"] = analysis
                    while analysis.poll() is None:
                        state.update(heartbeat_at=utc_now(), analysis_pid=analysis.pid)
                        atomic_json(directory / "supervisor_state.json", state)
                        try:
                            analysis.wait(timeout=settings["heartbeat_seconds"])
                        except subprocess.TimeoutExpired:
                            pass
                    children.pop("analysis")
                    state.update(analysis_return_code=analysis.returncode, phase="complete" if analysis.returncode == 0 else "attention_analysis",
                                 attention=analysis.returncode != 0, finished_at=utc_now())
                break
            state["phase"] = "running"
            atomic_json(directory / "supervisor_state.json", state)
            time.sleep(settings["heartbeat_seconds"])
        return 2 if state["attention"] else 0
    except StopRequested:
        state.update(phase="interrupted", attention=True, reason="signal", finished_at=utc_now())
        return 130
    except Exception as error:
        # Persist only the class; runtime exception strings may contain endpoint
        # bodies or request information. Local raw logs remain available to audit.
        safe_message = str(error) if isinstance(error, ValueError) and str(error).startswith(("Frozen ", "Run manifest", "Disk reserve", "Sealed ", "Attempt ", "Terminal attempt", "Saved attempt", "Completed accepted", "Conflicting accepted", "Prior ", "Selected ")) else None
        state.update(phase="attention_supervisor", attention=True, reason=type(error).__name__, safe_message=safe_message, finished_at=utc_now())
        return 2
    finally:
        stop_owned(children)
        for arm, (batch_path, _batch) in batches.items():
            if arm in children:
                seal_batch(batch_path, children[arm].poll(), "supervisor_cleanup")
        for log in logs.values():
            log.close()
        if awake is not None and awake.poll() is None:
            awake.terminate()
            try:
                awake.wait(timeout=5)
            except subprocess.TimeoutExpired:
                awake.kill()
                awake.wait(timeout=5)
        state.update(heartbeat_at=utc_now(), active={})
        atomic_json(directory / "supervisor_state.json", state)
        event(directory, "supervisor_stopped", phase=state["phase"], attention=state["attention"])
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "status", "resume"))
    parser.add_argument("--run-dir", "--output_dir", dest="run_dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "final_dataset/questions_answers_sql_fhir.csv")
    parser.add_argument("--families", type=Path, default=ROOT / "manifests/question_families.csv")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--max-total-tokens", "--max_total_tokens", dest="max_total_tokens", type=int, default=200000)
    parser.add_argument("--max-tool-calls", "--max_tool_calls", dest="max_tool_calls", type=int, default=30)
    parser.add_argument("--num-processes", "--num_processes", dest="num_processes", type=int, default=2)
    parser.add_argument("--save-interval", "--save_interval", dest="save_interval", type=int, default=1)
    parser.add_argument("--heartbeat-seconds", type=int, default=15)
    parser.add_argument("--stall-seconds", type=int, default=45 * 60)
    parser.add_argument("--min-free-gb", type=float, default=5)
    parser.add_argument("--max-runtime-retries", type=int, default=20)
    parser.add_argument("--retry-delay-seconds", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--only-arm", action="append", choices=("single_agent",),
                        help="Resume only the explicitly pinned single-agent scope; never runs full analysis")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.run_dir, args.input, args.families = [path.resolve() for path in (args.run_dir, args.input, args.families)]
    if args.only_arm is not None and args.command != "resume":
        raise ValueError("Selected arms are supported only for resume")
    if args.command == "status":
        path = args.run_dir / "supervisor_state.json"
        print(json.dumps(compact_state(json.loads(path.read_text())) if path.exists() else {"phase": "not_started"}, indent=2))
        return 0
    if any(getattr(args, key) < 1 for key in ("max_total_tokens", "max_tool_calls", "num_processes", "heartbeat_seconds", "stall_seconds", "batch_size")):
        raise ValueError("Run limits must be positive")
    if args.save_interval != 1 or args.max_runtime_retries < 0 or args.retry_delay_seconds < 0 or args.min_free_gb < 0:
        raise ValueError("Invalid checkpoint, retry, or disk guard setting")
    if args.command == "resume":
        manifest = json.loads((args.run_dir / "run_manifest.json").read_text())
        execution_arms(manifest, args.only_arm, resume=True)
        validate_manifest(manifest, args.run_dir)
        patients = dataset_cohort(manifest["input"]["path"])
    else:
        manifest = manifest_for(args)
        patients = dataset_cohort(args.input)
        if args.command == "plan":
            print(json.dumps({key: manifest[key] for key in ("protocol", "classifier_version", "arms", "input", "settings", "policy", "analysis_command")}, indent=2))
            return 0
        args.run_dir.mkdir(parents=True, exist_ok=True)
        if any((args.run_dir / name).exists() for name in ("run_manifest.json", "supervisor_state.json", "attempts", "acceptance.json")):
            raise ValueError("Run already exists; use resume")
        with (args.run_dir / "run_manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    return run_supervisor(args.run_dir, manifest, patients, resume=args.command == "resume", only_arms=args.only_arm)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"V3 full benchmark setup needs attention ({type(error).__name__}).", file=sys.stderr)
        raise SystemExit(2)
