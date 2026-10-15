"""Isolated sequential request-cap diagnostic. Never changes project/config files.

prepare/plan are offline. run is for AFTER V3b ends, uses the existing bootstrap,
and sends six fixed nonclinical or development prompts once each, with no retries
or scoring. prepare_health creates nonclinical inputs without project imports.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

CAPS = (16384, 4096)
HEADERS = ("x-request-id", "apim-request-id", "x-ms-request-id", "x-ms-region",
           "x-ratelimit-limit-requests", "x-ratelimit-limit-tokens",
           "x-ratelimit-remaining-requests", "x-ratelimit-remaining-tokens",
           "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens", "retry-after", "retry-after-ms")
PHASE_TOKENS = {"contract": 10000, "solver": 175000, "audit": 15000}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def paths(project):
    for path in (project / ".local-setup/phase2-dependencies", project):
        sys.path.insert(0, str(path))


def cases_file(path):
    value = json.loads(Path(path).read_text())
    cases = value.get("cases", [])
    if value.get("version") != 1 or len(cases) != 3 or {c.get("phase") for c in cases} != set(PHASE_TOKENS):
        raise ValueError("Provide exactly one completed-development contract, solver and static-audit prompt")
    for case in cases:
        if set(case) != {"case_id", "phase", "model", "messages", "tools", "max_completion_tokens"}:
            raise ValueError("Case schema must contain only the documented non-credential input fields")
        if case["model"] != "gpt-6-luna" or not case["messages"] or case["max_completion_tokens"] < 1:
            raise ValueError("Cases require the fixed model, messages and a positive original caller ceiling")
        if not all(isinstance(m, dict) for m in case["messages"]):
            raise ValueError("Messages must be API dictionaries")
    return value


def schedule(cases):
    # Reversed middle pair reduces fixed-order/cache/time bias; not randomization.
    return [(case, cap) for index, case in enumerate(cases)
            for cap in (CAPS if index % 2 == 0 else CAPS[::-1])]


def prepare(args):
    # This source is a COMPLETED development fixture, not the live V3b run.
    source = json.loads(args.development_output.read_text())
    trace = source["trace"]
    prefixes = {"contract": "Independently derive a short question contract",
                "solver": "You answer questions", "audit": "Check the solver's actual executed Python"}
    paths(args.project_root)
    from tools import get_tool_definitions
    from utils.core_utils import count_tokens_in_messages
    definitions = get_tool_definitions()
    selected = ("get_resources_by_patient_fhir_id", "get_resources_by_resource_id", "execute_python_code")
    solver_tools = [d for d in definitions if d["function"]["name"] in selected]
    solver_tools += [d for d in definitions if d["function"]["name"] == "check_fields"]
    cases = []
    for phase, prefix in prefixes.items():
        index = next(i for i, m in enumerate(trace) if m.get("role") == "system"
                     and (m.get("content") or "").startswith(prefix))
        messages = [trace[index], trace[index + 1]]
        if messages[1].get("role") != "user":
            raise ValueError("Development trace does not preserve the expected initial prompt")
        # Zero-tool Role calls include a final-task user instruction; the
        # solver includes its visible budget signal. Keep all initial users.
        cursor = index + 2
        while cursor < len(trace) and trace[cursor].get("role") == "user":
            messages.append(trace[cursor])
            cursor += 1
        if phase == "solver":
            if len(messages) < 3 or not messages[-1].get("content", "").startswith("Solver allocation remaining"):
                raise ValueError("The solver's original budget signal is missing")
        tools = solver_tools if phase == "solver" else None
        prompt = count_tokens_in_messages(messages, tools)
        remaining = PHASE_TOKENS[phase] - prompt - max(64, math.ceil(prompt * .05))
        cases.append({"case_id": "completed-development-" + phase, "phase": phase,
                      "model": "gpt-6-luna", "messages": messages, "tools": tools,
                      "max_completion_tokens": remaining})
    args.cases.write_text(json.dumps({"version": 1, "source_sha256": sha(args.development_output),
        "source_kind": "reconstructed_completed_development_initial_phase_prompts_no_scoring", "cases": cases}, indent=2) + "\n")
    print(json.dumps({"cases": 3, "source_sha256": sha(args.development_output), "model_calls": 0}))


def prepare_health(args):
    requests = {
        "contract": "Return only JSON with operation='count', item='blue blocks', constraint='before Friday' for this sentence: Count the blue blocks before Friday.",
        "solver": "Return only JSON with answer=3 and operation='count' for this toy list: blue, red, blue, green, blue. This is a nonclinical transport test.",
        "audit": "Return only JSON with decision='pass' and issues=[] if this toy computation is correct: len([2, 4, 6]) = 3. This is a nonclinical transport test.",
    }
    cases = [{"case_id": "nonclinical-health-" + phase, "phase": phase,
              "model": "gpt-6-luna", "messages": [
                  {"role": "system", "content": "Follow the requested JSON format. Be concise. No tools are available."},
                  {"role": "user", "content": prompt}], "tools": None,
              "max_completion_tokens": 32768} for phase, prompt in requests.items()]
    args.cases.write_text(json.dumps({"version": 1,
        "source_kind": "nonclinical_fixed_health_prompts_not_benchmark_phases", "cases": cases}, indent=2) + "\n")
    print(json.dumps({"cases": 3, "model_calls": 0, "clinical_data": False}))


def metadata(args, value):
    selected_schedule = schedule(value["cases"])[:args.request_limit]
    return {"version": 1, "diagnostic_only": True, "model": "gpt-6-luna", "workers": 1,
        "sdk_max_retries": 0, "application_max_retries": 1, "sdk_timeout_seconds": 60,
        "case_file_sha256": sha(args.cases), "cases_source_sha256": value.get("source_sha256"),
        "source_kind": value.get("source_kind"), "reasoning_effort": "none",
        "planned_sdk_calls": len(selected_schedule), "caps": sorted({cap for _, cap in selected_schedule}, reverse=True),
        "request_scoring": False,
        "schedule": [{"case_id": case["case_id"], "phase": case["phase"], "cap": cap,
                      "effective_requested_completion_tokens": min(cap, case["max_completion_tokens"]),
                      "input_sha256": digest({"messages": case["messages"], "tools": case["tools"]})}
                     for case, cap in selected_schedule]}


def usage(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict):
        return None
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {key: value[key] for key in keys} if all(
        isinstance(value.get(key), int) and not isinstance(value[key], bool) and value[key] >= 0 for key in keys) else None


def classify(error):
    status = getattr(error, "status_code", None)
    kind = type(error).__name__
    if status is not None:
        return "server_http_408" if status == 408 else "server_http_error"
    if kind == "APITimeoutError":
        return "client_timeout"
    return "connection_error" if kind == "APIConnectionError" else "host_error"


def child(args):
    import azure_adapter as adapter
    import httpx
    from utils.core_utils import safe_llm_call
    # Keep potentially detailed HTTP/error logging private to SDK internals;
    # the diagnostic persists only whitelisted response metadata below.
    for name in ("openai", "httpx", "httpcore", "httpx2", "httpcore2"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    settings = adapter._settings  # installed by the normal sitecustomize bootstrap
    if (settings is None or not settings.matches("gpt-6-luna")
            or settings.max_completion_tokens != 16384 or settings.reasoning_effort != "none"):
        raise ValueError("Normal bootstrap/profile does not match the recorded base policy")
    value = cases_file(args.cases)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = metadata(args, value)
    base_profile = settings.public()
    base_profile.pop("api_key_configured", None)
    record["provider_base_profile_sha256"] = digest(base_profile)
    record["provider_identity_profile_sha256"] = hashlib.sha256(json.dumps(base_profile, sort_keys=True).encode()).hexdigest()
    record["arm_profile_sha256"] = {str(cap): digest({**base_profile, "max_completion_tokens": cap}) for cap in CAPS}
    record["source_sha256"] = {str(p.relative_to(args.project_root)): sha(p) for p in
        [args.project_root / "utils/core_utils.py", args.project_root / ".local-setup/azure_adapter.py",
         args.project_root / ".local-setup/sitecustomize.py", args.project_root / ".local-setup/azure.py"]}
    record["harness_sha256"] = sha(__file__)
    record["runtime_versions"] = {name: importlib.metadata.version(name) for name in ("openai", "httpx", "litellm")}
    (args.output_dir / "probe_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    (args.output_dir / "cases.json").write_bytes(args.cases.read_bytes())
    current = {}

    def response_hook(response):
        current["http_status"] = response.status_code
        current["response_headers_seconds"] = time.monotonic() - current["started"]
        current["response_headers"] = {name: response.headers[name] for name in HEADERS if name in response.headers}

    original = adapter.azure_completion
    adapter._client = adapter.make_client(settings)
    sdk_records = []

    def observed_completion(**kwargs):
        current["started"] = time.monotonic()  # exact SDK/adapter interval, after caller token preflight
        current["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        current["sdk_default_transport_type"] = type(adapter._client._client).__module__ + "." + type(adapter._client._client).__name__
        current["sdk_default_transport_version"] = importlib.metadata.version("httpx2")
        try:
            response = original(**kwargs)
            current["sdk_usage"] = usage(getattr(response, "usage", None))
            current["outcome"] = "success"
            current["finish_reasons"] = [c.finish_reason for c in response.choices]
            current["tool_call_count"] = sum(len(c.message.tool_calls or []) for c in response.choices)
            current["service_tier"] = getattr(response, "service_tier", None)
            raw = response.model_dump(exclude_none=True)
            with (args.output_dir / current["response_file"]).open("wb") as raw_file:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw_file, mtime=0) as stream:
                    stream.write(json.dumps(raw, ensure_ascii=False, allow_nan=False).encode())
            return response
        except Exception as error:
            current["outcome"] = classify(error)
            current["exception_type"] = type(error).__name__
            causes = []
            seen = set()
            cause = error
            while cause is not None and id(cause) not in seen and len(causes) < 8:
                seen.add(id(cause))
                message = str(cause).lower()
                causes.append({"type": type(cause).__name__,
                               "errno": getattr(cause, "errno", None) if isinstance(getattr(cause, "errno", None), int) else None,
                               "client_closed": "client has been closed" in message,
                               "network_unreachable": "network is unreachable" in message,
                               "dns_resolution": any(x in message for x in ("nodename nor servname", "name or service not known", "getaddrinfo", "name resolution")),
                               "certificate_verification": "certificate_verify_failed" in message})
                cause = cause.__cause__ or cause.__context__
            current["exception_causes"] = causes
            current["http_status"] = getattr(error, "status_code", current.get("http_status"))
            body = getattr(error, "body", None)
            current["sdk_usage"] = usage(body.get("usage")) if isinstance(body, dict) else None
            details = body.get("error", body) if isinstance(body, dict) else {}
            code = details.get("code") if isinstance(details, dict) else None
            current["service_error_code"] = code if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", code) else None
            # Never persist an error message, request headers, endpoint, or key.
            raise
        finally:
            current["elapsed_seconds"] = time.monotonic() - current["started"]
            current["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            current["usage_unknown"] = current.get("sdk_usage") is None
            current["cost_usd"] = None  # never infer zero cost from an error/no reported usage
            row = {k: v for k, v in current.items() if k != "started"}
            sdk_records.append(row)
            with (args.output_dir / "sdk_events.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

    import litellm
    litellm.completion = observed_completion
    try:
        for index, (case, cap) in enumerate(schedule(value["cases"])[:args.request_limit]):
            adapter._settings = replace(settings, max_completion_tokens=cap)
            current.clear()
            current.update({"request_index": index, "case_id": case["case_id"], "phase": case["phase"],
                "cap": cap, "effective_requested_completion_tokens": min(cap, case["max_completion_tokens"]),
                "input_sha256": digest({"messages": case["messages"], "tools": case["tools"]}),
                "started": time.monotonic(), "response_file": f"response_{index}.json.gz"})
            caller_started = time.monotonic()
            with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
                _, error, reported = safe_llm_call(model=case["model"], messages=case["messages"], tools=case["tools"],
                    max_retries=1, max_tokens=32000, max_completion_tokens=case["max_completion_tokens"])
            with (args.output_dir / "caller_attempts.jsonl").open("a") as stream:
                stream.write(json.dumps({"request_index": index, "caller_error_present": bool(error),
                    "caller_elapsed_seconds": time.monotonic() - caller_started,
                    "unknown_usage": any(c.get("usage_available") is False for c in (reported or {}).get("per_call", [])),
                    "reported_success_tokens_only": (reported or {}).get("total_tokens"),
                    "llm_calls": (reported or {}).get("llm_calls")}) + "\n")
            # Each condition is one first request. Failures are never replayed.
    finally:
        adapter._settings = settings
        adapter._client.close()
        litellm.completion = original
    summary = {str(cap): {"requests": sum(r["cap"] == cap for r in sdk_records),
        "server_http_408": sum(r["cap"] == cap and r["outcome"] == "server_http_408" for r in sdk_records),
        "client_timeout": sum(r["cap"] == cap and r["outcome"] == "client_timeout" for r in sdk_records),
        "unknown_usage_requests": sum(r["cap"] == cap and r["usage_unknown"] for r in sdk_records),
        "reported_usage_tokens_only": sum(r["sdk_usage"]["total_tokens"] for r in sdk_records if r["cap"] == cap and r["sdk_usage"] is not None),
        "length_finish_requests": sum(r["cap"] == cap and "length" in r.get("finish_reasons", []) for r in sdk_records)} for cap in CAPS}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "prepare_health", "plan", "run", "_child"))
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--development_output", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--after_v3b_finished", action="store_true")
    parser.add_argument("--request_limit", type=int, choices=range(1, 7), default=6,
                        help="1 makes a single unchanged-16,384-ceiling nonclinical availability/region request")
    args = parser.parse_args()
    args.project_root, args.cases = args.project_root.resolve(), args.cases.resolve()
    if args.command == "prepare_health":
        return prepare_health(args)
    if args.command == "prepare":
        if not args.development_output:
            parser.error("prepare requires a completed development output; never use a live result file")
        return prepare(args)
    value = cases_file(args.cases)
    if args.command == "plan":
        print(json.dumps(metadata(args, value), indent=2))
        return 0
    if not args.after_v3b_finished:
        parser.error("Live probe is disabled until root confirms the frozen V3b run has ended")
    if not args.output_dir or args.output_dir.exists():
        parser.error("Use a fresh --output_dir; this diagnostic never resumes or retries")
    args.output_dir = args.output_dir.resolve()
    if args.command == "_child":
        return child(args)
    # Existing normal provider/bootstrap path. No configuration is edited or
    # printed; the normal loader runs only inside this explicitly invoked child.
    sys.path.insert(0, str(args.project_root / ".local-setup"))
    import azure
    azure.PROJECT = args.project_root
    environment = azure.child_environment()
    environment["PYTHONPATH"] += os.pathsep + str(args.project_root / ".local-setup/phase2-dependencies")
    return subprocess.call([str(args.project_root / ".venv/bin/python"), str(Path(__file__).resolve()),
        "_child", "--project_root", str(args.project_root), "--cases", str(args.cases),
        "--output_dir", str(args.output_dir), "--after_v3b_finished",
        "--request_limit", str(args.request_limit)], cwd=args.project_root, env=environment)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("Probe stopped: input, fresh-output or runtime check failed; no secret-bearing exception details printed.", file=sys.stderr)
        raise SystemExit(2)
