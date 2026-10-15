"""One-shot live saved-file audit: provenance/accounting only, never scores."""
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
canonical = lambda value: json.dumps(value, sort_keys=True, allow_nan=False)


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def module_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def integer(value, nonnegative=False):
    return value is None or type(value) is int and (not nonnegative or value >= 0)


def class_schema(value):
    if value is None:
        return
    require(isinstance(value, dict) and set(value) == {"class", "mro"}, "Journal class schema")
    pattern = r"[A-Za-z0-9_.]{1,200}"
    require(isinstance(value["class"], str) and re.fullmatch(pattern, value["class"]), "Journal class name")
    require(isinstance(value["mro"], list) and 1 <= len(value["mro"]) <= 16, "Journal MRO bound")
    require(all(isinstance(item, str) and re.fullmatch(pattern, item) for item in value["mro"]), "Journal MRO values")


def client_schema(value):
    require(isinstance(value, dict) and set(value) == {"sdk", "http", "transport", "pool", "sdk_closed", "http_closed", "client_id", "versions"}, "Journal client schema")
    for key in ("sdk", "http", "transport", "pool"):
        class_schema(value[key])
    require(all(value[key] is None or type(value[key]) is bool for key in ("sdk_closed", "http_closed")), "Journal closed flags")
    require(integer(value["client_id"], True), "Journal client identifier")
    require(isinstance(value["versions"], dict) and set(value["versions"]) == {"openai", "httpx2"}, "Journal package schema")
    require(all(item is None or isinstance(item, str) and re.fullmatch(r"[0-9A-Za-z.+_-]{1,64}", item) for item in value["versions"].values()), "Journal package values")


def event_schema(event):
    common = {"version", "pid", "ppid", "counter", "process_session", "started_utc", "event", "client", "fd_count"}
    end = {"finished_utc", "duration_seconds", "outcome", "exception_chain", "usage", "http_status", "request_ids"}
    require(isinstance(event, dict) and event.get("event") in {"start", "end"}, "Journal event kind")
    require(set(event) == (common if event["event"] == "start" else common | end), "Journal key allowlist")
    require(type(event["version"]) is int and event["version"] == 1, "Journal version")
    require(all(type(event[key]) is int and event[key] > 0 for key in ("pid", "ppid", "counter")), "Journal process/counter values")
    require(isinstance(event["process_session"], str) and re.fullmatch(r"[a-f0-9]{32}", event["process_session"]), "Journal session")
    started = datetime.fromisoformat(event["started_utc"])
    require(started.tzinfo is not None, "Journal UTC timestamp")
    require(integer(event["fd_count"], True), "Journal descriptor count")
    client_schema(event["client"])
    if event["event"] == "start":
        return
    require(datetime.fromisoformat(event["finished_utc"]) >= started, "Journal finish timestamp")
    require(type(event["duration_seconds"]) in (int, float) and math.isfinite(event["duration_seconds"]) and event["duration_seconds"] >= 0, "Journal duration")
    require(event["outcome"] in {"success", "error"}, "Journal outcome")
    require(integer(event["http_status"]) and (event["http_status"] is None or 100 <= event["http_status"] <= 599), "Journal HTTP status")
    require(isinstance(event["request_ids"], dict) and set(event["request_ids"]) <= {"x-request-id", "x-ms-request-id", "apim-request-id"}, "Journal request-ID allowlist")
    require(all(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value) for value in event["request_ids"].values()), "Journal request-ID values")
    require(isinstance(event["usage"], dict) and set(event["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens", "cached_prompt_tokens"}, "Journal usage schema")
    require(all(integer(value, True) for value in event["usage"].values()), "Journal usage values")
    require(isinstance(event["exception_chain"], list) and len(event["exception_chain"]) <= 8, "Journal cause bound")
    for cause in event["exception_chain"]:
        require(isinstance(cause, dict) and set(cause) == {"type", "errno", "winerror", "verify_code", "flags"}, "Journal cause schema")
        class_schema(cause["type"])
        require(all(integer(cause[key]) for key in ("errno", "winerror", "verify_code")), "Journal numeric cause codes")
        require(isinstance(cause["flags"], dict) and set(cause["flags"]) == {"dns", "tls", "connect", "read", "write", "pool", "timeout", "resource_fd"}, "Journal cause flag schema")
        require(all(type(value) is bool for value in cause["flags"].values()), "Journal cause flag values")


def main():
    started = datetime.now(timezone.utc).isoformat()
    pre = module_file("v4_prelaunch_pure", BASE / "prelaunch_audit_source.py")
    store = module_file("v4_audit_store_only", ROOT / "scripts/audit_store.py")
    manifest = pre.load(BASE / "manifest.json")
    runtime = pre.load(BASE / "runtime_adapter_manifest.json")
    expected_binding = pre.load(BASE / "expected_public_binding.json")
    output = Path(manifest["output_directory"])
    pre.check_archive(BASE / "source.zip", manifest["source_sha256"], manifest["source_archive_sha256"])
    require(len(manifest["source_sha256"]) == 142, "Frozen source count")
    pre.check_archive(BASE / "runtime_adapter_sources.zip", runtime["source_sha256"], runtime["archive_sha256"])
    require(len(runtime["source_sha256"]) == 4, "Runtime source count")
    require(sha(BASE / "design.txt") == manifest["design_sha256"], "Design hash")
    require(sha(ROOT / "final_dataset/questions_answers_sql_fhir.csv") == manifest["dataset_sha256"], "Dataset hash")
    require(sha(ROOT / "manifests/split.json") == manifest["split_sha256"], "Split hash")
    require(canonical(pre.load(output / ".local-provider.json")) == canonical(expected_binding), "Actual provider binding")
    prior = pre.load(BASE / "prelaunch_audit.json")
    expected_ids = pre.load(ROOT / "manifests/split.json")["splits"]["held_out"]["question_ids"]
    # Parser-only extraction; no agent, settings or runtime bootstrap import.
    plan_module = ROOT / "agent/plan_conditioned_agent.py"
    plan_ns = dict(json=json, deepcopy=deepcopy,
        OPERATORS=frozenset({"population", "filter", "reference", "time_window", "order", "select", "project", "aggregate", "combine", "multiplicity", "units", "output"}))
    pre.extract(plan_module, {"parse_plan"}, plan_ns)
    arms, anomalies = {}, []
    observed_plans = boundaries = artifact_bytes = 0
    for arm, strategy in [("ordinary_single_agent", "single_agent"), ("ordinary_multi_agent", "multi_agent_v4")]:
        identity = pre.load(output / (arm + ".json.run.json"))
        label = "single" if strategy == "single_agent" else "team"
        require(canonical(identity) == canonical(prior["prospective_run_identities"][label]), "Actual run identity differs: " + arm)
        raw = (output / (arm + ".json")).read_bytes()  # one atomic-save byte snapshot
        rows = json.loads(raw)
        require(isinstance(rows, list) and len(rows) == 300 and [row["question_id"] for row in rows] == expected_ids, "Exact scheduled order")
        require(all(not row.get("attempt_history") for row in rows), "Unexpected repeated attempts")
        # Same input identity serialization; no answer/reference comparison.
        input_json = pre.pd.DataFrame(rows)[["question_id", "question", "true_answer", "assumption", "patient_fhir_id"]].to_json(orient="records", date_format="iso")
        require(hashlib.sha256(input_json.encode()).hexdigest() == identity["input_hash"], "Saved input identities differ")
        saved = [row for row in rows if type(row.get("latency")) in (int, float) and math.isfinite(row["latency"]) or row.get("error")]
        phases_seen, policies, visible = Counter(), Counter(), Counter()
        checked = errors = unknown_usage = overshoots = phase_fatals = 0
        per_row_metadata = []
        for row in saved:
            ref = row.get("audit_artifacts")
            require(isinstance(ref, dict) and ref.get("version") == 2 and ref.get("codec") == "json+gzip", "Missing completed audit artifact")
            require(ref.get("question_id") == row["question_id"] and re.fullmatch(r"audit/[a-f0-9]{64}\.json\.gz", ref.get("path", "")), "Audit owner/path")
            require(ref["path"] == "audit/" + ref["sha256"] + ".json.gz", "Audit content address")
            path = (output / ref["path"]).resolve()
            require(path.is_relative_to(output.resolve()), "Escaping audit path")
            with gzip.open(path, "rb") as stream:
                payload_bytes = stream.read(512 * 1024 * 1024 + 1)  # consumes gzip trailer/CRC
            require(len(payload_bytes) <= 512 * 1024 * 1024, "Audit restoration limit")
            require(hashlib.sha256(payload_bytes).hexdigest() == ref["sha256"], "Independent audit SHA")
            payload = json.loads(payload_bytes)
            require(payload["_audit_meta"]["question_id"] == row["question_id"], "Payload owner")
            require(set(payload) == {"_audit_meta", *ref["fields"]}, "Payload field declaration")
            restored = store.restore_result_audit(row, output, question_id=row["question_id"])
            for field in ref["fields"]:
                actual = restored["revision"]["messages"] if field == "revision.messages" else restored[field]
                require(canonical(actual) == canonical(payload[field]), "Lossless restored audit field")
            for field in ("agent_answer", "evidence", "limitation", "tokens", "usage", "tool_calls", "budget", "error", "latency", "revision"):
                if field == "revision" and "revision.messages" in ref["fields"]:
                    continue
                require(canonical(restored.get(field)) == canonical(row.get(field)), "Restoration changed inline field")
            artifact_bytes += len(payload_bytes)
            checked += 1
            errors += bool(row.get("error"))
            tokens, budget = row.get("tokens") or {}, row.get("budget") or {}
            require((budget["max_total_tokens"], budget["max_tool_calls"]) == (200000, 30), "Actual aggregate caps")
            require(type(budget["tool_calls_run"]) is int and 0 <= budget["tool_calls_run"] <= 30, "Tool accounting")
            overshoots += bool(tokens.get("total", 0) > 200000)
            calls = tokens.get("per_call", [])
            unknown_usage += sum(call.get("usage_available") is False for call in calls)
            if strategy == "single_agent":
                continue
            revision = restored.get("revision") or {}
            require(revision.get("protocol") == "plan-conditioned-v4", "Actual V4 protocol")
            require(revision.get("targeted_revised") is False, "Unexpected targeted revision")
            phases = revision.get("phases") or []
            require([phase["phase"] for phase in phases] in [["planner"], ["planner", "solver"]], "Unexpected late audit/targeted phase")
            require(sum(phase["tokens"] for phase in phases) == tokens["total"], "Phase token sum")
            require(sum(phase["tool_calls"] for phase in phases) == budget["tool_calls_run"], "Phase tool sum")
            for phase in phases:
                require((phase["token_cap"], phase["tool_cap"]) == {"planner": (10000, 0), "solver": (190000, 30)}[phase["phase"]], "Actual local phase caps")
                phases_seen[phase["phase"]] += 1
                phase_fatals += bool(phase["fatal"])
                overshoots += bool(phase["tokens"] > phase["token_cap"] or phase["tool_calls"] > phase["tool_cap"])
            require(tokens["llm_calls"] <= 21, "Actual model call ceiling")
            if calls and calls[0].get("usage_available"):
                require(calls[0]["prompt"] + calls[0]["completion"] == phases[0]["tokens"], "Planner usage attribution")
            trace = restored["trace"]
            systems = [index for index, message in enumerate(trace) if isinstance(message, dict) and message.get("role") == "system"]
            require(systems and systems[0] == 0, "Missing planner system")
            planner_user = trace[1]
            context = json.loads(planner_user["content"])
            require(set(context) == {"question", "patient_fhir_id", "current_time", "assumptions"}, "Planner context allowlist")
            require(context["question"] == row["question"] and context["patient_fhir_id"] == row["patient_fhir_id"], "Planner original question/owner")
            plan = revision.get("plan")
            is_visible = revision.get("plan_visible") is True
            visible[str(is_visible)] += 1
            if is_visible:
                require(isinstance(plan, dict), "Visible plan absent")
                digest = hashlib.sha256(json.dumps(plan, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()
                require(digest == revision.get("plan_sha256"), "Frozen plan hash")
                require(len(systems) >= 2, "Visible plan has no solver system")
                solver_system = trace[systems[1]]["content"]
                marker = "Frozen advisory plan JSON:\n"
                require(marker in solver_system, "Advice absent from solver input")
                require(canonical(json.loads(solver_system.split(marker, 1)[1])) == canonical(plan), "Solver saw different advice")
                raw_replies = [message for message in trace[:systems[1]] if isinstance(message, dict) and message.get("role") == "assistant"]
                require(len(raw_replies) == 1 and not raw_replies[0].get("tool_calls"), "Planner was not a single tool-free pass")
                parsed, error = plan_ns["parse_plan"](raw_replies[0].get("content"), context["question"])
                require(error is None and canonical(parsed) == canonical(plan), "Question-only plan normalization differs")
                observed_plans += 1
            else:
                require(plan is None and revision.get("plan_sha256") is None, "Ignored plan still exposed")
                require(not any("Frozen advisory plan JSON:" in (trace[index].get("content") or "") for index in systems[1:]), "Ignored advice leaked into solver")
            row_boundaries = [message for message in trace if isinstance(message, dict) and message.get("phase") == "solver_boundary"]
            require(len(row_boundaries) <= 2 and all(message.get("model_call") is False for message in row_boundaries), "Host boundary accounting")
            boundaries += len(row_boundaries)
            policy = revision.get("release_policy")
            policies[str(policy)] += 1
            if policy == "source_valid_primary_preserved":
                require(revision.get("final_check_issues") == [] and not row.get("error"), "Release source contract")
                final = revision["final"]
                for display, artifact in [("agent_answer", "answer"), ("evidence", "evidence"), ("limitation", "limitation")]:
                    require(canonical(row.get(display)) == canonical(final.get(artifact)), "Released artifact/display binding")
            require(not any(phase["fatal"] for phase in phases) or policy != "source_valid_primary_preserved", "Fatal accounting released an artifact")
            per_row_metadata.append({"question_id": row["question_id"], "audit_sha256": ref["sha256"], "plan_sha256": revision.get("plan_sha256"), "plan_visible": is_visible,
                                     "llm_calls": tokens["llm_calls"], "phases": [{key: phase[key] for key in ("phase", "tokens", "tool_calls", "token_cap", "tool_cap", "fatal")} for phase in phases]})
        arms[arm] = {"scheduled_rows": 300, "saved_first_attempts_checked": len(saved), "gzip_owner_sha_crc_lossless_restorations": checked,
            "terminal_errors": errors, "unknown_usage_calls": unknown_usage, "reported_overshoots": overshoots, "phase_fatals": phase_fatals,
            "phase_counts": dict(phases_seen), "plan_visibility": dict(visible), "release_policies": dict(policies),
            "raw_snapshot_sha256": hashlib.sha256(raw).hexdigest(), "raw_snapshot_bytes": len(raw),
            "snapshot_observed_utc": datetime.now(timezone.utc).isoformat(), "team_operational_metadata": per_row_metadata}

    journal_reports, total_events = [], Counter()
    for path in sorted((output / "provider-events").glob("*.jsonl")):
        require(re.fullmatch(r"azure-diagnostics-[0-9]+-[a-f0-9]{32}\.jsonl", path.name), "Journal filename")
        raw = path.read_bytes()
        lines = raw.split(b"\n")
        tail = lines.pop()  # legitimate concurrent append may leave a partial tail
        events = [json.loads(line) for line in lines]
        starts, ends, kinds, fds = {}, {}, Counter(), []
        native = unavailable = unknowns = 0
        for event in events:
            event_schema(event)
            require(path.name == f"azure-diagnostics-{event['pid']}-{event['process_session']}.jsonl", "Journal owner/filename")
            key = (event["pid"], event["process_session"], event["counter"])
            target = starts if event["event"] == "start" else ends
            require(key not in target, "Duplicate journal event")
            target[key] = event
            kinds[event["event"]] += 1
            if event["fd_count"] is not None:
                fds.append(event["fd_count"])
            if event["event"] == "end":
                kinds[event["outcome"]] += 1
                unknowns += event["usage"]["total_tokens"] is None
                client = event["client"]
                if client["transport"] is None:
                    unavailable += 1
                else:
                    require(client["transport"]["class"] == "httpx2.HTTPTransport", "Nondefault SDK transport")
                    require(client["http"]["class"] == "openai._base_client.SyncHttpxClientWrapper" and "httpx2.Client" in client["http"]["mro"], "Nondefault SDK client")
                    require(client["versions"] == {"openai": "3.26.1", "httpx2": "2.13.1"}, "Unexpected SDK versions")
                    native += 1
        require(not set(ends) - set(starts), "Journal end without start")
        for key in set(starts) & set(ends):
            require(all(starts[key][field] == ends[key][field] for field in ("pid", "ppid", "counter", "process_session", "started_utc")), "Journal pair identity")
        pending = sorted(set(starts) - set(ends))
        newest = max((key[2] for key in starts), default=0)
        total_events.update(kinds)
        journal_reports.append({"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "counts": dict(kinds),
            "matched_pairs": len(set(starts) & set(ends)), "unmatched_latest_starts": sum(key[2] == newest for key in pending),
            "older_unmatched_starts_unknown_not_proven_inflight": sum(key[2] != newest for key in pending),
            "trailing_partial_bytes": len(tail), "native_default_httpx2_ends": native, "transport_metadata_unavailable_ends": unavailable,
            "unknown_usage_ends": unknowns, "fd_count_min": min(fds) if fds else None, "fd_count_max": max(fds) if fds else None,
            "snapshot_observed_utc": datetime.now(timezone.utc).isoformat()})
    require(journal_reports, "No provider journal evidence")
    require(all(sha(ROOT / name) == digest for name, digest in manifest["source_sha256"].items()), "Source changed during snapshot")
    require(all(sha(ROOT / name) == digest for name, digest in runtime["source_sha256"].items()), "Runtime changed during snapshot")
    report = {"status": "PASS", "scope": "Initial saved-file and journal snapshots; operational/provenance only, no scoring/ranking/completion claim",
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "helper_sha256": sha(__file__),
        "dependencies": {"prelaunch_audit_source.py": sha(BASE / "prelaunch_audit_source.py"), "scripts/audit_store.py": sha(ROOT / "scripts/audit_store.py")},
        "frozen_source_files_verified": 142, "source_archive_verified": True, "runtime_sources_and_exact4_archive_verified": True,
        "design_dataset_split_verified": True, "actual_provider_binding": expected_binding,
        "actual_identities_equal_prelaunch": True, "same_exact_ordered300_inputs_each": True, "global_caps": [200000, 30],
        "phase_caps": {"planner": [10000, 0], "solver": [190000, 30]}, "no_late_audit_or_targeted_phase_observed": True,
        "validated_plans_observed_in_solver_context": observed_plans, "no_model_host_boundaries_observed": boundaries,
        "independently_verified_uncompressed_artifact_bytes": artifact_bytes, "arms": arms,
        "journals": journal_reports, "journal_event_totals": dict(total_events), "anomalies": anomalies,
        "limits": ["Each raw result and journal was read once; snapshots occur at different times while owned runs continue.",
            "All completed attempts present in those two raw snapshots were checked; later saves are outside this report.",
            "Latest unmatched starts and partial journal tails may be in-flight; older unmatched starts remain unknown logging evidence.",
            "A public default transport class does not identify the selected proxy/network route; successful status may remain unknown.",
            "Usage/status missing from best-effort journals remains unknown. No per-row accuracy or reference comparison was performed.",
            "Source/owner/time checks and advice provenance do not establish question semantics or forecast benchmark accuracy."]}
    with (BASE / "initial_protocol_audit.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("PASS initial protocol snapshot; no scoring")
    print(json.dumps({"saved_checked": {arm: value["saved_first_attempts_checked"] for arm, value in arms.items()},
                      "visible_plans": observed_plans, "journal_events": dict(total_events),
                      "report_sha256": sha(BASE / "initial_protocol_audit.json")}))


if __name__ == "__main__":
    main()
