"""Read-only independent statistical QA of an already gated COMPLETE paired export.

No analyzer, scorer, launcher, live supervisor or provider modules are imported.
The existing exporter owns full first-clean classifier/audit replay. This utility
checks its immutable receipt and snapshot bindings, then independently reconciles
exported measurements and reported statistics. It never computes answer scores.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

N = 2931
ARMS = ("single_agent", "multi_agent")
STATUS = "V3_SINGLE_COMPLETE_DIRECT_PENDING"
FINGERPRINT = "65d2bbfc7da05f97e646acb22d20f0367d7f7a4e1a2a7c8b9c244fd77ef8d409"
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_prompt_tokens",
                "uncached_prompt_tokens", "repeated_prompt_tokens")
MEASURES = ("latency_seconds", "estimated_cost_usd", "reported_cost_usd", *TOKEN_FIELDS,
            "llm_calls", "tool_calls_executed", "tool_calls_recorded", "evidence_count",
            "evidence_valid", "valid_evidence_fraction", "evidence_precision", "evidence_recall")
PRIMARY = ("latency_seconds", "estimated_cost_usd", "total_tokens")
BOOTSTRAP, SEED = 10000, 5106


def require(test, message):
    if not test:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def semantic_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def same(actual, expected, location):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), location + ": keys differ")
        for key in expected:
            same(actual[key], expected[key], location + "." + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), location + ": list differs")
        for i, (a, e) in enumerate(zip(actual, expected)):
            same(a, e, location + "." + str(i))
    elif type(expected) in (float, int):
        require(type(actual) in (float, int) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=2e-11, abs_tol=1e-11), location + ": numeric mismatch")
    else:
        require(type(actual) is type(expected) and actual == expected, location + ": value mismatch")


def valid(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def number(value):
    return value if valid(value) else None


def linear_percentile(sorted_values, fraction):
    position = (len(sorted_values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] * (high - position) + sorted_values[high] * (position - low) if low != high else sorted_values[low]


def measure(values):
    observed = [v for v in values if valid(v)]
    missing = len(values) - len(observed)
    if observed:
        ordered = sorted(observed)
        stats = {"total": sum(observed), "mean": statistics.fmean(observed),
                 "median": statistics.median(ordered), "p95": linear_percentile(ordered, .95),
                 "stdev": statistics.stdev(observed) if len(observed) > 1 else None,
                 "min": ordered[0], "max": ordered[-1]}
    else:
        stats = dict.fromkeys(("total", "mean", "median", "p95", "stdev", "min", "max"))
    return {"n": len(values), "observed_n": len(observed), "missing_n": missing,
            **{key: None if missing else value for key, value in stats.items()}, "observed_only": stats}


def cluster_interval(values, patients):
    require(len(values) == len(patients) and all(type(x) in (int, float) and math.isfinite(x) for x in values),
            "Bootstrap values invalid")
    buckets = defaultdict(list)
    for value, patient in zip(values, patients):
        require(isinstance(patient, str) and bool(patient.strip()), "Bootstrap patient missing")
        buckets[patient].append(value)
    result = {"n": len(values), "patients": len(buckets),
              "estimate": float(np.mean(values)) if values else None, "ci_low": None, "ci_high": None}
    if len(buckets) < 2:
        return result
    # Independently aggregate Python buckets. Only the RNG family and exact draw
    # scheme intentionally match the frozen protocol, to reproduce its seeded CI.
    totals = np.array([math.fsum(buckets[p]) for p in sorted(buckets)], dtype=float)
    counts = np.array([len(buckets[p]) for p in sorted(buckets)], dtype=float)
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(len(buckets), np.repeat(1 / len(buckets), len(buckets)), size=BOOTSTRAP)
    means = weights.dot(totals) / weights.dot(counts)
    ordered = sorted(means.tolist())
    result.update(ci_low=linear_percentile(ordered, .025), ci_high=linear_percentile(ordered, .975))
    return result


def metric_values(rows, key):
    return [None if key in TOKEN_FIELDS and row["provider_usage_unknown"] else row[key] for row in rows]


def arm_summary(rows, intervals=True):
    patients = [r["patient_fhir_id"] for r in rows]
    flags = [int(r["correct"]) for r in rows]
    result = {"questions": len(rows), "correct": sum(flags),
              "accuracy": cluster_interval(flags, patients) if intervals else {
                  "n": len(rows), "patients": len(set(patients)), "estimate": statistics.fmean(flags) if rows else None,
                  "ci_low": None, "ci_high": None},
              "intrinsic_errors": sum(r["intrinsic_error"] for r in rows),
              "abstentions": sum(r["abstained"] for r in rows),
              "limitations": sum(r["has_limitation"] for r in rows),
              "budget_stops": sum(r["budget_stopped"] for r in rows),
              "provider_usage_unknown": sum(r["provider_usage_unknown"] for r in rows),
              "evidence_validated_questions": sum(r["evidence_validation_available"] for r in rows),
              "metrics": {key: measure(metric_values(rows, key)) for key in MEASURES}}
    if intervals:
        for key in PRIMARY:
            values = metric_values(rows, key)
            if values and all(valid(v) for v in values):
                result["metrics"][key]["mean_patient_bootstrap"] = cluster_interval(values, patients)
    return result


def cost(tokens, rates):
    values = [number(tokens.get(k)) for k in ("prompt", "cached_prompt", "completion")]
    if any(call.get("usage_available") is False for call in tokens.get("per_call") or [] if isinstance(call, dict)):
        return None
    if None in values or any(not valid(rates.get(k)) for k in ("input", "cached_input", "output")):
        return None
    prompt, cached, completion = values
    if cached > prompt:
        return None
    return ((prompt - cached) * rates["input"] + cached * rates["cached_input"] + completion * rates["output"]) / 1e6


def checked_path(base, relative):
    require(isinstance(relative, str), "Artifact path must be a string")
    path = base / relative
    require(not Path(relative).is_absolute() and path.resolve().is_relative_to(base), "Artifact path escapes export")
    require(path.is_file() and not path.is_symlink(), "Artifact missing or symlinked")
    return path


def final_gate(export):
    """Complete cohort/provenance gate before reading derived correctness fields."""
    export = Path(export).resolve()
    receipt_path = export / "V3_SINGLE_COMPLETE_DIRECT_PENDING.json"
    inventory_path = export / "delivery_artifact_hashes.json"
    receipt_bytes = receipt_path.read_bytes()
    inventory_bytes = inventory_path.read_bytes()
    stability = {receipt_path:hashlib.sha256(receipt_bytes).hexdigest(),
                 inventory_path:hashlib.sha256(inventory_bytes).hexdigest()}
    receipt = json.loads(receipt_bytes)
    require(receipt.get("status") == STATUS and receipt.get("paired_scope_complete") is True
            and receipt.get("benchmark_complete") is False and receipt.get("questions_per_arm") == N
            and set(receipt.get("participating_arms", [])) == set(ARMS), "Complete matched paired receipt required")
    require(receipt.get("inference_fingerprint") == FINGERPRINT
            and receipt.get("direct_llm", {}).get("included_or_scored") is False
            and receipt.get("original_three_arm_final_gate_changed") is False, "Receipt scope or identity differs")
    inventory = json.loads(inventory_bytes)
    require(isinstance(inventory, dict) and inventory, "Artifact inventory missing")
    for relative, detail in inventory.items():
        path = checked_path(export, relative)
        require(path.stat().st_size == detail["bytes"] and sha256(path) == detail["sha256"], "Delivery artifact changed: " + relative)
    for required in ("V3_SINGLE_COMPLETE_DIRECT_PENDING.json", "paired_complete/summary.json",
                     "paired_complete/per_question.jsonl", "paired_complete/per_question.csv"):
        require(required in inventory, "Required artifact is not hashed: " + required)
    snapshot = checked_path(export, "validated_paired_input_snapshot/run_manifest.json").parent
    manifest = read_json(snapshot / "run_manifest.json")
    ledger = read_json(snapshot / "acceptance.json")
    require(sha256(snapshot / "run_manifest.json") == receipt["current_manifest_sha256"], "Manifest binding differs")
    require(manifest["questions_per_arm"] == N and manifest["selected_arms"] == ["single_agent"], "Manifest scope differs")
    require(ledger["protocol"] == "v3-first-runtime-clean-terminal" and ledger["selection"] == "first_runtime_clean_terminal", "Ledger protocol differs")
    source = Path(manifest["input"]["path"])
    require(source.is_file() and sha256(source) == manifest["input"]["sha256"], "Frozen input missing or changed")
    stability[source] = manifest["input"]["sha256"]
    with source.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    ids = {}
    for row in source_rows:
        require(row.get("question_id") and row.get("patient_fhir_id") and row["question_id"] not in ids, "Input identity missing or duplicated")
        ids[row["question_id"]] = row
    require(len(ids) == N, "Full 2,931 question input required")
    canonical = {}
    for arm in ARMS:
        selections = ledger["arms"][arm]
        require(set(selections) == set(ids), arm + ": incomplete acceptance cohort")
        require(semantic_hash({"selections": selections, "counts": ledger["counts"][arm]})
                == receipt["accepted_ledger_slice_sha256"][arm], arm + ": ledger binding differs")
        path = snapshot / ("full_" + arm + ".json")
        require(sha256(path) == receipt["canonical_sha256"][arm], arm + ": canonical binding differs")
        identity = read_json(Path(str(path) + ".run.json"))
        require(identity == manifest["run_identities"][arm] and identity["code_hash"] == FINGERPRINT, arm + ": inference identity differs")
        indexed = {}
        for row in read_json(path):
            qid = row.get("question_id")
            require(qid in ids and qid not in indexed and row.get("patient_fhir_id") == ids[qid]["patient_fhir_id"], arm + ": canonical identity differs")
            require(not row.get("attempt_history") and "agent_answer" in row, arm + ": invalid canonical attempt")
            selection = selections[qid]
            require(selection.get("runtime_clean") is True and selection.get("selection") in ("runtime_clean_terminal", "intrinsic_terminal_error"), arm + ": first-clean proof missing")
            indexed[qid] = row
        require(set(indexed) == set(ids), arm + ": incomplete canonical cohort")
        replay = receipt["first_clean_replay_counts"][arm]
        require(replay == ledger["counts"][arm] and replay["raw_attempts"] - replay["runtime_failures"] == N,
                arm + ": runtime exclusion/replay counts differ")
        canonical[arm] = indexed
    # The completed exporter already replayed all sealed rounds and raw audits.
    # We verify its bound outputs, not execute a second classifier implementation.
    return export, receipt, manifest, ledger, ids, canonical, inventory, stability


def verify(export):
    export, receipt, manifest, ledger, source, canonical, inventory, stability = final_gate(export)
    report = read_json(export / "paired_complete/summary.json")
    require(report.get("status") == STATUS and report.get("paired_scope_complete") is True
            and report.get("benchmark_complete") is False and set(report["arms"]) == set(ARMS), "Projected report scope differs")
    same(report["statistics"], receipt["statistics"], "statistics receipt binding")
    require(report["statistics"]["bootstrap_replicates"] == BOOTSTRAP and report["statistics"]["seed"] == SEED
            and report["statistics"]["unit"] == "patient cluster" and report["statistics"]["confidence_level"] == .95, "Frozen statistics protocol differs")
    same(report["pricing"], receipt["pricing"], "pricing receipt binding")
    rates = report["pricing"]["per_million_tokens"]
    require(rates == {"input": .1, "cached_input": .01, "output": .5}, "Authorized official-price snapshot differs")
    require(read_json(export / "validated_paired_input_snapshot/pricing.json") == report["pricing"], "Pricing snapshot differs")
    by_arm = {arm: {} for arm in ARMS}
    with (export / "paired_complete/per_question.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            arm, qid = row.get("arm"), row.get("question_id")
            require(arm in by_arm and qid in source and qid not in by_arm[arm], "Per-question identity missing/duplicated/foreign")
            require(row["patient_fhir_id"] == source[qid]["patient_fhir_id"], "Per-question patient differs")
            by_arm[arm][qid] = row
    require(all(set(indexed) == set(source) for indexed in by_arm.values()), "Full per-question paired cohort required")
    usage_checks, per_call_checks = 0, 0
    for arm in ARMS:
        require(report["completeness"][arm] == {"expected":N,"accepted":N,"missing":0,"missing_question_ids":[],"complete":True}, "Report completeness differs")
        for qid, row in by_arm[arm].items():
            raw, reference = canonical[arm][qid], source[qid]
            for exported, original in (("final_answer", "agent_answer"), ("error", "error"),
                                        ("evidence", "evidence"), ("audit_artifacts", "audit_artifacts")):
                same(row.get(exported), raw.get(original), arm + ": canonical " + exported)
            require(row["question"] == reference["question"] and row["reference_answer"] == reference["true_answer"], "Reference binding differs")
            require(row["runtime_clean_selection_verified"] is True and row["acceptance_entry"] == ledger["arms"][arm][qid], "Per-question acceptance differs")
            for field in ("correct", "typed_answer_matches", "intrinsic_error", "abstained", "has_limitation", "budget_stopped", "provider_usage_unknown", "evidence_validation_available"):
                require(type(row[field]) is bool, "Non-boolean derived flag: " + field)
            intrinsic = raw.get("error") is not None and str(raw["error"]).strip() not in ("", "nan")
            require(row["intrinsic_error"] == intrinsic and row["correct"] == (row["typed_answer_matches"] and not intrinsic), "Terminal-error correctness conjunction differs")
            tokens = raw.get("tokens") if isinstance(raw.get("tokens"), dict) else {}
            calls = tokens.get("per_call")
            unknown = any(call.get("usage_available") is False for call in calls or [] if isinstance(call, dict))
            require(row["provider_usage_unknown"] == unknown, "Provider-usage missingness differs")
            same(row["per_call_usage"], calls, "Per-call usage binding")
            same(row["latency_seconds"], number(raw.get("latency")), "Latency binding")
            same(row["estimated_cost_usd"], cost(tokens, rates), "Cached-subset cost formula")
            same(row["reported_cost_usd"], number(tokens.get("cost")), "Reported billing binding")
            for exported, original in (("prompt_tokens","prompt"),("completion_tokens","completion"),("total_tokens","total"),
                                       ("cached_prompt_tokens","cached_prompt"),("repeated_prompt_tokens","repeated_prompt"),("llm_calls","llm_calls")):
                same(row[exported], number(tokens.get(original)), "Usage binding " + exported)
            p, c, o = row["prompt_tokens"], row["cached_prompt_tokens"], row["completion_tokens"]
            same(row["uncached_prompt_tokens"], p - c if p is not None and c is not None and c <= p else None, "Uncached token binding")
            if not unknown and all(valid(v) for v in (p, c, o, row["total_tokens"])):
                require(c <= p and p + o == row["total_tokens"], "Token balance/cache subset invalid")
                usage_checks += 1
            if calls and all(isinstance(call, dict) and call.get("usage_available") is not False
                             and all(valid(call.get(k)) for k in ("prompt","completion","cached")) for call in calls):
                for aggregate, key in ((p,"prompt"),(o,"completion"),(c,"cached")):
                    same(aggregate, sum(call[key] for call in calls), "Per-call aggregate " + key)
                per_call_checks += 1
            budget = raw.get("budget") if isinstance(raw.get("budget"), dict) else {}
            require(row["has_limitation"] == bool(raw.get("limitation"))
                    and row["budget_stopped"] == bool(budget.get("stopped")), "Limitation/budget flags differ")
            same(row["tool_calls_executed"], number(budget.get("tool_calls_run")), "Executed tools binding")
            same(row["tool_calls_recorded"], len(raw["tool_calls"]) if isinstance(raw.get("tool_calls"), list) else None, "Recorded tools binding")
            same(row["evidence_count"], len(raw["evidence"]) if isinstance(raw.get("evidence"), list) else None, "Evidence count binding")
    ordered = {arm: [by_arm[arm][qid] for qid in sorted(source)] for arm in ARMS}
    independent, cross_tabs = {}, {}
    for arm, rows in ordered.items():
        independent[arm] = arm_summary(rows)
        same(report["arms"][arm], independent[arm], "Arm statistics " + arm)
        cross_tabs[arm] = {"empty_no_value_flag_set": {"questions":sum(r["abstained"] for r in rows),
                              "correct":sum(r["correct"] and r["abstained"] for r in rows),
                              "wrong":sum(not r["correct"] and r["abstained"] for r in rows)},
                          "empty_no_value_flag_unset": {"questions":sum(not r["abstained"] for r in rows),
                              "correct":sum(r["correct"] and not r["abstained"] for r in rows),
                              "wrong":sum(not r["correct"] and not r["abstained"] for r in rows)},
                          "empty_reference_by_flag": dict(Counter(str((r["answer_type"] == "empty", r["abstained"], r["correct"])) for r in rows))}
        same(receipt["missing_measurements"][arm], {k:v["missing_n"] for k,v in independent[arm]["metrics"].items()}, "Receipt missingness " + arm)
    group_count = 0
    require(set(report["groups"]) == {"family","patient_fhir_id","template_id","answer_type","analysis_split","exposure"},
            "Expected aggregate group tables omitted or changed")
    for field, table in report["groups"].items():
        buckets = defaultdict(list)
        for arm, rows in ordered.items():
            for row in rows:
                buckets[(arm, str(row[field] or "unknown"))].append(row)
        seen = set()
        for group in table:
            key = (group["arm"], group["group"])
            require(key in buckets and key not in seen, "Aggregate group missing/duplicated")
            seen.add(key)
            same(group, {"arm":key[0],"group":key[1], **arm_summary(buckets[key], intervals=field not in ("patient_fhir_id","template_id"))}, "Group " + field)
            group_count += 1
        require(seen == buckets.keys(), "Aggregate group omitted")
    phases = defaultdict(list)
    for arm, rows in ordered.items():
        for row in rows:
            for phase in row.get("phases") or []:
                if isinstance(phase, dict):
                    phases[(arm, str(phase.get("phase", "unknown")))].append(phase)
    seen_phases = set()
    for item in report["phase_usage"]:
        key = (item["arm"],item["phase"])
        require(key in phases and key not in seen_phases, "Phase aggregate missing/duplicated")
        seen_phases.add(key)
        values = phases[key]
        same(item,{"arm":key[0],"phase":key[1],"questions_with_phase":len(values),
                   "tokens":measure([p.get("tokens") for p in values]),
                   "tool_calls":measure([p.get("tool_calls") for p in values]),
                   "fatal":sum(bool(p.get("fatal")) for p in values),
                   "has_diagnostic_errors":sum(bool(p.get("errors")) for p in values)},"Phase aggregate")
    require(seen_phases == phases.keys(), "Phase aggregate omitted")
    require(len(report["comparisons"]) == 1, "Exactly one matched comparison required")
    pair = report["comparisons"][0]
    left, right = pair["left"], pair["right"]
    require(left != right and {left,right} == set(ARMS), "Paired arms differ")
    aa, bb = ordered[left], ordered[right]
    patients = [r["patient_fhir_id"] for r in aa]
    paired = {"left":left,"right":right,"matched_questions":N,"left_only":0,"right_only":0,
              "left_wins":sum(x["correct"] and not y["correct"] for x,y in zip(aa,bb)),
              "right_wins":sum(y["correct"] and not x["correct"] for x,y in zip(aa,bb)),
              "both_correct":sum(x["correct"] and y["correct"] for x,y in zip(aa,bb)),
              "both_wrong":sum(not x["correct"] and not y["correct"] for x,y in zip(aa,bb)),
              "accuracy_difference":cluster_interval([int(x["correct"])-int(y["correct"]) for x,y in zip(aa,bb)],patients)}
    for key in PRIMARY:
        x, y = metric_values(aa,key), metric_values(bb,key)
        complete = [(a-b,p) for a,b,p in zip(x,y,patients) if valid(a) and valid(b)]
        paired[key + "_difference_observed_pairs"] = cluster_interval([d for d,p in complete],[p for d,p in complete])
        paired[key + "_missing_pairs"] = N - len(complete)
    same(pair, paired, "Paired left-minus-right statistics")
    # Cross-check the scalar CSV exports, including the literal JSON-encoding
    # convention used for None/bools/lists/dicts in the untouched exporter.
    csv.field_size_limit(sys.maxsize)
    csv_counts = {}
    for path in (export / "paired_complete").glob("*.csv"):
        with path.open(newline="",encoding="utf-8") as stream:
            values = list(csv.DictReader(stream))
        require(all(None not in row for row in values), "Malformed CSV fields")
        csv_counts[path.name] = len(values)
        if path.name == "per_question.csv":
            require(len(values) == 2*N, "CSV full cohort missing")
            seen = set()
            for row in values:
                key = (row["arm"],row["question_id"])
                require(key not in seen and key[0] in ARMS and key[1] in source, "CSV identity differs")
                seen.add(key)
                original = by_arm[key[0]][key[1]]
                for field,value in original.items():
                    rendered = json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list,tuple)) else str(value) if value is not None else ""
                    require(row[field] == rendered, "CSV/JSONL field differs: " + field)
    # Detect mutations during QA; the export/snapshot itself is never written.
    for relative, detail in inventory.items():
        require(sha256(checked_path(export,relative)) == detail["sha256"], "Export changed during QA")
    for path, digest in stability.items():
        require(sha256(path) == digest, "Receipt, inventory or input changed during QA")
    return {"status":"PASS","generated_at":datetime.now(timezone.utc).isoformat(),"export_directory":str(export),
            "questions_per_arm":N,"patients":len(set(patients)),"participating_arms":list(ARMS),
            "arm_statistics":independent,"empty_answer_flag_cross_tabs":cross_tabs,"paired_statistics":paired,
            "aggregate_groups_verified":group_count,"measurement_fields_verified":len(MEASURES),
            "delivery_artifacts_verified":len(inventory),"csv_rows_parsed":csv_counts,
            "phase_aggregates_verified":len(seen_phases),
            "token_balance_rows_verified":usage_checks,"per_call_aggregate_rows_verified":per_call_checks,
            "bootstrap":{"independent_implementation":True,"replicates":BOOTSTRAP,"seed":SEED,"unit":"patient cluster","level":.95},
            "no_answer_scorer_or_analyzer_called":True,"actual_incomplete_data_scored":False,
            "benchmark_complete":False,"paired_scope_complete":True,
            "limits":["Correctness and empty/no-value flags are reconciled as frozen scorer outputs, not independently rescored.",
                       "Raw first-clean classifier and compressed-audit replay remains the separate exporter gate; this utility checks its immutable receipt, ledger, canonical identities and artifacts.",
                       "The exact RNG family, seed and multinomial draw scheme match the frozen statistics protocol; aggregation and scalar statistics are independently implemented.",
                       "Dollar estimates use authorized OpenAI rates as an Azure proxy; actual billing remains unknown when unavailable."],
            "verifier_sha256":sha256(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "QA output must be NEW")
    require(not args.output.resolve().is_relative_to(args.export_dir.resolve()), "QA output must be outside immutable export")
    result = verify(args.export_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("status","questions_per_arm","patients","aggregate_groups_verified","delivery_artifacts_verified","verifier_sha256")}))


if __name__ == "__main__":
    main()
