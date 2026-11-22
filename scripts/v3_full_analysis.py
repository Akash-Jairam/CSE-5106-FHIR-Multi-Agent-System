"""Offline accepted-attempt analysis for the frozen V3 full benchmark.

Only full_{arm}.json is scored. Raw runtime failures under attempts/ are never
loaded, scored, or added to latency/token/cost totals. Accepted files must have
one runtime-clean outcome per question and no retry history. The supervisor's
acceptance.json is the selection audit trail; scorer output never selects rows.
"""
import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.experiment_stats import clustered_mean, paired_difference
from scripts.full_benchmark import iter_result_rows
from utils.answer_scorer import answer_type, is_abstention, score_answer
from utils.evidence_scorer import cited_refs, fetch_from_ndjson, parse_evidence, score_evidence

ARMS = ("direct_llm", "single_agent", "multi_agent")
QUESTIONS = 2931
BOOTSTRAP = 10000
SEED = 5106
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_prompt_tokens",
                "uncached_prompt_tokens", "repeated_prompt_tokens")
MEASURES = ("latency_seconds", "estimated_cost_usd", "reported_cost_usd", *TOKEN_FIELDS,
            "llm_calls", "tool_calls_executed", "tool_calls_recorded", "evidence_count",
            "evidence_valid", "valid_evidence_fraction", "evidence_precision", "evidence_recall")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def numeric(value):
    return value if finite(value) else None


def failed(row):
    return row.get("error") is not None and str(row["error"]).strip() not in ("", "nan")


def percentiles(values, percentile):
    """Linear interpolation, matching numpy's default percentile convention."""
    values = sorted(values)
    index = (len(values) - 1) * percentile / 100
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def measure(values):
    """Totals/means are complete-cohort values only; observed values are explicit."""
    valid = [value for value in values if finite(value)]
    missing = len(values) - len(valid)
    observed = {"total": sum(valid), "mean": statistics.fmean(valid),
                "median": statistics.median(valid), "p95": percentiles(valid, 95),
                "stdev": statistics.stdev(valid) if len(valid) > 1 else None,
                "min": min(valid), "max": max(valid)} if valid else dict.fromkeys(
                    ("total", "mean", "median", "p95", "stdev", "min", "max"))
    return {"n": len(values), "observed_n": len(valid), "missing_n": missing,
            **{key: value if not missing else None for key, value in observed.items()},
            "observed_only": observed}


def cost_estimate(tokens, pricing):
    """Cached tokens are a subset of prompt tokens, charged at their own rate."""
    rates = pricing.get("per_million_tokens") or {}
    prompt, completion, cached = (numeric(tokens.get(key)) for key in ("prompt", "completion", "cached_prompt"))
    unknown = any(call.get("usage_available") is False for call in tokens.get("per_call") or []
                  if isinstance(call, dict))
    values = (prompt, completion, cached, rates.get("input"), rates.get("cached_input"), rates.get("output"))
    if unknown or any(not finite(value) for value in values) or cached > prompt:
        return None
    return ((prompt - cached) * rates["input"] + cached * rates["cached_input"]
            + completion * rates["output"]) / 1_000_000


def issue_types(value):
    if value is None:
        return None
    if not isinstance(value, list):
        return ["malformed_issue_list"]
    return sorted(set(str(item.get("type", "untyped")) if isinstance(item, dict)
                      else str(item).partition(":")[0] for item in value))


def exposure_index(split):
    questions, patients = {}, defaultdict(set)
    for name, detail in split.get("splits", {}).items():
        for qid in detail.get("question_ids", []):
            if qid in questions:
                raise ValueError("Split manifest has overlapping question IDs")
            questions[qid] = name
        for patient in detail.get("patients", []):
            patient = patient.get("patient_fhir_id") if isinstance(patient, dict) else patient
            if patient:
                patients[patient].add(name)
    return questions, patients


def question_record(arm, row, reference, family, pricing, split_index, run_dir):
    """Compact inline diagnostics; never restore every compressed trace at once."""
    if row.get("attempt_history"):
        raise ValueError(f"{arm}: accepted results must not contain attempt_history")
    if "agent_answer" not in row:
        raise ValueError(f"{arm}: accepted result lacks agent_answer")
    if row.get("patient_fhir_id") != reference["patient_fhir_id"]:
        raise ValueError(f"{arm}: accepted patient identity differs from input")
    tokens = row.get("tokens") if isinstance(row.get("tokens"), dict) else {}
    revision = row.get("revision") if isinstance(row.get("revision"), dict) else {}
    draft, primary = revision.get("draft"), revision.get("primary")
    budget = row.get("budget") if isinstance(row.get("budget"), dict) else {}
    tool_calls = row.get("tool_calls")
    evidence = row.get("evidence")
    tool_counts = Counter(str(call.get("name", "unknown")) for call in tool_calls
                          if isinstance(call, dict)) if isinstance(tool_calls, list) else None
    qid = row["question_id"]
    splits, patient_splits = split_index
    exposed = bool(patient_splits.get(reference["patient_fhir_id"], set()) & {"dev", "pilot"})
    exposure = ("development_exposed_patient" if exposed else "held_out_patient_descriptive"
                if "held_out" in patient_splits.get(reference["patient_fhir_id"], set()) else "unassigned_patient")
    question, truth, template = reference["question"], reference["true_answer"], family["template"]

    def score(value):
        return bool(score_answer(value, truth, question, template)["typed_correct"])

    final_details = score_answer(row["agent_answer"], truth, question, template)
    final_score = bool(final_details["typed_correct"])
    draft_present = isinstance(draft, dict) and "answer" in draft
    primary_present = isinstance(primary, dict) and "answer" in primary
    per_call = tokens.get("per_call")
    usage_unknown = any(call.get("usage_available") is False for call in per_call or []
                        if isinstance(call, dict))
    prompt, cached = numeric(tokens.get("prompt")), numeric(tokens.get("cached_prompt"))
    resources = row.get("agent_fhir_resources")
    resource_counts = {str(key): len(value) for key, value in resources.items()
                       if isinstance(value, list)} if isinstance(resources, dict) else None
    audit = revision.get("audit") if isinstance(revision.get("audit"), dict) else {}
    verdicts = Counter(str(check.get("verdict", "unknown")) for check in audit.get("checks") or []
                       if isinstance(check, dict))
    proof = revision.get("targeted_proof") if isinstance(revision.get("targeted_proof"), dict) else {}
    final_issues = revision.get("final_check_issues")
    result = {
        "arm": arm, "question_id": qid, "patient_fhir_id": reference["patient_fhir_id"],
        "family": family.get("family", "unknown"), "template_id": family.get("template_id"),
        "template": template, "answer_type": answer_type(truth, template),
        "question": question, "reference_answer": truth,
        "dataset_split": family.get("split"), "analysis_split": splits.get(qid, "unassigned"),
        "exposure": exposure, "final_answer": row["agent_answer"], "typed_answer_matches": final_score,
        "typed_reason": final_details.get("typed_reason"), "typed_parsed": final_details.get("typed_parsed"),
        "correct": final_score and not failed(row), "intrinsic_error": failed(row), "error": row.get("error"),
        "abstained": is_abstention(row["agent_answer"]),
        "draft_present": draft_present, "draft_answer": draft.get("answer") if draft_present else None,
        "draft_correct": score(draft["answer"]) if draft_present else None,
        "primary_present": primary_present, "primary_answer": primary.get("answer") if primary_present else None,
        "primary_correct": score(primary["answer"]) if primary_present else None,
        "revised": revision.get("revised"), "answer_changed": revision.get("answer_changed"),
        "solver_source_revised": revision.get("solver_source_revised"),
        "targeted_revised": revision.get("targeted_revised"), "release_policy": revision.get("release_policy"),
        "revision_protocol": revision.get("protocol"), "limitation": row.get("limitation"),
        "has_limitation": bool(row.get("limitation")), "budget_stopped": bool(budget.get("stopped")),
        "budget_stop_reason": budget.get("stopped"), "latency_seconds": numeric(row.get("latency")),
        "estimated_cost_usd": cost_estimate(tokens, pricing), "reported_cost_usd": numeric(tokens.get("cost")),
        "prompt_tokens": prompt, "completion_tokens": numeric(tokens.get("completion")),
        "total_tokens": numeric(tokens.get("total")), "cached_prompt_tokens": cached,
        "uncached_prompt_tokens": prompt - cached if prompt is not None and cached is not None and cached <= prompt else None,
        "repeated_prompt_tokens": numeric(tokens.get("repeated_prompt")),
        "provider_usage_unknown": usage_unknown, "per_call_usage": per_call,
        "llm_calls": numeric(tokens.get("llm_calls")), "tool_calls_executed": numeric(budget.get("tool_calls_run")),
        "tool_calls_recorded": len(tool_calls) if isinstance(tool_calls, list) else None,
        "tool_counts_by_name": dict(tool_counts) if tool_counts is not None else None,
        "evidence_count": len(evidence) if isinstance(evidence, list) else None,
        "evidence": evidence, "gold_fhir_ids": reference.get("true_fhir_ids"),
        "evidence_validation_available": False, "evidence_valid": None,
        "valid_evidence_fraction": None, "evidence_precision": None, "evidence_recall": None,
        "evidence_issues": None,
        "has_evidence": bool(evidence) if isinstance(evidence, list) else None,
        "evidence_resource_ids": sorted(set(str(item.get("resource_id")) for item in evidence
                                          if isinstance(item, dict) and item.get("resource_id"))) if isinstance(evidence, list) else None,
        "retrieved_resource_counts": resource_counts,
        "source_check_available": final_issues is not None,
        "source_check_clean": final_issues == [] if final_issues is not None else None,
        "source_issue_types": issue_types(final_issues), "initial_source_issue_types": issue_types(revision.get("check_issues")),
        "contract_present": revision.get("contract") is not None if revision else None,
        "contract_error": revision.get("contract_error"), "audit_error": revision.get("audit_error"),
        "audit_verdict_counts": dict(verdicts), "targeted_proof_status": proof.get("status"),
        "targeted_proof_assurance": proof.get("assurance"), "unresolved_scope": revision.get("unresolved_scope"),
        "complete_types": revision.get("complete_types"), "phases": revision.get("phases"),
        "context_compactions": revision.get("context_compactions"),
        "audit_artifacts": row.get("audit_artifacts"),
        "accepted_results_path": str((run_dir / f"full_{arm}.json").resolve()),
        "acceptance_ledger_path": str((run_dir / "acceptance.json").resolve()),
    }
    return result


def metric_values(records, name):
    return [None if name in TOKEN_FIELDS and row["provider_usage_unknown"] else row[name] for row in records]


def summarize(records, n_boot=BOOTSTRAP, intervals=True):
    accuracy = (clustered_mean([int(row["correct"]) for row in records],
                              [row["patient_fhir_id"] for row in records], n_boot=n_boot, seed=SEED)
                if intervals else {"n": len(records), "patients": len({row["patient_fhir_id"] for row in records}),
                    "estimate": statistics.fmean(row["correct"] for row in records) if records else None,
                    "ci_low": None, "ci_high": None})
    result = {"questions": len(records), "correct": sum(row["correct"] for row in records),
              "accuracy": accuracy, "intrinsic_errors": sum(row["intrinsic_error"] for row in records),
              "abstentions": sum(row["abstained"] for row in records),
              "limitations": sum(row["has_limitation"] for row in records),
              "budget_stops": sum(row["budget_stopped"] for row in records),
              "provider_usage_unknown": sum(row["provider_usage_unknown"] for row in records),
              "evidence_validated_questions": sum(row["evidence_validation_available"] for row in records),
              "metrics": {name: measure(metric_values(records, name)) for name in MEASURES}}
    if intervals:
        for name in ("latency_seconds", "estimated_cost_usd", "total_tokens"):
            values = metric_values(records, name)
            if values and all(finite(value) for value in values):
                result["metrics"][name]["mean_patient_bootstrap"] = clustered_mean(
                    values, [row["patient_fhir_id"] for row in records], n_boot=n_boot, seed=SEED)
    return result


def mechanism_summary(records):
    drafts = [row for row in records if row["draft_present"]]
    targeted = [row for row in records if row["targeted_revised"]]
    counts = lambda key: dict(Counter(str(row[key]) for row in records if row[key] is not None))
    return {"questions": len(records), "drafts": len(drafts), "draft_correct": sum(row["draft_correct"] for row in drafts),
            "wrong_to_right": sum(not row["draft_correct"] and row["correct"] for row in drafts),
            "right_to_wrong": sum(row["draft_correct"] and not row["correct"] for row in drafts),
            "revised": sum(bool(row["revised"]) for row in records),
            "answer_changed": sum(bool(row["answer_changed"]) for row in records),
            "solver_source_revised": sum(bool(row["solver_source_revised"]) for row in records),
            "targeted_revised": len(targeted),
            "targeted_primary_wrong_to_final_right": sum(row["primary_correct"] is False and row["correct"] for row in targeted),
            "targeted_primary_right_to_final_wrong": sum(row["primary_correct"] is True and not row["correct"] for row in targeted),
            "source_checks_available": sum(row["source_check_available"] for row in records),
            "source_checks_clean": sum(row["source_check_clean"] is True for row in records),
            "release_policies": counts("release_policy"), "proof_statuses": counts("targeted_proof_status"),
            "revision_protocols": counts("revision_protocol"),
            "note": "Source-check flags are agent diagnostics, not independent proof of answer correctness. "
                    "Targeted proofs establish local constraints only; draft transitions include all revisions."}


def paired(records_by_arm, n_boot):
    result = []
    for left, right in itertools.combinations(ARMS, 2):
        a = {row["question_id"]: row for row in records_by_arm[left]}
        b = {row["question_id"]: row for row in records_by_arm[right]}
        ids = sorted(a.keys() & b.keys())
        aa, bb = [a[qid] for qid in ids], [b[qid] for qid in ids]
        detail = {"left": left, "right": right, "matched_questions": len(ids),
                  "left_only": len(a.keys() - b.keys()), "right_only": len(b.keys() - a.keys()),
                  "left_wins": sum(x["correct"] and not y["correct"] for x, y in zip(aa, bb)),
                  "right_wins": sum(y["correct"] and not x["correct"] for x, y in zip(aa, bb)),
                  "both_correct": sum(x["correct"] and y["correct"] for x, y in zip(aa, bb)),
                  "both_wrong": sum(not x["correct"] and not y["correct"] for x, y in zip(aa, bb)),
                  "accuracy_difference": paired_difference([int(row["correct"]) for row in aa],
                      [int(row["correct"]) for row in bb], [row["patient_fhir_id"] for row in aa], n_boot=n_boot, seed=SEED)}
        for key in ("latency_seconds", "estimated_cost_usd", "total_tokens"):
            values_a, values_b = metric_values(aa, key), metric_values(bb, key)
            complete = [(x, y, row["patient_fhir_id"]) for x, y, row in zip(values_a, values_b, aa)
                        if finite(x) and finite(y)]
            detail[f"{key}_difference_observed_pairs"] = paired_difference(
                [item[0] for item in complete], [item[1] for item in complete], [item[2] for item in complete], n_boot=n_boot, seed=SEED)
            detail[f"{key}_missing_pairs"] = len(ids) - len(complete)
        result.append(detail)
    return result


def resolved_input(manifest, key, fallback):
    detail = manifest.get(key) or {}
    value = detail.get("path") if isinstance(detail, dict) else detail
    path = Path(value or fallback)
    path = path if path.is_absolute() else ROOT / path
    if isinstance(detail, dict) and detail.get("sha256") and sha256(path) != detail["sha256"]:
        raise ValueError(f"{key}: source hash differs from frozen run manifest")
    return path.resolve()


def csv_index(path, required):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(required) <= set(reader.fieldnames or []):
            raise ValueError(f"{path.name}: missing required columns")
        result = {}
        for row in reader:
            qid = row["question_id"]
            if not qid or qid in result:
                raise ValueError(f"{path.name}: question IDs must be unique and nonempty")
            result[qid] = row
        return result


def evidence_analysis(records, fhir_dir):
    """Validate local fields and gold resource overlap, without contacting FHIR."""
    evidence_lists = [parse_evidence(row["evidence"]) for row in records]
    resources = fetch_from_ndjson(cited_refs(evidence_lists), str(fhir_dir))
    for row, evidence in zip(records, evidence_lists):
        gold = row["gold_fhir_ids"]
        try:
            gold = ast.literal_eval(gold) if isinstance(gold, str) and gold.strip() else gold
        except (ValueError, SyntaxError):
            gold = None
        gold_known = isinstance(gold, (list, dict))
        metrics = score_evidence(evidence, resources, gold if gold_known else [])
        row["evidence_validation_available"] = True
        for key in ("evidence_valid", "valid_evidence_fraction", "evidence_precision", "evidence_recall"):
            row[key] = numeric(float(metrics[key])) if gold_known or key not in ("evidence_precision", "evidence_recall") else None
        row["evidence_issues"] = metrics["evidence_issues"]
    return {"mode": "offline_refs_only_ndjson", "path": str(Path(fhir_dir).resolve()),
            "resource_index_entries": len(resources), "questions": len(records),
            "note": "Checks cited field values and resource overlap; this does not prove that cited evidence answers the question. "
                    "No-citation valid fractions and undefined ratios are null; empty-set overlap follows the unchanged evidence scorer."}


def analyze(run_dir, partial=False, n_boot=BOOTSTRAP, expected_questions=QUESTIONS, fhir_dir=None):
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        manifest_path = run_dir / "full_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    acceptance_path = run_dir / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text()) if acceptance_path.exists() else {}
    if not partial and (acceptance.get("protocol") != "v3-first-runtime-clean-terminal"
                        or acceptance.get("selection") != "first_runtime_clean_terminal"
                        or acceptance.get("complete") is not True):
        raise ValueError("A final report requires the complete first-runtime-clean acceptance ledger")
    pricing_path = run_dir / "pricing.json"
    pricing = json.loads(pricing_path.read_text()) if pricing_path.exists() else {
        "label": "Pricing unavailable; estimated cost unknown", "per_million_tokens": {}}
    pricing_hash = (manifest.get("pinned_files_sha256") or {}).get("pricing.json")
    if pricing_hash and (not pricing_path.exists() or sha256(pricing_path) != pricing_hash):
        raise ValueError("Pricing schedule differs from the frozen manifest")
    scorer_hash = (manifest.get("source_sha256") or {}).get("utils/answer_scorer.py")
    if scorer_hash and sha256(ROOT / "utils/answer_scorer.py") != scorer_hash:
        raise ValueError("Typed scorer differs from the frozen manifest")
    input_path = resolved_input(manifest, "input", ROOT / "final_dataset/questions_answers_sql_fhir.csv")
    families_path = resolved_input(manifest, "families", ROOT / "manifests/question_families.csv")
    split_path = run_dir / "split.json"
    if not split_path.exists():
        split_path = resolved_input(manifest, "split_manifest", ROOT / "manifests/split.json")
    elif (manifest.get("split_manifest") or {}).get("sha256") and sha256(split_path) != manifest["split_manifest"]["sha256"]:
        raise ValueError("Split snapshot differs from the frozen manifest")
    split = json.loads(split_path.read_text())
    split_index = exposure_index(split)
    source = csv_index(input_path, ("question_id", "patient_fhir_id", "question", "true_answer"))
    families = csv_index(families_path, ("question_id", "family", "template"))
    if len(source) != expected_questions:
        raise ValueError(f"Full input must have {expected_questions} questions")
    if not source.keys() <= families.keys():
        raise ValueError("Family manifest does not cover the full input")
    by_arm, provenance, completeness = {}, {}, {}
    identity_binding = None
    for arm in ARMS:
        path = run_dir / f"full_{arm}.json"
        arm_ledger = (acceptance.get("arms") or {}).get(arm) or {}
        if not partial and set(arm_ledger) != set(source):
            raise ValueError(f"{arm}: acceptance ledger question cohort differs from the full input")
        selected = {}
        if path.exists():
            for row in iter_result_rows(path):
                if not isinstance(row, dict) or row.get("question_id") not in source:
                    raise ValueError(f"{arm}: unknown or malformed accepted question")
                qid = row["question_id"]
                if qid in selected:
                    raise ValueError(f"{arm}: duplicate accepted question")
                selection = arm_ledger.get(qid)
                if selection is not None:
                    if (not isinstance(selection, dict) or selection.get("runtime_clean") is not True
                            or selection.get("selection") not in ("runtime_clean_terminal", "intrinsic_terminal_error")):
                        raise ValueError(f"{arm}: accepted question lacks runtime-clean selection proof")
                    raw_path = (run_dir / selection.get("attempt_file", "")).resolve()
                    if not raw_path.is_relative_to(run_dir) or not raw_path.is_file():
                        raise ValueError(f"{arm}: acceptance raw attempt pointer is missing or escapes run directory")
                elif not partial:
                    raise ValueError(f"{arm}: accepted question has no selection ledger entry")
                selected[qid] = question_record(arm, row, source[qid], families[qid], pricing, split_index, run_dir)
                selected[qid]["acceptance_entry"] = selection
                selected[qid]["runtime_clean_selection_verified"] = selection is not None
        missing = sorted(source.keys() - selected.keys())
        completeness[arm] = {"expected": expected_questions, "accepted": len(selected), "missing": len(missing),
                             "missing_question_ids": missing, "complete": not missing}
        if missing and not partial:
            raise ValueError(f"{arm}: {len(missing)} questions missing; use --partial only for labeled progress analysis")
        by_arm[arm] = [selected[qid] for qid in sorted(selected)]
        identity_path = Path(str(path) + ".run.json")
        identity = json.loads(identity_path.read_text()) if identity_path.exists() else None
        expected_identity = (manifest.get("run_identities") or {}).get(arm)
        if not partial and (identity is None or expected_identity is None):
            raise ValueError(f"{arm}: final report requires saved and manifest-pinned arm identities")
        if expected_identity is not None and identity != expected_identity and (identity is not None or not partial):
            raise ValueError(f"{arm}: accepted identity differs from the frozen manifest identity")
        if identity is not None:
            binding = {key: identity.get(key) for key in ("model", "max_total_tokens", "max_tool_calls", "input_hash", "code_hash")}
            if identity_binding is not None and binding != identity_binding:
                raise ValueError("Accepted arm identities disagree on model, budgets, input, or frozen code")
            identity_binding = binding
        provenance[arm] = {"path": str(path), "sha256": sha256(path) if path.exists() else None, "identity": identity}
    all_records = [row for arm in ARMS for row in by_arm[arm]]
    fhir_dir = fhir_dir or manifest.get("offline_fhir_dir")
    evidence_provenance = evidence_analysis(all_records, fhir_dir) if fhir_dir else {
        "mode": "not_run", "reason": "Local --fhir-dir or manifest offline_fhir_dir was not provided; evidence retained for later analysis."}
    complete = all(item["complete"] for item in completeness.values())
    groups = {}
    for field in ("family", "patient_fhir_id", "template_id", "answer_type", "analysis_split", "exposure"):
        table = []
        for arm, rows in by_arm.items():
            buckets = defaultdict(list)
            for row in rows:
                buckets[str(row[field] or "unknown")].append(row)
            for group, selected in sorted(buckets.items()):
                table.append({"arm": arm, "group": group,
                              **summarize(selected, n_boot=n_boot, intervals=field not in ("patient_fhir_id", "template_id"))})
        groups[field] = table
    phase_rows = []
    for arm, rows in by_arm.items():
        phases = defaultdict(list)
        for row in rows:
            for phase in row["phases"] or []:
                if isinstance(phase, dict):
                    phases[str(phase.get("phase", "unknown"))].append(phase)
        for name, phase in sorted(phases.items()):
            phase_rows.append({"arm": arm, "phase": name, "questions_with_phase": len(phase),
                               "tokens": measure([item.get("tokens") for item in phase]),
                               "tool_calls": measure([item.get("tool_calls") for item in phase]),
                               "fatal": sum(bool(item.get("fatal")) for item in phase),
                               "has_diagnostic_errors": sum(bool(item.get("errors")) for item in phase)})
    report = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
              "status": "COMPLETE" if complete else "PARTIAL_PROGRESS_DO_NOT_REPORT_AS_FINAL",
              "partial_mode": partial, "expected_questions_per_arm": expected_questions,
              "completeness": completeness, "run_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
              "inputs": {"dataset": {"path": str(input_path), "sha256": sha256(input_path)},
                         "families": {"path": str(families_path), "sha256": sha256(families_path)},
                         "split": {"path": str(split_path), "sha256": sha256(split_path)}},
              "accepted_results": provenance, "pricing": pricing,
              "evidence_analysis": evidence_provenance,
              "scoring": {"path": str(ROOT / "utils/answer_scorer.py"), "sha256": sha256(ROOT / "utils/answer_scorer.py"),
                          "rule": "Unchanged typed scorer on accepted runtime-clean final answers; intrinsic terminal errors are incorrect. "
                                  "No raw runtime failures or retry history contribute to reported measurements."},
              "statistics": {"bootstrap_replicates": n_boot, "seed": SEED, "confidence_level": 0.95,
                             "unit": "patient cluster", "estimand": "question-weighted mean; paired differences left minus right",
                             "percentile_convention": "linear interpolation", "stdev": "sample standard deviation; undefined for n<2"},
              "interpretation": ["All latency/cost/token results use only selected runtime-clean attempts. Runtime retries remain separate operational records.",
                                 "Summed per-question latency is not wall-clock benchmark duration; arms and workers may overlap.",
                                 "A missing measurement makes a complete-cohort total/mean unavailable. Observed-only statistics are separately labeled.",
                                 "Pricing estimates use the frozen official OpenAI schedule; the user authorized assuming the same rates for Azure. Actual billed charges are not measured.",
                                 "Split/exposure results are descriptive. Development used repeated subsets and later designs; the historic split does not certify an untouched test set.",
                                 "Architecture comparisons include shared tools/contracts and topology; they do not isolate a pure topology effect."],
              "arms": {arm: summarize(rows, n_boot=n_boot) for arm, rows in by_arm.items()},
              "all_arms_accepted_metrics": {name: measure(metric_values(all_records, name)) for name in MEASURES},
              "groups": groups, "comparisons": paired(by_arm, n_boot),
              "mechanisms": {arm: mechanism_summary(rows) for arm, rows in by_arm.items()}, "phase_usage": phase_rows,
              "audit_provenance": {"acceptance_ledger": str(run_dir / "acceptance.json"),
                                   "acceptance_sha256": sha256(run_dir / "acceptance.json") if (run_dir / "acceptance.json").exists() else None,
                                   "raw_attempt_directory": str(run_dir / "attempts"),
                                   "note": "Audit descriptors are retained per question. Full compressed resources/traces were not inflated for this report."}}
    state_path = run_dir / "supervisor_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    events_path = run_dir / "operations.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()] if events_path.exists() else []
    starts = [event["at"] for event in events if event.get("event") == "supervisor_started" and event.get("at")]
    started = min(starts) if starts else state.get("started_at")
    endpoint = state.get("finished_at") or report["generated_at"]
    elapsed = None
    if started:
        elapsed = (datetime.fromisoformat(endpoint) - datetime.fromisoformat(started)).total_seconds()
    arm_wall = {}
    for arm in ARMS:
        arm_starts = [event["at"] for event in events if event.get("arm") == arm and event.get("event") == "batch_started"]
        arm_ends = [event["at"] for event in events if event.get("arm") == arm and event.get("event") == "batch_sealed"]
        first = min(arm_starts) if arm_starts else None
        last = max(arm_ends) if arm_ends else None
        arm_wall[arm] = {"first_batch_started_at": first, "last_batch_sealed_at": last,
                         "wall_clock_seconds": (datetime.fromisoformat(last) - datetime.fromisoformat(first)).total_seconds()
                                               if first and last else None}
    report["operational_context"] = {"excluded_attempt_counts": acceptance.get("counts"),
                                     "supervisor_started_at": started,
                                     "supervisor_finished_at": state.get("finished_at"), "wall_clock_seconds": elapsed,
                                     "wall_clock_as_of": endpoint, "per_arm": arm_wall,
                                     "note": "Operational counts/wall-clock may include excluded runtime attempts. "
                                             "Whole-run duration ends at supervisor finish when available, otherwise at analysis generation. "
                                             "Per-arm duration is first batch start to last seal. These wall durations are distinct from summed accepted latency."}
    return report, all_records


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list))
                             else value for key, value in row.items()})


def display(value, decimals=4):
    return "unknown" if value is None else f"{value:.{decimals}f}"


def write_report(report, records, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "per_question.csv", records)
    with (destination / "per_question.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_csv(destination / "arm_summary.csv", ({"arm": arm, **flatten(value)} for arm, value in report["arms"].items()))
    for field, rows in report["groups"].items():
        write_csv(destination / f"by_{field}.csv", (flatten(row) for row in rows))
    write_csv(destination / "paired_comparisons.csv", (flatten(row) for row in report["comparisons"]))
    write_csv(destination / "mechanisms.csv", ({"arm": arm, **flatten(value)} for arm, value in report["mechanisms"].items()))
    write_csv(destination / "phase_usage.csv", (flatten(row) for row in report["phase_usage"]))
    lines = ["V3 full benchmark: " + report["status"], "", "Accepted runtime-clean attempts only; failed runtime attempts excluded.",
             f"Patient-cluster bootstrap: {report['statistics']['bootstrap_replicates']} replicates, seed {SEED}, 95% percentile intervals.",
             "Cost schedule: " + str(report["pricing"].get("label", "Frozen pricing.json; Azure billing unknown")), ""]
    for arm, detail in report["arms"].items():
        accuracy, metrics = detail["accuracy"], detail["metrics"]
        lines.append(f"{arm}: {detail['correct']}/{detail['questions']} correct; accuracy={display(accuracy['estimate'])}; "
                     f"95% CI [{display(accuracy['ci_low'])}, {display(accuracy['ci_high'])}]")
        lines.append(f"  latency seconds: total={display(metrics['latency_seconds']['total'], 2)}, mean={display(metrics['latency_seconds']['mean'], 2)}, "
                     f"median={display(metrics['latency_seconds']['median'], 2)}, p95={display(metrics['latency_seconds']['p95'], 2)}")
        lines.append(f"  estimated USD: total={display(metrics['estimated_cost_usd']['total'], 6)}, "
                     f"mean={display(metrics['estimated_cost_usd']['mean'], 6)}; tokens total={display(metrics['total_tokens']['total'], 0)}")
        lines.append(f"  intrinsic errors={detail['intrinsic_errors']}; limitations={detail['limitations']}; "
                     f"budget stops={detail['budget_stops']}; missing latency={metrics['latency_seconds']['missing_n']}; "
                     f"missing cost={metrics['estimated_cost_usd']['missing_n']}")
    lines.append("")
    for item in report["comparisons"]:
        diff = item["accuracy_difference"]
        lines.append(f"{item['left']} minus {item['right']}: accuracy difference={display(diff['estimate'])}, "
                     f"95% CI [{display(diff['ci_low'])}, {display(diff['ci_high'])}], wins/losses={item['left_wins']}/{item['right_wins']}, "
                     f"matched questions={item['matched_questions']}")
    lines.extend(["", *report["interpretation"], "", "Per-question answers, stage scores, usage, mechanisms and raw audit pointers: per_question.jsonl / per_question.csv.",
                  "Detailed aggregates and provenance: summary.json; group tables and paired_comparisons.csv."])
    (destination / "report.txt").write_text("\n".join(lines) + "\n")
    (destination / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--partial", action="store_true", help="Clearly labeled progress only; incomplete cohorts otherwise fail")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    parser.add_argument("--fhir-dir", type=Path, help="Local NDJSON snapshot for independent offline evidence validation")
    args = parser.parse_args()
    report, records = analyze(args.run_dir, partial=args.partial, n_boot=args.bootstrap, fhir_dir=args.fhir_dir)
    destination = args.output_dir or args.run_dir / ("analysis_partial" if args.partial else "analysis")
    write_report(report, records, destination)
    print(f"{report['status']}: analysis saved to {destination.resolve()}")


if __name__ == "__main__":
    main()
