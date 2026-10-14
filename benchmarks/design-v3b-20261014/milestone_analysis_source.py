"""Post-stop milestone inventory, not a complete-cohort benchmark ranking.

Offline only. Gold answers enter this report, never an agent or its decisions.
All stopped first attempts and pending rows are retained without replacement.
"""
import collections
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.audit_store import verify_result_audit
from scripts.full_benchmark import iter_result_rows
from scripts.variability import first_attempt, is_failed
from utils.answer_scorer import score_answer

RECORD = Path(__file__).resolve().parent
OUTPUT = ROOT / "output/experiments/gpt-6-luna/query_contract_v3b_300q"
RESULT = OUTPUT / "ordinary_multi_agent.json"
CONTROL = ROOT / "output/experiments/gpt-6-luna/verified_team_v2_300q/ordinary_single_agent.json"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(row):
    row = first_attempt(row)
    return {key: row.get(key) for key in ("question_id", "agent_answer", "evidence", "limitation", "latency", "error", "revision", "budget")}


def main():
    manifest = json.loads((RECORD / "manifest.json").read_text())
    mismatches = [name for name, expected in manifest["source_sha256"].items()
                  if sha(ROOT / name) != expected]
    if mismatches or sha(RECORD / "source.zip") != manifest["source_archive_sha256"]:
        raise ValueError("Frozen design source has changed")
    references = {row["question_id"]: row for row in csv.DictReader(
        (ROOT / "final_dataset/questions_answers_sql_fhir.csv").open())}
    families = {row["question_id"]: row for row in csv.DictReader(
        (ROOT / "manifests/question_families.csv").open())}
    expected = set(json.loads((ROOT / "manifests/split.json").read_text())["splits"]["held_out"]["question_ids"])
    rows, verified = {}, 0
    for original in iter_result_rows(RESULT):
        qid = original["question_id"]
        if qid in rows:
            raise ValueError("Duplicate stopped result question")
        verify_result_audit(original, OUTPUT, question_id=qid)
        verified += 1
        rows[qid] = compact(original)
    if set(rows) != expected:
        raise ValueError("Scheduled stopped cohort differs from the fixed 300")
    control = {row["question_id"]: compact(row) for row in iter_result_rows(CONTROL)}
    if set(control) != expected:
        raise ValueError("Control differs from the scheduled fixed cohort")
    attempted = sorted(qid for qid, row in rows.items()
                       if row.get("latency") is not None or bool(row.get("error")))
    nonterminal = [qid for qid in attempted if not is_failed(rows[qid])]
    failures = [qid for qid in attempted if is_failed(rows[qid])]
    pending = sorted(expected - set(attempted))

    def correct(index, qid):
        row, ref = index[qid], references[qid]
        return bool(not is_failed(row) and score_answer(
            row.get("agent_answer"), ref["true_answer"], ref["question"],
            families[qid]["template"])["typed_correct"])

    team_correct = sum(correct(rows, qid) for qid in nonterminal)
    control_correct = sum(correct(control, qid) for qid in nonterminal)
    def consistent_source_release(qid):
        row = rows[qid]
        revision = row.get("revision") or {}
        final = revision.get("final")
        if revision.get("final_check_issues") != [] or not isinstance(final, dict):
            return False
        displayed = {"answer": row.get("agent_answer"), "evidence": row.get("evidence"), "limitation": row.get("limitation")}
        recorded = {key: final.get(key) for key in displayed}
        return json.dumps(displayed, sort_keys=True, allow_nan=False) == json.dumps(recorded, sort_keys=True, allow_nan=False)

    source_released = [qid for qid in nonterminal if consistent_source_release(qid)]
    def error_kind(qid):
        error = str(rows[qid]["error"])
        status = re.search(r"Error code: (\d{3})", error)
        if status:
            return "azure_http_" + status.group(1)
        if re.search(r"APITimeout|Request timed out", error, re.I):
            return "client_sdk_timeout"
        return "other_terminal"

    errors = collections.Counter(error_kind(qid) for qid in failures)
    archive = ROOT / "output/archives/query_contract_v3b_300q_stopped_azure_incident.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    allowed = {RESULT, Path(str(RESULT) + ".run.json"), OUTPUT / ".local-provider.json", OUTPUT / "run.log"}
    allowed.update((OUTPUT / "audit").glob("*.json.gz"))
    files = {str(path.relative_to(OUTPUT)): {"sha256": sha(path), "bytes": path.stat().st_size}
             for path in sorted(allowed) if path.is_file()}
    if not archive.exists():
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as snapshot:
            for name in files:
                snapshot.write(OUTPUT / name, name)
    with zipfile.ZipFile(archive) as snapshot:
        if snapshot.testzip() is not None or set(snapshot.namelist()) != set(files):
            raise ValueError("Stopped raw archive verification failed")
        for name, identity in files.items():
            digest = hashlib.sha256()
            with snapshot.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != identity["sha256"]:
                raise ValueError("Preserved raw archive content differs from its source")
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "milestone_benchmark_pending_stopped_at_user_request",
        "stop_reason": "User requested stopping because of the reported Azure incident",
        "stopped_at_utc": "2026-10-14T16:02:22Z", "owned_process_group": 38353,
        "launcher_exit_code": 143, "owned_processes_verified_exited": True,
        "protocol": "query-contract-v3.2-static-audit",
        "branch": "experiments/query-contract-v3-compat", "source_commit": manifest["code_commit"],
        "source_files_verified_unchanged": len(manifest["source_sha256"]),
        "source_archive_sha256": sha(RECORD / "source.zip"),
        "scheduled_questions": len(expected), "recorded_first_attempts": len(attempted),
        "terminal_errors": len(failures), "terminal_error_types": dict(errors),
        "recorded_nonterminal_attempts": len(nonterminal),
        "not_saved_as_completed": len(pending),
        "pending_note": "Includes unattempted or interrupted in-flight questions; their usage/outcome is unknown, not zero",
        "question_ids": {"recorded": attempted, "terminal": failures, "not_saved_as_completed": pending},
        "source_valid_releases": len(source_released),
        "host_boundary_serialization_errors": sum("HostBoundaryMessage" in str(rows[qid]["error"]) for qid in failures),
        "verified_scheduled_row_audits": verified,
        "success_selected_comparison": {
            "scope": "Only recorded team attempts without terminal errors; identical question IDs for the historical refreshed single",
            "questions": len(nonterminal), "team_correct": team_correct,
            "team_accuracy": team_correct / len(nonterminal),
            "refreshed_single_correct_on_same_questions": control_correct,
            "refreshed_single_accuracy_on_same_questions": control_correct / len(nonterminal),
            "team_wins": sum(correct(rows, qid) and not correct(control, qid) for qid in nonterminal),
            "team_losses": sum(correct(control, qid) and not correct(rows, qid) for qid in nonterminal),
            "correct_without_clean_source_release": sum(correct(rows, qid) for qid in nonterminal if qid not in source_released),
            "warning": "Descriptive milestone evidence only. Incomplete cohort and provider-success selection; different run dates and incident exposure. Not a full-300 accuracy, superiority claim, confidence interval, or acceptance result."
        },
        "scoring": "Unchanged typed scorer; terminal errors incorrect; no retry replacement. Only the selected nonterminal comparison is summarized, not a complete-cohort score.",
        "reference_sha256": sha(ROOT / "final_dataset/questions_answers_sql_fhir.csv"),
        "families_sha256": sha(ROOT / "manifests/question_families.csv"),
        "scorer_sha256": sha(ROOT / "utils/answer_scorer.py"),
        "control_path": str(CONTROL), "control_sha256": sha(CONTROL),
        "milestone_analysis_source_sha256": sha(__file__),
        "raw_archive_path": str(archive), "raw_archive_sha256": sha(archive),
        "raw_archive_bytes": archive.stat().st_size, "raw_files": files,
        "benchmark_pending": True, "full_benchmark_started": False,
        "main_promoted": False, "further_model_or_health_probes_started": False,
        "required_next_validation": "After service recovery and explicit resumption, run all fixed300 first attempts in a fresh directory with a contemporaneous single-agent control, then select/promote and start/verify the real full2931 benchmark."
    }
    (RECORD / "milestone.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("recorded_first_attempts", "terminal_errors", "recorded_nonterminal_attempts", "not_saved_as_completed", "source_files_verified_unchanged", "source_valid_releases", "host_boundary_serialization_errors")}))
    print(json.dumps(report["success_selected_comparison"]))


if __name__ == "__main__":
    main()
