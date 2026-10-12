"""Demo backend (P3-23): patients, questions, one evaluated-pipeline run, and its verification status.

The demo runs the same code path as the experiment: run_agent_with_input()
with the same strategy, model, base URL, per-question budget, and tool
cache. It checks the current code and settings against the evaluated run
full identity and complete result batch of the matching ordinary arm.
Identity validation reads reference labels offline through the runner;
they are never sent to the page or model. Example questions come from the
non-held-out splits by default, and the picker parses only context columns.
"""
import hashlib
from datetime import datetime
import json
import math
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import run_agent  # noqa: E402
from utils import curate_input_dataset, parse_question_context, run_agent_with_input  # noqa: E402

SYSTEMS = {"multi_agent": "Multi-agent system", "single_agent": "Single agent", "direct_llm": "Direct LLM"}
DEFAULT_SPLITS = ("pilot", "dev")
CURRENT_TIME_RE = re.compile(r"current time is (\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?)")
TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?$")
CONTEXT_COLUMNS = ["question_id", "question", "assumption", "patient_fhir_id"]


def _file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _input_context(input_file, manifest):
    """Read only context columns, after checking the frozen source binding."""
    if manifest.get("source_sha256") != _file_hash(input_file):
        raise ValueError("The split was built from a different input dataset")
    return pd.read_csv(input_file, usecols=CONTEXT_COLUMNS)


def load_patients(input_file, split_manifest, splits=DEFAULT_SPLITS):
    """Patients of the chosen splits with their example questions (no reference answers)."""
    with open(split_manifest, encoding="utf-8") as stream:
        manifest = json.load(stream)
    frame = _input_context(input_file, manifest)
    patients = []
    for split in splits:
        entry = manifest["splits"][split]
        chosen = run_agent.select_split(frame, manifest, split).sort_values("question_id")
        for patient in entry["patients"]:
            rows = chosen[chosen["patient_fhir_id"] == patient["patient_fhir_id"]]
            if rows.empty:
                continue
            patients.append({**patient, "split": split, "questions": [
                {"question_id": r.question_id, "question": r.question,
                 "current_time": current_time_of(r.assumption)} for r in rows.itertuples()]})
    return patients


def current_time_of(assumption):
    match = CURRENT_TIME_RE.search(assumption) if isinstance(assumption, str) else None
    if not match:
        return None
    return match.group(1) if len(match.group(1)) == 19 else match.group(1) + ":00"


def build_input(patient_fhir_id, question, current_time=None):
    """The exact context string the runner gives agents for a dataset row."""
    question = (question or "").strip()
    if not question:
        raise ValueError("Enter a question")
    if current_time and not TIME_RE.match(current_time.strip()):
        raise ValueError("Current time must look like 2150-12-31 23:59:00")
    if current_time:
        try:
            parsed_time = datetime.fromisoformat(current_time.strip())
        except ValueError as exc:
            raise ValueError("Current time must be a valid calendar date and time") from exc
    assumption = f"Assume the current time is {current_time.strip()}." if current_time else None
    row = pd.DataFrame([{"question": question, "assumption": assumption, "patient_fhir_id": patient_fhir_id}])
    input_str = curate_input_dataset(row, True)[0]
    expected = (question, patient_fhir_id, parsed_time.strftime("%Y-%m-%d %H:%M:%S") if current_time else None)
    if parse_question_context(input_str) != expected:
        raise ValueError("Question context conflicts with the selected patient or time; "
                         "remove embedded patient IDs, current-time statements, or Context sections")
    return input_str


def evaluated_run(system, settings, output_dir, input_file=None, split_manifest=None):
    """Match the runner's full identity and a complete ordinary result batch.

    `settings`: model, base_url, max_total_tokens, max_tool_calls.
    Reference labels are read only for the offline input-identity hash.
    """
    if system not in SYSTEMS:
        raise ValueError(f"Unknown system: {system}")
    input_file = input_file or os.path.join(ROOT, "final_dataset", "questions_answers_sql_fhir.csv")
    split_manifest = split_manifest or os.path.join(ROOT, "manifests", "split.json")
    result_path = os.path.join(output_dir, f"ordinary_{system}.json")
    path = result_path + ".run.json"
    status = {"found": os.path.exists(path), "path": path, "matches": {}, "valid": False, "all_match": False}
    if not os.path.exists(path):
        return status
    try:
        with open(path, encoding="utf-8") as stream:
            saved = json.load(stream)
        if not isinstance(saved, dict):
            raise ValueError("Run metadata must be an object")
    except (OSError, ValueError) as exc:
        return {**status, "reason": f"Invalid evaluated run metadata: {exc}"}
    try:
        with open(split_manifest, encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("source_sha256") != _file_hash(input_file):
            raise ValueError("The split was built from a different input dataset")
        argv = ["--agent_strategy", system, "--model", settings["model"], "--input", input_file,
                "--split_manifest", split_manifest, "--split", "held_out", "--enable_cache", "true"]
        for key in ("base_url", "max_total_tokens", "max_tool_calls"):
            if settings.get(key) is not None:
                argv.extend([f"--{key}", str(settings[key])])
        run_settings = run_agent.build_parser().parse_args(argv)
        frame = run_agent.load_inputs(input_file, split_manifest=split_manifest, split="held_out")
        current = run_agent.run_identity(run_settings, frame)
        matches = {key: key in saved and saved[key] == value for key, value in current.items()}
        if set(saved) != set(current):
            matches["identity_fields"] = False
        status["matches"] = matches
        expected = frame.set_index("question_id")[CONTEXT_COLUMNS[1:]].to_dict("index")
        if not expected:
            raise ValueError("No held-out questions are planned")
        with open(result_path, encoding="utf-8") as stream:
            rows = json.load(stream)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Saved results must be a list of result objects")
        ids = [row.get("question_id") for row in rows]
        if any(not isinstance(qid, str) for qid in ids) or len(set(ids)) != len(ids) or set(ids) != set(expected):
            raise ValueError("Saved results must cover every held-out question exactly once")
        for row in rows:
            context = expected[row["question_id"]]
            if row.get("patient_fhir_id") != context["patient_fhir_id"] or row.get("question") != context["question"]:
                raise ValueError("Saved results do not match their source question and patient")
            assumption = context["assumption"] if pd.notna(context["assumption"]) else None
            if row.get("assumption") != assumption:
                raise ValueError("Saved results do not match their source time context")
            latency = row.get("latency")
            if latency is not None and (isinstance(latency, bool) or not isinstance(latency, (int, float))
                                        or not math.isfinite(latency) or latency < 0):
                raise ValueError("Saved latency must be a finite non-negative number or null")
            if latency is None and not row.get("error"):
                raise ValueError("Some held-out questions have not been attempted")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {**status, "reason": f"No complete evaluated result batch: {exc}"}
    return {**status, "valid": True, "all_match": saved == current}


def ask(system, input_str, settings):
    """One run of the evaluated pipeline; returns the runner's saved result columns."""
    if system not in SYSTEMS:
        raise ValueError(f"Unknown system: {system}")
    budget = {"max_total_tokens": settings.get("max_total_tokens"), "max_tool_calls": settings.get("max_tool_calls")}
    return run_agent_with_input((input_str, system, settings["model"], False, settings.get("base_url"), True, budget, None))


def describe_issue(issue):
    """Issues are dicts from the multi-agent verifier and strings from the single agent's field check."""
    if isinstance(issue, dict):
        return {"type": issue.get("type"), "record": issue.get("record"), "field": issue.get("field"),
                "detail": issue.get("action") or ""}
    kind, _, detail = str(issue).partition(":")
    return {"type": kind.strip() if detail else "issue", "record": None, "field": None,
            "detail": detail.strip() or str(issue)}


def remaining_issues(output):
    """Issues from the last check of the released or withheld candidate."""
    revision = output.get("revision") if isinstance(output.get("revision"), dict) else {}
    final_issues = revision.get("final_check_issues")
    return [describe_issue(i) for i in (final_issues if final_issues is not None else revision.get("check_issues") or [])]


def _check_recorded(output):
    revision = output.get("revision")
    if not isinstance(revision, dict) or output.get("error"):
        return False
    return isinstance(revision.get("final_check_issues"), list) or (
        not revision.get("revised") and isinstance(revision.get("check_issues"), list))


def accepted_empty_result(system, output):
    """A null final independently accepted by the multi-agent verifier."""
    revision = output.get("revision")
    final = revision.get("final") if isinstance(revision, dict) else None
    return (system == "multi_agent" and output.get("agent_answer") is None
            and isinstance(final, dict) and "answer" in final and final["answer"] is None
            and revision.get("final_check_issues") == [] and not output.get("error")
            and not revision.get("revision_error") and not (output.get("budget") or {}).get("stopped"))


def verification_status(system, output):
    """State, label, explanation, and remaining issues of one run, for display."""
    revision = output.get("revision") if isinstance(output.get("revision"), dict) else None
    released = output.get("agent_answer") is not None
    if output.get("error"):
        return {"state": "error", "label": "Error", "detail": str(output["error"]), "issues": []}
    if system == "direct_llm":
        return {"state": "unchecked", "label": "Not checked",
                "detail": "The direct baseline answers from the starting packet in one call; nothing checks it.",
                "issues": []}
    if not _check_recorded(output):
        issues = remaining_issues(output)
        if issues:
            return {"state": "unverified", "label": "Not verified",
                    "detail": "No completed final check was recorded; earlier issues remain.", "issues": issues}
        if released:
            return {"state": "unchecked", "label": "Not checked",
                    "detail": "No completed check was recorded for the released answer.", "issues": remaining_issues(output)}
        stopped = (output.get("budget") or {}).get("stopped")
        return {"state": "abstained", "label": "No answer",
                "detail": f"Stopped by the {stopped}." if stopped else "The system produced no draft to check.",
                "issues": []}
    issues = remaining_issues(output)
    repaired = bool(revision.get("revised"))
    repair = (f" after one repair (from {revision['repair_role']})" if repaired and revision.get("repair_role")
              else " after one revision" if repaired else "")
    if accepted_empty_result(system, output):
        return {"state": "verified", "label": "Verified empty result", "empty_result": True,
                "detail": f"The independent check confirmed that no matching value was found{repair}.", "issues": []}
    if released and not issues:
        return {"state": "verified", "label": "Verified",
                "detail": f"Every cited field matched its source record{repair}.", "issues": []}
    if issues:
        return {"state": "unverified", "label": "Not verified",
                "detail": f"{len(issues)} issue(s) remained{repair}; the answer was "
                          + ("released." if released else "withheld."), "issues": issues}
    return {"state": "abstained", "label": "No answer", "detail": f"The system abstained{repair}.", "issues": []}


def _display(value):
    return value if isinstance(value, str) else json.dumps(value, default=str)


def cited_fields(output):
    """Citation rows, each marked with any remaining issue on its record and field (or on the whole record)."""
    flagged = {}
    for issue in remaining_issues(output):
        if issue["record"]:
            # An issue without a field applies to every citation of its record
            flagged.setdefault((issue["record"], issue["field"]), issue["type"])
    rows = []
    default = "ok" if _check_recorded(output) and not remaining_issues(output) else "not confirmed"
    for citation in output.get("evidence") or []:
        if not isinstance(citation, dict):
            continue
        record, field = citation.get("resource_id"), citation.get("field_path")
        rows.append({"resource": record, "field": field, "value": _display(citation.get("value")),
                     "check": flagged.get((record, field)) or flagged.get((record, None))
                              or (default if record and field else "incomplete citation")})
    return rows

