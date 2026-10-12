"""Demo backend (P3-23): patients, questions, one evaluated-pipeline run, and its verification status.

The demo runs the same code path as the experiment: run_agent_with_input()
with the same strategy, model, base URL, per-question budget, and tool
cache. It checks the current code and settings against the evaluated run
identity of the matching ordinary arm. Example questions come from the
non-held-out splits by default. Reference answers are never loaded or sent
to the page.
"""
import hashlib
import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import run_agent  # noqa: E402
from utils import curate_input_dataset, run_agent_with_input  # noqa: E402

SYSTEMS = {"multi_agent": "Multi-agent system", "single_agent": "Single agent", "direct_llm": "Direct LLM"}
DEFAULT_SPLITS = ("pilot", "dev")
CURRENT_TIME_RE = re.compile(r"current time is (\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?)")
TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?$")


def load_patients(input_file, split_manifest, splits=DEFAULT_SPLITS):
    """Patients of the chosen splits with their example questions (no reference answers)."""
    with open(split_manifest, encoding="utf-8") as stream:
        manifest = json.load(stream)
    frame = run_agent.read_input_data(input_file)[["question_id", "question", "assumption", "patient_fhir_id"]]
    patients = []
    for split in splits:
        entry = manifest["splits"][split]
        chosen = frame[frame["question_id"].isin(entry["question_ids"])].sort_values("question_id")
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
    assumption = f"Assume the current time is {current_time.strip()}." if current_time else None
    row = pd.DataFrame([{"question": question, "assumption": assumption, "patient_fhir_id": patient_fhir_id}])
    return curate_input_dataset(row, True)[0]


def evaluated_run(system, settings, output_dir):
    """Whether this demo uses the evaluated ordinary arm's code, model, budget, and cache.

    `settings`: model, base_url, max_total_tokens, max_tool_calls.
    """
    path = os.path.join(output_dir, f"ordinary_{system}.json.run.json")
    if not os.path.exists(path):
        return {"found": False, "path": path, "matches": {}}
    with open(path, encoding="utf-8") as stream:
        saved = json.load(stream)
    current = {"code_hash": run_agent.code_fingerprint(), "model": settings["model"],
               "base_url_hash": hashlib.sha256((settings.get("base_url") or "").encode()).hexdigest(),
               "max_total_tokens": settings.get("max_total_tokens"), "max_tool_calls": settings.get("max_tool_calls"),
               "enable_cache": True, "agent_strategy": system}
    matches = {key: saved.get(key) == value for key, value in current.items()}
    return {"found": True, "path": path, "matches": matches, "all_match": all(matches.values())}


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
    if revision is None:
        stopped = (output.get("budget") or {}).get("stopped")
        return {"state": "abstained", "label": "No answer",
                "detail": f"Stopped by the {stopped}." if stopped else "The system produced no draft to check.",
                "issues": []}
    issues = remaining_issues(output)
    repaired = bool(revision.get("revised"))
    repair = (f" after one repair (from {revision['repair_role']})" if repaired and revision.get("repair_role")
              else " after one revision" if repaired else "")
    if released and not issues:
        return {"state": "verified", "label": "Verified",
                "detail": f"Every cited field matched its source record{repair}.", "issues": []}
    if issues:
        return {"state": "unverified", "label": "Not verified",
                "detail": f"The answer was withheld: {len(issues)} issue(s) remained{repair}.", "issues": issues}
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
    for citation in output.get("evidence") or []:
        if not isinstance(citation, dict):
            continue
        record, field = citation.get("resource_id"), citation.get("field_path")
        rows.append({"resource": record, "field": field, "value": _display(citation.get("value")),
                     "check": flagged.get((record, field)) or flagged.get((record, None)) or "ok"})
    return rows

