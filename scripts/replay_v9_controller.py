"""Offline replay of the V9 dual-solver controller, and its mechanism table (docs/diagnosis_v8/design_v9.txt).

predict    Replays controller steps 3-4 (agreement, null rule) exactly over existing
           arm files: the single agent's released answer is the primary and the V8
           team's solver candidate is the recompute. Recompute availability is
           estimated from the V8 solver's own usage against the V9 residual allowance
           (agent.dual_solver_agent.budget_plan). Both-non-null disagreements are
           counted as adjudication cases and, for the prediction, default to the
           primary; the oracle and always-recompute bounds are reported beside it.
mechanism  For a finished V9 team arm: release-policy counts, draft (primary) ->
           released transitions per policy, adjudication outcomes, recompute usage.

Only counts, scores and token statistics are printed or written; answers never are.
Never called by agents.

  python scripts/replay_v9_controller.py predict \
      --single <dir>/ordinary_single_agent.json --team <dir>/ordinary_multi_agent.json --output replay.json
  python scripts/replay_v9_controller.py mechanism --team <v9 dir>/ordinary_multi_agent.json --output mech.json
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.dual_solver_agent import RELEASE_POLICIES, budget_plan, cited_ids, decide, source_valid
from scripts.full_benchmark import iter_result_rows
from scripts.variability import first_attempt, is_failed
from utils.answer_scorer import score_answer

DEFAULT_INPUT = ROOT / "final_dataset/questions_answers_sql_fhir.csv"
DEFAULT_FAMILIES = ROOT / "manifests/question_families.csv"


def load_references(input_path=DEFAULT_INPUT, families_path=DEFAULT_FAMILIES):
    with open(input_path, newline="", encoding="utf-8") as stream:
        source = {r["question_id"]: r for r in csv.DictReader(stream)}
    with open(families_path, newline="", encoding="utf-8") as stream:
        families = {r["question_id"]: r for r in csv.DictReader(stream)}
    return {qid: {"question": row["question"], "true_answer": row["true_answer"],
                  "template": families.get(qid, {}).get("template", "")} for qid, row in source.items()}


def load_rows(path):
    rows = {}
    for row in iter_result_rows(Path(path)):
        if row.get("question_id") in rows:
            raise ValueError(f"{path}: duplicate question {row.get('question_id')}")
        rows[row["question_id"]] = first_attempt(row)
    return rows


def correct(answer, reference):
    return score_answer(answer, reference["true_answer"], reference["question"], reference["template"])["typed_correct"]


def _released(row):
    return None if is_failed(row) else row.get("agent_answer")


def _phase(revision, name):
    return next((p for p in revision.get("phases") or [] if p.get("phase") == name), None)


def _quantiles(values):
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return {"n": 0, "median": None, "p90": None, "mean": None}
    return {"n": len(values), "median": statistics.median(values),
            "p90": values[min(len(values) - 1, int(0.9 * len(values)))], "mean": statistics.fmean(values)}


def recompute_candidate(team_row, field="primary"):
    """The V8 solver's candidate in V9 draft-mode terms: answer and source validity, or None."""
    revision = team_row.get("revision") or {}
    if field == "draft":
        candidate = revision.get("draft")
        issues = (revision.get("primary_revision") or {}).get("check_issues")
    else:
        candidate = revision.get("primary")
        issues = revision.get("final_check_issues")
    if not isinstance(candidate, dict):
        return None
    return {"answer": candidate.get("answer"),
            "source_valid": source_valid(issues, revision.get("unresolved_scope"), cited_ids(candidate))}


def replay(single_rows, team_rows, references, max_total_tokens, max_tool_calls, field="primary"):
    """Predicted V9 score and caseloads over paired single / V8-team rows."""
    if set(single_rows) != set(team_rows):
        raise ValueError("single and team arms must cover the same questions")
    counts, availability = Counter(), Counter()
    adjudication = Counter()
    scores = {"single": 0, "recompute_candidate": 0, "predicted_default_primary": 0,
              "bound_adjudication_oracle": 0, "bound_adjudication_recompute": 0, "oracle_pick": 0}
    pairs = Counter()
    single_tokens, solver_tokens = [], []
    for qid in sorted(single_rows):
        single, team, reference = single_rows[qid], team_rows[qid], references[qid]
        primary = _released(single)
        tokens = (single.get("tokens") or {}).get("total") or 0
        tools = (single.get("budget") or {}).get("tool_calls_run") or 0
        single_tokens.append(tokens)
        plan = budget_plan(max_total_tokens, max_tool_calls, tokens, tools)
        candidate = recompute_candidate(team, field)
        solver = _phase(team.get("revision") or {}, "solver") or {}
        solver_tokens.append(solver.get("tokens"))
        if plan["skip"]:
            availability["skipped_residual"] += 1
            candidate_available = None
        elif (solver.get("tokens") is None or solver["tokens"] > plan["recompute_tokens"]
              or (solver.get("tool_calls") or 0) > plan["recompute_tools"]):
            # the recompute would have been truncated; approximated as unavailable
            availability["truncated_estimate"] += 1
            candidate_available = None
        else:
            availability["available"] += 1
            candidate_available = candidate
        policy = decide(primary, candidate_available, reference["question"])
        counts[policy] += 1
        p_right = correct(primary, reference)
        r_right = correct((candidate or {}).get("answer"), reference) if candidate else 0
        scores["single"] += p_right
        scores["recompute_candidate"] += r_right
        scores["oracle_pick"] += max(p_right, r_right)
        pairs[("P" if p_right else "-") + ("R" if r_right else "-")] += 1
        if policy == "null_rule_recompute":
            released = r_right
            counts[f"null_rule_recompute_{'right' if r_right else 'wrong'}"] += 1
        else:
            released = p_right
        if policy == "adjudicate":
            adjudication[("primary_right" if p_right else "primary_wrong") + "/" +
                         ("recompute_right" if r_right else "recompute_wrong")] += 1
            scores["bound_adjudication_oracle"] += max(p_right, r_right)
            scores["bound_adjudication_recompute"] += r_right
        else:
            scores["bound_adjudication_oracle"] += released
            scores["bound_adjudication_recompute"] += released
        scores["predicted_default_primary"] += released
    return {"questions": len(single_rows), "recompute_field": field,
            "ceilings": {"max_total_tokens": max_total_tokens, "max_tool_calls": max_tool_calls},
            "scores": scores, "policies": dict(sorted(counts.items())),
            "correctness_pairs (primary, recompute)": dict(sorted(pairs.items())),
            "adjudication_cases": dict(sorted(adjudication.items())),
            "recompute_availability": dict(sorted(availability.items())),
            "tokens": {"single": _quantiles(single_tokens), "v8_solver_phase": _quantiles(solver_tokens)}}


def mechanism(team_rows, references):
    """Release policies and draft -> released transitions of a finished V9 arm."""
    policies, transitions, adjudication = Counter(), Counter(), Counter()
    phase_tokens = {"primary": [], "recompute": [], "adjudicate": []}
    total_tokens, protocols, primary_errors_rescued = [], Counter(), 0
    for qid in sorted(team_rows):
        row, reference = team_rows[qid], references[qid]
        revision = row.get("revision") or {}
        protocols[revision.get("protocol")] += 1
        policy = revision.get("release_policy") if not is_failed(row) else "terminal_error"
        policies[policy] += 1
        total_tokens.append((row.get("tokens") or {}).get("total"))
        draft = (revision.get("draft") or {}).get("answer")
        before, after = correct(draft, reference), correct(_released(row), reference)
        key = {(1, 1): "right_to_right", (1, 0): "right_to_wrong",
               (0, 1): "wrong_to_right", (0, 0): "wrong_to_wrong"}[(before, after)]
        transitions[f"{policy}:{key}"] += 1
        record = revision.get("adjudication") or {}
        if record:
            adjudication["cases"] += 1
            adjudication[f"check_{record.get('check_status') or 'not_run'}"] += 1
            adjudication["malformed_or_failed"] += bool(record.get("error"))
        primary_errors_rescued += bool(revision.get("primary_error")) and policy in (
            "null_rule_recompute", "adjudicated_recompute")
        for phase in revision.get("phases") or []:
            if phase.get("phase") in phase_tokens:
                phase_tokens[phase["phase"]].append(phase.get("tokens"))
    unknown = sorted(p for p in policies if p not in RELEASE_POLICIES and p != "terminal_error")
    return {"questions": len(team_rows), "protocols": dict(protocols), "policies": dict(sorted(policies.items())),
            "unknown_policies": unknown, "transitions": dict(sorted(transitions.items())),
            "replacements": {p: {k.split(":", 1)[1]: v for k, v in transitions.items() if k.startswith(p + ":")}
                             for p in ("null_rule_recompute", "adjudicated_recompute")},
            "adjudication": dict(sorted(adjudication.items())),
            "primary_errors_rescued": primary_errors_rescued,
            "tokens": {"total": _quantiles(total_tokens), **{k: _quantiles(v) for k, v in phase_tokens.items()}}}


def _ceilings(path, args):
    identity_path = Path(str(path) + ".run.json")
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    return (args.max_total_tokens if args.max_total_tokens is not None else identity.get("max_total_tokens"),
            args.max_tool_calls if args.max_tool_calls is not None else identity.get("max_tool_calls"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["predict", "mechanism"])
    parser.add_argument("--single", help="predict: single-agent arm file")
    parser.add_argument("--team", required=True, help="predict: V8 team arm file; mechanism: V9 team arm file")
    parser.add_argument("--recompute_field", choices=["primary", "draft"], default="primary",
                        help="predict: which V8 solver candidate stands in for the recompute")
    parser.add_argument("--max_total_tokens", type=int, default=None, help="Default: the team arm's run identity")
    parser.add_argument("--max_tool_calls", type=int, default=None, help="Default: the team arm's run identity")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--families", default=str(DEFAULT_FAMILIES))
    parser.add_argument("--output", default=None, help="Write the JSON report here")
    args = parser.parse_args(argv)
    references = load_references(args.input, args.families)
    team = load_rows(args.team)
    if args.command == "predict":
        if not args.single:
            parser.error("predict needs --single")
        report = replay(load_rows(args.single), team, references, *_ceilings(args.team, args), args.recompute_field)
    else:
        report = mechanism(team, references)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
