"""Run-variability statistics (P3-20): how much one system's results change between repeats.

Each repeat is the same system answering the same fixed subset with the same
model, budget, and tool cache. Outcomes and primary usage describe the first
attempt; a later retry cannot replace its failure with a success. Retry counts
and cumulative spend describe all attempts separately. Answers are scored
with the typed scorer; an errored attempt counts as wrong.
Only aggregates are returned, never answers.
"""
import json
import math
import statistics

from utils import RESULT_COLUMNS
from utils.answer_scorer import is_abstention, score_answer


def _sd(values):
    return statistics.stdev(values) if len(values) > 1 else None


def _spread(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "sd": None, "min": None, "max": None, "n": 0}
    return {"mean": statistics.fmean(values), "sd": _sd(values), "min": min(values), "max": max(values),
            "n": len(values)}


def is_failed(row):
    error = row.get("error")
    return error is not None and str(error).strip() not in ("", "nan")


def attempts(row):
    """Recorded attempts in execution order, including the current saved result."""
    history = row.get("attempt_history")
    if history is None:
        history = []
    if not isinstance(history, list) or any(not isinstance(attempt, dict) for attempt in history):
        raise ValueError("Attempt history must be a list of recorded result objects")
    return [*history, row]


def first_attempt(row):
    """First recorded outcome, retaining the question metadata on the saved row."""
    first = attempts(row)[0]
    return {**row, **{key: first.get(key) for key in RESULT_COLUMNS}}


def score_row(row, reference):
    """1 if the first attempt is correct under the typed scorer, else 0."""
    row = first_attempt(row)
    if is_failed(row):
        return 0
    return score_answer(row.get("agent_answer"), reference["true_answer"], reference["question"],
                        reference.get("template", ""))["typed_correct"]


def _answer_key(row):
    answer = None if is_failed(row) else row.get("agent_answer")
    if isinstance(answer, float) and math.isnan(answer):
        answer = None
    return json.dumps(answer, sort_keys=True, default=str)


def _measure(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Usage and latency must be finite non-negative numbers or null")
    return value


def _attempt_measure(attempt, key):
    if key == "tokens":
        tokens = attempt.get("tokens")
        if tokens is not None and not isinstance(tokens, dict):
            raise ValueError("Token usage must be an object or null")
        return _measure((tokens or {}).get("total"))
    return _measure(attempt.get(key))


def cumulative_measure(row, key):
    """Retry-inclusive tokens or latency, unavailable if any attempt is unmeasured."""
    if key not in ("tokens", "latency"):
        raise ValueError("Cumulative measure must be tokens or latency")
    values = [_attempt_measure(attempt, key) for attempt in attempts(row)]
    return _measure(sum(values)) if None not in values else None


def summarize(repeats, references):
    """Variance of one system over repeated runs.

    `repeats`: one {question_id: result row} per complete repeat.
    `references`: {question_id: {"question", "true_answer", "template"}} for the subset.
    """
    question_ids = sorted(references)
    if repeats and not question_ids:
        raise ValueError("A repeat must contain at least one subset question")
    if any(set(rows) != set(question_ids) for rows in repeats):
        raise ValueError("Every repeat must answer exactly the subset's questions")
    observations = [{qid: first_attempt(rows[qid]) for qid in question_ids} for rows in repeats]
    correct = [[score_row(rows[qid], references[qid]) for qid in question_ids] for rows in repeats]
    per_repeat = []
    for original, rows, scores in zip(repeats, observations, correct):
        tokens = [_attempt_measure(rows[q], "tokens") for q in question_ids]
        latency = [_attempt_measure(rows[q], "latency") for q in question_ids]
        all_tokens = [cumulative_measure(original[q], "tokens") for q in question_ids]
        all_latency = [cumulative_measure(original[q], "latency") for q in question_ids]
        per_repeat.append({
            "accuracy": statistics.fmean(scores),
            "abstention_rate": statistics.fmean(not is_failed(rows[q]) and is_abstention(rows[q].get("agent_answer"))
                                                 for q in question_ids),
            "error_rate": statistics.fmean(is_failed(rows[q]) for q in question_ids),
            "mean_tokens": statistics.fmean(tokens) if None not in tokens else None,
            "median_latency": statistics.median(latency) if None not in latency else None,
            "retry_count": sum(len(attempts(original[q])) - 1 for q in question_ids),
            "failed_attempts": sum(is_failed(attempt) for q in question_ids for attempt in attempts(original[q])),
            "retry_inclusive_tokens_total": sum(all_tokens) if None not in all_tokens else None,
            "retry_inclusive_latency_total": sum(all_latency) if None not in all_latency else None,
        })

    n = len(repeats)
    by_question = list(zip(*correct)) if n else []
    pairs = n * (n - 1) / 2
    agreement = [sum(a == b for i, a in enumerate(q) for b in q[i + 1:]) / pairs for q in by_question] if n > 1 else []
    same_answer = [not any(is_failed(rows[qid]) for rows in observations)
                   and len({_answer_key(rows[qid]) for rows in observations}) == 1 for qid in question_ids] if n else []
    return {
        "repeats": n,
        "questions": len(question_ids),
        "accuracy": _spread([r["accuracy"] for r in per_repeat]),
        "abstention_rate": _spread([r["abstention_rate"] for r in per_repeat]),
        "error_rate": _spread([r["error_rate"] for r in per_repeat]),
        "mean_tokens": _spread([r["mean_tokens"] for r in per_repeat]),
        "median_latency": _spread([r["median_latency"] for r in per_repeat]),
        "retry_count": _spread([r["retry_count"] for r in per_repeat]),
        "retry_inclusive_tokens_total": _spread([r["retry_inclusive_tokens_total"] for r in per_repeat]),
        "retry_inclusive_latency_total": _spread([r["retry_inclusive_latency_total"] for r in per_repeat]),
        "always_correct": sum(all(q) for q in by_question),
        "never_correct": sum(not any(q) for q in by_question),
        "unstable": sum(0 < sum(q) < n for q in by_question),
        # Mean over questions of the share of repeat pairs that agree on correctness
        "correctness_agreement": statistics.fmean(agreement) if agreement else None,
        # Share of questions answered identically in every repeat
        "identical_answers": statistics.fmean(same_answer) if same_answer else None,
        "per_repeat": per_repeat,
    }
