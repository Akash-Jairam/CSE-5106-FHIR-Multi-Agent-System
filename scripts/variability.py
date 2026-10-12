"""Run-variability statistics (P3-20): how much one system's results change between repeats.

Each repeat is the same system answering the same fixed subset with the same
model, budget, and tool cache, so differences come from the model's sampling
and any nondeterminism in the pipeline. Answers are scored with the typed
scorer; an errored attempt counts as wrong, as in evaluation_metrics.py.
Only aggregates are returned, never answers.
"""
import json
import math
import statistics

from utils.answer_scorer import score_answer


def _sd(values):
    return statistics.stdev(values) if len(values) > 1 else None


def _spread(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "sd": None, "min": None, "max": None}
    return {"mean": statistics.fmean(values), "sd": _sd(values), "min": min(values), "max": max(values)}


def _errored(row):
    error = row.get("error")
    return error is not None and str(error).strip() not in ("", "nan")


def score_row(row, reference):
    """1 if a completed row's answer is correct under the typed scorer, else 0."""
    if _errored(row):
        return 0
    return score_answer(row.get("agent_answer"), reference["true_answer"], reference["question"],
                        reference.get("template", ""))["typed_correct"]


def _answer_key(row):
    answer = None if _errored(row) else row.get("agent_answer")
    if isinstance(answer, float) and math.isnan(answer):
        answer = None
    return json.dumps(answer, sort_keys=True, default=str)


def summarize(repeats, references):
    """Variance of one system over repeated runs.

    `repeats`: one {question_id: result row} per complete repeat.
    `references`: {question_id: {"question", "true_answer", "template"}} for the subset.
    """
    question_ids = sorted(references)
    if any(set(rows) != set(question_ids) for rows in repeats):
        raise ValueError("Every repeat must answer exactly the subset's questions")
    correct = [[score_row(rows[qid], references[qid]) for qid in question_ids] for rows in repeats]
    per_repeat = []
    for rows, scores in zip(repeats, correct):
        tokens = [(rows[q].get("tokens") or {}).get("total") for q in question_ids]
        latency = [rows[q].get("latency") for q in question_ids if rows[q].get("latency") is not None]
        per_repeat.append({
            "accuracy": statistics.fmean(scores),
            "abstention_rate": statistics.fmean(rows[q].get("agent_answer") is None for q in question_ids),
            "mean_tokens": statistics.fmean(tokens) if None not in tokens else None,
            "median_latency": statistics.median(latency) if latency else None,
        })

    n = len(repeats)
    by_question = list(zip(*correct)) if n else []
    pairs = n * (n - 1) / 2
    agreement = [sum(a == b for i, a in enumerate(q) for b in q[i + 1:]) / pairs for q in by_question] if n > 1 else []
    same_answer = [len({_answer_key(rows[qid]) for rows in repeats}) == 1 for qid in question_ids] if n else []
    return {
        "repeats": n,
        "questions": len(question_ids),
        "accuracy": _spread([r["accuracy"] for r in per_repeat]),
        "abstention_rate": _spread([r["abstention_rate"] for r in per_repeat]),
        "mean_tokens": _spread([r["mean_tokens"] for r in per_repeat]),
        "median_latency": _spread([r["median_latency"] for r in per_repeat]),
        "always_correct": sum(all(q) for q in by_question),
        "never_correct": sum(not any(q) for q in by_question),
        "unstable": sum(0 < sum(q) < n for q in by_question),
        # Mean over questions of the share of repeat pairs that agree on correctness
        "correctness_agreement": statistics.fmean(agreement) if agreement else None,
        # Share of questions answered identically in every repeat
        "identical_answers": statistics.fmean(same_answer) if same_answer else None,
        "per_repeat": per_repeat,
    }
