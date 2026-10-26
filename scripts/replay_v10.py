"""Offline replay for the V10 design over the B2 rows, and the V10 mechanism table (design_v10.txt).

predict    Over a finished V9 pair (single arm + team arm, same model and ceilings):
           (a) two-draw statistics, with the team's primary (the single scaffold, a
               second draw) as draw 1 and the fresh single as draw 2: agreement and
               its precision, disagreements by which draw is right, draw-2 and
               draw-3 availability under the V10 residual scheme, and the vote's
               expected change over draw 1 if a third draw reproduces either earlier
               answer with equal probability;
           (b) per-template finisher reachability on the fresh single's misses
               (and on the team primary's), by the audited signatures: a bare
               member of a larger gold tie set (T051/T052), Observation-only
               evidence (T054), a null or single organism (T055), a count citing
               prescriptions (T001), a timestamp on the gold day (T064/T073); plus
               the right answers a replacement condition would reach (harm exposure);
           (c) the predicted V10 score under the design's assumptions.
mechanism  For a finished V10 team arm: votes, release policies, finisher outcomes and
           reasons per template, candidate -> released transitions of finisher
           replacements (gate item 3), draw-1 -> released transitions per policy,
           adjudication outcomes and per-phase tokens.

Templates come from the family manifest here (an offline analysis); the agent itself
routes by question text only. Only counts and token statistics are printed or
written; answers never are. Never called by agents.

  python scripts/replay_v10.py predict \\
      --single <b2>/ordinary_single_agent.json --team <b2>/ordinary_multi_agent.json --output replay_v10.json
  python scripts/replay_v10.py mechanism --team <v10>/ordinary_multi_agent.json --output mechanism.json
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.dual_solver_agent import answers_agree
from agent.finisher_team_agent import (FINISHER_RESERVE, MIN_DRAW, RELEASE_POLICIES, THIRD_DRAW_MIN, VOTES,
                                       third_draw_allowed)
from agent.finishers import FINISHER_OUTCOMES
from agent.template_router import classify_template
from scripts.replay_v9_controller import (DEFAULT_FAMILIES, DEFAULT_INPUT, _phase, _quantiles, _released,
                                          correct, load_references, load_rows)
from utils.answer_scorer import parse_reference

FINISHER_TEMPLATES = ("T001", "T051", "T052", "T054", "T055", "T064", "T073")
TIMESTAMP_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2})?.*)?\s*$")
# Design §3 assumptions (pre-registered; changed only by passing the flags explicitly)
REALISATION = 0.6
HARMFUL = 2
ADJUDICATION_GAIN = (0, 2)


def template_ids(families_path=DEFAULT_FAMILIES):
    with open(families_path, newline="", encoding="utf-8") as stream:
        return {r["question_id"]: r["template_id"] for r in csv.DictReader(stream)}


def _norm(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _single_string(answer):
    if isinstance(answer, str):
        return answer
    if isinstance(answer, list) and len(answer) == 1 and isinstance(answer[0], str):
        return answer[0]
    return None


def _cited_types(evidence, contributing_ids=()):
    ids = [c.get("resource_id") for c in evidence or [] if isinstance(c, dict)] + list(contributing_ids or [])
    return {rid.split("/", 1)[0] for rid in ids if isinstance(rid, str) and "/" in rid}


def signature(template, answer, cited_types, reference):
    """The audited wrong-answer signature a finisher's replacement condition can reach, or None."""
    gold = parse_reference(reference["true_answer"])
    if template in ("T051", "T052"):
        member = _single_string(answer)
        if ("MedicationRequest" in cited_types and member is not None and len(gold) > 1
                and _norm(member) in {_norm(g) for g in gold}):
            return "bare_tie_member"
    elif template == "T054":
        if "Observation" in cited_types and "Specimen" not in cited_types and answer is not None:
            return "observation_only"
    elif template == "T055":
        if "Observation" in cited_types and (answer is None or _single_string(answer) is not None):
            return "null_or_single_organism"
    elif template == "T001":
        if "MedicationRequest" in cited_types and isinstance(answer, (int, float)) and not isinstance(answer, bool):
            return "count_citing_prescriptions"
    elif template in ("T064", "T073"):
        expected = "Observation" if template == "T064" else "MedicationRequest"
        match = TIMESTAMP_RE.match(answer) if isinstance(answer, str) else None
        gold_day = str(gold[0])[:10] if gold else None
        if expected in cited_types and match and match.group(1) == gold_day:
            return "timestamp_on_gold_day"
    return None


def _reachability(rows, references, templates, field):
    """Per template: misses, misses with a reachable signature, right answers a condition would reach."""
    table = defaultdict(Counter)
    for qid, (answer, cited, right) in rows.items():
        template = templates.get(qid)
        if template not in FINISHER_TEMPLATES:
            continue
        sig = signature(template, answer, cited, references[qid])
        entry = table[template]
        entry["questions"] += 1
        entry["wrong"] += not right
        if sig and not right:
            entry[f"reachable:{sig}"] += 1
            entry["reachable"] += 1
        if sig and right:
            entry["exposed_right"] += 1
    total = Counter()
    for entry in table.values():
        total.update({k: v for k, v in entry.items() if k in ("questions", "wrong", "reachable", "exposed_right")})
    return {"arm": field, "by_template": {t: dict(sorted(c.items())) for t, c in sorted(table.items())},
            "total": dict(total)}


def predict(single_rows, team_rows, references, templates, max_total_tokens, max_tool_calls,
            realisation=REALISATION, harmful=HARMFUL, adjudication_gain=ADJUDICATION_GAIN):
    if set(single_rows) != set(team_rows):
        raise ValueError("single and team arms must cover the same questions")
    pairs, availability, router = Counter(), Counter(), Counter()
    scores = Counter()
    expected_vote_change = 0.0
    two_draw_tokens, single_tokens = [], []
    single_view, primary_view = {}, {}
    for qid in sorted(single_rows):
        single, team, reference = single_rows[qid], team_rows[qid], references[qid]
        revision = team.get("revision") or {}
        primary = revision.get("primary") or {}
        d1, d2 = primary.get("answer"), _released(single)
        c1, c2 = correct(d1, reference), correct(d2, reference)
        scores["fresh_single"] += c2
        scores["team_primary_draw1"] += c1
        scores["oracle_two_draws"] += max(c1, c2)
        single_view[qid] = (d2, _cited_types(single.get("evidence")), c2)
        primary_view[qid] = (d1, _cited_types(primary.get("evidence"), primary.get("contributing_ids")), c1)
        template = templates.get(qid)
        expected = template if template in FINISHER_TEMPLATES else None
        router["agree" if classify_template(reference["question"]) == expected else "disagree"] += 1

        t1 = (_phase(revision, "primary") or {}).get("tokens") or 0
        k1 = (_phase(revision, "primary") or {}).get("tool_calls") or 0
        t2 = (single.get("tokens") or {}).get("total") or 0
        k2 = (single.get("budget") or {}).get("tool_calls_run") or 0
        single_tokens.append(t2)
        two_draw_tokens.append(t1 + t2)
        draw2 = ((max_total_tokens is None or max_total_tokens - t1 - FINISHER_RESERVE[0] >= MIN_DRAW[0])
                 and (max_tool_calls is None or max_tool_calls - k1 - FINISHER_RESERVE[1] >= MIN_DRAW[1]))
        availability["draw2_available"] += draw2
        if not draw2:
            pairs["draw2_unavailable"] += 1
            continue
        if answers_agree(d1, d2, reference["question"]):
            pairs["agree_right" if c1 else "agree_wrong"] += 1
            continue
        key = {(1, 1): "disagree_both_right", (1, 0): "disagree_draw1_right", (0, 1): "disagree_draw2_right",
               (0, 0): "disagree_both_wrong"}[(c1, c2)]
        pairs[key] += 1
        if third_draw_allowed(max_total_tokens, max_tool_calls, t1 + t2, k1 + k2):
            availability["draw3_available"] += 1
            if c1 != c2:
                # a third draw reproduces either answer with equal probability: majority right half the time
                expected_vote_change += 0.5 - c1
        else:
            availability["draw3_unavailable"] += 1
    agreements = pairs["agree_right"] + pairs["agree_wrong"]
    reach_single = _reachability(single_view, references, templates, "fresh_single")
    reach_primary = _reachability(primary_view, references, templates, "team_primary_draw1")
    finisher_gain = realisation * reach_single["total"].get("reachable", 0) - harmful
    base = scores["fresh_single"]
    low = base + expected_vote_change + finisher_gain + adjudication_gain[0]
    high = base + expected_vote_change + finisher_gain + adjudication_gain[1]
    budget = max_total_tokens or 0
    return {"questions": len(single_rows),
            "ceilings": {"max_total_tokens": max_total_tokens, "max_tool_calls": max_tool_calls},
            "scores": dict(scores),
            "two_draws": {"pairs": dict(sorted(pairs.items())),
                          "agreement_precision": round(pairs["agree_right"] / agreements, 3) if agreements else None,
                          "availability": dict(sorted(availability.items())),
                          "expected_vote_change_over_draw1": round(expected_vote_change, 2)},
            "finisher_reachability": {"fresh_single": reach_single, "team_primary_draw1": reach_primary},
            "router_check": dict(router),
            "tokens": {"single": _quantiles(single_tokens), "two_draws": _quantiles(two_draw_tokens),
                       "three_single_fit_under_ceiling_minus_15k":
                           sum(3 * t <= budget - 15_000 for t in single_tokens) if budget else None},
            "assumptions": {"finisher_realisation": realisation, "harmful_replacements": harmful,
                            "adjudication_gain": list(adjudication_gain)},
            "prediction": {"fresh_single": base, "vote": round(expected_vote_change, 2),
                           "finishers": round(finisher_gain, 2), "adjudication": list(adjudication_gain),
                           "team_low": round(low, 1), "team_high": round(high, 1)}}


def mechanism(team_rows, references, templates):
    """Votes, policies, finisher outcomes and transitions of a finished V10 arm."""
    votes, policies, protocols = Counter(), Counter(), Counter()
    finisher_outcomes, finisher_reasons = defaultdict(Counter), defaultdict(Counter)
    transitions, replacements, adjudication = Counter(), defaultdict(Counter), Counter()
    phase_tokens, totals = defaultdict(list), []
    draw1_correct = released_correct = 0
    for qid in sorted(team_rows):
        row, reference = team_rows[qid], references[qid]
        revision = row.get("revision") or {}
        protocols[revision.get("protocol")] += 1
        failed = _released(row) is None and bool(row.get("error"))
        policy = "terminal_error" if failed else revision.get("release_policy")
        policies[policy] += 1
        votes[revision.get("vote")] += 1
        totals.append((row.get("tokens") or {}).get("total"))
        draft = (revision.get("draft") or {}).get("answer")
        before, after = correct(draft, reference), correct(_released(row), reference)
        draw1_correct += before
        released_correct += after
        key = {(1, 1): "right_to_right", (1, 0): "right_to_wrong",
               (0, 1): "wrong_to_right", (0, 0): "wrong_to_wrong"}[(before, after)]
        transitions[f"{policy}:{key}"] += 1
        finisher = revision.get("finisher") or {}
        template = finisher.get("template") or templates.get(qid) or "none"
        finisher_outcomes[template][finisher.get("finisher")] += 1
        if finisher.get("reason"):
            finisher_reasons[template][finisher["reason"]] += 1
        if policy == "finisher_replaced":
            candidate = (revision.get("candidate") or {}).get("answer")
            c_before = correct(candidate, reference)
            replacements[template][{(1, 1): "right_to_right", (1, 0): "right_to_wrong", (0, 1): "wrong_to_right",
                                    (0, 0): "wrong_to_wrong"}[(c_before, after)]] += 1
        record = revision.get("adjudication") or {}
        if record:
            adjudication["cases"] += 1
            adjudication[f"check_{record.get('check_status') or 'not_run'}"] += 1
            adjudication["malformed_or_failed"] += bool(record.get("error"))
        for phase in revision.get("phases") or []:
            phase_tokens[phase.get("phase")].append(phase.get("tokens"))
    replaced = Counter()
    for counts in replacements.values():
        replaced.update(counts)
    unknown = sorted(p for p in policies if p not in RELEASE_POLICIES and p != "terminal_error")
    return {"questions": len(team_rows), "protocols": dict(protocols), "votes": dict(sorted(votes.items(), key=str)),
            "unknown_votes": sorted(str(v) for v in votes if v not in VOTES),
            "policies": dict(sorted(policies.items(), key=str)), "unknown_policies": unknown,
            "draw1_correct": draw1_correct, "released_correct": released_correct,
            "transitions": dict(sorted(transitions.items())),
            "finisher_outcomes": {t: dict(c) for t, c in sorted(finisher_outcomes.items())},
            "unknown_finisher_outcomes": sorted({str(o) for c in finisher_outcomes.values() for o in c
                                                 if o not in FINISHER_OUTCOMES}),
            "finisher_reasons": {t: dict(c) for t, c in sorted(finisher_reasons.items())},
            "finisher_replacements": {"total": dict(replaced),
                                      "by_template": {t: dict(c) for t, c in sorted(replacements.items())}},
            "gate_finisher_harm_ok": replaced["right_to_wrong"] <= replaced["wrong_to_right"],
            "adjudication": dict(sorted(adjudication.items())),
            "tokens": {"total": _quantiles(totals), **{k: _quantiles(v) for k, v in sorted(phase_tokens.items())}}}


def _ceilings(path, args):
    identity_path = Path(str(path) + ".run.json")
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    return (args.max_total_tokens if args.max_total_tokens is not None else identity.get("max_total_tokens"),
            args.max_tool_calls if args.max_tool_calls is not None else identity.get("max_tool_calls"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["predict", "mechanism"])
    parser.add_argument("--single", help="predict: fresh single-agent arm file")
    parser.add_argument("--team", required=True, help="predict: V9 team arm file; mechanism: V10 team arm file")
    parser.add_argument("--max_total_tokens", type=int, default=None, help="Default: the team arm's run identity")
    parser.add_argument("--max_tool_calls", type=int, default=None, help="Default: the team arm's run identity")
    parser.add_argument("--realisation", type=float, default=REALISATION)
    parser.add_argument("--harmful", type=int, default=HARMFUL)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--families", default=str(DEFAULT_FAMILIES))
    parser.add_argument("--output", default=None, help="Write the JSON report here")
    args = parser.parse_args(argv)
    references = load_references(args.input, args.families)
    templates = template_ids(args.families)
    team = load_rows(args.team)
    if args.command == "predict":
        if not args.single:
            parser.error("predict needs --single")
        report = predict(load_rows(args.single), team, references, templates, *_ceilings(args.team, args),
                         realisation=args.realisation, harmful=args.harmful)
    else:
        report = mechanism(team, references, templates)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
