"""V10 team: self-consistent primary, deterministic finishers (docs/diagnosis_v8/design_v10.txt).

Three contexts on the same base model under one aggregate accountant:

1. analysis      the unchanged single-agent scaffold answers twice, in independent
                 trajectories (no shared transcript). Draw 1 has the full allowance
                 (the team never has less than the single); draw 2 runs on the
                 residual minus the finisher reserve. If the two typed-normalised
                 answers differ and at least THIRD_DRAW_MIN remains, a third draw
                 runs (residual minus the finisher and adjudication reserves) and the
                 majority answer is the candidate. Agreement merges the agreeing
                 draws' cited evidence. No majority keeps draw 1 (vote "split").
2. verification  the template-routed deterministic finisher (agent.finishers)
                 recomputes the convention-sensitive last step from the candidate's
                 cited anchor record; no model call.
3. adjudication  only when the vote is split and no finisher decided (replaced the
                 candidate or recomputed the same answer): the V9 adjudicator picks
                 between draw 1 and the first later draw with a different non-null
                 answer; its pick is released only if its cited check verifies.

Release policies: vote_candidate, finisher_replaced, adjudicated_draw1,
adjudicated_other, adjudication_defaulted. When a draw's own answer is released,
that draw's output contract (error, budget stop) is reproduced.
"""
import json
from copy import deepcopy

from .controller import FixedController
from .dual_solver_agent import ADJUDICATION_RESERVE, DualSolverAgent, answers_agree, record_index
from .finishers import Fetcher, run_finisher
from .single_agent import SingleAgent

PROTOCOL_V10 = "finisher-team-v10"
# Held back from draws 2 and 3 for the finisher: tokens, and its patient-scoped fetches (it makes at most 2).
FINISHER_RESERVE = (10_000, 3)
# A third draw runs only when the two draws disagree and at least this residual remains.
THIRD_DRAW_MIN = (60_000, 8)
# Below this allowance (after the reserve) draw 2 is skipped and draw 1 is the candidate.
MIN_DRAW = (25_000, 4)
VOTES = ("agree2", "majority3", "split", "single_draw")
RELEASE_POLICIES = ("vote_candidate", "finisher_replaced", "adjudicated_draw1", "adjudicated_other",
                    "adjudication_defaulted")
FATAL_STOPS = ("provider did not report token usage",)


def run_v10_job(args_tuple):
    """Same job contract as run_v9_job."""
    from utils import run_agent_safe
    import tools.cache as cache_module
    input_data, strategy, model, verbose, base_url, enable_cache, *rest = args_tuple
    if strategy != "multi_agent_v10":
        raise ValueError("V10 worker received a different strategy")
    cache_module.CACHE_ENABLED = enable_cache
    agent = FinisherTeamAgent(model, verbose=verbose, base_url=base_url)
    agent.set_budget(**((rest[0] if rest else None) or {}))
    agent.set_perturbation(rest[1] if len(rest) > 1 else None)
    include_full_audit = bool(rest[2]) if len(rest) > 2 else False
    audit_context = rest[3] if len(rest) > 3 else None
    if include_full_audit:
        output = run_agent_safe(agent, input_data, include_full_audit=True)
        if audit_context is not None:
            from scripts.audit_store import externalize_result_audit
            output = externalize_result_audit(output, audit_context["output_dir"],
                                               question_id=audit_context["question_id"])
        return output
    return run_agent_safe(agent, input_data)


# ---------------------------------------------------------------------------
# Deterministic controller rules (shared with scripts/replay_v10.py)
# ---------------------------------------------------------------------------

def draw_allowance(max_total_tokens, max_tool_calls, used_tokens, used_tools, reserve=FINISHER_RESERVE):
    """{"tokens", "tools", "skip"} for a later draw: the residual minus `reserve`; None means unlimited."""
    tokens = None if max_total_tokens is None else max_total_tokens - used_tokens - reserve[0]
    tools = None if max_tool_calls is None else max_tool_calls - used_tools - reserve[1]
    skip = None
    if tokens is not None and tokens < MIN_DRAW[0]:
        skip = f"draw allowance {tokens} tokens < {MIN_DRAW[0]}"
    elif tools is not None and tools < MIN_DRAW[1]:
        skip = f"draw allowance {tools} tools < {MIN_DRAW[1]}"
    return {"tokens": tokens, "tools": tools, "skip": skip}


def third_draw_allowed(max_total_tokens, max_tool_calls, used_tokens, used_tools):
    return ((max_total_tokens is None or max_total_tokens - used_tokens >= THIRD_DRAW_MIN[0])
            and (max_tool_calls is None or max_tool_calls - used_tools >= THIRD_DRAW_MIN[1]))


def vote(answers, question=""):
    """(vote, indices of the draws behind the candidate); the first index supplies the released answer."""
    if len(answers) == 1:
        return "single_draw", [0]
    groups = []
    for index, answer in enumerate(answers):
        group = next((g for g in groups if answers_agree(answers[g[0]], answer, question)), None)
        if group is None:
            groups.append([index])
        else:
            group.append(index)
    best = max(groups, key=len)
    if len(best) >= 2:
        return ("agree2" if len(answers) == 2 else "majority3"), best
    return "split", [0]


def merge_candidates(candidates):
    """The first candidate's answer with the union of every agreeing candidate's citations."""
    merged = deepcopy(candidates[0])
    seen = {json.dumps(c, sort_keys=True, default=str) for c in merged.get("evidence") or []}
    ids = list(merged.get("contributing_ids") or [])
    for other in candidates[1:]:
        for citation in other.get("evidence") or []:
            key = json.dumps(citation, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                merged.setdefault("evidence", []).append(deepcopy(citation))
        ids += [rid for rid in other.get("contributing_ids") or [] if rid not in ids]
    merged["contributing_ids"] = ids
    return merged


def adjudication_pair(answers):
    """(0, j): draw 1 and the first later draw with a different non-null answer, or None."""
    if answers[0] is None:
        return None
    other = next((j for j in range(1, len(answers))
                  if answers[j] is not None and not answers_agree(answers[0], answers[j])), None)
    return None if other is None else (0, other)


def _by_type(index):
    grouped = {}
    for key, record in index.items():
        grouped.setdefault(key.split("/", 1)[0], []).append(record)
    return grouped


# ---------------------------------------------------------------------------
# The team
# ---------------------------------------------------------------------------

class FinisherTeamAgent(DualSolverAgent):
    """k-draw single-agent primary, a deterministic finisher, and the V9 adjudicator for split votes."""
    protocol = PROTOCOL_V10

    def _draw(self, question, index, allowance=None):
        """One independent run of the single-agent scaffold, optionally capped below the shared ceiling."""
        start_tokens, start_tools = self.total_usage["total_tokens"], self.tool_calls_made
        ceilings = (self.max_total_tokens, self.max_tool_calls)
        self.budget_stop, self.last_llm_error = None, None
        if allowance is not None:
            if allowance["tokens"] is not None:
                self.max_total_tokens = start_tokens + allowance["tokens"]
            if allowance["tools"] is not None:
                self.max_tool_calls = start_tools + allowance["tools"]
        try:
            result = SingleAgent._run(self, question)
        except Exception as exc:
            if index == 1:
                raise  # draw 1 is the single agent itself and fails exactly as it does
            self.final_computation = None
            result = {"answer": None, "evidence": [], "limitation": None, "trace": self.messages,
                      "retrieved_fhir_resources": {}, "error": f"draw {index} failed: {type(exc).__name__}"}
        finally:
            self.max_total_tokens, self.max_tool_calls = ceilings
        stop = self.budget_stop
        over = ceilings[0] is not None and self.total_usage["total_tokens"] > ceilings[0]
        return {"index": index, "candidate": self._primary_candidate(result), "result": result,
                "workspace": result.get("retrieved_fhir_resources") or {}, "trace": self.messages,
                "stop": stop, "last_error": self.last_llm_error,
                "fatal": over or any(marker in (stop or "") for marker in FATAL_STOPS),
                "phase": {"phase": f"draw{index}", "tokens": self.total_usage["total_tokens"] - start_tokens,
                          "tool_calls": self.tool_calls_made - start_tools, "stopped": stop,
                          "allowance": allowance}}

    def _finisher_trace(self, record):
        """Host-side audit entries for the finisher's charged fetches (no model call)."""
        messages = []
        for number, fetch in enumerate(record["fetches"], 1):
            call_id = f"finisher_fetch_{number}"
            messages.append({"role": "assistant", "content": None, "tool_calls": [{
                "id": call_id, "type": "function", "function": {
                    "name": "get_resources_by_patient_fhir_id",
                    "arguments": json.dumps({"resource_type": fetch["resource_type"]})}}]})
            messages.append({"role": "tool", "tool_call_id": call_id, "name": "get_resources_by_patient_fhir_id",
                             "content": json.dumps({k: v for k, v in fetch.items() if k != "resource_type"})})
        return messages

    def _run(self, question):
        context = FixedController.question_context(question)
        draws = [self._draw(question, 1)]
        plan = {"draw2": None, "draw3": None}
        if not draws[0]["fatal"]:
            allowance = draw_allowance(self.max_total_tokens, self.max_tool_calls,
                                       self.total_usage["total_tokens"], self.tool_calls_made)
            plan["draw2"] = allowance
            if allowance["skip"] is None:
                draws.append(self._draw(question, 2, allowance))
        answers = lambda: [d["candidate"]["answer"] for d in draws if not d["fatal"]]
        if (len(draws) == 2 and not draws[1]["fatal"]
                and not answers_agree(*answers(), context["question"])):
            if third_draw_allowed(self.max_total_tokens, self.max_tool_calls,
                                  self.total_usage["total_tokens"], self.tool_calls_made):
                reserve = (FINISHER_RESERVE[0] + ADJUDICATION_RESERVE[0], FINISHER_RESERVE[1] + ADJUDICATION_RESERVE[1])
                allowance = draw_allowance(self.max_total_tokens, self.max_tool_calls,
                                           self.total_usage["total_tokens"], self.tool_calls_made, reserve)
                plan["draw3"] = allowance
                if allowance["skip"] is None:
                    draws.append(self._draw(question, 3, allowance))
            else:
                plan["draw3"] = {"skip": f"residual below {THIRD_DRAW_MIN[0]} tokens / {THIRD_DRAW_MIN[1]} tools"}
        valid = [d for d in draws if not d["fatal"]] or draws[:1]
        fatal = any(d["fatal"] for d in draws)
        outcome, group = vote([d["candidate"]["answer"] for d in valid], context["question"])
        candidate = merge_candidates([valid[i]["candidate"] for i in group])
        source = valid[group[0]]

        finisher = {"template": None, "finisher": "none", "reason": "aggregate usage could not be verified",
                    "anchors": [], "result": None, "evidence": [], "fetches": [], "detail": {}}
        fetcher = None
        if not fatal:
            fetcher = Fetcher(self._take_tool_call, context["patient_fhir_id"], context["current_time"])
            finisher = run_finisher(context, candidate, fetcher)
        decided = finisher["finisher"] == "replaced" or finisher["reason"] == "result_equals_primary"

        policy, adjudication, released = "vote_candidate", None, candidate
        workspaces = [valid[i]["workspace"] for i in group]
        if finisher["finisher"] == "replaced":
            policy = "finisher_replaced"
            released = {"answer": finisher["result"], "evidence": deepcopy(finisher["evidence"]), "limitation": None}
            workspaces.append(fetcher.workspace)
        elif outcome == "split" and not decided and not fatal:
            pair = adjudication_pair([d["candidate"]["answer"] for d in valid])
            if pair is not None:
                first, other = (valid[i] for i in pair)
                pick, adjudication = self._adjudicate(context, first["candidate"], other["candidate"],
                                                      (first["workspace"], other["workspace"]))
                adjudication["other_draw"] = other["index"]
                if pick is None:
                    policy = "adjudication_defaulted"
                elif pick == "primary":
                    policy = "adjudicated_draw1"
                else:
                    policy, source, released = "adjudicated_other", other, other["candidate"]
                    workspaces = [other["workspace"]]

        if policy in ("vote_candidate", "adjudicated_draw1", "adjudication_defaulted", "adjudicated_other"):
            # A draw's own answer is released: reproduce that draw's output contract.
            self.budget_stop, self.last_llm_error = source["stop"], source["last_error"]
            error = source["candidate"]["error"]
        else:
            self.budget_stop, self.last_llm_error, error = source["stop"], None, None
        trace = [message for d in draws for message in d["trace"]] + self._finisher_trace(finisher)
        phases = [d["phase"] for d in draws]
        if finisher["fetches"]:
            phases.append({"phase": "finisher", "tokens": 0, "tool_calls": len(finisher["fetches"])})
        if adjudication and adjudication.get("phase"):
            phases.append(adjudication["phase"])
        summary = lambda d: {**{k: deepcopy(v) for k, v in d["candidate"].items() if k != "program"},
                             "index": d["index"], "stop": d["stop"], "fatal": d["fatal"]}
        first = draws[0]["candidate"]
        return {"answer": released["answer"], "evidence": released["evidence"], "limitation": released["limitation"],
                "error": error, "trace": trace,
                "retrieved_fhir_resources": _by_type(record_index(*workspaces)),
                "revision": {"protocol": self.protocol,
                             "draft": {"answer": first["answer"], "evidence": first["evidence"],
                                       "limitation": first["limitation"]},
                             "candidate": {"answer": candidate["answer"], "evidence": candidate["evidence"],
                                           "limitation": candidate["limitation"],
                                           "contributing_ids": candidate["contributing_ids"]},
                             "final": {"answer": released["answer"], "evidence": released["evidence"],
                                       "limitation": released["limitation"]},
                             "revised": policy in ("finisher_replaced", "adjudicated_other"),
                             "answer_changed": not answers_agree(first["answer"], released["answer"],
                                                                 context["question"]),
                             "vote": outcome, "vote_group": [valid[i]["index"] for i in group],
                             "finisher": finisher, "release_policy": policy,
                             "draws": [summary(d) for d in draws],
                             "primary_revision": draws[0]["result"].get("revision"),
                             "primary_error": first["error"], "budget_plan": plan,
                             "adjudication": adjudication, "phases": phases,
                             "context_compactions": self.context_compactions}}

