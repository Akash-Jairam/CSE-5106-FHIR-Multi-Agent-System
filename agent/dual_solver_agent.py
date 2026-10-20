"""V9 team: dual-solver verified release (docs/diagnosis_v8/design_v9.txt).

Three contexts on the same base model under one aggregate accountant:

1. analysis      the unchanged single-agent scaffold answers first with the full
                 allowance (the team never has less than the single);
2. recompute     the V3/V8 query-contract solver (LedgerSolver) re-derives an
                 answer from the original records without seeing the primary,
                 on the residual budget minus the adjudication reserve. Its
                 source-scope and convention checks are informational flags,
                 never a veto, and it gets no repair;
3. adjudication  one bounded call, only when both answers are non-null and
                 differ: it picks a candidate and names the deciding field.

The fixed controller releases the primary on agreement, applies the null rule
when exactly one candidate is non-null, and releases the adjudicated pick only
if the cited field verifies against the retrieved records. Every other path
releases the primary. The single repair is replacing the primary with the
recomputed candidate.
"""
import hashlib
import json
import re
from copy import deepcopy

from .controller import FixedController
from .convention_checks import convention_issues
from .ehr_semantics import SEMANTIC_GUIDANCE
from .framework import Role, Toolbox
from .output_schema import _json_objects
from .query_contract_agent import PHASE_CAPS, LedgerSolver, PhaseHost, _reply
from .single_agent import SingleAgent
from .verification_agent import _canonical
from utils.evidence_scorer import check_citation

PROTOCOL_V9 = "dual-solver-v9"
# Adjudication reserve held back from the recompute: tokens, and the adjudicator's fetches.
ADJUDICATION_RESERVE = (15_000, 2)
# Below this residual (before the reserve) the recompute is skipped and the primary released.
MIN_RESIDUAL_TOKENS = 25_000
# The recompute needs two source-check slots plus at least two working tools.
MIN_RECOMPUTE_TOOLS = 4
ADJUDICATOR_TURNS = 3
REFERENCE_ONLY_TYPES = frozenset({"Medication", "Location"})
SCOPE_ISSUE_PREFIX = "source_scope: unresolved patient/time binding for fetched records: "
PROGRAM_CHARS = 6_000
RECORD_ID_RE = re.compile(r"[A-Za-z]+/[A-Za-z0-9.-]{1,64}")
RELEASE_POLICIES = ("agree_primary", "null_rule_primary", "null_rule_recompute", "adjudicated_primary",
                    "adjudicated_recompute", "recompute_skipped_budget", "adjudication_defaulted")

ADJUDICATION_PROMPT = """You compare two candidate answers to one question about a patient's FHIR record.
Two solvers worked independently from the same records. You see each candidate's released answer, its cited
evidence, its executed final program and the host's automatic check flags, but not its reasoning. Decide which
candidate answers the question as asked under the shared conventions below. Do not compute a new answer and do
not merge the candidates. Flags are hints, not verdicts: a flagged candidate can still be right.
You may fetch at most two records with get_resources_by_resource_id to read a field the excerpts do not show.
Return only JSON:
{"pick":"A|B","check":{"resource_id":"<ResourceType>/<id>","field_path":"<dot/index path>",
"rule":"<the question requirement or shared convention that decides>","observed_value":<exact value at that path>}}
The check names one record field whose actual value decides between the candidates. The host verifies that the
record is among the retrieved records and holds exactly that value at that path; an unverifiable check is
treated as no decision.
""" + SEMANTIC_GUIDANCE


def run_v9_job(args_tuple):
    """Same job contract as run_v3_job / run_v8_job."""
    from utils import run_agent_safe
    import tools.cache as cache_module
    input_data, strategy, model, verbose, base_url, enable_cache, *rest = args_tuple
    if strategy != "multi_agent_v9":
        raise ValueError("V9 worker received a different strategy")
    cache_module.CACHE_ENABLED = enable_cache
    agent = DualSolverAgent(model, verbose=verbose, base_url=base_url)
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
# Deterministic controller rules (shared with scripts/replay_v9_controller.py)
# ---------------------------------------------------------------------------

def answers_agree(first, second, question=""):
    """Typed-normalised equality (the scorer's timestamp precision, numeric rounding, case, 1-lists)."""
    return _canonical(first, question) == _canonical(second, question)


def cited_ids(candidate):
    candidate = candidate or {}
    ids = list(candidate.get("contributing_ids") or [])
    ids += [c.get("resource_id") for c in candidate.get("evidence") or [] if isinstance(c, dict)]
    return {rid for rid in ids if isinstance(rid, str)}


def _scope_keys(issue, unresolved_scope):
    if unresolved_scope:
        return set(unresolved_scope)
    return {part.split("=", 1)[0].strip() for part in issue[len(SCOPE_ISSUE_PREFIX):].split(";") if part.strip()}


def source_valid(issues, unresolved_scope=None, cited=()):
    """True when the candidate's final source check passed for its cited resources.

    The recompute solver's unresolved-scope note is ignored when every unresolved
    record is reference-only metadata (Medication/Location can never be bound to
    the patient on their own) or is not cited by the candidate. Any other issue
    blocks. `issues` None means no check ran, which is not valid.
    """
    if issues is None:
        return False
    cited = set(cited)
    for issue in issues:
        if isinstance(issue, str) and issue.startswith(SCOPE_ISSUE_PREFIX):
            keys = _scope_keys(issue, unresolved_scope)
            if all(key.split("/", 1)[0] in REFERENCE_ONLY_TYPES or key not in cited for key in keys):
                continue
        return False
    return True


def budget_plan(max_total_tokens, max_tool_calls, used_tokens, used_tools):
    """The recompute allowance on the residual after the primary, minus the adjudication reserve."""
    reserve_tokens, reserve_tools = ADJUDICATION_RESERVE
    solver_tokens, solver_tools = PHASE_CAPS["solver"]
    residual_tokens = None if max_total_tokens is None else max_total_tokens - used_tokens
    residual_tools = None if max_tool_calls is None else max_tool_calls - used_tools
    tokens = solver_tokens if residual_tokens is None else min(solver_tokens, residual_tokens - reserve_tokens)
    tools = solver_tools if residual_tools is None else min(solver_tools, residual_tools - reserve_tools)
    skip = None
    if residual_tokens is not None and residual_tokens < MIN_RESIDUAL_TOKENS:
        skip = f"residual {residual_tokens} tokens < {MIN_RESIDUAL_TOKENS}"
    elif tools < MIN_RECOMPUTE_TOOLS:
        skip = f"recompute allowance {tools} tools < {MIN_RECOMPUTE_TOOLS}"
    return {"residual_tokens": residual_tokens, "residual_tools": residual_tools,
            "recompute_tokens": max(0, tokens), "recompute_tools": max(0, tools),
            "reserve_tokens": reserve_tokens, "reserve_tools": reserve_tools, "skip": skip}


def decide(primary_answer, recompute, question=""):
    """Controller steps 3-4: a release policy, or "adjudicate" for a both-non-null disagreement.

    `recompute` is None when the recompute was skipped, else {"answer", "source_valid"}.
    """
    if recompute is None:
        return "recompute_skipped_budget"
    other = recompute.get("answer")
    if answers_agree(primary_answer, other, question):
        return "agree_primary"
    if primary_answer is not None and other is None:
        return "null_rule_primary"
    if primary_answer is None:
        return "null_rule_recompute" if recompute.get("source_valid") else "null_rule_primary"
    return "adjudicate"


def candidate_order(question):
    """Fixed, question-derived A/B order so neither solver always appears first."""
    return ("primary", "recompute") if hashlib.sha256(question.encode()).digest()[0] % 2 == 0 \
        else ("recompute", "primary")


def parse_adjudication(content):
    """(verdict, None) or (None, reason) for the strict adjudicator schema."""
    if not isinstance(content, str):
        return None, "adjudication reply must be text containing JSON"
    verdict = next((obj for obj in _json_objects(content) if isinstance(obj, dict) and "pick" in obj), None)
    if verdict is None:
        return None, "no JSON object with a pick"
    if verdict.get("pick") not in ("A", "B"):
        return None, "pick must be A or B"
    check = verdict.get("check")
    if not isinstance(check, dict):
        return None, "check must be an object"
    rid, path, rule = check.get("resource_id"), check.get("field_path"), check.get("rule")
    if not isinstance(rid, str) or not RECORD_ID_RE.fullmatch(rid.strip()):
        return None, "check.resource_id must be ResourceType/id"
    if not isinstance(path, str) or not path.strip() or len(path) > 200:
        return None, "check.field_path must be a nonempty path"
    if not isinstance(rule, str) or not rule.strip() or len(rule) > 1000:
        return None, "check.rule must be a nonempty string"
    if "observed_value" not in check:
        return None, "check.observed_value is required"
    try:
        json.dumps(verdict, allow_nan=False)
    except (TypeError, ValueError):
        return None, "adjudication must be finite JSON"
    return {"pick": verdict["pick"], "check": {"resource_id": rid.strip(), "field_path": path.strip(),
                                               "rule": rule.strip(), "observed_value": check["observed_value"]}}, None


def record_index(*workspaces):
    """{"Type/id": record} over retrieved workspaces (later workspaces win)."""
    index = {}
    for workspace in workspaces:
        for rtype, records in (workspace or {}).items():
            for record in records or []:
                if isinstance(record, dict) and isinstance(record.get("id"), str):
                    index[f"{rtype}/{record['id']}"] = {**record, "resourceType": record.get("resourceType", rtype)}
    return index


def verify_check(check, index):
    """Status of the adjudicator's cited check against the held records ("valid" verifies)."""
    return check_citation({"resource_id": check["resource_id"], "field_path": check["field_path"],
                           "value": check["observed_value"]}, index)


# ---------------------------------------------------------------------------
# The team
# ---------------------------------------------------------------------------

class DualSolverAgent(SingleAgent):
    """The single-agent scaffold is the analysis context; the class adds recompute and adjudication."""
    protocol = PROTOCOL_V9

    def __init__(self, model, verbose=False, base_url=None):
        super().__init__(model, verbose=verbose, base_url=base_url)
        self.phase_caps = {}

    def _primary_candidate(self, result):
        computation = self.final_computation or {}
        revision = result.get("revision") or {}
        return {"answer": result.get("answer"), "evidence": deepcopy(result.get("evidence") or []),
                "limitation": result.get("limitation"),
                "contributing_ids": list(computation.get("contributing_ids") or []),
                "program": computation.get("code"), "executed": bool(computation),
                "source_issues": revision.get("final_check_issues"), "error": result.get("error")}

    def _recompute(self, question, context, plan):
        """Context 2: the draft-mode query-contract solver on the residual allowance."""
        self.phase_caps["recompute"] = (plan["recompute_tokens"], plan["recompute_tools"])
        phase = PhaseHost(self, "recompute")
        solver = LedgerSolver(self, phase)
        try:
            result = solver._run(question)
            error = result.get("error")
        except Exception as exc:
            result, error = {}, f"recompute failed: {type(exc).__name__}"
        finally:
            self.messages.extend(solver.messages)
        report = phase.report()
        workspace = result.get("retrieved_fhir_resources") or {}
        revision = result.get("revision") or {}
        computed = solver._computed_candidate(workspace) if not error else None
        if computed is not None and not computed.get("error"):
            candidate = {**computed, "program": (solver.final_computation or {}).get("code"), "executed": True}
        else:
            # Draft mode keeps the solver's last stated final; the flags say it was not executed.
            stated = revision.get("final") or revision.get("draft") or {}
            candidate = {"answer": stated.get("answer"), "evidence": deepcopy(stated.get("evidence") or []),
                         "limitation": stated.get("limitation"), "contributing_ids": [], "program": None,
                         "executed": False}
        if error or phase.fatal:
            candidate = {**candidate, "answer": None, "evidence": []}
        unresolved = deepcopy(solver._scoped_box.scope_failures if solver._scoped_box else {})
        issues = revision.get("final_check_issues")
        try:
            conventions = convention_issues(context["question"], candidate, workspace, solver.complete_types())
        except Exception as exc:
            conventions = [{"check": "convention_checks", "type": "check_error", "action": type(exc).__name__}]
        candidate.update(source_issues=issues, unresolved_scope=unresolved,
                         source_valid=source_valid(issues, unresolved, cited_ids(candidate)),
                         convention_issues=conventions, complete_types=sorted(solver.complete_types()),
                         error=error, usage_verified=not phase.fatal)
        return candidate, workspace, report

    def _adjudicate(self, context, primary, recompute, workspaces):
        """Context 3: one bounded comparison. Returns (pick or None, record)."""
        remaining = (None if self.max_total_tokens is None
                     else self.max_total_tokens - self.total_usage["total_tokens"])
        tokens = ADJUDICATION_RESERVE[0] if remaining is None else max(0, min(ADJUDICATION_RESERVE[0], remaining))
        tools = ADJUDICATION_RESERVE[1]
        if self.max_tool_calls is not None:
            tools = max(0, min(tools, self.max_tool_calls - self.tool_calls_made))
        self.phase_caps["adjudicate"] = (tokens, tools)
        order = candidate_order(context["question"])
        candidates = {"primary": primary, "recompute": recompute}
        held = record_index(*workspaces)
        packet = {"context": context, "candidates": {}}
        for label, name in zip("AB", order):
            candidate = candidates[name]
            records = {rid: held[rid] for rid in sorted(cited_ids(candidate)) if rid in held}
            packet["candidates"][label] = {
                "answer": candidate.get("answer"), "evidence": candidate.get("evidence"),
                "limitation": candidate.get("limitation"),
                "executed_final_program": (candidate.get("program") or "")[:PROGRAM_CHARS] or None,
                "flags": {"source_issues": candidate.get("source_issues"),
                          "convention_issues": [i.get("action") for i in candidate.get("convention_issues") or []]},
                "cited_records": json.loads(_reply(records))}
        record = {"order": dict(zip("AB", order)), "phase": None, "verdict": None, "error": None,
                  "check_status": None, "pick": None}
        phase = PhaseHost(self, "adjudicate")
        resources = {}
        for rid, item in held.items():
            resources.setdefault(rid.split("/", 1)[0], []).append(deepcopy(item))
        toolbox = Toolbox(("get_resources_by_resource_id",), resources=resources,
                          patient_fhir_id=context["patient_fhir_id"], current_time=context["current_time"])
        role = Role("adjudicator", ADJUDICATION_PROMPT, toolbox, phase,
                    max_turns=ADJUDICATOR_TURNS, max_tool_calls=tools)
        try:
            result = role.run(json.dumps(packet, default=str))
            verdict, error = (None, result.error) if result.error else parse_adjudication(result.content)
        except Exception as exc:
            verdict, error = None, f"adjudication failed: {type(exc).__name__}"
        record["phase"] = phase.report()
        record["verdict"], record["error"] = verdict, error
        if verdict is None or phase.fatal:
            record["error"] = record["error"] or "aggregate usage could not be verified"
            return None, record
        record["check_status"] = verify_check(verdict["check"], record_index(*workspaces, toolbox.resources))
        if record["check_status"] != "valid":
            return None, record
        record["pick"] = record["order"][verdict["pick"]]
        return record["pick"], record

    def _run(self, question):
        context = FixedController.question_context(question)
        # Context 1: the baseline scaffold itself, on the full shared allowance.
        primary_result = super()._run(question)
        primary_trace = self.messages
        primary_last_error, primary_stop = self.last_llm_error, self.budget_stop
        primary = self._primary_candidate(primary_result)
        primary_workspace = primary_result.get("retrieved_fhir_resources") or {}
        plan = budget_plan(self.max_total_tokens, self.max_tool_calls,
                           self.total_usage["total_tokens"], self.tool_calls_made)
        phases = [{"phase": "primary", "tokens": self.total_usage["total_tokens"],
                   "tool_calls": self.tool_calls_made, "stopped": primary_stop}]
        recompute, recompute_workspace, adjudication = None, {}, None
        if plan["skip"] is None:
            recompute, recompute_workspace, report = self._recompute(question, context, plan)
            phases.append(report)
            if not recompute["usage_verified"]:
                plan = {**plan, "skip": "recompute usage could not be verified"}
                recompute = {**recompute, "answer": None}
        policy = decide(primary["answer"], None if plan["skip"] else recompute, context["question"])
        if policy == "adjudicate":
            pick, adjudication = self._adjudicate(context, primary, recompute,
                                                  (primary_workspace, recompute_workspace))
            if adjudication["phase"]:
                phases.append(adjudication["phase"])
            policy = "adjudication_defaulted" if pick is None else f"adjudicated_{pick}"
        released_recompute = policy in ("null_rule_recompute", "adjudicated_recompute")
        released = recompute if released_recompute else primary
        if released_recompute:
            self.last_llm_error, error = None, None
        else:
            # Releasing the primary reproduces the single agent's own output contract.
            self.last_llm_error, self.budget_stop, error = primary_last_error, primary_stop, primary["error"]
        summary = lambda c: None if c is None else {k: deepcopy(v) for k, v in c.items() if k != "program"}
        return {"answer": released["answer"], "evidence": released["evidence"], "limitation": released["limitation"],
                "error": error, "trace": primary_trace,
                "retrieved_fhir_resources": recompute_workspace if released_recompute else primary_workspace,
                "revision": {"protocol": self.protocol,
                             "draft": {"answer": primary["answer"], "evidence": primary["evidence"],
                                       "limitation": primary["limitation"]},
                             "final": {"answer": released["answer"], "evidence": released["evidence"],
                                       "limitation": released["limitation"]},
                             "revised": released_recompute,
                             "answer_changed": released_recompute and not answers_agree(
                                 primary["answer"], released["answer"], context["question"]),
                             "release_policy": policy, "primary": summary(primary), "recompute": summary(recompute),
                             "primary_revision": primary_result.get("revision"),
                             "primary_error": primary["error"], "budget_plan": plan,
                             "adjudication": adjudication, "phases": phases,
                             "context_compactions": self.context_compactions}}
