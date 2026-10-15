"""Selected candidate for paired experiment, not a production default.

No late model audit, second answer computation, or plan-driven tool restriction.
Source/execution gates, deferred scope, and SDK host-boundary conversion are
inherited from the frozen V3 module. This additive strategy is selected for
the next paired experiment; its accuracy has not been benchmarked.
"""
from copy import deepcopy
import json

from .base_agent import BaseAgent
from .controller import FixedController
from .ehr_semantics import SEMANTIC_GUIDANCE
from .framework import Role, Toolbox
from .query_contract_agent import LedgerSolver, PhaseHost, fingerprint

PHASE_CAPS = {"planner": (10_000, 0), "solver": (190_000, 30)}
PROTOCOL = "plan-conditioned-v4"
OPERATORS = frozenset({"population", "filter", "reference", "time_window",
                      "order", "select", "project", "aggregate", "combine",
                      "multiplicity", "units", "output"})

PLAN_PROMPT = """Derive a compact advisory plan for the original patient question.
You see only the original question/context and shared conventions: no solver
answer, code, patient records, selections or prior interpretation. Do not compute
or predict an answer, output executable code, select resource IDs, or request tools.
Preserve each operand's population, concept and reference dependencies, exact
predicates/connectives, temporal anchor/field/endpoints, ordering and rank,
requested output field/unit, multiplicity and meaning of empty or missing data.
Use the original question and shared guide. Distinguish explicit requirements
from tentative field mappings. Never add normal clinical ranges, forced joins,
deduplication, non-null filtering or fallback field substitutions. An empty
result is not an error; missing scope or failed retrieval does not establish zero.
Keep separate operands separate. Retain ordering before projecting its value.
Unknown mappings remain unknown, not invented instructions.

Return one JSON object:
{"status":"READY|UNKNOWN","steps":[{"id":"p1","operator":"population|filter|reference|time_window|order|select|project|aggregate|combine|multiplicity|units|output",
"inputs":[],"quote":"exact original question substring","span":[0,1],
"requirement":"one concise obligation or explicitly tentative mapping"}],
"unknowns":["unresolved mapping or assumption"]}.
Inputs refer only to earlier step IDs. Every step quotes a unique exact substring
of the original question; the host checks provenance, not semantic correctness.
Use UNKNOWN and no steps if you cannot offer anchored advice. Do not emit answer,
code, SQL, resources, selected IDs, template IDs or benchmark annotations. Keep
the reply under 2,000 tokens. The solver may reject unsupported advice; this plan
cannot restrict tools or determine the final answer.
""" + SEMANTIC_GUIDANCE

ADVICE_PROMPT = """ADVISORY QUESTION PLAN, derived before any solver draft or records.
The original question and shared conventions remain authoritative. This plan
can be wrong. Use it as a checklist, and correct or ignore unsupported mappings
rather than forcing the data to fit. It changes neither your available tools nor
the execution/source contract. Do not treat a plan as evidence of completeness,
an answer, or permission to rewrite a valid empty result. Keep exact source values
and units in citations. Preserve event identity and order until selection, then
project only the requested output. Explain interpretation deviations briefly
in your existing operation/limitation fields; do not add unrequested output.
Frozen advisory plan JSON:
"""


def parse_plan(content, question):
    """Strict, bounded question-only interface; unique quotes normalize spans.

    Exact anchoring proves textual provenance, not correct clinical semantics.
    Invalid/unknown plans are ignored as a whole; no partial guessed plan leaks
    into the solver and no malformed-plan veto affects a source-valid result.
    """
    if not isinstance(content, str) or len(content) > 16_000:
        return None, "advisory plan is not bounded text"
    try:
        plan = json.loads(content)
        json.dumps(plan, allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        return None, "advisory plan is not finite JSON"
    if not isinstance(plan, dict) or set(plan) != {"status", "steps", "unknowns"}:
        return None, "advisory plan schema requires status, steps and unknowns only"
    if not isinstance(plan["status"], str) or plan["status"] not in {"READY", "UNKNOWN"}:
        return None, "advisory plan status is invalid"
    unknowns = plan["unknowns"]
    if (not isinstance(unknowns, list) or len(unknowns) > 16
            or any(not isinstance(item, str) or not item.strip() or len(item) > 800 for item in unknowns)):
        return None, "advisory plan unknowns are invalid"
    if plan["status"] == "UNKNOWN":
        return None, "planner declared UNKNOWN; advice ignored"
    if not isinstance(plan["steps"], list) or not 1 <= len(plan["steps"]) <= 16:
        return None, "advisory plan needs 1-16 anchored steps"
    normalized, previous_ids = [], set()
    for step in plan["steps"]:
        if (not isinstance(step, dict) or set(step) != {"id", "operator", "inputs", "quote", "span", "requirement"}
                or not isinstance(step["id"], str) or not step["id"].strip()
                or len(step["id"]) > 32 or step["id"] in previous_ids
                or not isinstance(step["operator"], str) or step["operator"] not in OPERATORS):
            return None, "advisory step identity/operator is invalid"
        inputs = step["inputs"]
        if (not isinstance(inputs, list) or len(inputs) > 16
                or any(not isinstance(item, str) or item not in previous_ids for item in inputs)
                or len(set(inputs)) != len(inputs)):
            return None, "advisory step inputs must refer uniquely to earlier steps"
        quote, span, requirement = step["quote"], step["span"], step["requirement"]
        if (not isinstance(quote, str) or not quote.strip() or len(quote) > 2_000
                or not isinstance(span, list) or len(span) != 2
                or any(type(value) is not int for value in span)
                or not isinstance(requirement, str) or not requirement.strip() or len(requirement) > 800):
            return None, "advisory step quote/span/requirement is invalid"
        # Search each possible start, including overlapping repeated substrings.
        starts = [index for index in range(len(question)) if question.startswith(quote, index)]
        if len(starts) != 1:
            return None, "advisory step quote is missing or ambiguous in the original question"
        derived_span = [starts[0], starts[0] + len(quote)]
        normalized.append({**deepcopy(step), "span": derived_span,
                           "supplied_span": deepcopy(span),
                           "span_normalization": "unchanged" if span == derived_span else "host_unique_quote"})
        previous_ids.add(step["id"])
    return {"status": "READY", "steps": normalized, "unknowns": deepcopy(unknowns)}, None


class V4PhaseHost(PhaseHost):
    """Reuse the verified aggregate accountant with V4's two local caps."""
    def __init__(self, parent, name):
        if name not in PHASE_CAPS:
            raise ValueError("V4 permits only planner and solver phases")
        super().__init__(parent, "contract" if name == "planner" else "solver")
        self.name = name
        self.token_cap, self.tool_cap = PHASE_CAPS[name]


class PlanLedgerSolver(LedgerSolver):
    """Existing V3 lifecycle/gates with optional visible advisory context."""
    def __init__(self, parent, phase, plan=None):
        super().__init__(parent, phase)
        # Twenty real solver invocations may require two separate no-model
        # boundaries: the first source check and the corrected final check.
        # The inherited _actual_calls guard still caps real invocations at 20.
        self.max_iterations = 22
        self.system_msg = deepcopy(self.system_msg)
        if plan is not None:
            self.system_msg[0]["content"] += "\n\n" + ADVICE_PROMPT + json.dumps(plan, allow_nan=False)


class PlanConditionedAgent(BaseAgent):
    def _run(self, question):
        context = FixedController.question_context(question)
        phases, plan, plan_error = [], None, None
        planner = V4PhaseHost(self, "planner")
        role = Role("advisory_question_plan", PLAN_PROMPT, Toolbox(()), planner,
                    max_turns=1, max_tool_calls=0)
        result = role.run(json.dumps(context))
        if result.error:
            plan_error = result.error
        else:
            plan, plan_error = parse_plan(result.content, context["question"])
        phases.append(planner.report())
        plan_hash = fingerprint(plan) if plan is not None else None
        common = {"protocol": PROTOCOL, "plan": plan, "plan_error": plan_error,
                  "plan_sha256": plan_hash, "plan_visible": plan is not None,
                  "phases": phases, "targeted_revised": False}
        if planner.fatal:
            return {"answer": None, "evidence": [], "limitation": "Aggregate usage could not be verified",
                    "error": result.error or "Planner aggregate accounting failed", "trace": self.messages,
                    "revision": {**common, "draft": None,
                        "final": {"answer": None, "evidence": [], "limitation": "Aggregate usage could not be verified"},
                        "revised": False, "solver_source_revised": False,
                        "final_check_issues": [{"type": "accounting", "action": "Planner aggregate usage unverified"}],
                        "release_policy": "aggregate_accounting_failed"}}
        phase = V4PhaseHost(self, "solver")
        solver = PlanLedgerSolver(self, phase, plan=plan)
        result = solver._run(question)
        self.messages.extend(solver.messages)
        phases.append(phase.report())
        workspace = deepcopy(result.get("retrieved_fhir_resources", {}))
        primary = solver._computed_candidate(workspace)
        revision = result.get("revision") or {}
        draft = revision.get("draft") or primary
        issues = revision.get("final_check_issues")
        common.update({"primary": primary, "primary_revision": result.get("revision"),
                       "draft": draft, "solver_source_revised": bool(revision.get("revised")),
                       "revised": bool(revision.get("revised")), "check_issues": revision.get("check_issues"),
                       "retrieval_ledger": deepcopy(solver.retrieval_ledger),
                       "complete_types": sorted(solver.complete_types()),
                       "unresolved_scope": deepcopy(solver._scoped_box.scope_failures if solver._scoped_box else {}),
                       "context_compactions": deepcopy(self.context_compactions)})
        if (phase.fatal or result.get("error") or primary is None or primary.get("error") or issues != []):
            limitation = result.get("limitation") or "No source-validated explicit final computation"
            final = {"answer": None, "evidence": [], "limitation": limitation}
            return {**result, **final, "trace": self.messages,
                    "revision": {**common, "final": final,
                        "answer_changed": bool(draft is not None and draft.get("answer") is not None),
                        "final_check_issues": issues if issues else [{"type": "execution_contract",
                            "action": "No source-validated explicit final"}],
                        "release_policy": "primary_failed_execution_or_source_contract"}}
        self.last_llm_error = None  # nonfatal plan availability/format errors are retained above
        return {"answer": primary["answer"], "evidence": primary["evidence"],
                "limitation": primary["limitation"], "trace": self.messages,
                "retrieved_fhir_resources": workspace,
                "revision": {**common, "final": primary,
                    "answer_changed": draft["answer"] != primary["answer"], "final_check_issues": [],
                    "release_policy": "source_valid_primary_preserved"}}


def run_v4_job(args_tuple):
    """Normal budgeted worker path, including lossless audit storage before IPC."""
    from utils import run_agent_safe
    import tools.cache as cache_module
    input_data, strategy, model, verbose, base_url, enable_cache, *rest = args_tuple
    if strategy != "multi_agent_v4":
        raise ValueError("V4 worker received a different strategy")
    cache_module.CACHE_ENABLED = enable_cache
    agent = PlanConditionedAgent(model, verbose=verbose, base_url=base_url)
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
