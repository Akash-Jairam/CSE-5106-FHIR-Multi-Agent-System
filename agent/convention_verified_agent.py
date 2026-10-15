"""V8 team: V3 query-contract pipeline + deterministic convention verification.

docs/diagnosis_v8/decision.md (D2). After the V3 release decision the host runs
agent.convention_checks on the released, source-valid draft. If a check fires,
the fixed routing sends ONE repair to the responsible role: the solver (the
analysis/retrieval context) continues its own trajectory with the issue list,
fetching what is missing or recomputing with the stated convention. The repair
is released only if it is source-valid and the triggering checks no longer
fire; otherwise the original source-valid draft stands. Null is never
substituted for a valid draft.

Optionally the static program audit (the model-based verifier role) runs on a
different model family (`verifier_model`), breaking the same-model error
correlation measured in V5-V7; retrieval and analysis stay on the base model
and all usage is counted in the one aggregate accountant.
"""
import json
import os
from copy import deepcopy

from .convention_checks import convention_issues, repair_message
from .controller import FixedController
from .query_contract_agent import HostBoundaryMessage, PHASE_CAPS, PhaseHost, QueryContractAgent
from .output_schema import parse_final_message

PHASE_CAPS["repair"] = (45_000, 6)
PROTOCOL_V8 = "convention-verified-v8"
REPAIR_MODEL_CALLS = 6
VERIFIER_MODEL_ENV = "VERIFIER_MODEL"


def run_v8_job(args_tuple):
    """Same job contract as run_v3_job; the verifier model comes from $VERIFIER_MODEL."""
    from utils import run_agent_safe
    import tools.cache as cache_module
    input_data, strategy, model, verbose, base_url, enable_cache, *rest = args_tuple
    if strategy != "multi_agent_v8":
        raise ValueError("V8 worker received a different strategy")
    cache_module.CACHE_ENABLED = enable_cache
    agent = ConventionVerifiedAgent(model, verbose=verbose, base_url=base_url,
                                    verifier_model=os.environ.get(VERIFIER_MODEL_ENV) or None)
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


class ConventionVerifiedAgent(QueryContractAgent):
    protocol = PROTOCOL_V8

    def __init__(self, model, verbose=False, base_url=None, verifier_model=None):
        super().__init__(model, verbose=verbose, base_url=base_url)
        self.verifier_model = verifier_model if verifier_model and verifier_model != model else None

    def _run_audit(self, audit_role, packet):
        if self.verifier_model is None:
            return super()._run_audit(audit_role, packet)
        base = self.model
        self.model = self.verifier_model
        try:
            return super()._run_audit(audit_role, packet)
        finally:
            self.model = base

    def _post_release(self, released, policy, context, solver, primary_result, phases):
        workspace = primary_result.get("retrieved_fhir_resources", {})
        issues = convention_issues(context["question"], released, workspace, solver.complete_types())
        extension = {"verifier_model": self.verifier_model, "convention_issues": issues,
                     "convention_repair": None, "routing": None}
        if not issues:
            return released, policy, extension
        routable = [issue for issue in issues if issue["type"] != "check_error"]
        router = FixedController.__new__(FixedController)
        router._workspace = workspace
        role, decisions = router.route(routable, released) if routable else (None, [])
        extension["routing"] = {"role": role, "decisions": decisions}
        if role is None:
            extension["convention_repair"] = {"status": "not_run", "reason": "no repairable convention issue"}
            return released, policy, extension
        phase = PhaseHost(self, "repair")
        offset = len(solver.messages)
        try:
            candidate, report = self._repair(solver, phase, workspace, routable, context)
        except Exception as exc:
            candidate, report = None, {"status": "failed", "reason": f"repair failed: {type(exc).__name__}"}
        finally:
            self.messages.extend(solver.messages[offset:])
        phases.append(phase.report())
        if phase.fatal:
            report = {**report, "status": "failed", "reason": "aggregate usage could not be verified"}
            candidate = None
        extension["convention_repair"] = report
        if candidate is None:
            return released, policy, extension
        return candidate, "convention_repair_released", extension

    def _repair(self, solver, phase, workspace, issues, context):
        """One routed repair inside the solver's own trajectory; returns (candidate or None, report)."""
        triggered = {issue["check"] for issue in issues}
        solver._phase, solver._actual_calls, solver._normal_tools = phase, 0, 0
        solver._current_workspace = None
        solver._invalidate_computation("Convention verification requested a repaired final computation")
        solver._append_message({"role": "user", "content": repair_message(issues)})
        candidate = None
        for _ in range(REPAIR_MODEL_CALLS + 2):
            response, error, _usage = solver._llm_call(solver.model_messages, tools=solver.tools)
            if error:
                return None, {"status": "failed", "reason": error, "triggered": sorted(triggered)}
            if isinstance(response, HostBoundaryMessage):
                candidate = solver._computed_candidate(workspace)
                break
            solver._append_message(response)
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    if not solver._take_tool_call():
                        solver._append_message(solver._tool_budget_message(tool_call))
                        continue
                    message = solver._handle_tool_call(tool_call, workspace)
                    if message is None:
                        message = {"tool_call_id": tool_call.id, "role": "tool", "name": tool_call.function.name,
                                   "content": f"Unknown tool '{tool_call.function.name}'"}
                    solver._append_message(message)
                continue
            candidate = solver._computed_candidate(workspace)
            if candidate is not None:
                break
            if solver._is_final(response.content):
                # prose finals are not executed computations; the repair needs mode=final
                answer, _evidence, _limitation = parse_final_message(response.content)
                return None, {"status": "rejected", "reason": "repair ended without an executed final computation",
                              "prose_answer_present": answer is not None, "triggered": sorted(triggered)}
        if candidate is None or candidate.get("error"):
            return None, {"status": "failed", "reason": (candidate or {}).get("error") or "no repaired final computation",
                          "triggered": sorted(triggered)}
        source_issues = solver._automatic_check((candidate["answer"], candidate["evidence"], candidate["limitation"]), 3)
        remaining = convention_issues(context["question"], candidate, workspace, solver.complete_types())
        still = sorted({issue["check"] for issue in remaining} & triggered)
        report = {"status": "released", "triggered": sorted(triggered), "source_issues": source_issues,
                  "remaining_convention_issues": remaining, "candidate": candidate}
        if source_issues:
            return None, {**report, "status": "rejected", "reason": "repaired final is not source-valid"}
        if still:
            return None, {**report, "status": "rejected", "reason": f"triggering checks still fire: {', '.join(still)}"}
        if candidate["answer"] is None:
            return None, {**report, "status": "rejected", "reason": "a repair may not replace a valid draft with null"}
        return candidate, report
