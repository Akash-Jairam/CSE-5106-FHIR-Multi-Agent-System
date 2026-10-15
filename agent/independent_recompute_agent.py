"""A continuous SingleAgent primary plus bounded independent recomputation.

This is an experimental selection method, not a semantic correctness oracle.
The primary keeps its natural solving loop; the optional second computation
uses the same frozen evidence. Every admitted model/tool operation is charged.
"""
from copy import deepcopy
import hashlib
import json

from .base_agent import BaseAgent
from .continuous_primary import run_continuous_primary
from .controller import FixedController
from .independent_recompute_aux import run_independent_recompute


PROTOCOL = "continuous-primary-independent-recompute-v6"
FINAL_FIELDS = ("answer", "evidence", "operation", "contributing_ids", "limitation")


def _same(left, right):
    """Exact finite JSON equality, with bool/number and missing/null distinct."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


def _rendered(candidate):
    return {key: deepcopy(candidate.get(key)) for key in ("answer", "evidence", "limitation")}


class IndependentRecomputeAgent(BaseAgent):
    """Budget-reallocated actual Single policy, then one independent candidate.

    Primary: 175k tokens, 27 inclusive tools, 30 natural model iterations.
    Auxiliary: 25k tokens, two explicit model calls and at most three tools
    (the implemented simple path uses two). The potential model-call maximum
    is 32, versus the contemporary SingleAgent's 30, under the same aggregate
    200k-token/30-tool ceilings. No shared source-check rule is relaxed.
    """

    def _aggregate_error(self):
        tokens = self.total_usage.get("total_tokens")
        tools = self.tool_calls_made
        if type(tokens) is not int or tokens < 0 or type(tools) is not int or tools < 0:
            return "Aggregate accounting counters are malformed"
        token_cap = 200_000 if self.max_total_tokens is None else min(200_000, self.max_total_tokens)
        tool_cap = 30 if self.max_tool_calls is None else min(30, self.max_tool_calls)
        if tokens > token_cap or tools > tool_cap:
            return "Reported aggregate usage exceeded the declared ceiling"
        return None

    def _failure(self, common, workspace, reason):
        final = {"answer": None, "evidence": [], "limitation": reason}
        # Do not use the ordinary budget-refusal marker: BaseAgent correctly
        # treats known pre-call refusals differently from accounting failures.
        self.last_llm_error = None
        return {**final, "error": "V6 execution/accounting failure: " + reason,
                "trace": self.messages, "retrieved_fhir_resources": workspace,
                "revision": {**common, "final": final,
                             "release_policy": "fatal_execution_or_accounting_failure"}}

    def _run(self, question):
        context = FixedController.question_context(question)
        primary = run_continuous_primary(self, question)
        released = deepcopy(primary.released)
        previous = deepcopy(released.get("revision") or {})
        workspace = deepcopy(primary.workspace)
        common = {
            "protocol": PROTOCOL,
            "primary": _rendered(released),
            "primary_artifact": deepcopy(primary.artifact),
            "primary_revision": previous,
            "primary_diagnostics": deepcopy(primary.diagnostics),
            "draft": deepcopy(previous.get("draft")),
            "phases": [deepcopy(primary.diagnostics)],
            "primary_source_revised": bool(previous.get("revised")),
            "independent_candidate_selected": False,
            "revised": bool(previous.get("revised")),
            "answer_changed": previous.get("answer_changed", False),
            "context_compactions": deepcopy(self.context_compactions),
        }
        problem = self._aggregate_error()
        if primary.diagnostics.get("fatal") or problem:
            return self._failure(common, workspace, problem or "Primary accounting could not be verified")

        if primary.artifact is None:
            # A prose-only or otherwise ineligible primary is still the actual
            # Single output. Ineligibility disables the extra stage; it never
            # synthesizes a program or imposes an additional answer veto.
            self.last_llm_error = None
            return {**_rendered(released), "error": released.get("error"),
                    "trace": self.messages, "retrieved_fhir_resources": workspace,
                    "revision": {**common, "final": _rendered(released),
                                 "final_check_issues": previous.get("final_check_issues"),
                                 "release_policy": "continuous_primary_preserved_auxiliary_ineligible"}}

        candidate_a = {key: deepcopy(primary.artifact[key]) for key in FINAL_FIELDS}
        frozen = hashlib.sha256(json.dumps(workspace, sort_keys=True, default=str,
                                          allow_nan=False).encode()).hexdigest()
        try:
            chosen, diagnostic = run_independent_recompute(
                self, candidate_a, primary.artifact, workspace, context, primary.source_check)
        except Exception as exc:
            # An unexpected error cannot attest to whether a provider invocation
            # was admitted. Expected known failures are handled inside the extra
            # stage and preserve A; unexpected ones remain explicit failed rows.
            common["recompute"] = {"status": "unexpected_runtime_failure",
                                    "exception_type": type(exc).__name__, "fatal": True}
            return self._failure(common, workspace, "Independent stage failed unexpectedly")
        common["recompute"] = deepcopy(diagnostic)
        common["phases"].append(deepcopy(diagnostic.get("phase") or diagnostic))
        problem = self._aggregate_error()
        if hashlib.sha256(json.dumps(workspace, sort_keys=True, default=str,
                                     allow_nan=False).encode()).hexdigest() != frozen:
            problem = "The frozen primary workspace changed"
        if diagnostic.get("fatal") or problem:
            return self._failure(common, workspace, problem or "Independent-stage accounting could not be verified")
        if not isinstance(chosen, dict) or set(chosen) != set(FINAL_FIELDS):
            return self._failure(common, workspace, "Independent stage returned an invalid candidate envelope")

        switched = not _same(chosen, candidate_a)
        self.last_llm_error = None
        draft = common["draft"]
        answer_changed = (not _same(chosen["answer"], draft["answer"])
                          if isinstance(draft, dict) and "answer" in draft else None)
        return {**_rendered(chosen), "trace": self.messages,
                "retrieved_fhir_resources": workspace,
                "revision": {**common, "final": deepcopy(chosen),
                             "revised": bool(previous.get("revised")) or switched,
                             "answer_changed": answer_changed,
                             "independent_candidate_selected": switched,
                             "final_check_issues": [],
                             "release_policy": ("selected_executed_independent_candidate" if switched
                                                else "continuous_primary_preserved")}}


def run_v6_job(args_tuple):
    """Common worker entry, including ownership-bound audit storage before IPC."""
    from utils import run_agent_safe
    import tools.cache as cache_module

    input_data, strategy, model, verbose, base_url, enable_cache, *rest = args_tuple
    if strategy != "multi_agent_v6":
        raise ValueError("V6 worker received a different strategy")
    cache_module.CACHE_ENABLED = enable_cache
    agent = IndependentRecomputeAgent(model, verbose=verbose, base_url=base_url)
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
