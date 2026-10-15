"""V7: full natural Single A, fresh residual Single B, fallible existing-choice selector."""
from copy import deepcopy
import json
import re

from .base_agent import BaseAgent
from .ehr_semantics import SEMANTIC_GUIDANCE
from .controller import FixedController
from .residual_trajectory_phase import (fingerprint, json_text, message_dict, reserve,
    run_single_phase, same, selector_call, strict_object)

PROTOCOL = "full-single-residual-trajectories-v7"
ENTRY_BYTES = 32768
BOUNDARY_TOKENS = 512
SELECTOR_COMPLETION = 2000
FIELDS = ("answer", "evidence", "operation", "contributing_ids", "limitation")
RENDERED = ("answer", "evidence", "limitation")
SELECTOR_PROMPT = """Compare two EXISTING released candidates for the original question/context.
A is the full natural Single result. B is a fresh residual-capped natural Single
with its OWN original starting packet, retrieval, program execution and original
source checks. Shared store/model/cache can correlate errors; neither retrieval
metadata nor source-field validity proves complete search or correct semantics.
A may be explicitly unverified, prose-only, limited or erroneous. Never replace
it by a computed-but-unreleased draft. Read each actual released object, program,
source check, limitations and successful retrieval metadata in its own namespace.

First derive the material requirements from the exact question/context (scope,
operands, chronology, concepts, units and requested output as applicable). For
EACH requirement compare actual candidate evidence/program behavior. Use supported,
not_found or contradicted. Each basis must name concrete field/program/retrieval
facts available in that candidate entry; absence of a claim is not contradiction.
Successful retrieval request/count replies are not proof of exhaustive retrieval;
physical/cache/page metadata may be unavailable in the unchanged Single trace.
Do not invent context, records, expected answers or hidden tests. An empty result
can be supported by an actual completed search; a limitation remains visible and
must not be deleted to manufacture support. Source checks are not semantic proof.
Select B only if ALL its requirements are supported AND at least one A requirement
is concretely contradicted. Otherwise preserve A or return UNKNOWN. A/B labels
are fallible model judgments, not machine-verified witnesses. Never choose B just
for being nonempty, longer, different, or source-checked while A is unverified.
No new answer, code, computation, retrieval, tools or evidence merge is allowed.
Return one finite JSON object exactly:
{"requirements":[{"requirement":"question-derived obligation","A":{"status":"supported|not_found|contradicted","basis":"concrete facts"},"B":{"status":"supported|not_found|contradicted","basis":"concrete facts"}}],"choice":"A|B|UNKNOWN","reason":"brief concrete comparison"}.
""" + SEMANTIC_GUIDANCE


def rendered(snapshot):
    return {key: deepcopy(snapshot["released"].get(key)) for key in RENDERED}


def successful_retrievals(trace):
    """Actual successful request/reply metadata only, not a completeness certificate."""
    requests, result = {}, []
    names = {"get_resources_by_patient_fhir_id", "get_resources_by_resource_id"}
    for message in trace:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                if function.get("name") in names:
                    requests[call.get("id")] = function
        if message.get("role") != "tool" or message.get("name") not in names:
            continue
        function = requests.get(message.get("tool_call_id"))
        if not function:
            continue
        try:
            output = json.loads(message["content"])
            arguments = json.loads(function["arguments"])
        except (TypeError, ValueError, KeyError):
            try:
                arguments = json.loads(function["arguments"])
            except (TypeError, ValueError, KeyError):
                continue
            content = message.get("content")
            match = re.fullmatch(r"retrieved_resources\['([^']+)'\] \((\d+) items; (\d+) fetched\)", content or "")
            if not match or arguments.get("resource_type") != match[1]:
                continue
            output = {"recorded_success_reply": content, "held_count": int(match[2]), "fetched_count": int(match[3])}
        if not isinstance(output, dict) or output.get("error"):
            continue
        result.append({"tool_call_id": message.get("tool_call_id"), "tool": function["name"],
                       "arguments": arguments, "recorded_reply": deepcopy(output),
                       "physical_metadata_available": "retrieval_metadata" in output,
                       "assurance": "request_and_retained_reply_not_search_completeness"})
    return result


def candidate_entry(snapshot, trace):
    artifact = snapshot["artifact"]
    return {"released": {key: deepcopy(snapshot["released"].get(key)) for key in (*RENDERED, "error")},
        "verification": "released_execution_and_original_source_bound" if artifact else "released_unverified",
        "program": artifact.get("code") if artifact else None,
        "artifact": {key: deepcopy(artifact[key]) for key in FIELDS} if artifact else None,
        "source_check": deepcopy(snapshot["source_check"]),
        "successful_retrievals": successful_retrievals(trace),
        "workspace_sha256": snapshot["workspace_sha256"], "snapshot_sha256": fingerprint(snapshot)}


def selection_object(response):
    message = message_dict(response)
    if message.get("tool_calls"):
        raise ValueError("selector returned native tools")
    value = strict_object(message.get("content"), 16000)
    if set(value) != {"requirements", "choice", "reason"} or value["choice"] not in {"A", "B", "UNKNOWN"}:
        raise ValueError("selector choice schema differs")
    if not isinstance(value["reason"], str) or not value["reason"].strip() or len(value["reason"]) > 2000:
        raise ValueError("selector reason is absent or oversized")
    requirements = value["requirements"]
    if not isinstance(requirements, list) or not 1 <= len(requirements) <= 32:
        raise ValueError("selector requirements absent or oversized")
    for item in requirements:
        if (not isinstance(item, dict) or set(item) != {"requirement", "A", "B"}
                or not isinstance(item["requirement"], str) or not item["requirement"].strip()
                or len(item["requirement"]) > 1000):
            raise ValueError("selector requirement schema differs")
        for label in ("A", "B"):
            fact = item[label]
            if (not isinstance(fact, dict) or set(fact) != {"status", "basis"}
                    or fact["status"] not in {"supported", "not_found", "contradicted"}
                    or not isinstance(fact["basis"], str) or not fact["basis"].strip() or len(fact["basis"]) > 2000):
                raise ValueError("selector requirement basis/status differs")
    return value


class ResidualTrajectoryAgent(BaseAgent):
    """Both trajectories plus selector share actual200k/30tools/30SDK attempts."""
    def _aggregate_error(self):
        if (self.total_usage["total_tokens"] > self.max_total_tokens
                or self.tool_calls_made > self.max_tool_calls or self.total_usage["llm_calls"] > 30):
            return "actual aggregate ceiling exceeded"
        return None

    def _release(self, snapshots, phases, chosen, status, selection=None, admission=None, fatal=None):
        snapshot = snapshots[chosen]
        released = snapshot["released"]
        self.last_llm_error = None
        # Local phase refusals must not rewrite the actual released A at outer normalization.
        self.budget_stop = None
        index = {key: fingerprint(value) for key, value in snapshots.items()}
        revision = {"protocol": PROTOCOL, "status": status, "selected": chosen,
            "A_only": {"available": True, "snapshot_sha256": index["A"],
                       "error": snapshots["A"]["released"].get("error"), "source_bound": snapshots["A"]["artifact"] is not None},
            "candidate_snapshot_sha256": index, "phases": deepcopy(phases),
            "selection": deepcopy(selection), "admission": deepcopy(admission),
            "messages": [deepcopy(value) for value in snapshots.values()],
            "final_check_issues": [] if snapshot["artifact"] is not None and not fatal else None,
            "independent_candidate_selected": chosen == "B", "fatal": bool(fatal)}
        if fatal:
            return {"answer": None, "evidence": [], "limitation": fatal,
                "error": "V7 execution/accounting failure: " + fatal, "trace": self.messages,
                "retrieved_fhir_resources": deepcopy(snapshot["workspace"]), "revision": revision}
        return {**{key: deepcopy(released.get(key)) for key in RENDERED}, "error": released.get("error"),
                "trace": self.messages, "retrieved_fhir_resources": deepcopy(snapshot["workspace"]), "revision": revision}

    def _run(self, question):
        self.set_budget(max_total_tokens=min(200000, self.max_total_tokens) if self.max_total_tokens is not None else 200000,
                        max_tool_calls=min(30, self.max_tool_calls) if self.max_tool_calls is not None else 30)
        self._v7_merged_phases = set()
        snapshots, phases = {}, []
        a, phase = run_single_phase(self, question, "natural_primary", self.max_total_tokens, self.max_tool_calls, 30)
        snapshots["A"], phases = a, [phase]
        frozen_A = fingerprint(a)
        if phase["fatal"] or self._aggregate_error():
            return self._release(snapshots, phases, "A", "fatal_primary", fatal=self._aggregate_error() or "primary usage unverifiable")
        try:
            context = FixedController.question_context(question)
            trace_A = [message for message in self.messages if message.get("phase") == "natural_primary"]
            entry_A = candidate_entry(a, trace_A)
            if len(json_text(entry_A).encode()) > ENTRY_BYTES:
                return self._release(snapshots, phases, "A", "A_entry_coverage_skip")
            base_packet = {"context": context, "A": entry_A, "B": None}
            base_messages = [{"role": "system", "content": SELECTOR_PROMPT}, {"role": "user", "content": json_text(base_packet)}]
            fixed_tokens = base_module_count(base_messages)
            selector_bound = fixed_tokens + ENTRY_BYTES + BOUNDARY_TOKENS
            selector_reservation = reserve(selector_bound) + SELECTOR_COMPLETION
            remaining_tokens = self.max_total_tokens - self.total_usage["total_tokens"]
            remaining_tools = self.max_tool_calls - self.tool_calls_made
            remaining_sdk = 30 - self.total_usage["llm_calls"]
            admission = {"A_entry_bytes": len(json_text(entry_A).encode()), "future_B_entry_byte_bound": ENTRY_BYTES,
                "selector_fixed_prompt_tokens": fixed_tokens, "selector_prompt_bound": selector_bound,
                "selector_token_reservation": selector_reservation, "residual_tokens": remaining_tokens,
                "residual_tools": remaining_tools, "residual_sdk_attempts": remaining_sdk,
                "B_token_cap": remaining_tokens - selector_reservation,
                "B_natural_tool_cap": remaining_tools - 1, "B_sdk_cap": min(30, remaining_sdk - 1)}
            if remaining_tokens <= selector_reservation or remaining_tools < 3 or remaining_sdk < 3:
                return self._release(snapshots, phases, "A", "residual_coverage_skip", admission=admission)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            return self._release(snapshots, phases, "A", "A_packet_unavailable", admission={"exception": type(exc).__name__})
        b, phase = run_single_phase(self, question, "natural_alternative", admission["B_token_cap"],
            admission["B_natural_tool_cap"], admission["B_sdk_cap"], initialization_charge=True)
        snapshots["B"] = b
        phases.append(phase)
        if fingerprint(a) != frozen_A:
            return self._release(snapshots, phases, "A", "fatal_mutation", admission=admission, fatal="frozen A mutated")
        if phase["fatal"] or self._aggregate_error():
            return self._release(snapshots, phases, "A", "fatal_alternative", admission=admission,
                                 fatal=self._aggregate_error() or "alternative usage unverifiable")
        if b["artifact"] is None:
            return self._release(snapshots, phases, "A", "B_ineligible", admission=admission)
        if same(a["released"].get("answer"), b["released"].get("answer")):
            return self._release(snapshots, phases, "A", "agreement_preserved", admission=admission)
        try:
            trace_B = [message for message in self.messages if message.get("phase") == "natural_alternative"]
            entry_B = candidate_entry(b, trace_B)
            if len(json_text(entry_B).encode()) > ENTRY_BYTES:
                return self._release(snapshots, phases, "A", "B_entry_coverage_skip", admission=admission)
            packet = {**base_packet, "B": entry_B}
            messages = [{"role": "system", "content": SELECTOR_PROMPT}, {"role": "user", "content": json_text(packet)}]
            actual_prompt = base_module_count(messages)
            admission.update(B_entry_bytes=len(json_text(entry_B).encode()), selector_prompt_tokens=actual_prompt,
                             selector_packet_sha256=fingerprint(packet))
            if actual_prompt > selector_bound:
                return self._release(snapshots, phases, "A", "selector_coverage_skip", admission=admission)
            self.messages.extend([{**message, "phase": "selection"} for message in messages])
            response, phase = selector_call(self, messages, selector_reservation)
            phases.append(phase)
            if phase["fatal"] or self._aggregate_error():
                return self._release(snapshots, phases, "A", "fatal_selection", admission=admission,
                    fatal=self._aggregate_error() or "selector usage unverifiable")
            if response is None:
                return self._release(snapshots, phases, "A", "selector_unavailable", admission=admission)
            self.messages.append({**message_dict(response), "phase": "selection"})
            selection = selection_object(response)
            allow_B = (selection["choice"] == "B" and all(r["B"]["status"] == "supported" for r in selection["requirements"])
                       and any(r["A"]["status"] == "contradicted" for r in selection["requirements"]))
            if fingerprint(a) != frozen_A:
                return self._release(snapshots, phases, "A", "fatal_mutation", admission=admission, fatal="frozen A mutated")
            return self._release(snapshots, phases, "B" if allow_B else "A",
                                 "selected_B" if allow_B else "selector_preserved_A", selection, admission)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            return self._release(snapshots, phases, "A", "selector_malformed_or_unavailable", admission={**admission, "exception": type(exc).__name__})


def base_module_count(messages):
    from . import base_agent
    return base_agent.count_tokens_in_messages(messages)


def run_v7_job(args_tuple):
    from utils import run_agent_safe
    import tools.cache as cache_module
    input_data, strategy, model, verbose, base_url, enable_cache, *rest = args_tuple
    if strategy != "multi_agent_v7":
        raise ValueError("V7 worker received a different strategy")
    cache_module.CACHE_ENABLED = enable_cache
    agent = ResidualTrajectoryAgent(model, verbose=verbose, base_url=base_url)
    agent.set_budget(**((rest[0] if rest else None) or {}))
    agent.set_perturbation(rest[1] if len(rest) > 1 else None)
    include_full_audit = bool(rest[2]) if len(rest) > 2 else False
    audit_context = rest[3] if len(rest) > 3 else None
    if include_full_audit:
        output = run_agent_safe(agent, input_data, include_full_audit=True)
        if audit_context is not None:
            from scripts.audit_store import externalize_result_audit
            output = externalize_result_audit(output, audit_context["output_dir"], question_id=audit_context["question_id"])
        return output
    return run_agent_safe(agent, input_data)
