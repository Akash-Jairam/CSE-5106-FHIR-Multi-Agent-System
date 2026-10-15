"""Compose the actual SingleAgent policy, then observe its released result.

The primary has thirty natural model turns and an inclusive tool budget. This
module does not intercept its loop, prompts, retrieval, or source checker. An
unbound result remains the released primary; only auxiliary eligibility changes.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math

from .analysis_agent import AnalysisAgent
from .single_agent import SingleAgent

PRIMARY_TOKENS = 175_000
PRIMARY_TOOLS = 27
PRIMARY_ITERATIONS = 30
FIELDS = ("answer", "evidence", "operation", "contributing_ids", "limitation")
RENDERED = ("answer", "evidence", "limitation")
COUNTERS = ("prompt_tokens", "completion_tokens", "total_tokens", "llm_calls")


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _integer(value):
    return type(value) is int and value >= 0


def _tagged(value):
    """Quarantine malformed accounting without unencodable NaN/custom objects."""
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite_float": str(value)}
    if isinstance(value, (list, tuple)):
        return [_tagged(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _tagged(item) for key, item in value.items()}
    return {"unsupported_type": type(value).__module__ + "." + type(value).__name__}


@dataclass(frozen=True)
class ContinuousPrimaryResult:
    """Defensive views of one completed primary and its frozen audit snapshot."""
    _released: dict
    _artifact: object
    _workspace: dict
    _diagnostics: dict
    _source_check: object

    @property
    def released(self):
        return deepcopy(self._released)

    @property
    def artifact(self):
        return deepcopy(self._artifact)

    @property
    def workspace(self):
        return deepcopy(self._workspace)

    @property
    def diagnostics(self):
        return deepcopy(self._diagnostics)

    @property
    def source_check(self):
        return deepcopy(self._source_check)


def _clean_host(host):
    totals = host.total_usage
    if (getattr(host, "_continuous_primary_adopted_ledger", None) is totals
            or not isinstance(totals, dict)
            or any(not _integer(totals.get(key)) or totals[key] != 0 for key in COUNTERS)
            or type(totals.get("cost")) not in (int, float) or totals["cost"] != 0
            or host.call_usage != [] or type(host.tool_calls_made) is not int or host.tool_calls_made != 0
            or host.messages != [] or host.budget_stop is not None or host.last_llm_error is not None):
        raise ValueError("Continuous primary requires a clean, not-yet-adopted host ledger")


def _adopt(host, primary):
    """Adopt once; malformed counters are retained tagged and globally fatal."""
    _clean_host(host)
    totals, calls, tools = deepcopy(primary.total_usage), deepcopy(primary.call_usage), primary.tool_calls_made
    problems = []
    if not isinstance(totals, dict) or any(not _integer(totals.get(key)) for key in COUNTERS):
        problems.append("malformed primary aggregate counters")
    cost = totals.get("cost") if isinstance(totals, dict) else None
    if cost is not None and (type(cost) not in (int, float)
                             or isinstance(cost, float) and not math.isfinite(cost) or cost < 0):
        problems.append("malformed primary cost")
    valid_calls = isinstance(calls, list) and all(isinstance(call, dict)
        and all(_integer(call.get(key)) for key in ("prompt", "completion"))
        and ("cached" not in call or _integer(call["cached"]))
        and type(call.get("usage_available")) is bool for call in calls)
    if not valid_calls:
        problems.append("malformed primary per-call usage")
    if not _integer(tools):
        problems.append("malformed primary tool counter")
    try:
        json.dumps({"total_usage": totals, "call_usage": calls}, allow_nan=False)
    except (TypeError, ValueError):
        problems.append("nonfinite or non-JSON primary usage")
    if not problems:
        if (totals["prompt_tokens"] + totals["completion_tokens"] != totals["total_tokens"]
                or totals["llm_calls"] != len(calls)
                or sum(call["prompt"] for call in calls) != totals["prompt_tokens"]
                or sum(call["completion"] for call in calls) != totals["completion_tokens"]):
            problems.append("unreconciled primary aggregate/per-call usage")
        if any(call["usage_available"] is False for call in calls) and cost is not None:
            problems.append("unknown primary usage has a falsely known aggregate cost")
    raw = {"total_usage": _tagged(totals), "call_usage": _tagged(calls), "tool_calls_made": _tagged(tools)}
    if problems:
        safe_totals = {key: totals.get(key) if isinstance(totals, dict) and _integer(totals.get(key)) else 0
                       for key in COUNTERS}
        safe_totals["cost"] = None
        safe_calls = []
        for call in calls if isinstance(calls, list) else []:
            call = call if isinstance(call, dict) else {}
            safe_calls.append({"prompt": call.get("prompt") if _integer(call.get("prompt")) else 0,
                               "completion": call.get("completion") if _integer(call.get("completion")) else 0,
                               "cached": call.get("cached") if _integer(call.get("cached")) else 0,
                               "usage_available": False, "accounting_quarantined": True})
        if not safe_calls:
            # An unknown marker is not a claim of a zero-token/free SDK send.
            safe_calls = [{"prompt": 0, "completion": 0, "cached": 0,
                           "usage_available": False, "accounting_quarantined": True}]
        host.total_usage.update(safe_totals)
        host.call_usage = safe_calls
        host.tool_calls_made = tools if _integer(tools) else 0
    else:
        host.total_usage.clear()
        host.total_usage.update(totals)
        host.call_usage = calls
        host.tool_calls_made = tools
    host._continuous_primary_adopted_ledger = host.total_usage
    host.context_compactions.extend(deepcopy(primary.context_compactions))
    unknown = bool(problems) or any(call.get("usage_available") is not True for call in host.call_usage)
    return raw, problems, unknown


def _binding(released, artifact, workspace, trace):
    if released.get("error"):
        return None, "primary returned an error", {}
    if not isinstance(artifact, dict) or artifact.get("mode") != "final":
        return None, "no surviving explicit final artifact", {}
    try:
        if artifact.get("workspace_sha256") != SingleAgent._workspace_sha256(workspace):
            return None, "final workspace differs from execution", {}
        code = artifact.get("code")
        if not isinstance(code, str) or not code.strip():
            return None, "executed final program is missing", {}
        candidate = AnalysisAgent._draft(json.dumps({key: artifact.get(key) for key in FIELDS}, allow_nan=False), workspace)
        if candidate.get("error"):
            return None, "executed final schema is invalid", {}
        rendered = {key: candidate[key] for key in RENDERED}
        revision = released.get("revision") or {}
        if (not isinstance(revision, dict) or revision.get("final_check_issues") != []
                or not isinstance(revision.get("final"), dict)
                or not _same(rendered, {key: released.get(key) for key in RENDERED})
                or not _same(rendered, {key: revision["final"].get(key) for key in RENDERED})):
            return None, "released answer does not bind a clean executed final", {}
        requests, executions, checks = {}, [], []
        for index, message in enumerate(trace):
            if _get(message, "role") == "assistant":
                for call in _get(message, "tool_calls", []) or []:
                    if _get(call, "id") in requests:
                        return None, "ambiguous repeated primary tool-call identity", {}
                    function = _get(call, "function")
                    try:
                        arguments = json.loads(_get(function, "arguments", ""))
                    except (TypeError, ValueError):
                        arguments = None
                    requests[_get(call, "id")] = (_get(function, "name"), arguments, index)
            if _get(message, "role") != "tool":
                continue
            identifier = _get(message, "tool_call_id")
            request = requests.get(identifier)
            try:
                output = json.loads(_get(message, "content", ""))
            except (TypeError, ValueError):
                continue
            if not request or not isinstance(request[1], dict) or not isinstance(output, dict):
                continue
            name, args, _ = request
            if (name == "execute_python_code" and args.get("mode", "auto") == "final"
                    and args.get("code") == code and output.get("mode") == "final" and not output.get("error")):
                observed = AnalysisAgent._draft(json.dumps({key: output.get(key) for key in FIELDS}, allow_nan=False), workspace)
                if _same(observed, candidate):
                    executions.append((index, identifier))
            if name == "check_fields" and identifier in ("automatic_field_check_1", "automatic_field_check_2"):
                checks.append((index, identifier, args, output))
        if not executions or not checks:
            return None, "actual final execution or automatic source-check trace is missing", {}
        execution_index, execution_id = executions[-1]
        check_index, check_id, args, check = checks[-1]
        results = check.get("results")
        if (check_index <= execution_index or not _same(args.get("citations"), candidate["evidence"])
                or check.get("issues") != [] or not isinstance(results, list)
                or len(results) != len(candidate["evidence"])
                or type(check.get("n_valid")) is not int or check["n_valid"] != len(results)
                or any(not isinstance(item, dict) or item.get("status") != "valid" or item.get("warnings") != []
                       or not _same(item.get("citation"), citation)
                       for citation, item in zip(candidate["evidence"], results))):
            return None, "automatic source-check result does not bind the released artifact", {}
        # A cap-refused mutation never reaches Single._handle_tool_call's invalidation.
        for message in trace[execution_index + 1:]:
            if _get(message, "role") == "tool" and str(_get(message, "content", "")).startswith("Not run: the tool-call budget"):
                name, args, _ = requests.get(_get(message, "tool_call_id"), (None, None, None))
                if name in ("get_resources_by_patient_fhir_id", "get_resources_by_resource_id") or (
                        name == "execute_python_code" and (not isinstance(args, dict) or args.get("mode", "auto") != "inspect")):
                    return None, "a later requested mutation was refused before dispatch", {}
        result = deepcopy({**artifact, **candidate})
        return result, None, {"execution_tool_call_id": execution_id, "execution_trace_index": execution_index,
                              "source_check_tool_call_id": check_id, "source_check_trace_index": check_index,
                              "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                              "workspace_sha256": artifact["workspace_sha256"], "source_check": deepcopy(check)}
    except (TypeError, ValueError, KeyError, AttributeError):
        return None, "final execution/source binding is malformed or nonfinite", {}


def run_continuous_primary(host, question):
    """Run the actual Single to completion, adopting its one phase exactly once.

    Global lower caps reduce the local caps; None/higher caps never expand this
    phase. Local budget stops are diagnostics, not a global host budget stop.
    """
    _clean_host(host)
    token_cap = min(PRIMARY_TOKENS, host.max_total_tokens) if host.max_total_tokens is not None else PRIMARY_TOKENS
    tool_cap = min(PRIMARY_TOOLS, host.max_tool_calls) if host.max_tool_calls is not None else PRIMARY_TOOLS
    primary = SingleAgent(host.model, max_iterations=PRIMARY_ITERATIONS, verbose=host.verbose, base_url=host.base_url)
    primary.set_budget(max_total_tokens=token_cap, max_tool_calls=tool_cap)
    primary.set_perturbation(host.perturbation)
    exception = None
    try:
        released = primary.run(question)
    except Exception as exc:
        exception = type(exc).__name__
        # No common-contract A exists if its formatter threw. Do not recover a
        # computed-but-unreleased candidate as though it were the released A.
        released = {"answer": None, "evidence": [], "limitation": None,
                    "error": "Primary output formatting failed: " + exception,
                    "trace": deepcopy(primary.messages), "retrieved_fhir_resources": {}}
    raw, problems, unknown = _adopt(host, primary)
    trace = deepcopy(released.get("trace") or primary.messages)
    host.messages.extend(trace)
    workspace = deepcopy(released.get("retrieved_fhir_resources") or {})
    fatal_reasons = list(problems)
    if unknown:
        fatal_reasons.append("primary admitted usage is unknown")
    if exception:
        fatal_reasons.append("primary common-output formatter failed")
    if host.total_usage["total_tokens"] > token_cap:
        fatal_reasons.append("primary reported token ceiling overshoot")
    if host.tool_calls_made > tool_cap:
        fatal_reasons.append("primary reported tool ceiling overshoot")
    if host.total_usage["llm_calls"] > PRIMARY_ITERATIONS:
        fatal_reasons.append("primary reported model-call ceiling overshoot")
    artifact, reason, binding = _binding(released, deepcopy(primary.final_computation), workspace, trace)
    if fatal_reasons:
        artifact, reason = None, "primary accounting is globally fatal"
    source_check = binding.pop("source_check", None) if artifact else None
    diagnostics = {"phase": "primary", "nominal_token_cap": PRIMARY_TOKENS, "nominal_tool_cap": PRIMARY_TOOLS,
                   "token_cap": token_cap, "tool_cap": tool_cap, "max_iterations": PRIMARY_ITERATIONS,
                   "tokens": host.total_usage["total_tokens"], "tool_calls": host.tool_calls_made,
                   "model_call_attempts_recorded": host.total_usage["llm_calls"],
                   "fatal": bool(fatal_reasons), "errors": fatal_reasons, "usage_known": not unknown,
                   "raw_accounting": raw, "primary_budget_stop": primary.budget_stop,
                   "primary_budget": deepcopy(released.get("budget")), "released_available": exception is None,
                   "eligible": artifact is not None, "skip_reason": reason, "binding": binding if artifact else {}}
    frozen_release = deepcopy(released)
    # Internal views share one snapshot; callers receive independent copies.
    frozen_workspace = frozen_release.get("retrieved_fhir_resources") or {}
    return ContinuousPrimaryResult(frozen_release, artifact, frozen_workspace, diagnostics, source_check)
