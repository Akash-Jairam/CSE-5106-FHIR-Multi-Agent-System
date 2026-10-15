"""Bounded, candidate-blind recomputation followed by fallible A/B selection.

This module never changes the primary policy. Its caller supplies an already
released, execution/source-bound A. All fallback paths preserve that A; an
accounting-fatal diagnostic requires the caller to withhold the whole attempt.
Source checks establish cited facts, not semantic correctness or completeness.
"""
from copy import deepcopy
import hashlib
import json
import math

from .analysis_agent import AnalysisAgent
from .ehr_semantics import SEMANTIC_GUIDANCE
from .single_agent import SingleAgent
from tools import get_tool_definitions
from tools.field_check import check_fields
from tools.python_executor import execute_python_code
from utils.core_utils import count_tokens_in_messages

PROTOCOL = "independent-frozen-recomputation-v6"
TOKEN_CAP, TOOL_CAP, CALL_CAP = 25_000, 3, 2
FIELDS = ("answer", "evidence", "operation", "contributing_ids", "limitation")
LIMITS = {"generation_completion": 4_000, "selection_completion": 1_800,
          "program_chars": 6_000, "candidate_entry_bytes": 8_000,
          "selector_boundary_tokens": 512, "selection_chars": 4_000}
ASSURANCE = "fallible_model_selection_of_executed_source_checked_candidates"

GENERATION_PROMPT = """Independently answer the ORIGINAL question from the frozen
retrieved_resources available to Python. No candidate answer, program, evidence,
operation or prior conversation is visible. The inventory contains complete type
counts/concept labels and one source example per held type, not all records.
The Python worker's retrieved_resources contains ALL held records in full,
including records beyond those displayed examples. Read and iterate that mapping
at runtime; the displayed sample is not the worker's data limit. Intermediate
field inspection, filtering, ordering and calculations can all happen inside
your one final program. Do not declare records unavailable solely because their
full contents are absent from the prompt's inventory preview.
Held-record access does not establish complete retrieval. There is no intermediate
tool feedback or second execution.
You have exactly one execute_python_code invocation, no retrieval or inspection
round, and no follow-up correction. Produce one self-contained final computation
on these held records. Missing records or ambiguous mapping require an explicit
limitation; they do not establish an empty result. Do not invent context facts.
Use the native execute_python_code tool exactly once, with mode="final".
Set answer to the requested finite JSON scalar/list/null; evidence to exact
{resource_id,field_path,value} source citations; contributing_ids to every used
full Type/id; operation to a nonempty string explaining the actual completed
search/filter/calculation, even for empty results; limitation to null if
established, otherwise a brief explanation. A fresh worker receives only copied
retrieved_resources. The host executes your code once, then checks original
source fields. Null, zero, false and empty lists are legitimate results.
Keep the complete program within 6,000 characters and metadata concise. There
is no answer key. Source validity does not prove correct question interpretation.
""" + SEMANTIC_GUIDANCE

SELECTION_PROMPT = """Choose between two existing executed candidates for the
ORIGINAL question and bound context. Both programs, actual five-field artifacts
and source-check summaries are intact. Source checks verify citations, not the
interpretation, calculation or complete search. Both used the SAME frozen held
workspace, so agreement and shared omissions are not independent evidence.
Read the actual programs; operation descriptions can be wrong. Select B only
when its actual computation is better supported by the question/context and held
evidence. Preserve A on uncertainty. Do not prefer nonempty, longer or unfamiliar
answers. Valid null/zero/false/empty results are not errors. You have no tools,
new records, hidden tests or gold answers. Do not emit a new answer, code, patch,
probe or third computation. Return exact finite JSON only:
{"choice":"A|B|UNKNOWN","reason":"brief concrete comparison"}.
This is a fallible model decision, not a proof of semantic correctness.
""" + SEMANTIC_GUIDANCE


def _finite(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _finite(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _finite(item)
        return
    raise ValueError("not strict finite JSON")


def _json(value):
    _finite(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def same(left, right):
    """Typed finite JSON equality, without scorer normalization or bool coercion."""
    return _json(left) == _json(right)


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _strict_object(content, limit):
    if not isinstance(content, str) or len(content) > limit:
        raise ValueError("response text bound exceeded")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(content, object_pairs_hook=pairs)
    _finite(value)
    if not isinstance(value, dict):
        raise ValueError("response is not an object")
    return value


def _summary_shape(count, digest):
    return {"issues": [], "n_valid": count, "citation_count": count,
            "all_results_valid": True, "all_warnings_empty": True, "raw_check_sha256": digest,
            "assurance": "source_fields_only_not_semantics_or_search_completeness"}


def _summary(candidate, check):
    """Bind every actual checker result; no synthetic clean-source certificate."""
    _finite(check)
    if (not isinstance(check, dict)
            or not set(check) <= {"issues", "results", "n_valid", "question_scopes", "scope_membership_only"}
            or check.get("issues") != [] or not isinstance(check.get("results"), list)
            or type(check.get("n_valid")) is not int
            or check["n_valid"] != len(candidate["evidence"])
            or len(check["results"]) != len(candidate["evidence"])):
        raise ValueError("source check is malformed, incomplete or has issues")
    for citation, result in zip(candidate["evidence"], check["results"]):
        if (not isinstance(result, dict) or result.get("status") != "valid"
                or result.get("warnings") != [] or not same(result.get("citation"), citation)):
            raise ValueError("source result does not bind a valid candidate citation")
    answer = candidate["answer"]
    established_empty = answer is None or answer is False or answer == [] or (
        type(answer) in (int, float) and answer == 0)
    if not candidate["evidence"] and not established_empty:
        raise ValueError("no evidence for nonempty answer")
    return _summary_shape(check["n_valid"], fingerprint(check))


def _inventory(workspace):
    result = {}
    for kind, rows in workspace.items():
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("workspace is not normalized")
        labels = sorted({str(item.get("display") or item.get("code") or "")
            for row in rows for item in (row.get("code") or {}).get("coding", [])})
        result[kind] = {"count": len(rows), "source_example": deepcopy(rows[:1]),
                        "concept_labels": labels}
    return result


def _messages(prompt, packet):
    return [{"role": "system", "content": prompt}, {"role": "user", "content": _json(packet)}]


def _entry(candidate, code, summary, workspace_sha):
    return {"program": code, "artifact": deepcopy(candidate), "source_check": deepcopy(summary),
            "program_sha256": hashlib.sha256(code.encode()).hexdigest(), "workspace_sha256": workspace_sha}


def _reserve(prompt):
    return prompt + max(64, math.ceil(prompt * .05))


class AuxPhase:
    """One host ledger, local caps; no retries, hidden inference or tool probes."""
    def __init__(self, host):
        self.host = host
        self.start_tokens = host.total_usage["total_tokens"]
        self.start_tools = host.tool_calls_made
        self.actual_calls, self.fatal, self.errors = 0, False, []

    def remaining(self):
        remaining = TOKEN_CAP - (self.host.total_usage["total_tokens"] - self.start_tokens)
        if self.host.max_total_tokens is not None:
            remaining = min(remaining, self.host.max_total_tokens - self.host.total_usage["total_tokens"])
        return remaining

    def take_tool(self):
        if self.host.tool_calls_made - self.start_tools >= TOOL_CAP:
            self.errors.append("auxiliary tool allowance exhausted")
            return False
        return self.host._take_tool_call()

    def report(self):
        return {"phase": "independent_recompute", "token_cap": TOKEN_CAP, "tool_cap": TOOL_CAP,
                "model_call_cap": CALL_CAP, "tokens": self.host.total_usage["total_tokens"] - self.start_tokens,
                "tool_calls": self.host.tool_calls_made - self.start_tools, "model_calls": self.actual_calls,
                "errors": list(self.errors), "fatal": self.fatal}

    def _unknown(self, previous, call_start):
        host = self.host
        self.fatal = True
        host.total_usage["cost"] = None
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "llm_calls"):
            value = host.total_usage.get(key)
            if type(value) is not int or value < previous[key]:
                host.total_usage[key] = previous[key]
        if len(host.call_usage) == call_start:
            host._update_usage({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "llm_calls": max(0, previous["llm_calls"] + 1 - host.total_usage["llm_calls"]),
                "cost": None, "per_call": [{"prompt": 0, "completion": 0, "usage_available": False}]})
        else:
            for index in range(call_start, len(host.call_usage)):
                raw = host.call_usage[index]
                raw = raw if isinstance(raw, dict) else {}
                item = {key: raw[key] if type(raw.get(key)) is int and raw[key] >= 0 else 0
                        for key in ("prompt", "completion")}
                if "cached" in raw:
                    item["cached"] = raw["cached"] if type(raw["cached"]) is int and raw["cached"] >= 0 else 0
                host.call_usage[index] = {**item, "usage_available": False, "malformed_usage": True}

    def call(self, messages, completion, tools=None):
        host = self.host
        prompt = count_tokens_in_messages(messages, tools)
        reserve = _reserve(prompt) + completion
        if self.actual_calls >= CALL_CAP or reserve > self.remaining():
            self.errors.append("known auxiliary call/token refusal")
            return None, "known auxiliary call/token refusal"
        previous = deepcopy(host.total_usage)
        call_start = len(host.call_usage)
        old_limit, old_stop, old_error = host.max_total_tokens, host.budget_stop, host.last_llm_error
        host.max_total_tokens = min(old_limit, previous["total_tokens"] + reserve) if old_limit is not None else previous["total_tokens"] + reserve
        self.actual_calls += 1
        try:
            response, error, usage = host._llm_call(messages, tools=tools)
        except Exception as exc:
            self._unknown(previous, call_start)
            self.errors.append("admitted model invocation raised " + type(exc).__name__)
            return None, "admitted model invocation failed with unknown usage"
        finally:
            host.max_total_tokens = old_limit
        calls = usage.get("per_call") if isinstance(usage, dict) else None
        known = (isinstance(calls, list) and len(calls) == 1
            and type(usage.get("llm_calls")) is int and usage["llm_calls"] == 1
            and all(type(usage.get(key)) is int and usage[key] >= 0
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
            and (usage.get("cost") is None or (type(usage.get("cost")) in (int, float)
                 and math.isfinite(usage["cost"]) and usage["cost"] >= 0))
            and ("cached_prompt_tokens" not in usage or (type(usage["cached_prompt_tokens"]) is int and usage["cached_prompt_tokens"] >= 0))
            and all(isinstance(item, dict) and set(item) <= {"prompt", "completion", "cached", "usage_available"}
                and item.get("usage_available") is True
                and all(type(item.get(key)) is int and item[key] >= 0 for key in ("prompt", "completion"))
                and ("cached" not in item or (type(item["cached"]) is int and item["cached"] >= 0)) for item in calls))
        if known:
            known = (usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
                and calls[0]["prompt"] == usage["prompt_tokens"] and calls[0]["completion"] == usage["completion_tokens"]
                and ("cached_prompt_tokens" not in usage or calls[0].get("cached", 0) == usage["cached_prompt_tokens"]))
        if not known:
            self._unknown(previous, call_start)
            self.errors.append("provider usage unavailable or malformed")
            return None, "provider usage unavailable or malformed"
        # Reconcile the actually adopted ledger, not merely the returned usage.
        try:
            cost = host.total_usage.get("cost")
            expected_cost = None if previous.get("cost") is None or usage.get("cost") is None else previous["cost"] + usage["cost"]
            cost_bound = (cost is None and expected_cost is None) or (
                type(cost) in (int, float) and math.isfinite(cost) and cost >= 0
                and type(expected_cost) in (int, float) and math.isfinite(expected_cost) and cost == expected_cost)
            ledger_bound = (len(host.call_usage) == call_start + 1 and same(host.call_usage[-1], calls[0])
                and cost_bound
                and all(type(host.total_usage.get(key)) is int
                        and host.total_usage[key] == previous[key] + usage[key]
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "llm_calls")))
        except (TypeError, ValueError):
            ledger_bound = False
        if not ledger_bound:
            self._unknown(previous, call_start)
            self.errors.append("provider usage does not bind adopted host ledger")
            return None, "provider usage does not bind adopted host ledger"
        if (usage["completion_tokens"] > completion or host.total_usage["total_tokens"] > previous["total_tokens"] + reserve
                or host.total_usage["total_tokens"] - self.start_tokens > TOKEN_CAP
                or (old_limit is not None and host.total_usage["total_tokens"] > old_limit)
                or (host.max_tool_calls is not None and host.tool_calls_made > host.max_tool_calls)):
            self.fatal = True
            self.errors.append("reported completion/phase/global ceiling overshoot")
            return None, "reported completion/phase/global ceiling overshoot"
        host.budget_stop, host.last_llm_error = old_stop, old_error
        if error:
            self.errors.append("provider returned an error with verified usage")
        return response, error


def _append_response(host, response, phase):
    raw = response if isinstance(response, dict) else response.model_dump(exclude_none=True)
    _finite(raw)
    host.messages.append({**deepcopy(raw), "phase": phase})


def _trace_tool(host, name, args, output, identifier, *, request=True):
    if request:
        host.messages.append({"role": "assistant", "content": None, "phase": "independent_recompute",
            "tool_calls": [{"id": identifier, "type": "function", "function": {"name": name, "arguments": _json(args)}}]})
    host.messages.append({"role": "tool", "name": name, "tool_call_id": identifier,
                          "content": _json(output), "phase": "independent_recompute"})


def run_independent_recompute(host, primary, artifact, workspace, context, primary_source_check,
                              *, executor=None, source_checker=None):
    """Return (existing A/B five-field candidate, diagnostics); never invent C.

    Host is the outer BaseAgent ledger after the authentic primary was adopted.
    Injection points only replace deterministic tools in offline tests. A fatal
    diagnostic is authoritative even though A is returned for audit preservation.
    """
    phase = AuxPhase(host)
    diagnostic = {"protocol": PROTOCOL, "status": "not_run", "fatal": False,
                  "assurance": ASSURANCE, "selected": "A", "switched": False,
                  "packet_admitted": False, "selector_admitted": False}
    original = deepcopy(primary)
    executor = execute_python_code if executor is None else executor
    source_checker = check_fields if source_checker is None else source_checker
    try:
        frozen = deepcopy(workspace)
        workspace_sha = SingleAgent._workspace_sha256(frozen)
        primary_sha, artifact_sha, context_sha = fingerprint(primary), fingerprint(artifact), fingerprint(context)
        diagnostic.update(primary_sha256=primary_sha, primary_artifact_sha256=artifact_sha,
                          workspace_sha256=workspace_sha, context_sha256=context_sha)
        def intact():
            if (fingerprint(primary) != primary_sha or fingerprint(artifact) != artifact_sha
                    or fingerprint(context) != context_sha or SingleAgent._workspace_sha256(workspace) != workspace_sha
                    or SingleAgent._workspace_sha256(frozen) != workspace_sha):
                raise ValueError("frozen primary/context/workspace changed")
        if (not isinstance(artifact, dict) or artifact.get("mode") != "final"
                or artifact.get("workspace_sha256") != workspace_sha
                or not isinstance(artifact.get("code"), str) or not artifact["code"].strip()
                or not same(primary, {key: artifact.get(key) for key in FIELDS})):
            raise ValueError("primary does not bind a final artifact/workspace")
        parsed = AnalysisAgent._draft(_json(primary), frozen)
        if parsed.get("error") or not same(parsed, primary) or primary.get("limitation"):
            raise ValueError("primary schema/limitation is not auxiliary eligible")
        a_summary = _summary(primary, primary_source_check)
        a = _entry(primary, artifact["code"], a_summary, workspace_sha)
        blind_packet = {"context": deepcopy(context), "workspace_sha256": workspace_sha,
                        "held_inventory": _inventory(frozen), "retrieval_allowed": False}
        generation = _messages(GENERATION_PROMPT, blind_packet)
        native = [deepcopy(item) for item in get_tool_definitions() if item["function"]["name"] == "execute_python_code"]
        if len(native) != 1:
            raise ValueError("native Python tool definition unavailable")
        selector_base = {"context": deepcopy(context), "workspace_sha256": workspace_sha, "A": a, "B": None}
        first_tokens = count_tokens_in_messages(generation, native)
        second_base_tokens = count_tokens_in_messages(_messages(SELECTION_PROMPT, selector_base))
        # B is ASCII JSON with a byte bound. Each UTF-8 byte contributes at most
        # one cl100k token; an explicit boundary/framing allowance is added.
        second_bound = second_base_tokens + LIMITS["candidate_entry_bytes"] + LIMITS["selector_boundary_tokens"]
        reservation = _reserve(first_tokens) + LIMITS["generation_completion"] + _reserve(second_bound) + LIMITS["selection_completion"]
        diagnostic.update(generation_packet_sha256=fingerprint(blind_packet), generation_prompt_tokens=first_tokens,
                          selector_fixed_prompt_tokens=second_base_tokens, selector_prompt_reserved_bound=second_bound,
                          two_call_token_reservation=reservation, limits=deepcopy(LIMITS))
        available_tools = TOOL_CAP
        if host.max_tool_calls is not None:
            available_tools = min(available_tools, host.max_tool_calls - host.tool_calls_made)
        if reservation > phase.remaining() or available_tools < 2:
            diagnostic.update(status="coverage_skip", reason="intact two-call packet or two counted tools unavailable")
            return original, diagnostic
        intact()
        diagnostic["packet_admitted"] = True
        host.messages.extend([{**deepcopy(message), "phase": "independent_generation"} for message in generation])
        response, error = phase.call(generation, LIMITS["generation_completion"], native)
        if error or response is None:
            raise ValueError("independent generation failed")
        _append_response(host, response, "independent_generation")
        calls = _get(response, "tool_calls") or []
        if len(calls) != 1 or _get(calls[0], "type") != "function":
            raise ValueError("generation must request exactly one native function")
        function = _get(calls[0], "function")
        if _get(function, "name") != "execute_python_code":
            raise ValueError("generation requested an unavailable function")
        args = _strict_object(_get(function, "arguments"), LIMITS["program_chars"] * 6 + 1_000)
        if (not {"code", "mode"} <= set(args) or not set(args) <= {"code", "mode", "timeout"}
                or args["mode"] != "final" or not isinstance(args["code"], str)
                or not args["code"].strip() or len(args["code"]) > LIMITS["program_chars"]
                or ("timeout" in args and (type(args["timeout"]) is not int or not 0 < args["timeout"] <= 60))):
            raise ValueError("generation program/mode/argument bound invalid")
        identifier = _get(calls[0], "id")
        if not isinstance(identifier, str) or not identifier or len(identifier) > 200:
            raise ValueError("native tool call identifier invalid")
        intact()
        if not phase.take_tool():
            raise ValueError("final execution tool allowance unavailable")
        output = executor(args["code"], mode="final", timeout=args.get("timeout", 30),
                          global_vars={"retrieved_resources": deepcopy(frozen)})
        _finite(output)
        _trace_tool(host, "execute_python_code", args, output, identifier, request=False)
        diagnostic["B_execution"] = deepcopy(output)
        intact()
        if not isinstance(output, dict) or output.get("error") or output.get("mode") != "final":
            raise ValueError("independent final execution failed")
        b = AnalysisAgent._draft(_json({key: output.get(key) for key in FIELDS}), frozen)
        if b.get("error") or b.get("limitation"):
            raise ValueError("independent executed artifact is ineligible")
        diagnostic["B"] = deepcopy(b)
        # This is a byte-length calculation only, never a source certificate.
        # The eventual checked summary has exactly this shape and a 64-byte SHA.
        bounded_entry = _entry(b, args["code"], _summary_shape(len(b["evidence"]), "0" * 64), workspace_sha)
        diagnostic["B_entry_bytes"] = len(_json(bounded_entry).encode())
        if diagnostic["B_entry_bytes"] > LIMITS["candidate_entry_bytes"]:
            raise ValueError("intact independent candidate exceeds reserved selector entry")
        if not phase.take_tool():
            raise ValueError("original source-check allowance unavailable")
        check = source_checker(b["evidence"], question=context["question"],
                               current_time=context.get("current_time"), patient_fhir_id=context.get("patient_fhir_id"))
        _trace_tool(host, "check_fields", {"citations": b["evidence"]}, check, "v6-independent-source-check")
        diagnostic["B_source_check"] = deepcopy(check)
        b_summary = _summary(b, check)
        intact()
        b_entry = _entry(b, args["code"], b_summary, workspace_sha)
        diagnostic["B_entry_bytes"] = len(_json(b_entry).encode())
        if diagnostic["B_entry_bytes"] > LIMITS["candidate_entry_bytes"]:
            raise ValueError("intact independent candidate exceeds reserved selector entry")
        diagnostic["B_eligible"] = True
        diagnostic["answer_agreement"] = same(primary["answer"], b["answer"])
        if diagnostic["answer_agreement"]:
            diagnostic.update(status="agreement_preserved", reason="exact typed answer agreement; retain original A")
            return original, diagnostic
        selector_packet = {**selector_base, "B": b_entry}
        selection = _messages(SELECTION_PROMPT, selector_packet)
        actual_tokens = count_tokens_in_messages(selection)
        diagnostic.update(selector_packet_sha256=fingerprint(selector_packet), selector_prompt_tokens=actual_tokens)
        if actual_tokens > second_bound or _reserve(actual_tokens) + LIMITS["selection_completion"] > phase.remaining():
            raise ValueError("intact selector packet exceeds remaining reservation")
        intact()
        diagnostic["selector_admitted"] = True
        host.messages.extend([{**deepcopy(message), "phase": "independent_selection"} for message in selection])
        response, error = phase.call(selection, LIMITS["selection_completion"])
        if error or response is None:
            raise ValueError("selector call failed")
        _append_response(host, response, "independent_selection")
        if _get(response, "tool_calls"):
            raise ValueError("selector cannot request tools")
        choice = _strict_object(_get(response, "content"), LIMITS["selection_chars"])
        if (set(choice) != {"choice", "reason"} or choice["choice"] not in ("A", "B", "UNKNOWN")
                or not isinstance(choice["reason"], str) or not choice["reason"].strip() or len(choice["reason"]) > 2_000):
            raise ValueError("selector must choose one existing candidate or UNKNOWN")
        diagnostic["selection"] = choice
        intact()
        if choice["choice"] == "B":
            diagnostic.update(status="switched", selected="B", switched=True,
                              reason="fallible_selector_chose_existing_executed_source_checked_B")
            return deepcopy(b), diagnostic
        diagnostic.update(status="primary_preserved", reason="selector chose " + choice["choice"])
        return original, diagnostic
    except Exception as exc:
        diagnostic.update(status="primary_preserved", reason="auxiliary unavailable: " + type(exc).__name__ + ": " + str(exc))
        return original, diagnostic
    finally:
        diagnostic.update(fatal=phase.fatal, phase=phase.report(), model_calls=phase.actual_calls,
                          tool_calls=host.tool_calls_made - phase.start_tools)
        if phase.fatal:
            diagnostic["status"] = "accounting_failed"
