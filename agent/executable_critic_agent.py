"""Frozen-workspace executable critic; an experimental, fallible repair protocol.

The primary is the unchanged V3 LedgerSolver. The generic critic has one factual
inspection, one bounded code replacement, and one original-source check. Textual
anchors and executable witnesses establish provenance, not semantic truth.
"""
import ast
from copy import deepcopy
import hashlib
import json
import math
import re

from .analysis_agent import AnalysisAgent, _is_resource_id
from .base_agent import BaseAgent, BUDGET_EXCEEDED
from .controller import FixedController
from .ehr_semantics import SEMANTIC_GUIDANCE
from .framework import PYTHON_TOOL, _bound_tool_reply
from .query_contract_agent import LedgerSolver, PhaseHost, fingerprint
from .verification_agent import _empty_answer, field_check_issues
from tools.field_check import check_fields
from tools.python_executor import execute_python_code
from utils.core_utils import count_tokens_in_messages
from utils.evidence_scorer import PATH_SEGMENT_RE

PROTOCOL = "frozen-workspace-executable-critic-v5"
PHASE_CAPS = {"solver": (175_000, 27), "critic": (25_000, 3)}
LIMITS = {"first_prompt_tokens": 5_500, "first_completion": 1_800,
          "second_completion": 4_000, "first_response_tokens": 2_500,
          "probe_tokens": 1_500, "first_chars": 12_000, "second_chars": 24_000,
          "inspect_chars": 6_000, "region_chars": 8_000, "program_chars": 32_000}
ASSURANCE = "fallible_model_semantics_not_global_proof"

CRITIC_PROMPT = """Review the actual executed program against the ORIGINAL question and context.
The primary already passed its execution and source checks. Preserve it unless you
can identify a concrete question-anchored discrepancy and inspect its actual source
or intermediate behavior. Code/records may reveal values: you are not an independent
answer-blind verifier. Source previews do not establish a complete search. You have
no retrieval, no new records, no answer key, and no whole second solve. Missing data
or ambiguous semantics require UNKNOWN; valid null/zero/false results are not errors.
Do not invent context facts, constraints, defaults or domain rules absent from the
original context and declared adapter.
Textual anchors, observations and source-field equality are not semantic proofs.

First response: exact finite JSON, no Markdown or tool calls:
{"decision":"KEEP|UNKNOWN","reason":"brief reason"}
or {"decision":"INSPECT","anchor":{"channel":"question|assumptions",
"quote":"unique exact original substring"},
"claim":"one falsifiable discrepancy","inspect_code":"focused Python"}.
The host executes inspect_code once on a copy of retrieved_resources, in inspect
mode. Set answer to exactly {"facts":[{"resource_id":"Type/id",
"field_path":"concrete.path[0]","value":"actual source value"}],
"observations":[{"name":"unique name","value":"actual derived fact"}]}.
Use 1..16 source facts and 1..16 observations, small finite JSON values. A source
fact must match the immutable original record, not a mutated or invented value.
An observation is a program output, not a trusted expected-answer oracle. Keep
inspection focused and bounded. Every invocation starts a new worker. No code
execution in prose can replace the actual final execution.

After the inspection, the second/final response is KEEP/UNKNOWN as above, or:
{"decision":"PATCH","anchor":<same original anchor>,
"observed_discrepancy":"actual behavior and why it conflicts",
"expected_obligation":"what the original question/context requires",
"preserved_obligations":["other obligations still satisfied"],
"witnesses":[{"fact":<exact inspected source fact>,"observation_name":"name",
"observed_value":<exact inspected value>,"relevance":"causal relevance"}],
"patch":{"old_text":"one unique exact region of actual original code",
"new_text":"replacement region"}}.
Use 1..8 witnesses and 1..8 preserved obligations. The patch anchor must equal the
first claim anchor. One region only; each old/new string <=8,000 characters. You
may delete a region. Explain downstream consequences; changing outputs alone does
not prove correctness. Same contributing IDs may support a corrected projection,
grouping or arithmetic. If the inspection is inconclusive, return UNKNOWN rather
than patching. A valid source check cannot certify your semantic interpretation.
The host will execute the full patched program and re-fetch its source citations.
""" + SEMANTIC_GUIDANCE


class V5PhaseHost(PhaseHost):
    def __init__(self, parent, name):
        if name not in PHASE_CAPS:
            raise ValueError("V5 permits solver and critic phases only")
        super().__init__(parent, "solver" if name == "solver" else "audit")
        self.name = name
        self.token_cap, self.tool_cap = PHASE_CAPS[name]


def _text(value, limit, empty=False):
    return isinstance(value, str) and len(value) <= limit and (empty or bool(value.strip()))


def _strict_json(content, chars):
    if not isinstance(content, str) or len(content) > chars:
        raise ValueError("response exceeds text bounds")
    def pairs(items):
        output = {}
        for key, value in items:
            if key in output:
                raise ValueError("duplicate JSON keys")
            output[key] = value
        return output
    value = json.loads(content, object_pairs_hook=pairs)
    json.dumps(value, allow_nan=False)
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    return value


def _same(left, right):
    """Strict finite JSON equality; bool/number and missing/null stay distinct."""
    return fingerprint(left) == fingerprint(right)


def validate_anchor(anchor, context):
    if not isinstance(anchor, dict) or set(anchor) != {"channel", "quote"}:
        raise ValueError("anchor schema invalid")
    if anchor["channel"] not in ("question", "assumptions"):
        raise ValueError("anchor channel invalid")
    quote = anchor["quote"]
    source = context[anchor["channel"]]
    if not _text(quote, 2_000):
        raise ValueError("anchor quote invalid")
    starts = [index for index in range(len(source)) if source.startswith(quote, index)]
    if len(starts) != 1:
        raise ValueError("anchor quote missing or ambiguous in original channel")
    return {**deepcopy(anchor), "span": [starts[0], starts[0] + len(quote)]}


def parse_first(content, context):
    value = _strict_json(content, LIMITS["first_chars"])
    if value.get("decision") in ("KEEP", "UNKNOWN"):
        if set(value) != {"decision", "reason"} or not _text(value["reason"], 1_000):
            raise ValueError("nonrepair response schema invalid")
    elif value.get("decision") == "INSPECT":
        if (set(value) != {"decision", "anchor", "claim", "inspect_code"}
                or not _text(value["claim"], 1_500)
                or not _text(value["inspect_code"], LIMITS["inspect_chars"])):
            raise ValueError("inspection schema invalid")
        validate_anchor(value["anchor"], context)
    else:
        raise ValueError("first response decision invalid")
    return value


def _field(record, path):
    """Resolve one concrete path without flattening or matching unindexed lists."""
    current = record
    for segment in path.split("."):
        match = PATH_SEGMENT_RE.fullmatch(segment)
        if match is None or not isinstance(current, dict) or match.group(1) not in current:
            raise ValueError("witness source field missing or nonconcrete")
        current = current[match.group(1)]
        for index in re.findall(r"\d+", match.group(2)):
            position = int(index)
            if not isinstance(current, list) or position >= len(current):
                raise ValueError("witness source index missing")
            current = current[position]
    return current


def validate_fact(fact, workspace):
    if (not isinstance(fact, dict) or set(fact) != {"resource_id", "field_path", "value"}
            or not _is_resource_id(fact["resource_id"]) or not _text(fact["field_path"], 256)
            or len(json.dumps(fact["value"], allow_nan=False)) > 1_024):
        raise ValueError("source fact schema invalid")
    kind, identifier = fact["resource_id"].split("/", 1)
    records = [row for row in workspace.get(kind, []) if row.get("id") == identifier]
    if len(records) != 1 or not _same(_field(records[0], fact["field_path"]), fact["value"]):
        raise ValueError("source fact differs from immutable original")
    return deepcopy(fact)


def validate_probe(output, workspace):
    if not isinstance(output, dict) or output.get("error"):
        raise ValueError("inspection failed")
    value = output.get("answer")
    if (not isinstance(value, dict) or set(value) != {"facts", "observations"}
            or not isinstance(value["facts"], list) or not 1 <= len(value["facts"]) <= 16
            or not isinstance(value["observations"], list) or not 1 <= len(value["observations"]) <= 16):
        raise ValueError("inspection answer schema invalid")
    json.dumps(value, allow_nan=False)
    for fact in value["facts"]:
        validate_fact(fact, workspace)
    names = set()
    for observation in value["observations"]:
        if (not isinstance(observation, dict) or set(observation) != {"name", "value"}
                or not _text(observation["name"], 80) or observation["name"] in names
                or len(json.dumps(observation["value"], allow_nan=False)) > 1_024):
            raise ValueError("inspection observation schema invalid")
        names.add(observation["name"])
    return deepcopy(value)


def parse_patch(content, context, first, probe, code, workspace):
    value = _strict_json(content, LIMITS["second_chars"])
    if value.get("decision") in ("KEEP", "UNKNOWN"):
        if set(value) != {"decision", "reason"} or not _text(value["reason"], 1_000):
            raise ValueError("nonrepair response schema invalid")
        return value, None
    required = {"decision", "anchor", "observed_discrepancy", "expected_obligation",
                "preserved_obligations", "witnesses", "patch"}
    if value.get("decision") != "PATCH" or set(value) != required:
        raise ValueError("patch response schema invalid")
    validate_anchor(value["anchor"], context)
    if not _same(value["anchor"], first["anchor"]):
        raise ValueError("patch switched the inspected question/context anchor")
    if not all(_text(value[key], 1_500) for key in ("observed_discrepancy", "expected_obligation")):
        raise ValueError("patch explanation invalid")
    preserved = value["preserved_obligations"]
    if not isinstance(preserved, list) or not 1 <= len(preserved) <= 8 or not all(_text(i, 500) for i in preserved):
        raise ValueError("preserved obligations invalid")
    witnesses = value["witnesses"]
    if not isinstance(witnesses, list) or not 1 <= len(witnesses) <= 8:
        raise ValueError("patch witnesses invalid")
    observations = {item["name"]: item["value"] for item in probe["observations"]}
    for witness in witnesses:
        if (not isinstance(witness, dict)
                or set(witness) != {"fact", "observation_name", "observed_value", "relevance"}
                or not _text(witness["observation_name"], 80) or not _text(witness["relevance"], 500)):
            raise ValueError("patch witness schema invalid")
        validate_fact(witness["fact"], workspace)
        if not any(_same(witness["fact"], fact) for fact in probe["facts"]):
            raise ValueError("patch source fact was not observed in the probe")
        name = witness["observation_name"]
        if name not in observations or not _same(observations[name], witness["observed_value"]):
            raise ValueError("patch behavior was not observed in the probe")
    patch = value["patch"]
    if (not isinstance(patch, dict) or set(patch) != {"old_text", "new_text"}
            or not _text(patch["old_text"], LIMITS["region_chars"])
            or not _text(patch["new_text"], LIMITS["region_chars"], empty=True)
            or patch["old_text"] == patch["new_text"]):
        raise ValueError("patch code region invalid")
    starts = [index for index in range(len(code)) if code.startswith(patch["old_text"], index)]
    if len(starts) != 1:
        raise ValueError("old region must occur exactly once in original code")
    start = starts[0]
    replaced = code[:start] + patch["new_text"] + code[start + len(patch["old_text"]):]
    if len(replaced) > LIMITS["program_chars"]:
        raise ValueError("replacement program exceeds code bounds")
    ast.parse(replaced)
    return value, replaced


def _prompt_reserve(tokens):
    return tokens + max(64, math.ceil(tokens * 0.05))


def critic_packet(context, artifact, workspace, ledger, complete, unresolved):
    """Exact input builder shared with score-blind offline coverage diagnostics."""
    code_sha = hashlib.sha256(artifact["code"].encode()).hexdigest()
    packet = {"context": context, "final_code": artifact["code"], "code_sha256": code_sha,
              "workspace_sha256": artifact["workspace_sha256"], "retrieval_ledger": ledger,
              "complete_types": complete, "unresolved_scope": unresolved,
              "inventory_preview": json.loads(_bound_tool_reply({"content": json.dumps({
                  kind: {"count": len(rows), "sample": rows[:1]} for kind, rows in workspace.items()},
                  default=str)})["content"])}
    messages = [{"role": "system", "content": CRITIC_PROMPT},
                {"role": "user", "content": json.dumps(packet, allow_nan=False)}]
    return packet, messages


class ExecutableCriticAgent(BaseAgent):
    def _trace_tool(self, phase, name, arguments, output, identifier):
        self.messages.append({"role": "assistant", "content": None, "phase": phase,
            "tool_calls": [{"id": identifier, "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments)}}]})
        self.messages.append({"role": "tool", "name": name, "tool_call_id": identifier,
                              "content": json.dumps(output, default=str), "phase": phase})

    def _remaining(self, phase):
        remaining = phase.token_cap - (self.total_usage["total_tokens"] - phase.start_tokens)
        if self.max_total_tokens is not None:
            remaining = min(remaining, self.max_total_tokens - self.total_usage["total_tokens"])
        return remaining

    def _critic_call(self, phase, messages, completion):
        prompt = count_tokens_in_messages(messages)
        reserve = _prompt_reserve(prompt) + completion
        if reserve > self._remaining(phase):
            phase.errors.append("critic known-budget refusal before model call")
            return None, "critic known-budget refusal"
        previous = self.max_total_tokens
        self.max_total_tokens = self.total_usage["total_tokens"] + reserve
        if previous is not None:
            self.max_total_tokens = min(previous, self.max_total_tokens)
        phase.actual_calls = getattr(phase, "actual_calls", 0) + 1
        previous_usage_calls = len(self.call_usage)
        previous_totals = deepcopy(self.total_usage)
        def unknown_invocation():
            phase.fatal = True
            self.total_usage["cost"] = None
            # Invalid shapes are not usable counters. Retain every valid
            # already recorded increment and preserve earlier known totals.
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "llm_calls"):
                value = self.total_usage.get(key)
                if type(value) is not int or value < previous_totals[key]:
                    self.total_usage[key] = previous_totals[key]
            if len(self.call_usage) == previous_usage_calls:
                self._update_usage({"prompt_tokens": 0, "completion_tokens": 0,
                    "total_tokens": 0, "llm_calls": max(0, previous_totals["llm_calls"] + 1 - self.total_usage["llm_calls"]), "cost": None,
                    "per_call": [{"prompt": 0, "completion": 0, "usage_available": False}]})
            else:
                # Preserve any reported numeric counters, but never pass a
                # malformed per-call object into the shared output formatter.
                for index in range(previous_usage_calls, len(self.call_usage)):
                    raw = self.call_usage[index]
                    item = {}
                    if not isinstance(raw, dict):
                        raw = {}
                    for key in ("prompt", "completion"):
                        value = raw.get(key)
                        item[key] = value if type(value) is int and value >= 0 else 0
                    if "cached" in raw:
                        value = raw["cached"]
                        item["cached"] = value if type(value) is int and value >= 0 else 0
                    item.update(usage_available=False, malformed_usage=True)
                    self.call_usage[index] = item
        try:
            response, error, usage = phase._llm_call(messages, tools=None)
        except Exception:
            unknown_invocation()  # An admitted invocation has unknown billed usage.
            raise
        finally:
            self.max_total_tokens = previous
        # All pre-call budget refusals happen above. A provider error without
        # usage cannot be treated as a free, harmless auxiliary failure.
        calls = usage.get("per_call") if isinstance(usage, dict) else None
        known = (isinstance(calls, list) and len(calls) == 1
                 and (usage.get("cost") is None or (type(usage.get("cost")) in (int, float)
                      and math.isfinite(usage["cost"]) and usage["cost"] >= 0))
                 and ("cached_prompt_tokens" not in usage or (type(usage["cached_prompt_tokens"]) is int
                      and usage["cached_prompt_tokens"] >= 0))
                 and type(usage.get("llm_calls")) is int and usage["llm_calls"] == 1
                 and all(type(usage.get(key)) is int and usage[key] >= 0
                         for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
                 and all(isinstance(item, dict)
                         and set(item) <= {"prompt", "completion", "cached", "usage_available"}
                         and item.get("usage_available") is True
                         and all(type(item.get(key)) is int and item[key] >= 0
                                 for key in ("prompt", "completion"))
                         and ("cached" not in item or (type(item["cached"]) is int and item["cached"] >= 0))
                         for item in calls))
        if known:
            known = (usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
                     and sum(item["prompt"] for item in calls) == usage["prompt_tokens"]
                     and sum(item["completion"] for item in calls) == usage["completion_tokens"]
                     and ("cached_prompt_tokens" not in usage or
                          sum(item.get("cached", 0) for item in calls) == usage["cached_prompt_tokens"]))
        if not known:
            unknown_invocation()
            return None, error or "Critic provider usage unavailable or malformed"
        if usage["completion_tokens"] > completion:
            phase.fatal = True
            phase.errors.append("critic reported completion exceeded requested subcap")
            return None, "Critic reported completion exceeded requested subcap"
        return response, error

    @staticmethod
    def _snapshot(workspace, artifact, code_sha):
        if (fingerprint(workspace) != artifact["workspace_sha256"]
                or hashlib.sha256(artifact["code"].encode()).hexdigest() != code_sha):
            raise ValueError("frozen primary workspace/program identity changed")

    def _sandbox(self, code, mode, workspace, identifier):
        try:
            output = execute_python_code(code, mode=mode,
                global_vars={"retrieved_resources": deepcopy(workspace)})
        except Exception as exc:
            output = {"error": "sandbox exception: " + type(exc).__name__}
        self._trace_tool("critic", PYTHON_TOOL, {"code": code, "mode": mode}, output, identifier)
        return output

    def _candidate_check(self, candidate, context, unresolved):
        try:
            check = check_fields(candidate["evidence"], question=context["question"],
                current_time=context["current_time"], patient_fhir_id=context["patient_fhir_id"])
        except Exception as exc:
            check = {"issues": ["source_check: " + type(exc).__name__], "results": []}
        self._trace_tool("critic", "check_fields", {"citations": candidate["evidence"]},
                         check, "v5-critic-source-check")
        if (not isinstance(check, dict)
                or not set(check) <= {"issues", "results", "n_valid", "question_scopes", "scope_membership_only"}
                or not isinstance(check.get("issues"), list)
                or not all(isinstance(item, str) for item in check["issues"])
                or not isinstance(check.get("results"), list)
                or len(check["results"]) != len(candidate["evidence"])):
            raise ValueError("source checker returned malformed/incomplete results")
        for citation, result in zip(candidate["evidence"], check["results"]):
            if (not isinstance(result, dict) or not _same(result.get("citation"), citation)
                    or not isinstance(result.get("warnings"), list)):
                raise ValueError("source checker result does not bind candidate citation")
        issues = list(check["issues"])
        structured = field_check_issues(candidate, check)
        if _empty_answer(candidate["answer"]):
            structured = [issue for issue in structured if issue["type"] != "no_evidence"]
        issues.extend(structured)
        if unresolved:
            issues.append({"type": "source_scope", "unresolved_scope": deepcopy(unresolved)})
        return check, issues

    def _critique(self, primary, artifact, workspace, context, ledger, complete, unresolved, phase):
        try:
            return self._critique_impl(primary, artifact, workspace, context, ledger, complete, unresolved, phase)
        except Exception as exc:
            return primary, {"status": "primary_preserved", "assurance": ASSURANCE,
                "model_calls": getattr(phase, "actual_calls", 0), "tool_calls": self.tool_calls_made - phase.start_tools,
                "reason": "critic preflight failed: " + type(exc).__name__}

    def _critique_impl(self, primary, artifact, workspace, context, ledger, complete, unresolved, phase):
        diagnostic = {"status": "not_run", "assurance": ASSURANCE,
                      "model_calls": 0, "tool_calls": 0}
        code_sha = hashlib.sha256(artifact["code"].encode()).hexdigest()
        diagnostic.update(primary_code_sha256=code_sha, workspace_sha256=fingerprint(workspace))
        packet, messages = critic_packet(context, artifact, workspace, ledger, complete, unresolved)
        prompt = count_tokens_in_messages(messages)
        reserved_second = prompt + LIMITS["first_response_tokens"] + LIMITS["probe_tokens"] + 300
        reservation = (_prompt_reserve(prompt) + LIMITS["first_completion"]
                       + _prompt_reserve(reserved_second) + LIMITS["second_completion"])
        diagnostic.update(packet_sha256=fingerprint(packet), first_prompt_tokens=prompt,
                          two_call_token_reservation=reservation)
        if prompt > LIMITS["first_prompt_tokens"] or reservation > self._remaining(phase):
            diagnostic.update(status="coverage_skip", reason="intact essential packet/two-call reservation exceeds critic allowance")
            return primary, diagnostic
        try:
            self._snapshot(workspace, artifact, code_sha)
            self.messages.extend(deepcopy(messages))
            response, error = self._critic_call(phase, messages, LIMITS["first_completion"])
            diagnostic["model_calls"] = getattr(phase, "actual_calls", 0)
            if error:
                raise ValueError(error)
            self.messages.append(response)
            if getattr(response, "tool_calls", None):
                raise ValueError("native tool calls are unavailable to the JSON-only critic")
            if count_tokens_in_messages([{"role": "assistant", "content": response.content}]) > LIMITS["first_response_tokens"]:
                raise ValueError("first response exceeds reserved second-input bound")
            first = parse_first(response.content, context)
            diagnostic["first"] = first
            if first["decision"] == "INSPECT":
                diagnostic["anchor"] = validate_anchor(first["anchor"], context)
            if first["decision"] != "INSPECT":
                diagnostic.update(status="primary_preserved", reason=first["reason"])
                return primary, diagnostic
            if not phase._take_tool_call():
                raise ValueError("critic inspection tool allowance unavailable")
            diagnostic["tool_calls"] += 1
            output = self._sandbox(first["inspect_code"], "inspect", workspace, "v5-critic-inspect")
            diagnostic["probe_output"] = deepcopy(output)
            self._snapshot(workspace, artifact, code_sha)
            probe = validate_probe(output, workspace)
            if count_tokens_in_messages([{"role": "user", "content": json.dumps(probe)}]) > LIMITS["probe_tokens"]:
                raise ValueError("inspection facts exceed reserved second-input bound")
            diagnostic["validated_probe"] = probe
            feedback = {"role": "user", "content": "Actual inspection result; observations are not a semantic oracle. "
                        "Return KEEP/UNKNOWN or the exact PATCH schema now. No tools or further calls.\n" + json.dumps(probe)}
            messages = [*messages, {"role": "assistant", "content": response.content}, feedback]
            self.messages.append(deepcopy(feedback))
            response, error = self._critic_call(phase, messages, LIMITS["second_completion"])
            diagnostic["model_calls"] = getattr(phase, "actual_calls", 0)
            if error:
                raise ValueError(error)
            self.messages.append(response)
            if getattr(response, "tool_calls", None):
                raise ValueError("native tool calls are unavailable to the JSON-only critic")
            proposal, code = parse_patch(response.content, context, first, probe, artifact["code"], workspace)
            diagnostic["proposal"] = proposal
            if code is None:
                diagnostic.update(status="primary_preserved", reason=proposal["reason"])
                return primary, diagnostic
            self._snapshot(workspace, artifact, code_sha)
            if not phase._take_tool_call():
                raise ValueError("critic final execution tool allowance unavailable")
            diagnostic["tool_calls"] += 1
            output = self._sandbox(code, "final", workspace, "v5-critic-final")
            diagnostic["alternate_execution"] = deepcopy(output)
            if not isinstance(output, dict) or output.get("error"):
                raise ValueError("replacement final execution failed")
            candidate = AnalysisAgent._draft(json.dumps(output), workspace)
            if candidate.get("error"):
                raise ValueError(candidate["error"])
            diagnostic["alternate"] = candidate
            if _same({key: primary[key] for key in ("answer", "evidence", "contributing_ids")},
                     {key: candidate[key] for key in ("answer", "evidence", "contributing_ids")}):
                raise ValueError("replacement has no substantive executed effect")
            if not phase._take_tool_call():
                raise ValueError("critic original-source check allowance unavailable")
            diagnostic["tool_calls"] += 1
            check, issues = self._candidate_check(candidate, context, unresolved)
            diagnostic.update(source_check=check, candidate_issues=issues)
            self._snapshot(workspace, artifact, code_sha)
            if issues or candidate.get("limitation"):
                raise ValueError("replacement did not establish a clean source-valid repair")
            diagnostic.update(status="released", reason="executed_source_checked_critic_proposal",
                              candidate_code_sha256=hashlib.sha256(code.encode()).hexdigest())
            return candidate, diagnostic
        except Exception as exc:
            diagnostic["model_calls"] = getattr(phase, "actual_calls", 0)
            diagnostic.update(status="primary_preserved", reason=f"{type(exc).__name__}: {exc}")
            return primary, diagnostic

    def _run(self, question):
        context = FixedController.question_context(question)
        phase = V5PhaseHost(self, "solver")
        solver = LedgerSolver(self, phase)
        result = solver._run(question)
        self.messages.extend(solver.messages)
        phases = [phase.report()]
        workspace = deepcopy(result.get("retrieved_fhir_resources", {}))
        artifact = deepcopy(solver.final_computation)
        primary = solver._computed_candidate(workspace)
        previous = result.get("revision") or {}
        draft = previous.get("draft") or primary
        issues = previous.get("final_check_issues")
        unresolved = deepcopy(solver._scoped_box.scope_failures if solver._scoped_box else {})
        common = {"protocol": PROTOCOL, "primary": primary, "primary_revision": previous,
                  "draft": draft, "phases": phases, "retrieval_ledger": deepcopy(solver.retrieval_ledger),
                  "complete_types": sorted(solver.complete_types()), "unresolved_scope": unresolved,
                  "solver_source_revised": bool(previous.get("revised")), "critic_revised": False,
                  "targeted_revised": False, "check_issues": previous.get("check_issues"),
                  "context_compactions": deepcopy(self.context_compactions)}
        if (phase.fatal or result.get("error") or primary is None or primary.get("error")
                or issues != [] or unresolved or artifact is None):
            final = {"answer": None, "evidence": [], "limitation": result.get("limitation") or
                     "No source-validated explicit final computation"}
            return {**result, **final, "trace": self.messages, "revision": {**common, "final": final,
                "revised": bool(previous.get("revised")), "answer_changed": bool(draft and draft.get("answer") is not None),
                "final_check_issues": issues or [{"type": "execution_contract", "action": final["limitation"]}],
                "release_policy": "primary_failed_execution_or_source_contract"}}
        phase = V5PhaseHost(self, "critic")
        released, diagnostic = self._critique(primary, artifact, workspace, context,
            deepcopy(solver.retrieval_ledger), sorted(solver.complete_types()), unresolved, phase)
        phases.append(phase.report())
        common["critic"] = diagnostic
        if phase.fatal:
            diagnostic["status"] = "accounting_failed"
            final = {"answer": None, "evidence": [], "limitation": "Aggregate usage could not be verified"}
            return {**final, "trace": self.messages, "retrieved_fhir_resources": workspace,
                    "error": "Critic aggregate accounting failed", "revision": {**common, "final": final,
                        "revised": bool(previous.get("revised")), "answer_changed": bool(draft and draft.get("answer") is not None),
                        "final_check_issues": [{"type": "accounting", "action": final["limitation"]}],
                        "release_policy": "aggregate_accounting_failed"}}
        self.last_llm_error = None
        changed = released is not primary
        return {"answer": released["answer"], "evidence": released["evidence"], "limitation": released["limitation"],
                "trace": self.messages, "retrieved_fhir_resources": workspace, "revision": {**common,
                    "final": released, "revised": bool(previous.get("revised")) or changed,
                    "answer_changed": not _same(released["answer"], draft["answer"]),
                    "critic_revised": changed, "targeted_revised": changed, "final_check_issues": [],
                    "release_policy": "executed_source_checked_critic_proposal" if changed else "source_valid_primary_preserved"}}


def run_v5_job(args_tuple):
    from utils import run_agent_safe
    import tools.cache as cache_module
    input_data, strategy, model, verbose, base_url, enable_cache, *rest = args_tuple
    if strategy != "multi_agent_v5":
        raise ValueError("V5 worker received a different strategy")
    cache_module.CACHE_ENABLED = enable_cache
    agent = ExecutableCriticAgent(model, verbose=verbose, base_url=base_url)
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
