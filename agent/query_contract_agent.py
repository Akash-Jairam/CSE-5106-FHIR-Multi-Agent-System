"""Integrated solver + draft-blind intent contract + bounded program audit.

Only one question-anchored, mechanically restricted expression change may
replace a source-valid primary. Auditor prose/UNKNOWN/format failures do not
veto it. This protocol has no mandatory third solve or final synthesis call.
"""
import ast
from copy import deepcopy
import hashlib
import json
import math
import re

from .base_agent import BaseAgent, BUDGET_EXCEEDED
from .controller import FixedController
from .analysis_agent import AnalysisAgent
from .ehr_semantics import SEMANTIC_GUIDANCE
from .framework import PYTHON_TOOL, RETRIEVAL_TOOLS, Role, Toolbox, _bound_tool_reply
from .output_schema import _json_objects
from .single_agent import SingleAgent
from .verification_agent import _canonical, _empty_answer, field_check_issues
from tools.field_check import check_fields, _wall_clock
from tools.python_executor import execute_python_code
from utils.core_utils import count_tokens_in_messages


def run_v3_job(args_tuple):
    """Common budgeted agent execution with optional audit storage before IPC."""
    from utils import run_agent_safe
    import tools.cache as cache_module
    input_data, strategy, model, verbose, base_url, enable_cache, *rest = args_tuple
    if strategy != "multi_agent_v3":
        raise ValueError("V3 worker received a different strategy")
    cache_module.CACHE_ENABLED = enable_cache
    agent = QueryContractAgent(model, verbose=verbose, base_url=base_url)
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

PHASE_CAPS = {"contract": (10_000, 0), "solver": (175_000, 27),
              "audit": (15_000, 0), "targeted": (0, 3)}
PROTOCOL = "query-contract-v3.2-static-audit"

CONTRACT_PROMPT = """Independently derive a short question contract. No solver
answer, code, record selections or prior interpretation is visible. Do not
compute the answer. Use only the question, bound context and shared conventions.
Return JSON {"requirements":[{"id":"c1","kind":"predicate|ordinal|concept|scope|aggregation|time_field|output",
"quote":"exact question substring","span":[0,1],"expected":"precise requirement"}],
"unknowns":["uncertain mapping"]}. Each explicit requirement has an exact
0-based [start,end) span. Do not add clinical normal ranges or constraints absent
from the question. Unknown mappings remain unknown. Substring provenance is
not proof that your interpretation is correct. Keep the contract under 2,000 tokens.
""" + SEMANTIC_GUIDANCE

AUDIT_PROMPT = """Check the solver's actual executed Python against the frozen
question contract derived before its code was visible. The contract itself can
be wrong: mark an overreaching requirement UNKNOWN. Source record previews are
bounded, not complete search results. The rendered candidate answer is hidden.
This is one static code-audit pass with no tools. Do not request inspection,
perform a whole second solve, or emit a replacement answer. The host executes
and source-checks any proposed supported local repair. Zero/null/false
can be correct; unfamiliar values and vague doubts are not errors.
Return JSON {"checks":[{"requirement_id":"c1","verdict":"PASS|FAIL|UNKNOWN|N/A","reason":"brief evidence"}],
"repair":null|{"requirement_id":"c1","old_expression":"exact Python expression",
"new_expression":"one proposed expression","reason":"specific semantic discrepancy"}}.
Only a FAIL on an exact question-anchored predicate or measurement ordinal may
request one repair. Supported automatic changes are one numeric comparison
operator with an unchanged literal threshold and source value expression, or
one ordinal index of a list explicitly sorted by effectiveDateTime. Keep every
other expression unchanged. Numeric repairs require an isolated How-many count;
ordinal repairs require a directly selected source quantity. Compound outputs,
negated or multiple numeric/rank clauses remain unsupported. Changes to dates/concepts/visits/units/aggregation,
missing retrieval, aliases you cannot establish, or multiple changes remain
UNKNOWN for this variant; they cannot veto the source-valid primary. A proposed
change is a hypothesis awaiting host execution and source checks.
""" + SEMANTIC_GUIDANCE


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def _reply(content):
    return _bound_tool_reply({"content": json.dumps(content, default=str)})["content"]


class HostBoundaryMessage:
    role = "assistant"
    content = "Host phase boundary: executed final ready for source checking. No model call."
    tool_calls = None
    model_call = False

    def model_dump(self, **_):
        # Remains a valid API message if source feedback requires a later call.
        return {"role": self.role, "content": self.content}

    def to_dict(self):
        return {**self.model_dump(), "model_call": False, "phase": "solver_boundary"}


class PhaseHost:
    """All roles use the same host/model and aggregate accountant, plus local caps.

    Temporary ceilings cap completions before the provider call. Local refusals
    remain in phase diagnostics but cannot contaminate a later valid release.
    Unknown provider usage/global overshoot remains fatal for the whole attempt.
    """
    def __init__(self, parent, name):
        self.parent, self.name = parent, name
        self.token_cap, self.tool_cap = PHASE_CAPS[name]
        self.start_tokens = parent.total_usage["total_tokens"]
        self.start_tools = parent.tool_calls_made
        self.errors = []
        self.fatal = False

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def _llm_call(self, messages, tools=None):
        parent = self.parent
        global_limit = parent.max_total_tokens
        local_limit = self.start_tokens + self.token_cap
        previous_stop, previous_error = parent.budget_stop, parent.last_llm_error
        if parent.total_usage["total_tokens"] >= local_limit:
            error = f"{BUDGET_EXCEEDED}: {self.name} phase token ceiling"
            self.errors.append(error)
            return None, error, None
        parent.max_total_tokens = min(global_limit, local_limit) if global_limit is not None else local_limit
        try:
            response, error, usage = parent._llm_call(messages, tools=tools)
        finally:
            parent.max_total_tokens = global_limit
        if error:
            self.errors.append(error)
        unknown_usage = bool(usage and any(c.get("usage_available") is False for c in usage.get("per_call", [])))
        self.fatal = (self.fatal or unknown_usage or parent.total_usage["total_tokens"] > local_limit
                      or (global_limit is not None and parent.total_usage["total_tokens"] > global_limit))
        if not self.fatal:
            parent.budget_stop, parent.last_llm_error = previous_stop, previous_error
        return response, error, usage

    def _take_tool_call(self):
        if self.parent.tool_calls_made - self.start_tools >= self.tool_cap:
            self.errors.append(f"{BUDGET_EXCEEDED}: {self.name} phase tool ceiling")
            return False
        return self.parent._take_tool_call()

    def report(self):
        return {"phase": self.name, "token_cap": self.token_cap, "tool_cap": self.tool_cap,
                "tokens": self.parent.total_usage["total_tokens"] - self.start_tokens,
                "tool_calls": self.parent.tool_calls_made - self.start_tools,
                "errors": list(self.errors), "fatal": self.fatal}


class ScopedLedgerToolbox(Toolbox):
    """Retain unresolved original records until later source context can bind them.

    Recovery retries the unchanged shared patient/time validator against original
    fetched records; it does not infer ownership or dates from auditor claims.
    Known other-patient/future exclusions are outside the requested source scope.
    Unknown ownership/time cannot establish search completeness or absence.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.originals = {}

    def _in_scope(self, records):
        for record in records:
            self.originals[f"{record['resourceType']}/{record['id']}"] = deepcopy(record)
        return self._validate_scope(records)

    def _validate_scope(self, records):
        kept, excluded = super()._in_scope(records)
        kept_ids = {f"{record['resourceType']}/{record['id']}" for record in kept}
        # Patient-linked closure can arrive before its ownership roots are held.
        # HTTP completion alone cannot certify that this held metadata is usable.
        for record in records:
            key = f"{record['resourceType']}/{record['id']}"
            if record["resourceType"] in {"Medication", "Location"} and key not in kept_ids:
                self.scope_failures[key] = "unlinked_metadata"
        return kept, excluded

    def _retrieve(self, name, args):
        output = json.loads(super()._retrieve(name, args))
        if output.get("error"):
            return json.dumps(output)
        deferred = [deepcopy(self.originals[key]) for key in self.scope_failures if key in self.originals]
        recovered, _ = self._validate_scope(deferred)
        recovered_ids = []
        for record in recovered:
            rtype = record["resourceType"]
            held = {item["id"]: item for item in self.resources.get(rtype, [])}
            held[record["id"]] = deepcopy(record)
            self.resources[rtype] = list(held.values())
            recovered_ids.append(f"{rtype}/{record['id']}")
        output["recovered_scope_ids"] = recovered_ids
        output["unresolved_scope"] = deepcopy(self.scope_failures)
        return json.dumps(output)


class LedgerSolver(SingleAgent):
    """Keep the single solver's code/debug/final/source-check trajectory intact.

    Retrieval uses the existing validated Toolbox dispatcher to retain complete
    search metadata, patient/time bindings and linked-closure errors. Starting
    packets never establish a patient-wide search's completeness.
    """
    def __init__(self, parent, phase):
        # Two host-only slots cover the initial and revised final source checks
        # after tool batches. Actual model calls remain capped at 20.
        super().__init__(parent.model, max_iterations=22, verbose=parent.verbose, base_url=parent.base_url)
        self._phase = phase
        self._llm_call = self._solver_llm_call
        self._take_tool_call = self._solver_take_tool_call
        self._actual_calls = 0
        self._normal_tools = 0
        self._source_check_active = False
        self._current_workspace = None
        self.perturbation = parent.perturbation
        self.retrieval_ledger = []
        self._scoped_box = None

    def _append_message(self, message):
        if isinstance(message, HostBoundaryMessage):
            # The SDK accepts message mappings, not arbitrary host objects.
            # Keep host-only annotations in the saved audit, outside API input.
            self.messages.append(message.to_dict())
            self.model_messages.append(message.model_dump())
            return
        return super()._append_message(message)

    def complete_types(self):
        unresolved = set(self._scoped_box.scope_failures if self._scoped_box else {})
        unresolved_types = {key.split("/", 1)[0] for key in unresolved}
        return {item["arguments"].get("resource_type") for item in self.retrieval_ledger
                if item.get("success") and item.get("tool") == "get_resources_by_patient_fhir_id"
                and item["arguments"].get("resource_type") not in unresolved_types}

    def _check(self, citations, encounter_id=None):
        result = super()._check(citations, encounter_id)
        unresolved = self._scoped_box.scope_failures if self._scoped_box else {}
        if unresolved:
            result["issues"].append("source_scope: unresolved patient/time binding for fetched records: " +
                                   "; ".join(f"{key}={reason}" for key, reason in sorted(unresolved.items())))
            result["unresolved_scope"] = deepcopy(unresolved)
        return result

    def _solver_llm_call(self, messages, tools=None):
        # SingleAgent initializes this to False. The team requires an executed
        # final from the first turn, so early prose receives its existing
        # final-execution feedback instead of terminating before the outer gate.
        if self._actual_calls == 0:
            self._requires_final_computation = True
            self._final_computation_error = "This solver requires an explicit mode=final execution"
        # Check only at the boundary after the entire preceding batch. A later
        # retrieval/final failure in that batch has already invalidated readiness.
        if self._current_workspace is not None:
            candidate = self._computed_candidate(self._current_workspace)
            if candidate is not None and not candidate.get("error"):
                return HostBoundaryMessage(), None, None
        if self._actual_calls >= 20:
            return None, "Solver did not finalize within its 20 model-call allocation", None
        remaining_tokens = self._phase.token_cap - (self._phase.parent.total_usage["total_tokens"] - self._phase.start_tokens)
        remaining_tools = max(0, self._phase.tool_cap - 2 - self._normal_tools)
        if self._phase.parent.max_total_tokens is not None:
            remaining_tokens = min(remaining_tokens, self._phase.parent.max_total_tokens - self._phase.parent.total_usage["total_tokens"])
        if self._phase.parent.max_tool_calls is not None:
            remaining_tools = min(remaining_tools, max(0, self._phase.parent.max_tool_calls - self._phase.parent.tool_calls_made - 2))
        prompt_size = count_tokens_in_messages(messages, tools)
        guidance = {"role": "user", "content": (
            f"Solver allocation remaining: about {remaining_tokens} total input/output tokens and {remaining_tools} model-requested tools. "
            "Two automatic source-check calls are reserved separately. "
            + ("Finalize now in mode=final if the completed search establishes an answer; otherwise finalize null with the truthful missing-data limitation. "
               "Further inspection may prevent finalization. Do not guess or use diagnostics as the final answer."
               if remaining_tools <= 3 or remaining_tokens < 2 * prompt_size + 8000 else
               "Keep diagnostics bounded and finalize promptly once the calculation is established."))}
        visible = [*messages, guidance]
        self._actual_calls += 1
        # Append this budget signal to both saved and visible histories only once
        # for this call, keeping the main solver trajectory intact.
        self.messages.append(deepcopy(guidance))
        return self._phase._llm_call(visible, tools=tools)

    def _solver_take_tool_call(self):
        if not self._source_check_active and self._normal_tools >= self._phase.tool_cap - 2:
            self._phase.errors.append("Solver model-requested tool ceiling; two source checks reserved")
            return False
        if (not self._source_check_active and self._phase.parent.max_tool_calls is not None
                and self._phase.parent.max_tool_calls - self._phase.parent.tool_calls_made <= 2):
            self._phase.errors.append("Global tool allowance reserved for automatic source checks")
            return False
        accepted = self._phase._take_tool_call()
        if accepted and not self._source_check_active:
            self._normal_tools += 1
        return accepted

    def _automatic_check(self, output, index):
        self._source_check_active = True
        try:
            return super()._automatic_check(output, index)
        finally:
            self._source_check_active = False

    def _tool_budget_message(self, tool_call):
        # A requested mutation/retrieval that did not run cannot silently leave
        # an earlier artifact ready at the same batch's boundary.
        if tool_call.function.name in RETRIEVAL_TOOLS:
            self._invalidate_computation("A requested retrieval could not run within the allocated tools")
        elif tool_call.function.name == PYTHON_TOOL:
            try:
                mode = json.loads(tool_call.function.arguments).get("mode", "auto")
            except (ValueError, TypeError, AttributeError):
                mode = "auto"
            if mode != "inspect":
                self._invalidate_computation("A requested calculation could not run within the allocated tools")
        return super()._tool_budget_message(tool_call)

    def _handle_tool_call(self, tool_call, resources):
        self._current_workspace = resources
        if tool_call.function.name not in RETRIEVAL_TOOLS:
            return super()._handle_tool_call(tool_call, resources)
        self._invalidate_computation("Retrieval requires a new final calculation")
        if self._scoped_box is None:
            self._scoped_box = ScopedLedgerToolbox(RETRIEVAL_TOOLS, resources,
                patient_fhir_id=self._patient_fhir_id, current_time=self._current_time)
        reply = self._scoped_box.call(tool_call)
        try:
            args = json.loads(tool_call.function.arguments)
            output = json.loads(reply["content"])
            self.retrieval_ledger.append({"tool": tool_call.function.name, "arguments": args,
                                          "success": not bool(output.get("error")),
                                          "complete_type": (args.get("resource_type") if not output.get("error")
                                              and tool_call.function.name == "get_resources_by_patient_fhir_id"
                                              and not any(key.startswith(args.get("resource_type", "") + "/")
                                                          for key in self._scoped_box.scope_failures) else None),
                                          "excluded": deepcopy(output.get("excluded", {})),
                                          "unresolved_scope": deepcopy(self._scoped_box.scope_failures),
                                          "recovered_scope_ids": output.get("recovered_scope_ids", []),
                                          "metadata": output.get("retrieval_metadata"),
                                          "workspace_sha256": self._workspace_sha256(resources)})
        except (ValueError, TypeError, KeyError):
            self.retrieval_ledger.append({"success": False, "error": "malformed retrieval response"})
        return reply


def parse_contract(content, question):
    if not isinstance(content, str):
        return None, "contract reply must be text containing JSON"
    contract = next((obj for obj in _json_objects(content or "") if "requirements" in obj), None)
    if not isinstance(contract, dict) or not isinstance(contract.get("requirements"), list):
        return None, "auditor did not return a requirements object"
    try:
        json.dumps(contract, allow_nan=False)
    except (ValueError, TypeError):
        return None, "contract must be finite JSON"
    if len(contract["requirements"]) > 24:
        return None, "contract has more than 24 requirements"
    requirements, seen = [], set()
    for item in contract["requirements"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or item["id"] in seen:
            return None, "requirement IDs must be unique strings"
        seen.add(item["id"])
        quote, span = item.get("quote"), item.get("span")
        positions = []
        if isinstance(quote, str) and quote:
            cursor = 0
            while (position := question.find(quote, cursor)) >= 0:
                positions.append(position)
                cursor = position + 1
                if len(positions) > 1:
                    break
        # Deterministic question-only normalization happens before solver reveal.
        # A span can identify provenance, but cannot make a repeated quote unique.
        valid_anchor = len(positions) == 1
        actual_span = [positions[0], positions[0] + len(quote)] if valid_anchor else None
        requirements.append({**item, "span_supplied": deepcopy(span), "span": actual_span,
                             "span_normalized": valid_anchor and span != actual_span,
                             "anchor_valid": valid_anchor,
                             "anchor_issue": None if valid_anchor else "quote missing, repeated, or not an exact nonempty substring"})
    return {**contract, "requirements": requirements}, None


def parse_audit(content, contract):
    if not isinstance(content, str):
        return None, "audit reply must be text containing JSON"
    audit = next((obj for obj in _json_objects(content or "") if "checks" in obj), None)
    if not isinstance(audit, dict) or not isinstance(audit.get("checks"), list):
        return None, "auditor did not return a checks object"
    try:
        json.dumps(audit, allow_nan=False)
    except (ValueError, TypeError):
        return None, "audit must be finite JSON"
    identifiers = {item["id"] for item in contract["requirements"]}
    seen = set()
    for item in audit["checks"]:
        if (not isinstance(item, dict) or item.get("requirement_id") not in identifiers
                or item["requirement_id"] in seen or item.get("verdict") not in {"PASS", "FAIL", "UNKNOWN", "N/A"}):
            return None, "audit checks need unique known requirement IDs and supported verdicts"
        seen.add(item["requirement_id"])
    if not isinstance(audit.get("repair"), (dict, type(None))):
        return None, "repair must be an object or null"
    return audit, None


THRESHOLD_RE = re.compile(r"\b(greater than or equal to|less than or equal to|greater than|less than|at least|at most)\s+(-?\d+(?:\.\d+)?)\b", re.I)
THRESHOLD_OP = {"greater than": ast.Gt, "less than": ast.Lt, "at least": ast.GtE,
                "at most": ast.LtE, "greater than or equal to": ast.GtE, "less than or equal to": ast.LtE}
ORDINAL_RE = re.compile(r"\b(second to last|second-to-last|first|last|second)\s+(?:measured|measurement|recorded|reading)\b", re.I)
ORDINAL_TOKEN_RE = re.compile(r"\b(second to last|second-to-last|first|last|second)\b", re.I)
UNBOUND_THRESHOLD_RE = re.compile(r"\b(above|below|higher|lower|more than|fewer than|between|equal to)\b", re.I)
NEGATED_CONSTRAINT_RE = re.compile(r"\b(not|no|never|neither|except|excluding|exclude)\b", re.I)


def _constant_number(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float) and math.isfinite(node.value):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _constant_number(node.operand)
        return -value if value is not None else None
    return None


def _value_path(node):
    """Require a visible record-field access, not an ungrounded scalar alias."""
    if isinstance(node, ast.Name):
        return node.id, []
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        base = _value_path(node.value)
        return (base[0], base[1] + [node.slice.value]) if base else None
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
            and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
        base = _value_path(node.func.value)
        return (base[0], base[1] + [node.args[0].value]) if base else None
    return None


def patched_program(code, audit, contract, question):
    """One supported AST expression change; unknown semantics preserve the primary.

    Literal anchors are checked against the question; numeric operators are
    resolved deterministically. Ordinal repairs additionally require an explicit
    ascending effectiveDateTime sort. This is a local constraint check, not a
    proof of complete clinical query correctness.
    """
    repair = audit.get("repair")
    if not repair:
        return None, "no material repair proposed"
    requirement = next((item for item in contract["requirements"]
                        if item["id"] == repair.get("requirement_id")), None)
    failed = any(item.get("requirement_id") == repair.get("requirement_id") and item.get("verdict") == "FAIL"
                 for item in audit["checks"])
    if requirement is None or not requirement["anchor_valid"] or not failed:
        return None, "repair lacks a valid anchored FAIL"
    old, new = repair.get("old_expression"), repair.get("new_expression")
    if not isinstance(old, str) or not isinstance(new, str) or len(old) > 1500 or len(new) > 1500:
        return None, "repair expressions must be bounded strings"
    try:
        tree = ast.parse(code)
        old_node, new_node = ast.parse(old, mode="eval").body, ast.parse(new, mode="eval").body
    except SyntaxError:
        return None, "repair is not a Python expression"
    old_dump = ast.dump(old_node, include_attributes=False)
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.expr) and ast.dump(node, include_attributes=False) == old_dump]
    if len(nodes) != 1:
        return None, "old expression must occur exactly once in the final program"
    detail = {"requirement": requirement, "old_expression": old, "new_expression": new}
    if requirement.get("kind") == "predicate":
        all_predicates = list(THRESHOLD_RE.finditer(question))
        if len(all_predicates) != 1:
            return None, "multiple or unrecognized question predicates lack a clause-to-expression binding"
        remainder = question[:all_predicates[0].start()] + question[all_predicates[0].end():]
        if UNBOUND_THRESHOLD_RE.search(remainder) or NEGATED_CONSTRAINT_RE.search(question):
            return None, "negated or additional threshold language lacks a supported clause binding"
        found = [match for match in all_predicates
                 if requirement["span"][0] <= match.start() and match.end() <= requirement["span"][1]]
        if len(found) != 1 or not isinstance(old_node, ast.Compare) or not isinstance(new_node, ast.Compare):
            return None, "numeric repair needs one explicit threshold and one comparison"
        if len(old_node.ops) != 1 or len(new_node.ops) != 1:
            return None, "chained comparisons are not a local supported repair"
        numerical_comparisons = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)
                                 and any(_constant_number(value) is not None for value in node.comparators)]
        if len(numerical_comparisons) != 1 or numerical_comparisons[0] is not nodes[0]:
            return None, "multiple numerical program comparisons lack a clause-to-expression binding"
        expected = THRESHOLD_OP[found[0].group(1).lower()]
        threshold = float(found[0].group(2))
        path = _value_path(old_node.left)
        if (path is None or path[1] != ["valueQuantity", "value"]
                or ast.dump(old_node.left) != ast.dump(new_node.left)
                or ast.dump(old_node.comparators[0]) != ast.dump(new_node.comparators[0])
                or _constant_number(new_node.comparators[0]) != threshold
                or not isinstance(new_node.ops[0], expected) or type(old_node.ops[0]) is expected):
            return None, "repair may change only the question's numerical operator over valueQuantity.value"
        detail.update(kind="predicate", field_path="valueQuantity.value", threshold=threshold,
                      old_operator=type(old_node.ops[0]).__name__, new_operator=type(new_node.ops[0]).__name__)
    elif requirement.get("kind") == "ordinal":
        all_ordinals = list(ORDINAL_RE.finditer(question))
        if len(all_ordinals) != 1 or len(list(ORDINAL_TOKEN_RE.finditer(question))) != 1:
            return None, "multiple or unrecognized question ranks lack a clause-to-expression binding"
        found = [match for match in all_ordinals
                 if requirement["span"][0] <= match.start() and match.end() <= requirement["span"][1]]
        if len(found) != 1 or not isinstance(old_node, ast.Subscript) or not isinstance(new_node, ast.Subscript):
            return None, "ordinal repair needs one explicit measurement rank and one index"
        if not isinstance(old_node.value, ast.Name) or ast.dump(old_node.value) != ast.dump(new_node.value):
            return None, "ordinal repair must index the same explicitly sorted variable"
        rank = found[0].group(1).lower()
        index = {"first": 0, "second": 1, "last": -1, "second to last": -2, "second-to-last": -2}[rank]
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == old_node.value.id for target in node.targets)]
        if len(assignments) != 1:
            return None, "ordinal source needs one unambiguous sorted assignment"
        sort = assignments[0].value
        key = next((kw.value for kw in sort.keywords if kw.arg == "key"), None) if isinstance(sort, ast.Call) else None
        reverse = next((kw.value for kw in sort.keywords if kw.arg == "reverse"), None) if isinstance(sort, ast.Call) else None
        sort_path = _value_path(key.body) if isinstance(key, ast.Lambda) else None
        if (not isinstance(sort, ast.Call) or not isinstance(sort.func, ast.Name) or sort.func.id != "sorted"
                or len(sort.args) != 1 or any(kw.arg not in {"key", "reverse"} for kw in sort.keywords)
                or sort_path is None or sort_path[1] != ["effectiveDateTime"]
                or not isinstance(key, ast.Lambda) or len(key.args.args) != 1
                or sort_path[0] != key.args.args[0].arg
                or (reverse is not None and not (isinstance(reverse, ast.Constant) and reverse.value is False))
                or _constant_number(new_node.slice) != index or _constant_number(old_node.slice) == index):
            return None, "rank repair needs a proven ascending effectiveDateTime sort and exact requested index"
        mutations = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                     and node.func.value.id == old_node.value.id]
        aliases = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and isinstance(node.value, ast.Name) and node.value.id == old_node.value.id]
        if mutations or aliases:
            return None, "rank source must not be mutated or aliased after sorting"
        rebound = [node for node in ast.walk(tree)
                   if (isinstance(node, ast.Name) and node.id == old_node.value.id and isinstance(node.ctx, (ast.Store, ast.Del))
                       and node not in assignments[0].targets)
                   or (isinstance(node, ast.arg) and node.arg == old_node.value.id)]
        if rebound or any(isinstance(node, (ast.AugAssign, ast.Delete))
                          and any(isinstance(child, ast.Name) and child.id == old_node.value.id for child in ast.walk(node))
                          for node in ast.walk(tree)):
            return None, "rank variable has a deletion, augmented assignment or shadowing binding"
        if any(isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
               and any(isinstance(child, ast.Subscript) and isinstance(child.ctx, ast.Store)
                       and any(isinstance(part, ast.Name) and part.id == old_node.value.id for part in ast.walk(child.value))
                       for child in ast.walk(node)) for node in ast.walk(tree)):
            return None, "rank source has an item assignment that can change its ordering"
        detail.update(kind="ordinal", field_path="effectiveDateTime", rank=rank,
                      sorted_variable=old_node.value.id, old_index=_constant_number(old_node.slice), new_index=index)
    else:
        return None, "this conservative variant does not automatically repair that semantic dimension"
    target = nodes[0]
    class ReplaceOne(ast.NodeTransformer):
        def visit(self, node):
            return deepcopy(new_node) if node is target else super().visit(node)
    patched = ast.fix_missing_locations(ReplaceOne().visit(tree))
    return {**detail, "code": ast.unparse(patched)}, None


class QueryContractAgent(BaseAgent):
    """Production candidate strategy; keeps V2/single strategies unchanged."""
    protocol = PROTOCOL

    def __init__(self, model, verbose=False, base_url=None):
        super().__init__(model, verbose=verbose, base_url=base_url)

    def _run_audit(self, audit_role, packet):
        """One static audit model call; subclasses may run it on another model."""
        return audit_role.run(json.dumps(packet, default=str))

    def _post_release(self, released, policy, context, solver, primary_result, phases):
        """Hook after the V3 release decision: (released, policy, revision extension)."""
        return released, policy, {}

    def _trace_tool(self, phase, name, arguments, output, identifier):
        self.messages.append({"role": "assistant", "content": None, "phase": phase,
            "tool_calls": [{"id": identifier, "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments)}}]})
        self.messages.append({"role": "tool", "name": name, "tool_call_id": identifier,
                              "content": json.dumps(output, default=str), "phase": phase})

    def _execute(self, phase, code, workspace, identifier):
        if not phase._take_tool_call():
            return {"error": "targeted execution tool budget exhausted"}
        output = execute_python_code(code, mode="final", global_vars={"retrieved_resources": deepcopy(workspace)})
        self._trace_tool(phase.name, PYTHON_TOOL, {"code": code, "mode": "final"}, output, identifier)
        if not isinstance(output, dict):
            return {"error": "sandbox returned a malformed result"}
        if output.get("error"):
            return output
        candidate = AnalysisAgent._draft(json.dumps(output), workspace)
        return candidate if not candidate.get("error") else candidate

    def _targeted(self, patch, primary, artifact, workspace, context, complete_types, phase):
        required = {rid.split("/", 1)[0] for rid in primary["contributing_ids"]}
        if not required or not required.issubset(complete_types):
            return None, {"status": "unsupported", "reason": "targeted proof needs complete relevant type searches"}
        ordering = None
        if patch["kind"] == "ordinal":
            if not phase._take_tool_call():
                return None, {"status": "failed", "reason": "no budget for ordered source replay"}
            capture = ("\nanswer = {'replayed': {'answer': answer, 'evidence': evidence, "
                       "'contributing_ids': contributing_ids, 'operation': operation, 'limitation': limitation}, "
                       f"'ordered_ids': ['Observation/' + r['id'] for r in {patch['sorted_variable']}]}}\n")
            replay_code = artifact["code"] + capture
            output = execute_python_code(replay_code, mode="inspect", global_vars={"retrieved_resources": deepcopy(workspace)})
            self._trace_tool(phase.name, PYTHON_TOOL, {"code": replay_code, "mode": "inspect"}, output, "v3-primary-order-replay")
            data = output.get("answer") if isinstance(output, dict) and not output.get("error") else None
            if not isinstance(data, dict) or not isinstance(data.get("ordered_ids"), list):
                return None, {"status": "failed", "reason": "ordered source replay did not complete"}
            ordering = data["ordered_ids"]
            replay = AnalysisAgent._draft(json.dumps(data.get("replayed")), workspace)
        else:
            replay = self._execute(phase, artifact["code"], workspace, "v3-primary-replay")
        if replay.get("error") or fingerprint(replay) != fingerprint(primary):
            return None, {"status": "failed", "reason": "original final calculation did not reproduce on frozen records", "replay": replay}
        alternate = self._execute(phase, patch["code"], workspace, "v3-targeted-final")
        if alternate.get("error"):
            return None, {"status": "failed", "reason": "patched final contract failed", "alternate": alternate}
        if _canonical(primary["answer"], context["question"]) == _canonical(alternate["answer"], context["question"]):
            return None, {"status": "no_effect", "alternate": alternate}
        old_ids, new_ids = set(primary["contributing_ids"]), set(alternate["contributing_ids"])
        if patch["kind"] == "predicate":
            old_observations = {rid for rid in old_ids if rid.startswith("Observation/")}
            new_observations = {rid for rid in new_ids if rid.startswith("Observation/")}
            if (not re.match(r"\s*How many\b", context["question"], re.I)
                    or type(primary["answer"]) is not int or type(alternate["answer"]) is not int
                    or primary["answer"] != len(old_observations) or alternate["answer"] != len(new_observations)):
                return None, {"status": "unsupported", "reason": "predicate repair needs a direct count of contributing Observation IDs", "alternate": alternate}
        changed_ids = old_ids ^ new_ids
        if not changed_ids:
            return None, {"status": "unsupported", "reason": "answer changed without changed contributing IDs", "alternate": alternate}
        if {rid for rid in old_ids if not rid.startswith("Observation/")} != {rid for rid in new_ids if not rid.startswith("Observation/")}:
            return None, {"status": "unsupported", "reason": "repair changed non-Observation dependencies", "alternate": alternate}
        # The witness is reconstructed from held original records, not auditor claims.
        ids = changed_ids
        witnesses = []
        original_records = {}
        for rtype, items in workspace.items():
            for record in items:
                rid = f"{rtype}/{record['id']}"
                original_records[rid] = record
                if rtype != "Observation" or rid not in ids:
                    continue
                if patch["kind"] == "predicate":
                    value = record.get("valueQuantity", {}).get("value")
                    def passes(operator):
                        return {"Gt": value > patch["threshold"], "GtE": value >= patch["threshold"],
                                "Lt": value < patch["threshold"], "LtE": value <= patch["threshold"]}.get(operator)
                    if (type(value) not in (int, float) or not math.isfinite(value)
                            or passes(patch["old_operator"]) is passes(patch["new_operator"])
                            or passes(patch["old_operator"]) != (rid in old_ids)
                            or passes(patch["new_operator"]) != (rid in new_ids)):
                        return None, {"status": "unsupported", "reason": "changed source IDs do not witness the actual old/new predicate", "alternate": alternate}
                witnesses.append({"resource_id": rid, "effectiveDateTime": record.get("effectiveDateTime"),
                                  "valueQuantity": deepcopy(record.get("valueQuantity"))})
        if not witnesses:
            return None, {"status": "unsupported", "reason": "no original-field counterexample witness", "alternate": alternate}
        if patch["kind"] == "ordinal":
            removed, added = old_ids - new_ids, new_ids - old_ids
            try:
                moments = [_wall_clock(original_records[rid].get("effectiveDateTime")) for rid in ordering]
                if (len(set(ordering)) != len(ordering) or not moments or any(not item for item in moments)
                        or moments != sorted(moments) or len(removed) != 1 or len(added) != 1
                        or removed != {ordering[patch["old_index"]]} or added != {ordering[patch["new_index"]]}
                        or moments[patch["old_index"]] == moments[patch["new_index"]]):
                    raise ValueError("rank discrepancy not established")
                old_value = original_records[ordering[patch["old_index"]]].get("valueQuantity", {}).get("value")
                new_value = original_records[ordering[patch["new_index"]]].get("valueQuantity", {}).get("value")
                if (type(primary["answer"]) not in (int, float) or type(alternate["answer"]) not in (int, float)
                        or primary["answer"] != old_value or alternate["answer"] != new_value):
                    raise ValueError("rank answer is not the directly selected source quantity")
            except (KeyError, IndexError, TypeError, ValueError):
                return None, {"status": "unsupported", "reason": "changed IDs and source times do not establish the exact requested rank", "alternate": alternate}
        if not phase._take_tool_call():
            return None, {"status": "failed", "reason": "no budget for alternate source check", "alternate": alternate}
        check_error = None
        try:
            check = check_fields(alternate["evidence"], question=context["question"],
                current_time=context["current_time"], patient_fhir_id=context["patient_fhir_id"])
        except Exception as exc:
            check_error = f"source_check: {type(exc).__name__}"
            check = {"issues": [check_error], "results": []}
        self._trace_tool(phase.name, "check_fields", {"citations": alternate["evidence"]}, check, "v3-targeted-source-check")
        try:
            if (not isinstance(check, dict) or not isinstance(check.get("issues"), list)
                    or not isinstance(check.get("results"), list)
                    or len(check["results"]) != len(alternate["evidence"])):
                raise ValueError("source checker returned an incomplete result")
            issues = field_check_issues(alternate, check)
            # field_check_issues localizes per-citation findings; raw global
            # failures must not disappear when no result objects were returned.
            if check_error or any(str(item).startswith("source_check:") for item in check["issues"]):
                issues.append({"type": "source_check", "action": check_error or "source unavailable"})
        except (KeyError, TypeError, ValueError):
            issues = [{"type": "source_check", "action": check_error or "source checker result malformed/incomplete"}]
        if _empty_answer(alternate["answer"]):
            issues = [issue for issue in issues if issue["type"] != "no_evidence"]
        proof = {"status": "source_checked_local_constraint", "assurance": "local_constraint_not_global_semantic_proof",
                 "patch": patch, "source_witnesses": witnesses, "primary": primary, "alternate": alternate,
                 "changed_contributing_ids": sorted(changed_ids), "ordered_ids": ordering,
                 "workspace_sha256": fingerprint(workspace), "source_check": check, "issues": issues}
        if issues or alternate.get("limitation"):
            proof["status"] = "rejected"
            return None, proof
        return alternate, proof

    def _run(self, question):
        context = FixedController.question_context(question)
        phases, audit_error, contract, audit, proof = [], None, None, None, None
        planner = PhaseHost(self, "contract")
        contract_role = Role("intent_contract", CONTRACT_PROMPT, Toolbox(()), planner, max_turns=1, max_tool_calls=0)
        result = contract_role.run(json.dumps(context))
        if not result.error:
            contract, audit_error = parse_contract(result.content, context["question"])
        else:
            audit_error = result.error
        phases.append(planner.report())
        if planner.fatal:
            return {"answer": None, "evidence": [], "limitation": "Aggregate usage could not be verified", "error": result.error}
        contract_hash = fingerprint(contract) if contract is not None else None

        phase = PhaseHost(self, "solver")
        solver = LedgerSolver(self, phase)
        primary_result = solver._run(question)
        self.messages.extend(solver.messages)
        phases.append(phase.report())
        workspace = deepcopy(primary_result.get("retrieved_fhir_resources", {}))
        artifact = deepcopy(solver.final_computation)
        primary = solver._computed_candidate(workspace)
        primary_revision = primary_result.get("revision") or {}
        original_draft = primary_revision.get("draft") or primary
        source_issues = primary_revision.get("final_check_issues")
        if (phase.fatal or primary_result.get("error") or primary is None or primary.get("error")
                or source_issues != []):
            return {**primary_result, "answer": None, "evidence": [],
                "limitation": primary_result.get("limitation") or "No source-validated explicit final computation",
                "trace": self.messages, "revision": {"protocol": self.protocol,
                "primary": primary, "primary_revision": primary_result.get("revision"),
                "draft": original_draft, "final": {"answer": None, "evidence": [],
                    "limitation": primary_result.get("limitation") or "No source-validated explicit final computation"},
                "revised": bool(primary_revision.get("revised")),
                "answer_changed": bool(original_draft is not None and original_draft.get("answer") is not None),
                "check_issues": primary_revision.get("check_issues"),
                "solver_source_revised": bool(primary_revision.get("revised")), "targeted_revised": False,
                "retrieval_ledger": solver.retrieval_ledger,
                "complete_types": sorted(solver.complete_types()),
                "unresolved_scope": deepcopy(solver._scoped_box.scope_failures if solver._scoped_box else {}),
                "contract": contract, "contract_error": audit_error, "phases": phases,
                "final_check_issues": source_issues if source_issues else [{"type": "execution_contract", "action": "No source-validated explicit final"}],
                "release_policy": "primary_failed_execution_or_source_contract"}}
        released, policy = primary, "source_valid_primary_preserved"
        if contract is not None and len(artifact["code"]) <= 24_000:
            phase = PhaseHost(self, "audit")
            audit_role = Role("program_audit", AUDIT_PROMPT, Toolbox(()), phase,
                              max_turns=1, max_tool_calls=0)
            inventory = {rtype: {"count": len(rows), "sample": rows[:2]} for rtype, rows in workspace.items()}
            packet = {"context": context, "contract": contract, "contract_sha256": contract_hash,
                      "final_code": artifact["code"], "workspace_sha256": artifact["workspace_sha256"],
                      "inventory_preview": inventory, "retrieval_ledger": solver.retrieval_ledger}
            packet["complete_types"] = sorted(solver.complete_types())
            packet["unresolved_scope"] = deepcopy(solver._scoped_box.scope_failures if solver._scoped_box else {})
            # Code is kept intact; bound only raw data previews, not the audited program.
            packet["inventory_preview"] = json.loads(_reply(inventory))
            try:
                result = self._run_audit(audit_role, packet)
                if not result.error:
                    audit, audit_error = parse_audit(result.content, contract)
                else:
                    audit_error = result.error
            except Exception as exc:
                # Auxiliary availability/format errors cannot veto a valid P.
                audit_error = f"program audit failed: {type(exc).__name__}"
            phases.append(phase.report())
            if phase.fatal:
                return {"answer": None, "evidence": [], "limitation": "Aggregate usage could not be verified",
                        "revision": {"protocol": self.protocol, "primary": primary, "phases": phases}}
            if audit is not None:
                try:
                    patch, patch_error = patched_program(artifact["code"], audit, contract, context["question"])
                except Exception as exc:
                    patch, patch_error = None, f"repair validation failed: {type(exc).__name__}"
                if patch is not None:
                    phase = PhaseHost(self, "targeted")
                    complete_types = solver.complete_types()
                    try:
                        alternate, proof = self._targeted(patch, primary, artifact, workspace, context, complete_types, phase)
                    except Exception as exc:
                        alternate, proof = None, {"status": "failed", "reason": f"targeted check failed: {type(exc).__name__}"}
                    phases.append(phase.report())
                    if alternate is not None:
                        released, policy = alternate, "source_checked_targeted_local_constraint"
                else:
                    proof = {"status": "not_run", "reason": patch_error}
        elif contract is not None:
            audit_error = "program exceeds 24,000-character audit cap; primary preserved"
        released, policy, extension = self._post_release(released, policy, context, solver, primary_result, phases)
        workspace = deepcopy(primary_result.get("retrieved_fhir_resources", workspace))
        self.last_llm_error = None  # nonfatal auxiliary errors remain in the revision audit
        return {"answer": released["answer"], "evidence": released["evidence"], "limitation": released["limitation"],
                "trace": self.messages, "retrieved_fhir_resources": workspace,
                "revision": {**extension, "protocol": self.protocol, "draft": original_draft, "primary": primary, "final": released,
                    "revised": bool(primary_revision.get("revised")) or released is not primary,
                    "answer_changed": released["answer"] != original_draft["answer"],
                    "check_issues": primary_revision.get("check_issues"),
                    "solver_source_revised": bool(primary_revision.get("revised")), "targeted_revised": released is not primary,
                    "primary_revision": primary_result.get("revision"), "contract": contract,
                    "contract_sha256": contract_hash, "audit": audit, "audit_error": audit_error,
                    "targeted_proof": proof, "release_policy": policy, "phases": phases, "final_check_issues": [],
                    "retrieval_ledger": solver.retrieval_ledger, "complete_types": sorted(solver.complete_types()),
                    "unresolved_scope": deepcopy(solver._scoped_box.scope_failures if solver._scoped_box else {}),
                    "context_compactions": self.context_compactions}}
