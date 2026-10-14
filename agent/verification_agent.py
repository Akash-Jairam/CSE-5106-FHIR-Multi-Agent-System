"""Verification role (P2-15) for the fixed controller.

Computes from a fresh, candidate-blind context. The verifier never sees the
retrieval/analysis conversations, their workspace, or their candidate answer:

1. Re-fetch cited fields: the deterministic field check (tools/field_check.py)
   re-reads every citation from the store and checks the value, the time
   field, the unit, the patient, the encounter scope, and the current time.
   As for the single agent, this check counts as one tool call.
2. Recompute: the model starts from an empty workspace, re-searches the
   patient's records within the question's scope, and recomputes the answer
   in the sandbox. It reports issues the field check cannot see, such as
   records the draft missed or a wrong calculation.
3. Merge: the field-check issues, the model's issues, and a
   `recompute_mismatch` issue if the recomputed answer differs from the draft.
   A complete executed final is frozen for the question; repaired candidates
   get new source checks against that same computation.

Every issue is {"type", "record", "field", "responsible_role", "action"}.
responsible_role is "retrieval" or "analysis", or None when no role can
repair the issue.
"""
import json
import math
import hashlib
from copy import deepcopy

from .analysis_agent import AnalysisAgent
from .framework import PYTHON_TOOL, RETRIEVAL_TOOLS, Role, Toolbox, _bound_tool_reply
from .ehr_semantics import SEMANTIC_GUIDANCE
from .output_schema import _json_objects
from tools.field_check import check_fields
from tools.resource_tools import supported_types
from utils.answer_scorer import DATETIME_RE, _parse_datetime, timestamp_precision

REPAIR_ROLES = ("retrieval", "analysis")

SYSTEM_PROMPT = f"""You are the verification agent in a team answering questions about one patient's FHIR record.
Compute an independent answer from the question and original records. The candidate answer and its reasoning
are intentionally hidden from you. The host will compare your executed answer with the candidate afterwards.

Your tools fetch the patient's records (get_resources_by_patient_fhir_id, get_resources_by_resource_id) into
`retrieved_resources`, which starts empty, and run Python over them (execute_python_code; set `answer`).
Available resource types: {', '.join(supported_types)}. Only the question's patient can be retrieved, and records
dated after the assumed current time are left out.

To inspect records or intermediate values, use execute_python_code with
mode="inspect" and set `answer` to the JSON value you need returned. Each call starts a fresh worker.
Printed output is returned only as bounded diagnostics; it never establishes a final answer.
For the final independent calculation, use mode="final" and set `answer` to the actual finite scalar, list of
answer values, or null for an established empty result. Set `evidence` to source-field citations,
`operation` to a concise account of the filters and calculation, and `contributing_ids` to the full IDs used.
Set `limitation` to null when complete, or a concise reason when records or scope cannot establish the answer.
These variables are captured directly from executed code. A diagnostic object is never a final answer.
Once the final calculation succeeds, return its result promptly. Do not run more diagnostic calculations.

Preferred inspection example (small bounded structured values):
  answer = {{"count": len(retrieved_resources.get("Observation", [])),
             "labels": [r.get("code", {{}}) for r in retrieved_resources.get("Observation", [])[:3]]}}
Use mode="inspect" for that example. Setting answer=None does not establish an empty result from that inspection.

Final metadata schema is exact. `evidence` is a list of citations; each citation has exactly
`resource_id`, `field_path`, and `value`. Use one concrete dotted/indexed field path per citation.
Do not use resource/field/fields aliases, bare IDs, comma-separated paths, or omit source values.
After independently selecting the resource and computing the question's answer, code can use:
  evidence = [{{"resource_id": selected["resourceType"] + "/" + selected["id"],
               "field_path": "valueQuantity.value", "value": selected["valueQuantity"]["value"]}}]
  contributing_ids = [r["resourceType"] + "/" + r["id"] for r in contributing_records]
  operation = "<actual scope, filters, ordering, fields and calculation>"
  limitation = None
Here `selected` and `contributing_records` must be computed from fetched records in this same invocation;
the citation field must match the field actually used by this question. Counts and empty answers still need
an explicit operation describing the completed search. Every cited resource must occur in contributing_ids.

{SEMANTIC_GUIDANCE}

1. Fetch the records the question needs yourself; do not assume an excerpt is complete.
   Fetch patient-owned events and their Encounters before fetching linked Medication or Location metadata.
2. Recompute the answer in Python, applying the question's scope (visit, time window, current time).
3. Report your executed answer. The host checks candidate citations and compares answers deterministically.

Report each problem as an issue, using one of these types where it fits (they decide which agent repairs it):
- "missing_evidence": records the answer needs were never retrieved, or the wrong visit or time window was
  searched (repaired by retrieval). Name a needed record in "record" when you can.
- "wrong_filter", "wrong_calculation", "wrong_field": the records were there but were filtered, computed, or
  cited wrongly (repaired by analysis).
Reply with only a JSON object:
{{"recomputed_answer": <the answer returned by your successful Python calculation, in the same answer format,
                       including null for an empty result>,
 "issues": [{{"type": "<one of the types above, or short_snake_case>", "record": "<ResourceType/id or null>",
             "field": "<field path or null>", "responsible_role": "retrieval" | "analysis",
             "action": "<what the responsible agent should do>"}}]}}
Reply with "issues": [] only after a successful independent fetch and final Python calculation establish your answer.
Do not claim a recomputation if a tool failed. Zero, false, or an empty list can have no cited records when your
independent search and calculation establish that empty result; missing evidence is not itself an empty result."""


def _canonical(value, question=""):
    """Comparable answers using the primary scorer's requested wall-clock precision."""
    if isinstance(value, list):
        if len(value) == 1:
            return _canonical(value[0], question)
        return tuple(sorted(repr(_canonical(v, question)) for v in value))
    if isinstance(value, bool):
        return ("bool", value)  # True == 1.0 in Python; a yes/no answer is not a count
    if value is None:
        return value
    if isinstance(value, str):
        text = value.strip()
        if DATETIME_RE.fullmatch(text):
            moment, has_time = _parse_datetime(text)
            if moment:
                precision = timestamp_precision(question)
                if precision == "day":
                    return ("timestamp", precision, moment.date().isoformat())
                if not has_time:
                    return ("timestamp", "date_only", moment.date().isoformat())
                moment = moment.replace(microsecond=0, **({"second": 0} if precision == "minute" else {}))
                return ("timestamp", precision, moment.isoformat())
        text = text.casefold()
        try:
            value = float(text)
        except ValueError:
            return text
    if isinstance(value, (int, float)) and math.isfinite(value):
        return round(float(value), 6)
    return value


def _issue(kind, record=None, field=None, responsible_role="analysis", action=""):
    return {"type": kind, "record": record, "field": field, "responsible_role": responsible_role, "action": action}


def _valid_answer(value):
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return isinstance(value, list) and all(_valid_answer(item) for item in value)


def _empty_answer(value):
    return value is None or value is False or value == [] or (
        isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0)


class VerificationToolbox(Toolbox):
    """Keep acceptance evidence from actual successful tools, not the model's claim."""

    def reset(self):
        self.fetched = False
        self.computation = None
        self.explicit_final = False
        self.final_computation = None
        self.last_execution = None
        self.last_final_call = None
        self.final_contract_error = None
        getattr(self, "scope_failures", {}).clear()

    def call(self, tool_call):
        name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except (ValueError, TypeError):
            arguments = {}
        mode = arguments.get("mode", "auto") if isinstance(arguments, dict) else "auto"
        if name == PYTHON_TOOL and mode == "final":
            self.last_final_call = deepcopy(arguments)
            self.final_contract_error = None
        if name in RETRIEVAL_TOOLS:
            self.fetched = False
            self.computation = None
            self.explicit_final = False
            self.last_final_call = None
            self.final_contract_error = None
        elif name == PYTHON_TOOL and mode != "inspect":
            self.computation = None
            self.explicit_final = False
            self.final_computation = None
            self.last_execution = None
        reply = super().call(tool_call)
        if (self.explicit_final and self.computation is not None
                and self.computation.get("workspace_sha256") != self.workspace_sha256()):
            self.computation = None
            self.explicit_final = False
        try:
            output = json.loads(reply["content"])
        except (ValueError, TypeError):
            return reply
        if not isinstance(output, dict) or "error" in output:
            if (name == PYTHON_TOOL and mode == "final" and isinstance(output, dict)
                    and output.get("error_type") == "final_contract"):
                self.final_contract_error = str(output.get("error"))
            return reply
        if name in RETRIEVAL_TOOLS:
            self.fetched = True
            # A calculation over a previous workspace cannot verify later fetched records.
            self.computation = None
        elif name == PYTHON_TOOL and self.fetched:
            # Toolbox retains the full execution envelope separately from any model preview.
            final = getattr(self, "final_computation", None)
            executed = final or getattr(self, "last_execution", None) or output
            if final is not None and _valid_answer(final.get("answer")):
                self.computation = final
                self.explicit_final = True
            elif mode != "inspect" and "answer" in executed and _valid_answer(executed["answer"]):
                self.computation = executed
        return reply


class _FinalCorrectionTools:
    """One metadata correction can execute a final calculation, never restart retrieval."""

    def __init__(self, toolbox):
        self.toolbox = toolbox
        self.definitions = [definition for definition in toolbox.definitions
                            if definition["function"]["name"] == PYTHON_TOOL]

    def call(self, tool_call):
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except (ValueError, TypeError):
            arguments = {}
        if (tool_call.function.name != PYTHON_TOOL or not isinstance(arguments, dict)
                or arguments.get("mode") != "final"):
            return self.toolbox._error(tool_call, "Metadata correction requires one execute_python_code mode=final call")
        return self.toolbox.call(tool_call)


def field_check_issues(draft, check):
    """Structured issues from a check_fields result for `draft`."""
    issues = []
    for text in check["issues"]:
        if text.startswith("encounter_scope:"):
            issues.append(_issue("encounter_scope", action=text.split(":", 1)[1].strip()))
    for result in check["results"]:
        citation = result["citation"] if isinstance(result["citation"], dict) else {}
        record, field = citation.get("resource_id"), citation.get("field_path")
        if result["status"] != "valid":
            actual = f" The value there is {json.dumps(result['actual'], default=str)}." if "actual" in result else ""
            issues.append(_issue(result["status"], record, field,
                                 action=f"Cite a field that exists in the record with its exact value.{actual}"))
        for warning in result["warnings"]:
            kind, _, detail = warning.partition(":")
            issues.append(_issue(kind.strip(), record, field, action=detail.strip()))
    if draft.get("answer") is not None and not draft.get("evidence"):
        issues.append(_issue("no_evidence", action="Cite the fields that support the answer"))
    return issues


class VerificationAgent:
    """Handler for the controller's verification tasks. Owns its workspace; nothing is shared."""

    def __init__(self, host, context, max_turns=10, require_explicit_final=False):
        self.host = host
        self.context = context
        self.workspace = {}
        self.checks = 0
        self.require_explicit_final = require_explicit_final
        self._independent_cache = None
        self.toolbox = VerificationToolbox(RETRIEVAL_TOOLS + (PYTHON_TOOL,), self.workspace,
                                           patient_fhir_id=context.get("patient_fhir_id"),
                                           current_time=context.get("current_time"))
        self.role = Role("verification", SYSTEM_PROMPT, self.toolbox, host, max_turns=max_turns,
                         max_tool_calls=11)

    def __call__(self, task):
        draft = task["draft"]
        cached = self._independent_cache
        if cached is not None:
            # The independent source computation belongs to this question, not to a particular draft.
            # Recheck repaired citations from the store without recomputing a correlated new answer.
            issues = self._field_check(draft)
            if issues is None:
                return {"issues": [_issue("source_check", responsible_role=None,
                                          action="The tool-call budget did not allow the evidence check")]}
            if cached["identity"] != self._independent_identity():
                return self._failed_verdict(issues, "The independent workspace or question scope changed")
            computation = deepcopy(cached["computation"])
            verdict = {"recomputed_answer": computation["answer"], "issues": []}
            return self._finish(draft, issues, verdict, computation, False, {}, True, reused=True)
        self.workspace.clear()
        self.toolbox.reset()
        issues = self._field_check(draft)
        if issues is None:
            return {"issues": [_issue("source_check", responsible_role=None,
                                      action="The tool-call budget did not allow the evidence check")]}
        if any(i["responsible_role"] is None for i in issues):
            return {"issues": issues, "recomputed_answer": None}
        budget_stop_before = self.host.budget_stop
        role_start_calls = self.host.tool_calls_made
        result = self.role.run(self._prompt())
        prior_independent_issues = []
        if not result.error and not self.host.budget_stop:
            problem = self.toolbox.final_contract_error
            if self.toolbox.explicit_final and self.toolbox.computation is not None:
                _, problem = self._final_artifact(self.toolbox.computation)
            if problem and self.host.tool_calls_made - role_start_calls < self.role.max_tool_calls:
                prior_independent_issues = self._reported_issues(result.content)
                result = self._repair_final_metadata(problem)
        refused_budget = self.host.budget_stop and self.host.budget_stop != budget_stop_before
        if refused_budget:
            issues.append(_issue("source_check", responsible_role=None,
                                 action=f"Verification did not complete all requested checks: {self.host.budget_stop}"))
        unresolved = getattr(self.toolbox, "scope_failures", {})
        for record, reason in sorted(unresolved.items()):
            issues.append(_issue("source_check", record=record, responsible_role=None,
                                 action=f"The source record's {reason.replace('_', ' ')} prevents establishing the answer's scope"))
        if result.error:
            return self._failed_verdict(issues, result.error)
        computation = self.toolbox.computation
        if self.require_explicit_final and not self.toolbox.explicit_final:
            return self._failed_verdict(issues, "Independent verification requires a validated mode=final execution; diagnostics cannot certify an answer")
        # Explicit finalization makes the sandbox result authoritative. Formatting a later model reply
        # cannot discard an already executed result; scope, source and budget gates still apply below.
        if self.toolbox.explicit_final and computation is not None:
            independent, problem = self._final_artifact(computation)
            if problem:
                return self._failed_verdict(issues, problem)
            independent_issues = self._field_check(independent)
            if independent_issues is None:
                return self._failed_verdict(issues + [_issue("source_check", responsible_role=None,
                    action="The tool-call budget did not allow checking independent source evidence")],
                    "Independent source evidence was not verified")
            if _empty_answer(independent["answer"]):
                independent_issues = [issue for issue in independent_issues if issue["type"] != "no_evidence"]
            for issue in independent_issues:
                issues.append(_issue("independent_evidence", issue.get("record"), issue.get("field"),
                                     responsible_role=None,
                                     action=f"{issue['type']}: {issue.get('action', '')}"))
            # A valid warning about the independent search remains material after a metadata correction.
            verdict = {"recomputed_answer": computation["answer"],
                       "issues": prior_independent_issues + self._reported_issues(result.content)}
            if computation.get("limitation"):
                verdict["issues"].append(_issue("independent_check", responsible_role=None,
                                               action=str(computation["limitation"])))
        else:
            verdict = next((obj for obj in _json_objects(result.content or "") if "issues" in obj), None)
        if verdict is None:
            return self._failed_verdict(issues, "the reply had no JSON verdict with issues")
        if not isinstance(verdict["issues"], list) or not all(
                isinstance(i, dict) and isinstance(i.get("type"), str) and i["type"].strip()
                for i in verdict["issues"]):
            return self._failed_verdict(issues, "verdict issues must be a list of objects with a type")
        if "recomputed_answer" not in verdict or not _valid_answer(verdict["recomputed_answer"]):
            return self._failed_verdict(issues, "the verdict must include a finite scalar, list, or null recomputed_answer")
        if (self.toolbox.explicit_final and computation is not None and not unresolved and not refused_budget
                and not any(i["responsible_role"] is None for i in issues)
                and not verdict["issues"]):
            self._independent_cache = {"identity": self._independent_identity(),
                                       "computation": deepcopy(computation)}
        return self._finish(draft, issues, verdict, computation, refused_budget, unresolved,
                            self.toolbox.explicit_final)

    def _final_artifact(self, computation):
        computed = {key: computation.get(key) for key in
                    ("answer", "evidence", "operation", "contributing_ids", "limitation")}
        independent = AnalysisAgent._draft(json.dumps(computed, default=str), self.workspace)
        if independent.get("error"):
            return None, "Invalid independent final artifact: " + independent["error"]
        if not _empty_answer(independent["answer"]) and not independent["evidence"]:
            return None, "A nonempty independent answer requires source evidence"
        return independent, None

    @staticmethod
    def _reported_issues(content):
        reported = next((obj for obj in _json_objects(content or "") if "issues" in obj), None)
        if (reported is not None and isinstance(reported["issues"], list) and all(
                isinstance(item, dict) and isinstance(item.get("type"), str) and item["type"].strip()
                for item in reported["issues"])):
            return reported["issues"]
        return []

    def _repair_final_metadata(self, problem):
        previous = (self.toolbox.last_final_call or {}).get("code", "")
        feedback = _bound_tool_reply({"content": json.dumps({"metadata_error": problem,
                "previous_independent_code": previous,
                "workspace_counts": {rtype: len(records) for rtype, records in self.workspace.items()}})})["content"]
        prompt = self._prompt() + "\n\n" + (
            "The independent final execution had a metadata-format problem. You have ONE counted final Python "
            "correction, with the same fetched workspace. Repair the evidence/contributor schema while preserving "
            "your independently selected scope and calculation. Each invocation starts fresh: rerun self-contained "
            "code and set answer, evidence, contributing_ids, operation, and limitation. Do not fetch records or "
            "use inspect mode. The candidate remains hidden.\n" + feedback)
        correction = Role("verification_format", SYSTEM_PROMPT, _FinalCorrectionTools(self.toolbox), self.host,
                          max_turns=1, max_tool_calls=1)
        return correction.run(prompt)

    def _independent_identity(self):
        return hashlib.sha256(json.dumps({"context": self.context, "workspace": self.workspace},
                                         sort_keys=True, allow_nan=False, default=str).encode()).hexdigest()

    def _finish(self, draft, issues, verdict, computation, refused_budget, unresolved, explicit_final, reused=False):
        for item in verdict["issues"]:
            role = item.get("responsible_role")
            issues.append(_issue(item["type"].strip(), item.get("record"), item.get("field"),
                                 role if role in REPAIR_ROLES else None, str(item.get("action") or "")))
        recomputed = verdict.get("recomputed_answer")
        question = self.context["question"]
        if computation is None:
            issues.append(_issue("independent_check", responsible_role=None,
                                 action="Verification needs a successful independent fetch and Python recomputation"))
        elif _canonical(computation["answer"], question) != _canonical(recomputed, question):
            issues.append(_issue("independent_check", responsible_role=None,
                                 action="The reported recomputation differs from the successful Python result"))
        elif (not unresolved and not refused_budget and _empty_answer(recomputed)
              and _canonical(recomputed, question) == _canonical(draft.get("answer"), question)):
            issues = [i for i in issues if i["type"] != "no_evidence"]
        if computation is not None and _canonical(computation["answer"], question) != _canonical(draft.get("answer"), question):
            issues.append(_issue("recompute_mismatch", action=(
                f"The verifier recomputed {json.dumps(computation['answer'])}; the draft answered "
                f"{json.dumps(draft.get('answer'))}. Recheck the filters and calculation.")))
        unique = {}
        for issue in issues:
            key = (issue["type"], json.dumps(issue["record"]), json.dumps(issue["field"]))
            # Deterministic findings precede model findings and cannot be rerouted by them.
            if key not in unique or issue["responsible_role"] is None:
                unique[key] = issue
        result = {"issues": list(unique.values()), "recomputed_answer": recomputed}
        if explicit_final and computation is not None:
            # Metadata comes from code, never the candidate or a model-written final summary. It allows
            # the controller to inspect an independent alternative without changing its release policy.
            result["independent_computation"] = {
                key: computation.get(key) for key in
                ("answer", "evidence", "operation", "contributing_ids", "limitation", "workspace_sha256")}
            result["independent_candidate"] = {
                key: computation.get(key) for key in ("answer", "evidence", "operation", "contributing_ids", "limitation")}
            result["independent_check_reused"] = reused
        return result

    @staticmethod
    def _failed_verdict(issues, reason):
        return {"issues": issues + [_issue("verification_failure", responsible_role=None, action=reason)],
                "recomputed_answer": None}

    def _field_check(self, draft):
        """check_fields on the draft's citations, recorded in the trace like the single agent's check;
        None if the tool-call budget is used up."""
        if not self.host._take_tool_call():
            return None
        self.checks += 1
        tool_id = f"verification_field_check_{self.checks}"
        arguments = {"citations": draft.get("evidence") or []}
        self.host.messages.append({"role": "assistant", "content": None, "tool_calls": [{
            "id": tool_id, "type": "function",
            "function": {"name": "check_fields", "arguments": json.dumps(arguments, default=str)}}]})
        try:
            check = check_fields(arguments["citations"], question=self.context["question"],
                                 current_time=self.context.get("current_time"),
                                 patient_fhir_id=self.context.get("patient_fhir_id"))
        except Exception as exc:
            check = {"error": f"Source evidence check failed ({type(exc).__name__})"}
        self.host.messages.append({"role": "tool", "tool_call_id": tool_id, "name": "check_fields",
                                   "content": json.dumps(check, default=str)})
        if "error" in check:
            return [_issue("source_check", responsible_role=None, action=check["error"])]
        return field_check_issues(draft, check)

    def _prompt(self):
        return "\n".join([
            f"Question: {self.context['question']}", f"Context: {self.context.get('assumptions', '')}", "",
            f"Patient FHIR ID: {self.context.get('patient_fhir_id', '')}",
            f"Assumed current time: {self.context.get('current_time', '')}", "",
            "Independently retrieve the original records and compute the answer. No candidate is provided.",
            "Use mode=inspect for diagnostics and mode=final for the complete answer with executed citations."])
