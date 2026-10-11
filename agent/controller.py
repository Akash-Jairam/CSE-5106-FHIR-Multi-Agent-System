"""Fixed controller for the multi-agent system (P2-12).

No model call decides the order of work. For each question the controller:

1. parses the shared question context (question, patient, assumed current
   time, and the dataset's assumption text) and sends it with every task;
2. runs retrieval -> analysis -> verification, routing each role's
   structured result to the next over the Bus;
3. if verification reports issues, runs at most one repair starting from the
   earliest responsible role, then verifies again. The repaired answer is
   released only if that check is clean; otherwise the system abstains with
   a limitation that lists the remaining issues.

A role is a handler that maps a task object to a result object, both JSON.
Subclasses supply the handlers in build_roles(). The LLM roles are P2-13 to
P2-15, and P2-17 refines route().

Results by role:
- retrieval: any object (e.g. record IDs; records stay in the workspace)
- analysis: a draft {"answer", "evidence": [...], "limitation"}
- verification: {"issues": [{"type", "record", "field", "responsible_role", "action"}]}
Any result may instead carry {"error": "..."}, which stops the run.
"""
import re
from copy import deepcopy

from .base_agent import BaseAgent
from .framework import Bus
from utils import parse_question_context

MAX_REPAIRS = 1
ROLES = ("retrieval", "analysis", "verification")
REPAIRABLE = ("retrieval", "analysis")  # pipeline order; a repair reruns every later stage
RESULT_KINDS = {"retrieval": "records", "analysis": "draft", "verification": "verdict"}
CONTEXT_RE = re.compile(r"\nContext:(.*)\Z", re.S)


class RoleFailure(Exception):
    """A role reported an error or broke its result contract."""


class FixedController(BaseAgent):

    def build_roles(self, context, workspace):
        """{"retrieval", "analysis", "verification"} -> handler(task) -> result.

        `workspace` holds working records by resource type for the run's
        retrieved_fhir_resources. Retrieval and analysis may share it; the
        verifier should not, so its check is independent of their copies.
        """
        raise NotImplementedError

    def route(self, issues):
        """The role a repair starts from, or None if no issue names a role
        that can repair it. The earliest responsible role wins, since the
        stages after it depend on its output."""
        roles = {issue.get("responsible_role") for issue in issues}
        return next((role for role in REPAIRABLE if role in roles), None)

    @staticmethod
    def question_context(question):
        """What every role is told about the question."""
        bare, patient_fhir_id, current_time = parse_question_context(question)
        match = CONTEXT_RE.search(question)
        return {"question": bare, "patient_fhir_id": patient_fhir_id, "current_time": current_time,
                "assumptions": match.group(1).strip() if match else ""}

    def _ask(self, role, payload):
        """Send one task to a role and return its result after checking the result contract."""
        task = self._bus.send("controller", role, "task", {"context": self._context, **payload})
        try:
            result = self._roles[role](deepcopy(task.payload))
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        if isinstance(result, dict) and result.get("error"):
            self._bus.send(role, "controller", "error", {"error": str(result["error"])})
            raise RoleFailure(f"{role}: {result['error']}")
        problem = self._contract_problem(role, result)
        if problem:
            self._bus.send(role, "controller", "error", {"error": problem})
            raise RoleFailure(f"{role}: {problem}")
        try:
            message = self._bus.send(role, "controller", RESULT_KINDS[role], result)
        except ValueError as exc:
            raise RoleFailure(f"{role}: {exc}") from None
        return deepcopy(message.payload)

    @staticmethod
    def _contract_problem(role, result):
        if not isinstance(result, dict):
            return "result must be an object"
        if role == "analysis":
            if "answer" not in result:
                return "a draft must have an answer (null to abstain)"
            if not isinstance(result.get("evidence", []), list):
                return "draft evidence must be a list"
        if role == "verification":
            issues = result.get("issues")
            if not isinstance(issues, list) or not all(
                    isinstance(i, dict) and isinstance(i.get("type"), str) for i in issues):
                return "issues must be a list of objects with a type"
        return None

    def _verify(self, draft):
        return self._ask("verification", {"draft": draft})["issues"]

    def _run(self, question):
        self._context = self.question_context(question)
        self._bus = Bus()
        workspace = {}
        self._roles = self.build_roles(self._context, workspace)
        missing = [role for role in ROLES if role not in self._roles]
        if missing:
            raise ValueError(f"The controller needs handlers for: {', '.join(missing)}")

        draft = check_issues = final = final_check_issues = repair_role = error = None
        try:
            records = self._ask("retrieval", {})
            draft = self._ask("analysis", {"records": records})
            check_issues = self._verify(draft)
            final, final_check_issues = draft, check_issues
            repairs = 0
            while final_check_issues and repairs < MAX_REPAIRS:
                repair_role = self.route(final_check_issues)
                if repair_role is None:
                    break
                repairs += 1
                if repair_role == "retrieval":
                    records = self._ask("retrieval", {"issues": final_check_issues, "previous_records": records})
                final = self._ask("analysis", {"records": records, "issues": final_check_issues,
                                               "previous_draft": final})
                final_check_issues = self._verify(final)
        except RoleFailure as exc:
            error = str(exc)

        released = final if final is not None and final_check_issues == [] else None
        if released is not None:
            answer, evidence, limitation = released["answer"], released.get("evidence") or [], released.get("limitation")
        else:
            candidate = final or draft or {}
            answer, evidence = None, candidate.get("evidence") or []
            limitation = self._unverified_reason(candidate, final_check_issues, repair_role, error)
        self._bus.send("controller", "user", "final", {"answer": answer, "evidence": evidence,
                                                         "limitation": limitation})
        return {
            "answer": answer,
            "evidence": evidence,
            "limitation": limitation,
            "error": error,
            "trace": self.messages,
            "retrieved_fhir_resources": workspace,
            "revision": {
                "draft": self._summary(draft),
                "check_issues": check_issues,
                "repair_role": repair_role,
                "revised": final is not None and final is not draft,
                "answer_changed": final is not None and draft is not None and final["answer"] != draft["answer"],
                "final": self._summary(final),
                "final_check_issues": final_check_issues,
                "revision_error": error if draft is not None else None,
                "messages": self._bus.to_list(),
            },
        }

    @staticmethod
    def _summary(draft):
        return None if draft is None else {key: draft.get(key) for key in ("answer", "evidence", "limitation")}

    @staticmethod
    def _unverified_reason(candidate, issues, repair_role, error):
        if error and not candidate:
            return None
        reasons = []
        if issues:
            described = "; ".join(
                issue["type"] + "".join(f" {issue[k]}" for k in ("record", "field") if issue.get(k))
                for issue in issues)
            reasons.append("Evidence could not be verified: " + described)
            if repair_role is None:
                reasons.append("No role could repair these issues")
        if error:
            reasons.append(f"Stopped: {error}")
        reason = ". ".join(reasons)
        own = candidate.get("limitation")
        return f"{own}. {reason}" if own else reason
