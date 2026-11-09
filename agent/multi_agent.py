"""The multi-agent system: the fixed controller (P2-12) with its three LLM roles.

Retrieval (P2-13) and analysis (P2-14) share one workspace; the verifier
(P2-15) starts from a copy of it. All roles share the host's model, token
ceiling, and tool-call ceiling.

- Retrieval fetches the complete records of the types the question needs and
  also reads the starting packet once without tools (the `direct` answer).
- Analysis computes the draft in Python with the packet, worked examples and
  EHR conventions in its prompt.
- Verification re-checks the cited fields and recomputes the answer; its
  issues route one repair, as before, but never withhold an answer.

Release is a vote among the answers the team produced: the original draft,
the repaired answer, the packet reading and the verifier's recomputation.
Abstentions do not vote; ties go to the earlier voter, so a repair prompted
by an unreliable verifier does not displace the draft on a tie. A deterministic finisher (agent/finishers.py) then applies the
benchmark's conventions for the seven templates it covers. A role failure
after a draft or packet answer exists is recorded, not the run's error.
"""
from .analysis_agent import AnalysisAgent
from .controller import FixedController
from .finishers import Fetcher, run_finisher
from .retrieval_agent import RetrievalAgent
from .verification_agent import VerificationAgent, _canonical

# Verdicts whose recomputation is not an independent successful calculation
FAILED_CHECKS = {"independent_check", "verification_failure", "source_check"}


class MultiAgentSystem(FixedController):

    def build_roles(self, context, workspace):
        self._recomputed = []
        self._retrieval = RetrievalAgent(self, context, workspace, max_turns=4, prefetch=True, direct=True)
        return {"retrieval": self._retrieval,
                "analysis": AnalysisAgent(self, context, workspace, packet=lambda: self._retrieval.packet),
                "verification": VerificationAgent(self, context, seed=lambda: workspace)}

    def _verify(self, draft):
        verdict = self._ask("verification", {"draft": draft})
        if not any(issue["type"] in FAILED_CHECKS for issue in verdict["issues"]):
            self._recomputed.append(verdict.get("recomputed_answer"))
        return verdict["issues"]

    def candidates(self, state):
        """(voter, candidate) pairs in tie-break order."""
        draft, final = state["draft"], state["final"]
        pairs = []
        if draft is not None:
            pairs.append(("draft", draft))
        if final is not None and final is not draft and state["revised"]:
            pairs.append(("repaired", final))
        direct = self._retrieval.direct_candidate
        if direct is not None:
            pairs.append(("direct", direct))
        if self._recomputed:
            pairs.append(("verifier", {"answer": self._recomputed[-1], "evidence": []}))
        return pairs

    def release(self, state):
        question = self._context["question"]
        pairs = self.candidates(state)
        votes = {}
        for voter, candidate in pairs:
            if candidate.get("answer") is None:
                continue
            key = repr(_canonical(candidate["answer"], question))
            votes.setdefault(key, []).append((voter, candidate))
        details = {"candidates": {voter: c.get("answer") for voter, c in pairs}, "winner": None, "finisher": None}
        if not votes:
            # Nothing to release: a failure before any answer stays the run's (retryable) error
            return None, state["error"] if not pairs else None, details
        # Most votes; among equals, the group holding the earliest voter
        order = {voter: index for index, (voter, _) in enumerate(pairs)}
        group = max(votes.values(), key=lambda g: (len(g), -min(order[v] for v, _ in g)))
        voter, candidate = min(group, key=lambda item: order[item[0]])
        # A computed candidate's citations, else any agreeing candidate's
        cited = next((c for _, c in group if c.get("evidence")), candidate)
        released = {**candidate, "evidence": cited.get("evidence") or [],
                    "contributing_ids": cited.get("contributing_ids") or []}
        details.update(winner=voter, votes=len(group))

        fetcher = Fetcher(self._take_tool_call, self._context.get("patient_fhir_id"), self._context.get("current_time"))
        finished = run_finisher(self._context, released, fetcher)
        details["finisher"] = {k: finished[k] for k in ("template", "finisher", "reason")}
        if finished["finisher"] == "replaced":
            released = {**released, "answer": finished["result"], "evidence": finished["evidence"]}
        # A failed model call after an answer exists (e.g. in verification) must not void the answer
        self.last_llm_error = None
        return released, None, details
