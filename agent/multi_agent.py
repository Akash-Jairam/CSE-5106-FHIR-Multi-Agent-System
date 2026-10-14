"""The multi-agent system: the fixed controller (P2-12) with its three LLM roles.

Retrieval (P2-13) and analysis (P2-14) share one workspace; the verifier
(P2-15) keeps its own, so its check does not depend on their copies. All
roles share the host's model, token ceiling, and tool-call ceiling.
"""
from .analysis_agent import AnalysisAgent
from .controller import FixedController
from .retrieval_agent import RetrievalAgent
from .verification_agent import VerificationAgent


class MultiAgentSystem(FixedController):

    def build_roles(self, context, workspace):
        return {"retrieval": RetrievalAgent(self, context, workspace),
                "analysis": AnalysisAgent(self, context, workspace),
                "verification": VerificationAgent(self, context, require_explicit_final=True)}
