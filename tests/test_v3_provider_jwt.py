"""Network-free private regression cases; no real credentials or scoring."""
from copy import deepcopy
import unittest

from scripts import v3_full_run as runner

JWT = runner.PROVIDER_JWT_REJECTION
classify_candidate = runner.classify_attempt


def row(*, error=None, revision=None, trace=None, unknown=True, answer="arbitrary"):
    return {"question_id": "fixture", "agent_answer": answer, "error": error,
            "latency": 2.973, "tokens": {"total": 1471,
                "per_call": [{"usage_available": True}, {"usage_available": not unknown}]},
            "revision": revision or {}, "trace": trace or []}


class JwtCandidateTests(unittest.TestCase):
    def status(self, value):
        return classify_candidate(value)["status"]

    def test_exact_top_provider_failure_is_runtime(self):
        self.assertEqual(self.status(row(error=JWT)), "runtime")

    def test_phase_contract_and_audit_host_failures_are_runtime(self):
        for revision in ({"phases": [{"phase": "solver", "errors": [JWT]}]},
                         {"contract_error": JWT}, {"audit_error": JWT}):
            self.assertEqual(self.status(row(revision=revision)), "runtime")

    def test_failed_provider_usage_evidence_is_required(self):
        self.assertEqual(self.status(row(error=JWT, unknown=False)), "attention")

    def test_generic_auth_and_other_jwt_messages_stay_attention(self):
        for error in ("Error code: 401", "AuthenticationError", "Invalid API key",
                      "Max retries exceeded: Incorrect API key provided",
                      "Max retries exceeded: JWT verification failed",
                      JWT + " additional unclassified detail"):
            with self.subTest(error=error):
                self.assertEqual(self.status(row(error=error)), "attention")

    def test_model_owned_fields_never_trigger_runtime(self):
        value = row(revision={"contract": {"error": JWT}, "audit": {"errors": [JWT]},
                              "code": "raise Exception('Jwt verification fails')"},
                    trace=[{"role": "assistant", "content": JWT},
                           {"role": "user", "content": JWT}], unknown=False, answer=JWT)
        self.assertEqual(self.status(value), "accepted")
        value["tokens"]["per_call"][-1]["usage_available"] = False
        self.assertEqual(self.status(value), "attention")

    def test_python_code_error_cannot_masquerade_as_provider(self):
        for error in (JWT, "RuntimeError: " + JWT):
            value = row(trace=[{"role": "tool", "name": "execute_python_code",
                                "content": {"error": error}}])
            self.assertEqual(self.status(value), "attention")

    def test_unknown_argument_echo_remains_intrinsic(self):
        value = row(trace=[{"role": "tool", "name": "get_resources_by_patient_fhir_id",
                            "content": {"error": "Unknown arguments: " + JWT}}], unknown=False)
        self.assertEqual(self.status(value), "accepted")

    def test_answers_do_not_change_recovery_decision(self):
        for answer in (None, "arbitrary", 0, False, [1, 2]):
            self.assertEqual(self.status(row(error=JWT, answer=answer)), "runtime")

    def test_structural_attention_is_not_bypassed(self):
        value = row(error=JWT)
        value["latency"] = float("nan")
        self.assertEqual(self.status(value), "attention")


if __name__ == "__main__":
    unittest.main()
