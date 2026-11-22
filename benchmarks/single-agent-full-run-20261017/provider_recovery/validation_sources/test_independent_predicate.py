"""Private, network-free regression cases for the independent proposal."""

from copy import deepcopy
import unittest

from independent_predicate import (
    MAX_ERROR_CHARS, SDK_ERROR_PREFIX, is_observed_provider_prompt_rejection,
)


def payload():
    return {"error": {
        "code": "content_filter", "param": "prompt", "status": 400,
        "type": None, "message": "Synthetic provider prompt rejection",
        "innererror": {
            "code": "ResponsibleAIPolicyViolation",
            "content_filter_result": {
                "jailbreak": {"detected": True, "filtered": True},
                "hate": {"filtered": False, "severity": "safe"},
                "self_harm": {"filtered": False, "severity": "safe"},
                "sexual": {"filtered": False, "severity": "safe"},
                "violence": {"filtered": False, "severity": "safe"},
            },
        },
    }}


def row():
    return {
        "error": SDK_ERROR_PREFIX + repr(payload()), "agent_answer": None,
        "tokens": {"per_call": [{"usage_available": False}]},
        "trace": [], "revision": None,
    }


def with_error(error):
    result = row()
    result["error"] = error
    return result


def with_calls(calls):
    result = row()
    result["tokens"]["per_call"] = calls
    return result


def with_payload_value(path, value):
    body = payload()
    node = body
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return with_error(SDK_ERROR_PREFIX + repr(body))


def exclusion_cases():
    """Each case removes trustworthy evidence or moves it to an untrusted field."""
    cases = []
    for label, error in [
        ("generic400", "HTTP 400 content_filter ResponsibleAIPolicyViolation jailbreak"),
        ("sdk_wrapper_missing", "Error code: 400 - " + repr(payload())),
        ("unobserved_bad_request_wrapper", "BadRequestError: Error code: 400 - " + repr(payload())),
        ("unowned_prefix", "Model says " + row()["error"]),
        ("trailing_prose", row()["error"] + " unknown trailing diagnostic"),
        ("python_exception_echo", "TypeError: " + row()["error"]),
        ("malformed_literal", SDK_ERROR_PREFIX + "not a dict"),
        ("code_would_execute_if_eval_used", SDK_ERROR_PREFIX + "__import__('os').system('NEVER_EXECUTED')"),
        ("excessive_depth", SDK_ERROR_PREFIX + "[" * 1000 + "]" * 1000),
        ("overlong_payload", SDK_ERROR_PREFIX + "x" * MAX_ERROR_CHARS),
        ("nonstring_error", {"code": "content_filter"}),
        ("missing_error", None),
        ("empty_error", ""),
    ]:
        cases.append((label, with_error(error)))
    for label, calls in [
        ("usage_missing", [{"other": False}]),
        ("usage_reported", [{"usage_available": True}]),
        ("usage_string_false", [{"usage_available": "false"}]),
        ("usage_zero", [{"usage_available": 0}]),
        ("usage_null", [{"usage_available": None}]),
        ("calls_missing", None),
        ("calls_empty", []),
        ("calls_not_list", {}),
        ("call_not_mapping", [None]),
        ("previous_unknown_then_clean", [{"usage_available": False}, {"usage_available": True}]),
    ]:
        cases.append((label, with_calls(calls)))
    for label, path, value in [
        ("provider_code_other", ["error", "code"], "other"),
        ("param_completion", ["error", "param"], "completion"),
        ("status_string", ["error", "status"], "400"),
        ("status_boolean", ["error", "status"], True),
        ("status_other", ["error", "status"], 401),
        ("inner_code_other", ["error", "innererror", "code"], "other"),
        ("inner_not_mapping", ["error", "innererror"], None),
        ("filters_not_mapping", ["error", "innererror", "content_filter_result"], []),
        ("jailbreak_not_mapping", ["error", "innererror", "content_filter_result", "jailbreak"], "detected"),
        ("detected_false", ["error", "innererror", "content_filter_result", "jailbreak", "detected"], False),
        ("filtered_false", ["error", "innererror", "content_filter_result", "jailbreak", "filtered"], False),
        ("detected_string", ["error", "innererror", "content_filter_result", "jailbreak", "detected"], "true"),
        ("filtered_integer", ["error", "innererror", "content_filter_result", "jailbreak", "filtered"], 1),
    ]:
        cases.append((label, with_payload_value(path, value)))
    body = payload()
    body["other"] = "Unknown provider structure"
    cases.append(("extra_root_key", with_error(SDK_ERROR_PREFIX + repr(body))))
    cases.append(("root_list", with_error(SDK_ERROR_PREFIX + repr([payload()]))))
    for label, field, value in [
        ("assistant_prose", "trace", [{"role": "assistant", "content": row()["error"]}]),
        ("user_prose", "trace", [{"role": "user", "content": row()["error"]}]),
        ("fhir_tool_echo", "trace", [{"role": "tool", "name": "fhir_request_get", "content": {"error": row()["error"]}}]),
        ("python_tool_echo", "trace", [{"role": "tool", "name": "execute_python_code", "content": {"error": row()["error"]}}]),
        ("model_answer_echo", "agent_answer", row()["error"]),
        ("model_code_echo", "code", row()["error"]),
        ("model_contract_echo", "revision", {"contract": {"error": row()["error"]}}),
        ("model_audit_echo", "revision", {"audit": {"error": row()["error"]}}),
        ("out_of_scope_contract_error", "revision", {"contract_error": row()["error"]}),
        ("out_of_scope_phase_error", "revision", {"phases": [{"errors": [row()["error"]]}]}),
        ("out_of_scope_audit_error", "revision", {"audit_error": row()["error"]}),
        ("out_of_scope_revision_error", "revision", {"revision_error": row()["error"]}),
    ]:
        result = row()
        result["error"] = None
        result[field] = value
        cases.append((label, result))
    for label, tokens in [("tokens_absent", None), ("tokens_not_mapping", [])]:
        result = row()
        result["tokens"] = tokens
        cases.append((label, result))
    assert len(cases) == 52, len(cases)
    return cases


class IndependentPredicateTests(unittest.TestCase):
    def test_exact_observed_provider_structure(self):
        self.assertTrue(is_observed_provider_prompt_rejection(row()))

    def test_answer_is_not_part_of_the_decision(self):
        for answer in (None, "arbitrary wrong answer", 0, False, [1, 2], {"error": "unrelated"}):
            with self.subTest(answer=answer):
                result = row()
                result["agent_answer"] = deepcopy(answer)
                self.assertTrue(is_observed_provider_prompt_rejection(result))

    def test_successful_earlier_calls_do_not_hide_the_final_failed_call(self):
        result = with_calls([{"usage_available": True}, {"usage_available": False}])
        self.assertTrue(is_observed_provider_prompt_rejection(result))

    def test_non_mapping_row_fails_closed(self):
        for value in (None, [], "provider error", 0):
            with self.subTest(value=value):
                self.assertFalse(is_observed_provider_prompt_rejection(value))


def add_exclusion_test(label, fixture):
    def test(self):
        self.assertFalse(is_observed_provider_prompt_rejection(deepcopy(fixture)))
    test.__name__ = "test_exclusion_" + label
    setattr(IndependentPredicateTests, test.__name__, test)


for case_label, case_fixture in exclusion_cases():
    add_exclusion_test(case_label, case_fixture)


if __name__ == "__main__":
    unittest.main()
