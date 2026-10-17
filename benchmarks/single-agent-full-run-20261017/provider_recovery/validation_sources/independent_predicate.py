"""Independent, offline-only proposal for the observed Azure prompt rejection.

This file is not imported by the live benchmark. It recognizes only the exact
host error wrapper observed in the saved attempts. Model answers and trace or
revision content do not participate in the decision.
"""

import ast


SDK_ERROR_PREFIX = "Max retries exceeded: Error code: 400 - "
MAX_ERROR_CHARS = 8192


def is_observed_provider_prompt_rejection(row):
    """Return whether a row contains the narrowly observed host rejection.

    The failed final provider call must lack reported usage. The SDK exception's
    Python dictionary representation is parsed with literal_eval, never eval.
    Unrecognized formats fail closed and remain the live classifier's concern.
    """
    if not isinstance(row, dict):
        return False
    error = row.get("error")
    tokens = row.get("tokens")
    calls = tokens.get("per_call") if isinstance(tokens, dict) else None
    if not (
        isinstance(error, str)
        and error.startswith(SDK_ERROR_PREFIX)
        and len(error) <= MAX_ERROR_CHARS
        and isinstance(calls, list)
        and calls
        and isinstance(calls[-1], dict)
        and calls[-1].get("usage_available") is False
    ):
        return False
    try:
        payload = ast.literal_eval(error[len(SDK_ERROR_PREFIX):])
    except (ValueError, SyntaxError, TypeError, RecursionError):
        return False
    if not (
        isinstance(payload, dict)
        and set(payload) == {"error"}
        and isinstance(payload["error"], dict)
    ):
        return False
    provider_error = payload["error"]
    if not (
        provider_error.get("code") == "content_filter"
        and provider_error.get("param") == "prompt"
        and type(provider_error.get("status")) is int
        and provider_error["status"] == 400
    ):
        return False
    inner = provider_error.get("innererror")
    if not (
        isinstance(inner, dict)
        and inner.get("code") == "ResponsibleAIPolicyViolation"
    ):
        return False
    filters = inner.get("content_filter_result")
    jailbreak = filters.get("jailbreak") if isinstance(filters, dict) else None
    return (
        isinstance(jailbreak, dict)
        and jailbreak.get("detected") is True
        and jailbreak.get("filtered") is True
    )
