"""MODEL-FREE diagnostics: synthetic settings, default SDK transport, no send."""
import builtins
from datetime import datetime, timezone
import errno
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "local_diagnostic_test_adapter", ROOT / ".local-setup/azure_adapter.py")
adapter = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {spec.name: adapter}):
    spec.loader.exec_module(adapter)

POISON = "PRIVATE_SENTINEL_MUST_NEVER_BE_JOURNALED"


def all_values(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from all_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from all_values(child)
    else:
        yield value


def values_for_key(value, key):
    if isinstance(value, dict):
        for name, child in value.items():
            if name == key:
                yield child
            yield from values_for_key(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from values_for_key(child, key)


class AzureDiagnosticJournalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "diagnostics"
        environment = patch.dict(os.environ, {
            "FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR": str(self.directory)}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.settings = adapter.AzureSettings(
            endpoint="https://PRIVATE-ENDPOINT.example.invalid",
            api_version="synthetic-version", deployment="synthetic-deployment",
            model="synthetic-model", max_completion_tokens=1234,
            reasoning_effort="none", api_key=POISON + "_KEY")
        for target in ((adapter.AzureSettings, "load"), (socket, "getaddrinfo"),
                       (socket.socket, "connect")):
            blocked = patch.object(*target, side_effect=AssertionError(
                "MODEL-FREE fixture attempted configuration or network access"))
            blocked.start()
            self.addCleanup(blocked.stop)
        for name in ("read_text", "read_bytes"):
            original = getattr(Path, name)

            def safe_read(path, *args, _original=original, **kwargs):
                if path.name.startswith("config") and path.suffix in {".yml", ".yaml"}:
                    raise AssertionError("Configuration contents must never be read")
                return _original(path, *args, **kwargs)

            guard = patch.object(Path, name, safe_read)
            guard.start()
            self.addCleanup(guard.stop)

    def make_default_client(self):
        import httpx2 as httpx
        import openai
        client = adapter.make_client(self.settings)
        self.addCleanup(client.close)
        self.assertIsInstance(client, openai.AzureOpenAI)
        self.assertIsInstance(client._client, httpx.Client)
        self.assertIsInstance(client._client._transport, httpx.HTTPTransport)
        self.assertTrue(hasattr(client._client._transport, "_pool"))
        self.assertEqual(client.timeout, 60.0)
        self.assertEqual(client.max_retries, 0)
        return client

    def invoke(self, client, **overrides):
        messages = [{"role": "user", "content": POISON + "_MESSAGE"}]
        arguments = {"model": self.settings.model, "messages": messages}
        arguments.update(overrides)
        litellm = ModuleType("litellm")
        litellm.ModelResponse = Mock(side_effect=lambda **payload: payload)
        with patch.dict(sys.modules, {"litellm": litellm}), \
                patch.object(adapter, "_settings", self.settings), \
                patch.object(adapter, "_client", client):
            return adapter.azure_completion(**arguments)

    def records(self):
        paths = sorted(self.directory.glob("azure-diagnostics-*.jsonl"))
        self.assertTrue(paths, "enabled diagnostics did not create a per-process journal")
        records = []
        for path in paths:
            text = path.read_text()
            self.assertNotIn(POISON, text)
            self.assertNotIn("PRIVATE-ENDPOINT", text)
            self.assertNotIn("Authorization", text)
            for line in text.splitlines():
                record = json.loads(line)
                self.assertIn(str(record["pid"]), path.name)
                self.assertIn(record["process_session"], path.name)
                records.append(record)
        return records

    def assert_pair(self, records, outcome):
        self.assertEqual(len(records), 2)
        start, end = records
        self.assertEqual([start["event"], end["event"]], ["start", "end"])
        for name in ("pid", "ppid", "counter", "process_session", "started_utc"):
            self.assertEqual(start[name], end[name])
        self.assertIsInstance(start["pid"], int)
        self.assertIsInstance(start["ppid"], int)
        self.assertIsInstance(start["counter"], int)
        self.assertGreater(start["counter"], 0)
        self.assertTrue(start["process_session"])
        started = datetime.fromisoformat(start["started_utc"].replace("Z", "+00:00"))
        finished = datetime.fromisoformat(end["finished_utc"].replace("Z", "+00:00"))
        self.assertEqual(started.utcoffset(), timezone.utc.utcoffset(started))
        self.assertEqual(finished.utcoffset(), timezone.utc.utcoffset(finished))
        self.assertGreaterEqual(finished, started)
        self.assertIsInstance(end["duration_seconds"], (int, float))
        self.assertNotIsInstance(end["duration_seconds"], bool)
        self.assertGreaterEqual(end["duration_seconds"], 0)
        self.assertEqual(end["outcome"], outcome)
        return start, end

    def synthetic_response(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18,
                                  prompt_tokens_details=SimpleNamespace(cached_tokens=3)),
            status_code=201,
            headers={"x-request-id": "safe-request-id", "x-ms-request-id": "safe-ms-id",
                     "apim-request-id": "safe-apim-id", "Authorization": POISON + "_HEADER",
                     "x-private-id": POISON + "_NONALLOWLISTED"},
            _request_id="safe-request-id", request_id="safe-request-id",
            body=POISON + "_BODY", content=POISON + "_CONTENT")
        response.model_dump = Mock(return_value={"result": POISON + "_RETURNED_CONTENT"})
        return response

    def test_unset_directory_has_no_journal_filesystem_effect_and_keeps_exact_request(self):
        client = self.make_default_client()
        response = self.synthetic_response()
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(Path, "mkdir", side_effect=AssertionError("journal mkdir while disabled")), \
                patch.object(Path, "open", side_effect=AssertionError("journal open while disabled")), \
                patch.object(builtins, "open", side_effect=AssertionError("journal open while disabled")), \
                patch.object(client.chat.completions, "create", return_value=response) as create:
            self.assertIsNone(adapter._diagnostic_start(client))
            result = self.invoke(client, max_tokens=17, temperature=0.7,
                                 tools=[{"type": "function", "function": {"name": "fixture_tool"}}],
                                 parallel_tool_calls=False, tool_choice="auto",
                                 response_format={"type": "json_object"}, stop=[], seed=0)
        create.assert_called_once_with(
            model="synthetic-deployment", messages=[{"role": "user", "content": POISON + "_MESSAGE"}],
            max_completion_tokens=17, reasoning_effort="none",
            tools=[{"type": "function", "function": {"name": "fixture_tool"}}],
            parallel_tool_calls=False, tool_choice="auto",
            response_format={"type": "json_object"}, stop=[], seed=0)
        self.assertEqual(result, {"result": POISON + "_RETURNED_CONTENT"})
        self.assertFalse(self.directory.exists())

    def test_installed_default_sdk_success_parses_in_memory_response_without_network(self):
        import httpx2 as httpx
        client = self.make_default_client()
        payload = {"id": POISON + "_RESPONSE_ID", "object": "chat.completion", "created": 0,
                   "model": "synthetic-model", "choices": [{"index": 0, "finish_reason": "stop",
                   "message": {"role": "assistant", "content": POISON + "_RESPONSE_CONTENT"}}],
                   "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18,
                             "prompt_tokens_details": {"cached_tokens": 3}}}
        captured = []

        def send(request, **kwargs):
            captured.append((request, kwargs))
            return httpx.Response(200, json=payload, request=request,
                                  headers={"x-request-id": "safe-sdk-request-id",
                                           "Authorization": POISON + "_RESPONSE_HEADER"})

        with patch.object(client._client, "send", side_effect=send) as send_call:
            result = self.invoke(client, max_completion_tokens=9000, temperature=0.7)
        send_call.assert_called_once()
        request, _ = captured[0]
        self.assertEqual(json.loads(request.content), {
            "model": "synthetic-deployment", "messages": [{"role": "user", "content": POISON + "_MESSAGE"}],
            "max_completion_tokens": 1234, "reasoning_effort": "none"})
        self.assertEqual(result["choices"][0]["message"]["content"], POISON + "_RESPONSE_CONTENT")
        start, end = self.assert_pair(self.records(), "success")
        self.assertEqual(start["pid"], os.getpid())
        self.assertEqual(start["ppid"], os.getppid())
        serialized = json.dumps([start, end])
        for instance in (client, client._client, client._client._transport,
                         client._client._transport._pool):
            self.assertIn(type(instance).__module__ + "." + type(instance).__qualname__, serialized)
        self.assertTrue(list(values_for_key([start, end], "mro")))
        self.assertEqual(start["client"]["sdk_closed"], False)
        self.assertEqual(start["client"]["http_closed"], False)
        self.assertIn("openai", serialized)
        self.assertIn("httpx", serialized)
        self.assertIn("safe-sdk-request-id", serialized)

    def test_safe_success_metadata_contains_only_numeric_usage_and_allowlisted_ids(self):
        client = self.make_default_client()
        response = self.synthetic_response()
        response.usage.extra = POISON + "_USAGE_EXTRA"
        state = adapter._diagnostic_start(client)
        adapter._diagnostic_finish(state, client, response=response)
        _, end = self.assert_pair(self.records(), "success")
        # A parsed SDK completion exposes its request ID, but no HTTP response.
        self.assertIsNone(end["http_status"])
        self.assertEqual(end["request_ids"], {"x-request-id": "safe-request-id"})
        self.assertEqual(set(end["usage"].values()), {11, 7, 18, 3})
        for value in end["usage"].values():
            self.assertIsInstance(value, (int, float))
            self.assertNotIsInstance(value, bool)
        self.assertEqual(len(end["usage"]), 4)
        for record in self.records():
            counts = list(values_for_key(record, "fd_count"))
            self.assertTrue(counts)
            self.assertTrue(all(value is None or type(value) is int for value in counts))

    def test_installed_sdk_connection_error_preserves_identical_exception_and_safe_chain(self):
        import httpx2 as httpx
        import openai
        client = self.make_default_client()
        request = httpx.Request("POST", self.settings.endpoint,
                                headers={"Authorization": POISON + "_REQUEST_HEADER"},
                                content=POISON + "_REQUEST_BODY")
        transport_error = httpx.ConnectError(POISON + "_TRANSPORT_EXCEPTION", request=request)
        transport_error.__cause__ = OSError(errno.EMFILE, POISON + "_OS_EXCEPTION")
        actual_create = client.chat.completions.create
        captured = []

        def create(**kwargs):
            try:
                return actual_create(**kwargs)
            except Exception as error:
                captured.append(error)
                raise

        with patch.object(client._client, "send", side_effect=transport_error) as send, \
                patch.object(client.chat.completions, "create", side_effect=create), \
                self.assertRaises(openai.APIConnectionError) as raised:
            self.invoke(client)
        send.assert_called_once()
        self.assertIs(raised.exception, captured[0])
        self.assertIs(raised.exception.__cause__, transport_error)
        _, end = self.assert_pair(self.records(), "error")
        self.assertIn("openai.APIConnectionError", json.dumps(end))
        self.assertIn(type(transport_error).__module__ + "." + type(transport_error).__qualname__, json.dumps(end))
        self.assertIn(errno.EMFILE, list(all_values(end)))
        self.assertIn(True, list(values_for_key(end, "resource_fd")))
        self.assertIn(True, list(values_for_key(end, "connect")))

    def test_error_chain_is_bounded_and_rejects_unsafe_numeric_fields(self):
        client = self.make_default_client()
        errors = [RuntimeError(POISON + str(index)) for index in range(12)]
        for current, cause in zip(errors, errors[1:]):
            current.__cause__ = cause
        errors[0].errno = POISON + "_ERRNO"
        errors[0].winerror = POISON + "_WINERROR"
        errors[0].verify_code = POISON + "_TLS_CODE"
        errors[0].request = SimpleNamespace(headers={"Authorization": POISON}, content=POISON)
        state = adapter._diagnostic_start(client)
        adapter._diagnostic_finish(state, client, error=errors[0])
        _, end = self.assert_pair(self.records(), "error")
        self.assertLessEqual(len(end["exception_chain"]), 8)
        self.assertGreater(len(end["exception_chain"]), 0)
        for entry in end["exception_chain"]:
            for field in ("errno", "winerror", "verify_code"):
                if field in entry:
                    self.assertTrue(entry[field] is None or type(entry[field]) is int)
        self.assertNotIn("traceback", json.dumps(end).lower())

    def test_exception_causes_context_suppression_and_cycles_follow_python_semantics(self):
        client = self.make_default_client()
        explicit = ValueError(POISON)
        explicit.__cause__ = FileNotFoundError(errno.ENOENT, POISON)
        explicit.__context__ = TypeError(POISON)
        unsuppressed = RuntimeError(POISON)
        unsuppressed.__context__ = LookupError(POISON)
        suppressed = RuntimeError(POISON)
        suppressed.__context__ = OSError(POISON)
        suppressed.__suppress_context__ = True
        cyclic = ValueError(POISON)
        cyclic.__cause__ = TypeError(POISON)
        cyclic.__cause__.__cause__ = cyclic
        cases = [(explicit, ["builtins.ValueError", "builtins.FileNotFoundError"]),
                 (unsuppressed, ["builtins.RuntimeError", "builtins.LookupError"]),
                 (suppressed, ["builtins.RuntimeError"]),
                 (cyclic, ["builtins.ValueError", "builtins.TypeError"])]
        for error, expected in cases:
            with self.subTest(expected=expected):
                state = adapter._diagnostic_start(client)
                adapter._diagnostic_finish(state, client, error=error)
                _, end = self.assert_pair(self.records()[-2:], "error")
                self.assertEqual([item["type"]["class"] for item in end["exception_chain"]], expected)

    def test_sdk_http_error_exports_only_status_and_allowlisted_request_ids(self):
        import httpx2 as httpx
        import openai
        client = self.make_default_client()
        response = httpx.Response(429, request=httpx.Request("POST", self.settings.endpoint,
                                  headers={"Authorization": POISON + "_REQUEST_HEADER"}),
                                  headers={"x-request-id": "safe-request-id", "x-ms-request-id": "safe-ms-id",
                                           "apim-request-id": "safe-apim-id", "x-private-id": POISON,
                                           "Authorization": POISON}, json={"private": POISON})
        error = openai.RateLimitError(POISON + "_EXCEPTION", response=response,
                                     body={"private": POISON + "_BODY"})
        with patch.object(client.chat.completions, "create", side_effect=error), \
                self.assertRaises(openai.RateLimitError) as raised:
            self.invoke(client)
        self.assertIs(raised.exception, error)
        _, end = self.assert_pair(self.records(), "error")
        self.assertEqual(end["http_status"], 429)
        self.assertEqual(end["request_ids"], {"x-request-id": "safe-request-id",
                         "x-ms-request-id": "safe-ms-id", "apim-request-id": "safe-apim-id"})
        self.assertTrue(all(value is None for value in end["usage"].values()))

    def test_invalid_usage_and_request_ids_never_export_unsafe_values(self):
        client = self.make_default_client()
        response = self.synthetic_response()
        response.usage = SimpleNamespace(prompt_tokens=POISON, completion_tokens=True, total_tokens=-2,
                                         prompt_tokens_details=SimpleNamespace(cached_tokens=POISON))
        response._request_id = POISON + " with spaces and unsafe punctuation/"
        state = adapter._diagnostic_start(client)
        adapter._diagnostic_finish(state, client, response=response)
        _, end = self.assert_pair(self.records(), "success")
        self.assertTrue(all(value is None for value in end["usage"].values()))
        self.assertEqual(end["request_ids"], {})

    def test_each_append_closes_its_file_before_start_or_finish_returns(self):
        client = self.make_default_client()
        opened = []
        original_path_open = Path.open
        original_open = builtins.open

        def track(handle, path):
            if str(path).endswith(".jsonl"):
                opened.append(handle)
            return handle

        def path_open(path, *args, **kwargs):
            return track(original_path_open(path, *args, **kwargs), path)

        def plain_open(path, *args, **kwargs):
            return track(original_open(path, *args, **kwargs), path)

        with patch.object(Path, "open", path_open), patch.object(builtins, "open", plain_open):
            state = adapter._diagnostic_start(client)
            self.assertTrue(opened)
            self.assertTrue(all(handle.closed for handle in opened))
            adapter._diagnostic_finish(state, client, response=self.synthetic_response())
            self.assertGreaterEqual(len(opened), 2)
            self.assertTrue(all(handle.closed for handle in opened))

    def test_journal_directory_or_append_failure_never_changes_success_or_error(self):
        client = self.make_default_client()
        response = self.synthetic_response()
        failure = RuntimeError(POISON + "_ORIGINAL_FAILURE")
        for filesystem_call in ("mkdir", "open"):
            with self.subTest(filesystem_call=filesystem_call), \
                    patch.object(Path, filesystem_call, side_effect=OSError(errno.EIO, POISON + "_LOG_FAILURE")), \
                    patch.object(client.chat.completions, "create", return_value=response) as create, \
                    patch.object(sys, "stderr", new_callable=io.StringIO) as stderr:
                self.assertEqual(self.invoke(client), {"result": POISON + "_RETURNED_CONTENT"})
                create.assert_called_once()
                self.assertNotIn(POISON, stderr.getvalue())
            with self.subTest(filesystem_call=filesystem_call, outcome="error"), \
                    patch.object(Path, filesystem_call, side_effect=OSError(errno.EIO, POISON + "_LOG_FAILURE")), \
                    patch.object(client.chat.completions, "create", side_effect=failure), \
                    self.assertRaises(RuntimeError) as raised:
                self.invoke(client)
            self.assertIs(raised.exception, failure)

    def test_end_append_failure_never_replaces_success_or_original_sdk_error(self):
        client = self.make_default_client()
        response = self.synthetic_response()
        original_append = adapter._diagnostic_append
        failure = RuntimeError(POISON + "_ORIGINAL_EXCEPTION")

        def fail_end(path, record):
            if record["event"] == "end":
                raise OSError(errno.EIO, POISON + "_END_APPEND_FAILURE")
            return original_append(path, record)

        with patch.object(adapter, "_diagnostic_append", side_effect=fail_end), \
                patch.object(client.chat.completions, "create", return_value=response) as create:
            self.assertEqual(self.invoke(client), {"result": POISON + "_RETURNED_CONTENT"})
        create.assert_called_once()
        with patch.object(adapter, "_diagnostic_append", side_effect=fail_end), \
                patch.object(client.chat.completions, "create", side_effect=failure) as create, \
                self.assertRaises(RuntimeError) as raised:
            self.invoke(client)
        create.assert_called_once()
        self.assertIs(raised.exception, failure)
        self.assertEqual([record["event"] for record in self.records()], ["start", "start"])

    def test_write_failure_closes_the_append_handle_without_changing_sdk_result(self):
        client = self.make_default_client()
        response = self.synthetic_response()
        original_open = Path.open
        opened = []

        class WriteFailure:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def write(self, value):
                raise OSError(errno.EIO, POISON + "_WRITE_FAILURE")

        def failing_open(path, *args, **kwargs):
            handle = original_open(path, *args, **kwargs)
            if path.suffix == ".jsonl":
                opened.append(handle)
                return WriteFailure(handle)
            return handle

        with patch.object(Path, "open", failing_open), \
                patch.object(client.chat.completions, "create", return_value=response):
            self.assertEqual(self.invoke(client), {"result": POISON + "_RETURNED_CONTENT"})
        self.assertTrue(opened)
        self.assertTrue(all(handle.closed for handle in opened))

    def test_unavailable_versions_and_fd_counts_keep_request_success_and_error(self):
        import openai
        import httpx2
        client = self.make_default_client()
        response = self.synthetic_response()
        error = RuntimeError(POISON + "_ORIGINAL_EXCEPTION")
        with patch.object(adapter.os, "listdir", side_effect=OSError(errno.EACCES, POISON)), \
                patch.object(adapter.metadata, "version", side_effect=OSError(errno.EIO, POISON)), \
                patch.object(openai, "__version__", None), \
                patch.object(httpx2, "__version__", None), \
                patch.object(client.chat.completions, "create", return_value=response) as create:
            self.assertEqual(self.invoke(client), {"result": POISON + "_RETURNED_CONTENT"})
            create.assert_called_once()
            create.reset_mock(side_effect=True)
            create.side_effect = error
            with self.assertRaises(RuntimeError) as raised:
                self.invoke(client)
            create.assert_called_once()
            self.assertIs(raised.exception, error)
        records = self.records()
        self.assert_pair(records[:2], "success")
        self.assert_pair(records[2:], "error")
        for record in records:
            self.assertIsNone(record["fd_count"])
            self.assertEqual(record["client"]["versions"], {"openai": None, "httpx2": None})

    def test_unexpected_metadata_or_fd_failure_never_changes_sdk_outcome(self):
        client = self.make_default_client()
        response = self.synthetic_response()
        error = RuntimeError(POISON + "_ORIGINAL_EXCEPTION")
        for helper in ("_client_metadata", "_fd_count"):
            with self.subTest(helper=helper), \
                    patch.object(adapter, helper, side_effect=RuntimeError(POISON + "_METADATA_FAILURE")), \
                    patch.object(client.chat.completions, "create", return_value=response) as create:
                self.assertEqual(self.invoke(client), {"result": POISON + "_RETURNED_CONTENT"})
                create.assert_called_once()
            with self.subTest(helper=helper, outcome="error"), \
                    patch.object(adapter, helper, side_effect=RuntimeError(POISON + "_METADATA_FAILURE")), \
                    patch.object(client.chat.completions, "create", side_effect=error) as create, \
                    self.assertRaises(RuntimeError) as raised:
                self.invoke(client)
            create.assert_called_once()
            self.assertIs(raised.exception, error)

    def test_per_process_files_and_counters_pair_without_reusing_another_pid(self):
        client = self.make_default_client()
        for pid in (600001, 600002):
            with patch.object(adapter.os, "getpid", return_value=pid), \
                    patch.object(adapter.os, "getppid", return_value=600000):
                for _ in range(2):
                    state = adapter._diagnostic_start(client)
                    adapter._diagnostic_finish(state, client, response=self.synthetic_response())
        records = self.records()
        self.assertEqual(len(records), 8)
        for pid in (600001, 600002):
            selected = [record for record in records if record["pid"] == pid]
            self.assertEqual(len(selected), 4)
            self.assert_pair(selected[:2], "success")
            self.assert_pair(selected[2:], "success")
            self.assertGreater(selected[2]["counter"], selected[0]["counter"])


if __name__ == "__main__":
    unittest.main()
