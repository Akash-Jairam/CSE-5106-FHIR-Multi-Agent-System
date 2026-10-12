"""Standalone sandbox worker. Never imports repository configuration or tools."""
import contextlib
import datetime
import json
import math
import re
import resource
import statistics
import sys


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 * 1024, 4 * 1024 * 1024))
    payload = json.load(sys.stdin)
    records = payload["resources"]
    namespace = {"json": json, "datetime": datetime, "math": math, "re": re,
                 "statistics": statistics,
                 "retrieved_resources": records, "retrieved_fhir_resources": records}
    try:
        # stdout is diagnostic; only the explicit result becomes the tool response.
        with open("/dev/null", "w") as sink, contextlib.redirect_stdout(sink):
            exec(payload["code"], namespace)
        result = {"answer": namespace["answer"]} if "answer" in namespace else {
            "error": "Code executed successfully (no answer variable set)"}
    except SystemExit:
        result = {"error": "Code called exit()/SystemExit. Set the 'answer' variable instead."}
    except Exception as error:
        result = {"error": f"{type(error).__name__}: {error}"}
    try:
        serialized = json.dumps(result, allow_nan=False)
    except (TypeError, ValueError):
        serialized = json.dumps({"error": "The answer must be a finite JSON value"})
    if len(serialized) > 4 * 1024 * 1024:
        serialized = json.dumps({"error": "Python result exceeds the 4 MiB limit"})
    sys.stdout.write(serialized)


if __name__ == "__main__":
    main()
