import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from .registry import tool_registry


def normalize_retrieved_resources(resources: dict) -> dict:
    """Auto-parse JSON strings and ensure consistent list format."""
    normalized = {}
    
    for key, value in resources.items():
        try:
            # Parse JSON strings
            if isinstance(value, str) and value.strip().startswith(('{', '[')):
                parsed = json.loads(value)
                normalized[key] = [parsed] if isinstance(parsed, dict) else parsed
            # Wrap single dicts in lists
            elif isinstance(value, dict):
                normalized[key] = [value]
            else:
                normalized[key] = value
        except:
            normalized[key] = value
            
    return normalized


def execute_python_code(code: str, timeout: int = 60, global_vars: dict = None) -> str:
    """Execute calculations in an OS sandbox with only copied task data and a deadline."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        return {"error": "timeout must be positive"}
    timeout = min(timeout, 60)
    try:
        task_data = global_vars or {}
        task_data = {k: v for k, v in task_data.items() if k in {"retrieved_resources", "retrieved_fhir_resources"}}
        resources = normalize_retrieved_resources(task_data.get("retrieved_resources", task_data.get("retrieved_fhir_resources", {})))
        payload = json.dumps({"code": code, "resources": resources}, allow_nan=False)
        with tempfile.TemporaryDirectory(prefix="fhir-python-") as temporary:
            command = _sandbox_command(temporary)
            if command is None:
                return {"error": "Python sandbox unavailable: requires sandbox-exec on macOS or bubblewrap on Linux"}
            env = {"PATH": "/usr/bin:/bin", "HOME": temporary, "TMPDIR": temporary,
                   "LANG": "en_US.UTF-8", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
            result = subprocess.run(command, input=payload, text=True, capture_output=True,
                                    timeout=timeout, cwd=temporary, env=env)
            if result.returncode:
                return {"error": "Python sandbox worker failed", "returncode": result.returncode}
            return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return {"error": f"Python execution exceeded {timeout:g} seconds"}
    except Exception as e:
        return {"error": f"Python execution failed: {type(e).__name__}"}


def _sandbox_command(temporary):
    """Use kernel-enforced isolation; never fall back to host-process exec."""
    worker = Path(__file__).with_name("python_worker.py").resolve()
    executable = str(Path(sys.executable).resolve())
    runtime_paths = sorted({str(Path(sys.prefix).resolve()), str(Path(sys.base_prefix).resolve())})
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        read_paths = runtime_paths + ["/System", "/usr/lib", "/usr/share", "/Library/Apple",
                                      "/private/var/db/dyld", "/private/etc", "/dev",
                                      str(Path(temporary).resolve())]
        outside = " ".join(f"(require-not (subpath {json.dumps(p)}))" for p in read_paths)
        outside += f' (require-not (literal "/")) (require-not (literal {json.dumps(str(worker))}))'
        profile = ("(version 1) (allow default) (deny network*) (deny process-fork) "
                   f'(deny file-read-data (require-all (regex #"^/") {outside})) '
                   f'(deny file-map-executable (require-all (regex #"^/") {outside})) '
                   f'(deny file-write* (require-all (regex #"^/") (require-not (subpath {json.dumps(str(Path(temporary).resolve()))})) '
                   '(require-not (literal "/dev/null"))))')
        return ["sandbox-exec", "-p", profile, executable, "-I", str(worker)]
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        command = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
                   "--clearenv", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                   "--setenv", "HOME", "/tmp", "--setenv", "TMPDIR", "/tmp"]
        for path in sorted(set(runtime_paths + ["/usr", "/lib", "/lib64"] )):
            if Path(path).exists():
                command += ["--ro-bind", path, path]
        return command + ["--ro-bind", str(worker), "/worker.py", "--chdir", "/tmp", executable, "-I", "/worker.py"]
    return None


# Register tool
tool_registry.register_tool("execute_python_code", execute_python_code, {
    "type": "function",
    "function": {
        "name": "execute_python_code",
        "description": "Execute Python in an isolated worker using only retrieved_resources. Set the answer variable to a JSON value. Execution has a real timeout; repository files, credentials, and network are unavailable.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute. Use 'answer' variable to store final result."},
                "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30}
            },
            "required": ["code"]
        }
    }
})
