"""The one base model shared by every system (CC-28).

The direct baseline, the single agent and the multi-agent system must run on
the same base LLM. Every entry point (run_agent.py, scripts/run_experiments.py,
the demo, the setup check, the judge, and the shell launchers) takes its
default from get_default_model(), so changing the model in one place changes
it everywhere. Order: the MODEL environment variable, then MODEL in
config.yml, then DEFAULT_MODEL. An explicit --model still overrides it for
one command; scripts/run_experiments.py passes its one model to every arm.

get_default_base_url() picks the OpenAI-compatible endpoint the same way:
$BASE_URL, then BASE_URL in config.yml, then a local Ollama server for Qwen
models (e.g. qwen3:4b), else None (the provider's hosted API).

Run as a script to print the model; the shell launchers do this. It imports
nothing from the project so that it stays fast.
"""
import os
from pathlib import Path

import yaml

DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yml"


def _configured(value):
    """A usable model name, or None for empty values and config.example.yml placeholders."""
    if not isinstance(value, str) or not value.strip() or value.strip().startswith("your-"):
        return None
    return value.strip()


def _from_environment_or_config(key):
    if _configured(os.getenv(key)):
        return _configured(os.getenv(key))
    if CONFIG_PATH.exists():
        try:
            config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            # YAML diagnostics can contain credential-bearing source lines.
            raise ValueError(f"Could not read the model configuration at {CONFIG_PATH}") from None
        if isinstance(config, dict) and _configured(config.get(key)):
            return _configured(config[key])
    return None


def get_default_model():
    return _from_environment_or_config("MODEL") or DEFAULT_MODEL


def get_default_base_url(model):
    """The endpoint for `model`: $BASE_URL, then BASE_URL in config.yml, then Ollama for Qwen, else None."""
    configured = _from_environment_or_config("BASE_URL")
    if configured:
        return configured
    return DEFAULT_OLLAMA_BASE_URL if "qwen" in (model or "").lower() else None


if __name__ == "__main__":
    print(get_default_model())
