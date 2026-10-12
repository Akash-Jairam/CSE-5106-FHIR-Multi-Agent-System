"""The one base model shared by every system (CC-28).

The direct baseline, the single agent and the multi-agent system must run on
the same base LLM. Every entry point (run_agent.py, scripts/run_experiments.py,
the demo, the setup check, the judge, and the shell launchers) takes its
default from get_default_model(), so changing the model in one place changes
it everywhere. Order: the MODEL environment variable, then MODEL in
config.yml, then DEFAULT_MODEL. An explicit --model still overrides it for
one command; scripts/run_experiments.py passes its one model to every arm.

Run as a script to print the model; the shell launchers do this. It imports
nothing from the project so that it stays fast.
"""
import os
from pathlib import Path

import yaml

DEFAULT_MODEL = "gpt-6-luna"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yml"


def _configured(value):
    """A usable model name, or None for empty values and config.example.yml placeholders."""
    if not isinstance(value, str) or not value.strip() or value.strip().startswith("your-"):
        return None
    return value.strip()


def get_default_model():
    if _configured(os.getenv("MODEL")):
        return _configured(os.getenv("MODEL"))
    if CONFIG_PATH.exists():
        try:
            config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"Could not read the model from {CONFIG_PATH}: {exc}") from None
        if isinstance(config, dict) and _configured(config.get("MODEL")):
            return _configured(config["MODEL"])
    return DEFAULT_MODEL


if __name__ == "__main__":
    print(get_default_model())
