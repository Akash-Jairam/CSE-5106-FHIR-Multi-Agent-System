"""Personal Azure adapter, enabled only by the local fhir-azure launcher."""
from dataclasses import dataclass, field
import os
from pathlib import Path
from urllib.parse import urlsplit

import yaml


@dataclass(frozen=True)
class AzureSettings:
    endpoint: str
    api_version: str
    deployment: str
    model: str
    max_completion_tokens: int
    reasoning_effort: str
    api_key: str = field(repr=False)

    @classmethod
    def load(cls, path):
        config = yaml.safe_load(Path(path).read_text()) or {}
        azure = config.get("AZURE_OPENAI", {})
        required = ["ENDPOINT", "API_VERSION", "DEPLOYMENT", "MODEL",
                    "MAX_COMPLETION_TOKENS", "REASONING_EFFORT"]
        missing = [name for name in required if azure.get(name) in (None, "")]
        if missing:
            raise ValueError("Missing Azure settings: " + ", ".join(missing))
        endpoint = str(azure["ENDPOINT"])
        url = urlsplit(endpoint)
        if (url.scheme != "https" or not url.hostname or url.username
                or url.password or url.query or url.fragment):
            raise ValueError("AZURE_OPENAI.ENDPOINT must be an HTTPS resource endpoint")
        maximum = int(azure["MAX_COMPLETION_TOKENS"])
        if maximum < 1:
            raise ValueError("MAX_COMPLETION_TOKENS must be positive")
        if azure["REASONING_EFFORT"] != "none":
            raise ValueError("This GPT-6 Luna Chat Completions tool setup requires REASONING_EFFORT: none")
        key = (os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY")
               or azure.get("API_KEY") or "")
        return cls(endpoint, str(azure["API_VERSION"]), str(azure["DEPLOYMENT"]),
                   str(azure["MODEL"]), maximum, "none", str(key).strip())

    @property
    def key_present(self):
        return bool(self.api_key) and not self.api_key.lower().startswith(
            ("your-", "<", "paste", "replace"))

    def matches(self, model):
        return model in {self.model, self.deployment,
                         "azure/" + self.deployment}

    def public(self):
        return {"provider": "azure", "endpoint": self.endpoint,
                "api_version": self.api_version, "deployment": self.deployment,
                "model": self.model, "reasoning_effort": self.reasoning_effort,
                "max_completion_tokens": self.max_completion_tokens,
                "api_key_configured": self.key_present}


_settings = None
_client = None
_originals = None


def make_client(settings, http_client=None):
    from openai import AzureOpenAI
    if not settings.key_present:
        raise ValueError("Fill AZURE_OPENAI.API_KEY in your local config.yml")
    return AzureOpenAI(azure_endpoint=settings.endpoint,
                       api_version=settings.api_version, api_key=settings.api_key,
                       timeout=60.0, max_retries=0, http_client=http_client)


def azure_completion(*args, **kwargs):
    """Preserve the repo's LiteLLM response format using the Azure SDK."""
    import litellm
    global _client
    if args:
        raise TypeError("The local Azure adapter requires keyword arguments")
    if not _settings.matches(kwargs.get("model")):
        raise ValueError("The local Azure launcher supports only its configured deployment")
    if kwargs.get("base_url"):
        raise ValueError("Azure ENDPOINT is configured in config.yml; omit --base_url")
    if _client is None:
        _client = make_client(_settings)
    request = {
        "model": _settings.deployment,
        "messages": kwargs["messages"],
        "max_completion_tokens": min(_settings.max_completion_tokens,
                                     kwargs.get("max_completion_tokens") or kwargs.get("max_tokens") or _settings.max_completion_tokens),
        "reasoning_effort": _settings.reasoning_effort,
    }
    # Omit temperature, matching the supplied Azure sample. The SDK bypasses
    # the old LiteLLM model registry, which predates GPT-6.
    if kwargs.get("tools"):
        request["tools"] = kwargs["tools"]
        if kwargs.get("parallel_tool_calls") is not None:
            request["parallel_tool_calls"] = kwargs["parallel_tool_calls"]
    for name in ("tool_choice", "response_format", "stop", "seed"):
        if kwargs.get(name) is not None:
            request[name] = kwargs[name]
    response = _client.chat.completions.create(**request)
    return litellm.ModelResponse(**response.model_dump(exclude_none=True))


def install(settings):
    """Install process-local hooks before importing the project entry point."""
    import litellm
    import utils
    import utils.core_utils as core
    global _settings, _client, _originals
    _settings, _client = settings, None
    if _originals is None:
        _originals = {name: getattr(core, name) for name in
                      ("get_model_provider", "validate_model_api_key", "is_reasoning_llm")}

    def provider(model):
        return "azure" if _settings.matches(model) else _originals["get_model_provider"](model)

    def validate_key(model):
        return _settings.key_present if _settings.matches(model) else _originals["validate_model_api_key"](model)

    def setup_keys():
        # The Azure key is passed only to the Azure client, never installed as
        # OPENAI_API_KEY or printed by the launcher.
        return _settings.key_present

    def reasoning(model):
        return _settings.matches(model) or _originals["is_reasoning_llm"](model)

    hooks = {"get_model_provider": provider, "validate_model_api_key": validate_key,
             "setup_api_keys": setup_keys, "is_reasoning_llm": reasoning}
    for name, value in hooks.items():
        setattr(core, name, value)
        setattr(utils, name, value)
    litellm.completion = azure_completion

    # Azure pricing is not in the pinned LiteLLM registry. Preserve token
    # counts and label monetary cost unknown rather than reporting zero.
    from agent.base_agent import BaseAgent
    original_update = _originals.setdefault("update_usage", BaseAgent._update_usage)

    def update_usage(self, usage_info):
        if not usage_info:
            return
        self.total_usage["cost"] = 0.0
        original_update(self, usage_info)
        self.total_usage.update({"cost": None, "cost_status": "not_estimated",
                                 "provider": "azure", "model": _settings.model,
                                 "deployment": _settings.deployment,
                                 "api_version": _settings.api_version,
                                 "reasoning_effort": _settings.reasoning_effort})

    BaseAgent._update_usage = update_usage
