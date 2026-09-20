import sys
from types import SimpleNamespace

import pytest

from augplot import ConfigurationError, ProviderError
from augplot.provider import complete


def test_adapter_disables_retries_and_passes_configuration(monkeypatch):
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="result"))])

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    assert complete(model="anthropic/example", messages=[], api_base=None, timeout=60) == "result"
    assert calls[0]["num_retries"] == 0
    assert calls[0]["caching"] is False
    assert calls[0]["model"] == "anthropic/example"


def test_provider_error_does_not_expose_raw_exception(monkeypatch):
    class AuthenticationError(Exception):
        pass

    def completion(**kwargs):
        raise AuthenticationError("Authorization: SECRET_KEY; request: PRIVATE_DATA")

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    with pytest.raises(ProviderError) as caught:
        complete(model="openai/example", messages=[], api_base=None, timeout=60)
    assert "SECRET_KEY" not in str(caught.value)
    assert "PRIVATE_DATA" not in str(caught.value)
    assert "API-key" in str(caught.value)


def test_strict_response_format_is_sent_to_supported_models(monkeypatch):
    calls = []
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "result", "strict": True, "schema": {"type": "object"}},
    }

    def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(
            completion=completion,
            supports_response_schema=lambda **kwargs: kwargs["model"] == "openai/supported",
        ),
    )
    complete(
        model="openai/supported",
        messages=[],
        api_base=None,
        timeout=60,
        response_format=response_format,
    )
    assert calls[0]["response_format"] == response_format


def test_model_without_strict_response_schema_support_is_rejected(monkeypatch):
    def unexpected_completion(**kwargs):
        raise AssertionError("Unsupported models must fail before inference")

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(
            completion=unexpected_completion,
            supports_response_schema=lambda **kwargs: False,
        ),
    )
    with pytest.raises(ConfigurationError, match="strict JSON Schema"):
        complete(
            model="other/unsupported",
            messages=[],
            api_base=None,
            timeout=60,
            response_format={"type": "json_schema"},
        )
