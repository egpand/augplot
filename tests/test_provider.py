import sys
from types import SimpleNamespace

import pytest

from augplot import ProviderError
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
