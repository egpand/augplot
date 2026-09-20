"""Small, replaceable inference boundary; imports and authentication are lazy."""

from .errors import ConfigurationError, ProviderError


def complete(
    *,
    model: str,
    messages: list[dict],
    api_base: str | None,
    timeout: float,
    response_format: dict | None = None,
) -> str:
    # Importing augplot must never initialize an SDK or contact a provider.
    import litellm

    if response_format is not None:
        try:
            supported = litellm.supports_response_schema(model=model)
        except Exception:
            supported = False
        if not supported:
            raise ConfigurationError(
                "The configured model must support strict JSON Schema responses. "
                "Choose a model that LiteLLM reports as supporting response schemas."
            )

    try:
        request = dict(
            model=model,
            messages=messages,
            api_base=api_base,
            timeout=timeout,
            num_retries=0,
            caching=False,
        )
        if response_format is not None:
            request["response_format"] = response_format
        response = litellm.completion(**request)
        content = response.choices[0].message.content
    except Exception as exc:
        # Do not echo SDK exceptions: they can contain headers or request payloads.
        name = type(exc).__name__
        hints = {
            "AuthenticationError": "Check the provider's API-key environment variable.",
            "RateLimitError": "The provider rate limit or quota was reached; retry later.",
            "Timeout": "The model request timed out; increase timeout or retry later.",
            "NotFoundError": "Check that the configured model and endpoint exist.",
            "APIConnectionError": "Check your network connection and api_base.",
            "BadRequestError": "Check your model identifier and provider configuration.",
        }
        hint = hints.get(name, "Check your model, endpoint, and provider credentials.")
        raise ProviderError(f"LLM request failed ({name}). {hint}") from None
    if not isinstance(content, str) or not content.strip():
        # Empty/refused responses are generation failures and can be repaired.
        return ""
    return content
