"""Strict, streamed Responses requests using a user's authorized ChatGPT plan."""

from . import auth
from .errors import AuthenticationError, ConfigurationError, ProviderError

_MAX_RESPONSE_CHARS = 200_000


def _failure(code=None, status=None):
    if (
        code
        in {
            "subscription_sharing_usage_limit_exceeded",
            "subscription_sharing_usage_unavailable",
        }
        or status == 429
    ):
        return ProviderError("ChatGPT usage limit reached. Manage usage: " + auth.MANAGE_USAGE_URL)
    if (
        code
        in {
            "subscription_sharing_invalid_user",
            "chatpass_v2_scope_not_authorized",
            "chatpass_v2_invalid_authorization_context",
        }
        or status == 401
    ):
        return AuthenticationError(
            "ChatGPT could not authorize this request. Check the selected account "
            "and reconnect with ap.sign_in()."
        )
    if code == "subscription_sharing_unsupported_capability" or status == 400:
        return ConfigurationError(
            "This ChatGPT model or request does not support Augplot's strict response contract. "
            "Inspect ap.models() and select another model."
        )
    if status == 403:
        return ProviderError("ChatGPT blocked this request under the account's access policy.")
    if status == 404:
        return ConfigurationError("ChatGPT model is unavailable. Inspect ap.models().")
    return ProviderError("ChatGPT inference failed. Check access or connectivity and retry later.")


def complete(*, model, messages, timeout, response_format, profile_id=None):
    """Return only completed text, never execute or expose partial stream output."""
    from openai import APIStatusError, OpenAI

    state = auth._read_state()
    selected, profile = auth._profile(state, profile_id)
    if model not in {item["id"] for item in profile.get("catalog", [])}:
        raise ConfigurationError("Model is unavailable for this account. Inspect ap.models().")
    token = auth._access_token(selected)
    instructions = "\n\n".join(
        message["content"] for message in messages if message["role"] == "system"
    )
    inputs = [dict(message) for message in messages if message["role"] != "system"]
    request = {
        "model": model,
        "instructions": instructions,
        "input": inputs,
        "store": False,
        "stream": True,
    }
    if response_format is not None:
        schema = response_format["json_schema"]
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": schema["name"],
                "strict": schema["strict"],
                "schema": schema["schema"],
            }
        }
    fragments = []
    length = 0
    completed = False
    try:
        with (
            OpenAI(
                api_key=token,
                base_url=auth.RESOURCE,
                max_retries=0,
                timeout=timeout,
            ) as client,
            client.responses.create(**request) as stream,
        ):
            for event in stream:
                if event.type == "response.output_text.delta":
                    length += len(event.delta)
                    if length > _MAX_RESPONSE_CHARS:
                        raise ProviderError("ChatGPT response exceeded Augplot's size limit.")
                    fragments.append(event.delta)
                elif event.type == "response.completed":
                    if getattr(event.response, "status", None) != "completed":
                        raise ProviderError("ChatGPT returned an incomplete result. Retry later.")
                    completed = True
                    break
                elif event.type == "response.failed":
                    error = event.response.error
                    raise _failure(error.code if error else None)
                elif event.type in {"response.incomplete", "error"}:
                    raise _failure(getattr(event, "code", None))
    except (ProviderError, ConfigurationError):
        raise
    except APIStatusError as exc:
        body = exc.body
        error = body.get("error", body) if isinstance(body, dict) else {}
        code = error.get("code") if isinstance(error, dict) else None
        raise _failure(code if isinstance(code, str) else None, exc.status_code) from None
    except Exception:
        # SDK exceptions can contain credentials or the private request payload.
        raise ProviderError("ChatGPT request interrupted or timed out. Retry later.") from None
    if not completed:
        raise ProviderError("ChatGPT stream ended before completion. Retry later.")
    return "".join(fragments)
