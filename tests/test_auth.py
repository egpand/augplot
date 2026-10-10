"""Exercise OAuth and renewable sessions without real accounts or OS credentials."""

import json
import time
from threading import Thread
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from augplot import AuthenticationError, ConfigurationError, ProviderError, auth


class MemoryKeyring:
    def __init__(self):
        self.values = {}

    def get_password(self, service, profile):
        return self.values.get((service, profile))

    def set_password(self, service, profile, value):
        self.values[service, profile] = value

    def delete_password(self, service, profile):
        self.values.pop((service, profile), None)


@pytest.fixture
def keyring(monkeypatch):
    backend = MemoryKeyring()
    monkeypatch.setattr(auth, "_secure_backend", lambda: backend)
    return backend


@pytest.fixture
def saved_account(keyring):
    state = {
        "version": 1,
        "host_id": "urn:uuid:test-host",
        "active": "first",
        "profiles": {
            "first": {
                "label": "test@example.invalid",
                "client_id": "oaiapp_test",
                "subject": "user-test",
                "connected": True,
                "plan_enabled": True,
                "model": "model-a",
                "catalog": [
                    {"id": "model-a", "name": "Model A"},
                    {"id": "model-b", "name": "Model B"},
                ],
            },
        },
    }
    auth._write_state(state)
    auth._credentials(
        "first",
        {
            "access_token": "SECRET_ACCESS",
            "refresh_token": "SECRET_REFRESH",
            "id_token": "SECRET_ID",
            "expires_at": time.time() + 3600,
            "scopes": [auth.PLAN_SCOPE],
        },
    )
    return state


@pytest.fixture
def discovery(monkeypatch):
    data = {
        "issuer": auth.ISSUER,
        "authorization_endpoint": auth.ISSUER + "/api/accounts/authorize",
        "token_endpoint": auth.ISSUER + "/api/accounts/oauth/token",
        "jwks_uri": auth.ISSUER + "/.well-known/jwks.json",
        "revocation_endpoint": auth.ISSUER + "/oauth/revoke",
    }
    monkeypatch.setattr(auth, "_discovery", lambda: data)
    return data


def response(body, status=200):
    return httpx.Response(status, json=body, request=httpx.Request("GET", auth.ISSUER))


def test_accounts_are_offline_safe_metadata(saved_account, monkeypatch):
    monkeypatch.setattr(auth, "_credentials", lambda *a, **k: pytest.fail("No secrets needed"))
    monkeypatch.setattr(auth, "_request", lambda *a, **k: pytest.fail("No network needed"))
    accounts = auth.accounts()
    assert accounts[0]["active"]
    assert accounts[0]["model"] == "model-a"
    assert "SECRET" not in repr(accounts)
    assert "SECRET" not in (auth._directory() / "profiles.json").read_text()


def test_secure_storage_rejects_plaintext_backend(monkeypatch):
    import keyring

    monkeypatch.setattr(keyring, "get_keyring", lambda: SimpleNamespace(priority=1))
    monkeypatch.setattr(auth.webbrowser, "open", lambda *a: pytest.fail("Do not open browser"))
    with pytest.raises(AuthenticationError, match="secure operating-system"):
        auth.sign_in()


def test_model_catalog_order_visibility_and_saved_default(saved_account, monkeypatch):
    calls = []

    def request(*args, **kwargs):
        calls.append((args, kwargs))
        return response(
            {
                "models": [
                    {"slug": "hidden", "display_name": "Hidden", "visibility": "hide"},
                    {"slug": "model-b", "display_name": "Model B", "visibility": "list"},
                    {"slug": "model-a", "display_name": "Model A", "visibility": "list"},
                ]
            }
        )

    monkeypatch.setattr(auth, "_request", request)
    table = auth.models()
    assert list(table["id"]) == ["model-b", "model-a"]
    assert list(table["selected"]) == [False, True]
    assert calls[0][0] == ("GET", auth.RESOURCE + "/models")
    assert calls[0][1]["headers"]["Authorization"] == "Bearer SECRET_ACCESS"
    assert auth.set_model("model-b") == "model-b"
    assert auth.connection() == ("first", "model-b")
    with pytest.raises(ConfigurationError, match="unavailable"):
        auth.set_model("unknown")


def test_empty_catalog_does_not_erase_saved_model(saved_account, monkeypatch):
    monkeypatch.setattr(auth, "_request", lambda *a, **k: response({"models": []}))
    with pytest.raises(ConfigurationError, match="No models"):
        auth.models()
    assert auth.connection() == ("first", "model-a")


def test_refresh_rotates_all_fields_and_reuses_issued_client(saved_account, monkeypatch):
    tokens = auth._credentials("first")
    tokens["expires_at"] = 0
    auth._credentials("first", tokens)
    calls = []

    def refresh(form):
        calls.append(form)
        return {
            "access_token": "ROTATED",
            "refresh_token": "ROTATED_REFRESH",
            "expires_in": 3600,
            "token_type": "Bearer",
        }

    monkeypatch.setattr(auth, "_token_request", refresh)
    assert auth._access_token("first") == "ROTATED"
    assert auth._access_token("first") == "ROTATED"
    assert len(calls) == 1
    assert calls[0]["client_id"] == "oaiapp_test"
    assert "scope" not in calls[0]
    assert auth._credentials("first")["refresh_token"] == "ROTATED_REFRESH"
    assert auth._credentials("first")["scopes"] == [auth.PLAN_SCOPE]


def test_concurrent_refresh_is_serialized(saved_account, monkeypatch):
    tokens = auth._credentials("first")
    tokens["expires_at"] = 0
    auth._credentials("first", tokens)
    calls = []
    results = []

    def refresh(form):
        calls.append(form)
        time.sleep(0.05)
        return {
            "access_token": "NEW",
            "refresh_token": "NEW_REFRESH",
            "expires_in": 3600,
            "token_type": "Bearer",
        }

    monkeypatch.setattr(auth, "_token_request", refresh)
    threads = [Thread(target=lambda: results.append(auth._access_token("first"))) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert results == ["NEW", "NEW"]
    assert len(calls) == 1


@pytest.mark.parametrize(
    "status,code,cleared",
    [
        (400, "invalid_grant", True),
        (503, "temporary_failure", False),
    ],
)
def test_refresh_error_recovers_without_erasing_temporary_sessions(
    saved_account,
    discovery,
    monkeypatch,
    status,
    code,
    cleared,
):
    tokens = auth._credentials("first")
    tokens["expires_at"] = 0
    auth._credentials("first", tokens)
    monkeypatch.setattr(auth, "_request", lambda *a, **k: response({"error": code}, status))
    with pytest.raises(AuthenticationError):
        auth._access_token("first")
    assert (auth._credentials("first") is None) is cleared
    assert auth.accounts()[0]["connected"] is not cleared


def test_refresh_removes_revoked_scope(saved_account, monkeypatch):
    tokens = auth._credentials("first")
    tokens["expires_at"] = 0
    auth._credentials("first", tokens)
    monkeypatch.setattr(
        auth,
        "_token_request",
        lambda *a: {
            "access_token": "NEW",
            "expires_in": 3600,
            "token_type": "Bearer",
            "scope": "openid",
        },
    )
    with pytest.raises(AuthenticationError, match="permission changed"):
        auth._access_token("first")
    assert not auth.accounts()[0]["plan_enabled"]


@pytest.mark.parametrize("status", [200, 503])
def test_logout_clears_secrets_keeps_registration_and_offline_model(
    saved_account,
    discovery,
    monkeypatch,
    capsys,
    status,
):
    calls = []

    def request(*args, **kwargs):
        calls.append(kwargs["data"])
        return response({}, status)

    monkeypatch.setattr(auth, "_request", request)
    result = auth.sign_out()
    assert not result["connected"]
    assert auth._credentials("first") is None
    assert auth._read_state()["profiles"]["first"]["client_id"] == "oaiapp_test"
    assert auth.connection() == ("first", "model-a")
    assert calls[0]["token_type_hint"] == "refresh_token"
    assert ("not confirmed" in capsys.readouterr().out) is (status == 503)


def test_selecting_accounts_preserves_credentials(saved_account, monkeypatch):
    state = auth._read_state()
    state["profiles"]["second"] = {
        **state["profiles"]["first"],
        "client_id": "oaiapp_second",
        "model": "model-b",
    }
    auth._write_state(state)
    auth._credentials("second", {"access_token": "SECOND"})
    monkeypatch.setattr(auth, "models", lambda: None)
    assert auth.select_account("second")["model"] == "model-b"
    assert auth._credentials("first")["access_token"] == "SECRET_ACCESS"
    assert auth._credentials("second")["access_token"] == "SECOND"


def test_signin_stores_registration_before_exchange_and_selects_default(
    keyring,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(auth, "_authorize", lambda *a: ("oaiapp_new", "CODE", "V", "URI", "N"))

    def exchange(form, **kwargs):
        pending = next(iter(auth._read_state()["profiles"].values()))
        assert pending["client_id"] == "oaiapp_new"
        assert auth._read_state()["active"] is None
        assert form["client_id"] == "oaiapp_new"
        return {
            "id_token": "ID",
            "access_token": "ACCESS",
            "refresh_token": "REFRESH",
            "expires_in": 3600,
            "token_type": "Bearer",
            "scope": auth.SCOPES,
        }

    monkeypatch.setattr(auth, "_token_request", exchange)
    monkeypatch.setattr(
        auth,
        "_verify_identity",
        lambda *a: {
            "sub": "subject",
            "email": "test@example.invalid",
        },
    )
    monkeypatch.setattr(
        auth,
        "_request",
        lambda *a, **k: response(
            {
                "models": [
                    {"slug": "first-model", "display_name": "First Model", "visibility": "list"},
                ]
            }
        ),
    )
    connection = auth.sign_in()
    assert connection["plan_enabled"]
    assert connection["model"] == "first-model"
    assert auth._read_state()["host_id"].startswith("urn:uuid:")
    assert "using your ChatGPT plan" in capsys.readouterr().out
    metadata = (auth._directory() / "profiles.json").read_text()
    assert "ACCESS" not in metadata and "REFRESH" not in metadata


def test_identity_only_signin_never_calls_inference(keyring, monkeypatch):
    monkeypatch.setattr(auth, "_authorize", lambda *a: ("oaiapp_new", "C", "V", "URI", "N"))
    monkeypatch.setattr(
        auth, "_token_request", lambda *a, **k: {"id_token": "ID", "scope": "openid"}
    )
    monkeypatch.setattr(auth, "_verify_identity", lambda *a: {"sub": "subject"})
    monkeypatch.setattr(auth, "models", lambda: pytest.fail("No catalog without scope"))
    connection = auth.sign_in()
    assert connection["connected"]
    assert not connection["plan_enabled"]
    with pytest.raises(AuthenticationError, match="plan use is disabled"):
        auth._access_token(connection["profile_id"])


def test_returning_signin_rejects_other_identity_without_overwriting(saved_account, monkeypatch):
    monkeypatch.setattr(auth, "_authorize", lambda *a: ("oaiapp_test", "C", "V", "URI", "N"))
    monkeypatch.setattr(auth, "_token_request", lambda *a, **k: {"id_token": "ID"})
    monkeypatch.setattr(auth, "_verify_identity", lambda *a: {"sub": "OTHER"})
    with pytest.raises(AuthenticationError, match="does not match"):
        auth.sign_in()
    assert auth._credentials("first")["access_token"] == "SECRET_ACCESS"


def test_catalog_failure_leaves_new_connection_recoverable(keyring, monkeypatch):
    monkeypatch.setattr(auth, "_authorize", lambda *a: ("oaiapp_new", "C", "V", "URI", "N"))
    monkeypatch.setattr(
        auth,
        "_token_request",
        lambda *a, **k: {
            "id_token": "ID",
            "access_token": "ACCESS",
            "scope": auth.SCOPES,
            "expires_in": 3600,
            "token_type": "Bearer",
        },
    )
    monkeypatch.setattr(auth, "_verify_identity", lambda *a: {"sub": "subject"})
    monkeypatch.setattr(auth, "_request", lambda *a, **k: response({}, 503))
    with pytest.raises(ProviderError, match="list ChatGPT models"):
        auth.sign_in()
    assert auth.accounts()[0]["connected"]


@pytest.mark.parametrize(
    "override",
    [
        {"nonce": "wrong"},
        {"aud": "OTHER_CLIENT"},
        {"iss": "https://attacker.invalid"},
        {"exp": 1},
        {"sub": ""},
    ],
)
def test_identity_verification_rejects_bad_signed_claims(discovery, monkeypatch, override):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    jwk["kid"] = "test-key"
    monkeypatch.setattr(auth, "_request", lambda *a, **k: response({"keys": [jwk]}))
    claims = {
        "sub": "user",
        "iss": auth.ISSUER,
        "aud": "CLIENT",
        "iat": time.time(),
        "exp": time.time() + 3600,
        "nonce": "NONCE",
        **override,
    }
    token = jwt.encode(claims, private, algorithm="RS256", headers={"kid": "test-key"})
    with pytest.raises(AuthenticationError, match="invalid identity"):
        auth._verify_identity(token, "CLIENT", "NONCE")


def test_identity_verification_accepts_valid_signature_and_rejects_tampering(
    discovery, monkeypatch
):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    jwk["kid"] = "test-key"
    monkeypatch.setattr(auth, "_request", lambda *a, **k: response({"keys": [jwk]}))
    token = jwt.encode(
        {
            "sub": "user",
            "iss": auth.ISSUER,
            "aud": "CLIENT",
            "iat": time.time(),
            "exp": time.time() + 3600,
            "nonce": "NONCE",
        },
        private,
        algorithm="RS256",
        headers={"kid": "test-key"},
    )
    assert auth._verify_identity(token, "CLIENT", "NONCE")["sub"] == "user"
    chunks = token.split(".")
    chunks[2] = ("A" if chunks[2][0] != "A" else "B") + chunks[2][1:]
    with pytest.raises(AuthenticationError):
        auth._verify_identity(".".join(chunks), "CLIENT", "NONCE")


@pytest.mark.parametrize("callback", ["success", "denied", "different_client", "wrong_issuer"])
def test_real_loopback_callback_state_pkce_and_cleanup(discovery, monkeypatch, callback, capsys):
    captured = {}
    requests = []
    workers = []

    def browser(url):
        params = parse_qs(urlsplit(url).query)
        captured.update(params)

        def submit():
            redirect = params["redirect_uri"][0]
            # Invalid callbacks must not consume the authorization attempt.
            requests.append(httpx.get(redirect + "?" + urlencode({"state": "wrong"})).status_code)
            data = {"state": params["state"][0], "code": "PRIVATE_CODE"}
            if callback == "denied":
                data["error"] = "access_denied"
            elif callback == "different_client":
                data["client_id"] = "OTHER"
            elif callback == "wrong_issuer":
                data["iss"] = "https://attacker.invalid"
            requests.append(httpx.get(redirect + "?" + urlencode(data)).status_code)

        worker = Thread(target=submit, daemon=True)
        workers.append(worker)
        worker.start()
        return True

    monkeypatch.setattr(auth.webbrowser, "open", browser)
    if callback == "success":
        result = auth._authorize({"client_id": "SAVED"}, "urn:uuid:host", None, 5)
        assert result[0] == "SAVED"
        assert result[1] == "PRIVATE_CODE"
        import base64
        import hashlib

        challenge = base64.urlsafe_b64encode(hashlib.sha256(result[2].encode()).digest()).rstrip(
            b"="
        )
        assert captured["code_challenge"] == [challenge.decode()]
    else:
        with pytest.raises(AuthenticationError):
            auth._authorize({"client_id": "SAVED"}, "urn:uuid:host", None, 5)
    for worker in workers:
        worker.join(timeout=5)
    assert requests == [400, 200]
    assert "PRIVATE_CODE" not in capsys.readouterr().out
    assert captured["client_id"] == ["SAVED"]
    assert "agent_name_hint" not in captured
    assert captured["resource"] == [auth.RESOURCE]


def test_callback_timeout_and_browser_failure(discovery, monkeypatch):
    monkeypatch.setattr(auth.webbrowser, "open", lambda *a: True)
    with pytest.raises(AuthenticationError, match="timed out"):
        auth._authorize({}, "urn:uuid:host", None, 0.01)
    monkeypatch.setattr(auth.webbrowser, "open", lambda *a: False)
    with pytest.raises(AuthenticationError, match="Could not open"):
        auth._authorize({}, "urn:uuid:host", None, 5)


@pytest.fixture
def stream_client(monkeypatch):
    import openai

    calls = []
    events = []

    class Stream:
        def __enter__(self):
            return iter(events)

        def __exit__(self, *args):
            return False

    class Client:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.responses = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            calls.append(kwargs)
            return Stream()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(openai, "OpenAI", Client)
    return calls, events


def completed_events(text):
    return [
        SimpleNamespace(type="response.output_text.delta", delta=text),
        SimpleNamespace(type="response.completed", response=SimpleNamespace(status="completed")),
    ]


def test_plan_request_contract_and_no_litellm(saved_account, stream_client, monkeypatch):
    from augplot import provider
    from augplot.prompts import RESPONSE_FORMAT

    monkeypatch.setitem(__import__("sys").modules, "litellm", None)
    calls, events = stream_client
    events.extend(completed_events('{"status":"ok"}'))
    content = provider.complete(
        model="model-a",
        messages=[{"role": "system", "content": "RULES"}, {"role": "user", "content": "PROFILE"}],
        api_base=None,
        timeout=45,
        response_format=RESPONSE_FORMAT,
        auth="chatgpt",
    )
    assert content == '{"status":"ok"}'
    assert calls[0]["api_key"] == "SECRET_ACCESS"
    assert calls[0]["max_retries"] == 0
    assert calls[1]["instructions"] == "RULES"
    assert calls[1]["input"] == [{"role": "user", "content": "PROFILE"}]
    assert calls[1]["store"] is False and calls[1]["stream"] is True
    assert calls[1]["text"]["format"]["schema"] == RESPONSE_FORMAT["json_schema"]["schema"]
    assert "temperature" not in calls[1] and "max_output_tokens" not in calls[1]


@pytest.mark.parametrize("terminal", ["response.failed", "response.incomplete", None])
def test_partial_responses_are_never_returned(saved_account, stream_client, terminal):
    from augplot import provider

    _, events = stream_client
    events.append(SimpleNamespace(type="response.output_text.delta", delta="PRIVATE_PARTIAL"))
    if terminal:
        events.append(
            SimpleNamespace(
                type=terminal,
                response=SimpleNamespace(
                    error=SimpleNamespace(
                        code="subscription_sharing_usage_limit_exceeded",
                    )
                ),
            )
        )
    with pytest.raises(ProviderError) as caught:
        provider.complete(model="model-a", messages=[], api_base=None, timeout=30, auth="chatgpt")
    assert "PRIVATE_PARTIAL" not in str(caught.value)
    assert "SECRET" not in str(caught.value)
    if terminal == "response.failed":
        assert "Manage usage" in str(caught.value)


def test_unknown_model_and_custom_endpoint_fail_before_inference(saved_account, stream_client):
    from augplot import provider

    calls, _ = stream_client
    with pytest.raises(ConfigurationError, match="unavailable"):
        provider.complete(model="other", messages=[], api_base=None, timeout=30, auth="chatgpt")
    with pytest.raises(ConfigurationError, match="Custom api_base"):
        provider.complete(
            model="model-a",
            messages=[],
            api_base="https://attacker.invalid",
            timeout=30,
            auth="chatgpt",
        )
    assert calls == []


def test_sdk_error_does_not_expose_credentials_or_payload(saved_account, monkeypatch):
    import openai

    from augplot import provider

    def broken(**kwargs):
        raise RuntimeError("SECRET_ACCESS; PRIVATE_PROFILE")

    monkeypatch.setattr(openai, "OpenAI", broken)
    with pytest.raises(ProviderError) as caught:
        provider.complete(model="model-a", messages=[], api_base=None, timeout=30, auth="chatgpt")
    assert "SECRET_ACCESS" not in str(caught.value)
    assert "PRIVATE_PROFILE" not in str(caught.value)


def test_model_and_account_are_pinned_for_refinement_and_replay(
    saved_account,
    stream_client,
    monkeypatch,
):
    import matplotlib.pyplot as plt
    from conftest import ARRAY_CODE
    from conftest import response as plot_response

    from augplot import plot

    monkeypatch.delenv("AUGPLOT_MODEL", raising=False)
    monkeypatch.delenv("AUGPLOT_API_BASE", raising=False)
    calls, events = stream_client
    events.extend(completed_events(plot_response(ARRAY_CODE)))
    viz = plot([1, 2], show=False)
    state = auth._read_state()
    state["profiles"]["first"]["model"] = "model-b"
    state["profiles"]["second"] = {
        **state["profiles"]["first"],
        "client_id": "oaiapp_second",
        "label": "Second",
    }
    state["active"] = "second"
    auth._write_state(state)
    auth._credentials("second", {**auth._credentials("first"), "access_token": "SECOND"})
    viz.refine("Add labels", show=False)
    assert calls[2]["api_key"] == "SECRET_ACCESS"
    assert calls[3]["model"] == "model-a"
    next_viz = plot([3, 4], show=False)
    assert calls[4]["api_key"] == "SECOND"
    assert calls[5]["model"] == "model-b"
    state["active"] = "first"
    state["profiles"]["first"].update(connected=False, plan_enabled=False, model="model-a")
    auth._write_state(state)
    auth._credentials("first", clear=True)
    calls_before = len(calls)
    replay = plot([1, 2], auth="chatgpt", show=False)
    replay.refine("Add labels", show=False)
    assert replay.cache_hit
    assert len(calls) == calls_before
    assert replay.code == viz.code
    with pytest.raises(AuthenticationError, match="signed out"):
        plot([1, 2], auth="chatgpt", show=False, regenerate=True)
    for chart in (viz, next_viz, replay):
        plt.close(chart.figure)


def test_explicit_model_cache_replays_with_no_saved_connection(saved_account, stream_client):
    from conftest import ARRAY_CODE
    from conftest import response as plot_response

    from augplot import plot

    calls, events = stream_client
    events.extend(completed_events(plot_response(ARRAY_CODE)))
    plot([1, 2], auth="chatgpt", model="model-a", show=False)
    (auth._directory() / "profiles.json").unlink()
    replay = plot([1, 2], auth="chatgpt", model="model-a", show=False)
    assert replay.cache_hit
    assert len(calls) == 2


def test_explicit_api_configuration_preserves_existing_route(
    saved_account, monkeypatch, fake_model
):
    from conftest import ARRAY_CODE
    from conftest import response as plot_response

    from augplot import plot

    calls = fake_model(plot_response(ARRAY_CODE))
    plot([1, 2], model="test/fake-model", show=False)
    assert calls[0]["model"] == "test/fake-model"
    assert "auth" not in calls[0]


def test_unsigned_identity_is_rejected_before_jwks_request(discovery, monkeypatch):
    monkeypatch.setattr(auth, "_request", lambda *a, **k: pytest.fail("No keys for unsigned JWT"))
    token = jwt.encode({"sub": "user"}, key="", algorithm="none")
    with pytest.raises(AuthenticationError):
        auth._verify_identity(token, "CLIENT", "NONCE")


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"version": 1, "profiles": {}},
        {"version": 1, "active": [], "host_id": None, "profiles": {}},
        {"version": 1, "active": None, "host_id": None, "profiles": {"broken": []}},
    ],
)
def test_corrupt_account_metadata_is_reported_without_raw_contents(value):
    directory = auth._directory()
    directory.mkdir()
    (directory / "profiles.json").write_text(json.dumps(value))
    with pytest.raises(ConfigurationError, match="account settings"):
        auth.accounts()


@pytest.mark.parametrize(
    "invalid",
    [
        {"expires_in": float("nan")},
        {"token_type": []},
        {"access_token": ""},
        {"refresh_token": []},
        {"scope": ["openid"]},
    ],
)
def test_invalid_credential_responses_are_rejected(invalid):
    data = {"access_token": "ACCESS", "token_type": "Bearer", "expires_in": 3600, **invalid}
    with pytest.raises(AuthenticationError, match="invalid plan-access credentials"):
        auth._token_set(data)


def test_real_openai_sdk_serializes_and_reads_sse_without_network(saved_account, monkeypatch):
    import openai

    from augplot import provider
    from augplot.prompts import RESPONSE_FORMAT

    sdk = openai.OpenAI
    requests = []
    event_data = [
        {
            "type": "response.output_text.delta",
            "delta": '{"status":"ok"}',
            "item_id": "message-test",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
        },
        {
            "type": "response.completed",
            "sequence_number": 2,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "model-a",
                "output": [],
            },
        },
    ]
    stream = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in event_data
    )

    def handle(request):
        requests.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=stream)

    def client(**kwargs):
        return sdk(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))

    monkeypatch.setattr(openai, "OpenAI", client)
    content = provider.complete(
        model="model-a",
        messages=[{"role": "system", "content": "RULES"}, {"role": "user", "content": "PROFILE"}],
        api_base=None,
        timeout=30,
        auth="chatgpt",
        response_format=RESPONSE_FORMAT,
    )
    assert content == '{"status":"ok"}'
    assert len(requests) == 1
    assert str(requests[0].url) == auth.RESOURCE + "/responses"
    assert requests[0].headers["authorization"] == "Bearer SECRET_ACCESS"
    body = json.loads(requests[0].content)
    assert body["instructions"] == "RULES" and body["stream"] is True
    assert body["text"]["format"]["strict"] is True
