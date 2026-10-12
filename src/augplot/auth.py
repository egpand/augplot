"""Local ChatGPT connections. Credentials never enter the notebook-facing API."""

import base64
import hashlib
import json
import math
import os
import secrets
import tempfile
import time
import uuid
import webbrowser
from contextlib import contextmanager
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit

from .errors import AuthenticationError, ConfigurationError, ProviderError

ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = f"openid profile email offline_access resource.invoke {PLAN_SCOPE}"
MANAGE_USAGE_URL = "https://chatgpt.com/#settings/Usage"
_SERVICE = "augplot.chatgpt"
_SECURE_BACKENDS = (
    "keyring.backends.macOS",
    "keyring.backends.Windows",
    "keyring.backends.SecretService",
    "keyring.backends.kwallet",
    "keyring.backends.libsecret",
)
_TERMINAL_REFRESH_ERRORS = {
    "invalid_grant",
    "invalid_refresh_token",
    "token_expired",
    "refresh_token_expired",
    "refresh_token_invalidated",
    "refresh_token_reused",
}


def _directory():
    from platformdirs import user_config_path

    return user_config_path("augplot") / "chatgpt"


def _read_state():
    try:
        state = json.loads((_directory() / "profiles.json").read_text())
        if (
            state.get("version") != 1
            or not {"active", "host_id", "profiles"} <= state.keys()
            or not isinstance(state["profiles"], dict)
        ):
            raise ValueError
        if state.get("active") is not None and not isinstance(state["active"], str):
            raise ValueError
        for profile in state["profiles"].values():
            if not isinstance(profile, dict) or not isinstance(profile.get("label"), str):
                raise ValueError
        return state
    except FileNotFoundError:
        return {"version": 1, "host_id": None, "active": None, "profiles": {}}
    except (OSError, ValueError, TypeError, AttributeError):
        raise ConfigurationError("Could not read Augplot's ChatGPT account settings.") from None


def _write_state(state):
    directory = _directory()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    filename = None
    try:
        fd, filename = tempfile.mkstemp(dir=directory, prefix="profiles-", suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            json.dump(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(filename, directory / "profiles.json")
    except OSError:
        raise ConfigurationError("Could not save Augplot's ChatGPT account settings.") from None
    finally:
        if filename is not None:
            try:
                os.unlink(filename)
            except FileNotFoundError:
                pass


@contextmanager
def _locked():
    from filelock import FileLock, Timeout

    directory = _directory()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with FileLock(directory / "session.lock", timeout=30):
            yield
    except Timeout:
        raise AuthenticationError(
            "Another kernel is updating ChatGPT access. Retry shortly."
        ) from None


def _secure_backend():
    import keyring

    try:
        backend = keyring.get_keyring()
        candidates = getattr(backend, "backends", [backend])
        for candidate in candidates:
            if type(candidate).__module__.startswith(_SECURE_BACKENDS) and candidate.priority > 0:
                return candidate
    except Exception:
        pass
    raise AuthenticationError(
        "A secure operating-system credential store is required for ChatGPT sign-in. "
        "Unlock or enable your macOS Keychain, Windows Credential Manager, "
        "or Linux Secret Service/KWallet, then retry ap.sign_in()."
    )


def _credentials(profile_id, value=None, *, clear=False):
    backend = _secure_backend()
    try:
        if clear:
            from keyring.errors import PasswordDeleteError

            try:
                backend.delete_password(_SERVICE, profile_id)
            except PasswordDeleteError:
                pass
            return None
        if value is not None:
            backend.set_password(_SERVICE, profile_id, json.dumps(value))
            return None
        raw = backend.get_password(_SERVICE, profile_id)
        result = json.loads(raw) if raw is not None else None
        if result is not None and not isinstance(result, dict):
            raise ValueError
        return result
    except Exception:
        raise AuthenticationError("Could not access the secure ChatGPT credential store.") from None


def _profile(state, profile_id=None):
    selected = profile_id if profile_id is not None else state["active"]
    if selected is not None and not isinstance(selected, str):
        raise ConfigurationError("profile_id must be an identifier returned by ap.accounts().")
    if selected not in state["profiles"]:
        raise AuthenticationError("No ChatGPT account selected. Run ap.sign_in() first.")
    return selected, state["profiles"][selected]


def _public(state, profile_id):
    profile = state["profiles"][profile_id]
    return {
        "profile_id": profile_id,
        "label": profile["label"],
        "connected": profile.get("connected", False),
        "plan_enabled": profile.get("plan_enabled", False),
        "model": profile.get("model"),
        "active": state["active"] == profile_id,
    }


def accounts():
    """List saved connections without reading credentials or contacting OpenAI."""
    state = _read_state()
    return [_public(state, profile_id) for profile_id in state["profiles"]]


def _request(method, url, *, timeout=30, **kwargs):
    import httpx

    try:
        response = httpx.request(method, url, timeout=timeout, follow_redirects=False, **kwargs)
    except httpx.HTTPError:
        raise ProviderError(
            "ChatGPT could not be reached. Check your connection and retry."
        ) from None
    return response


@lru_cache(maxsize=1)
def _discovery():
    response = _request("GET", f"{ISSUER}/.well-known/openid-configuration")
    try:
        response.raise_for_status()
        data = response.json()
        if data["issuer"] != ISSUER:
            raise ValueError
        for key in ("authorization_endpoint", "token_endpoint", "jwks_uri", "revocation_endpoint"):
            url = urlsplit(data[key])
            if url.scheme != "https" or url.netloc != "auth.openai.com":
                raise ValueError
        return data
    except Exception:
        raise AuthenticationError("Could not verify OpenAI's sign-in configuration.") from None


def _verify_identity(id_token, client_id, nonce):
    import jwt

    try:
        if not isinstance(id_token, str) or not id_token:
            raise ValueError
        header = jwt.get_unverified_header(id_token)
        if header.get("alg") not in {"RS256", "ES256"}:
            raise ValueError
        response = _request("GET", _discovery()["jwks_uri"])
        response.raise_for_status()
        keys = jwt.PyJWKSet.from_dict(response.json()).keys
        key = next(key for key in keys if key.key_id == header.get("kid"))
        claims = jwt.decode(
            id_token,
            key.key,
            algorithms=[header["alg"]],
            audience=client_id,
            issuer=ISSUER,
            leeway=5,
            options={"require": ["sub", "exp", "iat", "nonce"]},
        )
        if not secrets.compare_digest(str(claims["nonce"]), nonce):
            raise ValueError
        if not isinstance(claims["sub"], str) or not claims["sub"]:
            raise ValueError
        return claims
    except Exception:
        raise AuthenticationError(
            "ChatGPT returned an invalid identity. Run ap.sign_in() again."
        ) from None


def _authorize(profile, host_id, old_tokens, timeout):
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=")
    result = {}

    class CallbackHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Callback URLs contain authorization codes; never log them.

        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query, keep_blank_values=True)
            valid = (
                parsed.path == "/auth/callback"
                and not result
                and all(len(values) == 1 for values in query.values())
                and secrets.compare_digest(query.get("state", [""])[0], state)
            )
            if not valid:
                self.send_error(400, "Invalid sign-in callback")
                return
            result.update({key: values[0] for key, values in query.items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(
                b"<!doctype html><title>Augplot</title><h1>Return to your notebook</h1>"
                b"<p>Augplot will verify your ChatGPT connection there. You can close this tab.</p>"
            )

    class CallbackServer(HTTPServer):
        def get_request(self):
            connection, address = super().get_request()
            connection.settimeout(2)
            return connection, address

        def handle_error(self, request, client_address):
            pass  # Do not emit callback data or raw network exceptions.

    try:
        server = CallbackServer(("127.0.0.1", 0), CallbackHandler)
    except OSError:
        raise AuthenticationError("Could not start the local ChatGPT sign-in callback.") from None
    with server:
        redirect = f"http://127.0.0.1:{server.server_port}/auth/callback"
        parameters = {
            "client_id": profile.get("client_id") or "dynamic_agent_client",
            "ext_agent_host_id": host_id,
            "response_type": "code",
            "redirect_uri": redirect,
            "scope": SCOPES,
            "resource": RESOURCE,
            "state": state,
            "nonce": nonce,
            "code_challenge_method": "S256",
            "code_challenge": challenge.decode(),
        }
        if not profile.get("client_id"):
            parameters["agent_name_hint"] = "Augplot"
        elif old_tokens and old_tokens.get("id_token"):
            parameters["id_token_hint"] = old_tokens["id_token"]
        if profile.get("connected") and not profile.get("plan_enabled"):
            parameters["prompt"] = "consent"
        url = _discovery()["authorization_endpoint"] + "?" + urlencode(parameters)
        print("Augplot: Continue with ChatGPT in your browser.", flush=True)
        try:
            opened = webbrowser.open(url)
        except Exception:
            opened = False
        if not opened:
            raise AuthenticationError("Could not open your browser. Configure a browser and retry.")
        deadline = time.monotonic() + timeout
        while not result and time.monotonic() < deadline:
            server.timeout = min(0.5, max(0, deadline - time.monotonic()))
            server.handle_request()
    if not result:
        raise AuthenticationError("ChatGPT sign-in timed out. Run ap.sign_in() to try again.")
    if result.get("iss", ISSUER) != ISSUER:
        raise AuthenticationError("ChatGPT returned an invalid sign-in issuer.")
    if "error" in result:
        raise AuthenticationError("ChatGPT sign-in was declined. Run ap.sign_in() to try again.")
    issued = result.get("client_id") or profile.get("client_id")
    if not issued or issued == "dynamic_agent_client" or not result.get("code"):
        raise AuthenticationError("ChatGPT registration was incomplete. Run ap.sign_in() again.")
    if profile.get("client_id") and issued != profile["client_id"]:
        raise AuthenticationError("ChatGPT returned a different account registration.")
    return issued, result["code"], verifier, redirect, nonce


def _token_request(form, *, timeout=30):
    response = _request("POST", _discovery()["token_endpoint"], data=form, timeout=timeout)
    try:
        data = response.json()
    except ValueError:
        data = {}
    if response.status_code >= 400:
        code = data.get("error") if isinstance(data, dict) else None
        if isinstance(code, dict):
            code = code.get("code")
        terminal = (
            form["grant_type"] == "refresh_token"
            and isinstance(code, str)
            and code in _TERMINAL_REFRESH_ERRORS
        )
        error = AuthenticationError(
            "ChatGPT access expired or was revoked. Run ap.sign_in() again."
            if terminal
            else "ChatGPT could not renew or authorize access. Retry sign-in later."
        )
        error.terminal = terminal
        raise error
    if not isinstance(data, dict):
        raise AuthenticationError("ChatGPT returned an invalid authorization response.")
    return data


def _token_set(data, *, previous=None):
    previous = previous or {}
    try:
        lifetime = float(data.get("expires_in", 0))
        if not math.isfinite(lifetime) or lifetime <= 0:
            raise ValueError
        access = data["access_token"]
        if (
            not isinstance(access, str)
            or not access
            or not isinstance(data.get("token_type"), str)
            or data["token_type"].lower() != "bearer"
        ):
            raise ValueError
        scope = data.get("scope")
        if scope is not None and not isinstance(scope, str):
            raise ValueError
        for field in ("refresh_token", "id_token"):
            value = data.get(field, previous.get(field))
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError
        return {
            "access_token": access,
            "refresh_token": data.get("refresh_token", previous.get("refresh_token")),
            "id_token": data.get("id_token", previous.get("id_token")),
            "expires_at": time.time() + lifetime,
            "scopes": scope.split() if scope is not None else previous.get("scopes", []),
        }
    except (KeyError, ValueError, TypeError):
        raise AuthenticationError("ChatGPT returned invalid plan-access credentials.") from None


def sign_in(profile_id=None, *, new_account=False, timeout=180):
    """Open browser sign-in, verify identity, and remember the selected connection."""
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ConfigurationError("Sign-in timeout must be a finite positive number.")
    if new_account and profile_id is not None:
        raise ConfigurationError("Use either profile_id or new_account, not both.")
    _secure_backend()  # Fail before opening the browser if persistence cannot be protected.
    with _locked():
        state = _read_state()
        if state["host_id"] is None:
            state["host_id"] = "urn:uuid:" + str(uuid.uuid4())
            _write_state(state)
        chosen = None if new_account else (profile_id or state["active"])
        if chosen is None and not new_account and len(state["profiles"]) == 1:
            chosen = next(iter(state["profiles"]))
        if chosen is not None:
            chosen, profile = _profile(state, chosen)
            profile = dict(profile)
        else:
            chosen = str(uuid.uuid4())
            profile = {"label": f"Connection {len(state['profiles']) + 1}"}
        old_tokens = _credentials(chosen)
        host_id = state["host_id"]
    issued, code, verifier, redirect, nonce = _authorize(profile, host_id, old_tokens, timeout)
    # Retain registration even when a one-time code expires during exchange.
    with _locked():
        state = _read_state()
        profile["client_id"] = issued
        state["profiles"][chosen] = profile
        _write_state(state)
    data = _token_request(
        {
            "grant_type": "authorization_code",
            "client_id": issued,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect,
            "resource": RESOURCE,
        },
        timeout=min(timeout, 30),
    )
    identity = _verify_identity(data.get("id_token"), issued, nonce)
    if profile.get("subject") and identity["sub"] != profile["subject"]:
        raise AuthenticationError("ChatGPT identity does not match the selected account.")
    scope = data.get("scope", "")
    if not isinstance(scope, str):
        raise AuthenticationError("ChatGPT returned invalid permissions.")
    tokens = (
        _token_set(data)
        if "access_token" in data
        else {
            "id_token": data["id_token"],
            "scopes": scope.split(),
        }
    )
    plan_enabled = PLAN_SCOPE in tokens["scopes"] and bool(tokens.get("access_token"))
    with _locked():
        state = _read_state()
        # Never replace another registration's identity or token set.
        existing = state["profiles"][chosen]
        if existing.get("subject") and existing["subject"] != identity["sub"]:
            raise AuthenticationError("The selected ChatGPT registration changed during sign-in.")
        first_plan = plan_enabled and not existing.get("welcomed", False)
        profile.update(
            subject=identity["sub"],
            label=str(identity.get("email") or identity.get("name") or profile["label"]),
            connected=True,
            plan_enabled=plan_enabled,
            welcomed=existing.get("welcomed", False) or plan_enabled,
        )
        _credentials(chosen, tokens)
        state["profiles"][chosen] = profile
        state["active"] = chosen
        _write_state(state)
    if plan_enabled:
        if first_plan:
            print("Augplot: You're using your ChatGPT plan. " + MANAGE_USAGE_URL, flush=True)
        # The connection survives catalog/network failure, allowing models() to retry later.
        models()
    else:
        print("Augplot: signed in; plan usage is disabled. Run ap.sign_in() to enable it.")
    return next(account for account in accounts() if account["profile_id"] == chosen)


def select_account(profile_id):
    """Select a saved connected account and refresh its model catalog."""
    with _locked():
        state = _read_state()
        chosen, profile = _profile(state, profile_id)
        if not profile.get("connected"):
            raise AuthenticationError("Reconnect this account with ap.sign_in(profile_id=...).")
        if not _credentials(chosen):
            raise AuthenticationError("Saved credentials are missing. Run ap.sign_in().")
        state["active"] = chosen
        _write_state(state)
    if profile.get("plan_enabled"):
        models()
    return next(account for account in accounts() if account["profile_id"] == chosen)


def connection():
    """Resolve persisted account/model metadata, including for offline cache replay."""
    state = _read_state()
    profile_id, profile = _profile(state)
    if not profile.get("model"):
        if profile.get("connected") and profile.get("plan_enabled"):
            models()
            state = _read_state()
            profile_id, profile = _profile(state)
        else:
            raise AuthenticationError("Run ap.sign_in() and enable ChatGPT plan usage first.")
    return profile_id, profile["model"]


def _clear_session(state, profile_id):
    _credentials(profile_id, clear=True)
    state["profiles"][profile_id].update(connected=False, plan_enabled=False)
    _write_state(state)


def _access_token(profile_id):
    # Re-read after obtaining the interprocess lock: refresh tokens can rotate in another kernel.
    with _locked():
        state = _read_state()
        chosen, profile = _profile(state, profile_id)
        if not profile.get("connected"):
            raise AuthenticationError("ChatGPT is signed out. Run ap.sign_in() again.")
        if not profile.get("plan_enabled"):
            raise AuthenticationError(
                "ChatGPT plan use is disabled. Run ap.sign_in() to enable it."
            )
        tokens = _credentials(chosen)
        if not tokens or not tokens.get("access_token"):
            raise AuthenticationError("Saved ChatGPT credentials are missing. Run ap.sign_in().")
        if tokens.get("expires_at", 0) <= time.time() + 60:
            if not tokens.get("refresh_token"):
                _clear_session(state, chosen)
                raise AuthenticationError("ChatGPT access expired. Run ap.sign_in() again.")
            try:
                data = _token_request(
                    {
                        "grant_type": "refresh_token",
                        "client_id": profile["client_id"],
                        "refresh_token": tokens["refresh_token"],
                        "resource": RESOURCE,
                    }
                )
            except AuthenticationError as exc:
                if getattr(exc, "terminal", False):
                    _clear_session(state, chosen)
                raise
            tokens = _token_set(data, previous=tokens)
            _credentials(chosen, tokens)
            profile["plan_enabled"] = PLAN_SCOPE in tokens["scopes"]
            _write_state(state)
            if not profile["plan_enabled"]:
                raise AuthenticationError("ChatGPT plan permission changed. Run ap.sign_in().")
        return tokens["access_token"]


def models():
    """Fetch this account's available models as a table with the default marked."""
    import pandas as pd

    state = _read_state()
    profile_id, _ = _profile(state)
    token = _access_token(profile_id)
    response = _request("GET", RESOURCE + "/models", headers={"Authorization": f"Bearer {token}"})
    if response.status_code >= 400:
        raise ProviderError("Could not list ChatGPT models. Check account access and retry.")
    try:
        catalog = response.json()["models"]
        rows = [
            {"id": item["slug"], "name": item["display_name"]}
            for item in catalog
            if item.get("visibility") == "list"
        ]
        if any(not isinstance(row["id"], str) or not row["id"] for row in rows):
            raise ValueError
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ProviderError("ChatGPT returned an invalid model catalog.") from None
    with _locked():
        state = _read_state()
        _, profile = _profile(state, profile_id)
        if not profile.get("model") and rows:
            profile["model"] = rows[0]["id"]
        profile["catalog"] = rows
        _write_state(state)
        selected = profile.get("model")
    if not rows:
        raise ConfigurationError("No models are available for this ChatGPT account.")
    return pd.DataFrame(
        [{**row, "selected": row["id"] == selected} for row in rows],
        columns=["id", "name", "selected"],
    )


def set_model(model_id):
    """Save the selected account's default for new plots; existing plots keep their model."""
    if not isinstance(model_id, str) or not model_id.strip():
        raise ConfigurationError("model_id must be an available ChatGPT model identifier.")
    profile_id, _ = _profile(_read_state())
    available = models()
    if model_id not in available["id"].tolist():
        raise ConfigurationError("Model is unavailable for this account. Inspect ap.models().")
    with _locked():
        state = _read_state()
        if state["active"] != profile_id:
            raise ConfigurationError("The active account changed. Retry ap.set_model().")
        state["profiles"][profile_id]["model"] = model_id
        _write_state(state)
    return model_id


def sign_out(profile_id=None):
    """Revoke the renewable session and clear local tokens; keep offline plot configuration."""
    with _locked():
        state = _read_state()
        chosen, profile = _profile(state, profile_id)
        tokens = _credentials(chosen)
        revoked = True
        if tokens and tokens.get("refresh_token"):
            revoked = False
            try:
                endpoint = _discovery()["revocation_endpoint"]
                for attempt in range(2):
                    response = _request(
                        "POST",
                        endpoint,
                        timeout=10,
                        data={
                            "token": tokens["refresh_token"],
                            "token_type_hint": "refresh_token",
                            "client_id": profile["client_id"],
                        },
                    )
                    if response.status_code == 200:
                        revoked = True
                        break
                    if response.status_code < 500:
                        break
                    if attempt == 0:
                        time.sleep(0.25)
            except ProviderError:
                pass
            except AuthenticationError:
                pass
        _clear_session(state, chosen)
        result = _public(state, chosen)
    if not revoked:
        print(
            "Augplot: signed out locally; remote revocation was not confirmed. "
            "Disconnect Augplot in ChatGPT Settings > Security and login."
        )
    return result
