"""Public exceptions. Provider/runtime diagnostics never include raw data or credentials."""

from pathlib import Path
from typing import TypedDict


class ValidationViolation(TypedDict):
    """A machine-readable generated-code validation failure."""

    rule: str
    message: str


class AugplotError(Exception):
    """Base class for Augplot errors."""

    model_calls = 0


class ConfigurationError(AugplotError, ValueError):
    """Missing or invalid configuration."""


class DataError(AugplotError, ValueError):
    """Unsupported or unusable input data."""


class ProviderError(AugplotError):
    """The configured model could not be called."""


class AuthenticationError(ProviderError):
    """A ChatGPT connection could not be established or renewed."""


class ScopeError(AugplotError, ValueError):
    """The request needs upstream modeling or prediction inputs, not plotting code."""


class InputRequiredError(AugplotError, ValueError):
    """A supported plot needs a choice or input the request did not provide."""


class GenerationError(AugplotError):
    """The model did not produce a valid, executable plot."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        violations: list[ValidationViolation] | None = None,
        diagnostics_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.violations = list(violations or [])
        self.diagnostics_path = diagnostics_path
