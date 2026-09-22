"""Public exceptions. Provider/runtime diagnostics never include raw data or credentials."""


class AugplotError(Exception):
    """Base class for Augplot errors."""


class ConfigurationError(AugplotError, ValueError):
    """Missing or invalid configuration."""


class DataError(AugplotError, ValueError):
    """Unsupported or unusable input data."""


class ProviderError(AugplotError):
    """The configured model could not be called."""


class ScopeError(AugplotError, ValueError):
    """The request needs upstream modeling or prediction inputs, not plotting code."""


class GenerationError(AugplotError):
    """The model did not produce a valid, executable plot."""

    def __init__(self, message: str, *, code: str | None = None, violations=None):
        super().__init__(message)
        self.code = code
        self.violations = list(violations or [])
