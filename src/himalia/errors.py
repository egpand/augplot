"""Public exceptions. Provider/runtime diagnostics never include raw data or credentials."""


class HimaliaError(Exception):
    """Base class for Himalia errors."""


class ConfigurationError(HimaliaError, ValueError):
    """Missing or invalid configuration."""


class DataError(HimaliaError, ValueError):
    """Unsupported or unusable input data."""


class ProviderError(HimaliaError):
    """The configured model could not be called."""


class GenerationError(HimaliaError):
    """The model did not produce a valid, executable plot."""

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        self.code = code
