"""Notebook visualizations with inspectable, reusable Python source."""

from .core import plot
from .errors import (
    ConfigurationError,
    DataError,
    GenerationError,
    AugplotError,
    ProviderError,
    ScopeError,
)

__all__ = [
    "plot",
    "AugplotError",
    "ConfigurationError",
    "DataError",
    "GenerationError",
    "ProviderError",
    "ScopeError",
]
__version__ = "0.1.0"
