"""Notebook visualizations with inspectable, reusable Python source."""

from .core import plot
from .datasets import load_dataset
from .errors import (
    AugplotError,
    ConfigurationError,
    DataError,
    GenerationError,
    ProviderError,
    ScopeError,
)

__all__ = [
    "plot",
    "load_dataset",
    "AugplotError",
    "ConfigurationError",
    "DataError",
    "GenerationError",
    "ProviderError",
    "ScopeError",
]
__version__ = "0.1.0"
