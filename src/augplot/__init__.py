"""Notebook visualizations with inspectable, reusable Python source."""

from .core import plot
from .datasets import SNS_DATASETS, load_sns_dataset
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
    "load_sns_dataset",
    "SNS_DATASETS",
    "AugplotError",
    "ConfigurationError",
    "DataError",
    "GenerationError",
    "ProviderError",
    "ScopeError",
]
__version__ = "0.1.1"
