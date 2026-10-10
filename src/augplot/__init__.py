"""Notebook visualizations with inspectable, reusable Python source."""

from .auth import accounts, models, select_account, set_model, sign_in, sign_out
from .core import plot
from .datasets import SNS_DATASETS, load_sns_dataset
from .errors import (
    AugplotError,
    AuthenticationError,
    ConfigurationError,
    DataError,
    GenerationError,
    InputRequiredError,
    ProviderError,
    ScopeError,
)

__all__ = [
    "plot",
    "sign_in",
    "sign_out",
    "accounts",
    "select_account",
    "models",
    "set_model",
    "load_sns_dataset",
    "SNS_DATASETS",
    "AugplotError",
    "AuthenticationError",
    "ConfigurationError",
    "DataError",
    "GenerationError",
    "InputRequiredError",
    "ProviderError",
    "ScopeError",
]
__version__ = "0.2.0"
