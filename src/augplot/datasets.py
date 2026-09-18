"""Convenient access to Seaborn's online example datasets."""

from typing import Any

import pandas as pd
import seaborn as sns

SNS_DATASETS = (
    "anagrams",
    "anscombe",
    "attention",
    "brain_networks",
    "car_crashes",
    "diamonds",
    "dots",
    "dowjones",
    "exercise",
    "flights",
    "fmri",
    "geyser",
    "glue",
    "healthexp",
    "iris",
    "mpg",
    "penguins",
    "planets",
    "seaice",
    "taxis",
    "tips",
    "titanic",
)


def load_sns_dataset(
    name: str,
    *,
    cache: bool = True,
    data_home: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load one of Seaborn's example datasets without importing Seaborn yourself.

    The first load may download the dataset from Seaborn's public data repository.
    Seaborn caches downloads locally by default. Extra keyword arguments are passed
    to :func:`seaborn.load_dataset` and then to ``pandas.read_csv``.
    """
    if name not in SNS_DATASETS:
        available = ", ".join(SNS_DATASETS)
        raise ValueError(f"Unknown Seaborn dataset {name!r}. Available datasets: {available}.")

    return sns.load_dataset(name, cache=cache, data_home=data_home, **kwargs)
