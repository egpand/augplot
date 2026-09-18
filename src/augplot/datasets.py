"""Small offline datasets used by the documentation and examples."""

from importlib.resources import files

import pandas as pd

_DATASETS = ("cv_results", "forecast_results", "training_history")


def load_dataset(name: str) -> pd.DataFrame:
    """Return a fresh DataFrame for one of Augplot's bundled example datasets."""
    if name not in _DATASETS:
        available = ", ".join(_DATASETS)
        raise ValueError(f"Unknown dataset {name!r}. Available datasets: {available}.")

    resource = files("augplot").joinpath("data", f"{name}.csv")
    with resource.open("rb") as handle:
        frame = pd.read_csv(handle)

    for column in frame.columns:
        if any(token in column.lower() for token in ("date", "datetime", "timestamp", "week")):
            frame[column] = pd.to_datetime(frame[column])
    return frame
