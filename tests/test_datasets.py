import pandas as pd
import pytest

import augplot as ap


@pytest.mark.parametrize(
    "name,columns,rows",
    [
        ("training_history", {"epoch", "training_loss", "validation_loss"}, 30),
        ("cv_results", {"model", "fold", "accuracy", "roc_auc"}, 25),
        (
            "forecast_results",
            {
                "week",
                "observed_latency_ms",
                "forecast_latency_ms",
                "lower_95_ms",
                "upper_95_ms",
            },
            28,
        ),
    ],
)
def test_load_dataset(name, columns, rows):
    frame = ap.load_dataset(name)
    assert isinstance(frame, pd.DataFrame)
    assert set(frame.columns) == columns
    assert len(frame) == rows


def test_load_dataset_returns_fresh_frame():
    first = ap.load_dataset("training_history")
    first.loc[0, "training_loss"] = -1
    assert ap.load_dataset("training_history").loc[0, "training_loss"] > 0


def test_load_dataset_parses_dates_and_rejects_unknown_names():
    assert pd.api.types.is_datetime64_any_dtype(ap.load_dataset("forecast_results")["week"])
    with pytest.raises(ValueError, match="Available datasets"):
        ap.load_dataset("penguins")
