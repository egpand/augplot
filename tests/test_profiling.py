import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from augplot import DataError
from augplot.profiling import copy_data, profile_data


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame({"a": [1, None, 3], "when": pd.date_range("2025-01-01", periods=3)}),
        pd.Series([1, pd.NA, 3], dtype="Int64", name="scores"),
        np.array([1, 2, 3]),
        np.array([[1, 2], [3, 4]]),
        [{"label": "A", "score": 1}, {"label": "B", "score": 2}],
        {"test_score": np.array([0.7, 0.8]), "fit_time": np.array([1.0, 1.2])},
        {"model": {"score": [1, np.nan, np.inf]}},
        pd.DataFrame({"a": pd.Categorical(["x", "y"])}),
    ],
)
def test_supported_inputs_are_deterministic_and_json_safe(data):
    first = profile_data(data)
    assert first == profile_data(data)
    assert len(json.dumps(first, allow_nan=False)) <= 20_000


def test_profile_is_bounded_and_uses_spread_sample():
    data = pd.DataFrame({"value": np.arange(10_000)})
    profile = profile_data(data)
    assert [r["values"][0] for r in profile["data"]["sample_rows"]] == [0, 2499, 4999, 7499, 9999]
    assert profile["data"]["columns"][0]["stats"]["sampled"] is True
    large = {f"column_{i}": ["x" * 500] * 10 for i in range(100)}
    small = profile_data(large, max_chars=500)
    assert len(json.dumps(small)) <= 500
    assert small["truncated"]


def test_samples_can_be_disabled():
    profile = profile_data(pd.DataFrame({"a": [1, 2]}), sample_rows=0)
    assert profile["data"]["sample_rows"] == []


def test_duplicate_columns_and_nonstring_keys():
    frame = pd.DataFrame([[1, 2]], columns=["score", "score"])
    assert len(profile_data(frame)["data"]["columns"]) == 2
    assert profile_data({("score", 1): [1, 2]})["data"]["items"][0]["key"] == ["score", 1]


@pytest.mark.parametrize(
    "data",
    [
        [],
        {},
        np.zeros((2, 2, 2)),
        object(),
        {"a": object()},
        '[{"x": 1}]',
        Path("results.json"),
        iter([1, 2]),
        pd.Index([1, 2]),
    ],
)
def test_unsupported_or_empty_data(data):
    with pytest.raises(DataError):
        profile_data(data)


def test_cycles_rejected_but_shared_references_supported():
    values = [1, 2]
    profile_data({"a": values, "b": values})
    values.append(values)
    with pytest.raises(DataError, match="Cyclic"):
        profile_data(values)


def test_never_calls_custom_repr():
    class Custom:
        def __repr__(self):
            raise AssertionError("Do not inspect me")

    with pytest.raises(DataError):
        profile_data(pd.DataFrame({"value": [Custom()]}))


@pytest.mark.parametrize(
    "data", [np.array(1), np.empty((2, 0)), pd.DataFrame(index=[1, 2])]
)
def test_degenerate_shapes_raise_data_error(data):
    with pytest.raises(DataError):
        profile_data(data)


@pytest.mark.parametrize(
    "data",
    [
        np.array([1e308, 1e308]),
        np.array([-1e308, 1e308]),
        [np.longdouble("1.2")],
    ],
)
def test_extreme_numeric_values_are_json_safe(data):
    json.dumps(profile_data(data), allow_nan=False)


def test_copy_nested_pandas_objects():
    frame = pd.DataFrame({"values": [[1, 2]]})
    data = {"frame": frame, "series": frame["values"]}
    copied = copy_data(data)
    copied["frame"].iloc[0, 0].append(3)
    copied["series"].iloc[0].append(4)
    assert frame.iloc[0, 0] == [1, 2]
    assert copied["frame"].iloc[0, 0] == [1, 2, 3]
    assert copied["series"].iloc[0] == [1, 2, 4]


@pytest.mark.parametrize("kwargs", [{"sample_rows": -1}, {"max_chars": 0}])
def test_profile_options_validated(kwargs):
    with pytest.raises(ValueError):
        profile_data([1], **kwargs)
