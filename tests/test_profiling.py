import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from augplot import DataError
from augplot.profiling import _category_counts, copy_data, profile_data


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
    profile = profile_data(pd.DataFrame({"a": ["x", "y"]}), sample_rows=0)
    assert profile["data"]["sample_rows"] == []
    summary = profile["data"]["columns"][0]["category_counts"]
    assert {entry["value"] for entry in summary["common"] + summary["rare"]} == {"x", "y"}


def test_full_category_counts_include_rare_labels_outside_row_sample():
    labels = ["Main sequence"] * 1001
    labels[1] = "White dwarfs"
    labels[2] = "main sequence"
    labels[3] = None
    frame = pd.DataFrame({"stellar_class": labels})

    profile = profile_data(frame)

    assert all(
        row["values"][0] == "Main sequence" for row in profile["data"]["sample_rows"]
    )
    summary = profile["data"]["columns"][0]["category_counts"]
    assert summary["distinct"] == 3
    assert summary["missing"] == 1
    assert summary["omitted_distinct"] == 0
    assert {entry["value"]: entry["count"] for entry in summary["common"] + summary["rare"]} == {
        "Main sequence": 998,
        "White dwarfs": 1,
        "main sequence": 1,
    }


def test_series_boolean_and_declared_numeric_categories():
    flag = pd.Series([True, False, pd.NA, False], dtype="boolean", name="flag")
    summary = profile_data(flag, sample_rows=0)["data"]["category_counts"]
    assert summary["missing"] == 1
    assert {entry["value"]: entry["count"] for entry in summary["common"] + summary["rare"]} == {
        False: 2,
        True: 1,
    }

    categories = pd.Series(pd.Categorical([1, 2, 1, None]))
    summary = profile_data(categories)["data"]["category_counts"]
    assert summary["missing"] == 1
    assert summary["distinct"] == 2
    assert "category_counts" not in profile_data(pd.Series([1, 2, 1]))["data"]


@pytest.mark.parametrize(
    "dtype, values",
    [
        ("float64", [np.nan, np.nan]),
        ("float64", [np.nan, 1.0]),
        ("Float64", [pd.NA, pd.NA]),
        ("Int64", [pd.NA, 1]),
        ("int64", [1, 2]),
        ("complex128", [complex(np.nan, 0), 1j]),
    ],
)
def test_numeric_category_counts_do_not_scan_values(monkeypatch, dtype, values):
    series = pd.Series(values, dtype=dtype)

    def reject_iteration(self):
        raise AssertionError("Numeric category counting must not scan observations")

    monkeypatch.setattr(pd.Series, "__iter__", reject_iteration)
    assert _category_counts(series) is None


@pytest.mark.parametrize("container", [list, tuple])
def test_record_fields_get_exact_category_counts(container):
    records = container([{"kind": "A"}, {"kind": "B"}, {"kind": "A"}, {}])
    profile = profile_data(records, sample_rows=0)

    assert profile["data"]["sample"] == []
    assert profile["data"]["fields"][0]["name"] == "kind"
    summary = profile["data"]["fields"][0]["category_counts"]
    assert (summary["distinct"], summary["missing"], summary["omitted_distinct"]) == (2, 1, 0)
    assert {entry["value"]: entry["count"] for entry in summary["common"] + summary["rare"]} == {
        "A": 2,
        "B": 1,
    }


def test_high_cardinality_shows_common_and_rare_with_omitted_count():
    labels = [f"group_{i:02d}" for i in range(30) for _ in range(i + 1)]
    summary = profile_data(pd.Series(labels))["data"]["category_counts"]

    assert summary["distinct"] == 30
    assert summary["omitted_distinct"] == 10
    assert [entry["value"] for entry in summary["common"]] == [
        f"group_{i:02d}" for i in range(29, 19, -1)
    ]
    assert [entry["value"] for entry in summary["rare"]] == [
        f"group_{i:02d}" for i in range(10)
    ]
    assert [entry["count"] for entry in summary["rare"]] == list(range(1, 11))

    bounded = profile_data(pd.Series(labels), max_chars=500)
    trimmed = bounded["data"]["category_counts"]
    assert len(json.dumps(bounded)) <= 500
    assert trimmed["omitted_distinct"] == 30 - len(trimmed["common"]) - len(trimmed["rare"])
    assert trimmed["common"] and trimmed["rare"]


def test_profile_budget_trims_samples_before_category_counts():
    frame = pd.DataFrame(
        {"kind": ["common"] * 30 + ["rare"]},
        index=[f"row-{i}-" + "x" * 200 for i in range(31)],
    )
    profile = profile_data(frame, max_chars=800)

    assert len(json.dumps(profile)) <= 800
    assert len(profile["data"]["sample_rows"]) < 5
    summary = profile["data"]["columns"][0]["category_counts"]
    assert summary["distinct"] == 2
    assert summary["omitted_distinct"] == 0


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


def test_grouped_relationships_retain_nonlinear_coverage_and_rare_groups():
    x = np.linspace(-4, 4, 1201)
    labels = np.full(len(x), "usual", dtype=object)
    labels[1:4] = "rare"
    frame = pd.DataFrame({"input": x, "response": x**2, "kind": labels})
    relationship = profile_data(frame)["data"]["relationships"][0]

    assert relationship["count_scope"] == "exact"
    assert relationship["finite_pairs"] == len(x)
    assert relationship["group_by"] == "kind"
    groups = {group["value"]: group for group in relationship["groups"]}
    assert groups["rare"]["count"] == 3
    usual = groups["usual"]
    assert len(usual["bins"]) == 12
    assert sum(bin_["count"] for bin_ in usual["bins"]) == usual["finite_pairs"]
    medians = [bin_["y"][1] for bin_ in usual["bins"]]
    assert medians[0] > medians[5] < medians[-1]
    assert relationship == profile_data(frame.iloc[::-1])["data"]["relationships"][0]


def test_relationship_request_ranking_and_column_positions():
    frame = pd.DataFrame(np.arange(30).reshape(5, 6), columns=["a", "b", "c", 4, "a", "e"])
    profile = profile_data(frame, request="Compare c with b and a")
    assert profile["profile_version"] == 3
    pairs = profile["data"]["relationships"]
    assert [(pair["x_index"], pair["y_index"]) for pair in pairs] == [(2, 1), (2, 0)]
    assert len(profile_data(frame)["data"]["relationships"]) == 2


def test_finite_positive_quantiles_and_log_relationship_bins():
    frame = pd.DataFrame({
        "x": [0, 1, 2, 3, 4, 5],
        "y": [1e-5, 1, 1e5, np.inf, np.nan, -1],
    })
    profile = profile_data(frame)
    stats = profile["data"]["columns"][1]["stats"]
    assert (stats["finite"], stats["positive"], stats["missing"]) == (4, 3, 1)
    assert stats["quantiles"][0] == -1
    assert stats["positive_log10_quantiles"] == [-5, -2.5, 0, 2.5, 5]
    pair = profile["data"]["relationships"][0]
    assert pair["finite_pairs"] == 4
    assert pair["groups"][0]["bin_y_scale"] == "linear"
    positive = profile_data(frame.iloc[:3])["data"]["relationships"][0]["groups"][0]
    assert positive["bin_y_scale"] == "log10"
    assert positive["bins"][0]["y"] == [-5, -5, -5]
    json.dumps(profile, allow_nan=False)


def test_relationship_sample_scope_and_empty_finite_groups():
    frame = pd.DataFrame({
        "x": np.arange(100_001), "y": np.ones(100_001),
        "kind": ["missing"] + ["valid"] * 100_000,
    })
    frame.loc[0, "y"] = np.nan
    relationship = profile_data(frame)["data"]["relationships"][0]
    assert relationship["observations"] == 100_000
    assert relationship["sampled"] is True
    assert relationship["count_scope"] == "sampled"
    group = next(group for group in relationship["groups"] if group["value"] == "missing")
    assert group == {"value": "missing", "count": 1, "finite_pairs": 0}


@pytest.mark.parametrize("budget", [500, 1000, 5000, 20_000])
def test_relationships_obey_budget_and_mark_removed_bins(budget):
    frame = pd.DataFrame({
        "x": np.linspace(-4, 4, 21636),
        "y": np.linspace(-4, 4, 21636)**2,
        "z": np.geomspace(1e-6, 1e6, 21636),
        "kind": np.tile(list("abcdefg"), 3091)[:21636],
    })
    profile = profile_data(frame, max_chars=budget)
    assert len(json.dumps(profile, allow_nan=False)) <= budget
    for pair in profile["data"].get("relationships", []):
        for group in pair["groups"]:
            if "bins" in group:
                assert sum(bin_["count"] for bin_ in group["bins"]) == group["finite_pairs"]
            elif group["finite_pairs"]:
                assert group["bins_omitted"] is True
    if budget == 20_000:
        assert all(len(pair["groups"]) == 7 for pair in profile["data"]["relationships"])


def test_high_cardinality_is_not_used_for_grouping_and_extremes_are_json_safe():
    frame = pd.DataFrame({
        "x": [np.finfo(float).max] * 30,
        "y": np.arange(30),
        "id": [f"row-{i}" for i in range(30)],
    })
    profile = profile_data(frame)
    assert "group_by" not in profile["data"]["relationships"][0]
    json.dumps(profile, allow_nan=False)
