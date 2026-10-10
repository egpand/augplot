"""Bounded, JSON-safe profiles of ordinary in-memory data science objects."""

import copy
import datetime as dt
import json
import math
from itertools import islice

import numpy as np
import pandas as pd

from .errors import DataError

_SCALARS = (str, bool, int, float, dt.date, dt.timedelta, np.generic)


def validate_data(data):
    """Reject cycles/custom objects before copying or inspecting their representations."""
    if not isinstance(data, (dict, list, tuple, np.ndarray, pd.DataFrame, pd.Series)):
        raise DataError("Expected a dictionary, list, tuple, NumPy array, DataFrame, or Series.")
    if isinstance(data, np.ndarray) and data.ndim not in (1, 2):
        raise DataError("Only one- and two-dimensional NumPy arrays are supported.")
    if (
        len(data) == 0
        or isinstance(data, np.ndarray) and data.size == 0
        or isinstance(data, pd.DataFrame) and data.empty
    ):
        raise DataError("Cannot visualize empty data.")

    def visit(value, ancestors, depth):
        if depth > 32:
            raise DataError("Input nesting exceeds 32 levels.")
        if value is None or value is pd.NA or value is pd.NaT:
            return
        if isinstance(value, _SCALARS):
            return
        identity = id(value)
        if identity in ancestors:
            raise DataError("Cyclic input containers are not supported.")
        chain = ancestors | {identity}
        if isinstance(value, pd.DataFrame):
            for column in value.columns:
                visit(column, chain, depth + 1)
            visit(value.index.tolist(), chain, depth + 1)
            for index in range(len(value.columns)):
                visit(value.iloc[:, index], chain, depth + 1)
        elif isinstance(value, pd.Series):
            visit(value.name, chain, depth + 1)
            visit(value.index.tolist(), chain, depth + 1)
            if value.dtype == object or isinstance(value.dtype, pd.CategoricalDtype):
                for item in value:
                    visit(item, chain, depth + 1)
        elif isinstance(value, np.ndarray):
            if value.ndim not in (1, 2):
                raise DataError("Only one- and two-dimensional NumPy arrays are supported.")
            if value.dtype.hasobject:
                for item in value.flat:
                    visit(item, chain, depth + 1)
        elif isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, (str, int, float, bool, tuple, np.generic)):
                    raise DataError("Dictionary keys must be scalar values or tuples.")
                visit(key, chain, depth + 1)
                visit(item, chain, depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, chain, depth + 1)
        else:
            raise DataError(
                "Input contains an unsupported object; convert it to ordinary data first."
            )

    visit(data, set(), 0)


def copy_data(data):
    """Copy nested values too: Pandas' deep copy leaves object-dtype cells shared."""
    if isinstance(data, pd.DataFrame):
        result = data.copy(deep=True)
        for index in range(len(data.columns)):
            column = data.iloc[:, index]
            if column.dtype == object:
                result.isetitem(index, column.map(copy_data))
        return result
    if isinstance(data, pd.Series):
        result = data.copy(deep=True)
        if data.dtype == object:
            for index in range(len(data)):
                result.iloc[index] = copy_data(data.iloc[index])
        return result
    if isinstance(data, dict):
        return {key: copy_data(value) for key, value in data.items()}
    if isinstance(data, list):
        return [copy_data(value) for value in data]
    if isinstance(data, tuple):
        return tuple(copy_data(value) for value in data)
    if isinstance(data, np.ndarray) and data.dtype.hasobject:
        result = data.copy()
        for index in range(data.size):
            result.flat[index] = copy_data(data.flat[index])
        return result
    return copy.deepcopy(data)


def _indices(length, count):
    return np.linspace(0, length - 1, min(length, count), dtype=int).tolist() if length else []


def _value(value, depth=0):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        if isinstance(value, (np.datetime64, np.timedelta64)):
            return str(value)
        if isinstance(value, np.floating):
            return _value(float(value), depth)
        return _value(value.item(), depth)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 200 else value[:200] + "…[truncated]"
    if isinstance(value, (dt.date, dt.timedelta)):
        return str(value)
    if depth >= 4:
        return {"type": type(value).__name__, "truncated": True}
    if isinstance(value, dict):
        return {
            "items": [
                [_value(k, depth + 1), _value(v, depth + 1)] for k, v in islice(value.items(), 5)
            ],
            "length": len(value),
        }
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_value(value[i], depth + 1) for i in _indices(len(value), 5)]
    return {"type": type(value).__name__}


def _stats(values):
    """Bound statistical work to 1,000 evenly spaced observations; label the sample."""
    if not len(values):
        return {}
    sample = pd.Series(values).iloc[_indices(len(values), 1000)]
    result = {
        "observations": len(sample),
        "sampled": len(sample) < len(values),
        "missing": int(sample.isna().sum()),
    }
    if pd.api.types.is_numeric_dtype(sample.dtype) and not pd.api.types.is_complex_dtype(
        sample.dtype
    ):
        numeric = sample.to_numpy(dtype=float, na_value=np.nan)
        finite = numeric[np.isfinite(numeric)]
        if len(finite):
            # Scale first to avoid overflow while averaging large, finite values.
            scale = float(np.abs(finite).max()) or 1.0
            normalized = finite / scale
            result.update(
                min=float(finite.min()),
                max=float(finite.max()),
                mean=float(normalized.mean()) * scale,
            )
            if len(finite) > 1:
                result["std"] = _value(float(normalized.std(ddof=1)) * scale)
    return result


def _category_counts(values):
    """Count observed scalar categories without sampling or coercing numeric columns."""
    dtype = getattr(values, "dtype", None)
    categorical = isinstance(dtype, pd.CategoricalDtype)
    if (
        dtype is not None
        and not categorical
        and pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
    ):
        return None
    declared = categorical or (
        dtype is not None
        and (
            pd.api.types.is_bool_dtype(dtype)
            or (
                pd.api.types.is_string_dtype(dtype)
                and not pd.api.types.is_object_dtype(dtype)
            )
        )
    )
    counts = {}
    missing = 0
    for value in values:
        absent = pd.isna(value)
        if isinstance(absent, (bool, np.bool_)) and absent:
            missing += 1
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if not categorical and not isinstance(value, (str, bool)):
            return None
        try:
            key = (type(value), value)
            hash(key)
        except TypeError:
            return None
        counts[key] = counts.get(key, 0) + 1
    if not counts and not declared:
        return None

    def tie_break(key):
        return (key[0].__name__, json.dumps(_value(key[1]), sort_keys=True))

    tie_breaks = {key: tie_break(key) for key in counts}
    ordered = sorted(counts, key=lambda key: (-counts[key], tie_breaks[key]))
    if len(ordered) <= 20:
        split = (len(ordered) + 1) // 2
        common = ordered[:split]
        rare = ordered[split:]
    else:
        common = ordered[:10]
        rare = ordered[-10:]
    rare = sorted(rare, key=lambda key: (counts[key], tie_breaks[key]))

    def entries(keys):
        return [{"value": _value(key[1]), "count": counts[key]} for key in keys]

    return {
        "distinct": len(ordered),
        "missing": missing,
        "common": entries(common),
        "rare": entries(rare),
        "omitted_distinct": len(ordered) - len(common) - len(rare),
    }


def profile_data(data, *, sample_rows=5, max_chars=20_000):
    """Return a profile capped by serialized character count, never a sliced JSON string."""
    if not isinstance(sample_rows, int) or not 0 <= sample_rows <= 100:
        raise ValueError("sample_rows must be an integer between 0 and 100.")
    if not isinstance(max_chars, int) or not 500 <= max_chars <= 100_000:
        raise ValueError("max_chars must be between 500 and 100,000.")
    validate_data(data)
    node_count = 0

    def describe(value, depth=0):
        nonlocal node_count
        node_count += 1
        result = {"type": type(value).__name__}
        if depth >= 6 or node_count > 300:
            return {**result, "truncated": True}
        if isinstance(value, pd.DataFrame):
            result.update(
                shape=list(value.shape),
                index_type=type(value.index).__name__,
                index_names=[_value(name) for name in value.index.names],
            )
            result["columns"] = []
            for i in range(min(len(value.columns), 100)):
                column = value.iloc[:, i]
                description = {
                    "name": _value(value.columns[i]),
                    "dtype": str(column.dtype),
                    "stats": _stats(column),
                }
                categories = _category_counts(column)
                if categories is not None:
                    description["category_counts"] = categories
                result["columns"].append(description)
            result["sample_rows"] = [
                {
                    "index": _value(value.index[i]),
                    "values": [_value(x) for x in value.iloc[i, :100]],
                }
                for i in _indices(len(value), sample_rows)
            ]
            if len(value.columns) > 100:
                result["truncated"] = True
        elif isinstance(value, pd.Series):
            result.update(
                length=len(value),
                name=_value(value.name),
                dtype=str(value.dtype),
                stats=_stats(value),
                sample=[
                    {"index": _value(value.index[i]), "value": _value(value.iloc[i])}
                    for i in _indices(len(value), sample_rows)
                ],
            )
            categories = _category_counts(value)
            if categories is not None:
                result["category_counts"] = categories
        elif isinstance(value, np.ndarray):
            result.update(
                shape=list(value.shape),
                dtype=str(value.dtype),
                sample=[_value(value[i]) for i in _indices(len(value), sample_rows)],
            )
            if value.ndim == 1:
                result["stats"] = _stats(value)
        elif isinstance(value, dict):
            result.update(
                length=len(value),
                items=[
                    {"key": _value(key), "value": describe(item, depth + 1)}
                    for key, item in islice(value.items(), 100)
                ],
            )
            if len(value) > 100:
                result["truncated"] = True
        elif isinstance(value, (list, tuple)):
            result.update(
                length=len(value),
                sample=[
                    {"position": i, "value": describe(value[i], depth + 1)}
                    for i in _indices(len(value), sample_rows)
                ],
            )
            if value and all(isinstance(v, (int, float, np.number)) for v in value):
                result["stats"] = _stats(value)
            if value and all(isinstance(item, dict) for item in value):
                names = {}
                for record in value:
                    for name in record:
                        if name not in names and len(names) < 100:
                            names[name] = None
                        elif name not in names:
                            result["truncated"] = True
                fields = []
                for name in names:
                    categories = _category_counts(record.get(name) for record in value)
                    if categories is not None:
                        fields.append({"name": _value(name), "category_counts": categories})
                if fields:
                    result["fields"] = fields
        else:
            result["value"] = _value(value)
        return result

    profile = {"profile_version": 2, "data": describe(data)}

    def shrink(node):
        samples, categories, other = [], [], []

        def collect(value):
            if isinstance(value, dict):
                is_category_summary = "omitted_distinct" in value and "distinct" in value
                for key, item in value.items():
                    if isinstance(item, list) and item:
                        candidate = (len(json.dumps(item, ensure_ascii=True)), value, key)
                        if key in {"sample", "sample_rows"}:
                            samples.append(candidate)
                        elif is_category_summary and key in {"common", "rare"}:
                            categories.append(candidate)
                        else:
                            other.append(candidate)
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(node)
        for group in (samples, categories, other):
            if group:
                return max(group, key=lambda candidate: candidate[0])
        return None

    while len(json.dumps(profile, ensure_ascii=True)) > max_chars:
        candidates = shrink(profile)
        profile["truncated"] = True
        if candidates is None:
            profile = {
                "profile_version": 2,
                "data": {"type": type(data).__name__},
                "truncated": True,
            }
            break
        _, parent, key = candidates
        original_length = len(parent[key])
        parent[key] = parent[key][: original_length // 2]
        if key in {"common", "rare"} and "omitted_distinct" in parent:
            parent["omitted_distinct"] += original_length - len(parent[key])
        parent["truncated"] = True
    return profile
