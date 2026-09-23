import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from conftest import ARRAY_CODE, CV_CODE, DF_CODE, response

from augplot import ConfigurationError, GenerationError, ProviderError, ScopeError, plot
from augplot.core import _Visualization


def test_fit_refine_and_render(cv_data, fake_model):
    horizontal = CV_CODE.replace("ax.bar(names, means)", "ax.barh(names, means)")
    calls = fake_model(response(CV_CODE), response(horizontal))
    original = copy.deepcopy(cv_data)
    viz = plot(cv_data, show=False)
    assert len(viz.figure.axes[0].patches) == 2
    assert viz.refine("Make it horizontal", show=False) is viz
    context = json.loads(calls[1]["messages"][1]["content"])
    assert context["operation"] == "refine"
    assert context["previous_code"] == CV_CODE
    assert context["original_request"] == "auto"
    assert "# Conditional domain guidance: cross-validation results" in calls[1]["messages"][0][
        "content"
    ]
    changed = {"ridge": {"r2": [0.2]}}
    assert viz.render(changed, title="New", figsize=(6, 3), show=False) is viz
    assert viz.figure.axes[0].patches[0].get_width() == pytest.approx(0.2)
    assert viz.figure.axes[0].get_title() == "New"
    assert tuple(viz.figure.get_size_inches()) == (6, 3)
    assert len(calls) == 2
    assert cv_data == original


def test_plot_accepts_nested_cross_validation_arrays(fake_model):
    data = {
        "baseline": {
            "fit_time": np.array([1.0, 1.1, 1.2]),
            "test_f1": np.array([0.3, 0.4, 0.5]),
        },
        "candidate": {
            "fit_time": np.array([1.3, 1.4, 1.5]),
            "test_f1": np.array([0.5, 0.6, 0.7]),
        },
    }
    code = CV_CODE.replace('"r2"', '"test_f1"')
    calls = fake_model(response(code))

    viz = plot(data, show=False)

    assert len(viz.figure.axes[0].patches) == 2
    assert "nested result dictionaries" in calls[0]["messages"][0]["content"]


RECORD_CODE = '''def plot_data(data, *, title=None, figsize=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(data)
    fig, ax = plt.subplots()
    ax.scatter(frame["x"], frame["y"])
    return fig
'''

SCALAR_DICT_CODE = '''def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    names = list(data)
    values = [data[name] for name in names]
    fig, ax = plt.subplots()
    ax.bar(names, values)
    return fig
'''


@pytest.mark.parametrize(
    ("data", "code"),
    [
        ((1, 2, 3), ARRAY_CODE),
        (pd.Series([1, 2, 3], index=pd.date_range("2026-01-01", periods=3)), ARRAY_CODE),
        (np.array([[1, 2], [2, 3]]), ARRAY_CODE),
        (json.loads('[{"x": 1, "y": 2}, {"x": 2, "y": 3}]'), RECORD_CODE),
        ({"x": [1, 2], "y": [2, 3]}, RECORD_CODE),
        (json.loads('{"baseline": 0.4, "candidate": 0.6}'), SCALAR_DICT_CODE),
    ],
    ids=["tuple", "series", "matrix", "json-records", "column-dict", "json-mapping"],
)
def test_plot_accepts_documented_input_forms(data, code, fake_model):
    fake_model(response(code))

    viz = plot(data, show=False)

    assert viz.figure.axes[0].has_data()


def test_fitted_snapshot_does_not_follow_user_mutations(cv_data, fake_model):
    fake_model(response(CV_CODE))
    viz = plot(cv_data, show=False)
    cv_data["ridge"]["r2"][:] = [9]
    viz.render(show=False)
    assert viz.figure.axes[0].patches[0].get_height() == pytest.approx(0.73)


def test_generated_code_gets_a_copy(cv_data, fake_model):
    mutating = CV_CODE.replace(
        "names = list(data)", 'data["ridge"]["r2"].append(99)\n    names = list(data)'
    )
    fake_model(response(mutating))
    original = copy.deepcopy(cv_data)
    viz = plot(cv_data, show=False)
    assert cv_data == original
    assert viz._data == original


def test_repair_once_and_sanitize_diagnostics(fake_model):
    bad = ARRAY_CODE.replace("ax.plot(data)", 'raise ValueError("SECRET_RAW_VALUE")')
    calls = fake_model(response(bad), response(ARRAY_CODE))
    viz = plot([1, 2, 3], show=False)
    assert viz.figure is not None
    assert len(calls) == 2
    diagnostic = json.loads(calls[1]["messages"][-1]["content"])
    assert "ValueError" in diagnostic["diagnostic"]
    assert "SECRET_RAW_VALUE" not in diagnostic["diagnostic"]


def test_repair_receives_safe_scatter_shape_hint(fake_model):
    bad = ARRAY_CODE.replace("ax.plot(data)", "ax.scatter(0, data)")
    calls = fake_model(response(bad), response(ARRAY_CODE))

    viz = plot([17, 23], show=False)

    assert viz.figure is not None
    diagnostic = json.loads(calls[1]["messages"][-1]["content"])["diagnostic"]
    assert "Scatter x and y must have the same number of values" in diagnostic
    assert "17" not in diagnostic
    assert "23" not in diagnostic


@pytest.mark.parametrize("first", ["not json", '{"code": 123}', ""])
def test_malformed_response_is_repaired(fake_model, first):
    calls = fake_model(first, response(ARRAY_CODE))
    assert plot([1, 2], show=False).figure is not None
    assert len(calls) == 2


def test_failure_preserves_state_and_exposes_candidate(cv_data, fake_model):
    bad = CV_CODE.replace("return fig", "return None")
    calls = fake_model(response(CV_CODE), response(bad), response(bad))
    viz = plot(cv_data, show=False)
    previous = (viz.code, viz.figure, viz.explanation, viz.profile)
    with pytest.raises(GenerationError) as caught:
        viz.refine("Change it", show=False)
    assert caught.value.code == bad
    assert (viz.code, viz.figure, viz.explanation, viz.profile) == previous
    assert len(calls) == 3


def test_repairs_disabled_and_provider_errors_not_repaired(fake_model):
    calls = fake_model("invalid")
    with pytest.raises(GenerationError):
        plot([1, 2], max_repairs=0, show=False)
    assert len(calls) == 1
    calls.clear()
    fake_model(ProviderError("bad credentials"))
    with pytest.raises(ProviderError):
        plot([1, 2], show=False)
    assert len(calls) == 1


def test_out_of_scope_request_stops_without_execution_or_repair(fake_model, monkeypatch):
    from augplot import core

    def unexpected_execution(*args, **kwargs):
        raise AssertionError("Out-of-scope requests must not execute code")

    monkeypatch.setattr(core, "execute", unexpected_execution)
    calls = fake_model(
        json.dumps(
            {
                "status": "out_of_scope",
                "code": "",
                "explanation": (
                    "Provide predictions from an upstream model to visualize a forecast."
                ),
            }
        )
    )
    with pytest.raises(ScopeError, match="upstream model"):
        plot([1, 2], prompt="Train a model and forecast next month", show=False)
    assert len(calls) == 1
    assert not Path(".augplot").exists()


@pytest.mark.parametrize("operation", ["fit", "refine"])
def test_scope_refusal_preserves_previous_visualization(fake_model, operation):
    calls = fake_model(
        response(ARRAY_CODE),
        json.dumps(
            {
                "status": "out_of_scope",
                "code": "",
                "explanation": "Supply the prediction intervals; Augplot does not estimate them.",
            }
        ),
    )
    viz = plot([1, 2], show=False)
    previous = (viz.code, viz.figure, viz.history_path)
    saved_files = set(Path(".augplot").rglob("*"))
    with pytest.raises(ScopeError, match="prediction intervals"):
        if operation == "fit":
            viz.fit([3, 4], prompt="Estimate prediction intervals", show=False)
        else:
            viz.refine("Estimate prediction intervals", show=False)
    assert (viz.code, viz.figure, viz.history_path) == previous
    assert set(Path(".augplot").rglob("*")) == saved_files
    assert len(calls) == 2


def test_payload_contains_profile_not_full_data(fake_model):
    calls = fake_model(response(ARRAY_CODE))
    plot(np.arange(10_000), show=False)
    context = json.loads(calls[0]["messages"][1]["content"])
    assert context["operation"] == "generate"
    assert len(json.dumps(context["data_profile"])) <= 20_000
    assert len(context["data_profile"]["data"]["sample"]) == 5
    assert calls[0]["response_format"]["json_schema"]["strict"] is True


def test_config_precedence_and_environment_resolved_at_call(fake_model, monkeypatch):
    calls = fake_model(response(ARRAY_CODE), response(ARRAY_CODE))
    monkeypatch.setenv("AUGPLOT_API_BASE", "https://environment.example")
    viz = _Visualization()
    monkeypatch.setenv("AUGPLOT_MODEL", "new/model")
    viz.fit([1, 2], show=False)
    assert calls[0]["model"] == "new/model"
    assert calls[0]["api_base"] == "https://environment.example"
    plot([1, 2], model="explicit/model", api_base="http://localhost:11434", timeout=90, show=False)
    assert calls[1]["model"] == "explicit/model"
    assert calls[1]["api_base"] == "http://localhost:11434"
    assert calls[1]["timeout"] == 90


def test_missing_model_and_unfitted_operations(monkeypatch):
    monkeypatch.delenv("AUGPLOT_MODEL", raising=False)
    viz = _Visualization()
    with pytest.raises(ConfigurationError, match="AUGPLOT_MODEL"):
        viz.fit([1, 2])
    for operation in (lambda: viz.refine("change"), viz.render, viz.to_python):
        with pytest.raises(ConfigurationError, match="fit"):
            operation()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "unknown"},
        {"backend": "plotly"},
        {"sample_rows": -1},
        {"max_profile_chars": 5},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"max_repairs": 3},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ConfigurationError):
        _Visualization(**kwargs)


@pytest.mark.parametrize(
    "backend,code",
    [
        ("auto", DF_CODE),
        ("seaborn", DF_CODE),
        (
            "matplotlib",
            DF_CODE.replace("    import seaborn as sns\n", "").replace(
                'sns.scatterplot(data=data, x="spend", y="revenue", hue="channel", ax=ax)',
                'ax.scatter(data["spend"], data["revenue"])',
            ),
        ),
    ],
)
def test_backends(backend, code, fake_model):
    fake_model(response(code))
    data = pd.DataFrame({"spend": [1, 2], "revenue": [4, 8], "channel": ["a", "b"]})
    viz = plot(data, backend=backend, show=False)
    assert viz.figure.axes


def test_cleanup_and_style_restoration(fake_model):
    existing = plt.figure()
    before = dict(mpl.rcParams)
    numbers = plt.get_fignums()
    bad = ARRAY_CODE.replace("return fig", "return None")
    fake_model(response(bad), response(ARRAY_CODE))
    plot([1, 2], show=False)
    assert plt.get_fignums() == numbers
    assert dict(mpl.rcParams) == before
    plt.close(existing)


def test_import_without_credentials_does_not_import_sdk():
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.endswith("API_KEY") and key != "AUGPLOT_MODEL"
    }
    result = subprocess.run(
        [sys.executable, "-c", "import sys, augplot; assert 'litellm' not in sys.modules"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
