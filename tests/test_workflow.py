import copy
import json
import os
import subprocess
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from conftest import ARRAY_CODE, CV_CODE, DF_CODE, PLOTLY_CODE, response

from himalia import ConfigurationError, GenerationError, ProviderError, Visualizer, plot


def test_fit_refine_and_render(cv_data, fake_model):
    horizontal = CV_CODE.replace("ax.bar(names, means)", "ax.barh(names, means)")
    calls = fake_model(response(CV_CODE), response(horizontal))
    original = copy.deepcopy(cv_data)
    viz = plot(cv_data, show=False)
    assert len(viz.figure.axes[0].patches) == 2
    assert viz.refine("Make it horizontal", show=False) is viz
    context = json.loads(calls[1]["messages"][1]["content"])
    assert context["previous_code"] == CV_CODE
    assert context["original_request"] == "auto"
    changed = {"ridge": {"r2": [0.2]}}
    assert viz.render(changed, title="New", figsize=(6, 3), show=False) is viz
    assert viz.figure.axes[0].patches[0].get_width() == pytest.approx(0.2)
    assert viz.figure.axes[0].get_title() == "New"
    assert tuple(viz.figure.get_size_inches()) == (6, 3)
    assert len(calls) == 2
    assert cv_data == original


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


def test_payload_contains_profile_not_full_data(fake_model):
    calls = fake_model(response(ARRAY_CODE))
    plot(np.arange(10_000), show=False)
    context = json.loads(calls[0]["messages"][1]["content"])
    assert len(json.dumps(context["data_profile"])) <= 20_000
    assert len(context["data_profile"]["data"]["sample"]) == 5


def test_config_precedence_and_environment_resolved_at_call(fake_model, monkeypatch):
    calls = fake_model(response(ARRAY_CODE), response(ARRAY_CODE))
    monkeypatch.setenv("HIMALIA_API_BASE", "https://environment.example")
    viz = Visualizer()
    monkeypatch.setenv("HIMALIA_MODEL", "new/model")
    viz.fit([1, 2], show=False)
    assert calls[0]["model"] == "new/model"
    assert calls[0]["api_base"] == "https://environment.example"
    plot([1, 2], model="explicit/model", api_base="http://localhost:11434", timeout=90, show=False)
    assert calls[1]["model"] == "explicit/model"
    assert calls[1]["api_base"] == "http://localhost:11434"
    assert calls[1]["timeout"] == 90


def test_missing_model_and_unfitted_operations(monkeypatch):
    monkeypatch.delenv("HIMALIA_MODEL", raising=False)
    viz = Visualizer()
    with pytest.raises(ConfigurationError, match="HIMALIA_MODEL"):
        viz.fit([1, 2])
    for operation in (lambda: viz.refine("change"), viz.render, viz.save):
        with pytest.raises(ConfigurationError, match="fit"):
            operation()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "unknown"},
        {"sample_rows": -1},
        {"max_profile_chars": 5},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"max_repairs": 3},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ConfigurationError):
        Visualizer(**kwargs)


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
        ("plotly", PLOTLY_CODE),
    ],
)
def test_backends(backend, code, fake_model):
    if backend == "plotly":
        pytest.importorskip("plotly")
    fake_model(response(code))
    data = pd.DataFrame({"spend": [1, 2], "revenue": [4, 8], "channel": ["a", "b"]})
    viz = plot(data, backend=backend, show=False)
    if backend == "plotly":
        assert len(viz.figure.data) == 2
        viz.render(title="New", figsize=(7, 4), show=False)
        assert viz.figure.layout.width == 700
    else:
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
        if not key.endswith("API_KEY") and key != "HIMALIA_MODEL"
    }
    result = subprocess.run(
        [sys.executable, "-c", "import sys, himalia; assert 'litellm' not in sys.modules"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
