import json

import pytest
from conftest import CV_CODE, response

from augplot import (
    ConfigurationError,
    GenerationError,
    InputRequiredError,
    ProviderError,
    core,
    plot,
)


@pytest.mark.parametrize("budget", [0, -1, True, 1.0, "1"])
def test_invalid_model_call_budgets(budget, cv_data, fake_model):
    fake_model()
    with pytest.raises(ConfigurationError, match="max_model_calls"):
        plot(cv_data, max_model_calls=budget, show=False)


def test_one_call_budget_stops_repair_and_reports_failure(cv_data, fake_model):
    calls = fake_model(response(CV_CODE.replace("np.mean", "np.load")), response(CV_CODE))
    with pytest.raises(GenerationError) as caught:
        plot(cv_data, max_repairs=2, max_model_calls=1, cache_dir=None, show=False)
    assert caught.value.model_calls == 1
    assert len(calls) == 1
    assert "after 1 attempt(s)" in str(caught.value)


def test_call_counts_reset_for_refine_replay_render_and_export(cv_data, fake_model, tmp_path):
    calls = fake_model(response(CV_CODE), response(CV_CODE))
    viz = plot(cv_data, max_model_calls=1, show=False)
    assert viz.model_calls == 1
    replay = plot(cv_data, max_model_calls=1, show=False)
    assert replay.cache_hit and replay.model_calls == 0
    viz.refine("Use the same measured scores", show=False)
    assert viz.model_calls == 1
    viz.render(show=False)
    assert viz.model_calls == 0
    viz.to_python(tmp_path / "budget_export.py")
    assert viz.model_calls == 0
    assert len(calls) == 2


def test_one_call_budget_does_not_prompt_for_unusable_clarification(
    cv_data, fake_model, monkeypatch,
):
    calls = fake_model(
        json.dumps({"status": "needs_input", "code": "", "explanation": "Which score?"})
    )
    monkeypatch.setattr(core, "_has_notebook_kernel", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("No budget to use an answer"))
    with pytest.raises(InputRequiredError) as caught:
        plot(cv_data, max_model_calls=1, cache_dir=None, show=False)
    assert caught.value.model_calls == len(calls) == 1


def test_clarification_and_repair_share_the_budget(cv_data, fake_model, monkeypatch):
    calls = fake_model(
        json.dumps({"status": "needs_input", "code": "", "explanation": "Which score?"}),
        response(CV_CODE.replace("np.mean", "np.load")),
        response(CV_CODE),
    )
    monkeypatch.setattr(core, "_has_notebook_kernel", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: "R²")
    with pytest.raises(GenerationError) as caught:
        plot(cv_data, max_model_calls=2, max_repairs=2, cache_dir=None, show=False)
    assert caught.value.model_calls == len(calls) == 2


def test_provider_failure_reports_attempted_call(cv_data, fake_model, monkeypatch):
    fake_model()

    def unavailable(**kwargs):
        raise ProviderError("Unavailable")

    monkeypatch.setattr(core.provider, "complete", unavailable)
    with pytest.raises(ProviderError) as caught:
        plot(cv_data, max_model_calls=1, show=False)
    assert caught.value.model_calls == 1


def test_helper_implementation_version_changes_history_identity(cv_data, fake_model, monkeypatch):
    calls = fake_model(response(CV_CODE), response(CV_CODE))
    first = plot(cv_data, show=False)
    monkeypatch.setattr(core, "PLOTTING_VERSION", "future-version")
    second = plot(cv_data, show=False)
    assert not first.cache_hit and not second.cache_hit
    assert first.history_path != second.history_path
    assert len(calls) == 2
