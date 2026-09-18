import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import ARRAY_CODE, CV_CODE, response

from augplot import ConfigurationError, GenerationError, plot
from augplot.history import fingerprint


def test_original_and_refinement_chain_replay_in_fresh_process(cv_data, fake_model):
    horizontal = CV_CODE.replace("ax.bar(names, means)", "ax.barh(names, means)")
    titled = horizontal.replace('"Model comparison"', '"Refined comparison"')
    calls = fake_model(response(CV_CODE), response(horizontal), response(titled))
    viz = plot(cv_data, show=False)
    original = viz.history_path
    viz.refine("Horizontal", show=False)
    first = viz.history_path
    viz.refine("Change title", show=False)
    second = viz.history_path
    assert len(calls) == 3
    assert len(set((original, first, second))) == 3
    assert original.name.startswith("initial_")
    assert first.name.startswith("refined_1_")
    assert second.name.startswith("refined_2_")
    assert original.read_text() == CV_CODE

    script = f'''
from augplot import plot, provider
def forbidden(**kwargs):
    raise AssertionError("Replay must never call a provider")
provider.complete = forbidden
viz = plot({cv_data!r}, show=False)
assert viz.cache_hit and str(viz.history_path) == {str(original)!r}
assert viz.code == {CV_CODE!r}
viz.refine("Horizontal", show=False)
assert viz.cache_hit and str(viz.history_path) == {str(first)!r}
viz.refine("Change title", show=False)
assert viz.cache_hit and str(viz.history_path) == {str(second)!r}
assert viz.figure.axes[0].get_title() == "Refined comparison"
assert "litellm" not in __import__("sys").modules
'''
    env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY")}
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_refinement_branches_do_not_replace_one_another(fake_model):
    code_a = ARRAY_CODE.replace('"Values"', '"A"')
    code_b = ARRAY_CODE.replace('"Values"', '"B"')
    calls = fake_model(response(ARRAY_CODE), response(code_a), response(code_b))
    initial = plot([1, 2], show=False)
    initial.refine("Title A", show=False)
    first = initial.history_path
    branch = plot([1, 2], show=False).refine("Title B", show=False)
    assert branch.code == code_b
    replay = plot([1, 2], show=False).refine("Title A", show=False)
    assert replay.history_path == first
    assert replay.code == code_a
    assert len(calls) == 3


def test_same_prompt_on_different_parents_is_a_new_step(fake_model):
    calls = fake_model(response(ARRAY_CODE), response(ARRAY_CODE), response(ARRAY_CODE))
    viz = plot([1, 2], show=False)
    viz.refine("Improve labels", show=False)
    first = viz.history_path
    viz.refine("Improve labels", show=False)
    assert viz.history_path != first
    assert len(calls) == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": "New instruction"},
        {"model": "test/other"},
        {"api_base": "http://localhost:1234"},
        {"backend": "matplotlib"},
        {"sample_rows": 0},
        {"max_profile_chars": 1000},
    ],
)
def test_generation_setting_changes_miss_cache(fake_model, kwargs):
    calls = fake_model(response(ARRAY_CODE), response(ARRAY_CODE))
    original = plot([1, 2], show=False)
    changed = plot([1, 2], show=False, **kwargs)
    assert len(calls) == 2
    assert original.history_path != changed.history_path


def test_unsampled_data_changes_miss_cache(fake_model):
    calls = fake_model(response(ARRAY_CODE), response(ARRAY_CODE))
    data = np.arange(5000)
    original = plot(data, show=False)
    data[123] = -999
    changed = plot(data, show=False)
    assert len(calls) == 2
    assert original.history_path != changed.history_path


def test_display_settings_and_render_do_not_generate_versions(fake_model):
    calls = fake_model(response(ARRAY_CODE))
    viz = plot([1, 2], show=False)
    files = set(Path(".augplot").rglob("*"))
    viz.render([3, 4], show=False)
    replay = plot([1, 2], display_format="svg", timeout=90, show=False)
    assert replay.cache_hit
    assert files == set(Path(".augplot").rglob("*"))
    assert len(calls) == 1


def test_explicit_regeneration_preserves_old_source_and_replaces_lookup(fake_model):
    revised = ARRAY_CODE.replace('"Values"', '"New"')
    calls = fake_model(response(ARRAY_CODE), response(revised))
    old = plot([1, 2], show=False)
    new = plot([1, 2], regenerate=True, show=False)
    assert new.history_path != old.history_path
    assert old.history_path.read_text() == ARRAY_CODE
    assert plot([1, 2], show=False).code == revised
    assert len(calls) == 2


def test_regenerating_refinement_keeps_original_and_previous_revision(fake_model):
    revised = ARRAY_CODE.replace('"Values"', '"New"')
    calls = fake_model(response(ARRAY_CODE), response(ARRAY_CODE), response(revised))
    old = plot([1, 2], show=False).refine("Improve", show=False)
    new = plot([1, 2], show=False).refine("Improve", regenerate=True, show=False)
    assert old.history_path.read_text() == ARRAY_CODE
    assert new.code == revised
    assert plot([1, 2], show=False).code == ARRAY_CODE
    assert plot([1, 2], show=False).refine("Improve", show=False).code == revised
    assert len(calls) == 3


def test_failed_regeneration_preserves_lookup_and_state(fake_model):
    calls = fake_model(response(ARRAY_CODE), "invalid")
    viz = plot([1, 2], max_repairs=0, show=False)
    before = (viz.code, viz.figure, viz.history_path)
    with pytest.raises(GenerationError):
        viz.fit([1, 2], regenerate=True, show=False)
    assert (viz.code, viz.figure, viz.history_path) == before
    assert plot([1, 2], show=False).history_path == before[2]
    assert len(calls) == 2


def test_corrupt_history_fails_without_inference(fake_model):
    calls = fake_model(response(ARRAY_CODE), response(ARRAY_CODE))
    viz = plot([1, 2], show=False)
    viz.history_path.write_text("corrupt source")
    with pytest.raises(ConfigurationError, match="regenerate=True"):
        plot([1, 2], show=False)
    assert len(calls) == 1
    assert plot([1, 2], regenerate=True, show=False).figure is not None


def test_cached_execution_failure_does_not_call_model(fake_model, monkeypatch):
    calls = fake_model(response(ARRAY_CODE))
    plot([1, 2], show=False)

    def broken(*args, **kwargs):
        raise GenerationError("Execution failed")

    monkeypatch.setattr("augplot.core.execute", broken)
    with pytest.raises(GenerationError, match="no LLM request"):
        plot([1, 2], show=False)
    assert len(calls) == 1


def test_cache_can_be_disabled_or_relocated(fake_model, tmp_path):
    calls = fake_model(*(response(ARRAY_CODE) for _ in range(4)))
    for _ in range(2):
        viz = plot([1, 2], cache_dir=None, show=False)
        assert viz.history_path is None
    viz.refine("Change", show=False)
    assert not Path(".augplot").exists()
    viz = plot([1, 2], cache_dir=tmp_path / "custom", show=False)
    assert viz.history_path.is_relative_to(tmp_path / "custom")
    assert len(calls) == 4


def test_to_python_uses_current_refinement(fake_model, tmp_path):
    revised = ARRAY_CODE.replace('"Values"', '"New"')
    fake_model(response(ARRAY_CODE), response(revised))
    plot([1, 2], show=False).refine("New title", show=False)
    replay = plot([1, 2], show=False).refine("New title", show=False)
    path = replay.to_python(tmp_path / "vis_utils.py", function_name="plot_results")
    assert "def plot_results(" in path.read_text()
    assert "'New'" in path.read_text()


def test_history_does_not_store_data_or_settings_credentials(fake_model):
    fake_model(response(ARRAY_CODE))
    viz = plot([193847.65789, 2], api_base="https://SECRET_ENDPOINT", show=False)
    for path in viz.history_path.parent.iterdir():
        contents = path.read_text()
        assert "193847.65789" not in contents
        assert "SECRET_ENDPOINT" not in contents
    metadata = json.loads(viz.history_path.with_suffix(".json").read_text())
    assert metadata["depth"] == 0
    assert metadata["data_fingerprint"] == viz.data_fingerprint
    assert viz.data_fingerprint == fingerprint([193847.65789, 2])


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, None, float("nan")], "b": (True, "x")},
        np.array([[1, 2], [3, 4]], dtype=np.int32),
        np.array([{"x": 1}, [2]], dtype=object),
        pd.DataFrame({"x": [[1, 2], [3, 4]], "y": [1, 2]}),
        pd.Series(pd.Categorical(["a", "b"], categories=["a", "b", "c"], ordered=True)),
        pd.Series(pd.date_range("2020-01-01", periods=2, tz="UTC")),
        pd.Series([1, pd.NA], dtype="Int64"),
        pd.Series(pd.period_range("2020", periods=2, freq="M")),
    ],
)
def test_fingerprints_are_stable_for_supported_nested_data(data):
    assert fingerprint(data) == fingerprint(copy.deepcopy(data))


def test_fingerprint_includes_schema_types_order_and_categories():
    values = [
        [1, 2], (1, 2), [1.0, 2.0], [2, 1],
        np.array([1, 2], dtype=np.int32), np.array([1, 2], dtype=np.int64),
        {"a": 1, "b": 2}, {"b": 2, "a": 1},
        pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"b": [1, 2]}),
        pd.DataFrame({"a": [1, 2]}, index=["a", "b"]),
        pd.Series([1, 2], name="a"), pd.Series([1, 2], name="b"),
        pd.Series(pd.Categorical(["a"], categories=["a", "b"])),
        pd.Series(pd.Categorical(["a"], categories=["b", "a"])),
    ]
    assert len({fingerprint(value) for value in values}) == len(values)
