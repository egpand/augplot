import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import CV_CODE, response

from augplot import plot


def test_bundled_helpers_preserve_generated_local_names(tmp_path, fake_model):
    import numpy as np

    code = '''def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    import augplot.plotting as apx
    smooth_trend = 123
    augplot_helper_smooth_trend = 456
    fig, ax = plt.subplots()
    for x, y in apx.smooth_trend(data["x"], data["y"]):
        ax.plot(x, y)
    ax.set_title(str(smooth_trend + augplot_helper_smooth_trend))
    return fig
'''
    fake_model(response(code))
    data = {"x": np.linspace(0, 1, 100), "y": np.linspace(1, 2, 100)}
    viz = plot(data, show=False, backend="matplotlib")
    target = viz.to_python(tmp_path / "local_names.py", function_name="draw")
    namespace = {}
    exec(target.read_text(), namespace)
    figure = namespace["draw"](data)
    assert figure.axes[0].get_title() == "579"
    assert len(figure.axes[0].lines) == 1
    assert "import augplot.plotting" not in target.read_text()


def test_to_python_runs_without_augplot_or_credentials(tmp_path, cv_data, fake_model, capsys):
    calls = fake_model(response(CV_CODE))
    viz = plot(cv_data, show=False)
    path = viz.to_python(tmp_path / "vis_utils.py", function_name="plot_cv_results")
    assert not hasattr(viz, "save")
    assert "from vis_utils import plot_cv_results" in capsys.readouterr().out
    assert len(calls) == 1
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.endswith("API_KEY") and not key.startswith("AUGPLOT_")
    }
    script = """
import sys
class BlockAugplot:
    def find_spec(self, fullname, *args):
        if fullname == "augplot" or fullname.startswith("augplot."):
            raise ImportError("Augplot is unavailable")
sys.meta_path.insert(0, BlockAugplot())
import matplotlib
matplotlib.use("Agg")
from vis_utils import plot_cv_results
fig = plot_cv_results({"new_model": {"r2": [0.2, 0.4]}}, title="New data")
assert len(fig.axes[0].patches) == 1
assert abs(fig.axes[0].patches[0].get_height() - 0.3) < 1e-8
assert fig.axes[0].get_title() == "New data"
assert any(line.get_visible() for line in fig.axes[0].get_ygridlines())
assert "augplot" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert json.dumps(cv_data) not in path.read_text()


def test_append_and_conflicts_are_atomic(tmp_path, cv_data, fake_model):
    fake_model(response(CV_CODE))
    viz = plot(cv_data, show=False)
    path = tmp_path / "utils.py"
    path.write_text("# Keep this\nexisting = 1\n")
    viz.to_python(path, function_name="plot_one")
    viz.to_python(path, function_name="plot_two")
    viz.to_python(path, function_name="plot_one")
    before = path.read_bytes()
    assert b"# Keep this" in before
    assert path.read_text().count("def plot_one") == 1
    for name in ("existing", "not-valid", "class"):
        with pytest.raises(ValueError):
            viz.to_python(path, function_name=name)
        assert path.read_bytes() == before
    path.write_text("this is not valid python !")
    with pytest.raises(ValueError, match="valid Python"):
        viz.to_python(path)
    assert path.read_text() == "this is not valid python !"


def test_to_python_updates_generated_function_in_stable_default_module(cv_data, fake_model):
    horizontal = CV_CODE.replace("ax.bar(names, means)", "ax.barh(names, means)")
    fake_model(response(CV_CODE), response(horizontal))
    viz = plot(cv_data, show=False)

    first = viz.to_python(function_name="plot_results")
    viz.refine("Make it horizontal", show=False)
    second = viz.to_python(function_name="plot_results")

    assert first == Path("augplot_utils.py")
    assert second == first
    source = first.read_text()
    assert source.count("def plot_results") == 1
    assert "ax.barh(names, means)" in source


def test_to_python_rejects_symbol_import_collision(tmp_path, cv_data, fake_model):
    fake_model(response(CV_CODE))
    viz = plot(cv_data, show=False)
    path = tmp_path / "utils.py"
    path.write_text("import numpy as plot_visualization\n")
    with pytest.raises(ValueError, match="already exists"):
        viz.to_python(path)
