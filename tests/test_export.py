import json
import os
import subprocess
import sys

import pytest
from conftest import CV_CODE, response

from augplot import plot


def test_export_runs_without_augplot_or_credentials(tmp_path, cv_data, fake_model, capsys):
    calls = fake_model(response(CV_CODE))
    viz = plot(cv_data, show=False)
    path = viz.save(tmp_path / "vis_utils.py", function_name="plot_cv_results")
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
    viz.save(path, function_name="plot_one")
    viz.save(path, function_name="plot_two")
    before = path.read_bytes()
    assert b"# Keep this" in before
    for name in ("existing", "plot_one", "not-valid", "class"):
        with pytest.raises(ValueError):
            viz.save(path, function_name=name)
        assert path.read_bytes() == before
    path.write_text("this is not valid python !")
    with pytest.raises(ValueError, match="valid Python"):
        viz.save(path)
    assert path.read_text() == "this is not valid python !"


def test_export_rejects_symbol_import_collision(tmp_path, cv_data, fake_model):
    fake_model(response(CV_CODE))
    viz = plot(cv_data, show=False)
    path = tmp_path / "utils.py"
    path.write_text("import numpy as plot_visualization\n")
    with pytest.raises(ValueError, match="already exists"):
        viz.save(path)
