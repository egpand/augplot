"""Exercise plausible generated programs against reusable plotting helpers offline."""

import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from conftest import response
from matplotlib.backends.backend_agg import FigureCanvasAgg

from augplot import plot

GROUPED_CODE = '''def plot_data(data, *, title=None, figsize=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import augplot.plotting as apx
    fig, ax = plt.subplots(figsize=figsize or (9, 5))
    curves = []
    for label, group in data.groupby("group", sort=True):
        ax.scatter(group["x"], group["y"], s=3, alpha=0.15)
        for x, y in apx.smooth_trend(group["x"], group["y"]):
            ax.plot(x, y)
            curves.append((label, x, y))
    ax.margins(x=0.12, y=0.15)
    apx.label_curves(ax, curves)
    return fig
'''

GAPPED_CODE = '''def plot_data(data, *, title=None, figsize=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import augplot.plotting as apx
    fig, ax = plt.subplots(figsize=figsize or (9, 5))
    ax.scatter(data["day"], data["rate"], s=4, alpha=0.2)
    curves = apx.smooth_trend(data["day"], np.log10(data["rate"]), max_gap=10)
    for x, log_y in curves:
        ax.plot(x, 10 ** log_y)
    ax.set_yscale("log")
    ax.set_xlabel("Days since monitoring began")
    ax.set_ylabel("Event rate")
    return fig
'''

DENSE_CODE = '''def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    import augplot.plotting as apx
    fig, axes = plt.subplots(1, 2, figsize=figsize or (14, 6))
    for ax, (panel, frame) in zip(axes, data.groupby("panel", sort=True)):
        curves = []
        for label, group in frame.groupby("group", sort=True):
            ax.scatter(group["x"], group["y"], s=2, alpha=0.1)
            for x, y in apx.smooth_trend(group["x"], group["y"]):
                ax.plot(x, y)
                curves.append((label, x, y))
        ax.set_title(panel)
        ax.margins(x=0.18, y=0.25)
        apx.label_curves(ax, curves, fontsize=9)
    return fig
'''


def _generate(data, code, fake_model):
    before = data.copy(deep=True)
    calls = fake_model(response(code))
    chart = plot(
        data, prompt="Show the observations and supported smooth trends with readable labels.",
        backend="matplotlib", show=False, cache_dir=None, sample_rows=2,
        max_repairs=0, max_model_calls=1,
    )
    assert len(calls) == chart.model_calls == 1
    assert not chart.cache_hit
    pd.testing.assert_frame_equal(data, before)
    return chart


def _assert_labels_readable(fig, expected):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    labels = [text for ax in fig.axes for text in ax.texts if text.get_text() in expected]
    assert sorted(text.get_text() for text in labels) == sorted(expected)
    boxes = [text.get_window_extent(renderer) for text in labels]
    for index, box in enumerate(boxes):
        assert fig.bbox.contains(box.x0, box.y0)
        assert fig.bbox.contains(box.x1, box.y1)
        assert all(not box.overlaps(other) for other in boxes[index + 1:])


def test_grouped_nonlinear_trends_use_all_observations(fake_model):
    x = np.linspace(0, 8, 401)
    data = pd.concat([
        pd.DataFrame({"x": x, "y": np.sin(x) + offset, "group": label})
        for label, offset in [("Control", 0), ("Treatment", 3)]
    ], ignore_index=True)
    chart = _generate(data, GROUPED_CODE, fake_model)
    ax = chart.figure.axes[0]
    assert sum(len(artist.get_offsets()) for artist in ax.collections) == len(data)
    assert len(ax.lines) == 2
    for line, offset in zip(ax.lines, [0, 3], strict=True):
        fit_x, fit_y = line.get_data()
        assert min(fit_x) >= 0 and max(fit_x) <= 8
        assert np.sqrt(np.mean((fit_y - np.sin(fit_x) - offset) ** 2)) < 0.3
        # A straight line can satisfy counts and ranges but misses these extrema.
        assert np.ptp(fit_y) > 1.5
    _assert_labels_readable(chart.figure, ["Control", "Treatment"])
    plt.close(chart.figure)


def test_monitoring_trend_preserves_gap_and_log_coordinates(fake_model):
    days = np.concatenate([np.linspace(0, 20, 151), np.linspace(50, 70, 151)])
    data = pd.DataFrame({"day": days, "rate": 10 ** (0.02 * days + 0.2 * np.sin(days / 8))})
    chart = _generate(data, GAPPED_CODE, fake_model)
    ax = chart.figure.axes[0]
    assert ax.get_yscale() == "log"
    assert len(ax.collections[0].get_offsets()) == len(data)
    assert len(ax.lines) == 2
    for line in ax.lines:
        x, y = line.get_data()
        assert not (min(x) < 20 and max(x) > 50)
        assert np.all((x <= 20) | (x >= 50))
        assert np.all(y > 0)
        expected = 0.02 * x + 0.2 * np.sin(x / 8)
        assert np.max(np.abs(np.log10(y) - expected)) < 0.15
    plt.close(chart.figure)


def test_dense_labels_remain_readable_in_multiple_panels(fake_model):
    x = np.linspace(0, 10, 121)
    data = pd.concat([
        pd.DataFrame({
            "x": x, "y": 0.05 * x + index * 0.13 + shift,
            "group": f"Series {index + 1}", "panel": panel,
        })
        for panel, shift in [("Morning", 0), ("Evening", 2)]
        for index in range(7)
    ], ignore_index=True)
    chart = _generate(data, DENSE_CODE, fake_model)
    assert len(chart.figure.axes) == 2
    for ax in chart.figure.axes:
        assert sum(len(artist.get_offsets()) for artist in ax.collections) == len(data) // 2
        assert len(ax.lines) == 7
    _assert_labels_readable(chart.figure, [f"Series {i + 1}" for i in range(7)] * 2)
    plt.close(chart.figure)


def test_helper_plot_exports_without_augplot(tmp_path, fake_model, monkeypatch):
    x = np.linspace(0, 8, 101)
    data = pd.DataFrame({"x": x, "y": np.sin(x), "group": "Control"})
    chart = _generate(data, GROUPED_CODE, fake_model)
    target = chart.to_python(tmp_path / "standalone.py", function_name="plot_groups")

    class BlockAugplot:
        def find_spec(self, fullname, *args):
            if fullname == "augplot" or fullname.startswith("augplot."):
                raise ImportError("Augplot is unavailable")

    # The exported function must work even when already-imported package modules disappear.
    with monkeypatch.context() as patch:
        for name in list(sys.modules):
            if name == "augplot" or name.startswith("augplot."):
                patch.delitem(sys.modules, name)
        patch.setattr(sys, "meta_path", [BlockAugplot(), *sys.meta_path])
        spec = importlib.util.spec_from_file_location("standalone", target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        figure = module.plot_groups(data)
    assert len(figure.axes[0].collections[0].get_offsets()) == len(data)
    _assert_labels_readable(figure, ["Control"])
    plt.close(figure)
    plt.close(chart.figure)
