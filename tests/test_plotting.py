"""Geometric and numerical contracts of the reusable plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.text import Text

from augplot.plotting import label_curves, smooth_trend


def test_smoothing_recovers_uneven_nonlinear_trend_with_outliers():
    rng = np.random.default_rng(18)
    x = np.r_[rng.uniform(0, 1.2, 4500), rng.uniform(1.2, 2, 160)]

    def truth(t):
        return 2 - t - 5 * np.maximum(t - 1.4, 0) ** 2

    y = truth(x) + rng.normal(0, 0.16, len(x))
    y[rng.choice(len(y), 250, replace=False)] += 6
    [(cx, cy)] = smooth_trend(x, y)
    assert cx[-1] > 1.95
    assert np.sqrt(np.mean((cy - truth(cx)) ** 2)) < 0.11
    assert np.mean(np.abs(cy[cx > 1.5] - truth(cx[cx > 1.5]))) < 0.15
    assert cy[-1] < cy[np.argmin(abs(cx - 1.4))] - 1.8


def test_gaps_duplicates_nonfinite_and_supported_ranges():
    x = np.r_[np.linspace(0, 1, 50), np.linspace(4, 5, 50)]
    y = np.sin(x)
    curves = smooth_trend(np.r_[x, x, np.nan, 3], np.r_[y, y, 2, np.inf])
    assert len(curves) == 2
    for (cx, cy), (lo, hi) in zip(curves, [(0, 1), (4, 5)], strict=True):
        assert np.all(np.isfinite(cx)) and np.all(np.isfinite(cy))
        assert cx.min() >= lo and cx.max() <= hi
        assert np.all(np.diff(cx) > 0)
    assert len(smooth_trend(x, y, max_gap=4)) == 1


def test_sparse_fallback_and_normalization():
    x = np.array([1, 2, 4, 7])
    [(cx, cy)] = smooth_trend(x, 10 - x)
    np.testing.assert_allclose(cy, 10 - cx, atol=1e-12)
    [(large_x, large_y)] = smooth_trend(x * 1e8 + 1e12, (10 - x) * 1e-8)
    np.testing.assert_allclose(large_y, cy * 1e-8, atol=1e-18)
    np.testing.assert_allclose((large_x - 1e12) / 1e8, cx)


@pytest.mark.parametrize(
    "x,y,kwargs,match",
    [
        ([1, 1], [2, 3], {}, "distinct"),
        ([1, 2], [2], {}, "matching"),
        ([1, 2], [2, 3], {"points": 1}, "points"),
        ([1, 2], [2, 3], {"points": 2001}, "points"),
        ([1, 2], [2, 3], {"smoothing": -1}, "smoothing"),
        ([0, 2, 4], [0, 1, 2], {"max_gap": 1}, "supported"),
    ],
)
def test_actionable_invalid_input(x, y, kwargs, match):
    with pytest.raises(ValueError, match=match):
        smooth_trend(x, y, **kwargs)


def test_labels_crossings_log_axes_and_disconnected_segments():
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.geomspace(1, 100, 200)
    curves = [("ascending", x, x), ("descending", x, 100 / x), ("ascending", x[:30], x[:30])]
    for label, cx, cy in curves:
        ax.plot(cx, cy, label=label)
    ax.set(xscale="log", yscale="log", xlim=(0.7, 140), ylim=(0.7, 140))
    labels = label_curves(ax, curves)
    fig.canvas.draw()
    assert {label.get_text() for label in labels} == {"ascending", "descending"}
    assert len(labels) == 2
    boxes = [Text.get_window_extent(label, fig.canvas.get_renderer()) for label in labels]
    assert not boxes[0].overlaps(boxes[1])
    bounds = ax.get_window_extent()
    for label, box in zip(labels, boxes, strict=True):
        assert bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)
        assert label.arrow_patch is not None
        for _, cx, cy in curves:
            display = ax.transData.transform(np.column_stack([cx, cy]))
            assert not any(box.contains(*point) for point in display)
    plt.close(fig)


def test_impossible_label_is_reported_without_partial_annotations():
    fig, ax = plt.subplots(figsize=(1, 1))
    x = np.linspace(0, 1, 10)
    ax.plot(x, x)
    with pytest.raises(ValueError, match="Cannot place required label"):
        label_curves(ax, [("a label far wider than the whole axes", x, x)])
    assert not ax.texts
    plt.close(fig)


def test_sparse_guide_smooths_instead_of_interpolating_noise():
    x = np.array([0, 1, 2, 3])
    y = np.array([0, 2, 1, 3])
    [(cx, cy)] = smooth_trend(x, y)
    np.testing.assert_allclose(np.diff(cy), np.diff(cy)[0], atol=1e-12)
    assert abs(np.interp(1, cx, cy) - 2) > 0.5


def test_label_geometry_limits_are_actionable():
    fig, ax = plt.subplots()
    x = np.arange(10001)
    with pytest.raises(ValueError, match="Resample"):
        label_curves(ax, [("too many points", x, x)])
    assert not ax.texts
    plt.close(fig)


def test_isolated_observation_does_not_invalidate_supported_guide():
    x = np.r_[np.linspace(0, 1, 100), 10]
    y = np.r_[np.linspace(2, 3, 100), -20]
    [(cx, cy)] = smooth_trend(x, y)
    assert cx.min() >= 0 and cx.max() <= 1
    np.testing.assert_allclose(cy, cx + 2, atol=1e-9)


def test_sparse_contaminated_tail_uses_supported_robust_anchors():
    rng = np.random.default_rng(22)
    x = np.r_[
        rng.uniform(0, 1.2, 4000),
        np.linspace(1.2, 1.9, 45),
        [1.61, 1.63, 1.67, 1.69, 1.86],
    ]

    def truth(t):
        return 2 - t - 5 * np.maximum(t - 1.3, 0) ** 2

    y = truth(x) + rng.normal(0, 0.05, len(x))
    y[-5:] += 6
    [(cx, cy)] = smooth_trend(x, y)
    assert cx[-1] > 1.7
    assert np.mean(abs(cy[cx > 1.3] - truth(cx[cx > 1.3]))) < 0.15
    assert np.interp(1.2, cx, cy) - cy[-1] > 1.5


def test_concentrated_duplicate_sample_keeps_a_sparse_guide():
    [(cx, cy)] = smooth_trend([0] * 9 + [1], [2] * 9 + [3])
    np.testing.assert_allclose(cy, 2 + cx, atol=1e-12)


def test_repeated_label_calls_avoid_existing_text_and_annotations():
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.linspace(0, 1, 100)
    ax.plot(x, x)
    original = ax.annotate(
        "Existing note", xy=(0.4, 0.6), xytext=(0.3, 0.8), arrowprops={"arrowstyle": "->"}
    )
    first = label_curves(ax, [("Alpha", x, x)])
    second = label_curves(ax, [("Bravo", x, x)])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [Text.get_window_extent(text, renderer) for text in [original, *first, *second]]
    for index, box in enumerate(boxes):
        assert not any(box.overlaps(other) for other in boxes[index + 1 :])
    with pytest.raises(ValueError, match="Cannot place required label"):
        label_curves(ax, [("impossibly long label " * 30, x, x)])
    assert list(ax.texts) == [original, *first, *second]
    plt.close(fig)


def test_wholly_invisible_curve_cannot_silently_receive_a_clipped_label():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    x, y = np.array([-0.02, -0.01]), np.array([0.5, 0.6])
    ax.plot(x, y)
    with pytest.raises(ValueError, match="Cannot place required label"):
        label_curves(ax, [("Invisible", x, y)])
    assert not ax.texts
    plt.close(fig)


def test_partially_visible_curve_uses_a_visible_label_anchor():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    # Even a two-point line whose endpoints are outside has a visible interior.
    x = np.array([-0.2, 1.2])
    y = 0.4 + x / 4
    ax.plot(x, y)
    [label] = label_curves(ax, [("Visible portion", x, y)])
    fig.canvas.draw()
    bounds = ax.get_window_extent()
    assert bounds.contains(*ax.transData.transform(label.xy))
    box = Text.get_window_extent(label, fig.canvas.get_renderer())
    assert box.width > 10 and box.height > 5
    assert bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)
    assert label.get_annotation_clip() is False
    plt.close(fig)
