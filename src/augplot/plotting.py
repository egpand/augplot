"""Numerical trend guides and renderer-aware direct curve labels.

The helpers operate only on supplied data and axes. They do not change global styles.
"""

PLOTTING_VERSION = "2"


def smooth_trend(x, y, *, smoothing=None, max_gap=None, points=200):
    """Return supported segments of a robust, count-weighted smoothing spline.

    Supply y in the units in which the fit should be performed (for example,
    log10(y) for a logarithmic trend). Initial equal-width bins adapt to sample
    size. Adjacent sparse bins are merged toward eight observations per median
    anchor, with a final sparse tail merged backward. Small samples use a lower
    target to preserve up to five anchors. Counts determine fit weights; no
    occupied bin or observation within a supported segment is discarded.
    Coordinates are normalized before SciPy's GCV
    choice of smoothing; ``smoothing`` optionally sets its nonnegative lambda in
    these normalized units. With fewer than five bin centers, a count-weighted
    straight-line fit is used instead.

    By default a gap splits the fit only if it exceeds both eight times the
    median positive spacing and 10% of the full x range. ``max_gap`` overrides
    this threshold in x units. Segments never extend beyond their bin centers.
    Isolated gap-separated observations cannot support a guide and are omitted;
    they remain observations for the caller to show in a scatter plot. A ValueError
    is raised if no segment contains at least two distinct finite x values.
    """
    import numpy as np
    from scipy.interpolate import make_smoothing_spline

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("smooth_trend requires matching one-dimensional x and y arrays")
    if not isinstance(points, (int, np.integer)) or not 2 <= points <= 2000:
        raise ValueError("points must be an integer between 2 and 2000")
    if smoothing is not None and (not np.isfinite(smoothing) or smoothing < 0):
        raise ValueError("smoothing must be None or a finite nonnegative number")
    if max_gap is not None and (not np.isfinite(max_gap) or max_gap <= 0):
        raise ValueError("max_gap must be a finite positive number")
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    unique = np.unique(x)
    if len(unique) < 2:
        raise ValueError("smooth_trend needs at least two distinct finite x values")
    spacing = np.diff(unique)
    gap = max_gap if max_gap is not None else max(8 * np.median(spacing), np.ptp(x) / 10)
    cuts = np.flatnonzero(np.diff(x) > gap) + 1
    result = []
    for indices in np.split(np.arange(len(x)), cuts):
        sx, sy = x[indices], y[indices]
        if len(np.unique(sx)) < 2:
            continue
        bins = min(60, max(8, int(len(sx) ** 0.4)))
        edges = np.linspace(sx[0], sx[-1], bins + 1)
        groups = np.minimum(np.searchsorted(edges, sx, side="right") - 1, bins - 1)
        min_support = min(8, max(1, len(sx) // 5))
        occupied = np.split(np.arange(len(sx)), np.flatnonzero(np.diff(groups)) + 1)
        aggregated = []
        pending = []
        pending_count = 0
        for members in occupied:
            pending.append(members)
            pending_count += len(members)
            if pending_count >= min_support:
                aggregated.append(np.concatenate(pending))
                pending, pending_count = [], 0
        if pending:
            tail = np.concatenate(pending)
            if aggregated:
                aggregated[-1] = np.concatenate((aggregated[-1], tail))
            else:
                aggregated.append(tail)
        centers, medians, counts = [], [], []
        for members in aggregated:
            centers.append(np.median(sx[members]))
            medians.append(np.median(sy[members]))
            counts.append(len(members))
        # Concentrated samples can occupy only one initial bin after merging.
        # Two distinct median anchors are the minimum honest sparse guide.
        if len(centers) < 2:
            midpoint = np.searchsorted(sx, np.median(sx), side="right")
            if midpoint == len(sx):
                midpoint = np.searchsorted(sx, sx[-1], side="left")
            if midpoint == 0:
                midpoint = np.searchsorted(sx, sx[0], side="right")
            halves = [np.arange(midpoint), np.arange(midpoint, len(sx))]
            centers = [np.median(sx[members]) for members in halves]
            medians = [np.median(sy[members]) for members in halves]
            counts = [len(members) for members in halves]
        cx, cy = np.asarray(centers), np.asarray(medians)
        origin, width = cx[0], cx[-1] - cx[0]
        nx = (cx - origin) / width
        baseline = np.median(cy)
        scale = max(np.ptp(cy), np.finfo(float).eps * max(1, abs(baseline)))
        ny = (cy - baseline) / scale
        grid = np.linspace(0, 1, points)
        if len(cx) >= 5:
            weights = np.asarray(counts, dtype=float)
            weights /= weights.mean()
            fit = make_smoothing_spline(nx, ny, w=weights, lam=smoothing)
        else:
            # Counts approximate the inverse variance of aggregated medians.
            root_weights = np.sqrt(np.asarray(counts, dtype=float))
            design = np.column_stack((np.ones(len(nx)), nx))
            coefficients = np.linalg.lstsq(
                design * root_weights[:, None], ny * root_weights, rcond=None
            )[0]
            values = (coefficients[0] + coefficients[1] * grid) * scale + baseline
        if len(cx) >= 5:
            values = fit(grid) * scale + baseline
        if not np.all(np.isfinite(values)):
            raise ValueError("The trend fit is not finite; rescale the supplied fitting units")
        result.append((grid * width + origin, values))
    if not result:
        raise ValueError(
            "No supported trend segments remain after gap splitting; increase max_gap "
            "or supply at least two distinct observations within a segment"
        )
    return result


def label_curves(ax, curves, *, color=None, fontsize=9):
    """Directly label plotted ``(label, x, y)`` segments using measured text boxes.

    Coordinates must be in the actual axis units, including for log axes.
    Repeated labels represent segments of one curve and receive one annotation.
    Text stays within the axes and clear of curves and visible existing axes text. Offset
    placements receive a short leader. An impossible placement raises ValueError.
    """
    import numpy as np
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox

    groups = {}
    total_points = 0
    for curve_index, (label, x, y) in enumerate(curves):
        if curve_index >= 100:
            raise ValueError("At most 100 curve segments can be labeled in one call")
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 1 or x.shape != y.shape:
            raise ValueError("Each labeled curve needs matching one-dimensional x and y")
        total_points += x.size
        if x.size > 10000 or total_points > 100000:
            raise ValueError("Resample curves to at most 10000 points per segment and 100000 total")
        finite = np.isfinite(x) & np.isfinite(y)
        # Invalid log-domain coordinates must not turn into clipped display points.
        if ax.get_xscale() == "log":
            finite &= x > 0
        if ax.get_yscale() == "log":
            finite &= y > 0
        if np.count_nonzero(finite) < 2:
            raise ValueError(f"Curve {label!r} needs at least two finite visible points")
        groups.setdefault(str(label), []).append(np.column_stack((x[finite], y[finite])))
    if not groups:
        return []
    canvas = ax.figure.canvas
    canvas.draw()
    renderer = canvas.get_renderer()
    axes_bounds = ax.get_window_extent(renderer)
    bounds = axes_bounds.padded(-3)
    display = {}
    for label, segments in groups.items():
        display[label] = [ax.transData.transform(segment) for segment in segments]
    # Sample display-space line segments densely enough to detect bbox intersections.
    clouds = []
    cloud_size = 0
    for segments in display.values():
        for segment_index, segment in enumerate(segments):
            dense_parts = []
            for start, stop in zip(segment[:-1], segment[1:], strict=True):
                count = min(1000, max(2, int(np.linalg.norm(stop - start) / 3) + 1))
                cloud_size += count
                if cloud_size > 200000:
                    raise ValueError(
                        "Curve display geometry is too large; resample or narrow limits"
                    )
                dense = np.linspace(start, stop, count)
                clouds.append(dense)
                dense_parts.append(dense)
            segments[segment_index] = np.concatenate(dense_parts)
    cloud = np.concatenate(clouds)
    annotations = []
    boxes = [
        Text.get_window_extent(existing, renderer).padded(3)
        for existing in ax.texts
        if existing.get_visible() and existing.get_text()
    ]
    for label, segments in display.items():
        annotation = ax.annotate(
            label,
            xy=(0, 0),
            xytext=(0, 0),
            textcoords="data",
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color if color is not None else "black",
            annotation_clip=False,
        )
        best = None
        for segment in segments:
            visible = segment[
                (segment[:, 0] >= axes_bounds.x0)
                & (segment[:, 0] <= axes_bounds.x1)
                & (segment[:, 1] >= axes_bounds.y0)
                & (segment[:, 1] <= axes_bounds.y1)
            ]
            candidates = np.unique(
                np.linspace(0, len(visible) - 1, min(15, len(visible))).astype(int)
            )
            for index in candidates:
                anchor = visible[index]
                for dx, dy in [
                    (0, 14),
                    (0, -14),
                    (24, 14),
                    (-24, 14),
                    (24, -14),
                    (-24, -14),
                    (0, 28),
                    (0, -28),
                    (40, 24),
                    (-40, 24),
                    (40, -24),
                    (-40, -24),
                ]:
                    position = anchor + np.array([dx, dy])
                    data_position = ax.transData.inverted().transform(position)
                    annotation.set_position(data_position)
                    box = annotation.get_window_extent(renderer).padded(3)
                    if not (bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)):
                        continue
                    if any(box.overlaps(other) for other in boxes):
                        continue
                    distances = np.hypot(
                        np.maximum(np.maximum(box.x0 - cloud[:, 0], cloud[:, 0] - box.x1), 0),
                        np.maximum(np.maximum(box.y0 - cloud[:, 1], cloud[:, 1] - box.y1), 0),
                    )
                    clearance = distances.min()
                    if clearance < 3:
                        continue
                    score = min(clearance, 20) - np.hypot(dx, dy) / 8
                    if best is None or score > best[0]:
                        best = (score, anchor, data_position, Bbox.from_extents(*box.extents))
        if best is None:
            annotation.remove()
            for previous in annotations:
                previous.remove()
            raise ValueError(
                f"Cannot place required label {label!r} clearly; enlarge the axes, "
                "reduce fontsize, or provide more space around the curves"
            )
        _, anchor, position, box = best
        annotation.remove()
        anchor_data = ax.transData.inverted().transform(anchor)
        annotation = ax.annotate(
            label,
            xy=anchor_data,
            xytext=position,
            textcoords="data",
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color if color is not None else "black",
            annotation_clip=False,
            arrowprops={
                "arrowstyle": "-",
                "color": color if color is not None else "black",
                "lw": 0.6,
                "shrinkA": 3,
                "shrinkB": 2,
            },
        )
        annotations.append(annotation)
        boxes.append(box)
    return annotations
