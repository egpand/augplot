import json

import pandas as pd
import pytest
from conftest import ARRAY_CODE, response
from notebook_charts import FLIGHTS_HEATMAP, FLIGHTS_LINES, PENGUIN_FACETS, PENGUIN_SCATTER
from titanic_fixture import titanic_timeline_events

from augplot import GenerationError, ScopeError
from augplot.execution import execute, parse_response, validate_code

TIMELINE_CODE = """def plot_data(data, *, title=None, figsize=None):
    import numpy as np
    import matplotlib.pyplot as plt
    events = data.sort_values("x").head(50)
    x_values = events["x"]
    baseline = np.asarray(x_values) * 0.0
    levels = np.where(events["side"] > 0, 0.35, -0.35)
    fig, ax = plt.subplots(figsize=figsize or (15, 5), constrained_layout=True)
    ax.set_ylim(-1.4, 1.4)
    ax.set_xlim(0, 10)
    ax.axhline(0, xmin=0.05, xmax=0.95, color="#4a4a4a", zorder=1)
    ax.scatter(x_values, baseline, s=45, color="#4a4a4a", zorder=3)
    ax.vlines(x_values, baseline, levels, color="#4a4a4a", linewidth=1.2, zorder=2)
    labels = zip(
        x_values,
        events["time"],
        events["event"],
        events["highlight"],
    )
    for idx, (x_value, time_label, event_label, highlight) in enumerate(labels):
        time_y = 0.48 if idx % 2 == 0 else -0.48
        event_y = 0.42 if idx % 2 == 0 else -0.42
        alignment = "bottom" if idx % 2 == 0 else "top"
        text_color = "#e3120b" if highlight else "#4a4a4a"
        ax.text(
            x_value,
            time_y,
            f"{time_label}",
            ha="center",
            va=alignment,
            fontfamily="serif",
            fontweight="bold",
            color=text_color,
        )
        ax.text(
            x_value,
            event_y,
            event_label,
            ha="center",
            va=alignment,
            fontfamily="serif",
            color=text_color,
        )
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or "Titanic Timeline", fontweight="bold", fontfamily="serif")
    return fig
"""


RIDGELINE_CODE = """def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, panels = plt.subplots(
        3,
        1,
        figsize=figsize or (12, 8),
        gridspec_kw={"hspace": -0.55},
    )
    axes = [panels[0], panels[1], panels[2]]
    classes = sorted(data["Pclass"].dropna().unique())[:3]
    colors = ["#022133", "#5c693b", "#51371c"]
    show_ticks = [False, False, True]
    for ax, cls, color, keep_ticks in zip(axes, classes, colors, show_ticks):
        sns.kdeplot(
            x="Age",
            data=data[data["Pclass"] == cls],
            fill=True,
            ax=ax,
            cut=0,
            bw_method=0.25,
            linewidth=1.4,
            edgecolor="lightgray",
            color=color,
            alpha=1,
        )
        ax.set_ylim(0, 0.08)
        ax.set_xlim(0, 85)
        ax.set_yticks([])
        if not keep_ticks:
            ax.set_xticks([])
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_frame_on(False)
        ax.set_facecolor("none")
        ax.text(
            -0.2,
            0,
            f"Pclass {cls}",
            fontweight="light",
            fontfamily="serif",
            fontsize=11,
            ha="right",
        )
    fig.text(
        0.13,
        0.81,
        title or "Age distribution by Pclass in Titanic",
        fontweight="bold",
        fontfamily="serif",
        fontsize=16,
    )
    return fig
"""


def assert_rejected_by(code, rule, backend="auto"):
    with pytest.raises(GenerationError) as caught:
        validate_code(code, backend)
    assert [violation["rule"] for violation in caught.value.violations] == [rule]


@pytest.mark.parametrize(
    "statement",
    [
        "import os as os",
        "import pathlib as paths",
        "import requests as requests",
        "import numpy as _np",
        "import numpy._core._methods as methods",
        "from pandas import read_csv",
        'open("/tmp/private")',
        "data.__class__",
        'getattr(data, "shape")',
        'pd.read_csv("https://example.com")',
        'np.load("data.npy")',
        'ax.figure.savefig("out.png")',
        'exec("print(1)")',
        "while True:\n        pass",
        "plot_data(data)",
        'plt.rcParams["font.size"] = 99',
        'plt.style.use("dark_background")',
    ],
)
def test_rejects_prohibited_constructs(statement):
    code = ARRAY_CODE.replace("ax.plot(data)", statement)
    with pytest.raises(GenerationError):
        validate_code(code, "auto")


@pytest.mark.parametrize(
    "code",
    [
        "print('side effect')\n" + ARRAY_CODE,
        "@decorator\n" + ARRAY_CODE,
        ARRAY_CODE.replace("figsize=None", "figsize=func()"),
        ARRAY_CODE.replace("plot_data(data,", "plot_data(data: func(),"),
        ARRAY_CODE + "\ndef another():\n    pass\n",
        ARRAY_CODE.replace("ax.plot(data)", "def nested():\n        pass"),
    ],
)
def test_rejects_executable_definition_metadata_and_extra_definitions(code):
    with pytest.raises(GenerationError):
        validate_code(code, "auto")


def test_backend_import_enforced():
    with pytest.raises(GenerationError):
        validate_code(ARRAY_CODE.replace("ax.plot(data)", "import seaborn as sns"), "matplotlib")


@pytest.mark.parametrize(
    "statement",
    [
        "pd.io.common.os.environ.get('SECRET')",
        "pd.io.common.urlopen('https://example.invalid')",
        "pd.io.common.os.remove('/tmp/never')",
        "plt.imread('/tmp/never')",
        "plt.imsave('/tmp/never', data)",
        "pd.read_pickle('/tmp/never')",
        "np.save('/tmp/never', data)",
        "fn = ax.plot\n    fn(data)",
        "data['plot'](data)",
        "plt = pd\n    plt.read_csv('/tmp/never')",
        "__import__('os').system('false')",
        "import os as os",
        "fig.savefig('/tmp/never')",
    ],
)
def test_security_validator_rejects_indirect_and_io_capabilities(statement):
    code = ARRAY_CODE.replace("ax.plot(data)", statement)
    with pytest.raises(GenerationError) as caught:
        validate_code(code, "auto")
    assert caught.value.violations


def test_generated_cross_validation_panels_work_with_bounded_model_names():
    code = """def plot_data(data, *, title=None, figsize=None):
    import numpy as np
    import matplotlib.pyplot as plt
    names = list(data)[:200]
    metrics = ["test_accuracy", "test_f1", "fit_time"]
    labels = ["Accuracy", "F1 score", "Fit time (seconds)"]
    fig, axes = plt.subplots(1, 3, figsize=figsize or (12, 4))
    if len(names) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "No results", ha="center")
        return fig
    positions = np.arange(len(names))
    for ax, metric, label in zip(axes, metrics, labels):
        observations = [np.asarray(data[name][metric], dtype=float) for name in names]
        finite = [values[np.isfinite(values)] for values in observations]
        means = [np.mean(values) if len(values) else np.nan for values in finite]
        deviations = [np.std(values, ddof=1) if len(values) > 1 else 0 for values in finite]
        ax.errorbar(positions, means, yerr=deviations, fmt="o")
        ax.set_title(label)
        ax.set_xticks(positions)
        ax.set_xticklabels(names, rotation=45, ha="right")
        if metric.startswith("test_"):
            ax.set_ylim(0, 1)
        else:
            ax.set_yscale("log")
    fig.tight_layout()
    return fig
"""
    data = {
        "baseline": {
            "test_accuracy": [0.95, 0.96, 0.97],
            "test_f1": [0.3, 0.4, 0.5],
            "fit_time": [1.0, 1.2, 1.1],
        },
        "candidate": {
            "test_accuracy": [0.96, 0.97, 0.98],
            "test_f1": [0.4, 0.5, 0.6],
            "fit_time": [1.3, 1.5, 1.4],
        },
    }

    figure = execute(code, data, backend="matplotlib")

    assert len(figure.axes) == 3
    assert [axis.get_title() for axis in figure.axes] == [
        "Accuracy",
        "F1 score",
        "Fit time (seconds)",
    ]
    assert len(figure.axes[0].lines) > 0
    assert len(execute(code, {}, backend="matplotlib").axes[0].texts) == 1


def test_generated_horizontal_cross_validation_panels_can_label_y_ticks():
    code = """def plot_data(data, *, title=None, figsize=None):
    import numpy as np
    import matplotlib.pyplot as plt
    names = list(data)[:200]
    metrics = ["test_f1", "fit_time"]
    fig, axes = plt.subplots(2, 1)
    positions = np.arange(len(names))
    for ax, metric in zip(axes, metrics):
        means = np.asarray([np.nanmean(data[name][metric]) for name in names])
        ax.barh(positions, means)
        ax.set_yticks(positions)
        ax.set_yticklabels(names)
    return fig
"""
    data = {
        "baseline": {"test_f1": [0.3, 0.4], "fit_time": [1.0, 1.1]},
        "candidate": {"test_f1": [0.5, 0.6], "fit_time": [1.2, 1.3]},
    }

    figure = execute(code, data, backend="matplotlib")

    assert len(figure.axes) == 2
    assert [tick.get_text() for tick in figure.axes[0].get_yticklabels()] == list(data)


def test_numpy_arange_still_requires_a_proven_bounded_length():
    code = ARRAY_CODE.replace("ax.plot(data)", "ax.plot(np.arange(len(data)))").replace(
        "import matplotlib.pyplot as plt", "import numpy as np\n    import matplotlib.pyplot as plt"
    )
    validate_code(code, "auto")


def test_data_sized_positions_and_axis_styling_from_live_generation():
    code = """def plot_data(data, *, title=None, figsize=None):
    import numpy as np
    import matplotlib.pyplot as plt
    values = np.asarray(data, dtype=float)
    positions = np.arange(1, values.size + 1)
    valid = np.isfinite(values)
    fig, ax = plt.subplots()
    ax.scatter(positions[valid], values[valid])
    if valid.any():
        ax.axhline(values[valid].mean())
    ax.set_axisbelow(True)
    return fig
"""
    figure = execute(code, [3, 4, 5], backend="matplotlib")
    assert len(figure.axes[0].collections) == 1


def test_passive_figsize_value_can_be_initialized_in_a_branch():
    code = """def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    if figsize is None:
        figsize = (10, 5.5)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data)
    return fig
"""

    figure = execute(code, [1, 2, 3], backend="matplotlib")

    assert tuple(figure.get_size_inches()) == (10, 5.5)


def test_series_date_index_can_be_plotted_without_mutation():
    code = """def plot_data(data, *, title=None, figsize=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    converted_index = pd.to_datetime(data.index, errors="coerce")
    fig, ax = plt.subplots()
    ax.plot(converted_index, data.to_numpy())
    return fig
"""
    data = pd.Series([3, 5], index=["2026-01-01", "2026-01-02"])

    figure = execute(code, data, backend="matplotlib")

    assert len(figure.axes[0].lines) == 1


def test_aliased_array_column_extent_can_label_two_dimensional_plot():
    code = """def plot_data(data, *, title=None, figsize=None):
    import numpy as np
    import matplotlib.pyplot as plt
    values = np.asarray(data)
    row_order = np.arange(values.shape[0]) + 1
    column_count = values.shape[1]
    labels = [f"Column {index + 1}" for index in np.arange(column_count)]
    fig, ax = plt.subplots()
    ax.plot(row_order, values)
    ax.legend(labels)
    return fig
"""

    figure = execute(code, [[1, 2], [3, 4], [5, 6]], backend="matplotlib")

    assert len(figure.axes[0].lines) == 2
    assert [text.get_text() for text in figure.axes[0].get_legend().texts] == [
        "Column 1",
        "Column 2",
    ]


def test_scatter_shape_error_reports_safe_repair_hint():
    code = """def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(0, data)
    return fig
"""

    with pytest.raises(GenerationError) as caught:
        execute(code, [17, 23], backend="matplotlib")

    assert "Scatter x and y must have the same number of values" in str(caught.value)
    assert "17" not in str(caught.value)
    assert "23" not in str(caught.value)


def test_numpy_arange_still_rejects_large_data_multipliers():
    code = ARRAY_CODE.replace("ax.plot(data)", "ax.plot(np.arange(data.size * 1000000))").replace(
        "import matplotlib.pyplot as plt", "import numpy as np\n    import matplotlib.pyplot as plt"
    )
    assert_rejected_by(code, "resource_limit")


@pytest.mark.parametrize(
    ("operation", "rule"),
    [
        ('frame.plot(backend="attacker_module")', "dynamic_backend"),
        ('frame.hist(backend="attacker_module")', "dynamic_backend"),
        ('frame.apply("to_pickle", args=("/tmp/augplot-never",))', "dynamic_dispatch"),
        ('frame.agg("to_pickle", path="/tmp/augplot-never")', "dynamic_dispatch"),
        ('frame.aggregate("to_pickle", path="/tmp/augplot-never")', "dynamic_dispatch"),
        ('frame.transform("to_pickle", path="/tmp/augplot-never")', "dynamic_dispatch"),
    ],
)
def test_rejects_pandas_dynamic_dispatch_and_backend_loading(operation, rule):
    code = f"""def plot_data(data, *, title=None, figsize=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    frame = pd.DataFrame({{"value": data}})
    {operation}
    fig, ax = plt.subplots()
    ax.plot(data)
    return fig
"""
    assert_rejected_by(code, rule)


@pytest.mark.parametrize(
    "operation",
    [
        'ax.text(0, 0, "click", url="https://example.invalid")',
        'fig.text(0, 0, "click", url="https://example.invalid")',
        'ax.text(0, 0, "click", url="javascript:alert(1)")',
        'ax.plot(data, url="https://example.invalid")',
        'ax.scatter(data, data, urls=["https://example.invalid"])',
        'artist = ax.text(0, 0, "click")\n    artist.set(url="https://example.invalid")',
        'ax.set(url="https://example.invalid")',
        'ax.text(0, 0, r"$x$", usetex=True)',
        'ax.set_title("title", usetex=True)',
        'ax.text(0, 0, "text", font="/tmp/untrusted.ttf")',
        'ax.text(0, 0, "text", fontproperties="/tmp/untrusted.ttf")',
        "ax.plot(data, picker=True)",
        "artist = ax.plot(data)[0]\n    artist.set(picker=True)",
    ],
)
def test_rejects_active_content_and_dangerous_render_keywords(operation):
    code = ARRAY_CODE.replace("ax.plot(data)", operation)
    assert_rejected_by(code, "dangerous_keyword")


@pytest.mark.parametrize(
    "setup",
    [
        "values = np.zeros((100000, 100000))",
        "values = np.ones((100000, 100000))",
        "values = np.full((100000, 100000), 1)",
        "values = np.arange(1000000000)",
        "values = np.linspace(0, 1, 1000000000)",
        "values = np.repeat(data, 1000000000)",
        "values = np.histogram(data, bins=1000000000)",
        "values = [0] * 1000000000",
        "values = [(i, j) for i in range(10000) for j in range(10000)]",
    ],
)
def test_rejects_static_allocation_and_comprehension_bombs(setup):
    code = f"""def plot_data(data, *, title=None, figsize=None):
    import numpy as np
    import matplotlib.pyplot as plt
    {setup}
    fig, ax = plt.subplots()
    ax.plot(data)
    return fig
"""
    assert_rejected_by(code, "resource_limit")


@pytest.mark.parametrize(
    "plot_call",
    [
        "fig, ax = plt.subplots(10000, 10000)",
        "fig, ax = plt.subplots(nrows=10000, ncols=10000)",
        "fig = plt.figure(figsize=(100000, 100000))\n    ax = fig.add_subplot()",
        "fig, ax = plt.subplots()\n    ax.hist(data, bins=1000000000)",
    ],
)
def test_rejects_matplotlib_resource_bombs(plot_call):
    code = f"""def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    {plot_call}
    return fig
"""
    assert_rejected_by(code, "resource_limit")


def test_rejects_figure_level_subplot_grid_bomb():
    code = """def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    axes = fig.subplots(10000, 10000)
    return fig
"""
    assert_rejected_by(code, "resource_limit")


def test_rejects_regex_enabled_data_transform():
    code = """def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    frame = data.copy()
    cleaned = frame.replace("(a+)+$", "x", regex=True)
    fig, ax = plt.subplots()
    ax.plot(cleaned)
    return fig
"""
    assert_rejected_by(code, "resource_limit")


def test_rejection_happens_before_dynamic_dispatch_side_effect(tmp_path):
    destination = tmp_path / "must-not-exist.pkl"
    code = f"""def plot_data(data, *, title=None, figsize=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    frame = pd.DataFrame({{"value": data}})
    frame.apply("to_pickle", args=({str(destination)!r},))
    fig, ax = plt.subplots()
    return fig
"""
    with pytest.raises(GenerationError):
        execute(code, [1, 2], backend="auto")
    assert not destination.exists()


def test_rejection_happens_before_backend_import(monkeypatch):
    imported = []

    def record_import(name):
        imported.append(name)
        raise AssertionError("validation must run before backend import")

    monkeypatch.setattr("pandas.plotting._core.importlib.import_module", record_import)
    code = """def plot_data(data, *, title=None, figsize=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    frame = pd.DataFrame({"value": data})
    frame.plot(backend="attacker_module")
    fig, ax = plt.subplots()
    return fig
"""
    with pytest.raises(GenerationError):
        execute(code, [1, 2], backend="auto")
    assert imported == []


def test_rejection_happens_before_render_time_subprocess(monkeypatch):
    calls = []

    def record_subprocess(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("validation must run before subprocess execution")

    monkeypatch.setattr("subprocess.check_output", record_subprocess)
    code = ARRAY_CODE.replace("ax.plot(data)", 'ax.text(0, 0, r"$x$", usetex=True)')
    with pytest.raises(GenerationError):
        execute(code, [1, 2], backend="auto")
    assert calls == []


@pytest.mark.parametrize(
    "statement",
    [
        "pd.io.common.os.environ.get('SECRET')",
        "pd.io.common.urlopen('https://example.invalid')",
        "pd.io.common.os.system('false')",
        "pd.io.common.os.remove('/tmp/augplot-never')",
        "np.ctypeslib.load_library('library', '.')",
        "plt.imread('/etc/passwd')",
    ],
)
def test_external_capability_attempts_remain_rejected(statement):
    code = ARRAY_CODE.replace("ax.plot(data)", statement)
    with pytest.raises(GenerationError) as caught:
        validate_code(code, "auto")
    assert caught.value.violations


def test_security_validator_accepts_tick_formatters_and_figure_level_seaborn():
    ticks = ARRAY_CODE.replace(
        'ax.set_title(title or "Values")',
        'ax.xaxis.set_major_locator(ticker.MaxNLocator(5))\n    ax.set_title(title or "Values")',
    ).replace(
        "import matplotlib.pyplot as plt",
        "import matplotlib.pyplot as plt\n    import matplotlib.ticker as ticker",
    )
    validate_code(ticks, "matplotlib")
    seaborn_grid = """def plot_data(data, *, title=None, figsize=None):
    import seaborn as sns
    grid = sns.displot(data=data)
    fig = grid.figure
    return fig
"""
    validate_code(seaborn_grid, "seaborn")


@pytest.mark.parametrize("code", [PENGUIN_SCATTER, PENGUIN_FACETS, FLIGHTS_LINES, FLIGHTS_HEATMAP])
def test_quickstart_generated_plot_corpus_passes_validation(code):
    validate_code(code, "auto")


def test_timeline_annotations_accept_explicitly_bounded_data_loop():
    data = pd.DataFrame(
        {
            "x": [1.5, 2.4, 3.4, 4.5, 6.5, 7.6, 8.0],
            "time": [
                "1:30 PM",
                "9:00 AM",
                "7:15 PM",
                "11:30 PM",
                "12:20 AM",
                "2:00 AM",
                "2:20 AM",
            ],
            "event": [
                "Titanic sets sail.",
                "Message received.",
                "Return of the message requested.",
                "Iceberg warning bells.",
                "Lifeboats lowered.",
                "Rear begins to rise.",
                "Titanic sinks.",
            ],
            "side": [1, -1, 1, -1, 1, -1, 1],
            "highlight": [False, False, False, False, False, False, True],
        }
    )

    validate_code(TIMELINE_CODE, "matplotlib")
    figure = execute(TIMELINE_CODE, data, backend="matplotlib")

    assert len(figure.axes) == 1
    assert figure.axes[0].get_title() == "Titanic Timeline"
    assert len(figure.axes[0].texts) == 2 * len(data)
    assert figure.axes[0].texts[-1].get_color() == "#e3120b"


def test_supplied_titanic_timeline_fixture_renders_all_events():
    events = titanic_timeline_events()

    figure = execute(TIMELINE_CODE, events, backend="matplotlib")

    assert len(events) == 11
    assert len(figure.axes[0].texts) == 22
    assert figure.axes[0].texts[-1].get_text() == "Titanic sinks"
    assert figure.axes[0].texts[-1].get_color() == "#e3120b"


def test_supplied_timeline_can_hide_spines_with_bounded_style_loop():
    code = """def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(data)
    for spine in ["left", "top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    return fig
"""

    figure = execute(code, [1, 2, 3], backend="matplotlib")

    assert all(not spine.get_visible() for spine in figure.axes[0].spines.values())


def test_capped_dataframe_iterrows_can_annotate_timeline():
    code = """def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    events = data.sort_values("x").head(20)
    fig, ax = plt.subplots()
    for index, row in events.iterrows():
        ax.text(row["x"], 0, row["event"])
    return fig
"""

    figure = execute(code, titanic_timeline_events(), backend="matplotlib")

    assert len(figure.axes[0].texts) == 11


def test_same_capability_on_both_branches_can_annotate_a_row():
    code = """def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for index, row in data.head(20).iterrows():
        if row["x"]:
            if row["side"] < 0:
                side = -1
            else:
                side = 1
            ax.text(row["x"], 0, str(side))
    return fig
"""

    figure = execute(code, titanic_timeline_events(), backend="matplotlib")

    assert len(figure.axes[0].texts) == 11


def test_ridgeline_small_multiples_pass_existing_axes_loop_capabilities():
    data = pd.DataFrame(
        {
            "Pclass": [class_number for class_number in (1, 2, 3) for _ in range(8)],
            "Age": [
                age + class_number * 4
                for class_number in (1, 2, 3)
                for age in (8, 16, 24, 32, 40, 48, 56, 64)
            ],
        }
    )

    validate_code(RIDGELINE_CODE, "seaborn")
    figure = execute(RIDGELINE_CODE, data, backend="seaborn")

    assert len(figure.axes) == 3
    assert all(axis.collections for axis in figure.axes)
    assert all(not axis.get_frame_on() for axis in figure.axes)
    assert [axis.texts[0].get_text() for axis in figure.axes] == [
        "Pclass 1",
        "Pclass 2",
        "Pclass 3",
    ]
    assert figure.texts[0].get_text() == "Age distribution by Pclass in Titanic"


@pytest.mark.parametrize("bounded_call", ["head(2)", "head(n=2)", "tail(2)", "tail(n=2)"])
def test_accepts_explicitly_bounded_column_iteration(bounded_call):
    code = f"""def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    rows = data.{bounded_call}
    values = rows["value"]
    for value in values:
        ax.text(0, 0, f"{{value}}")
    return fig
"""
    validate_code(code, "matplotlib")


def test_accepts_single_json_fence():
    code, explanation = parse_response("```json\n" + response(ARRAY_CODE) + "\n```")
    assert code == ARRAY_CODE
    assert explanation


@pytest.mark.parametrize(
    "payload",
    [
        {"code": ARRAY_CODE, "explanation": "Missing status"},
        {"status": "unknown", "code": ARRAY_CODE, "explanation": "Bad status"},
        {"status": "ok", "code": "", "explanation": "Missing code"},
        {"status": "ok", "code": ARRAY_CODE, "explanation": "Fine", "extra": True},
        {"status": "out_of_scope", "code": ARRAY_CODE, "explanation": "Must be empty"},
    ],
)
def test_rejects_responses_outside_unified_schema(payload):
    with pytest.raises(GenerationError):
        parse_response(json.dumps(payload))


def test_parses_unified_out_of_scope_response():
    payload = {
        "status": "out_of_scope",
        "code": "",
        "explanation": "Provide upstream predictions.",
    }
    with pytest.raises(ScopeError, match="upstream predictions"):
        parse_response(json.dumps(payload))


@pytest.mark.parametrize(
    "statement",
    [
        "pd.DataFrame(data).to_clipboard()",
        "np.genfromtxt('/etc/passwd')",
        "plt.imread('/etc/passwd')",
        "plt.imsave('/tmp/plot.png', data)",
        "pd.ExcelWriter('/tmp/plot.xlsx')",
        "fig.canvas.print_png('/tmp/plot.png')",
        "ax.set_url('https://example.invalid')",
        "data.query('value > 1')",
        "pd.set_option('plotting.backend', 'attacker_module')",
        "pd.DataFrame(data).to_gbq('dataset.table')",
        "list(map(pd.DataFrame(data).apply, ['to_pickle']))",
        "list(map(plot_data, data))",
        "np.nan = 1",
        "pd.DataFrame.plot = data",
        "sns.get_dataset_names()",
        "plt.setp(ax, 'usetex', True)",
        "plt.get_current_fig_manager()",
    ],
)
def test_file_and_escape_apis_are_rejected(statement):
    code = ARRAY_CODE.replace(
        "import matplotlib.pyplot as plt",
        "import numpy as np\n    import pandas as pd\n    import seaborn as sns\n"
        "    import matplotlib.pyplot as plt",
    ).replace("ax.plot(data)", statement)
    with pytest.raises(GenerationError) as caught:
        validate_code(code, "auto")
    assert caught.value.violations


def test_common_pandas_aggregation_and_index_assignment_are_allowed():
    code = """def plot_data(data, *, title=None, figsize=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(data).copy()
    frame["scaled"] = frame["value"] * 2
    summary = frame.groupby("group")["scaled"].agg("mean")
    summary.index = summary.index.astype(str)
    fig, ax = plt.subplots()
    ax.bar(summary.index, summary.to_numpy())
    return fig
"""
    figure = execute(code, {"group": ["a", "b", "a"], "value": [1, 2, 3]}, backend="matplotlib")
    assert [patch.get_height() for patch in figure.axes[0].patches] == [4, 4]


def test_series_map_with_local_label_mapping_is_allowed():
    code = """def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    labels = {"a": "Alpha", "b": "Beta"}
    renamed = data["group"].map(labels)
    fig, ax = plt.subplots()
    ax.bar(renamed, data["value"])
    return fig
"""
    figure = execute(
        code, pd.DataFrame({"group": ["a", "b"], "value": [1, 2]}), backend="matplotlib"
    )
    assert [tick.get_text() for tick in figure.axes[0].get_xticklabels()] == ["Alpha", "Beta"]


def test_data_sized_and_nested_annotation_loops_are_allowed():
    code = """def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(data)
    for index in range(len(data)):
        for offset in (0, 1):
            ax.text(index, data[index] + offset, f"{data[index]:.1f}")
    return fig
"""
    figure = execute(code, [1.0, 2.0], backend="matplotlib")
    assert len(figure.axes[0].texts) == 4


def test_safe_numpy_allocation_and_render_keyword_are_allowed():
    code = """def plot_data(data, *, title=None, figsize=None):
    import numpy as np
    import matplotlib.pyplot as plt
    baseline = np.zeros(len(data))
    fig, ax = plt.subplots()
    ax.fill_between(range(len(data)), baseline, data, color="skyblue")
    return fig
"""
    figure = execute(code, [1, 3, 2], backend="matplotlib")
    assert figure.axes[0].collections


def test_huge_f_string_width_is_rejected():
    code = ARRAY_CODE.replace("ax.plot(data)", 'ax.text(0, 0, f"{len(data):10000000}")')
    assert_rejected_by(code, "format_string")
