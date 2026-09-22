import json

import pytest
from conftest import ARRAY_CODE, response
from notebook_charts import FLIGHTS_HEATMAP, FLIGHTS_LINES, PENGUIN_FACETS, PENGUIN_SCATTER

from augplot import GenerationError, ScopeError
from augplot.execution import execute, parse_response, validate_code


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
def test_manifest_rejects_indirect_and_io_capabilities(statement):
    code = ARRAY_CODE.replace("ax.plot(data)", statement)
    with pytest.raises(GenerationError) as caught:
        validate_code(code, "auto")
    assert caught.value.violations


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
    code = f'''def plot_data(data, *, title=None, figsize=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    frame = pd.DataFrame({{"value": data}})
    {operation}
    fig, ax = plt.subplots()
    ax.plot(data)
    return fig
'''
    assert_rejected_by(code, rule)


@pytest.mark.parametrize(
    "operation",
    [
        'ax.text(0, 0, "click", url="https://example.invalid")',
        'ax.text(0, 0, "click", url="javascript:alert(1)")',
        'ax.plot(data, url="https://example.invalid")',
        'ax.scatter(data, data, urls=["https://example.invalid"])',
        'artist = ax.text(0, 0, "click")\n    artist.set(url="https://example.invalid")',
        'ax.set(url="https://example.invalid")',
        'ax.text(0, 0, r"$x$", usetex=True)',
        'ax.set_title("title", usetex=True)',
        'ax.text(0, 0, "text", font="/tmp/untrusted.ttf")',
        'ax.text(0, 0, "text", fontproperties="/tmp/untrusted.ttf")',
        'ax.plot(data, picker=True)',
        'artist = ax.plot(data)[0]\n    artist.set(picker=True)',
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
    code = f'''def plot_data(data, *, title=None, figsize=None):
    import numpy as np
    import matplotlib.pyplot as plt
    {setup}
    fig, ax = plt.subplots()
    ax.plot(data)
    return fig
'''
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
    code = f'''def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    {plot_call}
    return fig
'''
    assert_rejected_by(code, "resource_limit")


def test_rejects_figure_level_subplot_grid_bomb():
    code = '''def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    axes = fig.subplots(10000, 10000)
    return fig
'''
    assert_rejected_by(code, "resource_limit")


def test_rejects_regex_enabled_data_transform():
    code = '''def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    frame = data.copy()
    cleaned = frame.replace("(a+)+$", "x", regex=True)
    fig, ax = plt.subplots()
    ax.plot(cleaned)
    return fig
'''
    assert_rejected_by(code, "resource_limit")


@pytest.mark.parametrize(
    "body",
    [
        """candidate = data
    if title:
        candidate = data
    else:
        candidate = fig
    candidate.mean()""",
        """if title:
        candidate = data
    candidate.mean()""",
        """values = [item for item in data]
    item + 1""",
    ],
)
def test_rejects_ambiguous_branch_and_comprehension_provenance(body):
    code = f'''def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    {body}
    return fig
'''
    assert_rejected_by(code, "provenance")


def test_rejection_happens_before_dynamic_dispatch_side_effect(tmp_path):
    destination = tmp_path / "must-not-exist.pkl"
    code = f'''def plot_data(data, *, title=None, figsize=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    frame = pd.DataFrame({{"value": data}})
    frame.apply("to_pickle", args=({str(destination)!r},))
    fig, ax = plt.subplots()
    return fig
'''
    with pytest.raises(GenerationError):
        execute(code, [1, 2], backend="auto")
    assert not destination.exists()


def test_rejection_happens_before_backend_import(monkeypatch):
    imported = []

    def record_import(name):
        imported.append(name)
        raise AssertionError("validation must run before backend import")

    monkeypatch.setattr("pandas.plotting._core.importlib.import_module", record_import)
    code = '''def plot_data(data, *, title=None, figsize=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    frame = pd.DataFrame({"value": data})
    frame.plot(backend="attacker_module")
    fig, ax = plt.subplots()
    return fig
'''
    with pytest.raises(GenerationError):
        execute(code, [1, 2], backend="auto")
    assert imported == []


def test_rejection_happens_before_render_time_subprocess(monkeypatch):
    calls = []

    def record_subprocess(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("validation must run before subprocess execution")

    monkeypatch.setattr("subprocess.check_output", record_subprocess)
    code = ARRAY_CODE.replace(
        "ax.plot(data)", 'ax.text(0, 0, r"$x$", usetex=True)'
    )
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


def test_manifest_accepts_tick_formatters_and_figure_level_seaborn():
    ticks = ARRAY_CODE.replace(
        'ax.set_title(title or "Values")',
        'ax.xaxis.set_major_locator(ticker.MaxNLocator(5))\n    ax.set_title(title or "Values")',
    ).replace(
        "import matplotlib.pyplot as plt",
        "import matplotlib.pyplot as plt\n    import matplotlib.ticker as ticker",
    )
    validate_code(ticks, "matplotlib")
    seaborn_grid = '''def plot_data(data, *, title=None, figsize=None):
    import seaborn as sns
    grid = sns.displot(data=data)
    fig = grid.figure
    return fig
'''
    validate_code(seaborn_grid, "seaborn")


@pytest.mark.parametrize(
    "code", [PENGUIN_SCATTER, PENGUIN_FACETS, FLIGHTS_LINES, FLIGHTS_HEATMAP]
)
def test_quickstart_generated_plot_corpus_passes_manifest(code):
    validate_code(code, "auto")


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
